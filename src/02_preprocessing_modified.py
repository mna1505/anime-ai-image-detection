from pathlib import Path
from PIL import Image, ImageOps
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

# ============================================================
# PREPROCESSING MODIFIED
# ============================================================
# Tahap ini dilakukan SETELAH dataset preparation dan split.
#
# Tujuan utama preprocessing modified adalah memberikan variasi
# tambahan pada DATA TRAINING agar model tidak hanya menghafal
# pola gambar yang terlihat selama proses training.
#
# Gambar training diberi beberapa perubahan secara acak, seperti:
# - crop dan resize,
# - flip horizontal,
# - rotasi ringan,
# - perubahan warna,
# - Gaussian Blur,
# - simulasi kompresi JPEG,
# - Random Erasing.
#
# Perubahan tersebut digunakan untuk membantu model menjadi lebih
# tahan terhadap variasi kualitas, posisi, warna, dan kondisi
# gambar ketika menemukan data baru.
#
# Untuk validation dan test, augmentasi acak TIDAK digunakan.
# Data evaluasi harus tetap deterministik agar performa model
# dapat dibandingkan secara adil.
#
# Catatan:
# ------------------------------------------------------------
# File ini mengatur TRANSFORMASI GAMBAR.
# Strategi training seperti MixUp, Label Smoothing, Learning
# Rate Scheduler, dan Discriminative Learning Rate diatur pada
# file 04_train_modified.py.
# ============================================================

MAX_SIDE = 1024
# Batas ukuran sisi terpanjang gambar.
#
# Gambar dari internet memiliki resolusi yang sangat beragam.
# Dengan menetapkan sisi terpanjang menjadi 1024 piksel, ukuran
# awal gambar menjadi lebih seragam sebelum proses berikutnya.
#
# Aspect ratio tetap dipertahankan sehingga gambar tidak gepeng
# atau memanjang.

MODEL_INPUT_SIZE = 224
# Ukuran akhir gambar yang akan diberikan kepada model.
#
# Keempat arsitektur penelitian menggunakan input 224 x 224
# pada konfigurasi pretrained ImageNet yang digunakan.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Mean dan standard deviation dari ImageNet.
#
# Model menggunakan pretrained weights ImageNet. Oleh karena itu,
# nilai piksel input dinormalisasi menggunakan statistik yang sama
# dengan distribusi data ketika pretrained model belajar.

# ============================================================
# PARAMETER AUGMENTASI
# ============================================================
# Nilai P berarti probability atau peluang sebuah efek diterapkan.
# Contoh p=0.25 berarti efek tersebut memiliki peluang 25%
# untuk diterapkan pada gambar.
#
# Nilai parameter dibuat tetap agar perlakuan preprocessing
# konsisten pada seluruh arsitektur model.
# ============================================================
COLOR_JITTER_STRENGTH = 0.20 # Kekuatan perubahan warna.
GAUSSIAN_BLUR_P = 0.25        # 25% peluang gambar menjadi buram/blur.
JPEG_COMPRESSION_P = 0.35     # 35% peluang gambar dibuat pecah-pecah kotak (efek internet lemot).
RANDOM_ERASING_P = 0.15       # 15% peluang ada bagian gambar yang dihapus/ditutup kotak noise.
JPEG_QUALITY_RANGE = (55, 90) # Rentang kualitas kompresi JPEG (semakin kecil angka, semakin buram hancur).


class ResizeLongestSide:
    """
    Mengubah ukuran sisi terpanjang gambar menjadi 1024 piksel
    tanpa mengubah aspect ratio.

    Contoh:
        400 x 600 -> sisi terpanjang 600
        -> sisi terpanjang diperbesar menjadi 1024
        -> sisi lainnya ikut dihitung secara proporsional.

    Tujuannya agar gambar memiliki ukuran awal yang lebih seragam
    tanpa merusak bentuk asli objek.
    """
    def __init__(self, max_side=MAX_SIDE):
        self.max_side = max_side

    def __call__(self, image):
        width, height = image.size

        # Menentukan panjang sisi kotak yang dibutuhkan.
        longest = max(width, height)

        if longest <= 0:
            raise ValueError("Gambar tidak valid.")

        scale = self.max_side / float(longest)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        return image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )


# Mengapa aspect ratio dipertahankan?
# ------------------------------------------------------------
# Jika gambar langsung dipaksa menjadi 224 x 224, gambar yang
# awalnya portrait atau landscape dapat mengalami distorsi.
# Distorsi tersebut dapat mengubah bentuk karakter dan objek.
#
# LANCZOS digunakan untuk menjaga detail ketika gambar diperbesar
# atau diperkecil.


class PadToSquare:
    """
    Menambahkan padding pada sisi gambar yang lebih pendek agar
    gambar menjadi persegi tanpa melakukan stretching.

    Padding berwarna hitam secara default dan dibagi sehingga
    gambar tetap berada di tengah.
    """
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        side = max(width, height)

        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top

        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill
        )


# Mengapa menggunakan padding?
# ------------------------------------------------------------
# Semua gambar dalam satu batch harus memiliki ukuran yang sama.
# Padding membuat gambar menjadi persegi tanpa merusak proporsi
# karakter atau objek di dalam gambar.


class RandomJPEGCompression:
    """
    Mensimulasikan artefak akibat kompresi JPEG.

    Proses:
    1. Program menentukan secara acak apakah efek diterapkan.
    2. Jika diterapkan, kualitas JPEG dipilih secara acak.
    3. Gambar disimpan sementara di memory sebagai JPEG.
    4. Gambar dibuka kembali dan digunakan untuk training.

    Tujuannya agar model tidak terlalu bergantung pada detail
    piksel yang sangat bersih.
    """
    def __init__(self, p=0.35, quality_range=(55, 90)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, image):
        # Jika peluang efek tidak terpenuhi, gambar dikembalikan
        # tanpa perubahan.
        if random.random() > self.p:
            return image

        quality = random.randint(*self.quality_range)

        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality) # Gambar disimpan di memori dengan kualitas rendah.
        buffer.seek(0)

        # Gambar dibuka kembali setelah mengalami kompresi.
        return Image.open(buffer).convert("RGB")


# ============================================================
# TRANSFORMASI DASAR
# ============================================================
# Semua data melewati tiga langkah dasar berikut:
#
# 1. ResizeLongestSide -> sisi terpanjang menjadi 1024.
# 2. PadToSquare       -> gambar menjadi persegi.
# 3. Resize            -> ukuran akhir menjadi 224 x 224.
#
# Transformasi dasar ini digunakan baik sebelum augmentasi TRAIN
# maupun sebelum evaluasi.
BASE_TRANSFORM = [
    ResizeLongestSide(MAX_SIDE),
    PadToSquare(fill=0),
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), antialias=True),
]


def get_train_transform(model_name="resnet50", level=1):
    """
    Membuat pipeline preprocessing dan augmentation untuk TRAIN.

    Parameter:
    ------------------------------------------------------------
    model_name
        Nama arsitektur model. Parameter dipertahankan agar
        fungsi dapat dipanggil oleh pipeline untuk semua model.

    level
        Tingkat strategi training:
        - Level 1, 2, 3 menggunakan augmentasi visual terarah.
        - Level 4 menggunakan RandAugment yang lebih agresif.

    Catatan:
    ------------------------------------------------------------
    Komponen seperti MixUp, Label Smoothing, scheduler, dan
    Discriminative Learning Rate bukan bagian dari transformasi
    gambar ini. Komponen tersebut berada di file training.
    """
    if level not in (1, 2, 3, 4):
        raise ValueError("Level training harus di antara angka 1 sampai 4.")

    # ========================================================
    # LEVEL 4: AUGMENTASI LEBIH AGRESIF
    # ========================================================
    # Level 4 menggunakan RandAugment.
    # RandAugment memilih beberapa operasi augmentasi secara
    # otomatis sehingga variasi gambar training menjadi lebih luas.
    # ========================================================
    if level == 4:
        return transforms.Compose(
            BASE_TRANSFORM
            + [
                # RandAugment memilih dan menerapkan operasi
                # augmentasi secara otomatis.
                transforms.RandAugment(num_ops=2, magnitude=9),
                # Simulasi kompresi JPEG tetap ditambahkan untuk
                # memberikan variasi artefak kompresi.
                RandomJPEGCompression(p=JPEG_COMPRESSION_P, quality_range=JPEG_QUALITY_RANGE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    # ========================================================
    # LEVEL 1, 2, DAN 3: AUGMENTASI VISUAL TERARAH
    # ========================================================
    # Pipeline visual ini digunakan bersama oleh Level 1, 2,
    # dan 3. Perbedaan strategi Level 2 dan Level 3 terutama
    # berada pada file training.
    # ========================================================
    blur_p = GAUSSIAN_BLUR_P
    jpeg_p = JPEG_COMPRESSION_P
    erase_p = RANDOM_ERASING_P
    jitter = COLOR_JITTER_STRENGTH

    return transforms.Compose(
        BASE_TRANSFORM
        + [
            # 1. RANDOM RESIZED CROP
            # Mengambil area gambar secara acak kemudian
            # mengubahnya menjadi ukuran input model.
            #
            # scale 0.85-1.0 berarti area crop berada pada
            # kisaran 85%-100% dari area gambar.
            transforms.RandomResizedCrop(
                MODEL_INPUT_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.90, 1.10),
                antialias=True,
            ),
            # 2. RANDOM HORIZONTAL FLIP
            # Membalik gambar secara horizontal seperti cermin.
            # p=0.5 berarti peluang penerapan sebesar 50%.
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. RANDOM ROTATION
            # Memutar gambar secara acak hingga 10 derajat.
            # Tujuannya agar model tidak hanya mengenali objek
            # pada posisi yang benar-benar tegak.
            transforms.RandomRotation(degrees=10),
            # 4. COLOR JITTER
            # Mengubah brightness, contrast, saturation, dan hue
            # secara ringan.
            #
            # brightness/contrast/saturation = 20%
            # hue = 3%
            #
            # Tujuannya memberikan variasi warna tanpa mengubah
            # karakteristik gambar secara ekstrem.
            transforms.ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=0.03,
            ),
            # 5. GAUSSIAN BLUR
            # Memberikan blur ringan dengan peluang 25%.
            # Efek ini mensimulasikan gambar yang kehilangan
            # sebagian ketajaman detail.
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=blur_p,
            ),
            # 6. RANDOM JPEG COMPRESSION
            # Memberikan simulasi artefak kompresi JPEG dengan
            # peluang 35% dan kualitas acak 55-90.
            RandomJPEGCompression(p=jpeg_p, quality_range=JPEG_QUALITY_RANGE),
            # 7. TO TENSOR
            # Mengubah gambar PIL menjadi Tensor berupa data
            # numerik yang dapat diproses oleh PyTorch dan GPU.
            transforms.ToTensor(),
            # 8. NORMALIZATION
            # Menyesuaikan distribusi nilai piksel dengan standar
            # ImageNet yang digunakan oleh pretrained model.
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # 9. RANDOM ERASING
            # Menutup atau menghapus sebagian kecil area gambar
            # secara acak dan mengisinya dengan nilai acak.
            #
            # Tujuannya agar model tidak terlalu bergantung pada
            # satu bagian tertentu dari gambar.
            transforms.RandomErasing(
                p=erase_p,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


def get_eval_transform():
    """
    Membuat preprocessing untuk validation dan test.

    Tidak ada augmentasi acak pada fungsi ini.

    Tujuannya agar seluruh model dievaluasi menggunakan kondisi
    input yang konsisten dan dapat dibandingkan secara objektif.
    """
    return transforms.Compose(
        BASE_TRANSFORM
        + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_device():
    """
    Memeriksa ketersediaan CUDA.

    Jika GPU dengan CUDA tersedia, program menggunakan GPU.
    Jika tidak tersedia, program menggunakan CPU.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(
            f"CUDA tersedia: {name}. "
            "Pemrosesan menggunakan GPU siap dijalankan!"
        )
        return torch.device("cuda")

    print("CUDA tidak tersedia. Menggunakan CPU.")
    return torch.device("cpu")


# ============================================================
# QUICK CHECK / SANITY CHECK
# ============================================================
# Bagian ini digunakan untuk menguji pipeline secara mandiri.
# Program membuat satu gambar contoh berukuran 400 x 600,
# menjalankan preprocessing evaluasi, kemudian memeriksa bentuk
# Tensor yang dihasilkan.
#
# Tujuannya memastikan preprocessing dapat berjalan tanpa error
# sebelum digunakan pada dataset penelitian.
# ============================================================

if __name__ == "__main__":
    sample = Image.new("RGB", (400, 600), "white")

    tensor = get_eval_transform()(sample)

    # Bentuk yang diharapkan:
    # (3, 224, 224)
    print(
        "Hasil uji coba modifikasi sukses diubah menjadi "
        "tensor berukuran:",
        tuple(tensor.shape)
    )
