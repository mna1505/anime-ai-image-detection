from PIL import Image, ImageOps
import torch
from torchvision import transforms

# ============================================================
# PREPROCESSING
# ============================================================
# Tahap ini dilakukan SETELAH dataset preparation dan split.
#
# Strategi:
# 1. Sisi terpanjang -> 1024 px.
# 2. Aspect ratio dipertahankan.
# 3. Citra diberi padding menjadi 1024 x 1024 agar tidak terdistorsi
#    dan dapat dibentuk menjadi batch.
# 4. Baru diturunkan menjadi 224 x 224 untuk input model.
# 5. Training mendapat augmentasi.
# 6. Validation/test TIDAK mendapat augmentasi.
# 7. Semua dinormalisasi dengan ImageNet mean/std karena menggunakan
#    pretrained weights ImageNet.
# ============================================================

MAX_SIDE = 1024 # Batas piksel untuk sisi terpanjang gambar.

# Mengapa harus seperti itu? Gambar anime dari internet memiliki 
# resolusi yang sangat acak (ada yang kecil, ada yang sangat besar). 
# Menyeragamkan sisi terpanjang ke 1024 piksel menjadi langkah awal yang adil 
# agar gambar resolusi tinggi diperkecil, dan gambar resolusi rendah diperbesar 
# ke standar yang sama sebelum diproses lebih lanjut.

MODEL_INPUT_SIZE = 224 # Ukuran akhir resolusi gambar (224x224 piksel) yang akan masuk ke model AI.

# Mengapa harus seperti itu? Model AI jenis Vision populer 
# (seperti ResNet, EfficientNet, atau ViT) dirancang secara arsitektural 
# untuk menerima input berukuran tepat 224x224 piksel.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Nilai rata-rata (mean) dan standar deviasi (standard deviation) 
# warna dari dataset raksasa bernama ImageNet.

# Mengapa harus seperti itu? Karena model yang dipakai menggunakan 
# Pretrained Weights (bobot AI yang sudah pintar karena pernah dilatih 
# menggunakan jutaan gambar ImageNet). Agar model tersebut tidak bingung 
# saat membaca gambar anime Anda, karakteristik warna gambar anime Anda 
# harus disesuaikan (dinormalisasi) mengikuti standar warna ImageNet tempat ia belajar dulu.

class ResizeLongestSide:
    """Resize dengan sisi terpanjang tepat MAX_SIDE tanpa mengubah aspect ratio."""

    def __init__(self, max_side=MAX_SIDE):
        self.max_side = max_side

    def __call__(self, image):
        width, height = image.size
        longest = max(width, height)

        if longest <= 0:
            raise ValueError("Ukuran gambar tidak valid.")

        # Jika sudah lebih kecil dari 1024, gambar tetap diperbesar sehingga
        # sisi terpanjang menjadi 1024. Jika sudah 1024, tidak berubah.
        scale = self.max_side / float(longest)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        return image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

# Fungsi __init__ & __call__: Mengambil sebuah gambar, mencari sisi mana yang paling panjang 
# (lebar atau tinggi), lalu mengubah ukuran sisi terpanjang tersebut menjadi tepat 1024 piksel. 
# Sisi yang lebih pendek akan ikut menyesuaikan secara otomatis secara proporsional.

# Mengapa harus seperti itu? Jika gambar langsung dipaksa ditarik menjadi kotak 224x224 tanpa 
# mempertahankan aspek rasio, gambar anime akan menjadi gepeng atau lonjong (distorsi). 
# AI akan kesulitan mengenali objek jika proporsinya rusak. Metode Image.Resampling.LANCZOS 
# digunakan karena merupakan algoritma pemrosesan gambar terbaik untuk menjaga ketajaman 
# detail piksel saat gambar diperbesar/diperkecil.


class PadToSquare:
    """Pad gambar ke kotak tanpa melakukan stretching."""

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
            fill=self.fill,
        )

# Fungsi __init__ & __call__: Mengambil gambar hasil resize sebelumnya 
# (yang sekarang salah satu sisinya sudah 1024), 
# lalu menambahkan ruang kosong berwarna hitam (fill=0) di sisi yang lebih pendek 
# hingga gambar tersebut menjadi kotak sempurna berukuran 1024x1024 piksel. 
# Sisa ruang dibagi rata ke kiri-kanan atau atas-bawah agar posisi gambar tetap di tengah.

# Mengapa harus seperti itu? Komputer memproses gambar dalam kelompok (batch). 
# Agar bisa dikelompokkan, semua gambar wajib berbentuk kotak dengan dimensi yang sama persis. 
# Memberikan bantalan (padding) hitam adalah cara teraman untuk membuat gambar menjadi kotak 
# tanpa merusak isi konten aslinya.


BASE_TRANSFORM = [
    ResizeLongestSide(MAX_SIDE),
    PadToSquare(fill=0),
    transforms.Resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        antialias=True,
    ),
] # Rangkaian tiga langkah dasar (Ubah Sisi ➔ Beri Bantalan Kotak ➔ Kecilkan ke 224x224).

# Mengapa harus seperti itu? Variabel ini dibuat agar langkah dasar pengubahan bentuk 
# gambar tidak perlu ditulis berulang-ulang, baik untuk data latihan maupun data ujian.


def get_train_transform():
    """Preprocessing + augmentation untuk TRAIN saja."""
    return transforms.Compose(
        BASE_TRANSFORM
        + [
            transforms.RandomHorizontalFlip(p=0.5), #  Membalik gambar secara horizontal (seperti cermin) dengan peluang 50%.
            transforms.RandomRotation(degrees=10), # Memutar gambar sedikit ke kiri atau kanan maksimal 10 derajat.
            transforms.ColorJitter( # Mengubah tingkat kecerahan, kontras, dan kepekatan warna gambar secara acak.
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(), # Mengubah gambar biasa menjadi matriks angka (Tensor) agar bisa dihitung oleh kartu grafis (GPU).
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), # Menerapkan rumus matematika ImageNet pada angka piksel gambar.
        ]
    )

#  Fungsi ini menggabungkan tiga langkah dasar (BASE_TRANSFORM) 
# dengan teknik Augmentasi dan Normalisasi khusus untuk data TRAINING.

# Mengapa harus seperti itu? Data latihan harus diberi augmentasi (variasi buatan). 
# Tujuannya agar model AI tidak cepat bosan dan tidak menghafal satu posisi gambar saja (Overfitting). 
# Jika AI dilatih dengan gambar yang sedikit miring atau warnanya agak gelap, 
# AI akan menjadi lebih tangguh saat mendeteksi gambar serupa di dunia nyata nanti.


def get_eval_transform():
    """Preprocessing deterministic untuk validation dan test."""
    return transforms.Compose(
        BASE_TRANSFORM
        + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

# Kegunaan: Fungsi preprocessing untuk data VALIDASI dan TEST. 
# Fungsi ini hanya berisi langkah dasar, diubah ke Tensor, lalu dinormalisasi. 
# Tidak ada unsur acak atau pembalikan gambar di sini.

# Mengapa harus seperti itu? Saat menguji kemampuan AI, kita harus bersikap adil dan objektif. 
# Gambar tidak boleh dimanipulasi (tidak boleh dibalik atau diputar secara acak) 
# agar kita tahu performa asli AI dalam mendeteksi gambar apa adanya.


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"CUDA tersedia: {name}")
        return torch.device("cuda")

    print("CUDA tidak tersedia. Menggunakan CPU.")
    return torch.device("cpu")

# Kegunaan: Memeriksa apakah komputer memiliki kartu grafis (GPU) Nvidia yang mendukung CUDA. 
# Jika ada, program akan berjalan di GPU; jika tidak, program akan beralih menggunakan prosesor biasa (CPU).

# Mengapa harus seperti itu? Proses pelatihan Deep Learning melibatkan miliaran perhitungan matematika matriks. 
# Menggunakan GPU (CUDA) bisa mempercepat proses latihan hingga puluhan kali lipat lebih cepat 
# dibanding menggunakan CPU biasa.


if __name__ == "__main__":
    # Quick check transform tanpa memerlukan dataset.
    sample = Image.new("RGB", (400, 600), "white")
    transform = get_eval_transform()
    tensor = transform(sample)
    print("Contoh input 400x600 -> tensor:", tuple(tensor.shape))

# Kegunaan: Area uji coba mandiri. Kode di bawahnya membuat satu gambar tiruan putih berukuran 400x600 piksel, 
# lalu memasukkannya ke fungsi get_eval_transform() 
# untuk melihat apakah sistem berhasil mengubahnya menjadi format tensor tanpa eror.

# Mengapa harus seperti itu? Ini adalah praktik pemrograman yang baik (sanity check). 
# Kita bisa memastikan logika matematika pada fungsi pembubuhan bantalan (padding) dan pengubahan ukuran (resize) 
# sudah berjalan lancar sebelum kita menjalankan seluruh kode pada jutaan gambar dataset yang asli.