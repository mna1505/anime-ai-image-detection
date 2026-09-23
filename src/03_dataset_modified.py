from pathlib import Path
from importlib import import_module
import random

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ============================================================
# ALUR 1: SETTING DIRETORI DAN KONFIGURASI GLOBAL
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT_DIR / "dataset" / "splits"

# Kegunaan: Menunjuk otomatis jalur folder utama proyek dan folder tempat pecahan data (train, validation, test) berada.
#
# Mengapa harus seperti itu? Agar skrip ini otomatis tahu di mana harus mencari folder data tanpa 
# perlu mengetik jalur folder secara manual (hardcode). Langkah ini menjaga kode Anda tetap aman 
# dan tidak memicu eror jika folder proyek dipindahkan ke komputer atau laptop lain.

preprocessing = import_module("02_preprocessing_modified")

# Kegunaan: Mengimpor modul skrip preprocessing versi MODIFIED secara dinamis.
#
# Mengapa harus seperti itu? Skrip modified ini memiliki fungsi pengacak dan perusak gambar tingkat lanjut (Level 1-4). 
# Dengan menghubungkannya secara dinamis, pipa dataset kita bisa menyuplai gambar yang sudah dimanipulasi 
# secara bervariasi sesuai level eksperimen yang sedang kita jalankan.

BATCH_SIZE_DEFAULT = 32 

# Kegunaan: Jumlah gambar anime yang akan dikirim ke kartu grafis (GPU) dalam satu kali suapan komputasi.
#
# Mengapa harus seperti itu? Kartu grafis (GPU) memiliki batas memori yang disebut VRAM. 
# Kita tidak bisa memasukkan ribuan gambar sekaligus karena akan membuat GPU kehabisan memori dan crash (Out of Memory). 
# Ukuran 32 gambar per batch adalah angka standar yang paling stabil untuk menjaga keseimbangan kecepatan dan kapasitas memori GPU.

NUM_WORKERS_DEFAULT = 2 

# Kegunaan: Jumlah inti prosesor (CPU sub-threads) yang ditugaskan khusus untuk membuka dan menyiapkan gambar dari harddisk.
#
# Mengapa harus seperti itu? Ini adalah trik kerja sama tim (Multi-processing). Selagi GPU Anda sedang sibuk 
# menghitung dan melatih 32 gambar pertama, 2 pekerja CPU ini sudah bergerak di latar belakang untuk membuka 32 gambar 
# berikutnya. Efeknya, waktu tunggu (bottleneck) hilang dan proses latihan menjadi jauh lebih cepat.

RANDOM_SEED = 42 

# Kegunaan: Angka pengunci fungsi acak untuk pengocokan urutan data dan efek manipulasi gambar.
#
# Mengapa harus seperti itu? Dalam riset AI, eksperimen harus bisa diuji ulang dengan hasil yang sama persis (reproducible). 
# Angka 42 bertindak sebagai "jangkar" agar pola pengacakan data latihan tidak berubah-ubah setiap kali kode dijalankan ulang.


# ============================================================
# ALUR 2: FUNGSI FONDASI BACA DATA & NYALAKAN PEKERJA CPU
# ============================================================

def create_dataset(folder, transform):
    """Membaca folder split menggunakan ImageFolder."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")

    return ImageFolder(
        root=str(folder),
        transform=transform,
    )

# Kegunaan: Mengubah folder biasa di komputer menjadi objek dataset terstruktur yang dipahami oleh PyTorch.
#
# Mengapa harus seperti itu? Fungsi ImageFolder bawaan PyTorch sangat cerdas. Ia otomatis membaca sub-folder 
# di dalamnya (seperti folder 'human' dan 'ai') dan langsung menganggap nama folder tersebut sebagai label kelasnya. 
# Proteksi `if not folder.exists()` wajib dipasang agar program langsung mogok dengan pesan eror yang jelas jika Anda lupa menaruh dataset.


def seed_worker(worker_id):
    """Seed worker agar augmentasi/data loading tetap reproducible."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)

# Kegunaan: Mengunci angka acak pada setiap sub-thread pekerja CPU pembuka gambar.
#
# Mengapa harus seperti itu? Karena pada versi modified ini kita menggunakan teknik perusakan gambar yang sangat acak (seperti blur dan kompresi JPEG), 
# setiap pekerja CPU bisa menghasilkan urutan acak yang berantakan jika tidak dikunci. Fungsi ini memastikan sifat 
# eksperimen yang dapat diulang (reproducible) tetap terjaga hingga ke level sub-thread terkecil.


# ============================================================
# ALUR 3: INTI PROGRAM - PEMBUATAN EMPAT PIPA DATALOADER
# ============================================================

def get_dataloaders(
    batch_size=BATCH_SIZE_DEFAULT,
    num_workers=NUM_WORKERS_DEFAULT,
    model_name="resnet50",
    level=1,
):
    """Membuat empat DataLoader dinamis dengan pemisahan dataset ketat."""

    # 1. Pipa Data Latihan (TRAIN): Satu-satunya yang menggunakan augmentasi ekstrem (Level 1-4).
    train_dataset = create_dataset(
        SPLIT_DIR / "train",
        preprocessing.get_train_transform(
            model_name=model_name,
            level=level,
        ),
    )

    # 2. Pipa Ujian Tengah Semester (VALIDATION): Menggunakan mode evaluasi bersih tanpa manipulasi gambar.
    validation_dataset = create_dataset(
        SPLIT_DIR / "validation",
        preprocessing.get_eval_transform(),
    )

    # 3. Pipa Ujian Akhir Internal (TEST IN-DOMAIN): Menggunakan mode evaluasi bersih.
    test_in_domain_dataset = create_dataset(
        SPLIT_DIR / "test_in_domain",
        preprocessing.get_eval_transform(),
    )

    # 4. Pipa Ujian Tingkat Dewa (TEST CROSS-GENERATOR): Menggunakan mode evaluasi bersih.
    test_cross_dataset = create_dataset(
        SPLIT_DIR / "test_cross_generator",
        preprocessing.get_eval_transform(),
    )

    # Mengapa pemisahan data ini sangat ketat?
    # Sesuai aturan riset ketahanan (robustness), dataset dari sumber luar seperti Human Danbooru2021 dan NovelAI3 
    # HANYA boleh ditaruh di dalam pipa 'test_cross_generator'. Mereka tidak boleh menyusup ke data latihan (train). 
    # Ini adalah ujian pamungkas untuk melihat apakah AI Anda tetap tangguh saat melihat generator gambar baru yang belum pernah dipelajari sebelumnya.

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    # Kamus konfigurasi efisiensi memori yang dipakai bersama oleh keempat dataloader.
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # Mengapa isi parameter common = dict(...) harus seperti itu?
    #
    # pin_memory=torch.cuda.is_available(): Jika Anda memakai GPU, fitur ini akan mengunci posisi memori di RAM 
    # agar proses transfer gambar ke VRAM kartu grafis berjalan secepat kilat.
    #
    # persistent_workers=(num_workers > 0): Menjaga agar pekerja CPU tidak mematikan dirinya sendiri tiap kali 
    # batch gambar berganti. Mereka tetap siaga (stand-by) sehingga menghemat waktu jeda komputasi.

    return {
        # shuffle=True: Urutan gambar latihan wajib diacak di setiap putaran latihan (epoch) agar AI tidak menghafal urutan urutan file.
        "train": DataLoader(train_dataset, shuffle=True, **common),
        # shuffle=False: Data evaluasi dan tes tidak boleh diacak agar hasil penilaian performa berjalan konsisten dan berurutan.
        "validation": DataLoader(validation_dataset, shuffle=False, **common),
        "test_in_domain": DataLoader(test_in_domain_dataset, shuffle=False, **common),
        "test_cross_generator": DataLoader(test_cross_dataset, shuffle=False, **common),
    }


# ============================================================
# TAHAP 4: DIAGNOSTIK & AUDIT DATASET MANDIRI
# ============================================================

def print_dataset_information():
    loaders = get_dataloaders()

    print("\n" + "=" * 60)
    print("MODIFIED DATASET INFORMATION (LAPORAN PIPELINE MODIFIKASI)")
    print("=" * 60)

    for name, loader in loaders.items():
        dataset = loader.dataset

        print(f"\n[{name}]")
        print("Jumlah gambar:", len(dataset))
        print("Class:", dataset.classes)
        print("Mapping (Konversi Angka):", dataset.class_to_idx)

        # Menghitung otomatis sebaran data per kategori kelas.
        counts = {class_name: 0 for class_name in dataset.classes}
        for _, label in dataset.samples:
            counts[dataset.classes[label]] += 1

        print("Distribusi:", counts)

# Kegunaan: Membuka seluruh bungkusan pipa data loader untuk diaudit fungsinya sebelum training dimulai.
#
# Mengapa harus seperti itu? Untuk memastikan secara instan lewat teks terminal bahwa jumlah file seimbang, 
# kelas data tidak tertukar (misal kelas human jadi angka 0, kelas ai jadi angka 1), dan sistem folder splits Anda sudah terbaca dengan sempurna.


if __name__ == "__main__":
    # Menjalankan fungsi audit otomatis jika skrip ini diklik running secara langsung.
    print_dataset_information()
