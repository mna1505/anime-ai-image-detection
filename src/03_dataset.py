from pathlib import Path
from importlib import import_module
import random

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT_DIR / "dataset" / "splits"

# Menunjuk jalur folder utama proyek dan folder tempat pecahan data (train, validation, test) berada.

# Mengapa harus seperti itu? Agar skrip ini otomatis tahu 
# di mana harus mencari folder data tanpa perlu mengetik 
# jalur folder secara manual (hardcode) yang bisa memicu eror 
# jika kode dipindah ke komputer lain.

preprocessing = import_module("02_preprocessing")

# Mengimpor modul skrip preprocessing (Tahap 2) 
# secara dinamis menggunakan import_module.

# Mengapa harus seperti itu? Langkah ini menghubungkan 
# alur kerja sehingga fungsi modifikasi gambar 
# (get_train_transform dan get_eval_transform) 
# bisa langsung ditempelkan otomatis pada gambar saat dibaca oleh program.

BATCH_SIZE_DEFAULT = 32 # Jumlah gambar yang akan dikirim ke komputer AI dalam satu kali komputasi/suapan.

# Mengapa harus seperti itu? Kartu grafis (GPU) 
# memiliki batas memori (VRAM). 
# Kita tidak bisa memasukkan semua gambar (ribuan file)
# sekaligus karena GPU akan kehabisan memori (Out of Memory). 
# Membaginya menjadi porsi kecil isi 32 gambar membuat proses 
# latihan stabil dan optimal.

NUM_WORKERS_DEFAULT = 2 # Jumlah inti prosesor (CPU sub-threads) yang ditugaskan khusus untuk membuka, memotong, dan menyiapkan gambar dari harddisk.

# Mengapa harus seperti itu? Ini adalah teknik Multi-processing. 
# Selagi GPU sedang menghitung dan melatih 32 gambar pertama, 
# 2 workers CPU ini sudah sibuk di latar belakang membuka 32 gambar 
# berikutnya dari harddisk. Langkah ini menghilangkan waktu tunggu (bottleneck) 
# agar GPU tidak menganggur

RANDOM_SEED = 42 # Angka pengunci fungsi acak untuk pengocokan urutan gambar.

# Mengapa harus seperti itu? Menjamin urutan pengacakan data latihan 
# akan selalu konsisten sama setiap kali kode dijalankan ulang, 
# demi keabsahan dokumentasi eksperimen riset ilmiah.





def create_dataset(folder, transform):
    """Membaca folder split menggunakan ImageFolder."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")

    return ImageFolder(
        root=str(folder),
        transform=transform,
    )

# Membaca folder target menggunakan fungsi ImageFolder 
# bawaan PyTorch dan menempelkan aturan transformasinya.

# Mengapa harus seperti itu? ImageFolder sangat cerdas. 
# Fungsi ini otomatis membaca struktur folder 
# (misal folder human dan ai), lalu otomatis menganggap 
# nama folder tersebut sebagai label kelasnya tanpa 
# perlu membuat daftarnya secara manual. Fungsi ini juga dilengkapi 
# proteksi FileNotFoundError untuk menghentikan program dengan pesan
# eror yang jelas jika lupa menaruh folder dataset-nya.


def seed_worker(worker_id):
    """Seed worker agar augmentasi/data loading lebih reproducible."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)

# Mengunci angka acak (seed) pada setiap worker CPU yang aktif membantu memproses data.

# Mengapa harus seperti itu? Jika menggunakan augmentasi acak 
# (seperti memutar gambar secara acak) dan mengaktifkan multi-workers, 
# setiap worker bisa menghasilkan urutan acak yang tidak sinkron atau 
# tidak dapat diulang kembali. Fungsi ini menjamin sifat reproducible 
# tetap terjaga hingga ke level sub-thread terkecil.


def get_dataloaders(
    batch_size=BATCH_SIZE_DEFAULT,
    num_workers=NUM_WORKERS_DEFAULT,
):
    train_dataset = create_dataset(
        SPLIT_DIR / "train",
        preprocessing.get_train_transform(),
    )

    validation_dataset = create_dataset(
        SPLIT_DIR / "validation",
        preprocessing.get_eval_transform(),
    )

    test_in_domain_dataset = create_dataset(
        SPLIT_DIR / "test_in_domain",
        preprocessing.get_eval_transform(),
    )

    test_cross_dataset = create_dataset(
        SPLIT_DIR / "test_cross_generator",
        preprocessing.get_eval_transform(),
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return {
        "train": DataLoader(
            train_dataset,
            shuffle=True,
            **common,
        ),
        "validation": DataLoader(
            validation_dataset,
            shuffle=False,
            **common,
        ),
        "test_in_domain": DataLoader(
            test_in_domain_dataset,
            shuffle=False,
            **common,
        ),
        "test_cross_generator": DataLoader(
            test_cross_dataset,
            shuffle=False,
            **common,
        ),
    }

# Inti utama program yang mengemas 4 jalur dataset
# (train, validation, test in-domain, test cross-generator)
# menjadi objek DataLoader siap pakai.

# Mengapa parameter di dalam common = dict(...) harus seperti itu?
#
# pin_memory=torch.cuda.is_available(): Jika komputer punya GPU,
# gambar yang sudah siap di RAM komputer akan dikunci posisinya agar
# bisa ditransfer ke VRAM GPU dengan kecepatan super cepat.
#
# persistent_workers=...: Menjaga agar 2 pekerja CPU tadi tidak
# mematikan dirinya sendiri setelah selesai memproses satu batch,
# melainkan tetap siaga untuk batch berikutnya. Ini menghemat waktu
# jeda komputasi.
#
# shuffle=True (Hanya untuk Train): Mengacak urutan gambar latihan
# di setiap epoch (putaran latihan).Jika data tidak diacak,
# AI akan menghafal urutan gambar, bukan mempelajari fitur visualnya.
# Sebaliknya, untuk data evaluasi dan tes, shuffle=False karena kita
# ingin menguji performa secara berurutan dan konsisten.


def print_dataset_information():
    loaders = get_dataloaders()

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    for name, loader in loaders.items():
        dataset = loader.dataset

        print(f"\n[{name}]")
        print("Jumlah gambar:", len(dataset))
        print("Class:", dataset.classes)
        print("Mapping:", dataset.class_to_idx)

        counts = {class_name: 0 for class_name in dataset.classes}
        for _, label in dataset.samples:
            counts[dataset.classes[label]] += 1

        print("Distribusi:", counts)

# Membuka seluruh bungkusan data loader, menghitung jumlah file di dalamnya, 
# dan mencetak distribusi datanya ke layar terminal.

# Mengapa harus seperti itu? Sebagai fungsi audit mandiri sebelum pelatihan dimulai. 
# Kita bisa memastikan secara instan lewat teks terminal bahwa angka pemetaan label
# kelas tidak tertukar (misal: human dipetakan ke angka 0, ai dipetakan ke angka 1) 
# dan jumlah distribusinya benar-benar seimbang.

if __name__ == "__main__":
    print_dataset_information()
