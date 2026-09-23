from pathlib import Path
from importlib import import_module
import csv
import json
import os
import threading
import time

import psutil
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

# Artinya: Jalur otomatis ke folder utama proyek,
# folder tempat menyimpan file otak AI (models), 
# dan folder penyimpanan hasil laporan (results).

# Mengapa harus seperti itu? Agar program bisa membaca 
# otak AI hasil latihan secara otomatis tanpa risiko salah jalur folder, 
# meskipun kode ini dijalankan di komputer yang berbeda.

preprocessing = import_module("02_preprocessing")
dataset_module = import_module("03_dataset")

# Artinya: Menghubungkan skrip ini ke Kode Tahap 2 (pemrosesan gambar) 
# dan Kode Tahap 3 (penyuap data/data loader).

MODELS = (
    "mobilenetv3",
    "resnet50",
    "efficientnetv2",
    "vit",
) # Daftar 4 arsitektur model AI yang akan diuji kekuatannya.

CONDITIONS = (
    "in_domain",
    "cross_generator",
) # Dua jenis skenario ujian yang wajib dilewati model.

# Mengapa harus dua? in_domain menguji AI dengan gambar dari generator
# yang sudah ia kenali saat latihan. cross_generator menguji AI dengan 
# generator baru (NovelAI3) yang belum pernah ia lihat sama sekali. 
# Ini dilakukan untuk membuktikan apakah AI benar-benar pintar atau hanya menghafal.

BATCH_SIZE = 32
NUM_WORKERS = 2

# Ukuran pengelompokan gambar (32 buah sekaligus) 
# dan jumlah asisten CPU (2 pekerja) saat proses membaca gambar.

WARMUP_BATCHES = 3

# Aturan untuk melakukan pemanasan sebanyak 3 kelompok gambar di awal sebelum penilaian dimulai.

DEVICE = preprocessing.get_device()

# Detektor otomatis apakah program berjalan menggunakan kartu grafis (GPU/CUDA) atau prosesor biasa (CPU).





# ============================================================
# RAM MONITOR
# ============================================================

class PeakMemoryMonitor:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss = self.process.memory_info().rss
        self.running = False
        self.thread = None

    def _run(self):
        while self.running:
            try:
                rss = self.process.memory_info().rss
                self.peak_rss = max(self.peak_rss, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

# Kegunaan: Kelas asisten latar belakang (threading) yang bertugas mencatat titik paling boros 
# (puncak tertinggi) penggunaan RAM komputer selama proses deteksi gambar berlangsung.

# Mengapa harus menggunakan threading (_run, start, stop)?Komputer memproses gambar sangat cepat. 
# Jika kita hanya mengecek RAM di awal dan di akhir, kita akan melewatkan momen ketika RAM 
# mendadak melonjak tinggi di tengah-tengah proses komputasi. Fungsi _run mengecek RAM setiap 
# 0.02 detik secara paralel agar lonjakan sekecil apa pun tetap terekam.

    @property
    def peak_ram_mb(self):
        return self.peak_rss / (1024 ** 2)

# Kegunaan: Mengubah satuan data memori komputer dari bentuk bita (bytes) 
# menjadi bentuk Megabytes (MB) agar mudah dibaca manusia.





# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(model_name):
    """Membuat arsitektur yang sama persis dengan yang digunakan saat training."""

    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            2,
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif model_name == "efficientnetv2":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            2,
        )

    elif model_name == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            2,
        )

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    return model

# Kegunaan: Mengambil kerangka kosong dari fungsi create_model, 
# lalu memasukkan file "otak pintar" (best_model.pth) hasil latihan ke dalam kerangka tersebut.

# Mengapa dilakukan? Arsitektur dasar model di internet dibuat untuk menebak 1000 objek. 
# Karena tugas proyek hanya menebak 2 pilihan (Human atau AI), 
# strukturnya wajib disamakan persis seperti saat latihan.


def load_model(model_name):
    model = create_model(model_name)
    checkpoint = MODEL_DIR / model_name / "best_model.pth"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint}. "
            "Jalankan 04_train.py terlebih dahulu."
        )

    state_dict = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()

# Kegunaan: Mengambil kerangka kosong dari fungsi create_model, 
# lalu memasukkan file "otak pintar" (best_model.pth) hasil latihan ke dalam kerangka tersebut.

# Mengapa ada fungsi .eval()?Ini sangat krusial. Perintah .eval() memberi tahu model: 
# "Kamu sekarang sedang ujian, matikan fungsi acak (seperti dropout atau augmentasi) 
# dan fokus berikan jawaban terbaikmu!".





# ============================================================
# DEVICE MEMORY
# ============================================================

def reset_gpu_peak_memory():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_vram_mb():
    if DEVICE.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

# Kegunaan: Berfungsi untuk membersihkan sampah memori kartu grafis, 
# mengunci sinkronisasi waktu komputasi, dan mencatat puncak tertinggi 
# penggunaan memori kartu grafis (VRAM GPU).

# Mengapa harus di-synchronize? GPU bekerja secara asinkronus 
# (menebak gambar dan menghitung waktu berjalan terpisah). 
# Fungsi synchronize() memaksa kode Python menunggu sampai GPU benar-benar selesai menghitung, 
# sehingga pencatatan durasi waktu deteksi menjadi sangat akurat.





# ============================================================
# EVALUATION
# Fungsi ini adalah inti dari seluruh lembar pengujian. 
# Di dalamnya terdapat beberapa sub-tahap penting:
# ============================================================

def evaluate_model(model_name, condition):
    loaders = dataset_module.get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    if condition == "in_domain":
        loader = loaders["test_in_domain"]
    elif condition == "cross_generator":
        loader = loaders["test_cross_generator"]
    else:
        raise ValueError("condition harus in_domain atau cross_generator")

    model = load_model(model_name)

    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------
    # Warm-up dilakukan sebelum pencatatan latency agar startup CUDA,
    # kernel compilation, dan overhead pertama tidak mencemari hasil.
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            synchronize()
            _ = model(images)
            synchronize()

            if batch_index + 1 >= WARMUP_BATCHES:
                break

    # --------------------------------------------------------
    # RESET METRICS SETELAH WARM-UP
    # --------------------------------------------------------
    reset_gpu_peak_memory()

    ram_monitor = PeakMemoryMonitor()
    baseline_ram_mb = (
        ram_monitor.process.memory_info().rss / (1024 ** 2)
    )
    ram_monitor.start()

    labels_all = []
    predictions_all = []
    probabilities_all = []

    total_forward_seconds = 0.0
    total_images = 0

    # Mengapa dilakukan? Saat pertama kali model AI dijalankan di GPU, 
    # komputer membutuhkan waktu tambahan untuk memuat sistem internal 
    # (kernel compilation overhead). Jika waktu batch pertama ini langsung dihitung, 
    # hasil catatan kecepatan (latency) model akan terlihat sangat lambat secara tidak adil.
    # Tiga batch awal sengaja dibuang (dipakai untuk pemanasan) agar 
    # penilaian kecepatan berikutnya murni objektif.





    # --------------------------------------------------------
    # REAL INFERENCE MEASUREMENT
    # --------------------------------------------------------
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            synchronize()
            start = time.perf_counter()

            outputs = model(images)

            synchronize()
            elapsed = time.perf_counter() - start

            total_forward_seconds += elapsed
            total_images += images.size(0)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = outputs.argmax(dim=1)

            labels_all.extend(labels.cpu().numpy().tolist())
            predictions_all.extend(predictions.cpu().numpy().tolist())
            probabilities_all.extend(probabilities.cpu().numpy().tolist())

    ram_monitor.stop()

    # Proses di mana AI menebak seluruh gambar tanpa melihat kunci jawaban (torch.no_grad()). 
    # Skor tebakan disimpan dalam bentuk probabilitas dan nilai mutlak.





    # --------------------------------------------------------
    # CLASSIFICATION METRICS
    # --------------------------------------------------------
    cm = confusion_matrix(
        labels_all,
        predictions_all,
        labels=[0, 1],
    )

    accuracy = accuracy_score(labels_all, predictions_all)
    precision = precision_score(
        labels_all,
        predictions_all,
        zero_division=0,
    )
    recall = recall_score(
        labels_all,
        predictions_all,
        zero_division=0,
    )
    f1 = f1_score(
        labels_all,
        predictions_all,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        labels_all,
        probabilities_all,
    )

    # Menghitung nilai statistik keakuratan: Accuracy (skor ketepatan total), 
    # Precision (keabsahan tebakan AI), Recall (kemampuan menjaring semua gambar AI), 
    # F1-score (keseimbangan precision & recall), serta Confusion Matrix 
    # (tabel detail untuk melihat berapa gambar Human yang salah ditebak sebagai AI, atau sebaliknya).





    # --------------------------------------------------------
    # EFFICIENCY METRICS
    # --------------------------------------------------------
    inference_latency_ms_per_image = (
        total_forward_seconds / total_images
    ) * 1000.0

    throughput_images_per_second = (
        total_images / total_forward_seconds
        if total_forward_seconds > 0
        else 0.0
    )

    peak_gpu_vram = get_peak_gpu_vram_mb()
    peak_ram = ram_monitor.peak_ram_mb

    result = {
        "model": model_name,
        "condition": condition,
        "number_of_images": total_images,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "total_inference_time_seconds": total_forward_seconds,
        "inference_latency_ms_per_image": inference_latency_ms_per_image,
        "throughput_images_per_second": throughput_images_per_second,
        "peak_gpu_vram_mb_inference": peak_gpu_vram,
        "baseline_ram_mb_inference": baseline_ram_mb,
        "peak_ram_mb_inference": peak_ram,
        "peak_ram_delta_mb_inference": peak_ram - baseline_ram_mb,
    }

    output_dir = RESULT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / f"metrics_{condition}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=4)

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} - {condition.upper()}")
    print("=" * 70)
    print(f"Images       : {total_images}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print(f"Latency      : {inference_latency_ms_per_image:.4f} ms/image")
    print(f"Throughput   : {throughput_images_per_second:.2f} image/s")

    if peak_gpu_vram is None:
        print("Peak VRAM    : N/A (CPU)")
    else:
        print(f"Peak VRAM    : {peak_gpu_vram:.2f} MB")

    print(f"Peak RAM     : {peak_ram:.2f} MB")
    print(f"RAM delta    : {peak_ram - baseline_ram_mb:.2f} MB")
    print("Confusion matrix [Human, AI]:")
    print(cm)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result

    # Menghitung kecepatan deteksi per satu gambar (latency ms/image) 
    # dan produktivitas jumlah gambar per detik (throughput). Seluruh hasil 
    # disimpan otomatis menjadi file .json rapi di dalam folder hasil.





# ============================================================
# MAIN
# ============================================================

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Semua model diuji pada DUA kondisi.
    # Tidak ada pemilihan model sebelum seluruh 8 hasil diperoleh.
    for model_name in MODELS:
        for condition in CONDITIONS:
            result = evaluate_model(model_name, condition)
            all_results.append(result)

    with open(
        RESULT_DIR / "all_evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(all_results, file, indent=4)

    # Ringkasan 8 hasil agar langsung bisa dipakai untuk tabel skripsi/paper.
    with open(
        RESULT_DIR / "evaluation_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
            "model",
            "condition",
            "number_of_images",
            "confusion_matrix",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "total_inference_time_seconds",
            "inference_latency_ms_per_image",
            "throughput_images_per_second",
            "peak_gpu_vram_mb_inference",
            "baseline_ram_mb_inference",
            "peak_ram_mb_inference",
            "peak_ram_delta_mb_inference",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in all_results:
            writer.writerow({key: row.get(key) for key in fields})

    print("\n" + "=" * 70)
    print("SEMUA 8 HASIL PENGUJIAN SELESAI")
    print("4 model x 2 kondisi = 8 hasil evaluasi")
    print("=" * 70)


if __name__ == "__main__":
    main()

# Kegunaan: Sutradara utama program. Fungsi ini melakukan perulangan otomatis: 
# mengambil model pertama, mengujinya di skenario 1, mengujinya di skenario 2, 
# lalu lanjut ke model berikutnya sampai total 8 hasil pengujian terkumpul lengkap.

# Mengapa hasil akhirnya diekspor ke evaluation_summary.csv?Menyulap data digital 
# menjadi format tabel siap pakai. Hasil akhir berupa file CSV ini bisa langsung 
# salin-tempel (copy-paste) ke dalam tabel Excel untuk kebutuhan pembuatan grafik laporan, 
# paper ilmiah, atau dokumen skripsi tanpa perlu mengetik manual satu per satu.
