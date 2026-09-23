# ============================================================
# EVALUASI MODEL MODIFIED
# ============================================================
# File ini digunakan untuk menguji checkpoint hasil training dari
# 04_train_modified.py. Evaluasi dilakukan setelah proses training
# selesai dan TIDAK melakukan training ulang.
#
# Tujuan utama evaluasi:
# ------------------------------------------------------------
# 1. Mengukur performa klasifikasi pada data IN-DOMAIN.
# 2. Mengukur kemampuan generalisasi pada data CROSS-GENERATOR.
# 3. Menghitung metrik klasifikasi secara lengkap.
# 4. Mengukur kecepatan inference.
# 5. Mengukur penggunaan GPU VRAM dan RAM selama inference.
# 6. Menyimpan hasil dalam JSON dan CSV agar mudah dibandingkan.
# 7. Menggabungkan training history dan training summary yang
#    telah dibuat oleh 04_train_modified.py.
#
# Model yang dievaluasi:
# ------------------------------------------------------------
# MobileNetV3-Large, ResNet50, EfficientNetV2-S, dan ViT-B/16.
#
# Kondisi pengujian:
# ------------------------------------------------------------
# in_domain       : data test dari domain yang digunakan saat training.
# cross_generator : data test dari sumber/generator yang tidak digunakan
#                   dalam training.
#
# Metrik klasifikasi:
# ------------------------------------------------------------
# Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC.
#
# Metrik efisiensi inference:
# ------------------------------------------------------------
# Total inference time, latency, throughput, peak GPU VRAM,
# baseline RAM, peak RAM, dan peak RAM delta.
#
# Aturan label penelitian:
# ------------------------------------------------------------
#     label 0 = Human
#     label 1 = AI
#
# Karena probabilitas kelas 1 digunakan untuk ROC-AUC, nilai probabilitas
# yang disimpan evaluator adalah probabilitas bahwa gambar merupakan AI.
#
# Alur umum:
# ------------------------------------------------------------
# Load dataset -> Load checkpoint -> Warm-up -> Inference ->
# Hitung metrik -> Hitung efisiensi -> Simpan hasil ->
# Rekap training output -> Hitung generalization gap.
# ============================================================

from pathlib import Path
from importlib import import_module
import argparse
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
MODEL_DIR = ROOT_DIR / "models_modified"
RESULT_DIR = ROOT_DIR / "results_modified"

preprocessing = import_module("02_preprocessing_modified")
dataset_module = import_module("03_dataset_modified")

MODELS = (
    "mobilenetv3",
    "resnet50",
    "efficientnetv2",
    "vit",
)

CONDITIONS = (
    "in_domain",
    "cross_generator",
)

BATCH_SIZE = 32
NUM_WORKERS = 2
WARMUP_BATCHES = 3

DEVICE = preprocessing.get_device()


# ============================================================
# MONITORING RAM
# ============================================================
# PeakMemoryMonitor memantau RSS (Resident Set Size) dari proses Python
# selama inference. Monitoring dilakukan secara berkala menggunakan
# psutil.
#
# Nilai yang digunakan dalam laporan:
# - baseline_ram_mb : RAM sebelum pengukuran inference dimulai.
# - peak_ram_mb     : RAM tertinggi yang terpantau selama inference.
# - peak_ram_delta  : peak RAM dikurangi baseline RAM.
#
# Dengan tiga nilai tersebut, penggunaan RAM dapat dibandingkan
# antar-model dan antar-level secara lebih informatif.
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

    @property
    def peak_ram_mb(self):
        return self.peak_rss / (1024 ** 2)


# ============================================================
# MONITORING GPU VRAM
# ============================================================
# Statistik peak memory CUDA di-reset sebelum pengukuran dimulai.
# Tujuannya agar nilai peak VRAM yang dilaporkan berasal dari
# inference yang sedang diuji, bukan sisa penggunaan memori sebelumnya.
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


# ============================================================
# MEMBUAT ARSITEKTUR MODEL
# ============================================================
# Struktur arsitektur dibuat kembali tanpa pretrained weights.
# Bobot yang digunakan nanti berasal dari best_model.pth.
#
# Semua classification head mempunyai 2 output dengan aturan:
#     output 0 = Human
#     output 1 = AI
#
# Struktur model harus sama dengan struktur saat training agar
# state_dict checkpoint dapat dimuat dengan benar.
# ============================================================

def create_model(model_name):
    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(
            weights=None
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)

    elif model_name == "resnet50":
        model = models.resnet50(
            weights=None
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif model_name == "efficientnetv2":
        model = models.efficientnet_v2_s(
            weights=None
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)

    elif model_name == "vit":
        model = models.vit_b_16(
            weights=None
        )
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, 2)

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    return model


# ============================================================
# MEMUAT CHECKPOINT MODEL
# ============================================================
# Checkpoint dicari berdasarkan kombinasi level dan nama model.
# File best_model.pth berisi bobot model terbaik yang dipilih
# selama training.
#
# Setelah bobot dimuat, model dipindahkan ke DEVICE dan menggunakan
# mode eval() agar layer yang memiliki perilaku berbeda saat training
# dan evaluasi bekerja dalam mode evaluasi.
# ============================================================

def load_model(model_name, level):
    model = create_model(model_name)

    checkpoint = (
        MODEL_DIR
        / f"level{level}"
        / model_name
        / "best_model.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint}\n"
            f"Jalankan 04_train_modified.py --level {level} terlebih dahulu."
        )

    state_dict = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    return model.to(DEVICE).eval()


# ============================================================
# FUNGSI EVALUASI MODEL
# ============================================================
# Fungsi ini menjalankan evaluasi untuk satu kombinasi:
#     model + condition + level.
#
# Tahap yang dilakukan:
# 1. Mengambil DataLoader.
# 2. Memilih test_in_domain atau test_cross_generator.
# 3. Memuat checkpoint.
# 4. Melakukan warm-up.
# 5. Mengukur inference sebenarnya.
# 6. Mengumpulkan label, prediksi, dan probabilitas.
# 7. Menghitung metrik klasifikasi.
# 8. Menghitung latency dan throughput.
# 9. Mengukur VRAM dan RAM.
# 10. Mengembalikan hasil sebagai dictionary.
# ============================================================

def evaluate_model(model_name, condition, level):
    loaders = dataset_module.get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=model_name,
        level=level,
    )

    if condition == "in_domain":
        loader = loaders["test_in_domain"]
    elif condition == "cross_generator":
        loader = loaders["test_cross_generator"]
    else:
        raise ValueError(
            "condition harus in_domain atau cross_generator"
        )

    model = load_model(
        model_name,
        level,
    )

    # ========================================================
    # WARM-UP INFERENCE
    # ========================================================
    # Tiga batch awal digunakan sebagai pemanasan dan tidak dimasukkan
    # ke perhitungan waktu inference utama.
    #
    # Warm-up diperlukan karena eksekusi pertama pada GPU dapat memiliki
    # overhead inisialisasi runtime. Dengan membuang batch awal tersebut,
    # pengukuran latency lebih fokus pada kondisi inference yang sudah
    # berjalan stabil.
    #
    # WARMUP_BATCHES = 3 adalah protokol pengukuran yang ditetapkan
    # dalam penelitian ini. Warm-up tidak mengubah bobot model dan
    # tidak digunakan untuk menghitung metrik klasifikasi.
    # ========================================================
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            synchronize()
            _ = model(images)
            synchronize()

            if batch_index + 1 >= WARMUP_BATCHES:
                break

    reset_gpu_peak_memory()

    ram_monitor = PeakMemoryMonitor()

    # Catat RAM sebelum pengukuran inference.
    # Ini mengikuti standar evaluator BASELINE agar RAM absolut
    # dan RAM delta dapat dibandingkan secara konsisten.
    baseline_ram_mb = (
        ram_monitor.process.memory_info().rss
        / (1024 ** 2)
    )

    ram_monitor.start()

    labels_all = []
    predictions_all = []
    probabilities_all = []

    total_forward_seconds = 0.0
    total_images = 0

    # ========================================================
    # REAL INFERENCE MEASUREMENT
    # ========================================================
    # torch.no_grad() digunakan karena evaluasi tidak membutuhkan
    # gradient maupun backpropagation.
    #
    # Ketika menggunakan CUDA, synchronize() dipanggil sebelum dan
    # sesudah forward pass. Hal ini penting karena operasi GPU dapat
    # berjalan asynchronous. Sinkronisasi membuat waktu yang diukur
    # lebih merepresentasikan durasi forward pass yang sebenarnya.
    #
    # Selama loop dikumpulkan:
    # - label asli,
    # - prediksi kelas,
    # - probabilitas kelas AI,
    # - total waktu forward,
    # - jumlah gambar.
    # ========================================================
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                DEVICE,
                non_blocking=True,
            )
            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            synchronize()
            start = time.perf_counter()

            outputs = model(images)

            synchronize()
            elapsed = time.perf_counter() - start

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_forward_seconds += elapsed
            total_images += images.size(0)

            labels_all.extend(
                labels.cpu().tolist()
            )

            predictions_all.extend(
                predictions.cpu().tolist()
            )

            probabilities_all.extend(
                probabilities[:, 1].cpu().tolist()
            )

    ram_monitor.stop()

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================
    # Setelah seluruh data selesai diprediksi, prediksi dibandingkan
    # dengan label sebenarnya.
    #
    # Accuracy  : persentase seluruh prediksi yang benar.
    # Precision : ketepatan ketika model memprediksi kelas AI.
    # Recall    : kemampuan menemukan gambar yang benar-benar AI.
    # F1-Score  : keseimbangan antara precision dan recall.
    # ROC-AUC   : kemampuan membedakan dua kelas berdasarkan skor
    #             probabilitas.
    #
    # Confusion matrix menggunakan urutan label [0, 1]:
    #     [[TN, FP],
    #      [FN, TP]]
    # dengan aturan penelitian:
    #     0 = Human
    #     1 = AI
    # ========================================================

    accuracy = accuracy_score(
        labels_all,
        predictions_all,
    )

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

    try:
        roc_auc = roc_auc_score(
            labels_all,
            probabilities_all,
        )
    except ValueError:
        roc_auc = None

    cm = confusion_matrix(
        labels_all,
        predictions_all,
        labels=[0, 1],
    )

    # ========================================================
    # EFFICIENCY METRICS
    # ========================================================
    # Latency rata-rata per gambar dihitung dengan:
    #
    #     latency = total forward time / jumlah gambar
    #
    # kemudian dikonversi dari detik menjadi milidetik.
    #
    # Throughput dihitung dengan:
    #
    #     throughput = jumlah gambar / total forward time
    #
    # Nilai ini membantu membandingkan kebutuhan waktu inference
    # antar arsitektur dan antar level training.
    # ========================================================

    inference_time_ms = (
        total_forward_seconds
        / max(total_images, 1)
        * 1000.0
    )

    throughput = (
        total_images
        / total_forward_seconds
        if total_forward_seconds > 0
        else 0.0
    )

    # ========================================================
    # HASIL EVALUASI
    # ========================================================
    # Semua metrik disimpan dalam satu dictionary agar dapat digunakan
    # kembali untuk JSON, CSV, dan analisis generalization gap.
    # Urutan dan nama field dibuat konsisten dengan evaluator baseline
    # sehingga hasil setiap level dapat dibandingkan langsung.
    # ========================================================
    # Struktur hasil sengaja disamakan dengan evaluator BASELINE.
    # Dengan demikian CSV baseline, Level 1, Level 2, dan Level 3
    # mempunyai nama kolom yang identik dan dapat dibandingkan
    # langsung di Excel/Pandas tanpa mengubah nama variabel lagi.
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
        "inference_latency_ms_per_image": inference_time_ms,
        "throughput_images_per_second": throughput,
        "peak_gpu_vram_mb_inference": get_peak_gpu_vram_mb(),
        "baseline_ram_mb_inference": baseline_ram_mb,
        "peak_ram_mb_inference": ram_monitor.peak_ram_mb,
        "peak_ram_delta_mb_inference": (
            ram_monitor.peak_ram_mb - baseline_ram_mb
        ),
    }

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result



# ========================================================
# REKAP OUTPUT TRAINING
# ========================================================
# Bagian ini tidak melakukan training dan tidak membuat angka training
# baru. Evaluator hanya membaca file hasil 04_train_modified.py.
#
# File yang direkap:
# - training_summary.json : ringkasan performa dan efisiensi training.
# - training_history.csv  : riwayat loss, accuracy, validation,
#                           waktu epoch, dan learning rate.
#
# Data tersebut kemudian digabungkan menjadi file tingkat level agar
# lebih mudah dianalisis. Jika file sumber belum ada, evaluator hanya
# memberikan peringatan dan tetap melanjutkan evaluasi.
# ========================================================
# TRAINING OUTPUT AGGREGATION
# ========================================================
# Catatan penting:
# training_history.csv dan training_efficiency_summary.csv adalah
# keluaran dari proses TRAINING (04_train_modified.py), bukan hasil
# inferensi (05_evaluate_modified.py).
#
# Namun evaluator ini tetap menyediakan sinkronisasi/rekapitulasi:
# - Jika training_summary.json per model sudah ada, evaluator akan
#   membuat training_efficiency_summary.csv pada folder level.
# - Jika training_history.csv per model sudah ada, evaluator akan
#   membuat training_history.csv gabungan pada folder level.
#
# Tidak ada angka training yang dibuat-buat dari checkpoint .pth.
# Jika file sumber training belum ada, evaluator hanya memberikan
# peringatan dan tetap melanjutkan evaluasi.

TRAINING_EFFICIENCY_FIELDS = [
    "model",
    "architecture",
    "best_epoch",
    "best_validation_accuracy",
    "training_time_seconds",
    "average_epoch_time_seconds",
    "total_parameters",
    "trainable_parameters",
    "model_size_mb",
    "flops",
    "macs",
    "peak_gpu_vram_mb_training",
    "peak_ram_mb_training",
]


def aggregate_training_outputs(level):
    # Menggabungkan output training dari seluruh model pada level ini.
    """Menggabungkan output training per model menjadi file level."""
    level_dir = RESULT_DIR / f"level{level}"
    level_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # TRAINING EFFICIENCY SUMMARY
    # --------------------------------------------------------
    summaries = []

    for model_name in MODELS:
        summary_path = (
            level_dir / model_name / "training_summary.json"
        )

        if not summary_path.exists():
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as file:
                summary = json.load(file)
            summaries.append(summary)
        except Exception as error:
            print(
                f"[WARNING] Gagal membaca {summary_path}: {error}"
            )

    if summaries:
        with open(
            level_dir / "training_efficiency_summary.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summaries, file, indent=4)

        with open(
            level_dir / "training_efficiency_summary.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=TRAINING_EFFICIENCY_FIELDS,
            )
            writer.writeheader()
            for summary in summaries:
                writer.writerow({
                    key: summary.get(key)
                    for key in TRAINING_EFFICIENCY_FIELDS
                })

        print(
            f"Training efficiency summary: "
            f"{level_dir / 'training_efficiency_summary.csv'}"
        )
    else:
        print(
            "[INFO] training_summary.json per model belum ditemukan; "
            "training_efficiency_summary.csv tidak dibuat."
        )

    # --------------------------------------------------------
    # TRAINING HISTORY GABUNGAN
    # --------------------------------------------------------
    history_rows = []

    for model_name in MODELS:
        history_path = (
            level_dir / model_name / "training_history.csv"
        )

        if not history_path.exists():
            continue

        try:
            with open(
                history_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                reader = csv.DictReader(file)
                for row in reader:
                    row["model"] = model_name
                    row["level"] = level
                    history_rows.append(row)
        except Exception as error:
            print(
                f"[WARNING] Gagal membaca {history_path}: {error}"
            )

    if history_rows:
        history_fields = [
            "model",
            "level",
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "epoch_time_seconds",
            "learning_rate",
        ]

        with open(
            level_dir / "training_history.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history_fields,
            )
            writer.writeheader()
            writer.writerows(history_rows)

        print(
            f"Training history: "
            f"{level_dir / 'training_history.csv'}"
        )
    else:
        print(
            "[INFO] training_history.csv per model belum ditemukan; "
            "training history gabungan tidak dibuat."
        )

# ========================================================
# MAIN PROGRAM
# ========================================================
# Command utama dapat mengevaluasi satu model atau seluruh model.
# Level wajib diberikan melalui command line.
#
# Contoh seluruh model:
#     python 05_evaluate_modified.py --level 1
#
# Contoh satu model:
#     python 05_evaluate_modified.py --level 1 --model resnet50
#
# Jika --model tidak diberikan, seluruh model pada MODELS akan diuji
# pada dua kondisi: in_domain dan cross_generator.
# ========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate modified models."
    )

    # Level menentukan checkpoint dan pipeline dataset yang digunakan.
    # Pilihan 1-4 disediakan untuk mengevaluasi seluruh level modified.
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Level modifikasi training yang dievaluasi (1, 2, atau 3).",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS),
        default=None,
    )

    args = parser.parse_args()

    # Rekap output training yang SUDAH dihasilkan oleh 04_train_modified.py.
    # Proses ini hanya membaca dan menggabungkan hasil training yang
    # sudah tersedia; tidak ada retraining pada tahap evaluasi.
    # Ini tidak melakukan retraining dan tidak membuat nilai training palsu.
    aggregate_training_outputs(args.level)

    models_to_evaluate = (
        [args.model]
        if args.model
        else list(MODELS)
    )

    all_results = []

    for model_name in models_to_evaluate:
        for condition in CONDITIONS:
            print(
                "\n"
                + "=" * 70
            )
            print(
                f"EVALUATION | LEVEL {args.level} | "
                f"{model_name.upper()} | {condition}"
            )
            print(
                "=" * 70
            )

            result = evaluate_model(
                model_name,
                condition,
                args.level,
            )

            all_results.append(result)

            print(
                f"Accuracy  : {result['accuracy']:.4f}"
            )
            print(
                f"Precision : {result['precision']:.4f}"
            )
            print(
                f"Recall    : {result['recall']:.4f}"
            )
            print(
                f"F1        : {result['f1_score']:.4f}"
            )
            print(
                f"ROC-AUC   : "
                f"{result['roc_auc']:.4f}"
                if result["roc_auc"] is not None
                else "ROC-AUC   : N/A"
            )
            print(
                f"Latency   : "
                f"{result['inference_latency_ms_per_image']:.4f} ms/image"
            )
            print(
                f"Throughput: "
                f"{result['throughput_images_per_second']:.2f} image/s"
            )
            print(
                f"Peak RAM  : "
                f"{result['peak_ram_mb_inference']:.2f} MB"
            )

    # ========================================================
    # OUTPUT HASIL EVALUASI
    # ========================================================
    # Hasil disimpan dalam:
    # 1. JSON detail untuk menyimpan struktur hasil secara lengkap.
    # 2. CSV utama untuk memudahkan analisis dan perbandingan.
    #
    # Schema CSV dibuat eksplisit agar nama dan urutan kolom konsisten
    # dengan evaluator baseline.
    # ========================================================
    # CSV utama sengaja dibuat langsung di results_modified dengan
    # nama yang konsisten:
    #   evaluation_summary_1.csv
    #   evaluation_summary_2.csv
    #   evaluation_summary_3.csv
    #
    # Kolom CSV mengikuti evaluator BASELINE 100%.
    # Informasi "level" tetap tersimpan pada JSON dan folder detail,
    # tetapi TIDAK dimasukkan ke CSV utama agar schema identik.
    output_dir = RESULT_DIR
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_dir = (
        RESULT_DIR / f"level{args.level}"
    )
    detail_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        detail_dir
        / "evaluation_summary.json"
    )

    csv_path = (
        output_dir
        / f"evaluation_summary_{args.level}.csv"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=4,
        )

    if all_results:
        # ====================================================
        # SCHEMA CSV = BASELINE
        # ====================================================
        # Jangan menggunakan list(all_results[0].keys()) di sini.
        # Field order ditetapkan secara eksplisit supaya Level 1-3
        # selalu identik dengan evaluation_summary_baseline.csv.
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

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()
            for row in all_results:
                writer.writerow({
                    key: row.get(key)
                    for key in fields
                })

    # ========================================================
    # GENERALIZATION GAP
    # ========================================================
    # Generalization gap membandingkan performa pada in-domain dengan
    # cross-generator.
    #
    # Accuracy gap:
    #     in-domain accuracy - cross-generator accuracy
    #
    # F1 gap:
    #     in-domain F1 - cross-generator F1
    #
    # Nilai gap membantu menunjukkan seberapa besar performa berubah
    # ketika model berpindah dari domain yang dikenal ke domain/generator
    # yang tidak digunakan saat training.
    # ========================================================
    gap_rows = []

    for model_name in models_to_evaluate:
        in_domain = next(
            (
                row
                for row in all_results
                if row["model"] == model_name
                and row["condition"] == "in_domain"
            ),
            None,
        )

        cross = next(
            (
                row
                for row in all_results
                if row["model"] == model_name
                and row["condition"] == "cross_generator"
            ),
            None,
        )

        if in_domain and cross:
            gap_rows.append({
                "model": model_name,
                "level": args.level,
                "in_domain_accuracy": in_domain["accuracy"],
                "cross_generator_accuracy": cross["accuracy"],
                "generalization_gap": (
                    in_domain["accuracy"]
                    - cross["accuracy"]
                ),
                "in_domain_f1": in_domain["f1_score"],
                "cross_generator_f1": cross["f1_score"],
                "f1_gap": (
                    in_domain["f1_score"]
                    - cross["f1_score"]
                ),
            })

    with open(
        detail_dir / "generalization_gap.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
            "model",
            "level",
            "in_domain_accuracy",
            "cross_generator_accuracy",
            "generalization_gap",
            "in_domain_f1",
            "cross_generator_f1",
            "f1_gap",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(gap_rows)

    print("\nSemua evaluasi selesai.")
    print(f"CSV utama : {csv_path}")
    print(f"JSON detail: {json_path}")


# ============================================================
# RUN PROGRAM
# ============================================================
# main() hanya dijalankan ketika file ini dieksekusi secara langsung.
# ============================================================
if __name__ == "__main__":
    main()
