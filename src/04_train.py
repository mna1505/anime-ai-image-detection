from pathlib import Path
from importlib import import_module
import csv
import json
import os
import random
import threading
import time

import psutil
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

preprocessing = import_module("02_preprocessing")
dataset_module = import_module("03_dataset")

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODELS = (
    "mobilenetv3",
    "resnet50",
    "efficientnetv2",
    "vit",
)

# Daftar 4 model kecerdasan buatan berbeda yang akan diuji dan saling dibandingkan kekuatannya.

# Mengapa harus seperti itu? Dalam riset, kita tidak boleh berasumsi 
# model mana yang terbaik. Kita menguji model ringan (MobileNet), 
# model standar (ResNet, EfficientNet), dan model berbasis Transformer (ViT) 
# untuk mencari tahu mana yang paling akurat dan efisien mendeteksi anime buatan AI.


BATCH_SIZE = 32 # Pengaturan jumlah suapan data per hitungan (32 gambar)
NUM_EPOCHS = 20 # Batas maksimal AI melakukan putaran belajar ulang pada seluruh dataset (maksimal 20 kali bolak-balik membaca data).
LEARNING_RATE = 1e-4 #Kecepatan atau ukuran langkah AI saat memperbaiki kesalahannya (0.0001)

# Mengapa harus kecil? Jika terlalu besar, AI akan melompati pola penting dan gagal pintar. 
# Nilai 1e-4 adalah standar aman untuk teknik Fine-Tuning (melatih model yang dasarnya sudah pintar).

EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 2 # Pengaturan CPU (2 sub-prosesor) yang dioper ke proses latihan.
RANDOM_SEED = 42 # Angka pengunci fungsi acak.
PRETRAINED = True # Memakai model yang otaknya sudah diisi pengetahuan dasar dari jutaan gambar umum lain.

DEVICE = preprocessing.get_device() # Menentukan apakah latihan berjalan di kartu grafis (CUDA/GPU) atau prosesor biasa (CPU).





# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Membuat eksperimen lebih reproducible.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Kegunaan: Mengunci keacakan komputasi matematika GPU dan CPU.

# Mengapa dilakukan? Menjamin bobot awal AI selalu sama setiap kali 
# kode dijalankan ulang demi keabsahan hasil riset.


# ============================================================
# MEMORY MONITOR
# ============================================================

class PeakMemoryMonitor:
    """Memantau peak resident RAM proses Python selama fase eksperimen."""

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


def reset_gpu_peak_memory():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_vram_mb():
    if DEVICE.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

# Kegunaan: Mengaktifkan asisten latar belakang (threading) untuk mencatat titik 
# tertinggi penggunaan RAM dan memori kartu grafis (VRAM GPU).

# Mengapa dilakukan? Di dunia nyata, keakuratan tinggi tidak ada gunanya jika 
# model tersebut terlalu berat hingga membuat server jebol. Data ini krusial 
# untuk mengukur efisiensi model.





# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(model_name, pretrained=PRETRAINED):
    """
    Membuat 4 arsitektur yang dibandingkan.

    - MobileNetV3-Large
    - ResNet50
    - EfficientNetV2-S
    - ViT-B/16

    Semua diubah menjadi classifier 2 kelas: Human dan AI.
    """

    if model_name == "mobilenetv3":
        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained else None
        )
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)

    elif model_name == "resnet50":
        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif model_name == "efficientnetv2":
        weights = (
            models.EfficientNet_V2_S_Weights.DEFAULT
            if pretrained else None
        )
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)

    elif model_name == "vit":
        weights = (
            models.ViT_B_16_Weights.DEFAULT
            if pretrained else None
        )
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, 2)

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    return model

# Kegunaan: Memanggil arsitektur asli dari library PyTorch, 
# lalu memotong bagian ujung saraf terakhirnya (Classifier Layer) 
# dan menggantinya dengan gerbang Linear baru yang bercabang dua.

# Mengapa dilakukan? Model asli bawaan internet dirancang untuk 
# menebak 1000 benda dunia nyata. Karena proyek hanya mendeteksi 
# 2 hal (Human vs AI), layernya wajib diubah menjadi hanya mendeteksi 2 kelas saja.

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Kegunaan: Menghitung total jumlah saraf digital di dalam model AI tersebut.





# ============================================================
# MODEL COMPLEXITY: FLOPs / MACs
# ============================================================

def calculate_complexity(model_name):
    """
    Mengukur MACs/FLOPs pada input 1 x 3 x 224 x 224.

    Prioritas:
    1. fvcore
    2. thop
    3. None jika keduanya tidak tersedia.

    Nilai ini merupakan estimasi computational complexity, bukan waktu
    eksekusi aktual. Waktu aktual tetap diukur melalui training/inference.
    """

    model = create_model(model_name, pretrained=False).cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)

    # ---------- fvcore ----------
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, dummy)
        flops = float(analysis.total())

        # Secara konvensi umum: FLOPs ~ 2 x MACs.
        macs = flops / 2.0

        del model
        return {
            "flops": flops,
            "macs": macs,
            "complexity_tool": "fvcore",
        }

    except Exception as fvcore_error:
        fvcore_message = str(fvcore_error)

    # ---------- thop fallback ----------
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = 2.0 * float(macs)

        del model
        return {
            "flops": flops,
            "macs": float(macs),
            "complexity_tool": "thop",
        }

    except Exception as thop_error:
        del model
        return {
            "flops": None,
            "macs": None,
            "complexity_tool": "unavailable",
            "complexity_error": {
                "fvcore": fvcore_message,
                "thop": str(thop_error),
            },
        }

# Kegunaan: Menguji model menggunakan gambar bohongan (dummy) 
# untuk menghitung nilai FLOPs/MACs (jumlah kalkulasi matematika 
# yang harus diselesaikan model per satu gambar).

# Mengapa ada sistem fallback? Fungsi ini mencoba modul fvcore dulu, 
# jika tidak ada baru beralih ke thop. Nilai FLOPs memberi tahu kita 
# seberapa kompleks struktur matematika model tersebut secara teori.





# ============================================================
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# Kegunaan: Proses belajar 1 putaran. AI menebak gambar, 
# melihat kunci jawaban, menghitung nilai kesalahan (loss), 
# lalu memperbarui arah sarafnya (optimizer.step()).

# Mengapa menggunakan set_to_none=True? Menghapus jejak memori 
# kalkulasi gradien lama secara total agar memori GPU tetap lega 
# untuk batch selanjutnya.


def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# Kegunaan: Proses ujian tengah semester. AI menguji kemampuannya pada 
# data validation menggunakan fungsi torch.no_grad() (hanya menebak 
# tanpa mengubah struktur otaknya).

# Mengapa dilakukan? Untuk melihat performa asli AI saat menghadapi 
# gambar yang tidak dipakai untuk belajar.




# ============================================================
# SAVE HISTORY
# ============================================================

def save_training_history(history, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "epoch_time_seconds",
        ])

        for row in history:
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                row["train_accuracy"],
                row["validation_loss"],
                row["validation_accuracy"],
                row["epoch_time_seconds"],
            ])

# Kegunaan: Menyimpan catatan perkembangan skor ujian AI per putaran ke dalam file CSV.





# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(model_name):
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 70)

    loaders = dataset_module.get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    train_loader = loaders["train"]
    validation_loader = loaders["validation"]

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = create_model(model_name, pretrained=PRETRAINED).to(DEVICE)

    total_parameters, trainable_parameters = count_parameters(model)
    complexity = calculate_complexity(model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    model_dir = MODEL_DIR / model_name
    result_dir = RESULT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "best_model.pth"

    # --------------------------------------------------------
    # TRAINING METRICS
    # --------------------------------------------------------
    best_val_accuracy = -1.0
    best_epoch = 0
    patience = 0
    history = []

    reset_gpu_peak_memory()
    ram_monitor = PeakMemoryMonitor()
    ram_monitor.start()

    training_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = validate(
            model,
            validation_loader,
            criterion,
        )

        epoch_time = time.perf_counter() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "epoch_time_seconds": epoch_time,
        })

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Time {epoch_time:.2f}s"
        )

        # ====================================================
        # CHECKPOINT SELECTION
        # ====================================================
        # Validation dipakai untuk memilih checkpoint TERBAIK
        # di dalam model yang sama. Test set belum disentuh.
        # Ini bukan berarti memilih model terbaik secara keseluruhan.
        # Semua 4 model tetap akan diuji pada dua test set.
        # ====================================================
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            patience = 0

            torch.save(model.state_dict(), checkpoint_path)
            print("  -> checkpoint validation terbaik disimpan")
        else:
            patience += 1

        if patience >= EARLY_STOPPING_PATIENCE:
            print("  -> early stopping")
            break

    training_time = time.perf_counter() - training_start

    ram_monitor.stop()
    peak_gpu_vram = get_peak_gpu_vram_mb()

    if not checkpoint_path.exists():
        raise RuntimeError("Checkpoint terbaik tidak berhasil dibuat.")

    model_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
    average_epoch_time = training_time / len(history)

    save_training_history(
        history,
        result_dir / "training_history.csv",
    )

    summary = {
        "model": model_name,
        "architecture": {
            "mobilenetv3": "MobileNetV3-Large",
            "resnet50": "ResNet50",
            "efficientnetv2": "EfficientNetV2-S",
            "vit": "ViT-B/16",
        }[model_name],
        "pretrained": PRETRAINED,
        "input_size": [3, 224, 224],
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": NUM_EPOCHS,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60.0,
        "average_epoch_time_seconds": average_epoch_time,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "model_size_mb": model_size_mb,
        "flops": complexity["flops"],
        "macs": complexity["macs"],
        "complexity_tool": complexity["complexity_tool"],
        "peak_gpu_vram_mb_training": peak_gpu_vram,
        "peak_ram_mb_training": ram_monitor.peak_ram_mb,
    }

    if "complexity_error" in complexity:
        summary["complexity_error"] = complexity["complexity_error"]

    with open(
        result_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n[TRAINING EFFICIENCY]")
    print(f"Training time       : {training_time / 60:.2f} min")
    print(f"Average epoch time  : {average_epoch_time:.2f} s")
    print(f"Parameters          : {total_parameters:,}")
    print(f"Model size          : {model_size_mb:.2f} MB")
    print(f"FLOPs               : {complexity['flops']}")
    print(f"MACs                : {complexity['macs']}")
    print(
        "Peak GPU VRAM       : "
        f"{peak_gpu_vram:.2f} MB" if peak_gpu_vram is not None
        else "Peak GPU VRAM       : N/A"
    )
    print(f"Peak RAM            : {ram_monitor.peak_ram_mb:.2f} MB")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return summary

# Kegunaan: Mengatur urutan eksekusi latihan satu model dari awal hingga selesai 
# (panggil data loader, siapkan model, monitor RAM/GPU, jalankan loop putaran, 
# dan ekspor ringkasan JSON).

# Mengapa menyimpan checkpoint best_model.pth berdasarkan val_acc? Kode ini hanya 
# menyimpan otak AI jika nilai akurasi pada lembar validasi meningkat. 
# Jadi, jika di putaran ke-15 AI mulai mengalami penurunan kemampuan, 
# tetap mendapatkan file otak AI terbaiknya yang sempat terekam.


# ============================================================
# MAIN
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_name in MODELS:
        summary = train_model(model_name)
        all_summaries.append(summary)

    with open(
        RESULT_DIR / "training_efficiency_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(all_summaries, file, indent=4)

    # Ringkasan CSV agar mudah dipindahkan ke Excel/Google Sheets.
    with open(
        RESULT_DIR / "training_efficiency_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
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
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in all_summaries:
            writer.writerow({key: row.get(key) for key in fields})

    print("\nSemua model selesai dilatih.")


if __name__ == "__main__":
    main()

# Kegunaan: Pemimpin utama jalannya program. Fungsi ini memanggil fungsi 
# train_model berulang kali secara otomatis untuk ke-4 model secara bergantian.

# Mengapa di akhir disimpan ke JSON dan CSV bersama? Agar seluruh metrik 
# performa ke-4 model tersusun rapi dalam satu tabel terpusat yang siap 
# salin ke lembar dokumen laporan atau skripsi.