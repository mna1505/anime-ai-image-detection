from pathlib import Path
from importlib import import_module
import argparse
import csv
import json
import os
import random
import threading
import time

import psutil
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models_modified"
RESULT_DIR = ROOT_DIR / "results_modified"

preprocessing = import_module("02_preprocessing_modified")
dataset_module = import_module("03_dataset_modified")

# ============================================================
# MODIFIED TRAINING CONFIGURATION
# ============================================================
#
# File ini merupakan tahap TRAINING untuk model modified.
#
# Berbeda dengan baseline, training modified menggunakan beberapa
# strategi tambahan yang dibagi secara bertingkat:
#
# LEVEL 1
# ------------------------------------------------------------
# Fokus: memperkuat proses training dasar dan membuat model lebih
# tahan terhadap variasi data.
#
# Komponen yang digunakan:
# - AdamW sebagai optimizer.
# - Weight decay = 1e-4.
# - Cosine Annealing Learning Rate.
# - Gradient clipping dengan max_norm = 1.0.
# - Early stopping dengan patience = 6.
# - Augmentasi visual Level 1 diatur oleh
#   02_preprocessing_modified.py.
#
# LEVEL 2
# ------------------------------------------------------------
# Fokus: regularisasi tambahan dan kestabilan proses training.
#
# Level 2 mempertahankan komponen Level 1, kemudian menambahkan:
# - MixUp dengan alpha = 0.2.
# - Label smoothing = 0.10.
# - Linear learning-rate warm-up selama 3 epoch.
# - Setelah warm-up, scheduler dilanjutkan dengan
#   Cosine Annealing.
#
# LEVEL 3
# ------------------------------------------------------------
# Fokus: discriminative fine-tuning.
#
# Level 3 mempertahankan strategi Level 2, kemudian membagi
# parameter model menjadi dua kelompok:
# - Backbone  -> learning rate 1e-5.
# - Head      -> learning rate 1e-4.
#
# Tujuannya adalah membuat perubahan pada backbone lebih kecil,
# sedangkan classification head dapat beradaptasi lebih cepat
# terhadap tugas klasifikasi Human vs AI.
#
# LEVEL 4
# ------------------------------------------------------------
# Level 4 mempertahankan kebijakan learning rate Level 3 dan
# menambahkan SAM (Sharpness-Aware Minimization).
#
# SAM melakukan dua kali forward/backward pada satu batch:
# - langkah pertama mencari arah parameter yang paling sensitif;
# - langkah kedua melakukan update pada kondisi yang lebih stabil.
#
# ------------------------------------------------------------
# ATURAN PEMILIHAN CHECKPOINT
# ------------------------------------------------------------
# best_model.pth selalu dipilih berdasarkan VALIDATION ACCURACY.
#
# Data test:
# - test_in_domain
# - test_cross_generator
#
# tidak digunakan selama training dan tidak digunakan untuk
# memilih checkpoint. Kedua test set hanya digunakan pada tahap
# evaluasi setelah training selesai.
# ============================================================

MODELS = (
    "mobilenetv3",
    "resnet50",
    "efficientnetv2",
    "vit",
)

# ============================================================
# PARAMETER TRAINING
# ============================================================
#
# Nilai berikut dibuat tetap agar seluruh arsitektur mendapat
# kondisi eksperimen yang sama.
# ============================================================

BATCH_SIZE = 32
# Jumlah gambar yang diproses dalam satu batch.
#
# Batch size 32 berarti model menerima 32 gambar sebelum
# melakukan pembaruan parameter.

NUM_EPOCHS = 25
# Batas maksimum jumlah epoch.
#
# Satu epoch berarti seluruh data training telah digunakan
# satu kali untuk proses pembelajaran.

NUM_WORKERS = 2
# Jumlah worker untuk membantu DataLoader membaca data.

RANDOM_SEED = 42
# Angka pengunci proses acak agar eksperimen dapat diulang
# dengan kondisi yang sama.

PRETRAINED = True
# Menggunakan pretrained weights ImageNet sebagai titik awal.
# Model tidak memulai pembelajaran dari bobot acak sepenuhnya.

EARLY_STOPPING_PATIENCE = 6
# Training dihentikan jika validation accuracy tidak membaik
# selama 6 epoch berturut-turut.

# ------------------------------------------------------------
# LEVEL 1 DAN LEVEL 2: LEARNING RATE DASAR
# ------------------------------------------------------------

BASE_LR = 1e-4
# Learning rate dasar.
#
# Learning rate menentukan seberapa besar perubahan parameter
# model pada setiap proses update.

# ------------------------------------------------------------
# LEVEL 2: MIXUP DAN LABEL SMOOTHING
# ------------------------------------------------------------

MIXUP_ALPHA = 0.2
# Parameter alpha pada distribusi Beta yang digunakan untuk
# menentukan seberapa kuat dua gambar dicampurkan.

LABEL_SMOOTHING = 0.10
# Tingkat label smoothing sebesar 10%.
#
# Target tidak diperlakukan terlalu absolut sehingga model
# tidak didorong menjadi terlalu percaya diri.

# ------------------------------------------------------------
# LEVEL 3: DISCRIMINATIVE LEARNING RATE
# ------------------------------------------------------------

BACKBONE_LR = 1e-5
# Learning rate kecil untuk backbone.
# Tujuannya menjaga pengetahuan umum dari pretrained model.

HEAD_LR = 1e-4
# Learning rate lebih besar untuk classification head.
# Head perlu beradaptasi dengan cepat terhadap tugas Human vs AI.

# ------------------------------------------------------------
# REGULARISASI DAN LEARNING RATE SCHEDULER
# ------------------------------------------------------------

WEIGHT_DECAY = 1e-4
# Penalti kecil pada bobot model untuk membantu mengurangi
# kecenderungan overfitting.

WARMUP_EPOCHS = 3
# Jumlah epoch untuk fase warm-up pada Level 2-4.

WARMUP_START_FACTOR = 0.1
# Learning rate saat awal warm-up dimulai dari 10% dari
# learning rate target.

DEVICE = preprocessing.get_device()


# ============================================================
# REPRODUCIBILITY
# ============================================================
# Fungsi ini mengunci sumber keacakan utama yang digunakan
# selama eksperimen.
#
# Mengapa penting?
# ------------------------------------------------------------
# Training Deep Learning menggunakan proses acak, misalnya
# inisialisasi, pengacakan batch, dan MixUp. Tanpa seed yang
# sama, dua training dapat menghasilkan hasil yang berbeda.
#
# Seed membantu membuat eksperimen lebih mudah diulang dan
# dibandingkan.
# ============================================================

def set_seed(seed=RANDOM_SEED):
    # Mengunci generator random Python.
    random.seed(seed)

    # Mengunci generator random PyTorch pada CPU.
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # Mengunci generator random untuk seluruh GPU CUDA.
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        # Memilih operasi CUDA yang lebih deterministik.
        # Benchmark dimatikan agar algoritma yang dipilih tidak
        # berubah-ubah berdasarkan pengukuran performa.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# MEMORY MONITOR
# ============================================================
# Kelas ini digunakan untuk mencatat penggunaan RAM tertinggi
# selama training.
#
# RSS (Resident Set Size) menunjukkan jumlah RAM sistem yang
# sedang digunakan oleh proses Python.
#
# Monitoring dilakukan pada thread terpisah sehingga proses
# training tetap berjalan.
# ============================================================

class PeakMemoryMonitor:
    """Memantau peak resident RAM proses Python selama eksperimen."""

    def __init__(self, interval=0.02):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss = self.process.memory_info().rss
        self.running = False
        self.thread = None

    def _run(self):
        # Membaca penggunaan RAM secara berkala selama training.
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
# GPU MEMORY MONITOR
# ============================================================
# Sebelum setiap eksperimen, statistik peak VRAM CUDA
# di-reset agar pengukuran tidak tercampur dengan proses
# sebelumnya.
# ============================================================

def reset_gpu_peak_memory():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_vram_mb():
    if DEVICE.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ============================================================
# SAM OPTIMIZER — KHUSUS LEVEL 4
# ============================================================
# SAM (Sharpness-Aware Minimization) adalah optimizer yang
# berusaha mencari parameter model yang tidak terlalu sensitif
# terhadap perubahan kecil pada bobot.
#
# Implementasi di bawah mengikuti pola dua langkah:
#
# FIRST STEP
# ------------------------------------------------------------
# Gradien digunakan untuk menggeser parameter sementara ke arah
# gangguan yang dianggap paling sensitif.
#
# SECOND STEP
# ------------------------------------------------------------
# Parameter dikembalikan ke posisi semula, kemudian optimizer
# melakukan update sebenarnya berdasarkan gradien dari kondisi
# tersebut.
#
# SAM hanya aktif ketika level == 4.
# ============================================================

class SAM(torch.optim.Optimizer):
    """Implementasi Sharpness-Aware Minimization (SAM) Optimizer."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                # Geser bobot sementara ke arah yang paling sensitif.
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # Kembalikan bobot ke posisi sebelum perturbasi SAM.
                p.sub_(self.state[p]["e_w"])

        # Setelah kembali ke posisi awal, base optimizer melakukan
        # pembaruan parameter yang sebenarnya.
        self.base_optimizer.step()
        if zero_grad: self.zero_grad(set_to_none=True)

    def _grad_norm(self):
        # Deteksi device secara dinamis dari parameter pertama yang memiliki gradien
        shared_device = next(p.device for group in self.param_groups for p in group["params"] if p.grad is not None)
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ============================================================
# MODEL FACTORY
# ============================================================
# Fungsi ini membuat model yang digunakan dalam penelitian.
#
# Empat arsitektur:
# - MobileNetV3-Large
# - ResNet50
# - EfficientNetV2-S
# - ViT-B/16
#
# Semua model menggunakan pretrained weights ImageNet jika
# PRETRAINED=True.
#
# Classification head bawaan kemudian diganti menjadi output
# 2 kelas:
#   0 = Human
#   1 = AI
#
# Mapping label harus konsisten dengan dataset dan file evaluasi.
# ============================================================

def create_model(model_name, pretrained=PRETRAINED):
    """Membuat salah satu dari empat arsitektur dengan classifier 2 kelas."""

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


# ============================================================
# MODEL COMPLEXITY
# ============================================================
# Selain performa klasifikasi, penelitian juga memperhatikan
# efisiensi model.
#
# Jumlah parameter menunjukkan ukuran jumlah bobot yang dimiliki
# model. Semakin banyak parameter, biasanya model membutuhkan
# sumber daya lebih besar.
# ============================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def calculate_complexity(model_name):
    """
    Mengukur kompleksitas komputasi model menggunakan input
    dummy berukuran 1 x 3 x 224 x 224.

    FLOPs:
        perkiraan jumlah operasi floating-point.

    MACs:
        jumlah operasi Multiply-Accumulate.

    fvcore digunakan terlebih dahulu. Jika tidak tersedia atau
    gagal menghitung model, program mencoba menggunakan thop.
    Jika keduanya gagal, nilai kompleksitas dicatat sebagai None.
    """
    model = create_model(model_name, pretrained=False).cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)

    # ---------- fvcore ----------
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(model, dummy)
        flops = float(analysis.total())
        macs = flops / 2.0
        del model
        return {"flops": flops, "macs": macs, "complexity_tool": "fvcore"}
    except Exception as fvcore_error:
        fvcore_message = str(fvcore_error)

    # ---------- thop fallback ----------
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = 2.0 * float(macs)
        del model
        return {"flops": flops, "macs": float(macs), "complexity_tool": "thop"}
    except Exception as thop_error:
        del model
        return {
            "flops": None,
            "macs": None,
            "complexity_tool": "unavailable",
            "complexity_error": {"fvcore": fvcore_message, "thop": str(thop_error)}
        }


# ============================================================
# LEVEL 3 — DISCRIMINATIVE FINE-TUNING
# ============================================================
# Pada Level 3, seluruh parameter model tidak lagi menerima
# learning rate yang sama.
#
# Backbone:
#   1e-5
#
# Classification Head:
#   1e-4
#
# Struktur backbone berbeda untuk setiap arsitektur, sehingga
# parameter yang masuk ke kelompok backbone disesuaikan dengan
# struktur masing-masing model.
#
# Kebijakan learning rate tetap sama agar perlakuan eksperimen
# antararsitektur dapat dibandingkan.
# ============================================================

def configure_level3(model, model_name):
    """
    Level 3 menggunakan kebijakan discriminative learning rate yang SAMA
    untuk seluruh arsitektur:

        Backbone             -> BACKBONE_LR = 1e-5
        Classification Head  -> HEAD_LR = 1e-4

    Struktur parameter backbone berbeda mengikuti arsitektur masing-masing,
    tetapi kebijakan learning rate dibuat identik agar perbandingan
    antararsitektur tetap terkontrol.
    """
    if model_name == "mobilenetv3":
        backbone_parameters = model.features.parameters()
        head_parameters = model.classifier.parameters()
    elif model_name == "resnet50":
        backbone_parameters = (
            list(model.conv1.parameters()) + list(model.bn1.parameters())
            + list(model.layer1.parameters()) + list(model.layer2.parameters())
            + list(model.layer3.parameters()) + list(model.layer4.parameters())
        )
        head_parameters = model.fc.parameters()
    elif model_name == "efficientnetv2":
        backbone_parameters = model.features.parameters()
        head_parameters = model.classifier.parameters()
    elif model_name == "vit":
        backbone_parameters = (
            list(model.conv_proj.parameters()) + list(model.encoder.parameters())
        )
        head_parameters = model.heads.parameters()
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    return [
        {"params": backbone_parameters, "lr": BACKBONE_LR},
        {"params": head_parameters, "lr": HEAD_LR},
    ]


def build_optimizer_and_scheduler(model, model_name, level):
    """
    Menentukan optimizer dan learning-rate scheduler sesuai level.

    Level 1
    ------------------------------------------------------------
    AdamW + weight decay + Cosine Annealing.

    Level 2
    ------------------------------------------------------------
    AdamW + weight decay + Linear Warm-up + Cosine Annealing.

    Level 3
    ------------------------------------------------------------
    Discriminative Learning Rate:
        backbone = 1e-5
        head     = 1e-4

    Level 4
    ------------------------------------------------------------
    Discriminative Learning Rate + SAM.
    Scheduler warm-up dan cosine tetap digunakan.

    Scheduler hanya mengatur learning rate. Scheduler tidak
    mengubah data gambar dan tidak mengubah label.
    """
    if level == 1:
        optimizer = AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif level == 2:
        optimizer = AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        warmup = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    elif level == 3:
        optimizer = AdamW(configure_level3(model, model_name), weight_decay=WEIGHT_DECAY)
        warmup = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    elif level == 4:
        # Level 4 mempertahankan pembagian learning rate Level 3,
        # kemudian menambahkan SAM sebagai optimizer.
        optimizer = SAM(configure_level3(model, model_name), AdamW, rho=0.05, weight_decay=WEIGHT_DECAY)
        warmup = LinearLR(optimizer.base_optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(optimizer.base_optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS))
        scheduler = SequentialLR(optimizer.base_optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    else:
        raise ValueError("level harus 1, 2, 3, atau 4")
    return optimizer, scheduler


# ============================================================
# MIXUP — LEVEL 2, LEVEL 3, DAN LEVEL 4
# ============================================================
# MixUp membuat sampel training baru dengan mencampurkan
# dua gambar dari batch.
#
# Secara sederhana:
#
# mixed_image = λ * image_A + (1-λ) * image_B
#
# Nilai λ diambil dari Beta Distribution dengan alpha = 0.2.
# Label A dan label B juga disimpan karena loss kemudian
# dihitung berdasarkan proporsi campuran tersebut.
#
# MixUp hanya digunakan pada training. Validation dan test tetap
# menggunakan gambar asli tanpa pencampuran.
# ============================================================

def mixup_data(images, labels, alpha=MIXUP_ALPHA):
    """Membuat batch campuran untuk Level 2, Level 3, dan Level 4."""

    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]

    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """
    Menghitung loss untuk hasil MixUp.

    Loss dihitung dua kali:
    - loss terhadap label A
    - loss terhadap label B

    Keduanya kemudian digabungkan menggunakan bobot lambda.
    """
    return (
        lam * criterion(outputs, labels_a)
        + (1.0 - lam) * criterion(outputs, labels_b)
    )


# ============================================================
# TRAINING DAN VALIDATION
# ============================================================
# Bagian ini merupakan inti proses pembelajaran.
#
# TRAIN:
# - model berada pada mode train().
# - data training diproses.
# - loss dihitung.
# - gradien dihitung melalui backward().
# - parameter model diperbarui oleh optimizer.
#
# VALIDATION:
# - model berada pada mode eval().
# - parameter tidak diperbarui.
# - torch.no_grad() digunakan untuk menghemat memori dan
#   menghindari pembuatan graph gradien.
#
# Validation digunakan untuk memilih checkpoint terbaik.
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, level):
    # Mode train mengaktifkan perilaku khusus training seperti
    # Dropout dan Batch Normalization sesuai kebutuhan model.
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # Memindahkan batch dari CPU ke device training.
        # non_blocking=True dapat membantu transfer data ketika
        # DataLoader menggunakan pinned memory.
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # --------------------------------------------------------
        # JALUR KHUSUS LEVEL 4: SAM OPTIMIZATION LOOP
        # --------------------------------------------------------
        if level == 4:
            # Pipa pencampuran data MixUp tetap diaktifkan (mengikuti skema Level 2 & 3)
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

            # ------------------------------------------------
            # FIRST PASS
            # ------------------------------------------------
            # Model menghitung gradien pada data MixUp untuk
            # menentukan arah perturbasi SAM.
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # ------------------------------------------------
            # SECOND PASS
            # ------------------------------------------------
            # Model dihitung kembali setelah perturbasi. Gradien
            # dari kondisi ini digunakan untuk update sebenarnya.
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.second_step(zero_grad=True)

            predictions = outputs.argmax(dim=1)
            batch_correct = (predictions == labels).sum().item()

        # --------------------------------------------------------
        # JALUR LEVEL 1, 2, DAN 3
        # --------------------------------------------------------
        # Level 1:
        #   gambar asli -> loss standar
        #
        # Level 2-3:
        #   gambar -> MixUp -> loss MixUp
        # --------------------------------------------------------
        else:
            optimizer.zero_grad(set_to_none=True)

            if level >= 2:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                predictions = outputs.argmax(dim=1)
                batch_correct = (predictions == labels).sum().item()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)
                batch_correct = (predictions == labels).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += batch_correct
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion):
    """
    Mengukur loss dan accuracy pada data validation.

    Tidak ada pembaruan parameter di sini.
    Validation hanya digunakan untuk memantau kemampuan model
    pada data yang tidak digunakan untuk update bobot.
    """
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
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================
# Setiap epoch dicatat ke CSV agar perkembangan training dapat
# dianalisis kembali.
#
# Kolom yang disimpan:
# - epoch
# - train_loss
# - train_accuracy
# - validation_loss
# - validation_accuracy
# - epoch_time_seconds
# - learning_rate
#
# Learning rate juga dicatat karena setiap level memiliki
# scheduler yang berbeda.
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
            "learning_rate",
        ])

        for row in history:
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                row["train_accuracy"],
                row["validation_loss"],
                row["validation_accuracy"],
                row["epoch_time_seconds"],
                row["learning_rate"],
            ])


def get_current_lr(optimizer):
    """
    Mengambil learning rate terbesar dari seluruh parameter group.

    Pada Level 3 dan 4 terdapat dua kelompok learning rate
    (backbone dan head), sehingga fungsi ini mengambil nilai
    terbesar untuk ditampilkan pada history.
    """
    return max(
        group["lr"]
        for group in optimizer.param_groups
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================
# Fungsi ini mengatur seluruh siklus training untuk satu kombinasi:
#
#   satu arsitektur + satu level
#
# Urutan proses:
# 1. Mengunci seed.
# 2. Membuat DataLoader.
# 3. Membuat model pretrained.
# 4. Menghitung parameter dan kompleksitas.
# 5. Menentukan loss.
# 6. Menentukan optimizer dan scheduler.
# 7. Menjalankan training epoch demi epoch.
# 8. Menguji validation setiap epoch.
# 9. Menyimpan checkpoint dengan validation accuracy terbaik.
# 10. Menghentikan training jika early stopping terpenuhi.
# 11. Mengukur waktu dan penggunaan memori.
# 12. Menyimpan history dan summary.
# ============================================================

def train_model(model_name, level):
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 75)
    print(f"MODIFIED TRAINING: {model_name.upper()} | LEVEL {level}")
    print("=" * 75)

    loaders = dataset_module.get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=model_name,
        level=level,
    )

    train_loader = loaders["train"]
    validation_loader = loaders["validation"]

    model = create_model(
        model_name,
        pretrained=PRETRAINED,
    ).to(DEVICE)

    total_parameters, trainable_parameters = count_parameters(model)

    complexity = calculate_complexity(model_name)

    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------
    # Level 1 menggunakan Cross Entropy Loss standar.
    #
    # Level 2-4 menggunakan Cross Entropy Loss dengan
    # label smoothing sebesar 0.10.
    if level == 1:
        criterion = nn.CrossEntropyLoss()

    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=LABEL_SMOOTHING
        )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        model_name,
        level,
    )

    model_dir = MODEL_DIR / f"level{level}" / model_name
    result_dir = RESULT_DIR / f"level{level}" / model_name

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "best_model.pth"

    best_val_accuracy = -1.0
    best_epoch = 0
    patience = 0
    history = []

    reset_gpu_peak_memory()

    ram_monitor = PeakMemoryMonitor()
    ram_monitor.start()

    training_start = time.perf_counter()

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    # Setiap epoch terdiri dari:
    #   TRAIN -> VALIDATION -> pencatatan -> checkpoint ->
    #   scheduler -> pemeriksaan early stopping.
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            level,
        )

        val_loss, val_acc = validate(
            model,
            validation_loader,
            criterion,
        )

        epoch_time = time.perf_counter() - epoch_start
        current_lr = get_current_lr(optimizer)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "epoch_time_seconds": epoch_time,
            "learning_rate": current_lr,
        })

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"LR {current_lr:.2e} | "
            f"Time {epoch_time:.2f}s"
        )

        # ====================================================
        # CHECKPOINT SELECTION
        # ====================================================
        # Checkpoint dipilih berdasarkan validation accuracy
        # tertinggi selama training.
        #
        # Test set tidak disentuh pada tahap ini. Dengan demikian,
        # pemilihan checkpoint tidak dipengaruhi oleh hasil
        # cross-generator maupun test in-domain.
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            patience = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print("  -> checkpoint validation terbaik disimpan")

        else:
            patience += 1

        # Memperbarui learning rate sesuai scheduler pada level
        # yang sedang digunakan.
        scheduler.step()

        # Jika validation accuracy tidak membaik dalam jumlah
        # epoch sesuai patience, training dihentikan lebih awal.
        if patience >= EARLY_STOPPING_PATIENCE:
            print("  -> early stopping")
            break

    training_time = time.perf_counter() - training_start

    ram_monitor.stop()
    peak_gpu_vram = get_peak_gpu_vram_mb()

    if not checkpoint_path.exists():
        raise RuntimeError("Checkpoint terbaik tidak berhasil dibuat.")

    average_epoch_time = training_time / len(history)

    save_training_history(
        history,
        result_dir / "training_history.csv",
    )

    # ========================================================
    # TRAINING SUMMARY
    # ========================================================
    # Ringkasan berikut menyimpan informasi penting mengenai
    # konfigurasi, performa validation, waktu training,
    # kompleksitas model, dan penggunaan resource.
    #
    # File ini nantinya digunakan oleh tahap evaluasi dan
    # analisis statistik.
    summary = {
        "model": model_name,
        "level": level,
        "architecture": {
            "mobilenetv3": "MobileNetV3-Large",
            "resnet50": "ResNet50",
            "efficientnetv2": "EfficientNetV2-S",
            "vit": "ViT-B/16",
        }[model_name],
        "pretrained": PRETRAINED,
        "input_size": [3, 224, 224],
        "batch_size": BATCH_SIZE,
        "max_epochs": NUM_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60.0,
        "average_epoch_time_seconds": average_epoch_time,
        "total_parameters": total_parameters,
        "flops": complexity["flops"],
        "macs": complexity["macs"],
        "complexity_tool": complexity["complexity_tool"],
        "mixup": level >= 2,
        "label_smoothing": (
            LABEL_SMOOTHING if level >= 2 else 0.0
        ),
        "sam_optimizer": level == 4,
        "model_checkpoint": str(checkpoint_path),
        "peak_gpu_vram_mb_training": peak_gpu_vram,
        "peak_ram_mb_training": ram_monitor.peak_ram_mb,
    }

    with open(
        result_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return summary


# ============================================================
# MAIN / COMMAND LINE
# ============================================================
# Program dapat dijalankan untuk:
#
# 1. Melatih satu level untuk semua model:
#       python 04_train_modified.py --level 1
#
# 2. Melatih satu model pada level tertentu:
#       python 04_train_modified.py --level 2 --model resnet50
#
# Argumen --level menentukan strategi training.
# Argumen --model membatasi arsitektur yang dilatih.
# Jika --model tidak diberikan, seluruh empat model dilatih.
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Modified training untuk menguji peningkatan "
            "generalisasi dan robustness cross-generator."
        )
    )

    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Level modifikasi: 1, 2, 3, 4",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS),
        default=None,
        help="Latih satu model saja. Jika kosong, semua model dilatih.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    models_to_train = (
        [args.model]
        if args.model
        else list(MODELS)
    )

    all_summaries = []

    for model_name in models_to_train:
        summary = train_model(
            model_name,
            args.level,
        )
        all_summaries.append(summary)

    # ========================================================
    # AGGREGASI HASIL SATU LEVEL
    # ========================================================
    # Setelah seluruh model pada level tersebut selesai dilatih,
    # summary tiap model digabungkan menjadi satu JSON dan satu
    # CSV agar lebih mudah dianalisis.
    summary_dir = RESULT_DIR / f"level{args.level}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(
        summary_dir / "training_efficiency_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_summaries,
            file,
            indent=4,
        )

    with open(
        summary_dir / "training_efficiency_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
            "model",
            "level",
            "architecture",
            "best_epoch",
            "best_validation_accuracy",
            "training_time_seconds",
            "average_epoch_time_seconds",
            "total_parameters",
            "flops",
            "macs",
            "mixup",
            "label_smoothing",
            "peak_gpu_vram_mb_training",
            "peak_ram_mb_training",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in all_summaries:
            writer.writerow({
                key: row.get(key)
                for key in fields
            })

    print("\nSemua model selesai dilatih.")


if __name__ == "__main__":
    main()
