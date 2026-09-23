"""
===============================================================================
06_gui_predict.py
===============================================================================
Antarmuka GUI Web (Gradio) untuk melakukan prediksi gambar anime.

Tujuan:
-------------------------------------------------------------------------------
File ini digunakan sebagai tahap DEPLOYMENT sederhana dari model yang sudah
dilatih. Pengguna dapat memilih arsitektur model, mengunggah satu gambar anime,
kemudian sistem akan menampilkan hasil klasifikasi:

    0 = Human-made
    1 = AI-generated

Alur kerja aplikasi:
-------------------------------------------------------------------------------
1. Menentukan lokasi folder project dan folder model.
2. Memuat fungsi preprocessing yang sama dengan tahap evaluasi.
3. Membuat arsitektur model sesuai pilihan pengguna.
4. Memuat bobot model terbaik dari file best_model.pth.
5. Mengubah gambar input menjadi format yang sesuai dengan model.
6. Melakukan inferensi tanpa menghitung gradient.
7. Mengambil kelas dengan probabilitas tertinggi.
8. Mengubah hasil angka menjadi label Human-made atau AI-generated.
9. Menampilkan label dan confidence score melalui GUI Gradio.

Catatan penting:
-------------------------------------------------------------------------------
Model TIDAK dilatih ulang oleh file ini. File ini hanya menggunakan checkpoint
hasil training yang sudah tersedia di folder models/.

Preprocessing untuk prediksi menggunakan get_eval_transform(), sehingga gambar
yang diuji diproses dengan pipeline evaluasi yang sama dan tidak mendapatkan
augmentasi acak.

Teknologi:
-------------------------------------------------------------------------------
- PyTorch       : membangun model dan melakukan inferensi.
- Torchvision   : menyediakan arsitektur pretrained yang digunakan.
- Gradio        : membuat antarmuka web sederhana.
- PIL           : membaca dan mengubah gambar.
- CUDA          : digunakan jika GPU tersedia.

===============================================================================
"""

from pathlib import Path
from importlib import import_module
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models

# ============================================================
# INISIALISASI JALUR FOLDER
# ============================================================
# ROOT_DIR digunakan untuk mencari folder utama project.
# MODEL_DIR menunjuk ke folder yang menyimpan checkpoint model.
#
# Struktur yang diharapkan:
#
# project/
# ├── models/
# │   ├── mobilenetv3/
# │   │   └── best_model.pth
# │   ├── resnet50/
# │   │   └── best_model.pth
# │   ├── efficientnetv2/
# │   │   └── best_model.pth
# │   └── vit/
# │       └── best_model.pth
# └── 06_gui_predict.py
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

# ============================================================
# IMPOR PREPROCESSING
# ============================================================
# Menggunakan modul preprocessing yang telah dibuat sebelumnya.
# Dengan cara ini, preprocessing pada GUI tetap konsisten dengan
# preprocessing yang digunakan ketika evaluasi.
# ============================================================
preprocessing = import_module("02_preprocessing")
DEVICE = preprocessing.get_device()

# ============================================================
# MEMBUAT ARSITEKTUR MODEL
# ============================================================
# Fungsi ini membuat struktur model sesuai pilihan pengguna.
#
# weights=None digunakan karena bobot akan diambil dari file
# best_model.pth, bukan mengunduh pretrained weights baru.
#
# Output terakhir setiap model diubah menjadi 2 kelas:
#   0 = Human-made
#   1 = AI-generated
# ============================================================
def create_model(model_name):
    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "efficientnetv2":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif model_name == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    return model

# ============================================================
# MEMUAT CHECKPOINT MODEL
# ============================================================
# Fungsi ini:
# 1. Membuat arsitektur model.
# 2. Mencari best_model.pth.
# 3. Membaca state_dict dari checkpoint.
# 4. Memasukkan bobot ke arsitektur.
# 5. Memindahkan model ke DEVICE.
# 6. Mengubah model ke mode evaluasi dengan eval().
#
# Jika checkpoint tidak ditemukan, fungsi mengembalikan None.
# ============================================================
def load_model(model_name):
    model = create_model(model_name)
    checkpoint = MODEL_DIR / model_name / "best_model.pth"
    if not checkpoint.exists():
        return None
    state_dict = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()

# ============================================================
# FUNGSI PREDIKSI GUI
# ============================================================
# Fungsi ini dipanggil ketika tombol "Mulai Cek Gambar" ditekan.
#
# Alurnya:
#   gambar pengguna
#       ↓
#   preprocessing evaluasi
#       ↓
#   Tensor + batch dimension
#       ↓
#   model
#       ↓
#   logits
#       ↓
#   softmax probability
#       ↓
#   kelas dengan nilai tertinggi
#       ↓
#   label + confidence
# ============================================================
def predict_gui(model_name, input_image):
    if input_image is None:
        return "Silakan unggah gambar terlebih dahulu.", ""

    # --------------------------------------------------------
    # 1. MEMUAT MODEL
    # --------------------------------------------------------
    # Arsitektur dipilih berdasarkan dropdown GUI.
    # Checkpoint best_model.pth kemudian dimuat.
    # --------------------------------------------------------
    model = load_model(model_name)
    if model is None:
        return f"File 'best_model.pth' untuk {model_name} tidak ditemukan di folder models/. Selesaikan training terlebih dahulu.", ""

    transform = preprocessing.get_eval_transform()
    
    # --------------------------------------------------------
    # 2. PRAPEMROSESAN GAMBAR
    # --------------------------------------------------------
    # Gambar dari GUI diubah menjadi PIL RGB agar formatnya
    # konsisten. Setelah itu digunakan preprocessing evaluasi.
    #
    # unsqueeze(0) menambahkan dimensi batch:
    #   (3, 224, 224) -> (1, 3, 224, 224)
    # --------------------------------------------------------
    image = Image.fromarray(input_image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction = int(outputs.argmax(dim=1).item())

    label = "AI-generated" if prediction == 0 else "Human-made"
    confidence = float(probabilities[prediction].item() * 100.0)

    # --------------------------------------------------------
    # 4. PEMBERSIHAN MEMORI GPU
    # --------------------------------------------------------
    # Model tidak lagi dibutuhkan setelah satu prediksi selesai.
    # Model dihapus dan cache CUDA dibersihkan jika GPU digunakan.
    # Hal ini membantu mengurangi penggunaan memori ketika GUI
    # digunakan berkali-kali.
    # --------------------------------------------------------
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return label, f"{confidence:.2f}%"

# ============================================================
# DESAIN ANTARMUKA WEB
# ============================================================
# Gradio digunakan untuk membuat antarmuka sederhana tanpa harus
# membuat HTML/CSS/JavaScript secara manual.
#
# Komponen utama:
# - Dropdown  : memilih model.
# - Image     : mengunggah gambar anime.
# - Button    : menjalankan prediksi.
# - Textbox   : menampilkan hasil dan confidence.
# ============================================================
with gr.Blocks(title="Anime AI Detector") as demo:
    gr.Markdown("# 🎨 Anime AI vs Human-Made Detector")
    gr.Markdown("Aplikasi riset berbasis GUI untuk mendeteksi keaslian gambar anime menggunakan akselerasi GPU NVIDIA RTX.")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["mobilenetv3", "resnet50", "efficientnetv2", "vit"],
                value="resnet50",
                label="Pilih Model AI yang Digunakan"
            )
            image_input = gr.Image(label="Unggah atau Tarik Gambar Anime ke Sini")
            btn = gr.Button("Mulai Cek Gambar", variant="primary")
            
        with gr.Column():
            output_label = gr.Textbox(label="Hasil Prediksi Detektor")
            output_confidence = gr.Textbox(label="Tingkat Keyakinan Akurasi Model")

    btn.click(
        fn=predict_gui,
        inputs=[model_dropdown, image_input],
        outputs=[output_label, output_confidence]
    )

if __name__ == "__main__":
    # ========================================================
    # MENJALANKAN APLIKASI
    # ========================================================
    # demo.launch(inbrowser=True) menjalankan server Gradio
    # lokal dan meminta browser membuka halaman GUI otomatis.
    # ========================================================
    demo.launch(inbrowser=True)
