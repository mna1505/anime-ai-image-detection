"""
===============================================================================
06_gui_predict_modified.py
===============================================================================
Antarmuka GUI Web (Gradio) untuk melakukan prediksi menggunakan model hasil
training modified.

Tujuan:
-------------------------------------------------------------------------------
File ini merupakan tahap DEPLOYMENT dari model yang telah melalui strategi
training modified. Pengguna dapat memilih arsitektur model, mengunggah gambar
anime, lalu sistem menampilkan:

    - hasil klasifikasi,
    - confidence score,
    - waktu inferensi,
    - informasi model dan performa cross-generator.

Aturan label yang digunakan:
-------------------------------------------------------------------------------
    0 = Human-made
    1 = AI-generated

Alur kerja aplikasi:
-------------------------------------------------------------------------------
1. Menentukan lokasi project dan folder checkpoint.
2. Mencoba menggunakan preprocessing modified.
3. Menentukan checkpoint terbaik berdasarkan prioritas level eksperimen.
4. Jika checkpoint level terbaik tidak tersedia, sistem melakukan fallback
   ke level berikutnya sampai menemukan checkpoint yang tersedia.
5. Membuat arsitektur model sesuai pilihan pengguna.
6. Memuat bobot dari file best_model.pth.
7. Model dimuat secara lazy/on-demand, yaitu hanya ketika dibutuhkan.
8. Gambar diproses menggunakan transform evaluasi.
9. Model melakukan inferensi tanpa gradient.
10. Jika CUDA tersedia, inferensi menggunakan AMP Autocast.
11. Sistem menghitung waktu inferensi.
12. Hasil prediksi diterjemahkan menjadi Human-made atau AI-generated.
13. Informasi model dan performa ditampilkan melalui GUI Gradio.

Strategi efisiensi deployment:
-------------------------------------------------------------------------------
- Lazy Loading:
  Model tidak semuanya dimuat ke RAM/VRAM sejak aplikasi dimulai.
  Model hanya dimuat ketika pengguna memilihnya.

- Memory Cleanup:
  Model sebelumnya dihapus ketika pengguna berpindah model.
  garbage collection dan CUDA cache juga dibersihkan.

- inference_mode():
  Digunakan karena proses deployment hanya melakukan prediksi dan tidak
  membutuhkan gradient.

- AMP Autocast:
  Pada GPU CUDA, inferensi menggunakan autocast untuk membantu mengurangi
  beban komputasi/memori.

Catatan penting:
-------------------------------------------------------------------------------
Model TIDAK dilatih ulang oleh file ini. Semua bobot berasal dari checkpoint
hasil training yang sudah tersedia.

Preprocessing gambar untuk prediksi menggunakan get_eval_transform(), bukan
transformasi training. Dengan demikian, gambar pengguna tidak diberi
augmentasi acak ketika diprediksi.

===============================================================================
"""


import gc
from importlib import import_module
import os
from pathlib import Path
import time
from PIL import Image

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models

# ============================================================
# INISIALISASI PATH DAN LINGKUNGAN
# ============================================================
# Bagian ini menentukan lokasi project dan folder checkpoint.
#
# Sistem terlebih dahulu mencari models_modified. Jika tidak ada,
# sistem mencoba lokasi models_modified pada folder parent.
# Folder models digunakan sebagai checkpoint baseline.
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[0]
if not (ROOT_DIR / "models_modified").exists() and not (ROOT_DIR / "models").exists():
  ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_MODIFIED_DIR = ROOT_DIR / "results_modified" / "models_modified"
if not MODEL_MODIFIED_DIR.exists():
  MODEL_MODIFIED_DIR = ROOT_DIR / "models_modified"

MODEL_BASELINE_DIR = ROOT_DIR / "models"

# ============================================================
# IMPOR PREPROCESSING
# ============================================================
# Program mencoba menggunakan 02_preprocessing_modified.py.
# Jika modul modified tidak ditemukan, program menggunakan
# 02_preprocessing.py sebagai fallback.
#
# Dengan fallback ini, GUI tetap dapat dijalankan selama modul
# preprocessing baseline tersedia.
# ============================================================
try:
  preprocessing = import_module("02_preprocessing_modified")
except ImportError:
  preprocessing = import_module("02_preprocessing")

DEVICE = preprocessing.get_device()

# ============================================================#
# PEMETAAN CHECKPOINT TERBAIK DENGAN FALLBACK AUTOMATIS
# ============================================================#
BEST_MODEL_CONFIGS = {
    "mobilenetv3": {
        "name": "MobileNetV3-Large",
        "optimal_level": 4,
        "search_priority": [4, 3, 2, 1, 0],  # 0 = baseline
        "best_acc_cross_gen": "78.70% (Level 4 SAM)",
        "params": "~4.2M Params",
    },
    "resnet50": {
        "name": "ResNet50",
        "optimal_level": 3,
        "search_priority": [3, 4, 2, 1, 0],
        "best_acc_cross_gen": "76.80% (Level 3 Discrim-LR)",
        "params": "~23.5M Params",
    },
    "efficientnetv2": {
        "name": "EfficientNetV2-S",
        "optimal_level": 3,
        "search_priority": [3, 4, 2, 1, 0],
        "best_acc_cross_gen": "73.20% (Level 3 Discrim-LR)",
        "params": "~21.5M Params",
    },
    "vit": {
        "name": "ViT-B/16 (Vision Transformer)",
        "optimal_level": 3,
        "search_priority": [3, 4, 2, 1, 0],
        "best_acc_cross_gen": "70.60% (Level 3 Discrim-LR)",
        "params": "~86.0M Params",
    },
}

CURRENT_LOADED_MODEL = None
CURRENT_MODEL_KEY = None


# ============================================================
# MODEL FACTORY
# ============================================================
# Membuat arsitektur model sesuai pilihan pengguna.
#
# weights=None digunakan karena bobot final akan dimuat dari
# checkpoint hasil training.
#
# Semua model memiliki 2 output kelas:
#   0 = Human-made
#   1 = AI-generated
# ============================================================
def create_model_architecture(model_key):
  """Membangun kerangka arsitektur modelPyTorch biner (2 kelas output)."""
  if model_key == "mobilenetv3":
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
  elif model_key == "resnet50":
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
  elif model_key == "efficientnetv2":
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
  elif model_key == "vit":
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
  else:
    raise ValueError(f"Model key tidak dikenal: {model_key}")
  return model


# ============================================================
# MENCARI CHECKPOINT TERBAIK
# ============================================================
# Fungsi ini memeriksa checkpoint berdasarkan search_priority.
#
# Jika file best_model.pth ditemukan, jalur file dan nama level
# dikembalikan.
#
# Jika seluruh checkpoint tidak ditemukan, fungsi mengembalikan
# None sehingga GUI dapat menampilkan pesan error.
# ============================================================
def resolve_best_checkpoint(model_key):
  """Mencari jalur checkpoint .pth terbaik berdasarkan prioritas level hasil eksperimen."""
  config = BEST_MODEL_CONFIGS[model_key]

  for level in config["search_priority"]:
    if level == 0:
      checkpoint_path = MODEL_BASELINE_DIR / model_key / "best_model.pth"
      level_name = "Baseline"
    else:
      checkpoint_path = (
          MODEL_MODIFIED_DIR / f"level{level}" / model_key / "best_model.pth"
      )
      level_name = f"Level {level}"

    if checkpoint_path.exists():
      return checkpoint_path, level_name

  return None, None


# ============================================================
# LAZY LOADING MODEL
# ============================================================
# Model hanya dimuat ketika benar-benar diperlukan.
#
# Jika model yang sama sudah aktif, model tersebut digunakan
# kembali sehingga tidak perlu membaca checkpoint dari disk lagi.
#
# Jika pengguna berpindah model:
#   model lama dihapus
#       ↓
#   Python garbage collector dijalankan
#       ↓
#   CUDA cache dibersihkan
#       ↓
#   checkpoint model baru dimuat
#
# Tujuannya menghindari semua model berada di RAM/VRAM sekaligus.
# ============================================================
def load_optimized_model(model_key):
  """Memuat model secara 'Lazy-Loading' dan membebaskan model lama dari VRAM/RAM."""
  global CURRENT_LOADED_MODEL, CURRENT_MODEL_KEY

  if CURRENT_MODEL_KEY == model_key and CURRENT_LOADED_MODEL is not None:
    return CURRENT_LOADED_MODEL, "Model sudah aktif di memori."

  # 1. Meringankan Server: Bersihkan VRAM/RAM dari model sebelumnya
  if CURRENT_LOADED_MODEL is not None:
    del CURRENT_LOADED_MODEL
    CURRENT_LOADED_MODEL = None
    CURRENT_MODEL_KEY = None
    gc.collect()
    if DEVICE.type == "cuda":
      torch.cuda.empty_cache()

  checkpoint_path, level_name = resolve_best_checkpoint(model_key)
  if checkpoint_path is None:
    return (
        None,
        f"Checkpoint .pth untuk '{model_key}' tidak ditemukan di folder"
        " models/.",
    )

  # 2. Muat Arsitektur dan Bobot
  model = create_model_architecture(model_key)
  state_dict = torch.load(
      checkpoint_path, map_location=DEVICE, weights_only=True
  )
  model.load_state_dict(state_dict)
  model = model.to(DEVICE).eval()

  # Simpan state aktif
  CURRENT_LOADED_MODEL = model
  CURRENT_MODEL_KEY = model_key

  status_msg = f"Model {BEST_MODEL_CONFIGS[model_key]['name']} ({level_name}) berhasil dimuat."
  return model, status_msg


# ============================================================
# FUNGSI PREDIKSI DEPLOYMENT
# ============================================================
# Fungsi ini menerima:
#   - model_key
#   - gambar dari pengguna
#
# Kemudian menghasilkan:
#   - label prediksi
#   - confidence
#   - latency
#   - informasi model
#
# Inferensi menggunakan inference_mode() karena tidak diperlukan
# gradient untuk proses prediksi.
#
# Jika CUDA tersedia, autocast digunakan untuk membantu efisiensi
# penggunaan memori dan komputasi saat inferensi.
# ============================================================
def predict_image(model_key, input_image):
  """Melakukan inferensi cepat menggunakan `torch.inference_mode()` dan FP16 Autocast."""
  if input_image is None:
    return "Silakan unggah gambar terlebih dahulu.", "0.00%", "N/A", "N/A"

  # 1. Lazy Loading Model
  model, status_msg = load_optimized_model(model_key)
  if model is None:
    return status_msg, "0.00%", "Error", "Error"

  # 2. Prapemrosesan Gambar
  transform = preprocessing.get_eval_transform()
  pil_image = Image.fromarray(input_image).convert("RGB")
  tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

  # 3. Inferensi Teroptimasi (Inference Mode + AMP Autocast)
  start_time = time.perf_counter()

  with torch.inference_mode():
    if DEVICE.type == "cuda":
      with torch.amp.autocast("cuda"):
        outputs = model(tensor)
    else:
      outputs = model(tensor)

    probabilities = torch.softmax(outputs, dim=1)[0]
    prediction = int(outputs.argmax(dim=1).item())

  elapsed_ms = (time.perf_counter() - start_time) * 1000.0

  # --------------------------------------------------------
    # 4. INTERPRETASI LABEL
    # --------------------------------------------------------
    # Aturan label penelitian:
    #   0 = Human-made
    #   1 = AI-generated
    #
    # Nilai confidence diambil dari probabilitas softmax pada
    # kelas yang dipilih oleh model.
    # --------------------------------------------------------
  label = "AI-generated" if prediction == 0 else "Human-made"
  confidence = float(probabilities[prediction].item() * 100.0)

  cfg = BEST_MODEL_CONFIGS[model_key]
  info_str = f"{cfg['name']} ({cfg['params']})"

  return (
      label,
      f"{confidence:.2f}%",
      f"{elapsed_ms:.2f} ms",
      f"{info_str} — Top Cross-Gen Acc: {cfg['best_acc_cross_gen']}",
  )


# ============================================================
# DESAIN ANTARMUKA WEB (GRADIO)
# ============================================================
# GUI terdiri dari:
# - Dropdown model
# - Upload gambar
# - Tombol prediksi
# - Hasil klasifikasi
# - Confidence
# - Latency
# - Informasi model dan performa
# ============================================================
custom_css = """
.container { max-width: 900px; margin: auto; }
.title-box { text-align: center; margin-bottom: 20px; }
.status-badge { font-weight: bold; color: #2e7d32; }
"""

with gr.Blocks(title="Anime AI Illustration Detector", css=custom_css) as demo:
  gr.Markdown(
      """
        # 🎨 Anime AI vs Human-Made Illustration Detector
        ### Portal Deteksi Forensik Seni Anime Berbasis Deep Learning & Vision Transformer
        ---
        """
  )

  with gr.Row():
    with gr.Column(scale=1):
      model_dropdown = gr.Dropdown(
          choices=[
              ("MobileNetV3-Large (Rekomendasi Utama - Paling Cepat & Akurat)", "mobilenetv3"),
              ("ResNet50 (Deep CNN - Level 3 Fine-Tuning)", "resnet50"),
              ("EfficientNetV2-S (Efficient CNN - Level 3)", "efficientnetv2"),
              ("ViT-B/16 (Vision Transformer - Level 3)", "vit"),
          ],
          value="mobilenetv3",
          label="Pilih Arsitektur Model AI",
      )

      image_input = gr.Image(
          label="Unggah Gambar Ilustrasi Anime", type="numpy"
      )

      btn = gr.Button("🔍 Deteksi Gambar Sekarang", variant="primary")

      gr.Markdown(
          """
            *💡 **Strategi Optimasi Server:** Web menggunakan teknik **Lazy Loading** dan **AMP Autocast**. Model hanya dimuat saat dibutuhkan dan VRAM/RAM dibersihkan secara otomatis demi respon cepat.*
            """
      )

    with gr.Column(scale=1):
      output_label = gr.Textbox(
          label="Hasil Prediksi Klasifikasi",
          placeholder="Menunggu gambar...",
          interactive=False,
      )
      output_confidence = gr.Textbox(
          label="Tingkat Keyakinan Probabilitas (Confidence Score)",
          placeholder="-",
          interactive=False,
      )
      output_latency = gr.Textbox(
          label="Waktu Inferensi Server (Latency)",
          placeholder="-",
          interactive=False,
      )
      output_model_info = gr.Textbox(
          label="Spesifikasi & Performa Lintas-Dataset Model",
          placeholder="-",
          interactive=False,
      )

  btn.click(
      fn=predict_image,
      inputs=[model_dropdown, image_input],
      outputs=[
          output_label,
          output_confidence,
          output_latency,
          output_model_info,
      ],
  )

if __name__ == "__main__":
  # Menjalankan server Gradio secara lokal dan membuka GUI
  # secara otomatis di browser.
  demo.launch(inbrowser=True)
