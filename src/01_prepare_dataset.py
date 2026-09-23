from pathlib import Path
import random
import shutil
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1] # Menentukan folder utama proyek berdasarkan posisi file kode ini.
RAW_DIR = ROOT_DIR / "dataset" / "raw" #  Jalur menuju folder gambar mentah asli yang belum diproses.
PREPARED_DIR = ROOT_DIR / "dataset" / "prepared" # Jalur folder hasil kurasi dan penyaringan awal.
SPLIT_DIR = ROOT_DIR / "dataset" / "splits" # Jalur folder akhir di mana gambar sudah terbagi menjadi folder train, val, dan test.

# Mengapa harus seperti ini? Memisahkan folder mentah (raw),
# folder siap pakai (prepared), dan folder pecahan (splits)
# menjaga manajemen data tetap rapi. Jika salah melakukan
# pembagian data, cukup menghapus folder splits tanpa perlu
# menyaring ulang data mentah dari awal.

# ============================================================
# PEMBAGIAN JUMLAH GAMBAR
# ============================================================

# --------------------------------------------------------------------------------------------------------------------------------
HUMAN_ANIMEDL2M_COUNT = 3250       # Human AnimeDL-2M Real untuk in-domain pool.
# AnimeDL-2M Real: 6,396 raw -> dipilih 3,250 untuk in-domain.

AI_IN_DOMAIN_TOTAL = 3250
# AnimeDL-2M Fake: 15,081 raw -> dipilih 3,250 untuk AI in-domain.

# --------------------------------------------------------------------------------------------------------------------------------
FLUX_COUNT = 1083
SDXL_COUNT = 1083
SD_COUNT = 1084
# SD, SDXL, dan Flux adalah nama-nama arsitektur model AI pembuat gambar (text-to-image).

# Mengapa harus dibagi rata? Ini disebut Stratifikasi. Jika gambar didominasi oleh FLUX saja,
# model AI Deteksi hanya akan pintar mendeteksi gambar buatan FLUX,
# --------------------------------------------------------------------------------------------------------------------------------

DANBOORU_COUNT = 500 # Total gambar Human dari Danbooru2021_SQLite khusus untuk cross-generator.
# Danbooru2021_SQLite: 7,393 raw -> dipilih 500 untuk Human cross-generator.

NOVELAI_TOTAL = 500 # Total gambar dari NovelAI3 untuk ujian akhir.
NOVELAI_832_COUNT = 167 # Memilih 167 gambar dari gambar ukuran 832x1216
NOVELAI_1024_COUNT = 167 # Memilih 167 gambar dari gambar ukuran 1024x1024
NOVELAI_1216_COUNT = 166 # Memilih 166 gambar dari gambar ukuran 1216x832
# NovelAI3: 2,357 raw -> Dipilih 500 untuk AI Cross
# --------------------------------------------------------------------------------------------------------------------------------


# In-domain: 3,250 human + 3,250 AI = 6,500
TRAIN_PER_CLASS = 2250 #Jumlah gambar per kelas (Human/AI) untuk latihan.
VAL_PER_CLASS = 500 #  Jumlah gambar per kelas untuk validasi saat latihan.
TEST_IN_DOMAIN_PER_CLASS = 500 # Jumlah gambar per kelas untuk ujian standar.

RANDOM_SEED = 42 # Angka pengunci fungsi acak.

# Mengapa harus dikunci? Komputer mengacak data menggunakan algoritma.
# Jika angka seed dikunci (misal di angka 42),
# urutan pengacakan akan selalu sama setiap kali kode dijalankan.
# Ini sangat penting dalam riset ilmiah agar eksperimen dapat diulang
# kembali (reproducible) dengan hasil pembagian data yang identik.
# --------------------------------------------------------------------------------------------------------------------------------





# ============================================================
# FILTER HUMAN
# ============================================================
# Alasan:
# AnimeDL-2M Real ditemukan memiliki beberapa citra dengan dimensi
# ekstrem, misalnya 100 x 50,000. Citra seperti ini tidak representatif
# untuk eksperimen klasifikasi yang akan dibandingkan dengan sumber lain.
# Filter ini menjaga ukuran minimum/maksimum dan membatasi rasio aspek.
# Aturan filter yang sama juga diterapkan pada Danbooru2021_SQLite
# agar data Human pada cross-generator memenuhi standar kualitas yang sama.

MIN_WIDTH = 512
MIN_HEIGHT = 512
MAX_WIDTH = 2048
MAX_HEIGHT = 2048
# Batas ukuran piksel terkecil dan terbesar.

MAX_ASPECT_RATIO = 2.0
# Batas kelonjongan gambar (panjang tidak boleh lebih dari 2 kali lebar, atau sebaliknya).

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
} # Daftar ekstensi file gambar resmi yang diizinkan untuk dibaca.

ANIMEDL2M_REAL = RAW_DIR / "animedl2m_real" / "images"
ANIMEDL2M_FAKE = RAW_DIR / "animedl2m_fake" / "images"
DANBOORU2021 = RAW_DIR / "danbooru2021_SQLite" / "images"
NOVELAI3 = RAW_DIR / "novelai3" / "images"
# folder yang diraih untuk diambil gambar nya


def get_images(folder):
    """Mengambil seluruh file gambar dari folder dan seluruh subfoldernya."""
    if not folder.exists():
        print(f"[WARNING] Folder tidak ditemukan: {folder}")
        return []

    return sorted(
        [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
# Kegunaan: Mencari semua file di dalam folder (dan subfolder di dalamnya)
# yang memiliki ekstensi gambar sesuai daftar IMAGE_EXTENSIONS,
# lalu mengurutkannya berdasarkan nama file.

# Mengapa harus seperti ini? Menggunakan fungsi bawaan rglob("*")
# memastikan tidak ada gambar yang tertinggal di dalam subfolder
# terdalam sekalipun. Fungsi sorted() digunakan agar urutan pembacaan
# data konsisten di sistem operasi mana pun (Windows, Linux, Mac).


def recreate_directory(folder):
    """Hapus folder lama lalu buat ulang agar hasil eksperimen bersih."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

# Kegunaan: Menghapus folder lama jika sudah ada, lalu membuat folder kosong yang baru.

# Mengapa harus seperti ini? Untuk mencegah tercampurnya sisa data dari eksperimen lama.
# Jika kita mengubah konfigurasi jumlah data dan menjalankan ulang kode
# tanpa menghapus folder lama, file lama yang tidak terpakai akan tetap
# menumpuk di sana dan mengacaukan hasil latihan model AI.3


def create_class_directory(base_dir):
    """Buat struktur kelas ImageFolder: human/ dan ai/."""
    (base_dir / "human").mkdir(parents=True, exist_ok=True)
    (base_dir / "ai").mkdir(parents=True, exist_ok=True)

# Kegunaan: Membuat dua subfolder wajib di dalam folder tujuan, yaitu folder human dan folder ai.

# Mengapa harus seperti ini? Struktur folder terpisah seperti train/human/ dan train/ai/
# adalah format standar (ImageFolder) yang dikenali secara otomatis oleh framework
# pembelajaran mesin populer seperti PyTorch atau TensorFlow untuk menentukan label kelas
# secara otomatis.


def copy_images(images, destination, prefix="image"):
    """Salin gambar ke folder tujuan dengan nama file yang konsisten."""
    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(images):
        destination_file = (
            destination / f"{prefix}_{index:05d}{source.suffix.lower()}"
        )
        shutil.copy2(source, destination_file)

# Kegunaan: Menyalin gambar ke folder baru sekaligus mengubah namanya menjadi
# format yang seragam (misal: human_A_00001.png, human_A_00002.png).

# Mengapa harus seperti ini? Nama file asli dari internet seringkali berantakan,
# mengandung karakter aneh, atau bahkan memiliki nama yang sama antar folder.
# Penamaan ulang dengan nomor urut (00001, 00002) mencegah file tertimpa
# (overwrite) dan mempermudah pelacakan data jika terjadi eror.




# ============================================================
# 1. KURASI ANIMEDL-2M REAL / HUMAN
# ============================================================


def is_valid_human_image(image_path):
    """
    Validasi citra Human.

    Aturan:
    - width >= 512
    - height >= 512
    - width <= 2048
    - height <= 2048
    - aspect ratio <= 2.0
    - file harus dapat dibuka oleh Pillow
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False

            if width > MAX_WIDTH or height > MAX_HEIGHT:
                return False

            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False

            return True

    except Exception:
        return False

# Kegunaan: Membuka gambar menggunakan modul Pillow (PIL) untuk mengecek apakah dimensinya
# memenuhi syarat minimum, maksimum, rasio aspek, dan memastikan filenya tidak rusak (corrupted).

# Mengapa harus seperti ini? Dataset internet sering kali mengandung gambar
# rusak atau dimensi ekstrem (contoh: gambar vertikal super panjang (100 \times 50000\) piksel).
# Jika gambar ekstrem ini dipaksa masuk ke model AI, proses pengubahan ukuran (resize)
# di tahap selanjutnya akan membuat gambar menjadi sangat gepeng
# dan merusak fitur visual yang harus dipelajari oleh AI.


def filter_human_images(images, dataset_name="HUMAN"):
    valid = []

    for image in images:
        if is_valid_human_image(image):
            valid.append(image)

    print(f"\n[{dataset_name}]")
    print(f"Raw      : {len(images)}")
    print(f"Valid    : {len(valid)}")
    print(f"Rejected : {len(images) - len(valid)}")

    return valid


def prepare_animedl2m_human(human_images):
    valid = filter_human_images(human_images, "HUMAN / ANIMEDL-2M REAL")

    if len(valid) < HUMAN_ANIMEDL2M_COUNT:
        raise RuntimeError(
            f"AnimeDL-2M Real valid hanya {len(valid)}, tetapi diperlukan "
            f"{HUMAN_ANIMEDL2M_COUNT}."
        )

    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(valid, HUMAN_ANIMEDL2M_COUNT)

    print("\n[ANIMEDL-2M HUMAN SELECTION]")
    print(f"Total selected : {len(selected)}")

    return selected


# ============================================================
# 2. KURASI ANIMEDL-2M FAKE
# ============================================================
# AnimeDL-2M Fake tidak diberi filter ukuran seperti Human karena
# fokus kurasinya berbeda: sumber AI dipertahankan berdasarkan generator.
# Nama file dipakai untuk memisahkan FLUX, SDXL, dan SD.
# Setelah itu jumlah tiap generator diseimbangkan.


def classify_animedl2m_fake(fake_images):
    groups = {
        "flux": [],
        "sdxl": [],
        "sd": [],
    }

    for image in fake_images:
        name = image.name.lower()

        if name.startswith("flux"):
            groups["flux"].append(image)
        elif name.startswith("sdxl"):
            groups["sdxl"].append(image)
        elif name.startswith("sd_"):
            groups["sd"].append(image)

    return groups

# Kegunaan: Membaca nama file gambar AI dari dataset AnimeDL-2M,
# lalu mengelompokkannya ke dalam dictionary berdasarkan teks awal
# nama filenya (flux, sdxl, atau sd_).


def prepare_animedl2m_fake(fake_images):
    groups = classify_animedl2m_fake(fake_images)

    required = {
        "flux": FLUX_COUNT,
        "sdxl": SDXL_COUNT,
        "sd": SD_COUNT,
    }

    print("\n[ANIMEDL-2M FAKE / GENERATOR STRATIFICATION]")

    for key in ("flux", "sdxl", "sd"):
        print(f"{key.upper():5s} available : {len(groups[key])}")

        if len(groups[key]) < required[key]:
            raise RuntimeError(
                f"{key.upper()} hanya memiliki {len(groups[key])}, "
                f"tetapi diperlukan {required[key]}."
            )

    rng = random.Random(RANDOM_SEED)
    selected = []

    for key in ("flux", "sdxl", "sd"):
        part = rng.sample(groups[key], required[key])
        selected.extend(part)
        print(f"{key.upper():5s} selected   : {len(part)}")

    # Campur urutan setelah kuota tiap generator terpenuhi.
    rng.shuffle(selected)

    print(f"TOTAL selected : {len(selected)}")
    return selected

# Kegunaan: Memastikan jumlah gambar dari masing-masing jenis AI
# (FLUX, SDXL, SD) mencukupi kuota, mengambil sampel acak sesuai kuota,
# lalu mengacak ulang (shuffle) seluruh hasil gabungannya.

# Mengapa diacak ulang setelah digabung? Jika tidak diacak ulang,
# susunan datanya akan berurutan: semua gambar FLUX di awal, diikuti SDXL, lalu SD di akhir.
# Saat data ini dipecah menjadi data Train, Val, dan Test, ada risiko data Test
# hanya kebagian gambar jenis SD, sementara data Train hanya kebagian FLUX.
# Pengacakan akhir menjamin distribusi jenis AI tersebar merata di semua pecahan data.




# ============================================================
# 3. KURASI DANBOORU2021_SQLITE / HUMAN CROSS-GENERATOR
# ============================================================
# Danbooru2021_SQLite hanya digunakan sebagai sumber Human untuk
# cross-generator test. Data ini TIDAK masuk ke train, validation,
# maupun test in-domain.


def prepare_danbooru2021(danbooru_images):
    valid = filter_human_images(danbooru_images, "HUMAN / DANBOORU2021_SQLITE")

    if len(valid) < DANBOORU_COUNT:
        raise RuntimeError(
            f"Danbooru2021_SQLite valid hanya {len(valid)}, tetapi diperlukan "
            f"{DANBOORU_COUNT}."
        )

    rng = random.Random(RANDOM_SEED + 2)
    selected = rng.sample(valid, DANBOORU_COUNT)

    print("\n[DANBOORU2021_SQLITE / CROSS-GENERATOR HUMAN]")
    print(f"Total selected : {len(selected)}")
    print("Isolation      : khusus cross-generator test")

    return selected


# ============================================================
# 4. KURASI NOVELAI3
# ============================================================
# NovelAI3 dibagi berdasarkan kelompok resolusi yang tersedia.
# Tujuannya menjaga representasi tiga bentuk dimensi tetap seimbang.


def classify_novelai3(novelai_images):
    groups = {
        "832x1216": [],
        "1024x1024": [],
        "1216x832": [],
    }

    for image in novelai_images:
        path_string = str(image).lower()

        if "832x1216" in path_string or "832_1216" in path_string:
            groups["832x1216"].append(image)
        elif "1024x1024" in path_string or "1024_1024" in path_string:
            groups["1024x1024"].append(image)
        elif "1216x832" in path_string or "1216_832" in path_string:
            groups["1216x832"].append(image)

    return groups


def prepare_novelai(novelai_images):
    groups = classify_novelai3(novelai_images)

    required = {
        "832x1216": NOVELAI_832_COUNT,
        "1024x1024": NOVELAI_1024_COUNT,
        "1216x832": NOVELAI_1216_COUNT,
    }

    print("\n[NOVELAI3 / RESOLUTION STRATIFICATION]")

    rng = random.Random(RANDOM_SEED)
    selected = []

    for key in ("832x1216", "1024x1024", "1216x832"):
        print(f"{key:10s} available : {len(groups[key])}")

        if len(groups[key]) < required[key]:
            raise RuntimeError(
                f"NovelAI3 {key} hanya memiliki {len(groups[key])}, "
                f"tetapi diperlukan {required[key]}."
            )

        part = rng.sample(groups[key], required[key])
        selected.extend(part)
        print(f"{key:10s} selected   : {len(part)}")

    rng.shuffle(selected)

    print(f"TOTAL selected : {len(selected)}")
    return selected

# Kegunaan: Memiliki logika yang sama dengan penyaringan data AI di atas,
# namun pengelompokannya bukan berdasarkan nama generator,
# melainkan berdasarkan string resolusi (832x1216, dll) yang ada pada jalur filenya.
# Alasan penerapannya sama, yaitu untuk menjaga keseimbangan bentuk gambar (resolution stratification).




# ============================================================
# 5. CREATE PREPARED DATASET
# ============================================================


def create_prepared_dataset(human_animedl2m, animedl2m_fake, human_danbooru, novelai):
    recreate_directory(PREPARED_DIR)

    copy_images(
        human_animedl2m,
        PREPARED_DIR / "human_animedl2m_real" / "images",
        prefix="human_animedl2m",
    )

    copy_images(
        animedl2m_fake,
        PREPARED_DIR / "ai_train_animedl2m" / "images",
        prefix="animedl2m",
    )

    copy_images(
        human_danbooru,
        PREPARED_DIR / "human_danbooru2021" / "images",
        prefix="danbooru2021",
    )

    copy_images(
        novelai,
        PREPARED_DIR / "ai_cross_novelai3" / "images",
        prefix="novelai3",
    )

# Kegunaan: Menjadi "dirigen" yang bertugas memanggil fungsi recreate_directory
# untuk membersihkan folder tujuan, lalu memicu fungsi copy_images untuk
# memindahkan ke-4 kelompok data yang sudah lolos kurasi:
# Human AnimeDL-2M, AnimeDL Fake, Human Danbooru2021, dan NovelAI3.

# Pemisahan folder dilakukan berdasarkan PERAN EKSPERIMEN, bukan hanya label kelas.
# Human AnimeDL-2M + AnimeDL Fake = in-domain pool.
# Human Danbooru2021 + NovelAI3 = cross-generator test pool.




# ============================================================
# 6. EXACT SPLIT IN-DOMAIN
# ============================================================


def split_exact(images, train_count, val_count, test_count, seed):
    total = train_count + val_count + test_count

    if len(images) != total:
        raise ValueError(
            f"Jumlah gambar {len(images)} tidak sama dengan "
            f"total split {total}."
        )

    rng = random.Random(seed)
    items = list(images)
    rng.shuffle(items)

    train = items[:train_count]
    validation = items[train_count:train_count + val_count]
    test = items[train_count + val_count:]

    return train, validation, test

# Kegunaan: Membagi daftar gambar menjadi 3 bagian (Train, Val, Test)
# dengan jumlah angka yang pasti/eksak sesuai parameter input.

# Mengapa harus seperti ini? Umumnya, orang membagi data menggunakan
# persentase (misal 70% Train, 20% Val, 10% Test).
# Namun, pembagian berbasis persentase sering menghasilkan
# angka desimal yang harus dibulatkan, sehingga jumlah akhir gambar per kelas
# bisa selisih sedikit. Fungsi ini menjamin jumlah gambar
# di tiap kelas akan tepat sama persis hingga satuan unit terakhir.


def create_in_domain_split():
    human = get_images(PREPARED_DIR / "human_animedl2m_real" / "images")
    ai = get_images(PREPARED_DIR / "ai_train_animedl2m" / "images")

    if len(human) != HUMAN_ANIMEDL2M_COUNT:
        raise RuntimeError(
            f"Human AnimeDL-2M harus {HUMAN_ANIMEDL2M_COUNT}, ditemukan {len(human)}."
        )

    if len(ai) != AI_IN_DOMAIN_TOTAL:
        raise RuntimeError(f"AI Train harus {AI_IN_DOMAIN_TOTAL}, ditemukan {len(ai)}.")

    human_train, human_val, human_test = split_exact(
        human,
        TRAIN_PER_CLASS,
        VAL_PER_CLASS,
        TEST_IN_DOMAIN_PER_CLASS,
        RANDOM_SEED,
    )

    ai_train, ai_val, ai_test = split_exact(
        ai,
        TRAIN_PER_CLASS,
        VAL_PER_CLASS,
        TEST_IN_DOMAIN_PER_CLASS,
        RANDOM_SEED + 1,
    )

    for name in ("train", "validation", "test_in_domain"):
        recreate_directory(SPLIT_DIR / name)
        create_class_directory(SPLIT_DIR / name)

    # TRAIN = 2,250 Human + 2,250 AI = 4,500
    copy_images(human_train, SPLIT_DIR / "train" / "human", "human")
    copy_images(ai_train, SPLIT_DIR / "train" / "ai", "ai")

    # VALIDATION = 500 Human + 500 AI = 1,000
    copy_images(human_val, SPLIT_DIR / "validation" / "human", "human")
    copy_images(ai_val, SPLIT_DIR / "validation" / "ai", "ai")

    # TEST IN-DOMAIN = 500 Human + 500 AI = 1,000
    copy_images(human_test, SPLIT_DIR / "test_in_domain" / "human", "human")
    copy_images(ai_test, SPLIT_DIR / "test_in_domain" / "ai", "ai")

    print("\n[IN-DOMAIN SPLIT]")
    print(f"Train      : {len(human_train)} Human + {len(ai_train)} AI = 4500")
    print(f"Validation : {len(human_val)} Human + {len(ai_val)} AI = 1000")
    print(f"Test       : {len(human_test)} Human + {len(ai_test)} AI = 1000")

# Kegunaan: Mengambil gambar dari folder Human AnimeDL-2M dan AI AnimeDL-2M,
# memecah keduanya menggunakan fungsi split_exact, lalu menyalin hasilnya
# ke struktur folder ImageFolder di dalam folder splits/train, splits/validation,
# dan splits/test_in_domain.

# Mengapa menggunakan RANDOM_SEED + 1 untuk data AI? Kode ini menggunakan seed 42
# untuk data manusia dan seed 43 untuk data AI. Ini adalah trik agar pola pengacakan
# urutan antara kelompok manusia dan kelompok AI tidak identik,
# sehingga indeks urutan gambar tidak saling mengikat secara linier saat dipecah.




# ============================================================
# 7. CROSS-GENERATOR TEST
# ============================================================
# Human Danbooru2021 dan NovelAI3 tidak pernah masuk train, validation,
# atau test-in-domain.
# Model yang telah selesai training langsung diuji di sini tanpa retraining.


def create_cross_generator_test():
    human_danbooru = get_images(PREPARED_DIR / "human_danbooru2021" / "images")
    novelai = get_images(PREPARED_DIR / "ai_cross_novelai3" / "images")

    if len(human_danbooru) != DANBOORU_COUNT:
        raise RuntimeError(
            f"Human Danbooru2021_SQLite harus {DANBOORU_COUNT}, ditemukan {len(human_danbooru)}."
        )

    if len(novelai) != NOVELAI_TOTAL:
        raise RuntimeError(
            f"NovelAI3 harus {NOVELAI_TOTAL}, ditemukan {len(novelai)}."
        )

    cross_dir = SPLIT_DIR / "test_cross_generator"
    recreate_directory(cross_dir)
    create_class_directory(cross_dir)

    copy_images(human_danbooru, cross_dir / "human", "human")
    copy_images(novelai, cross_dir / "ai", "ai")

    print("\n[CROSS-GENERATOR TEST]")
    print(f"Human Danbooru2021_SQLite : {len(human_danbooru)}")
    print(f"AI NovelAI3               : {len(novelai)}")
    print(f"TOTAL                     : {len(human_danbooru) + len(novelai)}")

# Kegunaan: Mengambil dua kelompok eksternal yang sejak awal diisolasi,
# yaitu Human Danbooru2021_SQLite dan AI NovelAI3, lalu menyatukannya ke dalam folder
# pengujian eksternal bernama splits/test_cross_generator.

# Mengapa harus dilakukan seperti ini? Ini adalah inti dari validasi model yang kuat.
# Model AI dilatih menggunakan data AnimeDL-2M Real dan AnimeDL-2M Fake
# (FLUX/SDXL/SD). Dengan mengujinya menggunakan NovelAI3 sebagai sumber AI
# dan Danbooru2021_SQLite sebagai sumber Human, model menghadapi distribusi data
# eksternal yang tidak pernah digunakan selama proses pembelajaran.




# ============================================================
# 8. SUMMARY
# ============================================================

def print_final_summary():
    print("""
============================================================
FINAL DATASET DESIGN
============================================================
RAW
  AnimeDL-2M Real       : 6,396
  AnimeDL-2M Fake       : 15,081
  Danbooru2021_SQLite   : 7,393
  NovelAI3              : 2,357

PREPARED
  Human AnimeDL-2M Real : 3,250
  AI AnimeDL-2M Fake    : 3,250
    - FLUX              : 1,083
    - SDXL              : 1,083
    - SD                : 1,084
  Human Danbooru2021    :   500
  AI NovelAI3           :   500
    - 832x1216          :   167
    - 1024x1024         :   167
    - 1216x832          :   166

IN DOMAIN POOL
  Human AnimeDL-2M + AI AnimeDL-2M : 6,500
  Train                 : 4,500 (2,250 Human + 2,250 AI)
  Validation            : 1,000 (500 Human + 500 AI)
  Test In-Domain        : 1,000 (500 Human + 500 AI)

CROSS GENERATOR TEST
  Human Danbooru2021 + AI NovelAI3 : 1,000 (500 Human + 500 AI)

SEPARATION RULE
  Human Danbooru2021 : tidak pernah masuk in-domain
  NovelAI3           : tidak pernah masuk in-domain
  AnimeDL-2M Real    : tidak pernah masuk cross-generator
  AnimeDL-2M Fake    : tidak pernah masuk cross-generator
============================================================
""")

# Kegunaan: Mencetak rangkuman cetak biru (blueprint)
# desain dataset berupa teks di terminal setelah seluruh proses selesai.

# Mengapa harus seperti ini? Sebagai dokumentasi instan bagi peneliti atau programmer.
# Melalui rangkuman ini, kita bisa langsung memverifikasi secara visual
# bahwa matematika pembagian data sudah sinkron (misal: 2250 + 500 + 500 = 3250)
# tanpa harus menghitung manual jumlah file di dalam folder sistem operasi.


def load_raw_datasets():
    human_animedl2m = get_images(ANIMEDL2M_REAL)
    fake = get_images(ANIMEDL2M_FAKE)
    danbooru = get_images(DANBOORU2021)
    novelai = get_images(NOVELAI3)

    print("\n[RAW DATASET]")
    print(f"AnimeDL-2M Real      : {len(human_animedl2m)}")
    print(f"AnimeDL-2M Fake      : {len(fake)}")
    print(f"Danbooru2021_SQLite  : {len(danbooru)}")
    print(f"NovelAI3             : {len(novelai)}")

    if not human_animedl2m:
        raise RuntimeError("AnimeDL-2M Real kosong atau folder tidak ditemukan.")
    if not fake:
        raise RuntimeError("AnimeDL-2M Fake kosong atau folder tidak ditemukan.")
    if not danbooru:
        raise RuntimeError("Danbooru2021_SQLite kosong atau folder tidak ditemukan.")
    if not novelai:
        raise RuntimeError("NovelAI3 kosong atau folder tidak ditemukan.")

    return human_animedl2m, fake, danbooru, novelai


def main():
    print("\n" + "=" * 60)
    print("ANIME AI DETECTOR - DATASET PREPARATION")
    print("=" * 60)

    human_animedl2m, fake, danbooru, novelai = load_raw_datasets()

    selected_human_animedl2m = prepare_animedl2m_human(human_animedl2m)
    selected_fake = prepare_animedl2m_fake(fake)
    selected_danbooru = prepare_danbooru2021(danbooru)
    selected_novelai = prepare_novelai(novelai)

    create_prepared_dataset(
        selected_human_animedl2m,
        selected_fake,
        selected_danbooru,
        selected_novelai,
    )

    recreate_directory(SPLIT_DIR)
    create_in_domain_split()
    create_cross_generator_test()
    print_final_summary()

    print("Dataset preparation selesai.")


# Kegunaan: load_raw_datasets berfungsi mengecek keberadaan folder
# mentah di awal program dan menghentikan kode secara paksa (raise RuntimeError)
# jika folder kosong. Sementara main() bertugas mengontrol urutan eksekusi
# seluruh fungsi di atas dari awal hingga selesai.


if __name__ == "__main__":
    main()
