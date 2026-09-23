# Dataset

Dataset penelitian tidak disertakan dalam repository karena terdiri dari
sejumlah besar file gambar dan memiliki ukuran penyimpanan yang besar.

Dataset diperoleh dari beberapa sumber publik dan diproses kembali sesuai
kebutuhan penelitian.

## Dataset Sources

### 1. AnimeDL-2M

Source:
https://huggingface.co/datasets/FlyTweety/AnimeDL-2M/tree/main/fake_images

Digunakan sebagai sumber data AI-generated pada dataset in-domain.

### 2. Danbooru2024-SFW

Source:
https://huggingface.co/datasets/deepghs/danbooru2024-sfw

Digunakan sebagai sumber data human-made pada dataset in-domain.

### 3. NovelAI3

Source:
https://huggingface.co/datasets/shareAI/novelai3

Digunakan sebagai sumber data AI-generated pada pengujian cross-dataset.

### 4. Danbooru2021-SQLite

Source:
https://huggingface.co/datasets/KBlueLeaf/Danbooru2021-SQLite

Digunakan sebagai sumber data human-made pada pengujian cross-dataset.

## Dataset Structure

```text
dataset/
├── prepared/
│   ├── ai_cross_novelai3/
│   │   └── images/
│   ├── ai_train_animedl2m/
│   │   └── images/
│   ├── human_animedl2m_real/
│   │   └── images/
│   └── human_danbooru2021/
│       └── images/
│
├── processed/
│
├── raw/
│   ├── animedl2m_fake/
│   │   └── images/
│   ├── animedl2m_real/
│   │   └── images/
│   ├── danbooru2021_SQLite/
│   │   └── images/
│   └── novelai3/
│       └── images/
│
└── splits/
    ├── test_cross_generator/
    │   ├── ai/
    │   └── human/
    ├── test_in_domain/
    │   ├── ai/
    │   └── human/
    ├── train/
    │   ├── ai/
    │   └── human/
    └── validation/
        ├── ai/
        └── human/
