"""
07_statistical_visualization.py
Visualisasi statistik Baseline, Level 1, Level 2, Level 3, Level 4.

CSV harus memiliki schema:
model, condition, number_of_images, confusion_matrix,
accuracy, precision, recall, f1_score, roc_auc,
total_inference_time_seconds, inference_latency_ms_per_image,
throughput_images_per_second, peak_gpu_vram_mb_inference,
baseline_ram_mb_inference, peak_ram_mb_inference,
peak_ram_delta_mb_inference
"""

from pathlib import Path
import ast
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# KONFIGURASI
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]  # Menuju folder anime-ai-detector

RESULTS_BASELINE = ROOT_DIR / "results"
RESULTS_MODIFIED = ROOT_DIR / "results_modified"
OUTPUT_DIR = ROOT_DIR / "statistical_visualization"

# Path kandidat baseline relatif terhadap RESULTS_BASELINE
BASELINE_CSV_CANDIDATES = [
    RESULTS_BASELINE / "evaluation_summary.csv",
]

LEVEL_ORDER = ["Baseline", "Level 1", "Level 2", "Level 3", "Level 4"]

REQUIRED_COLUMNS = [
    "model", "condition", "number_of_images", "confusion_matrix",
    "accuracy", "precision", "recall", "f1_score", "roc_auc",
    "total_inference_time_seconds",
    "inference_latency_ms_per_image",
    "throughput_images_per_second",
    "peak_gpu_vram_mb_inference",
    "baseline_ram_mb_inference",
    "peak_ram_mb_inference",
    "peak_ram_delta_mb_inference",
]

CLASSIFICATION_METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "roc_auc": "ROC-AUC",
}

EFFICIENCY_METRICS = {
    "total_inference_time_seconds": ("Total Inference Time", "Seconds"),
    "inference_latency_ms_per_image": ("Inference Latency", "ms / image"),
    "throughput_images_per_second": ("Throughput", "Images / second"),
    "peak_gpu_vram_mb_inference": ("Peak GPU VRAM", "MB"),
    "baseline_ram_mb_inference": ("Baseline RAM", "MB"),
    "peak_ram_mb_inference": ("Peak RAM", "MB"),
    "peak_ram_delta_mb_inference": ("RAM Delta", "MB"),
}


# ============================================================
# UTILITAS
# ============================================================

def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_baseline_csv():
    for path in BASELINE_CSV_CANDIDATES:
        if path.exists():
            return path

    if RESULTS_BASELINE.exists():
        candidates = sorted(RESULTS_BASELINE.glob("*evaluation_summary*.csv"))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        "CSV baseline tidak ditemukan. Sesuaikan BASELINE_CSV_CANDIDATES."
    )


def load_csv(path, level_name):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"\nSchema tidak sesuai: {path}\n"
            f"Kolom hilang: {missing}\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["level"] = level_name

    numeric = [
        "number_of_images", "accuracy", "precision", "recall", "f1_score",
        "roc_auc", "total_inference_time_seconds",
        "inference_latency_ms_per_image",
        "throughput_images_per_second",
        "peak_gpu_vram_mb_inference",
        "baseline_ram_mb_inference",
        "peak_ram_mb_inference",
        "peak_ram_delta_mb_inference",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        valid = df[col].dropna()
        if ((valid < 0) | (valid > 1)).any():
            raise ValueError(f"Nilai {col} harus berada pada 0–1: {path}")

    return df


def load_all_data():
    frames = [load_csv(find_baseline_csv(), "Baseline")]

    for level in [1, 2, 3, 4]:
        path = RESULTS_MODIFIED / f"evaluation_summary_{level}.csv"
        frames.append(load_csv(path, f"Level {level}"))

    df = pd.concat(frames, ignore_index=True)

    models = sorted(df["model"].dropna().astype(str).unique())
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    df["level"] = pd.Categorical(
        df["level"], categories=LEVEL_ORDER, ordered=True
    )

    return df


def parse_confusion_matrix(value):
    if pd.isna(value):
        return None

    try:
        arr = np.asarray(ast.literal_eval(str(value)), dtype=float)
    except Exception:
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(value))
        if len(nums) != 4:
            return None
        arr = np.array([[float(nums[0]), float(nums[1])],
                        [float(nums[2]), float(nums[3])]])

    if arr.size != 4:
        return None

    return arr.reshape(2, 2)


# ============================================================
# 1. CLASSIFICATION — PER MODEL
# ============================================================

def plot_classification_per_model(df):
    out = OUTPUT_DIR / "01_classification_per_model"

    for model in df["model"].cat.categories:
        for condition in sorted(df["condition"].dropna().unique()):
            d = df[
                (df["model"] == model) &
                (df["condition"] == condition)
            ].sort_values("level")

            if d.empty:
                continue

            x = np.arange(len(d))
            width = 0.15

            fig, ax = plt.subplots(figsize=(12, 7))

            for i, (col, label) in enumerate(CLASSIFICATION_METRICS.items()):
                values = d[col].to_numpy() * 100
                bars = ax.bar(
                    x + (i - 2) * width,
                    values,
                    width,
                    label=label
                )

                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.5,
                        f"{value:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(d["level"].astype(str))
            ax.set_ylim(0, 105)
            ax.set_xlabel("Training Stage")
            ax.set_ylabel("Score (%)")
            ax.set_title(
                f"{model} — {str(condition).replace('_', ' ').title()}\n"
                "Classification Performance"
            )
            ax.legend(ncol=3)
            ax.grid(axis="y", alpha=0.25)

            save_figure(
                fig,
                out / f"{safe_filename(model)}_{safe_filename(condition)}.png"
            )


# ============================================================
# 2. CLASSIFICATION — SEMUA MODEL
# ============================================================

def plot_classification_all_models(df):
    out = OUTPUT_DIR / "02_classification_all_models"

    for condition in sorted(df["condition"].dropna().unique()):
        d = df[df["condition"] == condition]
        models = list(df["model"].cat.categories)
        x = np.arange(len(models))
        width = 0.14  # FIX: Mengecilkan lebar batang agar 5 kluster muat berjejer

        for col, label in CLASSIFICATION_METRICS.items():
            fig, ax = plt.subplots(figsize=(14, 7))

            for i, level in enumerate(LEVEL_ORDER):
                values = []

                for model in models:
                    row = d[
                        (d["model"] == model) &
                        (d["level"].astype(str) == level)
                    ]
                    values.append(
                        row[col].iloc[0] * 100 if not row.empty else np.nan
                    )

                bars = ax.bar(
                    x + (i - 2) * width,  # FIX: Pemetaan simetris untuk 5 kategori level
                    values,
                    width,
                    label=level
                )

                for bar, value in zip(bars, values):
                    if not np.isnan(value):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value + 0.4,
                            f"{value:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Model")
            ax.set_ylabel("Score (%)")
            ax.set_title(
                f"{label} — {str(condition).replace('_', ' ').title()}\n"
                "Baseline vs Level 1 vs Level 2 vs Level 3 vs Level 4"
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            save_figure(
                fig,
                out / f"{safe_filename(condition)}_{safe_filename(label)}.png"
            )


# ============================================================
# 3. EFISIENSI / RESOURCE
# ============================================================

def plot_efficiency_per_model(df):
    out = OUTPUT_DIR / "03_efficiency_per_model"

    for model in df["model"].cat.categories:
        for condition in sorted(df["condition"].dropna().unique()):
            d = df[
                (df["model"] == model) &
                (df["condition"] == condition)
            ].sort_values("level")

            if d.empty:
                continue

            for col, (title, ylabel) in EFFICIENCY_METRICS.items():
                fig, ax = plt.subplots(figsize=(11, 6))
                x = np.arange(len(d))
                values = d[col].to_numpy()
                bars = ax.bar(x, values, width=0.6)
                
                for bar, value in zip(bars, values):
                    if not np.isnan(value):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value,
                            f"{value:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=9
                        )
                
                ax.set_xticks(x)
                ax.set_xticklabels(d["level"].astype(str))
                ax.set_xlabel("Training Stage")
                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"{model} — {str(condition).replace('_', ' ').title()}\n"
                    f"{title}"
                )
                ax.grid(axis="y", alpha=0.25)
                
                save_figure(
                    fig,
                    out / safe_filename(model) / f"{safe_filename(condition)}_{col}.png"
                )


def plot_efficiency_all_models(df):
    out = OUTPUT_DIR / "04_efficiency_all_models"
    for condition in sorted(df["condition"].dropna().unique()):
        d = df[df["condition"] == condition]
        models = list(df["model"].cat.categories)
        x = np.arange(len(models))
        width = 0.14  # FIX: Penyelarasan geometri lebar data efisiensi
        
        for col, (title, ylabel) in EFFICIENCY_METRICS.items():
            fig, ax = plt.subplots(figsize=(14, 7))
            
            for i, level in enumerate(LEVEL_ORDER):
                values = []
                for model in models:
                    row = d[
                        (d["model"] == model) &
                        (d["level"].astype(str) == level)
                    ]
                    values.append(
                        row[col].iloc[0] if not row.empty else np.nan
                    )
                
                bars = ax.bar(
                    x + (i - 2) * width,  # FIX: Penyelarasan posisi kluster 5 level efisiensi
                    values,
                    width,
                    label=level
                )
                
                for bar, value in zip(bars, values):
                    if not np.isnan(value):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value,
                            f"{value:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=7
                        )
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15)
            ax.set_xlabel("Model")
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{title} — {str(condition).replace('_', ' ').title()}\n"
                "Baseline vs Level 1 vs Level 2 vs Level 3 vs Level 4"
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            save_figure(
                fig,
                out / f"{safe_filename(condition)}_{col}.png"
            )


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrices(df):
    out = OUTPUT_DIR / "05_confusion_matrix"
    for _, row in df.iterrows():
        cm = parse_confusion_matrix(row["confusion_matrix"])
        if cm is None:
            continue
        
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        image = ax.imshow(cm)
        
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{int(round(cm[i, j]))}",
                    ha="center", va="center", fontsize=14
                )
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Human", "AI"])
        ax.set_yticklabels(["Human", "AI"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(
            f"{row['model']} — {row['level']} — {row['condition']}\n"
            "Confusion Matrix"
        )
        fig.colorbar(image, ax=ax, label="Number of Images")
        
        save_figure(
            fig,
            out / safe_filename(row["model"]) / 
            f"{safe_filename(row['level'])}_{safe_filename(row['condition'])}.png"
        )


# ============================================================
# 5. DELTA TERHADAP BASELINE
# ============================================================

def plot_delta_from_baseline(df):
    out = OUTPUT_DIR / "06_delta_from_baseline"
    for condition in sorted(df["condition"].dropna().unique()):
        for model in df["model"].cat.categories:
            d = df[
                (df["model"] == model) &
                (df["condition"] == condition)
            ]
            base = d[d["level"].astype(str) == "Baseline"]
            if base.empty:
                continue
            base = base.iloc[0]
            
            levels = ["Level 1", "Level 2", "Level 3", "Level 4"]
            fig, ax = plt.subplots(figsize=(12, 7))
            x = np.arange(4)
            width = 0.14  # FIX: Diubah menjadi 0.14 agar kluster 5 metrik muat sejajar sempurna
            
            for i, (col, label) in enumerate(CLASSIFICATION_METRICS.items()):
                values = []
                for level in levels:
                    row = d[d["level"].astype(str) == level]
                    if row.empty:
                        values.append(np.nan)
                    else:
                        values.append((row.iloc[0][col] - base[col]) * 100)
                
                bars = ax.bar(
                    x + (i - 2) * width,  # FIX: Penyelarasan matematis sebaran kluster metrik terhadap sumbu X level
                    values,
                    width,
                    label=label
                )
                
                for bar, value in zip(bars, values):
                    if not np.isnan(value):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value + (0.15 if value >= 0 else -0.15),
                            f"{value:+.2f}",
                            ha="center",
                            va="bottom" if value >= 0 else "top",
                            fontsize=8
                        )
            
            ax.axhline(0, linewidth=1, color='black', alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(levels)
            ax.set_xlabel("Training Stage")
            ax.set_ylabel("Change from Baseline (percentage point)")
            ax.set_title(
                f"{model} — {str(condition).replace('_', ' ').title()}\n"
                "Classification Change from Baseline"
            )
            ax.legend(ncol=3)
            ax.grid(axis="y", alpha=0.25)
            
            save_figure(
                fig,
                out / f"{safe_filename(model)}_{safe_filename(condition)}.png"
            )


# ============================================================
# 6. FOKUS CROSS-GENERATOR
# ============================================================

def plot_cross_generator(df):
    out = OUTPUT_DIR / "07_cross_generator_focus"
    d = df[df["condition"] == "cross_generator"]
    if d.empty:
        return
    
    models = list(df["model"].cat.categories)
    x = np.arange(len(models))
    width = 0.14  # FIX: Penyelarasan geometri kluster lintas generator
    
    for col, label in CLASSIFICATION_METRICS.items():
        fig, ax = plt.subplots(figsize=(14, 7))
        for i, level in enumerate(LEVEL_ORDER):
            values = []
            for model in models:
                row = d[
                    (d["model"] == model) &
                    (d["level"].astype(str) == level)
                ]
                values.append(
                    row[col].iloc[0] * 100 if not row.empty else np.nan
                )
            
            ax.bar(
                x + (i - 2) * width,  # FIX: Penyelarasan simetris 5 kluster level pada grafik cross-generator
                values,
                width,
                label=level
            )
            
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score (%)")
        ax.set_title(
            f"Cross-Generator {label}\n"
            "Generalization: Baseline vs Level 1 vs Level 2 vs Level 3 vs Level 4"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        
        save_figure(
            fig,
            out / f"cross_generator_{safe_filename(label)}.png"
        )


# ============================================================
# 7. CSV RINGKASAN
# ============================================================

def create_summary_tables(df):
    out = OUTPUT_DIR / "08_summary_tables"
    out.mkdir(parents=True, exist_ok=True)
    
    combined = df.copy()
    combined["model"] = combined["model"].astype(str)
    combined["level"] = combined["level"].astype(str)
    combined.to_csv(out / "combined_all_stages.csv", index=False)
    
    classification = df[
        [
            "model", "level", "condition", "number_of_images",
            "accuracy", "precision", "recall", "f1_score", "roc_auc"
        ]
    ].copy()
    
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        classification[col] *= 100
        
    classification.rename(columns={
        "accuracy": "accuracy_percent",
        "precision": "precision_percent",
        "recall": "recall_percent",
        "f1_score": "f1_score_percent",
        "roc_auc": "roc_auc_percent",
    }, inplace=True)
    
    classification.to_csv(
        out / "classification_metrics_percent.csv", index=False
    )
    
    efficiency = df[
        [
            "model", "level", "condition", "number_of_images",
            "total_inference_time_seconds",
            "inference_latency_ms_per_image",
            "throughput_images_per_second",
            "peak_gpu_vram_mb_inference",
            "baseline_ram_mb_inference",
            "peak_ram_mb_inference",
            "peak_ram_delta_mb_inference",
        ]
    ].copy()
    efficiency.to_csv(out / "efficiency_metrics.csv", index=False)
    
    cross = classification[
        classification["condition"] == "cross_generator"
    ].copy()
    
    if not cross.empty:
        cross.sort_values(
            ["f1_score_percent", "accuracy_percent", "roc_auc_percent"],
            ascending=False
        ).to_csv(out / "cross_generator_ranking.csv", index=False)


# ============================================================
# 8. LINE-DOT STATISTICS — SEMUA METRIK
# ============================================================
#
# Menambahkan visualisasi garis + titik (line-dot) tanpa
# menghapus grafik lama.
#
# Output utama:
#   09_line_dot_statistics/
#
# Untuk SETIAP condition:
#   - in_domain
#   - cross_generator
#
# Dibuat grafik untuk:
#   Classification:
#       accuracy
#       precision
#       recall
#       f1_score
#       roc_auc
#
#   Efficiency:
#       total_inference_time_seconds
#       inference_latency_ms_per_image
#       throughput_images_per_second
#       peak_gpu_vram_mb_inference
#       baseline_ram_mb_inference
#       peak_ram_mb_inference
#       peak_ram_delta_mb_inference
#
#   Confusion Matrix:
#       TN
#       FP
#       FN
#       TP
#
# Setiap grafik membandingkan:
#   Baseline -> Level 1 -> Level 2 -> Level 3 -> Level 4
# untuk seluruh model.
#
# ============================================================

def _line_dot_base(df, condition, metric):
    """
    Mengambil data satu condition dan satu metric,
    lalu memastikan urutan level:
    Baseline -> Level 1 -> Level 2 -> Level 3 -> Level 4.
    """
    d = df[df["condition"] == condition].copy()

    if d.empty or metric not in d.columns:
        return d

    d["level"] = pd.Categorical(
        d["level"].astype(str),
        categories=LEVEL_ORDER,
        ordered=True
    )

    return d.sort_values(["model", "level"])


def _plot_line_dot(
    df,
    condition,
    metric,
    title,
    ylabel,
    output_path,
    percent=False,
):
    """
    Grafik garis + titik:
        X = Baseline -> Level 4
        Y = metric
        satu garis = satu model
    """
    d = _line_dot_base(df, condition, metric)

    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(df["model"].cat.categories)

    x = np.arange(len(LEVEL_ORDER))

    for model in models:
        md = d[d["model"] == model]

        values = []
        for level in LEVEL_ORDER:
            row = md[md["level"].astype(str) == level]

            if row.empty:
                values.append(np.nan)
            else:
                value = row.iloc[0][metric]

                if pd.isna(value):
                    values.append(np.nan)
                else:
                    values.append(
                        value * 100 if percent else value
                    )

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            markersize=6,
            label=str(model),
        )

        for xi, value in zip(x, values):
            if not np.isnan(value):
                suffix = "%" if percent else ""
                ax.annotate(
                    f"{value:.2f}{suffix}",
                    (xi, value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_ORDER)
    ax.set_xlabel("Training Stage")
    ax.set_ylabel(ylabel)

    ax.set_title(
        f"{title}\n"
        f"Condition: {str(condition).replace('_', ' ').title()}"
    )

    ax.legend(
        title="Model",
        loc="best",
    )

    ax.grid(
        axis="both",
        alpha=0.25,
    )

    if percent:
        ax.set_ylim(0, 105)

    save_figure(fig, output_path)


def plot_line_dot_classification(df):
    """
    Line-dot untuk seluruh classification metrics.
    """
    out = OUTPUT_DIR / "09_line_dot_statistics" / "classification"

    for condition in sorted(
        df["condition"].dropna().unique()
    ):
        for metric, label in CLASSIFICATION_METRICS.items():

            _plot_line_dot(
                df=df,
                condition=condition,
                metric=metric,
                title=f"{label} — Baseline to Level 4",
                ylabel="Score (%)",
                output_path=(
                    out
                    / f"{safe_filename(condition)}_"
                      f"{safe_filename(metric)}_line_dot.png"
                ),
                percent=True,
            )


def plot_line_dot_efficiency(df):
    """
    Line-dot untuk seluruh efficiency/resource metrics.
    """
    out = OUTPUT_DIR / "09_line_dot_statistics" / "efficiency"

    for condition in sorted(
        df["condition"].dropna().unique()
    ):
        for metric, (title, ylabel) in EFFICIENCY_METRICS.items():

            _plot_line_dot(
                df=df,
                condition=condition,
                metric=metric,
                title=f"{title} — Baseline to Level 4",
                ylabel=ylabel,
                output_path=(
                    out
                    / f"{safe_filename(condition)}_"
                      f"{safe_filename(metric)}_line_dot.png"
                ),
                percent=False,
            )


def _extract_confusion_values(row):
    """
    Mengambil TN, FP, FN, TP dari confusion_matrix.
    Format yang diharapkan:
        [[TN, FP],
         [FN, TP]]
    """
    cm = parse_confusion_matrix(row["confusion_matrix"])

    if cm is None:
        return None

    return {
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
    }


def plot_line_dot_confusion_matrix(df):
    """
    Line-dot confusion matrix dari Baseline -> Level 4.

    Dibuat 4 grafik per condition:
        TN
        FP
        FN
        TP

    Setiap garis = model.
    """
    out = OUTPUT_DIR / "09_line_dot_statistics" / "confusion_matrix"

    cm_frames = []

    for _, row in df.iterrows():
        values = _extract_confusion_values(row)

        if values is None:
            continue

        base = {
            "model": str(row["model"]),
            "condition": row["condition"],
            "level": str(row["level"]),
        }

        base.update(values)
        cm_frames.append(base)

    if not cm_frames:
        return

    cm_df = pd.DataFrame(cm_frames)

    for condition in sorted(
        cm_df["condition"].dropna().unique()
    ):
        d = cm_df[
            cm_df["condition"] == condition
        ].copy()

        for metric in ["TN", "FP", "FN", "TP"]:
            fig, ax = plt.subplots(figsize=(12, 7))

            x = np.arange(len(LEVEL_ORDER))

            for model in sorted(
                d["model"].dropna().unique()
            ):
                md = d[d["model"] == model]

                values = []

                for level in LEVEL_ORDER:
                    row = md[
                        md["level"] == level
                    ]

                    if row.empty:
                        values.append(np.nan)
                    else:
                        values.append(
                            row.iloc[0][metric]
                        )

                ax.plot(
                    x,
                    values,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=model,
                )

                for xi, value in zip(x, values):
                    if not np.isnan(value):
                        ax.annotate(
                            f"{int(round(value))}",
                            (xi, value),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha="center",
                            fontsize=8,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(LEVEL_ORDER)
            ax.set_xlabel("Training Stage")
            ax.set_ylabel("Number of Images")

            ax.set_title(
                f"Confusion Matrix — {metric}\n"
                f"Condition: "
                f"{str(condition).replace('_', ' ').title()}"
            )

            ax.legend(title="Model")
            ax.grid(axis="both", alpha=0.25)

            save_figure(
                fig,
                (
                    out
                    / f"{safe_filename(condition)}_"
                      f"{metric}_line_dot.png"
                ),
            )


def create_line_dot_csv_tables(df):
    """
    Membuat CSV khusus untuk data line-dot sehingga data grafik
    juga tersedia dalam bentuk tabel.

    Output:
        09_line_dot_statistics/csv/
    """
    out = OUTPUT_DIR / "09_line_dot_statistics" / "csv"
    out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Classification line-dot table
    # --------------------------------------------------------

    classification = df[
        [
            "model",
            "level",
            "condition",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]
    ].copy()

    for col in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]:
        classification[col] *= 100

    classification.rename(
        columns={
            "accuracy": "accuracy_percent",
            "precision": "precision_percent",
            "recall": "recall_percent",
            "f1_score": "f1_score_percent",
            "roc_auc": "roc_auc_percent",
        },
        inplace=True,
    )

    classification["level"] = pd.Categorical(
        classification["level"].astype(str),
        categories=LEVEL_ORDER,
        ordered=True,
    )

    classification.sort_values(
        ["condition", "model", "level"]
    ).to_csv(
        out / "line_dot_classification_all.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Efficiency line-dot table
    # --------------------------------------------------------

    efficiency = df[
        [
            "model",
            "level",
            "condition",
            "total_inference_time_seconds",
            "inference_latency_ms_per_image",
            "throughput_images_per_second",
            "peak_gpu_vram_mb_inference",
            "baseline_ram_mb_inference",
            "peak_ram_mb_inference",
            "peak_ram_delta_mb_inference",
        ]
    ].copy()

    efficiency["level"] = pd.Categorical(
        efficiency["level"].astype(str),
        categories=LEVEL_ORDER,
        ordered=True,
    )

    efficiency.sort_values(
        ["condition", "model", "level"]
    ).to_csv(
        out / "line_dot_efficiency_all.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix line-dot table
    # --------------------------------------------------------

    cm_rows = []

    for _, row in df.iterrows():
        values = _extract_confusion_values(row)

        if values is None:
            continue

        cm_rows.append({
            "model": row["model"],
            "level": row["level"],
            "condition": row["condition"],
            "TN": values["TN"],
            "FP": values["FP"],
            "FN": values["FN"],
            "TP": values["TP"],
        })

    if cm_rows:
        cm_table = pd.DataFrame(cm_rows)

        cm_table["level"] = pd.Categorical(
            cm_table["level"].astype(str),
            categories=LEVEL_ORDER,
            ordered=True,
        )

        cm_table.sort_values(
            ["condition", "model", "level"]
        ).to_csv(
            out / "line_dot_confusion_matrix_all.csv",
            index=False,
        )


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STATISTICAL VISUALIZATION")
    print("Baseline vs Level 1 vs Level 2 vs Level 3 vs Level 4")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[1/8] Membaca CSV...")
    df = load_all_data()
    
    print("[2/8] Classification per model...")
    plot_classification_per_model(df)
    
    print("[3/8] Classification semua model...")
    plot_classification_all_models(df)
    
    print("[4/8] Efficiency/resource...")
    plot_efficiency_per_model(df)
    plot_efficiency_all_models(df)
    
    print("[5/8] Confusion matrix...")
    plot_confusion_matrices(df)
    
    print("[6/8] Delta terhadap baseline...")
    plot_delta_from_baseline(df)
    
    print("[7/8] Cross-generator focus...")
    plot_cross_generator(df)
    
    print("[8/8] Tabel ringkasan...")
    create_summary_tables(df)
    
    print("[9/9] Line-dot statistics — semua metric...")
    plot_line_dot_classification(df)
    plot_line_dot_efficiency(df)
    plot_line_dot_confusion_matrix(df)
    create_line_dot_csv_tables(df)

    print("\nSELESAI.")
    print(f"Hasil: {OUTPUT_DIR.resolve()}")
    print("\nSatuan grafik:")
    print("- Accuracy / Precision / Recall / F1 / ROC-AUC : %")
    print("- Total inference time : seconds")
    print("- Latency : ms/image")
    print("- Throughput : images/second")
    print("- VRAM / RAM : MB")


if __name__ == "__main__":
    main()