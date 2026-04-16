import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/01_data_exploration.ipynb")


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md_cell(
        """
# 01 - Data Exploration

This notebook explores the local ISIC 2020 dataset stored under `data/`.

**Exploration goals**

- validate the local image and metadata layout
- understand the raw binary imbalance
- profile the available clinical descriptors
- inspect repeated `patient_id` patterns before splitting
- compare the raw dataset with a proposed training-effective subset
        """
    ),
    code_cell(
        """
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
from IPython import get_ipython
from IPython.display import display

ipython = get_ipython()
if ipython is None or ipython.__class__.__name__ != "ZMQInteractiveShell":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


def resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent]
    for candidate in candidates:
        meta_path = candidate / "data" / "ISIC_2020_Training_GroundTruth.csv"
        img_dir = candidate / "data" / "train"
        if meta_path.exists() and img_dir.exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root containing data/ISIC_2020_Training_GroundTruth.csv.")


ROOT_DIR = resolve_repo_root()
DATA_DIR = ROOT_DIR / "data"
IMG_DIR = DATA_DIR / "train"
META_PATH = DATA_DIR / "ISIC_2020_Training_GroundTruth.csv"
FIG_DIR = ROOT_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NON_MELANOMA_TO_MELANOMA_RATIO = 6.0
MIN_STRATUM_COUNT = 5
TARGET_NAMES = {0: "Non-melanoma", 1: "Melanoma"}
        """
    ),
    md_cell(
        """
## 1. Load the Metadata

The ISIC 2020 training CSV already provides a binary target:

- `1` = melanoma
- `0` = non-melanoma
        """
    ),
    code_cell(
        """
def clean_category(series: pd.Series, missing_label: str = "missing") -> pd.Series:
    return series.fillna(missing_label).astype(str).str.strip().replace("", missing_label).str.lower()


df = pd.read_csv(META_PATH).copy()
df = df.rename(columns={"image_name": "image_id"})

df["binary_label"] = df["target"].astype(int)
df["target_name"] = df["binary_label"].map(TARGET_NAMES)
df["diagnosis_clean"] = clean_category(df["diagnosis"])
df["sex_clean"] = clean_category(df["sex"])
df["site_clean"] = clean_category(df["anatom_site_general_challenge"])
df["img_path"] = df["image_id"].map(lambda image_id: IMG_DIR / f"{image_id}.jpg")
df["image_exists"] = df["img_path"].map(Path.exists)

missing_metadata = (
    df[["sex", "age_approx", "anatom_site_general_challenge"]]
    .isna()
    .sum()
    .rename("missing_count")
    .to_frame()
)
missing_metadata["missing_pct"] = (missing_metadata["missing_count"] / len(df) * 100).round(2)

integrity_summary = pd.DataFrame(
    [
        {"metric": "rows", "value": len(df)},
        {"metric": "unique_images", "value": int(df["image_id"].nunique())},
        {"metric": "duplicate_images", "value": int(df["image_id"].duplicated().sum())},
        {"metric": "unique_patients", "value": int(df["patient_id"].nunique())},
        {"metric": "missing_image_files", "value": int((~df["image_exists"]).sum())},
        {"metric": "melanoma_images", "value": int((df["binary_label"] == 1).sum())},
        {"metric": "non_melanoma_images", "value": int((df["binary_label"] == 0).sum())},
    ]
)

target_summary = (
    df["binary_label"]
    .value_counts()
    .sort_index()
    .rename(index=TARGET_NAMES)
    .rename_axis("target")
    .reset_index(name="count")
)
target_summary["pct"] = (target_summary["count"] / len(df) * 100).round(2)

display(integrity_summary)
display(target_summary)
display(missing_metadata)
df.head()
        """
    ),
    md_cell(
        """
## 2. Build a Training-Effective Dataset

The raw dataset is extremely imbalanced. For exploration, we compare it with a proposed training-effective subset that:

- keeps every melanoma image
- downsamples only the negative class
- preserves the negative diagnosis mix as much as possible
        """
    ),
    code_cell(
        """
def build_negative_sampling_stratum(frame: pd.DataFrame, min_count: int = MIN_STRATUM_COUNT) -> pd.Series:
    diagnosis = frame["diagnosis_clean"].copy()
    counts = diagnosis.value_counts()
    rare_labels = set(counts[counts < min_count].index)
    diagnosis = diagnosis.where(~diagnosis.isin(rare_labels), "other_rare_negative")
    return diagnosis


def build_effective_dataset(df: pd.DataFrame, non_melanoma_ratio: float, seed: int) -> pd.DataFrame:
    melanoma_df = df[df["binary_label"] == 1].copy()
    non_melanoma_df = df[df["binary_label"] == 0].copy()

    target_non_melanoma = min(
        len(non_melanoma_df),
        int(round(len(melanoma_df) * non_melanoma_ratio)),
    )

    if target_non_melanoma <= 0:
        raise ValueError("The requested non-melanoma ratio produced an empty selection.")

    non_melanoma_df["sampling_stratum"] = build_negative_sampling_stratum(non_melanoma_df)

    if target_non_melanoma == len(non_melanoma_df):
        selected_non_melanoma = non_melanoma_df.copy()
    else:
        selected_non_melanoma, _ = train_test_split(
            non_melanoma_df,
            train_size=target_non_melanoma,
            stratify=non_melanoma_df["sampling_stratum"],
            random_state=seed,
        )

    melanoma_df["selection_source"] = "kept_all_melanoma"
    selected_non_melanoma["selection_source"] = "downsampled_non_melanoma"

    effective_df = (
        pd.concat([melanoma_df, selected_non_melanoma], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return effective_df


effective_df = build_effective_dataset(df, NON_MELANOMA_TO_MELANOMA_RATIO, SEED)

raw_binary_summary = (
    df["binary_label"]
    .value_counts()
    .sort_index()
    .rename(index=TARGET_NAMES)
    .rename_axis("target")
    .reset_index(name="count")
)
raw_binary_summary["dataset"] = "raw"

effective_binary_summary = (
    effective_df["binary_label"]
    .value_counts()
    .sort_index()
    .rename(index=TARGET_NAMES)
    .rename_axis("target")
    .reset_index(name="count")
)
effective_binary_summary["dataset"] = "effective"

comparison_summary = pd.concat([raw_binary_summary, effective_binary_summary], axis=0).reset_index(drop=True)
comparison_summary["pct"] = comparison_summary.groupby("dataset")["count"].transform(lambda s: (s / s.sum() * 100).round(2))

negative_mix = pd.DataFrame(
    {
        "raw_negative_count": df[df["binary_label"] == 0]["diagnosis_clean"].value_counts(),
        "effective_negative_count": effective_df[effective_df["binary_label"] == 0]["diagnosis_clean"].value_counts(),
    }
).fillna(0).astype(int).sort_values("raw_negative_count", ascending=False)
negative_mix["retention_pct"] = (
    negative_mix["effective_negative_count"] / negative_mix["raw_negative_count"] * 100
).round(2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

raw_counts = raw_binary_summary.set_index("target").loc[["Non-melanoma", "Melanoma"], "count"]
effective_counts = effective_binary_summary.set_index("target").loc[["Non-melanoma", "Melanoma"], "count"]

axes[0].bar(raw_counts.index, raw_counts.values, color=["#2b6cb0", "#c53030"])
axes[0].set_title("Raw Binary Distribution", fontweight="bold")
axes[0].set_ylabel("Count")

axes[1].bar(effective_counts.index, effective_counts.values, color=["#2b6cb0", "#c53030"])
axes[1].set_title("Effective Training Distribution", fontweight="bold")

top_negative_mix = negative_mix.head(6).sort_values("raw_negative_count", ascending=True)
axes[2].barh(top_negative_mix.index, top_negative_mix["retention_pct"], color="#4c72b0")
axes[2].set_title("Negative Diagnosis Retention", fontweight="bold")
axes[2].set_xlabel("Retention %")

plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

display(comparison_summary)
display(negative_mix.head(10))
        """
    ),
    md_cell(
        """
## 3. Patient-Level Structure

`patient_id` repeats strongly in this dataset, so the dataset needs to be understood at patient level before any split policy is defined.
        """
    ),
    code_cell(
        """
patient_df = (
    df.groupby("patient_id")
    .agg(
        n_images=("image_id", "count"),
        positive_images=("binary_label", "sum"),
        has_melanoma=("binary_label", "max"),
        unique_targets=("binary_label", "nunique"),
    )
    .reset_index()
)
patient_df["patient_type"] = np.select(
    [
        patient_df["unique_targets"] > 1,
        patient_df["has_melanoma"] == 1,
    ],
    [
        "mixed_target_patient",
        "melanoma_only_patient",
    ],
    default="non_melanoma_only_patient",
)

patient_summary = pd.DataFrame(
    [
        {"metric": "unique_patients", "value": int(len(patient_df))},
        {"metric": "patients_with_multiple_images", "value": int((patient_df["n_images"] > 1).sum())},
        {"metric": "mixed_target_patients", "value": int((patient_df["unique_targets"] > 1).sum())},
        {"metric": "max_images_per_patient", "value": int(patient_df["n_images"].max())},
        {"metric": "median_images_per_patient", "value": float(patient_df["n_images"].median())},
    ]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(patient_df["n_images"], bins=30, kde=True, ax=axes[0], color="#2b6cb0")
axes[0].set_title("Images per Patient", fontweight="bold")
axes[0].set_xlabel("Number of images")

patient_type_counts = patient_df["patient_type"].value_counts()
axes[1].bar(patient_type_counts.index, patient_type_counts.values, color=["#dd8452", "#c53030", "#4c72b0"])
axes[1].set_title("Patient Categories", fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_ylabel("Patient count")

plt.tight_layout()
plt.savefig(FIG_DIR / "patient_profile.png", dpi=150, bbox_inches="tight")
plt.show()

display(patient_summary)
display(patient_df["n_images"].describe().round(2).to_frame(name="value"))
        """
    ),
    md_cell(
        """
## 4. Clinical Metadata Profile

The new dataset brings structured descriptors that were not available in the old segmentation-only flow.
        """
    ),
    code_cell(
        """
df["age_bucket"] = pd.cut(
    df["age_approx"],
    bins=[0, 20, 40, 60, 80, 120],
    labels=["0-20", "21-40", "41-60", "61-80", "81+"],
    include_lowest=True,
)
df["age_bucket"] = df["age_bucket"].astype("object").fillna("missing")

sex_pct = pd.crosstab(df["sex_clean"], df["target_name"], normalize="columns").mul(100).round(2)
site_pct = (
    pd.crosstab(df["site_clean"], df["target_name"], normalize="columns")
    .mul(100)
    .round(2)
    .sort_values(by="Melanoma", ascending=False)
)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

missing_metadata["missing_pct"].sort_values().plot(kind="barh", ax=axes[0, 0], color="#7a7a7a")
axes[0, 0].set_title("Missing Metadata", fontweight="bold")
axes[0, 0].set_xlabel("Missing %")

sns.boxplot(data=df, x="target_name", y="age_approx", palette=["#2b6cb0", "#c53030"], ax=axes[0, 1])
axes[0, 1].set_title("Age by Target", fontweight="bold")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Approximate age")

sex_pct.plot(kind="bar", ax=axes[1, 0], color=["#2b6cb0", "#c53030"])
axes[1, 0].set_title("Sex Distribution by Target", fontweight="bold")
axes[1, 0].tick_params(axis="x", rotation=0)
axes[1, 0].set_ylabel("% within target")

site_pct.head(6).plot(kind="barh", ax=axes[1, 1], color=["#2b6cb0", "#c53030"])
axes[1, 1].set_title("Top Anatomical Sites by Target", fontweight="bold")
axes[1, 1].set_xlabel("% within target")

plt.tight_layout()
plt.savefig(FIG_DIR / "demographic_profile.png", dpi=150, bbox_inches="tight")
plt.show()

display(sex_pct)
display(site_pct.head(10))
display(pd.crosstab(df["age_bucket"], df["target_name"], normalize="columns").mul(100).round(2))
        """
    ),
    md_cell(
        """
## 5. Diagnosis Breakdown

For negatives, most labels are `unknown`, while `nevus` remains the main named benign subgroup.
        """
    ),
    code_cell(
        """
diagnosis_counts = (
    df.groupby(["diagnosis_clean", "target_name"])
    .size()
    .rename("count")
    .reset_index()
)

diagnosis_pivot = pd.crosstab(df["diagnosis_clean"], df["target_name"]).sort_values(
    by=["Melanoma", "Non-melanoma"],
    ascending=False,
)

negative_diagnosis_effective = (
    effective_df[effective_df["binary_label"] == 0]["diagnosis_clean"]
    .value_counts()
    .rename("effective_count")
    .to_frame()
)
negative_diagnosis_effective["raw_count"] = (
    df[df["binary_label"] == 0]["diagnosis_clean"].value_counts()
)
negative_diagnosis_effective = negative_diagnosis_effective.fillna(0).astype(int)
negative_diagnosis_effective["effective_pct_of_negative"] = (
    negative_diagnosis_effective["effective_count"] / negative_diagnosis_effective["effective_count"].sum() * 100
).round(2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

diagnosis_pivot.head(8).sort_values("Non-melanoma").plot(
    kind="barh",
    ax=axes[0],
    color=["#2b6cb0", "#c53030"],
)
axes[0].set_title("Top Diagnoses in the Raw Dataset", fontweight="bold")
axes[0].set_xlabel("Count")

negative_diagnosis_effective.head(8).sort_values("effective_count").plot(
    y=["raw_count", "effective_count"],
    kind="barh",
    ax=axes[1],
    color=["#9ecae1", "#4c72b0"],
)
axes[1].set_title("Negative Diagnoses: Raw vs Effective", fontweight="bold")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig(FIG_DIR / "diagnosis_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

display(diagnosis_pivot.head(12))
display(negative_diagnosis_effective.head(12))
        """
    ),
    md_cell(
        """
## 6. Visual Inspection

Sampling images by target and by common negative diagnoses helps verify whether the new dataset is visually coherent.
        """
    ),
    code_cell(
        """
sample_groups = [
    ("Melanoma", df[df["binary_label"] == 1]),
    ("Unknown", df[df["diagnosis_clean"] == "unknown"]),
    ("Nevus", df[df["diagnosis_clean"] == "nevus"]),
]

N_SAMPLES = 4
fig, axes = plt.subplots(len(sample_groups), N_SAMPLES, figsize=(N_SAMPLES * 3.2, len(sample_groups) * 3.2))

for row_idx, (title, frame) in enumerate(sample_groups):
    samples = frame.sample(n=min(N_SAMPLES, len(frame)), random_state=SEED).reset_index(drop=True)
    for col_idx in range(N_SAMPLES):
        ax = axes[row_idx, col_idx]
        if col_idx < len(samples):
            sample_row = samples.iloc[col_idx]
            ax.imshow(Image.open(sample_row["img_path"]).convert("RGB"))
            ax.set_title(title if col_idx == 0 else sample_row["site_clean"], fontsize=10)
        ax.axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
        """
    ),
    md_cell(
        """
## 7. Image Geometry and Pixel Statistics

Unlike the old dataset, the ISIC 2020 images do not share a single resolution, so preprocessing must normalize shape explicitly.
        """
    ),
    code_cell(
        """
dimension_sample = df.sample(n=min(600, len(df)), random_state=SEED).copy()
size_records = []

for image_path in dimension_sample["img_path"]:
    with Image.open(image_path) as image_file:
        width, height = image_file.size
        size_records.append(
            {
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
            }
        )

size_df = pd.DataFrame(size_records)
size_counts = Counter(list(zip(size_df["width"], size_df["height"])))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=size_df, x="width", y="height", alpha=0.6, s=40, ax=axes[0], color="#2b6cb0")
axes[0].set_title("Sampled Image Dimensions", fontweight="bold")

sns.histplot(size_df["aspect_ratio"], bins=30, kde=True, ax=axes[1], color="#dd8452")
axes[1].set_title("Aspect Ratio Distribution", fontweight="bold")
axes[1].set_xlabel("Width / Height")

plt.tight_layout()
plt.savefig(FIG_DIR / "image_dimensions.png", dpi=150, bbox_inches="tight")
plt.show()

display(pd.DataFrame(size_counts.most_common(10), columns=["size", "count"]))
        """
    ),
    code_cell(
        """
def channel_stats(paths: list[Path], n: int = 80, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_count = min(n, len(paths))
    indices = rng.choice(len(paths), size=sample_count, replace=False)
    pixels = []

    for idx in indices:
        arr = np.array(Image.open(paths[idx]).convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        pixels.append(arr.reshape(-1, 3))

    pixels = np.vstack(pixels)
    return pixels.mean(axis=0), pixels.std(axis=0)


mel_paths = df[df["binary_label"] == 1]["img_path"].tolist()
nonmel_paths = df[df["binary_label"] == 0]["img_path"].tolist()

mel_mean, mel_std = channel_stats(mel_paths)
nonmel_mean, nonmel_std = channel_stats(nonmel_paths)

channels = ["R", "G", "B"]
x = np.arange(3)
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(x - w / 2, mel_mean, w, label="Melanoma", color="#c53030", alpha=0.9)
axes[0].bar(x + w / 2, nonmel_mean, w, label="Non-melanoma", color="#2b6cb0", alpha=0.9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(channels)
axes[0].set_title("Mean Channel Intensity")
axes[0].legend()

axes[1].bar(x - w / 2, mel_std, w, label="Melanoma", color="#c53030", alpha=0.9)
axes[1].bar(x + w / 2, nonmel_std, w, label="Non-melanoma", color="#2b6cb0", alpha=0.9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(channels)
axes[1].set_title("Channel Std Dev")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "pixel_stats.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Melanoma mean RGB: {mel_mean} | std: {mel_std}")
print(f"Non-melanoma mean RGB: {nonmel_mean} | std: {nonmel_std}")
        """
    ),
    md_cell(
        """
## 8. Summary
        """
    ),
    code_cell(
        """
raw_ratio = (df["binary_label"] == 0).sum() / (df["binary_label"] == 1).sum()
effective_ratio = (effective_df["binary_label"] == 0).sum() / (effective_df["binary_label"] == 1).sum()
mixed_patient_count = int((patient_df["unique_targets"] > 1).sum())
top_negative_diagnosis = negative_mix.index[0]
top_negative_retention = negative_mix.iloc[0]["retention_pct"]
size_mode, size_mode_count = size_counts.most_common(1)[0]
size_mode_pct = size_mode_count / len(size_df) * 100

summary_table = pd.DataFrame(
    [
        {"metric": "Raw total images", "value": len(df)},
        {"metric": "Raw melanoma", "value": int((df["binary_label"] == 1).sum())},
        {"metric": "Raw non-melanoma", "value": int((df["binary_label"] == 0).sum())},
        {"metric": "Raw imbalance ratio", "value": f"{raw_ratio:.2f} : 1"},
        {"metric": "Effective total images", "value": len(effective_df)},
        {"metric": "Effective imbalance ratio", "value": f"{effective_ratio:.2f} : 1"},
        {"metric": "Unique patients", "value": int(len(patient_df))},
        {"metric": "Mixed-target patients", "value": mixed_patient_count},
        {"metric": "Most common sampled size", "value": f"{size_mode[0]}x{size_mode[1]} ({size_mode_pct:.1f}% of sampled images)"},
        {"metric": "Dominant negative diagnosis", "value": top_negative_diagnosis},
    ]
)

display(summary_table)

print("Insights:")
print(
    f"- O dataset bruto é extremamente desbalanceado, com razão de {raw_ratio:.2f}:1 "
    f"(não melanoma:melanoma), então algum tipo de balanceamento apenas no treino é justificável."
)
print(
    f"- A repetição de pacientes é estrutural neste dataset: {len(patient_df)} pacientes únicos geram todas as {len(df)} imagens, "
    f"e {mixed_patient_count} pacientes contêm os dois valores de target, o que torna inseguro fazer split aleatório por imagem."
)
print(
    f"- O subconjunto efetivo de treino reduz o desbalanceamento para {effective_ratio:.2f}:1, preservando "
    f"{top_negative_retention:.1f}% do grupo diagnóstico negativo dominante ({top_negative_diagnosis})."
)
print(
    f"- A geometria das imagens é heterogênea; o tamanho mais comum na amostra é {size_mode[0]}x{size_mode[1]}, "
    f"mas coexistem várias resoluções de captura."
)
print(
    f"- A classe negativa é dominada por rótulos 'unknown', enquanto 'nevus' continua sendo o principal hard negative nomeado; "
    f"isso limita análises finas por subclasse, mas ainda preserva um grupo de contraste clinicamente relevante."
)
print(
    f"- Os metadados clínicos são úteis, mas incompletos, especialmente em sítio anatômico; por isso, preprocessing e modelagem "
    f"não devem depender dessas colunas totalmente preenchidas."
)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
