import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/02_preprocessing.ipynb")


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
# 02 - Preprocessing & Training-Ready Data Pipeline

This notebook converts the local ISIC 2020 dataset under `data/` into a training-ready binary melanoma pipeline.

**What this notebook does**

- validates the metadata and image layout
- creates patient-aware train / validation / test splits
- keeps validation and test close to the raw distribution
- balances only the training split by downsampling negatives
- applies deterministic image preprocessing without segmentation masks
- exports processed folders and split manifests
- computes train-only normalization statistics
- builds PyTorch `DataLoader` objects from the exported folders
        """
    ),
    code_cell(
        """
import json
import os
import random
import shutil
import warnings
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from PIL import Image
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


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
OUTPUT_DIR = ROOT_DIR / "notebooks" / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
PREP_DIR = OUTPUT_DIR / "preprocessing"
PROCESSED_DIR = DATA_DIR / "processed"

for directory in [FIG_DIR, PREP_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(PREP_DIR / ".mplconfig"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TRAIN_NON_MELANOMA_TO_MELANOMA_RATIO = 6.0
MIN_STRATUM_COUNT = 5
HAIR_KERNEL_SIZE = 17
HAIR_THRESHOLD = 10
HAIR_INPAINT_RADIUS = 1
BATCH_SIZE = 32
NUM_WORKERS = 0
HARD_NEGATIVE_DIAGNOSIS = "nevus"
HARD_NEGATIVE_MULTIPLIER = 1.5
NORMALIZATION_MODE = "dataset"  # choose between "dataset" and "imagenet"
EXPORT_BASELINE_NAME = "without_augmentation"
EXPORT_AUGMENTED_NAME = "with_augmentation"
AUGMENTED_EXPORT_COPIES = 2
INCLUDE_ORIGINAL_IN_AUGMENTED_TRAIN = True
EXPORT_IMAGE_FORMAT = "png"
BINARY_CLASS_DIRS = {0: "0_non_melanoma", 1: "1_melanoma"}
TARGET_NAMES = {0: "non_melanoma", 1: "melanoma"}

config = {
    "seed": SEED,
    "target_size": TARGET_SIZE,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "train_non_melanoma_to_melanoma_ratio": TRAIN_NON_MELANOMA_TO_MELANOMA_RATIO,
    "min_stratum_count": MIN_STRATUM_COUNT,
    "hair_kernel_size": HAIR_KERNEL_SIZE,
    "hair_threshold": HAIR_THRESHOLD,
    "hair_inpaint_radius": HAIR_INPAINT_RADIUS,
    "batch_size": BATCH_SIZE,
    "hard_negative_diagnosis": HARD_NEGATIVE_DIAGNOSIS,
    "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER,
    "normalization_mode": NORMALIZATION_MODE,
    "processed_dir": str(PROCESSED_DIR),
    "export_baseline_name": EXPORT_BASELINE_NAME,
    "export_augmented_name": EXPORT_AUGMENTED_NAME,
    "augmented_export_copies": AUGMENTED_EXPORT_COPIES,
    "include_original_in_augmented_train": INCLUDE_ORIGINAL_IN_AUGMENTED_TRAIN,
    "export_image_format": EXPORT_IMAGE_FORMAT,
    "binary_class_dirs": BINARY_CLASS_DIRS,
}

print(json.dumps(config, indent=2))
        """
    ),
    md_cell(
        """
## 1. Load the Raw Metadata

The new dataset already provides the binary label, but it also includes patient identifiers and auxiliary descriptors that affect how the pipeline should split and sample the data.
        """
    ),
    code_cell(
        """
def clean_category(series: pd.Series, missing_label: str = "missing") -> pd.Series:
    return series.fillna(missing_label).astype(str).str.strip().replace("", missing_label).str.lower()


raw_df = pd.read_csv(META_PATH).copy()
raw_df = raw_df.rename(columns={"image_name": "image_id"})

required_columns = {
    "image_id",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "diagnosis",
    "benign_malignant",
    "target",
}
missing_columns = required_columns - set(raw_df.columns)
assert not missing_columns, f"Metadata is missing required columns: {sorted(missing_columns)}"

raw_df["binary_label"] = raw_df["target"].astype(int)
raw_df["target_name"] = raw_df["binary_label"].map(TARGET_NAMES)
raw_df["diagnosis_clean"] = clean_category(raw_df["diagnosis"])
raw_df["sex_clean"] = clean_category(raw_df["sex"])
raw_df["site_clean"] = clean_category(raw_df["anatom_site_general_challenge"])
raw_df["img_path"] = raw_df["image_id"].map(lambda image_id: IMG_DIR / f"{image_id}.jpg")
raw_df["image_exists"] = raw_df["img_path"].map(Path.exists)

raw_integrity = pd.DataFrame(
    [
        {"metric": "rows", "value": len(raw_df)},
        {"metric": "unique_images", "value": int(raw_df["image_id"].nunique())},
        {"metric": "duplicate_images", "value": int(raw_df["image_id"].duplicated().sum())},
        {"metric": "unique_patients", "value": int(raw_df["patient_id"].nunique())},
        {"metric": "missing_images", "value": int((~raw_df["image_exists"]).sum())},
        {"metric": "melanoma_images", "value": int((raw_df["binary_label"] == 1).sum())},
        {"metric": "non_melanoma_images", "value": int((raw_df["binary_label"] == 0).sum())},
    ]
)

raw_label_summary = (
    raw_df["binary_label"]
    .value_counts()
    .sort_index()
    .rename(index=TARGET_NAMES)
    .rename_axis("target")
    .reset_index(name="count")
)
raw_label_summary["pct"] = (raw_label_summary["count"] / len(raw_df) * 100).round(2)

display(raw_integrity)
display(raw_label_summary)
raw_df.head()
        """
    ),
    md_cell(
        """
## 2. Patient-Aware Split First, Balance Only the Training Split

This dataset repeats `patient_id` heavily and even contains mixed-target patients. The split is therefore created at patient level before any class balancing is applied.
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


def split_patients(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_df = (
        df.groupby("patient_id")
        .agg(
            patient_target=("binary_label", "max"),
            n_images=("image_id", "count"),
            unique_targets=("binary_label", "nunique"),
        )
        .reset_index()
    )

    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=(1.0 - train_ratio),
        stratify=patient_df["patient_target"],
        random_state=seed,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test_ratio,
        stratify=temp_patients["patient_target"],
        random_state=seed,
    )

    patient_split_map = {}
    for split_name, patient_frame in [
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients),
    ]:
        for patient_id in patient_frame["patient_id"]:
            patient_split_map[patient_id] = split_name

    split_df = df.copy()
    split_df["split"] = split_df["patient_id"].map(patient_split_map)
    assert split_df["split"].notna().all(), "Some rows were not assigned to a split."

    return split_df, patient_df


def build_balanced_training_subset(df: pd.DataFrame, non_melanoma_ratio: float, seed: int) -> pd.DataFrame:
    melanoma_df = df[df["binary_label"] == 1].copy()
    non_melanoma_df = df[df["binary_label"] == 0].copy()

    target_non_melanoma = min(
        len(non_melanoma_df),
        int(round(len(melanoma_df) * non_melanoma_ratio)),
    )
    if target_non_melanoma <= 0:
        raise ValueError("The requested non-melanoma ratio produced an empty training selection.")

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

    melanoma_df["selection_source"] = "train_kept_all_melanoma"
    selected_non_melanoma["selection_source"] = "train_downsampled_non_melanoma"

    balanced_train_df = (
        pd.concat([melanoma_df, selected_non_melanoma], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return balanced_train_df


split_df, patient_df = split_patients(raw_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)

train_raw_df = split_df[split_df["split"] == "train"].copy().reset_index(drop=True)
val_df = split_df[split_df["split"] == "val"].copy().reset_index(drop=True)
test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)

train_balanced_df = build_balanced_training_subset(
    train_raw_df,
    TRAIN_NON_MELANOMA_TO_MELANOMA_RATIO,
    SEED,
)
val_df["selection_source"] = "validation_full_distribution"
test_df["selection_source"] = "test_full_distribution"

model_df = pd.concat([train_balanced_df, val_df, test_df], axis=0).reset_index(drop=True)

patient_split_sets = {
    split_name: set(frame["patient_id"].unique())
    for split_name, frame in [("train", train_raw_df), ("val", val_df), ("test", test_df)]
}
assert patient_split_sets["train"].isdisjoint(patient_split_sets["val"])
assert patient_split_sets["train"].isdisjoint(patient_split_sets["test"])
assert patient_split_sets["val"].isdisjoint(patient_split_sets["test"])

split_summary = (
    split_df.groupby(["split", "binary_label"])
    .size()
    .rename("count")
    .reset_index()
)
split_summary["target_name"] = split_summary["binary_label"].map(TARGET_NAMES)

patient_split_summary = (
    split_df.groupby("split")["patient_id"]
    .nunique()
    .rename("patient_count")
    .reset_index()
)

train_selection_summary = pd.DataFrame(
    {
        "raw_train_count": train_raw_df["binary_label"].value_counts().sort_index(),
        "balanced_train_count": train_balanced_df["binary_label"].value_counts().sort_index(),
    }
).rename(index=TARGET_NAMES)
train_selection_summary["retention_pct"] = (
    train_selection_summary["balanced_train_count"] / train_selection_summary["raw_train_count"] * 100
).round(2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(
    data=split_summary,
    x="split",
    y="count",
    hue="target_name",
    palette={"non_melanoma": "#2b6cb0", "melanoma": "#c53030"},
    ax=axes[0],
)
axes[0].set_title("Raw Split Distribution", fontweight="bold")

sns.barplot(data=patient_split_summary, x="split", y="patient_count", color="#4c72b0", ax=axes[1])
axes[1].set_title("Patients per Split", fontweight="bold")

train_selection_summary[["raw_train_count", "balanced_train_count"]].plot(
    kind="bar",
    ax=axes[2],
    color=["#9ecae1", "#2b6cb0"],
)
axes[2].set_title("Training Selection Policy", fontweight="bold")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(FIG_DIR / "preprocessing_split_summary.png", dpi=150, bbox_inches="tight")
plt.show()

display(patient_split_summary)
display(train_selection_summary)
        """
    ),
    md_cell(
        """
## 3. Integrity Checks on the Final Modeling Dataset

At this point, the modeling dataset is defined as:

- balanced training split
- raw validation split
- raw test split
        """
    ),
    code_cell(
        """
inspection_df = model_df.sample(n=min(800, len(model_df)), random_state=SEED).copy()
size_records = []
missing_images = []

for row in inspection_df.itertuples():
    if not row.img_path.exists():
        missing_images.append(str(row.img_path))
        continue

    with Image.open(row.img_path) as image_file:
        width, height = image_file.size
        size_records.append(
            {
                "split": row.split,
                "binary_label": row.binary_label,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
            }
        )

size_df = pd.DataFrame(size_records)

integrity_summary = pd.DataFrame(
    [
        {"metric": "rows_in_model_dataset", "value": len(model_df)},
        {"metric": "unique_images", "value": int(model_df["image_id"].nunique())},
        {"metric": "missing_images_in_sample", "value": len(missing_images)},
        {"metric": "train_rows", "value": int(len(train_balanced_df))},
        {"metric": "val_rows", "value": int(len(val_df))},
        {"metric": "test_rows", "value": int(len(test_df))},
    ]
)

assert len(missing_images) == 0, "There are missing images referenced by the final modeling dataset."
assert model_df["image_id"].nunique() == len(model_df), "Duplicate image ids found in the final modeling dataset."

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=size_df, x="width", y="height", hue="split", alpha=0.6, s=40, ax=axes[0])
axes[0].set_title("Image Dimensions Across Splits", fontweight="bold")

sns.histplot(data=size_df, x="aspect_ratio", hue="split", bins=30, kde=True, ax=axes[1], element="step")
axes[1].set_title("Aspect Ratio Distribution", fontweight="bold")
axes[1].set_xlabel("Width / Height")

plt.tight_layout()
plt.savefig(FIG_DIR / "preprocessing_integrity.png", dpi=150, bbox_inches="tight")
plt.show()

display(integrity_summary)
display(size_df[["width", "height"]].value_counts().head(10).to_frame(name="count"))
        """
    ),
    md_cell(
        """
## 4. Deterministic Image Preprocessing

Because the new dataset does not include segmentation masks, preprocessing is image-centric rather than lesion-mask-centric:

1. load RGB image
2. attenuate thin dark hair artifacts
3. pad the image to a square canvas
4. resize to `224 x 224`
        """
    ),
    code_cell(
        """
def load_rgb_image(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def remove_hair_artifacts(
    image_rgb: np.ndarray,
    kernel_size: int = HAIR_KERNEL_SIZE,
    threshold: int = HAIR_THRESHOLD,
    inpaint_radius: int = HAIR_INPAINT_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(image_rgb, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned, hair_mask


def pad_to_square(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    side = max(height, width)

    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left

    if array.ndim == 3:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

    return np.pad(array, pad_width=pad_width, mode="edge")


def preprocess_image(
    image_path: Path,
    target_size: int = TARGET_SIZE,
    remove_hair: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    image = load_rgb_image(image_path)

    if remove_hair:
        image, hair_mask = remove_hair_artifacts(image)
    else:
        hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    original_shape = image.shape[:2]
    squared = pad_to_square(image)
    resized = cv2.resize(squared, (target_size, target_size), interpolation=cv2.INTER_AREA)

    debug = {
        "original_shape": original_shape,
        "square_shape": squared.shape[:2],
        "final_shape": resized.shape[:2],
        "hair_pixels_detected": int((hair_mask > 0).sum()),
    }
    return resized, hair_mask, debug
        """
    ),
    code_cell(
        """
preview_specs = [
    ("melanoma", train_balanced_df[train_balanced_df["binary_label"] == 1]),
    ("nevus", train_balanced_df[train_balanced_df["diagnosis_clean"] == "nevus"]),
    ("unknown", train_balanced_df[train_balanced_df["diagnosis_clean"] == "unknown"]),
]

preview_rows = []
for label, frame in preview_specs:
    if len(frame) > 0:
        preview_rows.append((label, frame.sample(1, random_state=SEED).iloc[0]))

fig, axes = plt.subplots(len(preview_rows), 3, figsize=(12, 4 * len(preview_rows)))
if len(preview_rows) == 1:
    axes = np.array([axes])

for row_idx, (label, row) in enumerate(preview_rows):
    raw_image = load_rgb_image(row["img_path"])
    processed_image, hair_mask, debug = preprocess_image(
        row["img_path"],
        target_size=TARGET_SIZE,
        remove_hair=True,
    )

    axes[row_idx, 0].imshow(raw_image)
    axes[row_idx, 0].set_title(f"{label} - raw")
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(hair_mask, cmap="gray")
    axes[row_idx, 1].set_title("hair mask")
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(processed_image)
    axes[row_idx, 2].set_title(f"processed {debug['final_shape'][0]}x{debug['final_shape'][1]}")
    axes[row_idx, 2].axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "preprocessing_steps.png", dpi=150, bbox_inches="tight")
plt.show()

uniformity_checks = []
for _, row in model_df.sample(min(128, len(model_df)), random_state=SEED).iterrows():
    processed_image, _, _ = preprocess_image(
        row["img_path"],
        target_size=TARGET_SIZE,
        remove_hair=True,
    )
    uniformity_checks.append(processed_image.shape)

assert all(image_shape == (TARGET_SIZE, TARGET_SIZE, 3) for image_shape in uniformity_checks)
print(f"All sampled preprocessed images have fixed shape: {uniformity_checks[0]}.")
        """
    ),
    md_cell(
        """
## 5. Train-Only Normalization Statistics

Normalization is measured only on the balanced training split after deterministic preprocessing.
        """
    ),
    code_cell(
        """
def compute_normalization_stats(frame: pd.DataFrame, sample_size: int = 1024) -> tuple[list[float], list[float], int]:
    sample_frame = frame.sample(n=min(sample_size, len(frame)), random_state=SEED)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for image_path in sample_frame["img_path"]:
        image, _, _ = preprocess_image(image_path, target_size=TARGET_SIZE, remove_hair=True)
        pixels = image.astype(np.float32).reshape(-1, 3) / 255.0
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels ** 2).sum(axis=0)
        pixel_count += len(pixels)

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sq_sum / pixel_count - mean ** 2)
    return mean.round(6).tolist(), std.round(6).tolist(), len(sample_frame)


train_mean, train_std, normalization_sample_count = compute_normalization_stats(train_balanced_df)
normalization_stats = {
    "mean": train_mean,
    "std": train_std,
    "sample_count": normalization_sample_count,
    "target_size": TARGET_SIZE,
}

(PREP_DIR / "normalization_stats.json").write_text(
    json.dumps(normalization_stats, indent=2),
    encoding="utf-8",
)
(PREP_DIR / "preprocessing_config.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

display(pd.DataFrame({"channel": ["R", "G", "B"], "mean": train_mean, "std": train_std}))
print(f"Saved normalization stats to: {(PREP_DIR / 'normalization_stats.json').resolve()}")
        """
    ),
    md_cell(
        """
## 6. Export the Processed Dataset

Two exports are created:

1. `without_augmentation`: deterministic preprocessing only
2. `with_augmentation`: deterministic preprocessing plus offline augmentation copies in the training split
        """
    ),
    code_cell(
        """
try:
    import albumentations as A
except ImportError as exc:
    raise ImportError("Albumentations is required for the export section of this notebook.") from exc


offline_aug_export_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.08,
            rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
    ]
)


def save_rgb_image(image_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array.astype(np.uint8)).save(output_path)


def prepare_export_root(export_root: Path) -> None:
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)


def export_split(
    frame: pd.DataFrame,
    split_name: str,
    export_root: Path,
    augmenter=None,
    augmented_copies: int = 0,
    include_original: bool = True,
) -> pd.DataFrame:
    manifest_rows = []

    for row in frame.itertuples():
        base_image, _, debug = preprocess_image(row.img_path, target_size=TARGET_SIZE, remove_hair=True)
        class_dir = export_root / split_name / BINARY_CLASS_DIRS[row.binary_label]

        if include_original:
            base_output_path = class_dir / f"{row.image_id}.{EXPORT_IMAGE_FORMAT}"
            save_rgb_image(base_image, base_output_path)
            manifest_rows.append(
                {
                    "image_id": row.image_id,
                    "patient_id": row.patient_id,
                    "split": split_name,
                    "binary_label": row.binary_label,
                    "target_name": TARGET_NAMES[row.binary_label],
                    "diagnosis_clean": row.diagnosis_clean,
                    "sex_clean": row.sex_clean,
                    "site_clean": row.site_clean,
                    "age_approx": row.age_approx,
                    "selection_source": row.selection_source,
                    "is_augmented": False,
                    "augmentation_copy": 0,
                    "export_path": str(base_output_path),
                    "original_height": debug["original_shape"][0],
                    "original_width": debug["original_shape"][1],
                }
            )

        if split_name == "train" and augmenter is not None and augmented_copies > 0:
            for copy_idx in range(1, augmented_copies + 1):
                transformed = augmenter(image=base_image)
                aug_image = transformed["image"]
                aug_output_path = class_dir / f"{row.image_id}_aug{copy_idx}.{EXPORT_IMAGE_FORMAT}"
                save_rgb_image(aug_image, aug_output_path)
                manifest_rows.append(
                    {
                        "image_id": row.image_id,
                        "patient_id": row.patient_id,
                        "split": split_name,
                        "binary_label": row.binary_label,
                        "target_name": TARGET_NAMES[row.binary_label],
                        "diagnosis_clean": row.diagnosis_clean,
                        "sex_clean": row.sex_clean,
                        "site_clean": row.site_clean,
                        "age_approx": row.age_approx,
                        "selection_source": row.selection_source,
                        "is_augmented": True,
                        "augmentation_copy": copy_idx,
                        "export_path": str(aug_output_path),
                        "original_height": debug["original_shape"][0],
                        "original_width": debug["original_shape"][1],
                    }
                )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(export_root / f"{split_name}_manifest.csv", index=False)
    return manifest_df


def export_experiment(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    export_root: Path,
    train_augmenter=None,
    train_augmented_copies: int = 0,
    include_original_train: bool = True,
) -> dict[str, pd.DataFrame]:
    prepare_export_root(export_root)

    manifests = {
        "train": export_split(
            train_frame,
            "train",
            export_root,
            augmenter=train_augmenter,
            augmented_copies=train_augmented_copies,
            include_original=include_original_train,
        ),
        "val": export_split(val_frame, "val", export_root, include_original=True),
        "test": export_split(test_frame, "test", export_root, include_original=True),
    }

    pd.concat(manifests.values(), axis=0).to_csv(export_root / "full_manifest.csv", index=False)
    return manifests


baseline_root = PROCESSED_DIR / EXPORT_BASELINE_NAME
augmented_root = PROCESSED_DIR / EXPORT_AUGMENTED_NAME

baseline_manifests = export_experiment(
    train_balanced_df,
    val_df,
    test_df,
    baseline_root,
    train_augmenter=None,
    train_augmented_copies=0,
    include_original_train=True,
)

augmented_manifests = export_experiment(
    train_balanced_df,
    val_df,
    test_df,
    augmented_root,
    train_augmenter=offline_aug_export_transform,
    train_augmented_copies=AUGMENTED_EXPORT_COPIES,
    include_original_train=INCLUDE_ORIGINAL_IN_AUGMENTED_TRAIN,
)

export_summary = pd.DataFrame(
    [
        {
            "experiment": "without_augmentation",
            "train_images": len(baseline_manifests["train"]),
            "val_images": len(baseline_manifests["val"]),
            "test_images": len(baseline_manifests["test"]),
        },
        {
            "experiment": "with_augmentation",
            "train_images": len(augmented_manifests["train"]),
            "val_images": len(augmented_manifests["val"]),
            "test_images": len(augmented_manifests["test"]),
        },
    ]
)

display(export_summary)
print(f"Baseline export root : {baseline_root.resolve()}")
print(f"Augmented export root: {augmented_root.resolve()}")
        """
    ),
    md_cell(
        """
## 7. Build PyTorch DataLoaders

The loaders below read directly from the exported folders so later training notebooks can reuse the same on-disk splits.
        """
    ),
    code_cell(
        """
try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms
except ImportError as exc:
    raise ImportError("PyTorch and torchvision are required for the loader section of this notebook.") from exc


torch.manual_seed(SEED)

if NORMALIZATION_MODE == "dataset":
    normalization_mean = train_mean
    normalization_std = train_std
else:
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose(
    [
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std),
    ]
)

baseline_train_dataset = datasets.ImageFolder(baseline_root / "train", transform=inference_transform)
baseline_val_dataset = datasets.ImageFolder(baseline_root / "val", transform=inference_transform)
baseline_test_dataset = datasets.ImageFolder(baseline_root / "test", transform=inference_transform)

augmented_train_dataset = datasets.ImageFolder(augmented_root / "train", transform=inference_transform)

baseline_train_manifest = pd.read_csv(baseline_root / "train_manifest.csv")
baseline_train_manifest["file_name"] = baseline_train_manifest["export_path"].map(lambda path: Path(path).name)

positive_count = max(int((baseline_train_manifest["binary_label"] == 1).sum()), 1)
negative_count = max(int((baseline_train_manifest["binary_label"] == 0).sum()), 1)
positive_weight = negative_count / positive_count

weight_map = {}
for row in baseline_train_manifest.itertuples():
    weight = positive_weight if row.binary_label == 1 else 1.0
    if row.binary_label == 0 and row.diagnosis_clean == HARD_NEGATIVE_DIAGNOSIS:
        weight *= HARD_NEGATIVE_MULTIPLIER
    weight_map[row.file_name] = weight

sample_weights = [weight_map[Path(sample_path).name] for sample_path, _ in baseline_train_dataset.samples]
train_sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)

baseline_train_loader = DataLoader(
    baseline_train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
)
baseline_val_loader = DataLoader(
    baseline_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
baseline_test_loader = DataLoader(
    baseline_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
augmented_train_loader = DataLoader(
    augmented_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

loader_summary = pd.DataFrame(
    [
        {"loader": "baseline_train", "dataset_size": len(baseline_train_dataset)},
        {"loader": "baseline_val", "dataset_size": len(baseline_val_dataset)},
        {"loader": "baseline_test", "dataset_size": len(baseline_test_dataset)},
        {"loader": "augmented_train", "dataset_size": len(augmented_train_dataset)},
    ]
)
display(loader_summary)

images, labels = next(iter(baseline_train_loader))
preview_count = min(8, len(images))
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for idx in range(preview_count):
    image = images[idx].permute(1, 2, 0).numpy()
    image = image * np.array(normalization_std) + np.array(normalization_mean)
    image = np.clip(image, 0, 1)
    axes[idx].imshow(image)
    axes[idx].set_title(TARGET_NAMES[int(labels[idx].item())])
    axes[idx].axis("off")

for idx in range(preview_count, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "batch_sanity.png", dpi=150, bbox_inches="tight")
plt.show()
        """
    ),
    md_cell(
        """
## 8. Recommended Use in the Training Notebook

Training can now read directly from:

1. `data/processed/without_augmentation/train`, `val`, `test`
2. `data/processed/with_augmentation/train`, `val`, `test`

Each experiment root also contains `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`, and `full_manifest.csv`.
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
