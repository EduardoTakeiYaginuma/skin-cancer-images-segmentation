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

This notebook converts the HAM10000 segmentation dataset into a training-ready binary melanoma pipeline.

**What this notebook does**

- validates metadata, images, and masks end-to-end
- guarantees a fixed output size for every sample
- applies lesion-centric preprocessing using the segmentation mask
- removes hair artifacts with a classical inpainting step
- creates reproducible train / validation / test splits
- computes train-set normalization statistics
- prepares two experimental tracks:
  - `baseline`: deterministic preprocessing without augmentation
  - `augmented`: same preprocessing plus stochastic data augmentation
- builds PyTorch `DataLoader` objects for both tracks from the exported folders

The split membership is the same in both experiments. What changes is only the transform applied to the training set, which is the correct way to compare whether augmentation helps or hurts.
        """
    ),
    code_cell(
        """
import os
import json
import random
import warnings
from collections import Counter
from pathlib import Path

def resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "docs").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root containing 'data/' and 'docs/'.")


ROOT_DIR = resolve_repo_root()
OUTPUT_DIR = ROOT_DIR / "notebooks" / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
PREP_DIR = OUTPUT_DIR / "preprocessing"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

for directory in [FIG_DIR, PREP_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(PREP_DIR / ".mplconfig"))

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from PIL import Image
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = ROOT_DIR / "data"
IMG_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"
META_PATH = DATA_DIR / "metadata.csv"
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

TARGET_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MASK_THRESHOLD = 127
BBOX_MARGIN_RATIO = 0.10
HAIR_KERNEL_SIZE = 17
HAIR_THRESHOLD = 10
HAIR_INPAINT_RADIUS = 1
BATCH_SIZE = 32
NUM_WORKERS = 0
HARD_NEGATIVE_LABEL = "NV"
HARD_NEGATIVE_MULTIPLIER = 2.0
NORMALIZATION_MODE = "dataset"  # choose between "dataset" and "imagenet"
EXPORT_BASELINE_NAME = "without_augmentation"
EXPORT_AUGMENTED_NAME = "with_augmentation"
AUGMENTED_EXPORT_COPIES = 2
INCLUDE_ORIGINAL_IN_AUGMENTED_TRAIN = True
EXPORT_IMAGE_FORMAT = "png"
BINARY_CLASS_DIRS = {0: "0_non_melanoma", 1: "1_melanoma"}

config = {
    "seed": SEED,
    "target_size": TARGET_SIZE,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "mask_threshold": MASK_THRESHOLD,
    "bbox_margin_ratio": BBOX_MARGIN_RATIO,
    "hair_kernel_size": HAIR_KERNEL_SIZE,
    "hair_threshold": HAIR_THRESHOLD,
    "hair_inpaint_radius": HAIR_INPAINT_RADIUS,
    "batch_size": BATCH_SIZE,
    "hard_negative_label": HARD_NEGATIVE_LABEL,
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
## 1. Load Metadata and Build Binary Labels

We keep the original 7-class annotation to preserve the hard-negative subclasses, but the training target is binary:

- `1` = melanoma (`MEL`)
- `0` = every other class
        """
    ),
    code_cell(
        """
df = pd.read_csv(META_PATH).copy()

assert set(CLASSES).issubset(df.columns), "The metadata file is missing one or more diagnostic columns."
assert (df[CLASSES].sum(axis=1) == 1).all(), "Each row must belong to exactly one original class."

df["label"] = df[CLASSES].idxmax(axis=1)
df["binary_label"] = (df["label"] == "MEL").astype(int)
df["img_path"] = df["image"].map(lambda image_id: IMG_DIR / f"{image_id}.jpg")
df["mask_path"] = df["image"].map(lambda image_id: MASK_DIR / f"{image_id}.png")

label_summary = (
    df["label"]
    .value_counts()
    .rename_axis("label")
    .reset_index(name="count")
)
label_summary["pct"] = (label_summary["count"] / len(df) * 100).round(2)

binary_summary = (
    df["binary_label"]
    .value_counts()
    .sort_index()
    .rename(index={0: "non_melanoma", 1: "melanoma"})
    .rename_axis("binary_label")
    .reset_index(name="count")
)
binary_summary["pct"] = (binary_summary["count"] / len(df) * 100).round(2)

display(df.head())
display(label_summary)
display(binary_summary)
        """
    ),
    md_cell(
        """
## 2. Integrity Checks

Before designing the pipeline, verify that:

- every metadata row maps to an image and a mask
- there are no duplicated image ids
- the raw image and mask resolutions are consistent
- the masks contain valid lesion coverage information
        """
    ),
    code_cell(
        """
def load_mask_array(mask_path: Path) -> np.ndarray:
    return (np.array(Image.open(mask_path).convert("L")) > MASK_THRESHOLD).astype(np.uint8)


image_sizes = Counter()
mask_sizes = Counter()
mask_coverages = []
bbox_height_ratios = []
bbox_width_ratios = []

missing_images = []
missing_masks = []

for idx, row in df.iterrows():
    img_path = row["img_path"]
    mask_path = row["mask_path"]

    if not img_path.exists():
        missing_images.append(str(img_path))
        continue
    if not mask_path.exists():
        missing_masks.append(str(mask_path))
        continue

    with Image.open(img_path) as image_file:
        image_sizes[image_file.size] += 1

    mask = load_mask_array(mask_path)
    mask_sizes[(mask.shape[1], mask.shape[0])] += 1
    mask_coverages.append(mask.mean())

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        bbox_width_ratios.append(0.0)
        bbox_height_ratios.append(0.0)
    else:
        bbox_width_ratios.append((xs.max() - xs.min() + 1) / mask.shape[1])
        bbox_height_ratios.append((ys.max() - ys.min() + 1) / mask.shape[0])

df["mask_coverage"] = mask_coverages
df["bbox_width_ratio"] = bbox_width_ratios
df["bbox_height_ratio"] = bbox_height_ratios

integrity_summary = pd.DataFrame(
    {
        "metric": [
            "rows_in_metadata",
            "unique_image_ids",
            "missing_images",
            "missing_masks",
            "distinct_image_sizes",
            "distinct_mask_sizes",
        ],
        "value": [
            len(df),
            df["image"].nunique(),
            len(missing_images),
            len(missing_masks),
            dict(image_sizes),
            dict(mask_sizes),
        ],
    }
)

coverage_summary = (
    df.groupby("label")["mask_coverage"]
    .agg(["mean", "median", "min", "max"])
    .round(4)
    .sort_index()
)

display(integrity_summary)
display(coverage_summary)

assert len(missing_images) == 0, "There are missing images referenced in the metadata."
assert len(missing_masks) == 0, "There are missing masks referenced in the metadata."
assert df["image"].nunique() == len(df), "Duplicated image ids found in metadata."
assert len(image_sizes) == 1, "The raw images do not all share the same size."
assert len(mask_sizes) == 1, "The raw masks do not all share the same size."

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["mask_coverage"], bins=40, kde=True, ax=axes[0], color="#2b6cb0")
axes[0].set_title("Mask Coverage Distribution")
axes[0].set_xlabel("Lesion coverage ratio")

sns.scatterplot(
    data=df.sample(1500, random_state=SEED),
    x="bbox_width_ratio",
    y="bbox_height_ratio",
    hue="binary_label",
    palette={0: "#2b6cb0", 1: "#c53030"},
    alpha=0.7,
    ax=axes[1],
)
axes[1].set_title("Bounding Box Ratio by Sample")
axes[1].set_xlabel("Bounding box width / image width")
axes[1].set_ylabel("Bounding box height / image height")

plt.tight_layout()
plt.savefig(FIG_DIR / "preprocessing_integrity.png", dpi=150, bbox_inches="tight")
plt.show()
        """
    ),
    md_cell(
        """
## 3. Deterministic Preprocessing Strategy

The exploratory analysis showed that the raw images already share one resolution (`600 x 450`), but the training pipeline still needs a deterministic preprocessing stage.

The chosen strategy is:

1. load RGB image and binary mask
2. remove thin dark hair artifacts with a classical black-hat + inpainting step
3. crop around the lesion mask with a safety margin
4. pad to square using edge padding
5. resize to a fixed `224 x 224`

This ensures every downstream model receives tensors with identical spatial dimensions while keeping the lesion centered.
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


def compute_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        height, width = mask.shape
        return 0, 0, width, height
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float = BBOX_MARGIN_RATIO,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0
    box_height = y1 - y0
    margin = max(int(max(box_width, box_height) * margin_ratio), 1)

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)
    return x0, y0, x1, y1


def crop_array(array: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return array[y0:y1, x0:x1]


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


def preprocess_image_and_mask(
    image_path: Path,
    mask_path: Path,
    target_size: int = TARGET_SIZE,
    remove_hair: bool = True,
    crop_to_mask: bool = True,
    apply_background_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    image = load_rgb_image(image_path)
    mask = load_mask_array(mask_path)

    if remove_hair:
        image, hair_mask = remove_hair_artifacts(image)
    else:
        hair_mask = np.zeros(mask.shape, dtype=np.uint8)

    original_shape = image.shape[:2]
    bbox = compute_mask_bbox(mask)

    if crop_to_mask:
        bbox = expand_bbox(bbox, width=image.shape[1], height=image.shape[0])
        image = crop_array(image, bbox)
        mask = crop_array(mask, bbox)

    if apply_background_mask:
        image = image * mask[..., None]

    image = pad_to_square(image)
    mask = pad_to_square(mask)

    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)

    debug = {
        "original_shape": original_shape,
        "final_shape": image.shape[:2],
        "bbox": bbox,
        "mask_coverage_before_crop": float(load_mask_array(mask_path).mean()),
        "mask_coverage_after_resize": float(mask.mean()),
        "hair_pixels_detected": int((hair_mask > 0).sum()),
    }
    return image, mask, debug
        """
    ),
    code_cell(
        """
preview_rows = []
for label in ["MEL", "NV", "BKL"]:
    preview_rows.append(df[df["label"] == label].sample(1, random_state=SEED).iloc[0])

fig, axes = plt.subplots(len(preview_rows), 4, figsize=(16, 4 * len(preview_rows)))

for row_idx, row in enumerate(preview_rows):
    raw_image = load_rgb_image(row["img_path"])
    raw_mask = load_mask_array(row["mask_path"])
    hair_removed, hair_mask = remove_hair_artifacts(raw_image)
    processed_image, processed_mask, debug = preprocess_image_and_mask(
        row["img_path"],
        row["mask_path"],
        target_size=TARGET_SIZE,
        remove_hair=True,
        crop_to_mask=True,
        apply_background_mask=False,
    )

    axes[row_idx, 0].imshow(raw_image)
    axes[row_idx, 0].set_title(f"{row['label']} - raw")
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(hair_mask, cmap="gray")
    axes[row_idx, 1].set_title("hair mask")
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(hair_removed)
    axes[row_idx, 2].contour(raw_mask, levels=[0.5], colors="lime", linewidths=1)
    axes[row_idx, 2].set_title("hair removed + lesion contour")
    axes[row_idx, 2].axis("off")

    axes[row_idx, 3].imshow(processed_image)
    axes[row_idx, 3].set_title(f"processed {debug['final_shape'][0]}x{debug['final_shape'][1]}")
    axes[row_idx, 3].axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "preprocessing_steps.png", dpi=150, bbox_inches="tight")
plt.show()
        """
    ),
    code_cell(
        """
uniformity_checks = []
for _, row in df.sample(256, random_state=SEED).iterrows():
    processed_image, processed_mask, _ = preprocess_image_and_mask(
        row["img_path"],
        row["mask_path"],
        target_size=TARGET_SIZE,
        remove_hair=True,
        crop_to_mask=True,
        apply_background_mask=False,
    )
    uniformity_checks.append((processed_image.shape, processed_mask.shape))

assert all(image_shape == (TARGET_SIZE, TARGET_SIZE, 3) for image_shape, _ in uniformity_checks)
assert all(mask_shape == (TARGET_SIZE, TARGET_SIZE) for _, mask_shape in uniformity_checks)

print(
    f"All sampled preprocessed images and masks have fixed shapes: "
    f"{uniformity_checks[0][0]} for images and {uniformity_checks[0][1]} for masks."
)
        """
    ),
    md_cell(
        """
## 4. Reproducible Train / Validation / Test Split

We stratify on the binary target so that melanoma prevalence stays stable across splits.

- train: `70%`
- validation: `15%`
- test: `15%`
        """
    ),
    code_cell(
        """
train_df, temp_df = train_test_split(
    df,
    test_size=(1.0 - TRAIN_RATIO),
    stratify=df["binary_label"],
    random_state=SEED,
)

relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
val_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_ratio,
    stratify=temp_df["binary_label"],
    random_state=SEED,
)

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split_df["split"] = split_name

split_sets = {
    "train": set(train_df["image"]),
    "val": set(val_df["image"]),
    "test": set(test_df["image"]),
}

assert split_sets["train"].isdisjoint(split_sets["val"])
assert split_sets["train"].isdisjoint(split_sets["test"])
assert split_sets["val"].isdisjoint(split_sets["test"])

split_summary = (
    pd.concat([train_df, val_df, test_df], axis=0)
    .groupby("split")["binary_label"]
    .agg(total="size", melanoma="sum")
    .reset_index()
)
split_summary["non_melanoma"] = split_summary["total"] - split_summary["melanoma"]
split_summary["melanoma_pct"] = (split_summary["melanoma"] / split_summary["total"] * 100).round(2)

display(split_summary)
        """
    ),
    md_cell(
        """
## 5. Train-Only Statistics and Sampling Weights

Normalization statistics must be computed only on the training split. We also create a weighted sampling strategy that:

- compensates the melanoma class imbalance
- boosts `NV` samples because they are the main hard negatives
        """
    ),
    code_cell(
        """
def compute_channel_stats(split_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for row_idx, (_, row) in enumerate(split_df.iterrows(), start=1):
        processed_image, _, _ = preprocess_image_and_mask(
            row["img_path"],
            row["mask_path"],
            target_size=TARGET_SIZE,
            remove_hair=True,
            crop_to_mask=True,
            apply_background_mask=False,
        )
        processed_image = processed_image.astype(np.float32) / 255.0
        flattened = processed_image.reshape(-1, 3)

        channel_sum += flattened.sum(axis=0)
        channel_sq_sum += np.square(flattened).sum(axis=0)
        pixel_count += flattened.shape[0]

        if row_idx % 1000 == 0 or row_idx == len(split_df):
            print(f"processed {row_idx:4d} / {len(split_df)} images for normalization stats")

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sq_sum / pixel_count - np.square(mean))
    return mean, std


dataset_mean, dataset_std = compute_channel_stats(train_df)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

stats_table = pd.DataFrame(
    {
        "channel": ["R", "G", "B"],
        "dataset_mean": dataset_mean.round(6),
        "dataset_std": dataset_std.round(6),
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }
)

display(stats_table)

stats_payload = {
    "dataset_mean": dataset_mean.tolist(),
    "dataset_std": dataset_std.tolist(),
    "imagenet_mean": IMAGENET_MEAN.tolist(),
    "imagenet_std": IMAGENET_STD.tolist(),
}

with open(PREP_DIR / "normalization_stats.json", "w", encoding="utf-8") as fp:
    json.dump(stats_payload, fp, indent=2)

class_counts = train_df["binary_label"].value_counts().sort_index()
class_weights = (len(train_df) / (2.0 * class_counts)).rename(index={0: "non_melanoma", 1: "melanoma"})

positive_weight = class_counts[0] / class_counts[1]

def assign_sample_weight(row: pd.Series) -> float:
    if row["binary_label"] == 1:
        return float(positive_weight)
    if row["label"] == HARD_NEGATIVE_LABEL:
        return float(HARD_NEGATIVE_MULTIPLIER)
    return 1.0


train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["sample_weight"] = train_df.apply(assign_sample_weight, axis=1)
val_df["sample_weight"] = 1.0
test_df["sample_weight"] = 1.0

weight_summary = (
    train_df.groupby("label")["sample_weight"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
    .sort_index()
)

display(class_weights.to_frame(name="balanced_class_weight"))
display(weight_summary)
        """
    ),
    md_cell(
        """
## 6. Transforms for Baseline vs Augmented Experiments

Both experiments use the same deterministic preprocessing function. The only difference is:

- `baseline`: normalize only
- `augmented`: normalize after stochastic spatial and photometric augmentation on the training split

Validation and test sets stay deterministic in both experiments. That keeps the comparison fair.
        """
    ),
    code_cell(
        """
if NORMALIZATION_MODE == "dataset":
    normalization_mean = dataset_mean
    normalization_std = dataset_std
elif NORMALIZATION_MODE == "imagenet":
    normalization_mean = IMAGENET_MEAN
    normalization_std = IMAGENET_STD
else:
    raise ValueError(f"Unsupported normalization mode: {NORMALIZATION_MODE}")


augmentation_ops = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(-0.05, 0.05),
        rotate=(-20, 20),
        shear=(-5, 5),
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.6,
    ),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=10, val_shift_limit=8, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
]

baseline_train_transform = A.Compose(
    [
        A.Normalize(mean=normalization_mean, std=normalization_std),
        ToTensorV2(),
    ]
)

augmented_train_transform = A.Compose(
    augmentation_ops
    + [
        A.Normalize(mean=normalization_mean, std=normalization_std),
        ToTensorV2(),
    ]
)

offline_aug_export_transform = A.Compose(augmentation_ops)

eval_transform = A.Compose(
    [
        A.Normalize(mean=normalization_mean, std=normalization_std),
        ToTensorV2(),
    ]
)

print("Baseline train transform:")
print(baseline_train_transform)
print("\\nAugmented train transform:")
print(augmented_train_transform)
        """
    ),
    code_cell(
        """
def denormalize(image_tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return np.clip(image, 0.0, 1.0)


preview_row = train_df[train_df["binary_label"] == 1].sample(1, random_state=SEED).iloc[0]
preprocessed_preview, _, _ = preprocess_image_and_mask(
    preview_row["img_path"],
    preview_row["mask_path"],
    target_size=TARGET_SIZE,
    remove_hair=True,
    crop_to_mask=True,
    apply_background_mask=False,
)

baseline_tensor = baseline_train_transform(image=preprocessed_preview)["image"]
augmented_tensors = [
    augmented_train_transform(image=preprocessed_preview)["image"]
    for _ in range(6)
]

fig, axes = plt.subplots(1, 7, figsize=(21, 3))
axes[0].imshow(preprocessed_preview)
axes[0].set_title("preprocessed")
axes[0].axis("off")

for plot_idx, tensor in enumerate(augmented_tensors, start=1):
    axes[plot_idx].imshow(denormalize(tensor, normalization_mean, normalization_std))
    axes[plot_idx].set_title(f"aug {plot_idx}")
    axes[plot_idx].axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "augmentation_preview_complete.png", dpi=150, bbox_inches="tight")
plt.show()
        """
    ),
    md_cell(
        """
## 7. Export the Processed Datasets to Disk

This section exports the final training-ready images into folder structures that can be consumed directly by directory-based training pipelines such as `torchvision.datasets.ImageFolder`.

The notebook creates two roots:

- `data/processed/without_augmentation/`
- `data/processed/with_augmentation/`

The baseline root contains:

- `train/0_non_melanoma` and `train/1_melanoma`
- `val/0_non_melanoma` and `val/1_melanoma`
- `test/0_non_melanoma` and `test/1_melanoma`

The augmented root contains:

- `train_aug/0_non_melanoma` and `train_aug/1_melanoma`
- `val_aug/0_non_melanoma` and `val_aug/1_melanoma`
- `test_aug/0_non_melanoma` and `test_aug/1_melanoma`

Only `train_aug` receives augmented variants. The `val_aug` and `test_aug` folders are deterministic copies, preserved only to mirror the experiment folder structure while keeping evaluation fair.
        """
    ),
    code_cell(
        """
def save_array_image(image_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array.astype(np.uint8)).save(output_path)


def split_folder_name(split_name: str, use_aug_suffix: bool) -> str:
    return f"{split_name}_aug" if use_aug_suffix else split_name


def class_folder_name(binary_label: int) -> str:
    return BINARY_CLASS_DIRS[int(binary_label)]


def export_split_to_folder(
    split_df: pd.DataFrame,
    split_name: str,
    experiment_name: str,
    augment_train_split: bool = False,
    augmented_copies: int = 1,
    include_original: bool = False,
) -> dict:
    experiment_root = PROCESSED_DIR / experiment_name
    exported_split_name = split_folder_name(split_name, use_aug_suffix=(experiment_name == EXPORT_AUGMENTED_NAME))
    split_output_dir = experiment_root / exported_split_name
    exported_rows = 0
    positive_rows = 0

    for row_idx, row in enumerate(split_df.itertuples(index=False), start=1):
        base_image, base_mask, debug = preprocess_image_and_mask(
            row.img_path,
            row.mask_path,
            target_size=TARGET_SIZE,
            remove_hair=True,
            crop_to_mask=True,
            apply_background_mask=False,
        )

        if augment_train_split and include_original:
            export_specs = [("original", base_image, base_mask, "")]
            for version_idx in range(augmented_copies):
                transformed = offline_aug_export_transform(image=base_image, mask=base_mask)
                export_specs.append(
                    (
                        f"aug_{version_idx + 1:02d}",
                        transformed["image"],
                        (np.array(transformed["mask"]) > 0).astype(np.uint8),
                        f"__aug_{version_idx + 1:02d}",
                    )
                )
        elif augment_train_split:
            export_specs = []
            for version_idx in range(augmented_copies):
                transformed = offline_aug_export_transform(image=base_image, mask=base_mask)
                export_specs.append(
                    (
                        f"aug_{version_idx + 1:02d}",
                        transformed["image"],
                        (np.array(transformed["mask"]) > 0).astype(np.uint8),
                        f"__aug_{version_idx + 1:02d}",
                    )
                )
        else:
            export_specs = [("original", base_image, base_mask, "")]

        for export_tag, export_image, export_mask, suffix in export_specs:
            class_dir = split_output_dir / class_folder_name(row.binary_label)
            file_stem = f"{row.label}__{row.image}{suffix}"
            image_output_path = class_dir / f"{file_stem}.{EXPORT_IMAGE_FORMAT}"

            save_array_image(export_image, image_output_path)
            exported_rows += 1
            positive_rows += int(row.binary_label)

        if row_idx % 500 == 0 or row_idx == len(split_df):
            print(f"[{experiment_name}/{split_name}] exported {row_idx:4d} / {len(split_df)} source images")

    return {
        "experiment": experiment_name,
        "split": split_name,
        "exported_split_name": exported_split_name,
        "exported_rows": exported_rows,
        "positive_rows": positive_rows,
        "output_dir": str(split_output_dir),
    }


baseline_export_summaries = [
    export_split_to_folder(train_df, "train", EXPORT_BASELINE_NAME, augment_train_split=False),
    export_split_to_folder(val_df, "val", EXPORT_BASELINE_NAME, augment_train_split=False),
    export_split_to_folder(test_df, "test", EXPORT_BASELINE_NAME, augment_train_split=False),
]

augmented_export_summaries = [
    export_split_to_folder(
        train_df,
        "train",
        EXPORT_AUGMENTED_NAME,
        augment_train_split=True,
        augmented_copies=AUGMENTED_EXPORT_COPIES,
        include_original=INCLUDE_ORIGINAL_IN_AUGMENTED_TRAIN,
    ),
    export_split_to_folder(val_df, "val", EXPORT_AUGMENTED_NAME, augment_train_split=False),
    export_split_to_folder(test_df, "test", EXPORT_AUGMENTED_NAME, augment_train_split=False),
]

export_summary = pd.DataFrame(
    baseline_export_summaries + augmented_export_summaries
)

display(export_summary)
print(f"Processed folders saved under: {PROCESSED_DIR.resolve()}")
        """
    ),
    md_cell(
        """
## 8. PyTorch DataLoaders from the Exported Folders

At this point the preprocessing is finished. Training can read directly from the exported directories, without recomputing lesion cropping, artifact removal, or augmentation.
        """
    ),
    code_cell(
        """
folder_tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(normalization_mean), std=list(normalization_std)),
    ]
)


def parse_original_label_from_path(image_path: str) -> str:
    return Path(image_path).stem.split("__")[0]


def build_weighted_sampler_from_imagefolder(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    weights = []
    for image_path, class_idx in dataset.samples:
        original_label = parse_original_label_from_path(image_path)
        if class_idx == 1:
            weights.append(float(positive_weight))
        elif original_label == HARD_NEGATIVE_LABEL:
            weights.append(float(HARD_NEGATIVE_MULTIPLIER))
        else:
            weights.append(1.0)

    weights = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_folder_loaders(experiment_root: Path, use_aug_suffix: bool) -> tuple[dict[str, datasets.ImageFolder], dict[str, DataLoader]]:
    split_names = {
        "train": split_folder_name("train", use_aug_suffix=use_aug_suffix),
        "val": split_folder_name("val", use_aug_suffix=use_aug_suffix),
        "test": split_folder_name("test", use_aug_suffix=use_aug_suffix),
    }

    datasets_by_split = {
        split_key: datasets.ImageFolder(root=str(experiment_root / split_folder), transform=folder_tensor_transform)
        for split_key, split_folder in split_names.items()
    }

    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=BATCH_SIZE,
            sampler=build_weighted_sampler_from_imagefolder(datasets_by_split["train"]),
            num_workers=NUM_WORKERS,
            pin_memory=False,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        ),
    }
    return datasets_by_split, loaders


baseline_datasets, baseline_loaders = build_folder_loaders(PROCESSED_DIR / EXPORT_BASELINE_NAME, use_aug_suffix=False)
augmented_datasets, augmented_loaders = build_folder_loaders(PROCESSED_DIR / EXPORT_AUGMENTED_NAME, use_aug_suffix=True)

experiment_summary = pd.DataFrame(
    [
        {
            "experiment": "baseline",
            "train_folder": str(PROCESSED_DIR / EXPORT_BASELINE_NAME / "train"),
            "val_folder": str(PROCESSED_DIR / EXPORT_BASELINE_NAME / "val"),
            "test_folder": str(PROCESSED_DIR / EXPORT_BASELINE_NAME / "test"),
            "train_size": len(baseline_datasets["train"]),
            "val_size": len(baseline_datasets["val"]),
            "test_size": len(baseline_datasets["test"]),
            "class_to_idx": str(baseline_datasets["train"].class_to_idx),
        },
        {
            "experiment": "augmented",
            "train_folder": str(PROCESSED_DIR / EXPORT_AUGMENTED_NAME / "train_aug"),
            "val_folder": str(PROCESSED_DIR / EXPORT_AUGMENTED_NAME / "val_aug"),
            "test_folder": str(PROCESSED_DIR / EXPORT_AUGMENTED_NAME / "test_aug"),
            "train_size": len(augmented_datasets["train"]),
            "val_size": len(augmented_datasets["val"]),
            "test_size": len(augmented_datasets["test"]),
            "class_to_idx": str(augmented_datasets["train"].class_to_idx),
        },
    ]
)

display(experiment_summary)
        """
    ),
    code_cell(
        """
def inspect_batch(loader: DataLoader, title: str, file_name: str) -> None:
    images, labels = next(iter(loader))
    positive_count = int(labels.sum().item())
    print(f"{title}: batch shape = {tuple(images.shape)}, positives = {positive_count} / {len(labels)}")

    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    for idx in range(8):
        axes[idx].imshow(denormalize(images[idx], normalization_mean, normalization_std))
        axes[idx].set_title("MEL" if labels[idx].item() == 1 else "non-MEL", color="#c53030" if labels[idx].item() == 1 else "#2b6cb0", fontsize=9)
        axes[idx].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / file_name, dpi=150, bbox_inches="tight")
    plt.show()


inspect_batch(baseline_loaders["train"], "Baseline train batch", "baseline_batch_sanity.png")
inspect_batch(augmented_loaders["train"], "Augmented train batch", "augmented_batch_sanity.png")
        """
    ),
    code_cell(
        """
def expected_positive_ratio_from_imagefolder(dataset: datasets.ImageFolder) -> float:
    total_weight = 0.0
    positive_total = 0.0
    for image_path, class_idx in dataset.samples:
        original_label = parse_original_label_from_path(image_path)
        if class_idx == 1:
            sample_weight = float(positive_weight)
            positive_total += sample_weight
        elif original_label == HARD_NEGATIVE_LABEL:
            sample_weight = float(HARD_NEGATIVE_MULTIPLIER)
        else:
            sample_weight = 1.0
        total_weight += sample_weight

    return positive_total / total_weight


def folder_positive_ratio(dataset: datasets.ImageFolder) -> float:
    positives = sum(class_idx for _, class_idx in dataset.samples)
    return positives / len(dataset)


balance_table = pd.DataFrame(
    [
        {
            "experiment": "baseline",
            "train_positive_ratio_after_sampler": round(expected_positive_ratio_from_imagefolder(baseline_datasets["train"]) * 100, 2),
            "val_positive_ratio": round(folder_positive_ratio(baseline_datasets["val"]) * 100, 2),
            "test_positive_ratio": round(folder_positive_ratio(baseline_datasets["test"]) * 100, 2),
        },
        {
            "experiment": "augmented",
            "train_positive_ratio_after_sampler": round(expected_positive_ratio_from_imagefolder(augmented_datasets["train"]) * 100, 2),
            "val_positive_ratio": round(folder_positive_ratio(augmented_datasets["val"]) * 100, 2),
            "test_positive_ratio": round(folder_positive_ratio(augmented_datasets["test"]) * 100, 2),
        },
    ]
)

display(balance_table)
        """
    ),
    md_cell(
        """
## 9. Save the Final Preprocessing Configuration

This final artifact records the exact settings used to generate the folder-based datasets and normalization statistics.
        """
    ),
    code_cell(
        """
final_payload = {
    **config,
    "dataset_mean": dataset_mean.tolist(),
    "dataset_std": dataset_std.tolist(),
    "imagenet_mean": IMAGENET_MEAN.tolist(),
    "imagenet_std": IMAGENET_STD.tolist(),
    "train_size": int(len(train_df)),
    "val_size": int(len(val_df)),
    "test_size": int(len(test_df)),
    "train_positive_ratio": float(train_df["binary_label"].mean()),
    "val_positive_ratio": float(val_df["binary_label"].mean()),
    "test_positive_ratio": float(test_df["binary_label"].mean()),
    "baseline_export_root": str(PROCESSED_DIR / EXPORT_BASELINE_NAME),
    "augmented_export_root": str(PROCESSED_DIR / EXPORT_AUGMENTED_NAME),
}

with open(PREP_DIR / "preprocessing_config.json", "w", encoding="utf-8") as fp:
    json.dump(final_payload, fp, indent=2)

print(f"Saved preprocessing artifacts to: {PREP_DIR.resolve()}")
        """
    ),
    md_cell(
        """
## 10. Recommended Use in the Training Notebook

From this point forward, training can read directly from the exported directories:

1. `data/processed/without_augmentation/train`, `val`, `test`
2. `data/processed/with_augmentation/train_aug`, `val_aug`, `test_aug`

If you use `torchvision.datasets.ImageFolder`, the binary classes are already encoded by the folder names:

- `0_non_melanoma`
- `1_melanoma`

That means you do not need to rerun preprocessing in order to start training.
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
