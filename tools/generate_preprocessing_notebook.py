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
# 02 - Preprocessing do Dataset com Máscaras

Este notebook prepara o dataset antigo, agora organizado em:

- `data/metadata.csv`
- `data/images/`
- `data/masks/`

**Objetivos**

- validar imagens, máscaras e metadata
- montar o dataset efetivo com downsampling apenas da classe negativa
- aplicar preprocessing lesion-centric com apoio das máscaras
- criar splits `train/val/test`
- calcular estatísticas de normalização
- exportar versões com e sem augmentation
- montar `DataLoader`s prontos para treino
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
        if (candidate / "data" / "metadata.csv").exists() and (candidate / "docs").exists():
            return candidate
    raise FileNotFoundError("Nao foi possivel localizar a raiz com data/metadata.csv.")


ROOT_DIR = resolve_repo_root()
DATA_DIR = ROOT_DIR / "data"
IMG_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"
META_PATH = DATA_DIR / "metadata.csv"
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

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
TARGET_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NON_MELANOMA_TO_MELANOMA_RATIO = 3.0
MASK_THRESHOLD = 127
BBOX_MARGIN_RATIO = 0.10
HAIR_KERNEL_SIZE = 17
HAIR_THRESHOLD = 10
HAIR_INPAINT_RADIUS = 1
NORMALIZATION_SAMPLE_SIZE = 512
BATCH_SIZE = 32
NUM_WORKERS = 0
HARD_NEGATIVE_LABEL = "NV"
HARD_NEGATIVE_MULTIPLIER = 2.0
NORMALIZATION_MODE = "dataset"
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
    "non_melanoma_to_melanoma_ratio": NON_MELANOMA_TO_MELANOMA_RATIO,
    "mask_threshold": MASK_THRESHOLD,
    "bbox_margin_ratio": BBOX_MARGIN_RATIO,
    "hair_kernel_size": HAIR_KERNEL_SIZE,
    "hair_threshold": HAIR_THRESHOLD,
    "hair_inpaint_radius": HAIR_INPAINT_RADIUS,
    "normalization_sample_size": NORMALIZATION_SAMPLE_SIZE,
    "batch_size": BATCH_SIZE,
    "hard_negative_label": HARD_NEGATIVE_LABEL,
    "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER,
    "normalization_mode": NORMALIZATION_MODE,
    "export_baseline_name": EXPORT_BASELINE_NAME,
    "export_augmented_name": EXPORT_AUGMENTED_NAME,
    "augmented_export_copies": AUGMENTED_EXPORT_COPIES,
    "include_original_in_augmented_train": INCLUDE_ORIGINAL_IN_AUGMENTED_TRAIN,
    "export_image_format": EXPORT_IMAGE_FORMAT,
    "processed_dir": str(PROCESSED_DIR),
}

print(json.dumps(config, indent=2))
        """
    ),
    md_cell(
        """
## 1. Carregamento do dataset bruto
        """
    ),
    code_cell(
        """
raw_df = pd.read_csv(META_PATH).copy()

assert set(CLASSES).issubset(raw_df.columns), "Metadata sem uma ou mais classes."
assert (raw_df[CLASSES].sum(axis=1) == 1).all(), "Cada imagem deve pertencer a uma unica classe."

raw_df["label"] = raw_df[CLASSES].idxmax(axis=1)
raw_df["binary_label"] = (raw_df["label"] == "MEL").astype(int)
raw_df["img_path"] = raw_df["image"].map(lambda image_id: IMG_DIR / f"{image_id}.jpg")
raw_df["mask_path"] = raw_df["image"].map(lambda image_id: MASK_DIR / f"{image_id}.png")

raw_summary = pd.DataFrame(
    [
        {"metrica": "linhas_metadata", "valor": len(raw_df)},
        {"metrica": "imagens_faltantes", "valor": int((~raw_df["img_path"].map(Path.exists)).sum())},
        {"metrica": "masks_faltantes", "valor": int((~raw_df["mask_path"].map(Path.exists)).sum())},
        {"metrica": "melanomas", "valor": int((raw_df["binary_label"] == 1).sum())},
        {"metrica": "nao_melanomas", "valor": int((raw_df["binary_label"] == 0).sum())},
    ]
)

label_summary = raw_df["label"].value_counts().reindex(CLASSES).rename_axis("classe").reset_index(name="count")
label_summary["pct"] = (label_summary["count"] / len(raw_df) * 100).round(2)

display(raw_summary)
display(label_summary)
print(
    f"Resumo rapido: {len(raw_df):,} imagens, "
    f"{int((raw_df['binary_label'] == 1).sum()):,} melanomas e "
    f"{int((raw_df['binary_label'] == 0).sum()):,} nao melanomas."
)
        """
    ),
    md_cell(
        """
## 2. Dataset efetivo

Mantemos todos os melanomas e reduzimos apenas a classe negativa para um ratio configurável.
        """
    ),
    code_cell(
        """
def build_effective_dataset(df: pd.DataFrame, non_melanoma_ratio: float, seed: int) -> pd.DataFrame:
    melanoma_df = df[df["binary_label"] == 1].copy()
    non_melanoma_df = df[df["binary_label"] == 0].copy()

    target_non_melanoma = min(
        len(non_melanoma_df),
        int(round(len(melanoma_df) * non_melanoma_ratio)),
    )

    if target_non_melanoma <= 0:
        raise ValueError("O ratio solicitado produziu um subconjunto negativo vazio.")

    if target_non_melanoma == len(non_melanoma_df):
        selected_non_melanoma = non_melanoma_df.copy()
    else:
        selected_non_melanoma, _ = train_test_split(
            non_melanoma_df,
            train_size=target_non_melanoma,
            stratify=non_melanoma_df["label"],
            random_state=seed,
        )

    melanoma_df["selection_source"] = "kept_all_melanoma"
    selected_non_melanoma["selection_source"] = "downsampled_non_melanoma"

    return (
        pd.concat([melanoma_df, selected_non_melanoma], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


df = build_effective_dataset(raw_df, NON_MELANOMA_TO_MELANOMA_RATIO, SEED)

effective_summary = pd.DataFrame(
    {
        "raw_count": raw_df["binary_label"].value_counts().sort_index(),
        "effective_count": df["binary_label"].value_counts().sort_index(),
    }
).rename(index={0: "nao_melanoma", 1: "melanoma"})
effective_summary["retention_pct"] = (
    effective_summary["effective_count"] / effective_summary["raw_count"] * 100
).round(2)

display(effective_summary)
print(
    f"Dataset efetivo criado com {len(df):,} imagens e ratio "
    f"{(df['binary_label'] == 0).sum() / (df['binary_label'] == 1).sum():.2f}:1."
)
        """
    ),
    md_cell(
        """
## 3. Checagens de integridade

Antes do preprocessing, precisamos confirmar que imagens e máscaras batem com a metadata e que os shapes são consistentes.
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

for _, row in df.iterrows():
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
        "metrica": [
            "linhas_dataset_efetivo",
            "imagens_unicas",
            "imagens_faltantes",
            "masks_faltantes",
            "distinct_image_sizes",
            "distinct_mask_sizes",
        ],
        "valor": [
            len(df),
            df["image"].nunique(),
            len(missing_images),
            len(missing_masks),
            dict(image_sizes),
            dict(mask_sizes),
        ],
    }
)

display(integrity_summary)

assert len(missing_images) == 0, "Existem imagens faltantes."
assert len(missing_masks) == 0, "Existem mascaras faltantes."
assert df["image"].nunique() == len(df), "Existem image ids duplicados."
assert len(image_sizes) == 1, "As imagens nao possuem tamanho unico."
assert len(mask_sizes) == 1, "As mascaras nao possuem tamanho unico."

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["mask_coverage"], bins=40, kde=True, ax=axes[0], color="#2b6cb0")
axes[0].set_title("Distribuicao da Cobertura das Mascaras")
axes[0].set_xlabel("Cobertura da mascara")

sns.scatterplot(
    data=df.sample(min(1500, len(df)), random_state=SEED),
    x="bbox_width_ratio",
    y="bbox_height_ratio",
    hue="binary_label",
    palette={0: "#2b6cb0", 1: "#c53030"},
    alpha=0.7,
    ax=axes[1],
)
axes[1].set_title("Bounding Box por Amostra")
axes[1].set_xlabel("bbox width / image width")
axes[1].set_ylabel("bbox height / image height")

plt.tight_layout()
plt.savefig(FIG_DIR / "preprocessing_integrity.png", dpi=150, bbox_inches="tight")
plt.show()

print("Checagens de integridade concluidas: imagens e mascaras estao consistentes.")
        """
    ),
    md_cell(
        """
## 4. Preprocessing lesion-centric

O pipeline determinístico usa a máscara para localizar a lesão:

1. carregar imagem RGB e máscara binária
2. remover pelos finos
3. localizar a bounding box da lesão
4. expandir a região com uma margem de segurança
5. fazer crop lesion-centric
6. aplicar padding para quadrado
7. redimensionar para `224 x 224`
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
) -> tuple[np.ndarray, np.ndarray, dict]:
    image = load_rgb_image(image_path)
    mask = load_mask_array(mask_path)

    if remove_hair:
        image, hair_mask = remove_hair_artifacts(image)
    else:
        hair_mask = np.zeros(mask.shape, dtype=np.uint8)

    bbox = compute_mask_bbox(mask)
    bbox = expand_bbox(bbox, width=image.shape[1], height=image.shape[0])

    image = crop_array(image, bbox)
    mask = crop_array(mask, bbox)

    image = pad_to_square(image)
    mask = pad_to_square(mask)

    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)

    debug = {
        "bbox": bbox,
        "final_shape": image.shape[:2],
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
    )

    axes[row_idx, 0].imshow(raw_image)
    axes[row_idx, 0].set_title(f"{row['label']} - original")
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(hair_mask, cmap="gray")
    axes[row_idx, 1].set_title("mascara de pelos")
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(hair_removed)
    axes[row_idx, 2].contour(raw_mask, levels=[0.5], colors="lime", linewidths=1)
    axes[row_idx, 2].set_title("hair removed + contorno")
    axes[row_idx, 2].axis("off")

    axes[row_idx, 3].imshow(processed_image)
    axes[row_idx, 3].set_title(f"processada {debug['final_shape'][0]}x{debug['final_shape'][1]}")
    axes[row_idx, 3].axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "preprocessing_steps.png", dpi=150, bbox_inches="tight")
plt.show()

uniformity_checks = []
for _, row in df.sample(min(128, len(df)), random_state=SEED).iterrows():
    processed_image, processed_mask, _ = preprocess_image_and_mask(
        row["img_path"],
        row["mask_path"],
        target_size=TARGET_SIZE,
        remove_hair=True,
    )
    uniformity_checks.append((processed_image.shape, processed_mask.shape))

assert all(image_shape == (TARGET_SIZE, TARGET_SIZE, 3) for image_shape, _ in uniformity_checks)
assert all(mask_shape == (TARGET_SIZE, TARGET_SIZE) for _, mask_shape in uniformity_checks)

print(
    f"Todas as amostras testadas apos preprocessing possuem shapes fixos: "
    f"{uniformity_checks[0][0]} para imagem e {uniformity_checks[0][1]} para mascara."
)
        """
    ),
    md_cell(
        """
## 5. Split reprodutível em train / val / test
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

combined_split_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

split_summary = (
    combined_split_df.groupby(["split", "binary_label"])
    .size()
    .rename("count")
    .reset_index()
)
split_summary["target_name"] = split_summary["binary_label"].map({0: "nao_melanoma", 1: "melanoma"})

display(split_summary)
print(
    f"Split concluido: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}."
)
        """
    ),
    md_cell(
        """
## 6. Estatísticas de normalização

As estatísticas são medidas apenas no treino após o preprocessing.
        """
    ),
    code_cell(
        """
def compute_normalization_stats(frame: pd.DataFrame, sample_size: int = NORMALIZATION_SAMPLE_SIZE) -> tuple[list[float], list[float], int]:
    sample_frame = frame.sample(n=min(sample_size, len(frame)), random_state=SEED)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for row in sample_frame.itertuples():
        image, _, _ = preprocess_image_and_mask(row.img_path, row.mask_path, target_size=TARGET_SIZE, remove_hair=True)
        pixels = image.astype(np.float32).reshape(-1, 3) / 255.0
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels ** 2).sum(axis=0)
        pixel_count += len(pixels)

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sq_sum / pixel_count - mean ** 2)
    return mean.round(6).tolist(), std.round(6).tolist(), len(sample_frame)


train_mean, train_std, normalization_sample_count = compute_normalization_stats(train_df)
normalization_stats = {
    "mean": train_mean,
    "std": train_std,
    "sample_count": normalization_sample_count,
    "target_size": TARGET_SIZE,
}

(PREP_DIR / "normalization_stats.json").write_text(json.dumps(normalization_stats, indent=2), encoding="utf-8")
(PREP_DIR / "preprocessing_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

display(pd.DataFrame({"canal": ["R", "G", "B"], "mean": train_mean, "std": train_std}))
print(f"Normalizacao calculada sobre {normalization_sample_count} imagens do treino.")
        """
    ),
    md_cell(
        """
## 7. Exportação do dataset processado
        """
    ),
    code_cell(
        """
try:
    import albumentations as A
except ImportError as exc:
    raise ImportError("Albumentations e necessario para a etapa de exportacao.") from exc


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
        base_image, base_mask, debug = preprocess_image_and_mask(
            row.img_path,
            row.mask_path,
            target_size=TARGET_SIZE,
            remove_hair=True,
        )
        class_dir = export_root / split_name / BINARY_CLASS_DIRS[row.binary_label]

        if include_original:
            base_output_path = class_dir / f"{row.image}.{EXPORT_IMAGE_FORMAT}"
            save_rgb_image(base_image, base_output_path)
            manifest_rows.append(
                {
                    "image_id": row.image,
                    "split": split_name,
                    "binary_label": row.binary_label,
                    "label": row.label,
                    "selection_source": row.selection_source,
                    "is_augmented": False,
                    "augmentation_copy": 0,
                    "export_path": str(base_output_path),
                    "mask_coverage_after_resize": debug["mask_coverage_after_resize"],
                }
            )

        if split_name == "train" and augmenter is not None and augmented_copies > 0:
            for copy_idx in range(1, augmented_copies + 1):
                transformed = augmenter(image=base_image, mask=base_mask)
                aug_image = transformed["image"]
                aug_output_path = class_dir / f"{row.image}_aug{copy_idx}.{EXPORT_IMAGE_FORMAT}"
                save_rgb_image(aug_image, aug_output_path)
                manifest_rows.append(
                    {
                        "image_id": row.image,
                        "split": split_name,
                        "binary_label": row.binary_label,
                        "label": row.label,
                        "selection_source": row.selection_source,
                        "is_augmented": True,
                        "augmentation_copy": copy_idx,
                        "export_path": str(aug_output_path),
                        "mask_coverage_after_resize": debug["mask_coverage_after_resize"],
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
    train_df,
    val_df,
    test_df,
    baseline_root,
    train_augmenter=None,
    train_augmented_copies=0,
    include_original_train=True,
)

augmented_manifests = export_experiment(
    train_df,
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
            "experimento": "without_augmentation",
            "train_images": len(baseline_manifests["train"]),
            "val_images": len(baseline_manifests["val"]),
            "test_images": len(baseline_manifests["test"]),
        },
        {
            "experimento": "with_augmentation",
            "train_images": len(augmented_manifests["train"]),
            "val_images": len(augmented_manifests["val"]),
            "test_images": len(augmented_manifests["test"]),
        },
    ]
)

display(export_summary)
print(
    f"Exportacao concluida. Baseline train={len(baseline_manifests['train']):,} e "
    f"augmented train={len(augmented_manifests['train']):,}."
)
        """
    ),
    md_cell(
        """
## 8. DataLoaders
        """
    ),
    code_cell(
        """
try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms
except ImportError as exc:
    raise ImportError("PyTorch e torchvision sao necessarios para os loaders.") from exc


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
    if row.label == HARD_NEGATIVE_LABEL:
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
print(f"DataLoaders prontos. Classes detectadas: {baseline_train_dataset.class_to_idx}")
print(f"Peso relativo da classe positiva no sampler: {positive_weight:.2f}")
        """
    ),
    md_cell(
        """
## 9. Conclusões
        """
    ),
    code_cell(
        """
raw_ratio = (raw_df["binary_label"] == 0).sum() / (raw_df["binary_label"] == 1).sum()
effective_ratio = (df["binary_label"] == 0).sum() / (df["binary_label"] == 1).sum()
baseline_train_binary = baseline_manifests["train"]["binary_label"].value_counts().sort_index()
augmented_train_binary = augmented_manifests["train"]["binary_label"].value_counts().sort_index()

baseline_nonmel = int(baseline_train_binary.get(0, 0))
baseline_mel = int(baseline_train_binary.get(1, 0))
augmented_nonmel = int(augmented_train_binary.get(0, 0))
augmented_mel = int(augmented_train_binary.get(1, 0))

summary_table = pd.DataFrame(
    [
        {"metrica": "ratio_bruto", "valor": round(raw_ratio, 2)},
        {"metrica": "ratio_efetivo", "valor": round(effective_ratio, 2)},
        {"metrica": "train_export_baseline", "valor": len(baseline_manifests["train"])},
        {"metrica": "train_export_augmented", "valor": len(augmented_manifests["train"])},
        {"metrica": "baseline_train_melanoma", "valor": baseline_mel},
        {"metrica": "baseline_train_nao_melanoma", "valor": baseline_nonmel},
        {"metrica": "augmented_train_melanoma", "valor": augmented_mel},
        {"metrica": "augmented_train_nao_melanoma", "valor": augmented_nonmel},
    ]
)
display(summary_table)

print("Conclusoes:")
print(
    f"- O preprocessing voltou a ser lesion-centric porque o dataset atual inclui mascaras confiaveis "
    f"para todas as imagens."
)
print(
    f"- O dataset bruto apresenta ratio de {raw_ratio:.2f}:1, enquanto o dataset efetivo reduz isso para "
    f"{effective_ratio:.2f}:1 mantendo todos os melanomas."
)
print(
    f"- Como imagens e mascaras possuem shape consistente, o pipeline pode usar crop guiado por mascara "
    f"antes do resize final para {TARGET_SIZE}x{TARGET_SIZE}."
)
print(
    f"- Sem data augmentation, o conjunto de treino exportado ficou com "
    f"{baseline_mel} imagens de melanoma e {baseline_nonmel} imagens de nao melanoma."
)
print(
    f"- Com data augmentation offline, o conjunto de treino exportado passou para "
    f"{augmented_mel} imagens de melanoma e {augmented_nonmel} imagens de nao melanoma."
)
print(
    f"- A augmentation aumentou o volume total de treino, mas manteve a mesma proporcao entre classes "
    f"({baseline_nonmel / baseline_mel:.2f}:1 no baseline e {augmented_nonmel / augmented_mel:.2f}:1 com augmentation)."
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
