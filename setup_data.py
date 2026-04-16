"""
setup_data.py
-------------
Validates the local ISIC 2020 dataset stored under ``data/`` and bootstraps the
derived-data folders used by the notebooks.

Usage:
    python3 setup_data.py
"""

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "image_name",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "diagnosis",
    "benign_malignant",
    "target",
]

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
IMG_DIR = DATA_DIR / "train"
META_PATH = DATA_DIR / "ISIC_2020_Training_GroundTruth.csv"

PROCESSED_DIR = DATA_DIR / "processed"
EDA_FIG_DIR = REPO_ROOT / "outputs" / "figures"
NB_FIG_DIR = REPO_ROOT / "notebooks" / "outputs" / "figures"
NB_PREP_DIR = REPO_ROOT / "notebooks" / "outputs" / "preprocessing"


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def main() -> None:
    ensure_exists(DATA_DIR, "data directory")
    ensure_exists(IMG_DIR, "training image directory")
    ensure_exists(META_PATH, "metadata CSV")

    metadata = pd.read_csv(META_PATH).copy()
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")

    metadata["img_path"] = metadata["image_name"].map(lambda image_id: IMG_DIR / f"{image_id}.jpg")

    image_count = len(list(IMG_DIR.glob("*.jpg")))
    row_count = len(metadata)
    missing_images = int((~metadata["img_path"].map(Path.exists)).sum())
    duplicate_images = int(metadata["image_name"].duplicated().sum())
    unique_patients = int(metadata["patient_id"].nunique())
    mixed_target_patients = int(
        (metadata.groupby("patient_id")["target"].nunique() > 1).sum()
    )

    for directory in [PROCESSED_DIR, EDA_FIG_DIR, NB_FIG_DIR, NB_PREP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    target_counts = metadata["target"].value_counts().sort_index().to_dict()
    diagnosis_counts = metadata["diagnosis"].fillna("missing").value_counts().head(8).to_dict()
    missing_metadata = metadata[
        ["sex", "age_approx", "anatom_site_general_challenge"]
    ].isna().sum().to_dict()

    print("Local ISIC 2020 dataset validated successfully.")
    print(f"Metadata rows         : {row_count:,}")
    print(f"Images on disk        : {image_count:,}")
    print(f"Missing image files   : {missing_images:,}")
    print(f"Duplicate image ids   : {duplicate_images:,}")
    print(f"Unique patients       : {unique_patients:,}")
    print(f"Mixed-target patients : {mixed_target_patients:,}")
    print(f"Target counts         : {target_counts}")
    print(f"Top diagnoses         : {diagnosis_counts}")
    print(f"Missing metadata      : {missing_metadata}")
    print(f"Processed dir         : {PROCESSED_DIR}")

    if image_count != row_count or missing_images > 0:
        print("[WARNING] The image folder and metadata file are not perfectly aligned.")


if __name__ == "__main__":
    main()
