"""Download HAM10000 (images + masks + metadata) into the data/ folder.

Lets external users reproduce the project end-to-end without access to
the team's private DVC S3 remote. The dataset is fetched from public
sources:

- Images and metadata: Kaggle (kmader/skin-cancer-mnist-ham10000)
- Lesion masks: Harvard Dataverse (Tschandl et al.)

Prerequisites:

- Free Kaggle account with API token at ~/.kaggle/kaggle.json.
  Token instructions: https://github.com/Kaggle/kaggle-api#api-credentials
- ~3.5 GB of free disk space.

Usage:

    python scripts/download_dataset.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# Harvard Dataverse persistent URL for the Tschandl lesion mask release.
# The release page lists the segmentations zip; the API endpoint below
# streams it directly without authentication.
MASKS_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/"
    ":persistentId?persistentId=doi:10.7910/DVN/DBW86T/Y5HEUS"
)


def _progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    percent = min(100, int(blocks * block_size / total_size * 100))
    sys.stdout.write(f"\r  {percent}%")
    sys.stdout.flush()


def _ensure_kaggle_installed() -> None:
    try:
        import kaggle  # noqa: F401,I001  # type: ignore[import-not-found]
    except ImportError:
        print("Installing the kaggle package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "kaggle"],
            check=True,
        )


def _download_images_and_metadata() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-not-found]

    api = KaggleApi()
    try:
        api.authenticate()
    except OSError as exc:
        print(f"\nKaggle API authentication failed: {exc}")
        print(
            "\nSteps to get your Kaggle token:\n"
            "  1. Sign up or log in at https://www.kaggle.com\n"
            "  2. Account, then 'Create New API Token' (downloads kaggle.json)\n"
            "  3. Place it at ~/.kaggle/kaggle.json\n"
            "  4. chmod 600 ~/.kaggle/kaggle.json\n"
        )
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} from Kaggle (about 3 GB)...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)


def _organize_images() -> None:
    """Move images from the Kaggle layout into data/images/."""
    IMAGES_DIR.mkdir(exist_ok=True)
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        part_dir = DATA_DIR / part
        if part_dir.exists():
            for img in part_dir.glob("*.jpg"):
                target = IMAGES_DIR / img.name
                if not target.exists():
                    shutil.move(str(img), target)
            shutil.rmtree(part_dir)


def _rename_metadata() -> None:
    src = DATA_DIR / "HAM10000_metadata.csv"
    dst = DATA_DIR / "metadata.csv"
    if src.exists() and not dst.exists():
        src.rename(dst)


def _download_masks() -> None:
    """Fetch the lesion mask release from Harvard Dataverse."""
    MASKS_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "_masks_tschandl.zip"
    if zip_path.exists():
        print("[skip] mask archive already downloaded")
    else:
        print("\nDownloading lesion masks from Harvard Dataverse...")
        try:
            urlretrieve(MASKS_URL, zip_path, reporthook=_progress)
            print()
        except Exception as exc:
            print(f"\n  Failed to download masks: {exc}")
            print(
                "  Masks are optional for classification but required for the\n"
                "  U-Net segmentation notebook. You can download them manually\n"
                "  from doi:10.7910/DVN/DBW86T and place the PNGs under data/masks/."
            )
            return

    print("Extracting masks...")
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.filename.lower().endswith(".png"):
                target = MASKS_DIR / Path(info.filename).name
                if not target.exists():
                    with z.open(info) as src_f, target.open("wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
    zip_path.unlink(missing_ok=True)


def main() -> None:
    _ensure_kaggle_installed()
    _download_images_and_metadata()
    _organize_images()
    _rename_metadata()
    _download_masks()

    image_count = sum(1 for _ in IMAGES_DIR.glob("*.jpg")) if IMAGES_DIR.exists() else 0
    mask_count = sum(1 for _ in MASKS_DIR.glob("*.png")) if MASKS_DIR.exists() else 0

    print(f"\nDataset ready under {DATA_DIR}")
    print(f"  Images:  data/images/   ({image_count} files)")
    print(f"  Masks:   data/masks/    ({mask_count} files)")
    print("  Splits:  data/metadata/ (committed in git)")
    print("  Labels:  data/metadata.csv")


if __name__ == "__main__":
    main()
