"""
setup_data.py
-------------
Downloads the HAM10000 Skin Cancer Dataset from Kaggle via kagglehub,
moves images and masks to the correct folders, and removes redundant files.

Usage:
    python setup_data.py

Requirements:
    pip install kagglehub
"""

import shutil
from pathlib import Path

import kagglehub

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_HANDLE = "farjanakabirsamanta/skin-cancer-dataset"

REPO_ROOT  = Path(__file__).resolve().parent
IMG_DIR    = REPO_ROOT / "data" / "images"
MASK_DIR   = REPO_ROOT / "data" / "masks"

# ── Download ───────────────────────────────────────────────────────────────────
print("Downloading dataset from Kaggle...")
download_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
print(f"Downloaded to: {download_path}")

# ── Locate images and masks ────────────────────────────────────────────────────
img_src = mask_src = None
for candidate in download_path.rglob("*"):
    if candidate.is_dir() and candidate.name in ("images", "HAM10000_images", "HAM10000_images_part1"):
        img_src = candidate.parent
        break

# Try common structures
for name in ["images", "HAM10000_images_part_1", "HAM10000_images_part1"]:
    candidate = download_path / name
    if candidate.exists():
        img_src = candidate
        break

for name in ["masks", "HAM10000_segmentations_lesion_tschandl"]:
    candidate = download_path / name
    if candidate.exists():
        mask_src = candidate
        break

# ── Move images ────────────────────────────────────────────────────────────────
if img_src and not IMG_DIR.exists():
    shutil.move(str(img_src), str(IMG_DIR))
    count = len(list(IMG_DIR.glob("*.jpg")))
    print(f"[OK] Images → data/images/ ({count:,} files)")
elif IMG_DIR.exists():
    print(f"[SKIP] data/images/ already exists.")
else:
    print(f"[WARNING] Could not locate images folder in {download_path}")

# ── Move masks ─────────────────────────────────────────────────────────────────
if mask_src and not MASK_DIR.exists():
    shutil.move(str(mask_src), str(MASK_DIR))
    count = len(list(MASK_DIR.glob("*.png")))
    print(f"[OK] Masks → data/masks/ ({count:,} files)")
elif MASK_DIR.exists():
    print(f"[SKIP] data/masks/ already exists.")
else:
    print(f"[WARNING] Could not locate masks folder in {download_path}")

# ── Done ───────────────────────────────────────────────────────────────────────
img_count  = len(list(IMG_DIR.glob("*.jpg")))  if IMG_DIR.exists()  else 0
mask_count = len(list(MASK_DIR.glob("*.png"))) if MASK_DIR.exists() else 0
print(f"\nDone. {img_count:,} images and {mask_count:,} masks ready.")
