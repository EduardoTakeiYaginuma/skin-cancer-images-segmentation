# Binary Melanoma Screening from Dermatoscopic Images

**Authors:** Gabriel Fernando Missaka Mendes | Eduardo Takei Yaginuma  
**Course:** Artificial Intelligence in Medicine and Healthcare

---

## Project Overview

A deep learning-based system for binary classification of dermatoscopic images, distinguishing melanoma from non-melanoma lesions.

Rather than multi-class classification, the task is reformulated as binary — all non-melanoma categories are grouped into a single class. The model focuses on learning discriminative features that separate melanoma from visually similar lesions (hard negatives), minimizing false negatives given their life-threatening consequences.

**Primary metric:** Sensitivity (Recall) — missing a melanoma can be fatal.  
**Secondary metrics:** Specificity, AUC-ROC.

## Dataset

**HAM10000 (Human Against Machine with 10,000 training images)** — [Kaggle](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset)

- 10,015 dermatoscopic images across 7 diagnostic classes
- Binary label: melanoma (1) vs. non-melanoma (0)
- >50% of lesions confirmed via histopathology
- Size: ~2.77 GB | Source: Harvard Dataverse | License: CC BY-NC-SA 4.0

### Classes

| Class | Label | Count |
|-------|-------|-------|
| Melanocytic Nevi (NV) | Non-Melanoma | 6,705 |
| Melanoma (MEL) | Melanoma | 1,113 |
| Benign Keratosis (BKL) | Non-Melanoma | 1,099 |
| Basal Cell Carcinoma (BCC) | Non-Melanoma | 514 |
| Actinic Keratosis (AKIEC) | Non-Melanoma | 327 |
| Vascular Lesions (VASC) | Non-Melanoma | 142 |
| Dermatofibroma (DF) | Non-Melanoma | 115 |
| **Total** | | **10,015** |

**Class imbalance:** ~8:1 (non-melanoma : melanoma)

## Project Structure

```
skin-cancer-images-segmentation/
├── data/
│   ├── images/              # Dataset images (not tracked by git)
│   ├── masks/               # Segmentation masks (not tracked by git)
│   └── metadata/            # Train/val/test split CSVs
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset analysis ✅
│   ├── 02_preprocessing.ipynb      # Augmentation & split strategy ✅
│   └── 03_classification.ipynb     # Binary classifier training & threshold selection
├── outputs/
│   ├── figures/             # Plots saved by notebooks
│   └── models/              # Saved model checkpoints (not tracked by git)
├── src/                     # Reusable Python modules
├── docs/                    # Project documents (proposal, report)
├── setup_data.py            # Downloads and organises the dataset automatically
├── requirements.txt         # Pinned dependencies
├── .gitignore
└── README.md
```

## Setup

**1. Install dependencies**

Option A — from requirements file (recommended):
```bash
pip install -r requirements.txt
```

Option B — core packages manually:
```bash
pip install torch torchvision pillow pandas numpy matplotlib seaborn scikit-learn
```

**2. Download the dataset**

```bash
pip install kagglehub   # if not already installed
python setup_data.py
```

> Requires Kaggle credentials configured (`~/.kaggle/kaggle.json`). See [Kaggle API docs](https://www.kaggle.com/docs/api) for setup instructions.

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `01_data_exploration` | Class distribution, sample images, pixel stats, masks, lesion coverage | Done |
| `02_preprocessing` | Augmentation pipeline, dataset split, batch sanity check | Done |
| `03_classification` | Binary classifier training, threshold selection | In progress |
