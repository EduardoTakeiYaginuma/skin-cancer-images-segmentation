# Binary Melanoma Screening from Dermatoscopic Images

**Authors:** Gabriel Fernando Missaka Mendes | Eduardo Takei Yaginuma  
**Course:** Artificial Intelligence in Medicine and Healthcare

## Project Overview

Binary classification of dermatoscopic images: melanoma (`1`) versus non-melanoma (`0`).

The project now uses the local ISIC 2020 training dataset stored under `data/`. The pipeline is organized around:

- raw image exploration from `data/train/`
- metadata exploration from `data/ISIC_2020_Training_GroundTruth.csv`
- patient-aware splitting to reduce leakage
- deterministic image preprocessing without segmentation masks
- optional training-time or offline augmentation after preprocessing

## Dataset

Expected local layout:

```text
data/
├── ISIC_2020_Training_GroundTruth.csv
├── train/
└── processed/
```

Current local dataset statistics:

- `33,126` images
- `2,056` unique patients
- `584` melanoma images
- `32,542` non-melanoma images

Important consequence of this dataset version:

- `patient_id` repeats heavily across the dataset
- many patients have multiple images
- some patients contain both positive and negative samples
- image-level random split is therefore unsafe and can leak patient information

## Preprocessing Policy

The notebooks now assume:

- the raw source of truth is `data/`
- all melanoma samples are kept inside each raw split
- only the training split is downsampled on the non-melanoma side
- negative downsampling preserves diagnosis mix as much as possible
- validation and test remain closer to the original raw distribution

The default training ratio in the notebook generators is `6.0` non-melanoma samples for each melanoma sample. This remains configurable in the generator scripts.

## Project Structure

```text
skin-cancer-images-segmentation/
├── data/
│   ├── ISIC_2020_Training_GroundTruth.csv
│   ├── train/
│   └── processed/           # Exported train/val/test folders and manifests
├── docs/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── outputs/             # Notebook-generated figures/configs
├── outputs/
│   └── figures/             # EDA figures
├── tools/
│   ├── generate_exploration_notebook.py
│   └── generate_preprocessing_notebook.py
├── setup_data.py
└── README.md
```

## Setup

Validate the local dataset:

```bash
python3 setup_data.py
```

Regenerate the notebooks from the tracked generators:

```bash
python3 tools/generate_exploration_notebook.py
python3 tools/generate_preprocessing_notebook.py
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Dataset integrity, imbalance analysis, metadata profiling, patient analysis, image inspection |
| `02_preprocessing.ipynb` | Patient-aware split, train balancing, deterministic preprocessing, export, manifests, loaders |

## Pipeline Outputs

The preprocessing notebook exports:

- `data/processed/without_augmentation/`
- `data/processed/with_augmentation/`
- split manifests for each experiment
- normalization stats and preprocessing config under `notebooks/outputs/preprocessing/`
