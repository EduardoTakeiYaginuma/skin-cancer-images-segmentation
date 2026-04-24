# Binary Melanoma Screening from Dermatoscopic Images

**Authors:** Gabriel Fernando Missaka Mendes | Eduardo Takei Yaginuma  
**Course:** Artificial Intelligence in Medicine and Healthcare

## Project Overview

Binary classification of dermatoscopic images: melanoma (`1`) versus non-melanoma (`0`).

The project currently uses the dataset stored locally under `data/`, with:

- `data/metadata.csv`
- `data/images/`
- `data/masks/`

The workflow is organized around:

- exploratory analysis of the original 7 classes
- conversion to a binary melanoma vs non-melanoma task
- lesion-centric preprocessing guided by segmentation masks
- comparison between baseline training and training with augmentation
- downstream classification experiments

## Dataset

Expected local layout:

```text
data/
├── metadata.csv
├── images/
├── masks/
├── metadata/
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
└── processed/
```

Current dataset statistics:

- `10,015` images
- `1,113` melanoma images
- `8,902` non-melanoma images
- original 7-class annotation preserved in the metadata
- masks available for lesion-aware preprocessing

## Preprocessing Policy

The current notebooks assume:

- the raw source of truth is `data/`
- all melanoma images are kept
- only non-melanoma images are downsampled when building the effective dataset
- the negative subclass mix is preserved as much as possible
- lesion masks are used to support lesion-centric cropping
- augmentation is evaluated as a separate experimental branch

The default effective ratio in preprocessing is `3.0` non-melanoma images for each melanoma image.

## Project Structure

```text
skin-cancer-images-segmentation/
├── data/
│   ├── metadata.csv
│   ├── images/
│   ├── masks/
│   ├── metadata/              # Saved train/val/test split CSVs
│   └── processed/             # Exported train/val/test folders and manifests
├── docs/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classification.ipynb
│   └── outputs/
├── outputs/
│   └── figures/
├── tools/
│   ├── generate_exploration_notebook.py
│   └── generate_preprocessing_notebook.py
├── requirements.txt
├── setup_data.py
└── README.md
```

## Setup

Validate the local dataset:

```bash
python3 setup_data.py
```

Regenerate the tracked notebooks:

```bash
python3 tools/generate_exploration_notebook.py
python3 tools/generate_preprocessing_notebook.py
```

Install dependencies when needed:

```bash
pip install -r requirements.txt
```

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `01_data_exploration.ipynb` | Class distribution, sample images, masks, lesion coverage and dataset insights | Done |
| `02_preprocessing.ipynb` | Effective dataset selection, lesion-centric preprocessing, export, loaders and augmentation branch | Done |
| `03_classification.ipynb` | Binary classifier training and threshold selection experiments | In progress |

## Pipeline Outputs

The preprocessing notebook exports:

- `data/processed/without_augmentation/`
- `data/processed/with_augmentation/`
- split manifests for each experiment
- normalization stats and preprocessing config under `notebooks/outputs/preprocessing/`
