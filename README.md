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
- offline data augmentation as a separate export step
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
- augmentation is generated after preprocessing as a separate experimental branch

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
│   ├── 03_data_augmentation.ipynb
│   ├── 04_classification.ipynb
│   └── outputs/
├── outputs/
│   └── figures/
├── tools/
│   ├── generate_exploration_notebook.py
│   ├── generate_preprocessing_notebook.py
│   └── generate_data_augmentation_notebook.py
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
python3 tools/generate_data_augmentation_notebook.py
```

Install dependencies when needed:

```bash
pip install -r requirements.txt
```

## Web Application

The repository now includes a richer Streamlit demo for dermatoscopic triage based on the
final `EfficientNet-B0` experiment documented in `docs/modeling_2_training_journal.md`.

Current app capabilities:

- upload one or multiple dermatoscopic images
- run the calibrated EfficientNet-B0 classifier
- return melanoma probability plus the three-zone clinical triage output
- generate Grad-CAM visual explanations for the classifier
- run a U-Net segmentation model to highlight the lesion area
- combine Grad-CAM and segmentation in the classifier frame
- generate occlusion-sensitivity maps and hotspot crops
- compute simple heuristic descriptors inspired by the ABCD rule
- retrieve visually similar cases from the local processed dataset
- compare predictions with the local dataset truth when the uploaded filename matches an ISIC case ID
- keep a session history and export a CSV/JSON summary

Main deployment settings:

- classification checkpoint: `outputs/models/model_comparison/efficientnet_b0_base_224x224_calibrated.pt`
- segmentation checkpoint: `outputs/models/unet_segmentation.pt`
- `T_LOW = 0.003779`
- `T_HIGH = 0.220775`

Run locally:

```bash
./venv/bin/streamlit run app.py
```

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `01_data_exploration.ipynb` | Class distribution, sample images, masks, lesion coverage and dataset insights | Done |
| `02_preprocessing.ipynb` | Effective dataset selection, lesion-centric preprocessing, baseline export and baseline loaders | Done |
| `03_data_augmentation.ipynb` | Offline augmentation export for the training split, creating the `with_augmentation` branches in `224x224` and `64x64` | Done |
| `04_classification.ipynb` | Binary classifier training and threshold selection experiments | In progress |

## Feature Store (Feast)

The project integrates [Feast](https://feast.dev) for versioned feature management, enabling consistent feature retrieval for both training and serving.

### Structure

```text
feature_store/
├── feature_repo/
│   ├── feature_store.yaml     # Configuração local (SQLite registry + online store)
│   ├── entities.py            # Entidade principal: image_id
│   ├── data_sources.py        # FileSource apontando para os Parquets gerados
│   ├── feature_views.py       # lesion_classification + preprocessing_stats
│   └── feature_services.py   # melanoma_training_features + melanoma_serving_features
├── data/
│   └── sources/               # Parquets gerados por prepare_sources.py (gitignored)
└── scripts/
    ├── prepare_sources.py     # Converte CSVs do projeto para Parquet
    ├── apply_registry.py      # Registra features no Feast (feast apply)
    ├── get_historical_features.py  # Exemplo de retrieval para treino
    └── materialize_online.py  # Materializa para online store
```

### Quickstart

```bash
# 0. Ativar o ambiente virtual do projeto
source venv/bin/activate

# 1. Instalar dependências (inclui feast e pyarrow)
python3 -m pip install -r requirements.txt

# 2. Converter os CSVs do projeto para Parquet (fontes do Feast)
python3 feature_store/scripts/prepare_sources.py

# 3. Registrar as features no registry local
python3 feature_store/scripts/apply_registry.py

# 4. Recuperar features históricas para treino
python3 feature_store/scripts/get_historical_features.py

# 5. (Opcional) Materializar para online store e servir em tempo real
python3 feature_store/scripts/materialize_online.py
```

### Feature Views

| Feature View | Entidade | Features |
|---|---|---|
| `lesion_classification` | `image_id` | MEL, NV, BCC, AKIEC, BKL, DF, VASC, binary_label, label, split |
| `preprocessing_stats` | `image_id` | mask_coverage_after_crop, hair_pixels_detected, final_height, final_width |

### Feature Services

| Serviço | Uso | Features incluídas |
|---|---|---|
| `melanoma_training_features` | Treino offline | Todas as features acima |
| `melanoma_serving_features` | Inferência online | mask_coverage_after_crop, hair_pixels_detected |

---

## Pipeline Outputs

The preprocessing and augmentation notebooks export:

- `data/processed/without_augmentation/` from `02_preprocessing.ipynb`
- `data/processed/without_augmentation_64x64/` from `02_preprocessing.ipynb`
- `data/processed/with_augmentation/` from `03_data_augmentation.ipynb`
- `data/processed/with_augmentation_64x64/` from `03_data_augmentation.ipynb`
- split manifests for each experiment
- normalization stats and preprocessing config under `notebooks/outputs/preprocessing/`
