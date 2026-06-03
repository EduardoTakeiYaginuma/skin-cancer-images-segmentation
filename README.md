# Binary Melanoma Screening from Dermatoscopic Images

**Authors:** Gabriel Fernando Missaka Mendes | Eduardo Takei Yaginuma
**Course:** MLOps at Insper (26.1)

## Project Video

A 3-5 minute video explaining the project, the dataset and the main MLOps decisions:

🎥 **[Watch the video](TODO-add-video-link-here)**

## Project Overview

This project takes a fine-tuned ResNet50 classifier for binary melanoma screening and operationalizes it end-to-end. The focus is not on pushing classification accuracy, but on demonstrating a complete MLOps stack: experiment tracking, data and feature versioning, automated container-based deployment, infrastructure as code, structured logging, statistical drift monitoring, automatic retraining on degradation, and full continuous integration.

The detailed report is in [`docs/project_report.md`](docs/project_report.md).

## Dataset

The project uses the **HAM10000** dataset from the ISIC archive: **10,015 dermatoscopic images** annotated across 7 diagnostic classes (`MEL`, `NV`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC`), collapsed into a binary `melanoma vs. non-melanoma` task. Class imbalance is significant: only ~11% of the images are melanoma.

HAM10000 is publicly available from:

- Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- ISIC Archive: https://challenge.isic-archive.com/data/

Expected local layout:

```
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

The split CSVs under `data/metadata/` are committed directly to git. The heavy data (`data/images/`, `data/masks/`, `data/metadata.csv`) is versioned with DVC pointers and stored on a private S3 remote configured in `.dvc/config`. The split CSVs in this repository remain valid as long as you use the same HAM10000 release.

## Project Structure

```
.
├── api/                  # FastAPI service (REST endpoints)
├── skin_app/             # Core inference and structured logging
├── monitoring/           # Drift detector, retrain trigger, reports
├── feature_store/        # Feast feature store (entities, views, services, scripts)
├── infra/                # IaC: CloudFormation + Terraform
├── notebooks/            # EDA, segmentation, preprocessing, modeling
├── scripts/              # deploy_lambda.sh, dvc_setup_s3.sh
├── tests/                # pytest suite
├── docs/                 # project_report.md
├── config/               # inference_config.json
├── data/                 # DVC pointers + metadata splits
├── outputs/              # DVC pointers for trained models
├── .github/workflows/    # CI (lint + tests + docker build + cron monitoring)
├── app.py                # Streamlit demo UI
├── train.py              # Training script with MLflow integration
├── evaluate.py           # Standalone evaluation
├── lambda_handler.py     # AWS Lambda entry point
├── mlflow_config.py      # MLflow tracking and registry config
├── dvc.yaml              # DVC pipeline: preprocess -> train -> evaluate
├── params.yaml           # DVC parameters
├── Dockerfile            # FastAPI image
├── Dockerfile.lambda     # AWS Lambda container image
├── docker-compose.yml    # Local API + Streamlit + MLflow UI
└── requirements*.txt     # base / api / ci / lambda
```

## What works without the dataset

The repository is fully readable even without the dataset. The five Jupyter notebooks under `notebooks/` are committed with their output cells preserved, so the EDA, the U-Net segmentation experiment, the preprocessing pipeline and the modeling benchmarks can be reviewed by simply opening them. The test suite, the linter and the Docker build run on every CI push and do not depend on the dataset.

What requires the dataset: the live FastAPI / Streamlit / Lambda inference (needs the model checkpoints) and the full DVC pipeline (`dvc repro`).

## Setup

### Option A: team members with access to the project S3 remote

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch dataset and model checkpoints from the DVC remote
dvc pull

# 3. Run the local API
uvicorn api.main:app --reload

# 4. (Optional) Run the Streamlit demo
streamlit run app.py
```

### Option B: external users (no S3 access)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download HAM10000 manually from the public sources listed above
#    and place the files under data/ following the layout described
#    in the Dataset section.

# 3. Run the test suite, lint and Docker build (work without the dataset)
pytest tests/ -v
ruff check .
docker build -f Dockerfile -t melanoma-api:dev .

# 4. To run the pipeline end-to-end, you must additionally execute
#    notebooks/03_preprocessing.ipynb so that the JSON artifacts under
#    notebooks/outputs/preprocessing/ (referenced by dvc.yaml) exist.
```

## Reproducing the pipeline

The DVC pipeline in `dvc.yaml` has three stages: `preprocess`, `train`, `evaluate`.

```bash
dvc repro        # rebuilds the full pipeline incrementally
mlflow ui        # inspect the tracked experiment locally
```

Note: `dvc repro` requires both the raw dataset (via `dvc pull` or a manual HAM10000 download) and the preprocessing artifacts under `notebooks/outputs/preprocessing/` generated by the EDA / preprocessing notebooks. Once both are in place, the pipeline rebuilds incrementally.

## Tests

```bash
pytest tests/ -v
```

CI runs lint (ruff), the pytest suite, and the Docker build on every push and pull request. A monitoring job runs the drift report on a cron schedule every Monday at 09:00.

## Deployment

The model is served as an AWS Lambda container image behind an HTTP API Gateway v2 route `POST /predict`. The pipeline is automated end-to-end:

```bash
bash scripts/deploy_lambda.sh <account-id> <region> <lambda-role-arn>
```

Infrastructure as Code is available in two equivalent forms under `infra/`:

- `infra/cloudformation.yaml` (AWS CloudFormation)
- `infra/terraform/` (HashiCorp Terraform)

## Monitoring

Statistical drift detection uses the Kolmogorov-Smirnov test (for continuous features such as predicted melanoma probability) and the Chi-squared test (for categorical features such as the assigned triage zone).

```bash
python -m monitoring.run_monitoring --shift 0.25
```

The detector writes a JSON summary and PNG visualizations to `monitoring/reports/<timestamp>/`. When drift is flagged, `monitoring/retrain_trigger.py` invokes `dvc repro --force` and appends an audit entry to `monitoring/retrain_log.jsonl`.

## Feature Store (Feast)

Two feature views are defined per `image_id`:

| Feature View | Features |
|---|---|
| `lesion_classification` | MEL, NV, BCC, AKIEC, BKL, DF, VASC, binary_label, label, split |
| `preprocessing_stats` | mask_coverage_after_crop, hair_pixels_detected, final_height, final_width |

Two feature services expose them for training and serving:

| Service | Use |
|---|---|
| `melanoma_training_features` | Offline training |
| `melanoma_serving_features` | Online inference |

Quickstart:

```bash
python feature_store/scripts/prepare_sources.py     # CSVs -> Parquet
python feature_store/scripts/apply_registry.py      # feast apply
python feature_store/scripts/get_historical_features.py
python feature_store/scripts/materialize_online.py
```

## Logging

Structured JSON logging via `python-json-logger` is centralized in [`skin_app/logging_config.py`](skin_app/logging_config.py) and reused across the FastAPI service, the Lambda handler, and the monitoring entry points.
