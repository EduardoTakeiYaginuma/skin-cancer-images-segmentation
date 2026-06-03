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

The heavy data (`data/images/`, `data/masks/`, `data/metadata.csv`) is versioned with DVC against an S3 remote. Run `dvc pull` to fetch.

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

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch dataset and model checkpoints from the S3 DVC remote
dvc pull

# 3. Run the local API
uvicorn api.main:app --reload

# 4. (Optional) Run the Streamlit demo
streamlit run app.py
```

## Reproducing the pipeline

The DVC pipeline declares three stages in `dvc.yaml`: `preprocess`, `train`, `evaluate`.

```bash
dvc repro        # rebuilds the full pipeline incrementally
mlflow ui        # inspect the tracked experiment locally
```

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
