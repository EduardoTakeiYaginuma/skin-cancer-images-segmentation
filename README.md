# Binary Melanoma Screening from Dermatoscopic Images

**Authors:** Gabriel Fernando Missaka Mendes | Eduardo Takei Yaginuma
**Course:** MLOps at Insper (26.1)

## Project Video

A 3-5 minute video explaining the project, the dataset and the main MLOps decisions:

🎥 **[Watch the video](https://youtu.be/8oEMfKIOY3Q)**

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

## What runs out of the box for anyone who clones the repo

The repository is designed so that a fresh clone is usable without any external credentials.

| Operation | Works on a fresh clone? |
|---|---|
| Read the report and all five Jupyter notebooks (outputs are committed) | yes |
| Run the test suite (`pytest tests/ -v`) | yes |
| Run the linter (`ruff check .`) | yes |
| Build the Docker image (`docker build -f Dockerfile`) | yes |
| Download the public dataset (`python scripts/download_dataset.py`) | yes, with a free Kaggle account |
| Run the notebooks from scratch with the downloaded dataset | yes |
| Train the model from scratch (`python train.py`) | yes, but takes hours without a GPU |
| Run `dvc repro` end-to-end | yes, after downloading the dataset and running notebook 03 |
| Hit the deployed Lambda endpoint with a real image | yes, the API Gateway URL is public |
| Run the local FastAPI / Streamlit with real inference | only if the user trained their own checkpoint, or has access to the team's DVC remote |

The pretrained model checkpoints (~459 MB) live on the team's private S3 (versioned via DVC) and are not redistributed in the repository. External users who need inference locally either retrain (the recipe is in `train.py`) or call the public Lambda endpoint.

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

# 2. Download HAM10000 (images + masks + metadata) into data/
#    Requires a free Kaggle account with API token at ~/.kaggle/kaggle.json
#    (token instructions: https://github.com/Kaggle/kaggle-api#api-credentials)
python scripts/download_dataset.py

# 3. Run the test suite, lint and Docker build (work even without the dataset)
pytest tests/ -v
ruff check .
docker build -f Dockerfile -t melanoma-api:dev .

# 4. To run the pipeline end-to-end, also execute notebooks/03_preprocessing.ipynb
#    so the JSON artifacts under notebooks/outputs/preprocessing/ exist
#    (they are referenced by dvc.yaml).
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

### Live endpoint

The current deployment is reachable at:

```
POST https://0tzj8c0o4f.execute-api.us-east-2.amazonaws.com/prod/predict
```

The handler expects the raw image bytes as the request body. Example:

```bash
curl -X POST "https://0tzj8c0o4f.execute-api.us-east-2.amazonaws.com/prod/predict?filename=test.jpg" \
  -H "Content-Type: image/jpeg" \
  --data-binary @your_image.jpg
```

Sample response:

```json
{
  "image_id": "59f9b0be4042",
  "melanoma_prob": 0.1675,
  "triage_zone": "positive",
  "triage_label": "Melanoma",
  "headline": "Alerta alto",
  "recommended_action": "Priorizar revisao especializada...",
  "latency_ms": 11082.66
}
```

First invocation can take 10-20 seconds because of the Lambda cold start (PyTorch and the model checkpoints are pulled from S3 into `/tmp`). Subsequent invocations on a warm container run inference in roughly 7-8 seconds on the configured 512 MB memory tier.

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
