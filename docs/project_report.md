# MLOps Final Project Report: Binary Melanoma Screening

**Authors:** Gabriel Fernando Missaka Mendes | Eduardo Takei Yaginuma
**Course:** MLOps at Insper (26.1)
**Repository:** [`insper-classroom/26-1-mlops-project-gabriel-e-edu`](https://github.com/insper-classroom/26-1-mlops-project-gabriel-e-edu)

---

## 1. Introduction

Melanoma is the most aggressive form of skin cancer, and early triage from dermatoscopic images can have direct clinical impact. While accurate models exist, the gap between a trained notebook and a model that is auditable, reproducible, observable, and safely deployable is the actual blocker to clinical adoption.

This project takes an existing binary-classification model (melanoma vs. non-melanoma) and **operationalizes it end-to-end**, applying the full MLOps stack required by the Insper rubric: experiment tracking, data versioning, feature store, automated deployment pipeline, infrastructure as code, structured logging, continuous integration, production monitoring with statistical drift detection, and automatic retraining on degradation. The goal of the project is not to push the state of the art on the modeling side, but to demonstrate professional operationalization of a non-trivial deep-learning model.

---

## 2. Dataset, Storage and Preprocessing

### 2.1 Data Collection and Storage

The project uses the **HAM10000** dataset (Human Against Machine with 10,000 training images) from the ISIC archive, comprising **10,015 dermatoscopic images** annotated across seven diagnostic classes (`MEL`, `NV`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC`), together with binary segmentation masks for each lesion.

The raw dataset (`data/images/`, `data/masks/`, `data/metadata.csv`) and the precomputed train/val/test splits (`data/metadata/*_split.csv`) are tracked locally with `.dvc` pointer files. Heavy binary content lives in an S3 remote configured via `.dvc/config`; this allows any contributor to reproduce the data state with a single `dvc pull`, without bloating Git history with multi-gigabyte assets.

### 2.2 Preprocessing Pipeline

The preprocessing pipeline is declared in `dvc.yaml` as a three-stage DAG:

- **`preprocess`**: runs `notebooks/03_preprocessing.ipynb` converted to a script. It performs (i) mask-guided cropping centered on the lesion, (ii) per-channel normalization using statistics computed once and persisted to `notebooks/outputs/preprocessing/normalization_stats.json`, (iii) resizing to `224×224` for the classifier and `64×64` for the segmentation model, and (iv) emits a `treated_manifest.csv` describing each exported sample.
- **`train`**: runs `train.py` using the treated manifest, the normalization stats and the splits as DVC dependencies. Output: a tracked MLflow run with the trained checkpoint.
- **`evaluate`**: runs `evaluate.py` against the held-out test split and writes `metrics.json` as a DVC-tracked metric file.

Because every stage declares its `deps`, `outs` and (where relevant) `params`, the pipeline is fully reproducible: changing a preprocessing parameter in `params.yaml` invalidates only the downstream stages, and `dvc repro` can run the minimum required work.

### 2.3 Class Balancing Strategy

Melanoma represents only ~11% of HAM10000 (1,113 of 10,015 images). The training procedure addresses this in two complementary ways: at the data layer, the non-melanoma class is downsampled to a 3:1 ratio relative to melanoma when building the effective training set; at the loader layer, a `WeightedRandomSampler` increases the sampling probability of melanoma cases and adds an extra multiplier to the melanocytic nevus (`NV`) class, which is the visually hardest negative.

---

## 3. Model

The classifier is a **ResNet50** initialized with ImageNet weights and fine-tuned end-to-end for binary classification with a single sigmoid output. The loss is `BCEWithLogitsLoss`; optimization uses AdamW with cosine annealing and early stopping on validation AUC.

To support clinical triage rather than a hard binary decision, the model exposes **three zones** via a dual-threshold strategy. A high threshold `T_HIGH` is selected on the validation set as the value that achieves at least 85% sensitivity while preserving an acceptable specificity floor. A low threshold `T_LOW` is set at the 2nd percentile of melanoma probabilities on the validation set. Predictions below `T_LOW` are routed to the automatic-dismissal zone ("negative"), predictions above `T_HIGH` to the high-risk zone ("positive"), and the remaining cases to a manual-review zone ("review"). Both thresholds are persisted alongside the checkpoint so the serving layer applies the same operating point used at training time.

The final selected configuration is **ResNet50 with online augmentation at 224×224**, with a test AUC of **0.9128** and a melanoma capture rate of 85.6% in the high-confidence zone.

---

## 4. MLOps Architecture Overview

The operational stack is composed of seven layers, each addressing one or more rubric items:

| Layer | Tool | Rubric item served |
|---|---|---|
| Data versioning | DVC + S3 remote | Data versioning (C) |
| Feature store | Feast (SQLite registry + online/offline store) | Feature store (C) |
| Experiment tracking | MLflow + Model Registry | MLOps framework (B/A) |
| Deployment pipeline | `deploy_lambda.sh` → ECR → AWS Lambda | Automated deployment pipeline (B/A) |
| Infrastructure as Code | CloudFormation **and** Terraform | IaC (B/A) |
| Production monitoring | KS + Chi² drift report (scipy), cron-scheduled | Monitor performance in production (B/A) |
| Degradation handling | `retrain_trigger.py` → `dvc repro --force` | Deals with degradation (B/A) |
| Continuous integration | GitHub Actions (lint + tests + docker build + cron monitoring) | Project runs without errors (C) |
| Structured logging | `python-json-logger` | Uses logging (C) |

---

## 5. Experiment Tracking: MLflow

All training runs are tracked with MLflow. The tracking URI and experiment name are centralized in `mlflow_config.py`; the URI defaults to a local SQLite backend and can be overridden via the `MLFLOW_TRACKING_URI` environment variable to point at a managed tracking server.

For each run, `train.py` logs the full set of hyperparameters (model architecture, image size, batch size, learning rate, epochs, augmentation flag, random seed and target device), per-epoch metrics (`train_loss`, `train_auc`, `val_loss`, `val_auc`), end-of-run test metrics (`test_auc`, `test_sensitivity`, `test_specificity`, `test_precision`, `test_f1`, `test_f2`), and two visual artifacts: the ROC curve and the confusion matrix.

The trained model is also registered in the **MLflow Model Registry** under the name `melanoma-classifier`. When the test AUC exceeds the production-promotion threshold of 0.85, the new version is automatically tagged with the `production` alias using `MlflowClient.set_registered_model_alias`. The inference config JSON (containing `T_LOW`, `T_HIGH`, normalization stats and metric snapshot) is logged as a run artifact so that the serving layer can always reproduce the exact operating point of the checkpoint it loads.

---

## 6. Data Versioning and Feature Store

The project satisfies the data-versioning rubric requirement **twice**, deliberately.

### 6.1 DVC

`dvc.yaml` defines the data + training pipeline; the S3 remote (`.dvc/config`) stores the actual binaries. The combination of `params.yaml` (pipeline parameters), DVC dependency tracking and remote storage means that a new contributor can rebuild any historical state of the dataset with `dvc pull && dvc repro <stage>`.

### 6.2 Feast

A complementary feature store is implemented under `feature_store/feature_repo/`. Two feature views materialize information per `image_id`:

- **`lesion_classification`**: the 7-way one-hot encoding plus the binary label and the split identifier, sourced from the train/val/test split CSVs.
- **`preprocessing_stats`**: per-image final dimensions, mask coverage after cropping, and hair-pixel count, sourced from the preprocessing manifest.

The registry uses SQLite, the online store is SQLite, and the offline store is a file provider. A `melanoma_serving_features` feature service exposes only the preprocessing stats for online retrieval, while `melanoma_training_features` exposes the full set for offline training. Scripts under `feature_store/scripts/` cover the full lifecycle: `prepare_sources.py` converts the project CSVs to Parquet, `apply_registry.py` runs `feast apply`, `get_historical_features.py` retrieves the training set, and `materialize_online.py` warms the online store.

---

## 7. Deployment Pipeline

The trained model is deployed as a container-image **AWS Lambda function** behind an HTTP API Gateway v2. The pipeline is automated end-to-end by `scripts/deploy_lambda.sh`:

1. Authenticate Docker against ECR.
2. Create the ECR repository if it does not exist.
3. Build the Lambda image from `Dockerfile.lambda` (Python 3.11 base image, PyTorch CPU wheels, OpenCV system libraries).
4. Tag and push the image to ECR.
5. Either create the Lambda function (first run) or update its image (subsequent runs) via `aws lambda create-function` / `update-function-code`.

The Lambda entry point (`lambda_handler.py`) downloads the classifier and segmentation checkpoints from S3 to `/tmp` on cold start, instantiates a cached predictor, and exposes the same triage contract used by the local FastAPI service. A live invocation returns a JSON body with `melanoma_prob`, `triage_zone` (`negative` / `review` / `positive`), `triage_label`, `recommended_action` and `latency_ms`.

In parallel, a local **FastAPI** service (`api/main.py`) provides `/health` and `/predict` endpoints for development and integration testing, and a **Streamlit** application (`app.py`) offers an interactive frontend for demonstration purposes.

---

## 8. Infrastructure as Code

Infrastructure is described declaratively in **two equivalent IaC implementations**, both maintained under `infra/`:

- **`infra/cloudformation.yaml`**: AWS CloudFormation template that provisions the Lambda function (image package, IAM role, S3 read access, configurable memory and timeout), the HTTP API Gateway v2 instance, the `AWS_PROXY` integration, the `POST /predict` route, the `prod` stage and the resource-based permission allowing API Gateway to invoke the Lambda.
- **`infra/terraform/`**: a Terraform module (`main.tf`, `variables.tf`, `outputs.tf`) provisioning the same set of resources via the `hashicorp/aws` provider, validated with `terraform validate`.

Both implementations are fully parameterized (region, ECR image URI, Lambda role ARN, model S3 bucket, memory size, timeout), so the entire production environment can be stood up or torn down with a single command.

---

## 9. Logging

Structured logging is implemented in `skin_app/logging_config.py` using `python-json-logger`. The configured handler emits one JSON object per log record with fields `timestamp`, `level`, `module` and `message`, suitable for ingestion by any modern log aggregator (CloudWatch Logs Insights, Elasticsearch, Datadog, etc.) without further parsing.

The same logger is reused across the FastAPI service (`api/main.py`), the Lambda handler (`lambda_handler.py`), and the monitoring entry points (`monitoring/drift_detector.py`, `monitoring/retrain_trigger.py`, `monitoring/run_monitoring.py`). This ensures consistent log shape regardless of execution environment.

---

## 10. Continuous Integration

The repository ships a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push and pull request, and additionally on a weekly cron schedule. The workflow contains four jobs:

- **`lint`**: runs `ruff check .` against the entire codebase.
- **`test`**: installs PyTorch CPU wheels and the project's CI requirements, then runs `pytest tests/ -v`. The suite covers FastAPI endpoint contracts (with a mocked predictor), the preprocessing utility functions, and the Feast Parquet schema. Current status: 17 tests collected, 15 passing, 2 skipped (the skipped tests require artifacts generated by the preprocessing notebook).
- **`docker-build`**: verifies that the production Dockerfile builds without errors on a clean Ubuntu runner.
- **`monitoring`**: scheduled job (cron `0 9 * * 1`, i.e., every Monday at 09:00) that runs the drift detection report and uploads the artifacts so degradation is visible without manual intervention.

The same workflow runs on both the working repository and the Insper Classroom repository. All recent pushes have completed green.

---

## 11. Production Monitoring

Once a model is live, drift is the operational risk that matters most. `monitoring/drift_detector.py` compares a reference distribution (typically taken from the training period) against a current distribution (taken from recent production predictions) using two complementary statistical tests:

- **Kolmogorov-Smirnov** on continuous features (e.g. the predicted melanoma probability), to detect a shift in the full predictive distribution. The KS test is non-parametric and sensitive to both location and shape changes.
- **Chi-squared** on categorical features (e.g. the assigned triage zone with categories `negative` / `review` / `positive`), to detect distributional shifts in the discrete output buckets that the downstream clinical workflow depends on.

Both tests use a significance level of 0.05. For each evaluated feature, the detector writes a `drift_summary.json` containing the test statistic, p-value, drift flag, alpha, and (for the categorical case) the full reference and current contingency tables. The detector additionally renders two PNG visualizations (histogram for the continuous feature, bar plot for the categorical feature) so that the report is both machine- and human-readable.

The monitoring entry point (`monitoring/run_monitoring.py`) supports both real production data (CSV inputs) and synthetic data with a configurable distribution shift, which is useful for demonstrating the alarm in a controlled way. Three example reports are committed under `monitoring/reports/`.

---

## 12. Handling Performance Degradation

When the drift detector flags one or more features, `monitoring/retrain_trigger.py` is responsible for the response. Its logic is intentionally minimal: read the drift summary, determine the set of drifted features, and (i) write a structured JSON-Lines entry to `monitoring/retrain_log.jsonl` for audit purposes and (ii) invoke `dvc repro --force` to rebuild the data and training pipelines from scratch using the current data state.

The trigger supports a `--dry-run` flag so that the monitoring job can run in observation mode in CI without consuming compute, and a live flag to perform the actual retraining when desired. The full chain (*drift detected → log → DVC repro → MLflow registered → conditional production promotion*) closes the operational loop without manual intervention.

---

## 13. Reproducibility

A new contributor can bring up the project end-to-end with the following sequence, fully documented in the repository's `README.md`:

1. Clone the repository.
2. `pip install -r requirements.txt` (or use the provided `Dockerfile` for a sealed environment).
3. `dvc pull` to fetch the dataset and model checkpoints from the S3 remote.
4. `dvc repro` to rebuild the preprocessing, training and evaluation pipeline.
5. `uvicorn api.main:app` to run the local FastAPI service, or `streamlit run app.py` to run the demo UI.

The repository ships four scoped `requirements*.txt` files (base, API-only, CI, Lambda) so that each environment installs only what it actually needs, and a `Dockerfile` plus `Dockerfile.lambda` for hermetic builds.

---

## 14. Authors and Contributions

Both authors contributed across all stages of the project through extensive pair programming, and every commit on the final branch credits both authors via Git `Co-authored-by` trailers.

**Gabriel Fernando Missaka Mendes** led the data exploration phase, the design of the preprocessing pipeline, the clinical interpretation of the threshold strategy (`T_LOW` / `T_HIGH`), the structure of the project repository and documentation, and the final report and video deliverables.

**Eduardo Takei Yaginuma** led the modeling experimentation across architectures and image resolutions, the implementation of the MLflow tracking and Model Registry integration, the construction of the AWS deployment pipeline (Docker image, ECR, Lambda, API Gateway), the two Infrastructure-as-Code implementations (CloudFormation and Terraform), and the drift-detection and retrain-trigger components.

---

## 15. Conclusion

The delivered project covers all C-level requirements of the rubric and all five B-level items:

- **Experiment tracking framework**: MLflow with Model Registry and conditional alias promotion.
- **Automated deployment pipeline**: `deploy_lambda.sh` automating Docker build, ECR push, and Lambda create/update; the endpoint is currently live.
- **Infrastructure as Code**: both a CloudFormation template and a Terraform module, fully parameterized and equivalent.
- **Production monitoring**: KS and Chi-squared drift detection executed on a weekly CI schedule, with JSON summary and visual artifacts.
- **Degradation handling**: automatic retrain trigger via `dvc repro --force`, logged for auditability.

The project demonstrates that a non-trivial deep-learning model for medical-image triage can be operationalized with the same engineering discipline as any other production system: every piece of state is versioned, every action is reproducible, every deployment is automated, and every output is observable.

---

## References

1. AMERICAN CANCER SOCIETY. *Cancer Facts & Figures 2024*. Atlanta: American Cancer Society, 2024.
2. CARAVIELLO, Camila et al. *Melanoma Skin Cancer: A Comprehensive Review of Current Knowledge*. *Cancers*, Basel, v. 17, n. 2920, p. 1-35, 2025.
3. VIEIRA, Larissa Silva Fontaine; BRANDÃO, Byron José Figueiredo. *Diagnosis and prevention of melanoma: a systematic review*. *BWS Journal*, [s. l.], v. 5, e220900160, p. 1-10, Sept. 2022.
4. TSCHANDL, Philipp; ROSENDAHL, Cliff; KITTLER, Harald. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. *Scientific Data*, v. 5, n. 180161, 2018.
