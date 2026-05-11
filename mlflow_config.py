from __future__ import annotations

import os

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
EXPERIMENT_NAME = "skin-cancer-classification"
REGISTERED_MODEL_NAME = "melanoma-classifier"
