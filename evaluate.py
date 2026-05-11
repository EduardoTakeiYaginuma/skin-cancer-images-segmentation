"""Standalone evaluation script — loads best MLflow model and writes metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mlflow_config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from train import SkinDataset, build_transforms, compute_metrics, evaluate

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    params_path = REPO_ROOT / "params.yaml"
    with params_path.open() as f:
        cfg = yaml.safe_load(f)["train"]

    norm_path = REPO_ROOT / "notebooks" / "outputs" / "preprocessing" / "normalization_stats.json"
    with norm_path.open() as f:
        norm = json.load(f)

    _, val_tf = build_transforms(
        input_size=cfg["input_size"],
        mean=norm["mean"],
        std=norm["std"],
        augmentation=False,
    )

    test_ds = SkinDataset(
        REPO_ROOT / "data" / "metadata" / "test_split.csv",
        transform=val_tf,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.to(device)

    _, test_labels, test_probs = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_labels, test_probs)
    print(f"Test metrics: {metrics}")

    metrics_path = REPO_ROOT / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {metrics_path}")


if __name__ == "__main__":
    main()
