"""Standalone training script with MLflow tracking and Model Registry."""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml
from mlflow import MlflowClient
from PIL import Image
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mlflow_config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SkinDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        transform: transforms.Compose,
        neg_ratio: float | None = None,
        seed: int = 42,
    ) -> None:
        df = pd.read_csv(csv_path)

        if neg_ratio is not None:
            mel = df[df["binary_label"] == 1]
            non_mel = df[df["binary_label"] == 0]
            n_neg = int(len(mel) * neg_ratio)
            rng = np.random.default_rng(seed)
            sampled_neg = non_mel.sample(n=min(n_neg, len(non_mel)), random_state=rng.integers(0, 2**31))
            df = pd.concat([mel, sampled_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

        self.records = df[["img_path", "binary_label"]].values.tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.records[idx]
        image = Image.open(REPO_ROOT / img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor(float(label), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(
    input_size: int,
    mean: list[float],
    std: list[float],
    augmentation: bool,
) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=mean, std=std)

    if augmentation:
        train_tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])

    val_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(arch: str) -> nn.Module:
    return timm.create_model(arch, pretrained=True, num_classes=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    auc_score = float(roc_auc_score(labels, probs))
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    precision = float(tp / max(tp + fp, 1))
    f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "auc": auc_score,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Plots (logged as MLflow artifacts)
# ---------------------------------------------------------------------------

def _roc_curve_bytes(labels: np.ndarray, probs: np.ndarray) -> bytes:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve — Validation")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _confusion_matrix_bytes(labels: np.ndarray, probs: np.ndarray, threshold: float) -> bytes:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-mel", "Melanoma"])
    ax.set_yticklabels(["Non-mel", "Melanoma"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, float, np.ndarray, np.ndarray]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state: dict = {}
    best_labels: np.ndarray = np.array([])
    best_probs: np.ndarray = np.array([])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= max(len(train_loader.dataset), 1)  # type: ignore[arg-type]
        val_loss, val_labels, val_probs = evaluate(model, val_loader, device)
        metrics = compute_metrics(val_labels, val_probs)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": metrics["auc"],
                "val_sensitivity": metrics["sensitivity"],
                "val_specificity": metrics["specificity"],
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"AUC={metrics['auc']:.4f} | "
            f"sens={metrics['sensitivity']:.4f} | "
            f"spec={metrics['specificity']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_labels = val_labels
            best_probs = val_probs

    model.load_state_dict(best_state)
    return model, best_auc, best_labels, best_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    params_path = REPO_ROOT / "params.yaml"
    with params_path.open() as f:
        cfg = yaml.safe_load(f)["train"]

    norm_stats_path = REPO_ROOT / "notebooks" / "outputs" / "preprocessing" / "normalization_stats.json"
    with norm_stats_path.open() as f:
        norm = json.load(f)
    mean: list[float] = norm["mean"]
    std: list[float] = norm["std"]

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    train_tf, val_tf = build_transforms(
        input_size=cfg["input_size"],
        mean=mean,
        std=std,
        augmentation=cfg["augmentation"],
    )

    train_ds = SkinDataset(
        REPO_ROOT / "data" / "metadata" / "train_split.csv",
        transform=train_tf,
        neg_ratio=cfg["neg_ratio"],
        seed=cfg["seed"],
    )
    val_ds = SkinDataset(
        REPO_ROOT / "data" / "metadata" / "val_split.csv",
        transform=val_tf,
    )
    test_ds = SkinDataset(
        REPO_ROOT / "data" / "metadata" / "test_split.csv",
        transform=val_tf,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params({
            "model_arch": cfg["model_arch"],
            "input_size": cfg["input_size"],
            "batch_size": cfg["batch_size"],
            "lr": cfg["lr"],
            "epochs": cfg["epochs"],
            "augmentation": cfg["augmentation"],
            "neg_ratio": cfg["neg_ratio"],
            "seed": cfg["seed"],
            "device": str(device),
            "norm_mean": mean,
            "norm_std": std,
        })

        model = build_model(cfg["model_arch"])
        model, best_val_auc, val_labels, val_probs = train(
            model, train_loader, val_loader,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            device=device,
        )

        # Test set evaluation
        _, test_labels, test_probs = evaluate(model, test_loader, device)
        test_metrics = compute_metrics(test_labels, test_probs)

        mlflow.log_metrics({
            "test_auc": test_metrics["auc"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_precision": test_metrics["precision"],
            "test_f1": test_metrics["f1"],
            "best_val_auc": best_val_auc,
        })

        print(f"\nTest metrics: {test_metrics}")

        # Log artifacts
        mlflow.log_image(
            image=plt.imread(io.BytesIO(_roc_curve_bytes(val_labels, val_probs))),  # type: ignore[arg-type]
            artifact_file="roc_curve_val.png",
        )
        mlflow.log_image(
            image=plt.imread(io.BytesIO(_confusion_matrix_bytes(test_labels, test_probs, threshold=0.5))),  # type: ignore[arg-type]
            artifact_file="confusion_matrix_test.png",
        )
        mlflow.log_artifact(str(params_path))

        # Save checkpoint locally and log
        ckpt_dir = REPO_ROOT / "outputs" / "models" / "mlflow"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{cfg['model_arch']}_{cfg['input_size']}x{cfg['input_size']}.pt"
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")

        # Register model
        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Transition to Production if test AUC meets minimum threshold
        if test_metrics["auc"] >= 0.85:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            latest_version = max(int(v.version) for v in versions)
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="production",
                version=str(latest_version),
            )
            print(f"Model v{latest_version} aliased as 'production' (AUC={test_metrics['auc']:.4f})")
        else:
            print(f"AUC {test_metrics['auc']:.4f} below 0.85 — model NOT promoted to production.")


if __name__ == "__main__":
    main()
