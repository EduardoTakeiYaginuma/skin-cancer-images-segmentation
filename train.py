"""Standalone training script — ResNet50 base_224x224 com MLflow tracking."""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split  # still used for downsampling
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mlflow_config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Hiperparâmetros (espelha o notebook 04_modeling — ResNet50 base_224x224)
# ---------------------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 16
PATIENCE = 4
NON_MELANOMA_RATIO = 3.0
MELANOMA_RECALL_TARGET = 0.85
LOW_FNR_BUDGET = 0.02        # T_LOW: max 2% melanomas abaixo do threshold
MIN_SPECIFICITY = 0.10
HARD_NEGATIVE_LABEL = "NV"
HARD_NEGATIVE_MULTIPLIER = 2.0
MODEL_ARCH = "resnet50"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

ONLINE_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.15),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=20,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
])


def _load_rgb(path: str | Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _to_tensor(img: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    t = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t - m) / s


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mean: list[float], std: list[float],
                 augment: bool = False) -> None:
        self.records = df.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[idx]
        img = _load_rgb(row["img_path"], IMAGE_SIZE)
        if self.augment:
            img = ONLINE_AUG(image=img)["image"]
        return _to_tensor(img, self.mean, self.std), torch.tensor(float(row["binary_label"]))


def _build_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    pos = max(int((df["binary_label"] == 1).sum()), 1)
    neg = max(int((df["binary_label"] == 0).sum()), 1)
    pos_w = neg / pos
    weights = [
        (pos_w if row.binary_label == 1 else 1.0) * (HARD_NEGATIVE_MULTIPLIER if getattr(row, "label", None) == HARD_NEGATIVE_LABEL else 1.0)
        for row in df.itertuples()
    ]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------

def compute_norm_stats(df: pd.DataFrame, sample_size: int = 512) -> dict[str, list[float]]:
    sample = df.sample(n=min(sample_size, len(df)), random_state=SEED)
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sq = np.zeros(3, dtype=np.float64)
    n_px = 0
    for row in sample.itertuples():
        img = _load_rgb(row.img_path, IMAGE_SIZE).astype(np.float32) / 255.0
        px = img.reshape(-1, 3)
        ch_sum += px.sum(0)
        ch_sq += (px ** 2).sum(0)
        n_px += len(px)
    mean = (ch_sum / n_px).round(6).tolist()
    std = np.sqrt(np.maximum(ch_sq / n_px - (ch_sum / n_px) ** 2, 1e-12)).round(6).tolist()
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Métricas e threshold
# ---------------------------------------------------------------------------

def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = f1_score(labels, preds, zero_division=0)
    beta_sq = 4.0
    f2 = (1 + beta_sq) * prec * sens / max(beta_sq * prec + sens, 1e-12) if (prec + sens) > 0 else 0.0
    return dict(auc=float(roc_auc_score(labels, probs)), threshold=threshold,
                sensitivity=sens, specificity=spec, precision=prec,
                f1=f1, f2=f2, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                preds=preds, labels=labels, probs=probs)


def select_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Escolhe threshold priorizando sensibilidade >= MELANOMA_RECALL_TARGET."""
    candidates = np.sort(np.unique(np.round(np.concatenate(([0.0], probs, [1.0])), 6)))
    rows = [_binary_metrics(probs, labels, float(t)) for t in candidates]
    feasible = [r for r in rows if r["sensitivity"] >= MELANOMA_RECALL_TARGET and r["specificity"] >= MIN_SPECIFICITY]
    if feasible:
        return max(feasible, key=lambda r: (r["specificity"], r["f2"]))["threshold"]
    feasible2 = [r for r in rows if r["sensitivity"] >= MELANOMA_RECALL_TARGET and r["threshold"] > 0]
    if feasible2:
        return max(feasible2, key=lambda r: (r["specificity"], r["f2"]))["threshold"]
    return max(rows, key=lambda r: r["f2"])["threshold"]


def compute_t_low(val_labels: np.ndarray, val_probs: np.ndarray) -> float:
    mel_probs = val_probs[val_labels == 1]
    return float(np.quantile(mel_probs, LOW_FNR_BUDGET))


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, probs_list, labels_list = 0.0, [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs).squeeze(1)
        total_loss += criterion(logits, lbls).item() * len(lbls)
        probs_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels_list.extend(lbls.cpu().numpy().tolist())
    n = max(len(labels_list), 1)
    return total_loss / n, np.array(labels_list), np.array(probs_list)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_state, patience_counter = -float("inf"), {}, 0
    best_val_labels: np.ndarray = np.array([])
    best_val_probs: np.ndarray = np.array([])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_probs, train_labels = 0.0, [], []
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(lbls)
            train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            train_labels.extend(lbls.cpu().numpy().tolist())
        scheduler.step()

        train_loss /= max(len(train_labels), 1)
        train_auc = float(roc_auc_score(train_labels, train_probs)) if len(set(train_labels)) > 1 else 0.0
        val_loss, val_labels, val_probs = _eval_epoch(model, val_loader, device)
        val_auc = float(roc_auc_score(val_labels, val_probs)) if len(set(val_labels)) > 1 else 0.0

        improved = val_auc > best_auc
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}" + (" <<<" if improved else ""))

        mlflow.log_metrics({"train_loss": train_loss, "train_auc": train_auc,
                            "val_loss": val_loss, "val_auc": val_auc}, step=epoch)

        if improved:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_labels = val_labels
            best_val_probs = val_probs
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping na época {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, best_val_labels, best_val_probs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _roc_png(labels: np.ndarray, probs: np.ndarray, title: str) -> bytes:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title); ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def _cm_png(labels: np.ndarray, preds: np.ndarray) -> bytes:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-mel", "Melanoma"]); ax.set_yticklabels(["Non-mel", "Melanoma"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    splits_dir = REPO_ROOT / "data" / "metadata"
    for name in ("train_split.csv", "val_split.csv", "test_split.csv"):
        p = splits_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Split CSV não encontrado: {p}  (rode dvc pull)")

    train_full = pd.read_csv(splits_dir / "train_split.csv")
    val_df = pd.read_csv(splits_dir / "val_split.csv")
    test_df = pd.read_csv(splits_dir / "test_split.csv")

    # Substitui img_path pelo export_path das imagens tratadas (pré-processadas/recortadas),
    # que é o mesmo input usado no notebook 04_modeling para o melhor resultado.
    treated_manifest_path = REPO_ROOT / "data" / "processed" / "treated_manifest.csv"
    if not treated_manifest_path.exists():
        raise FileNotFoundError(
            f"Execute 02_preprocessing.ipynb primeiro. Esperado: {treated_manifest_path}"
        )
    treated = pd.read_csv(treated_manifest_path)[["image_id", "export_path"]]
    for df_ in (train_full, val_df, test_df):
        df_.rename(columns={"image": "image_id"}, inplace=True)
        merged = df_.merge(treated, on="image_id", how="left")
        missing = merged["export_path"].isna().sum()
        if missing:
            raise ValueError(f"{missing} amostras sem imagem tratada — rode 02_preprocessing.ipynb")
        df_["img_path"] = merged["export_path"].values

    # Downsampling da classe negativa no treino (ratio 1:3)
    mel = train_full[train_full["binary_label"] == 1]
    non_mel = train_full[train_full["binary_label"] == 0]
    n_neg = min(len(non_mel), int(round(len(mel) * NON_MELANOMA_RATIO)))
    non_mel_sampled, _ = train_test_split(non_mel, train_size=n_neg,
                                          stratify=non_mel["label"], random_state=SEED)
    train_df = pd.concat([mel, non_mel_sampled]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Melanoma no treino: {int((train_df['binary_label']==1).sum())} / {len(train_df)}")

    # Normalização computada sobre o treino (igual ao notebook)
    print("Calculando estatísticas de normalização...")
    norm = compute_norm_stats(train_df)
    mean, std = norm["mean"], norm["std"]
    print(f"Mean: {mean} | Std: {std}")

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    train_ds = SkinDataset(train_df, mean, std, augment=True)   # aug_224x224: melhor resultado clínico
    val_ds = SkinDataset(val_df, mean, std, augment=False)
    test_ds = SkinDataset(test_df, mean, std, augment=False)

    sampler = _build_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{MODEL_ARCH}_aug_{IMAGE_SIZE}x{IMAGE_SIZE}"):
        mlflow.log_params({
            "model_arch": MODEL_ARCH,
            "input_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "augmentation": True,
            "neg_ratio": NON_MELANOMA_RATIO,
            "melanoma_recall_target": MELANOMA_RECALL_TARGET,
            "hard_negative_label": HARD_NEGATIVE_LABEL,
            "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER,
            "seed": SEED,
            "device": str(device),
            "norm_mean": mean,
            "norm_std": std,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        })

        model = timm.create_model(MODEL_ARCH, pretrained=True, num_classes=1)
        model, val_labels, val_probs = train(model, train_loader, val_loader, device)

        # Threshold clínico no val set
        t_high = select_threshold(val_probs, val_labels)
        t_low = compute_t_low(val_labels, val_probs)
        print(f"T_LOW={t_low:.6f} | T_HIGH={t_high:.6f}")

        # Avaliação no test set
        _, test_labels, test_probs = _eval_epoch(model, test_loader, device)
        test_m = _binary_metrics(test_probs, test_labels, t_high)

        mlflow.log_metrics({
            "test_auc": test_m["auc"],
            "test_sensitivity": test_m["sensitivity"],
            "test_specificity": test_m["specificity"],
            "test_precision": test_m["precision"],
            "test_f1": test_m["f1"],
            "test_f2": test_m["f2"],
            "test_tp": test_m["tp"],
            "test_fp": test_m["fp"],
            "test_fn": test_m["fn"],
            "test_tn": test_m["tn"],
            "t_high": t_high,
            "t_low": t_low,
        })

        print(f"\nTest | AUC={test_m['auc']:.4f} | sens={test_m['sensitivity']:.4f} | "
              f"spec={test_m['specificity']:.4f} | FN={test_m['fn']}")

        # Artefatos visuais
        mlflow.log_image(plt.imread(io.BytesIO(_roc_png(test_labels, test_probs, "ROC — Test Set"))),
                         artifact_file="roc_curve_test.png")
        mlflow.log_image(plt.imread(io.BytesIO(_cm_png(test_labels, test_m["preds"]))),
                         artifact_file="confusion_matrix_test.png")

        # Salva checkpoint local
        ckpt_dir = REPO_ROOT / "outputs" / "models" / "mlflow"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{MODEL_ARCH}_aug_{IMAGE_SIZE}x{IMAGE_SIZE}.pt"
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")

        # Salva config de inferência com T_LOW e T_HIGH
        config_out = {
            "model_arch": MODEL_ARCH,
            "input_size": IMAGE_SIZE,
            "norm_mean": mean,
            "norm_std": std,
            "t_low": t_low,
            "t_high": t_high,
            "test_auc": test_m["auc"],
            "test_sensitivity": test_m["sensitivity"],
            "test_specificity": test_m["specificity"],
        }
        config_path = ckpt_dir / "inference_config.json"
        with config_path.open("w") as f:
            json.dump(config_out, f, indent=2)
        mlflow.log_artifact(str(config_path))

        # Registra no Model Registry
        mlflow.pytorch.log_model(model, artifact_path="model",
                                 registered_model_name=REGISTERED_MODEL_NAME)

        if test_m["auc"] >= 0.85:
            versions = MlflowClient(MLFLOW_TRACKING_URI).search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'"
            )
            latest = max(int(v.version) for v in versions)
            MlflowClient(MLFLOW_TRACKING_URI).set_registered_model_alias(
                REGISTERED_MODEL_NAME, "production", str(latest)
            )
            print(f"Modelo v{latest} promovido para 'production'.")
        else:
            print(f"AUC {test_m['auc']:.4f} < 0.85 — modelo NÃO promovido.")


if __name__ == "__main__":
    main()
