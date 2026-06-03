"""Data drift detection using scipy stats: KS test (contínuo) + Chi-Square (categórico)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

logger = logging.getLogger(__name__)


def ks_test(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test para feature contínua."""
    stat, p_value = ks_2samp(reference, current)
    return {
        "test": "ks_2samp",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "drift_detected": bool(p_value < alpha),
        "alpha": alpha,
    }


def chi2_test(reference: pd.Series, current: pd.Series, alpha: float = 0.05) -> dict:
    """Chi-Square test para feature categórica."""
    categories = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts().reindex(categories, fill_value=0)
    cur_counts = current.value_counts().reindex(categories, fill_value=0)

    contingency = np.array([ref_counts.values, cur_counts.values])
    stat, p_value, dof, _ = chi2_contingency(contingency)
    return {
        "test": "chi2_contingency",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "dof": int(dof),
        "drift_detected": bool(p_value < alpha),
        "alpha": alpha,
        "categories": categories,
        "reference_counts": ref_counts.to_dict(),
        "current_counts": cur_counts.to_dict(),
    }


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    alpha: float = 0.05,
) -> dict:
    """Roda KS + Chi-Square, salva plots e JSON summary em output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if "melanoma_prob" in reference.columns and "melanoma_prob" in current.columns:
        result = ks_test(reference["melanoma_prob"].values, current["melanoma_prob"].values, alpha)
        results["melanoma_prob_ks"] = result
        _plot_histogram(
            reference["melanoma_prob"].values,
            current["melanoma_prob"].values,
            "melanoma_prob",
            result,
            output_dir / "melanoma_prob_dist.png",
        )
        logger.info(
            "KS test melanoma_prob → stat=%.4f  p=%.4f  drift=%s",
            result["statistic"], result["p_value"], result["drift_detected"],
        )

    if "triage_zone" in reference.columns and "triage_zone" in current.columns:
        result = chi2_test(reference["triage_zone"], current["triage_zone"], alpha)
        results["triage_zone_chi2"] = result
        _plot_bar(
            reference["triage_zone"],
            current["triage_zone"],
            result,
            output_dir / "triage_zone_dist.png",
        )
        logger.info(
            "Chi-Square triage_zone → stat=%.4f  p=%.4f  drift=%s",
            result["statistic"], result["p_value"], result["drift_detected"],
        )

    summary_path = output_dir / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Sumário salvo em %s", summary_path)

    return results


def _plot_histogram(ref: np.ndarray, cur: np.ndarray, name: str, result: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref, bins=30, alpha=0.55, label="Referência (treino)", density=True)
    ax.hist(cur, bins=30, alpha=0.55, label="Produção", density=True)
    status = "DRIFT DETECTADO" if result["drift_detected"] else "sem drift"
    ax.set_title(f"{name} :  KS p={result['p_value']:.4f}  ({status})")
    ax.set_xlabel(name)
    ax.set_ylabel("Densidade")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_bar(ref: pd.Series, cur: pd.Series, result: dict, path: Path) -> None:
    categories = result["categories"]
    ref_pct = np.array([result["reference_counts"].get(c, 0) for c in categories], dtype=float)
    cur_pct = np.array([result["current_counts"].get(c, 0) for c in categories], dtype=float)
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, ref_pct, width, label="Referência (treino)")
    ax.bar(x + width / 2, cur_pct, width, label="Produção")
    status = "DRIFT DETECTADO" if result["drift_detected"] else "sem drift"
    ax.set_title(f"triage_zone :  χ² p={result['p_value']:.4f}  ({status})")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Proporção")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
