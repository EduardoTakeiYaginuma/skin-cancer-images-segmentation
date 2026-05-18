"""Entry point para drift monitoring.

Uso:
    # Demo com dados sintéticos (sem argumentos):
    python -m monitoring.run_monitoring

    # Com dados reais:
    python -m monitoring.run_monitoring --reference data/ref_preds.csv --current data/prod_preds.csv

    # Aumentar shift para forçar drift (demo):
    python -m monitoring.run_monitoring --shift 0.3
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.drift_detector import run_drift_report
from monitoring.retrain_trigger import run as trigger_retrain

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _synthetic_reference(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Distribução de referência baseada no perfil real do HAM10000 (melanoma ~11%)."""
    rng = np.random.default_rng(seed)
    probs = np.clip(rng.beta(1.5, 8, size=n), 0, 1)
    zones = np.where(probs < 0.008, "negative", np.where(probs < 0.159, "review", "positive"))
    return pd.DataFrame({"melanoma_prob": probs, "triage_zone": zones})


def _synthetic_current(reference: pd.DataFrame, shift: float, n: int = 200) -> pd.DataFrame:
    """Produção simulada com shift na distribuição de probabilidades."""
    rng = np.random.default_rng(42)
    probs = np.clip(
        rng.choice(reference["melanoma_prob"].values, size=n, replace=True)
        + rng.normal(shift, 0.06, size=n),
        0, 1,
    )
    zones = np.where(probs < 0.008, "negative", np.where(probs < 0.159, "review", "positive"))
    return pd.DataFrame({"melanoma_prob": probs, "triage_zone": zones})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Detecta drift entre dados de treino e produção.")
    parser.add_argument("--reference", type=str, default=None, help="CSV de referência (treino/val)")
    parser.add_argument("--current", type=str, default=None, help="CSV de predições de produção")
    parser.add_argument("--shift", type=float, default=0.0,
                        help="Shift sintético na prob para demonstrar drift (default: 0.0 = sem drift)")
    parser.add_argument("--retrain", action="store_true",
                        help="Executa dvc repro quando drift detectado (default: dry-run)")
    args = parser.parse_args()

    if args.reference:
        reference = pd.read_csv(args.reference)
        logger.info("Referência carregada: %d linhas", len(reference))
    else:
        logger.info("Gerando referência sintética (demo)…")
        reference = _synthetic_reference()

    if args.current:
        current = pd.read_csv(args.current)
        logger.info("Produção carregada: %d linhas", len(current))
    else:
        logger.info("Gerando produção sintética com shift=%.2f…", args.shift)
        current = _synthetic_current(reference, shift=args.shift)

    output_dir = REPORTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run_drift_report(reference, current, output_dir)

    drifted = [k for k, v in results.items() if v.get("drift_detected")]
    if drifted:
        logger.warning("⚠  DRIFT DETECTADO em: %s", ", ".join(drifted))
        logger.warning("   Considerar re-treinamento do modelo.")
    else:
        logger.info("✓  Sem drift detectado. Modelo estável.")

    logger.info("Relatórios salvos em %s", output_dir)

    trigger_retrain(results, dry_run=not args.retrain)


if __name__ == "__main__":
    main()
