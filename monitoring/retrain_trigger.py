"""Trigger automático de re-treinamento quando drift é detectado.

Fluxo:
  1. Recebe o dict de resultados do run_drift_report()
  2. Verifica se alguma feature ultrapassou o threshold de drift
  3. Se sim, executa `dvc repro` para re-treinar via pipeline DVC
  4. Loga o evento com timestamp

Uso standalone:
    python -m monitoring.retrain_trigger --drift-summary monitoring/reports/.../drift_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def should_retrain(drift_results: dict) -> tuple[bool, list[str]]:
    """Retorna (True, [features com drift]) se re-treino é necessário."""
    drifted = [k for k, v in drift_results.items() if v.get("drift_detected")]
    return bool(drifted), drifted


def trigger_retrain(dry_run: bool = False) -> int:
    """Executa dvc repro para re-treinar o modelo. Retorna o exit code."""
    cmd = ["dvc", "repro", "--force"]
    logger.info("Executando: %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY RUN] Re-treino não executado.")
        return 0

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False)
    if result.returncode == 0:
        logger.info("Re-treino concluído com sucesso.")
    else:
        logger.error("Re-treino falhou (exit code %d).", result.returncode)
    return result.returncode


def run(drift_results: dict, dry_run: bool = False) -> None:
    needs_retrain, drifted_features = should_retrain(drift_results)

    if not needs_retrain:
        logger.info("✓ Sem drift — re-treino não necessário.")
        return

    logger.warning(
        "⚠ Drift detectado em: %s — iniciando re-treino automático.",
        ", ".join(drifted_features),
    )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = REPO_ROOT / "monitoring" / "retrain_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": ts,
            "drifted_features": drifted_features,
            "action": "dvc repro" if not dry_run else "dry_run",
        }) + "\n")

    trigger_retrain(dry_run=dry_run)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-summary", required=True, help="Path para drift_summary.json")
    parser.add_argument("--dry-run", action="store_true", help="Simula sem executar dvc repro")
    args = parser.parse_args()

    with open(args.drift_summary) as f:
        drift_results = json.load(f)

    run(drift_results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
