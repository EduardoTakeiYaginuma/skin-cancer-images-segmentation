"""Entry point for drift monitoring. Run: python monitoring/run_monitoring.py [--log <path>]"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from monitoring.drift_detector import detect_drift, load_current_data, load_reference_data

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FEATURE_COLS = ["mask_coverage_after_crop", "hair_pixels_detected", "final_height", "final_width"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Gera relatório de drift com Evidently.")
    parser.add_argument("--log", type=str, default=None, help="CSV com dados de inferência recentes")
    args = parser.parse_args()

    reference = load_reference_data()

    if args.log:
        current = load_current_data(args.log)
    else:
        logger.warning("Nenhum log de inferência fornecido — usando amostra aleatória do val set como proxy.")
        current = reference.sample(frac=0.2, random_state=42)

    available_cols = [c for c in FEATURE_COLS if c in reference.columns and c in current.columns]
    if not available_cols:
        logger.error("Nenhuma feature de overlap entre reference e current. Abortando.")
        return

    report = detect_drift(reference, current, feature_cols=available_cols)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(str(report_path))
    logger.info("Relatório salvo em %s", report_path)


if __name__ == "__main__":
    main()
