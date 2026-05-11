"""Validates that Feast Parquet sources have the expected schema."""
from __future__ import annotations

import pytest
from pathlib import Path

PARQUET_DIR = Path(__file__).resolve().parent.parent / "feature_store" / "data" / "sources"


@pytest.mark.skipif(
    not (PARQUET_DIR / "lesion_classification.parquet").exists(),
    reason="Parquets não gerados — execute prepare_sources.py primeiro.",
)
def test_lesion_classification_schema() -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET_DIR / "lesion_classification.parquet")
    required = {"image_id", "event_timestamp", "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "binary_label", "label", "split"}
    assert required.issubset(set(table.schema.names)), f"Colunas faltando: {required - set(table.schema.names)}"


@pytest.mark.skipif(
    not (PARQUET_DIR / "preprocessing_stats.parquet").exists(),
    reason="Parquets não gerados — execute prepare_sources.py primeiro.",
)
def test_preprocessing_stats_schema() -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET_DIR / "preprocessing_stats.parquet")
    required = {"image_id", "event_timestamp", "mask_coverage_after_crop", "hair_pixels_detected", "final_height", "final_width"}
    assert required.issubset(set(table.schema.names)), f"Colunas faltando: {required - set(table.schema.names)}"
