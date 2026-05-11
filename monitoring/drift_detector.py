"""Data drift detection using Evidently."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataQualityPreset
from evidently import Report


_FEATURE_STORE_PARQUET = (
    Path(__file__).resolve().parent.parent
    / "feature_store" / "data" / "sources" / "lesion_classification.parquet"
)
_PREPROCESSING_PARQUET = (
    Path(__file__).resolve().parent.parent
    / "feature_store" / "data" / "sources" / "preprocessing_stats.parquet"
)


def load_reference_data() -> pd.DataFrame:
    df = pd.read_parquet(_FEATURE_STORE_PARQUET)
    prep = pd.read_parquet(_PREPROCESSING_PARQUET)
    merged = df.merge(prep, on="image_id", how="left")
    return merged[merged["split"] == "train"]


def load_current_data(inference_log_path: str | Path) -> pd.DataFrame:
    """Load inference log CSV produced by the API/Lambda for drift comparison."""
    return pd.read_csv(inference_log_path)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
) -> Report:
    definition = DataDefinition(numerical_columns=feature_cols)
    ref_ds = Dataset.from_pandas(reference[feature_cols], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[feature_cols], data_definition=definition)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_ds, current_data=cur_ds)
    return report
