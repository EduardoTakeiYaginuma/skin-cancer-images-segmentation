from pathlib import Path
from feast import FileSource

_SOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"

classification_source = FileSource(
    name="lesion_classification_source",
    path=str(_SOURCES_DIR / "lesion_classification.parquet"),
    timestamp_field="event_timestamp",
    description="One-hot class labels e target binário de melanoma por imagem",
)

preprocessing_source = FileSource(
    name="preprocessing_stats_source",
    path=str(_SOURCES_DIR / "preprocessing_stats.parquet"),
    timestamp_field="event_timestamp",
    description="Estatísticas do pipeline de preprocessamento por imagem",
)
