from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Float64, Int64, String

from entities import image
from data_sources import classification_source, preprocessing_source

# TTL longo: dataset estático, sem risco de dados expirarem
_STATIC_TTL = timedelta(days=3650)

lesion_classification_fv = FeatureView(
    name="lesion_classification",
    entities=[image],
    ttl=_STATIC_TTL,
    schema=[
        Field(name="MEL", dtype=Float32),
        Field(name="NV", dtype=Float32),
        Field(name="BCC", dtype=Float32),
        Field(name="AKIEC", dtype=Float32),
        Field(name="BKL", dtype=Float32),
        Field(name="DF", dtype=Float32),
        Field(name="VASC", dtype=Float32),
        Field(name="binary_label", dtype=Int64),
        Field(name="label", dtype=String),
        Field(name="split", dtype=String),
    ],
    source=classification_source,
    description="Labels dermatológicos (one-hot 7 classes + binário melanoma) por imagem",
)

preprocessing_stats_fv = FeatureView(
    name="preprocessing_stats",
    entities=[image],
    ttl=_STATIC_TTL,
    schema=[
        Field(name="final_height", dtype=Int64),
        Field(name="final_width", dtype=Int64),
        Field(name="mask_coverage_after_crop", dtype=Float64),
        Field(name="hair_pixels_detected", dtype=Int64),
    ],
    source=preprocessing_source,
    description="Estatísticas do pipeline de preprocessamento (máscara, cabelo, dimensões finais)",
)
