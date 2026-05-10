from feast import FeatureService
from feature_views import lesion_classification_fv, preprocessing_stats_fv

# Feature service para treino: inclui targets e features tabulares
training_service = FeatureService(
    name="melanoma_training_features",
    features=[
        lesion_classification_fv[
            ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "binary_label", "label", "split"]
        ],
        preprocessing_stats_fv[
            ["mask_coverage_after_crop", "hair_pixels_detected", "final_height", "final_width"]
        ],
    ],
    description=(
        "Conjunto completo de features para treinamento do classificador de melanoma. "
        "Inclui labels multi-classe, target binário, split de dados e estatísticas de preprocessamento."
    ),
)

# Feature service para serving: apenas features disponíveis em tempo de inferência
serving_service = FeatureService(
    name="melanoma_serving_features",
    features=[
        preprocessing_stats_fv[["mask_coverage_after_crop", "hair_pixels_detected"]],
    ],
    description=(
        "Features para inferência online. Apenas métricas computáveis no momento da predição, "
        "sem vazamento de labels de treino."
    ),
)
