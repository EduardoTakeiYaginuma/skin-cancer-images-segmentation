"""
Exemplo de recuperação de features históricas para treino.

Uso:
    python feature_store/scripts/get_historical_features.py

O script demonstra como usar o FeatureStore para obter um DataFrame de treino
que combina labels e estatísticas de preprocessamento, pronto para usar em
notebooks ou pipelines de treinamento.
"""

from pathlib import Path

import pandas as pd
from feast import FeatureStore

REPO_PATH = Path(__file__).resolve().parent.parent / "feature_repo"


def load_entity_df(split: str | None = None) -> pd.DataFrame:
    """
    Carrega os image_ids do projeto e monta o entity DataFrame para o Feast.

    Args:
        split: Se fornecido ("train", "val", "test"), filtra apenas esse split.
               None retorna todas as imagens.
    """
    sources_dir = Path(__file__).resolve().parent.parent / "data" / "sources"
    classification_path = sources_dir / "lesion_classification.parquet"

    if not classification_path.exists():
        raise FileNotFoundError(
            f"{classification_path} não encontrado.\n"
            "Execute: python feature_store/scripts/prepare_sources.py"
        )

    df = pd.read_parquet(classification_path, columns=["image_id", "event_timestamp", "split"])

    if split is not None:
        df = df[df["split"] == split].copy()

    return df[["image_id", "event_timestamp"]].reset_index(drop=True)


def get_training_dataframe(split: str = "train") -> pd.DataFrame:
    """Retorna DataFrame com todas as features de treino para o split solicitado."""
    store = FeatureStore(repo_path=str(REPO_PATH))
    entity_df = load_entity_df(split=split)

    print(f"Buscando features para {len(entity_df)} imagens (split={split})...")

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service("melanoma_training_features"),
    ).to_df()

    return training_df


def main() -> None:
    store = FeatureStore(repo_path=str(REPO_PATH))
    print(f"Feature store: {store.project}\n")

    # Recupera features do split de treino
    df = get_training_dataframe(split="train")

    print(f"Shape: {df.shape}")
    print(f"Colunas: {list(df.columns)}\n")
    print("Primeiras linhas:")
    print(df.head())

    melanoma_ratio = df["binary_label"].mean()
    print(f"\nProporção de melanoma no treino: {melanoma_ratio:.1%}")

    # Exemplo com features individuais (sem feature service)
    print("\n--- Exemplo: recuperar apenas preprocessing_stats ---")
    entity_df = load_entity_df(split="val")
    preprocessing_df = store.get_historical_features(
        entity_df=entity_df,
        features=["preprocessing_stats:mask_coverage_after_crop", "preprocessing_stats:hair_pixels_detected"],
    ).to_df()

    print(f"Cobertura média da máscara (val): {preprocessing_df['mask_coverage_after_crop'].mean():.3f}")
    print(f"Pixels de cabelo detectados (mediana): {preprocessing_df['hair_pixels_detected'].median():.0f}")


if __name__ == "__main__":
    main()
