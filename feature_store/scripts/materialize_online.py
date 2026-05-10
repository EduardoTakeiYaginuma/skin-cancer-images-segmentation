"""
Materializa features para o online store (SQLite local).

Necessário para servir features em tempo real via store.get_online_features().
Execute após feast apply e ao atualizar os dados de fonte.

Uso:
    python feature_store/scripts/materialize_online.py
"""

from datetime import datetime, timezone
from pathlib import Path

from feast import FeatureStore

REPO_PATH = Path(__file__).resolve().parent.parent / "feature_repo"

# Janela de materialização: cobre o período do dataset HAM10000
START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)


def main() -> None:
    store = FeatureStore(repo_path=str(REPO_PATH))
    print(f"Materializando features de {START_DATE.date()} a {END_DATE.date()}...")

    store.materialize(start_date=START_DATE, end_date=END_DATE)

    print("\nMaterialização concluída.")
    print("Você pode agora servir features online via store.get_online_features().")

    # Exemplo de leitura online
    example_ids = [
        {"image_id": "ISIC_0024306"},
        {"image_id": "ISIC_0024307"},
    ]
    online_response = store.get_online_features(
        features=["preprocessing_stats:mask_coverage_after_crop", "preprocessing_stats:hair_pixels_detected"],
        entity_rows=example_ids,
    ).to_dict()

    print("\nExemplo de resposta online:")
    for key, values in online_response.items():
        print(f"  {key}: {values}")


if __name__ == "__main__":
    main()
