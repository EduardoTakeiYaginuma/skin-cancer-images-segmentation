"""
Prepara as fontes de dados (Parquet) para o Feast a partir dos CSVs do projeto.

Execute uma vez antes de rodar `feast apply` ou qualquer script de features:
    python feature_store/scripts/prepare_sources.py

Requer que os splits de dados já existam em data/metadata/.
Se treated_manifest.csv não existir, rode o notebook 03_preprocessing.ipynb primeiro.
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"

# Data de referência: HAM10000 foi publicado em 2018-2020
# Usamos timestamp fixo pois o dataset é estático (sem evolução temporal real)
REFERENCE_TIMESTAMP = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")


def prepare_classification_source() -> None:
    """Combina os splits train/val/test num único Parquet de features de classificação."""
    split_files = {
        "train": DATA_DIR / "metadata" / "train_split.csv",
        "val": DATA_DIR / "metadata" / "val_split.csv",
        "test": DATA_DIR / "metadata" / "test_split.csv",
    }

    missing = [str(p) for p in split_files.values() if not p.exists()]
    if missing:
        print("Erro: arquivos de split não encontrados:")
        for m in missing:
            print(f"  {m}")
        print("Execute setup_data.py ou regenere os splits pelo notebook 01.")
        sys.exit(1)

    dfs = []
    for split_name, path in split_files.items():
        df = pd.read_csv(path)
        df["split"] = split_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.rename(columns={"image": "image_id"}, inplace=True)
    combined["event_timestamp"] = REFERENCE_TIMESTAMP

    feature_cols = [
        "image_id", "event_timestamp",
        "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC",
        "label", "binary_label", "split",
    ]
    combined = combined[feature_cols]

    float_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    combined[float_cols] = combined[float_cols].astype("float32")
    combined["binary_label"] = combined["binary_label"].astype("int64")

    # Força string (pa.string / utf8) em vez de large_string para compatibilidade com Feast 0.42
    schema = pa.schema([
        pa.field("image_id", pa.string()),
        pa.field("event_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("MEL", pa.float32()),
        pa.field("NV", pa.float32()),
        pa.field("BCC", pa.float32()),
        pa.field("AKIEC", pa.float32()),
        pa.field("BKL", pa.float32()),
        pa.field("DF", pa.float32()),
        pa.field("VASC", pa.float32()),
        pa.field("label", pa.string()),
        pa.field("binary_label", pa.int64()),
        pa.field("split", pa.string()),
    ])

    out_path = SOURCES_DIR / "lesion_classification.parquet"
    table = pa.Table.from_pandas(combined, schema=schema, preserve_index=False)
    pq.write_table(table, out_path)
    print(f"[OK] lesion_classification.parquet — {len(combined)} registros → {out_path}")


def prepare_preprocessing_source() -> None:
    """Exporta as estatísticas de preprocessamento do treated_manifest para Parquet."""
    manifest_path = DATA_DIR / "processed" / "treated_manifest.csv"

    if not manifest_path.exists():
        print(f"Erro: {manifest_path} não encontrado.")
        print("Execute o notebook 03_preprocessing.ipynb para gerar o manifesto.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    df["event_timestamp"] = REFERENCE_TIMESTAMP

    feature_cols = [
        "image_id", "event_timestamp",
        "final_height", "final_width",
        "mask_coverage_after_crop", "hair_pixels_detected",
    ]
    df = df[feature_cols]

    df["final_height"] = df["final_height"].astype("int64")
    df["final_width"] = df["final_width"].astype("int64")
    df["hair_pixels_detected"] = df["hair_pixels_detected"].astype("int64")
    df["mask_coverage_after_crop"] = df["mask_coverage_after_crop"].astype("float64")

    # Força string (pa.string / utf8) para compatibilidade com Feast 0.42
    schema = pa.schema([
        pa.field("image_id", pa.string()),
        pa.field("event_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("final_height", pa.int64()),
        pa.field("final_width", pa.int64()),
        pa.field("mask_coverage_after_crop", pa.float64()),
        pa.field("hair_pixels_detected", pa.int64()),
    ])

    out_path = SOURCES_DIR / "preprocessing_stats.parquet"
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, out_path)
    print(f"[OK] preprocessing_stats.parquet — {len(df)} registros → {out_path}")


if __name__ == "__main__":
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Preparando fontes em: {SOURCES_DIR}\n")
    prepare_classification_source()
    prepare_preprocessing_source()
    print("\nFontes prontas. Próximo passo: python feature_store/scripts/apply_registry.py")
