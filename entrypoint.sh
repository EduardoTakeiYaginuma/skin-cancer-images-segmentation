#!/bin/bash
set -e

echo "[deploy] Sincronizando modelos do S3..."
aws s3 cp "s3://${S3_BUCKET}/models/unet_segmentation.pt" \
    outputs/models/unet_segmentation.pt \
    --no-progress
aws s3 sync "s3://${S3_BUCKET}/models/model_comparison" \
    outputs/models/model_comparison \
    --no-progress

echo "[deploy] Baixando metadata..."
aws s3 cp "s3://${S3_BUCKET}/data/metadata.csv" \
    data/metadata.csv \
    --no-progress

echo "[deploy] Iniciando Streamlit..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
