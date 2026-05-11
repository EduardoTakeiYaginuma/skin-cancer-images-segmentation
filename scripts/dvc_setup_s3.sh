#!/usr/bin/env bash
# Configure DVC remote S3 e rastreie os artefatos grandes.
# Uso: bash scripts/dvc_setup_s3.sh <nome-do-bucket>

set -euo pipefail

BUCKET="${1:?Forneça o nome do bucket S3: bash scripts/dvc_setup_s3.sh meu-bucket}"
REMOTE_NAME="s3remote"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "==> Configurando remote DVC: s3://${BUCKET}/dvc-store (região: ${REGION})"
dvc remote add -f -d "${REMOTE_NAME}" "s3://${BUCKET}/dvc-store"
dvc remote modify "${REMOTE_NAME}" region "${REGION}"

echo "==> Rastreando artefatos de dados..."
dvc add data/images data/masks data/metadata.csv

echo "==> Rastreando modelos..."
dvc add outputs/models/model_comparison outputs/models/unet_segmentation.pt

echo "==> Empurrando para S3..."
dvc push

echo ""
echo "Feito! Adicione os arquivos .dvc ao git:"
echo "  git add data/images.dvc data/masks.dvc data/metadata.csv.dvc"
echo "  git add outputs/models/model_comparison.dvc outputs/models/unet_segmentation.pt.dvc"
echo "  git add .dvc/config"
echo "  git commit -m 'chore: rastrear dados e modelos com DVC + S3'"
