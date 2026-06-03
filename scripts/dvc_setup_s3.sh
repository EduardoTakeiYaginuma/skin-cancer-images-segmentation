#!/usr/bin/env bash
# Configure the DVC S3 remote and track the heavy artifacts.
# Usage: bash scripts/dvc_setup_s3.sh <bucket-name>

set -euo pipefail

BUCKET="${1:?Provide the S3 bucket name: bash scripts/dvc_setup_s3.sh my-bucket}"
REMOTE_NAME="s3remote"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "==> Configuring DVC remote: s3://${BUCKET}/dvc-store (region: ${REGION})"
dvc remote add -f -d "${REMOTE_NAME}" "s3://${BUCKET}/dvc-store"
dvc remote modify "${REMOTE_NAME}" region "${REGION}"

echo "==> Tracking data artifacts..."
dvc add data/images data/masks data/metadata.csv

echo "==> Tracking model artifacts..."
dvc add outputs/models/model_comparison outputs/models/unet_segmentation.pt

echo "==> Pushing to S3..."
dvc push

echo ""
echo "Done. Add the generated .dvc files to git:"
echo "  git add data/images.dvc data/masks.dvc data/metadata.csv.dvc"
echo "  git add outputs/models/model_comparison.dvc outputs/models/unet_segmentation.pt.dvc"
echo "  git add .dvc/config"
echo "  git commit -m 'chore: track data and models with DVC + S3'"
