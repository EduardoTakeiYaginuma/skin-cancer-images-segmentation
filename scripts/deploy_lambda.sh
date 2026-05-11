#!/usr/bin/env bash
# Deploy da Lambda via Docker image no ECR.
# Uso: bash scripts/deploy_lambda.sh <account-id> <region> [<lambda-role-arn>]

set -euo pipefail

ACCOUNT_ID="${1:?Forneça o AWS Account ID}"
REGION="${2:-us-east-1}"
ROLE_ARN="${3:-}"
REPO_NAME="melanoma-lambda"
FUNCTION_NAME="melanoma-predictor"
IMAGE_TAG="latest"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "==> Autenticando no ECR..."
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Criando repositório ECR (se não existir)..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

echo "==> Build da imagem Lambda..."
docker build -f Dockerfile.lambda -t "${REPO_NAME}:${IMAGE_TAG}" .

echo "==> Tag e push para ECR..."
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}"
docker push "${ECR_URI}"

# Criar ou atualizar Lambda
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" 2>/dev/null; then
  echo "==> Atualizando Lambda existente..."
  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --image-uri "${ECR_URI}" \
    --region "${REGION}"
else
  if [ -z "${ROLE_ARN}" ]; then
    echo "ERRO: Lambda não existe. Forneça o Role ARN como 3º argumento."
    exit 1
  fi
  echo "==> Criando Lambda..."
  aws lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --package-type Image \
    --code "ImageUri=${ECR_URI}" \
    --role "${ROLE_ARN}" \
    --memory-size 3008 \
    --timeout 30 \
    --region "${REGION}"
fi

echo ""
echo "Deploy concluído: ${ECR_URI}"
echo "Para expor via API Gateway, execute:"
echo "  bash scripts/setup_api_gateway.sh ${ACCOUNT_ID} ${REGION}"
