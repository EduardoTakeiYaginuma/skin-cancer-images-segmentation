#!/usr/bin/env bash
# Deploy the Lambda function via a Docker image hosted on ECR.
# Usage: bash scripts/deploy_lambda.sh <account-id> <region> [<lambda-role-arn>]

set -euo pipefail

ACCOUNT_ID="${1:?Provide the AWS account ID}"
REGION="${2:-us-east-1}"
ROLE_ARN="${3:-}"
REPO_NAME="melanoma-lambda"
FUNCTION_NAME="melanoma-predictor"
IMAGE_TAG="latest"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "==> Authenticating to ECR..."
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Creating the ECR repository if needed..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

echo "==> Building the Lambda image..."
docker build -f Dockerfile.lambda -t "${REPO_NAME}:${IMAGE_TAG}" .

echo "==> Tagging and pushing to ECR..."
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}"
docker push "${ECR_URI}"

# Create or update the Lambda function
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" 2>/dev/null; then
  echo "==> Updating the existing Lambda function..."
  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --image-uri "${ECR_URI}" \
    --region "${REGION}"
else
  if [ -z "${ROLE_ARN}" ]; then
    echo "ERROR: Lambda does not exist. Provide the IAM role ARN as the third argument."
    exit 1
  fi
  echo "==> Creating the Lambda function..."
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
echo "Deploy finished: ${ECR_URI}"
echo "To provision the API Gateway in front of this Lambda, apply the IaC under infra/"
echo "(CloudFormation: infra/cloudformation.yaml, or Terraform: infra/terraform/)."
