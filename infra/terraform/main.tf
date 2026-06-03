terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Lambda Function
resource "aws_lambda_function" "melanoma_predictor" {
  function_name = "melanoma-predictor"
  package_type  = "Image"
  image_uri     = var.ecr_image_uri
  role          = var.lambda_role_arn

  memory_size = var.memory_size
  timeout     = var.timeout

  architectures = ["x86_64"]

  ephemeral_storage {
    size = 1024
  }

  environment {
    variables = {
      MODEL_S3_BUCKET  = var.model_s3_bucket
      MODEL_S3_KEY     = "models/resnet50_aug_224x224.pt"
      SEG_MODEL_S3_KEY = "models/unet_segmentation.pt"
    }
  }
}

# HTTP API Gateway (v2)
resource "aws_apigatewayv2_api" "melanoma_api" {
  name          = "melanoma-api"
  protocol_type = "HTTP"
  description   = "Public API for melanoma triage"
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.melanoma_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.melanoma_predictor.arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predict_route" {
  api_id    = aws_apigatewayv2_api.melanoma_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.melanoma_api.id
  name        = "prod"
  auto_deploy = true
}

# Permission: API Gateway -> Lambda
resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.melanoma_predictor.arn
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.melanoma_api.execution_arn}/*/*/predict"
}
