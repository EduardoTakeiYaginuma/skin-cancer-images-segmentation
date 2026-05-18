output "api_endpoint" {
  description = "URL do endpoint de predição"
  value       = "${aws_apigatewayv2_stage.prod.invoke_url}/predict"
}

output "lambda_arn" {
  description = "ARN da função Lambda"
  value       = aws_lambda_function.melanoma_predictor.arn
}
