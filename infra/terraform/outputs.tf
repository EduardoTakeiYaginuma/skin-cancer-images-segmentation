output "api_endpoint" {
  description = "URL of the prediction endpoint"
  value       = "${aws_apigatewayv2_stage.prod.invoke_url}/predict"
}

output "lambda_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.melanoma_predictor.arn
}
