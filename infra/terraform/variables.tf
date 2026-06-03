variable "ecr_image_uri" {
  type        = string
  description = "Full ECR image URI (e.g. 354413487200.dkr.ecr.us-east-2.amazonaws.com/melanoma-lambda:latest)"
}

variable "lambda_role_arn" {
  type        = string
  description = "ARN of the IAM role with Lambda execution and S3 ReadOnly permissions"
}

variable "model_s3_bucket" {
  type        = string
  default     = "skin-cancer-segmentation"
  description = "S3 bucket where the model checkpoints are stored"
}

variable "memory_size" {
  type        = number
  default     = 3008
  description = "Memory allocated to the Lambda function (MB)"
}

variable "timeout" {
  type        = number
  default     = 120
  description = "Lambda function timeout (seconds)"
}

variable "aws_region" {
  type    = string
  default = "us-east-2"
}
