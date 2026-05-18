variable "ecr_image_uri" {
  type        = string
  description = "URI completa da imagem ECR (ex: 354413487200.dkr.ecr.us-east-2.amazonaws.com/melanoma-lambda:latest)"
}

variable "lambda_role_arn" {
  type        = string
  description = "ARN da role IAM com permissões Lambda + S3 ReadOnly"
}

variable "model_s3_bucket" {
  type        = string
  default     = "skin-cancer-segmentation"
  description = "Bucket S3 onde os checkpoints do modelo estão armazenados"
}

variable "memory_size" {
  type        = number
  default     = 3008
  description = "Memória alocada para a Lambda (MB)"
}

variable "timeout" {
  type        = number
  default     = 120
  description = "Timeout da Lambda (segundos)"
}

variable "aws_region" {
  type    = string
  default = "us-east-2"
}
