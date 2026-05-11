"""SQS consumer: processa mensagens de predição em batch."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import os

import boto3

from skin_app.inference import SkinCancerPredictor

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "inference_config.json"
_QUEUE_NAME = os.environ.get("SQS_QUEUE_NAME", "melanoma-batch-queue")
_RESULTS_BUCKET = os.environ.get("S3_RESULTS_BUCKET", "melanoma-batch-results")

logger = logging.getLogger(__name__)


def _get_predictor() -> SkinCancerPredictor:
    return SkinCancerPredictor(config_path=_CONFIG_PATH)


def _save_result_to_s3(s3: object, job_id: str, file_name: str, result: dict) -> None:
    key = f"results/{job_id}/{file_name}.json"
    s3.put_object(  # type: ignore[union-attr]
        Bucket=_RESULTS_BUCKET,
        Key=key,
        Body=json.dumps(result),
        ContentType="application/json",
    )
    logger.info("Resultado salvo em s3://%s/%s", _RESULTS_BUCKET, key)


def run(max_messages: int = 10, wait_seconds: int = 20, idle_sleep: float = 2.0) -> None:
    sqs = boto3.client("sqs")
    s3 = boto3.client("s3")
    queue_url = sqs.get_queue_url(QueueName=_QUEUE_NAME)["QueueUrl"]

    predictor = _get_predictor()
    logger.info("Consumer iniciado. Fila: %s", queue_url)

    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
            MessageAttributeNames=["All"],
        )
        messages = response.get("Messages", [])

        if not messages:
            time.sleep(idle_sleep)
            continue

        for msg in messages:
            attrs = msg.get("MessageAttributes", {})
            job_id = attrs.get("job_id", {}).get("StringValue", "unknown")
            file_name = attrs.get("file_name", {}).get("StringValue", "upload.jpg")

            try:
                file_bytes = bytes.fromhex(msg["Body"])
                analysis = predictor.analyze_upload(file_bytes=file_bytes, file_name=file_name)
                result = {
                    "job_id": job_id,
                    "file_name": file_name,
                    "image_id": analysis.case_id,
                    "melanoma_prob": analysis.prediction.probability,
                    "triage_zone": analysis.prediction.zone_key,
                    "triage_label": analysis.prediction.zone_label,
                }
                _save_result_to_s3(s3, job_id, file_name, result)
                logger.info("Processado %s — prob=%.4f zona=%s", file_name, result["melanoma_prob"], result["triage_zone"])
            except Exception:
                logger.exception("Falha ao processar mensagem %s", msg["MessageId"])
                continue

            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
