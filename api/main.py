"""FastAPI REST service for melanoma inference."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import os

import boto3
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

load_dotenv()

from api.dependencies import get_predictor
from api.schemas import BatchJobResponse, HealthResponse, PredictionResponse
from skin_app.inference import SkinCancerPredictor

_SQS_QUEUE_NAME = os.environ.get("SQS_QUEUE_NAME", "melanoma-batch-queue")

app = FastAPI(
    title="Melanoma Screening API",
    description="API REST para triagem de melanoma a partir de imagens dermatoscópicas.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health(predictor: SkinCancerPredictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_arch=predictor.config["model_name"],
        device=str(predictor.device),
        checkpoint_hash=predictor.model_checkpoint_hash,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Imagem dermatoscópica (JPEG/PNG)"),
    predictor: SkinCancerPredictor = Depends(get_predictor),
) -> PredictionResponse:
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Formato não suportado. Use JPEG ou PNG.")

    t0 = time.perf_counter()
    file_bytes = await file.read()
    analysis = predictor.analyze_upload(file_bytes=file_bytes, file_name=file.filename or "upload.jpg")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        image_id=analysis.case_id,
        melanoma_prob=analysis.prediction.probability,
        triage_zone=analysis.prediction.zone_key,
        triage_label=analysis.prediction.zone_label,
        headline=analysis.prediction.headline,
        recommended_action=analysis.prediction.recommended_action,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchJobResponse)
async def predict_batch(
    files: list[UploadFile] = File(..., description="Múltiplas imagens para processamento assíncrono"),
) -> BatchJobResponse:
    if not files:
        raise HTTPException(status_code=422, detail="Envie ao menos uma imagem.")

    sqs = boto3.client("sqs")
    try:
        queue_url = sqs.get_queue_url(QueueName=_SQS_QUEUE_NAME)["QueueUrl"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Fila SQS indisponível: {exc}") from exc

    job_id = str(uuid.uuid4())
    sent = 0
    for file in files:
        file_bytes = await file.read()
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=file_bytes.hex(),
            MessageAttributes={
                "job_id": {"StringValue": job_id, "DataType": "String"},
                "file_name": {"StringValue": file.filename or "upload.jpg", "DataType": "String"},
            },
        )
        sent += 1

    return BatchJobResponse(job_id=job_id, queue_url=queue_url, message_count=sent)
