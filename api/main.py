"""FastAPI REST service for melanoma inference."""
from __future__ import annotations

import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

load_dotenv()

from api.dependencies import get_predictor
from api.schemas import HealthResponse, PredictionResponse
from skin_app.inference import SkinCancerPredictor

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
