from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    image_id: str
    melanoma_prob: float = Field(ge=0.0, le=1.0)
    triage_zone: str
    triage_label: str
    headline: str
    recommended_action: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_arch: str
    device: str
    checkpoint_hash: str | None
