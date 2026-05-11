"""AWS Lambda handler for melanoma inference via API Gateway."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from skin_app.inference import SkinCancerPredictor

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "inference_config.json"
_predictor: SkinCancerPredictor | None = None


def _get_predictor() -> SkinCancerPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SkinCancerPredictor(config_path=_CONFIG_PATH)
    return _predictor


def lambda_handler(event: dict, context: object) -> dict:
    t0 = time.perf_counter()

    # API Gateway sends body as base64 when isBase64Encoded=True
    body = event.get("body", "")
    is_b64 = event.get("isBase64Encoded", False)
    if is_b64:
        file_bytes = base64.b64decode(body)
    else:
        file_bytes = body.encode("utf-8") if isinstance(body, str) else body

    file_name = "upload.jpg"
    if event.get("queryStringParameters"):
        file_name = event["queryStringParameters"].get("filename", file_name)

    try:
        predictor = _get_predictor()
        analysis = predictor.analyze_upload(file_bytes=file_bytes, file_name=file_name)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "image_id": analysis.case_id,
                "melanoma_prob": analysis.prediction.probability,
                "triage_zone": analysis.prediction.zone_key,
                "triage_label": analysis.prediction.zone_label,
                "headline": analysis.prediction.headline,
                "recommended_action": analysis.prediction.recommended_action,
                "latency_ms": latency_ms,
            }),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }
