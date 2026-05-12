"""AWS Lambda handler for melanoma inference via API Gateway."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

import boto3

from skin_app.inference import SkinCancerPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("MODEL_S3_BUCKET", "skin-cancer-segmentation")
_S3_CLASSIFIER_KEY = os.environ.get("MODEL_S3_KEY", "models/resnet50_aug_224x224.pt")
_S3_SEG_KEY = os.environ.get("SEG_MODEL_S3_KEY", "models/unet_segmentation.pt")
_TMP_DIR = Path("/tmp/melanoma_models")

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "inference_config.json"
_predictor: SkinCancerPredictor | None = None


def _ensure_models() -> None:
    """Download model checkpoints from S3 to /tmp on cold start."""
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    for s3_key, local_name in [
        (_S3_CLASSIFIER_KEY, "resnet50_aug_224x224.pt"),
        (_S3_SEG_KEY, "unet_segmentation.pt"),
    ]:
        dest = _TMP_DIR / local_name
        if not dest.exists():
            logger.info("Downloading s3://%s/%s → %s", _S3_BUCKET, s3_key, dest)
            s3.download_file(_S3_BUCKET, s3_key, str(dest))
            logger.info("Downloaded %s (%.1f MB)", local_name, dest.stat().st_size / 1e6)


def _make_tmp_config() -> Path:
    """Write a config JSON to /tmp with checkpoint paths pointing to /tmp."""
    import json as _json

    with open(_CONFIG_PATH) as f:
        cfg = _json.load(f)

    cfg["checkpoint_path"] = str(_TMP_DIR / "resnet50_aug_224x224.pt")
    cfg.setdefault("segmentation", {})["checkpoint_path"] = str(_TMP_DIR / "unet_segmentation.pt")

    tmp_cfg = _TMP_DIR / "inference_config.json"
    with open(tmp_cfg, "w") as f:
        _json.dump(cfg, f)
    return tmp_cfg


def _get_predictor() -> SkinCancerPredictor:
    global _predictor
    if _predictor is None:
        _ensure_models()
        tmp_cfg = _make_tmp_config()
        _predictor = SkinCancerPredictor(config_path=tmp_cfg)
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
        logger.exception("Inference error")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }
