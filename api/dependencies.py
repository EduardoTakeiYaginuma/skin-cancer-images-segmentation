from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from skin_app.inference import SkinCancerPredictor

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "inference_config.json"


@lru_cache(maxsize=1)
def get_predictor() -> SkinCancerPredictor:
    return SkinCancerPredictor(config_path=_CONFIG_PATH)
