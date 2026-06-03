"""Tests for FastAPI endpoints using TestClient (without loading real models)."""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.dependencies import get_predictor
from skin_app.inference import CaseAnalysis, DatasetMatch, PredictionResult


def _make_fake_analysis() -> CaseAnalysis:
    prediction = PredictionResult(
        probability=0.12,
        zone_key="review",
        zone_label="Possivel melanoma",
        headline="Caso intermediario",
        recommended_action="Encaminhar para revisão.",
        threshold_low=0.004,
        threshold_high=0.22,
        device="cpu",
        predicted_binary=0,
        preprocessing_source="fallback_square",
    )
    img = Image.new("RGB", (224, 224))
    return CaseAnalysis(
        case_id="abc123",
        file_name="test.jpg",
        display_name="test",
        prediction=prediction,
        dataset_match=DatasetMatch(found=False),
        original_image=img,
        uploaded_image=img,
        classifier_source_image=img,
        classifier_input_image=img,
        segmentation_overlay=None,
        segmentation_mask_image=None,
        lesion_bbox_image=None,
        classifier_segmentation_overlay=None,
        combined_explainability_overlay=None,
        ground_truth_overlay=None,
        gradcam_overlay=None,
        occlusion_overlay=None,
        gradcam_hotspots=[],
        occlusion_hotspots=[],
        similar_cases=[],
        lesion_heuristics=[],
        segmentation_coverage=None,
        classifier_segmentation_coverage=None,
        segmentation_stats=None,
        classifier_segmentation_stats=None,
        image_quality={},
        pipeline_timings={"total": 0.5},
        model_info={"build_id": None, "checkpoint_path": "", "checkpoint_hash": None,
                    "seg_checkpoint_path": None, "seg_checkpoint_hash": None},
        dataset_stats={},
        preprocessing_note="",
        truth_note=None,
    )


@pytest.fixture()
def client() -> TestClient:
    mock_predictor = MagicMock()
    mock_predictor.config = {"model_name": "resnet50"}
    mock_predictor.device = "cpu"
    mock_predictor.model_checkpoint_hash = "abc123"
    mock_predictor.analyze_upload.return_value = _make_fake_analysis()

    app.dependency_overrides[get_predictor] = lambda: mock_predictor
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(200, 150, 100)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_arch" in data


def test_predict_returns_json(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "melanoma_prob" in data
    assert "triage_zone" in data
    assert 0.0 <= data["melanoma_prob"] <= 1.0


def test_predict_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("data.csv", b"col1,col2\n1,2", "text/csv")},
    )
    assert response.status_code == 415
