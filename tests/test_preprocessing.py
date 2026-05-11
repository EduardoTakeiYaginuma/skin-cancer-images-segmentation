"""Unit tests for image preprocessing utilities in skin_app.inference."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from skin_app.inference import (
    compute_bbox_from_mask,
    expand_bbox,
    normalize_minmax,
    pad_to_square,
    pil_to_rgb_array,
    rgb_array_to_pil,
)


class TestNormalizeMinmax:
    def test_normal_range(self) -> None:
        arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        result = normalize_minmax(arr)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_constant_array_returns_zeros(self) -> None:
        arr = np.full((4, 4), 5.0, dtype=np.float32)
        result = normalize_minmax(arr)
        assert (result == 0.0).all()

    def test_output_dtype_is_float32(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = normalize_minmax(arr)
        assert result.dtype == np.float32


class TestComputeBboxFromMask:
    def test_single_pixel(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 7] = 1
        bbox = compute_bbox_from_mask(mask)
        assert bbox == (7, 5, 8, 6)

    def test_empty_mask_returns_none(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert compute_bbox_from_mask(mask) is None

    def test_full_mask(self) -> None:
        mask = np.ones((8, 6), dtype=np.uint8)
        bbox = compute_bbox_from_mask(mask)
        assert bbox == (0, 0, 6, 8)


class TestExpandBbox:
    def test_expansion_stays_in_bounds(self) -> None:
        bbox = (5, 5, 15, 15)
        result = expand_bbox(bbox, width=20, height=20, margin_ratio=0.5)
        x0, y0, x1, y1 = result
        assert x0 >= 0 and y0 >= 0
        assert x1 <= 20 and y1 <= 20

    def test_expansion_is_larger_than_original(self) -> None:
        bbox = (5, 5, 15, 15)
        result = expand_bbox(bbox, width=100, height=100, margin_ratio=0.2)
        assert result[0] < 5
        assert result[1] < 5
        assert result[2] > 15
        assert result[3] > 15


class TestPilConversions:
    def test_roundtrip(self) -> None:
        image = Image.new("RGB", (8, 8), color=(128, 64, 32))
        arr = pil_to_rgb_array(image)
        restored = rgb_array_to_pil(arr)
        assert restored.size == image.size
        assert np.array_equal(np.array(restored), np.array(image))

    def test_output_shape(self) -> None:
        image = Image.new("RGB", (12, 8))
        arr = pil_to_rgb_array(image)
        assert arr.shape == (8, 12, 3)


class TestPadToSquare:
    def test_portrait_becomes_square(self) -> None:
        arr = np.zeros((10, 6, 3), dtype=np.uint8)
        result = pad_to_square(arr)
        assert result.shape[0] == result.shape[1] == 10

    def test_already_square_unchanged(self) -> None:
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        result = pad_to_square(arr)
        assert result.shape == (8, 8, 3)
