"""Unit tests for BibRecognizer (PaddleOCR 3.x / PP-OCRv5)."""
from __future__ import annotations

import os

# Must set env vars before any paddle import
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

import cv2
import numpy as np
import pytest

from src.ml.bibs.recognizer import BibRecognizer


@pytest.fixture(scope="module")
def recognizer() -> BibRecognizer:
    """Module-scoped to avoid reloading PaddleOCR models for each test."""
    return BibRecognizer(use_gpu=False)


def _make_number_image(text: str, size: tuple[int, int] = (400, 600)) -> np.ndarray:
    """Create a white image with black text drawn on it."""
    h, w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    font_scale = min(h, w) / 80
    thickness = max(2, int(font_scale * 2))
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return img


def _make_blank_image(size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Create a uniform gray image with no text."""
    return np.full((size[0], size[1], 3), 128, dtype=np.uint8)


class TestBibRecognizerInit:
    def test_ocr_engine_loaded(self, recognizer: BibRecognizer):
        assert recognizer.ocr is not None

    def test_recognizer_type(self, recognizer: BibRecognizer):
        assert isinstance(recognizer, BibRecognizer)


class TestBibRecognizerRecognize:
    def test_returns_required_keys(self, recognizer: BibRecognizer):
        img = _make_number_image("42")
        result = recognizer.recognize(img)
        assert "bib_number" in result
        assert "confidence" in result
        assert "all_candidates" in result

    def test_recognizes_two_digit_number(self, recognizer: BibRecognizer):
        img = _make_number_image("42")
        result = recognizer.recognize(img)
        assert result["bib_number"] == "42"
        assert result["confidence"] > 0.8

    def test_recognizes_four_digit_number(self, recognizer: BibRecognizer):
        img = _make_number_image("1234")
        result = recognizer.recognize(img)
        assert result["bib_number"] == "1234"
        assert result["confidence"] > 0.8

    def test_recognizes_large_number(self, recognizer: BibRecognizer):
        img = _make_number_image("98765")
        result = recognizer.recognize(img)
        assert result["bib_number"] == "98765"
        assert result["confidence"] > 0.5

    def test_no_text_returns_empty(self, recognizer: BibRecognizer):
        img = _make_blank_image()
        result = recognizer.recognize(img)
        assert result["bib_number"] == ""
        assert result["confidence"] == 0.0
        assert result["all_candidates"] == []

    def test_confidence_range(self, recognizer: BibRecognizer):
        img = _make_number_image("42")
        result = recognizer.recognize(img)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_candidates_are_sorted_by_confidence(self, recognizer: BibRecognizer):
        img = _make_number_image("42")
        result = recognizer.recognize(img)
        candidates = result["all_candidates"]
        if len(candidates) > 1:
            confs = [c["confidence"] for c in candidates]
            assert confs == sorted(confs, reverse=True)

    def test_candidate_has_required_keys(self, recognizer: BibRecognizer):
        img = _make_number_image("42")
        result = recognizer.recognize(img)
        if result["all_candidates"]:
            candidate = result["all_candidates"][0]
            assert "text" in candidate
            assert "confidence" in candidate

    def test_strips_non_numeric_characters(self, recognizer: BibRecognizer):
        """Even if OCR reads mixed text, only digits should remain."""
        img = _make_number_image("42")
        result = recognizer.recognize(img)
        assert result["bib_number"].isdigit() or result["bib_number"] == ""

    def test_different_image_sizes(self, recognizer: BibRecognizer):
        """Recognizer should handle various image dimensions."""
        for size in [(200, 300), (400, 600), (100, 500)]:
            img = _make_number_image("42", size=size)
            result = recognizer.recognize(img)
            assert "bib_number" in result
