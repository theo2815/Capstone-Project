"""Integration tests for POST /api/v1/bibs/recognize endpoint."""
from __future__ import annotations

import io
import os

# Must set env vars before any paddle import
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def app():
    from src.main import create_app

    return create_app()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _make_number_jpeg(text: str, width: int = 600, height: int = 400) -> bytes:
    """Create a JPEG with numbers drawn on it."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font_scale = min(height, width) / 80
    thickness = max(2, int(font_scale * 2))
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_blank_jpeg(width: int = 256, height: int = 256) -> bytes:
    """Create a JPEG with no text (uniform gray)."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a PNG image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


class TestBibEndpointSuccess:
    def test_recognizes_number_image(self, client: TestClient):
        data = _make_number_jpeg("1234")
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("bib.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["bibs_detected"] >= 1
        assert body["data"]["detections"][0]["bib_number"] == "1234"
        assert body["data"]["detections"][0]["confidence"] > 0.8

    def test_recognizes_two_digit_number(self, client: TestClient):
        data = _make_number_jpeg("42")
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("bib42.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        detections = body["data"]["detections"]
        assert len(detections) >= 1
        assert detections[0]["bib_number"] == "42"

    def test_blank_image_returns_no_bibs(self, client: TestClient):
        data = _make_blank_jpeg()
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("blank.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["bibs_detected"] == 0
        assert body["data"]["detections"] == []

    def test_response_envelope_format(self, client: TestClient):
        data = _make_number_jpeg("99")
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert "success" in body
        assert "request_id" in body
        assert "timestamp" in body
        assert "data" in body
        assert "error" in body

    def test_response_data_structure(self, client: TestClient):
        data = _make_number_jpeg("42")
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        d = body["data"]
        assert "bibs_detected" in d
        assert "detections" in d
        assert "image_dimensions" in d
        assert "processing_time_ms" in d
        assert d["processing_time_ms"] > 0
        assert d["image_dimensions"] == [600, 400]

    def test_detection_has_bbox(self, client: TestClient):
        data = _make_number_jpeg("42")
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        if body["data"]["detections"]:
            det = body["data"]["detections"][0]
            assert "bbox" in det
            bbox = det["bbox"]
            assert "x1" in bbox
            assert "y1" in bbox
            assert "x2" in bbox
            assert "y2" in bbox

    def test_detection_has_candidates(self, client: TestClient):
        data = _make_number_jpeg("42")
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        if body["data"]["detections"]:
            det = body["data"]["detections"][0]
            assert "all_candidates" in det
            assert len(det["all_candidates"]) >= 1
            cand = det["all_candidates"][0]
            assert "text" in cand
            assert "confidence" in cand

    def test_png_accepted(self, client: TestClient):
        data = _make_png_bytes()
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("test.png", data, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestBibEndpointValidation:
    def test_unsupported_content_type(self, client: TestClient):
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400

    def test_corrupt_image(self, client: TestClient):
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("bad.jpg", b"\xff\xd8\xff\xe0garbage", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_no_file_uploaded(self, client: TestClient):
        resp = client.post("/api/v1/bibs/recognize")
        assert resp.status_code == 422

    def test_image_too_small(self, client: TestClient):
        # 16x16 is below MIN_DIMENSION (32)
        arr = np.full((16, 16, 3), 128, dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        resp = client.post(
            "/api/v1/bibs/recognize",
            files={"file": ("tiny.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 400
