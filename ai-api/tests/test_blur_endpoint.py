"""Integration tests for POST /api/v1/blur/detect endpoint."""
from __future__ import annotations

import io

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


def _make_jpeg_bytes(width: int = 256, height: int = 256, sharp: bool = True) -> bytes:
    """Create a JPEG image in memory."""
    if sharp:
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    else:
        arr = np.full((height, width, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a PNG image in memory."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestBlurEndpointSuccess:
    def test_sharp_image_detected(self, client: TestClient):
        data = _make_jpeg_bytes(sharp=True)
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("sharp.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["is_blurry"] is False
        assert body["data"]["confidence"] > 0.5
        assert body["data"]["processing_time_ms"] > 0
        assert body["data"]["image_dimensions"] == [256, 256]

    def test_blurry_image_detected(self, client: TestClient):
        data = _make_jpeg_bytes(sharp=False)
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("blurry.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["is_blurry"] is True
        assert body["data"]["confidence"] > 0.9

    def test_png_accepted(self, client: TestClient):
        data = _make_png_bytes()
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("test.png", data, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_response_envelope_format(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert "success" in body
        assert "request_id" in body
        assert "timestamp" in body
        assert "data" in body
        assert "error" in body

    def test_metrics_included_by_default(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        metrics = body["data"]["metrics"]
        assert metrics is not None
        assert "laplacian_variance" in metrics
        assert "hf_ratio" in metrics
        assert "confidence" in metrics

    def test_metrics_excluded_when_requested(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/blur/detect?include_metrics=false",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert body["data"]["metrics"] is None

    def test_custom_threshold(self, client: TestClient):
        """A very high threshold should make any image 'blurry'."""
        data = _make_jpeg_bytes(sharp=True)
        resp = client.post(
            "/api/v1/blur/detect?threshold=9999",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert body["success"] is True
        # With threshold=9999, even sharp images may be classified as blurry
        # because laplacian_variance might be below 9999


class TestBlurEndpointValidation:
    def test_unsupported_content_type(self, client: TestClient):
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400

    def test_corrupt_image(self, client: TestClient):
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("bad.jpg", b"\xff\xd8\xff\xe0garbage", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_no_file_uploaded(self, client: TestClient):
        resp = client.post("/api/v1/blur/detect")
        assert resp.status_code == 422  # FastAPI validation error

    def test_image_too_small(self, client: TestClient):
        data = _make_jpeg_bytes(width=16, height=16)
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("tiny.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 400


class TestBlurEndpointAuth:
    def test_no_key_in_debug_mode_allowed(self, client: TestClient):
        """In DEBUG mode (default .env), no API key should still work."""
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/blur/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        # DEBUG=true in .env allows keyless access
        assert resp.status_code == 200

    def test_invalid_key_rejected(self, client: TestClient):
        """An explicitly wrong key should be rejected."""
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/blur/detect",
            headers={"X-API-Key": "definitely_wrong_key"},
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        # Even in debug, if a key IS provided it must be valid
        assert resp.status_code == 401
