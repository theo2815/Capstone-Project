"""Integration tests for batch processing endpoints (Phase 5).

Requires: docker compose up db redis -d
All tests use Celery eager mode so tasks run in-process.
"""
from __future__ import annotations

import io
import os
import uuid

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


@pytest.fixture(autouse=True, scope="module")
def setup_celery_and_models(app, client):
    """Configure Celery eager mode and populate worker model globals.

    The `client` fixture dependency ensures the app lifespan has run
    (DB initialized, models loaded) before we copy models to the worker.
    """
    from src.workers.celery_app import celery_app

    import src.workers.model_loader as ml

    # Enable eager mode: tasks run synchronously in-process
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=False)

    # Copy models from app's registry to worker module globals
    registry = app.state.model_registry
    ml._blur_detector = registry.get("blur")
    ml._face_embedder = registry.get("face")
    ml._bib_detector = registry.get("bib_detector")
    ml._bib_recognizer = registry.get("bib_ocr")

    # Initialize sync DB for task execution
    from src.db.sync_session import init_sync_db

    init_sync_db()

    yield

    # Restore
    celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)
    ml._blur_detector = None
    ml._face_embedder = None
    ml._bib_detector = None
    ml._bib_recognizer = None


def _make_jpeg(width: int = 256, height: int = 256) -> bytes:
    """Create a plain JPEG image."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()


def _make_number_jpeg(text: str, width: int = 600, height: int = 400) -> bytes:
    """Create a JPEG with numbers drawn on it."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font_scale = min(height, width) / 80
    thickness = max(2, int(font_scale * 2))
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ---------- Blur Batch ----------


class TestBlurBatch:
    def test_submit_returns_202(self, client: TestClient):
        data = _make_jpeg()
        resp = client.post(
            "/api/v1/blur/detect/batch",
            files=[
                ("files", ("img1.jpg", data, "image/jpeg")),
                ("files", ("img2.jpg", data, "image/jpeg")),
            ],
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["total_items"] == 2
        assert "job_id" in body["data"]
        assert "poll_url" in body["data"]

    def test_job_poll_shows_completed(self, client: TestClient):
        """Submit batch, then poll — task runs eagerly so job is already done."""
        data = _make_jpeg()
        resp = client.post(
            "/api/v1/blur/detect/batch",
            files=[("files", ("img.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code == 202
        job_id = resp.json()["data"]["job_id"]

        # Poll for results (should be completed because eager mode)
        poll = client.get(f"/api/v1/jobs/{job_id}")
        assert poll.status_code == 200
        body = poll.json()
        assert body["success"] is True
        assert body["data"]["status"] == "completed"
        assert body["data"]["progress"] == 1.0
        assert body["data"]["result"] is not None
        assert len(body["data"]["result"]) == 1

    def test_blur_result_has_detection_fields(self, client: TestClient):
        """Verify each result item contains blur detection fields."""
        data = _make_jpeg()
        resp = client.post(
            "/api/v1/blur/detect/batch",
            files=[("files", ("img.jpg", data, "image/jpeg"))],
        )
        job_id = resp.json()["data"]["job_id"]
        poll = client.get(f"/api/v1/jobs/{job_id}")
        result = poll.json()["data"]["result"][0]
        assert "index" in result
        assert "is_blurry" in result
        assert "confidence" in result

    def test_no_files_returns_422(self, client: TestClient):
        """Submitting no files triggers FastAPI validation error."""
        resp = client.post("/api/v1/blur/detect/batch")
        assert resp.status_code == 422


# ---------- Bib Batch ----------


class TestBibBatch:
    def test_submit_returns_202(self, client: TestClient):
        data = _make_number_jpeg("42")
        resp = client.post(
            "/api/v1/bibs/recognize/batch",
            files=[("files", ("bib.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["total_items"] == 1

    def test_bib_batch_completed_with_results(self, client: TestClient):
        data = _make_number_jpeg("1234")
        resp = client.post(
            "/api/v1/bibs/recognize/batch",
            files=[("files", ("bib.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code == 202
        job_id = resp.json()["data"]["job_id"]

        poll = client.get(f"/api/v1/jobs/{job_id}")
        body = poll.json()
        assert body["data"]["status"] == "completed"
        results = body["data"]["result"]
        assert len(results) == 1
        assert results[0]["index"] == 0
        assert "bibs" in results[0]

    def test_multi_image_bib_batch(self, client: TestClient):
        img1 = _make_number_jpeg("42")
        img2 = _make_number_jpeg("99")
        resp = client.post(
            "/api/v1/bibs/recognize/batch",
            files=[
                ("files", ("bib1.jpg", img1, "image/jpeg")),
                ("files", ("bib2.jpg", img2, "image/jpeg")),
            ],
        )
        assert resp.status_code == 202
        job_id = resp.json()["data"]["job_id"]

        poll = client.get(f"/api/v1/jobs/{job_id}")
        body = poll.json()
        assert body["data"]["status"] == "completed"
        assert body["data"]["total_items"] == 2
        assert len(body["data"]["result"]) == 2


# ---------- Face Batch ----------


class TestFaceBatch:
    def test_submit_returns_202(self, client: TestClient):
        data = _make_jpeg()
        resp = client.post(
            "/api/v1/faces/search/batch",
            files=[("files", ("face.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["total_items"] == 1

    def test_face_detect_batch_completed(self, client: TestClient):
        data = _make_jpeg()
        resp = client.post(
            "/api/v1/faces/search/batch?operation=detect",
            files=[("files", ("face.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code == 202
        job_id = resp.json()["data"]["job_id"]

        poll = client.get(f"/api/v1/jobs/{job_id}")
        body = poll.json()
        assert body["data"]["status"] == "completed"
        results = body["data"]["result"]
        assert len(results) == 1
        assert "faces_detected" in results[0]


# ---------- Job Endpoint ----------


class TestJobEndpoint:
    def test_nonexistent_job_not_found(self, client: TestClient):
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/v1/jobs/{fake_id}")
        assert resp.status_code == 200  # wrapped in APIResponse
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NOT_FOUND"

    def test_invalid_job_id_422(self, client: TestClient):
        resp = client.get("/api/v1/jobs/not-a-uuid")
        assert resp.status_code == 422
