"""Integration tests for batch processing endpoints (Phase 5).

Requires: docker compose up db redis -d
All tests use Celery eager mode so tasks run in-process.
"""
from __future__ import annotations

import io
import uuid

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

    # Initialize sync DB for task execution
    from src.db.sync_session import init_sync_db

    init_sync_db()

    yield

    # Restore
    celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)
    ml._blur_detector = None


def _make_jpeg(width: int = 256, height: int = 256) -> bytes:
    """Create a plain JPEG image."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
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
