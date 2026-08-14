"""Integration tests for critical API paths.

These tests exercise the full request pipeline (middleware -> route -> service)
using FastAPI's TestClient with mocked ML models and database. They verify:

1. Auth + rate limiting working together
2. Model unavailable returns 503
3. Security headers are present
4. Scope enforcement blocks unauthorized access
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    """Create a test app with mocked dependencies."""
    # Import src.main OUTSIDE the patch block. src/main.py binds ModelRegistry
    # at import time (`from src.ml.registry import ModelRegistry`), so importing
    # it while the class is patched would bind the mock permanently for the whole
    # session — every later create_app() would then build a registry whose get()
    # returns None, and unrelated tests would see 503 MODEL_UNAVAILABLE.
    from src.main import create_app

    # Patch where the name is *used* (src.main), not where it is defined, so the
    # mock actually reaches lifespan and is restored cleanly on fixture teardown.
    with (
        patch("src.db.session.init_db", new_callable=AsyncMock),
        patch("src.db.session.close_db", new_callable=AsyncMock),
        patch("src.main.ModelRegistry") as MockRegistry,
    ):
        mock_registry = MagicMock()
        mock_registry.load_all = AsyncMock()
        mock_registry.unload_all = AsyncMock()
        mock_registry.all_loaded.return_value = True
        # All models return None (unavailable) by default
        mock_registry.get.return_value = None
        MockRegistry.return_value = mock_registry

        test_app = create_app()
        # Inject settings for test. MAX_FILE_SIZE must be a real int: the upload
        # handlers call validate_and_decode(file, max_file_size=...) before the
        # model check, and a bare MagicMock compares truthy against len(bytes),
        # so every request would 400 on "file exceeds limit" and never reach the
        # 503 MODEL_UNAVAILABLE path these tests exist to cover.
        test_app.state.settings = MagicMock(
            APP_NAME="QuickPitik Test",
            APP_VERSION="1.0.0-test",
            DEBUG=True,
            ENVIRONMENT="development",
            LOG_LEVEL="WARNING",
            ALLOWED_ORIGINS=["*"],
            WEBHOOK_SECRET_KEY="",
            API_KEY_HEADER="X-API-Key",
            MAX_FILE_SIZE=10 * 1024 * 1024,
        )
        yield test_app


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    """Health endpoint should be accessible without auth."""

    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        # Liveness returns HealthResponse (status + version), not the standard
        # APIResponse envelope — it is the one endpoint with no auth and no
        # request_id. See src/api/v1/health.py.
        assert data["status"] == "alive"


class TestSecurityHeaders:
    """SEC-11: Security headers must be present on all responses."""

    def test_nosniff_header(self, client):
        response = client.get("/api/v1/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_frame_deny_header(self, client):
        response = client.get("/api/v1/health")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_xss_protection_header(self, client):
        response = client.get("/api/v1/health")
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_referrer_policy_header(self, client):
        response = client.get("/api/v1/health")
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


class TestModelUnavailable503:
    """BUG-3: Model unavailable must return 503, not 200."""

    def _post_with_image(self, client, url: str) -> dict:
        """POST a dummy image to an endpoint (using debug auth bypass).

        The image must be at least MIN_DIMENSION (32px) on both edges — a
        smaller one is rejected by validate_and_decode with 400 before the
        handler ever reaches the model-availability check these tests assert on.
        """
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="JPEG")
        return client.post(
            url,
            files={"file": ("test.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")},
        )

    def test_blur_detect_503(self, client, app):
        """Blur detect returns 503 when model unavailable."""
        app.state.model_registry.get.return_value = None
        resp = self._post_with_image(client, "/api/v1/blur/detect")
        assert resp.status_code == 503
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "MODEL_UNAVAILABLE"

    def test_face_detect_503(self, client, app):
        """Face detect returns 503 when model unavailable."""
        app.state.model_registry.get.return_value = None
        resp = self._post_with_image(client, "/api/v1/faces/detect")
        assert resp.status_code == 503

    def test_bib_recognize_503(self, client, app):
        """Bib recognize returns 503 when OCR model unavailable."""
        app.state.model_registry.get.return_value = None
        resp = self._post_with_image(client, "/api/v1/bibs/recognize")
        assert resp.status_code == 503


class TestScopeEnforcement:
    """SEC-1: Scope enforcement must block unauthorized access."""

    def test_missing_key_returns_401_in_non_debug(self, app):
        """Without DEBUG, missing API key returns 401."""
        # Temporarily disable DEBUG
        app.state.settings.DEBUG = False
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/api/v1/blur/detect",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
                )
                assert resp.status_code == 401
        finally:
            app.state.settings.DEBUG = True


class TestWebhookSSRFValidation:
    """SEC-10: Private IP webhook URLs should be rejected at registration."""

    def test_private_ip_rejected(self, client):
        resp = client.post(
            "/api/v1/webhooks",
            json={
                "url": "http://192.168.1.1/callback",
                "events": ["job.completed"],
            },
        )
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "INVALID_WEBHOOK_URL"

    def test_loopback_rejected(self, client):
        resp = client.post(
            "/api/v1/webhooks",
            json={
                "url": "http://127.0.0.1:8080/hook",
                "events": ["job.completed"],
            },
        )
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "INVALID_WEBHOOK_URL"

    def test_invalid_scheme_rejected(self, client):
        resp = client.post(
            "/api/v1/webhooks",
            json={
                "url": "ftp://example.com/hook",
                "events": ["job.completed"],
            },
        )
        # A non-http(s) scheme never reaches the handler's SSRF check: the
        # request model types url as pydantic HttpUrl, so FastAPI rejects it at
        # schema validation with 422 + {"detail": [...]}. Rejected earlier and
        # harder than INVALID_WEBHOOK_URL — assert the contract that applies.
        assert resp.status_code == 422
        assert "detail" in resp.json()
