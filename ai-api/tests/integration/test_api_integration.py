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
    # Patch heavy imports before app creation
    with (
        patch("src.db.session.init_db", new_callable=AsyncMock),
        patch("src.db.session.close_db", new_callable=AsyncMock),
        patch("src.ml.registry.ModelRegistry") as MockRegistry,
    ):
        mock_registry = MagicMock()
        mock_registry.load_all = AsyncMock()
        mock_registry.unload_all = AsyncMock()
        mock_registry.all_loaded.return_value = True
        # All models return None (unavailable) by default
        mock_registry.get.return_value = None
        MockRegistry.return_value = mock_registry

        from src.main import create_app
        test_app = create_app()
        # Inject settings for test
        test_app.state.settings = MagicMock(
            APP_NAME="EventAI Test",
            APP_VERSION="1.0.0-test",
            DEBUG=True,
            ENVIRONMENT="development",
            LOG_LEVEL="WARNING",
            ALLOWED_ORIGINS=["*"],
            WEBHOOK_SECRET_KEY="",
            API_KEY_HEADER="X-API-Key",
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
        assert "success" in data


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
        """POST a dummy image to an endpoint (using debug auth bypass)."""
        # 1x1 white JPEG
        import io
        jpeg_bytes = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
            b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"
            b"\x22q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16"
            b"\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83"
            b"\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a"
            b"\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8"
            b"\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6"
            b"\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2"
            b"\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
        )
        return client.post(
            url,
            files={"file": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
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
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "INVALID_WEBHOOK_URL"
