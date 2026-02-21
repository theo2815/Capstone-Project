"""Integration tests for face recognition endpoints."""
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


def _make_jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a JPEG image in memory."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestFaceDetect:
    def test_detect_returns_200(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "faces_detected" in body["data"]
        assert "faces" in body["data"]
        assert "processing_time_ms" in body["data"]

    def test_detect_response_format(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert "success" in body
        assert "request_id" in body
        assert "timestamp" in body

    def test_detect_no_faces_in_random_noise(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("noise.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert body["data"]["faces_detected"] == 0
        assert body["data"]["faces"] == []

    def test_detect_invalid_file_type(self, client: TestClient):
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400


class TestFaceEnroll:
    def test_enroll_no_face(self, client: TestClient):
        """Enrolling an image with no face should return NO_FACES error."""
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/enroll",
            files={"file": ("noise.jpg", data, "image/jpeg")},
            data={"person_name": "Test Person"},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NO_FACES"

    def test_enroll_missing_person_name(self, client: TestClient):
        """Enrolling without person_name should fail validation."""
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/enroll",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 422


class TestFaceSearch:
    def test_search_no_face_returns_empty(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search",
            files={"file": ("noise.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["faces_detected"] == 0
        assert body["data"]["matches"] == []

    def test_search_response_format(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search",
            files={"file": ("noise.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert "faces_detected" in body["data"]
        assert "matches" in body["data"]
        assert "unmatched_faces" in body["data"]
        assert "processing_time_ms" in body["data"]


class TestFaceCompare:
    def test_compare_no_faces(self, client: TestClient):
        data1 = _make_jpeg_bytes()
        data2 = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/compare",
            files=[
                ("file1", ("a.jpg", data1, "image/jpeg")),
                ("file2", ("b.jpg", data2, "image/jpeg")),
            ],
        )
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NO_FACES"


class TestPersonCRUD:
    def test_get_nonexistent_person(self, client: TestClient):
        resp = client.get("/api/v1/faces/persons/00000000-0000-0000-0000-000000000000")
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NOT_FOUND"

    def test_delete_nonexistent_person(self, client: TestClient):
        resp = client.delete("/api/v1/faces/persons/00000000-0000-0000-0000-000000000000")
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NOT_FOUND"

    def test_invalid_uuid_returns_422(self, client: TestClient):
        resp = client.get("/api/v1/faces/persons/not-a-uuid")
        assert resp.status_code == 422
