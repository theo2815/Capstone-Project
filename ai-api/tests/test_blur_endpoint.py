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


# ---------------------------------------------------------------------------
# POST /api/v1/blur/classify/stream  — the desktop's high-throughput path
# ---------------------------------------------------------------------------

SHARP = {
    "predicted_class": "sharp",
    "confidence": 0.97,
    "probabilities": {"sharp": 0.97, "motion_blurred": 0.03},
}


class _StubClassifier:
    """Minimal stand-in for BlurClassifier.

    The endpoint only touches ``min_detection_confidence`` and
    ``classify_batch``; keeping the stub to that surface makes these tests
    about the streaming contract rather than about model behaviour.
    """

    def __init__(self, results=None, raises: Exception | None = None,
                 min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self._results = results
        self._raises = raises
        self.calls: list[int] = []

    def classify_batch(self, images):
        self.calls.append(len(images))
        if self._raises is not None:
            raise self._raises
        if self._results is not None:
            return list(self._results)
        # Mirror the real contract: a None input (failed decode) maps to None.
        return [None if img is None else dict(SHARP) for img in images]


@pytest.fixture
def stub_registry(app):
    """Swap a stub classifier into the model registry, then restore it.

    ``app`` is module-scoped, so the original must go back or later tests in
    this file would run against the stub.
    """
    registry = app.state.model_registry
    original = registry.get("blur_classifier")
    installed = []

    def install(stub):
        registry._models["blur_classifier"] = stub
        installed.append(stub)
        return stub

    yield install
    registry._models["blur_classifier"] = original


def _ndjson(resp) -> list[dict]:
    import json

    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _files(datas: list[bytes]):
    return [("files", (f"img_{i}.jpg", d, "image/jpeg")) for i, d in enumerate(datas)]


class TestDecodeForClassify:
    """_decode_bgr_for_classify honours BLUR_CLASSIFY_DECODE_DIM.

    The cap is accuracy-relevant (640 beats 1280 on the labelled val set), so it
    is a calibrated setting rather than an arbitrary constant — worth pinning.
    """

    def test_downscales_long_edge_to_max_dim(self):
        from src.api.v1.blur import _decode_bgr_for_classify

        data = _make_jpeg_bytes(width=2000, height=1000)
        out = _decode_bgr_for_classify(data, 640)
        assert max(out.shape[:2]) == 640
        assert out.shape[:2] == (320, 640)  # aspect ratio preserved

    def test_leaves_small_images_untouched(self):
        from src.api.v1.blur import _decode_bgr_for_classify

        data = _make_jpeg_bytes(width=256, height=256)
        out = _decode_bgr_for_classify(data, 640)
        assert out.shape[:2] == (256, 256)

    def test_returns_none_on_undecodable_bytes(self):
        from src.api.v1.blur import _decode_bgr_for_classify

        assert _decode_bgr_for_classify(b"definitely not an image", 640) is None

    def test_default_setting_is_the_calibrated_640(self):
        from src.config import Settings

        assert Settings().BLUR_CLASSIFY_DECODE_DIM == 640


class TestClassifyStream:
    def test_streams_one_line_per_image_plus_summary(self, client, stub_registry):
        stub_registry(_StubClassifier())
        images = [_make_jpeg_bytes() for _ in range(3)]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 200
        assert resp.headers["X-Total-Images"] == "3"
        lines = _ndjson(resp)
        assert len(lines) == 4  # 3 results + 1 summary
        results, summary = lines[:3], lines[3]
        assert summary["_summary"] is True
        assert summary["total"] == 3
        # index/filename stay paired and in order
        for i, line in enumerate(results):
            assert line["index"] == i
            assert line["filename"] == f"img_{i}.jpg"
            assert line["predicted_class"] == "sharp"

    def test_undecodable_image_errors_without_killing_the_batch(self, client, stub_registry):
        stub_registry(_StubClassifier())
        images = [_make_jpeg_bytes(), b"not an image at all", _make_jpeg_bytes()]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 200
        lines = _ndjson(resp)
        assert len(lines) == 4
        assert lines[0]["predicted_class"] == "sharp"
        assert lines[1]["error"] == "Failed to decode image"
        assert lines[2]["predicted_class"] == "sharp"
        assert lines[3]["_summary"] is True

    def test_sub_batch_failure_emits_error_lines_and_still_summarises(
        self, client, stub_registry
    ):
        """Regression: classify_batch raising must not kill the stream.

        The 200 and headers are already sent by then, so an unhandled raise
        would truncate the body — no error lines, no _summary — and the desktop
        could not tell that apart from a completed run.
        """
        stub_registry(_StubClassifier(raises=RuntimeError("onnx session died")))
        images = [_make_jpeg_bytes() for _ in range(2)]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 200
        lines = _ndjson(resp)
        assert len(lines) == 3
        assert [line["index"] for line in lines[:2]] == [0, 1]
        assert all("onnx session died" in line["error"] for line in lines[:2])
        assert lines[2]["_summary"] is True  # stream terminated cleanly

    def test_blur_type_mode_respects_min_confidence(self, client, stub_registry):
        """detected must mirror BlurClassifier.detect_blur_type: the predicted
        class has to match AND clear the confidence floor."""
        low = {
            "predicted_class": "motion_blurred",
            "confidence": 0.30,
            "probabilities": {"motion_blurred": 0.30, "sharp": 0.70},
        }
        high = {
            "predicted_class": "motion_blurred",
            "confidence": 0.80,
            "probabilities": {"motion_blurred": 0.80, "sharp": 0.20},
        }
        stub_registry(_StubClassifier(results=[low, high], min_detection_confidence=0.5))
        images = [_make_jpeg_bytes() for _ in range(2)]

        resp = client.post(
            "/api/v1/blur/classify/stream?blur_type=motion_blurred",
            files=_files(images),
        )

        assert resp.status_code == 200
        lines = _ndjson(resp)
        assert lines[0]["detected"] is False  # 0.30 < floor
        assert lines[0]["blur_type_probability"] == 0.30
        assert lines[1]["detected"] is True   # 0.80 >= floor
        assert lines[1]["blur_type"] == "motion_blurred"

    def test_over_limit_returns_400(self, client, stub_registry, app, monkeypatch):
        stub_registry(_StubClassifier())
        monkeypatch.setattr(app.state.settings, "STREAM_CLASSIFY_MAX_SIZE", 2)
        images = [_make_jpeg_bytes() for _ in range(3)]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "BATCH_TOO_LARGE"

    def test_batches_in_sub_batch_sized_chunks(self, client, stub_registry, app, monkeypatch):
        """Inference is issued per INFERENCE_SUB_BATCH_SIZE chunk, not per image
        — that grouping is the whole point of the batched path."""
        stub = stub_registry(_StubClassifier())
        monkeypatch.setattr(app.state.settings, "INFERENCE_SUB_BATCH_SIZE", 2)
        images = [_make_jpeg_bytes() for _ in range(5)]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 200
        assert stub.calls == [2, 2, 1]
        assert len(_ndjson(resp)) == 6  # 5 results + summary
