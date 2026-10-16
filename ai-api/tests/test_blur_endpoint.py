"""Integration tests for POST /api/v1/blur/detect endpoint."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import Request
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


def _textured_jpeg() -> bytes:
    y, x = np.indices((800, 1280))
    pattern = np.uint8(128 + (2 * ((x // 8 + y // 8) % 2) - 1) * 6)
    buf = io.BytesIO()
    Image.fromarray(pattern).convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_png_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a PNG image in memory."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid_png(width: int, height: int) -> bytes:
    """Flat-colour PNG. Compresses to almost nothing, so a 12000px edge costs
    little — which is what makes testing against the real cap practical."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _make_oversized_png_bytes() -> bytes:
    """One pixel past MAX_IMAGE_DIMENSION on the long edge."""
    from src.utils.image_utils import MAX_DIMENSION

    return _solid_png(MAX_DIMENSION + 1, 32)


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

    def test_metrics_excluded_skips_fft_without_changing_decision(self, client):
        data = _make_jpeg_bytes(sharp=False)
        files = {"file": ("test.jpg", data, "image/jpeg")}
        expected = client.post("/api/v1/blur/detect", files=files).json()["data"]
        with patch("src.ml.blur.detector.np.fft.fft2",
                   side_effect=AssertionError("FFT should be skipped")):
            response = client.post(
                "/api/v1/blur/detect?include_metrics=false", files=files
            )
        assert response.status_code == 200
        result = response.json()["data"]
        assert result["metrics"] is None
        assert result["is_blurry"] == expected["is_blurry"]
        assert result["confidence"] == expected["confidence"]

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

    def test_invalid_key_rejected(self, client: TestClient, app):
        """An explicitly wrong key is rejected when auth dependencies are available."""
        redis = AsyncMock()
        redis.get.return_value = None
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session = AsyncMock()
        session.execute.return_value = result
        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = session
        data = _make_jpeg_bytes()
        with (
            patch.object(app.state, "redis", redis),
            patch("src.db.session.get_session_ctx", return_value=session_ctx),
        ):
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

    The two-stage classifier needs the whole desktop upload (1280): it makes its
    own smaller views. The setting is pinned so a change is always deliberate.
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

    def test_thin_image_downscale_keeps_both_axes(self):
        from src.api.v1.blur import _decode_bgr_for_classify

        out = _decode_bgr_for_classify(_solid_png(1, 2000), 640)
        assert out.shape[:2] == (640, 1)

    def test_normal_downscale_preserves_calibrated_opencv_mapping(self):
        from src.api.v1.blur import _decode_bgr_for_classify

        data = _make_jpeg_bytes(width=2000, height=999)
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        expected = cv2.resize(
            decoded, None, fx=640 / 2000, fy=640 / 2000, interpolation=cv2.INTER_AREA
        )
        assert np.array_equal(_decode_bgr_for_classify(data, 640), expected)

    def test_default_setting_is_the_full_desktop_upload(self):
        from src.config import Settings

        assert Settings().BLUR_CLASSIFY_DECODE_DIM == 1280


class TestClassifyStream:
    def test_desktop_single_and_stream_work_with_async_disabled(
        self, client, app, stub_registry, monkeypatch
    ):
        class DesktopClassifier(_StubClassifier):
            def classify(self, image):
                return {**SHARP, "probabilities": {
                    "sharp": 0.97, "motion_blurred": 0.03,
                    "defocused_blurred": 0.0, "defocused_object_portrait": 0.0,
                }}

        stub_registry(DesktopClassifier())
        monkeypatch.setattr(app.state.settings, "BLUR_ASYNC_ENABLED", False)
        image = _make_jpeg_bytes()

        single = client.post(
            "/api/v1/blur/classify",
            files={"file": ("photo.jpg", image, "image/jpeg")},
        )
        stream = client.post(
            "/api/v1/blur/classify/stream", files=_files([image])
        )

        assert single.status_code == 200
        assert single.json()["data"]["predicted_class"] == "sharp"
        assert stream.status_code == 200
        lines = _ndjson(stream)
        assert lines[0]["predicted_class"] == "sharp"
        assert lines[1]["_summary"] is True
        assert lines[1]["complete"] is True

    def test_deadline_marks_stream_incomplete_and_stops_next_sub_batch(
        self, client, app, stub_registry, monkeypatch
    ):
        class SlowClassifier(_StubClassifier):
            def classify_batch(self, images):
                self.calls.append(len(images))
                time.sleep(0.08)
                return [dict(SHARP) for _ in images]

        stub = stub_registry(SlowClassifier())
        monkeypatch.setattr(app.state.settings, "INFERENCE_SUB_BATCH_SIZE", 1)
        monkeypatch.setattr(app.state.settings, "BLUR_STREAM_TIMEOUT_SECONDS", 0.02)
        response = client.post(
            "/api/v1/blur/classify/stream", files=_files([_make_jpeg_bytes()] * 2)
        )
        lines = _ndjson(response)
        assert lines[-1]["_summary"] is True
        assert lines[-1]["complete"] is False
        assert lines[-1]["successful"] < 2
        assert len(stub.calls) <= 1

    def test_static_batch_stops_after_stream_deadline(
        self, client, app, stub_registry, monkeypatch
    ):
        from src.utils.timeout import start_timeout_executor

        start_timeout_executor()

        class StaticClassifier(_StubClassifier):
            _dynamic_batch = False

            def classify(self, image):
                self.calls.append(1)
                time.sleep(0.5)
                return dict(SHARP)

            def classify_batch(self, images):
                raise AssertionError("Static path must check the deadline between images")

        stub = stub_registry(StaticClassifier())
        monkeypatch.setattr(app.state.settings, "BLUR_STREAM_TIMEOUT_SECONDS", 0.3)
        response = client.post(
            "/api/v1/blur/classify/stream", files=_files([_make_jpeg_bytes()] * 3)
        )
        assert _ndjson(response)[-1]["complete"] is False
        assert len(stub.calls) == 1

    def test_busy_classifier_returns_service_envelope(self, client, stub_registry):
        from src.utils.timeout import InferenceBusyError

        class BusyClassifier(_StubClassifier):
            def classify(self, image):
                raise InferenceBusyError("private state")

        stub_registry(BusyClassifier())
        response = client.post(
            "/api/v1/blur/classify",
            files={"file": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "1"
        assert response.json()["error"]["code"] == "BLUR_BUSY"
        assert "private state" not in response.text

    @pytest.mark.parametrize("route,model", [
        ("classify", "blur_classifier"), ("detect", "blur"),
    ])
    def test_missing_model_rejected_before_stream(self, client, app, route, model):
        registry = app.state.model_registry
        original = registry.get(model)
        registry._models[model] = None
        try:
            resp = client.post(f"/api/v1/blur/{route}/stream", files=_files([_make_jpeg_bytes()]))
        finally:
            registry._models[model] = original

        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "MODEL_UNAVAILABLE"
        assert resp.headers.get("content-type", "").startswith("application/json")

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
        stub = stub_registry(_StubClassifier())
        images = [_make_jpeg_bytes(), b"", _make_jpeg_bytes()]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 200
        lines = _ndjson(resp)
        assert len(lines) == 4
        assert lines[0]["predicted_class"] == "sharp"
        assert lines[1]["error"] == "Failed to decode image"
        assert lines[2]["predicted_class"] == "sharp"
        assert lines[3]["_summary"] is True
        assert lines[3]["successful"] == 2
        assert lines[3]["errors"] == 1
        assert stub.calls == [2]

    def test_later_sub_batch_decode_failure_keeps_all_indices(
        self, client, stub_registry, app, monkeypatch
    ):
        stub = stub_registry(_StubClassifier())
        monkeypatch.setattr(app.state.settings, "INFERENCE_SUB_BATCH_SIZE", 2)
        from src.api.v1 import blur

        original = blur._decode_bgr_for_classify
        bad = _make_jpeg_bytes()

        def decode(data, max_dim):
            if data == bad:
                raise cv2.error("private/path/decoder.cpp: internal details")
            return original(data, max_dim)

        with patch("src.api.v1.blur._decode_bgr_for_classify", side_effect=decode):
            resp = client.post(
                "/api/v1/blur/classify/stream",
                files=_files([_make_jpeg_bytes(), _make_jpeg_bytes(), bad, _make_jpeg_bytes()]),
            )

        lines = _ndjson(resp)
        assert len(lines) == 5
        assert [row["index"] for row in lines[:4]] == [0, 1, 2, 3]
        assert lines[2]["error"] == "Failed to decode image"
        assert "private/path" not in resp.text
        assert [row["predicted_class"] for row in (lines[0], lines[1], lines[3])] == ["sharp"] * 3
        assert lines[4]["successful"] == 3
        assert lines[4]["errors"] == 1
        assert stub.calls == [2, 1]

    def test_unexpected_stream_failure_is_marked_incomplete(self, client, stub_registry):
        stub_registry(_StubClassifier())
        with patch(
            "src.api.v1.blur.validate_stream_file", side_effect=RuntimeError("private/path")
        ):
            resp = client.post("/api/v1/blur/classify/stream", files=_files([_make_jpeg_bytes()]))

        lines = _ndjson(resp)
        assert len(lines) == 1
        assert lines[0]["_summary"] is True
        assert (lines[0]["total"], lines[0]["successful"], lines[0]["errors"]) == (
            1, 0, 0,
        )
        assert lines[0]["complete"] is False
        assert "private/path" not in resp.text

    def test_disconnect_stops_before_next_sub_batch(self, client, stub_registry, app, monkeypatch):
        stub = stub_registry(_StubClassifier())
        monkeypatch.setattr(app.state.settings, "INFERENCE_SUB_BATCH_SIZE", 1)
        with patch.object(Request, "is_disconnected", new_callable=AsyncMock) as disconnected:
            disconnected.side_effect = [False, True]
            resp = client.post(
                "/api/v1/blur/classify/stream", files=_files([_make_jpeg_bytes()] * 2)
            )

        assert [line["index"] for line in _ndjson(resp)] == [0]
        assert stub.calls == [1]

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
        assert all(line["error"] == "Classification failed" for line in lines[:2])
        assert lines[2]["_summary"] is True  # stream terminated cleanly
        assert lines[2]["successful"] == 0
        assert lines[2]["errors"] == 2

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

    def test_oversize_file_errors_without_killing_the_batch(
        self, client, stub_registry, app, monkeypatch
    ):
        """A file over MAX_FILE_SIZE is refused per-image, not per-request.

        Until 2026-08-14 the stream endpoints validated nothing at all, so
        arbitrarily large frames went straight into cv2.imdecode.
        """
        stub_registry(_StubClassifier())
        valid = _make_jpeg_bytes()
        monkeypatch.setattr(app.state.settings, "MAX_FILE_SIZE", len(valid) + 16)
        images = [valid, b"x" * (len(valid) + 64), valid]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 200
        lines = _ndjson(resp)
        assert len(lines) == 4
        assert lines[0]["predicted_class"] == "sharp"
        assert "exceeds" in lines[1]["error"] and "limit" in lines[1]["error"]
        assert lines[2]["predicted_class"] == "sharp"  # neighbour unaffected
        assert lines[3]["_summary"] is True

    def test_over_dimension_file_errors_without_killing_the_batch(
        self, client, stub_registry
    ):
        """Checked against the real MAX_IMAGE_DIMENSION, not a patched one —
        cv2.imdecode has no decompression-bomb guard, so this header read is
        the only thing bounding the allocation."""
        stub_registry(_StubClassifier())
        images = [_make_jpeg_bytes(), _make_oversized_png_bytes(), _make_jpeg_bytes()]

        resp = client.post("/api/v1/blur/classify/stream", files=_files(images))

        assert resp.status_code == 200
        lines = _ndjson(resp)
        assert len(lines) == 4
        assert "12001x32" in lines[1]["error"]
        assert "exceed" in lines[1]["error"]
        assert lines[2]["predicted_class"] == "sharp"
        assert lines[3]["_summary"] is True

    def test_finished_line_carries_milliseconds_per_photo_per_stage(
        self, client, stub_registry, monkeypatch
    ):
        from src.api.v1 import blur as blur_module

        class TimedClassifier(_StubClassifier):
            """Stage totals that have grown by 0.004 / 0.3 / 0.06 s when read the second time."""

            snapshots = 0

            def timing_snapshot(self):
                self.snapshots += 1
                grown = self.snapshots > 1
                return {"stage1": 0.004 if grown else 0.0, "detect": 0.3 if grown else 0.0,
                        "measure": 0.06 if grown else 0.0}

        lines = []

        class Recorder:
            def info(self, event, **fields):
                lines.append((event, fields))

            def warning(self, event, **fields):
                pass

            error = exception = debug = warning

        monkeypatch.setattr(blur_module, "logger", Recorder())
        stub_registry(TimedClassifier())

        resp = client.post("/api/v1/blur/classify/stream", files=_files([_make_jpeg_bytes()] * 2))

        assert resp.status_code == 200
        finished = next(
            fields for event, fields in lines if event == "Stream blur classification finished"
        )
        assert finished["stage1_ms_per_photo"] == 2.0  # 0.004 s over 2 photos
        assert finished["detect_ms_per_photo"] == 150.0
        assert finished["measure_ms_per_photo"] == 30.0
        assert 0 <= finished["decode_ms_per_photo"] <= finished["ms_per_photo"]


# ---------------------------------------------------------------------------
# POST /api/v1/blur/detect/stream — the desktop's other high-throughput path
#
# This endpoint had no coverage at all until 2026-08-14 while its classify twin
# had a full class, which is how it kept its unvalidated read loop unnoticed.
# ---------------------------------------------------------------------------


class TestDetectStream:
    def test_stream_deadline_is_counted(self, client, app, monkeypatch):
        from src.utils.blur_metrics import timeouts

        before = timeouts.labels("stream")._value.get()
        monkeypatch.setattr(app.state.settings, "BLUR_STREAM_TIMEOUT_SECONDS", 0.000001)
        response = client.post(
            "/api/v1/blur/detect/stream", files=_files([_make_jpeg_bytes()])
        )
        assert _ndjson(response)[-1]["complete"] is False
        assert timeouts.labels("stream")._value.get() - before == 1

    @pytest.mark.parametrize("decode_dim", [640, 512])
    def test_classical_paths_agree_on_textured_jpeg(
        self, client, app, tmp_path, monkeypatch, decode_dim
    ):
        """A fine pattern previously flipped at 1280 vs 640 pixels."""
        from src.workers.tasks.blur_tasks import blur_detect_batch

        monkeypatch.setattr(app.state.settings, "BLUR_DETECT_DECODE_DIM", decode_dim)
        data = _textured_jpeg()

        single = client.post(
            "/api/v1/blur/detect", files={"file": ("pattern.jpg", data, "image/jpeg")}
        ).json()["data"]
        stream = _ndjson(client.post(
            "/api/v1/blur/detect/stream", files=_files([data])
        ))[0]
        stream_fft = _ndjson(client.post(
            "/api/v1/blur/detect/stream?include_hf_ratio=true", files=_files([data])
        ))[0]
        import uuid

        from src.utils.blob_store import store_batch

        monkeypatch.setattr(app.state.settings, "BLOB_STORE_PATH", str(tmp_path))
        job_id = str(uuid.uuid4())
        key = store_batch(job_id, [data])[0]
        monkeypatch.setattr("src.workers.helpers.touch_mega_job", lambda _: True)
        with patch("src.workers.model_loader.get_blur_detector",
                   return_value=app.state.model_registry.get("blur")), patch(
                   "src.config.get_settings", return_value=app.state.settings):
            worker = blur_detect_batch(job_id, [key], mega_child=True)[0]

        assert (
            single["is_blurry"] == stream["is_blurry"]
            == stream_fft["is_blurry"] == worker["is_blurry"]
        )
        assert single["metrics"]["laplacian_variance"] == pytest.approx(
            stream["laplacian_variance"], abs=0.01
        )
        assert worker["laplacian_variance"] == pytest.approx(
            stream["laplacian_variance"], abs=0.01
        )
        assert stream_fft["laplacian_variance"] == pytest.approx(
            stream["laplacian_variance"], abs=0.01
        )
        assert stream_fft["hf_ratio"] == pytest.approx(single["metrics"]["hf_ratio"])

    def test_classical_threshold_default_and_override(self, client, app, monkeypatch):
        detector = app.state.model_registry.get("blur")
        monkeypatch.setattr(detector, "laplacian_threshold", 300.0)
        data = _textured_jpeg()
        files = {"file": ("pattern.jpg", data, "image/jpeg")}

        default = client.post("/api/v1/blur/detect", files=files).json()["data"]
        explicit = client.post("/api/v1/blur/detect?threshold=300", files=files).json()["data"]
        override = client.post("/api/v1/blur/detect?threshold=100", files=files).json()["data"]
        stream = _ndjson(client.post(
            "/api/v1/blur/detect/stream", files=_files([data])
        ))[0]
        assert default["is_blurry"] == explicit["is_blurry"] == stream["is_blurry"]
        assert default["is_blurry"] != override["is_blurry"]

    @pytest.mark.parametrize("threshold", [0, 10001, "nan", "inf"])
    def test_classical_threshold_rejects_out_of_range(self, client, threshold):
        data = _make_jpeg_bytes()
        assert client.post(
            f"/api/v1/blur/detect?threshold={threshold}",
            files={"file": ("x.jpg", data, "image/jpeg")},
        ).status_code == 422
        assert client.post(
            f"/api/v1/blur/detect/stream?threshold={threshold}",
            files=_files([data]),
        ).status_code == 422

    def test_streams_one_line_per_image_plus_summary(self, client):
        images = [_make_jpeg_bytes() for _ in range(3)]

        resp = client.post("/api/v1/blur/detect/stream", files=_files(images))

        assert resp.status_code == 200
        assert resp.headers["X-Total-Images"] == "3"
        lines = _ndjson(resp)
        assert len(lines) == 4  # 3 results + summary
        summary = [line for line in lines if line.get("_summary")]
        assert len(summary) == 1
        assert summary[0]["total"] == 3
        # Results complete out of order here (asyncio.as_completed), so index is
        # the identity, not position.
        results = [line for line in lines if not line.get("_summary")]
        assert sorted(line["index"] for line in results) == [0, 1, 2]
        assert all("is_blurry" in line for line in results)

    def test_undecodable_image_errors_without_killing_the_batch(self, client):
        images = [_make_jpeg_bytes(), b"", _make_jpeg_bytes()]

        resp = client.post("/api/v1/blur/detect/stream", files=_files(images))

        assert resp.status_code == 200
        by_index = {line["index"]: line for line in _ndjson(resp) if not line.get("_summary")}
        assert by_index[1]["error"] == "Failed to decode image"
        assert "is_blurry" in by_index[0] and "is_blurry" in by_index[2]

    def test_decode_exception_is_sanitized_and_counted(self, client):
        with patch(
            "src.api.v1.blur.cv2.imdecode", side_effect=cv2.error("private/path/decoder.cpp")
        ):
            resp = client.post("/api/v1/blur/detect/stream", files=_files([_make_jpeg_bytes()]))

        lines = _ndjson(resp)
        assert lines[0]["error"] == "Failed to process image"
        assert "private/path" not in resp.text
        assert lines[1]["successful"] == 0
        assert lines[1]["errors"] == 1

    def test_oversize_file_errors_without_killing_the_batch(
        self, client, app, monkeypatch
    ):
        valid = _make_jpeg_bytes()
        monkeypatch.setattr(app.state.settings, "MAX_FILE_SIZE", len(valid) + 16)
        images = [valid, b"x" * (len(valid) + 64), valid]

        resp = client.post("/api/v1/blur/detect/stream", files=_files(images))

        assert resp.status_code == 200
        lines = _ndjson(resp)
        assert len(lines) == 4
        by_index = {line["index"]: line for line in lines if not line.get("_summary")}
        assert "exceeds" in by_index[1]["error"]
        assert "is_blurry" in by_index[0] and "is_blurry" in by_index[2]

    def test_over_dimension_file_errors_without_killing_the_batch(self, client):
        images = [_make_jpeg_bytes(), _make_oversized_png_bytes(), _make_jpeg_bytes()]

        resp = client.post("/api/v1/blur/detect/stream", files=_files(images))

        assert resp.status_code == 200
        by_index = {line["index"]: line for line in _ndjson(resp) if not line.get("_summary")}
        assert "12001x32" in by_index[1]["error"]
        assert "is_blurry" in by_index[0] and "is_blurry" in by_index[2]

    def test_over_limit_returns_400(self, client, app, monkeypatch):
        monkeypatch.setattr(app.state.settings, "STREAM_BATCH_MAX_SIZE", 2)
        images = [_make_jpeg_bytes() for _ in range(3)]

        resp = client.post("/api/v1/blur/detect/stream", files=_files(images))

        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "BATCH_TOO_LARGE"


# ---------------------------------------------------------------------------
# validate_stream_file — the per-file half of the 2026-08-14 limits work
# ---------------------------------------------------------------------------


class TestValidateStreamFile:
    """The lighter sibling of validate_batch_file used by the /stream paths.

    It deliberately drops verify() and the content-type check; what it must NOT
    drop is the size and pixel bounds, which are the two cv2.imdecode cannot
    enforce for itself.
    """

    def test_accepts_a_normal_image(self):
        from src.utils.image_utils import validate_stream_file

        assert validate_stream_file(_make_jpeg_bytes(), "a.jpg", 25 * 1024 * 1024) is None

    def test_rejects_oversize_bytes(self):
        from src.utils.image_utils import validate_stream_file

        msg = validate_stream_file(b"x" * 2048, "big.jpg", 1024)
        assert msg is not None and "big.jpg" in msg

    def test_rejects_over_dimension(self):
        from src.utils.image_utils import MAX_DIMENSION, validate_stream_file

        msg = validate_stream_file(
            _make_oversized_png_bytes(), "wide.png", 25 * 1024 * 1024
        )
        assert msg is not None
        assert f"{MAX_DIMENSION}px" in msg

    def test_accepts_exactly_the_cap(self):
        """Boundary matches validate_batch_file's `>` — the cap itself passes."""
        from src.utils.image_utils import MAX_DIMENSION, validate_stream_file

        data = _solid_png(MAX_DIMENSION, 32)
        assert validate_stream_file(data, "edge.png", 25 * 1024 * 1024) is None

    def test_rejects_unparseable_header_before_cv2(self):
        from src.utils.image_utils import validate_stream_file

        assert (
            validate_stream_file(b"not an image", "x.jpg", 25 * 1024 * 1024)
            == "Failed to decode image"
        )

    def test_rejects_a_decompression_bomb(self, monkeypatch):
        """PIL raises rather than reporting a size once past MAX_IMAGE_PIXELS,
        so that path has to reject rather than fall through to cv2."""
        from PIL import Image as PILImage

        from src.utils.image_utils import validate_stream_file

        monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 100)
        msg = validate_stream_file(_make_jpeg_bytes(), "bomb.jpg", 25 * 1024 * 1024)
        assert msg is not None and "bomb.jpg" in msg


# ---------------------------------------------------------------------------
# MAX_REQUEST_BODY — the aggregate bound
# ---------------------------------------------------------------------------


class TestBodySizeLimit:
    """Aggregate ASGI byte limits apply before multipart parsing."""

    def test_declared_body_over_the_cap_is_refused(self, client, app):
        over = app.state.settings.MAX_REQUEST_BODY + 1

        resp = client.post(
            "/api/v1/blur/detect/stream",
            content=b"x",
            headers={"Content-Length": str(over)},
        )

        assert resp.status_code == 413
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "REQUEST_TOO_LARGE"
        # Registered inside RequestIDMiddleware so the envelope is complete.
        assert body["request_id"]

    def test_normal_request_passes(self, client):
        resp = client.post(
            "/api/v1/blur/detect/stream", files=_files([_make_jpeg_bytes()])
        )

        assert resp.status_code == 200

    def test_actual_body_limit_without_content_length(self):
        from src.config import Settings
        from src.main import create_app

        settings = Settings(MAX_REQUEST_BODY=1024, MAX_FILE_SIZE=100)
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()

        @app.post("/probe")
        async def probe(request: Request):
            await request.body()
            return {"ok": True}

        def chunked():
            yield b"x" * 700
            yield b"x" * 700

        response = TestClient(app).post("/probe", content=chunked())
        assert response.status_code == 413, response.text
        assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


        assert response.json()["request_id"]

    def test_lying_content_length_is_rejected(self):
        from src.config import Settings
        from src.main import create_app

        settings = Settings(MAX_REQUEST_BODY=1024, MAX_FILE_SIZE=100)
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()

        @app.post("/probe")
        async def probe(request: Request):
            await request.body()
            return {"ok": True}

        response = TestClient(app).post(
            "/probe", content=iter([b"x" * 700, b"x" * 700]),
            headers={"Content-Length": "500"},
        )
        assert response.status_code == 413

    def test_blur_auth_precedes_multipart_parse(self):
        from src.config import Settings
        from src.main import create_app

        settings = Settings(DEBUG=False, MODEL_PROFILE="blur")
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()
        app.state.settings = settings
        response = TestClient(app).post(
            "/api/v1/blur/detect",
            content=b"broken multipart",
            headers={"Content-Type": "multipart/form-data; boundary=missing"},
        )
        assert response.status_code == 401

    def test_blur_key_rate_limited_once_before_parse(self):
        from src.config import Settings
        from src.main import create_app

        settings = Settings(DEBUG=False, MODEL_PROFILE="blur")
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()
        app.state.settings = settings
        app.state.redis = AsyncMock()
        app.state.redis.get.return_value = json.dumps({
            "key_id": "test", "scopes": ["blur:read"], "rate_tier": "internal",
        })
        with patch("src.middleware.auth._enforce_rate_limit", new_callable=AsyncMock) as limit:
            response = TestClient(app).post("/api/v1/blur/detect", headers={"X-API-Key": "test"})
        assert response.status_code == 422
        limit.assert_awaited_once()

    def test_blur_redis_outage_returns_503_then_recovers(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        from src.config import Settings
        from src.main import create_app

        settings = Settings(DEBUG=False, MODEL_PROFILE="blur", METRICS_ENABLED=False)
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()
        app.state.settings = settings
        redis = AsyncMock()
        app.state.redis = redis
        redis.get.side_effect = RedisConnectionError("down")
        client = TestClient(app)
        url = "/api/v1/blur/detect"
        headers = {"X-API-Key": "test"}

        response = client.post(url, headers=headers)
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "1"

        redis.get.side_effect = None
        redis.get.return_value = json.dumps({
            "key_id": "test", "scopes": ["blur:read"], "rate_tier": "internal"
        })
        redis.eval.side_effect = RedisConnectionError("down")
        assert client.post(url, headers=headers).status_code == 503

        redis.eval.side_effect = None
        redis.eval.return_value = [1, 999, 1]
        assert client.post(url, headers=headers).status_code == 422

    def test_blur_actual_multipart_bytes_rejected(self):
        from src.config import Settings
        from src.main import create_app

        settings = Settings(
            DEBUG=True, ENVIRONMENT="test", MODEL_PROFILE="blur", BLUR_MAX_REQUEST_BODY=1024,
            MAX_FILE_SIZE=100,
        )
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()
        app.state.settings = settings
        body = (
            b"--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.jpg\"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + b"x" * 1400 + b"\r\n--test--\r\n"
        )
        response = TestClient(app).post(
            "/api/v1/blur/detect", content=iter([body[:700], body[700:]]),
            headers={"Content-Type": "multipart/form-data; boundary=test"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"

    def test_nginx_client_max_body_size_matches_the_setting(self):
        """The ceiling lives in two files. This is what stops them drifting —
        the same failure mode that left MAX_REQUEST_BODY dead for months while
        nginx quietly enforced a different number.
        """
        import re
        from pathlib import Path

        from src.config import Settings

        conf = Path(__file__).resolve().parents[1] / "nginx.conf"
        # Leading `#` excludes the commented-out HTTPS block.
        found = re.findall(
            r"^\s*client_max_body_size\s+(\d+)([KMG]);",
            conf.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        assert found, "no active client_max_body_size directive in nginx.conf"
        units = {"K": 1024, "M": 1024**2, "G": 1024**3}
        for number, unit in found:
            assert int(number) * units[unit] == Settings().MAX_REQUEST_BODY

    def test_nginx_config_parses(self):
        """The value-matching test above passed for months against a file nginx
        could not load at all (`limit_req off;` — no such parameter). Regexing a
        directive proves nothing about the config being valid, so hand it to a
        real nginx.

        Fed on stdin rather than bind-mounted: mounting a Windows path into a
        container needs path translation this suite should not own. The
        --add-host is required — without a resolvable upstream nginx aborts on
        the `server ai-api:8000` line and never reaches the rest of the file.
        """
        import shutil
        import subprocess
        from pathlib import Path

        if shutil.which("docker") is None:
            pytest.skip("docker unavailable — cannot validate nginx.conf")

        conf = Path(__file__).resolve().parents[1] / "nginx.conf"
        result = subprocess.run(
            [
                "docker", "run", "--rm", "-i",
                "--add-host", "ai-api:127.0.0.1",
                "--entrypoint", "sh", "nginx:alpine",
                "-c", "cat > /etc/nginx/conf.d/default.conf && nginx -t",
            ],
            input=conf.read_bytes(),
            capture_output=True,
            timeout=180,
        )

        assert result.returncode == 0, result.stderr.decode(errors="replace")


async def test_blur_streams_share_decode_capacity_and_admission():
    import httpx

    from src.config import Settings
    from src.main import create_app

    settings = Settings(DEBUG=True, ENVIRONMENT="test", MODEL_PROFILE="blur", METRICS_ENABLED=False,
                        BLUR_DECODE_CONCURRENCY=2)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings

    class Classifier:
        min_detection_confidence = 0.5

        def classify_batch(self, images):
            return [{"predicted_class": "sharp", "confidence": 1.0,
                     "probabilities": {"sharp": 1.0}} for _ in images]

    class Registry:
        def get(self, name):
            return Classifier() if name == "blur_classifier" else None

    app.state.model_registry = Registry()
    lock = threading.Lock()
    active = 0
    peak = 0

    def decode(data, max_dim):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return np.zeros((32, 32, 3), dtype=np.uint8)

    image = _make_jpeg_bytes(width=32, height=32)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("src.api.v1.blur._decode_bgr_for_classify", side_effect=decode):
            responses = await asyncio.gather(*[
                client.post("/api/v1/blur/classify/stream", files=_files([image] * 4))
                for _ in range(3)
            ])
    assert sorted(r.status_code for r in responses) == [200, 200, 503]
    assert peak == 2
    assert all(len(r.text.splitlines()) == 5 for r in responses if r.status_code == 200)
    assert next(r for r in responses if r.status_code == 503).json()["error"]["code"] == "BLUR_BUSY"
    assert next(r for r in responses if r.status_code == 503).headers["Retry-After"] == "1"
