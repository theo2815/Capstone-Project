"""Tests for BlurClassifier and /blur/classify endpoint."""
from __future__ import annotations

import hashlib
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image

from src.ml.blur.classifier import BlurClassifier
from src.utils.timeout import InferenceTimeoutError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a JPEG image in memory."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_bgr_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Create a random BGR image."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def _make_mock_onnx_session():
    """Create a mock ONNX InferenceSession returning fake stage-2 probabilities.

    The stage-2 model has two outputs in BlurClassifier.MODEL_CLASSES order:
    [defocused_object_portrait, sharp]. The exported YOLOv8-cls graph ends in
    Softmax, so the session already emits probabilities that sum to 1 —
    `_logits_to_result` deliberately does NOT re-apply softmax. The mock must
    match that, or confidence comes back > 1.0.
    """
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 640, 640])]

    def mock_run(output_names, input_feed):
        # sharp (index 1) is the argmax; sums to 1.0
        probs = np.array([[0.30, 0.70]], dtype=np.float32)
        return [probs]

    session.run = mock_run
    return session


def _make_mock_onnx_session_for_class(class_idx: int, num_classes: int = 2):
    """Create a mock ONNX session that predicts a specific class index.

    Emits probabilities (post-Softmax), matching the real graph — see
    _make_mock_onnx_session.
    """
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 640, 640])]

    def mock_run(output_names, input_feed):
        remainder = 0.3 / (num_classes - 1)
        probs = np.full((1, num_classes), remainder, dtype=np.float32)
        probs[0, class_idx] = 0.7
        return [probs]

    session.run = mock_run
    return session


# ---------------------------------------------------------------------------
# Unit tests: BlurClassifier internals
# ---------------------------------------------------------------------------

class TestBlurClassifierInit:
    def test_verified_artifact_rejects_changed_labels_before_loading_onnx(self, tmp_path):
        model = tmp_path / "blur_classifier.onnx"
        labels = tmp_path / "class_names.json"
        manifest = tmp_path / "manifest.json"
        model.write_bytes(b"audited model")
        labels.write_text(json.dumps(["sharp", "defocused_object_portrait"]))  # wrong order
        manifest.write_text(json.dumps({"models": {"blur_classifier": {
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "classes": BlurClassifier.MODEL_CLASSES,
        }}}))
        classifier = BlurClassifier(
            model_path=str(model), class_names_path=str(labels),
            manifest_path=str(manifest),
        )
        assert classifier.session is None

    def test_verified_artifact_requires_both_files(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"models": {"blur_classifier": {
            "sha256": "0" * 64, "labels_sha256": "0" * 64,
            "classes": BlurClassifier.MODEL_CLASSES,
        }}}))
        classifier = BlurClassifier(
            model_path=str(tmp_path / "missing.onnx"),
            class_names_path=str(tmp_path / "missing.json"),
            manifest_path=str(manifest),
        )
        assert classifier.session is None

    @pytest.mark.parametrize("probs", [
        [[float("nan"), 1.0]],
        [[0.1, 0.1]],
        [[1.0, 0.0, 0.0]],
    ])
    def test_verified_artifact_rejects_bad_warm_inference(self, tmp_path, monkeypatch, probs):
        import onnxruntime

        model = tmp_path / "blur_classifier.onnx"
        labels = tmp_path / "class_names.json"
        manifest = tmp_path / "manifest.json"
        model.write_bytes(b"model")
        labels.write_text(json.dumps(BlurClassifier.MODEL_CLASSES))
        manifest.write_text(json.dumps({"models": {"blur_classifier": {
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "classes": BlurClassifier.MODEL_CLASSES,
        }}}))

        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="images", shape=["batch", 3, 640, 640])]

            def get_outputs(self):
                return [SimpleNamespace(shape=["batch", 2])]

            def run(self, *_args):
                return [np.array(probs, dtype=np.float32)]

        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_args, **_kwargs: Session())
        classifier = BlurClassifier(
            model_path=str(model), class_names_path=str(labels),
            manifest_path=str(manifest),
        )
        assert classifier.session is None

    def test_no_model_file(self, tmp_path):
        """Classifier gracefully handles missing model file."""
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        assert clf.session is None

    def test_with_class_names_file(self, tmp_path):
        """Classifier loads class names from JSON file."""
        names = ["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]
        names_file = tmp_path / "class_names.json"
        names_file.write_text(json.dumps(names))

        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(names_file),
        )
        # Session will be None (no ONNX file), but class names should load
        assert clf.session is None

    def test_default_class_names(self, tmp_path):
        """Classifier uses default class names when file missing."""
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        assert len(clf.class_names) == 4
        assert "sharp" in clf.class_names
        assert "defocused_object_portrait" in clf.class_names


class TestBlurClassifierClassify:
    @pytest.fixture
    def classifier(self, tmp_path):
        """Create a classifier with a mock ONNX session."""
        names = ["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]
        names_file = tmp_path / "class_names.json"
        names_file.write_text(json.dumps(names))

        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(names_file),
        )
        # Inject mock session
        clf.session = _make_mock_onnx_session()
        clf.class_names = names
        return clf

    def test_returns_none_without_session(self, tmp_path):
        """classify() returns None when session is not loaded."""
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        assert clf.classify(_make_bgr_image()) is None

    def test_single_native_call_has_a_deadline(self, classifier, monkeypatch):
        from src import config
        from src.utils.timeout import start_timeout_executor

        start_timeout_executor()
        release = threading.Event()
        completed = threading.Event()
        monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(
            MODEL_PROFILE="blur", INFERENCE_TIMEOUT=0.02, INFERENCE_BATCH_TIMEOUT=0.02,
            BLUR_INFERENCE_TIMEOUT_SECONDS=0.02,
        ))
        def slow_run(*_args):
            release.wait(1)
            completed.set()
            return [np.array([[0.3, 0.7]], dtype=np.float32)]

        classifier.session.run = slow_run
        try:
            with pytest.raises(InferenceTimeoutError):
                classifier.classify(_make_bgr_image())
        finally:
            release.set()
            assert completed.wait(1)

    def test_static_batch_native_calls_have_a_deadline(self, classifier, monkeypatch):
        from src import config
        from src.utils.timeout import start_timeout_executor

        start_timeout_executor()
        release = threading.Event()
        completed = threading.Event()
        monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(
            MODEL_PROFILE="blur", INFERENCE_TIMEOUT=0.02, INFERENCE_BATCH_TIMEOUT=0.02,
            BLUR_INFERENCE_TIMEOUT_SECONDS=0.02,
        ))
        classifier._dynamic_batch = False
        def slow_run(*_args):
            release.wait(1)
            completed.set()
            return [np.array([[0.3, 0.7]], dtype=np.float32)]

        classifier.session.run = slow_run
        try:
            with pytest.raises(InferenceTimeoutError):
                classifier.classify_batch([_make_bgr_image()])
        finally:
            release.set()
            assert completed.wait(1)

    def test_returns_dict_with_mock_session(self, classifier):
        """classify() returns proper dict with mock session."""
        result = classifier.classify(_make_bgr_image())
        assert result is not None
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predicted_class_is_string(self, classifier):
        result = classifier.classify(_make_bgr_image())
        assert isinstance(result["predicted_class"], str)
        assert result["predicted_class"] in classifier.class_names

    def test_confidence_range(self, classifier):
        result = classifier.classify(_make_bgr_image())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, classifier):
        result = classifier.classify(_make_bgr_image())
        probs = result["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_all_non_negative(self, classifier):
        result = classifier.classify(_make_bgr_image())
        for name, prob in result["probabilities"].items():
            assert prob >= 0.0, f"Probability for {name} is negative: {prob}"

    def test_probabilities_contain_all_classes(self, classifier):
        result = classifier.classify(_make_bgr_image())
        for name in classifier.class_names:
            assert name in result["probabilities"]

    def test_predicted_class_has_highest_probability(self, classifier):
        result = classifier.classify(_make_bgr_image())
        predicted = result["predicted_class"]
        predicted_prob = result["probabilities"][predicted]
        for name, prob in result["probabilities"].items():
            assert predicted_prob >= prob, (
                f"Predicted class '{predicted}' ({predicted_prob}) is not "
                f"the highest probability, '{name}' has {prob}"
            )

    def test_different_image_sizes(self, classifier):
        """Classifier handles various image sizes."""
        for w, h in [(64, 64), (128, 256), (640, 480)]:
            result = classifier.classify(_make_bgr_image(width=w, height=h))
            assert result is not None
            assert "predicted_class" in result


class TestBlurClassifierPreprocess:
    @pytest.fixture
    def classifier(self, tmp_path):
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        return clf

    def test_output_shape(self, classifier):
        """Preprocessed output should be (1, 3, 640, 640)."""
        image = _make_bgr_image(width=640, height=480)
        result = classifier._preprocess(image)
        assert result.shape == (1, 3, 640, 640)

    def test_whole_frame_is_kept_and_gray(self, classifier):
        """No centre crop: both ends of an upright frame survive, as three equal channels."""
        upright = np.zeros((1280, 853, 3), dtype=np.uint8)
        upright[:40] = 255      # top edge
        upright[-40:] = 255     # bottom edge: a centre crop would lose both
        result = classifier._preprocess(upright)[0]
        assert result[0, :15].max() == 1.0
        assert result[0, -15:].max() == 1.0
        assert np.array_equal(result[0], result[1])
        assert np.array_equal(result[1], result[2])

    def test_output_dtype(self, classifier):
        """Preprocessed output should be float32."""
        image = _make_bgr_image()
        result = classifier._preprocess(image)
        assert result.dtype == np.float32

    def test_output_range(self, classifier):
        """Preprocessed values should be in [0, 1]."""
        image = _make_bgr_image()
        result = classifier._preprocess(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_downscale_is_anti_aliased(self, classifier):
        """The resize to 640 must area-average, not point-sample.

        This is an accuracy property, not a style preference. A non-anti-aliased
        downscale undersamples high-frequency detail and manufactures fake sharp
        edges, which makes blurry photos read as `sharp` — the one failure the
        cull must not make. Measured on the 1057-image labelled val set, fixing
        it cut blurry-called-sharp from 10 to 4 (see the 2026-08-14 ADR).

        Period-3 stripes are pure high frequency at this scale. Area-averaging
        collapses them toward their mean; bilinear keeps sampling individual
        stripes and the output swings with the sampling phase. The mean survives
        either way — only the variance separates them (measured for 1500 -> 640:
        0.109 area vs 0.333 bilinear), so this asserts on variance.
        """
        stripes = np.zeros((1500, 1500, 3), dtype=np.uint8)
        stripes[:, ::3] = 255

        result = classifier._preprocess(stripes)

        assert result.std() < 0.15, (
            f"std={result.std():.4f} — the 640 resize is point-sampling instead "
            "of area-averaging, so high-frequency detail is aliasing through"
        )
        # Brightness is preserved; only the aliasing is gone.
        assert result.mean() == pytest.approx(1 / 3, abs=0.02)


# ---------------------------------------------------------------------------
# Unit tests: detect_blur_type
# ---------------------------------------------------------------------------

class TestBlurClassifierDetectBlurType:
    @pytest.fixture
    def classifier(self, tmp_path):
        """Create a classifier with mock session predicting sharp (index 1)."""
        names = ["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        clf.session = _make_mock_onnx_session()  # predicts sharp (index 1)
        clf.class_names = names
        return clf

    def test_returns_none_without_session(self, tmp_path):
        """detect_blur_type() returns None when session is not loaded."""
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        assert clf.detect_blur_type(_make_bgr_image(), "motion_blurred") is None

    def test_detected_when_prediction_matches(self, tmp_path):
        """detect_blur_type() returns detected=True when prediction matches."""
        names = ["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        # Mock session that predicts defocused_object_portrait (index 0)
        clf.session = _make_mock_onnx_session_for_class(0)
        clf.class_names = names

        result = clf.detect_blur_type(_make_bgr_image(), "defocused_object_portrait")
        assert result is not None
        assert result["detected"] is True
        assert result["blur_type"] == "defocused_object_portrait"

    def test_not_detected_when_prediction_differs(self, classifier):
        """detect_blur_type() returns detected=False when prediction doesn't match."""
        # Default mock predicts sharp (index 1)
        result = classifier.detect_blur_type(_make_bgr_image(), "defocused_object_portrait")
        assert result is not None
        assert result["detected"] is False
        assert result["blur_type"] == "defocused_object_portrait"
        assert result["predicted_class"] == "sharp"

    def test_response_contains_required_fields(self, classifier):
        """detect_blur_type() returns all required fields."""
        result = classifier.detect_blur_type(_make_bgr_image(), "motion_blurred")
        assert result is not None
        assert "detected" in result
        assert "confidence" in result
        assert "blur_type" in result
        assert "blur_type_probability" in result
        assert "predicted_class" in result
        assert "probabilities" in result

    def test_blur_type_probability_matches_probabilities(self, classifier):
        """blur_type_probability should match the value in probabilities dict."""
        blur_type = "motion_blurred"
        result = classifier.detect_blur_type(_make_bgr_image(), blur_type)
        assert result is not None
        assert abs(result["blur_type_probability"] - result["probabilities"][blur_type]) < 1e-6

    def test_all_blur_types(self, classifier):
        """detect_blur_type() works for all 3 blur types."""
        for bt in ["defocused_object_portrait", "defocused_blurred", "motion_blurred"]:
            result = classifier.detect_blur_type(_make_bgr_image(), bt)
            assert result is not None
            assert result["blur_type"] == bt
            assert isinstance(result["detected"], bool)


# ---------------------------------------------------------------------------
# Unit tests: the two stages
# ---------------------------------------------------------------------------

def _counting_session(probs=(0.30, 0.70)):
    """Mock stage-2 session that records the shape of every batch it is given."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="images", shape=["batch", 3, 640, 640])]
    session.calls = []

    def run(output_names, input_feed):
        batch = next(iter(input_feed.values()))
        session.calls.append(batch.shape)
        return [np.tile(np.array([probs], dtype=np.float32), (batch.shape[0], 1))]

    session.run = run
    return session


def _soft_frame() -> np.ndarray:
    """Nothing sharp anywhere, no direction: whole-frame soft."""
    return cv2.GaussianBlur(_make_bgr_image(width=600, height=900), (0, 0), 8)


def _smeared_frame() -> np.ndarray:
    """Nothing sharp anywhere, smeared left-right: motion."""
    return cv2.blur(cv2.GaussianBlur(_make_bgr_image(width=600, height=900), (0, 0), 3), (121, 1))


class TestBlurClassifierTwoStage:
    @pytest.fixture
    def classifier(self, tmp_path):
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
            soft_max_detail=34.1, motion_min_ratio=2.63, motion_max_detail=52.4,
        )
        clf.session = _counting_session()
        clf._dynamic_batch = True
        return clf

    def test_thresholds_default_to_the_settings(self, tmp_path):
        from src.config import get_settings

        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        settings = get_settings()
        assert clf.soft_max_detail == settings.BLUR_SOFT_MAX_DETAIL
        assert clf.motion_min_ratio == settings.BLUR_MOTION_MIN_RATIO
        assert clf.motion_max_detail == settings.BLUR_MOTION_MAX_DETAIL

    def test_stage_two_output_fills_the_four_class_result(self, classifier):
        result = classifier.classify(_make_bgr_image())
        assert result["predicted_class"] == "sharp"
        assert result["confidence"] == pytest.approx(0.70)
        assert result["probabilities"] == pytest.approx({
            "defocused_blurred": 0.0, "defocused_object_portrait": 0.30,
            "motion_blurred": 0.0, "sharp": 0.70,
        })

    def test_whole_frame_soft_never_reaches_the_model(self, classifier):
        result = classifier.classify(_soft_frame())
        assert result == {
            "predicted_class": "defocused_blurred", "confidence": 1.0,
            "probabilities": {"defocused_blurred": 1.0, "defocused_object_portrait": 0.0,
                              "motion_blurred": 0.0, "sharp": 0.0},
        }
        assert classifier.session.calls == []

    def test_sideways_smear_is_motion_and_never_reaches_the_model(self, classifier):
        result = classifier.classify(_smeared_frame())
        assert result["predicted_class"] == "motion_blurred"
        assert result["confidence"] == 1.0
        assert classifier.session.calls == []

    def test_batch_mixes_both_stages_and_keeps_positions(self, classifier):
        images = [_make_bgr_image(), _soft_frame(), None, _smeared_frame(), _make_bgr_image()]
        results = classifier.classify_batch(images)
        assert [r and r["predicted_class"] for r in results] == [
            "sharp", "defocused_blurred", None, "motion_blurred", "sharp"]
        assert classifier.session.calls == [(2, 3, 640, 640)]

    def test_batch_feeds_the_model_in_small_chunks(self, classifier):
        """A 50-photo sub-batch must not become one ~250 MB tensor."""
        results = classifier.classify_batch([_make_bgr_image() for _ in range(20)])
        assert all(r["predicted_class"] == "sharp" for r in results)
        assert classifier.session.calls == [(8, 3, 640, 640), (8, 3, 640, 640), (4, 3, 640, 640)]

    def test_batch_of_only_whole_frame_blur_skips_the_model(self, classifier):
        results = classifier.classify_batch([_soft_frame(), _smeared_frame()])
        assert [r["predicted_class"] for r in results] == ["defocused_blurred", "motion_blurred"]
        assert classifier.session.calls == []

    def test_targeted_detection_sees_whole_frame_blur(self, classifier):
        result = classifier.detect_blur_type(_soft_frame(), "defocused_blurred")
        assert result["detected"] is True
        assert result["blur_type_probability"] == 1.0

    def test_a_four_class_model_from_before_the_two_stage_design_is_refused(
        self, tmp_path, monkeypatch,
    ):
        import onnxruntime

        model = tmp_path / "blur_classifier.onnx"
        model.write_bytes(b"old four-class model")
        labels = tmp_path / "class_names.json"
        labels.write_text(json.dumps(BlurClassifier.CLASSES))

        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="images", shape=["batch", 3, 224, 224])]

            def get_outputs(self):
                return [SimpleNamespace(shape=["batch", 4])]

        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_args, **_kwargs: Session())
        clf = BlurClassifier(model_path=str(model), class_names_path=str(labels))
        assert clf.session is None


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestBlurClassificationSchemas:
    def test_blur_class_probabilities(self):
        from src.schemas.blur import BlurClassProbabilities

        probs = BlurClassProbabilities(
            sharp=0.8,
            defocused_object_portrait=0.1,
            defocused_blurred=0.05,
            motion_blurred=0.05,
        )
        assert probs.sharp == 0.8
        assert probs.defocused_object_portrait == 0.1

    def test_blur_classification_response(self):
        from src.schemas.blur import BlurClassificationResponse, BlurClassProbabilities

        probs = BlurClassProbabilities(
            sharp=0.8,
            defocused_object_portrait=0.1,
            defocused_blurred=0.05,
            motion_blurred=0.05,
        )
        response = BlurClassificationResponse(
            predicted_class="sharp",
            confidence=0.8,
            probabilities=probs,
            image_dimensions=(640, 480),
            processing_time_ms=12.34,
        )
        assert response.predicted_class == "sharp"
        assert response.confidence == 0.8
        assert response.image_dimensions == (640, 480)

    def test_serialization_roundtrip(self):
        from src.schemas.blur import BlurClassificationResponse, BlurClassProbabilities

        probs = BlurClassProbabilities(
            sharp=0.8,
            defocused_object_portrait=0.1,
            defocused_blurred=0.05,
            motion_blurred=0.05,
        )
        response = BlurClassificationResponse(
            predicted_class="sharp",
            confidence=0.8,
            probabilities=probs,
            image_dimensions=(640, 480),
            processing_time_ms=12.34,
        )
        data = response.model_dump(mode="json")
        restored = BlurClassificationResponse(**data)
        assert restored.predicted_class == "sharp"
        assert restored.probabilities.sharp == 0.8

    def test_blur_type_enum(self):
        from src.schemas.blur import BlurType

        assert BlurType.DEFOCUSED_OBJECT_PORTRAIT.value == "defocused_object_portrait"
        assert BlurType.DEFOCUSED_BLURRED.value == "defocused_blurred"
        assert BlurType.MOTION_BLURRED.value == "motion_blurred"
        # sharp should not be in the enum
        assert not hasattr(BlurType, "SHARP")

    def test_blur_type_detection_response(self):
        from src.schemas.blur import BlurTypeDetectionResponse

        response = BlurTypeDetectionResponse(
            blur_type="defocused_object_portrait",
            detected=True,
            confidence=0.92,
            blur_type_probability=0.92,
            image_dimensions=(640, 480),
            processing_time_ms=8.5,
        )
        assert response.detected is True
        assert response.blur_type == "defocused_object_portrait"
        assert response.confidence == 0.92

    def test_blur_type_detection_response_not_detected(self):
        from src.schemas.blur import BlurTypeDetectionResponse

        response = BlurTypeDetectionResponse(
            blur_type="motion_blurred",
            detected=False,
            confidence=0.85,
            blur_type_probability=0.05,
            image_dimensions=(1920, 1080),
            processing_time_ms=10.2,
        )
        assert response.detected is False
        assert response.blur_type == "motion_blurred"
        assert response.blur_type_probability == 0.05


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------

class TestBlurServiceClassify:
    def test_classify_without_classifier(self):
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        detector = BlurDetector()
        service = BlurService(detector=detector, classifier=None)
        result = service.classify(_make_bgr_image())
        assert result is None

    def test_classify_with_mock_classifier(self, tmp_path):
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        detector = BlurDetector()

        names = ["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]
        names_file = tmp_path / "class_names.json"
        names_file.write_text(json.dumps(names))

        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(names_file),
        )
        clf.session = _make_mock_onnx_session()
        clf.class_names = names

        service = BlurService(detector=detector, classifier=clf)
        result = service.classify(_make_bgr_image())
        assert result is not None
        assert "predicted_class" in result

    def test_detect_still_works_with_classifier(self):
        """Adding classifier doesn't break existing detect()."""
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        detector = BlurDetector()
        service = BlurService(detector=detector, classifier=None)
        result = service.detect(_make_bgr_image())
        assert "is_blurry" in result
        assert "laplacian_variance" in result

    def test_detect_blur_type_without_classifier(self):
        """detect_blur_type() returns None without classifier."""
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        detector = BlurDetector()
        service = BlurService(detector=detector, classifier=None)
        result = service.detect_blur_type(_make_bgr_image(), "defocused_blurred")
        assert result is None

    def test_detect_blur_type_with_mock_classifier(self, tmp_path):
        """detect_blur_type() works with mock classifier."""
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        detector = BlurDetector()

        names = ["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        clf.session = _make_mock_onnx_session()
        clf.class_names = names

        service = BlurService(detector=detector, classifier=clf)
        result = service.detect_blur_type(_make_bgr_image(), "motion_blurred")
        assert result is not None
        assert "detected" in result
        assert isinstance(result["detected"], bool)
