"""Tests for BlurClassifier and /blur/classify endpoint."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.ml.blur.classifier import BlurClassifier


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


def _make_mock_onnx_session(num_classes: int = 4):
    """Create a mock ONNX InferenceSession that returns fake logits."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 224, 224])]

    def mock_run(output_names, input_feed):
        # Return logits where class 3 (sharp) has highest score
        logits = np.array([[0.5, 0.1, 0.2, 2.0]], dtype=np.float32)
        return [logits]

    session.run = mock_run
    return session


def _make_mock_onnx_session_for_class(class_idx: int, num_classes: int = 4):
    """Create a mock ONNX session that predicts a specific class index."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 224, 224])]

    def mock_run(output_names, input_feed):
        logits = np.full((1, num_classes), 0.1, dtype=np.float32)
        logits[0, class_idx] = 2.0
        return [logits]

    session.run = mock_run
    return session


# ---------------------------------------------------------------------------
# Unit tests: BlurClassifier internals
# ---------------------------------------------------------------------------

class TestBlurClassifierInit:
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
        """Preprocessed output should be (1, 3, 224, 224)."""
        image = _make_bgr_image(width=640, height=480)
        result = classifier._preprocess(image)
        assert result.shape == (1, 3, 224, 224)

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


# ---------------------------------------------------------------------------
# Unit tests: detect_blur_type
# ---------------------------------------------------------------------------

class TestBlurClassifierDetectBlurType:
    @pytest.fixture
    def classifier(self, tmp_path):
        """Create a classifier with mock session predicting sharp (index 3)."""
        names = ["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]
        clf = BlurClassifier(
            model_path=str(tmp_path / "nonexistent.onnx"),
            class_names_path=str(tmp_path / "nonexistent.json"),
        )
        clf.session = _make_mock_onnx_session()  # predicts sharp (index 3)
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
        # Mock session that predicts defocused_object_portrait (index 1)
        clf.session = _make_mock_onnx_session_for_class(1)
        clf.class_names = names

        result = clf.detect_blur_type(_make_bgr_image(), "defocused_object_portrait")
        assert result is not None
        assert result["detected"] is True
        assert result["blur_type"] == "defocused_object_portrait"

    def test_not_detected_when_prediction_differs(self, classifier):
        """detect_blur_type() returns detected=False when prediction doesn't match."""
        # Default mock predicts sharp (index 3)
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
