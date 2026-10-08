"""Tests for BlurClassifier (two stages: whole-frame rule, then people) and /blur/classify."""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from src.ml.blur import people
from src.ml.blur.classifier import BlurClassifier
from src.ml.blur.person_detector import Detection
from src.utils.timeout import InferenceBusyError, InferenceTimeoutError

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


def _rule_spec(p_missed: float = 0.3, names=None) -> dict:
    """A rule whose answer is always p_missed: zero weights, the intercept carries the odds."""
    names = list(people.FEATURE_NAMES if names is None else names)
    return {
        "feature_names": names,
        "mean": [0.0] * len(names),
        "scale": [1.0] * len(names),
        "coef": [0.0] * len(names),
        "intercept": math.log(p_missed / (1 - p_missed)),
        "min_conf": 0.2,
        "big_area": 0.03,
        "soft_ratio": 0.5,
    }


class FakeDetector:
    """Returns the detections it was given (by default one person filling the frame) and
    counts how often it was asked."""

    def __init__(self, detections=None):
        self.detections = detections
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.detections is not None:
            return list(self.detections)
        h, w = image.shape[:2]
        return [
            Detection(
                box=(0.0, 0.0, float(w), float(h)), conf=0.9, mask=np.ones((h, w), dtype=bool)
            )
        ]


def _arm(clf: BlurClassifier, p_missed: float = 0.3, detections=None) -> BlurClassifier:
    """Give a classifier a working stage 2 without any model file."""
    clf.session = object()
    clf.detector = FakeDetector(detections)
    clf.rule = people.LogisticRule(_rule_spec(p_missed))
    return clf


def _unarmed(tmp_path, **kwargs) -> BlurClassifier:
    return BlurClassifier(
        detector_path=str(tmp_path / "missing.onnx"),
        rule_path=str(tmp_path / "missing.json"),
        **kwargs,
    )


class _DetectorSession:
    """What a real person-detector session looks like to the loader."""

    def __init__(
        self, input_shape=(1, 3, 640, 640), output_shapes=((1, 116, 8400), (1, 32, 160, 160))
    ):
        self._in, self._outs = list(input_shape), [list(s) for s in output_shapes]

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self._in)]

    def get_outputs(self):
        return [SimpleNamespace(shape=s) for s in self._outs]

    def run(self, *_args):
        # no person anywhere
        return [
            np.zeros((1, 116, 8400), dtype=np.float32),
            np.zeros((1, 32, 160, 160), dtype=np.float32),
        ]


def _files(tmp_path, rule_spec=None, detector_bytes=b"detector", classes=None):
    """Detector + rule + a manifest whose hashes match them; returns the three paths."""
    detector = tmp_path / "yolov8n_person.onnx"
    detector.write_bytes(detector_bytes)
    rule = tmp_path / "people_rule.json"
    rule.write_text(json.dumps(_rule_spec() if rule_spec is None else rule_spec))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": {
                    "person_detector": {
                        "sha256": hashlib.sha256(detector.read_bytes()).hexdigest()
                    },
                    "blur_classifier": {
                        "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
                        "classes": BlurClassifier.MODEL_CLASSES if classes is None else classes,
                    },
                }
            }
        )
    )
    return detector, rule, manifest


# ---------------------------------------------------------------------------
# Unit tests: loading and the manifest
# ---------------------------------------------------------------------------


class TestBlurClassifierInit:
    def test_no_model_files(self, tmp_path):
        """Classifier gracefully handles missing files."""
        clf = _unarmed(tmp_path)
        assert clf.session is None and clf.detector is None and clf.rule is None

    def test_default_class_names(self, tmp_path):
        clf = _unarmed(tmp_path)
        assert clf.class_names == BlurClassifier.CLASSES

    def test_verified_artifacts_load(self, tmp_path, monkeypatch):
        import onnxruntime

        detector, rule, manifest = _files(tmp_path)
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: _DetectorSession())
        clf = BlurClassifier(
            detector_path=str(detector), rule_path=str(rule), manifest_path=str(manifest)
        )
        assert clf.session is not None
        assert clf.rule.feature_names == list(people.FEATURE_NAMES)
        assert clf.detector.min_conf == 0.2

    def test_verified_artifact_rejects_a_changed_rule(self, tmp_path, monkeypatch):
        import onnxruntime

        detector, rule, manifest = _files(tmp_path)
        rule.write_text(json.dumps(_rule_spec(0.9)))  # edited after the manifest was written
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: _DetectorSession())
        clf = BlurClassifier(
            detector_path=str(detector), rule_path=str(rule), manifest_path=str(manifest)
        )
        assert clf.session is None

    def test_verified_artifact_rejects_a_changed_detector(self, tmp_path, monkeypatch):
        import onnxruntime

        detector, rule, manifest = _files(tmp_path)
        detector.write_bytes(b"another detector")
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: _DetectorSession())
        clf = BlurClassifier(
            detector_path=str(detector), rule_path=str(rule), manifest_path=str(manifest)
        )
        assert clf.session is None

    def test_verified_artifact_rejects_other_classes(self, tmp_path, monkeypatch):
        import onnxruntime

        detector, rule, manifest = _files(tmp_path, classes=BlurClassifier.CLASSES)
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: _DetectorSession())
        clf = BlurClassifier(
            detector_path=str(detector), rule_path=str(rule), manifest_path=str(manifest)
        )
        assert clf.session is None

    def test_verified_artifact_requires_both_files(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "models": {
                        "person_detector": {"sha256": "0" * 64},
                        "blur_classifier": {
                            "sha256": "0" * 64,
                            "classes": BlurClassifier.MODEL_CLASSES,
                        },
                    }
                }
            )
        )
        clf = _unarmed(tmp_path, manifest_path=str(manifest))
        assert clf.session is None

    def test_a_rule_fitted_on_other_features_is_refused(self, tmp_path, monkeypatch):
        import onnxruntime

        detector, rule, manifest = _files(tmp_path, rule_spec=_rule_spec(names=["something_else"]))
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: _DetectorSession())
        clf = BlurClassifier(
            detector_path=str(detector), rule_path=str(rule), manifest_path=str(manifest)
        )
        assert clf.session is None

    @pytest.mark.parametrize(
        ("input_shape", "output_shapes"),
        [
            (("batch", 3, 224, 224), (("batch", 4),)),  # the 2026-08-14 four-class model
            (("batch", 3, 640, 640), (("batch", 2),)),  # the 2026-10 two-class model
            ((1, 3, 640, 640), ((1, 84, 8400),)),  # the box-only detector: no masks
            ((1, 3, 640, 640), ((1, 116, 8400),)),  # mask prototypes missing
        ],
    )
    def test_a_file_that_is_not_the_person_detector_is_refused(
        self, tmp_path, monkeypatch, input_shape, output_shapes
    ):
        import onnxruntime

        detector, rule, _ = _files(tmp_path)
        monkeypatch.setattr(
            onnxruntime,
            "InferenceSession",
            lambda *_a, **_k: _DetectorSession(input_shape, output_shapes),
        )
        clf = BlurClassifier(detector_path=str(detector), rule_path=str(rule))
        assert clf.session is None

    def test_warm_run_failure_is_refused(self, tmp_path, monkeypatch):
        import onnxruntime

        spec = _rule_spec()
        spec["intercept"] = float("nan")
        detector, rule, manifest = _files(tmp_path, rule_spec=spec)
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: _DetectorSession())
        clf = BlurClassifier(
            detector_path=str(detector), rule_path=str(rule), manifest_path=str(manifest)
        )
        assert clf.session is None


# ---------------------------------------------------------------------------
# Unit tests: classify, the four-class result, deadlines
# ---------------------------------------------------------------------------


class TestBlurClassifierClassify:
    @pytest.fixture
    def classifier(self, tmp_path):
        return _arm(_unarmed(tmp_path), p_missed=0.3)

    def test_returns_none_without_session(self, tmp_path):
        assert _unarmed(tmp_path).classify(_make_bgr_image()) is None

    def test_single_native_call_has_a_deadline(self, classifier, monkeypatch):
        from src import config
        from src.utils.timeout import start_timeout_executor

        start_timeout_executor()
        release = threading.Event()
        completed = threading.Event()
        monkeypatch.setattr(
            config,
            "get_settings",
            lambda: SimpleNamespace(
                MODEL_PROFILE="blur",
                INFERENCE_TIMEOUT=0.02,
                INFERENCE_BATCH_TIMEOUT=0.02,
                BLUR_INFERENCE_TIMEOUT_SECONDS=0.02,
            ),
        )

        def slow_detect(image):
            release.wait(1)
            completed.set()
            return []

        classifier.detector.detect = slow_detect
        try:
            with pytest.raises(InferenceTimeoutError):
                classifier.classify(_make_bgr_image())
        finally:
            release.set()
            assert completed.wait(1)

    def test_batch_native_calls_have_a_deadline(self, classifier, monkeypatch):
        from src import config
        from src.utils.timeout import start_timeout_executor

        start_timeout_executor()
        release = threading.Event()
        completed = threading.Event()
        monkeypatch.setattr(
            config,
            "get_settings",
            lambda: SimpleNamespace(
                MODEL_PROFILE="blur",
                INFERENCE_TIMEOUT=0.02,
                INFERENCE_BATCH_TIMEOUT=0.02,
                BLUR_INFERENCE_TIMEOUT_SECONDS=0.02,
            ),
        )

        def slow_detect(image):
            release.wait(1)
            completed.set()
            return []

        classifier.detector.detect = slow_detect
        try:
            with pytest.raises(InferenceTimeoutError):
                classifier.classify_batch([_make_bgr_image()])
        finally:
            release.set()
            assert completed.wait(1)

    def test_batch_passes_on_server_busy(self, classifier):
        def busy_detect(image):
            raise InferenceBusyError

        classifier.detector.detect = busy_detect
        with pytest.raises(InferenceBusyError):
            classifier.classify_batch([_make_bgr_image()])

    def test_returns_dict_with_stage_two(self, classifier):
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
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5

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
        for prob in result["probabilities"].values():
            assert predicted_prob >= prob

    def test_different_image_sizes(self, classifier):
        for w, h in [(64, 64), (128, 256), (640, 480)]:
            result = classifier.classify(_make_bgr_image(width=w, height=h))
            assert result is not None
            assert "predicted_class" in result


# ---------------------------------------------------------------------------
# Unit tests: detect_blur_type
# ---------------------------------------------------------------------------


class TestBlurClassifierDetectBlurType:
    @pytest.fixture
    def classifier(self, tmp_path):
        return _arm(_unarmed(tmp_path), p_missed=0.3)  # predicts sharp

    def test_returns_none_without_session(self, tmp_path):
        assert _unarmed(tmp_path).detect_blur_type(_make_bgr_image(), "motion_blurred") is None

    def test_detected_when_prediction_matches(self, tmp_path):
        clf = _arm(_unarmed(tmp_path), p_missed=0.7)
        result = clf.detect_blur_type(_make_bgr_image(), "defocused_object_portrait")
        assert result is not None
        assert result["detected"] is True
        assert result["blur_type"] == "defocused_object_portrait"

    def test_not_detected_when_prediction_differs(self, classifier):
        result = classifier.detect_blur_type(_make_bgr_image(), "defocused_object_portrait")
        assert result is not None
        assert result["detected"] is False
        assert result["blur_type"] == "defocused_object_portrait"
        assert result["predicted_class"] == "sharp"

    def test_response_contains_required_fields(self, classifier):
        result = classifier.detect_blur_type(_make_bgr_image(), "motion_blurred")
        assert result is not None
        for key in (
            "detected",
            "confidence",
            "blur_type",
            "blur_type_probability",
            "predicted_class",
            "probabilities",
        ):
            assert key in result

    def test_blur_type_probability_matches_probabilities(self, classifier):
        result = classifier.detect_blur_type(_make_bgr_image(), "motion_blurred")
        assert (
            abs(result["blur_type_probability"] - result["probabilities"]["motion_blurred"]) < 1e-6
        )

    def test_all_blur_types(self, classifier):
        for bt in ["defocused_object_portrait", "defocused_blurred", "motion_blurred"]:
            result = classifier.detect_blur_type(_make_bgr_image(), bt)
            assert result is not None
            assert result["blur_type"] == bt
            assert isinstance(result["detected"], bool)


# ---------------------------------------------------------------------------
# Unit tests: the two stages
# ---------------------------------------------------------------------------


def _soft_frame() -> np.ndarray:
    """Nothing sharp anywhere, no direction: whole-frame soft."""
    return cv2.GaussianBlur(_make_bgr_image(width=600, height=900), (0, 0), 8)


def _smeared_frame() -> np.ndarray:
    """Nothing sharp anywhere, smeared left-right: motion."""
    return cv2.blur(cv2.GaussianBlur(_make_bgr_image(width=600, height=900), (0, 0), 3), (121, 1))


class TestBlurClassifierTwoStage:
    @pytest.fixture
    def classifier(self, tmp_path):
        clf = _unarmed(
            tmp_path, soft_max_detail=34.1, motion_min_ratio=2.63, motion_max_detail=52.4
        )
        return _arm(clf, p_missed=0.3)

    def test_thresholds_default_to_the_settings(self, tmp_path):
        from src.config import get_settings

        clf = _unarmed(tmp_path)
        settings = get_settings()
        assert clf.soft_max_detail == settings.BLUR_SOFT_MAX_DETAIL
        assert clf.motion_min_ratio == settings.BLUR_MOTION_MIN_RATIO
        assert clf.motion_max_detail == settings.BLUR_MOTION_MAX_DETAIL

    @pytest.mark.parametrize(
        ("limits", "soft", "smear"),
        [
            ((1.0, 2.63, 52.4), "sharp", "motion_blurred"),  # soft limit below the soft frame
            (
                (34.1, 100.0, 52.4),
                "defocused_blurred",
                "defocused_blurred",
            ),  # ratio above the smear
            ((34.1, 2.63, 1.0), "defocused_blurred", "sharp"),  # motion limit below the smear
        ],
    )
    def test_each_limit_is_used_for_its_own_blur_type(self, tmp_path, limits, soft, smear):
        """_soft_frame has detail ~1.9; _smeared_frame has detail ~1.25 and ratio ~13."""
        clf = _arm(
            _unarmed(
                tmp_path,
                soft_max_detail=limits[0],
                motion_min_ratio=limits[1],
                motion_max_detail=limits[2],
            )
        )
        assert clf.classify(_soft_frame())["predicted_class"] == soft
        assert clf.classify(_smeared_frame())["predicted_class"] == smear

    def test_stage_two_output_fills_the_four_class_result(self, classifier):
        result = classifier.classify(_make_bgr_image())
        assert result["predicted_class"] == "sharp"
        assert result["confidence"] == pytest.approx(0.70)
        assert result["probabilities"] == pytest.approx(
            {
                "defocused_blurred": 0.0,
                "defocused_object_portrait": 0.30,
                "motion_blurred": 0.0,
                "sharp": 0.70,
            }
        )
        assert classifier.detector.calls == 1

    def test_a_high_probability_names_missed_focus(self, tmp_path):
        clf = _arm(_unarmed(tmp_path), p_missed=0.8)
        result = clf.classify(_make_bgr_image())
        assert result["predicted_class"] == "defocused_object_portrait"
        assert result["confidence"] == pytest.approx(0.8)

    def test_nobody_found_is_sharp_whatever_the_rule_says(self, tmp_path):
        clf = _arm(_unarmed(tmp_path), p_missed=0.9, detections=())
        result = clf.classify(_make_bgr_image())
        assert result["predicted_class"] == "sharp" and result["confidence"] == 1.0
        assert result["probabilities"]["defocused_object_portrait"] == 0.0
        assert clf.detector.calls == 1

    def test_whole_frame_soft_never_reaches_the_detector(self, classifier):
        result = classifier.classify(_soft_frame())
        assert result == {
            "predicted_class": "defocused_blurred",
            "confidence": 1.0,
            "probabilities": {
                "defocused_blurred": 1.0,
                "defocused_object_portrait": 0.0,
                "motion_blurred": 0.0,
                "sharp": 0.0,
            },
        }
        assert classifier.detector.calls == 0

    def test_sideways_smear_is_motion_and_never_reaches_the_detector(self, classifier):
        result = classifier.classify(_smeared_frame())
        assert result["predicted_class"] == "motion_blurred"
        assert result["confidence"] == 1.0
        assert classifier.detector.calls == 0

    def test_batch_mixes_both_stages_and_keeps_positions(self, classifier):
        images = [_make_bgr_image(), _soft_frame(), None, _smeared_frame(), _make_bgr_image()]
        results = classifier.classify_batch(images)
        assert [r and r["predicted_class"] for r in results] == [
            "sharp",
            "defocused_blurred",
            None,
            "motion_blurred",
            "sharp",
        ]
        assert classifier.detector.calls == 2

    def test_batch_gives_each_photo_its_own_answer(self, classifier):
        """Stage 2's answer depends on the picture here, so swapped answers would show."""
        classifier._missed_focus_probability = lambda image: float(image.mean() > 128)
        dark, bright = _make_bgr_image() // 4, 192 + _make_bgr_image() // 4
        results = classifier.classify_batch([_soft_frame(), dark, bright] + [dark] * 6 + [bright])
        assert [r["predicted_class"] for r in results] == [
            "defocused_blurred",
            "sharp",
            "defocused_object_portrait",
            *["sharp"] * 6,
            "defocused_object_portrait",
        ]

    def test_batch_of_only_whole_frame_blur_skips_the_detector(self, classifier):
        results = classifier.classify_batch([_soft_frame(), _smeared_frame()])
        assert [r["predicted_class"] for r in results] == ["defocused_blurred", "motion_blurred"]
        assert classifier.detector.calls == 0

    def test_an_unreadable_photo_gives_none_only_at_its_position(self, classifier):
        def detect(image):
            if image.shape[0] == 257:
                raise ValueError("unreadable")
            return []

        classifier.detector.detect = detect
        odd = _make_bgr_image(
            width=256, height=257
        )  # passes stage 1 (noise is "sharp"), then fails
        results = classifier.classify_batch([_make_bgr_image(), odd, _make_bgr_image()])
        assert [r and r["predicted_class"] for r in results] == ["sharp", None, "sharp"]

    def test_targeted_detection_sees_whole_frame_blur(self, classifier):
        result = classifier.detect_blur_type(_soft_frame(), "defocused_blurred")
        assert result["detected"] is True
        assert result["blur_type_probability"] == 1.0

    def test_stage_two_runs_the_people_pipeline(self, tmp_path):
        """A rule weighing the largest person's sharpness reacts to a soft versus a crisp person."""
        names = list(people.FEATURE_NAMES)
        spec = _rule_spec(0.5)
        spec["coef"][
            names.index("largest_es")
        ] = -4.0  # sharper largest person -> less missed focus
        clf = _unarmed(tmp_path)
        clf.session = object()
        clf.rule = people.LogisticRule(spec)
        background = _make_bgr_image(
            width=300, height=400
        )  # noisy: stage 1 lets both photos through
        crisp = background.copy()
        crisp[100:300, 100:200] = 255  # a "person" ...
        crisp[100:300:8, 100:200] = 40  # ... wearing stripes
        soft = background.copy()
        soft[100:300, 100:200] = cv2.GaussianBlur(crisp, (0, 0), 6)[100:300, 100:200]
        mask = np.zeros((400, 300), dtype=bool)
        mask[100:300, 100:200] = True
        clf.detector = FakeDetector(
            [Detection(box=(100.0, 100.0, 200.0, 300.0), conf=0.9, mask=mask)]
        )
        p_crisp = clf.classify(crisp)["probabilities"]["defocused_object_portrait"]
        p_soft = clf.classify(soft)["probabilities"]["defocused_object_portrait"]
        assert p_soft > p_crisp


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

        service = BlurService(detector=BlurDetector(), classifier=None)
        assert service.classify(_make_bgr_image()) is None

    def test_classify_with_armed_classifier(self, tmp_path):
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        service = BlurService(detector=BlurDetector(), classifier=_arm(_unarmed(tmp_path)))
        result = service.classify(_make_bgr_image())
        assert result is not None
        assert "predicted_class" in result

    def test_detect_still_works_with_classifier(self):
        """Adding classifier doesn't break existing detect()."""
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        service = BlurService(detector=BlurDetector(), classifier=None)
        result = service.detect(_make_bgr_image())
        assert "is_blurry" in result
        assert "laplacian_variance" in result

    def test_detect_blur_type_without_classifier(self):
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        service = BlurService(detector=BlurDetector(), classifier=None)
        assert service.detect_blur_type(_make_bgr_image(), "defocused_blurred") is None

    def test_detect_blur_type_with_armed_classifier(self, tmp_path):
        from src.ml.blur.detector import BlurDetector
        from src.services.blur_service import BlurService

        service = BlurService(detector=BlurDetector(), classifier=_arm(_unarmed(tmp_path)))
        result = service.detect_blur_type(_make_bgr_image(), "motion_blurred")
        assert result is not None
        assert "detected" in result
        assert isinstance(result["detected"], bool)


def test_stage_timers_grow_only_for_the_stages_a_photo_went_through(tmp_path):
    classifier = _arm(_unarmed(tmp_path), p_missed=0.3)
    assert classifier.timing_snapshot() == {"stage1": 0.0, "detect": 0.0, "measure": 0.0}

    classifier.classify_batch([_soft_frame()])  # stage 1 decides; the detector never runs
    after_soft = classifier.timing_snapshot()
    assert after_soft["stage1"] > 0 and after_soft["detect"] == 0 and after_soft["measure"] == 0

    classifier.classify(_make_bgr_image())  # noise is "sharp" for stage 1, so stage 2 runs
    after_sharp = classifier.timing_snapshot()
    assert after_sharp["stage1"] > after_soft["stage1"]
    assert after_sharp["detect"] > 0 and after_sharp["measure"] > 0
    after_sharp["detect"] = -1.0  # a copy: changing it must not touch the counters
    assert classifier.timing_snapshot()["detect"] > 0
