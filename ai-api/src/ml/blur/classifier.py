from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np

from src.ml.blur import people, whole_frame
from src.ml.blur.person_detector import PersonDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BlurClassifier:
    """Classify images into blur categories in two stages.

    Stage 1 (no model): whole-frame measurements decide `defocused_blurred` and
    `motion_blurred`, the photos in which nothing is sharp (whole_frame.py).
    Stage 2 (people first): a person detector (ONNX, boxes and masks) finds the
    people, people.py measures how sharp each one is on its own pixels at the
    decoded upload's full detail, and a
    logistic rule fitted on the labelled event folds turns those numbers into
    P(`defocused_object_portrait`, subject missed focus) versus `sharp`.
    Callers still get all four classes. Falls back gracefully if a file is
    missing or does not match the manifest.
    """

    DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
    CLASSES = [
        "defocused_blurred",
        "defocused_object_portrait",
        "motion_blurred",
        "sharp",
    ]
    # What stage 2 decides, in this order; the manifest's `classes` must say the same.
    MODEL_CLASSES = ["defocused_object_portrait", "sharp"]
    # The person detector's fixed shapes (scripts/export_person_detector.py).
    DETECTOR_INPUT = [1, 3, 640, 640]
    DETECTOR_OUTPUTS = [[1, 116, 8400], [1, 32, 160, 160]]

    def __init__(
        self,
        detector_path: str = "./models/person_detector/yolov8n_seg_person.onnx",
        rule_path: str = "./models/blur_classifier/people_rule.json",
        use_gpu: bool = False,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        manifest_path: str | None = None,
        soft_max_detail: float | None = None,
        motion_min_ratio: float | None = None,
        motion_max_detail: float | None = None,
    ) -> None:
        self.detector_path = detector_path
        self.rule_path = rule_path
        self.use_gpu = use_gpu
        self.min_detection_confidence = min_detection_confidence
        self.manifest_path = manifest_path
        from src.config import get_settings

        settings = get_settings()
        self.soft_max_detail = (
            settings.BLUR_SOFT_MAX_DETAIL if soft_max_detail is None else soft_max_detail
        )
        self.motion_min_ratio = (
            settings.BLUR_MOTION_MIN_RATIO if motion_min_ratio is None else motion_min_ratio
        )
        self.motion_max_detail = (
            settings.BLUR_MOTION_MAX_DETAIL if motion_max_detail is None else motion_max_detail
        )
        self.session = None  # the detector's ONNX session; None = stage 2 unavailable
        self.detector: PersonDetector | None = None
        self.rule: people.LogisticRule | None = None
        self.class_names: list[str] = []
        self.stage_seconds = {"stage1": 0.0, "detect": 0.0, "measure": 0.0}
        self._timing_lock = Lock()
        self._load_model()

    def _load_model(self) -> None:
        # The four classes callers see; stage 2 itself only knows MODEL_CLASSES.
        self.class_names = self.CLASSES.copy()

        try:
            import onnxruntime as ort

            detector_file, rule_file = Path(self.detector_path), Path(self.rule_path)
            if not detector_file.exists() or not rule_file.exists():
                logger.warning(
                    "Blur stage-2 files not found",
                    detector_path=self.detector_path,
                    rule_path=self.rule_path,
                )
                return

            if self.manifest_path:
                models = json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))["models"]
                artifact = models["blur_classifier"]
                if artifact.get("classes") != self.MODEL_CLASSES:
                    raise ValueError("Blur classifier manifest classes mismatch")
                for path, digest in (
                    (rule_file, artifact["sha256"]),
                    (detector_file, models["person_detector"]["sha256"]),
                ):
                    with path.open("rb") as f:
                        if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                            raise ValueError("Blur classifier artifact digest mismatch")

            rule = people.LogisticRule.load(rule_file)
            if rule.feature_names != list(people.FEATURE_NAMES):
                raise ValueError("Blur rule features do not match this code")

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.use_gpu
                else ["CPUExecutionProvider"]
            )

            from src.config import get_settings

            _settings = get_settings()

            # Optimized session options for production inference
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = _settings.ONNX_INTRA_OP_THREADS
            sess_opts.inter_op_num_threads = _settings.ONNX_INTER_OP_THREADS
            sess_opts.enable_mem_pattern = True
            sess_opts.enable_cpu_mem_arena = True
            sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            if not getattr(_settings, "ONNX_ALLOW_SPINNING", True):
                sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

            session = ort.InferenceSession(
                str(detector_file),
                sess_options=sess_opts,
                providers=providers,
            )

            # Only the COCO person detector fits here. The 2026-08-14 four-class
            # model and the 2026-10 two-class model have other shapes and are
            # refused, so /blur/classify answers 503 instead of mislabelling.
            input_shape = list(session.get_inputs()[0].shape)
            output_shapes = [list(o.shape) for o in session.get_outputs()]
            if input_shape != self.DETECTOR_INPUT or output_shapes != self.DETECTOR_OUTPUTS:
                logger.error(
                    "Blur stage-2 file is not the person detector",
                    input_shape=input_shape,
                    output_shapes=output_shapes,
                )
                return

            detector = PersonDetector(session, rule.min_conf)
            if self.manifest_path:
                # Warm run through the whole stage: a blank frame has no people and
                # must still give a finite probability.
                blank = np.zeros(
                    (self.DETECTOR_INPUT[2], self.DETECTOR_INPUT[3], 3), dtype=np.uint8
                )
                probe = rule.probability(
                    people.features(
                        people.measure(blank, detector.detect(blank)),
                        rule.big_area,
                        rule.soft_ratio,
                    )
                )
                if not np.isfinite(probe) or not 0.0 <= probe <= 1.0:
                    raise ValueError("Blur stage-2 warm inference failed")

            self.session, self.detector, self.rule = session, detector, rule
            logger.info(
                "BlurClassifier loaded",
                detector_path=self.detector_path,
                rule_path=self.rule_path,
                classes=self.class_names,
                features=len(rule.feature_names),
            )
        except Exception as e:
            logger.warning(
                "BlurClassifier model not available",
                detector_path=self.detector_path,
                rule_path=self.rule_path,
                error_type=type(e).__name__,
            )
            self.session = self.detector = self.rule = None

    def _add_time(self, stage: str, seconds: float) -> None:
        with self._timing_lock:
            self.stage_seconds[stage] += seconds

    def timing_snapshot(self) -> dict[str, float]:
        """Seconds spent so far in each stage, process-wide, for a per-request timing line."""
        with self._timing_lock:
            return dict(self.stage_seconds)

    def _whole_frame_result(self, image: np.ndarray) -> dict | None:
        """Stage 1: a finished result when nothing in the frame is sharp, else None."""
        started = time.perf_counter()
        kind = whole_frame.decide(
            *whole_frame.measure(image),
            soft_max_detail=self.soft_max_detail,
            motion_min_ratio=self.motion_min_ratio,
            motion_max_detail=self.motion_max_detail,
        )
        self._add_time("stage1", time.perf_counter() - started)
        if kind is None:
            return None
        # ponytail: a rule gives a hard 1.0. Fit a curve on the distance to the
        # thresholds if callers ever need graded confidence for whole-frame blur.
        return {
            "predicted_class": kind,
            "confidence": 1.0,
            "probabilities": {name: float(name == kind) for name in self.CLASSES},
        }

    def _missed_focus_probability(self, image: np.ndarray) -> float:
        """Stage 2 on one BGR image: find the people, measure them, apply the rule.

        Nobody found: 0.0 (sharp) without asking the rule, which was never fitted on such
        a photo (owner decision 2026-10-07).
        """
        started = time.perf_counter()
        detections = self.detector.detect(image)
        detected = time.perf_counter()
        found = people.measure(image, detections)
        probability = (
            0.0
            if not found
            else self.rule.probability(
                people.features(found, self.rule.big_area, self.rule.soft_ratio)
            )
        )
        self._add_time("detect", detected - started)
        self._add_time("measure", time.perf_counter() - detected)
        return probability

    def _people_result(self, p_missed: float) -> dict:
        """Turn stage 2's probability into the four-class result callers expect.

        The whole-frame classes are 0.0 here: stage 1 already ruled them out.
        """
        prob_dict = {name: 0.0 for name in self.CLASSES}
        prob_dict["defocused_object_portrait"] = float(p_missed)
        prob_dict["sharp"] = float(1.0 - p_missed)
        predicted_class = max(prob_dict, key=prob_dict.get)
        return {
            "predicted_class": predicted_class,
            "confidence": prob_dict[predicted_class],
            "probabilities": prob_dict,
        }

    def _run_stage_two(self, image: np.ndarray) -> float:
        """Stage 2 under the blur profile's per-image deadline."""
        from src.config import get_settings
        from src.utils.timeout import run_direct, run_with_timeout

        settings = get_settings()
        if settings.MODEL_PROFILE == "blur":
            return run_with_timeout(
                self._missed_focus_probability,
                args=(image,),
                timeout_seconds=settings.BLUR_INFERENCE_TIMEOUT_SECONDS,
            )
        return run_direct(self._missed_focus_probability, args=(image,), timeout_seconds=0)

    def classify(self, image: np.ndarray) -> dict | None:
        """Classify an image into blur categories.

        Args:
            image: BGR numpy array from cv2.

        Returns:
            Dict with predicted_class, confidence, and per-class probabilities.
            Returns None if the stage-2 files are not loaded.
        """
        if self.session is None:
            return None

        whole_frame_blur = self._whole_frame_result(image)
        if whole_frame_blur is not None:
            return whole_frame_blur
        return self._people_result(self._run_stage_two(image))

    def classify_batch(self, images: list[np.ndarray]) -> list[dict | None]:
        """Classify multiple images.

        Stage 2 runs the detector on each photo on its own: a (1, 3, 640, 640)
        input is 5 MB, and the detector's fixed batch axis is 1, so there is no
        batched call to gain from. With BLUR_STAGE2_CONCURRENCY=2 two photos are
        scored at the same time, each on its own native inference slot.

        Args:
            images: List of BGR numpy arrays. None entries are skipped and
                    produce None in the output at the same index.

        Returns:
            List of result dicts (same format as classify()), with None at
            positions where the input image was None or unreadable.
        """
        if self.session is None:
            return [None] * len(images)
        from src.config import get_settings
        from src.utils.timeout import InferenceBusyError, InferenceTimeoutError

        results: list[dict | None] = [None] * len(images)

        def one(index: int, img: np.ndarray) -> None:
            try:
                result = self._whole_frame_result(img)
                if result is None:
                    result = self._people_result(self._run_stage_two(img))
                results[index] = result
            except Exception as e:
                # An unreadable image gives None at its position; the deadline and
                # busy errors are not swallowed here, callers decide what they mean.
                if isinstance(e, (InferenceTimeoutError, InferenceBusyError)):
                    raise
                logger.warning(
                    "Blur classification failed for one image", error_type=type(e).__name__
                )

        todo = [(i, img) for i, img in enumerate(images) if img is not None]
        workers = getattr(get_settings(), "BLUR_STAGE2_CONCURRENCY", 1)
        if workers <= 1:
            for i, img in todo:
                one(i, img)
            return results
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(one, i, img) for i, img in todo]:
                future.result()  # re-raises a deadline or busy error
        return results

    def detect_blur_type(self, image: np.ndarray, blur_type: str) -> dict | None:
        """Detect whether a specific blur type is present in an image.

        Args:
            image: BGR numpy array from cv2.
            blur_type: One of "defocused_object_portrait", "defocused_blurred",
                       "motion_blurred".

        Returns:
            Dict with detected (bool), confidence, blur_type, blur_type_probability,
            predicted_class, and probabilities.
            Returns None if model is not loaded.
        """
        result = self.classify(image)
        if result is None:
            return None

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        blur_type_probability = result["probabilities"].get(blur_type, 0.0)

        detected = predicted_class == blur_type and confidence >= self.min_detection_confidence

        return {
            "detected": detected,
            "confidence": confidence,
            "blur_type": blur_type,
            "blur_type_probability": blur_type_probability,
            "predicted_class": predicted_class,
            "probabilities": result["probabilities"],
        }
