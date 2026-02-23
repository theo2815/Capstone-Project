from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BlurClassifier:
    """Classify images into blur categories using an ONNX model.

    Classes: sharp, defocused_object_portrait, defocused_blurred, motion_blurred.
    Falls back gracefully if the model file is not found.
    """

    def __init__(
        self,
        model_path: str = "./models/blur_classifier/blur_classifier.onnx",
        class_names_path: str = "./models/blur_classifier/class_names.json",
        input_size: int = 224,
    ) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.input_size = input_size
        self.session = None
        self.class_names: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        # Default class ordering (alphabetical, as YOLOv8 sorts)
        self.class_names = [
            "defocused_blurred",
            "defocused_object_portrait",
            "motion_blurred",
            "sharp",
        ]

        try:
            import onnxruntime as ort

            if not Path(self.model_path).exists():
                logger.warning(
                    "Blur classifier model not found",
                    model_path=self.model_path,
                )
                return

            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )

            # Load class names from file (overrides defaults)
            names_path = Path(self.class_names_path)
            if names_path.exists():
                with open(names_path) as f:
                    self.class_names = json.load(f)

            logger.info(
                "BlurClassifier loaded",
                model_path=self.model_path,
                classes=self.class_names,
            )
        except Exception as e:
            logger.warning(
                "BlurClassifier model not available",
                model_path=self.model_path,
                error=str(e),
            )
            self.session = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess BGR image for ONNX inference.

        Matches YOLOv8 classify inference pipeline:
        1. Center-crop to square (using shorter dimension)
        2. Resize to input_size x input_size
        3. BGR -> RGB, normalize to [0, 1]
        4. HWC -> CHW, add batch dimension
        """
        h, w = image.shape[:2]
        # Center-crop to square (matches YOLOv8 CenterCrop)
        m = min(h, w)
        top, left = (h - m) // 2, (w - m) // 2
        cropped = image[top : top + m, left : left + m]
        # Resize to target size
        resized = cv2.resize(
            cropped, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        # BGR -> RGB, normalize to [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # HWC -> CHW
        chw = np.transpose(rgb, (2, 0, 1))
        # Add batch dimension: (1, 3, H, W)
        return np.expand_dims(chw, axis=0)

    def classify(self, image: np.ndarray) -> dict | None:
        """Classify an image into blur categories.

        Args:
            image: BGR numpy array from cv2.

        Returns:
            Dict with predicted_class, confidence, and per-class probabilities.
            Returns None if model is not loaded.
        """
        if self.session is None:
            return None

        input_tensor = self._preprocess(image)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]  # shape: (num_classes,)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        prob_dict = {
            name: float(probabilities[i]) for i, name in enumerate(self.class_names)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

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

        detected = predicted_class == blur_type

        return {
            "detected": detected,
            "confidence": confidence,
            "blur_type": blur_type,
            "blur_type_probability": blur_type_probability,
            "predicted_class": predicted_class,
            "probabilities": result["probabilities"],
        }
