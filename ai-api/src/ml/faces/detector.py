from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """Standalone face detection using RetinaFace via InsightFace.

    This is used when only detection (no embeddings) is needed.
    For detection + embeddings, use FaceEmbedder instead.
    """

    def __init__(
        self,
        model_dir: str = "./models",
        use_gpu: bool = False,
        det_size: int = 640,
    ) -> None:
        from insightface.app import FaceAnalysis

        ctx_id = 0 if use_gpu else -1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect faces and return bounding boxes + landmarks."""
        faces = self.app.get(image)
        return [
            {
                "bbox": {
                    "x1": float(face.bbox[0]),
                    "y1": float(face.bbox[1]),
                    "x2": float(face.bbox[2]),
                    "y2": float(face.bbox[3]),
                    "confidence": float(face.det_score),
                },
                "landmarks": face.kps.tolist() if face.kps is not None else None,
            }
            for face in faces
        ]
