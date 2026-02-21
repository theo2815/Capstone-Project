from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FaceEmbedder:
    """Face detection and embedding extraction using InsightFace (RetinaFace + ArcFace)."""

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
        logger.info("FaceEmbedder initialized", gpu=use_gpu, det_size=det_size)

    def detect_faces(self, image: np.ndarray) -> list[dict]:
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

    def get_embeddings(self, image: np.ndarray) -> list[dict]:
        """Detect faces and extract 512-dim embeddings."""
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
                "embedding": face.normed_embedding.tolist(),
                "landmarks": face.kps.tolist() if face.kps is not None else None,
            }
            for face in faces
        ]
