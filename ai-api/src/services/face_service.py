from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FaceService:
    """Business logic for face detection, enrollment, and search."""

    def __init__(self, embedder, face_repo=None) -> None:
        self.embedder = embedder
        self.face_repo = face_repo

    def detect_faces(self, image):
        return self.embedder.detect_faces(image)

    def get_embeddings(self, image):
        return self.embedder.get_embeddings(image)
