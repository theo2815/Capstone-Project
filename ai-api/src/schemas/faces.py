from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0, le=1.0)


class FaceDetection(BaseModel):
    bbox: BoundingBox
    landmarks: list[tuple[float, float]] | None = None


class FaceSearchResult(BaseModel):
    person_id: UUID
    person_name: str | None = None
    similarity: float = Field(ge=0, le=1.0)
    bbox: BoundingBox


class FaceDetectResponse(BaseModel):
    faces_detected: int
    faces: list[FaceDetection]
    image_dimensions: tuple[int, int]
    processing_time_ms: float


class FaceSearchResponse(BaseModel):
    faces_detected: int
    matches: list[FaceSearchResult]
    unmatched_faces: list[FaceDetection]
    processing_time_ms: float


class FaceEnrollResponse(BaseModel):
    person_id: UUID
    person_name: str
    faces_enrolled: int
    embeddings_stored: int
    processing_time_ms: float


class FaceCompareResponse(BaseModel):
    is_match: bool
    similarity: float = Field(ge=0, le=1.0)
    face1: FaceDetection | None = None
    face2: FaceDetection | None = None
    processing_time_ms: float


class PersonResponse(BaseModel):
    person_id: UUID
    person_name: str
    embeddings_count: int
    created_at: str
    updated_at: str
