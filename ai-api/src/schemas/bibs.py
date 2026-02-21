from __future__ import annotations

from pydantic import BaseModel, Field


class BibCandidate(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1.0)


class BibDetection(BaseModel):
    bib_number: str
    confidence: float = Field(ge=0, le=1.0)
    bbox: dict[str, float]
    all_candidates: list[BibCandidate]


class BibRecognitionResponse(BaseModel):
    bibs_detected: int
    detections: list[BibDetection]
    image_dimensions: tuple[int, int]
    processing_time_ms: float
