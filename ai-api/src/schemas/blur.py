from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BlurType(str, Enum):
    """Selectable blur types for targeted detection."""

    DEFOCUSED_OBJECT_PORTRAIT = "defocused_object_portrait"
    DEFOCUSED_BLURRED = "defocused_blurred"
    MOTION_BLURRED = "motion_blurred"


class BlurMetrics(BaseModel):
    laplacian_variance: float
    hf_ratio: float
    confidence: float = Field(ge=0, le=1.0)


class BlurDetectionResponse(BaseModel):
    is_blurry: bool
    confidence: float = Field(ge=0, le=1.0)
    metrics: BlurMetrics | None = None
    image_dimensions: tuple[int, int]
    processing_time_ms: float


class BlurClassProbabilities(BaseModel):
    sharp: float = Field(ge=0, le=1.0)
    defocused_object_portrait: float = Field(ge=0, le=1.0)
    defocused_blurred: float = Field(ge=0, le=1.0)
    motion_blurred: float = Field(ge=0, le=1.0)


class BlurClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1.0)
    probabilities: BlurClassProbabilities
    image_dimensions: tuple[int, int]
    processing_time_ms: float


class BlurTypeDetectionResponse(BaseModel):
    blur_type: str
    detected: bool
    confidence: float = Field(ge=0, le=1.0)
    blur_type_probability: float = Field(ge=0, le=1.0)
    image_dimensions: tuple[int, int]
    processing_time_ms: float
