from __future__ import annotations

from pydantic import BaseModel, Field


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
