from __future__ import annotations

from prometheus_client import Counter, Histogram

# Request metrics
REQUEST_COUNT = Counter(
    "eventai_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "eventai_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Model inference metrics
INFERENCE_LATENCY = Histogram(
    "eventai_inference_duration_seconds",
    "ML model inference latency in seconds",
    ["model_name"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

INFERENCE_COUNT = Counter(
    "eventai_inferences_total",
    "Total number of ML model inferences",
    ["model_name", "status"],
)

# Image processing metrics
IMAGE_SIZE = Histogram(
    "eventai_image_size_bytes",
    "Uploaded image size in bytes",
    buckets=[10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
)
