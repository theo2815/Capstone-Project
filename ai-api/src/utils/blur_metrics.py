from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

images = Counter(
    "blur_images_total", "Accepted and accounted blur images", ["operation", "outcome"]
)
rejected_requests = Counter(
    "blur_rejected_requests_total", "Blur requests rejected before processing", ["reason"]
)
active_requests = Gauge(
    "blur_active_requests", "Admitted blur requests, including timed-out native work"
)
stream_outcomes = Counter(
    "blur_streams_total", "Blur stream completion outcomes", ["operation", "outcome"]
)
first_result = Histogram(
    "blur_first_result_seconds", "Time from accepted stream to first image result",
    ["operation"],
)
stream_duration = Histogram(
    "blur_stream_duration_seconds", "Time from accepted stream to completion",
    ["operation"],
)
timeouts = Counter(
    "blur_timeouts_total", "Blur timeouts by work phase", ["phase"]
)
model_info = Gauge(
    "blur_model_info", "Verified blur classifier artifact", ["version", "sha256"]
)
dependency_ready = Gauge(
    "blur_dependency_ready", "Last readiness probe result", ["dependency"]
)
