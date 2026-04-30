# Architecture

## Overview

QuickPitik is a REST API that accepts images over HTTP and returns AI analysis results. It supports three computer vision features:

1. **Blur Detection** - Determines if an image is blurry
2. **Face Recognition** - Detects, enrolls, and matches faces
3. **Bib Number OCR** - Reads race bib numbers from photos

## The 4-Layer Architecture

Every request flows through up to 4 layers. Each layer has one responsibility and only talks to the layer directly below it.

```
HTTP Request from client (mobile app, website, backend service)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  API Layer  (src/api/v1/)                                │
│  Receives HTTP requests, validates input, returns JSON.  │
│  Thin handlers — no business logic.                      │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Service Layer  (src/services/)                          │
│  Business rules. Orchestrates ML models and DB.          │
│  Today only BlurService exists — face and bib handlers   │
│  call the ML + repo layers directly while the logic is   │
│  still thin. Add a service when orchestration grows.     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  ML Layer  (src/ml/)                                     │
│  Wraps AI model libraries. Runs inference.               │
│  Knows nothing about HTTP or databases.                  │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  DB Layer  (src/db/)                                     │
│  Stores and retrieves data. Persons + face embeddings,   │
│  jobs, webhook subscriptions, API keys.                  │
└──────────────────────────────────────────────────────────┘
```

### Why this separation matters

- **Testability**: You can test the blur detector without starting a web server.
- **Swappability**: You can replace PaddleOCR with Tesseract without touching any API code.
- **Readability**: A new developer knows exactly where to look for each concern.

## Key Architectural Decisions

### 1. Model Registry (singleton pattern)

ML models are large (100MB-500MB each) and take seconds to load. Loading them on every request would be disastrous for performance. Instead:

- Models are loaded **once** at app startup via `src/ml/registry.py`
- Stored in memory for the entire lifetime of the process
- All requests share the same model instances
- Loaded via FastAPI's `lifespan` context manager in `src/main.py`

```
Server starts
    │
    ├── Pre-import torch / insightface / ultralytics    (avoid Windows DLL race)
    │
    ├── Load BlurDetector (Laplacian + FFT)             — registered as "blur", required
    ├── Load BlurClassifier (YOLOv8n-cls ONNX)          — registered as "blur_classifier", optional
    ├── Load FaceEmbedder (InsightFace buffalo_l)       — registered as "face", required
    ├── Load BibDetector (YOLOv8n ONNX, Ultralytics)    — registered as "bib_detector", optional
    ├── Load BibRecognizer (PaddleOCR PP-OCRv5)         — registered as "bib_ocr", required
    │
    ▼
Server ready to accept requests (models stay in memory)
    │
    ▼
Server stops → registry.unload_all() frees ONNX sessions + engines
```

`required` models count toward `models_loaded` in the readiness probe. `optional` models are reported as unavailable in the endpoint response (`503 MODEL_UNAVAILABLE`) if they fail to load, but the server stays up.

### 2. Async everywhere

The API uses Python's async/await pattern:
- **I/O operations** (database queries, Redis calls, webhook delivery) are non-blocking
- **CPU-bound ML inference** runs in a thread pool via `asyncio.to_thread()` so it doesn't block the event loop
- This means the server can handle many concurrent requests even during inference

### 3. C++ is optional

Performance-critical code (batch cosine similarity, batch image preprocessing) can be accelerated with C++ via pybind11. But the app **always works without it**:

```python
try:
    from _quickpitik_cpp import batch_cosine_topk  # Fast C++ path
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False  # Falls back to NumPy
```

### 4. Background task processing (blob-store, not base64)

For batch operations the API never blocks the request and never serialises image bytes onto the Celery queue:

```
Client POSTs N images (multipart)
    │
    ▼
API validates size + count, applies per-key backpressure
(MAX_ACTIVE_JOBS_PER_KEY; 429 if exceeded)
    │
    ▼
API creates Job row, writes image bytes to
{BLOB_STORE_PATH}/{job_id}/NNNNN.bin (atomic tmp→rename)
    │
    ▼
Celery task dispatched with (job_id, [file_paths])
    │
    ▼
Returns 202: { "job_id": "...", "poll_url": "/jobs/..." }
    │
    ▼
Worker: parallel decode from paths → sub-batched inference
       → update_job_progress (throttled) → complete_job
       → cleanup_batch removes blob dir → webhook dispatch
    │
    ▼
Client polls GET /api/v1/jobs/{id}?offset=..&limit=..
OR receives a webhook (job.completed / job.failed)
```

Three batch modes exist per pipeline (see `api-reference.md`):

- `.../batch` — `MAX_BATCH_SIZE` (50) — single Celery task
- `.../mega`  — `MEGA_BATCH_MAX_SIZE` (500) — Celery chord, auto-chunked, merged by `finalize_mega_batch`
- `.../stream` (blur only) — `STREAM_BATCH_MAX_SIZE` (500) — synchronous NDJSON stream; no Celery

## Request Flow (complete example)

Here's what happens when a client calls `POST /api/v1/blur/detect`:

```
1. Client sends HTTP POST with an image file
   │
2. SecurityHeadersMiddleware adds X-Content-Type-Options/X-Frame-Options/
   Referrer-Policy (+ HSTS in production) — pure ASGI wrapper
   │
3. CORSMiddleware validates origin
   │
4. RequestIDMiddleware assigns / validates X-Request-ID → request.state
   │
5. TimeoutMiddleware starts a 60s wall-clock timer (504 if exceeded)
   │
6. verify_api_key (route dep) extracts X-API-Key
   ├── SHA-256 hashes the header
   ├── Checks Redis cache for "apikey:<hash>"
   ├── Falls back to the api_keys table on cache miss
   ├── Enforces rate limit (token bucket in Redis, per tier)
   └── Stores rate_info on request.state
   │
7. check_scope("blur:read", key_meta) — 403 if key lacks the scope
   │
8. Route handler:
   ├── validate_and_decode() — content type, 10MB cap, PIL magic check,
   │                          EXIF rotate, 32–4096 px bounds, BGR ndarray
   │                          (downscaled to MAX_INFERENCE_DIMENSION)
   ├── registry.get("blur") — BlurDetector
   ├── asyncio.to_thread(detector.detect, image) — runs Laplacian (+ FFT)
   │   off the event loop; uses _quickpitik_cpp when present
   └── Wraps result in BlurDetectionResponse → APIResponse envelope
   │
9. RateLimitHeadersMiddleware attaches X-RateLimit-*
   │
10. Client receives JSON response with X-Request-ID header
```

## Infrastructure Diagram

```
┌─────────────────────────────────────────────────────┐
│                    Docker Compose                     │
│                                                       │
│  ┌─────────────┐    ┌──────────────────┐             │
│  │   ai-api    │    │  celery-worker   │             │
│  │  (FastAPI)  │    │  (background)    │             │
│  │  port 8000  │    │                  │             │
│  └──────┬──────┘    └────────┬─────────┘             │
│         │                    │                        │
│         ▼                    ▼                        │
│  ┌─────────────┐    ┌──────────────────┐             │
│  │  PostgreSQL  │    │      Redis       │             │
│  │  + pgvector  │    │  (cache, queue)  │             │
│  │  port 5432   │    │  port 6379       │             │
│  └──────────────┘    └──────────────────┘             │
└─────────────────────────────────────────────────────┘
```

- **PostgreSQL + pgvector**: Stores face embeddings (512-dim vectors), persons, jobs, webhooks, API keys. pgvector enables fast vector similarity search.
- **Redis**: Three roles - Celery task queue broker, rate limit counters, API key cache.
- **Celery worker**: Separate process that picks up batch tasks from Redis and processes them.
