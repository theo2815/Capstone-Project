# Architecture

## Overview

EventAI is a REST API that accepts images over HTTP and returns AI analysis results. It supports three computer vision features:

1. **Blur Detection** - Determines if an image is blurry
2. **Face Recognition** - Detects, enrolls, and matches faces
3. **Bib Number OCR** - Reads race bib numbers from photos

## The 4-Layer Architecture

Every request flows through 4 layers. Each layer has one responsibility and only talks to the layer directly below it.

```
HTTP Request from client (mobile app, website, backend service)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  API Layer  (src/api/)                                   │
│  Receives HTTP requests, validates input, returns JSON.  │
│  No business logic lives here.                           │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Service Layer  (src/services/)                          │
│  Contains business rules. Orchestrates ML models and DB. │
│  "When someone uploads a photo, what should happen?"     │
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
│  Stores and retrieves data. Face embeddings, jobs,       │
│  webhook subscriptions, API keys.                        │
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
    ├── Load BlurDetector into memory
    ├── Load FaceEmbedder (RetinaFace + ArcFace) into memory
    ├── Load BibDetector (YOLOv8) into memory
    ├── Load BibRecognizer (PaddleOCR) into memory
    │
    ▼
Server ready to accept requests (models stay in memory)
    │
    ▼
Server stops → models unloaded, memory freed
```

### 2. Async everywhere

The API uses Python's async/await pattern:
- **I/O operations** (database queries, Redis calls, webhook delivery) are non-blocking
- **CPU-bound ML inference** runs in a thread pool via `asyncio.to_thread()` so it doesn't block the event loop
- This means the server can handle many concurrent requests even during inference

### 3. C++ is optional

Performance-critical code (batch cosine similarity, batch image preprocessing) can be accelerated with C++ via pybind11. But the app **always works without it**:

```python
try:
    from _eventai_cpp import batch_cosine_topk  # Fast C++ path
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False  # Falls back to NumPy
```

### 4. Background task processing

For batch operations (processing 100+ images), the API doesn't block:

```
Client sends 100 images
    │
    ▼
API creates a Job record in DB, queues tasks to Celery via Redis
    │
    ▼
Returns immediately: { "job_id": "abc-123", "status": "pending" }
    │
    ▼
Celery worker picks up tasks, processes images one by one
    │
    ▼
Client polls GET /api/v1/jobs/abc-123 to check progress
    OR
Client receives a webhook callback when the job completes
```

## Request Flow (complete example)

Here's what happens when a client calls `POST /api/v1/blur/detect`:

```
1. Client sends HTTP POST with an image file
   │
2. RequestIDMiddleware assigns a UUID (e.g., "req-abc-123")
   │
3. auth.py extracts X-API-Key header
   ├── Hashes the key with SHA-256
   ├── Checks Redis cache for the hash
   ├── If not cached, checks PostgreSQL api_keys table
   ├── Returns key metadata (scopes, rate tier)
   │
4. The route handler in blur.py runs:
   ├── Calls validate_and_decode() on the uploaded file
   │   ├── Checks content type (JPEG/PNG/WebP only)
   │   ├── Checks file size (10MB max)
   │   ├── Verifies image magic bytes (not just the header)
   │   ├── Checks dimensions (32px min, 4096px max)
   │   └── Strips EXIF data, decodes to numpy array
   │
   ├── Gets the BlurDetector from the model registry
   │
   ├── Calls detector.detect(image)
   │   ├── Converts to grayscale
   │   ├── Applies Laplacian filter, computes variance
   │   ├── Runs FFT, computes high-frequency ratio
   │   └── Returns: { is_blurry, confidence, metrics }
   │
   ├── Wraps result in BlurDetectionResponse (Pydantic schema)
   └── Wraps that in APIResponse envelope
       │
5. Client receives JSON response
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
