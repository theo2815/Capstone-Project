# API Reference

All endpoints are prefixed with `/api/v1/`. All responses use a standard envelope:

```json
{
  "success": true,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-04-23T12:34:56.789Z",
  "data": { ... },
  "error": null
}
```

On error:
```json
{
  "success": false,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-04-23T12:34:56.789Z",
  "data": null,
  "error": {
    "code": "ImageValidationError",
    "message": "File exceeds 25MB limit"
  }
}
```

**Authentication**: All endpoints except `GET /api/v1/health` require an `X-API-Key` header. When the server runs with `DEBUG=true`, a missing key is allowed and treated as an internal-tier key with full scopes. The production server refuses to start with `DEBUG=true` when `ENVIRONMENT=production`.

**Request ID**: Clients may pass `X-Request-ID` (up to 128 alphanumerics or `-`). If absent, the server generates a UUID. The ID is echoed in the response body and `X-Request-ID` response header.

**Request timeout**: Every request has a 60-second wall-clock timeout enforced by `TimeoutMiddleware`. Exceeding it returns `504 REQUEST_TIMEOUT`.

---

## Health

### GET /api/v1/health

Liveness probe. Returns 200 if the process is running. **No authentication required.**

**Response:**
```json
{
  "status": "alive",
  "version": "1.0.0"
}
```

### GET /api/v1/health/ready

Readiness probe. Checks that all required models are loaded and that the database and Redis are reachable. Requires authentication.

**Response (200):**
```json
{
  "success": true,
  "request_id": "healthcheck",
  "data": {
    "models_loaded": true,
    "database": true,
    "redis": true
  }
}
```

Returns `503` with `success: false` if any check fails.

---

## Blur Detection

The blur pipeline has a classical CV detector (Laplacian variance + optional FFT) and a CNN classifier (YOLOv8n-cls, 4 classes). Each has single, streaming, batch, and mega-batch endpoints.

### POST /api/v1/blur/detect

Check if a single image is blurry (Laplacian/FFT).

**Request:**
- `file` (multipart, required): Image file (JPEG, PNG, or WebP)
- `threshold` (query, optional): Laplacian variance threshold. Default `100.0`, range `1.0–10000.0`. Lower = stricter.
- `include_metrics` (query, optional): Include detailed metrics. Default `true`.

**Scope:** `blur:read`

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/blur/detect \
  -H "X-API-Key: sk_dev_quickpitik_test_key_12345" \
  -F "file=@photo.jpg" \
  -G -d "threshold=100" -d "include_metrics=true"
```

**Response (200):**
```json
{
  "success": true,
  "data": {
    "is_blurry": false,
    "confidence": 0.85,
    "metrics": {
      "laplacian_variance": 185.42,
      "hf_ratio": 0.72,
      "confidence": 0.85
    },
    "image_dimensions": [1920, 1080],
    "processing_time_ms": 12.34
  }
}
```

### POST /api/v1/blur/detect/stream

High-throughput synchronous blur detection. Streams NDJSON — one JSON object per line, then a summary line. Up to `STREAM_BATCH_MAX_SIZE` (500) images per request. No job, no polling.

**Request:**
- `files` (multipart, required): up to 500 images
- `threshold` (query, optional): Laplacian threshold override
- `include_hf_ratio` (query, optional, default `false`): Also compute FFT high-frequency ratio (adds ~8 ms per image)

**Response:** `application/x-ndjson`, one result per line, then a summary:

```
{"index":0,"filename":"IMG_001.jpg","is_blurry":false,"confidence":0.82,"laplacian_variance":185.42}
{"index":1,"filename":"IMG_002.jpg","is_blurry":true,"confidence":0.95,"laplacian_variance":4.12}
{"_summary":true,"total":2,"processing_time_ms":412.3}
```

Headers: `X-Total-Images` carries the request size.

### POST /api/v1/blur/detect/batch

Submit a batch for async processing via Celery. Up to `MAX_BATCH_SIZE` (50) images.

**Request:** `files` (multipart)

**Response (202):**
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-...",
    "status": "pending",
    "total_items": 10,
    "poll_url": "/api/v1/jobs/550e8400-..."
  }
}
```

Rejected with `429 TOO_MANY_JOBS` if the caller already has `MAX_ACTIVE_JOBS_PER_KEY` (10) active jobs.

### POST /api/v1/blur/detect/mega

Same as `/batch` but accepts up to `MEGA_BATCH_MAX_SIZE` (500) images. The server chunks the input and dispatches a Celery chord; results are merged back into the single parent job.

### POST /api/v1/blur/classify

Classify an image into blur categories using a CNN model (YOLOv8n-cls). Returns 503 `MODEL_UNAVAILABLE` if the ONNX file is missing.

Two modes:

- **Full classification** (default): predicted class + probability vector over `sharp`, `defocused_object_portrait`, `defocused_blurred`, `motion_blurred`.
- **Targeted detection**: when `blur_type` is set, returns a binary Detected / Not Detected answer for that type. `detected` requires both a matching predicted class **and** `confidence >= BLUR_DETECTION_MIN_CONFIDENCE` (0.5) — identical on `/classify`, `/classify/stream` and `/classify/batch`.

All classify paths downscale to `BLUR_CLASSIFY_DECODE_DIM` (640) before inference, so the single, streaming and Celery endpoints return the same class for the same image. `image_dimensions` reports the decoded size before that step.

**Request:**
- `file` (multipart)
- `blur_type` (query, optional): one of `defocused_object_portrait`, `defocused_blurred`, `motion_blurred`

**Response (full classification):**
```json
{
  "success": true,
  "data": {
    "predicted_class": "sharp",
    "confidence": 0.96,
    "probabilities": {
      "sharp": 0.96,
      "defocused_object_portrait": 0.02,
      "defocused_blurred": 0.01,
      "motion_blurred": 0.01
    },
    "image_dimensions": [1920, 1080],
    "processing_time_ms": 45.12
  }
}
```

**Response (targeted detection):**
```json
{
  "success": true,
  "data": {
    "blur_type": "defocused_object_portrait",
    "detected": true,
    "confidence": 0.94,
    "blur_type_probability": 0.94,
    "image_dimensions": [1920, 1080],
    "processing_time_ms": 42.8
  }
}
```

### POST /api/v1/blur/classify/stream

NDJSON stream of classifications, up to `STREAM_CLASSIFY_MAX_SIZE` (500) images. Same `blur_type` query param as the single endpoint.

### POST /api/v1/blur/classify/batch

Async batch classification via Celery. Up to `MAX_BATCH_SIZE` (50). Accepts `blur_type` query param.

### POST /api/v1/blur/classify/mega

Same as `/batch` but up to 500 images via Celery chord.

---

## Face Recognition

Pipeline: InsightFace (RetinaFace detection + ArcFace embedding, 512-dim) → pgvector cosine search. Face data is always scoped by `api_key_id`, and `event_id` is **required** on every enroll and search surface (root rule 5) — `detect` and `compare` are the exceptions, since neither touches stored data.

### POST /api/v1/faces/detect

Detect faces and return bounding boxes + 5-point landmarks.

**Request:** `file`

**Scope:** `faces:read`

**Response:**
```json
{
  "success": true,
  "data": {
    "faces_detected": 2,
    "faces": [
      {
        "bbox": { "x1": 120.5, "y1": 80.3, "x2": 250.1, "y2": 280.7, "confidence": 0.98 },
        "landmarks": [[145.2, 150.1], [210.3, 148.9], [180.5, 185.1], [160.8, 230.5], [200.2, 230.1]]
      }
    ],
    "image_dimensions": [1920, 1080],
    "processing_time_ms": 85.2
  }
}
```

### POST /api/v1/faces/enroll

Register a person's face. Detects, embeds, and stores. Faces below `FACE_MIN_ENROLLMENT_CONFIDENCE` (default 0.7) are skipped. If all detected faces are below the threshold, returns `LOW_QUALITY`.

**Request (multipart form):**
- `file` (required): Image containing the person
- `person_name` (required, 1–255 chars)
- `person_id` (optional, UUID): Add embeddings to an existing person. Must belong to the caller's API key **and** to `event_id`.
- `event_id` (**required**, 1–255 chars): Event to scope this enrollment to. Omitting it (or sending `""`) returns `422` — enrollment is fail-closed event isolation, like search. An embedding stored without an event is unreachable: every search path requires an `event_id`, and `DELETE /faces/persons?event_id=` cannot erase it.

**Scope:** `faces:write`

**Response:**
```json
{
  "success": true,
  "data": {
    "person_id": "550e8400-...",
    "person_name": "John Doe",
    "event_id": "marathon-2026",
    "faces_enrolled": 1,
    "embeddings_stored": 1,
    "processing_time_ms": 120.5
  }
}
```

**LOW_QUALITY response:**
```json
{
  "success": false,
  "error": {
    "code": "LOW_QUALITY",
    "message": "All 1 detected face(s) were below the minimum enrollment confidence of 0.7"
  }
}
```

### POST /api/v1/faces/enroll/batch

Async bulk enroll — every image in the batch is attached to the same person (new or existing). Up to `MAX_BATCH_SIZE` (50).

**Form fields:** same as `/enroll` plus `files` (multipart list). `event_id` is **required** here too (422 if omitted), and a supplied `person_id` must belong to the caller's API key and to that event.

### POST /api/v1/faces/search

Detect faces and search stored embeddings.

**Request:**
- `file` (required)
- `threshold` (query, optional, default `0.4`): minimum cosine similarity
- `top_k` (query, optional, default `10`, max `100`): max matches per detected face
- `event_id` (query, optional): restrict search to this event

**Response:**
```json
{
  "success": true,
  "data": {
    "faces_detected": 1,
    "matches": [
      {
        "person_id": "550e8400-...",
        "person_name": "John Doe",
        "similarity": 0.87,
        "bbox": { "x1": 120, "y1": 80, "x2": 250, "y2": 280, "confidence": 0.98 }
      }
    ],
    "unmatched_faces": [],
    "processing_time_ms": 150.3
  }
}
```

### POST /api/v1/faces/search/batch

Async face batch. Up to `MAX_BATCH_SIZE` (50).

**Query params:**
- `operation` (`detect` or `search`, default `search`)
- `event_id`, `threshold`, `top_k` (same as single `/search`)

### POST /api/v1/faces/search/mega

Same as `/search/batch` but up to 500 images via Celery chord.

### POST /api/v1/faces/compare

1:1 verification — are the two images the same person? Uses `FACE_SIMILARITY_THRESHOLD` (default 0.4).

**Request:** `file1`, `file2`

**Response:**
```json
{
  "success": true,
  "data": {
    "is_match": true,
    "similarity": 0.92,
    "face1": { "bbox": { ... } },
    "face2": { "bbox": { ... } },
    "processing_time_ms": 200.1
  }
}
```

### GET /api/v1/faces/persons

Paginated list of enrolled persons (tenant-isolated).

**Query params:** `event_id` (optional), `offset` (default 0), `limit` (default 50, max 200)

**Response:**
```json
{
  "success": true,
  "data": {
    "persons": [
      {
        "person_id": "550e8400-...",
        "person_name": "John Doe",
        "event_id": "marathon-2026",
        "embeddings_count": 3,
        "created_at": "2026-04-20T10:00:00Z",
        "updated_at": "2026-04-22T11:00:00Z"
      }
    ],
    "total": 42,
    "offset": 0,
    "limit": 50
  }
}
```

### GET /api/v1/faces/persons/{person_id}

Get metadata about one enrolled person (tenant-isolated). Returns `NOT_FOUND` if missing or owned by another key.

### DELETE /api/v1/faces/persons/{person_id}

Remove a person and all their stored embeddings (cascade delete). GDPR right-to-erasure.

**Scope:** `faces:delete`

**Response:**
```json
{
  "success": true,
  "data": { "deleted": true, "person_id": "550e8400-..." }
}
```

---

## Bib Number Recognition

Custom YOLOv8n bib detector (ONNX) → PaddleOCR 3.x (PP-OCRv5). When the detector model is absent, the endpoint falls back to OCR on the full image and attaches a warning.

### POST /api/v1/bibs/recognize

**Request:**
- `file` (required)
- `min_chars` (query, optional, 1–10): override `BIB_MIN_CHARS` (default 2) — minimum digit count for a candidate to qualify

**Scope:** `bibs:read`

**Response:**
```json
{
  "success": true,
  "data": {
    "bibs_detected": 2,
    "detections": [
      {
        "bib_number": "1234",
        "confidence": 0.95,
        "bbox": { "x1": 300, "y1": 200, "x2": 450, "y2": 350, "confidence": 0.91 },
        "all_candidates": [
          { "text": "1234", "confidence": 0.95 },
          { "text": "1284", "confidence": 0.72 }
        ]
      }
    ],
    "image_dimensions": [1920, 1080],
    "processing_time_ms": 95.6,
    "warnings": null
  }
}
```

If the detector is unavailable, `warnings` contains a note and the bounding box covers the whole image.

### POST /api/v1/bibs/recognize/batch

Async batch. Up to `MAX_BATCH_SIZE` (50).

### POST /api/v1/bibs/recognize/mega

Async mega-batch. Up to 500 images via Celery chord.

---

## Async Jobs

### GET /api/v1/jobs/{job_id}

Status and results of an async job. Tenant-isolated — callers only see their own jobs.

**Query params:** `offset` (default 0), `limit` (default 100, max 500) — pagination for the `result` array once the job is complete.

**Response (in progress):**
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-...",
    "status": "processing",
    "progress": 0.45,
    "total_items": 100,
    "processed_items": 45,
    "created_at": "2026-04-20T10:00:00Z",
    "completed_at": null,
    "result": null,
    "error": null
  }
}
```

**Response (completed):**
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-...",
    "status": "completed",
    "progress": 1.0,
    "total_items": 100,
    "processed_items": 100,
    "completed_at": "2026-04-20T10:05:00Z",
    "result": [ /* paginated slice — first `limit` items from offset */ ],
    "result_total": 100,
    "result_offset": 0,
    "result_limit": 100,
    "error": null
  }
}
```

`status` is one of `pending`, `processing`, `completed`, `failed`. The `error` field is populated only for `failed` jobs.

---

## Webhooks

### POST /api/v1/webhooks

Register a callback URL. Only public URLs are accepted — IP-literal URLs that resolve to private ranges are rejected at registration and delivery time (SSRF protection).

**Scope:** `webhooks:write`

**Body (JSON):**
```json
{
  "url": "https://your-backend.com/api/webhooks/ai-results",
  "events": ["job.completed", "job.failed"],
  "secret": "optional_hmac_secret"
}
```

Allowed events today: `job.completed`, `job.failed`. `secret` is encrypted at rest with `WEBHOOK_SECRET_KEY` (Fernet) when configured, otherwise stored in plaintext and the server logs a warning at startup.

**Response (200):**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-...",
    "url": "https://your-backend.com/api/webhooks/ai-results",
    "events": ["job.completed", "job.failed"],
    "active": true,
    "created_at": "2026-04-20T10:00:00Z"
  }
}
```

### GET /api/v1/webhooks

Paginated list (`limit` 1–100 default 50, `offset` default 0) of the caller's webhooks.

**Scope:** `webhooks:read`

### DELETE /api/v1/webhooks/{webhook_id}

Remove a webhook. Tenant-isolated.

**Scope:** `webhooks:write`

### Webhook callback payload

Delivered as `POST <your url>` with `Content-Type: application/json`. The body for job events:

```json
{
  "event": "job.completed",
  "job_id": "550e8400-...",
  "result_count": 100
}
```

(`job.failed` replaces `result_count` with `error`.) If a `secret` was registered, the request includes:

```
X-QuickPitik-Signature: sha256=<HMAC-SHA256(secret, raw body)>
```

Verify with `hmac.compare_digest`. Delivery retries up to 3 times with exponential backoff on HTTP errors.

---

## Metrics

### GET /metrics

Prometheus text format. In `DEBUG=true` mode the endpoint is open. In production it requires an `X-API-Key` header that maps to an active key (401 if missing, 403 if invalid, 503 if the DB cannot validate).

---

## Error Codes

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `ImageValidationError` | Bad image (wrong type, corrupt, too small, too large) |
| 400 | `EMPTY_BATCH` | Batch upload had zero files |
| 400 | `BATCH_TOO_LARGE` | More files than the endpoint allows |
| 400 | `INVALID_WEBHOOK_URL` | Webhook URL failed validation |
| 400 | `INVALID_INPUT` | Malformed form/query input |
| 401 | *(FastAPI `HTTPException`)* | Missing or invalid API key |
| 403 | *(FastAPI `HTTPException`)* | API key lacks the required scope |
| 404 | `NOT_FOUND` | Person, job, or webhook not found / not owned by this key |
| 422 | *(FastAPI validation)* | Pydantic validation error |
| 429 | *(FastAPI `HTTPException`)* | Rate limit exceeded — honor `Retry-After` header |
| 429 | `TOO_MANY_JOBS` | `MAX_ACTIVE_JOBS_PER_KEY` (10) already active |
| 503 | `MODEL_UNAVAILABLE` | Required model not loaded at startup |
| 504 | `REQUEST_TIMEOUT` | Request exceeded 60-second wall clock |

Errors raised from `QuickPitikError` subclasses (`ImageValidationError`, `ModelNotLoadedError`, `AuthenticationError`, `RateLimitExceededError`, `JobNotFoundError`) are translated into the standard envelope by an exception handler in `main.py`. Raw `HTTPException`s from middleware (auth, rate limit) use FastAPI's default `{"detail": ...}` body.

---

## Rate Limits

Rate limiting is **enforced** on every authenticated endpoint via `verify_api_key → check_rate_limit`. A token bucket per API key is maintained in Redis (Lua script, atomic). If Redis is unavailable the limiter becomes a no-op — convenient in dev, expected always-on in production.

| Tier | Max burst | Refill | Effective |
|---|---|---|---|
| `free` | 60 | 1/s | ~60/min |
| `pro` | 300 | 5/s | ~300/min |
| `internal` | 1000 | ~16.7/s | ~1000/min |

Response headers set by `RateLimitHeadersMiddleware`:

- `X-RateLimit-Limit` — burst size for the tier
- `X-RateLimit-Remaining` — tokens left after this call
- `X-RateLimit-Reset` — Unix timestamp when the limiter resets

On 429, the same headers plus `Retry-After` are returned.

Extra backpressure for async batches: a caller with `MAX_ACTIVE_JOBS_PER_KEY` (10) jobs in `pending` or `processing` state gets `429 TOO_MANY_JOBS` on new batch submissions until earlier jobs finish.
