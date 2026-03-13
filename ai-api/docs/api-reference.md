# API Reference

All endpoints are prefixed with `/api/v1/`. All responses use a standard envelope:

```json
{
  "success": true,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-21T12:34:56.789Z",
  "data": { ... },
  "error": null
}
```

On error:
```json
{
  "success": false,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-21T12:34:56.789Z",
  "data": null,
  "error": {
    "code": "IMAGE_VALIDATION_ERROR",
    "message": "File exceeds 10MB limit"
  }
}
```

**Authentication**: All endpoints (except health) require an `X-API-Key` header.

---

## Health

### GET /api/v1/health

Liveness probe. Returns 200 if the process is running.

**Response:**
```json
{
  "status": "alive",
  "version": "1.0.0",
  "environment": "development"
}
```

### GET /api/v1/health/ready

Readiness probe. Checks that all dependencies are available.

**Response:**
```json
{
  "success": true,
  "data": {
    "models_loaded": true,
    "database": true,
    "redis": true
  }
}
```

---

## Blur Detection

### POST /api/v1/blur/detect

Upload an image to check if it's blurry.

**Request:**
- Content-Type: `multipart/form-data`
- `file` (required): Image file (JPEG, PNG, or WebP)
- `threshold` (optional, query param): Laplacian variance threshold. Default: 100.0. Range: 1.0-10000.0. Lower = stricter.
- `include_metrics` (optional, query param): Return detailed metrics. Default: true.

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/v1/blur/detect \
  -H "X-API-Key: sk_dev_eventai_test_key_12345" \
  -F "file=@photo.jpg" \
  -G -d "threshold=100" -d "include_metrics=true"
```

**Response (200):**
```json
{
  "success": true,
  "request_id": "abc-123",
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

**How to interpret results:**
- `is_blurry`: true/false verdict
- `laplacian_variance`: Higher = sharper image. Below threshold = blurry.
- `hf_ratio`: Ratio of high-frequency content (0-1). Lower = blurrier.
- `confidence`: How confident the detection is (0-1).

### POST /api/v1/blur/classify

Classify an image into blur categories using a CNN model (YOLOv8n-cls). Requires the trained ONNX model to be loaded at startup.

**Two modes:**
- **Full classification** (default): Returns the predicted class and probabilities for all 4 categories.
- **Targeted detection**: When `blur_type` is provided, returns a binary Detected/Not Detected response for that specific blur type.

**Request:**
- Content-Type: `multipart/form-data`
- `file` (required): Image file (JPEG, PNG, or WebP)
- `blur_type` (optional, query param): Specific blur type to detect. Options: `defocused_object_portrait`, `defocused_blurred`, `motion_blurred`

**Example (curl) — full classification:**
```bash
curl -X POST http://localhost:8000/api/v1/blur/classify \
  -H "X-API-Key: sk_dev_eventai_test_key_12345" \
  -F "file=@photo.jpg"
```

**Response (200) — full classification:**
```json
{
  "success": true,
  "request_id": "abc-123",
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

**Example (curl) — targeted detection:**
```bash
curl -X POST "http://localhost:8000/api/v1/blur/classify?blur_type=defocused_object_portrait" \
  -H "X-API-Key: sk_dev_eventai_test_key_12345" \
  -F "file=@photo.jpg"
```

**Response (200) — targeted detection:**
```json
{
  "success": true,
  "request_id": "abc-123",
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

**How to interpret results:**
- **Full classification mode** (`blur_type` omitted):
  - `predicted_class`: The top predicted category (sharp, defocused_object_portrait, defocused_blurred, motion_blurred)
  - `confidence`: Model confidence for the top prediction (0-1)
  - `probabilities`: Probability distribution across all 4 classes
- **Targeted detection mode** (`blur_type` provided):
  - `detected`: true/false — whether the selected blur type is the top prediction
  - `confidence`: Model confidence for the top prediction
  - `blur_type_probability`: Probability the model assigned to the selected blur type

**Error (503):** Returns `MODEL_UNAVAILABLE` if the blur classifier ONNX model is not loaded.

### POST /api/v1/blur/classify/batch

Submit a batch of images for async blur classification via Celery.

**Request:**
- Content-Type: `multipart/form-data`
- `files` (required): Multiple image files
- `blur_type` (optional, query param): Specific blur type to detect per image

**Response (202):**
```json
{
  "success": true,
  "request_id": "abc-123",
  "data": {
    "job_id": "550e8400-...",
    "status": "pending",
    "total_items": 10,
    "poll_url": "/api/v1/jobs/550e8400-..."
  }
}
```

Poll `GET /api/v1/jobs/{job_id}` for results.

---

## Face Recognition

### POST /api/v1/faces/detect

Detect faces in an image. Returns bounding boxes and landmarks.

**Request:**
- `file` (required): Image file

**Response (200):**
```json
{
  "success": true,
  "data": {
    "faces_detected": 2,
    "faces": [
      {
        "bbox": {
          "x1": 120.5,
          "y1": 80.3,
          "x2": 250.1,
          "y2": 280.7,
          "confidence": 0.98
        },
        "landmarks": [[145.2, 150.1], [210.3, 148.9], ...]
      }
    ],
    "image_dimensions": [1920, 1080],
    "processing_time_ms": 85.2
  }
}
```

### POST /api/v1/faces/enroll

Register a person's face in the database. Detects faces, extracts embeddings, and stores them.

Faces below the minimum enrollment confidence (`FACE_MIN_ENROLLMENT_CONFIDENCE`, default 0.7) are skipped to prevent low-quality embeddings from degrading search accuracy.

**Request:**
- `file` (required): Image file containing the person's face
- `person_name` (required, form field): Name of the person
- `person_id` (optional, form field): UUID of an existing person (to add more photos)

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/faces/enroll \
  -H "X-API-Key: sk_dev_eventai_test_key_12345" \
  -F "file=@john_photo.jpg" \
  -F "person_name=John Doe"
```

**Response (200):**
```json
{
  "success": true,
  "data": {
    "person_id": "550e8400-e29b-41d4-a716-446655440000",
    "person_name": "John Doe",
    "faces_enrolled": 1,
    "embeddings_stored": 1,
    "processing_time_ms": 120.5
  }
}
```

**Error — Low Quality (200, success: false):**
```json
{
  "success": false,
  "error": {
    "code": "LOW_QUALITY",
    "message": "All 1 detected face(s) were below the minimum enrollment confidence of 0.7"
  }
}
```

### POST /api/v1/faces/search

Upload a photo and find matching people in the database.

**Request:**
- `file` (required): Image file
- `threshold` (optional, query): Minimum similarity score (0-1). Default: 0.4
- `top_k` (optional, query): Maximum matches per face. Default: 10

**Response (200):**
```json
{
  "success": true,
  "data": {
    "faces_detected": 1,
    "matches": [
      {
        "person_id": "550e8400-e29b-41d4-a716-446655440000",
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

### POST /api/v1/faces/compare

Compare two images for 1:1 face verification (are they the same person?).

**Request:**
- `file1` (required): First image
- `file2` (required): Second image

**Response (200):**
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

### GET /api/v1/faces/persons/{person_id}

Get metadata about an enrolled person.

**Response (200):**
```json
{
  "success": true,
  "data": {
    "person_id": "550e8400-...",
    "person_name": "John Doe",
    "embeddings_count": 3,
    "created_at": "2026-02-21T10:00:00Z",
    "updated_at": "2026-02-21T10:00:00Z"
  }
}
```

### DELETE /api/v1/faces/persons/{person_id}

Remove a person and all their stored embeddings (GDPR right-to-erasure).

**Response (200):**
```json
{
  "success": true,
  "data": { "deleted": true, "person_id": "550e8400-..." }
}
```

---

## Bib Number Recognition

### POST /api/v1/bibs/recognize

Upload an image to detect and read bib numbers.

**Request:**
- `file` (required): Image file containing runners with bibs

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/bibs/recognize \
  -H "X-API-Key: sk_dev_eventai_test_key_12345" \
  -F "file=@race_photo.jpg"
```

**Response (200):**
```json
{
  "success": true,
  "data": {
    "bibs_detected": 2,
    "detections": [
      {
        "bib_number": "1234",
        "confidence": 0.95,
        "bbox": { "x1": 300, "y1": 200, "x2": 450, "y2": 350 },
        "all_candidates": [
          { "text": "1234", "confidence": 0.95 },
          { "text": "1284", "confidence": 0.72 }
        ]
      },
      {
        "bib_number": "567",
        "confidence": 0.88,
        "bbox": { "x1": 600, "y1": 180, "x2": 720, "y2": 300 },
        "all_candidates": [
          { "text": "567", "confidence": 0.88 }
        ]
      }
    ],
    "image_dimensions": [1920, 1080],
    "processing_time_ms": 95.6
  }
}
```

---

## Async Jobs

### GET /api/v1/jobs/{job_id}

Check the status of a batch processing job.

**Response (200) - In progress:**
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-...",
    "status": "processing",
    "progress": 0.45,
    "total_items": 100,
    "processed_items": 45,
    "created_at": "2026-02-21T10:00:00Z",
    "completed_at": null,
    "result": null,
    "error": null
  }
}
```

**Response (200) - Completed:**
```json
{
  "success": true,
  "data": {
    "status": "completed",
    "progress": 1.0,
    "total_items": 100,
    "processed_items": 100,
    "completed_at": "2026-02-21T10:05:00Z",
    "result": [ ... ]
  }
}
```

---

## Webhooks

### POST /api/v1/webhooks

Register a URL to receive callbacks when events occur.

**Request body (JSON):**
```json
{
  "url": "https://your-backend.com/api/webhooks/ai-results",
  "events": ["job.completed", "job.failed"],
  "secret": "your_webhook_secret"
}
```

### GET /api/v1/webhooks

List all your registered webhooks.

### DELETE /api/v1/webhooks/{webhook_id}

Remove a webhook.

**Webhook callback format (sent to your URL):**
```json
{
  "event": "job.completed",
  "job_id": "550e8400-...",
  "results": [ ... ],
  "completed_at": "2026-02-21T12:34:56Z"
}
```

If you provided a secret, the request includes an `X-EventAI-Signature` header with an HMAC-SHA256 signature of the body.

---

## Error Codes

| HTTP Status | Code | Meaning |
|---|---|---|
| 400 | `IMAGE_VALIDATION_ERROR` | Bad image (wrong type, too large, corrupt) |
| 401 | `AuthenticationError` | Missing or invalid API key |
| 403 | Insufficient permissions | API key doesn't have required scope |
| 404 | `NOT_FOUND` | Person, job, or webhook not found |
| 413 | `IMAGE_VALIDATION_ERROR` | File exceeds size limit |
| 429 | `RateLimitExceededError` | Too many requests (check Retry-After header) |
| 503 | `MODEL_UNAVAILABLE` | ML model failed to load at startup |

---

## Rate Limits

| Tier | Requests/minute | Who |
|---|---|---|
| Free | 60 | Default for new API keys |
| Pro | 300 | Upgraded keys |
| Internal | 1000 | Backend-to-backend communication |

**Note:** The rate limiter code exists (`src/middleware/rate_limit.py`) but is **not yet wired into endpoints** (pending Phase 6.5). Once activated, rate-limited (429) responses will include these headers:
- `X-RateLimit-Limit`: Maximum requests allowed in the window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when the limit resets
- `Retry-After`: Seconds until the next request is allowed
