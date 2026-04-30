# Desktop App - Blur Detection API Integration Guide

This guide explains how to connect the desktop (Electron) app to the QuickPitik blur detection API using your own API key.

## Prerequisites

The ai-api service must be running with PostgreSQL and Redis:

```bash
cd ai-api
docker compose up db redis -d
alembic upgrade head
make dev       # API on http://localhost:8000
make celery    # only needed for batch endpoints
```

---

## 1. Generate and Register Your API Key

### Step 1: Generate a key

Run from the `ai-api/` directory:

```bash
python gen_api_key.py
```

Output:

```
Your API key: sk_test_<random_hex>

Run this SQL:
INSERT INTO api_keys (id, key_hash, name, scopes, rate_tier, active)
VALUES ('<uuid>', '<sha256_hash>', 'Desktop App', '["*"]', 'pro', true);
```

**Save the `sk_test_...` key immediately** -- it cannot be recovered. Only the hash is stored in the database.

### Step 2: Insert the key into the database

If running PostgreSQL via Docker Compose:

```bash
docker compose exec db psql -U postgres -d quickpitik -c "<the SQL from step 1>"
```

Or run the helper script if one was already generated:

```bash
python insert_key.py
```

### Scopes

The `["*"]` scope grants access to all endpoints. For a desktop app that only needs blur detection, you can restrict to:

```json
["blur:read"]
```

This allows `/blur/detect`, `/blur/classify`, and their batch variants, but blocks face and bib endpoints.

### Rate Tiers

| Tier | Requests/min | Use case |
|------|-------------|----------|
| `free` | 60 | Testing |
| `pro` | 300 | Desktop app production |
| `internal` | 1000 | Backend-to-backend |

---

## 2. Authentication

Every request must include the API key in the `X-API-Key` header:

```
X-API-Key: sk_test_abc123...
```

Responses on auth failure:

| Status | Meaning |
|--------|---------|
| `401` | Missing or invalid API key |
| `403` | Key is valid but lacks the required scope (needs `blur:read`) |
| `429` | Rate limit exceeded -- check `Retry-After` header |

---

## 3. Blur Detection Endpoints

**Base URL:** `http://localhost:8000/api/v1`

All responses use the standard envelope:

```json
{
  "success": true,
  "request_id": "abc-123",
  "timestamp": "2026-03-26T12:00:00Z",
  "data": { ... },
  "error": null
}
```

On error, `success` is `false` and `error` contains `{"code": "...", "message": "..."}`.

---

### 3.1 Quick Blur Check (Laplacian)

Fast binary yes/no blur detection. Always available -- no trained model required.

```
POST /api/v1/blur/detect
```

**Request** (multipart/form-data):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | yes | -- | Image file (JPEG, PNG, WebP) up to 10 MB |
| `threshold` | float | no | `100.0` | Blur threshold (1.0 - 10000.0). Lower = stricter |
| `include_metrics` | bool | no | `true` | Include Laplacian variance and FFT metrics |

**Example (cURL):**

```bash
curl -X POST http://localhost:8000/api/v1/blur/detect \
  -H "X-API-Key: sk_test_abc123..." \
  -F "file=@photo.jpg"
```

**Response (`data`):**

```json
{
  "is_blurry": true,
  "confidence": 0.85,
  "metrics": {
    "laplacian_variance": 42.7,
    "hf_ratio": 0.12,
    "confidence": 0.85
  },
  "image_dimensions": [4032, 3024],
  "processing_time_ms": 23.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_blurry` | bool | `true` if image is blurry |
| `confidence` | float | 0.0 - 1.0 confidence score |
| `metrics.laplacian_variance` | float | Sharpness score -- lower = blurrier |
| `metrics.hf_ratio` | float | High-frequency content ratio (FFT) |
| `image_dimensions` | [w, h] | Image width and height in pixels |
| `processing_time_ms` | float | Server-side processing time |

---

### 3.2 Blur Classification (CNN -- 4 classes)

Classifies the image into one of four blur types using a trained YOLOv8n-cls model. Requires the ONNX model file at `models/blur_classifier/blur_classifier.onnx`.

```
POST /api/v1/blur/classify
```

**Request** (multipart/form-data):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | yes | -- | Image file (JPEG, PNG, WebP) up to 10 MB |
| `blur_type` | string | no | -- | Target a specific blur type (see below) |

#### Mode A: Full Classification (no `blur_type` param)

Returns the predicted class and probability breakdown for all four classes.

```bash
curl -X POST http://localhost:8000/api/v1/blur/classify \
  -H "X-API-Key: sk_test_abc123..." \
  -F "file=@photo.jpg"
```

**Response (`data`):**

```json
{
  "predicted_class": "motion_blurred",
  "confidence": 0.92,
  "probabilities": {
    "sharp": 0.03,
    "defocused_object_portrait": 0.02,
    "defocused_blurred": 0.03,
    "motion_blurred": 0.92
  },
  "image_dimensions": [4032, 3024],
  "processing_time_ms": 45.2
}
```

The four classes:

| Class | Meaning |
|-------|---------|
| `sharp` | Image is in focus |
| `defocused_object_portrait` | Subject out of focus (portrait/bokeh style) |
| `defocused_blurred` | General out-of-focus blur |
| `motion_blurred` | Motion blur from camera or subject movement |

#### Mode B: Targeted Detection (with `blur_type` param)

Pass `blur_type` as a query parameter to get a simple Detected / Not Detected answer for one specific blur type.

Valid values: `defocused_object_portrait`, `defocused_blurred`, `motion_blurred`

```bash
curl -X POST "http://localhost:8000/api/v1/blur/classify?blur_type=motion_blurred" \
  -H "X-API-Key: sk_test_abc123..." \
  -F "file=@photo.jpg"
```

**Response (`data`):**

```json
{
  "blur_type": "motion_blurred",
  "detected": true,
  "confidence": 0.92,
  "blur_type_probability": 0.92,
  "image_dimensions": [4032, 3024],
  "processing_time_ms": 44.8
}
```

| Field | Type | Description |
|-------|------|-------------|
| `blur_type` | string | The blur type you queried |
| `detected` | bool | `true` if this blur type was detected |
| `confidence` | float | Model confidence for the predicted class |
| `blur_type_probability` | float | Probability specifically for the queried blur type |

---

### 3.3 Batch Blur Detection (Async)

For processing multiple images at once. Requires the Celery worker to be running (`make celery`).

#### Submit batch:

```
POST /api/v1/blur/detect/batch     (Laplacian)
POST /api/v1/blur/classify/batch   (CNN classifier, also accepts blur_type param)
```

**Request** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | file[] | yes | Up to 20 images per batch (configurable via `MAX_BATCH_SIZE`) |

**Response** (HTTP 202):

```json
{
  "success": true,
  "request_id": "abc-123",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending",
    "total_images": 5,
    "poll_url": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000"
  }
}
```

#### Poll for results:

```
GET /api/v1/jobs/{job_id}
```

```bash
curl http://localhost:8000/api/v1/jobs/550e8400-... \
  -H "X-API-Key: sk_test_abc123..."
```

Job statuses: `pending` -> `processing` -> `completed` or `failed`

When `completed`, the `data.result` field contains an array of per-image results.

---

## 4. Desktop App Integration Recommendations

### Suggested workflow for the desktop app

```
1. User imports photos into desktop app
2. Desktop app iterates each photo:
   a. POST /blur/detect  (fast Laplacian check ~20ms)
   b. If is_blurry == true:
      - POST /blur/classify  (CNN classification ~45ms)
      - Tag photo with blur type (motion_blurred, defocused_object_portrait, etc.)
   c. If is_blurry == false:
      - Tag photo as sharp, skip classification
3. Display results to user with blur type labels
```

This two-step approach saves time: the fast Laplacian gate filters out sharp images cheaply, and only blurry images go through the heavier CNN classifier.

### For bulk uploads (20+ photos)

Use the batch endpoints instead to avoid per-request overhead:

```
1. POST /blur/detect/batch with up to 20 files
2. Poll GET /jobs/{job_id} until completed
3. For images flagged as blurry, POST /blur/classify/batch
4. Poll again for classification results
```

### API key storage

- Store the API key securely using the OS keychain (e.g., Electron's `safeStorage` API)
- Never hardcode the key in source code or commit it to version control
- Pass it in every request via the `X-API-Key` header

### Error handling

| HTTP Status | Action |
|-------------|--------|
| `200` | Success -- parse `data` |
| `202` | Batch accepted -- poll the `poll_url` |
| `401` | Invalid key -- prompt user to re-enter |
| `403` | Scope issue -- key needs `blur:read` |
| `413` | File too large -- max 10 MB per image |
| `429` | Rate limited -- wait for `Retry-After` seconds, then retry |
| `503` | Model not loaded -- blur classifier may not be deployed yet, fall back to `/blur/detect` |

### Confidence threshold guidance

The minimum confidence floor is configurable server-side (`BLUR_DETECTION_MIN_CONFIDENCE`, default 0.5). On the desktop side, you may want an additional UI threshold:

- **confidence >= 0.8**: High confidence -- auto-tag
- **0.5 <= confidence < 0.8**: Medium -- show to user for review
- **confidence < 0.5**: Low -- treat as uncertain

---

## 5. Quick Reference

| What | Value |
|------|-------|
| Base URL | `http://localhost:8000/api/v1` |
| Auth header | `X-API-Key: sk_test_...` |
| Required scope | `blur:read` |
| Max file size | 10 MB |
| Max batch size | 20 images |
| Supported formats | JPEG, PNG, WebP |
| Fast check | `POST /blur/detect` |
| CNN classify | `POST /blur/classify` |
| Batch detect | `POST /blur/detect/batch` |
| Batch classify | `POST /blur/classify/batch` |
| Poll job | `GET /jobs/{job_id}` |
