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
| `files` | file[] | yes | Up to `MAX_BATCH_SIZE` images per batch (default 50) |

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

### 3.4 Streaming Batch (Synchronous) — the path the desktop actually uses

**This is the primary desktop path** and has been since 2026-06-03. It needs no Celery
worker, no job row, and no polling: results stream back as NDJSON while the server
works, one JSON object per line.

```
POST /api/v1/blur/detect/stream     (Laplacian, fast gate)
POST /api/v1/blur/classify/stream   (CNN classifier, also accepts blur_type)
```

**Request** (multipart/form-data): `files` — up to 500 images per request
(`STREAM_BATCH_MAX_SIZE` / `STREAM_CLASSIFY_MAX_SIZE`). Over the cap returns
`400 BATCH_TOO_LARGE`.

**Response**: `200` with `Content-Type: application/x-ndjson` and an
`X-Total-Images` header. One line per image, then a final summary line:

```
{"index":0,"filename":"IMG_001.jpg","predicted_class":"sharp","confidence":0.97,"probabilities":{...}}
{"index":1,"filename":"IMG_002.jpg","predicted_class":"motion_blurred","confidence":0.88,"probabilities":{...}}
{"_summary":true,"total":2,"processing_time_ms":412.7}
```

Client rules that matter:

- **Match results by `filename`, not arrival order.** `/detect/stream` completes
  images as they finish, so lines arrive out of order. (`/classify/stream` emits in
  sub-batch order, but do not depend on that.)
- **Treat a missing `_summary` line as a failed run.** The HTTP status is sent
  before processing begins, so a mid-stream server failure cannot become an error
  code — it shows up as a short body. The summary line is the completion signal.
- **A per-image `{"index":…,"filename":…,"error":…}` line is normal** for an
  undecodable file; keep reading, the rest of the batch is unaffected.
- For 5k–10k images, send several concurrent requests of 200–500 each rather than
  one giant request.

**Limits on the stream path** (added 2026-08-14 — it previously validated nothing):

- **Per file**, `MAX_FILE_SIZE` (25 MB) and `MAX_IMAGE_DIMENSION` (12000 px longest
  edge). Both are sized to pass any real camera, so a normal frame is never
  refused. A file that breaks either gets a per-image `error` line at its own
  index — the request still returns 200 and every other image is still scored.
  Content-Type is *not* checked on this path, so posting
  `application/octet-stream` is fine.
- **Per request**, `MAX_REQUEST_BODY` (1 GB total) — refused with `413` before any
  file is read. At the ~5 MB typical original this is roughly 200 images per
  request, which is why the 200–500 chunking advice below matters in practice:
  500 full-resolution originals will exceed it. Send more, smaller requests.

Both `/blur/classify` and `/blur/classify/stream` decode to
`BLUR_CLASSIFY_DECODE_DIM` (640) internally and score identically.

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

### For bulk culling (20+ photos) — use the streaming endpoints

```
1. POST /blur/classify/stream with 200-500 files
2. Read NDJSON lines as they arrive; map each to its photo by `filename`
3. Stop when the {"_summary":true,...} line arrives — no summary means the run failed
4. Repeat with the next chunk (several requests can be in flight at once)
```

Prefer this over the async `/batch` + poll flow for culling work. It needs no
Celery worker, gives incremental progress you can drive a progress bar from, and
avoids the job-row round trips. The async `/batch` and `/mega` endpoints remain
available and are the right choice when the client cannot hold a connection open
for the duration of the run — a queued job survives a disconnect, a stream does not.

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
| `400` | Validation failure on a single-image endpoint -- file over 25 MB (`MAX_FILE_SIZE`), over 12000 px (`MAX_IMAGE_DIMENSION`), wrong type, or corrupt. On `/stream` the same failures arrive as a per-image `error` line instead, with the request still `200`. |
| `413` | Whole request body over 1 GB (`MAX_REQUEST_BODY`) -- send fewer files per request. Not per-image: no file has been read yet, so the response names no filename. |
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
| Max file size | 25 MB (`MAX_FILE_SIZE`) |
| Max image dimension | 12000 px longest edge (`MAX_IMAGE_DIMENSION`) |
| Max total request body | 1 GB (`MAX_REQUEST_BODY`) -- ~200 full-resolution originals |
| Max async batch size | 50 images (`MAX_BATCH_SIZE`) |
| Max stream batch size | 500 images (`STREAM_*_MAX_SIZE`), subject to the 1 GB body limit |
| Supported formats | JPEG, PNG, WebP |
| Fast check | `POST /blur/detect` |
| CNN classify | `POST /blur/classify` |
| **Stream detect (primary)** | `POST /blur/detect/stream` |
| **Stream classify (primary)** | `POST /blur/classify/stream` |
| Batch detect | `POST /blur/detect/batch` |
| Batch classify | `POST /blur/classify/batch` |
| Poll job | `GET /jobs/{job_id}` |
