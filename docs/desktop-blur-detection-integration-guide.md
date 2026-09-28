# Desktop App - Blur Detection API Integration Guide

This guide documents the API and the BatchMyPhotos invited beta client. The Electron main process calls ai-api directly; its Express backend is not in the blur path. See `BatchMyPhotos/desktop/docs/blur-detection.md` for the app workflow and privacy rules.

## Prerequisites

The ai-api service must be running with PostgreSQL and Redis:

```bash
cd ai-api
docker compose up db redis -d
alembic upgrade head
make dev       # API on http://localhost:8000
make celery    # only needed for async batch endpoints, not the desktop beta
```

---

## 1. Provision the Desktop API Key

Use the private `docs/api-keys.md` workflow to create and register a key in
the target ai-api database. Give it exactly `["blur:read","jobs:read"]`;
`jobs:read` covers polling if async is enabled later. Store the plaintext key
in the desktop's secret configuration. The database stores only its hash. Issue
one restricted, revocable staging key per invited tester; record its owner,
rate tier, and revocation procedure privately. The packaged beta accepts the
key after installation and encrypts it through Electron `safeStorage`; it is
never bundled into the installer. Source runs use the development environment
configuration described in `BatchMyPhotos/desktop/src/main/config.js`.
`gen_api_key.py` prints a generic wildcard SQL example, so do not execute
that SQL unchanged for a desktop key. Do not run `seed_db.py` in staging or
production.

### Rate Tiers

| Tier | Requests/min | Use case |
|------|-------------|----------|
| `free` | 60 | Testing |
| `pro` | 300 | Higher-volume approved clients |
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
| `file` | file | yes | -- | Image file (JPEG, PNG, WebP) up to 25 MB (`MAX_FILE_SIZE`, raised 2026-08-14) |
| `threshold` | float | no | `BLUR_THRESHOLD` (`100.0` by default) | Blur threshold (1.0 - 10000.0). Lower marks fewer images blurry |
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
  "confidence": 0.573,
  "metrics": {
    "laplacian_variance": 42.7,
    "hf_ratio": 0.12,
    "confidence": 0.573
  },
  "image_dimensions": [640, 480],
  "processing_time_ms": 23.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_blurry` | bool | `true` if image is blurry |
| `confidence` | float | Threshold-ratio heuristic, not a measured probability; `1.0` can mean score saturation |
| `metrics.laplacian_variance` | float | Texture/edge score at the classical decode scale -- lower = blurrier |
| `metrics.hf_ratio` | float | Informational high-frequency content ratio (FFT); does not affect `is_blurry` |
| `image_dimensions` | [w, h] | Processed image width and height in pixels (long edge at most `BLUR_DETECT_DECODE_DIM`) |
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
| `file` | file | yes | -- | Image file (JPEG, PNG, WebP) up to 25 MB (`MAX_FILE_SIZE`, raised 2026-08-14) |
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

`/blur/classify/stream` is the primary desktop beta path. The detect stream is a
separate API endpoint and is not called by this client. Streaming needs no Celery
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
{"_summary":true,"total":2,"successful":2,"errors":0,"complete":true,"processing_time_ms":412.7}
```

Client rules that matter:

- **Match results by the uploaded index, not arrival order or the original filename.**
  The desktop sends each image under its chunk index as the multipart filename,
  checks both fields in each row, then maps the index back to its file group.
- **Treat a missing `_summary` line or `complete:false` as a failed run.** The HTTP status is sent
  before processing begins, so a mid-stream server failure cannot become an error
  code — it shows up as a short body. The summary line is the completion signal.
- **A per-image `{"index":…,"filename":…,"error":…}` line is normal** for an
  undecodable file; keep reading, the rest of the batch is unaffected.
- Respect server `BLUR_BUSY` responses and `Retry-After` before starting another
  stream. The server admission limit is configured in `ai-api/src/config.py`.
- A busy rejection can close the connection while a large multipart body is still
  uploading, so the client may see a network error instead of the `503` response.
  Treat either outcome as a failed stream; wait before retrying unaccounted images.

**Limits on the stream path** (added 2026-08-14 — it previously validated nothing):

- **Per file**, `MAX_FILE_SIZE` (25 MB), `MAX_IMAGE_DIMENSION` (12000 px longest
  edge), and 24 megapixels on the dedicated blur service. A file that breaks a
  limit gets a per-image `error` line at its own
  index — the request still returns 200 and every other image is still scored.
  Content-Type is *not* checked on this path, so posting
  `application/octet-stream` is fine.
- **Per request**, `BLUR_MAX_REQUEST_BODY` (128 MiB) on the dedicated blur
  service — refused with `413` before multipart parsing. The shared all-model
  service retains its separate `MAX_REQUEST_BODY` ceiling (2 GiB). Split a
  stream if its prepared JPEGs approach the blur service's body limit.

Both `/blur/classify` and `/blur/classify/stream` decode to
`BLUR_CLASSIFY_DECODE_DIM` (640) internally and score identically.

---

## 4. BatchMyPhotos invited beta

The photographer selects a folder, enables blur suggestions, reviews the upload
notice, and clicks **Start Analysis**. Import, folder changes, and sensitivity or
category changes do not silently upload images. For each group the main process
chooses one supported raster companion; RAW-only groups remain unanalysed. Sharp
auto-orients and resizes it, then encodes a JPEG before upload. Preparation,
request count/byte limits, concurrency, and deadlines live in
`BatchMyPhotos/desktop/src/main/constants.js` and `blurDetectionService.js`.

The main process sends JPEGs to `/blur/classify/stream`, checks every row and
the terminal summary, and reports progress per accounted image. A stream with
a missing summary, malformed or duplicate rows, or contradictory counts fails
visibly; its partial rows do not count as a successful run. After a valid
summary, omitted rows use the single `/blur/classify` endpoint. A 404/405 stream route
also uses the single endpoint. Retryable stream failures get one bounded retry;
individual image errors become **unanalysed**, never sharp. There is no desktop
Celery job, poll loop, or `/blur/detect` pre-check.

Sensitivity and selected categories are applied locally to the returned class
probabilities. The authoritative mapping is `mapClassification` and
`SENSITIVITY_TO_THRESHOLD` in `BatchMyPhotos/desktop/src/main/blurDetectionService.js`.
Suggestions remain advisory: they do not exclude JPG/RAW groups from preview,
move or copy operations, or rollback. Service errors do not silently switch to
the local Laplacian implementation.

The packaged beta enables blur only with a valid staging-only `blur-beta.json`
resource containing an HTTPS origin. It ignores packaged environment overrides
for the origin and key. Each tester enters a restricted staging key in Settings;
the main process stores it encrypted and returns only key presence to the
renderer. Source runs use `BATCH_BLUR_*` development settings. See
`BatchMyPhotos/desktop/src/main/config.js` and `blurBetaKeyStore.js`.

Analysis consent is separate from feedback consent. Labelling a suggestion
does not submit it. **Submit this example** sends only the selected prepared
JPEG and limited metadata through the existing signed-in Supabase session;
the folder path, original filename, RAW companion, and EXIF are excluded.
Packaged sign-in and feedback still target production Supabase. Local policy
tests alone cannot prove production bucket, policy, or migration readiness; see
`BatchMyPhotos/desktop/docs/blur-detection.md` for the retention and deletion
procedure.

| Condition | Desktop behavior |
|-----------|------------------|
| `401` / `403` | Fail analysis; replace or correct the tester's key. |
| `429` / retryable `503` | Honor a bounded `Retry-After`, then retry once; otherwise fail visibly. |
| Request `413` | Fail analysis; the desktop's prepared-JPEG byte budget should keep requests below the configured server limit. |
| Per-image validation error | Mark that group unanalysed and continue. |
| Timeout, disconnect, invalid or unfinished stream | Fail analysis visibly; no optimistic partial result. |

---

## 5. Quick Reference

| What | Value |
|------|-------|
| Base URL | `http://localhost:8000/api/v1` |
| Auth header | `X-API-Key: sk_test_...` |
| Required scopes | `blur:read`, `jobs:read` |
| Max file size | 25 MB (`MAX_FILE_SIZE`) |
| Max image dimension | 12000 px longest edge (`MAX_IMAGE_DIMENSION`) |
| Max total request body | 128 MiB on dedicated blur (`BLUR_MAX_REQUEST_BODY`); 2 GiB on shared all-model (`MAX_REQUEST_BODY`) |
| Max async batch size | 50 images (`MAX_BATCH_SIZE`) |
| Max stream batch size | 500 images (`STREAM_*_MAX_SIZE`), subject to the service body limit |
| Supported formats | JPEG, PNG, WebP |
| Fast check | `POST /blur/detect` |
| CNN classify | `POST /blur/classify` |
| Stream detect (API only; not used by beta) | `POST /blur/detect/stream` |
| **Desktop primary path** | `POST /blur/classify/stream` |
| Batch detect/classify | Optional async; disabled on the initial dedicated Railway blur service |
| Poll job (API only; not used by beta) | `GET /jobs/{job_id}` |
