# EventAI AI-API Integration Guide

**Last updated:** 2026-03-26
**For:** Desktop app (Electron), Web app, Mobile app — any client calling the AI-API

---

## 1. Getting the API Running

### Start infrastructure

```bash
cd ai-api

# Terminal 1 — database + redis
docker compose up db redis -d

# Run migrations
alembic upgrade head
```

### Start the API server

```bash
# Terminal 2 — API
make dev
# or: uvicorn src.main:create_app --factory --reload --port 8000
```

### Start the Celery worker (only needed for batch endpoints)

```bash
# Terminal 3 — worker
celery -A src.workers.celery_app worker -l info -Q default,blur,face,bib -c 2
```

### Verify

```bash
curl http://localhost:8000/api/v1/health
# {"status": "alive", "version": "1.0.0"}
```

---

## 2. Create an API Key

Generate a key and insert it into the database:

```bash
python -c "
import hashlib, uuid, secrets
raw_key = 'sk_test_' + secrets.token_hex(16)
key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
print(f'Your API key: {raw_key}')
print()
print(f'Run this SQL:')
print(f\"\"\"INSERT INTO api_keys (id, key_hash, name, scopes, rate_tier, active)
VALUES ('{uuid.uuid4()}', '{key_hash}', 'Desktop App', '[\"*\"]', 'pro', true);\"\"\")
"
```

Insert into the database:

```bash
docker compose exec db psql -U postgres -d eventai -c "INSERT INTO api_keys ..."
```

Save the `sk_test_...` key. That's what your app sends in headers.

> **Note:** In DEBUG mode (`DEBUG=true`), auth is bypassed when no key is provided. For real testing, always use a key.

---

## 3. Authentication

Every request must include the API key header:

```
X-API-Key: sk_test_YOUR_KEY
```

Every response follows the same envelope:

```json
{
  "success": true,
  "request_id": "uuid",
  "timestamp": "2026-03-26T10:00:00Z",
  "data": { ... },
  "error": null
}
```

On failure:

```json
{
  "success": false,
  "request_id": "uuid",
  "timestamp": "2026-03-26T10:00:00Z",
  "data": null,
  "error": { "code": "ERROR_CODE", "message": "Human-readable message" }
}
```

Always check `json.success` first, then read `json.data` or `json.error`.

---

## 4. Endpoint Reference

### 4.1 Blur Detection

#### Quick blur check (is it blurry? yes/no)

```
POST /api/v1/blur/detect
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | — | Image file (JPEG, PNG, WebP) |
| `threshold` | Query float | No | 100.0 | Laplacian variance threshold (1.0 - 10000.0) |
| `include_metrics` | Query bool | No | true | Include detailed metrics |

```bash
curl -X POST http://localhost:8000/api/v1/blur/detect \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@photo.jpg"
```

Response `data`:

```json
{
  "is_blurry": false,
  "confidence": 0.85,
  "metrics": {
    "laplacian_variance": 185.4,
    "hf_ratio": 0.72,
    "confidence": 0.85
  },
  "image_dimensions": [1920, 1080],
  "processing_time_ms": 45.2
}
```

#### Blur type classification (what KIND of blur?)

```
POST /api/v1/blur/classify
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | — | Image file |
| `blur_type` | Query string | No | null | Specific type to detect: `defocused_object_portrait`, `defocused_blurred`, `motion_blurred` |

```bash
# Full classification
curl -X POST http://localhost:8000/api/v1/blur/classify \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@photo.jpg"

# Targeted detection (is it motion blurred?)
curl -X POST "http://localhost:8000/api/v1/blur/classify?blur_type=motion_blurred" \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@photo.jpg"
```

Full classification response `data`:

```json
{
  "predicted_class": "sharp",
  "confidence": 0.95,
  "probabilities": {
    "sharp": 0.95,
    "defocused_blurred": 0.02,
    "defocused_object_portrait": 0.02,
    "motion_blurred": 0.01
  },
  "image_dimensions": [1920, 1080],
  "processing_time_ms": 120.5
}
```

Targeted detection response `data`:

```json
{
  "blur_type": "motion_blurred",
  "detected": false,
  "confidence": 0.95,
  "blur_type_probability": 0.01,
  "image_dimensions": [1920, 1080],
  "processing_time_ms": 115.3
}
```

---

### 4.2 Face Recognition

#### Enroll a runner (register their face)

```
POST /api/v1/faces/enroll
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | — | Clear photo with the runner's face |
| `person_name` | Form string | Yes | — | Runner's name (1-255 chars) |
| `person_id` | Form string | No | null | Existing person UUID (to add more photos) |
| `event_id` | Form string | No | null | Event identifier (e.g., "cebu-marathon-2026") |

```bash
curl -X POST http://localhost:8000/api/v1/faces/enroll \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@runner_selfie.jpg" \
  -F "person_name=Juan Dela Cruz" \
  -F "event_id=cebu-marathon-2026"
```

Response `data`:

```json
{
  "person_id": "a1b2c3d4-...",
  "person_name": "Juan Dela Cruz",
  "event_id": "cebu-marathon-2026",
  "faces_enrolled": 1,
  "embeddings_stored": 1,
  "processing_time_ms": 450.3
}
```

> Save the `person_id` — you need it to add more photos or manage the person later.

To add more photos for the same person (improves accuracy):

```bash
curl -X POST http://localhost:8000/api/v1/faces/enroll \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@runner_another_angle.jpg" \
  -F "person_name=Juan Dela Cruz" \
  -F "person_id=a1b2c3d4-..."
```

#### Search for a runner in event photos

```
POST /api/v1/faces/search
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | — | Event photo to search |
| `threshold` | Query float | No | 0.4 | Similarity threshold (0.0 - 1.0). Lower = more results, more false positives |
| `top_k` | Query int | No | 10 | Max matches per face (1-100) |
| `event_id` | Query string | No | null | Scope search to this event only |

```bash
curl -X POST "http://localhost:8000/api/v1/faces/search?threshold=0.4&event_id=cebu-marathon-2026" \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@event_photo.jpg"
```

Response `data`:

```json
{
  "faces_detected": 3,
  "matches": [
    {
      "person_id": "a1b2c3d4-...",
      "person_name": "Juan Dela Cruz",
      "similarity": 0.82,
      "bbox": { "x1": 100, "y1": 50, "x2": 200, "y2": 180, "confidence": 0.99 }
    }
  ],
  "unmatched_faces": [
    {
      "bbox": { "x1": 400, "y1": 60, "x2": 480, "y2": 170, "confidence": 0.95 },
      "landmarks": null
    }
  ],
  "processing_time_ms": 850.7
}
```

#### Compare two faces (1:1 verification)

```
POST /api/v1/faces/compare
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file1` | File | Yes | First face photo |
| `file2` | File | Yes | Second face photo |

```bash
curl -X POST http://localhost:8000/api/v1/faces/compare \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file1=@photo_a.jpg" \
  -F "file2=@photo_b.jpg"
```

Response `data`:

```json
{
  "is_match": true,
  "similarity": 0.87,
  "face1": { "bbox": { "x1": 50, "y1": 30, "x2": 180, "y2": 200, "confidence": 0.99 } },
  "face2": { "bbox": { "x1": 60, "y1": 40, "x2": 190, "y2": 210, "confidence": 0.98 } },
  "processing_time_ms": 620.4
}
```

#### Detect faces only (no search)

```
POST /api/v1/faces/detect
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file |

```bash
curl -X POST http://localhost:8000/api/v1/faces/detect \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@photo.jpg"
```

Response `data`:

```json
{
  "faces_detected": 2,
  "faces": [
    {
      "bbox": { "x1": 100, "y1": 50, "x2": 200, "y2": 180, "confidence": 0.99 },
      "landmarks": [[120.5, 80.2], [160.3, 79.8], [140.1, 110.0], [125.0, 140.5], [155.0, 139.8]]
    }
  ],
  "image_dimensions": [1920, 1080],
  "processing_time_ms": 320.1
}
```

#### List enrolled persons

```
GET /api/v1/faces/persons
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_id` | Query string | null | Filter by event |
| `offset` | Query int | 0 | Pagination offset |
| `limit` | Query int | 50 | Page size (1-200) |

```bash
curl "http://localhost:8000/api/v1/faces/persons?event_id=cebu-marathon-2026&limit=10" \
  -H "X-API-Key: sk_test_YOUR_KEY"
```

#### Get a specific person

```
GET /api/v1/faces/persons/{person_id}
```

#### Delete a person (and all embeddings)

```
DELETE /api/v1/faces/persons/{person_id}
```

---

### 4.3 Bib Number Recognition

#### Recognize bib numbers in a photo

```
POST /api/v1/bibs/recognize
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | — | Photo of runner(s) with visible bib |
| `min_chars` | Query int | No | null | Override minimum digit count (1-10) |

```bash
curl -X POST http://localhost:8000/api/v1/bibs/recognize \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "file=@runner_with_bib.jpg"
```

Response `data`:

```json
{
  "bibs_detected": 1,
  "detections": [
    {
      "bib_number": "1234",
      "confidence": 0.92,
      "bbox": { "x1": 150, "y1": 200, "x2": 300, "y2": 350 },
      "all_candidates": [
        { "text": "1234", "confidence": 0.92 },
        { "text": "12345", "confidence": 0.45 }
      ]
    }
  ],
  "image_dimensions": [1920, 1080],
  "processing_time_ms": 980.2,
  "warnings": null
}
```

---

### 4.4 Batch Processing

All three pipelines support batch mode. The pattern is the same:

1. Submit files — get a `job_id` back (HTTP 202)
2. Poll for results

#### Submit a batch

```bash
# Blur batch
curl -X POST http://localhost:8000/api/v1/blur/detect/batch \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "files=@photo1.jpg" -F "files=@photo2.jpg" -F "files=@photo3.jpg"

# Face search batch
curl -X POST "http://localhost:8000/api/v1/faces/search/batch?event_id=cebu-2026" \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "files=@photo1.jpg" -F "files=@photo2.jpg"

# Face enroll batch (all photos for one person)
curl -X POST http://localhost:8000/api/v1/faces/enroll/batch \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "files=@angle1.jpg" -F "files=@angle2.jpg" \
  -F "person_name=Juan Dela Cruz" -F "event_id=cebu-2026"

# Bib batch
curl -X POST http://localhost:8000/api/v1/bibs/recognize/batch \
  -H "X-API-Key: sk_test_YOUR_KEY" \
  -F "files=@photo1.jpg" -F "files=@photo2.jpg"
```

Submit response (HTTP 202):

```json
{
  "success": true,
  "data": {
    "job_id": "job-uuid-here",
    "status": "pending",
    "total_items": 3,
    "poll_url": "/api/v1/jobs/job-uuid-here"
  }
}
```

#### Poll for results

```bash
curl http://localhost:8000/api/v1/jobs/JOB_ID_HERE \
  -H "X-API-Key: sk_test_YOUR_KEY"
```

Poll response (in progress):

```json
{
  "data": {
    "status": "processing",
    "progress": 0.66,
    "processed_items": 2,
    "total_items": 3
  }
}
```

Poll response (completed):

```json
{
  "data": {
    "status": "completed",
    "progress": 1.0,
    "result": [
      { "index": 0, "is_blurry": false, "confidence": 0.9, ... },
      { "index": 1, "is_blurry": true, "confidence": 0.8, ... },
      { "index": 2, "is_blurry": false, "confidence": 0.7, ... }
    ]
  }
}
```

**Batch limits:** max 20 files per request, max 10MB per file, max 10 active jobs per API key.

---

### 4.5 Health Checks

```bash
# Liveness (is the process alive?) — no auth required
curl http://localhost:8000/api/v1/health

# Readiness (are all components healthy?) — requires auth
curl http://localhost:8000/api/v1/health/ready \
  -H "X-API-Key: sk_test_YOUR_KEY"
```

Readiness response:

```json
{
  "data": {
    "models_loaded": true,
    "database": true,
    "redis": true
  }
}
```

---

## 5. Desktop App Integration (Electron / JavaScript)

### Setup

```javascript
const API_BASE = 'http://localhost:8000/api/v1';
const API_KEY  = 'sk_test_YOUR_KEY';
const headers  = { 'X-API-Key': API_KEY };
```

### Blur Detection

```javascript
async function checkBlur(filePath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(filePath));

  const res = await fetch(`${API_BASE}/blur/detect`, {
    method: 'POST',
    headers,
    body: formData,
  });
  const json = await res.json();

  if (json.success) {
    const { is_blurry, confidence, metrics } = json.data;
    return { is_blurry, confidence, sharpness: metrics.laplacian_variance };
  }
  throw new Error(json.error.message);
}
```

### Blur Classification

```javascript
async function classifyBlur(filePath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(filePath));

  const res = await fetch(`${API_BASE}/blur/classify`, {
    method: 'POST',
    headers,
    body: formData,
  });
  const json = await res.json();

  if (json.success) {
    const { predicted_class, confidence, probabilities } = json.data;
    // predicted_class: "sharp" | "motion_blurred" | "defocused_blurred" | "defocused_object_portrait"
    return { predicted_class, confidence, probabilities };
  }
  throw new Error(json.error.message);
}
```

### Face Enrollment

```javascript
async function enrollRunner(filePath, personName, eventId, personId = null) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(filePath));
  formData.append('person_name', personName);
  if (eventId) formData.append('event_id', eventId);
  if (personId) formData.append('person_id', personId);

  const res = await fetch(`${API_BASE}/faces/enroll`, {
    method: 'POST',
    headers,
    body: formData,
  });
  const json = await res.json();

  if (json.success) {
    return {
      personId: json.data.person_id,       // save this!
      personName: json.data.person_name,
      facesEnrolled: json.data.faces_enrolled,
    };
  }
  throw new Error(json.error.message);
}
```

### Face Search

```javascript
async function searchFaces(filePath, eventId, threshold = 0.4) {
  const url = new URL(`${API_BASE}/faces/search`);
  url.searchParams.set('threshold', threshold);
  if (eventId) url.searchParams.set('event_id', eventId);

  const formData = new FormData();
  formData.append('file', fs.createReadStream(filePath));

  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });
  const json = await res.json();

  if (json.success) {
    return {
      facesDetected: json.data.faces_detected,
      matches: json.data.matches,
      unmatchedFaces: json.data.unmatched_faces,
    };
  }
  throw new Error(json.error.message);
}
```

### Face Compare

```javascript
async function compareFaces(filePath1, filePath2) {
  const formData = new FormData();
  formData.append('file1', fs.createReadStream(filePath1));
  formData.append('file2', fs.createReadStream(filePath2));

  const res = await fetch(`${API_BASE}/faces/compare`, {
    method: 'POST',
    headers,
    body: formData,
  });
  const json = await res.json();

  if (json.success) {
    return {
      isMatch: json.data.is_match,
      similarity: json.data.similarity,
    };
  }
  throw new Error(json.error.message);
}
```

### Bib Recognition

```javascript
async function recognizeBib(filePath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(filePath));

  const res = await fetch(`${API_BASE}/bibs/recognize`, {
    method: 'POST',
    headers,
    body: formData,
  });
  const json = await res.json();

  if (json.success) {
    return {
      bibsDetected: json.data.bibs_detected,
      detections: json.data.detections,  // [{bib_number, confidence, bbox}]
    };
  }
  throw new Error(json.error.message);
}
```

### Batch Processing with Polling

```javascript
async function batchProcess(endpoint, filePaths, extraParams = {}) {
  const formData = new FormData();
  for (const fp of filePaths) {
    formData.append('files', fs.createReadStream(fp));
  }

  const url = new URL(`${API_BASE}/${endpoint}`);
  for (const [key, val] of Object.entries(extraParams)) {
    url.searchParams.set(key, val);
  }

  // 1. Submit
  const res = await fetch(url, { method: 'POST', headers, body: formData });
  const json = await res.json();
  if (!json.success) throw new Error(json.error.message);

  const jobId = json.data.job_id;

  // 2. Poll
  while (true) {
    const pollRes = await fetch(`${API_BASE}/jobs/${jobId}`, { headers });
    const pollJson = await pollRes.json();
    const { status, progress, result, error } = pollJson.data;

    if (status === 'completed') return result;
    if (status === 'failed') throw new Error(error);

    // Optional: update UI progress bar (progress is 0.0 - 1.0)
    onProgress?.(progress);

    await new Promise(r => setTimeout(r, 1000));
  }
}

// Usage:
const results = await batchProcess('blur/detect/batch', ['a.jpg', 'b.jpg', 'c.jpg']);
const results = await batchProcess('faces/search/batch', files, { event_id: 'cebu-2026', threshold: '0.4' });
const results = await batchProcess('bibs/recognize/batch', files);
```

### Delete a Person

```javascript
async function deletePerson(personId) {
  const res = await fetch(`${API_BASE}/faces/persons/${personId}`, {
    method: 'DELETE',
    headers,
  });
  const json = await res.json();
  return json.success;
}
```

### List Enrolled Persons

```javascript
async function listPersons(eventId, page = 0, limit = 50) {
  const url = new URL(`${API_BASE}/faces/persons`);
  if (eventId) url.searchParams.set('event_id', eventId);
  url.searchParams.set('offset', page * limit);
  url.searchParams.set('limit', limit);

  const res = await fetch(url, { headers });
  const json = await res.json();

  if (json.success) {
    return {
      persons: json.data.persons,
      total: json.data.total,
    };
  }
  throw new Error(json.error.message);
}
```

---

## 6. Endpoint Quick Reference

| Endpoint | Method | Auth Scope | Description |
|----------|--------|------------|-------------|
| `/api/v1/health` | GET | None | Liveness check |
| `/api/v1/health/ready` | GET | Any key | Readiness check (models, DB, Redis) |
| `/api/v1/blur/detect` | POST | `blur:read` | Quick blur detection |
| `/api/v1/blur/classify` | POST | `blur:read` | Blur type classification |
| `/api/v1/blur/detect/batch` | POST | `blur:read` | Batch blur detection |
| `/api/v1/blur/classify/batch` | POST | `blur:read` | Batch blur classification |
| `/api/v1/faces/detect` | POST | `faces:read` | Face detection only |
| `/api/v1/faces/enroll` | POST | `faces:write` | Register a face |
| `/api/v1/faces/search` | POST | `faces:read` | Search for face matches |
| `/api/v1/faces/compare` | POST | `faces:read` | 1:1 face comparison |
| `/api/v1/faces/persons` | GET | `faces:read` | List enrolled persons |
| `/api/v1/faces/persons/{id}` | GET | `faces:read` | Get person details |
| `/api/v1/faces/persons/{id}` | DELETE | `faces:delete` | Delete person + embeddings |
| `/api/v1/faces/search/batch` | POST | `faces:read` | Batch face search |
| `/api/v1/faces/enroll/batch` | POST | `faces:write` | Batch face enrollment |
| `/api/v1/bibs/recognize` | POST | `bibs:read` | Recognize bib numbers |
| `/api/v1/bibs/recognize/batch` | POST | `bibs:read` | Batch bib recognition |
| `/api/v1/jobs/{id}` | GET | Any key | Poll batch job status |
| `/api/v1/webhooks` | POST | `webhooks:write` | Register a webhook |
| `/api/v1/webhooks` | GET | `webhooks:read` | List webhooks |
| `/api/v1/webhooks/{id}` | DELETE | `webhooks:write` | Delete a webhook |

---

## 7. Typical Marathon Photo Workflow

```
1. SETUP (once per event)
   ├── Create event_id: "cebu-marathon-2026"
   └── Runners register with selfies
       └── POST /faces/enroll (person_name, event_id)
       └── Save person_id in your app's database

2. DURING EVENT (photographer uploads photos)
   ├── Filter blurry photos
   │   └── POST /blur/detect/batch → discard is_blurry=true
   ├── Auto-tag bib numbers
   │   └── POST /bibs/recognize/batch → save bib_number per photo
   └── Match faces to runners
       └── POST /faces/search/batch (event_id) → link photos to persons

3. AFTER EVENT (runner searches for their photos)
   ├── Runner uploads selfie or searches by name
   │   └── POST /faces/search (event_id) → return matched photos
   └── Runner searches by bib number
       └── Look up bib_number in your app's database

4. CLEANUP
   └── DELETE /faces/persons/{id} for GDPR erasure if requested
```

---

## 8. Error Codes

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `MODEL_UNAVAILABLE` | 503 | ML model not loaded; check health endpoint |
| `NO_FACES` | 200 | No faces detected in the uploaded image |
| `LOW_QUALITY` | 200 | All faces below minimum enrollment confidence |
| `INVALID_INPUT` | 200 | Invalid parameter (e.g., bad person_id format) |
| `NOT_FOUND` | 200 | Person or webhook not found |
| `BATCH_TOO_LARGE` | 400 | Exceeded max 20 files per batch |
| `EMPTY_BATCH` | 400 | No files provided |
| `TOO_MANY_JOBS` | 429 | Max 10 active jobs per API key |
| `REQUEST_TIMEOUT` | 504 | Request exceeded 60s |
| Missing API key | 401 | No `X-API-Key` header |
| Invalid API key | 401 | Key not found or deactivated |
| Insufficient scope | 403 | Key lacks required permission |
| Rate limit exceeded | 429 | Too many requests; check `Retry-After` header |

---

## 9. Tips

- **Image format:** JPEG, PNG, or WebP only. Max 10MB per file. Max 4096px per dimension.
- **Face enrollment:** Use clear, well-lit photos with face visible. Enroll 2-3 photos from different angles for better accuracy.
- **Similarity threshold:** Default 0.4 is good for marathons. Lower (0.3) = more matches but more false positives. Higher (0.6) = fewer but more accurate matches.
- **Event scoping:** Always pass `event_id` on enroll and search to keep events isolated and searches fast.
- **Batch vs single:** Use single endpoints for real-time UI. Use batch endpoints for background processing of albums.
- **Polling interval:** Poll every 1-2 seconds for batch jobs. Check `progress` field to update UI.
