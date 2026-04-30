# Integration Contracts — Backend-to-ai-api

**Date:** 2026-04-23
**Companion doc:** `integration-architecture.md` (read that first for the full picture)
**Purpose:** Exact API usage patterns for each backend. Copy-paste ready.

---

## Contract 1: Desktop Backend (Blur Only)

The Desktop Backend uses ai-api for image quality assessment. It calls two endpoints.

### Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/blur/detect` | Quick blur check (Laplacian + optional FFT) |
| `POST /api/v1/blur/detect/stream` | NDJSON streaming check for up to 500 images (no polling) |
| `POST /api/v1/blur/detect/batch` | Async Celery batch, up to 50 |
| `POST /api/v1/blur/detect/mega` | Async Celery chord, up to 500 (server splits) |
| `POST /api/v1/blur/classify` | Detailed blur-type classification (CNN) |
| `POST /api/v1/blur/classify/stream` | NDJSON streaming classify |
| `POST /api/v1/blur/classify/batch` | Async batch classify |
| `POST /api/v1/blur/classify/mega` | Async mega-batch classify |
| `GET /api/v1/jobs/{job_id}` | Poll batch job status (supports `offset`/`limit`) |
| `GET /api/v1/health/ready` | Health check before processing |

### Authentication

```
X-API-Key: sk_desktop_prod_<your_key>
```

Scope: `blur:read` (covers both detect and classify endpoints).

### Flow 1: Single Photo Quality Check

Use when user imports one photo or selects a photo for review.

```
Desktop App                Desktop Backend                ai-api
    │                           │                           │
    │  "Check this photo"       │                           │
    │─────────────────────────►│                           │
    │                           │  POST /blur/detect        │
    │                           │  + file                   │
    │                           │─────────────────────────►│
    │                           │                           │
    │                           │  { is_blurry: false,      │
    │                           │    confidence: 0.85,      │
    │                           │    metrics: {             │
    │                           │      laplacian: 185.4,    │
    │                           │      hf_ratio: 0.72 } }  │
    │                           │◄─────────────────────────│
    │                           │                           │
    │                           │  (Optional) If not blurry │
    │                           │  but user wants detail:   │
    │                           │  POST /blur/classify      │
    │                           │  + file                   │
    │                           │─────────────────────────►│
    │                           │                           │
    │                           │  { predicted_class:       │
    │                           │    "sharp",               │
    │                           │    confidence: 0.96,      │
    │                           │    probabilities: {...} } │
    │                           │◄─────────────────────────│
    │                           │                           │
    │  "Sharp (96% confident)"  │                           │
    │◄─────────────────────────│                           │
```

**Desktop Backend logic:**
```python
# 1. Call ai-api blur detect
response = httpx.post(
    f"{AI_API_URL}/api/v1/blur/detect",
    headers={"X-API-Key": AI_API_KEY},
    files={"file": (filename, image_bytes, content_type)},
)
result = response.json()["data"]

# 2. Store result in desktop DB
photo.blur_score = result["metrics"]["laplacian_variance"]
photo.is_blurry = result["is_blurry"]
photo.blur_confidence = result["confidence"]
photo.hf_ratio = result["metrics"]["hf_ratio"]
db.save(photo)

# 3. Optionally classify for blur type
if not result["is_blurry"] or user_wants_detail:
    classify_resp = httpx.post(
        f"{AI_API_URL}/api/v1/blur/classify",
        headers={"X-API-Key": AI_API_KEY},
        files={"file": (filename, image_bytes, content_type)},
    )
    classify_result = classify_resp.json()["data"]
    photo.blur_type = classify_result["predicted_class"]
    photo.blur_type_confidence = classify_result["confidence"]
    db.save(photo)

# 4. Return to desktop UI
return {
    "quality": "sharp" if not photo.is_blurry else "blurry",
    "blur_score": photo.blur_score,
    "blur_type": photo.blur_type,  # e.g., "motion_blurred"
    "confidence": photo.blur_confidence,
}
```

### Flow 2: Batch Import (Multiple Photos)

Use when user imports a folder of photos. Desktop backend should batch them.

```
Desktop App                Desktop Backend                ai-api
    │                           │                           │
    │  "Import 50 photos"       │                           │
    │─────────────────────────►│                           │
    │                           │  POST /blur/detect/batch  │
    │                           │  + 50 files               │
    │                           │─────────────────────────►│
    │                           │                           │
    │                           │  { job_id: "abc-123",     │
    │                           │    status: "pending",     │
    │                           │    poll_url: "/jobs/..." }│
    │                           │◄─────────────────────────│
    │                           │                           │
    │  "Processing 50 photos…"  │                           │
    │◄─────────────────────────│                           │
    │                           │                           │
    │                           │  (poll every 2-5 seconds) │
    │                           │  GET /jobs/abc-123        │
    │                           │─────────────────────────►│
    │                           │  { status: "processing",  │
    │                           │    progress: 0.60 }       │
    │                           │◄─────────────────────────│
    │                           │                           │
    │  "60% complete"           │                           │
    │◄─────────────────────────│                           │
    │                           │                           │
    │                           │  GET /jobs/abc-123        │
    │                           │─────────────────────────►│
    │                           │  { status: "completed",   │
    │                           │    result: [50 results] } │
    │                           │◄─────────────────────────│
    │                           │                           │
    │  "Done! 42 sharp, 8 blurry"│                          │
    │◄─────────────────────────│                           │
```

**Desktop Backend logic:**
```python
# 1. Submit batch
files = [("files", (f.name, f.bytes, f.content_type)) for f in photos]
response = httpx.post(
    f"{AI_API_URL}/api/v1/blur/detect/batch",
    headers={"X-API-Key": AI_API_KEY},
    files=files,
)
job_id = response.json()["data"]["job_id"]

# 2. Poll for results (or use webhook)
while True:
    status = httpx.get(
        f"{AI_API_URL}/api/v1/jobs/{job_id}",
        headers={"X-API-Key": AI_API_KEY},
    ).json()["data"]

    if status["status"] == "completed":
        break
    if status["status"] == "failed":
        raise ProcessingError(status["error"])

    notify_progress(status["progress"])
    await asyncio.sleep(3)

# 3. Store all results
for item in status["result"]:
    photo = photos[item["index"]]
    photo.is_blurry = item["is_blurry"]
    photo.blur_score = item["metrics"]["laplacian_variance"]
    db.save(photo)
```

### Desktop Backend Response Shaping

The desktop UI doesn't need raw ai-api responses. Shape them:

```json
// What desktop UI receives from desktop backend:
{
  "photo_id": "local-123",
  "filename": "IMG_4521.jpg",
  "quality": {
    "status": "sharp",
    "score": 185.4,
    "blur_type": null,
    "confidence": 0.85
  },
  "imported_at": "2026-03-26T10:00:00Z"
}
```

---

## Contract 2: Web/Mobile Backend (Full Features)

The Web/Mobile Backend uses all ai-api features and manages events, participants, and photo galleries.

### Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/blur/detect` | Photo quality gate on upload |
| `POST /api/v1/blur/classify` | Detailed blur classification |
| `POST /api/v1/blur/detect/batch` · `/detect/mega` | Async batch / mega-batch quality check |
| `POST /api/v1/blur/classify/batch` · `/classify/mega` | Async batch / mega-batch classification |
| `POST /api/v1/faces/enroll` | Register participant face for event |
| `POST /api/v1/faces/enroll/batch` | Bulk-enroll multiple photos under one person |
| `POST /api/v1/faces/search` | Find participants in uploaded photo |
| `POST /api/v1/faces/search/batch` · `/search/mega` | Async batch / mega-batch face search |
| `POST /api/v1/faces/detect` | Count faces / return bounding boxes |
| `POST /api/v1/faces/compare` | 1:1 face verification |
| `GET /api/v1/faces/persons` | Paginated list (scoped by api_key_id + optional event_id) |
| `GET /api/v1/faces/persons/{id}` | Get enrolled person info |
| `DELETE /api/v1/faces/persons/{id}` | Remove person (GDPR) |
| `POST /api/v1/bibs/recognize` | Read bib number from photo (`min_chars` override available) |
| `POST /api/v1/bibs/recognize/batch` · `/recognize/mega` | Async batch / mega-batch |
| `GET /api/v1/jobs/{job_id}` | Poll batch job status (supports `offset`/`limit`) |
| `POST /api/v1/webhooks` · `GET` · `DELETE` | Manage webhook subscriptions |
| `GET /api/v1/health/ready` | Health check |

### Authentication

```
X-API-Key: sk_webmobile_prod_<your_key>
```

Required scopes (actual names used in code): `blur:read`, `faces:read`, `faces:write`, `faces:delete`, `bibs:read`, `jobs:read`, `webhooks:read`, `webhooks:write`. A key with the `*` super-scope passes every check.

### Flow 1: Event Setup — Enroll Participant Faces

Before an event starts, the admin creates a public event gallery and uploads participant photos. Backend enrolls each face with `event_id`.

```
Admin (Web)              Web/Mobile Backend              ai-api
    │                           │                           │
    │  "Upload participant      │                           │
    │   photos for Marathon"    │                           │
    │─────────────────────────►│                           │
    │                           │                           │
    │                           │  For each participant:    │
    │                           │  POST /faces/enroll       │
    │                           │  form: file,              │
    │                           │    person_name,           │
    │                           │    event_id=marathon-2026 │
    │                           │─────────────────────────►│
    │                           │                           │
    │                           │  { person_id: "p-123",   │
    │                           │    faces_enrolled: 1 }    │
    │                           │◄─────────────────────────│
    │                           │                           │
    │                           │  Backend stores mapping:  │
    │                           │  participant.ai_person_id │
    │                           │    = "p-123"              │
    │                           │                           │
    │  "Enrolled 500 participants"│                         │
    │◄─────────────────────────│                           │
```

**Web/Mobile Backend logic:**
```python
async def enroll_participant(participant, event, photo_file):
    """Enroll a participant's face in ai-api, linked to this event."""

    response = await httpx.post(
        f"{AI_API_URL}/api/v1/faces/enroll",
        headers={"X-API-Key": AI_API_KEY},
        files={"file": (photo_file.name, photo_file.bytes, "image/jpeg")},
        data={
            "person_name": participant.full_name,
            "event_id": str(event.id),  # CRITICAL: always pass event_id
        },
    )
    result = response.json()

    if not result["success"]:
        if result["error"]["code"] == "LOW_QUALITY":
            participant.enrollment_status = "low_quality_photo"
            db.save(participant)
            return {"status": "failed", "reason": "Photo quality too low for enrollment"}
        raise EnrollmentError(result["error"]["message"])

    # Store ai-api person_id in our participant record
    participant.ai_person_id = result["data"]["person_id"]
    participant.enrollment_status = "enrolled"
    db.save(participant)

    return {"status": "enrolled", "person_id": result["data"]["person_id"]}
```

### Flow 2: Photo Upload — Auto-tag with Face + Bib

When a photographer uploads a race photo, the backend runs face search AND bib recognition to automatically tag participants.

```
Photographer (Mobile)    Web/Mobile Backend              ai-api
    │                           │                           │
    │  "Upload race photo"      │                           │
    │─────────────────────────►│                           │
    │                           │                           │
    │                           │  Step 1: Quality check    │
    │                           │  POST /blur/detect        │
    │                           │  + file                   │
    │                           │─────────────────────────►│
    │                           │  { is_blurry: false }     │
    │                           │◄─────────────────────────│
    │                           │                           │
    │                           │  Step 2: Face search      │
    │                           │  POST /faces/search       │
    │                           │  ?event_id=marathon-2026  │
    │                           │  ?threshold=0.6           │
    │                           │  + file                   │
    │                           │─────────────────────────►│
    │                           │  { matches: [            │
    │                           │    { person_id: "p-123", │
    │                           │      person_name: "John",│
    │                           │      similarity: 0.87 }  │
    │                           │  ] }                      │
    │                           │◄─────────────────────────│
    │                           │                           │
    │                           │  Step 3: Bib recognition  │
    │                           │  POST /bibs/recognize     │
    │                           │  + file                   │
    │                           │─────────────────────────►│
    │                           │  { detections: [         │
    │                           │    { bib_number: "1023", │
    │                           │      confidence: 0.91 }  │
    │                           │  ] }                      │
    │                           │◄─────────────────────────│
    │                           │                           │
    │                           │  Step 4: Backend merges   │
    │                           │  face match + bib match   │
    │                           │  into final tagging       │
    │                           │                           │
    │  "Tagged: John Doe (#1023)"│                          │
    │◄─────────────────────────│                           │
```

**Web/Mobile Backend logic:**
```python
async def process_uploaded_photo(photo_file, event):
    """Full pipeline: blur check → face search → bib OCR → merge results."""

    image_bytes = await photo_file.read()

    # Step 1: Blur quality gate
    blur_resp = await httpx.post(
        f"{AI_API_URL}/api/v1/blur/detect",
        headers={"X-API-Key": AI_API_KEY},
        files={"file": (photo_file.name, image_bytes, "image/jpeg")},
    )
    blur = blur_resp.json()["data"]

    if blur["is_blurry"]:
        return {"status": "rejected", "reason": "Photo is too blurry"}

    # Step 2: Face search (scoped to this event)
    face_resp = await httpx.post(
        f"{AI_API_URL}/api/v1/faces/search",
        headers={"X-API-Key": AI_API_KEY},
        params={
            "event_id": str(event.id),
            "threshold": event.face_match_threshold,  # e.g., 0.6
            "top_k": 5,
        },
        files={"file": (photo_file.name, image_bytes, "image/jpeg")},
    )
    face_matches = face_resp.json()["data"]["matches"]

    # Step 3: Bib recognition
    bib_resp = await httpx.post(
        f"{AI_API_URL}/api/v1/bibs/recognize",
        headers={"X-API-Key": AI_API_KEY},
        files={"file": (photo_file.name, image_bytes, "image/jpeg")},
    )
    bib_detections = bib_resp.json()["data"]["detections"]

    # Step 4: Merge results in backend
    tagged_participants = []

    # From face matches: look up our participant by ai_person_id
    for match in face_matches:
        participant = db.query(Participant).filter_by(
            ai_person_id=match["person_id"],
            event_id=event.id,
        ).first()
        if participant:
            tagged_participants.append({
                "participant": participant,
                "method": "face",
                "confidence": match["similarity"],
            })

    # From bib matches: look up participant by bib number
    for detection in bib_detections:
        if detection["confidence"] < event.bib_confidence_threshold:  # e.g., 0.7
            continue  # Backend filters low-confidence OCR

        participant = db.query(Participant).filter_by(
            bib_number=detection["bib_number"],
            event_id=event.id,
        ).first()
        if participant:
            tagged_participants.append({
                "participant": participant,
                "method": "bib",
                "confidence": detection["confidence"],
            })

    # Deduplicate (same participant found by both face and bib)
    unique = deduplicate_by_participant_id(tagged_participants)

    # Store photo with tags
    photo = Photo(
        event_id=event.id,
        file_url=upload_to_storage(image_bytes),
        blur_score=blur["metrics"]["laplacian_variance"],
        tags=unique,
    )
    db.save(photo)

    return {"status": "processed", "tagged": [t["participant"].name for t in unique]}
```

### Flow 3: Batch Processing (Event Photo Dump)

For bulk uploads (e.g., photographer dumps 200 photos after event), use batch endpoints with webhooks.

```
Photographer (Web)       Web/Mobile Backend              ai-api
    │                           │                           │
    │  "Upload 200 photos"      │                           │
    │─────────────────────────►│                           │
    │                           │                           │
    │                           │  POST /faces/search/batch │
    │                           │  + 100 files (batch 1)    │
    │                           │  params: event_id         │
    │                           │─────────────────────────►│
    │                           │  { job_id: "face-job-1" } │
    │                           │◄─────────────────────────│
    │                           │                           │
    │                           │  POST /bibs/recognize/batch│
    │                           │  + 100 files (batch 1)    │
    │                           │─────────────────────────►│
    │                           │  { job_id: "bib-job-1" }  │
    │                           │◄─────────────────────────│
    │                           │                           │
    │  "Processing… 0%"         │                           │
    │◄─────────────────────────│                           │
    │                           │                           │
    │                           │  (webhook fires when done)│
    │                           │◄─ POST /webhooks/callback │
    │                           │   { job_id: "face-job-1", │
    │                           │     result_count: 100 }   │
    │                           │                           │
    │                           │  GET /jobs/face-job-1     │
    │                           │─────────────────────────►│
    │                           │  { result: [...] }        │
    │                           │◄─────────────────────────│
    │                           │                           │
    │  "Done! 180 tagged"       │                           │
    │◄─────────────────────────│                           │
```

### Flow 4: Participant Removal (GDPR)

When a participant requests data deletion:

```python
async def remove_participant(participant, event):
    """Remove participant from both backend and ai-api."""

    # 1. Delete face data from ai-api
    if participant.ai_person_id:
        await httpx.delete(
            f"{AI_API_URL}/api/v1/faces/persons/{participant.ai_person_id}",
            headers={"X-API-Key": AI_API_KEY},
        )

    # 2. Remove tags from photos in backend
    db.query(PhotoTag).filter_by(participant_id=participant.id).delete()

    # 3. Delete participant record in backend
    db.delete(participant)
    db.commit()
```

---

## Error Handling Contract

Both backends must handle these ai-api error scenarios. The `Error Code` column shows what appears in the `error.code` field of the standard envelope (or `detail` when the response is a raw FastAPI `HTTPException`).

| HTTP Status | Error Code | Response shape | Backend Action |
|------------|-----------|---------------|----------------|
| 200 (success: false) | `LOW_QUALITY` / `NO_FACES` / `NOT_FOUND` / `INVALID_INPUT` / `INVALID_WEBHOOK_URL` / `DELETE_FAILED` | Envelope | Flag, surface user-friendly message, or retry with better input |
| 400 | `ImageValidationError` (class name) | Envelope | Reject upload with user-friendly message (file type, size, dimensions, corrupt) |
| 400 | `EMPTY_BATCH` / `BATCH_TOO_LARGE` | Envelope | Recheck client-side batching logic |
| 401 | `AuthenticationError` (class name) **or** raw `{"detail": "Missing API key"}` | Envelope *or* raw | Log alert — API key may be expired or revoked |
| 403 | Raw `{"detail": "Insufficient permissions. Required scope: ..."}` | Raw | Request a key with the right scope |
| 404 | `NOT_FOUND` | Envelope | Fall through to "not found" in the caller |
| 429 | Raw `{"detail": "Rate limit exceeded. Retry after N seconds."}` | Raw (with `Retry-After` header) | Queue request for retry after `Retry-After` seconds |
| 429 | `TOO_MANY_JOBS` | Envelope | Wait for earlier batch jobs to finish before submitting more |
| 503 | `MODEL_UNAVAILABLE` | Envelope | Show "AI processing temporarily unavailable", queue for retry |
| 504 | `REQUEST_TIMEOUT` | Envelope | Retry with a smaller image or split the request |
| 5xx | Any | Any | Retry with exponential backoff (max 3 retries) |

**Why the shape varies:** `QuickPitikError` subclasses (e.g., `ImageValidationError`, `JobNotFoundError`) go through the handler in `src/main.py` and produce the envelope with `error.code = <class_name>`. `HTTPException`s raised inside middleware (auth, rate limit, scope check) use FastAPI's default `{"detail": ...}` body — there is no envelope for those responses.

### Retry Strategy

```python
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds

async def call_ai_api_with_retry(method, url, **kwargs):
    for attempt in range(MAX_RETRIES):
        response = await httpx.request(method, url, **kwargs)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", BACKOFF_BASE ** attempt))
            await asyncio.sleep(retry_after)
            continue

        if response.status_code >= 500:
            await asyncio.sleep(BACKOFF_BASE ** attempt)
            continue

        return response

    raise AIApiUnavailableError(f"ai-api failed after {MAX_RETRIES} retries")
```

---

## Health Check Contract

Both backends should check ai-api health before starting to process requests.

```python
async def check_ai_api_health():
    """Call on backend startup and periodically (every 30s)."""
    response = await httpx.get(
        f"{AI_API_URL}/api/v1/health/ready",
        timeout=5.0,
    )
    health = response.json()["data"]

    if not health["models_loaded"]:
        logger.warning("ai-api models not loaded — ML features unavailable")
    if not health["database"]:
        logger.error("ai-api database down — face search unavailable")
    if not health["redis"]:
        logger.warning("ai-api redis down — batch processing unavailable")

    return health
```

---

## Configuration Each Backend Needs

### Desktop Backend `.env`

```bash
# ai-api connection
AI_API_URL=http://ai-api.internal:8000
AI_API_KEY=sk_desktop_prod_xxxxxxxxxxxx

# Blur settings (backend decides these, not ai-api)
BLUR_AUTO_REJECT=false         # Desktop shows results, doesn't auto-reject
BLUR_CLASSIFY_ON_IMPORT=true   # Run classification on every import
```

### Web/Mobile Backend `.env`

```bash
# ai-api connection
AI_API_URL=http://ai-api.internal:8000
AI_API_KEY=sk_webmobile_prod_xxxxxxxxxxxx

# Blur settings
BLUR_AUTO_REJECT=true                 # Reject blurry photos on upload
BLUR_REJECT_THRESHOLD=100.0           # Laplacian variance threshold

# Face search settings (per-event defaults, overridable in event settings)
FACE_MATCH_THRESHOLD_DEFAULT=0.6      # Minimum similarity to show match
FACE_TOP_K_DEFAULT=5                  # Max matches per face

# Bib OCR settings
BIB_CONFIDENCE_THRESHOLD_DEFAULT=0.7  # Minimum OCR confidence to trust
```

---

## Summary: Who Calls What

```
Desktop Backend:
  ✅ blur/detect, blur/detect/stream, blur/detect/batch, blur/detect/mega
  ✅ blur/classify, blur/classify/stream, blur/classify/batch, blur/classify/mega
  ✅ jobs/{id}
  ✅ health/ready
  ❌ faces/*       (not used)
  ❌ bibs/*        (not used)
  ❌ webhooks      (optional — polling is fine for desktop)

Web/Mobile Backend:
  ✅ blur/detect (+ stream/batch/mega variants)
  ✅ blur/classify (+ stream/batch/mega variants)
  ✅ faces/enroll          (with event_id!)
  ✅ faces/enroll/batch
  ✅ faces/search          (with event_id!)
  ✅ faces/search/batch, faces/search/mega
  ✅ faces/detect
  ✅ faces/compare
  ✅ faces/persons         (list), faces/persons/{id}, DELETE faces/persons/{id}
  ✅ bibs/recognize, bibs/recognize/batch, bibs/recognize/mega
  ✅ jobs/{id}             (with offset/limit pagination for large results)
  ✅ webhooks              (recommended for batch workflows)
  ✅ health/ready
```
