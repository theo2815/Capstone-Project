# Integration Contracts — Backend-to-ai-api

**Date:** 2026-04-23
**Companion doc:** `integration-architecture.md` (read that first for the full picture)
**Purpose:** Exact API usage patterns for each backend. Copy-paste ready.

---

## Contract 1: BatchMyPhotos desktop (blur only)

BatchMyPhotos' Electron **main process** calls ai-api directly. Its Express backend is
not a blur proxy. In the invited beta, analysis starts only after **Start Analysis**;
results are suggestions and never change batch routing or original files. See
the [desktop integration guide](../../docs/desktop-blur-detection-integration-guide.md)
and the implementation in
BatchMyPhotos/desktop/src/main/blurDetectionService.js.

### Routes used by this client

| Endpoint | Use |
|----------|-----|
| `POST /api/v1/blur/classify/stream` | Main CNN path; multipart prepared JPEGs, NDJSON rows and terminal summary |
| `POST /api/v1/blur/classify` | One image when the stream route is unsupported or a valid stream summary omits a row |

The API also offers `/blur/detect`, `/batch`, `/mega`, and `/jobs/{id}` for other
clients. This desktop beta does not use a classical pre-check, Celery jobs,
polling, or webhooks.

### Credential and configuration

Give each invited tester a separate, revocable **staging** key with exactly
`blur:read` and `jobs:read`. Record owner, tier, expiry or revocation, and scope
in the private `docs/api-keys.md` workflow. The main process sends the key only
in the `X-API-Key` header. Packaged beta builds require a validated staging
`blur-beta.json` with an HTTPS origin, then accept the key through Settings
after installation. Electron `safeStorage` encrypts it in the dedicated
`blur-beta-key` store; IPC reveals only whether a key is configured. Neither
the installer nor key-status IPC returns the raw saved key. Source runs use the
`BATCH_BLUR_*` settings in BatchMyPhotos/desktop/src/main/config.js; packaged
environment overrides cannot redirect the origin or inject a key.

### Analysis and response handling

1. Choose one supported raster companion per JPG/RAW group; RAW-only and
   undecodable groups remain unanalysed. Sharp auto-orients, resizes, and
   re-encodes each selected image as a JPEG before upload.
2. Split prepared JPEGs by count **and byte budget**. The authoritative
   limits, concurrency and timeouts are in BatchMyPhotos/desktop/src/main/constants.js.
3. Match each NDJSON row by the uploaded chunk index, not arrival order or
   original filename. Require one consistent terminal `_summary` with
   `complete:true`; reject malformed, duplicate, or contradictory rows.
4. A per-image error marks only that group unanalysed. Recover missing rows
   through the single route only after a valid summary. An unsupported stream
   route also uses singles. Retry a retryable stream failure once within the
   configured backoff bound. A truncated or invalid stream fails visibly.
5. Map returned probabilities using the client's sensitivity and selected
   categories. This mapping is advisory and must not remove a group from
   preview, copy/move, or rollback.

Analysis uploads are separate from training feedback. A label alone sends
nothing; an explicit **Submit this example** action sends the selected
prepared JPEG through the user's Supabase session. The packaged app still
uses production Supabase for feedback. Local SQL/Storage tests do not prove
its production migration, bucket, or policies are ready. See
BatchMyPhotos/desktop/docs/blur-detection.md for retention and deletion.

---

## Contract 2: Web/Mobile Backend (Full Features)

The Web/Mobile Backend uses ai-api's **face and bib** features and manages events, participants, and photo galleries.

### Shared client account and event rules

These routes belong to Spring Boot under `/api/v1`; clients never send account data to ai-api.

- Photographer metadata sharing: `GET`/`PUT /me/photographer/metadata-settings` exchange `{metadataPublicByDefault:boolean}`. The default is private. It is copied to a newly created photographer event or to first upload coverage of a shared event; changing it never changes existing event choices.
- `GET`/`PUT /me/photographer/events/{eventId}/metadata-visibility` exchange `{metadataPublic:boolean}` for the caller's coverage. A legacy owned event without coverage reads as private; its owner may create that relationship by PUT. Uncovered foreign events return 404. Upload counter upserts preserve an explicit choice.
- The future typed photo metadata response uses `photoId`, `status`, nullable `originalFilename`, nullable `fileSizeBytes`, and nullable `fields`. Status is `pending`, `ready`, `empty`, `failed`, or `unavailable`; internal processing reads as pending and a missing legacy row as unavailable. `fields` is an object only for ready/empty, with nullable `cameraMake`, `cameraModel`, `lens`, `aperture` (positive finite number), `exposureTime`, `iso` (positive integer), `author`, `copyright`, `takenAtLocal`, `takenAtOffset`, `flash`, `exposureMode`, and `whiteBalance`. Metadata is owned and served by Spring Boot, never passed to ai-api; raw EXIF, GPS and device serials are not exposed.

- Google sign-in needs a verified Google ID token, never a QuickPitik password. Registration shares the same normalized, unique email namespace and returns `EMAIL_TAKEN` for an existing Google account.
- `PUT /me/password` requires the current password even when Google is linked. To set an initial password or recover one, use the existing `/auth/forgot-password` → `/auth/verify-reset-otp` → `/auth/reset-password` flow. The continuation token is issued only after email OTP verification; reset revokes every refresh token.
- `PUT /me/email` retains the Google-linked exemption from current-password verification (product decision, 2026-09-09). Confirmation goes to the new inbox. This accepts the risk that a stolen authenticated Google-linked session can move the sign-in email and then use password recovery.
- Mobile race-day labels use `deriveEventDisplayState`. Gallery sockets and photographer syncing use `deriveEventState`, preserving the existing post-race upload window. Discovery uses `extractCity` until events provide structured region data; `GET /regions` remains the canonical region catalog for photographer settings.
- Handle changes are validated server-side against `reserved_handles`; V62 adds the account routes introduced in the client guard. Before release, query `SELECT p.user_id, p.handle FROM photographer_settings p JOIN reserved_handles r ON r.handle = lower(p.handle)` and arrange replacement handles with affected owners through the existing settings flow. The migration warns about collisions and preserves ownership.
- The profile's `#race-log` and `#saved-events` fragments both reach Saved events.

> **Blur is desktop-only — the web/mobile backend does NOT call it.** Per root `CLAUDE.md` rule 6, web + mobile upload paths MUST NOT call any `/blur/*` endpoint (no `BLUR_REJECTED` gate, no `blur_score` writes), and the Spring Boot `AiApiClient` intentionally has no `blurDetect()` method. Photographers cull blurry shots in BatchMyPhotos (desktop) before upload, so uploaded photos are already culled. All blur usage lives in Contract 1 (Desktop) only.

### Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/faces/enroll` | Register participant face for event (`event_id` **required** — 422 if omitted) |
| `POST /api/v1/faces/enroll/batch` | Bulk-enroll multiple photos under **one** person (many selfies of one runner) — `event_id` **required** |
| `POST /api/v1/faces/enroll/mega` | Bulk **photo indexing** — one person **per image**, every face stored, `ref` echoed (up to 500) |
| `POST /api/v1/faces/search` | Find participants in uploaded photo (`event_id` **required** — 422 if omitted) |
| `POST /api/v1/faces/search/batch` · `/search/mega` | Async batch / mega-batch face search (`event_id` **required** when `operation=search`) |
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

Required scopes (actual names used in code): `faces:read`, `faces:write`, `faces:delete`, `bibs:read`, `jobs:read`, `webhooks:read`, `webhooks:write`. (No `blur:read` — blur is desktop-only, see note above.) A key with the `*` super-scope passes every check.

### Flow 1: Event Setup — Enroll Participant Faces

Before an event starts, the admin creates a public event gallery and uploads participant photos. Backend enrolls each face with `event_id`.

> **Multi-face enrollment (photo indexing).** `enroll` stores **one embedding per detected face**, each keyed by a per-face-distinct `source_image_hash`. A single image containing N runners returns `faces_enrolled: N` and makes **all** of them findable — essential when a *photo* (not one participant) is enrolled as a single person. Faces below `FACE_MIN_ENROLLMENT_CONFIDENCE` are skipped; re-enrolling the same image is idempotent via the `(person_id, source_image_hash)` unique index. (Before 2026-06-02 the hash was computed over the whole image, so all but one face in a crowd shot collided and were dropped.)

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
    │                           │  Step 1: Face search      │
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
    │                           │  Step 2: Bib recognition  │
    │                           │  POST /bibs/recognize     │
    │                           │  + file                   │
    │                           │─────────────────────────►│
    │                           │  { detections: [         │
    │                           │    { bib_number: "1023", │
    │                           │      confidence: 0.91 }  │
    │                           │  ] }                      │
    │                           │◄─────────────────────────│
    │                           │                           │
    │                           │  Step 3: Backend merges   │
    │                           │  face match + bib match   │
    │                           │  into final tagging       │
    │                           │                           │
    │  "Tagged: John Doe (#1023)"│                          │
    │◄─────────────────────────│                           │
```

**Event isolation (fail-closed):** `event_id` is **required** on every enroll and every search — `/faces/search` rejects a missing scope with `422`, `/search/batch`·`/search/mega` reject it when `operation=search`, and `/faces/enroll`·`/enroll/batch`·`/enroll/mega` reject it outright. The backend already passes `event_id` on every call; making it required just moves enforcement of root rule 5 to the ai-api boundary instead of trusting the caller.

The two directions fail differently, and both matter. An unscoped **search** would silently match across every event for the API key — a leak. An unscoped **enroll** is worse in a quieter way: the embedding is stored with `event_id = NULL`, which no search can ever return (they all require a scope, and `NULL` matches none) and which `DELETE /faces/persons?event_id=` cannot erase — a permanent, invisible, un-GDPR-erasable orphan. Enroll was fail-open until 2026-08-14; `/enroll/mega` had always required it.

When `person_id` is supplied to an enroll, the person is looked up scoped by `api_key_id` **and** `event_id`. An existence-only lookup would let one tenant attach faces to another tenant's person, or file an embedding under a different event than the caller declared.

**Web/Mobile Backend logic:**
```python
async def process_uploaded_photo(photo_file, event):
    """Full pipeline: face search → bib OCR → merge results.

    (No blur gate — photos arrive already culled by the desktop app; blur is
    desktop-only per root CLAUDE.md rule 6.)
    """

    image_bytes = await photo_file.read()

    # Step 1: Face search (scoped to this event)
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

    # Step 2: Bib recognition
    bib_resp = await httpx.post(
        f"{AI_API_URL}/api/v1/bibs/recognize",
        headers={"X-API-Key": AI_API_KEY},
        files={"file": (photo_file.name, image_bytes, "image/jpeg")},
    )
    bib_detections = bib_resp.json()["data"]["detections"]

    # Step 3: Merge results in backend
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

### Flow 3b: Bulk Photo Indexing — `person = photo` (mega + webhook-or-poll)

The Spring backend's async indexing drain (Phase C) does not call the
single-photo endpoints per file. It accumulates `PENDING` photos **per event**
and submits each event-batch as **one** face-mega job and **one** bib-mega job,
then ingests results via webhook (prod) or poll (dev/backstop).

**`POST /api/v1/faces/enroll/mega`** — the photo-indexing primitive.

| | |
|---|---|
| Request | multipart `files` (≤500); Form `event_id` (**required**); scope `faces:write` |
| Semantics | each image enrolled as its **own** person; **every** detected face stored (per-face hash). Contrast `/faces/enroll/batch`, which lumps all images under one shared person. |
| Filename = `ref` | set each file's filename to your own photo ID; it is echoed back per result so you can map without relying on order. |
| Response | `202 { job_id }` — poll `GET /jobs/{job_id}`. |

Per-image result (in `job.result`):
```jsonc
{ "index": 0, "ref": "<your photo id>", "person_id": "uuid-or-null",
  "faces_detected": 3, "faces_enrolled": 3, "skipped": 0 }
// person_id is null + faces_enrolled 0 when no enrollable face was found —
// a benign "no runner in this shot" outcome (mark the photo indexed anyway).
```

**`POST /api/v1/bibs/recognize/mega`** also echoes `ref` per result when filenames are supplied. Map `result.ref` → your photo ID (fall back to `index` if absent).

**Event isolation:** a mega job is always single-event — every person it creates is stamped with that one `event_id`. The same runner appearing in two events becomes two separate event-scoped persons; search stays scoped by `event_id`.

**Webhook tenant scoping:** `job.completed` / `job.failed` now fan out **only** to subscriptions owned by the API key whose job fired (previously matched by event name alone, which leaked job IDs across tenants). The webhook body carries only `{ event, job_id, result_count|error }` — map `job_id` to your own job record (which holds the ordered photo IDs + `event_id`), then `GET /jobs/{job_id}` for the results. Best-effort delivery → make the receiver idempotent and keep a poll backstop.

**Backpressure:** `internal`-tier keys get a higher active-job ceiling (`MAX_ACTIVE_JOBS_PER_KEY_INTERNAL`, default 50) than the public default (10), since one drain submits two jobs per event-batch.

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

### BatchMyPhotos desktop configuration

Source runs use the `BATCH_BLUR_*` development settings in
`BatchMyPhotos/desktop/src/main/config.js`. The packaged beta accepts only a
valid staging `blur-beta.json` resource for its origin and an OS-encrypted key
entered after install. It does not classify on import or auto-reject photos.

### Web/Mobile Backend `.env`

```bash
# ai-api connection
AI_API_URL=http://ai-api.internal:8000
AI_API_KEY=sk_webmobile_prod_xxxxxxxxxxxx

# (No blur settings — blur is desktop-only per root CLAUDE.md rule 6)

# Face search settings (per-event defaults, overridable in event settings)
FACE_MATCH_THRESHOLD_DEFAULT=0.6      # Minimum similarity to show match
FACE_TOP_K_DEFAULT=5                  # Max matches per face

# Bib OCR settings
BIB_CONFIDENCE_THRESHOLD_DEFAULT=0.7  # Minimum OCR confidence to trust
```

---

## Summary: Who Calls What

```
BatchMyPhotos Electron main:
  ✅ blur/classify/stream (primary), blur/classify (fallback/recovery)
  ❌ blur/detect, blur/*/batch, blur/*/mega, jobs/{id}, webhooks
  ❌ faces/*, bibs/*

Web/Mobile Backend:
  ❌ blur/*                (desktop-only — root CLAUDE.md rule 6)
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
