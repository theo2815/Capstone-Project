# Integration Contracts — Clients-to-ai-api

**Date:** 2026-04-23 · **Updated 2026-10-16:** face and bib retired; the desktop app is the only client
**Companion doc:** `integration-architecture.md` (read that first for the full picture)
**Purpose:** Exact API usage patterns for each client. Copy-paste ready.

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

## Contract 2: Web/Mobile Backend (retired)

Retired 2026-10-16: the backend uses AWS Rekognition (`AI_PROVIDER=rekognition`, the backend's default); ai-api serves blur only, to the desktop app. The face and bib routes this contract described (`/faces/*`, `/bibs/*`) no longer exist.

### Shared client account and event rules

These routes belong to Spring Boot under `/api/v1` and were never ai-api routes; they stay listed here after the face/bib retirement. Clients never send account data to ai-api.

- Photographer metadata sharing: `GET`/`PUT /me/photographer/metadata-settings` exchange `{metadataPublicByDefault:boolean}`. The default is private. It is copied to a newly created photographer event or to first upload coverage of a shared event; changing it never changes existing event choices.
- `GET`/`PUT /me/photographer/events/{eventId}/metadata-visibility` exchange `{metadataPublic:boolean}` for the caller's coverage. A legacy owned event without coverage reads as private; its owner may create that relationship by PUT. Uncovered foreign events return 404. Upload counter upserts preserve an explicit choice.
- `GET /me/photographer/photos/{photoId}/metadata` lets the owning photographer read retained photos; `GET /public/photos/{photoId}/metadata` requires a LIVE, untrashed photo, readable event, approved unsuspended photographer, and that photographer's current event-sharing choice. Both return the typed `photoId`, `status`, nullable `originalFilename`, nullable `fileSizeBytes`, and nullable `fields` response with `private, no-store`; policy denials are indistinguishable 404s. Status is `pending`, `ready`, `empty`, `failed`, or `unavailable`; internal processing reads as pending and a missing legacy row as unavailable. `fields` is an object only for ready/empty, with nullable `cameraMake`, `cameraModel`, `lens`, `aperture` (positive finite number), `exposureTime`, `iso` (positive integer), `author`, `copyright`, `takenAtLocal`, `takenAtOffset`, `flash`, `exposureMode`, and `whiteBalance`.
- Gallery `PhotoDto` adds `metadataAccess: "none" | "public" | "owner"` as a UI routing hint; the metadata endpoints repeat authorization and gallery lists never embed field values. Metadata is read from Spring Boot's database, never fetched from storage on GET or passed to ai-api; raw EXIF, GPS and device serials are not exposed. The visibility setting controls only this API/popup: original downloads deliberately retain their embedded metadata.
- After committed LIVE publication, metadata extraction reuses the watermark worker's original array on the bounded `metadataProcessing` pool (`AsyncConfig`). Oversized arrays, saturation and stopped-executor submissions leave unclaimed rows for fenced recovery; recovery uses that pool only while watermark work is idle and may read the original again. Pool capacity and the enforced byte ceiling live in `AsyncConfig` and `PhotoMetadataService`; metadata completion does not gate LIVE.

- Google sign-in needs a verified Google ID token, never a QuickPitik password. Registration shares the same normalized, unique email namespace and returns `EMAIL_TAKEN` for an existing Google account.
- `PUT /me/password` requires the current password even when Google is linked. To set an initial password or recover one, use the existing `/auth/forgot-password` → `/auth/verify-reset-otp` → `/auth/reset-password` flow. The continuation token is issued only after email OTP verification; reset revokes every refresh token.
- `PUT /me/email` retains the Google-linked exemption from current-password verification (product decision, 2026-09-09); since 2026-10 only from a session minted after the account's last credential change (credential version, V88). Confirmation goes to the new inbox. This accepts the risk that a stolen authenticated Google-linked session can move the sign-in email and then use password recovery.
- Mobile race-day labels use `deriveEventDisplayState`. Gallery sockets and photographer syncing use `deriveEventState`, preserving the existing post-race upload window. Discovery uses `extractCity` until events provide structured region data; `GET /regions` remains the canonical region catalog for photographer settings.
- Handle changes are validated server-side against `reserved_handles`; V62 adds the account routes introduced in the client guard. Before release, query `SELECT p.user_id, p.handle FROM photographer_settings p JOIN reserved_handles r ON r.handle = lower(p.handle)` and arrange replacement handles with affected owners through the existing settings flow. The migration warns about collisions and preserves ownership.
- The profile's `#race-log` and `#saved-events` fragments both reach Saved events.

---

## Error Handling Contract

Callers must handle these ai-api error scenarios. The `Error Code` column shows what appears in the `error.code` field of the standard envelope (or `detail` when the response is a raw FastAPI `HTTPException`).

| HTTP Status | Error Code | Response shape | Caller Action |
|------------|-----------|---------------|----------------|
| 200 (success: false) | `NOT_FOUND` / `INVALID_WEBHOOK_URL` / `DELETE_FAILED` | Envelope | Flag, surface user-friendly message, or retry with better input |
| 400 | `ImageValidationError` (class name) | Envelope | Reject upload with user-friendly message (file type, size, dimensions, corrupt) |
| 400 | `EMPTY_BATCH` / `BATCH_TOO_LARGE` | Envelope | Recheck client-side batching logic |
| 401 | `AuthenticationError` (class name) **or** raw `{"detail": "Missing API key"}` | Envelope *or* raw | Log alert — API key may be expired or revoked |
| 403 | Raw `{"detail": "Insufficient permissions. Required scope: ..."}` | Raw | Request a key with the right scope |
| 404 | `NOT_FOUND` | Envelope | Fall through to "not found" in the caller |
| 429 | Raw `{"detail": "Rate limit exceeded. Retry after N seconds."}` | Raw (with `Retry-After` header) | Queue request for retry after `Retry-After` seconds |
| 429 | `TOO_MANY_JOBS` | Envelope | Wait for earlier batch jobs to finish before submitting more |
| 429 | `BLUR_KEY_BUSY` | Envelope (with `Retry-After: 1`) | This API key already has a blur request waiting or running; send one at a time and retry |
| 503 | `MODEL_UNAVAILABLE` | Envelope | Show "AI processing temporarily unavailable", queue for retry |
| 503 | `BLUR_BUSY` | Envelope (with `Retry-After: 1`) | Every blur slot stayed taken while the request waited in the server's first-come line (or the line was full); wait `Retry-After`, then retry |
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

A caller can check ai-api health before starting to process requests.

```python
async def check_ai_api_health():
    """Call on startup and periodically (every 30s)."""
    response = await httpx.get(
        f"{AI_API_URL}/api/v1/health/ready",
        timeout=5.0,
    )
    health = response.json()["data"]

    if not health["models_loaded"]:
        logger.warning("ai-api models not loaded — ML features unavailable")
    if not health["database"]:
        logger.error("ai-api database down — key lookups and batch jobs unavailable")
    if not health["redis"]:
        logger.warning("ai-api redis down — batch processing unavailable")

    return health
```

---

## Configuration Each Client Needs

### BatchMyPhotos desktop configuration

Source runs use the `BATCH_BLUR_*` development settings in
`BatchMyPhotos/desktop/src/main/config.js`. The packaged beta accepts only a
valid staging `blur-beta.json` resource for its origin and an OS-encrypted key
entered after install. It does not classify on import or auto-reject photos.

---

## Summary: Who Calls What

```
BatchMyPhotos Electron main:
  ✅ blur/classify/stream (primary), blur/classify (fallback/recovery)
  ❌ blur/detect, blur/*/batch, blur/*/mega, jobs/{id}, webhooks

Web/Mobile Backend:
  (no ai-api calls — retired 2026-10-16: face/bib go through AWS Rekognition; blur is desktop-only, root rule 6)
```
