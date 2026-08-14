# Security

## Authentication

### API Keys (primary method)

Every request (except health checks) must include an `X-API-Key` header.

**How it works:**
```
Client sends: X-API-Key: sk_live_abc123
    │
    ▼
Server hashes it: SHA-256("sk_live_abc123") → "a1b2c3d4..."
    │
    ▼
Checks Redis cache for "apikey:a1b2c3d4..."
    │
    ├── Cache hit → return stored metadata (scopes, rate tier)
    │
    └── Cache miss → query PostgreSQL api_keys table
                      │
                      ├── Found → cache in Redis (5 min TTL), return metadata
                      └── Not found → 401 Unauthorized
```

**Key properties:**
- Keys are **never stored in plain text**. Only SHA-256 hashes are in the database.
- Each key has **scopes** that control access (e.g., `blur:read`, `faces:write`, `bibs:read`)
- Each key has a **rate tier** (free, pro, internal)
- Keys can be **deactivated** without deletion (set `active=false`)

**Development mode:** When `DEBUG=true`, requests without an API key are allowed with full access. This only applies in development.

### JWT (future, for mobile/web)

When the mobile app or website needs to call the AI API directly (instead of going through the backend), JWT-based auth will be used. `JWT_PUBLIC_KEY` and `JWT_ALGORITHM=RS256` are already in `Settings` but no endpoint currently validates JWTs — only API-key auth is enforced today. The dependency is `PyJWT` with `cryptography`.

---

## Rate Limiting

Rate limiting is **enforced** on every authenticated endpoint. `verify_api_key` (`src/middleware/auth.py`) calls `check_rate_limit` (`src/middleware/rate_limit.py`) after successful authentication and stores the result on `request.state.rate_info`. `RateLimitHeadersMiddleware` in `main.py` reads that and attaches the `X-RateLimit-*` headers to every response.

### How it works

Token bucket algorithm implemented in Redis using a Lua script (atomic operation):

```
Each API key has a bucket:
  - Starts full (e.g., 60 tokens for free tier)
  - Each request removes 1 token
  - Tokens refill at a steady rate (1/second for free tier)
  - If bucket is empty → 429 Too Many Requests
```

### Tiers

| Tier | Max Burst | Refill Rate | Effective Limit |
|---|---|---|---|
| Free | 60 | 1/second | ~60 requests/minute |
| Pro | 300 | 5/second | ~300 requests/minute |
| Internal | 1000 | 16.7/second | ~1000 requests/minute |

### Response Headers

`X-RateLimit-*` headers are attached to **every** authenticated response by `RateLimitHeadersMiddleware`:

```
X-RateLimit-Limit: 60              # burst size for the tier
X-RateLimit-Remaining: 42          # tokens left after this call
X-RateLimit-Reset: 1708567200      # Unix timestamp when the limiter resets
```

On 429 responses these same headers are set, plus `Retry-After: <seconds>`.

> CORS note: `src/middleware/cors.py` currently exposes `X-RateLimit-Remaining` and `X-RateLimit-Reset` to browser clients. `X-RateLimit-Limit` is set on responses but not in the CORS expose list — add it if browsers need to read the per-tier ceiling.

### Extra backpressure for async batches

Each API key is capped at `MAX_ACTIVE_JOBS_PER_KEY` (default 10) concurrent async jobs. Submitting past that returns `429 TOO_MANY_JOBS` until earlier jobs complete.

### No Redis fallback

If Redis is unavailable, rate limiting is **disabled** (not enforced). This is a deliberate choice for development convenience. In production, Redis should always be available.

---

## Input Validation

### Image Upload Validation

Every uploaded image goes through multiple checks before any processing:

| Check | What It Does | Why |
|---|---|---|
| Content-Type header | Must be `image/jpeg`, `image/png`, or `image/webp` | Reject non-image files early |
| File size | Maximum 25MB (`MAX_FILE_SIZE`) | Prevent memory exhaustion. Matches the Spring backend's multipart ceiling — it forwards ORIGINAL photo bytes for indexing, so a lower bound here would fail photos it had already accepted. |
| Magic bytes | Opens file with PIL and calls `.verify()` | A file renamed to .jpg is still detected as non-image |
| Dimensions | Max 12000px per side (`MAX_IMAGE_DIMENSION`); min 32px on the single-image path | Too large = memory bomb. This is a fail-closed guard, not a quality gate — it is sized to pass any real camera (102 MP = 11648×8736), since every path downscales to `MAX_INFERENCE_DIMENSION` anyway. |

| EXIF handling | Pillow applies `ImageOps.exif_transpose` (preserves orientation) then the image is converted to a BGR numpy array which carries no EXIF | Privacy: EXIF metadata (GPS, device info, timestamps) is discarded as soon as the bytes leave Pillow |
| Total request body | Maximum 2 GB (`MAX_REQUEST_BODY`) | The only bound on a multipart upload *in aggregate*. Per-file limits cannot supply it: the `/stream` endpoints hold every file in memory at once, so 500 files each under `MAX_FILE_SIZE` still sum to 12.5 GB. Sized by the backend's mega drain at its worst case — 50 photos (its `batch.max-size`) × the 25 MB ceiling = 1250 MB. |

**Which paths enforce which check.**

| Path | Size | Dimensions | Content-Type | Magic bytes |
|---|---|---|---|---|
| 7 single-image endpoints (`validate_and_decode`) | ✅ | ✅ (+ min 32px) | ✅ | ✅ `.verify()` |
| 11 batch/mega endpoints (`validate_batch_file`) | ✅ | ✅ | ✅ | ✅ `.verify()` |
| 2 `/stream` endpoints (`validate_stream_file`) | ✅ | ✅ | ❌ | ❌ |

`validate_stream_file` is deliberately the narrow one. It drops `.verify()` because that walks the whole file while `cv2.imdecode` already fails closed on corrupt bytes, and it drops the Content-Type check because that is the one test that could reject a correct desktop client posting `application/octet-stream`. It keeps size and dimensions — the two `cv2.imdecode` cannot enforce for itself, since unlike Pillow it has no decompression-bomb guard.

A `/stream` rejection is reported as a per-image NDJSON error line, not an HTTP error; the request still returns 200 and the remaining images are still scored.

`MAX_REQUEST_BODY` is enforced by `BodySizeLimitMiddleware` (`src/main.py`) from the `Content-Length` header, before multipart is parsed. A chunked request declares no length and is not caught by it — the per-file layer still applies. Keep the value equal to `client_max_body_size` in `nginx.conf`; `TestBodySizeLimit` both parses that file for the value and hands it to a real `nginx -t` (the value check alone passed for months against a config nginx could not load).

**Memory consequence of the 2 GB ceiling.** The batch/mega paths spill to the blob store, so they hold no more than one file at a time. `/stream` does not — it reads every file into a list before processing, so this ceiling *is* its worst-case resident size. At `WORKERS=2` two saturating requests reach 4 GB of raw bytes against the 8 GB container limit in `docker-compose.prod.yml`. The exposure is theoretical today: the only `/stream` client is the desktop, which downscales to 1280px/q90 before uploading and sends ~49 MB per 200-image chunk. Revisit if a client ever streams originals.

### Face Enrollment Quality Gate

Face enrollment enforces a minimum detection confidence (`FACE_MIN_ENROLLMENT_CONFIDENCE`, default 0.7). Faces detected with confidence below this threshold are skipped to prevent low-quality embeddings from degrading search accuracy. If all detected faces are below the threshold, enrollment returns `LOW_QUALITY` error.

### Bib Number Validation

Bib text is cleaned using a strict character filter (`[A-Za-z0-9\-_]`) that preserves only alphanumeric characters, hyphens, and underscores. Leading and trailing `-` or `_` are stripped. A minimum character count (`BIB_MIN_CHARS`, default 2) ensures noise is rejected.

### What happens on failure

```json
{
  "success": false,
  "error": {
    "code": "ImageValidationError",
    "message": "File exceeds 25MB limit"
  }
}
```

Status codes:
- 400: Any per-image validation failure (wrong file type, corrupt image, too small, too large). The backing `ImageValidationError` always uses `status_code=400`.
- 413 `REQUEST_TOO_LARGE`: the whole request body exceeds `MAX_REQUEST_BODY`. This is the one case that is not an `ImageValidationError` — it is refused by middleware before any file is parsed, so it names no filename.

---

## Image Data Privacy

### What is stored

| Data | Stored? | Where | Notes |
|---|---|---|---|
| Original images | **NO** | Never stored | Images are processed in memory and discarded |
| Face embeddings | Yes | PostgreSQL (pgvector) | 512 floats per face. Cannot be reversed back to an image. |
| Image content hash | Yes | PostgreSQL | SHA-256 hash for deduplication. Cannot recreate the image. |
| EXIF metadata | **NO** | Stripped on upload | GPS, device info, timestamps are removed |
| API request logs | Yes | Structured logs | Contains request_id, endpoint, timing. No image data. |

### GDPR Compliance

- **Right to erasure**: `DELETE /api/v1/faces/persons/{id}` removes the person record and ALL associated embeddings. Cascading delete ensures nothing remains.
- **Data minimization**: Only embeddings (not images) are stored. Embeddings cannot be reversed to reconstruct a face.
- **Audit trail**: Deletion events are logged with timestamps and request IDs.

---

## CORS (Cross-Origin Resource Sharing)

Controls which websites/apps can call the API from a browser.

```python
# Default (src/config.py):
ALLOWED_ORIGINS = ["http://localhost:3000"]

# Production example:
ALLOWED_ORIGINS = ["https://quickpitik.com", "https://app.quickpitik.com"]
```

**Allowed methods**: GET, POST, DELETE
**Allowed headers**: X-API-Key, Authorization, Content-Type, X-Request-ID
**Exposed headers**: X-Request-ID, X-RateLimit-Remaining, X-RateLimit-Reset
**Credentials**: allowed
**Max age**: 3600s (preflight cache)

---

## Request Tracing

Every request gets a unique ID:

1. Client can send `X-Request-ID: my-trace-id` header
2. If not provided, server generates a UUID
3. The ID appears in:
   - Every log line for that request
   - The response `X-Request-ID` header
   - The `request_id` field in the JSON response body

This makes it possible to trace a single request through API logs, Celery worker logs, and database queries.

---

## Webhook Security

When delivering webhook callbacks, if the subscriber provided a secret:

```
Body: {"event": "job.completed", "job_id": "abc-123", ...}
Secret: "my_webhook_secret"

X-QuickPitik-Signature: sha256=<HMAC-SHA256(secret, body)>
```

The subscriber should verify the signature before trusting the payload:
```python
import hmac, hashlib
expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
received = request.headers["X-QuickPitik-Signature"].removeprefix("sha256=")
assert hmac.compare_digest(expected, received)
```

---

## Dependency Security

- **pip-audit**: Run in CI pipeline to check for known vulnerabilities in dependencies
- **Dependabot / Renovate**: Automated dependency update PRs
- **Pinned versions**: All dependencies in `pyproject.toml` have upper bounds (e.g., `>=0.115,<1`) to prevent unexpected breaking changes

---

## Summary Checklist

- [x] API key authentication (SHA-256 hashed, scoped)
- [x] Rate limiting (token bucket via Redis — enforced on every authenticated endpoint)
- [x] Per-key concurrent batch job cap (`MAX_ACTIVE_JOBS_PER_KEY`)
- [x] Input validation (file type, size, dimensions, magic bytes, decompression bomb guard) — single-image and batch/mega paths; **not** the two `/stream` endpoints
- [x] EXIF rotation applied, EXIF metadata discarded (privacy)
- [x] No persistent image storage (only embeddings, hashes, and short-lived Celery blob staging)
- [x] GDPR right-to-erasure endpoint
- [x] CORS configuration
- [x] Request ID tracing
- [x] Security response headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy; HSTS in production)
- [x] Webhook HMAC signatures + SSRF protection (DNS resolve + blocked-network check, TOCTOU-safe). Blocked ranges: `10/8`, `172.16/12`, `192.168/16`, `127/8`, `169.254/16`, `0.0.0.0/8`, `::1/128`, `fc00::/7`, `fe80::/10`.
- [x] Fernet encryption at rest for webhook secrets (when `WEBHOOK_SECRET_KEY` is set)
- [x] Structured logging (no sensitive data in logs)
- [x] Production DEBUG guard (server refuses to start with `DEBUG=true` + `ENVIRONMENT=production`)
- [x] Metrics endpoint authenticated in production
- [ ] HTTPS enforcement (handled at load balancer / reverse proxy)
- [ ] JWT authentication for client-direct access (infrastructure ready; no endpoint wired yet)
- [ ] Celery message signing (requires X.509 PKI setup)
- [ ] pip-audit in CI
