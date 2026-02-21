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

When the mobile app or website needs to call the AI API directly (instead of going through the backend), JWT-based auth will be used:
- The backend service issues JWTs to authenticated users
- The AI API validates them using a shared RS256 public key
- JWTs contain user ID, roles, and rate limit tier

This is configured but not enforced yet (Phase 7).

---

## Rate Limiting

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

Every response includes rate limit information:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1708567200   (Unix timestamp)
```

When rate limited:
```
HTTP 429 Too Many Requests
Retry-After: 12   (seconds)
```

### No Redis fallback

If Redis is unavailable, rate limiting is **disabled** (not enforced). This is a deliberate choice for development convenience. In production, Redis should always be available.

---

## Input Validation

### Image Upload Validation

Every uploaded image goes through multiple checks before any processing:

| Check | What It Does | Why |
|---|---|---|
| Content-Type header | Must be `image/jpeg`, `image/png`, or `image/webp` | Reject non-image files early |
| File size | Maximum 10MB | Prevent memory exhaustion |
| Magic bytes | Opens file with PIL and calls `.verify()` | A file renamed to .jpg is still detected as non-image |
| Dimensions | Min 32px, Max 4096px per side | Too small = useless. Too large = memory bomb. |
| EXIF stripping | Decoded via OpenCV (ignores EXIF) | Privacy: EXIF may contain GPS coordinates, device info |

### What happens on failure

```json
{
  "success": false,
  "error": {
    "code": "ImageValidationError",
    "message": "File exceeds 10MB limit"
  }
}
```

Status codes:
- 400: Wrong file type, corrupt image, too small
- 413: File too large

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
# Default: only localhost:3000 (your dev frontend)
ALLOWED_ORIGINS = ["http://localhost:3000"]

# Production example:
ALLOWED_ORIGINS = ["https://eventai.com", "https://app.eventai.com"]
```

**Allowed methods**: GET, POST, DELETE
**Allowed headers**: X-API-Key, Authorization, Content-Type, X-Request-ID
**Exposed headers**: X-Request-ID, X-RateLimit-Remaining, X-RateLimit-Reset

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

X-EventAI-Signature: sha256=<HMAC-SHA256(secret, body)>
```

The subscriber should verify the signature before trusting the payload:
```python
import hmac, hashlib
expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
received = request.headers["X-EventAI-Signature"].removeprefix("sha256=")
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
- [x] Rate limiting (token bucket via Redis)
- [x] Input validation (file type, size, dimensions, magic bytes)
- [x] EXIF stripping (privacy)
- [x] No image storage (only embeddings and hashes)
- [x] GDPR right-to-erasure endpoint
- [x] CORS configuration
- [x] Request ID tracing
- [x] Webhook HMAC signatures
- [x] Structured logging (no sensitive data in logs)
- [ ] HTTPS enforcement (handled at load balancer level)
- [ ] JWT authentication (Phase 7)
- [ ] pip-audit in CI (Phase 7)
