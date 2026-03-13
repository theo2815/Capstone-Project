# Technical Audit Suggestions

> Generated: March 13, 2026
> Scope: `ai-api/` codebase and `docs/` architectural baseline

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Documentation-to-Code Alignment](#documentation-to-code-alignment)
- [Bugs (B)](#bugs-b)
- [Security (S)](#security-s)
- [Performance (P)](#performance-p)
- [Infrastructure (I)](#infrastructure-i)
- [Architecture Alignment Score](#architecture-alignment-score)
- [Recommended Priority Order](#recommended-priority-order)

---

## Executive Summary

The `ai-api` codebase is well-architected and closely follows its documented 4-layer design. The code is clean, consistently structured, and demonstrates strong engineering choices overall. This audit identified **27 findings** across four categories that should be addressed before production deployment. The most critical issues involve unenforced rate limiting, thread-safety in the blur service, silent data-loss edge cases in batch endpoints, and missing `asyncio.to_thread()` calls for CPU-bound inference.

---

## Documentation-to-Code Alignment

| Documented Requirement | Code Status | Notes |
|---|---|---|
| 4-layer architecture (API → Service → ML → DB) | **Partially followed** | Several API routes call ML models and DB directly, bypassing the service layer (see B-01). |
| Model Registry singleton at startup | **Aligned** | `ModelRegistry.load_all()` runs in the `lifespan` context manager. |
| `asyncio.to_thread()` for CPU-bound inference | **Not aligned** | No API route uses `asyncio.to_thread()` (see P-01). |
| C++ optional with fallback | **Aligned** | `try/except ImportError` pattern consistently used. |
| Celery for batch processing | **Aligned** | All batch endpoints: create job, queue task, return 202. |
| Rate limiting via token bucket | **Implemented, not wired** | Code exists in `rate_limit.py` but is never called from any route or middleware. |
| Prometheus metrics mounted at `/metrics` | **Not mounted** | Metrics defined in `metrics.py` but never recorded or exposed. |
| CORS exposes `X-RateLimit-Limit` | **Not aligned** | `cors.py` exposes `X-RateLimit-Remaining` and `X-RateLimit-Reset` but omits `X-RateLimit-Limit`. |
| Structured logging (never f-strings) | **Aligned** | `structlog` used consistently with key=value context. |
| Webhook HMAC signatures | **Aligned** | `deliver_webhook` signs with HMAC-SHA256 when secret is provided. |

---

## Bugs (B)

### B-01: Service Layer Bypass — API Routes Call ML and DB Directly

**Severity:** Medium
**Files:** `src/api/v1/blur.py`, `src/api/v1/faces.py`, `src/api/v1/bibs.py`

The architecture mandates that API routes call services, and services call ML/DB. However, every API route handler directly accesses `registry.get(...)` and calls model methods, and face/bib routes directly instantiate DB repositories. The `BlurService`, `FaceService`, `BibService`, and `JobService` classes are defined but **never used** by any API route.

**Impact:** Violates the documented architecture, makes testing harder (can't mock at the service layer), and scatters business logic across route handlers.

**Suggested fix:** Wire the service layer into the API routes via FastAPI dependency injection, or acknowledge this as an intentional simplification in the docs.

---

### B-02: Thread-Safety Bug in `BlurService.detect()` — Shared Mutable State

**Severity:** High
**File:** `src/services/blur_service.py` (lines 31–34)

```python
original = self.detector.laplacian_threshold
self.detector.laplacian_threshold = threshold
result = self.detector.detect(image)
self.detector.laplacian_threshold = original
```

This temporarily mutates the shared singleton `BlurDetector.laplacian_threshold`. Under concurrent requests with different thresholds, one request's threshold overwrite will affect another's detection. While the service is currently unused by routes (see B-01), if it were wired in as intended, this would be a race condition.

**Suggested fix:** Pass `threshold` as a parameter to `detector.detect()` instead of mutating shared state, or create a copy of the detector per request.

---

### B-03: `APIResponse.error` Field Type Mismatch

**Severity:** Medium
**File:** `src/schemas/common.py` (line 30) vs. usage throughout API routes

The schema defines `error: ErrorDetail | None` where `ErrorDetail` is a `BaseModel` with `code`, `message`, `field`. But every route handler passes a raw `dict`:

```python
error={"code": "MODEL_UNAVAILABLE", "message": "..."}
```

Pydantic v2 coerces dicts to models at runtime, so this works, but it's fragile. Any dict with an unexpected shape or missing field will cause a validation error at response time rather than at the point of error construction.

**Suggested fix:** Use `ErrorDetail(code="...", message="...")` explicitly in all route handlers.

---

### B-04: `datetime.utcnow()` Deprecation — Naive Datetime in Async Job Repository

**Severity:** Low
**File:** `src/db/repositories/job_repo.py` (lines 51, 59)

```python
job.completed_at = datetime.utcnow()  # Returns naive datetime (no timezone)
```

The sync repository (`sync_job_repo.py`) correctly uses `datetime.now(UTC)`, but the async repository uses the deprecated `datetime.utcnow()` which returns a tz-naive datetime. This is inconsistent with the database column type `DateTime(timezone=True)`.

**Suggested fix:** Replace `datetime.utcnow()` with `datetime.now(UTC)` in `job_repo.py`. Add `from datetime import UTC` to imports.

---

### B-05: `get_embeddings_count()` Loads All Embeddings Into Memory

**Severity:** Low
**File:** `src/db/repositories/face_repo.py` (lines 89–93)

```python
async def get_embeddings_count(self, person_id: uuid.UUID) -> int:
    result = await self.session.execute(
        select(FaceEmbedding).where(FaceEmbedding.person_id == person_id)
    )
    return len(result.scalars().all())
```

This loads every `FaceEmbedding` row (including the 512-dim vectors) into Python just to count them.

**Suggested fix:** Use a SQL count query:

```python
from sqlalchemy import func

result = await self.session.execute(
    select(func.count(FaceEmbedding.id)).where(FaceEmbedding.person_id == person_id)
)
return result.scalar_one()
```

---

### B-06: Route Handlers Can Exit Without Returning a Response

**Severity:** Medium
**File:** `src/api/v1/faces.py` (lines 107–145), also `webhooks.py`, `jobs.py`

The `return` statements are inside the `async for session in get_session():` block. If `get_session()` yields zero times (shouldn't happen, but defensively), or if an exception occurs before the return, the function will exit without returning an `APIResponse`, causing a 500 error with no structured error body.

**Suggested fix:** Add a fallback `return` after the `async for` block, or restructure to use `Depends(get_session)` for guaranteed lifecycle management.

---

### B-07: Batch Endpoints Read All Files Into Memory Before Any Validation

**Severity:** Medium
**Files:** `src/api/v1/blur.py`, `src/api/v1/faces.py`, `src/api/v1/bibs.py`

In all batch endpoints, every uploaded file is read and base64-encoded into a list before any image validation:

```python
for f in files:
    raw = await f.read()
    image_data_list.append(base64.b64encode(raw).decode("ascii"))
```

With 100 files at 10MB each, this could consume ~1.3GB of RAM (10MB × 1.33 base64 overhead × 100) with no image validation (type, size, dimensions) applied.

**Suggested fix:** Validate each file (at minimum size and content type) before base64 encoding. Consider streaming or storing batch images in object storage rather than loading all into memory.

---

## Security (S)

### S-01: Rate Limiting Is Not Enforced

**Severity:** Critical
**Files:** `src/middleware/rate_limit.py` (exists but unused), all API routes

The `check_rate_limit()` function is fully implemented but **never called** from any route handler or middleware. The API is completely open to abuse — any holder of a valid API key can make unlimited requests.

**Suggested fix:** Wire `check_rate_limit` as a FastAPI dependency or middleware. The code already supports this — it just needs to be called. Example:

```python
from src.middleware.rate_limit import check_rate_limit

@router.post("/detect")
async def detect_blur(
    request: Request,
    key_meta: dict = Depends(verify_api_key),
):
    await check_rate_limit(request, key_meta)
    # ... rest of handler
```

Or, create a combined dependency that runs auth + rate limit together.

---

### S-02: No Scope Enforcement on Endpoints

**Severity:** High
**Files:** All API routes, `src/middleware/auth.py`

While `verify_api_key` returns `key_meta` with scopes, and `check_scope()` exists in `auth.py`, **no endpoint calls `check_scope()`**. An API key with scope `["blur:read"]` can currently access face enrollment, webhook management, and job status.

**Suggested fix:** Add `check_scope()` calls in each route handler:

```python
@router.post("/enroll")
async def enroll_face(
    request: Request,
    key_meta: dict = Depends(verify_api_key),
):
    check_scope("faces:write", key_meta)
    # ...
```

Or create a scope-checking dependency factory:

```python
def require_scope(scope: str):
    async def _check(key_meta: dict = Depends(verify_api_key)):
        check_scope(scope, key_meta)
        return key_meta
    return _check
```

---

### S-03: Webhook URL SSRF (Server-Side Request Forgery) Risk

**Severity:** High
**Files:** `src/api/v1/webhooks.py`, `src/workers/tasks/webhook_tasks.py`

When registering a webhook, the `url` field is validated as an `HttpUrl` by Pydantic but there is no check against internal/private IP ranges. An attacker could register URLs like:

- `http://169.254.169.254/latest/meta-data/` (AWS metadata endpoint)
- `http://localhost:6379/` (Redis)
- `http://db:5432/` (PostgreSQL via Docker DNS)

The Celery worker would then make requests to these internal services.

**Suggested fix:** Validate webhook URLs against private IP ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x, localhost, and Docker service names). Consider using an allowlist or denying non-HTTPS URLs in production.

```python
import ipaddress
from urllib.parse import urlparse

BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "db", "redis", "ai-api"}

def validate_webhook_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname in BLOCKED_HOSTS:
        raise ValueError("Webhook URL points to an internal service")
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("Webhook URL points to a private IP range")
    except ValueError:
        pass  # hostname is a domain, not an IP — OK
```

---

### S-04: Webhook Secrets Stored in Plaintext

**Severity:** Medium
**File:** `src/db/models.py` (line 84)

```python
secret: Mapped[str | None] = mapped_column(String(255), nullable=True)
```

Webhook secrets are stored as plaintext in the database. If the database is compromised, all webhook secrets are exposed.

**Suggested fix:** Hash webhook secrets (similar to API keys) and only use the raw secret for HMAC signing at the moment of delivery. Alternatively, encrypt at rest if the secret needs to be recoverable for re-signing.

---

### S-05: Debug Mode Grants Full Admin Access Without Authentication

**Severity:** Medium
**File:** `src/middleware/auth.py` (lines 25–26)

```python
if settings.DEBUG and not api_key:
    return {"scopes": ["*"], "rate_tier": "internal", "key_id": "debug"}
```

In `DEBUG=true` mode, any request without an API key gets full `scopes: ["*"]` and the highest rate tier. If `DEBUG=true` accidentally leaks to production (e.g., a misconfigured environment variable), the entire API is unauthenticated.

**Suggested fix:** Add a secondary check (e.g., require the request to come from localhost), or log a prominent warning at startup when `DEBUG=true`. Consider using a separate `ALLOW_UNAUTHENTICATED` flag rather than overloading `DEBUG`.

---

### S-06: Client-Controlled `X-Request-ID` Without Sanitization

**Severity:** Low
**File:** `src/middleware/request_id.py` (line 16)

```python
request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
```

The client can inject arbitrary strings as the request ID. Since this value appears in logs and response headers, a malicious client could inject log-forging content (newlines, ANSI codes, or very long strings).

**Suggested fix:** Validate the client-provided `X-Request-ID` — accept only UUIDs or alphanumeric strings up to 64 characters, falling back to a generated UUID otherwise:

```python
import re

raw_id = request.headers.get("X-Request-ID", "")
if raw_id and re.fullmatch(r"[a-zA-Z0-9\-_]{1,64}", raw_id):
    request_id = raw_id
else:
    request_id = str(uuid.uuid4())
```

---

### S-07: No HTTPS Enforcement or HSTS Headers

**Severity:** Medium
**Context:** Infrastructure-level

The documentation notes this is "handled at load balancer level" but no middleware enforces HTTPS redirection or sets `Strict-Transport-Security` headers. If deployed without a properly configured reverse proxy, all API keys travel in plaintext.

**Suggested fix:** Add HSTS headers via middleware and consider a redirect middleware for production environments.

---

## Performance (P)

### P-01: CPU-Bound ML Inference Blocks the Async Event Loop

**Severity:** Critical
**Files:** `src/api/v1/blur.py`, `src/api/v1/faces.py`, `src/api/v1/bibs.py`

The architecture documentation states: *"CPU-bound ML inference runs in a thread pool via `asyncio.to_thread()`"*. However, **no API route uses `asyncio.to_thread()`**. All inference calls (`detector.detect()`, `embedder.get_embeddings()`, `bib_ocr.recognize()`) run synchronously on the main async event loop.

For face detection (~80ms CPU) and bib OCR (~30ms CPU), this blocks all other requests during inference.

**Suggested fix:** Wrap all ML inference calls in `asyncio.to_thread()`:

```python
import asyncio

# Before (blocks event loop):
result = detector.detect(image)

# After (runs in thread pool):
result = await asyncio.to_thread(detector.detect, image)
```

Apply to every route that calls ML model methods.

---

### P-02: Duplicate Image Parsing in `validate_and_decode()`

**Severity:** Low
**File:** `src/utils/image_utils.py` (lines 43–49)

The image is parsed by PIL twice — once for `verify()` (which invalidates the image object) and once for dimension checking. Then it's parsed a third time by OpenCV (`cv2.imdecode`). That's 3 complete parses of the same image data.

**Suggested fix:** After `verify()`, re-open once and reuse for dimension checking. The OpenCV decode is necessary for the BGR array, but the PIL double-open can be reduced:

```python
try:
    img = Image.open(io.BytesIO(contents))
    img.verify()
except Exception:
    raise ImageValidationError("Invalid or corrupt image file")

# verify() invalidates the image object, must re-open
img = Image.open(io.BytesIO(contents))
w, h = img.size
# ... dimension checks ...
# No need to call img again after this
```

This is already what the code does — the real savings would come from caching the decoded numpy array if PIL and OpenCV pipelines could be unified, but that's marginal.

---

### P-03: Webhook `list_by_event()` Fetches All Webhooks Then Filters in Python

**Severity:** Low
**Files:** `src/db/repositories/webhook_repo.py` (lines 44–52), `src/db/repositories/sync_webhook_repo.py`

```python
async def list_by_event(self, event: str) -> list[WebhookSubscription]:
    result = await self.session.execute(
        select(WebhookSubscription).where(WebhookSubscription.active.is_(True))
    )
    return [wh for wh in result.scalars().all() if event in wh.events]
```

This loads ALL active webhook subscriptions from the database and filters in Python.

**Suggested fix:** Use a PostgreSQL JSONB containment query to filter in the database:

```python
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import cast

query = select(WebhookSubscription).where(
    WebhookSubscription.active.is_(True),
    WebhookSubscription.events.contains([event]),
)
```

---

### P-04: Prometheus Metrics Defined But Never Recorded

**Severity:** Medium
**File:** `src/utils/metrics.py` and all route handlers

The metrics (`REQUEST_COUNT`, `REQUEST_LATENCY`, `INFERENCE_LATENCY`, etc.) are defined but never incremented or observed anywhere in the codebase. The `/metrics` scrape endpoint is not mounted.

**Suggested fix:** Integrate `prometheus-fastapi-instrumentator` for automatic request metrics, and manually record inference latency in ML wrappers:

```python
from src.utils.metrics import INFERENCE_LATENCY
import time

start = time.perf_counter()
result = model.predict(data)
INFERENCE_LATENCY.labels(model_name="blur").observe(time.perf_counter() - start)
```

Mount the scrape endpoint in `main.py`:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

---

### P-05: Base64 Encoding for Batch Processing Increases Memory by 33%

**Severity:** Medium
**Files:** All batch endpoints in `src/api/v1/`

Every batch endpoint base64-encodes file bytes for Celery JSON serialization, expanding data by ~33%. For 100 images at 10MB each, the Redis message is ~1.33GB.

**Suggested fix:** Consider one of:

1. Store batch images temporarily in a shared filesystem or object storage (MinIO/S3) and pass only references through Celery.
2. Use `msgpack` serialization with binary support instead of JSON + base64.
3. Compress the base64 data with `zlib` before sending to Redis.

---

## Infrastructure (I)

### I-01: No Celery Task Retry Configuration

**Severity:** High
**Files:** `src/workers/tasks/blur_tasks.py`, `face_tasks.py`, `bib_tasks.py`

Batch processing tasks have `bind=True` but define no retry policy. If a task fails due to a transient error (database connection drop, OOM), it fails permanently. Only the webhook delivery task has retries (via `tenacity`).

**Suggested fix:** Add retry configuration to Celery task decorators:

```python
@celery_app.task(
    bind=True,
    name="blur.detect_batch",
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(ConnectionError, TimeoutError),
)
def blur_detect_batch(self, job_id, image_data_list):
    # ...
```

---

### I-02: No Request Body Size Limit at the Server Level

**Severity:** Medium
**Context:** Infrastructure-level

While individual file uploads are checked against `MAX_FILE_SIZE` (10MB), there is no global request body limit. A crafted multipart request with many large parts could exhaust memory before the application-level check runs.

**Suggested fix:** Configure a body size limit on the reverse proxy (nginx: `client_max_body_size`) or use a FastAPI middleware. For Uvicorn, use `--limit-concurrency` and configure memory limits in Docker.

---

### I-03: PostgreSQL Connection Pool Not Sized for Production

**Severity:** Medium
**File:** `src/db/session.py` (lines 21–26)

`pool_size=20` is hardcoded. With `WORKERS=2` Uvicorn workers, each creates its own pool (40 baseline, 60 max connections). Adding Celery workers could exhaust PostgreSQL's default `max_connections=100`.

**Suggested fix:**

1. Make `pool_size` and `max_overflow` configurable via `Settings`.
2. Size them based on `WORKERS` count: e.g., `pool_size = max(5, 20 // WORKERS)`.
3. Consider using PgBouncer for connection pooling in production.

---

### I-04: `echo=settings.DEBUG` Logs All SQL Statements

**Severity:** Low
**File:** `src/db/session.py` (line 23)

When `DEBUG=true`, SQLAlchemy logs every SQL statement to stdout. With many concurrent requests, this generates enormous log volume and can degrade performance.

**Suggested fix:** Use a separate `DB_ECHO` config flag or only enable echo at `LOG_LEVEL=DEBUG`.

---

### I-05: Docker Compose Exposes DB and Redis Ports to Host

**Severity:** Medium
**File:** `docker-compose.yml` (lines 53, 66)

```yaml
ports:
  - "5432:5432"  # PostgreSQL exposed
  - "6379:6379"  # Redis exposed
```

In production, database and cache ports should not be exposed on the host network.

**Suggested fix:** Use a separate `docker-compose.prod.yml` that removes port mappings for `db` and `redis`, or bind only to localhost: `"127.0.0.1:5432:5432"`.

---

### I-06: No Graceful Handling of Redis Unavailability at Runtime

**Severity:** Low
**Files:** `src/main.py`, `src/middleware/auth.py`

Redis is checked at startup, but if Redis goes down during operation, `auth.py` will throw unhandled exceptions when trying to cache API key lookups.

**Suggested fix:** Wrap Redis operations in `auth.py` with try/except and fall back to database-only lookups:

```python
if redis:
    try:
        cached = await redis.get(f"apikey:{key_hash}")
        if cached:
            return json.loads(cached)
    except Exception:
        pass  # Redis unavailable, fall through to DB lookup
```

---

### I-07: `async for session` Pattern Is Not Idiomatic FastAPI

**Severity:** Low
**Files:** All route handlers that access the database

The `async for session in get_session():` pattern works but is not idiomatic FastAPI. It doesn't integrate with FastAPI's dependency injection lifecycle.

**Suggested fix:** Register `get_session` as a `Depends()` parameter:

```python
from fastapi import Depends

@router.get("/{job_id}")
async def get_job_status(
    request: Request,
    session: AsyncSession = Depends(get_session),
    key_meta: dict = Depends(verify_api_key),
):
    repo = JobRepository(session)
    # ...
```

---

## Architecture Alignment Score

| Area | Alignment | Key Gaps |
|---|---|---|
| 4-Layer Separation | 70% | Service layer exists but is entirely bypassed by API routes |
| Model Registry Pattern | 95% | Works as documented |
| Async Architecture | 75% | `asyncio.to_thread()` documented but not used for inference |
| Background Processing | 90% | Works well; needs retry policies |
| Security Implementation | 60% | Auth works; rate limiting and scopes not enforced |
| Observability | 40% | Structured logging works; metrics defined but unused |
| C++ Integration | 100% | Fallback pattern implemented correctly everywhere |
| Database Design | 90% | Models are clean; minor datetime and count query issues |

---

## Recommended Priority Order

| Priority | Finding | Category | Effort |
|---|---|---|---|
| 1 | **P-01** — Wrap inference in `asyncio.to_thread()` | Performance | Low |
| 2 | **S-01** — Wire rate limiting into endpoints | Security | Low |
| 3 | **S-02** — Enforce scope checks on endpoints | Security | Low |
| 4 | **S-03** — Block SSRF in webhook URLs | Security | Medium |
| 5 | **B-07** — Validate batch uploads before buffering | Bug | Medium |
| 6 | **I-01** — Add Celery task retries | Infrastructure | Low |
| 7 | **B-02** — Fix thread-safety in BlurService threshold | Bug | Low |
| 8 | **B-04** — Replace `datetime.utcnow()` with `datetime.now(UTC)` | Bug | Trivial |
| 9 | **B-05** — Use SQL `COUNT()` instead of loading all embeddings | Bug | Trivial |
| 10 | **P-04** — Mount Prometheus metrics endpoint | Performance | Medium |
| 11 | **S-04** — Hash or encrypt webhook secrets | Security | Medium |
| 12 | **I-03** — Make DB pool size configurable | Infrastructure | Low |
| 13 | **B-01** — Wire service layer into API routes | Bug | High |
| 14 | **S-05** — Harden DEBUG mode auth bypass | Security | Low |
| 15 | **S-06** — Sanitize client-provided X-Request-ID | Security | Trivial |
