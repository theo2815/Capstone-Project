# Post-Fix Rescan Audit Report: EventAI API

**Date:** 2026-03-25
**Scope:** Full codebase rescan after P0–P3 fix implementation
**Audited By:** Claude Code (Opus 4.6)
**Codebase State:** All 44 original findings fixed. This report covers NEW and MISSED issues only.

---

## Executive Summary

The P0–P3 fixes are well-implemented and the codebase is significantly more hardened. However, the rescan uncovered **13 new findings** — one is a **High severity cross-tenant data leak** introduced by an inconsistency between the async and sync face repositories. Several medium-severity issues relate to operational reliability and missing hardening at system boundaries.

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 3 |
| Medium | 5 |
| Low | 5 |

---

## High Severity

### RS-1 Batch face search leaks cross-tenant results — HIGH (Security)

- **File:** `src/db/repositories/sync_face_repo.py:13-45`
- **Description:** `SyncFaceRepository.search_similar()` does NOT accept or filter by `api_key_id`. When batch face search runs in a Celery worker via `face_tasks._search_single()`, it queries ALL face embeddings across all tenants. Meanwhile, the async `FaceRepository.search_similar()` correctly supports tenant filtering.
- **Impact:** API key A submits a batch face search and gets matches against faces enrolled by API key B. **Cross-tenant biometric data exposure.**
- **Root cause:** The SEC-2 tenant isolation fix was applied to the async repo but NOT to the sync repo used by Celery workers.
- **Code path:** `faces.py /search/batch` → `face_tasks.face_process_batch()` → `_search_single()` → `SyncFaceRepository.search_similar()` (no tenant filter)
- **Fix:**
```python
# sync_face_repo.py
def search_similar(
    self,
    query_embedding: list[float],
    threshold: float = 0.4,
    top_k: int = 10,
    api_key_id: str | None = None,  # ADD THIS
) -> list[dict]:
    query_vec = "[" + ",".join(str(f) for f in query_embedding) + "]"
    tenant_filter = ""
    params = {"query": query_vec, "threshold": threshold, "top_k": top_k}
    if api_key_id is not None:
        tenant_filter = "AND p.api_key_id = :api_key_id"
        params["api_key_id"] = api_key_id
    # ... use tenant_filter in SQL
```
Also update `_search_single()` in `face_tasks.py` to pass the `api_key_id` (requires propagating it through the Celery task arguments).
- **Priority:** **P0 — fix immediately**

---

### RS-2 SSRF DNS rebinding TOCTOU in webhook delivery — HIGH (Security)

- **File:** `src/workers/tasks/webhook_tasks.py:31-53, 96-99`
- **Description:** The SSRF check in `_validate_webhook_url()` resolves the hostname via `socket.getaddrinfo()` and validates the IP. But the actual HTTP request at line 98 (`client.post(url, ...)`) performs its own independent DNS resolution. Between the two lookups, an attacker-controlled DNS server can return a different IP (DNS rebinding), redirecting the request to `169.254.169.254` (cloud metadata), `127.0.0.1`, or other internal services.
- **Impact:** Full SSRF bypass. On cloud deployments, this can leak IAM credentials, instance metadata, and internal service data.
- **Fix:** Resolve DNS once, then make the HTTP request directly to the resolved IP with a `Host` header override:
```python
def _validate_webhook_url(url: str) -> tuple[str, str]:
    """Resolve and validate. Returns (resolved_ip, hostname)."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    addr_info = socket.getaddrinfo(hostname, None)
    ip = ipaddress.ip_address(addr_info[0][4][0])
    # ... validate ip ...
    return str(ip), hostname

# In deliver_webhook:
resolved_ip, hostname = _validate_webhook_url(url)
# Replace hostname with IP in URL, set Host header
target_url = url.replace(hostname, resolved_ip)
headers["Host"] = hostname
```
Alternatively, use `httpx` transport with a custom resolver or connect to the pre-resolved IP directly.
- **Priority:** **P1**

---

### RS-3 Stale jobs permanently consume backpressure slots — HIGH (Reliability)

- **File:** `src/db/repositories/job_repo.py:70-80`, `src/api/v1/batch_utils.py:82-84`
- **Description:** If a Celery worker crashes mid-task (OOM kill, hardware failure, network partition), the job remains in `"pending"` or `"processing"` status forever. The new backpressure system (`count_active_by_key`) counts these zombie jobs against the user's limit. After enough worker crashes, a user's job slots are permanently exhausted.
- **Impact:** Users get permanent 429 TOO_MANY_JOBS with no recovery path. Requires manual database intervention.
- **Fix:** Add a stale job reaper — either:
  - A) A periodic Celery beat task that marks jobs older than N minutes (e.g., `task_time_limit + buffer`) with status `"processing"` as `"failed"` with error `"Worker timeout — job expired"`.
  - B) Modify `count_active_by_key` to exclude jobs older than the time limit:
```python
async def count_active_by_key(self, api_key_id: str) -> int:
    cutoff = datetime.now(UTC) - timedelta(seconds=3600 + 300)  # task_time_limit + buffer
    result = await self.session.execute(
        select(func.count())
        .select_from(Job)
        .where(
            Job.api_key_id == api_key_id,
            Job.status.in_(["pending", "processing"]),
            Job.created_at > cutoff,
        )
    )
    return result.scalar_one()
```
- **Priority:** **P1**

---

## Medium Severity

### RS-4 No job record expiry or cleanup — MEDIUM (Scalability)

- **File:** `src/db/models.py:56-75`
- **Description:** Completed and failed jobs accumulate in the `jobs` table forever. The `result` column (JSONB) stores full batch results — for a 20-image batch, this could be several KB per job. At scale (thousands of batch jobs/day), the table grows unboundedly.
- **Impact:** Database bloat. Slower queries. Eventually disk exhaustion.
- **Fix:** Add a periodic cleanup job (Celery beat or cron) that deletes jobs older than a configurable retention period (e.g., 7 days). Or add a `TTL` column and a database-level scheduled cleanup.
- **Priority:** P2

### RS-5 Rate limit headers missing on successful responses — MEDIUM (API Quality)

- **File:** `src/middleware/rate_limit.py:48-95`, `src/middleware/auth.py:42,72`
- **Description:** `check_rate_limit()` returns a `rate_info` dict with `remaining`, `limit`, and `reset`, but this dict is discarded — no endpoint or middleware sets `X-RateLimit-*` response headers on successful requests. Users only discover their limits when they get a 429 error.
- **Impact:** Poor developer experience. No way for clients to implement proactive throttling.
- **Fix:** Add a middleware or `Response` dependency that reads `rate_info` from `request.state` and sets `X-RateLimit-Remaining`, `X-RateLimit-Limit`, `X-RateLimit-Reset` headers on every response.
- **Priority:** P2

### RS-6 Missing index on `jobs.api_key_id` — MEDIUM (Performance)

- **File:** `src/db/models.py:69`
- **Description:** `Job.api_key_id` has no database index. The new `count_active_by_key()` query filters by `api_key_id` + `status` on every batch request. While `status` is indexed, the composite filter requires scanning all rows matching a status to then filter by key.
- **Impact:** Slow backpressure checks under load. Every batch submission now does a COUNT with a partial index scan.
- **Fix:** Add `index=True` to `Job.api_key_id` and create an Alembic migration. Consider a composite index on `(api_key_id, status)` for optimal query performance.
- **Priority:** P2

### RS-7 Webhook events list accepts arbitrary strings — MEDIUM (Input Validation)

- **File:** `src/schemas/webhooks.py:11`
- **Description:** `WebhookCreateRequest.events` is typed as `list[str]` with no validation. An attacker can:
  - Register webhooks for non-existent events (no error, just never fires)
  - Submit extremely long event strings (stored in JSONB, fills database)
  - Submit an empty list (webhook that never fires)
- **Fix:** Validate events against an allowed list and add length/count constraints:
```python
ALLOWED_EVENTS = {"job.completed", "job.failed"}

class WebhookCreateRequest(BaseModel):
    url: HttpUrl
    events: list[str] = Field(..., min_length=1, max_length=10)
    secret: str | None = Field(default=None, max_length=256)

    @field_validator("events")
    @classmethod
    def validate_events(cls, v):
        invalid = set(v) - ALLOWED_EVENTS
        if invalid:
            raise ValueError(f"Invalid events: {invalid}")
        return v
```
- **Priority:** P2

### RS-8 No pagination on webhook list endpoint — MEDIUM (Scalability)

- **File:** `src/api/v1/webhooks.py:81-112`, `src/db/repositories/webhook_repo.py:40-45`
- **Description:** `GET /api/v1/webhooks` returns ALL webhooks for the API key in a single response. No `limit`/`offset` parameters. A user with hundreds of webhooks gets an enormous response.
- **Impact:** Memory spike on large responses. Potential timeout on the 60s middleware.
- **Fix:** Add `limit` and `offset` query parameters (default limit=50, max=100).
- **Priority:** P3

---

## Low Severity

### RS-9 `[{}] * total` creates shared mutable references — LOW (Code Quality)

- **Files:** `src/workers/tasks/blur_tasks.py:31,84`, `face_tasks.py:32`, `bib_tasks.py:32`
- **Description:** `results: list[dict] = [{}] * total` creates a list where ALL elements reference the SAME dict object. The current code is safe because every assignment does `results[i] = {...}` (replaces the reference), but if anyone later writes `results[i]["key"] = value` (mutates in-place), all elements would be affected.
- **Fix:** Use `[{} for _ in range(total)]` or simply `[None] * total` since every slot gets fully replaced.
- **Priority:** P3

### RS-10 Batch file validation skips content-type header check — LOW (Defense in Depth)

- **File:** `src/utils/image_utils.py:109-123`
- **Description:** `validate_batch_file()` validates via PIL's `verify()` but does not check the `Content-Type` header like `validate_and_decode()` does. This is LOW because `Content-Type` is client-controlled and easily spoofable — PIL's magic-byte verification is the real check. But for defense-in-depth consistency, both paths should behave identically.
- **Fix:** Optionally add content-type checking to `validate_batch_file`, or document the intentional difference.
- **Priority:** P3

### RS-11 Fernet instance cached with no runtime invalidation — LOW (Operations)

- **File:** `src/utils/crypto.py:10-28`
- **Description:** The `_fernet` global is cached on first call to `_get_fernet()`. If `WEBHOOK_SECRET_KEY` is rotated at runtime (e.g., via environment variable update), the old key continues to be used until the process restarts. Old secrets encrypted with the previous key become undecryptable (silently falling back to returning ciphertext as plaintext at line 48).
- **Impact:** Key rotation requires full restart. Silent decryption failures during transition.
- **Fix:** Document that key rotation requires process restart. Or remove the caching and re-read the setting each time (minimal overhead since it's called infrequently).
- **Priority:** P3

### RS-12 Pre-decode holds all batch images in memory simultaneously — LOW (Resource)

- **File:** `src/workers/tasks/blur_tasks.py:34-39` (and face/bib variants)
- **Description:** The PERF-8 pre-decode optimization decodes ALL images before processing any. For a 20-image batch at max resolution (4096x4096x3 bytes = ~50MB each), this peaks at ~1GB of numpy arrays in memory simultaneously. The previous sequential approach held only one decoded image at a time.
- **Impact:** Higher peak memory in workers. Within the 8GB Docker limit but reduces headroom.
- **Fix:** Consider chunked pre-decode (decode in batches of 5) if memory pressure is observed. Current approach is acceptable for the MAX_BATCH_SIZE=20 limit.
- **Priority:** P3

### RS-13 Sync job progress update lacks FOR UPDATE — LOW (Consistency)

- **File:** `src/db/repositories/sync_job_repo.py:22-30`
- **Description:** The async `JobRepository.update_progress()` uses `SELECT ... FOR UPDATE` (BUG-5 fix), but the sync `SyncJobRepository.update_progress()` does a plain SELECT. These are functionally equivalent since each Celery task processes one job serially, but the inconsistency means a future change (e.g., parallel sub-tasks for a single job) could introduce a race condition.
- **Fix:** Add `.with_for_update()` to the sync variant for consistency.
- **Priority:** P3

---

## Prioritized Action Items

### P0 — Fix Immediately (Security)
| # | Finding | Impact | Effort |
|---|---------|--------|--------|
| 1 | RS-1: Add tenant isolation to `SyncFaceRepository.search_similar()` and propagate `api_key_id` through batch face tasks | Cross-tenant biometric data leak | Small |

### P1 — Fix This Sprint
| # | Finding | Impact | Effort |
|---|---------|--------|--------|
| 2 | RS-2: Fix SSRF DNS rebinding by connecting to pre-resolved IP | Full SSRF bypass on cloud | Medium |
| 3 | RS-3: Add stale job reaper or time-bounded active job count | Users permanently locked out | Small |

### P2 — Fix Soon
| # | Finding | Impact | Effort |
|---|---------|--------|--------|
| 4 | RS-4: Add job record expiry/cleanup | Database bloat | Small |
| 5 | RS-5: Return rate limit headers on all responses | Poor DX | Small |
| 6 | RS-6: Add index on `jobs.api_key_id` | Slow backpressure queries | Small |
| 7 | RS-7: Validate webhook events against allowed list | Input validation gap | Small |

### P3 — Backlog
| # | Finding | Impact | Effort |
|---|---------|--------|--------|
| 8 | RS-8: Add pagination to webhook list | Scalability | Small |
| 9 | RS-9: Fix `[{}] * total` shared reference pattern | Code quality | Trivial |
| 10 | RS-10: Add content-type check to batch validation | Defense in depth | Trivial |
| 11 | RS-11: Document Fernet key rotation requires restart | Operations | Trivial |
| 12 | RS-12: Monitor worker memory with pre-decode pattern | Resource usage | N/A |
| 13 | RS-13: Add FOR UPDATE to sync job progress update | Consistency | Trivial |

---

## Summary of Changes Since Last Audit

The P3 implementation introduced 4 new code patterns worth noting:

1. **`batch_utils.py`** — Well-structured, clean extraction. The `create_batch_job` return type is `str | JSONResponse` which works but is an unusual pattern; callers must remember the isinstance check.

2. **Backpressure system** — Correct implementation, but RS-3 (stale jobs) and RS-6 (missing index) need follow-up.

3. **Pre-decode optimization** — Functional and correct, but the `[{}] * total` pattern (RS-9) is a footgun and peak memory increased (RS-12).

4. **Celery message signing** — The `security_key` config approach is Celery 5.x compatible. Note: the `"auth"` serializer requires `pyOpenSSL` and certificate files for full security. The current implementation enables the config but the actual signing depends on the worker having the right dependencies installed.

---

*Generated by Claude Code (Opus 4.6) — 2026-03-25*
*Rescan of codebase after P0-P3 fix implementation from deep-audit-report.md*
