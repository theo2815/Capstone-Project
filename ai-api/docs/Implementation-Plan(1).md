# Implementation Plan: Production Readiness Rescan-2 (All 30 Findings)

## Context

The third production readiness audit (`production-readiness-rescan-2.md`) found 30 issues: 3 HIGH, 13 MEDIUM, 10 LOW, 4 INFO. This plan implements all fixes across 7 batches ordered by priority and file co-location. Each batch can be a separate commit.

---

## Batch 1 — P0 Critical Fixes (PR2-1, PR2-2, PR2-3)

### PR2-1: Fix blur normalization formula
**File:** `src/ml/blur/detector.py:67-72`

Replace pixel-count normalization with linear-resolution normalization:
```python
# BEFORE:
ref_pixels = 640 * 640
actual_pixels = h * w
if actual_pixels > 0:
    laplacian_var = laplacian_var * ref_pixels / actual_pixels

# AFTER:
ref_dim = 640
actual_dim = max(h, w)
if actual_dim > 0:
    laplacian_var = laplacian_var * ref_dim / actual_dim
```

### PR2-2: Add downscale to batch worker image decoder
**File:** `src/workers/helpers.py` — `decode_base64_image()`

Add `downscale_for_inference()` call after BGR conversion. Also add `raw_bytes` param to avoid double-decode (helps PR2-13):
```python
def decode_base64_image(b64_data: str, *, raw_bytes: bytes | None = None) -> np.ndarray | None:
    try:
        import io
        from PIL import Image, ImageOps
        from src.utils.image_utils import downscale_for_inference

        if raw_bytes is None:
            raw_bytes = base64.b64decode(b64_data)
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb_array = np.array(pil_img.convert("RGB"))
        image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        image = downscale_for_inference(image)
        return image
    except Exception:
        return None
```

### PR2-3: Replace broken UNION ALL with per-embedding loop
**File:** `src/db/repositories/face_repo.py` — `batch_search_similar()`

Replace the UNION ALL query builder (lines ~149-185) with a loop calling the existing correct `search_similar()`:
```python
async def batch_search_similar(self, embeddings, threshold=0.4, top_k=10,
                                api_key_id=None, event_id=None):
    results = []
    for emb in embeddings:
        matches = await self.search_similar(
            query_embedding=emb, threshold=threshold, top_k=top_k,
            api_key_id=api_key_id, event_id=event_id,
        )
        results.append(matches)
    return results
```

---

## Batch 2 — P1 Fixes (PR2-4, PR2-9, PR2-10, PR2-11, PR2-13, PR2-14)

### PR2-4: Return HTTP 503 when unhealthy
**File:** `src/api/v1/health.py:52-58`

When `healthy` is False, return `JSONResponse(status_code=503, content=response_data.model_dump(mode="json"))`.

### PR2-9 + PR2-10: Restructure batch enrollment (session scope + transaction)
**File:** `src/workers/tasks/face_tasks.py` — `face_enroll_batch()`

Split into two phases:
- **Phase 1 (no DB):** Decode images + run ML inference, collect `(image_hash, faces)` tuples
- **Phase 2 (DB only):** Open session, create/get person, store embeddings per image with per-image error handling

This fixes both the DB session held during inference (PR2-9) and the all-or-nothing transaction (PR2-10).

### PR2-11: Clamp YOLO bbox coordinates
**File:** `src/workers/tasks/bib_tasks.py:66-70`

```python
x1 = max(0, int(bbox["x1"]))
y1 = max(0, int(bbox["y1"]))
x2 = min(w, int(bbox["x2"]))
y2 = min(h, int(bbox["y2"]))
if x2 <= x1 or y2 <= y1:
    continue
```

### PR2-13: Eliminate double base64 decode
**File:** `src/workers/tasks/face_tasks.py` (enrollment loop)

Decode base64 once to `raw_bytes`, pass to `decode_base64_image(b64_data, raw_bytes=raw_bytes)`. Already enabled by the `raw_bytes` param added in Batch 1's PR2-2 fix.

### PR2-14: Validate API key on metrics endpoint
**File:** `src/main.py:184-193`

Replace presence check with hash verification against the DB (same logic as `verify_api_key`).

---

## Batch 3 — P2 Database & Query Fixes (PR2-5, PR2-16, PR2-19)

### PR2-5: Fix N+1 query in list_persons
**File:** `src/db/repositories/face_repo.py` — Add `list_persons_with_counts()`

Use `LEFT JOIN + GROUP BY` to get person + embedding count in one query:
```python
stmt = (
    select(Person, func.count(FaceEmbedding.id).label("emb_count"))
    .outerjoin(FaceEmbedding, Person.id == FaceEmbedding.person_id)
    .group_by(Person.id)
    ...
)
```

**File:** `src/api/v1/faces.py:370-393` — Update `list_persons` to use the new method.

### PR2-16: Unify webhook list+count filters
**File:** `src/db/repositories/webhook_repo.py`

Add `list_with_count()` method that builds filters once, applies to both list and count queries. Update `src/api/v1/webhooks.py` to call it.

### PR2-19: Add FOR UPDATE to job complete/fail
**Files:** `src/db/repositories/job_repo.py`, `src/db/repositories/sync_job_repo.py`

Replace `self.get(job_id)` with `select(Job).where(Job.id == job_id).with_for_update()` in `complete()` and `fail()`.

---

## Batch 4 — P2 API & Middleware Fixes (PR2-6, PR2-7, PR2-8)

### PR2-6: Use DB state for enroll response
**File:** `src/api/v1/faces.py:189-192`

Change `person_name=person_name` to `person_name=person.name` and `event_id=event_id` to `event_id=person.event_id`. The `person` variable is already fetched at line 137 when `person_id` is provided.

### PR2-7: Validate request ID format
**File:** `src/middleware/request_id.py`

Add regex validation: `^[a-zA-Z0-9\-]{1,128}$`. If invalid or missing, generate UUID.

### PR2-8: Convert SecurityHeadersMiddleware to pure ASGI
**File:** `src/main.py`

Rewrite `SecurityHeadersMiddleware` as a raw ASGI middleware (inject headers in `send_wrapper`). Keep `TimeoutMiddleware` and `RateLimitHeadersMiddleware` as `BaseHTTPMiddleware` since they need request state or asyncio features. Reduces buffering from 4x to 2x.

---

## Batch 5 — P2 ML & Infra Fixes (PR2-12, PR2-15, PR2-25)

### PR2-12: Add per-inference timeout
**New file:** `src/utils/timeout.py` — threading-based timeout wrapper
**Files:** `src/ml/faces/embedder.py`, `src/ml/bibs/recognizer.py`

Add `_run_inference()` helper with configurable timeout (default 120s via `INFERENCE_TIMEOUT` config). Uses daemon thread + `thread.join(timeout)`.

**File:** `src/config.py` — Add `INFERENCE_TIMEOUT: int = 120`

### PR2-15: Release GPU/ONNX resources in unload_all()
**File:** `src/ml/registry.py:58-62`

Before `self._models.clear()`, iterate models and delete ONNX sessions, InsightFace app, PaddleOCR engine.

### PR2-25: Consistent ONNX thread counts
**File:** `src/config.py` — Add `ONNX_INTRA_OP_THREADS: int = 2`, `ONNX_INTER_OP_THREADS: int = 1`
**File:** `src/ml/blur/classifier.py:73-75` — Read from config instead of hardcoded
**File:** `src/workers/model_loader.py` — Set `OMP_NUM_THREADS` env var before model loading for InsightFace/YOLO

---

## Batch 6 — P3 LOW Fixes (PR2-17 through PR2-24)

### PR2-17: Fix Celery auth serializer config
**File:** `src/workers/celery_app.py:59-75`

Replace with a log warning noting that proper PKI setup is required. Remove the non-functional `task_serializer="auth"` activation.

### PR2-18: Add form field constraints
**File:** `src/api/v1/faces.py` (enroll + batch enroll endpoints)

Add `min_length=1, max_length=255` to `person_name`, `max_length=255, pattern=r"^[a-zA-Z0-9_\-]+$"` to `event_id`.

### PR2-19: Already handled in Batch 3.

### PR2-20: Clear settings cache in tests
**File:** `tests/conftest.py` — Add `get_settings.cache_clear()` to test fixtures.

### PR2-21: Sync session eager init
Already addressed — `model_loader.py:48` calls `init_sync_db()` eagerly. No code change needed; the lazy fallback in `get_sync_session()` is safe.

### PR2-22: Log warning on decrypt failure
**File:** `src/utils/crypto.py:51-53` — Add `logger.warning(...)` before returning ciphertext.

### PR2-23: Add person_id index
**File:** `src/db/models.py:44` — Add `index=True` to `person_id` column.
**New migration:** `add_person_id_index` — `op.create_index('ix_face_embeddings_person_id', 'face_embeddings', ['person_id'])`

### PR2-24: Fork-safety guard for DB engine
**File:** `src/db/session.py` — Track `_engine_pid = os.getpid()` in `init_db()`. Check PID match in `get_session()`/`get_session_ctx()`.

---

## Batch 7 — P3 INFO Fixes (PR2-26 through PR2-30)

### PR2-26: Detection-only mode for face embedder
**File:** `src/ml/faces/embedder.py` — Use `det_model.detect()` directly in `detect_faces()`, skipping ArcFace embedding. Keep full pipeline as fallback.

### PR2-27: Generic APIResponse
**File:** `src/schemas/common.py` — Make `APIResponse` generic: `class APIResponse(BaseModel, Generic[T]): data: T | None = None`. Backwards-compatible.

### PR2-28: Fix PersonResponse datetime types
**File:** `src/schemas/faces.py:64-65` — Change `created_at: str` and `updated_at: str` to `datetime`.
**File:** `src/api/v1/faces.py` — Remove `.isoformat()` calls, pass datetime objects directly.

### PR2-29: Check webhook delete return value
**File:** `src/api/v1/webhooks.py:143-149` — If `deleted` is False, return error response.

### PR2-30: Add read-only session helper
**File:** `src/db/session.py` — Add `get_readonly_session()` that rolls back instead of committing. Optional use by read-only endpoints.

---

## File Conflict Map

Files appearing in multiple batches (implement in batch order to avoid conflicts):

| File | Batches | Changes |
|------|---------|---------|
| `src/api/v1/faces.py` | 3, 4, 6, 7 | N+1 fix, enroll response, form constraints, datetime |
| `src/workers/helpers.py` | 1, 2 | Downscale + raw_bytes param |
| `src/db/repositories/face_repo.py` | 1, 3 | UNION ALL fix + list_persons_with_counts |
| `src/main.py` | 2, 4 | Metrics auth + ASGI middleware |
| `src/ml/faces/embedder.py` | 5, 7 | Timeout + detect-only optimization |
| `src/config.py` | 5 | Inference timeout + ONNX threads |

---

## Verification Strategy

After each batch:
1. Run `make lint` (ruff/mypy)
2. Run `make test` (unit tests)
3. For Batch 1: manually test blur detection on a known-sharp high-res image
4. For Batch 2: test batch enrollment with a partially-corrupt image set
5. For Batch 3: verify SQL query count with echo enabled
6. For Batch 6: run `alembic upgrade head` for new migration
