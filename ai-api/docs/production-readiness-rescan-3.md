# Production Readiness Rescan 3 -- Final Audit

**Date:** 2026-03-26
**Scope:** Face search, blur detection, bib number search -- full pipeline review
**Predecessor:** production-readiness-rescan-2.md (30 findings, all addressed)

---

## Executive Summary

Fourth and final production readiness scan before real-world testing. Focused deep-dive into all three ML pipelines end-to-end: API layer, ML inference, worker tasks, DB repositories, and cross-cutting middleware. Found 6 actionable issues (1 HIGH, 4 MEDIUM, 1 LOW) and 5 accepted/informational items. All actionable issues have been fixed.

---

## Findings Summary

| ID   | Severity | Component           | Finding                                              | Status  |
|------|----------|---------------------|------------------------------------------------------|---------|
| RS3-1 | HIGH    | `recognizer.py`     | Bib OCR timeout bypass -- generator not materialized | FIXED   |
| RS3-2 | MEDIUM  | `bibs.py`           | Unclamped YOLO bbox in API endpoint (worker was fixed)| FIXED   |
| RS3-3 | MEDIUM  | `face_tasks.py`     | Invalid person_id crashes batch enroll, leaves job stuck | FIXED |
| RS3-4 | MEDIUM  | `classifier.py`     | Missing inference timeout on blur classifier          | FIXED   |
| RS3-5 | MEDIUM  | `main.py`           | Security headers missing on timeout/rate-limit errors | FIXED   |
| RS3-6 | LOW     | `face_repo.py` +2   | Floating-point similarity can exceed [0,1] boundary  | FIXED   |
| RS3-7 | INFO    | `image_utils.py`    | image_dimensions reflects downscaled size, not original | ACCEPTED |
| RS3-8 | INFO    | `face_tasks.py`     | Progress reaches 100% before DB writes complete      | ACCEPTED |
| RS3-9 | INFO    | `timeout.py`        | Daemon threads continue running after timeout         | ACCEPTED |
| RS3-10| INFO    | `rate_limit.py`     | Rate limiting disabled when Redis is down (fail-open) | ACCEPTED |
| RS3-11| INFO    | `recognizer.py`     | PaddleOCR 3.x dict-like API may change across versions | ACCEPTED |

---

## Detailed Findings

### RS3-1 (HIGH) -- Bib OCR timeout bypass

**File:** `src/ml/bibs/recognizer.py:64-66`

**Problem:** `run_with_timeout(self.ocr.predict, ...)` wraps the call to `predict()`, but PaddleOCR 3.x `predict()` returns a **lazy generator**. The generator object is returned instantly (no computation), so the timeout considers the function "done." The actual OCR inference runs during `list()` iteration -- completely outside the timeout wrapper. A pathological image could hang the worker indefinitely.

**Fix:** Wrap `list(self.ocr.predict(img))` inside the timeout lambda so both generator creation AND iteration run under the timeout.

```python
# Before (broken -- generator escapes timeout):
results = list(
    run_with_timeout(self.ocr.predict, args=(img,), timeout_seconds=timeout)
)

# After (correct -- iteration inside timeout):
results = run_with_timeout(
    lambda img: list(self.ocr.predict(img)),
    args=(img,),
    timeout_seconds=timeout,
)
```

---

### RS3-2 (MEDIUM) -- Unclamped YOLO bbox in bib API endpoint

**File:** `src/api/v1/bibs.py:64-71`

**Problem:** The PR2-11 fix clamped bbox coordinates in `bib_tasks.py` (worker path) but missed the identical code in the sync API endpoint. YOLO can return coordinates slightly outside image bounds. In numpy, negative indices wrap around (`image[-3:100]` reads from near the end), producing a completely wrong crop that gets fed to OCR.

**Fix:** Added `max(0, ...)` / `min(dim, ...)` clamping and early `continue` for degenerate boxes, matching the worker task fix.

---

### RS3-3 (MEDIUM) -- Invalid person_id crashes batch enroll

**File:** `src/workers/tasks/face_tasks.py:191`

**Problem:** Two issues in batch enrollment when `person_id` is provided:
1. `_uuid.UUID(person_id)` is called without try/except. An invalid UUID string raises `ValueError` outside any exception handler, causing the Celery task to fail without marking the job as "failed" in the database. The job stays stuck in "processing" until `reap_stale_jobs()` eventually cleans it up.
2. Unlike the sync API endpoint which calls `repo.get_person(pid)` to verify the person exists, the batch path skips this check. A non-existent person_id causes every embedding insert to fail with an IntegrityError.

**Fix:** Added UUID parsing in try/except with `fail_job()`, and added person existence check before proceeding.

---

### RS3-4 (MEDIUM) -- Missing inference timeout on blur classifier

**File:** `src/ml/blur/classifier.py:167-168`

**Problem:** `FaceEmbedder._run_inference()` and `BibRecognizer.recognize()` both use `run_with_timeout()` to guard against hung ML inference. The `BlurClassifier.classify()` method calls `self.session.run()` (ONNX inference) without any timeout. In the API path, the `TimeoutMiddleware` provides a 60s safety net, but in the worker path (batch classification), a hung ONNX session could monopolize the worker indefinitely.

**Fix:** Wrapped `self.session.run()` with `run_with_timeout()` using the configured `INFERENCE_TIMEOUT`.

---

### RS3-5 (MEDIUM) -- Security headers missing on error responses

**File:** `src/main.py:160-165`

**Problem:** `SecurityHeadersMiddleware` was the **innermost** middleware (added first). The middleware stack was:

```
CORS > RequestID > Timeout > RateLimit > Security > endpoint
```

Error responses generated by outer middleware (504 from Timeout, 429 from RateLimit) flowed back through CORS and RequestID but **bypassed** SecurityHeadersMiddleware entirely. These responses lacked `X-Content-Type-Options`, `X-Frame-Options`, HSTS, etc.

**Fix:** Moved `SecurityHeadersMiddleware` to be added **last** (outermost), after CORS. Now all responses -- including timeout and rate-limit errors -- pass through it.

---

### RS3-6 (LOW) -- Floating-point similarity exceeds [0,1]

**Files:** `face_repo.py:128`, `sync_face_repo.py:93`, `faces.py:327`

**Problem:** pgvector's cosine distance `<=>` and numpy's dot product can produce floating-point values like `1.0000000000000002` for near-identical embeddings. The `FaceSearchResult` and `FaceCompareResponse` schemas enforce `Field(ge=0, le=1.0)`. A value of `1.0000000000000002` would trigger a Pydantic validation error and return HTTP 500 to the client.

**Fix:** Added `min(1.0, max(0.0, ...))` clamping at the point of computation in all three locations.

---

### RS3-7 (INFO/ACCEPTED) -- Image dimensions reflect downscaled size

`validate_and_decode()` applies `downscale_for_inference()` before returning the image. All endpoints call `get_image_dimensions(image)` on the downscaled image, so the response shows inference dimensions (e.g., 2048x1365) rather than the uploaded size (e.g., 4000x3000). Bounding box coordinates are also in the downscaled coordinate space.

**Accepted:** For EventAI's use case (runner matching, bib numbers), the primary outputs are person matches and bib text, not precise box coordinates. The desktop app can scale coordinates if needed using the reported dimensions.

---

### RS3-8 (INFO/ACCEPTED) -- Batch enroll progress reaches 100% before DB writes

In `face_enroll_batch()`, `update_job_progress()` runs during Phase 1 (inference). Phase 2 (DB writes) has no progress updates. A client polling progress sees 100% while DB writes are still in progress.

**Accepted:** Progress represents inference completion, which is the slow phase. DB writes are typically fast (< 1s per image). Clients should rely on job status ("completed"/"failed"), not progress percentage.

---

### RS3-9 (INFO/ACCEPTED) -- Daemon threads continue after timeout

When `run_with_timeout()` fires, the daemon thread continues running in the background consuming resources. Python doesn't support thread cancellation. The thread will eventually finish or be killed when the process exits.

**Accepted:** Known limitation of threading-based timeouts. The alternative (multiprocessing) has higher overhead. Timeout events are rare in practice.

---

### RS3-10 (INFO/ACCEPTED) -- Rate limiting disabled when Redis is down

`check_rate_limit()` returns a no-op when Redis is unavailable (fail-open). This is a deliberate availability-over-security tradeoff.

**Accepted:** For a capstone/startup deployment, availability is prioritized. Redis downtime should be monitored and alerted on.

---

### RS3-11 (INFO/ACCEPTED) -- PaddleOCR API reliance

`recognizer.py:73` uses `result.get("rec_texts", [])` on PaddleOCR 3.x result objects. The dict-like interface may change in future PaddleOCR versions.

**Accepted:** PaddleOCR version is pinned in requirements. This is a standard dependency risk.

---

## Files Modified

| File | Change |
|------|--------|
| `src/ml/bibs/recognizer.py` | RS3-1: Materialize generator inside timeout |
| `src/api/v1/bibs.py` | RS3-2: Clamp YOLO bbox coordinates |
| `src/workers/tasks/face_tasks.py` | RS3-3: Validate person_id UUID + existence check |
| `src/ml/blur/classifier.py` | RS3-4: Add inference timeout to ONNX run |
| `src/main.py` | RS3-5: Move SecurityHeadersMiddleware to outermost |
| `src/db/repositories/face_repo.py` | RS3-6: Clamp similarity to [0,1] |
| `src/db/repositories/sync_face_repo.py` | RS3-6: Clamp similarity to [0,1] |
| `src/api/v1/faces.py` | RS3-6: Clamp compare similarity to [0,1] |

---

## Production Readiness Assessment

After three full audit rounds (30 + 11 = 41 findings total, all addressed):

- **Security:** Auth, SSRF protection, input validation, security headers, rate limiting -- all in place
- **Reliability:** Inference timeouts, fork safety, connection pool guards, graceful shutdown -- all in place
- **Performance:** Image downscaling, batch processing, N+1 elimination, thread control -- all optimized
- **Error handling:** Per-image error isolation in batch, job lifecycle management, proper HTTP status codes

**Verdict: Ready for real-world testing.**
