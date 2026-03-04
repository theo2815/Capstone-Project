# Implementation Phases

## Current Status: Phase 6 Complete | Phase 6.5 Pending

---

## Phase 1: Foundation [COMPLETE]

Everything needed to start the server, accept requests, and connect to infrastructure.

**What was built:**
- Project skeleton (68 files)
- `pyproject.toml` with all dependencies
- FastAPI app factory with lifespan (model loading/unloading)
- Pydantic Settings configuration (all env vars)
- Database layer (async SQLAlchemy + pgvector models + Alembic migrations)
- API key authentication middleware
- Rate limiting middleware (token bucket via Redis)
- Request ID middleware
- CORS configuration
- Structured logging (structlog)
- Prometheus metrics definitions
- Custom exception hierarchy
- All Pydantic request/response schemas
- All API route handlers (health, blur, faces, bibs, jobs, webhooks)
- Model registry singleton
- All ML model wrappers (blur detector, face embedder, face matcher, bib detector, bib recognizer)
- All service layer classes
- Celery configuration + task stubs
- Docker files (production + dev + compose + GPU overlay)
- Makefile with dev shortcuts
- Scripts (download_models, seed_db, benchmark)
- Test configuration (conftest.py)

**What you can do after Phase 1:**
- Start the server (`make dev`)
- See Swagger UI at `/docs`
- Hit health endpoints
- Authentication and rate limiting work
- Database tables are defined (need migration)

---

## Phase 2: Blur Detection [COMPLETE]

Goal: First ML feature end-to-end. Validates the full stack works.

**Tasks:**
- [x] Install dependencies in a virtual environment
- [x] Run `docker compose up db redis -d`
- [x] Create and run first Alembic migration
- [x] Seed database with dev API key
- [x] Start the server and test the health endpoint
- [x] Test blur detection with real images
- [x] Tune the Laplacian threshold with sample images (default 100.0 works well)
- [x] Write unit tests for BlurDetector (15 tests)
- [x] Write integration tests for POST /api/v1/blur/detect (13 tests)

**Test results:**
- Upload a sharp photo → `is_blurry: false` ✓
- Upload a blurry photo → `is_blurry: true` ✓
- Upload a non-image file → 400 error ✓
- Upload without API key (debug mode) → allowed ✓
- Upload corrupt image → 400 error ✓
- Upload too-small image → 400 error ✓
- Invalid API key → 401 error ✓

**Bugs fixed during Phase 2:**
- `structlog.get_level_from_name` removed in v25.x → replaced with `logging.getLevelName`
- `datetime.utcnow()` deprecated → replaced with `datetime.now(UTC)`
- `redis.close()` deprecated → replaced with `redis.aclose()`
- PaddleOCR 3.x: `use_gpu` param removed, `use_angle_cls` → `use_textline_orientation`
- PaddleOCR 3.x: slow connectivity check bypassed via `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`
- Alembic migration: missing `pgvector.sqlalchemy.vector` import in template and generated file
- Alembic `env.py`: added `CREATE EXTENSION IF NOT EXISTS vector` before migrations run

---

## Phase 3: Face Recognition [COMPLETE]

Goal: Most complex feature. Detection + embedding + database vector search.

**Tasks:**
- [x] Verify InsightFace models download on first run (buffalo_l, 282MB)
- [x] Test face detection endpoint with sample photos
- [x] Test face enrollment (POST /faces/enroll)
- [x] Verify embeddings are stored in PostgreSQL with pgvector
- [x] Test face search (POST /faces/search)
- [x] Test face comparison (POST /faces/compare)
- [x] Test person deletion (DELETE /faces/persons/{id})
- [ ] Test with real face photos (multiple people enrolled)
- [x] Tune similarity threshold (default 0.4)
- [x] Write unit tests for face matcher (11 tests)
- [x] Write integration tests for face endpoints (12 tests)

**Test results:**
- Face detect on random noise → `faces_detected: 0` ✓
- Enroll with no face → `NO_FACES` error ✓
- Search with no face → empty matches ✓
- Compare two no-face images → `NO_FACES` error ✓
- Get nonexistent person → `NOT_FOUND` error ✓
- Delete nonexistent person → `NOT_FOUND` error ✓
- Invalid UUID → 422 validation error ✓
- Missing person_name on enroll → 422 validation error ✓

**Note:** Full face-matching accuracy tests require real face photos. The pipeline is verified end-to-end; accuracy tuning is ongoing.

---

## Phase 4: Bib Number Recognition [COMPLETE]

Goal: Object detection + OCR pipeline.

**Approach:**
- Full pipeline: YOLO bib detection → crop → PaddleOCR (when custom YOLO model is available)
- Fallback mode: PaddleOCR on full image (current, until custom YOLO model is trained)
- The endpoint works with OCR-only; YOLO detector is optional and activates automatically when model file exists

**Tasks:**
- [x] Fix PaddleOCR 3.x (PP-OCRv5) compatibility:
  - Removed deprecated `show_log`, `use_gpu`, `cls` params
  - Replaced deprecated `ocr()` with `predict()` API
  - Adapted result parsing for new `OCRResult` dict-like objects
  - Set `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` to fix PIR+oneDNN crash on Windows
  - Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip slow connectivity check
- [x] PP-OCRv5 models downloaded from HuggingFace (PP-LCNet_x1_0_doc_ori, UVDoc, PP-LCNet_x1_0_textline_ori, PP-OCRv5_server_det, en_PP-OCRv5_mobile_rec)
- [x] Implement OCR-only fallback in endpoint (when bib_detector is unavailable)
- [x] Update BibService with same fallback logic
- [x] Make bib_detector optional in model registry readiness check
- [x] Verify OCR on synthetic number images (42 → 99.98%, 1234 → 98.07%)
- [x] Write unit tests for BibRecognizer (12 tests)
- [x] Write integration tests for POST /api/v1/bibs/recognize (12 tests)
- [ ] Train custom YOLOv8n bib detector (future — requires labeled bib dataset)

**Test results:**
- Number "1234" on image → `bib_number: "1234"`, confidence > 0.98 ✓
- Number "42" on image → `bib_number: "42"`, confidence > 0.99 ✓
- Blank image → `bibs_detected: 0`, empty detections ✓
- Response envelope format validated ✓
- PNG accepted ✓
- Unsupported content type → 400 ✓
- Corrupt image → 400 ✓
- No file → 422 ✓
- Image too small → 400 ✓

**Bugs fixed during Phase 4:**
- PaddleOCR 3.x `ocr()` method deprecated → switched to `predict()`
- PaddleOCR 3.x output format completely changed: old `[[bbox], (text, conf)]` → new `OCRResult` with `rec_texts`, `rec_scores` attributes
- PaddlePaddle 3.x PIR+oneDNN crash on Windows (`NotImplementedError: ConvertPirAttribute2RuntimeAttribute`) → fixed by disabling mkldnn via `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False`

**Full test suite: 75/75 passed**

---

## Phase 5: Async Batch Processing [COMPLETE]

Goal: Handle bulk image processing without blocking API responses.

**Architecture:**
- Celery tasks with Redis broker/backend for async processing
- Sync SQLAlchemy sessions (psycopg2) for Celery workers (can't use asyncpg)
- Worker model loading via `worker_process_init` Celery signal (loads models once per process)
- Base64 image transfer: API encodes file bytes for Celery JSON serialization
- Job lifecycle: pending → processing (with progress updates) → completed/failed
- Webhook dispatch on job completion/failure

**New files created:**
- `src/db/sync_session.py` — Sync SQLAlchemy engine/session for Celery tasks
- `src/db/repositories/sync_job_repo.py` — Sync job repository (get, update_progress, complete, fail)
- `src/db/repositories/sync_webhook_repo.py` — Sync webhook repository (list_by_event)
- `src/db/repositories/sync_face_repo.py` — Sync pgvector similarity search for batch face operations
- `src/workers/model_loader.py` — Worker model loading via Celery signals
- `src/workers/helpers.py` — Shared task utils (base64 decode, job progress, webhook dispatch)

**Tasks:**
- [x] Add `psycopg2-binary` dependency to `pyproject.toml`
- [x] Implement sync DB session module for Celery workers
- [x] Implement sync repositories (job, webhook, face)
- [x] Implement worker model loader with Celery signals
- [x] Implement shared task helpers (base64 decode, job progress, webhook dispatch)
- [x] Register model loader in `celery_app.py`
- [x] Implement batch blur detection task (`blur_tasks.py`)
- [x] Implement batch bib recognition task (`bib_tasks.py`)
- [x] Implement batch face processing task (`face_tasks.py`) — supports detect + search operations
- [x] Add `POST /blur/detect/batch` endpoint (returns 202 with job ID)
- [x] Add `POST /bibs/recognize/batch` endpoint (returns 202 with job ID)
- [x] Add `POST /faces/search/batch` endpoint (returns 202, supports `operation=detect|search`)
- [x] Wire up job progress tracking (updates DB during processing)
- [x] Wire up webhook delivery on job completion/failure
- [x] Write integration tests for batch processing (11 tests)
- [x] Verify full async flow: submit batch → get job ID → poll → completed with results

**Batch endpoint pattern:**
1. Accept multiple files via multipart upload
2. Validate batch (empty check, size limit of 100)
3. Create job record in DB (async)
4. Base64-encode all files for Celery JSON serialization
5. Queue Celery task with `.delay(job_id, image_data_list)`
6. Return HTTP 202 with `job_id` and `poll_url`
7. Poll `GET /api/v1/jobs/{job_id}` for status and results

**Test results:**
- Blur batch submit → 202 with job_id ✓
- Blur batch poll → completed with detection results ✓
- Blur result has `is_blurry`, `confidence` fields ✓
- No files → 422 validation error ✓
- Bib batch submit → 202 ✓
- Bib batch completed with results ✓
- Multi-image bib batch (2 images) → both processed ✓
- Face batch submit → 202 ✓
- Face detect batch → completed with `faces_detected` ✓
- Nonexistent job → NOT_FOUND error ✓
- Invalid job UUID → 422 ✓

**Full test suite: 86/86 passed (75 existing + 11 new batch tests)**

---

## Phase 6: C++ Acceleration [COMPLETE]

Goal: Optimize performance-critical paths with a pybind11 C++ extension module.

**Architecture:**
- pybind11 C++ extension module `_eventai_cpp` (compiled as `.pyd` on Windows, `.so` on Linux)
- Python fallback paths preserved for environments without a C++ compiler
- Build system: CMake + Ninja + MSVC (or GCC/Clang on Linux)
- `build_cpp.py` script handles: cmake configure → build → copy artifact to project root
- AVX2 auto-vectorization enabled via compiler flags
- GIL released during heavy computation loops for thread concurrency

**New files created:**
- `src/cpp/face_ops.h` / `face_ops.cpp` — `cosine_similarity`, `batch_cosine_topk` with `TopKResult` struct
- `src/cpp/blur_ops.h` / `blur_ops.cpp` — `laplacian_variance`, `fft_hf_ratio`, `batch_blur_metrics` with `BlurMetrics` struct
- `src/cpp/preprocess_ops.h` / `preprocess_ops.cpp` — `bgr_to_gray`, `resize_gray`
- `src/cpp/bindings.cpp` — `PYBIND11_MODULE(_eventai_cpp, m)` exposing all 9 functions and 2 structs
- `CMakeLists.txt` — Top-level CMake config (C++17, pybind11, AVX2 flags)
- `build_cpp.py` — Build script with MSVC vcvarsall auto-detection
- `tests/test_cpp_extension.py` — 31 tests across 8 test classes
- `benchmarks/bench_cpp_vs_python.py` — Timing comparisons for all operations

**Tasks:**
- [x] Install Visual Studio Build Tools 2026 with C++ workload
- [x] Install pybind11, cmake, ninja via pip
- [x] Create `src/cpp/` directory with header and source files
- [x] Implement `face_ops.cpp` — cosine similarity (dot product), batch cosine top-K (partial_sort, GIL release)
- [x] Implement `blur_ops.cpp` — manual Laplacian kernel with single-pass variance, radix-2 Cooley-Tukey 2D FFT
- [x] Implement `preprocess_ops.cpp` — luminance-weighted BGR→gray, bilinear resize
- [x] Write `bindings.cpp` — pybind11 module with all functions, TopKResult and BlurMetrics structs
- [x] Write `CMakeLists.txt` and `build_cpp.py` build scaffolding
- [x] Fix MSVC `ssize_t` → `py::ssize_t` compatibility (MSVC lacks POSIX ssize_t)
- [x] Build extension: `python build_cpp.py` → `_eventai_cpp.cp312-win_amd64.pyd`
- [x] Verify import: `python -c "import _eventai_cpp; print(_eventai_cpp.__version__)"` → `1.0.0`
- [x] Modify `src/ml/blur/detector.py` with C++ fallback pattern (same as matcher.py)
- [x] Write 31 C++ extension tests (numerical parity, edge cases, struct validation)
- [x] Write benchmark script comparing C++ vs Python for all operations
- [x] Update `pyproject.toml` — add cmake, ninja to `[project.optional-dependencies] cpp`
- [x] Update `.gitignore` — add CMake artifacts (CMakeCache.txt, CMakeFiles/, etc.)
- [x] Run full test suite: 117/117 passed (86 existing + 31 new)
- [x] Run benchmarks and record speedup numbers

**Python integration pattern** (existing in `matcher.py`, added to `detector.py`):
```python
try:
    from _eventai_cpp import laplacian_variance as _cpp_laplacian_var
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
```

**Benchmark results (MSVC 19.50, AVX2, Python 3.12, Windows):**

| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| Laplacian variance (256x256) | 0.40 ms | 0.075 ms | **5.4x** |
| FFT HF ratio (256x256) | 2.25 ms | 1.88 ms | **1.2x** |
| Cosine top-K (N=100, D=512) | 0.008 ms | 0.004 ms | **1.8x** |
| Cosine top-K (N=1K, D=512) | 0.11 ms | 0.038 ms | **2.8x** |
| Cosine top-K (N=10K, D=512) | 0.31 ms | 0.55 ms | 0.6x |
| Cosine top-K (N=100K, D=512) | 8.6 ms | 9.1 ms | 0.9x |
| Batch blur (100 × 128x128) | 28.2 ms | 54.9 ms | 0.5x |
| BGR→gray (1080x1920) | 0.63 ms | 7.35 ms | 0.1x |
| Resize gray (1080→270x480) | 0.08 ms | 1.17 ms | 0.1x |

**Analysis:**
- **Laplacian variance is the clear winner (5.4x)**: Single-pass variance (sum + sum_sq) avoids creating intermediate arrays, outperforming OpenCV's two-step approach.
- **Cosine top-K wins for small/medium databases (1.8-2.8x)**: GIL release + partial_sort is faster than NumPy matmul + argsort for N < ~5K.
- **FFT HF ratio marginal (1.2x)**: Our Cooley-Tukey radix-2 FFT is comparable to NumPy's FFTPACK.
- **Large-N cosine and preprocessing are slower**: NumPy uses BLAS (MKL/OpenBLAS) with hand-tuned SIMD intrinsics; OpenCV uses SSE/AVX for image operations. Naive C++ loops cannot compete with these heavily optimized implementations.
- **Key architectural benefit**: GIL release (`py::gil_scoped_release`) in all C++ functions enables concurrent Python thread execution during computation, improving throughput in multi-threaded server contexts.
- **Graceful fallback**: If C++ extension is not built, all functionality works identically via Python paths.

**Full test suite: 117/117 passed (86 existing + 31 new C++ tests)**

---

## Phase 6.5: API Abuse Prevention & Usage Control [PENDING]

Goal: Protect the API from abuse, control per-key usage, and prevent resource exhaustion — all within the existing monolith.

**Current state:** API key auth and file/batch size limits work. The token bucket rate limiter (`src/middleware/rate_limit.py`) is fully implemented but **never wired into any endpoint**. No usage tracking, no quotas, no task timeouts exist.

### 6.5.1 Activate Rate Limiter (High Priority)

The Redis token bucket is already built with 3 tiers. It just needs to be connected.

- [ ] Wire `check_rate_limit()` into the endpoint dependency chain (all routes)
- [ ] Verify 429 responses with `Retry-After` header for each tier:
  - `free`: 60 requests/min
  - `pro`: 300 requests/min
  - `internal`: 1000 requests/min
- [ ] Add rate limit headers to all responses (`X-RateLimit-Remaining`, `X-RateLimit-Limit`, `X-RateLimit-Reset`)
- [ ] Write integration tests for rate limiting (token exhaustion, tier switching, Redis failure fallback)

### 6.5.2 Celery Task Timeouts & Worker Protection (High Priority)

Batch tasks can currently run forever with no memory or time limits.

- [ ] Set `task_time_limit=3600` (hard kill after 1 hour) in `celery_app.py`
- [ ] Set `task_soft_time_limit=3300` (graceful shutdown warning at 55 min)
- [ ] Set `worker_max_tasks_per_child=50` (restart worker every 50 tasks to prevent memory leaks)
- [ ] Handle `SoftTimeLimitExceeded` in each task to save partial results before hard kill

### 6.5.3 Per-Key Concurrent Job Limits (High Priority)

Prevents a single key from flooding the Celery queue.

- [ ] Before accepting a new batch job, query active job count for the requesting API key
- [ ] Enforce max concurrent jobs per tier:
  - `free`: 2 concurrent batch jobs
  - `pro`: 10 concurrent batch jobs
  - `internal`: 50 concurrent batch jobs
- [ ] Return 429 with clear error when limit reached
- [ ] Write tests for concurrent job enforcement

### 6.5.4 Usage Audit Logging (Medium Priority)

Track all API usage for visibility, billing, and abuse detection.

- [ ] Create `api_usage_log` table (api_key_id, endpoint, request_id, timestamp, processing_time_ms, status_code, operation_type)
- [ ] Create Alembic migration for the new table
- [ ] Log every request via middleware (async write, non-blocking)
- [ ] Add admin endpoint: `GET /api/v1/admin/usage?key_id=...&from=...&to=...`

### 6.5.5 Monthly Quota Enforcement (Medium Priority)

Hard limits on total requests per billing period.

- [ ] Add `monthly_request_limit` and `monthly_requests_used` fields to `APIKey` model
- [ ] Create Alembic migration for the new fields
- [ ] Check quota in rate limit middleware; return 403 when exhausted
- [ ] Add quota reset logic (cron or Celery beat task on 1st of each month)
- [ ] Return `X-Quota-Remaining` and `X-Quota-Limit` headers

### 6.5.6 Endpoint Cost Weighting (Low Priority)

Different operations consume different server resources. Account for this in rate limiting and quotas.

- [ ] Define cost weights per endpoint:
  - `blur/detect`: 1 credit
  - `blur/classify`: 2 credits
  - `faces/detect`: 1 credit
  - `faces/search`: 5 credits (vector DB query)
  - `faces/enroll`: 3 credits
  - `bibs/recognize`: 3 credits
  - Batch endpoints: weight × number of images
- [ ] Deduct weighted credits from rate bucket and monthly quota
- [ ] Document credit costs in API docs

---

## Phase 7: Production Hardening [PENDING]

Goal: Make it production-ready.

**Tasks:**
- [ ] Add Prometheus metrics endpoint (/metrics)
- [ ] Set up Grafana dashboard templates
- [ ] Implement JWT authentication for mobile/web
- [ ] Add Sentry error tracking integration
- [ ] Create CI/CD pipeline (GitHub Actions):
  - Lint + type check
  - Run tests
  - pip-audit for dependency vulnerabilities
  - Build and push Docker image
- [ ] Load testing with Locust
- [ ] Write comprehensive API documentation
- [ ] Set up SSL/TLS termination guide
- [ ] Add database backup strategy documentation
- [ ] Create runbook for common operations

### 7.8 Microservice Readiness

Prepare the monolith for a future split into separate APIs per feature (blur, face, bib) without doing the split now. The current architecture already has zero cross-feature imports — these tasks make the eventual split trivial.

- [ ] Separate Celery queues per feature (`blur_queue`, `face_queue`, `bib_queue`) — enables independent worker scaling
- [ ] Feature-flag model loading (`ENABLED_FEATURES` env var) — skip loading unused models to reduce startup time and memory
- [ ] API gateway setup (Traefik or Nginx) — centralize auth, rate limiting, and CORS outside the app
- [ ] Document microservice split procedure in `docs/` — step-by-step guide for when scale demands it
