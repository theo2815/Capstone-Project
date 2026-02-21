# Implementation Phases

## Current Status: Phase 4 Complete

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

## Phase 5: Async Batch Processing [PENDING]

Goal: Handle bulk image processing without blocking API responses.

**Tasks:**
- [ ] Implement batch blur detection task (blur_tasks.py)
- [ ] Implement batch face processing task (face_tasks.py)
- [ ] Implement batch bib recognition task (bib_tasks.py)
- [ ] Add batch endpoints (POST /blur/detect/batch, POST /bibs/recognize/batch)
- [ ] Implement job progress tracking (update DB during processing)
- [ ] Wire up webhook delivery on job completion
- [ ] Test the full async flow:
  - Submit batch → get job ID
  - Poll job status → see progress
  - Job completes → results available
  - Webhook callback fires
- [ ] Write integration tests for batch processing

---

## Phase 6: C++ Acceleration [PENDING]

Goal: Optimize performance-critical paths.

**Tasks:**
- [ ] Implement `distance_ops.cpp` (batch cosine similarity + top-K)
- [ ] Implement `blur_ops.cpp` (batch Laplacian variance)
- [ ] Implement `preprocess_ops.cpp` (fused image preprocessing)
- [ ] Implement `bindings.cpp` (pybind11 module)
- [ ] Set up CMake build
- [ ] Verify C++ extension compiles and imports
- [ ] Run benchmarks: Python vs C++ comparison
- [ ] Update Docker build to compile C++ in build stage
- [ ] Write C++ unit tests

**Expected outcomes:**
- Batch cosine similarity: 5-10x speedup
- Batch Laplacian: 2-3x speedup
- Fused preprocessing: 2-4x speedup

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
