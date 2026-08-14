# AI System Overview

Current state of the three ML pipelines in the QuickPitik `ai-api` service. All three are trained, exported, and wired into the API.

## Status Summary

| Module | Model | Accuracy | Artifact | Runtime |
|---|---|---|---|---|
| Blur Detection | Laplacian + FFT + YOLOv8n-cls | 98.68% top-1 (sharp: 100%) | `models/blur_classifier/blur_classifier.onnx` | ONNX Runtime (CPU/GPU) |
| Face Search | InsightFace `buffalo_l` (RetinaFace + ArcFace) | Production-grade pretrained | `models/models/buffalo_l/` | ONNX Runtime via InsightFace |
| Bib Search | YOLOv8n (custom) + PaddleOCR PP-OCRv5 | Dedicated bib detector (trained on 1,713 images) | `models/bib_detection/yolov8n_bib.onnx` | Ultralytics YOLO + PaddleOCR |

All models load once at startup via `src/ml/registry.py` and are served from `app.state.model_registry`. Unused InsightFace sub-models (genderage, extra landmarks) are dropped at load time to cut ~40% compute and memory.

---

## Blur Detection

**Endpoints:** `POST /api/v1/blur/detect`, `POST /api/v1/blur/classify`, plus `/stream` (NDJSON, up to 500 images), `/batch` (Celery, up to 50), `/mega` (Celery chord, up to 500) variants of each.

Two-tier pipeline:

1. **`BlurDetector`** (`src/ml/blur/detector.py`) — classical CV. Laplacian variance + optional FFT high-frequency ratio, normalised to a 640 px linear reference. `detect_fast(gray)` skips BGR→gray and FFT for the hot path used by streaming and Celery batches. No model file needed.
2. **`BlurClassifier`** (`src/ml/blur/classifier.py`) — YOLOv8n-cls (~5.5 MB ONNX). Classifies each image into one of four categories and powers `/blur/classify`:
   - `sharp` — subject in focus, background bokeh allowed
   - `defocused_object_portrait` — camera focused on wrong plane
   - `defocused_blurred` — whole frame soft
   - `motion_blurred` — camera shake or subject motion

**Key guarantee:** zero false positives on valid portraits (sharp subject + blurred background). Callers can pass `?blur_type=...` to get a targeted `detected: true/false` response; omitting it returns the full class probability vector.

**Served accuracy (2026-08-14, `scripts/blur_gate.py` over 1057 labelled val images through the live HTTP path):** 98.30% overall, with 4/698 blurry photos called `sharp` and 0/359 sharp photos called blurry. The 98.68% above is Ultralytics' training-time figure; the gap is the serving preprocess, which reimplements Ultralytics' PIL/torchvision transform in OpenCV. Both the decode step and the 224 px resize area-average — see `BLUR_CLASSIFY_DECODE_DIM` in `src/config.py` and `BlurClassifier._preprocess`. Re-check with `python scripts/blur_gate.py Training-Images/dataset/val` after any change to either.

## Face Search

**Endpoints:** `POST /api/v1/faces/detect`, `/enroll`, `/search` (with optional `event_id`), `/compare`, `GET /faces/persons`, `GET /faces/persons/{id}`, `DELETE /faces/persons/{id}`, plus `/search/batch`, `/search/mega`, `/enroll/batch`.

Pipeline:

1. **`FaceEmbedder`** (`src/ml/faces/embedder.py`) — InsightFace `buffalo_l` bundle. RetinaFace for detection, ArcFace R100 for 512-dim embeddings. Unused `genderage` and extra landmark sub-models are dropped at load time.
2. **Matcher** (`src/ml/faces/matcher.py`) — module-level `cosine_similarity` and `find_matches`. Uses the C++ `batch_cosine_topk` when available. Default similarity threshold 0.4.
3. **Storage** — embeddings stored as `pgvector` columns in Postgres. Search uses the `<=>` cosine operator with a LATERAL JOIN for batch queries. Tenant isolation enforced by `api_key_id`; `event_id` narrows further when provided.

Enrollment is quality-gated: faces below `FACE_MIN_ENROLLMENT_CONFIDENCE` (0.7) are skipped and `LOW_QUALITY` is returned if none pass.

The combined face+bib YOLO detector that appeared in earlier planning was superseded — face detection now runs through InsightFace directly, which gave better accuracy than the custom combined model.

## Bib Search

**Endpoints:** `POST /api/v1/bibs/recognize`, `/recognize/batch`, `/recognize/mega`.

Two-stage pipeline:

1. **`BibDetector`** (`src/ml/bibs/detector.py`) — custom YOLOv8n ONNX loaded via Ultralytics. Trained on 1,713 Roboflow-annotated event photos (1,370 train / 343 val, 2,461 bib boxes). Crops bib regions from the image. Refuses non-`.onnx` paths (no pickle).
2. **`BibRecognizer`** (`src/ml/bibs/recognizer.py`) — PaddleOCR 3.x (PP-OCRv5). Reads digits/alphanumerics from each crop through a thread pool (`OCR_MAX_WORKERS`). A regex filter (`[A-Za-z0-9\-_]`) and substitution table (`O→0`, `I→1`, `S→5`, `B→8`, `Z→2`) clean OCR output. `BIB_MIN_CHARS` (default 2) filters short noise; callers can override per request.

If the ONNX detector is missing, the endpoint falls back to running PaddleOCR on the full image and attaches a warning.

---

## Infrastructure

| Concern | Implementation |
|---|---|
| Async batch | Celery + Redis broker. Image bytes are written to `BLOB_STORE_PATH` and workers get file paths (not base64). `.../batch` → 202 + `job_id`; `.../mega` fans out via Celery chord and merges via `finalize_mega_batch`. |
| DB | PostgreSQL 16 + pgvector (jobs, webhooks, persons, API keys, face embeddings). Async engine (asyncpg) for the API; sync engine (psycopg2) for Celery workers. |
| C++ acceleration | pybind11 module `_quickpitik_cpp` (`src/cpp/`). Measured wins: Laplacian variance ~5×, cosine top-K ~1.8–2.8× for small/medium databases. All ops release the GIL. Pure-Python fallback always available. |
| Auth + rate limits | `X-API-Key` header (SHA-256 hashed, cached in Redis). Token-bucket rate limiter wired into every endpoint via `verify_api_key`. Tiers: free=60/min, pro=300/min, internal=1000/min. Per-key concurrent-job cap (`MAX_ACTIVE_JOBS_PER_KEY=10`). |
| Observability | structlog JSON logs with request_id context. Prometheus metrics auto-instrumented via `prometheus-fastapi-instrumentator`, exposed at `/metrics` (auth-protected in production, open when `DEBUG=true`). |

## Model Artifacts

```
ai-api/models/
  manifest.json                                  model registry metadata
  blur_classifier/
    blur_classifier.onnx                         YOLOv8n-cls, 4-class, ~5.5 MB
    class_names.json
  bib_detection/
    yolov8n_bib.onnx                             custom YOLOv8n bib detector
  models/buffalo_l/                              InsightFace pack (RetinaFace + ArcFace), auto-downloaded
```

Retraining scripts live in `ai-api/scripts/` (`train_blur_classifier.py`, `train_bib_detector.py`, plus the matching `export_*.py` scripts). Training data is under `ai-api/Training-Images/` (gitignored).

## Source Layout (ML only)

```
src/ml/
  registry.py                  loads all 5 models at startup
  blur/
    detector.py                Laplacian/FFT sharp-vs-blur check
    classifier.py              YOLOv8n-cls ONNX inference
  faces/
    embedder.py                InsightFace detect + embed
    matcher.py                 cosine similarity / top-K
  bibs/
    detector.py                YOLOv8n bib region detector
    recognizer.py              PaddleOCR PP-OCRv5 wrapper
```

See `architecture.md` for the full 4-layer design, `api-reference.md` for endpoint contracts, and `integration-contracts.md` for per-backend usage.
