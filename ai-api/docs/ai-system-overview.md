# AI System Overview

Current state of the ML pipeline in the QuickPitik `ai-api` service: blur detection, used by the desktop app. Face search and bib search were removed on 2026-10-16; the backend uses AWS Rekognition (`AI_PROVIDER=rekognition`) for both.

## Status Summary

| Module | Model | Accuracy | Artifact | Runtime |
|---|---|---|---|---|
| Blur Detection | Laplacian + FFT detector; two-stage classifier (whole-frame rule, then people-first: person detector + per-person sharpness + logistic rule) | See quality evidence below | `models/person_detector/yolov8n_seg_person.onnx`, `models/blur_classifier/people_rule.json` | ONNX Runtime (CPU/GPU) |

Both models (`blur`, `blur_classifier`) load once at startup via `src/ml/registry.py` and are served from `app.state.model_registry`.

---

## Blur Detection

**Endpoints:** `POST /api/v1/blur/detect`, `POST /api/v1/blur/classify`, plus `/stream` (NDJSON, up to 500 images), `/batch` (Celery, up to 50), `/mega` (Celery chord, up to 500) variants of each.

Two-tier pipeline:

1. **`BlurDetector`** (`src/ml/blur/detector.py`) — classical CV. Single, stream, and Celery use the same grayscale decode target (`BLUR_DETECT_DECODE_DIM`) and OpenCV Laplacian score, normalized to a 640 px reference. FFT is optional and informational. No model file needed.
2. **`BlurClassifier`** (`src/ml/blur/classifier.py`) — two stages, four classes reported, powers `/blur/classify`:
   - **Stage 1, no model** (`src/ml/blur/whole_frame.py`): two numbers measured on a 640-px gray copy of the upload — fine detail (the Laplacian score) and the up-down / left-right detail ratio. When nothing in the frame is sharp the answer is `defocused_blurred` (whole-frame soft) or `motion_blurred` (sideways smear), with probability 1.0. Thresholds: `BLUR_SOFT_MAX_DETAIL`, `BLUR_MOTION_MIN_RATIO`, `BLUR_MOTION_MAX_DETAIL`, learned by `scripts/fit_blur_whole_frame_rule.py`.
   - **Stage 2, people first** (`src/ml/blur/person_detector.py`, `people.py`): a COCO person detector (YOLOv8n, ONNX Runtime) finds the people; each person's edge sharpness is measured on its own pixels at the 1280-px upload; a logistic rule (`models/blur_classifier/people_rule.json`, fitted by `scripts/fit_blur_people_rule.py` on the labelled event folds) turns thirteen numbers about the people — how sharp the subject (the largest person the detector is confident about) is next to the sharpest, how big the blurred people are, whether a blurred person is cut off by the frame or stands at the subject's feet or nearer, and how big the boxes the detector is unsure about are — into P(`defocused_object_portrait`) versus `sharp`.
   - `sharp` — the runner is sharp; background blur is intended; slight softness still counts as sharp
   - `defocused_object_portrait` — subject missed focus: a blurred person in front cuts into the frame, or the runner is soft while the background or the road is sharp
   - `defocused_blurred` — whole-frame soft: nothing in the frame is sharp
   - `motion_blurred` — the whole frame is smeared sideways

Callers can pass `?blur_type=...` to get a targeted `detected: true/false` response; omitting it returns the full class probability vector. A sharp subject with intentional background bokeh belongs to `sharp`.

**Quality evidence (2026-09-24):** The old 98.68% training-time figure and 98.30% served figure came from a validation split with 31 train-overlapping files. After excluding those files, the unchanged ONNX model classified 1,009/1,026 correctly, missed 4/667 blurry photos, and culled 0/359 sharp photos. This is diagnostic because the historical validation data informed model and preprocessing choices. On a separately selected, visually labeled night-marathon sample, it culled **36/120 sharp runner photos** (30%; Wilson 95% interval 22.5–38.7%). All 120 were sharp, so blur misses and overall four-class accuracy cannot be assessed on that sample. The photos came from one event and camera; labels have not had a second human review. Near-burst correlation also limits the interval's generalizability. See the 2026-09-24 Task 8 vault journal and ignored `Training-Images/quality_evidence/` manifests for full provenance. No production accuracy or zero-over-cull guarantee is established.

**Root cause (2026-09-29):** Real-event over-culling is shortcut learning, not a pipeline fault. Training labels were assigned per shoot, and 99.1% of the historical validation images share a shoot with training, so validation rewarded recognising shoots (a frozen ImageNet feature probe with no blur training scores 96% on it). On 736 sharp frames from two new events the served model flagged 67%; brightening or keeping colour alone raises its flags. Do not quote 98% as accuracy. The recovery plan is in the vault journal `2026-09-29-blur-detection-investigation-and-improvement-plan`.

**People-first redesign (2026-10):** The 224-px centre-crop classifier and its 640-px whole-frame successor were replaced after the 2026-10-05 and 2026-10-06 investigations showed that a whole-frame model learns the look of the shoot (night, lens, blurred people in the background) instead of the runner's focus. Stage 2 now finds the people and measures each one. The first people-first rule (nine numbers, version `2026-10-07-people-first`) met every limit on the five event folds but failed its one-run final exam on a fresh event narrowly at the 0.70 flag threshold (2026-10-08: sharp photos wrongly flagged 2.19%, missed focus not caught 10.87%, tone flips 2.80%): costumes and inflatable arches gave low-confidence boxes that the rule took for blurred bodies, blurred runners beside the subject counted like a blurred person in front, and darkened copies grew ghost "people" in dark backgrounds. The round-three rule (thirteen numbers, version `2026-10-08-people-first-r3`) chooses the subject among confident boxes, ignores duplicate boxes, and tells a soft person at the subject's feet from one behind; the spent exam joined the training folds as two events. The detector is scored only on events it never saw: five event folds over 9,327 labelled photos from 24 events. Pooled result at the Moderate flag threshold 0.70: sharp photos wrongly flagged 67/5,443 = 1.23% (worst event with at least 100 sharp photos: `trilogy-run-asia-leg3-2026-08`, 6/165 = 3.6%); whole-frame soft and motion missed 0/1,720 = 0.00%; missed focus not caught 210/2,164 = 9.70%; verdict flips from brightness or colour alone 173/9,327 = 1.85%. The whole-frame rule alone sends 0 sharp photos to a blur group. Motion and most whole-frame-soft training examples are software-blurred by the owner's choice, so other kinds of motion blur are not covered. The next one-run final exam on a fresh event is pending.

---

## Infrastructure

| Concern | Implementation |
|---|---|
| Async batch | Celery + Redis broker. Image bytes are written to `BLOB_STORE_PATH` and workers get file paths (not base64). `.../batch` → 202 + `job_id`; `.../mega` fans out via Celery chord and merges via `finalize_mega_batch`. Staged images stay `BLOB_RETENTION_SECONDS` (24 h) after the job ends, then `cleanup_stale_blobs` deletes them. |
| DB | PostgreSQL 16 + pgvector (jobs, webhooks, API keys; the retired `persons` / `face_embeddings` tables remain). Async engine (asyncpg) for the API; sync engine (psycopg2) for Celery workers. |
| C++ acceleration | Nothing in the served code imports the optional `_quickpitik_cpp` module. Classical blur uses OpenCV/NumPy in every build so its score does not depend on whether the extension is installed. |
| Auth + rate limits | `X-API-Key` header (SHA-256 hashed, cached in Redis). Token-bucket rate limiter wired into every endpoint via `verify_api_key`. Tiers: free=60/min, pro=300/min, internal=1000/min. Per-key concurrent-job cap (`MAX_ACTIVE_JOBS_PER_KEY=10`). |
| Observability | Structured logs carry request IDs and stream completion counts; Prometheus HTTP and blur counters, timeouts, readiness, model version, and Linux process memory are at `/metrics`. Non-debug scrapes require explicit `metrics:read`; the dedicated blur profile uses one API process. See `docs/deployment.md`. |

## Model Artifacts

```
ai-api/models/
  manifest.json                                  model registry metadata
  blur_classifier/
    people_rule.json                             the stage-2 rule (plain numbers)
  person_detector/
    yolov8n_seg_person.onnx                      COCO person detector, 13.8 MB
```

Retraining scripts live in `ai-api/scripts/`: for the blur classifier `build_blur_folds.py` → `fit_blur_whole_frame_rule.py` → `export_person_detector.py` → `measure_blur_people.py` → `fit_blur_people_rule.py` → `ship_blur_rule.py`, in that order (they need `pip install -e ".[train]"`). Training data is under `ai-api/Training-Images/` (gitignored).

## Source Layout (ML only)

```
src/ml/
  registry.py                  loads the 2 blur models at startup
  blur/
    detector.py                Laplacian/FFT sharp-vs-blur check
    classifier.py              two-stage blur classifier
    whole_frame.py             whole-frame rule
    person_detector.py         letterbox + NMS around the ONNX detector
    people.py                  per-person sharpness, features, logistic rule
```

See `architecture.md` for the full 4-layer design, `api-reference.md` for endpoint contracts, and `integration-contracts.md` for the desktop contract.
