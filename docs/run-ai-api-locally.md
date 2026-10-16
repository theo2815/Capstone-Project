# Running ai-api locally — blur

Operational runbook for standing up ai-api on your machine and checking blur detection, the feature the BatchMyPhotos desktop app uses.

ai-api lost its face and bib features on 2026-10-16: the Spring Boot backend does face search and bib reading through AWS Rekognition (`AI_PROVIDER=rekognition`), so the website and the backend do not need ai-api running. To run the backend, follow `backend/AGENTS.md`.

First written **2026-05-20** on Windows 11 + Python 3.12 + Docker Desktop. See `ai-api/CLAUDE.md` for architecture rules and `docs/desktop-blur-detection-integration-guide.md` for desktop-side blur specifics.

---

## Architecture at a glance

```
Desktop app (or curl)  ─►  ai-api  ─►  Postgres + Redis
                           (port 8000)  (5433 + 6379)
```

| Layer | What runs where |
|---|---|
| Postgres (pgvector image) + Redis | Docker containers (`ai-api-db-1`, `ai-api-redis-1`) |
| ai-api FastAPI | Host Python: `uvicorn` on `:8000` |
| ai-api Celery worker | Host Python: `celery` (only for the batch endpoints) |

We run ai-api in **hybrid mode** (Docker for db + redis, host for FastAPI + worker) so code changes take effect without an image rebuild.

---

## One-time setup

Do these once per machine; skip on subsequent runs.

### 1. Python deps

Already installed if you've been developing ai-api. Verify with:

```powershell
python -c "import fastapi, onnxruntime, cv2; print('ok')"
```

If anything is missing: `cd ai-api && pip install -e ".[dev]"`.

### 2. Docker Desktop

Install + start. The whale icon must say **Engine running** before you proceed.

### 3. Backend Postgres port collision

The backend's `postgres:16-alpine` holds host port **5432**. ai-api's pgvector runs on host port **5433** (mapping is already checked in at `ai-api/docker-compose.yml`). Container-internal hostname stays `db:5432`. The `.env` at `ai-api/.env` points `DATABASE_URL` at `127.0.0.1:5433`.

### 4. Seed the ai-api dev API key

```powershell
cd ai-api
docker compose up db redis -d
# wait ~10s for pgvector to be healthy
alembic upgrade head
python scripts/seed_db.py
```

Output should include:

```
Created dev API key: sk_dev_quickpitik_test_key_12345
```

That key is for local testing only. It has `["*"]` scope. **Don't lose it** — if Docker volume gets wiped you'll need to re-run `seed_db.py`.

### 5. Models on disk

The blur classifier needs two files under `ai-api/models/` (both listed in `ai-api/models/manifest.json`):

- `models/person_detector/yolov8n_seg_person.onnx` — person detector
- `models/blur_classifier/people_rule.json` — the stage-2 rule

Already present if you've trained / downloaded them. If missing, see `ai-api/scripts/download_models.py` or the blur training scripts listed in `ai-api/AGENTS.md`.

---

## Daily startup sequence

Open **three PowerShell terminals** (two if you skip the Celery worker). Run each command in its own terminal so logs are separated and you can Ctrl+C any one without killing the others.

### Terminal 1 — Postgres + Redis (Docker)

```powershell
cd "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api"
docker compose up db redis -d
```

Verify both are healthy:

```powershell
docker ps --filter "name=ai-api" --format "table {{.Names}}`t{{.Status}}"
```

You should see both `ai-api-db-1` and `ai-api-redis-1` as `Up ... (healthy)`.

### Terminal 2 — ai-api FastAPI server

```powershell
cd "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api"
uvicorn src.main:create_app --factory --host 127.0.0.1 --port 8000
```

`127.0.0.1` keeps the API on this computer. To test from a phone or a second device, use `--host 0.0.0.0` for that session, on a trusted network.

If `ai-api/.env` sets `DEBUG=true`, it must also set `ENVIRONMENT=development`, or startup refuses. It must not contain the removed `FACE_SIMILARITY_THRESHOLD`, `FACE_DET_SIZE`, `FACE_MIN_ENROLLMENT_CONFIDENCE`, `BIB_MIN_CHARS` or `OCR_MAX_WORKERS`, or startup refuses too.

First boot loads the two blur models — wait for these lines:

```
Model load finished            model=blur available=True
Model load finished            model=blur_classifier available=True
ML models loaded               all_loaded=True
INFO:     Application startup complete.
```

### Terminal 3 — ai-api Celery worker

```powershell
cd "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api"
celery -A src.workers.celery_app worker --loglevel=info --pool=solo
```

The `--pool=solo` flag is required on Windows — the default prefork pool doesn't work on win32.

You only need the worker for the batch endpoints (`/blur/detect/batch`, `/blur/classify/mega`, etc.). The single and stream endpoints the desktop uses work without it.

---

## Smoke tests

Run these from any terminal once everything is up. Confirms the AI stack is alive without needing to touch the FE.

### Health check (all models loaded)

```powershell
curl.exe -H "X-API-Key: sk_dev_quickpitik_test_key_12345" http://localhost:8000/api/v1/health/ready
```

Expected: `{"success":true,..."data":{"models_loaded":true,"database":true,"redis":true},...}`.

### Blur detection

```powershell
$IMG = "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api\Training-Images\Sharp_images\103_HUAWEI-P20_S.jpg"
curl.exe -X POST -H "X-API-Key: sk_dev_quickpitik_test_key_12345" -F "file=@$IMG" http://localhost:8000/api/v1/blur/detect
```

Expected: `is_blurry: false, confidence: 1.0` on a sharp image.

---

## Stopping cleanly

Ctrl+C in each PowerShell terminal **in this order**:

1. ai-api Celery worker
2. ai-api uvicorn
3. Docker containers: `docker compose down` from `ai-api/` directory

`docker compose down` keeps the data volume (your API keys and jobs). Use `docker compose down -v` only if you want to wipe everything and re-seed.

---

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `connection refused` on `localhost:8000` | uvicorn didn't start / crashed | Check Terminal 2 output |
| 401 on every request | Wrong key, or `seed_db.py` never ran | Re-run `python scripts/seed_db.py`; the key is `sk_dev_quickpitik_test_key_12345` |
| Health endpoint 401 | ai-api auth-gates `/health/ready` when `DEBUG=false` | Pass `-H "X-API-Key: sk_dev_quickpitik_test_key_12345"` |

---

## What changed 2026-05-20 (for future-me)

Six bugs were fixed in one session to get this working. The face and bib files below were removed on 2026-10-16; the list is kept as history:

- `ai-api/src/ml/bibs/recognizer.py` — PaddleOCR 2.x compat shim
- `ai-api/src/db/repositories/face_repo.py` — uses `CAST(:query AS vector)`
- `ai-api/src/config.py` — `MAX_INFERENCE_DIMENSION: int = 1280`
- `ai-api/src/ml/bibs/detector.py` — default confidence `0.15`
- `ai-api/src/api/v1/bibs.py` — has the full-image OCR fallback after the YOLO branch
- `backend/.../dto/ai/AiApiResults.kt` — `bbox: FaceBBox?` (not `List<Double>`)
- `backend/.../service/photographer/PhotoUploadService.kt` — passes `personId = null` to `facesEnroll`
- `backend/.../repository/PhotoRepository.kt` — bib filter uses `LIKE CONCAT('%', UPPER(:bib), '%')`

Full context: vault `_journal/2026-05-20-ai-api-integration-end-to-end.md`.
