# Running ai-api locally — blur, face, bib

Operational runbook for standing up ai-api end-to-end and verifying blur detection, face search, and bib OCR work against the Spring Boot backend + website.

Confirmed working **2026-05-20** on Windows 11 + Python 3.12 + Docker Desktop. See `ai-api/CLAUDE.md` for architecture rules and `docs/desktop-blur-detection-integration-guide.md` for desktop-side blur specifics.

---

## Architecture at a glance

```
Website FE  ─►  Spring Boot backend  ─►  ai-api  ─►  pgvector + Redis
(port 3000)     (port 8080)              (port 8000)  (5433 + 6379)
```

Three running processes need to be alive together:

| Layer | What runs where |
|---|---|
| pgvector + Redis | Docker containers (`ai-api-db-1`, `ai-api-redis-1`) |
| ai-api FastAPI | Host Python: `uvicorn` on `:8000` |
| ai-api Celery worker | Host Python: `celery` |
| Spring Boot backend | Host Java: `gradlew.bat bootRun` on `:8080` |
| Website FE | Host Node: `pnpm dev` on `:3000` |

We run ai-api in **hybrid mode** (Docker for db + redis, host for FastAPI + worker) because the full Docker build pulls ~3 GB of ML deps and times out on slow connections. The host already has the wheels installed.

---

## One-time setup

Do these once per machine; skip on subsequent runs.

### 1. Python deps

Already installed if you've been developing ai-api. Verify with:

```powershell
python -c "import fastapi, insightface, paddleocr, ultralytics, onnxruntime; print('ok')"
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

That key is what the backend uses to call ai-api. It has `["*"]` scope. **Don't lose it** — if Docker volume gets wiped you'll need to re-run `seed_db.py`.

### 5. Models on disk

ai-api needs three model files under `ai-api/models/`:

- `models/models/buffalo_l/` — InsightFace bundle (auto-downloaded on first face call)
- `models/blur_classifier/blur_classifier.onnx` — YOLOv8n-cls 4-class blur classifier
- `models/bib_detection/yolov8n_bib.onnx` — custom YOLOv8n bib detector

Already present if you've trained / downloaded them. If missing, see `ai-api/scripts/download_models.py` or retrain via the training scripts in `ai-api/scripts/`.

---

## Daily startup sequence

Open **three PowerShell terminals**. Run each command in its own terminal so logs are separated and you can Ctrl+C any one without killing the others.

### Terminal 1 — pgvector + Redis (Docker)

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
uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000
```

First boot loads all five models — wait for these lines:

```
Model loaded                   model=blur
Model loaded                   model=blur_classifier
Model loaded                   model=face
Model loaded                   model=bib_detector
Model loaded                   model=bib_ocr
ML models loaded               all_loaded=True
INFO:     Application startup complete.
```

Takes ~10–20 seconds.

### Terminal 3 — ai-api Celery worker

```powershell
cd "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api"
celery -A src.workers.celery_app worker --loglevel=info --pool=solo
```

The `--pool=solo` flag is required on Windows — the default prefork pool doesn't work on win32.

You only need the worker if you're using batch endpoints (`/blur/detect/batch`, `/faces/search/batch`, etc.). The website's upload path uses synchronous calls so it works without the worker — but start it anyway so batch calls don't silently hang.

### Terminal 4 — Spring Boot backend (separate window)

```powershell
cd "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\backend"
$env:AI_API_ENABLED="true"; $env:AI_API_KEY="sk_dev_quickpitik_test_key_12345"; $env:AI_API_URL="http://localhost:8000"; $env:SPRING_PROFILES_ACTIVE="local"; .\gradlew.bat bootRun
```

What each var does:

| Variable | Purpose |
|---|---|
| `AI_API_ENABLED=true` | Flips the master switch in `AiApiProperties.enabled`. Without this, `PhotoUploadService` skips faces+bibs and `PhotoSearchService.searchByFace` 503s |
| `AI_API_KEY=sk_dev_quickpitik_test_key_12345` | The dev key from `seed_db.py`. Without this you'd get 401 from ai-api |
| `AI_API_URL=http://localhost:8000` | ai-api host. Also the default; explicit is safer |
| `SPRING_PROFILES_ACTIVE=local` | Loads `application-local.yml` with PayMongo test secret + Resend API key. Required for guest checkout + email receipts to work |

Wait for: `Started QuickPitikApplicationKt in ~6 seconds`.

### Terminal 5 — Website (separate window)

```powershell
cd "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\website"
pnpm dev
```

Open `http://localhost:3000` in a browser.

---

## Smoke tests

Run these from any terminal once everything is up. Confirms the AI stack is alive without needing to touch the FE.

### Health check (all models loaded)

```powershell
curl.exe -H "X-API-Key: sk_dev_quickpitik_test_key_12345" http://localhost:8000/api/v1/health/ready
```

Expected: `{"success":true,..."data":{"models_loaded":true,"database":true,"redis":true},...}`.

### Blur detection (desktop feature — not used by website but should work)

```powershell
$IMG = "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api\Training-Images\Sharp_images\103_HUAWEI-P20_S.jpg"
curl.exe -X POST -H "X-API-Key: sk_dev_quickpitik_test_key_12345" -F "file=@$IMG" http://localhost:8000/api/v1/blur/detect
```

Expected: `is_blurry: false, confidence: 1.0` on a sharp image.

### Face detection

```powershell
$IMG = "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api\Training-Images\face_bib_detection\images\train\IMG_0001.JPG"
curl.exe -X POST -H "X-API-Key: sk_dev_quickpitik_test_key_12345" -F "file=@$IMG" http://localhost:8000/api/v1/faces/detect
```

Expected: `faces_detected: N` where N > 0 if the photo contains faces.

### Bib recognition

```powershell
$IMG = "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api\Training-Images\bib_detection\images\train\IMG_0001_JPG.rf.179818135f5597574aa40ac1ea3928db.jpg"
curl.exe -X POST -H "X-API-Key: sk_dev_quickpitik_test_key_12345" -F "file=@$IMG" http://localhost:8000/api/v1/bibs/recognize
```

Expected: at least one `bib_number` with `confidence > 0.9`.

### Backend → ai-api proxy (face search)

Requires you to be logged in. Grab an access token via the auth flow, then:

```powershell
$TOKEN = "<paste access token>"
$IMG = "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api\Training-Images\face_bib_detection\images\train\IMG_0001.JPG"
curl.exe -X POST -H "Authorization: Bearer $TOKEN" -F "selfie=@$IMG" http://localhost:8080/api/v1/events/<event-slug>/photos/search-by-face
```

Expected: 200 + paginated photo list (empty if no enrolled faces yet for that event).

---

## End-to-end website flow

1. Log in as a **photographer** with completed verification.
2. Go to `/upload/{eventId}` and upload a photo with a visible face + bib.
3. Backend log should show no `Faces detect failed` / `Bibs recognize failed` warnings.
4. Response payload should carry `aiDetectionStatus: "ok"`.
5. Log out, log in as a **runner**, add a selfie at `/profile` (or use an existing one).
6. Go to that event, hit **Find photos** → choose your selfie. Matched photos appear.
7. Type a bib number (or partial — substring match) into the bib search field. Photos with that bib appear.

---

## Stopping cleanly

Ctrl+C in each PowerShell terminal **in this order**:

1. Website (`pnpm dev`)
2. Backend (`gradlew bootRun`)
3. ai-api Celery worker
4. ai-api uvicorn
5. Docker containers: `docker compose down` from `ai-api/` directory

`docker compose down` keeps the data volume (pgvector keeps your enrolled faces). Use `docker compose down -v` only if you want to wipe everything and re-seed.

---

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `Tomcat started on port 8080... Port 8080 was already in use` | A previous backend bootRun is still running, or another process holds 8080 | `Get-Process java` to find the old PID, then `Stop-Process -Id <pid>` |
| `connection refused` on `localhost:8000` from backend log | uvicorn didn't start / crashed | Check Terminal 2 output |
| `ai-api faces/search failed/offline ... Falling back to demo list` in backend log | ai-api returned 500 or is unreachable | Check uvicorn log for the traceback. Likely env config drift |
| Bib search returns nothing for a clearly visible bib | YOLO bib detector missed the crop, full-image OCR fallback also missed it | Check `photo_bibs` table directly. The detector misses side-angle and night bibs — known limitation |
| `Invalid person_id format` in backend log | Old build — `PhotoUploadService` sending non-UUID. Fixed 2026-05-20 | `./gradlew clean build` then restart |
| 401 on every ai-api call | `AI_API_KEY` wrong or `seed_db.py` never ran | Verify `python ai-api/scripts/seed_db.py` output. Key must match `sk_dev_quickpitik_test_key_12345` |
| Health endpoint 401 | ai-api auth-gates `/health/ready` when `DEBUG=false` | Pass `-H "X-API-Key: sk_dev_quickpitik_test_key_12345"` |
| `paddleocr` import errors in fresh Python shell | Windows torch DLL issue in `albumentations` import chain — does NOT affect uvicorn | Ignore for diagnostic scripts; the runtime works |

---

## When you upload photos and bibs are missing

The bib OCR pipeline is two-stage: YOLO finds a region → PaddleOCR reads the text. If YOLO returns 0 regions, full-image OCR runs as fallback. Either stage can miss.

To backfill bibs for photos that uploaded successfully but have no `photo_bibs` row:

```powershell
cd "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project"
python backend\scripts\backfill_bibs.py <event-id-1> <event-id-2>
```

Or with no event IDs to backfill all photos in the DB. The script is idempotent (`ON CONFLICT (photo_id, bib_number) DO NOTHING`).

---

## When you need to wipe a test event clean

```powershell
docker exec quickpitik-postgres psql -U quickpitik -d quickpitik -f /path/to/nuke.sql
```

Or copy/paste the inline SQL from `backend/scripts/nuke-test-event-photos.sql` with your event IDs swapped in. The script handles all dependent rows (orders, payments, downloads, disputes, transactions, photo_bibs, photo_face_persons) in dependency order.

Also clean ai-api face embeddings for those events:

```powershell
docker exec ai-api-db-1 psql -U postgres -d quickpitik -c "DELETE FROM persons WHERE event_id IN ('<event-id-1>','<event-id-2>');"
```

And the photo files on disk:

```powershell
Remove-Item -Recurse -Force "C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\backend\.storage\events\<event-id>"
```

---

## What changed 2026-05-20 (for future-me)

Six bugs were fixed in one session to get this working. The runbook above assumes those fixes are in place. If something behaves wildly differently from what's described, check:

- `ai-api/src/ml/bibs/recognizer.py` — PaddleOCR 2.x compat shim
- `ai-api/src/db/repositories/face_repo.py` — uses `CAST(:query AS vector)`
- `ai-api/src/config.py` — `MAX_INFERENCE_DIMENSION: int = 1280`
- `ai-api/src/ml/bibs/detector.py` — default confidence `0.15`
- `ai-api/src/api/v1/bibs.py` — has the full-image OCR fallback after the YOLO branch
- `backend/.../dto/ai/AiApiResults.kt` — `bbox: FaceBBox?` (not `List<Double>`)
- `backend/.../service/photographer/PhotoUploadService.kt` — passes `personId = null` to `facesEnroll`
- `backend/.../repository/PhotoRepository.kt` — bib filter uses `LIKE CONCAT('%', UPPER(:bib), '%')`

Full context: vault `_journal/2026-05-20-ai-api-integration-end-to-end.md`.
