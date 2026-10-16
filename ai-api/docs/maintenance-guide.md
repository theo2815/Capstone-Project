# QuickPitik AI-API Maintenance Guide

**Last updated:** 2026-04-23
**For:** Agents and operators managing the QuickPitik AI-API in production

---

## 1. System Overview

QuickPitik AI-API is a FastAPI-based computer vision service for marathon/running event photography, called by the desktop app. It provides one ML pipeline (face search and bib OCR were removed on 2026-10-16; the backend uses AWS Rekognition for both):

| Pipeline | Purpose | Models Used | Endpoint Prefix |
|----------|---------|-------------|-----------------|
| **Blur Detection** | Filter blurry photos | Laplacian/FFT detector + two-stage blur classifier (whole-frame rule, then person detector + per-person sharpness + logistic rule); four classes reported | `/api/v1/blur/` |

### Architecture

```
Client -> FastAPI (uvicorn) -> ML Inference (sync, via asyncio.to_thread)
                            -> PostgreSQL 16 + pgvector (jobs, webhooks, API keys)
                            -> Redis (rate limiting, API key cache)
                            -> Celery Worker (batch jobs, webhooks)
```

### Key Files

| File | Role |
|------|------|
| `src/main.py` | App factory, middleware stack, lifespan |
| `src/config.py` | All settings (`Settings` class, `.env` driven) |
| `src/ml/registry.py` | Model loading/unloading at startup |
| `src/workers/celery_app.py` | Celery config, task routing, beat schedule |
| `src/workers/model_loader.py` | Worker-side model loading on fork |
| `src/db/session.py` | Async DB engine + session management |
| `src/db/sync_session.py` | Sync DB engine for Celery workers |

---

## 2. Routine Maintenance Tasks

### Daily

| Task | How | What to Check |
|------|-----|---------------|
| Check health endpoint | `GET /api/v1/health` | Returns `{"status": "alive"}` with HTTP 200 |
| Check readiness | `GET /api/v1/health/ready` (requires API key) | `models_loaded`, `database`, `redis` all true; HTTP 200 |
| Review error logs | Check structured logs for `level=error` | Look for repeated errors, model failures, DB timeouts |
| Check stale jobs | Query `jobs` table for `status='processing'` older than 1 hour | `reap_stale_jobs` runs every 5 min via Celery Beat, but verify |

### Weekly

| Task | How |
|------|-----|
| Review rate limit hits | Check Redis keys `ratelimit:*` or logs for 429 responses |
| Check disk usage | Model files in `./models/` (~2GB total); DB size; Redis memory |
| Review webhook delivery failures | Check `deliver_webhook` task failures in Celery logs |
| Verify job cleanup | `cleanup_old_jobs` runs daily; confirm jobs older than `JOB_RETENTION_DAYS` (7) are deleted |

### Monthly

| Task | How |
|------|-----|
| Rotate API keys | Deactivate unused keys in `api_keys` table, call `invalidate_api_key_cache()` |
| Review WEBHOOK_SECRET_KEY | Rotate Fernet key if needed; re-encrypt existing secrets |
| Check dependency CVEs | Run `pip audit` or `safety check` against requirements |
| Review connection pool metrics | Check for pool exhaustion: async pool_size=20, sync pool_size=15 |

---

## 3. Monitoring

### Logs

Structured JSON logging via `structlog`. Key fields: `event`, `level`, `timestamp`, `request_id`.

**Log locations:**
- FastAPI: stdout (uvicorn)
- Celery worker: stdout (celery worker process)

**Critical log patterns to watch:**

| Pattern | Meaning | Action |
|---------|---------|--------|
| `"Database health check failed"` | PostgreSQL unreachable | Check DB connection, restart if needed |
| `"Inference timed out after Xs"` | ML model hung on an image | Check if recurring; may indicate bad model state |
| `"Worker: failed to load"` | Model didn't load on worker start | Check model files exist, disk space, memory |
| `"Redis not available"` | Redis down; authenticated requests return 503 | Restore Redis; the pooled client reconnects |
| `"Database engine was created in PID"` | Fork-safety violation | Ensure `--preload` is disabled for uvicorn/gunicorn |

### Metrics

Prometheus metrics at `/metrics` (requires API key in production).

**Key metrics:**

| Metric | What It Tells You |
|--------|-------------------|
| `http_requests_total` | Request volume by endpoint and status code |
| `http_request_duration_seconds` | Latency percentiles (p50, p95, p99) |
| `http_requests_in_progress` | Current concurrent requests |

**Alert thresholds (suggested):**

| Condition | Severity | Action |
|-----------|----------|--------|
| `/health` returns non-200 for > 30s | Critical | Restart service |
| `/health/ready` returns 503 | High | Check which component is down |
| p99 latency > 10s on `/blur/classify` | Medium | Check CPU load, model state |
| Error rate > 5% on any endpoint | Medium | Check logs for root cause |
| Redis memory > 80% of limit | Medium | Check for key leaks, adjust maxmemory |

### Celery Monitoring

```bash
# Check active workers
celery -A src.workers.celery_app inspect active

# Check queues
celery -A src.workers.celery_app inspect active_queues

# Check reserved (prefetched) tasks
celery -A src.workers.celery_app inspect reserved
```

---

## 4. Performance Checks and Scaling

### Performance Baselines

| Endpoint | Expected Latency | Notes |
|----------|------------------|-------|
| `POST /blur/detect` | 50-200ms | Laplacian + FFT, CPU only |
| `POST /blur/classify` | 100-300ms | Whole-frame rule, then ONNX person detector + per-person sharpness + logistic rule |

### Scaling Levers

| Bottleneck | Solution |
|------------|----------|
| **API throughput** | Increase `WORKERS` (uvicorn workers). Each worker holds all models in memory (~2GB). |
| **Batch processing** | Scale Celery workers. Dedicate workers per queue with `WORKER_QUEUES`: `WORKER_QUEUES=blur celery -A src.workers.celery_app worker -Q blur -c 4`. Default-queue workers handle webhooks + maintenance and do not need ML models loaded. |
| **Memory pressure** | Reduce `MAX_INFERENCE_DIMENSION` (default 1280). Lower = less memory per image, possibly lower accuracy. The blur classifier decodes to its own `BLUR_CLASSIFY_DECODE_DIM`, calibrated to the desktop's 1280-px upload. |
| **DB connections** | Async pool and sync pool sizes are configurable in `src/db/session.py` / `src/db/sync_session.py`. Increase if pool exhaustion is logged. |
| **Throughput on desktop** | Use `/blur/detect/stream` or `/blur/classify/stream` for synchronous NDJSON batches of up to 500 — no polling overhead. |
| **Large uploads** | Use `/.../mega` (up to 500) — server chunks into `MAX_BATCH_SIZE` sub-tasks and merges via Celery chord. |

### Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| API server | 2 CPU, 4GB RAM | 4 CPU, 8GB RAM |
| Celery worker | 2 CPU, 4GB RAM | 4 CPU, 8GB RAM |
| PostgreSQL | 1 CPU, 1GB RAM | 2 CPU, 4GB RAM |
| Redis | 0.5 CPU, 512MB RAM | 1 CPU, 1GB RAM |

---

## 5. Security Checks and Abuse Prevention

### Security Checklist

| Check | How to Verify |
|-------|---------------|
| `DEBUG=false` in production | Check `.env`; app refuses to start if `DEBUG=true` unless `ENVIRONMENT` is set to `development`, `local` or `test` |
| API docs disabled | `/docs` and `/redoc` return 404 when `DEBUG=false` |
| API key required on all endpoints | Test without `X-API-Key` header; expect 401 |
| Rate limiting active | Check Redis is up; test burst requests; expect 429 |
| WEBHOOK_SECRET_KEY set | Check `.env`; if unset, webhook secrets stored as plaintext (logged as warning on startup) |
| CORS origins restricted | Check `ALLOWED_ORIGINS` in `.env`; should NOT be `*` in production |
| Security headers present | Check response headers: `X-Content-Type-Options`, `X-Frame-Options`, HSTS |
| SSRF protection on webhooks | Webhook delivery blocks private/internal IPs: `10/8`, `172.16/12`, `192.168/16`, `127/8`, `169.254/16`, `0.0.0.0/8`, plus IPv6 `::1`, `fc00::/7`, `fe80::/10` (see `_BLOCKED_NETWORKS` in `src/workers/tasks/webhook_tasks.py`) |

### Abuse Vectors and Mitigations

| Vector | Mitigation | Config |
|--------|------------|--------|
| **Image bombs** | Pillow decompression bomb limit (tied to `MAX_IMAGE_DIMENSION`²), max file size, max dimension 12000px enforced on both the single-image and batch validators, downscale before inference | `MAX_FILE_SIZE`, `MAX_IMAGE_DIMENSION`, `MAX_INFERENCE_DIMENSION` |
| **Batch flooding** | Max batch size, max active jobs per key, rate limiting (429) | `MAX_BATCH_SIZE=50`, `MEGA_BATCH_MAX_SIZE=500`, `MAX_ACTIVE_JOBS_PER_KEY=10` |
| **Inference DoS** | Per-image inference timeout, 60s request timeout, Celery `task_time_limit=3600s` (soft 3300s) | `INFERENCE_TIMEOUT`, `TimeoutMiddleware` |
| **Webhook SSRF** | IP-literal rejection at registration, DNS resolve + blocked-IP check at delivery (TOCTOU-safe — request uses resolved IP) | Built-in |
| **API key brute force** | Rate limiting (token bucket), SHA-256 hashed storage, cache invalidation on revoke | Rate tiers: `free=60/min`, `pro=300/min`, `internal=1000/min` |
| **Celery queue pressure** | `worker_max_tasks_per_child=500`, `worker_max_memory_per_child=2GB`, prefetch 4 (safe: paths only), `reap_stale_jobs` every 5 min | Celery config |

### API Key Management

```sql
-- Create a new API key (do this, then give the raw key to the client)
-- The key_hash is SHA-256 of the raw key
INSERT INTO api_keys (id, key_hash, name, scopes, rate_tier, active)
VALUES (gen_random_uuid(), encode(sha256('raw-key-here'::bytea), 'hex'),
        'Client Name', '["*"]', 'free', true);

-- Deactivate a key
UPDATE api_keys SET active = false WHERE id = 'key-uuid';

-- After deactivating, invalidate the Redis cache (5-min window otherwise):
-- Call invalidate_api_key_cache(redis, 'raw-key-string') from Python
```

---

## 6. Model Lifecycle

### Model Inventory

| Model | Path | Format | Size | Required |
|-------|------|--------|------|----------|
| Blur person detector (stage 2) | `models/person_detector/yolov8n_seg_person.onnx` | ONNX | 13.8MB | No (optional; required in the blur profile) |
| Blur rule (stage 2) | `models/blur_classifier/people_rule.json` | JSON | ~18KB | With the person detector |

### Updating a Model

1. **Stop accepting new requests** (optional: set maintenance mode or drain)
2. **Place new model file** in the correct path under `models/`
3. **Restart the service** -- models are loaded at startup (`ModelRegistry.load_all()`) and at worker fork (`load_models_on_worker_start()`)
4. **Verify** via `/api/v1/health/ready` and test requests
5. **Keep the old model file** as a rollback copy

### Retraining the Blur Classifier

Run from `ai-api/`, in this order:

1. `scripts/build_blur_folds.py` -- lays the labelled event photos out as five event folds
2. `scripts/fit_blur_whole_frame_rule.py` -- learns the whole-frame thresholds; copy the printed values into `src/config.py`
3. `scripts/export_person_detector.py` -- exports the person detector to `models/person_detector/yolov8n_seg_person.onnx` and records its hash (only when the detector itself is replaced)
4. `scripts/measure_blur_people.py` -- measures the people in every sharp and missed-focus fold photo once and caches the numbers
5. `scripts/fit_blur_people_rule.py` -- fits the rule and scores it on the folds it never saw
6. `scripts/ship_blur_rule.py` -- installs the rule as `models/blur_classifier/people_rule.json` and records its hash in `models/manifest.json`; then restart

At load time the classifier checks both files against the hashes in the manifest -- a mismatch disables it (the blur profile refuses to start).

### Rollback

1. Replace the model file with the previous version
2. Restart the service
3. No database migration needed -- models are stateless

---

## 7. Data Handling and Storage

### Database Tables

| Table | Purpose | Growth Pattern | Cleanup |
|-------|---------|----------------|---------|
| `persons` | Retired: runner profiles from the removed face routes | No longer grows (nothing writes it since 2026-10-16) | Manual SQL only (see GDPR / Data Erasure) |
| `face_embeddings` | Retired: 512-dim vectors (pgvector) | No longer grows | Cascade-deleted with person |
| `jobs` | Batch job tracking | Grows with batch requests | Auto-cleaned after `JOB_RETENTION_DAYS` (7) by Celery Beat |
| `api_keys` | API key hashes + metadata | Slow growth | Manual management |
| `webhook_subscriptions` | Webhook registrations | Slow growth | Manual delete via API |

### Image Storage

Images are **NOT persistently stored on disk or in the database**. Two flows:

**Single / stream endpoints (sync):**
1. Client uploads image via multipart/form-data
2. Image is decoded to a numpy array in memory (+ EXIF rotation, downscale)
3. ML inference runs on the array
4. Nothing is stored; the blur result goes back in the response
5. Raw bytes are discarded when the request returns

**Batch / mega endpoints (async):**
1. Client POSTs multiple files
2. The API atomically writes each image to the **blob store** at `{BLOB_STORE_PATH}/{job_id}/NNNNN.bin` (default `/tmp/quickpitik-blobs`, shared volume in Docker)
3. The Celery task receives file paths, not base64 payloads
4. Worker decodes from paths, runs inference, updates job progress
5. `complete_job` / `fail_job` records the result but **keeps** the staged images, as a retry window
6. `maintenance.cleanup_stale_blobs` (every 30 min) deletes a job's staged images once the job is `completed` or `failed` and finished at least `BLOB_RETENTION_SECONDS` ago (default 86400 = 24 h). A staged batch with no job row is deleted once its files are that old. Images of pending or processing jobs are kept.

No image content is ever logged or stored beyond the blob staging directory.

### Embedding Management (retired face data)

Nothing reads or writes these tables since the face routes were removed on 2026-10-16; the queries below inspect what is left.

```sql
-- Count embeddings per person
SELECT p.name, COUNT(fe.id) as emb_count
FROM persons p
LEFT JOIN face_embeddings fe ON p.id = fe.person_id
GROUP BY p.id ORDER BY emb_count DESC;

-- Find persons with no embeddings (cleanup candidates)
SELECT p.id, p.name FROM persons p
LEFT JOIN face_embeddings fe ON p.id = fe.person_id
WHERE fe.id IS NULL;

-- Database size check
SELECT pg_size_pretty(pg_database_size('quickpitik'));

-- Embeddings table size
SELECT pg_size_pretty(pg_total_relation_size('face_embeddings'));
```

### GDPR / Data Erasure

The erasure endpoint was removed with the face routes on 2026-10-16. Face data stored before then can be erased only in the database:

```sql
DELETE FROM persons WHERE id = '<person-uuid>';  -- face_embeddings rows go with it (ON DELETE CASCADE)
```

---

## 8. Deployment and Update Process

### Initial Deployment

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env: set DATABASE_URL, REDIS_URL, WEBHOOK_SECRET_KEY, ALLOWED_ORIGINS

# 2. Start infrastructure
docker compose up -d db redis

# 3. Run database migrations
alembic upgrade head

# 4. Place model files in ./models/ directory

# 5. Start the API and worker
docker compose up -d ai-api celery-worker
```

### Updating the Application

```bash
# 1. Pull latest code
git pull origin main

# 2. Check for new migrations
alembic history --verbose  # compare with current: alembic current

# 3. Run migrations (if any)
alembic upgrade head

# 4. Rebuild and restart
docker compose build ai-api celery-worker
docker compose up -d ai-api celery-worker

# 5. Verify
curl http://localhost:8000/api/v1/health
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/api/v1/health/ready
```

### Rolling Back

```bash
# 1. Revert code
git checkout <previous-commit>

# 2. Downgrade migrations (if the update added any)
alembic downgrade -1   # one step back

# 3. Rebuild and restart
docker compose build ai-api celery-worker
docker compose up -d ai-api celery-worker
```

### Environment Variables Reference (Critical)

| Variable | Default | Must Set in Production |
|----------|---------|----------------------|
| `DEBUG` | `false` | Verify `false` |
| `ENVIRONMENT` | `development` | Set to `production` |
| `DATABASE_URL` | localhost | Set to production DB |
| `REDIS_URL` | localhost | Set to production Redis |
| `WEBHOOK_SECRET_KEY` | (empty) | Generate: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `ALLOWED_ORIGINS` | localhost:3000 | Set to actual frontend origin(s) |
| `WORKERS` | 2 | Scale based on CPU cores |

---

## 9. Failure Handling and Recovery

### Common Failures

| Symptom | Likely Cause | Recovery |
|---------|-------------|----------|
| `/health` returns non-200 | Process crashed | Check logs, restart: `docker compose restart ai-api` |
| `/health/ready` shows `database: false` | PostgreSQL down or unreachable | Check DB status, connection string, restart DB |
| `/health/ready` shows `redis: false` | Redis unavailable or required commands failing | Restore Redis; readiness and authenticated access recover without an API restart |
| `/health/ready` shows `models_loaded: false` | Model files missing or corrupt | Check `./models/` directory, re-download models, restart |
| Jobs stuck in `processing` | Worker crashed mid-task | `reap_stale_jobs` auto-marks them failed after ~65 min. Or manually: `UPDATE jobs SET status='failed', error='manual reset' WHERE status='processing' AND created_at < NOW() - INTERVAL '1 hour'` |
| 503 on ML endpoints | Model not loaded | Check startup logs for model loading errors |
| 429 Too Many Requests | Rate limit hit | Normal behavior. Client should respect `Retry-After` header |
| 504 Gateway Timeout | Request exceeded 60s | Check if image is unusually large; check model performance |

### PostgreSQL Recovery

```bash
# Check if DB is accepting connections
docker compose exec db pg_isready

# Check active connections (pool exhaustion)
docker compose exec db psql -U postgres -d quickpitik -c "SELECT count(*) FROM pg_stat_activity WHERE datname='quickpitik';"

# Kill idle connections if pool is exhausted
docker compose exec db psql -U postgres -d quickpitik -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='quickpitik' AND state='idle' AND state_change < NOW() - INTERVAL '10 minutes';"
```

### Celery Worker Recovery

```bash
# Check if workers are alive
celery -A src.workers.celery_app inspect ping

# Restart workers (graceful -- finishes current tasks)
docker compose restart celery-worker

# Force restart (kills current tasks -- use only if stuck)
docker compose kill celery-worker && docker compose up -d celery-worker

# Purge all pending tasks (nuclear option -- loses queued work)
celery -A src.workers.celery_app purge
```

### Full System Recovery

```bash
# 1. Stop everything
docker compose down

# 2. Verify infrastructure
docker compose up -d db redis
# Wait for health checks to pass

# 3. Run any pending migrations
alembic upgrade head

# 4. Start application
docker compose up -d ai-api celery-worker

# 5. Verify all components
curl http://localhost:8000/api/v1/health
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/api/v1/health/ready
```

---

## 10. Testing and Validation

### Before Deploying Updates

```bash
# 1. Lint
ruff check src/ tests/

# 2. Type check
mypy src/

# 3. Unit tests
pytest tests/unit/ -v --tb=short

# 4. Syntax check all Python files
python -m py_compile src/main.py
# (or batch: find src -name "*.py" -exec python -m py_compile {} \;)
```

### After Deploying Updates

Run these checks against the live service:

```bash
API_KEY="your-api-key"
BASE="http://localhost:8000/api/v1"

# 1. Health checks
curl -s $BASE/health | python -m json.tool
curl -s -H "X-API-Key: $API_KEY" $BASE/health/ready | python -m json.tool

# 2. Blur detection (smoke test)
curl -s -X POST -H "X-API-Key: $API_KEY" \
  -F "file=@test_image.jpg" \
  $BASE/blur/detect | python -m json.tool

# 3. Check security headers
curl -s -I -H "X-API-Key: $API_KEY" $BASE/health/ready | grep -i "x-content-type-options\|x-frame-options\|strict-transport"

# 4. Check rate limiting
curl -s -I -H "X-API-Key: $API_KEY" $BASE/health | grep -i "x-ratelimit"
```

### Validation Criteria

| Check | Pass Condition |
|-------|---------------|
| Health liveness | HTTP 200, `status: "alive"` |
| Health readiness | HTTP 200, all three checks true |
| Blur detect | HTTP 200, `success: true`, `is_blurry` field present |
| Security headers | `x-content-type-options: nosniff` present |
| Rate limit headers | `X-RateLimit-Remaining` present |
| Error on no API key | HTTP 401 when `X-API-Key` header omitted |

---

## Quick Reference

### Ports

| Service | Port |
|---------|------|
| FastAPI (uvicorn) | 8000 |
| PostgreSQL | 5432 |
| Redis | 6379 |

### Task Queues

| Queue | Tasks | Workers load |
|-------|-------|--------------|
| `default` | `webhooks.deliver`, `maintenance.*`, `finalize_mega_batch` | No ML models needed |
| `blur` | `blur.detect_batch`, `blur.classify_batch` | BlurDetector + BlurClassifier |

### Celery Beat Schedule

| Task | Interval | Purpose |
|------|----------|---------|
| `maintenance.reap_stale_jobs` | 5 minutes | Mark stuck pending/processing jobs as failed (age > 65 min) |
| `maintenance.cleanup_old_jobs` | 24 hours | Delete completed/failed jobs older than `JOB_RETENTION_DAYS` (7) |
| `maintenance.cleanup_stale_blobs` | 30 minutes | Delete a finished job's staged images `BLOB_RETENTION_SECONDS` (24 h) after it ended; orphan batches after the same age |
| `maintenance.finalize_mega_batch` | on demand | Chord callback that merges mega-batch sub-task results into the parent job |

### Rate Limit Tiers

| Tier | Requests/min | Burst |
|------|-------------|-------|
| `free` | 60 | 60 |
| `pro` | 300 | 300 |
| `internal` | 1000 | 1000 |
