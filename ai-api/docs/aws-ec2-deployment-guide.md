# Deploying EventAI API to AWS EC2

> Complete step-by-step guide for deploying the AI API (blur detection, face recognition, bib number OCR) to an AWS EC2 instance.

---

## Table of Contents

1. [Instance Selection](#1-instance-selection)
2. [Launch EC2 Instance](#2-launch-ec2-instance)
3. [Connect to Your Instance](#3-connect-to-your-instance)
4. [Set Up Server Environment](#4-set-up-server-environment)
5. [Transfer Project Files](#5-transfer-project-files)
6. [Configure Production Environment](#6-configure-production-environment)
7. [Deploy the Application](#7-deploy-the-application)
8. [Test the Deployment](#8-test-the-deployment)
9. [Secure the Deployment](#9-secure-the-deployment)
10. [Operations & Monitoring](#10-operations--monitoring)
11. [Quick Reference Card](#11-quick-reference-card)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Instance Selection

The app loads ML models (~300MB total) into memory. Each API worker uses ~1.5GB RAM.

| Use Case | Instance Type | vCPU | RAM | Approx. Cost |
|----------|--------------|------|-----|---------------|
| **Testing (start here)** | `t3.xlarge` | 4 | 16 GB | ~$0.17/hr |
| Staging | `t3.2xlarge` | 8 | 32 GB | ~$0.33/hr |
| Production (GPU) | `g4dn.xlarge` | 4 | 16 GB + T4 GPU | ~$0.53/hr |

**Recommendation**: Start with `t3.xlarge`. All features work on CPU — GPU just makes face/bib inference faster. You can upgrade later without redeploying.

### CPU vs GPU Performance

| Feature | CPU | GPU |
|---------|-----|-----|
| Blur detection | ~2ms | ~2ms (CPU only) |
| Face detection (RetinaFace) | ~80ms | ~8ms |
| Face embedding (ArcFace) | ~50ms | ~5ms |
| Bib detection (YOLOv8) | ~40ms | ~4ms |
| Bib OCR (PaddleOCR) | ~30ms | ~6ms |

---

## 2. Launch EC2 Instance

### Step 1 — Sign in to AWS

1. Go to https://aws.amazon.com and sign in (or create a free-tier account)
2. In the top-right corner, select a **region** close to your users
   - Philippines / Southeast Asia: `ap-southeast-1` (Singapore)
   - US: `us-east-1` (N. Virginia) or `us-west-2` (Oregon)

### Step 2 — Launch Instance

1. Go to **EC2 Dashboard** → click **Launch Instance**
2. Fill in the settings:

| Setting | Value |
|---------|-------|
| **Name** | `eventai-api` |
| **AMI** | Ubuntu Server 24.04 LTS (free tier eligible) |
| **Architecture** | 64-bit (x86) |
| **Instance type** | `t3.xlarge` |
| **Key pair** | Click **Create new key pair** → name it `eventai-key` → Type: RSA → Format: `.pem` → Download it → **keep this file safe** |
| **Storage** | Change root volume to **60 GB** gp3 SSD |

### Step 3 — Configure Security Group (Firewall)

Under **Network settings**, click **Edit** and configure these inbound rules:

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| SSH | 22 | **My IP** | Your SSH access |
| HTTP | 80 | 0.0.0.0/0 (Anywhere) | Web traffic via Nginx |
| HTTPS | 443 | 0.0.0.0/0 (Anywhere) | Secure web traffic |
| Custom TCP | 8000 | **My IP** | Direct API access (testing only, remove later) |

**Important**: Do NOT open ports 5432 (Postgres) or 6379 (Redis) to the internet. The production docker-compose binds these to `127.0.0.1` only.

### Step 4 — Launch

1. Click **Launch Instance**
2. Wait 1-2 minutes for it to start
3. Go to **Instances** → click your instance → note the **Public IPv4 address**

---

## 3. Connect to Your Instance

### From Git Bash (Windows)

```bash
# Move your key to the .ssh folder
mv ~/Downloads/eventai-key.pem ~/.ssh/eventai-key.pem

# Set correct permissions
chmod 400 ~/.ssh/eventai-key.pem

# Connect via SSH
ssh -i ~/.ssh/eventai-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### From PowerShell (Windows)

```powershell
# Move key
Move-Item ~\Downloads\eventai-key.pem ~\.ssh\eventai-key.pem

# Connect
ssh -i $HOME\.ssh\eventai-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### From PuTTY

1. Open **PuTTYgen** → Load → select `eventai-key.pem` → **Save private key** as `eventai-key.ppk`
2. Open **PuTTY** → Host Name: `ubuntu@YOUR_EC2_PUBLIC_IP`
3. Connection → SSH → Auth → Credentials → Private key file: select `eventai-key.ppk`
4. Click **Open**

> Replace `YOUR_EC2_PUBLIC_IP` with the actual IP from the AWS console throughout this guide.

---

## 4. Set Up Server Environment

Run all commands below **on the EC2 instance** after SSH-ing in.

### Step 1 — Update system packages

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2 — Install Docker and Docker Compose

```bash
# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add your user to the docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker $USER

# Install Docker Compose plugin
sudo apt install -y docker-compose-plugin

# Apply group changes
newgrp docker

# Verify installation
docker --version          # Should show Docker 27.x+
docker compose version    # Should show v2.x+
```

### Step 3 — Install helpful utilities

```bash
sudo apt install -y git htop tmux
```

### Step 4 — Add swap memory (safety net for ML workloads)

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h   # Should show ~4G swap
```

### Step 5 — Enable Docker on boot

```bash
sudo systemctl enable docker
```

---

## 5. Transfer Project Files

### Option A — Git Clone (recommended)

If your repository is hosted on GitHub or GitLab:

```bash
cd ~
git clone YOUR_REPO_URL Capstone-Project
cd Capstone-Project/ai-api
```

**Note**: If your `models/` directory is gitignored, you still need to transfer model files separately (see Option B for the SCP commands).

### Option B — SCP (direct file transfer)

Run these commands **from your local Windows machine** (Git Bash):

**Transfer entire project:**

```bash
scp -i ~/.ssh/eventai-key.pem -r \
  "/c/Users/Theo Cedric Chan/Documents/Start Up project/Capstone-Project/ai-api" \
  ubuntu@YOUR_EC2_PUBLIC_IP:~/ai-api
```

**If the upload is slow, compress first:**

```bash
# On your local machine
cd "/c/Users/Theo Cedric Chan/Documents/Start Up project/Capstone-Project"
tar -czf ai-api.tar.gz ai-api/

# Transfer the compressed archive
scp -i ~/.ssh/eventai-key.pem ai-api.tar.gz ubuntu@YOUR_EC2_PUBLIC_IP:~/

# On EC2 — decompress
cd ~
tar -xzf ai-api.tar.gz
rm ai-api.tar.gz
```

### What gets transferred

| File/Directory | Size | Purpose |
|----------------|------|---------|
| `models/blur_classifier/blur_classifier.onnx` | ~5.7 MB | Blur detection model |
| `models/blur_classifier/class_names.json` | <1 KB | Class labels |
| `models/bib_detection/yolov8n_bib.onnx` | ~12 MB | Bib number detector |
| `models/models/buffalo_l/` | ~289 MB | Face recognition (RetinaFace + ArcFace) |
| `src/` | ~500 KB | Application source code |
| `Dockerfile`, `docker-compose.prod.yml` | <5 KB | Deployment config |

---

## 6. Configure Production Environment

### Step 1 — Create the `.env` file

On the EC2 instance:

```bash
cd ~/ai-api

cat > .env << 'ENVFILE'
# === Application ===
APP_NAME=EventAI API
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
WORKERS=2

# === Database ===
DATABASE_URL=postgresql+asyncpg://postgres:CHANGE_THIS_DB_PASSWORD@db:5432/eventai
POSTGRES_PASSWORD=CHANGE_THIS_DB_PASSWORD

# === Redis ===
REDIS_URL=redis://:CHANGE_THIS_REDIS_PASSWORD@redis:6379/0
REDIS_PASSWORD=CHANGE_THIS_REDIS_PASSWORD

# === ML Models ===
MODEL_DIR=/app/models
USE_GPU=false
GPU_DEVICE=0

# === ML Thresholds ===
BLUR_THRESHOLD=100.0
BLUR_DETECTION_MIN_CONFIDENCE=0.5
FACE_SIMILARITY_THRESHOLD=0.4
FACE_DET_SIZE=640
FACE_MIN_ENROLLMENT_CONFIDENCE=0.7
BIB_MIN_CHARS=2

# === Auth ===
API_KEY_HEADER=X-API-Key

# === Rate Limiting ===
RATE_LIMIT_DEFAULT=60
RATE_LIMIT_BURST=10

# === CORS ===
# Change to your actual frontend domain(s) before going live
ALLOWED_ORIGINS=["*"]

# === Webhooks ===
WEBHOOK_TIMEOUT=10
WEBHOOK_MAX_RETRIES=3

# === File Upload ===
MAX_FILE_SIZE=10485760
MAX_BATCH_SIZE=20
ENVFILE
```

### Step 2 — Generate and set strong passwords

```bash
# Generate random passwords
DB_PASS=$(openssl rand -base64 24)
REDIS_PASS=$(openssl rand -base64 24)

# Display them (save these somewhere secure!)
echo "============================================"
echo "DB Password:    $DB_PASS"
echo "Redis Password: $REDIS_PASS"
echo "============================================"

# Replace placeholders in .env
sed -i "s/CHANGE_THIS_DB_PASSWORD/$DB_PASS/g" .env
sed -i "s/CHANGE_THIS_REDIS_PASSWORD/$REDIS_PASS/g" .env
```

**Save these passwords!** You'll need them if you ever connect to the database directly.

---

## 7. Deploy the Application

### Step 1 — Build and start all services

```bash
cd ~/ai-api

docker compose -f docker-compose.prod.yml up --build -d
```

What this starts:
- **ai-api**: FastAPI server (2 workers, production Dockerfile, non-root user)
- **celery-worker**: Background task processor (4 queues: default, blur, face, bib)
- **db**: PostgreSQL 16 + pgvector
- **redis**: Redis 7.4 with password auth
- **nginx**: Reverse proxy on port 80 with rate limiting
- **certbot**: SSL certificate management

**First build takes 5-10 minutes** (downloading images + installing ML packages).

### Step 2 — Monitor the startup

```bash
# Watch all service logs
docker compose -f docker-compose.prod.yml logs -f

# Or watch specific services
docker compose -f docker-compose.prod.yml logs -f ai-api
docker compose -f docker-compose.prod.yml logs -f celery-worker
```

**Wait for these log messages before proceeding:**

```
ai-api      | INFO:     Uvicorn running on http://0.0.0.0:8000
ai-api      | Model registry: all models loaded successfully
celery-worker | celery@... ready.
```

Press `Ctrl+C` to stop following logs (services keep running).

### Step 3 — Run database migrations

```bash
docker compose -f docker-compose.prod.yml exec ai-api \
  python -m alembic upgrade head
```

### Step 4 — Create a test API key

```bash
docker compose -f docker-compose.prod.yml exec ai-api \
  python scripts/seed_db.py
```

**Copy the API key** that gets printed — you need it for all authenticated requests.

### Step 5 — Verify everything is healthy

```bash
docker compose -f docker-compose.prod.yml ps
```

Expected output — all services should show `Up (healthy)`:

```
NAME              STATUS                  PORTS
ai-api            Up (healthy)            8000/tcp
celery-worker     Up
db                Up (healthy)            127.0.0.1:5432->5432/tcp
redis             Up (healthy)            127.0.0.1:6379->6379/tcp
nginx             Up                      0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
certbot           Up
```

---

## 8. Test the Deployment

### 8.1 — Health checks

**From EC2 (localhost):**

```bash
# Liveness — is the process alive?
curl http://localhost:8000/api/v1/health

# Readiness — are models loaded, DB and Redis reachable?
curl http://localhost:8000/api/v1/health/ready
```

**From your local machine (via Nginx on port 80):**

```bash
curl http://YOUR_EC2_PUBLIC_IP/api/v1/health
curl http://YOUR_EC2_PUBLIC_IP/api/v1/health/ready
```

Both should return:
```json
{"success": true, "data": {...}}
```

### 8.2 — Blur detection (single image)

```bash
curl -X POST http://YOUR_EC2_PUBLIC_IP/api/v1/blur/detect \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@/path/to/test-image.jpg"
```

Expected response:
```json
{
  "success": true,
  "data": {
    "is_blurry": false,
    "confidence": 0.92,
    "laplacian_score": 245.7
  }
}
```

### 8.3 — Blur detection (batch)

```bash
# Submit batch job
curl -X POST http://YOUR_EC2_PUBLIC_IP/api/v1/blur/batch \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

Response returns a `job_id`:
```json
{"success": true, "data": {"job_id": "abc-123-..."}}
```

Poll for results:
```bash
curl http://YOUR_EC2_PUBLIC_IP/api/v1/jobs/JOB_ID_HERE \
  -H "X-API-Key: YOUR_API_KEY"
```

### 8.4 — Face enrollment and search

```bash
# Enroll a face
curl -X POST http://YOUR_EC2_PUBLIC_IP/api/v1/face/enroll \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@person_photo.jpg" \
  -F "person_id=runner-001" \
  -F "person_name=John Doe"

# Search for that face in a marathon photo
curl -X POST http://YOUR_EC2_PUBLIC_IP/api/v1/face/search \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@marathon_crowd_photo.jpg"
```

### 8.5 — Bib number detection

```bash
curl -X POST http://YOUR_EC2_PUBLIC_IP/api/v1/bib/detect \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@runner_with_bib.jpg"
```

### 8.6 — Bib number search

```bash
curl -X POST http://YOUR_EC2_PUBLIC_IP/api/v1/bib/search \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@marathon_photo.jpg" \
  -F "bib_number=1234"
```

---

## 9. Secure the Deployment

### 9.1 — SSL with Let's Encrypt (requires a domain name)

**Prerequisite**: You need a domain (e.g., `api.eventai.com`). Point its DNS A record to `YOUR_EC2_PUBLIC_IP`.

```bash
# Wait for DNS to propagate (check with: dig api.eventai.com)

# Request SSL certificate
docker compose -f docker-compose.prod.yml run --rm certbot \
  certonly --webroot --webroot-path=/var/lib/letsencrypt \
  -d YOUR_DOMAIN --email YOUR_EMAIL --agree-tos --no-eff-email
```

Then edit `nginx.conf` on the EC2 instance:

```bash
nano ~/ai-api/nginx.conf
```

Make these changes:

1. **Uncomment** the entire HTTPS `server { ... }` block at the bottom
2. **Replace** `YOUR_DOMAIN` with your actual domain (2 places in the HTTPS block)
3. In the HTTP block, **uncomment** the redirect:
   ```nginx
   location / {
       return 301 https://$host$request_uri;
   }
   ```
4. **Comment out** or remove the temporary HTTP proxy `location /` block

Restart Nginx:

```bash
docker compose -f docker-compose.prod.yml restart nginx
```

Verify: `curl https://YOUR_DOMAIN/api/v1/health`

### 9.2 — Remove direct API port access

Once Nginx is confirmed working on port 80/443:

1. Go to AWS Console → EC2 → Security Groups
2. Find your instance's security group
3. **Delete** the port 8000 inbound rule

All traffic now flows through Nginx (with rate limiting and proper headers).

### 9.3 — Automatic security updates

```bash
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
# Select "Yes" when prompted
```

### 9.4 — Lock down CORS for production

Edit `.env` and replace the wildcard:

```bash
ALLOWED_ORIGINS=["https://your-frontend-domain.com"]
```

Restart the API:

```bash
docker compose -f docker-compose.prod.yml up -d ai-api
```

### 9.5 — SSH hardening (optional but recommended)

```bash
sudo nano /etc/ssh/sshd_config
```

Ensure these settings:

```
PermitRootLogin no
PasswordAuthentication no
MaxAuthTries 3
```

Restart SSH:

```bash
sudo systemctl restart sshd
```

---

## 10. Operations & Monitoring

### Common commands

```bash
# === Lifecycle ===
docker compose -f docker-compose.prod.yml up -d          # Start all
docker compose -f docker-compose.prod.yml down            # Stop all
docker compose -f docker-compose.prod.yml restart ai-api  # Restart one service
docker compose -f docker-compose.prod.yml up --build -d   # Rebuild + restart

# === Logs ===
docker compose -f docker-compose.prod.yml logs -f                 # All logs
docker compose -f docker-compose.prod.yml logs -f ai-api          # API logs
docker compose -f docker-compose.prod.yml logs -f celery-worker   # Worker logs
docker compose -f docker-compose.prod.yml logs --tail=100 ai-api  # Last 100 lines

# === Debugging ===
docker compose -f docker-compose.prod.yml exec ai-api bash        # Shell into container
docker compose -f docker-compose.prod.yml ps                      # Service status
docker stats                                                       # Resource usage

# === Database ===
docker compose -f docker-compose.prod.yml exec db psql -U postgres eventai  # DB shell
docker compose -f docker-compose.prod.yml exec ai-api python -m alembic upgrade head  # Migrations

# === Disk ===
df -h                    # Check disk space
docker system prune -f   # Clean unused Docker data (images, containers)
```

### Keep sessions alive with tmux

```bash
tmux new -s deploy        # Start named session
# ... run commands ...
# Ctrl+B then D           # Detach (services keep running)
tmux attach -t deploy     # Reattach later
```

### Deploying code updates

```bash
cd ~/ai-api

# Pull latest code
git pull origin main

# Rebuild and restart (zero-downtime isn't guaranteed — plan for brief interruption)
docker compose -f docker-compose.prod.yml up --build -d

# Run any new migrations
docker compose -f docker-compose.prod.yml exec ai-api python -m alembic upgrade head
```

### Monitoring memory usage

```bash
# Quick overview
free -h

# Per-container usage
docker stats --no-stream

# If a container is killed (OOMKilled), check:
docker inspect CONTAINER_ID | grep -i oom
```

If you see OOM kills, consider upgrading to `t3.2xlarge` (32GB RAM).

---

## 11. Quick Reference Card

| Task | Command |
|------|---------|
| Start all services | `docker compose -f docker-compose.prod.yml up -d` |
| Stop all services | `docker compose -f docker-compose.prod.yml down` |
| View live logs | `docker compose -f docker-compose.prod.yml logs -f` |
| Rebuild after code change | `docker compose -f docker-compose.prod.yml up --build -d` |
| Run DB migrations | `docker compose -f docker-compose.prod.yml exec ai-api python -m alembic upgrade head` |
| Create API key | `docker compose -f docker-compose.prod.yml exec ai-api python scripts/seed_db.py` |
| Health check | `curl http://localhost/api/v1/health/ready` |
| Monitor resources | `docker stats` |
| Shell into API | `docker compose -f docker-compose.prod.yml exec ai-api bash` |
| DB shell | `docker compose -f docker-compose.prod.yml exec db psql -U postgres eventai` |
| Clean Docker disk | `docker system prune -f` |

---

## 12. Troubleshooting

### Container won't start

```bash
# Check logs for the failing container
docker compose -f docker-compose.prod.yml logs ai-api

# Common causes:
# - .env file missing or malformed
# - Model files not transferred (check models/ directory)
# - Port already in use
```

### "Model not found" errors

```bash
# Verify models are in the right place
docker compose -f docker-compose.prod.yml exec ai-api ls -la /app/models/
docker compose -f docker-compose.prod.yml exec ai-api ls -la /app/models/blur_classifier/
docker compose -f docker-compose.prod.yml exec ai-api ls -la /app/models/bib_detection/
docker compose -f docker-compose.prod.yml exec ai-api ls -la /app/models/models/buffalo_l/
```

### Database connection errors

```bash
# Check if DB is running
docker compose -f docker-compose.prod.yml ps db

# Test connection from API container
docker compose -f docker-compose.prod.yml exec ai-api \
  python -c "from src.config import get_settings; print(get_settings().DATABASE_URL)"
```

### Out of memory

```bash
# Check memory usage
free -h
docker stats --no-stream

# Solutions:
# 1. Reduce Celery concurrency: edit docker-compose.prod.yml, change --concurrency=2 to --concurrency=1
# 2. Reduce API workers: change WORKERS=2 to WORKERS=1 in .env
# 3. Upgrade instance: stop instance in AWS Console → Change Instance Type → Start
```

### Redis connection refused

```bash
# Check Redis is healthy
docker compose -f docker-compose.prod.yml ps redis

# Test connection
docker compose -f docker-compose.prod.yml exec redis redis-cli -a YOUR_REDIS_PASSWORD ping
# Should return: PONG
```

### Celery tasks stuck / not processing

```bash
# Check worker is running
docker compose -f docker-compose.prod.yml logs celery-worker

# Restart the worker
docker compose -f docker-compose.prod.yml restart celery-worker

# Check Redis queue length
docker compose -f docker-compose.prod.yml exec redis \
  redis-cli -a YOUR_REDIS_PASSWORD llen default
```

### Nginx 502 Bad Gateway

The API container hasn't started yet or crashed.

```bash
# Check if ai-api is running
docker compose -f docker-compose.prod.yml ps ai-api

# Check its logs
docker compose -f docker-compose.prod.yml logs --tail=50 ai-api

# Common cause: models still loading (wait 60-90 seconds on first start)
```

### SSL certificate renewal

Certificates auto-renew via the certbot container. To manually renew:

```bash
docker compose -f docker-compose.prod.yml run --rm certbot renew
docker compose -f docker-compose.prod.yml restart nginx
```

---

## Architecture Diagram

```
                    Internet
                       │
                       ▼
                ┌─────────────┐
                │   AWS EC2    │
                │  t3.xlarge   │
                └──────┬──────┘
                       │
              ┌────────▼────────┐
              │     Nginx       │ ← Port 80/443
              │  (reverse proxy │
              │   rate limiting)│
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    FastAPI      │ ← Port 8000 (internal)
              │   (2 workers)   │
              │  Model Registry │
              │ blur│face│bib   │
              └───┬─────────┬───┘
                  │         │
          ┌───────▼───┐ ┌───▼────────┐
          │ PostgreSQL │ │   Redis    │
          │ 16+pgvector│ │   7.4      │
          │ (jobs, face│ │ (Celery    │
          │  embeddings│ │  broker,   │
          │  webhooks) │ │  cache)    │
          └───────────┘ └─────┬──────┘
                              │
                      ┌───────▼───────┐
                      │ Celery Worker │
                      │ Queues:       │
                      │  default      │
                      │  blur         │
                      │  face         │
                      │  bib          │
                      └───────────────┘
```

---

## Cost Estimate (t3.xlarge)

| Resource | Monthly Cost (approx) |
|----------|----------------------|
| EC2 t3.xlarge (on-demand, 24/7) | ~$122 |
| EBS 60GB gp3 | ~$5 |
| Data transfer (first 100 GB out) | ~$9 |
| **Total** | **~$136/month** |

**To reduce cost:**
- Use **Reserved Instances** (1-year commitment) → ~$77/month (37% savings)
- Use **Spot Instances** for testing → ~$50/month (can be interrupted)
- Stop the instance when not testing → pay only for EBS storage (~$5/month)
