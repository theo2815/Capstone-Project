# Training Guide: Bib-Only Detector

This guide covers how to train the **bib-only detection model** (YOLOv8) on a separate machine. The AI agent assisting you should read this file before proceeding.

## Background

The combined face+bib detector (trained earlier) achieved:
- Face mAP50: **0.993** (excellent — production-ready)
- Bib mAP50: **0.561** (weak — needs a dedicated model)

This bib-only model replaces the bib detection in the combined model while face detection continues using the existing combined model.

---

## Prerequisites

### Hardware

- **NVIDIA GPU recommended** (RTX 3060 or better). Training on CPU works but is significantly slower.
- At least **16GB RAM**
- At least **5GB free disk space** (dataset + model artifacts)

### Software

- **Python 3.11 – 3.14** (required; team uses 3.14.2)
- **Git** (to clone/pull the repo)
- **NVIDIA GPU drivers + CUDA 12.1** (if using GPU)

---

## Setup Steps

### 1. Pull the latest code

```bash
cd api-ai/ai-api
git pull
```

### 2. Install training dependencies

**GPU (recommended):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics pillow-heif
```

**CPU only (slower):**

```bash
pip install ultralytics pillow-heif
```

> `pillow-heif` is required if training images are HEIF-encoded (from iPhone). Without it, images will be skipped as corrupt.

### 3. Prepare the dataset

The bib-only dataset goes in `Training-Images/bib_detection/` with this structure:

```
ai-api/Training-Images/bib_detection/
  classes.yaml              <- already exists in repo
  images/
    train/                  <- training images
    val/                    <- validation images
  labels/
    train/                  <- YOLO-format .txt label files
    val/                    <- YOLO-format .txt label files
```

**Option A — Extract from the existing combined dataset:**

The combined face+bib dataset already has 1,863 bib annotations. Extract bib-only labels:

```bash
python scripts/extract_bib_labels.py --preview   # check counts first
python scripts/extract_bib_labels.py              # extract + split 80/20
```

This copies images with bibs and remaps class `1` (bib in combined) to class `0` (bib in bib-only).

**Option B — Manually annotate new bib images (RECOMMENDED):**

The team has gathered real marathon images for training. These need to be manually annotated before training. See [Manual Annotation Guide](#manual-annotation-guide) below for full instructions.

After annotation, place the exported YOLO-format files into the dataset folders:
- Images → `Training-Images/bib_detection/images/train/` (and `val/`)
- Labels → `Training-Images/bib_detection/labels/train/` (and `val/`)

Split 80% train / 20% val. Most annotation tools can do this automatically on export.

**Option C — Combine both:**

Run Option A first, then add the manually annotated images to the same directories. More data = better.

### 4. Verify dataset structure

```bash
# Check image counts
ls Training-Images/bib_detection/images/train/ | wc -l
ls Training-Images/bib_detection/images/val/ | wc -l

# Check label counts (should match image counts)
ls Training-Images/bib_detection/labels/train/ | wc -l
ls Training-Images/bib_detection/labels/val/ | wc -l

# Check classes.yaml
cat Training-Images/bib_detection/classes.yaml
```

**Expected `classes.yaml` content:**

```yaml
path: .
train: images/train
val: images/val

nc: 1
names:
  0: bib
```

### 5. YOLO label format

Each image gets a `.txt` label file with the same name (e.g., `IMG_0001.JPG` -> `IMG_0001.txt`):

```
# Format: class_id x_center y_center width height (all normalized 0-1)
0 0.337500 0.531250 0.231000 0.110000    # bib
0 0.612000 0.480000 0.195000 0.098000    # another bib
```

Class `0` = bib (the only class).

---

## Manual Annotation Guide

The team will manually annotate real marathon images to create the bib training dataset. Manual annotations are significantly more accurate than auto-generated labels.

### Recommended annotation tools

Pick one — all export to YOLO format:

| Tool | Best for | Notes |
|------|----------|-------|
| **Roboflow** (free tier) | Easiest setup, team collaboration | Web-based, 10,000 images free, auto-export to YOLO format. Recommended for this project. |
| **CVAT** | Self-hosted, large datasets | Open-source, web-based, more complex setup |
| **LabelImg** | Offline, simple | Desktop app, lightweight, supports YOLO format natively |

### Setup with Roboflow (recommended)

1. Create a free account at roboflow.com
2. Create a new project → Object Detection → name it `bib-detection`
3. Set the single class: **`bib`**
4. Upload all gathered marathon images
5. Annotate (see labeling rules below)
6. Generate a dataset version with 80/20 train/val split
7. Export as **YOLOv8** format → download and extract into `Training-Images/bib_detection/`

### Labeling rules

**What to annotate:**

- Draw a **tight bounding box** around each visible bib/race number
- A "bib" = the physical card/sticker pinned to a runner's chest or back that displays their race number
- Include bibs that are partially occluded (folded, behind arms) if **more than 30%** of the bib is visible
- If a single image has multiple runners with bibs, annotate **all** of them

**What to skip:**

- Bibs where less than 30% is visible (too occluded to be useful)
- Bibs smaller than roughly 20x20 pixels in the image (too small to detect)
- Numbers printed on clothing that are NOT race bibs (jersey numbers, brand logos)
- Timing chips or other attachments that aren't the bib card itself

**Bounding box guidelines:**

- Box should be **tight** — edges of the box touch the edges of the bib
- Do NOT include large margins of clothing/skin around the bib
- If the bib is tilted/rotated, the box should still be axis-aligned (horizontal rectangle) covering the full bib
- For folded/curled bibs, box the visible portion only

### Dataset variety checklist

For best results, make sure your dataset includes a mix of:

- [ ] **Distances**: close-up, medium, far away runners
- [ ] **Angles**: front-facing, side-view, slight angle
- [ ] **Lighting**: sunny, overcast, shaded, backlit
- [ ] **Occlusion**: arms crossing bibs, partially hidden by other runners
- [ ] **Bib positions**: chest-mounted, waist-mounted, back-mounted
- [ ] **Group shots**: multiple runners with multiple bibs in one image
- [ ] **Motion**: runners in motion (slight blur is fine — this is realistic)

### After annotation — placing files

After exporting from your annotation tool in YOLO format, your files should look like:

```
exported_dataset/
  train/
    images/
      img001.jpg
      img002.jpg
      ...
    labels/
      img001.txt
      img002.txt
      ...
  val/
    images/
      ...
    labels/
      ...
```

Copy them into the training directory:

```bash
# Copy training images and labels
cp exported_dataset/train/images/* Training-Images/bib_detection/images/train/
cp exported_dataset/train/labels/* Training-Images/bib_detection/labels/train/

# Copy validation images and labels
cp exported_dataset/val/images/* Training-Images/bib_detection/images/val/
cp exported_dataset/val/labels/* Training-Images/bib_detection/labels/val/
```

Then verify with Step 4 below.

### Target dataset size

- **Minimum**: 500 annotated images (expect ~0.75+ mAP50)
- **Recommended**: 1,000–1,500 annotated images (expect ~0.90+ mAP50)
- **For 99%+ accuracy**: 2,000+ diverse images with clean annotations

Quality matters more than quantity — 800 clean hand-labeled images will outperform 3,000 noisy auto-labels.

---

## Training

### Run the training script

```bash
# Default (YOLOv8n, 100 epochs, batch 8)
python scripts/train_bib_detector.py

# Higher accuracy (larger model, more epochs)
python scripts/train_bib_detector.py --model yolov8s.pt --epochs 150

# If GPU runs out of memory
python scripts/train_bib_detector.py --batch 4
```

### Training configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Base model | `yolov8n.pt` | Use `yolov8s.pt` for higher accuracy at ~2x cost |
| Epochs | 100 | Early stopping (patience=20) handles convergence |
| Image size | 640px | Standard YOLO input size |
| Batch size | 8 | Reduce to 4 if GPU runs out of memory |

### Training augmentations (already configured)

| Augmentation | Value | Purpose |
|-------------|-------|---------|
| HSV hue shift | 0.02 | Color variation |
| HSV saturation | 0.5 | Handle different lighting |
| HSV value | 0.4 | Brightness variation |
| Rotation | 15 degrees | Tilted runners/bibs |
| Translation | 0.15 | Off-center bibs |
| Scale | 0.4 | Distance variation |
| Horizontal flip | 50% | Left/right symmetry |
| Mosaic | 1.0 | Multi-object learning |
| Mixup | 0.15 | Occlusion robustness |

### Monitor training

```
runs/detect/bib_det/results.png            <- loss and metric curves
runs/detect/bib_det/confusion_matrix.png
runs/detect/bib_det/val_batch0_pred.jpg    <- sample predictions on val set
```

### Target metrics

| Metric | Target | Notes |
|--------|--------|-------|
| mAP50 (bib) | > 0.75 | Combined model only reached 0.561 |
| mAP50-95 (bib) | > 0.40 | Combined model only reached 0.215 |
| Inference time | < 30ms (GPU), < 150ms (CPU) | |

---

## After Training

### Output files

```
runs/detect/bib_det/
  weights/
    best.pt          <- best model (USE THIS ONE)
    last.pt          <- last epoch model
  results.csv        <- per-epoch metrics
  results.png        <- training curves
  confusion_matrix.png
```

### Export to ONNX

```bash
python scripts/export_bib_detector.py --force
```

This exports `best.pt` to ONNX and copies it to `models/bib_detection/yolov8n_bib.onnx`.

### What to send back after training

**Required (must send):**

| File | Why |
|------|-----|
| `runs/detect/bib_det/weights/best.pt` | The trained model — main deliverable (~6-12MB) |

**Nice to have (for reviewing training quality):**

| File | Why |
|------|-----|
| `runs/detect/bib_det/results.png` | Training curves — shows if training converged |
| `runs/detect/bib_det/results.csv` | Per-epoch numbers |
| `runs/detect/bib_det/confusion_matrix.png` | Detection quality breakdown |

### What happens after you send `best.pt` back

The team lead will:
1. Export to ONNX (`python scripts/export_bib_detector.py --force`)
2. Place the ONNX model in the production `models/bib_detection/` folder
3. Wire it into the API — the `BibDetector` class loads from this path automatically

---

## Troubleshooting

### "No module named 'pi_heif'" / all images skipped as corrupt

```bash
pip install pillow-heif
```

Then delete any label cache files:

```bash
# Windows
del Training-Images\bib_detection\labels\train.cache
del Training-Images\bib_detection\labels\val.cache

# Linux/Mac
rm Training-Images/bib_detection/labels/train.cache
rm Training-Images/bib_detection/labels/val.cache
```

### GPU out of memory

```bash
python scripts/train_bib_detector.py --batch 4
```

### CUDA not detected

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

If `False`, install the GPU version of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Ultralytics creates nested directory

Sometimes YOLO creates `runs/detect/detect/bib_det/` instead of `runs/detect/bib_det/`. Both the training and export scripts handle this automatically.

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/train_bib_detector.py` | Training script (run this) |
| `scripts/export_bib_detector.py` | ONNX export script (run after training) |
| `scripts/extract_bib_labels.py` | Extract bib-only labels from combined dataset |
| `Training-Images/bib_detection/classes.yaml` | Dataset class definitions (single class: bib) |
| `docs/training-guide-bib.md` | This guide |
