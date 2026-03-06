# Training Guide: Face + Bib Detector

This guide covers how to set up and train the combined **face + bib number detection model** (YOLOv8n) on a separate machine. The AI agent assisting you should read this file before proceeding.

---

## Prerequisites

### Hardware

- **NVIDIA GPU recommended** (RTX 3060 or better). Training on CPU is possible but significantly slower.
- At least **16GB RAM**
- At least **5GB free disk space** (dataset ~1.7GB + model artifacts)

### Software

- **Python 3.11 or 3.12** (required)
- **Git** (to clone the repo)
- **NVIDIA GPU drivers + CUDA 12.1** (if using GPU)

---

## Setup Steps

### 1. Clone the repository

```bash
git clone https://gitlab.com/theocedricchan28/api-ai.git
cd api-ai/ai-api
```

### 2. Install training dependencies

**GPU (recommended):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
```

**CPU only (slower, not recommended for full training):**

```bash
pip install ultralytics
```

> `ultralytics` will pull in `torch`, `opencv-python`, `numpy`, and other required packages automatically.

### 3. Get the dataset

The `Training-Images/` folder is **not tracked by git** (gitignored because it's 1.7GB). You need to obtain it separately.

**Option A: Copy from shared drive / cloud storage**

Ask the team lead for the `face_bib_detection/` dataset folder and place it at:

```
ai-api/Training-Images/face_bib_detection/
```

**Option B: Copy directly from the annotation machine**

If you have access to the machine that ran the annotation, copy the entire folder:

```bash
# From the annotation machine, compress:
cd ai-api
tar -czf face_bib_dataset.tar.gz Training-Images/face_bib_detection/

# On your training machine, extract into ai-api/:
cd ai-api
tar -xzf face_bib_dataset.tar.gz
```

### 4. Verify the dataset structure

After placing the dataset, verify this exact structure exists:

```
ai-api/Training-Images/face_bib_detection/
  images/
    train/          <- 1,315 images (.JPG)
    val/            <- 323 images (.JPG)
  labels/
    train/          <- 1,315 label files (.txt)
    val/            <- 323 label files (.txt)
  classes.yaml      <- class definitions
```

**Quick verification commands:**

```bash
# Check image counts
ls Training-Images/face_bib_detection/images/train/ | wc -l   # expect: 1315
ls Training-Images/face_bib_detection/images/val/ | wc -l     # expect: 323

# Check label counts
ls Training-Images/face_bib_detection/labels/train/ | wc -l   # expect: 1315
ls Training-Images/face_bib_detection/labels/val/ | wc -l     # expect: 323

# Check classes.yaml exists
cat Training-Images/face_bib_detection/classes.yaml
```

**Expected `classes.yaml` content:**

```yaml
path: .
train: images/train
val: images/val

names:
  0: face
  1: bib

nc: 2
```

---

## Training

### Run the training script

```bash
python scripts/train_face_bib_detector.py
```

### What the script does

1. Validates the dataset structure (checks all folders and files exist)
2. Downloads `yolov8n.pt` (pretrained COCO weights, ~6MB, automatic)
3. Trains for up to 100 epochs with early stopping (patience=20)
4. Saves the best model to `runs/detect/face_bib_det/weights/best.pt`
5. Runs validation and prints mAP50 and mAP50-95 scores

### Training configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `yolov8n.pt` | YOLOv8 nano, pretrained on COCO |
| Epochs | 100 | Maximum — early stopping may trigger sooner |
| Image size | 640px | Standard YOLO input size |
| Batch size | 8 | Reduce to 4 if GPU runs out of memory (see troubleshooting) |
| Early stopping | patience=20 | Stops if no improvement for 20 epochs |
| Classes | 2 | face (0), bib (1) |

### Training augmentations (already configured)

| Augmentation | Value |
|-------------|-------|
| HSV hue shift | 0.015 |
| HSV saturation | 0.4 |
| HSV value | 0.3 |
| Rotation | 10 degrees |
| Translation | 0.1 |
| Scale | 0.3 |
| Horizontal flip | 50% |
| Mosaic | 1.0 |
| Mixup | 0.1 |

### Monitor training progress

While training runs, the script prints per-epoch metrics. You can also view the live training plots:

```
runs/detect/face_bib_det/results.png       <- loss and metric curves
runs/detect/face_bib_det/confusion_matrix.png
runs/detect/face_bib_det/val_batch0_pred.jpg  <- sample predictions on val set
```

---

## After Training

### Output files

```
runs/detect/face_bib_det/
  weights/
    best.pt          <- best model (USE THIS ONE)
    last.pt          <- last epoch model
  results.csv        <- per-epoch metrics
  results.png        <- training curves
  confusion_matrix.png
  val_batch0_pred.jpg
```

### Verify the trained model

Quick test on a validation image:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/face_bib_det/weights/best.pt")
results = model("Training-Images/face_bib_detection/images/val/IMG_0001.JPG")
results[0].show()   # opens a window with bounding boxes drawn
print(results[0].boxes)  # prints detected boxes with class + confidence
```

### What to send back after training

Send these files back to the team lead:

**Required (must send):**

| File | Why |
|------|-----|
| `runs/detect/face_bib_det/weights/best.pt` | The trained model — this is the main deliverable (~6-12MB) |

**Nice to have (for reviewing training quality):**

| File | Why |
|------|-----|
| `runs/detect/face_bib_det/results.png` | Training loss/metric curves — shows if training converged |
| `runs/detect/face_bib_det/results.csv` | Per-epoch numbers — detailed training history |
| `runs/detect/face_bib_det/confusion_matrix.png` | Shows how well the model distinguishes face vs bib |

### What happens after you send `best.pt` back

You don't need to do anything else. The team lead will:

1. **Export `best.pt` to ONNX** — lightweight, runs on any machine (no GPU needed)
2. **Place the ONNX model** in the production `models/` folder
3. **Wire it into the API** — connect the model to the face search and bib search endpoints
4. **Future:** come back to you for training the face embedding model and bib OCR model (separate phase, separate dataset)

---

## Troubleshooting

### GPU out of memory

Edit `scripts/train_face_bib_detector.py` line 65 — change `batch=8` to `batch=4`:

```python
batch=4,   # reduced from 8
```

### CUDA not detected

Check your setup:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

If `False`, ensure:
- NVIDIA drivers are installed
- CUDA 12.1 is installed
- You installed the GPU version of PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Training too slow on CPU

If you must train on CPU, consider reducing epochs:

Edit `scripts/train_face_bib_detector.py` line 63 — change `epochs=100` to `epochs=50`:

```python
epochs=50,
```

The model may still converge with fewer epochs thanks to early stopping.

### Missing dataset files

If the dataset verification fails, you need to either:
1. Get the pre-annotated dataset from the team (see Setup Step 3)
2. Or re-run annotation yourself (requires InsightFace + PaddleOCR installed):
   ```bash
   pip install insightface paddleocr paddlepaddle
   python scripts/auto_annotate_face_bib.py
   ```
   Note: annotation takes ~4 hours on CPU.

---

## Dataset Summary

This dataset was auto-annotated using InsightFace (RetinaFace) for faces and PaddleOCR for bib numbers.

| Metric | Value |
|--------|-------|
| Total images | 1,638 |
| Total annotations | 5,179 |
| Faces detected | 3,316 |
| Bibs detected | 1,863 |
| Images with faces | 1,617 (98.7%) |
| Images with bibs | 1,372 (83.8%) |
| Images with both | 1,370 (83.6%) |
| Train split | 1,315 images (80%) |
| Val split | 323 images (20%) |
| Dataset size | ~1.7GB |

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/train_face_bib_detector.py` | Training script (run this) |
| `scripts/auto_annotate_face_bib.py` | Auto-annotation script (already run, not needed unless re-annotating) |
| `Training-Images/face_bib_detection/classes.yaml` | Dataset class definitions |
| `docs/phase-plan-face-bibnumber-training.md` | Full training plan and architecture details |
