# Face Recognition & Bib Number — Training Plan

## Current Training Status

**Blur detection training (Phases 1-5) is complete** — 98.68% accuracy, ONNX exported. See [phase-plan-for-blur-detection-training.md](phase-plan-for-blur-detection-training.md).

**Phase 6 (Face Recognition) and Phase 7 (Bib Number) — Auto-annotation complete, ready for training.**

- 1,638 training images collected and placed in dataset folder
- Auto-annotation script written and verified (InsightFace + PaddleOCR)
- Auto-annotation completed on all 1,638 images (3,316 faces + 1,863 bibs = 5,179 total annotations)
- Dataset split into 1,315 train / 323 val (80/20)
- Training script ready
- **Next step:** Train the combined YOLOv8n detector

---

## Training Roadmap

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| **Phase 6** | Face Recognition | **In Progress** | Combined face+bib detection → face embedding → matching |
| **Phase 7** | Bib Number Reading | **In Progress** | Combined face+bib detection → bib OCR → number extraction |

### Sub-Task Progress

| # | Task | Status | Details |
|---|------|--------|---------|
| 1 | Collect training images | **Done** | 1,638 event photos in `images/train/` |
| 2 | Write auto-annotation script | **Done** | `scripts/auto_annotate_face_bib.py` — InsightFace (faces) + PaddleOCR (bibs) |
| 3 | Optimize annotation speed | **Done** | Resize to 800px before OCR: ~9s/image (down from ~8min) |
| 4 | Preview and verify annotations | **Done** | 5-image preview: 100% face detection, 80% bib detection, no false positives |
| 5 | Run full auto-annotation (1,638 images) | **Done** | 1,638 images processed, 5,179 annotations (3,316 faces + 1,863 bibs), split 1,315 train / 323 val |
| 6 | Train combined YOLOv8n detector | **Not Started** | `scripts/train_face_bib_detector.py` ready, all pre-training fixes applied |
| 7 | Export to ONNX | **Ready** | `scripts/export_face_bib_detector.py` written and validated |
| 8 | Train/fine-tune face embedding model | **Not Started** | Requires face crops from trained detector |
| 9 | Train/fine-tune bib OCR model | **Not Started** | Requires bib crops from trained detector |

---

## Architecture: Combined Detection Pipeline

Face detection and bib detection share the **same training images** (event photos contain both faces and bibs on runners). Instead of training two separate detection models, a single combined YOLOv8n model detects both in one pass.

### 3-Model Architecture

```
Event Photo
    |
    v
+-------------------------------+
|  Model 1: Combined Detector   |
|  YOLOv8n (2 classes: face,bib)|
|  One inference pass            |
+----------+----------+---------+
           |          |
     Face crops    Bib crops
           |          |
           v          v
+--------------+  +--------------+
| Model 2:     |  | Model 3:     |
| Face Embedder|  | Bib OCR      |
| (feature     |  | (digit       |
|  vectors)    |  |  reader)     |
+------+-------+  +------+-------+
       |                 |
       v                 v
  Runner identity   Bib number "1234"
```

### Why Combined Detection

| Factor | Combined (1 model, 2 classes) | Separate (2 models) |
|--------|-------------------------------|---------------------|
| Accuracy | Same — faces and bibs are visually distinct, no confusion | Same |
| Inference speed | 1 pass per image | 2 passes per image |
| Memory | ~5-8MB (one model) | ~10-16MB (two models) |
| Annotation | One pass per image — label both face and bib boxes | Two separate annotation passes |
| Maintenance | One model to retrain | Two models to manage |
| YOLO capacity | Designed for 80+ classes; 2 classes is trivial | N/A |

The API endpoints remain separate — the combined model detects both, but each endpoint only uses the detections it needs:

```
POST /api/v1/faces/search     -> uses face crops only, ignores bib detections
POST /api/v1/bibs/recognize   -> uses bib crops only, ignores face detections
POST /api/v1/runner/identify  -> uses both for complete runner identification (planned — not yet implemented)
```

---

## Auto-Annotation Pipeline

Since manually annotating 1,638 images is impractical, an auto-annotation script was built using pre-trained models already in the project:

### How It Works

```
For each image:
  1. InsightFace (RetinaFace buffalo_l) detects faces -> [x1, y1, x2, y2, confidence]
  2. PaddleOCR detects text regions -> filter for 2+ digit text (bib numbers)
  3. Expand bib text boxes by 60% to approximate full bib card area
  4. Merge overlapping bib boxes (IoU > 0.3)
  5. Convert all boxes to YOLO format -> write .txt label file
After all images:
  6. Split 80/20 into train/val (images + labels)
```

### Speed Optimizations

| Optimization | Before | After |
|-------------|--------|-------|
| Resize to 800px before PaddleOCR | ~8 min/image | ~7s/image |
| Disable text orientation detection | Extra processing | Skipped |
| BIB_MIN_DIGITS=2 (filter false positives) | "2XU" sock text detected | Eliminated |
| **Total per image** | **~8 min** | **~9s (53x faster)** |

### Key Parameters

```python
# Face detection
FACE_CONFIDENCE_THRESHOLD = 0.5

# Bib detection
BIB_MIN_DIGITS = 2           # Require at least 2 digits to count as bib (matches BIB_MIN_CHARS config)
BIB_BOX_EXPAND_RATIO = 0.6   # Expand text box to approximate full bib card
BIB_MIN_AREA_RATIO = 0.003   # Min bib area as fraction of image
BIB_MAX_AREA_RATIO = 0.15    # Max bib area as fraction of image
BIB_MERGE_IOU_THRESHOLD = 0.3

# OCR resize for speed
OCR_MAX_DIM = 800
```

### Preview Results (5 images)

| Metric | Value |
|--------|-------|
| Images processed | 5 |
| Faces detected | 5 (100%) |
| Bibs detected | 4 (80%) |
| False positives | 0 |
| Time per image | ~9s |
| Estimated full run (1,638 images) | ~4 hours |

### Full Run Results (1,638 images)

| Metric | Value |
|--------|-------|
| Images processed | 1,638 |
| Skipped (unreadable) | 0 |
| Total faces detected | 3,316 |
| Total bibs detected | 1,863 |
| Total annotations | 5,179 |
| Images with faces | 1,617 (98.7%) |
| Images with bibs | 1,372 (83.8%) |
| Images with BOTH face + bib | 1,370 (83.6%) |
| Images with face only | 247 (15.1%) |
| Images with bib only | 2 (0.1%) |
| Images with neither | 19 (1.2%) |
| Avg time per image | ~9.7s |
| Total time | ~266 min |
| Train split | 1,315 images |
| Val split | 323 images |

### Usage

```bash
# Preview first N images (no train/val split)
python scripts/auto_annotate_face_bib.py --preview 5

# Save annotated preview images for visual verification
python scripts/auto_annotate_face_bib.py --visualize 10

# Run full annotation on all images + split train/val
python scripts/auto_annotate_face_bib.py
```

---

## Training Configuration

### Combined Detector (YOLOv8n)

```python
model = YOLO("yolov8n.pt")  # pretrained COCO weights
model.train(
    data="classes.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,           # early stopping
    # Augmentation
    hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
    degrees=10.0, translate=0.1, scale=0.3,
    fliplr=0.5, mosaic=1.0, mixup=0.1,
)
```

### Dataset Split

- **Train:** 1,315 images (80%)
- **Val:** 323 images (20%)
- Only images with at least one annotation are included in the split
- Split is deterministic (seed=42)

### Usage

```bash
# Train the combined face+bib detector
python scripts/train_face_bib_detector.py
```

---

## Phase 6: Face Recognition

### Objective

Identify runners in event photos by detecting faces and matching them against reference photos. Attendees can find all their photos by matching their face across the event gallery.

### Components

| # | Component | Model | Purpose |
|---|-----------|-------|---------|
| 1 | Face Detection | Combined YOLOv8n detector (shared with Phase 7) | Locate face bounding boxes in event photos |
| 2 | Face Embedding | Lightweight embedding model (e.g., MobileFaceNet, ArcFace-mobile) | Generate 128/512-dim feature vectors per face |
| 3 | Face Matching | Cosine similarity / nearest-neighbor search | Compare embeddings against reference database |

### Key Considerations

- **Outdoor conditions:** Varied lighting, angles, sunglasses, hats, sweat — event photos are harder than studio portraits
- **Partial faces:** Runners may be mid-stride, turning away, or partially occluded
- **Multiple faces per image:** Group shots, spectators in background — need to handle multiple detections
- **Reference photos:** Runners may register with a selfie or ID photo for matching
- **Privacy:** Face data must be handled securely and in compliance with data protection regulations

### Dataset Requirements

| Data Type | Description | Status |
|-----------|-------------|--------|
| Event photos with face bounding boxes | YOLO-format annotations marking face regions (shared annotation with bib) | **Done** — 1,638 images annotated, 3,316 faces detected (98.7% of images) |
| Face identity labels | Group cropped faces by runner identity for embedding training | **Not Started** |
| Reference photos | Clean front-facing photos for matching baseline | **Not Started** |

### Training Pipeline

```
1. Collect event photos (same images used for both face and bib annotation)     [DONE]
2. Auto-annotate face bounding boxes using InsightFace (RetinaFace)             [DONE]
3. Train combined face+bib detection model (YOLOv8n, 2 classes)                 [READY]
4. Crop detected faces from training images
5. Train/fine-tune face embedding model on cropped faces with identity labels
6. Build matching pipeline (cosine similarity, set threshold)
7. Export both models to ONNX for production inference
```

### Integration

- `POST /api/v1/face/search` — detect faces in an image and match against reference database
- `POST /api/v1/face/search/batch` — batch processing for multiple images
- Integrates with the existing EventAI API architecture (`api/` -> `services/` -> `ml/`)

### Accuracy Targets

- **Reliable face detection** — detect faces even in challenging outdoor/action conditions
- **Accurate matching** — minimize false matches while maximizing true positive identification
- **Performance** — fast enough for batch processing of large event galleries
- **Practical limit:** Faces not visible in the image (turned away, fully occluded) cannot be detected — this is expected, not a model failure

---

## Phase 7: Bib Number Reading

### Objective

Automatically read race bib numbers from event photos. This enables instant photo-to-runner matching by bib number — the most reliable identifier in race photography.

### Components

| # | Component | Model | Purpose |
|---|-----------|-------|---------|
| 1 | Bib Detection | Combined YOLOv8n detector (shared with Phase 6) | Locate bib bounding boxes in event photos |
| 2 | Bib OCR | CNN-based digit recognizer or lightweight OCR model | Read digits/text from cropped bib regions |

### Key Considerations

- **Bib variety:** Different races use different bib designs, fonts, colors, and layouts
- **Occlusion:** Bibs may be partially covered by arms, hydration belts, or other runners
- **Angles and distance:** Bibs shot from various angles, distances, and with motion blur
- **Multiple bibs per image:** Group shots may contain several visible bib numbers
- **Number formats:** Pure digits, alphanumeric, or with prefixes (e.g., "A-1234", "F502"). The bib character filter supports alphanumeric characters, hyphens (`-`), and underscores (`_`).

### Dataset Requirements

| Data Type | Description | Status |
|-----------|-------------|--------|
| Event photos with bib bounding boxes | YOLO-format annotations marking bib regions (shared annotation with face) | **Done** — 1,638 images annotated, 1,863 bibs detected (83.8% of images) |
| Bib number ground truth | Correct text/number for each annotated bib | **Not Started** |
| Varied bib designs | Samples from different race events and bib styles | **Not Started** |

### Training Pipeline

```
1. Collect event photos (same images used for both face and bib annotation)     [DONE]
2. Auto-annotate bib bounding boxes using PaddleOCR text detection              [DONE]
3. Train combined face+bib detection model (YOLOv8n, 2 classes)                 [READY]
4. Crop detected bibs from training images
5. Label cropped bibs with ground truth numbers
6. Train/fine-tune OCR model on cropped bibs
7. Build end-to-end pipeline (detect bib -> crop -> OCR -> output number)
8. Export both models to ONNX for production inference
```

### Integration

- `POST /api/v1/bib/search` — detect and read bib numbers from an image
- `POST /api/v1/bib/search/batch` — batch processing for multiple images
- Integrates with the existing EventAI API architecture (`api/` -> `services/` -> `ml/`)

### Accuracy Targets

- **Accurate bib detection** — detect bibs at various angles, distances, and occlusion levels
- **High OCR accuracy** — correctly read bib numbers including partial or angled text
- **Production speed** — efficient enough for real-time or near-real-time processing
- **Practical limit:** Fully occluded or off-frame bibs cannot be detected — this is expected

---

## Annotation Format

Since both phases share the same training images, annotation is done **once per image** with two label classes:

### YOLO Annotation Format

```
# Each image gets a .txt label file with both face and bib boxes
# Format: class_id x_center y_center width height (normalized 0-1)

# classes.yaml
# 0: face
# 1: bib

# Example annotation (IMG_0001.txt):
0 0.587522 0.402113 0.074585 0.058900    # face 1
0 0.365422 0.317864 0.094773 0.073658    # face 2
0 0.925610 0.406801 0.064587 0.056858    # face 3
1 0.337500 0.531250 0.231000 0.110000    # bib
```

### Annotation Method

Auto-annotation using pre-trained models (no manual labeling needed):

| Detector | Model | What It Finds |
|----------|-------|---------------|
| InsightFace (RetinaFace) | `buffalo_l` | Face bounding boxes |
| PaddleOCR | PP-OCRv5 | Text regions containing 2+ digits (bib numbers) |

---

## File Reference

### Scripts

| File | Purpose | Status |
|------|---------|--------|
| `scripts/auto_annotate_face_bib.py` | Auto-annotate images using InsightFace + PaddleOCR | **Ready** |
| `scripts/train_face_bib_detector.py` | Train YOLOv8n combined face+bib detector | **Ready** |
| `scripts/export_face_bib_detector.py` | Export trained model to ONNX | **Ready** |

### Related Docs

| File | Purpose |
|------|---------|
| [phase-plan-for-blur-detection-training.md](phase-plan-for-blur-detection-training.md) | Blur detection training plan (Phases 1-5, complete) |

### Dataset Location

```
ai-api/Training-Images/
  face_bib_detection/
    images/
      train/                           <- 1,315 training images
      val/                             <- 323 validation images
    labels/
      train/                           <- 1,315 YOLO annotation .txt files
      val/                             <- 323 YOLO annotation .txt files
    classes.yaml                       <- class definitions (face=0, bib=1)
    annotation_preview/                <- visual verification samples (generated with --visualize)
  face_embeddings/                     <- cropped faces grouped by identity (future)
  bib_ocr/                            <- cropped bibs with ground truth labels (future)
```

### Artifacts (Future)

| File | Purpose |
|------|---------|
| `runs/detect/face_bib_det/weights/best.pt` | Best trained model weights |
| `models/face_bib_detector/face_bib_detector.onnx` | Combined face+bib detection ONNX model |
| `models/face_bib_detector/class_names.json` | Class label mapping (`face`, `bib`) |
| `models/face_embedder/face_embedder.onnx` | Face embedding ONNX model |
| `models/bib_ocr/bib_ocr.onnx` | Bib OCR ONNX model |

### Source Code (Future)

| File | Purpose |
|------|---------|
| `src/ml/face/detector.py` | Combined face+bib detector — ONNX inference |
| `src/ml/face/embedder.py` | Face embedding model — feature vector extraction |
| `src/ml/bib/ocr.py` | Bib OCR model — digit/text reading |
| `src/services/face_service.py` | Face search service layer |
| `src/services/bib_service.py` | Bib search service layer |
| `src/api/v1/face.py` | Face search API endpoints |
| `src/api/v1/bib.py` | Bib search API endpoints |

---

## How to Run (Quick Reference)

```bash
# 1. Preview auto-annotation on 5 images (verify quality)
python scripts/auto_annotate_face_bib.py --preview 5 --visualize 5

# 2. Run full auto-annotation on all 1,638 images
python scripts/auto_annotate_face_bib.py --visualize 20

# 3. Train the combined face+bib detector
python scripts/train_face_bib_detector.py

# 4. Export trained model to ONNX for production
python scripts/export_face_bib_detector.py
```
