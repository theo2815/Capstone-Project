# Face Recognition & Bib Number — Training Plan

## Current Training Status

**Phase 6 (Face Recognition) and Phase 7 (Bib Number) have not started yet.** Blur detection training (Phases 1-5) is in progress — see [phase-plan-for-blur-detection-training.md](phase-plan-for-blur-detection-training.md).

---

## Training Roadmap

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| **Phase 6** | Face Recognition | **Not Started** | Combined face+bib detection → face embedding → matching |
| **Phase 7** | Bib Number Reading | **Not Started** | Combined face+bib detection → bib OCR → number extraction |

---

## Architecture: Combined Detection Pipeline

Face detection and bib detection share the **same training images** (event photos contain both faces and bibs on runners). Instead of training two separate detection models, a single combined YOLOv8n model detects both in one pass.

### 3-Model Architecture

```
Event Photo
    │
    ▼
┌─────────────────────────────────┐
│  Model 1: Combined Detector     │
│  YOLOv8n (2 classes: face, bib) │
│  One inference pass             │
└──────────┬──────────┬───────────┘
           │          │
     Face crops    Bib crops
           │          │
           ▼          ▼
┌──────────────┐  ┌──────────────┐
│ Model 2:     │  │ Model 3:     │
│ Face Embedder│  │ Bib OCR      │
│ (feature     │  │ (digit       │
│  vectors)    │  │  reader)     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
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
POST /api/v1/face/search  → uses face crops only, ignores bib detections
POST /api/v1/bib/search   → uses bib crops only, ignores face detections
POST /api/v1/runner/identify → uses both for complete runner identification
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
| Event photos with face bounding boxes | YOLO-format annotations marking face regions (shared annotation with bib) | **Not Started** |
| Face identity labels | Group cropped faces by runner identity for embedding training | **Not Started** |
| Reference photos | Clean front-facing photos for matching baseline | **Not Started** |

### Training Pipeline

```
1. Collect event photos (same images used for both face and bib annotation)
2. Annotate face bounding boxes (YOLO format) alongside bib annotations
3. Train combined face+bib detection model (YOLOv8n, 2 classes)
4. Crop detected faces from training images
5. Train/fine-tune face embedding model on cropped faces with identity labels
6. Build matching pipeline (cosine similarity, set threshold)
7. Export both models to ONNX for production inference
```

### Integration

- `POST /api/v1/face/search` — detect faces in an image and match against reference database
- `POST /api/v1/face/search/batch` — batch processing for multiple images
- Integrates with the existing EventAI API architecture (`api/` → `services/` → `ml/`)

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
- **Number formats:** Pure digits, alphanumeric, or with prefixes (e.g., "A-1234", "F502")

### Dataset Requirements

| Data Type | Description | Status |
|-----------|-------------|--------|
| Event photos with bib bounding boxes | YOLO-format annotations marking bib regions (shared annotation with face) | **Not Started** |
| Bib number ground truth | Correct text/number for each annotated bib | **Not Started** |
| Varied bib designs | Samples from different race events and bib styles | **Not Started** |

### Training Pipeline

```
1. Collect event photos (same images used for both face and bib annotation)
2. Annotate bib bounding boxes (YOLO format) alongside face annotations
3. Train combined face+bib detection model (YOLOv8n, 2 classes)
4. Crop detected bibs from training images
5. Label cropped bibs with ground truth numbers
6. Train/fine-tune OCR model on cropped bibs
7. Build end-to-end pipeline (detect bib → crop → OCR → output number)
8. Export both models to ONNX for production inference
```

### Integration

- `POST /api/v1/bib/search` — detect and read bib numbers from an image
- `POST /api/v1/bib/search/batch` — batch processing for multiple images
- Integrates with the existing EventAI API architecture (`api/` → `services/` → `ml/`)

### Accuracy Targets

- **Accurate bib detection** — detect bibs at various angles, distances, and occlusion levels
- **High OCR accuracy** — correctly read bib numbers including partial or angled text
- **Production speed** — efficient enough for real-time or near-real-time processing
- **Practical limit:** Fully occluded or off-frame bibs cannot be detected — this is expected

---

## Annotation Workflow

Since both phases share the same training images, annotation is done **once per image** with two label classes:

### YOLO Annotation Format

```
# Each image gets a .txt label file with both face and bib boxes
# Format: class_id x_center y_center width height (normalized 0-1)

# classes.yaml
# 0: face
# 1: bib

# Example annotation for one image:
0 0.45 0.25 0.08 0.10    # face bounding box
1 0.50 0.55 0.12 0.15    # bib bounding box
```

### Annotation Tools

Any YOLO-compatible annotation tool works:
- **CVAT** (free, web-based)
- **Roboflow** (free tier available, auto-export to YOLO format)
- **LabelImg** (free, desktop)
- **Label Studio** (free, self-hosted)

---

## File Reference

### Related Docs

| File | Purpose |
|------|---------|
| [phase-plan-for-blur-detection-training.md](phase-plan-for-blur-detection-training.md) | Blur detection training plan (Phases 1-5) |

### Artifacts (Future)

| File | Purpose |
|------|---------|
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

### Dataset Location (Future)

```
ai-api/Training-Images/
  face_bib_detection/                  <- event photos for combined detection training
    images/
      train/                           <- training images
      val/                             <- validation images
    labels/
      train/                           <- YOLO annotation .txt files
      val/                             <- YOLO annotation .txt files
    classes.yaml                       <- class definitions (face, bib)
  face_embeddings/                     <- cropped faces grouped by identity
    runner_001/
    runner_002/
    ...
  bib_ocr/                            <- cropped bibs with ground truth labels
    images/
    labels.csv                         <- image_filename, bib_number
```
