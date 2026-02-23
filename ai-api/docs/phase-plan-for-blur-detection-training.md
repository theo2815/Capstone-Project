# Blur Detection Classifier — Training Plan

## Current Training Status

**Blur detection is not yet fully trained.**

We are currently training only the `defocused_object_portrait` blur type, and this training is still incomplete. More real-world images of defocused objects in portrait shots need to be collected before this class reaches production-ready accuracy.

Once `defocused_object_portrait` training is completed, we will proceed to train the remaining blur types in order.

---

## Training Roadmap

| Phase | Blur Type | Status | Notes |
|-------|-----------|--------|-------|
| **Phase 1** | `defocused_object_portrait` | **In Progress** | Expanded from 23 to ~150 images; needs more real-world samples |
| **Phase 2** | `defocused_blurred` | Pending | Train after Phase 1 is complete |
| **Phase 3** | `motion_blurred` | Pending | Train after Phase 2 is complete |

Each phase follows the same workflow: collect images, prepare dataset, train, export, verify.

---

## Model Scope & Constraints

The blur detection AI must detect **only** the following three blur types:

1. **`defocused_object_portrait`** — Subject blurry, background sharp (back-focus / front-focus error)
2. **`defocused_blurred`** — Entire image uniformly out of focus
3. **`motion_blurred`** — Directional motion blur from camera or subject movement

**No other blur types should be detected or classified.** The `sharp` class exists solely as a training baseline so the model can distinguish "no blur detected" from the blur categories.

### Accuracy Requirement

The goal is **maximum accuracy (target: 100%)** for detecting these three blur types, especially on real-world running event images.

---

## Overview

Multi-class blur classifier using **YOLOv8n-cls** (1.4M parameters, 3.4 GFLOPs) fine-tuned to classify race event photography into 4 categories. The model runs on CPU via ONNX Runtime and integrates into the EventAI API as `POST /blur/classify`.

The API exposes **user-selectable blur type detection**: the caller chooses which blur type to check for and receives a binary `Detected / Not Detected` response with a confidence score. Internally the model classifies into all 4 categories, but only the user-selected blur type is evaluated.

---

## Blur Classes

| # | Class | Description | Visual Cues | API-Selectable |
|---|-------|-------------|-------------|----------------|
| 1 | `sharp` | Subject in focus, background may have natural bokeh | Clear edges on subject, readable bib numbers, sharp facial features | No (training baseline only) |
| 2 | `defocused_object_portrait` | Subject blurry, background sharp (back-focus / front-focus error) | Background elements are sharper than the runner; camera focused on wrong plane | **Yes** |
| 3 | `defocused_blurred` | Entire image uniformly out of focus | Everything is soft — no sharp edges anywhere; evenly blurred across the frame | **Yes** |
| 4 | `motion_blurred` | Directional motion blur from camera or subject movement | Light trails, directional streaks, ghosting on moving limbs | **Yes** |

Only the 3 blur types are selectable via the API. The `sharp` class exists solely as a training baseline so the model can distinguish "no blur detected" from the blur categories.

---

## API Behavior

When calling the blur classification endpoint:

1. **User selects a blur type** via the `blur_type` query parameter (e.g., `defocused_object_portrait`)
2. **Model classifies** the image into all 4 categories internally
3. **API returns** a focused response for the selected blur type:
   - `detected: true/false` — whether the selected blur type is the top prediction
   - `confidence` — model confidence for the top prediction
   - `blur_type_probability` — probability the model assigned to the selected blur type

```
POST /api/v1/blur/classify?blur_type=defocused_object_portrait
-> { "detected": true, "confidence": 0.94, "blur_type_probability": 0.94, ... }

POST /api/v1/blur/classify?blur_type=motion_blurred
-> { "detected": false, "confidence": 0.94, "blur_type_probability": 0.02, ... }
```

When `blur_type` is omitted, the endpoint returns the full classification with all probabilities (backward compatible).

---

## Dataset

### Dataset Clarification

The `sharp` images have been recently added for comparison. The folder `Portrait Photos Running event and bib numbers/` contains **real-world marathon and running event photos**, including visible bib numbers. These represent **actual production data** — the kind of images the model will encounter in real use.

### Original Images

| Class | Directory | Count | Status |
|-------|-----------|-------|--------|
| `sharp` | `Portrait Photos Running event and bib numbers/` | 468 | Real-world production images |
| `defocused_object_portrait` | `Defocused object in portrait/` | ~150 | **Needs more images** (expanded from 23) |
| `defocused_blurred` | `defocused_blurred/` | 350 | Good coverage |
| `motion_blurred` | `motion_blurred/` | 352 | Good coverage |
| **Total** | | **~1,320** | |

### After Augmentation (prepared dataset)

| Class | Train | Val | Total | Method |
|-------|-------|-----|-------|--------|
| `sharp` | ~375 | ~93 | ~468 | Original (no augmentation needed) |
| `defocused_object_portrait` | ~240 | ~60 | ~300 | ~150 originals + augmented to 300 |
| `defocused_blurred` | ~280 | ~70 | ~350 | Original (no augmentation needed) |
| `motion_blurred` | ~282 | ~70 | ~352 | Original (no augmentation needed) |
| **Total** | **~1,177** | **~293** | **~1,470** | |

---

## Training Results

### Round 1: Baseline (COMPLETE)

- **Dataset:** 1,193 originals + 295 augmented = 1,488 total
- **Result:** 95.9% top-1 accuracy, 100% top-5
- **Training stopped:** Epoch 22/100 (early stopping, patience=20)
- **Weakness:** `defocused_subject` had only 23 real images

### Round 2: Expanded Data + Renamed Classes (COMPLETE)

- **Changes:** Renamed `defocused_subject` -> `defocused_object_portrait`, added ~127 new real images
- **Dataset:** ~1,320 originals, augmented to ~1,470 total
- **Result:** **98.63% top-1 accuracy**, 100% top-5
- **Training stopped:** Epoch 55/100 (early stopping, patience=20, best at epoch 35)
- **Key improvement:** `defocused_object_portrait` now has ~150 real images (was 23), class naming fixed

### Round 3: More Data Collection (CURRENT — In Progress)

- **Focus:** Gathering more real-world `defocused_object_portrait` images to push accuracy toward 100%
- **Goal:** Maximize accuracy on all three blur types before moving to production
- **Status:** Paused for image collection

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image size | 224x224 | Standard for classification; balances speed vs detail |
| Batch size | 16 | Fits in CPU memory comfortably |
| Epochs | 100 (max) | Early stopping handles actual length |
| Patience | 20 | Generous patience to avoid premature stopping |
| Dropout | 0.3 | Prevents overfitting on small dataset |
| Weight decay | 0.001 | L2 regularization |
| Label smoothing | 0.1 | Softened targets, reduces overconfidence |
| LR schedule | Cosine (lr0=0.001 to lrf=0.01) | Gradual decay to fine-tune later layers |

### Augmentation (YOLOv8 built-in, applied during training)

| Augmentation | Value |
|-------------|-------|
| HSV hue | 0.015 |
| HSV saturation | 0.4 |
| HSV value | 0.3 |
| Rotation | +/- 15 degrees |
| Translation | 10% |
| Scale | 0.3 (zoom 0.7-1.3x) |
| Horizontal flip | 50% |
| Random erasing | 10% |

---

## Future Training Rounds

### Hyperparameter Tuning (If accuracy plateaus)

| Parameter | Current | Try | Why |
|-----------|---------|-----|-----|
| Dropout | 0.3 | 0.2, 0.1 | Less regularization with more data |
| Image size | 224 | 320 | Larger input preserves more blur detail |
| Label smoothing | 0.1 | 0.05 | Less smoothing once data is sufficient |
| Learning rate | 0.001 | 0.0005 | Finer tuning with balanced dataset |

### Edge Case Hardening (Production polish)

| Edge Case | Description | Resolution |
|-----------|-------------|------------|
| Panning shots | Background blurred, subject sharp | Label as `sharp` |
| Bokeh portraits | Intentional shallow DOF | Label as `sharp` |
| Mild motion + defocus | Both blur types present | Label by dominant type |
| Low light / high ISO | Noise can look like blur | Add noisy-but-sharp to `sharp` |

---

## Training Priority

- **Maximum accuracy (target: 100%)** — especially on real-world running event images
- **High precision, low false positives** — the model should not falsely flag sharp images as blurry
- **Clear separation between blur types** — no overlapping labels between categories
- **Accuracy-first** — the model should give definitive Detected/Not Detected answers without needing user-configurable thresholds

---

## File Reference

### Scripts

| File | Purpose | When to Run |
|------|---------|-------------|
| `scripts/prepare_blur_dataset.py` | Augment small classes, create ImageFolder layout | After adding new images |
| `scripts/train_blur_classifier.py` | Fine-tune YOLOv8n-cls on the dataset | After dataset preparation |
| `scripts/export_blur_classifier.py` | Export best.pt to ONNX, copy to models/ | After training completes |

### Source Code

| File | Purpose |
|------|---------|
| `src/ml/blur/classifier.py` | BlurClassifier — ONNX inference, preprocessing, `detect_blur_type()` |
| `src/schemas/blur.py` | BlurType enum, BlurClassProbabilities, BlurTypeDetectionResponse |
| `src/services/blur_service.py` | classify() and detect_blur_type() delegates to BlurClassifier |
| `src/ml/registry.py` | blur_classifier loader (optional model) |
| `src/api/v1/blur.py` | POST /blur/classify (with optional blur_type param), POST /blur/classify/batch |
| `src/workers/tasks/blur_tasks.py` | blur_classify_batch Celery task (supports blur_type) |

### Artifacts

| File | Purpose |
|------|---------|
| `runs/classify/blur_cls/weights/best.pt` | Best PyTorch model checkpoint |
| `models/blur_classifier/blur_classifier.onnx` | ONNX model for production inference |
| `models/blur_classifier/class_names.json` | Class label mapping |

### Dataset Location

```
ai-api/Training-Images/
  Portrait Photos Running event and bib numbers/   <- sharp (real-world production data)
  Defocused object in portrait/                     <- defocused_object_portrait
  defocused_blurred/                                <- defocused_blurred
  motion_blurred/                                   <- motion_blurred
  dataset/                                          <- auto-generated by prepare_blur_dataset.py
    train/{class}/
    val/{class}/
```

---

## Quick Re-Training Workflow

```bash
# Step 1: Re-prepare the dataset (handles augmentation + splits)
python scripts/prepare_blur_dataset.py

# Step 2: Re-train the model (early stopping will handle convergence)
python scripts/train_blur_classifier.py

# Step 3: Export to ONNX for production inference
python scripts/export_blur_classifier.py

# Step 4: Verify tests still pass
pytest tests/test_blur_classifier.py -v

# Step 5: Test the API endpoint
# Start the server, then:
# POST /api/v1/blur/classify?blur_type=defocused_object_portrait (targeted detection)
# POST /api/v1/blur/classify (full classification, backward compatible)
```
