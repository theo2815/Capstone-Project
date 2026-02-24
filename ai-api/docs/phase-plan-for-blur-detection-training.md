# Blur Detection Classifier — Training Plan

## Current Training Status

**Blur detection is not yet fully trained.**

We are currently gathering more images for `defocused_object_portrait` (Phase 1 paused). In the meantime, we are proceeding with Phase 2 (`defocused_blurred`) training first.

---

## Blur Detection Logic (Strict Rules)

These rules define what is VALID (sharp) vs INVALID (blur) and must be enforced during both training (labeling) and inference.

### What is SHARP / VALID

| Scenario | Classification | Reason |
|----------|---------------|--------|
| All pixels sharp, no blur anywhere | **SHARP (valid)** | Fully sharp image |
| Subject sharp + background blurred (bokeh/DOF) | **SHARP (valid)** | Intentional depth-of-field; background blur is not an error |
| Panning shot: subject sharp, background streaked | **SHARP (valid)** | Standard event photography technique |

**Background blur alone must never be treated as an error.** Portrait and event photos with a sharp subject and blurred background are the standard in marathon/event photography and must always be classified as SHARP.

### What is BLUR / INVALID

| Blur Type | Condition | Classification |
|-----------|-----------|---------------|
| `defocused_object_portrait` | Main subject is out of focus, but there is a clear sharp region elsewhere (background, foreground) — the camera focused on the wrong plane | **BLUR (invalid)** |
| `defocused_blurred` | The image is **predominantly** out of focus — the overall frame is soft, even if minor sharp edges exist in isolated spots (e.g. feet touching the ground, high-contrast edges). No clear "in-focus region" exists. | **BLUR (invalid)** |
| `motion_blurred` | Blur caused by camera shake or subject movement — streaks, ghosting | **BLUR (invalid)** |

**Key distinction between `defocused_object_portrait` and `defocused_blurred`:**
- **`defocused_object_portrait`**: There is an obvious wrong focus plane — the subject is soft but the background is clearly sharp
- **`defocused_blurred`**: The whole frame is predominantly soft — no clear "this part is in focus" region, even if some edges happen to retain mild sharpness due to depth-of-field gradients

### Detection Constraints for `defocused_blurred`

`defocused_blurred` must be detected **only** when:

1. There is **no clearly sharp primary subject** in the image
2. **Subject-level details** (faces, bib numbers, key identifiers) **are not in focus**
3. The image is **globally unusable** as a sharp event photo

Portrait photos remain valid **only** when the primary subject itself is clearly sharp (sharp face/torso), regardless of background blur.

**An image must still be classified as `defocused_blurred` even if it contains localized sharp edges, such as:**

- Road markings
- Clothing edges or seams
- High-contrast outlines
- Feet touching the ground
- Minor foreground or background details
- Noise or compression artifacts that may appear sharp

**The presence of partial sharp edges must not override defocused blur detection.** These incidental elements do not constitute a "clearly sharp primary subject."

**Core principle:** When Defocused Blur mode is selected, the system must be strict and prioritize **subject-level clarity** (face, torso, bib number) over pixel-level or edge-level sharpness. If the primary subject is not in focus, the image is defocused — period.

**What makes a portrait VALID (not defocused_blurred):**
- The primary subject (runner's face, torso, bib) is clearly sharp and in focus
- Background blur is irrelevant — only subject sharpness matters
- This applies regardless of how much of the frame is covered by bokeh

### Technical Approach

**Current (Phase 1-3): Single-classifier approach**

YOLOv8n-cls classifies the whole image into 4 categories. The CNN learns spatial patterns implicitly from training data — it can distinguish "sharp subject region + soft background" (= `sharp`) from "uniform softness across the frame" (= `defocused_blurred`) based on learned features. This works when:

- The `sharp` training set contains diverse portrait photos with varying bokeh levels (current: 468 real marathon photos)
- The `defocused_blurred` training set contains frames where the whole scene is soft (current: 350 images)
- Edge cases (heavy bokeh where 90% of frame is soft but subject is sharp) are well-represented in training data

**If accuracy falls short: Two-stage validation (future enhancement)**

If the single classifier cannot achieve zero false positives on valid portrait photos, a two-stage approach would be added:

| Stage | What it does | Purpose |
|-------|-------------|---------|
| Stage 1 | Detect primary subject (person/runner via object detection) | Find the region that matters |
| Stage 2 | Evaluate sharpness specifically on subject region vs full frame | Confirm blur is on the subject, not just background |

This provides explicit subject-level sharpness validation. It would only be implemented if per-class evaluation reveals portrait misclassification that training data improvements alone cannot resolve.

### Detection Behavior

- When the user selects a blur detection mode, detection is **mandatory (100%)** with no tolerance or optional threshold
- If any of the above blur types are detected, the image **must always** be flagged as BLUR / INVALID
- **Zero false positives** for valid portrait photos (sharp subject + blurred background)
- **100% detection accuracy** for selected blur types

### Labeling Rules for Training Data

| Image Content | Correct Label | Common Mistake to Avoid |
|--------------|---------------|------------------------|
| Runner sharp, background bokeh | `sharp` | Do NOT label as defocused — the subject is sharp |
| Runner sharp, background motion-streaked (panning) | `sharp` | Do NOT label as motion_blurred — the subject is sharp |
| Runner out of focus, background sharp | `defocused_object_portrait` | This is a focus error, not intentional bokeh |
| Mostly soft/defocused, but some minor sharp edges (e.g. feet, ground contact) | `defocused_blurred` | Do NOT label as sharp — the overall frame is predominantly out of focus |
| Everything soft, nothing sharp | `defocused_blurred` | Full-frame defocus |
| Runner has ghosting/streaks from movement | `motion_blurred` | Camera shake or subject motion |
| Noisy but sharp (low light / high ISO) | `sharp` | Noise is not blur |

---

## Training Roadmap

| Phase | Blur Type | Status | Notes |
|-------|-----------|--------|-------|
| **Phase 1** | `defocused_object_portrait` | **Paused** | Gathering more real-world images; ~150 current, needs more |
| **Phase 2** | `defocused_blurred` | **Up Next** | 350 images available; proceeding while Phase 1 collects data |
| **Phase 3** | `motion_blurred` | Pending | 352 images available; train after Phase 2 |

Phase 1 is paused for image collection. Training continues with Phase 2 (`defocused_blurred`) in the meantime.

---

## Model Scope & Constraints

The blur detection AI must detect **only** the following three blur types:

1. **`defocused_object_portrait`** — Main subject is out of focus; background may be sharp or mixed
2. **`defocused_blurred`** — Image is predominantly out of focus; the overall frame is soft even if minor sharp edges exist in isolated spots
3. **`motion_blurred`** — Blur caused by camera shake or subject movement

**No other blur types should be detected or classified.** The `sharp` class exists solely as a training baseline so the model can distinguish "no blur detected" from the blur categories.

### Accuracy Requirement

The goal is **maximum accuracy (target: 100%)** for detecting these three blur types, especially on real-world running event images. Zero false positives for valid portrait photos.

---

## Overview

Multi-class blur classifier using **YOLOv8n-cls** (1.4M parameters, 3.4 GFLOPs) fine-tuned to classify race event photography into 4 categories. The model runs on CPU via ONNX Runtime and integrates into the EventAI API as `POST /blur/classify`.

The API exposes **user-selectable blur type detection**: the caller chooses which blur type to check for and receives a binary `Detected / Not Detected` response with a confidence score. Internally the model classifies into all 4 categories, but only the user-selected blur type is evaluated.

---

## Blur Classes

| # | Class | Description | Visual Cues | API-Selectable |
|---|-------|-------------|-------------|----------------|
| 1 | `sharp` | Subject in focus, background may have natural bokeh | Clear edges on subject, readable bib numbers, sharp facial features | No (training baseline only) |
| 2 | `defocused_object_portrait` | Main subject out of focus, background may be sharp or mixed | Background elements are sharper than the runner; camera focused on wrong plane | **Yes** |
| 3 | `defocused_blurred` | Image predominantly out of focus | Overall frame is soft; no clear in-focus region, though minor sharp edges may exist in isolated spots (e.g. feet, ground contact, high-contrast edges) | **Yes** |
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

The `sharp` images have been recently added for comparison. The folder `Portrait Photos Running event and bib numbers/` contains **real-world marathon and running event photos**, including visible bib numbers. These represent **actual production data** — the kind of images the model will encounter in real use. Many of these photos feature intentional background bokeh with a sharp subject — this is the correct, valid output of event photography.

### Original Images

| Class | Directory | Count | Status |
|-------|-----------|-------|--------|
| `sharp` | `Portrait Photos Running event and bib numbers/` | 468 | Real-world production images (includes bokeh portraits) |
| `defocused_object_portrait` | `Defocused object in portrait/` | ~150 | **Needs more images** (expanded from 23, collection ongoing) |
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

### Round 3: Per-Phase Training (CURRENT)

- **Phase 1 (`defocused_object_portrait`):** Paused — gathering more real-world images
- **Phase 2 (`defocused_blurred`):** Up next — evaluate per-class accuracy, improve if needed
- **Phase 3 (`motion_blurred`):** Pending
- **Goal:** 100% detection accuracy per blur type, zero false positives on valid portraits

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
| Panning shots | Background blurred, subject sharp | Label as `sharp` — subject is sharp |
| Bokeh portraits | Intentional shallow DOF, subject sharp | Label as `sharp` — background blur is not an error |
| Mild motion + defocus | Both blur types present | Label by dominant type |
| Low light / high ISO | Noise can look like blur | Label as `sharp` — noise is not blur |

---

## Training Priority

- **Zero false positives** — valid portrait photos (sharp subject + blurred background) must never be flagged as blur
- **100% detection accuracy** — when blur is present on the subject, it must always be detected
- **Maximum accuracy (target: 100%)** — especially on real-world running event images
- **Clear separation between blur types** — no overlapping labels between categories
- **Mandatory detection** — when user selects a blur type, detection is enforced with no optional threshold

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
  Portrait Photos Running event and bib numbers/   <- sharp (real-world production data, includes bokeh portraits)
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
