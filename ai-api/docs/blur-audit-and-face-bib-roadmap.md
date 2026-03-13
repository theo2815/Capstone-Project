# Blur Detection Audit & Face+Bib Training Roadmap

> Generated: March 13, 2026
> Scope: Blur detection accuracy critique, edge-case analysis, and strategic face+bib training plan

---

## Table of Contents

- [Part 1: Blur Detection Technical Critique](#part-1-blur-detection-technical-critique)
  - [Current State Summary](#current-state-summary)
  - [Edge-Case Analysis: Three Target Scenarios](#edge-case-analysis-three-target-scenarios)
  - [Red Flags and Bugs](#red-flags-and-bugs)
  - [Suggestions for Reaching 100% Detection](#suggestions-for-reaching-100-detection)
- [Part 2: Face + Bib Search Training Roadmap](#part-2-face--bib-search-training-roadmap)
  - [Current State](#current-state)
  - [Phase-by-Phase Training Plan](#phase-by-phase-training-plan)
  - [Data Preparation Strategy](#data-preparation-strategy)
  - [Model Selection Recommendations](#model-selection-recommendations)
  - [Integration Strategy: Linking Face + Bib for High-Confidence Search](#integration-strategy-linking-face--bib-for-high-confidence-search)
  - [Handling Varying Lighting and Motion](#handling-varying-lighting-and-motion)

---

## Part 1: Blur Detection Technical Critique

### Current State Summary

The blur detection system uses two complementary approaches:

| System | File | Method | Accuracy | Always Available |
|---|---|---|---|---|
| Laplacian Detector | `src/ml/blur/detector.py` | Laplacian variance + FFT spectral analysis | Threshold-based (no trained model) | Yes |
| CNN Classifier | `src/ml/blur/classifier.py` | YOLOv8n-cls (4-class ONNX model) | 98.68% top-1 | Requires trained model |

**CNN classifier per-class accuracy (Round 3):**

| Class | Accuracy | Confusion |
|---|---|---|
| `sharp` | **100%** | Zero false positives |
| `defocused_object_portrait` | **99%** | 1% confused with `defocused_blurred` |
| `motion_blurred` | **99%** | 2% confused with `defocused_blurred` |
| `defocused_blurred` | **97%** | 1% to `defocused_object_portrait`, 1% to `motion_blurred` |

**Training dataset:** 5,294 images (1,799 sharp, 1,142 defocused_object_portrait, 1,300 defocused_blurred, 1,053 motion_blurred).

---

### Edge-Case Analysis: Three Target Scenarios

#### Scenario 1: Portrait Out-of-Focus (Subject Defocused, Background Sharp)

**Mapped to class:** `defocused_object_portrait`
**Current accuracy:** 99%
**Current risk level:** Low

**What works well:**
- 1,142 dedicated real-world training images for this exact scenario
- The training documentation has strict labeling rules that distinguish this from intentional bokeh
- Sharp class includes 1,475 intentional bokeh portraits, teaching the model that background blur alone is not a defect

**Remaining gap:**
- The 1% confusion with `defocused_blurred` likely occurs when the subject is only *slightly* out of focus — a subtle missed focus where the depth-of-field gradient is gentle rather than dramatic. These borderline cases are the hardest for a whole-image classifier because the visual difference between "slightly missed focus on subject" and "slightly soft everywhere" is minimal at 224x224 resolution.

**Recommendation:**
- Collect 50-100 images of *subtle* missed focus (where the subject is only slightly soft, not dramatically out of focus) and ensure they are labeled as `defocused_object_portrait`. These borderline images are what separate 99% from 100%.
- Consider increasing input resolution from 224 to 320 for this class — subtle focus differences are lost when a 4000px image is compressed to 224px.

---

#### Scenario 2: Global Out-of-Focus (Entire Image Soft)

**Mapped to class:** `defocused_blurred`
**Current accuracy:** 97%
**Current risk level:** Medium — this is the weakest class

**What works well:**
- 1,300 training images after data cleaning
- Documentation explicitly states that incidental sharp edges (feet, road markings, clothing seams) should NOT override the defocused classification
- The labeling rules correctly define that "no clearly sharp primary subject" = defocused

**Where the 3% failure occurs:**
The confusion matrix shows 1% leaking to `defocused_object_portrait` and 1% to `motion_blurred`, with another ~1% likely going to `sharp`. The probable failure modes are:

1. **Images with prominent sharp edges that trick the model.** An image where the overall frame is soft but a high-contrast element (e.g., a white lane marking against dark asphalt) retains apparent sharpness. The model may interpret this strong local edge as evidence of a sharp region, misclassifying as `sharp` or `defocused_object_portrait`.

2. **Images where defocus and motion blur coexist.** A slightly soft image with minor camera shake might show both defocus softness and faint directional streaking. The model can't easily decide whether this is primarily defocus or primarily motion.

3. **Mild global softness.** Images that are *slightly* out of focus everywhere (not dramatically soft) are the hardest to distinguish from sharp images, especially at 224x224 where compression already introduces softness.

**Recommendation:**
- This class needs the most attention to reach 100%. Focus on collecting "mild global defocus" images — frames that are clearly unusable as sharp photos but not dramatically blurry.
- Add 100-200 images where strong local edges coexist with overall softness (the false negative pattern described above).
- Consider the two-stage validation approach documented in the training plan: detect the primary subject first (via object detection), then evaluate sharpness on the subject region specifically. This would be the definitive fix for the "sharp edge in a soft frame" problem.

---

#### Scenario 3: Partial Motion Blur (Sharp Torso, Blurry Limbs)

**Mapped to class:** `motion_blurred`
**Current accuracy:** 99%
**Current risk level:** Medium-High for the *specific* partial motion blur scenario

**Critical finding: This is the scenario with the largest architectural gap.**

The current model classifies the *entire image* into one category at 224x224. For a marathon photo where a runner's torso is sharp but their swinging arms and legs show motion streaks, the model sees a mix of sharp and blurry regions. At 224x224, these localized motion blur patterns are small and may not dominate the overall feature map.

**Two distinct sub-cases:**

| Sub-case | What It Looks Like | Current Handling |
|---|---|---|
| Full-frame motion blur | Camera shake, everything streaked | Well-handled — 99% accuracy |
| Partial motion blur | Subject limbs blurry, torso/face sharp | Under-represented — success depends on training data composition |

The training data documentation refers to `motion_blurred` as "blur caused by camera shake or subject movement" but does not explicitly state whether partial motion blur images (sharp core, blurry extremities) are included or how they are labeled. If the 1,053 motion_blurred images are predominantly *full-frame* motion blur (camera shake), the model may not generalize well to *partial* motion blur.

**The labeling dilemma:** Under the current labeling rules, an image with a sharp face/torso and blurry arms could be argued as either:
- `motion_blurred` — because motion blur is present on part of the subject
- `sharp` — because the *primary subject* (face, bib) is sharp (per the portrait rules)

This ambiguity in the labeling rules is a potential source of both training confusion and production false negatives.

**Recommendation:**
1. **Clarify the labeling rule for partial motion blur.** Define explicitly: if the primary subject's face and bib are sharp but limbs show motion blur, is this `sharp` or `motion_blurred`? The answer depends on the business requirement — is the photo "usable" even with blurry limbs?
2. **If partial motion blur should be detected:** Add a dedicated sub-category or ensure the `motion_blurred` training set includes 200+ images of partial motion blur specifically. Train at 320x320 input size to preserve the spatial distinction between sharp and blurry regions within the same image.
3. **If partial motion blur should be ignored (photo is usable):** Ensure these images are explicitly labeled as `sharp` in the training data, and document this decision.

---

### Red Flags and Bugs

> **Update (March 2026):** Red Flags 1–4 have been resolved. See the "Resolution" note under each item.

#### RED FLAG 1: Laplacian Detector Cannot Handle Any of the Three Scenarios Reliably

**Severity:** High (architectural limitation, not a bug)
**File:** `src/ml/blur/detector.py`

The Laplacian detector computes a single variance value across the entire grayscale image:

```python
laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
is_blurry = laplacian_var < self.laplacian_threshold
```

This global metric fundamentally fails for spatially-varying blur:

| Scenario | Laplacian Behavior | Result |
|---|---|---|
| Portrait out-of-focus (sharp background) | Variance is moderate-high (sharp background contributes high edges) | **False negative** — reports as sharp |
| Global out-of-focus | Variance is uniformly low | Correct detection |
| Partial motion blur (sharp torso, blurry limbs) | Variance is moderate (sharp regions dominate) | **False negative** — reports as sharp |

The Laplacian detector is a good fast-path heuristic for obvious blur but cannot serve as the primary detector for the three scenarios. The CNN classifier must be the primary system for production use.

**Recommendation:** Document that `POST /blur/detect` (Laplacian-based) is a lightweight heuristic suitable for obvious blur detection, while `POST /blur/classify` (CNN-based) is the production-grade endpoint for the three target scenarios.

**Resolution:** RESOLVED. A clarifying docstring was added to `BlurDetector` in `detector.py` explicitly documenting that it is a fast coarse gate and directing users to `BlurClassifier` for fine-grained categorization.

---

#### RED FLAG 2: Augmentation Script Applies Gaussian Blur to Training Data

**Severity:** Medium (latent bug, currently not triggered)
**File:** `scripts/prepare_blur_dataset.py` (lines 93-95)

```python
if rng.random() < 0.2:
    ksize = rng.choice([3, 5])
    result = cv2.GaussianBlur(result, (ksize, ksize), 0)
```

The augmentation function applies random Gaussian blur to 20% of augmented images — including the `sharp` class. This would teach the model that some blurry images are "sharp", directly undermining the decision boundary.

**Currently safe:** All classes exceed 1,000 images and the `AUGMENTATION_TARGET` is 300, so the augmentation code path is never executed. However, if the dataset were ever restructured (e.g., splitting a class, removing images), this latent bug would activate and corrupt training.

**Recommendation:** Remove the Gaussian blur augmentation entirely, or at minimum exclude it for the `sharp` class. For blur classes, adding *more* blur is also undesirable since it distorts the blur signatures the model needs to learn.

**Resolution:** RESOLVED. The Gaussian blur augmentation step (step 6) was removed from `prepare_blur_dataset.py`. The augmentation pipeline now ends after Gaussian noise (step 5). The existing trained model was never affected since the augmentation code path was not triggered.

---

#### RED FLAG 3: CNN Classifier Ignores GPU Configuration

**Severity:** Low
**File:** `src/ml/blur/classifier.py` (line 56)

```python
self.session = ort.InferenceSession(
    self.model_path,
    providers=["CPUExecutionProvider"],
)
```

The `USE_GPU` setting from config is ignored. All other models (face embedder, bib detector) respect this setting and use `CUDAExecutionProvider` when GPU is available. The blur classifier always runs on CPU.

At 224x224 input and ~1.4M parameters, inference is fast enough on CPU (~45ms), but this is inconsistent and should be fixed for environments where GPU inference is expected.

**Recommendation:** Accept `use_gpu` as a constructor parameter (like `BibRecognizer` and `FaceEmbedder` do) and select providers accordingly.

**Resolution:** RESOLVED. `BlurClassifier.__init__` now accepts `use_gpu` and selects `CUDAExecutionProvider` when enabled, with automatic CPU fallback. The `ModelRegistry` and Celery `model_loader.py` both pass `settings.USE_GPU` to the classifier.

---

#### RED FLAG 4: No Confidence Floor on Blur Detection

**Severity:** Medium
**File:** `src/ml/blur/classifier.py` (line 160), `src/api/v1/blur.py` (line 186-210)

The targeted detection mode (`blur_type` parameter) returns `detected: true` whenever the selected blur type is the argmax prediction, regardless of confidence:

```python
detected = predicted_class == blur_type
```

An image that scores `defocused_blurred: 0.28, sharp: 0.27, motion_blurred: 0.25, defocused_object_portrait: 0.20` would be classified as `detected: true` for `defocused_blurred` with only 28% confidence. This is barely above random chance for a 4-class model (25%).

**Recommendation:** Add a configurable minimum confidence threshold (e.g., 0.5) below which the detection returns `detected: false` regardless of the argmax result. This prevents low-confidence false positives:

```python
MIN_DETECTION_CONFIDENCE = 0.5
detected = predicted_class == blur_type and confidence >= MIN_DETECTION_CONFIDENCE
```

**Resolution:** RESOLVED. `BlurClassifier` now accepts `min_detection_confidence` (default 0.5). `detect_blur_type()` enforces this floor. The threshold is configurable via `BLUR_DETECTION_MIN_CONFIDENCE` in `.env`.

---

#### RED FLAG 5: 224x224 Input Resolution Compresses Spatial Blur Patterns

**Severity:** Medium (for partial motion blur and subtle defocus)

The model resizes all input images to 224x224 before classification. A 4000x3000 marathon photo is compressed by ~18x in each dimension. At this resolution:

- Subtle focus differences between "sharp face" and "slightly soft face" are lost
- Localized motion blur on limbs (50-100px streaks in the original) becomes 3-5px streaks — potentially indistinguishable from compression artifacts
- The center-crop preprocessing drops 25-50% of the image when aspect ratios don't match, potentially cropping out the blurry regions entirely

**Recommendation:** Test training at 320x320 (the next standard size). This provides ~2x more spatial detail at the cost of ~2x inference time. The training plan already lists this as a hyperparameter to try if accuracy plateaus. Given the 97% accuracy on `defocused_blurred` and the partial motion blur challenge, this upgrade is warranted now.

---

### Suggestions for Reaching 100% Detection

#### Tier 1: Achievable with Current Architecture (target: 99.5%+)

| Action | Effort | Expected Impact |
|---|---|---|
| Increase input resolution to 320x320 | Low — change `imgsz=320` in training and `input_size=320` in classifier | +0.5-1% on `defocused_blurred` and partial motion blur |
| Add 100 "mild global defocus" training images | Medium — collect and label | +0.5-1% on `defocused_blurred` |
| Add 100 "partial motion blur" training images | Medium — collect and label | Closes the partial motion blur gap |
| Add confidence floor (0.5) to detection | Low — code change | Eliminates low-confidence false positives |
| Reduce label smoothing from 0.1 to 0.05 | Low — hyperparameter change | Sharper decision boundaries with larger dataset |
| Reduce dropout from 0.3 to 0.2 | Low — hyperparameter change | Less regularization now that data is sufficient |

#### Tier 2: Two-Stage Pipeline (target: 99.9%+)

If Tier 1 improvements plateau below 99.5% on any class, implement the two-stage approach already outlined in the training plan:

**Stage 1:** Run a person detector (YOLOv8n, the same model being trained for face+bib detection) to locate the primary subject.

**Stage 2:** Evaluate blur specifically on the subject crop vs. the background crop. Compare local Laplacian variance or CNN classification on each region.

```
Input Image
    │
    ├── Stage 1: Person/Subject Detection (YOLOv8n)
    │       → Subject bounding box
    │
    ├── Stage 2a: Blur classify on subject crop
    │       → "Is the subject itself blurry?"
    │
    ├── Stage 2b: Blur classify on full image
    │       → "Is the overall image blurry?"
    │
    └── Decision Logic:
        - Subject crop is sharp + full image is blurry
            → defocused_object_portrait (wrong focus plane)
        - Subject crop is blurry + full image is blurry
            → defocused_blurred (global softness)
        - Subject crop has motion streaks
            → motion_blurred
        - Subject crop is sharp
            → sharp (background blur is intentional)
```

This approach explicitly solves all three target scenarios because it evaluates subject-level sharpness rather than image-level statistics. It also reuses the face+bib detection model (Stage 1), so no additional model training is needed for the subject localization step.

**Estimated additional latency:** ~40-80ms for the person detection pass + a second classification pass on the crop = ~80-160ms total (vs. current ~45ms). Acceptable for a 100% accuracy target.

#### Tier 3: Ensemble or Larger Model (target: 100%)

If both Tier 1 and Tier 2 fail to reach the target:

| Approach | Description | Trade-off |
|---|---|---|
| Model ensemble | Run both YOLOv8n-cls and a second classifier (e.g., EfficientNet-B0), accept blur only if both agree | ~2x inference cost, eliminates single-model blind spots |
| Larger backbone | Replace YOLOv8n-cls (1.4M params) with YOLOv8s-cls (5.4M params) or EfficientNet-B2 | ~3-4x inference cost, more capacity for subtle patterns |
| Multi-scale input | Run the classifier at both 224x224 and 448x448, combine predictions | Captures both global structure and local detail |

These are last-resort options. The two-stage pipeline (Tier 2) should be attempted first since it addresses the root cause (spatial localization) rather than adding brute-force capacity.

---

## Part 2: Face + Bib Search Training Roadmap

### Current State

| Component | Status | Details |
|---|---|---|
| Training images | **Done** | 1,638 event photos collected |
| Auto-annotation | **Done** | 3,316 faces + 1,863 bibs = 5,179 annotations |
| Dataset split | **Done** | 1,315 train / 323 val (80/20) |
| Combined detector script | **Ready** | `scripts/train_face_bib_detector.py` |
| Combined detector training | **Not started** | Next immediate step |
| Face embedding model | **Not started** | Requires face crops from trained detector |
| Bib OCR fine-tuning | **Not started** | Requires bib crops from trained detector |
| ONNX export script | **Ready** | `scripts/export_face_bib_detector.py` — exports to ONNX with validation |
| Pre-training fixes | **Done** | Bib character filter (supports `-`, `_`, letters), face enrollment confidence gate, batch worker threshold from config |
| Integration into API | **Partially done** | Endpoints exist with pre-trained models (InsightFace + PaddleOCR) |

### Phase-by-Phase Training Plan

#### Phase A: Combined Face+Bib Detector (Weeks 1-2)

**Goal:** Train a single YOLOv8n model that detects both faces and bibs in one inference pass.

**Step 1: Annotation Quality Audit (Day 1)**

Before training, verify annotation quality on a random sample:

```bash
python scripts/auto_annotate_face_bib.py --visualize 50
```

Manually review the 50 visualized images for:

| Check | What to Look For | Action if Found |
|---|---|---|
| Missed faces | Faces visible but no green box | Lower `FACE_CONFIDENCE_THRESHOLD` from 0.5 to 0.3, re-annotate |
| Missed bibs | Bibs visible but no red box | Check if bib has < 2 digits (adjust `BIB_MIN_DIGITS`), or OCR failed at 800px resize |
| False positive bibs | Red box on non-bib text (jersey text, sponsor logos) | Add filtering logic to exclude common false positive text patterns |
| Oversized bib boxes | Red box extends well beyond the bib card | Reduce `BIB_BOX_EXPAND_RATIO` from 0.6 to 0.4 |
| Undersized bib boxes | Red box too tight, cuts off bib edges | Increase `BIB_BOX_EXPAND_RATIO` to 0.7 |

**Step 2: Train the Combined Detector (Days 2-5)**

```bash
python scripts/train_face_bib_detector.py
```

Expected training time: ~4-8 hours on CPU, ~1-2 hours on GPU.

Hyperparameter recommendations based on the dataset size (1,638 images, 5,179 annotations):

| Parameter | Current Value | Recommended | Rationale |
|---|---|---|---|
| `imgsz` | 640 | 640 | Standard for detection; faces and bibs are medium-sized objects |
| `batch` | 8 | 8 (CPU) / 16 (GPU) | Increase if GPU memory allows |
| `epochs` | 100 | 100 | Early stopping (patience=20) handles convergence |
| `mosaic` | 1.0 | 1.0 | Critical for learning multi-object detection |
| `mixup` | 0.1 | 0.15 | Slightly more mixup helps with occlusion robustness |

**Target metrics:**

| Metric | Target | Notes |
|---|---|---|
| mAP50 (face) | > 0.85 | Faces are relatively easy to detect |
| mAP50 (bib) | > 0.75 | Bibs are harder — variable sizes, occlusion, angles |
| mAP50 (overall) | > 0.80 | Combined across both classes |
| Inference time | < 30ms (GPU), < 150ms (CPU) | YOLOv8n is designed for this |

**Step 3: Export and Validate (Day 6)**

Write `scripts/export_face_bib_detector.py` (not yet written) following the same pattern as `export_blur_classifier.py`:

```bash
python scripts/export_face_bib_detector.py
# -> models/face_bib_detector/face_bib_detector.onnx
# -> models/face_bib_detector/class_names.json
```

Validate on held-out images not in the training set.

---

#### Phase B: Face Embedding Fine-Tuning (Weeks 3-4)

**Goal:** Produce 512-dim face embeddings that reliably match the same runner across different photos despite lighting, angle, and expression changes.

**Step 1: Generate Face Crops (Day 1)**

Use the trained Phase A detector to crop faces from all 1,638 training images:

```
For each image:
  1. Run detector → face bounding boxes
  2. Expand each box by 20% (capture forehead, chin)
  3. Crop and resize to 112x112 (ArcFace standard)
  4. Save to Training-Images/face_embeddings/{person_id}/
```

**Step 2: Identity Labeling (Days 2-7) — Manual step**

This is the most labor-intensive step. Each cropped face must be assigned to a person identity. Two approaches:

| Approach | Effort | Accuracy |
|---|---|---|
| **Manual clustering** | High — human reviews each crop and groups by person | Highest accuracy |
| **Semi-automated** | Medium — use existing InsightFace embeddings to cluster similar faces, then human verifies clusters | Good accuracy, much faster |

Recommended: Semi-automated approach.

```
1. Extract embeddings for all face crops using InsightFace (already in the codebase)
2. Cluster embeddings using DBSCAN or agglomerative clustering (cosine distance)
3. Save clusters as directories: face_embeddings/person_001/, person_002/, etc.
4. Human reviews each cluster:
   - Split clusters that contain multiple people
   - Merge clusters that are the same person
5. Result: clean identity-labeled face dataset
```

**Minimum requirement:** At least 5-10 images per person identity, across at least 50-100 unique identities. More identities = better generalization.

**Step 3: Fine-Tune Embedding Model (Days 8-12)**

Two options:

| Option | Model | Params | Embedding Dim | Speed |
|---|---|---|---|---|
| **A: MobileFaceNet** | Lightweight mobile architecture | ~1M | 128 | Fast (~5ms GPU) |
| **B: ArcFace-R50** | ResNet-50 backbone with ArcFace loss | ~25M | 512 | Moderate (~15ms GPU) |

**Recommendation: Start with the existing InsightFace ArcFace model (already in the codebase) as the baseline.** Fine-tune it on your event-specific face crops if the baseline accuracy is insufficient. Event photography faces are different from the datasets these models were trained on (outdoor, sunglasses, hats, sweat, motion), so fine-tuning may yield significant improvements.

Training with ArcFace loss:

```python
# ArcFace loss pushes embeddings of the same person together
# and embeddings of different people apart on a hypersphere
loss = ArcFaceLoss(
    in_features=512,
    out_features=num_identities,
    s=64.0,    # scale factor
    m=0.50,    # angular margin
)
```

**Accuracy targets:**

| Metric | Target | How to Measure |
|---|---|---|
| True Positive Rate @ FAR=0.01 | > 0.95 | 95% of same-person pairs are correctly matched at 1% false accept rate |
| Rank-1 accuracy | > 0.90 | Given a probe face, the correct identity is the top match 90%+ of the time |

---

#### Phase C: Bib OCR Fine-Tuning (Weeks 3-4, parallel with Phase B)

**Goal:** Accurately read bib numbers from cropped bib regions, handling varied fonts, angles, and partial occlusion.

**Step 1: Generate Bib Crops (Day 1)**

Use the trained Phase A detector to crop bibs from training images.

**Step 2: Ground Truth Labeling (Days 2-5) — Manual step**

Each cropped bib needs the correct number as a label:

```
bib_crops/
  crop_0001.jpg  ->  "1234"
  crop_0002.jpg  ->  "567"
  crop_0003.jpg  ->  "A-892"
  crop_0004.jpg  ->  ""        (unreadable)
```

This can be partially automated:
1. Run PaddleOCR (already in the codebase) on each crop
2. Save the predicted text alongside the image
3. Human corrects predictions that are wrong

**Step 3: Evaluate Baseline OCR (Day 6)**

Before fine-tuning, measure PaddleOCR's accuracy on the labeled bib crops:

```
For each labeled crop:
  predicted = paddleocr.recognize(crop)
  correct = (predicted == ground_truth)
```

If PaddleOCR achieves > 95% accuracy on the labeled crops, fine-tuning may not be necessary — focus effort on improving the detector instead.

**Step 4: Fine-Tune OCR if Needed (Days 7-12)**

If baseline OCR accuracy is < 95%, consider:

| Approach | When to Use | Effort |
|---|---|---|
| **PaddleOCR fine-tuning** | Accuracy is 80-95% — the model is close but misreads certain fonts or angles | Medium |
| **Custom CRNN** | Accuracy is < 80% or bib format is very specialized | High |
| **Synthetic data augmentation** | Not enough labeled bib crops | Medium |

For PaddleOCR fine-tuning, use the [PaddleOCR training docs](https://github.com/PaddlePaddle/PaddleOCR) to fine-tune the recognition model on your labeled bib crops.

For synthetic data augmentation (if you need more training data):

```
Generate synthetic bib images:
  1. Random 1-5 digit numbers on varied backgrounds
  2. Apply random rotation (-15° to +15°)
  3. Apply random perspective distortion
  4. Vary font size, color, and weight
  5. Add noise, compression artifacts, slight blur
  6. Result: 10,000+ labeled training samples for free
```

---

#### Phase D: Integration and End-to-End Testing (Week 5)

**Goal:** Wire the three trained models into the EventAI API and validate the full pipeline.

**Step 1: Update ML Wrappers**

Replace the current pre-trained model usage with the custom-trained models:

| Current Code | Change To |
|---|---|
| `FaceEmbedder` uses InsightFace `buffalo_l` directly | Load custom-trained face+bib YOLO detector for detection, keep InsightFace for embedding (or use fine-tuned embedding model) |
| `BibDetector` loads `yolov8n_bib.onnx` (single class) | Load combined `face_bib_detector.onnx` (2 classes), filter for bib class |
| `BibRecognizer` uses PaddleOCR directly | Keep PaddleOCR or replace with fine-tuned OCR model |

**Step 2: Shared Detection Architecture**

The combined detector enables a key optimization — a single inference pass serves both face and bib endpoints:

```python
class CombinedDetector:
    """Runs face+bib detection in one pass, caches results."""

    def detect_all(self, image) -> dict:
        """Returns {'faces': [...], 'bibs': [...]}"""
        results = self.model(image)
        faces = [r for r in results if r.class_id == 0]
        bibs = [r for r in results if r.class_id == 1]
        return {"faces": faces, "bibs": bibs}
```

When a future `POST /api/v1/runner/identify` endpoint needs both face and bib from the same image, it runs detection once instead of twice.

**Step 3: End-to-End Test Matrix**

| Test Case | Expected Result | Validates |
|---|---|---|
| Single runner, clear face and bib | Face detected + matched, bib number read correctly | Happy path |
| Group of runners | Multiple faces and bibs detected, correctly paired | Multi-object handling |
| Runner with obscured face (sunglasses, turned) | Face detected with lower confidence or not detected | Graceful degradation |
| Runner with partially covered bib (arm across chest) | Bib detected if > 50% visible, OCR reads partial number | Occlusion handling |
| Backlit / harsh shadow photo | Detection works despite lighting extremes | Lighting robustness |
| Motion-blurred photo (runner in motion) | Detection may fail — this is expected for severely blurred photos | Known limitation |
| Photo with no runners | Zero detections, no errors | Edge case |

---

### Data Preparation Strategy

#### Labeling Strategy for Overlapping Bibs

Marathon photos frequently have runners overlapping, with one runner's arm or body partially covering another's bib. Handle this by:

1. **Annotate what's visible.** If a bib is > 30% visible, annotate it. If < 30% visible, do not annotate — the model should not be trained to detect barely-visible bibs.

2. **Bib box should cover the visible portion only.** Do not extend the bounding box into the occluded region. The detector learns to find visible bib area, not to guess where occluded bibs might be.

3. **OCR ground truth for partial bibs.** If only "12__" is visible from bib "1234", label the ground truth as "12" (what's readable), not "1234". The system should return what it can confidently read.

#### Labeling Strategy for Obscured Faces

1. **Profile faces (> 45° turn):** Annotate as face. InsightFace handles profiles well.

2. **Faces with sunglasses/hats:** Annotate as face. These are extremely common in outdoor events and the model must detect them.

3. **Fully turned away (> 90°, back of head):** Do NOT annotate. The model should not learn that the back of a head is a "face".

4. **Faces < 20x20 pixels in the image:** Do NOT annotate. Too small for meaningful embedding extraction. Set a minimum annotation size:

```python
MIN_FACE_SIZE = 20  # pixels — faces smaller than this are ignored during annotation
```

#### Lighting and Environmental Diversity

Ensure the training dataset includes:

| Condition | Minimum Representation | Why |
|---|---|---|
| Direct sunlight | 25% of images | Most common outdoor condition |
| Overcast / cloudy | 20% | Different color temperature, softer shadows |
| Backlit (sun behind runner) | 15% | Challenging for both face detection and bib OCR |
| Early morning / golden hour | 10% | Warm light, long shadows |
| Shade / under trees | 10% | Dappled light, partial shadows on face/bib |
| Indoor (finish line tents) | 10% | Artificial lighting, mixed sources |
| Rain / wet conditions | 5% | Reflections, water on bib surface |
| Night / artificial light | 5% | Flash photography, high ISO noise |

If the current 1,638 images are predominantly one lighting condition, consider sourcing additional images from different events or times of day.

---

### Model Selection Recommendations

#### Face+Bib Detection: YOLOv8n (Already Selected — Confirmed Correct)

| Factor | Assessment |
|---|---|
| Speed | ~5ms GPU, ~40ms CPU — fast enough for batch processing |
| Accuracy on small objects | Good with 640x640 input, faces and bibs are medium-sized |
| Training data requirement | Works well with ~1,500 images (transfer learning from COCO) |
| ONNX export | Native support via Ultralytics |

The choice of YOLOv8n for the combined detector is well-justified. No change recommended.

**If accuracy is insufficient after training:** Consider upgrading to YOLOv8s (11.2M params vs. 3.2M for nano) which provides better small-object detection at ~2x the inference cost.

#### Face Embedding: InsightFace ArcFace (Recommended Baseline)

The existing InsightFace integration (`buffalo_l` model bundle) includes both RetinaFace detection and ArcFace embedding. For the initial integration:

1. **Use the custom YOLOv8n detector for face detection** (more accurate on event photos since it's trained on your data)
2. **Use the existing ArcFace model for embedding extraction** (pre-trained on millions of faces, strong out-of-the-box)
3. **Fine-tune the embedding model only if matching accuracy is insufficient** (< 90% rank-1 on your test set)

If fine-tuning is needed, consider **MobileFaceNet** for a lightweight alternative (1M params, 128-dim embeddings, ~5ms inference) or keep ArcFace-R50 for maximum accuracy.

#### Bib OCR: PaddleOCR PP-OCRv5 (Recommended Baseline)

PaddleOCR is already integrated and handles "text in the wild" well. For bib numbers specifically:

1. **Start with the existing PaddleOCR integration** — it already achieves 98%+ accuracy on clean synthetic numbers
2. **Evaluate on real bib crops** from your trained detector — real-world accuracy will be lower
3. **Fine-tune only if accuracy < 90%** on real bib crops

**Alternative for difficult cases:** If PaddleOCR struggles with specific bib fonts or heavily angled text, consider training a lightweight CRNN (Convolutional Recurrent Neural Network) specifically for digit recognition on bib-like crops. CRNNs are smaller and faster than full OCR models since they only need to recognize 0-9 + a few letters.

---

### Integration Strategy: Linking Face + Bib for High-Confidence Search

The most powerful feature is combining face recognition with bib number OCR to provide redundant identification. Here's the recommended integration architecture:

#### Runner Identification Pipeline

```
Event Photo
    │
    ▼
Combined Detector (YOLOv8n)
    │
    ├── Face crops + bounding boxes
    │       │
    │       ▼
    │   Face Embedder (ArcFace)
    │       │
    │       ▼
    │   Face Match against DB
    │       → person_id, similarity score
    │
    ├── Bib crops + bounding boxes
    │       │
    │       ▼
    │   Bib OCR (PaddleOCR)
    │       │
    │       ▼
    │   Bib Number Lookup in DB
    │       → runner_id, bib_number
    │
    ▼
Spatial Association
    │
    ├── Match face crops to nearby bib crops
    │   (by vertical proximity — face is above bib on same runner)
    │
    ▼
Fused Result
    → runner_id (from bib), person_id (from face), confidence
```

#### Spatial Association Logic

Faces and bibs on the same runner have a predictable spatial relationship:

```python
def associate_face_bib(faces, bibs, image_height):
    """Match each face to its closest bib below it."""
    associations = []
    for face in faces:
        face_center_x = (face.x1 + face.x2) / 2
        face_bottom_y = face.y2

        best_bib = None
        best_distance = float('inf')

        for bib in bibs:
            bib_center_x = (bib.x1 + bib.x2) / 2
            bib_top_y = bib.y1

            # Bib must be below face
            if bib_top_y < face_bottom_y:
                continue

            # Horizontal alignment check (within 2x face width)
            horizontal_offset = abs(face_center_x - bib_center_x)
            face_width = face.x2 - face.x1
            if horizontal_offset > face_width * 2:
                continue

            # Vertical distance (bib should be within ~30% of image height below face)
            vertical_dist = bib_top_y - face_bottom_y
            if vertical_dist > image_height * 0.3:
                continue

            distance = (horizontal_offset ** 2 + vertical_dist ** 2) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_bib = bib

        associations.append({"face": face, "bib": best_bib})
    return associations
```

#### Confidence Fusion

When both face and bib identify the same runner, confidence is high. When they disagree, the system should flag it:

| Face Match | Bib Match | Result | Confidence |
|---|---|---|---|
| Person A (similarity 0.85) | Bib #1234 → Runner A | **Both agree** | Very High |
| Person A (similarity 0.85) | Bib #1234 → Runner B | **Conflict** — flag for human review | Low |
| Person A (similarity 0.85) | No bib detected | **Face only** — use face match | Medium |
| No face detected | Bib #1234 → Runner A | **Bib only** — use bib match | Medium-High |
| No face detected | No bib detected | **No identification** | None |

#### Planned API Endpoint

```
POST /api/v1/runner/identify
  - Input: event photo
  - Output: list of identified runners in the photo
  - Each runner: {
      bib_number: "1234",
      person_id: "uuid",
      person_name: "John Doe",
      face_confidence: 0.85,
      bib_confidence: 0.98,
      identification_method: "face+bib" | "face_only" | "bib_only",
      bbox: { face: {...}, bib: {...} }
    }
```

---

### Handling Varying Lighting and Motion

#### Lighting Robustness Strategies

**During Training:**

| Augmentation | Purpose | Current Setting | Recommended |
|---|---|---|---|
| HSV hue | Color temperature variation | 0.015 | 0.02 (slightly more variation) |
| HSV saturation | Handle washed-out or oversaturated images | 0.4 | 0.5 (outdoor conditions vary widely) |
| HSV value | Brightness variation (backlit, shade) | 0.3 | 0.4 (event photos have extreme brightness range) |

**During Inference:**

- No preprocessing needed — YOLOv8 and InsightFace handle lighting variation well out of the box
- For extreme backlit situations where faces are in shadow, consider histogram equalization on face crops before embedding extraction (improves ArcFace accuracy in low-contrast conditions)

#### Motion Robustness Strategies

Athletes in motion create two challenges:

1. **Motion blur on the subject** — The face/bib detection model may fail if blur is severe. This is expected and acceptable (the blur detection module should flag these images first).

2. **Unusual body poses** — Mid-stride runners have non-standard body positions. Arms may cover bibs, faces may be contorted.

**Recommendations:**

| Strategy | Implementation |
|---|---|
| Augmentation: rotation ±10° | Already configured — handles tilted runners |
| Augmentation: mosaic | Already configured — teaches detection of partially visible objects |
| Blur-first pipeline | Run blur detection before face+bib detection. If the image is flagged as `motion_blurred`, skip face+bib detection and report "image too blurry for identification" |
| Lower detection confidence | For motion-heavy photos, allow lower detection confidence (0.3 vs. 0.5) and let the embedding/OCR step confirm quality |
| Multi-frame matching | For burst photography, if one frame is blurry, check adjacent frames. Not needed for v1 but valuable for future gallery processing |

---

### Summary: Complete Training Timeline

| Week | Phase | Key Deliverable |
|---|---|---|
| **Week 1** | Annotation audit + Combined detector training | Trained YOLOv8n face+bib ONNX model |
| **Week 2** | Detector validation + Face crop generation | Validated mAP50 > 0.80, face crops extracted |
| **Week 3** | Face identity labeling + Bib ground truth labeling | Labeled face clusters + labeled bib crops |
| **Week 4** | Face embedding fine-tuning + Bib OCR evaluation | Embedding model ready, OCR accuracy measured |
| **Week 5** | API integration + End-to-end testing | Full pipeline functional in staging |
| **Week 6** | Edge case testing + Performance benchmarks | Production-ready sign-off |

**Critical path:** The manual labeling steps (face identity clustering in Week 3, bib ground truth in Week 3) are the bottleneck. Everything else can be automated. Plan for 2-3 days of focused human labeling effort.
