# Python + C++ Integration

## Why C++?

Python is great for writing application logic but slow for tight loops over raw data. Not everything needs C++ though. The rule is:

**Only optimize the inner loops that process raw pixel data or compute distances over large arrays thousands of times per request.**

Most of the "heavy" work (model inference) is already in C++ under the hood via OpenCV, ONNX Runtime, and PaddleOCR. The C++ extensions here target the gaps between those libraries where Python overhead adds up.

## What Gets Accelerated

| Component | What Python Does | What C++ Does Better | Priority |
|---|---|---|---|
| Batch cosine similarity | `numpy.dot` for one query is fast, but 1-vs-50K with filtering needs temp arrays | Fused loop: compute distance + threshold + top-K in one pass, no temp arrays | **High** |
| Batch Laplacian variance | Calls `cv2.Laplacian` in a Python loop for N images (N round-trips to C++) | Single C++ call processes all N images, eliminates per-image Python overhead | Medium |
| Image preprocessing | Three separate calls: `cv2.resize` + `cv2.cvtColor` + normalize | Single function: resize + color convert + normalize + transpose in one memory pass | Medium |
| FFT blur metric | `numpy.fft.fft2` is already backed by FFTW | Marginal benefit | Low (skip) |

## How It Works

### Binding Technology: pybind11

pybind11 is a C++ library that creates Python modules from C++ code. It has first-class support for NumPy arrays -- you can pass a Python NumPy array to C++ and access the underlying memory directly, with zero copying.

### File Structure

```
src/cpp/
├── CMakeLists.txt          # Build instructions
├── include/                # Header files (.h)
│   ├── blur_ops.h
│   ├── distance_ops.h
│   └── preprocess_ops.h
├── src/                    # Implementation files (.cpp)
│   ├── blur_ops.cpp        # Batch Laplacian kernel
│   ├── distance_ops.cpp    # Batch cosine similarity + top-K
│   ├── preprocess_ops.cpp  # Fused image preprocessing
│   └── bindings.cpp        # pybind11 module definition (glue)
└── tests/
    ├── test_blur_ops.cpp
    └── test_distance_ops.cpp
```

### Example: Batch Cosine Similarity

**The problem**: You have a query face embedding (512 floats) and a database of 50,000 embeddings. You need the top 10 matches above a threshold.

**Pure Python/NumPy approach**:
```python
similarities = database @ query          # Creates a 50,000-element temp array
mask = similarities >= threshold         # Creates another temp array
valid_indices = np.where(mask)[0]        # Another temp array
valid_scores = similarities[valid_indices]  # Another temp array
top_order = np.argsort(valid_scores)[::-1][:top_k]  # Sort everything
```
This creates 4-5 temporary arrays and sorts all valid results even though we only need 10.

**C++ approach**:
```cpp
// Single pass: compute similarity, check threshold, maintain top-K heap
// No temporary arrays. Stops early when possible.
for (int i = 0; i < n_database; i++) {
    float sim = dot_product(query, database[i], 512);
    if (sim >= threshold) {
        min_heap.push({i, sim});
        if (min_heap.size() > top_k) min_heap.pop();
    }
}
```
One pass through memory, no allocations, partial sort via heap.

### The Graceful Fallback Pattern

Every Python file that uses C++ has this pattern:

```python
try:
    from _eventai_cpp import batch_cosine_topk
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

def find_matches(query, database, threshold=0.4, top_k=10):
    if _HAS_CPP:
        # Fast C++ path
        result = batch_cosine_topk(query, database, threshold, top_k)
        return [{"index": i, "score": s} for i, s in zip(result.indices, result.scores)]

    # Pure NumPy fallback (always works)
    similarities = database @ query
    mask = similarities >= threshold
    # ... rest of NumPy implementation ...
```

This means:
- **The app always runs**, even without compiled C++ code
- During development, you use the NumPy path (no C++ compiler needed)
- In production Docker builds, C++ is compiled and used automatically
- You can benchmark both paths with `python scripts/benchmark.py`

## Building the C++ Extension

### In Docker (automatic)
The Dockerfile has a build stage that compiles C++ before creating the runtime image. You don't need to do anything.

### Locally (manual)
```bash
# Install build dependencies
pip install -e ".[cpp]"

# This triggers scikit-build-core which:
# 1. Runs CMake to configure the build
# 2. Compiles C++ with pybind11
# 3. Places the compiled module where Python can import it
```

### Requirements
- **Linux**: GCC 10+ or Clang 12+
- **Windows**: MSVC (Visual Studio Build Tools)
- **macOS**: Clang (Xcode Command Line Tools)
- **All**: CMake 3.28+, OpenCV development headers

## When to Add C++

Don't add C++ until Phase 6. Get everything working in pure Python first. Then:

1. Run `python scripts/benchmark.py` to establish Python baselines
2. Implement the C++ versions
3. Run benchmarks again to measure the improvement
4. If the improvement is <2x, it's probably not worth the maintenance cost

Expected improvements:
- Batch cosine similarity: **5-10x** on large databases (>10K embeddings)
- Batch Laplacian: **2-3x** on batch sizes >50 images
- Fused preprocessing: **2-4x** (eliminates intermediate array copies)
