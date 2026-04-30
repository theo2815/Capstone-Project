# Python + C++ Integration

## Why C++?

Python is great for writing application logic but slow for tight loops over raw data. Not everything needs C++ though. The rule is:

**Only optimize the inner loops that process raw pixel data or compute distances over large arrays thousands of times per request.**

Most of the "heavy" work (model inference) is already in C++ under the hood via OpenCV, ONNX Runtime, and PaddleOCR. The C++ extensions here target the gaps between those libraries where Python overhead adds up.

## What Gets Accelerated

| Component | Function | Observed speedup | Notes |
|---|---|---|---|
| Laplacian variance (single image) | `laplacian_variance` | ~5.4× | Single-pass sum + sum_sq kernel; biggest win |
| Batch cosine top-K | `batch_cosine_topk` | 1.8–2.8× (N ≤ ~5K) | GIL release + partial_sort; NumPy BLAS wins on very large N |
| FFT high-frequency ratio | `fft_hf_ratio` | ~1.2× | Radix-2 Cooley-Tukey; ~parity with NumPy FFTPACK |
| BGR → gray, resize, classifier preprocess | `bgr_to_gray`, `resize_gray`, `classify_preprocess` | neutral to slower | OpenCV/NumPy already use hand-tuned SIMD; kept for the GIL-release benefit inside threaded pipelines |

Numbers are from `benchmarks/bench_cpp_vs_python.py` on MSVC 19.50 / Windows / AVX2 / Python 3.12. They will differ on Linux-glibc with GCC.

**Key non-speed benefit:** every C++ function releases the GIL via `py::gil_scoped_release`, so multiple Python threads can overlap computation. This matters inside the streaming and Celery batch pipelines, which use thread pools for decode + inference.

## How It Works

### Binding Technology: pybind11

pybind11 is a C++ library that creates Python modules from C++ code. It has first-class support for NumPy arrays -- you can pass a Python NumPy array to C++ and access the underlying memory directly, with zero copying.

### File Structure

```
ai-api/
├── CMakeLists.txt              # Build instructions (C++17, pybind11, AVX2)
├── build_cpp.py                # Build script (auto-detects MSVC on Windows)
│
└── src/cpp/                    # All C++ source files (flat layout)
    ├── bindings.cpp            # pybind11 module definition (glue)
    ├── blur_ops.h              # Blur detection headers
    ├── blur_ops.cpp            # Batch Laplacian variance + FFT HF ratio
    ├── face_ops.h              # Face matching headers
    ├── face_ops.cpp            # Batch cosine similarity + top-K (AVX2)
    ├── preprocess_ops.h        # Image preprocessing headers
    └── preprocess_ops.cpp      # Fused bgr_to_gray + resize_gray (bilinear)
```

C++ tests are Python-based: `tests/test_cpp_extension.py` (31 tests verifying numerical parity with the NumPy fallbacks).

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
    from _quickpitik_cpp import batch_cosine_topk
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
# Option A — via the pyproject extras (uses scikit-build-core)
pip install -e ".[cpp]"

# Option B — direct CMake build (helpful on Windows when pip misdetects MSVC)
python build_cpp.py
```

Both paths produce `_quickpitik_cpp.<tag>.pyd` (Windows) or `.so` (Linux/macOS) at the project root where Python can `import _quickpitik_cpp`.

### Requirements
- **Linux**: GCC 10+ or Clang 12+
- **Windows**: MSVC (Visual Studio Build Tools 2022+) — `build_cpp.py` auto-detects `vcvarsall.bat`
- **macOS**: Clang (Xcode Command Line Tools)
- **All**: CMake 3.18+, Ninja (pulled in by the `[cpp]` extra)

OpenCV development headers are **not** required — the extension only uses its own sources plus pybind11's NumPy bindings.

## When to Add Another C++ Function

The rule: profile first, then decide. New C++ is worth adding only when a pure Python version falls behind on a hot path.

1. Run `python benchmarks/bench_cpp_vs_python.py` or `python scripts/benchmark.py` to confirm the Python baseline.
2. Prototype in C++, keeping the NumPy path intact.
3. Re-run benchmarks.
4. If the speedup is below ~2× or the new code can't release the GIL, don't merge it — the maintenance cost isn't worth it.

The existing extension shows the ceiling clearly: NumPy-backed BLAS and OpenCV's SIMD paths are very fast, so naive C++ loops lose on pre-processing and large-N matmul-like workloads. Wins come from fusing passes (single-pass Laplacian variance) or from algorithms where NumPy creates large temp arrays (cosine top-K).
