#include "face_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eventai {

float cosine_similarity(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();

    if (buf_a.shape(0) != buf_b.shape(0)) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    const py::ssize_t n = buf_a.shape(0);
    const float* pa = buf_a.data(0);
    const float* pb = buf_b.data(0);

    float dot = 0.0f;
    // SIMD-friendly loop: compiler auto-vectorizes with /O2 + /arch:AVX2
    for (py::ssize_t i = 0; i < n; ++i) {
        dot += pa[i] * pb[i];
    }
    return dot;
}

TopKResult batch_cosine_topk(py::array_t<float> query,
                             py::array_t<float> database, float threshold,
                             int top_k) {
    auto q = query.unchecked<1>();
    auto db = database.unchecked<2>();

    const py::ssize_t N = db.shape(0);
    const py::ssize_t D = db.shape(1);

    if (q.shape(0) != D) {
        throw std::invalid_argument(
            "Query dimension must match database column dimension");
    }

    if (top_k <= 0) {
        return TopKResult{{}, {}};
    }

    const float* qp = q.data(0);

    // Compute all similarities and filter by threshold.
    // Use a flat vector of (score, index) pairs.
    struct ScoreIdx {
        float score;
        int64_t index;
    };
    std::vector<ScoreIdx> candidates;
    candidates.reserve(std::min(N, static_cast<py::ssize_t>(top_k * 4)));

    {
        // Release the GIL for the heavy computation loop
        py::gil_scoped_release release;

        for (py::ssize_t i = 0; i < N; ++i) {
            const float* row = db.data(i, 0);
            float dot = 0.0f;
            for (py::ssize_t j = 0; j < D; ++j) {
                dot += qp[j] * row[j];
            }
            if (dot >= threshold) {
                candidates.push_back({dot, i});
            }
        }

        // partial_sort for O(N + K log K) instead of full O(N log N)
        const size_t k = std::min(static_cast<size_t>(top_k),
                                  candidates.size());
        if (k > 0 && k < candidates.size()) {
            std::partial_sort(candidates.begin(), candidates.begin() + k,
                              candidates.end(),
                              [](const ScoreIdx& a, const ScoreIdx& b) {
                                  return a.score > b.score;
                              });
        } else if (k > 0) {
            std::sort(candidates.begin(), candidates.end(),
                      [](const ScoreIdx& a, const ScoreIdx& b) {
                          return a.score > b.score;
                      });
        }
    }

    const size_t result_count =
        std::min(static_cast<size_t>(top_k), candidates.size());
    TopKResult result;
    result.indices.reserve(result_count);
    result.scores.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        result.indices.push_back(candidates[i].index);
        result.scores.push_back(candidates[i].score);
    }
    return result;
}

}  // namespace eventai
