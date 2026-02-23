#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace eventai {

struct TopKResult {
    std::vector<int64_t> indices;
    std::vector<float> scores;
};

// Cosine similarity between two L2-normalized float32 vectors.
float cosine_similarity(py::array_t<float> a, py::array_t<float> b);

// Batch: query (D,) vs database (N, D), filter by threshold, return top-K.
TopKResult batch_cosine_topk(py::array_t<float> query,
                             py::array_t<float> database, float threshold,
                             int top_k);

}  // namespace eventai
