#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "blur_ops.h"
#include "face_ops.h"
#include "preprocess_ops.h"

namespace py = pybind11;

PYBIND11_MODULE(_quickpitik_cpp, m) {
    m.doc() = "QuickPitik C++ acceleration module";
    m.attr("__version__") = "1.0.0";

    // --- Face ops ---
    py::class_<quickpitik::TopKResult>(m, "TopKResult")
        .def_readonly("indices", &quickpitik::TopKResult::indices)
        .def_readonly("scores", &quickpitik::TopKResult::scores)
        .def("__repr__", [](const quickpitik::TopKResult& r) {
            return "<TopKResult with " + std::to_string(r.indices.size()) +
                   " results>";
        });

    m.def("cosine_similarity", &quickpitik::cosine_similarity,
          py::arg("a"), py::arg("b"),
          "Cosine similarity between two L2-normalized float32 vectors.");

    m.def("batch_cosine_topk", &quickpitik::batch_cosine_topk,
          py::arg("query"), py::arg("database"),
          py::arg("threshold") = 0.4f, py::arg("top_k") = 10,
          "Batch cosine similarity with threshold filtering and top-K "
          "selection.");

    // --- Blur ops ---
    py::class_<quickpitik::BlurMetrics>(m, "BlurMetrics")
        .def_readonly("laplacian_var", &quickpitik::BlurMetrics::laplacian_var)
        .def_readonly("hf_ratio", &quickpitik::BlurMetrics::hf_ratio)
        .def("__repr__", [](const quickpitik::BlurMetrics& b) {
            return "<BlurMetrics laplacian_var=" +
                   std::to_string(b.laplacian_var) +
                   " hf_ratio=" + std::to_string(b.hf_ratio) + ">";
        });

    m.def("laplacian_variance", &quickpitik::laplacian_variance,
          py::arg("gray"),
          "Laplacian variance of a grayscale uint8 image.");

    m.def("fft_hf_ratio", &quickpitik::fft_hf_ratio,
          py::arg("gray"),
          "FFT high-frequency magnitude ratio of a grayscale uint8 image.");

    m.def("batch_blur_metrics", &quickpitik::batch_blur_metrics,
          py::arg("images"),
          "Compute Laplacian variance and FFT HF ratio for a batch of "
          "grayscale images.");

    // --- Preprocess ops ---
    m.def("bgr_to_gray", &quickpitik::bgr_to_gray,
          py::arg("bgr"),
          "Convert BGR (H,W,3) uint8 image to grayscale (H,W) uint8.");

    m.def("resize_gray", &quickpitik::resize_gray,
          py::arg("gray"), py::arg("new_h"), py::arg("new_w"),
          "Resize grayscale (H,W) uint8 image using bilinear interpolation.");

    m.def("classify_preprocess", &quickpitik::classify_preprocess,
          py::arg("bgr"), py::arg("target_size"),
          "Fused classify preprocessing: center-crop + resize + BGR->RGB "
          "normalize + HWC->CHW. Returns (1, 3, S, S) float32 tensor.");
}
