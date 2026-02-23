#include "blur_ops.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace eventai {

// ---------- Laplacian Variance ----------

double laplacian_variance(py::array_t<uint8_t> gray) {
    auto buf = gray.unchecked<2>();
    const py::ssize_t H = buf.shape(0);
    const py::ssize_t W = buf.shape(1);

    if (H < 3 || W < 3) {
        return 0.0;
    }

    // Laplacian kernel [0,1,0; 1,-4,1; 0,1,0]
    // Single-pass variance: var = E[x^2] - E[x]^2
    double sum = 0.0;
    double sum_sq = 0.0;
    const py::ssize_t count = (H - 2) * (W - 2);

    {
        py::gil_scoped_release release;

        for (py::ssize_t y = 1; y < H - 1; ++y) {
            for (py::ssize_t x = 1; x < W - 1; ++x) {
                double val =
                    static_cast<double>(buf(y - 1, x)) +
                    static_cast<double>(buf(y + 1, x)) +
                    static_cast<double>(buf(y, x - 1)) +
                    static_cast<double>(buf(y, x + 1)) -
                    4.0 * static_cast<double>(buf(y, x));
                sum += val;
                sum_sq += val * val;
            }
        }
    }

    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    return variance;
}

// ---------- FFT helpers ----------

namespace {

// Radix-2 Cooley-Tukey 1D FFT (in-place). n must be a power of 2.
void fft1d(std::vector<std::complex<double>>& data, py::ssize_t n) {
    // Bit-reversal permutation
    for (py::ssize_t i = 1, j = 0; i < n; ++i) {
        py::ssize_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // Butterfly stages
    for (py::ssize_t len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        std::complex<double> wn(std::cos(angle), std::sin(angle));
        for (py::ssize_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (py::ssize_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = data[i + j];
                std::complex<double> t = w * data[i + j + len / 2];
                data[i + j] = u + t;
                data[i + j + len / 2] = u - t;
                w *= wn;
            }
        }
    }
}

// Next power of 2 >= n
py::ssize_t next_pow2(py::ssize_t n) {
    py::ssize_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

}  // anonymous namespace

double fft_hf_ratio(py::array_t<uint8_t> gray) {
    auto buf = gray.unchecked<2>();
    const py::ssize_t H = buf.shape(0);
    const py::ssize_t W = buf.shape(1);

    if (H < 4 || W < 4) {
        return 0.0;
    }

    const py::ssize_t pH = next_pow2(H);
    const py::ssize_t pW = next_pow2(W);

    // Allocate 2D complex array (row-major, pH x pW)
    std::vector<std::complex<double>> data(
        static_cast<size_t>(pH) * static_cast<size_t>(pW), {0.0, 0.0});

    double hf_ratio_val = 0.0;

    {
        py::gil_scoped_release release;

        // Copy image data (zero-padded)
        for (py::ssize_t y = 0; y < H; ++y) {
            for (py::ssize_t x = 0; x < W; ++x) {
                data[static_cast<size_t>(y) * static_cast<size_t>(pW) +
                     static_cast<size_t>(x)] = {
                    static_cast<double>(buf(y, x)), 0.0};
            }
        }

        // 1D FFT along rows
        std::vector<std::complex<double>> row_buf(static_cast<size_t>(pW));
        for (py::ssize_t y = 0; y < pH; ++y) {
            size_t row_offset = static_cast<size_t>(y) * static_cast<size_t>(pW);
            for (py::ssize_t x = 0; x < pW; ++x) {
                row_buf[static_cast<size_t>(x)] = data[row_offset + static_cast<size_t>(x)];
            }
            fft1d(row_buf, pW);
            for (py::ssize_t x = 0; x < pW; ++x) {
                data[row_offset + static_cast<size_t>(x)] = row_buf[static_cast<size_t>(x)];
            }
        }

        // 1D FFT along columns
        std::vector<std::complex<double>> col_buf(static_cast<size_t>(pH));
        for (py::ssize_t x = 0; x < pW; ++x) {
            for (py::ssize_t y = 0; y < pH; ++y) {
                col_buf[static_cast<size_t>(y)] =
                    data[static_cast<size_t>(y) * static_cast<size_t>(pW) +
                         static_cast<size_t>(x)];
            }
            fft1d(col_buf, pH);
            for (py::ssize_t y = 0; y < pH; ++y) {
                data[static_cast<size_t>(y) * static_cast<size_t>(pW) +
                     static_cast<size_t>(x)] = col_buf[static_cast<size_t>(y)];
            }
        }

        // Compute magnitude with fftshift (swap quadrants)
        // Center of shifted spectrum: (pH/2, pW/2)
        const py::ssize_t cy = pH / 2;
        const py::ssize_t cx = pW / 2;
        // Use ORIGINAL image dimensions for radius (match Python behavior)
        const py::ssize_t r = std::min(H, W) / 8;

        double total_mag = 0.0;
        double hf_mag = 0.0;

        for (py::ssize_t y = 0; y < pH; ++y) {
            for (py::ssize_t x = 0; x < pW; ++x) {
                double mag = std::abs(
                    data[static_cast<size_t>(y) * static_cast<size_t>(pW) +
                         static_cast<size_t>(x)]);
                total_mag += mag;

                // After fftshift, position (y, x) -> ((y + cy) % pH, (x + cx) % pW)
                py::ssize_t sy = (y + cy) % pH;
                py::ssize_t sx = (x + cx) % pW;

                // Check if shifted position is in the center LF block
                bool in_center = (sy >= cy - r && sy < cy + r &&
                                  sx >= cx - r && sx < cx + r);
                if (!in_center) {
                    hf_mag += mag;
                }
            }
        }

        if (total_mag > 0.0) {
            hf_ratio_val = hf_mag / total_mag;
        }
    }

    return hf_ratio_val;
}

// ---------- Batch ----------

std::vector<BlurMetrics> batch_blur_metrics(
    const std::vector<py::array_t<uint8_t>>& images) {
    std::vector<BlurMetrics> results;
    results.reserve(images.size());
    for (const auto& img : images) {
        BlurMetrics m;
        m.laplacian_var = laplacian_variance(img);
        m.hf_ratio = fft_hf_ratio(img);
        results.push_back(m);
    }
    return results;
}

}  // namespace eventai
