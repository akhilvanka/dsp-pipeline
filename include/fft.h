#pragma once
#include <array>
#include <complex>
#include <cmath>
#include <cstddef>
#include <cassert>
#include <vector>

namespace dsp {

using Complex = std::complex<double>;

// In-place FFT on a power-of-two length buffer (forward or inverse)
void fft_inplace(Complex* buf, size_t n, bool inverse = false);

// Convenience: FFT of a real signal — returns n/2+1 complex bins
std::vector<Complex> rfft(const double* signal, size_t n);

// Magnitude spectrum from rfft output (n/2+1 elements)
std::vector<double> magnitude_spectrum(const std::vector<Complex>& bins);

// Short-Time Fourier Transform: overlapping windowed FFTs.
// Returns a matrix [frames x (n_fft/2+1)] of magnitudes.
struct STFTResult {
    std::vector<std::vector<double>> magnitudes; // [frame][bin]
    size_t n_fft;
    size_t hop;
    double sample_rate;

    // Frequency of bin k
    double bin_freq(size_t k) const {
        return (double)k * sample_rate / (double)n_fft;
    }
};

STFTResult stft(const double* signal, size_t n_samples,
                size_t n_fft, size_t hop, double sample_rate = 1.0);

// Power spectral density (Welch's method averaged over frames)
std::vector<double> psd_welch(const double* signal, size_t n,
                               size_t n_fft = 256, size_t hop = 128);

} // namespace dsp
