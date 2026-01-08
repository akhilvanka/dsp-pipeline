#pragma once
#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <span>

namespace dsp {

// ---- FIR Filter ----

template<size_t N>
class FIRFilter {
    static_assert(N > 0);
public:
    using CoeffArray = std::array<double, N>;

    explicit FIRFilter(const CoeffArray& coeffs) : h_(coeffs) {
        buf_.fill(0.0);
    }

    double process(double x) {
        buf_[pos_] = x;
        double y = 0.0;
        for (size_t i = 0; i < N; ++i) {
            y += h_[i] * buf_[(pos_ + N - i) % N];
        }
        pos_ = (pos_ + 1) % N;
        return y;
    }

    void reset() { buf_.fill(0.0); pos_ = 0; }

    static constexpr size_t order() { return N; }

    // Compute frequency response at normalized frequency w ∈ [0, π]
    double magnitude_response(double w) const {
        double re = 0.0, im = 0.0;
        for (size_t n = 0; n < N; ++n) {
            re += h_[n] * std::cos(n * w);
            im -= h_[n] * std::sin(n * w);
        }
        return std::sqrt(re*re + im*im);
    }

private:
    CoeffArray            h_;
    std::array<double, N> buf_;
    size_t                pos_{0};
};

// ---- Factory: windowed sinc lowpass FIR ----
template<size_t N>
FIRFilter<N> make_lowpass_fir(double cutoff_normalized) {
    static_assert(N % 2 == 1, "FIR order should be odd for symmetric filter");
    std::array<double, N> h;
    int M = (int)N - 1;
    int center = M / 2;

    for (int n = 0; n < (int)N; ++n) {
        double t = n - center;
        // lim_{t→0} sin(π·fc·t)/(π·t) = fc, not 1
        double sinc = (t == 0.0) ? cutoff_normalized
                    : std::sin(M_PI * cutoff_normalized * t) / (M_PI * t);
        // Hann window
        double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / M));
        h[(size_t)n] = sinc * w;
    }
    // Normalize DC gain to 1
    double sum = 0.0;
    for (auto v : h) sum += v;
    for (auto& v : h) v /= sum;

    return FIRFilter<N>(h);
}

// ---- Biquad section (IIR second-order section) ----
struct Biquad {
    double b0, b1, b2; // numerator
    double a1, a2;     // denominator (a0 = 1 normalized)
    double s1{0}, s2{0}; // state (transposed direct form II)

    double process(double x) {
        double y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        return y;
    }
    void reset() { s1 = s2 = 0.0; }
};

// ---- IIR filter as cascade of biquad sections ----
template<size_t Sections>
class IIRFilter {
public:
    explicit IIRFilter(const std::array<Biquad, Sections>& sections)
        : sections_(sections) {}

    double process(double x) {
        for (auto& s : sections_) x = s.process(x);
        return x;
    }
    void reset() { for (auto& s : sections_) s.reset(); }

private:
    std::array<Biquad, Sections> sections_;
};

// Factory: 2nd-order Butterworth lowpass (single biquad)
inline IIRFilter<1> make_butterworth_lp(double cutoff_normalized) {
    // Bilinear transform of analog prototype
    double w = std::tan(M_PI * cutoff_normalized);
    double w2 = w * w;
    double sqrt2w = std::sqrt(2.0) * w;
    double denom = 1.0 + sqrt2w + w2;
    Biquad b;
    b.b0 = w2 / denom;
    b.b1 = 2.0 * b.b0;
    b.b2 = b.b0;
    b.a1 = 2.0 * (w2 - 1.0) / denom;
    b.a2 = (1.0 - sqrt2w + w2) / denom;
    return IIRFilter<1>({b});
}

// ---- Moving Average (O(1) ring buffer implementation) ----
template<size_t N>
class MovingAverage {
    static_assert(N > 0);
public:
    double process(double x) {
        sum_ -= buf_[pos_];
        buf_[pos_] = x;
        sum_ += x;
        pos_ = (pos_ + 1) % N;
        if (count_ < N) ++count_;
        return sum_ / (double)count_;
    }

    void reset() { buf_.fill(0.0); sum_ = 0.0; pos_ = 0; count_ = 0; }
    double current() const { return count_ > 0 ? sum_ / count_ : 0.0; }
    static constexpr size_t window() { return N; }

private:
    std::array<double, N> buf_{};
    double                sum_{0.0};
    size_t                pos_{0};
    size_t                count_{0};
};

} // namespace dsp
