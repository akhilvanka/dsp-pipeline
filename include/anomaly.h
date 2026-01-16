#pragma once
#include "filters.h"
#include "fft.h"
#include <cmath>
#include <cstddef>
#include <optional>

namespace dsp {

struct AnomalyEvent {
    size_t  sample_index;
    double  value;
    double  score;
    enum class Type { Z_SCORE, CUSUM_HIGH, CUSUM_LOW, SPECTRAL } type;
};

// ---- Z-score detector ----
class ZScoreDetector {
public:
    explicit ZScoreDetector(double threshold = 3.0, size_t warmup = 30)
        : threshold_(threshold), warmup_(warmup) {}

    std::optional<AnomalyEvent> update(double x, size_t idx = 0) {
        ++n_;
        double delta  = x - mean_;
        mean_        += delta / (double)n_;
        double delta2 = x - mean_;
        m2_          += delta * delta2;

        if (n_ <= warmup_) return std::nullopt;

        double var   = m2_ / (double)(n_ - 1);
        double sigma = std::sqrt(var);
        if (sigma < 1e-12) return std::nullopt;

        double z = std::abs(x - mean_) / sigma;
        if (z > threshold_) {
            return AnomalyEvent{idx, x, z, AnomalyEvent::Type::Z_SCORE};
        }
        return std::nullopt;
    }

    void reset() { n_ = 0; mean_ = 0; m2_ = 0; }
    double mean()    const { return mean_; }
    double std_dev() const {
        return n_ > 1 ? std::sqrt(m2_ / (n_ - 1)) : 0.0;
    }

private:
    double threshold_;
    size_t warmup_;
    size_t n_{0};
    double mean_{0.0}, m2_{0.0};
};

// ---- CUSUM detector ----
class CUSUMDetector {
public:
    // k: allowance (half the shift magnitude to detect)
    // h: decision threshold
    explicit CUSUMDetector(double k = 0.5, double h = 5.0)
        : k_(k), h_(h) {}

    std::optional<AnomalyEvent> update(double x, size_t idx = 0) {
        ++n_;
        // Estimate running mean and std for normalization
        double delta  = x - mean_;
        mean_        += delta / (double)n_;
        m2_          += delta * (x - mean_);
        double sigma  = n_ > 1 ? std::sqrt(m2_ / (n_ - 1)) : 1.0;
        if (sigma < 1e-9) sigma = 1.0;

        double z = (x - mean_) / sigma;

        // Tabular CUSUM
        cusum_hi_ = std::max(0.0, cusum_hi_ + z - k_);
        cusum_lo_ = std::max(0.0, cusum_lo_ - z - k_);

        if (cusum_hi_ > h_) {
            cusum_hi_ = 0.0; // reset after alarm
            return AnomalyEvent{idx, x, cusum_hi_, AnomalyEvent::Type::CUSUM_HIGH};
        }
        if (cusum_lo_ > h_) {
            cusum_lo_ = 0.0;
            return AnomalyEvent{idx, x, cusum_lo_, AnomalyEvent::Type::CUSUM_LOW};
        }
        return std::nullopt;
    }

    void reset() { cusum_hi_ = cusum_lo_ = 0; n_ = 0; mean_ = 0; m2_ = 0; }
    double cusum_hi() const { return cusum_hi_; }
    double cusum_lo() const { return cusum_lo_; }

private:
    double k_, h_;
    double cusum_hi_{0.0}, cusum_lo_{0.0};
    size_t n_{0};
    double mean_{0.0}, m2_{0.0};
};

// ---- Spectral anomaly detector ----
class SpectralAnomalyDetector {
public:
    explicit SpectralAnomalyDetector(size_t n_fft = 256,
                                      double threshold_db = 10.0,
                                      size_t baseline_frames = 20)
        : n_fft_(n_fft), threshold_db_(threshold_db),
          baseline_frames_(baseline_frames) {}

    // Feed one frame's magnitude spectrum; returns whether an anomaly exists
    bool update_spectrum(const std::vector<double>& magnitudes);

    bool baseline_ready() const { return frame_count_ >= baseline_frames_; }
    const std::vector<double>& baseline() const { return baseline_; }
    const std::vector<double>& last_diff_db() const { return diff_db_; }

private:
    size_t               n_fft_;
    double               threshold_db_;
    size_t               baseline_frames_;
    size_t               frame_count_{0};
    std::vector<double>  baseline_;
    std::vector<double>  diff_db_;
};

} // namespace dsp
