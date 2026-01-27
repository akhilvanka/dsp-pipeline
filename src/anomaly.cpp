#include "anomaly.h"
#include <cmath>
#include <numeric>

namespace dsp {

bool SpectralAnomalyDetector::update_spectrum(const std::vector<double>& magnitudes) {
    size_t nbins = magnitudes.size();

    if (frame_count_ < baseline_frames_) {
        if (baseline_.empty()) baseline_.resize(nbins, 0.0);
        for (size_t i = 0; i < nbins; ++i)
            baseline_[i] += magnitudes[i];
        ++frame_count_;
        if (frame_count_ == baseline_frames_)
            for (auto& v : baseline_) v /= (double)baseline_frames_;
        return false;
    }

    diff_db_.resize(nbins);
    bool anomaly = false;

    for (size_t i = 0; i < nbins && i < baseline_.size(); ++i) {
        double ref = baseline_[i] < 1e-12 ? 1e-12 : baseline_[i];
        double cur = magnitudes[i]         < 1e-12 ? 1e-12 : magnitudes[i];
        diff_db_[i] = 20.0 * std::log10(cur / ref);
        if (std::abs(diff_db_[i]) > threshold_db_)
            anomaly = true;
    }
    return anomaly;
}

} // namespace dsp
