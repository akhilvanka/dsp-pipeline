#include "anomaly.h"
#include <cmath>
#include <numeric>

namespace dsp {

bool SpectralAnomalyDetector::update_spectrum(const std::vector<double>& magnitudes) {
    size_t nbins = magnitudes.size();

    if (frame_count_ < baseline_frames_) {
        // Accumulate baseline
        if (baseline_.empty()) baseline_.resize(nbins, 0.0);
        for (size_t i = 0; i < nbins; ++i) {
            baseline_[i] += magnitudes[i];
        }
        ++frame_count_;
        if (frame_count_ == baseline_frames_) {
            // Average
            for (auto& v : baseline_) v /= (double)baseline_frames_;
        }
        return false;
    }
