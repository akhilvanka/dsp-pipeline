#include "filters.h"
#include "fft.h"
#include "anomaly.h"
#include "pipeline.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

static constexpr double SAMPLE_RATE   = 1000.0;  // Hz
static constexpr size_t N_SAMPLES     = 2048;
static constexpr double SIGNAL_FREQ   = 50.0;    // Hz — primary component
static constexpr double NOISE_SIGMA   = 0.1;
static constexpr double FAULT_FREQ    = 200.0;   // Hz — injected at sample 1024
static constexpr double FAULT_AMPLITUDE = 0.5;

int main() {
    std::printf("=== Real-Time DSP Pipeline Demo ===\n\n");

    // ---- Generate synthetic signal ----
    // Clean signal: 50 Hz sine + white noise
    // Fault event at sample 1024: additional 200 Hz component injected
    std::mt19937 rng(42);
    std::normal_distribution<double> noise_dist(0.0, NOISE_SIGMA);

    std::vector<double> signal(N_SAMPLES);
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        double t = (double)i / SAMPLE_RATE;
        signal[i] = std::sin(2.0 * M_PI * SIGNAL_FREQ * t)
                  + noise_dist(rng);
        // Inject fault at sample 1024
        if (i >= 1024) {
            signal[i] += FAULT_AMPLITUDE * std::sin(2.0 * M_PI * FAULT_FREQ * t);
        }
    }

    std::printf("Signal: %zu samples @ %.0f Hz\n", N_SAMPLES, SAMPLE_RATE);
    std::printf("  Component: %.0f Hz sine + N(0,%.2f) noise\n",
        SIGNAL_FREQ, NOISE_SIGMA);
    std::printf("  Fault injected at sample 1024: %.0f Hz, amplitude=%.2f\n\n",
        FAULT_FREQ, FAULT_AMPLITUDE);

    // ---- FIR lowpass filter (cutoff 100 Hz / Nyquist) ----
    auto fir = dsp::make_lowpass_fir<63>(100.0 / (SAMPLE_RATE / 2.0));

    std::vector<double> fir_out(N_SAMPLES);
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        fir_out[i] = fir.process(signal[i]);
    }

    // Verify FIR: magnitude at cutoff should be ~0.5 (-3dB)
    double fir_mag_pass = fir.magnitude_response(0.1 * M_PI); // well below cutoff
    double fir_mag_stop = fir.magnitude_response(0.8 * M_PI); // well above cutoff
    std::printf("FIR filter (63-tap, fc=100Hz):\n");
    std::printf("  Passband magnitude (@50Hz): %.4f (target ≈ 1.0)\n", fir_mag_pass);
    std::printf("  Stopband magnitude (@400Hz): %.6f (target ≈ 0)\n\n", fir_mag_stop);

    // ---- Butterworth IIR comparison ----
    auto iir = dsp::make_butterworth_lp(100.0 / (SAMPLE_RATE / 2.0));
    std::vector<double> iir_out(N_SAMPLES);
    for (size_t i = 0; i < N_SAMPLES; ++i) iir_out[i] = iir.process(signal[i]);

    // ---- Pipeline composition ----
    dsp::Pipeline pipeline;
    dsp::MovingAverage<8> ma;
    dsp::ZScoreDetector   zscore(3.5, 50);
    dsp::CUSUMDetector    cusum(0.5, 4.0);

    size_t z_alerts = 0, cusum_alerts = 0;

    // Stage 1: Smooth with moving average
    pipeline.add_stage("moving_avg_8", [&](dsp::SampleBlock in, dsp::MutableBlock out) {
        for (size_t i = 0; i < in.size(); ++i) {
            out[i] = ma.process(in[i]);
        }
    });

    // Stage 2: Z-score anomaly detection (sink)
    pipeline.add_sink("zscore_anomaly", [&](dsp::SampleBlock in) {
        for (size_t i = 0; i < in.size(); ++i) {
            if (zscore.update(in[i], i)) ++z_alerts;
            if (cusum.update(in[i], i))  ++cusum_alerts;
        }
    });

    std::vector<double> pipeline_out(N_SAMPLES);
    pipeline.process(signal.data(), pipeline_out.data(), N_SAMPLES);

    std::printf("Anomaly Detection (on noisy+fault signal):\n");
    std::printf("  Z-score alerts (>3.5σ):  %zu\n", z_alerts);
    std::printf("  CUSUM alerts (mean shift): %zu\n", cusum_alerts);
    std::printf("  Z-score mean: %.4f, std: %.4f\n\n",
        zscore.mean(), zscore.std_dev());

    // ---- FFT spectral analysis ----
    // Analyze pre-fault vs post-fault windows
    auto pre_bins  = dsp::rfft(signal.data(),        512);
    auto post_bins = dsp::rfft(signal.data() + 1024, 512);

    auto pre_mag  = dsp::magnitude_spectrum(pre_bins);
    auto post_mag = dsp::magnitude_spectrum(post_bins);

    // Find dominant frequencies
    auto find_peak = [&](const std::vector<double>& mag) -> std::pair<double,double> {
        size_t peak_idx = std::max_element(mag.begin(), mag.end()) - mag.begin();
        double freq = (double)peak_idx * SAMPLE_RATE / 512.0;
        return {freq, mag[peak_idx]};
    };

    auto [pre_freq, pre_amp]   = find_peak(pre_mag);
    auto [post_freq, post_amp] = find_peak(post_mag);

    std::printf("FFT Analysis (512-point, %.0f Hz):\n", SAMPLE_RATE);
    std::printf("  Pre-fault  dominant frequency: %.1f Hz (amplitude=%.3f)\n",
        pre_freq, pre_amp);
    std::printf("  Post-fault dominant frequency: %.1f Hz (amplitude=%.3f)\n",
        post_freq, post_amp);

    // Check that post-fault spectrum shows energy at fault frequency
    size_t fault_bin = (size_t)(FAULT_FREQ * 512 / SAMPLE_RATE);
    double fault_energy = post_mag[fault_bin];
    std::printf("  Post-fault energy @ %.0f Hz: %.3f (expected > 0.1)\n\n",
        FAULT_FREQ, fault_energy);

    // ---- STFT fault detection ----
    auto stft_result = dsp::stft(signal.data(), N_SAMPLES, 256, 128, SAMPLE_RATE);
    dsp::SpectralAnomalyDetector sad(256, 6.0, 8);

    size_t spectral_anomalies = 0;
    size_t first_anomaly_frame = SIZE_MAX;

    for (size_t f = 0; f < stft_result.magnitudes.size(); ++f) {
        if (sad.update_spectrum(stft_result.magnitudes[f])) {
            ++spectral_anomalies;
            if (first_anomaly_frame == SIZE_MAX) first_anomaly_frame = f;
        }
    }

    double first_anomaly_sample = first_anomaly_frame != SIZE_MAX
        ? (double)(first_anomaly_frame * 128) : -1.0;

    std::printf("STFT Spectral Anomaly Detection (%zu frames):\n",
        stft_result.magnitudes.size());
    std::printf("  Anomalous frames detected: %zu\n", spectral_anomalies);
    std::printf("  First anomaly at frame %zu (~sample %.0f)\n",
        first_anomaly_frame, first_anomaly_sample);
    std::printf("  Fault injected at sample 1024 — detection lag: ~%.0f samples\n\n",
        first_anomaly_sample > 0 ? first_anomaly_sample - 1024.0 : 0.0);

    // ---- Welch PSD ----
    auto psd = dsp::psd_welch(signal.data(), N_SAMPLES, 256, 128);
    double total_power = 0.0;
    for (auto v : psd) total_power += v;
    std::printf("Welch PSD: %zu bins, total power = %.4f\n", psd.size(), total_power);

    // Final pass/fail
    bool fir_ok       = fir_mag_stop < 0.01;
    bool fault_ok     = fault_energy > 0.1;
    bool anomaly_ok   = z_alerts > 0 || cusum_alerts > 0;
    bool spectral_ok  = spectral_anomalies > 0;

    std::printf("\n=== Verification ===\n");
    std::printf("  FIR stopband attenuation: %s\n", fir_ok ? "PASS" : "FAIL");
    std::printf("  Fault frequency detected in FFT: %s\n", fault_ok ? "PASS" : "FAIL");
    std::printf("  Time-domain anomaly detected: %s\n", anomaly_ok ? "PASS" : "FAIL");
    std::printf("  Spectral anomaly detected: %s\n", spectral_ok ? "PASS" : "FAIL");

    bool all_ok = fir_ok && fault_ok && anomaly_ok && spectral_ok;
    std::printf("\n[%s] DSP pipeline demo complete.\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
