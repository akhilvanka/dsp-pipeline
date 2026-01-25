#include "filters.h"
#include "fft.h"
#include "anomaly.h"
#include "pipeline.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <cstdlib>

#define PASS(n) std::printf("PASS: %s\n", n)
#define FAIL(n, m) do { std::fprintf(stderr, "FAIL: %s — %s\n", n, m); std::exit(1); } while(0)
#define ASSERT_NEAR(a, b, tol) do { \
    if (std::fabs((a)-(b)) > (tol)) { \
        std::fprintf(stderr, "FAIL: |%.6f - %.6f| = %.6f > %.6f at %s:%d\n", \
            (double)(a),(double)(b),std::fabs((double)(a)-(double)(b)),(double)(tol),__FILE__,__LINE__); \
        std::exit(1); } } while(0)

// ---- FIR tests ----

void test_fir_passthrough_dc() {
    // All-ones FIR with single tap = passthrough
    dsp::FIRFilter<1>::CoeffArray c = {1.0};
    dsp::FIRFilter<1> f(c);
    ASSERT_NEAR(f.process(5.0), 5.0, 1e-10);
    PASS("fir_passthrough_dc");
}

void test_fir_lowpass_attenuation() {
    auto fir = dsp::make_lowpass_fir<63>(0.25);
    double pass = fir.magnitude_response(0.05 * M_PI);
    double stop = fir.magnitude_response(0.9  * M_PI);
    if (pass < 0.8) FAIL("fir_lowpass_attenuation", "passband too attenuated");
    if (stop > 0.1) FAIL("fir_lowpass_attenuation", "stopband not attenuated");
    PASS("fir_lowpass_attenuation");
}

void test_fir_delay_line() {
    // Unit impulse response — output should equal coefficients
    dsp::FIRFilter<4>::CoeffArray c = {0.1, 0.2, 0.4, 0.3};
    dsp::FIRFilter<4> f(c);

    ASSERT_NEAR(f.process(1.0), 0.1, 1e-10);
    ASSERT_NEAR(f.process(0.0), 0.2, 1e-10);
    ASSERT_NEAR(f.process(0.0), 0.4, 1e-10);
    ASSERT_NEAR(f.process(0.0), 0.3, 1e-10);
    ASSERT_NEAR(f.process(0.0), 0.0, 1e-10);
    PASS("fir_delay_line");
}

// ---- IIR tests ----

void test_iir_butterworth_dc() {
    auto iir = dsp::make_butterworth_lp(0.1);
    // Feed DC → output should converge to 1.0
    double y = 0.0;
    for (int i = 0; i < 200; ++i) y = iir.process(1.0);
    ASSERT_NEAR(y, 1.0, 0.01);
    PASS("iir_butterworth_dc");
}

void test_iir_highfreq_rejection() {
    auto iir = dsp::make_butterworth_lp(0.05); // cutoff 5% Nyquist
    // Feed a sine at 40% Nyquist — should be heavily attenuated
    double max_out = 0.0;
    for (int i = 0; i < 1000; ++i) {
        double x = std::sin(2.0 * M_PI * 0.4 * i);
        double y = std::fabs(iir.process(x));
        if (y > max_out) max_out = y;
    }
    if (max_out > 0.1) FAIL("iir_highfreq_rejection", "high-freq not attenuated");
    PASS("iir_highfreq_rejection");
}

// ---- Moving average tests ----

void test_moving_average() {
    dsp::MovingAverage<4> ma;
    ASSERT_NEAR(ma.process(4.0), 4.0, 1e-9);   // 1 sample: mean=4
    ASSERT_NEAR(ma.process(2.0), 3.0, 1e-9);   // 2: mean=3
    ASSERT_NEAR(ma.process(4.0), 10.0/3.0, 1e-9); // 3: mean=10/3
    ASSERT_NEAR(ma.process(2.0), 3.0, 1e-9);   // 4: (4+2+4+2)/4=3
    ASSERT_NEAR(ma.process(4.0), 3.0, 1e-9);   // 5: (2+4+2+4)/4=3 (window slides)
    PASS("moving_average");
}

// ---- FFT tests ----

void test_fft_known_input() {
    // FFT of a cosine at bin 1 should give two peaks at +/-1
    constexpr size_t N = 8;
    std::vector<dsp::Complex> buf(N);
    for (size_t i = 0; i < N; ++i) {
        buf[i] = dsp::Complex(std::cos(2.0 * M_PI * (double)i / N), 0.0);
    }
    dsp::fft_inplace(buf.data(), N);
    // Expect peak at index 1 and N-1=7
    ASSERT_NEAR(std::abs(buf[1]), (double)N/2.0, 0.5);
    ASSERT_NEAR(std::abs(buf[0]), 0.0, 0.5);
    PASS("fft_known_input");
}

void test_fft_inverse_roundtrip() {
    constexpr size_t N = 64;
    std::vector<dsp::Complex> orig(N), buf(N);
    for (size_t i = 0; i < N; ++i) {
        orig[i] = dsp::Complex(std::sin(2.0 * M_PI * 3 * (double)i / N), 0.0);
        buf[i]  = orig[i];
    }
    dsp::fft_inplace(buf.data(), N, false);
    dsp::fft_inplace(buf.data(), N, true);
    for (size_t i = 0; i < N; ++i) {
        ASSERT_NEAR(buf[i].real(), orig[i].real(), 1e-9);
    }
    PASS("fft_inverse_roundtrip");
}

void test_rfft_frequency_detection() {
    constexpr size_t N = 256;
    constexpr double FS = 1000.0;
    constexpr double F0 = 100.0; // 100 Hz
    std::vector<double> signal(N);
    for (size_t i = 0; i < N; ++i) {
        signal[i] = std::sin(2.0 * M_PI * F0 * (double)i / FS);
    }
    auto bins = dsp::rfft(signal.data(), N);
    auto mag  = dsp::magnitude_spectrum(bins);

    size_t peak = std::max_element(mag.begin(), mag.end()) - mag.begin();
    double peak_freq = (double)peak * FS / N;

    ASSERT_NEAR(peak_freq, F0, 5.0); // within one bin
    PASS("rfft_frequency_detection");
}

// ---- Anomaly detector tests ----

void test_zscore_no_false_alarms() {
    dsp::ZScoreDetector z(3.0, 50);
    int alerts = 0;
    // Stationary N(0,1) — very few alerts expected
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < 500; ++i) {
        if (z.update(nd(rng), i)) ++alerts;
    }
    if (alerts > 10) FAIL("zscore_no_false_alarms", "too many false alarms");
    PASS("zscore_no_false_alarms");
}

void test_zscore_detects_spike() {
    dsp::ZScoreDetector z(3.0, 30);
    for (int i = 0; i < 100; ++i) z.update(1.0 + 0.01 * i, i);
    // Inject a huge spike
    bool fired = z.update(1000.0, 100).has_value();
    if (!fired) FAIL("zscore_detects_spike", "spike not detected");
    PASS("zscore_detects_spike");
}

void test_cusum_detects_step() {
    dsp::CUSUMDetector c(0.5, 4.0);
    // 200 samples of zero, then step to 3σ
    int alerts = 0;
    for (int i = 0; i < 200; ++i) c.update(0.0, i);
    for (int i = 0; i < 50; ++i) {
        if (c.update(3.0, 200 + i)) ++alerts;
    }
    if (alerts == 0) FAIL("cusum_detects_step", "step not detected");
    PASS("cusum_detects_step");
}

// ---- Pipeline tests ----

void test_pipeline_passthrough() {
    dsp::Pipeline p;
    // No stages → identity
    std::vector<double> in  = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> out(5);
    p.process(in.data(), out.data(), 5);
    for (size_t i = 0; i < 5; ++i) ASSERT_NEAR(out[i], in[i], 1e-12);
    PASS("pipeline_passthrough");
}

void test_pipeline_scale() {
    dsp::Pipeline p;
    p.add_stage("scale_x2", [](dsp::SampleBlock in, dsp::MutableBlock out) {
        for (size_t i = 0; i < in.size(); ++i) out[i] = in[i] * 2.0;
    });
    p.add_stage("add_1", [](dsp::SampleBlock in, dsp::MutableBlock out) {
        for (size_t i = 0; i < in.size(); ++i) out[i] = in[i] + 1.0;
    });

    std::vector<double> in  = {1.0, 2.0, 3.0};
    std::vector<double> out(3);
    p.process(in.data(), out.data(), 3);

    ASSERT_NEAR(out[0], 3.0, 1e-12); // (1*2)+1
    ASSERT_NEAR(out[1], 5.0, 1e-12);
    ASSERT_NEAR(out[2], 7.0, 1e-12);
    PASS("pipeline_scale");
}

int main() {
    std::printf("Running DSP pipeline unit tests...\n\n");

    test_fir_passthrough_dc();
    test_fir_lowpass_attenuation();
    test_fir_delay_line();
    test_iir_butterworth_dc();
    test_iir_highfreq_rejection();
    test_moving_average();
    test_fft_known_input();
    test_fft_inverse_roundtrip();
    test_rfft_frequency_detection();
    test_zscore_no_false_alarms();
    test_zscore_detects_spike();
    test_cusum_detects_step();
    test_pipeline_passthrough();
    test_pipeline_scale();

    std::printf("\nAll tests passed.\n");
    return 0;
}
