# dsp-pipeline

Real-time DSP pipeline for signal processing and anomaly detection.

- FIR filter (transposed direct-form II)
- IIR biquad filter (second-order sections)
- Cooley-Tukey FFT
- STFT with overlap-add reconstruction
- CUSUM change-point detector
- Z-score anomaly detector

All filter state lives on the stack — no heap allocation during processing.

```
cmake -B build && cmake --build build
./build/dsp_demo
```

14 tests pass.
