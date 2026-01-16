#pragma once
#include <functional>
#include <vector>
#include <cstddef>
#include <string>
#include <memory>
#include <cstdio>
#include <algorithm>

namespace dsp {

// Lightweight non-owning view (C++17 substitute for std::span)
struct SampleBlock {
    const double* data;
    size_t        n;
    size_t        size() const { return n; }
    const double& operator[](size_t i) const { return data[i]; }
    const double* begin() const { return data; }
    const double* end()   const { return data + n; }
};

struct MutableBlock {
    double* data;
    size_t  n;
    size_t  size() const { return n; }
    double& operator[](size_t i) { return data[i]; }
};

// A processing stage: reads input, writes to output (same length).
// Stages may also be sinks (they ignore the output span).
struct Stage {
    std::string name;
    std::function<void(SampleBlock in, MutableBlock out)> process;
};

class Pipeline {
public:
    // Add a processing stage in-place transform
    void add_stage(std::string name,
                   std::function<void(SampleBlock, MutableBlock)> fn) {
        stages_.push_back({std::move(name), std::move(fn)});
    }

    // Add a sink stage (reads input, writes nothing back)
    void add_sink(std::string name,
                  std::function<void(SampleBlock)> fn) {
        auto wrapped = [fn](SampleBlock in, MutableBlock out) {
            // Copy through so the next stage still sees data
            std::copy(in.begin(), in.end(), out.data);
            fn(in);
        };
        stages_.push_back({std::move(name), std::move(wrapped)});
    }

    // Process a block of samples through all stages.
    // Uses two internal scratch buffers and ping-pongs between them.
    void process(const double* input, double* output, size_t n) {
        if (stages_.empty()) {
            std::copy(input, input + n, output);
            return;
        }

        ensure_buffers(n);

        // Copy input into buf_a_
        std::copy(input, input + n, buf_a_.begin());

        for (size_t i = 0; i < stages_.size(); ++i) {
            auto& stage = stages_[i];
            bool  last  = (i == stages_.size() - 1);
            auto* src   = (i % 2 == 0) ? buf_a_.data() : buf_b_.data();
            // Last stage always writes directly to caller's output buffer
            auto* dst   = last ? output
                               : ((i % 2 == 0) ? buf_b_.data() : buf_a_.data());
            stage.process(SampleBlock{src, n}, MutableBlock{dst, n});
        }
    }

    void reset() {
        buf_a_.clear();
        buf_b_.clear();
    }

    size_t num_stages() const { return stages_.size(); }

private:
    std::vector<Stage>  stages_;
    std::vector<double> buf_a_, buf_b_;

    void ensure_buffers(size_t n) {
        if (buf_a_.size() < n) { buf_a_.resize(n); buf_b_.resize(n); }
    }
};

} // namespace dsp
