#include "fft.h"
#include <cmath>
#include <algorithm>
#include <cassert>

namespace dsp {

static size_t bit_reverse(size_t n, size_t bits) {
    size_t r = 0;
    for (size_t i = 0; i < bits; ++i) {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

void fft_inplace(Complex* buf, size_t n, bool inverse) {
    assert(n > 0 && (n & (n - 1)) == 0); // must be power of 2

    // Bit-reversal permutation
    size_t bits = 0;
    for (size_t tmp = n; tmp > 1; tmp >>= 1) ++bits;

    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, bits);
        if (j > i) std::swap(buf[i], buf[j]);
    }

    // Cooley-Tukey butterfly stages
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / (double)len * (inverse ? 1.0 : -1.0);
        Complex wlen(std::cos(ang), std::sin(ang));

        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                Complex u = buf[i + j];
                Complex v = buf[i + j + len / 2] * w;
                buf[i + j]           = u + v;
                buf[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        double inv_n = 1.0 / (double)n;
        for (size_t i = 0; i < n; ++i) buf[i] *= inv_n;
    }
}