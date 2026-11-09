#pragma once

// Independent CPU description of the represented BF16 and RK4V4-E8 cache boundary.
// No production codec, CUDA math, or attention kernel is included here.
#include "ops/op_check.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tensor_attention_test {

inline double half_value(std::uint16_t bits) {
    const int e = (bits >> 10) & 31, m = bits & 1023;
    const double a = e == 0 ? std::ldexp(double(m), -24)
                    : e == 31 ? (m ? std::numeric_limits<double>::quiet_NaN()
                                   : std::numeric_limits<double>::infinity())
                              : std::ldexp(double(1024 + m), e - 25);
    return bits & 32768 ? -a : a;
}

inline std::uint16_t half_bits(float x) {
    const auto u = std::bit_cast<std::uint32_t>(x);
    const unsigned sign = (u >> 16) & 32768;
    const unsigned exp = (u >> 23) & 255;
    unsigned mantissa = u & 0x7fffff;
    if (exp == 255) return sign | (mantissa ? 0x7e00 : 0x7c00);
    const int e = int(exp) - 112;
    if (e >= 31) return sign | 0x7c00;
    if (e < -10) return sign;
    const int shift = e <= 0 ? 14 - e : 13;
    if (e <= 0) mantissa |= 0x800000;
    unsigned rounded = mantissa >> shift;
    const unsigned tail = mantissa & ((1u << shift) - 1);
    const unsigned tie = 1u << (shift - 1);
    rounded += tail > tie || (tail == tie && (rounded & 1));
    return sign | ((e <= 0 ? 0u : unsigned(e) << 10) + rounded);
}

// Direct matrix-vector product, independent of the device butterfly. BF16 fixtures use
// bounded dyadic values, so their FP32 codec transform is exact throughout the butterfly.
template <typename T>
inline std::array<T, 64> rotate(const T* input) {
    std::array<T, 64> result{};
    for (unsigned i = 0; i < 64; ++i) {
        double sum = 0;
        for (unsigned j = 0; j < 64; ++j)
            sum += (std::popcount(i & j) & 1 ? -1. : 1.) * double(input[j]);
        result[i] = T(sum * .125);
    }
    return result;
}

// Nearest point in each E8 coset; ties select the integer coset and lowest coordinate.
// The persistent fork format deliberately rounds half-coset coordinates to even integers
// and clamps them to signed nibbles: it stores no coset bit.
inline std::array<int, 8> e8_codes(const float* x) {
    std::array<std::array<float, 8>, 2> candidates{};
    std::array<float, 2> distances{};
    for (int coset = 0; coset < 2; ++coset) {
        int parity = 0, worst = 0;
        float largest = -1;
        for (int d = 0; d < 8; ++d) {
            const float shifted = x[d] - float(coset) * .5f;
            const float rounded = std::nearbyint(shifted);
            candidates[coset][d] = rounded + float(coset) * .5f;
            parity += int(rounded);
            const float error = std::abs(shifted - rounded);
            if (error > largest) { largest = error; worst = d; }
        }
        if (parity & 1)
            candidates[coset][worst] += x[worst] >= candidates[coset][worst] ? 1.f : -1.f;
        std::array<float, 8> errors{};
        for (int d = 0; d < 8; ++d) {
            const float diff = x[d] - candidates[coset][d];
            errors[d] = diff * diff;
        }
        // The codec's FP32 distance boundary has pairwise addition; this is not part
        // of the attention oracle, which never reproduces internal GPU rounding.
        distances[coset] = ((errors[0] + errors[1]) + (errors[2] + errors[3])) +
                           ((errors[4] + errors[5]) + (errors[6] + errors[7]));
    }
    const int selected = distances[0] <= distances[1] ? 0 : 1;
    std::array<int, 8> result{};
    for (int d = 0; d < 8; ++d)
        result[d] = std::clamp(int(std::nearbyint(candidates[selected][d])), -8, 7);
    return result;
}

struct EncodedRow {
    std::array<std::uint8_t, 128> bytes{};
    std::array<std::uint16_t, 4> scales{};
};

inline EncodedRow encode_e8(const float* input, bool key) {
    EncodedRow out;
    for (int g = 0; g < 4; ++g) {
        auto rotated = rotate(input + 64 * g);
        float maximum = 0;
        for (float x : rotated) maximum = std::max(maximum, std::abs(x));
        out.scales[g] = half_bits(maximum / 7.f);
        const float scale = float(half_value(out.scales[g]));
        const float inverse = scale > 0 ? 1.f / scale : 0.f;
        for (float& x : rotated) x *= inverse;
        for (int block = 0; block < 8; ++block) {
            std::array<int, 8> codes{};
            if (key) codes = e8_codes(rotated.data() + block * 8);
            else for (int d = 0; d < 8; ++d)
                codes[d] = std::clamp(int(std::nearbyint(rotated[block * 8 + d])), -7, 7);
            for (int d = 0; d < 8; d += 2)
                out.bytes[g * 32 + block * 4 + d / 2] =
                    (codes[d] & 15) | ((codes[d + 1] & 15) << 4);
        }
    }
    return out;
}

} // namespace tensor_attention_test
