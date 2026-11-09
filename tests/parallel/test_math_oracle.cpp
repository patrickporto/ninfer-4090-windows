// CPU mathematical contracts only. No production decoder, placement, or CUDA headers.
// These fixtures establish independent reference mathematics, not GPU route qualification.
// Authorities: docs/maintainer/{tensor-formats,storage-layouts,qwen3.6-27b-model}.md.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void close(double actual, double expected, double budget, const char* message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > budget)
        throw std::runtime_error(std::string(message) + ": error=" +
                                 std::to_string(std::abs(actual - expected)) +
                                 " budget=" + std::to_string(budget));
}
// Float operations are explicit semantic roundings; compile with -fno-fast-math
// -ffp-contract=off. No underflow/overflow occurs in these finite fixtures.
float fadd(float a, float b) { volatile float out = a + b; return out; }
float fmul(float a, float b) { volatile float out = a * b; return out; }
double gamma(int operations) {
    const double nu = operations * std::ldexp(1.0, -24);
    require(nu < 1, "invalid FP32 error budget");
    return nu / (1 - nu);
}
std::size_t align(std::size_t value, std::size_t by) {
    return (value + by - 1) / by * by;
}
// Exact arithmetic expansion of positive finite binary16, independent of host/CUDA codecs.
double half(std::uint16_t bits) {
    require((bits & 0x8000) == 0 && (bits & 0x7c00) != 0x7c00, "invalid scale");
    const int e = bits >> 10, m = bits & 1023;
    return e == 0 ? std::ldexp(double(m), -24) : std::ldexp(1024.0 + m, e - 25);
}
struct Packed {
    int bits, rows, k, group, padded;
    std::size_t high_offset, scale_offset;
    std::vector<std::uint8_t> bytes;
    Packed(int b, int n, int columns) : bits(b), rows(n), k(columns),
        group(b == 8 ? 32 : 64), padded(int(align(columns, 128))) {
        const std::size_t count = std::size_t(n) * padded / group;
        high_offset = align(count * 32, 256);
        scale_offset = high_offset + align(b == 5 ? count * 8 : 0, 256);
        bytes.resize(scale_offset + 2 * count);
    }
    // Byte-oriented decoder: row-split-k128-v1, with explicit sign extension.
    double at(int row, int col) const {
        const std::size_t g = std::size_t(row) * (padded / group) + col / group;
        const int lane = col % group;
        unsigned word;
        if (bits == 8) word = bytes[g * 32 + lane];
        else {
            word = (bytes[g * 32 + lane / 2] >> (4 * (lane % 2))) & 15;
            if (bits == 5)
                word |= ((bytes[high_offset + g * 8 + lane / 8] >> (lane % 8)) & 1) << 4;
        }
        const int q = (word & (1u << (bits - 1))) ? int(word) - (1 << bits) : int(word);
        require(bits != 8 || q != -128, "illegal W8 code");
        const std::size_t s = scale_offset + 2 * g;
        return q * half(std::uint16_t(bytes[s] | unsigned(bytes[s + 1]) << 8));
    }
};
// Fixture generation uses a logical code list and sets individual stream bits.
// Expected values come from those logical inputs, never from a second decoder.
struct Fixture { Packed packed; std::vector<double> logical; };
Fixture fixture(int bits, int rows, int k, bool bounded_scales = false) {
    Fixture f{Packed(bits, rows, k), std::vector<double>(std::size_t(rows) * k)};
    constexpr std::array<std::uint16_t, 7> scales{0, 1, 0x03ff, 0x0400, 0x3800, 0x3c01, 0x7bff};
    const int group = f.packed.group, groups = f.packed.padded / group;
    for (int row = 0; row < rows; ++row) for (int g = 0; g < groups; ++g) {
        const std::size_t index = std::size_t(row) * groups + g;
        const auto scale = g * group >= k ? std::uint16_t(0) :
            (bounded_scales ? std::uint16_t(0x2800) : scales[(g + row) % scales.size()]);
        f.packed.bytes[f.packed.scale_offset + index * 2] = scale & 255;
        f.packed.bytes[f.packed.scale_offset + index * 2 + 1] = scale >> 8;
        for (int lane = 0; lane < group; ++lane) {
            const int col = g * group + lane;
            const int minimum = bits == 8 ? -127 : -(1 << (bits - 1));
            const int range = bits == 8 ? 255 : (1 << bits);
            const int q = col >= k || scale == 0 ? 0 : minimum + (lane * 13 + row * 7 + g) % range;
            if (col < k) f.logical[std::size_t(row) * k + col] = double(q) * half(scale);
            const unsigned u = unsigned(q) & ((1u << bits) - 1);
            for (int bit = 0; bit < bits; ++bit) {
                if (((u >> bit) & 1) == 0) continue;
                std::size_t byte; int shift;
                if (bits == 8) { byte = index * 32 + lane; shift = bit; }
                else if (bit < 4) { byte = index * 32 + lane / 2; shift = 4 * (lane % 2) + bit; }
                else { byte = f.packed.high_offset + index * 8 + lane / 8; shift = lane % 8; }
                f.packed.bytes[byte] |= std::uint8_t(1u << shift);
            }
        }
    }
    return f;
}
void codec_contract() {
    require(half(1) == std::ldexp(1.0, -24), "minimum subnormal scale");
    require(half(0x0400) == std::ldexp(1.0, -14), "minimum normal scale");
    require(half(0x7bff) == 65504, "maximum scale");
    for (int bits : {4, 5, 8}) for (int k : {63, 64, 65, 127, 128, 129, 130, 256}) {
        const auto f = fixture(bits, 3, k);
        for (int row = 0; row < 3; ++row) for (int col = 0; col < f.packed.padded; ++col)
            require(f.packed.at(row, col) == (col < k ? f.logical[std::size_t(row) * k + col] : 0),
                    "stored code/scale or K padding mismatch");
    }
    // Literal known bytes protect against a mutually mistaken fixture writer/reader.
    Packed q5(5, 2, 130);
    require(q5.high_offset == 256 && q5.scale_offset == 512 && q5.bytes.size() == 528,
            "Q5 documented [2,130] geometry");
    q5.bytes[0] = 0xf0; q5.bytes[q5.high_offset] = 1;
    q5.bytes[q5.scale_offset + 1] = 0x3c; // scale 1
    require(q5.at(0, 0) == -16 && q5.at(0, 1) == 15, "literal Q5 sign/high/nibble order");
    Packed q4(4, 1, 64); q4.bytes[0] = 0x78; q4.bytes[q4.scale_offset + 1] = 0x3c;
    require(q4.at(0, 0) == -8 && q4.at(0, 1) == 7, "literal Q4 nibble order");
    Packed w8(8, 1, 32); w8.bytes[0] = 0x81; w8.bytes[1] = 0x7f;
    w8.bytes[w8.scale_offset + 1] = 0x3c;
    require(w8.at(0, 0) == -127 && w8.at(0, 1) == 127, "literal W8 signed code");
}

struct Dot { double reference = 0, absolute_scale = 0; float rounded = 0; };
Dot dot(const Packed& w, int row, const std::vector<double>& x, int begin, int end) {
    Dot d;
    for (int k = begin; k < end; ++k) {
        const double product = w.at(row, k) * x[k];
        d.reference += product; d.absolute_scale += std::abs(product);
        d.rounded = fadd(d.rounded, fmul(float(w.at(row, k)), float(x[k])));
    }
    // 1 input cast, 1 multiplication, <=length additions per term. Weights are
    // exact binary32 (integer * binary16); x may require a binary32 cast.
    close(d.rounded, d.reference, gamma(end - begin + 2) * d.absolute_scale,
          "FP32 local dot against FP64 absolute-scale budget");
    return d;
}
void projection_contract() {
    // Actual model K extents, sampled output rows: no full giant model allocation.
    for (int bits : {4, 5, 8}) for (int k : {256, 5120, 6144, 17408}) {
        const auto f = fixture(bits, 4, k);
        std::vector<double> x(k);
        for (int c = 0; c < k; ++c)
            x[c] = c % 11 == 0 ? 0 : (c % 2 ? -1.0 : 1.0) * (1 + (c % 17) * 0.03125);
        require(k / 2 % 128 == 0, "row shard K128 boundary");
        for (int row = 0; row < 4; ++row) {
            const auto whole = dot(f.packed, row, x, 0, k);
            const auto a = dot(f.packed, row, x, 0, k / 2);
            const auto b = dot(f.packed, row, x, k / 2, k);
            const double local_budget = gamma(k / 2 + 2) * (a.absolute_scale + b.absolute_scale);
            const double sum_budget = gamma(1) * (std::abs(double(a.rounded)) + std::abs(double(b.rounded)));
            const float sum = fadd(a.rounded, b.rounded);
            // FP64 oracle summation rounding is separately bounded, not hidden in
            // a relative tolerance that would fail near cancellation.
            const double oracle_budget = 4 * std::numeric_limits<double>::epsilon() * k * whole.absolute_scale;
            close(sum, whole.reference, local_budget + sum_budget + oracle_budget, "TP2 row SUM");
            const double residual = row == 0 ? 7 : -0.125;
            const float y = fadd(float(residual), sum);
            close(y, residual + whole.reference, local_budget + sum_budget + oracle_budget +
                  gamma(1) * (std::abs(residual) + std::abs(double(sum))), "once-only residual budget");
        }
    }
    require(fadd(7, fadd(2, 3)) == 12, "7 + SUM(2,3) must equal 12");
    require(fadd(fadd(7, 2), fadd(7, 3)) == 19, "negative control: rank-local residual duplicates");
    // Exact cancellation with a small result: absolute component scales stay nonzero.
    const float cancellation = fadd(16777216.f, -16777216.f);
    close(cancellation, 0, gamma(1) * 33554432.0, "near-zero SUM contract");
    // Exact answer 1, rounded answer 0: a relative-only criterion is invalid.
    const float left = fadd(16777216.f, 1.f), right = -16777216.f;
    const float near_zero = fadd(left, right);
    require(near_zero == 0, "cancellation fixture must lose the unit term");
    close(near_zero, 1, gamma(2) * 33554433.0 +
          gamma(1) * (std::abs(double(left)) + std::abs(double(right))),
          "nonzero near-cancellation absolute-scale budget");
    const auto up = fixture(4, 256, 8), gate = fixture(5, 256, 8), down = fixture(8, 8, 256);
    std::vector<double> x{1, -1, 0, 0.5, -0.5, 2, -2, 0.125}, z(256);
    // Naive unsplit SiLU(gate)*up, computed from logical represented values.
    for (int channel = 0; channel < 256; ++channel) {
        double u = 0, g = 0;
        for (int k = 0; k < 8; ++k) {
            u += up.logical[channel * 8 + k] * x[k];
            g += gate.logical[channel * 8 + k] * x[k];
        }
        z[channel] = (g / (1 + std::exp(-g))) * u;
    }
    // Column-parallel ranks consume disjoint up/gate rows without all-gather;
    // row projection consumes those same locally produced channels.
    for (int output = 0; output < 8; ++output) {
        double reference = x[output], split = x[output], absolute_scale = std::abs(x[output]);
        for (int c = 0; c < 256; ++c) {
            const double product = down.logical[output * 256 + c] * z[c];
            reference += product; absolute_scale += std::abs(product);
        }
        for (int rank = 0; rank < 2; ++rank) {
            double partial = 0;
            for (int c = rank * 128; c < (rank + 1) * 128; ++c) {
                double u = 0, g = 0;
                for (int k = 0; k < 8; ++k) {
                    u += up.packed.at(c, k) * x[k]; g += gate.packed.at(c, k) * x[k];
                }
                partial += down.packed.at(output, c) * (g / (1 + std::exp(-g))) * u;
            }
            split += partial;
        }
        // 512 FP64 rounding units cover the 256-term differing sum order;
        // nonlinear inputs are identical represented values in this identity test.
        close(split, reference, 512 * std::numeric_limits<double>::epsilon() * absolute_scale,
              "TP column/nonlinear/row composition");
    }
}

// Direct FP64 -> BF16 RNE: no intermediate FP32 double-rounding. The finite
// fixtures stay in BF16 range; subnormal spacing is explicitly included.
double bf16(double value) {
    require(std::isfinite(value), "nonfinite BF16 fixture");
    if (value == 0) return value;
    int exponent = 0;
    std::frexp(std::abs(value), &exponent);
    const double step = std::ldexp(1.0, std::max(exponent - 8, -133));
    const double scaled = std::abs(value) / step, lower = std::floor(scaled);
    const double fraction = scaled - lower;
    const double rounded = lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2) != 0));
    return std::copysign(rounded * step, value);
}

// Consume independently derived full/global and rank-local mixer channels.
// Both are represented BF16 at the mixer-output/gating boundaries. Each rank
// retains FP32 partials; only SUM + once-only residual is rounded to BF16.
void mixer_projection(const std::vector<double>& reference_channels,
                      const std::vector<double>& rank_channels, const char* label) {
    require(bf16(1 + std::ldexp(1.0, -8)) == 1, "BF16 tie to even down");
    require(bf16(1 + 3 * std::ldexp(1.0, -8)) == 1 + std::ldexp(1.0, -6), "BF16 tie to even up");
    require(bf16(std::ldexp(1.0, -134)) == 0, "BF16 subnormal tie to zero");
    const int k = int(reference_channels.size());
    require(k == 6144 && rank_channels.size() == reference_channels.size(), "real mixer output width");
    const auto weights = fixture(5, 4, k, true); // sampled hidden output rows
    for (int row = 0; row < 4; ++row) {
        const double residual = row % 2 ? -3.5 : 7.0;
        double expected = residual, absolute_scale = std::abs(residual);
        for (int col = 0; col < k; ++col) {
            require(rank_channels[col] == reference_channels[col], label);
            const double product = weights.logical[row * k + col] * reference_channels[col];
            expected += product; absolute_scale += std::abs(product);
        }
        const auto left = dot(weights.packed, row, rank_channels, 0, k / 2);
        const auto right = dot(weights.packed, row, rank_channels, k / 2, k);
        const float sum = fadd(left.rounded, right.rounded);
        const float with_residual = fadd(sum, float(residual));
        const double arithmetic_budget = gamma(k / 2 + 2) * (left.absolute_scale + right.absolute_scale) +
            gamma(1) * (std::abs(double(left.rounded)) + std::abs(double(right.rounded))) +
            gamma(1) * (std::abs(double(sum)) + std::abs(residual)) +
            4 * k * std::numeric_limits<double>::epsilon() * absolute_scale;
        close(with_residual, expected, arithmetic_budget, label);
        // Final BF16 storage error is separate from FP32 local/SUM error. The
        // oracle is the unsplit FP64 equation, not another emulated reduction.
        const double stored = bf16(with_residual);
        const double cast_budget = std::abs(stored - double(with_residual));
        close(stored, expected, arithmetic_budget + cast_budget, label);
        // Residual duplication must be rejected before a coarse BF16 cast can
        // hide it. Every fixture row deliberately has residual above the budget.
        const float duplicate = fadd(fadd(left.rounded, float(residual)),
                                     fadd(right.rounded, float(residual)));
        require(std::abs(double(duplicate) - expected) > arithmetic_budget,
                "full mixer residual-duplication sensitivity");
    }
}

// Stable FP64 attention for one complete query head. The full oracle below is
// deliberately indexed by global Q/KV identities; ranks use their local KV arrays.
std::vector<double> attention(const std::vector<double>& q, const std::vector<double>& k,
                              const std::vector<double>& v, int tokens, int dim) {
    std::vector<double> logits(tokens), result(dim);
    for (int t = 0; t < tokens; ++t)
        for (int d = 0; d < dim; ++d) logits[t] += q[d] * k[t * dim + d] / std::sqrt(double(dim));
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double denominator = 0;
    for (double& score : logits) { score = std::exp(score - maximum); denominator += score; }
    for (int t = 0; t < tokens; ++t) for (int d = 0; d < dim; ++d)
        result[d] += logits[t] / denominator * v[t * dim + d];
    return result;
}
void gqa_contract() {
    constexpr int dim = 256;
    for (int tokens : {1, 2, 17, 128}) {
        std::array<std::vector<double>, 24> q;
        std::array<std::vector<double>, 4> k, v;
        for (int h = 0; h < 24; ++h) {
            q[h].resize(dim);
            for (int d = 0; d < dim; ++d) q[h][d] = ((h + 3 * d) % 19 - 9) * 0.0625;
        }
        for (int h = 0; h < 4; ++h) {
            k[h].resize(tokens * dim); v[h].resize(tokens * dim);
            for (int i = 0; i < tokens * dim; ++i) {
                k[h][i] = ((i + h * 7) % 23 - 11) * 0.03125;
                v[h][i] = ((i * 3 + h * 11) % 29 - 14) * 0.125;
            }
        }
        std::array<int, 4> owners{};
        std::vector<double> reference_channels(24 * dim), rank_channels(24 * dim);
        for (int rank = 0; rank < 2; ++rank) for (int local_kv = 0; local_kv < 2; ++local_kv) {
            const int global_kv = rank * 2 + local_kv;
            for (int within = 0; within < 6; ++within) {
                const int local_q = local_kv * 6 + within, global_q = rank * 12 + local_q;
                ++owners[global_kv];
                const auto actual = attention(q[global_q], k[global_kv], v[global_kv], tokens, dim);
                // Independent dense global GQA equation with explicit softmax.
                std::vector<double> scores(tokens); double denominator = 0;
                for (int t = 0; t < tokens; ++t) {
                    for (int d = 0; d < dim; ++d)
                        scores[t] += q[global_q][d] * k[global_q / 6][t * dim + d];
                    scores[t] = std::exp(scores[t] / 16); denominator += scores[t];
                }
                for (int d = 0; d < dim; ++d) {
                    double expected = 0;
                    for (int t = 0; t < tokens; ++t)
                        expected += scores[t] * v[global_q / 6][t * dim + d] / denominator;
                    close(actual[d], expected, 2e-13, "whole-head GQA mapping");
                    const double gate = ((global_q * 7 + d) % 25 - 12) * 0.125;
                    // Attention stores BF16, then sigmoid_mul consumes BF16
                    // gate and activation and stores another BF16 boundary.
                    reference_channels[global_q * dim + d] = bf16(bf16(expected) / (1 + std::exp(-gate)));
                    const double sigmoid = 1 / (1 + std::exp(-gate));
                    rank_channels[(rank * 12 + local_q) * dim + d] = bf16(bf16(actual[d]) * sigmoid);
                }
            }
        }
        for (int count : owners) require(count == 6, "six complete Q heads per KV head");
        mixer_projection(reference_channels, rank_channels, "attention gate/projection/SUM/residual");
    }
}

void gdn_contract() {
    // Full real head counts and dimension; each state belongs to one V head.
    constexpr int dim = 128, heads = 48;
    std::vector<double> global(heads * dim * dim), local(global.size());
    for (std::size_t i = 0; i < global.size(); ++i)
        global[i] = double(float((int(i % 17) - 8) * 0.000125));
    local = global;
    for (int token = 0; token < 3; ++token) {
        std::array<std::vector<double>, 16> q, k;
        for (int h = 0; h < 16; ++h) {
            q[h].resize(dim); k[h].resize(dim); double qn = 0, kn = 0;
            for (int d = 0; d < dim; ++d) {
                q[h][d] = ((d + h + token) % 13 - 6) * 0.125;
                k[h][d] = ((3 * d + h + token) % 19 - 9) * 0.0625;
                qn += q[h][d] * q[h][d]; kn += k[h][d] * k[h][d];
            }
            for (int d = 0; d < dim; ++d) { q[h][d] /= std::sqrt(qn); k[h][d] /= std::sqrt(kn); }
        }
        for (int h = 0; h < heads; ++h) {
            const int qh = h / 3;
            const double decay = std::exp(-0.125 * (1 + h % 3)), beta = 0.25 + (h % 4) * 0.125;
            for (int d = 0; d < dim; ++d) {
                const std::size_t base = std::size_t(h * dim + d) * dim;
                const double value = ((d + 7 * h + token) % 23 - 11) * 0.0625;
                double sk = 0;
                for (int j = 0; j < dim; ++j) { global[base + j] *= decay; sk += global[base + j] * k[qh][j]; }
                for (int j = 0; j < dim; ++j) global[base + j] += beta * (value - sk) * k[qh][j];
            }
        }
        // Rank-local recurrence uses un-decayed Sk and fuses decay into the update.
        // This is an algebraically independent ordering, not the full oracle wrapper.
        for (int rank = 0; rank < 2; ++rank) for (int lv = 0; lv < 24; ++lv) {
            const int h = rank * 24 + lv, qh = rank * 8 + lv / 3;
            require(qh == h / 3, "three complete V heads per Q/K head");
            const double decay = std::exp(-0.125 * (1 + h % 3)), beta = 0.25 + (h % 4) * 0.125;
            for (int d = 0; d < dim; ++d) {
                const std::size_t base = std::size_t(h * dim + d) * dim;
                double projection = 0;
                for (int j = 0; j < dim; ++j) projection += local[base + j] * k[qh][j];
                const double value = ((d + 7 * h + token) % 23 - 11) * 0.0625;
                double actual_output = 0, expected_output = 0;
                for (int j = 0; j < dim; ++j) {
                    local[base + j] = decay * local[base + j] + beta * (value - decay * projection) * k[qh][j];
                    close(local[base + j], global[base + j], 2e-15, "GDN isolated per-head state");
                    actual_output += local[base + j] * q[qh][j];
                    expected_output += global[base + j] * q[h / 3][j];
                }
                close(actual_output / std::sqrt(double(dim)), expected_output / std::sqrt(double(dim)),
                      2e-15, "GDN head output");
            }
        }
        std::vector<double> reference_channels(heads * dim), rank_channels(heads * dim);
        for (int h = 0; h < heads; ++h) {
            std::vector<double> raw(dim), rank_raw(dim);
            double square_sum = 0, rank_square_sum = 0;
            for (int d = 0; d < dim; ++d) {
                const std::size_t base = std::size_t(h * dim + d) * dim;
                double expected = 0, actual = 0;
                for (int j = 0; j < dim; ++j) expected += global[base + j] * q[h / 3][j];
                for (int j = dim - 1; j >= 0; --j) actual += local[base + j] * q[h / 3][j];
                raw[d] = bf16(expected / std::sqrt(double(dim)));
                rank_raw[d] = bf16(actual / std::sqrt(double(dim)));
                square_sum += raw[d] * raw[d];
            }
            // Independent ordering for rank-local head RMS reduction.
            for (int d = dim - 1; d >= 0; --d) rank_square_sum += rank_raw[d] * rank_raw[d];
            const double epsilon = double(1e-6f);
            const double inverse = 1 / std::sqrt(square_sum / dim + epsilon);
            const double rank_inverse = 1 / std::sqrt(rank_square_sum / dim + epsilon);
            for (int d = 0; d < dim; ++d) {
                const double weight = 1 + (d % 7) * 0.0625; // represented BF16, no unit offset
                const double z = ((h * 3 + d + token) % 21 - 10) * 0.125;
                reference_channels[h * dim + d] = bf16(raw[d] * inverse * weight * (z / (1 + std::exp(-z))));
                const int rank = h / 24, local_v = h % 24;
                rank_channels[(rank * 24 + local_v) * dim + d] =
                    bf16((rank_raw[d] * weight * rank_inverse * z) / (1 + std::exp(-z)));
            }
        }
        mixer_projection(reference_channels, rank_channels, "GDN gated norm/projection/SUM/residual");
        // Persistent recurrence state crosses the declared FP32 boundary before
        // the next token; neither path carries hidden FP64 state across tokens.
        for (std::size_t i = 0; i < global.size(); ++i) {
            global[i] = double(float(global[i])); local[i] = double(float(local[i]));
        }
    }
}

// Tiny nonlinear physical blocks distinguish order, duplication and omission.
// This is a PP composition identity fixture, not a Qwen decoder implementation.
std::vector<double> block(std::vector<double> x, int physical) {
    std::vector<double> out(x.size());
    double ss = 0; for (double value : x) ss += value * value;
    const double inverse = 1 / std::sqrt(ss / x.size() + 1e-6);
    for (std::size_t d = 0; d < x.size(); ++d)
        out[d] = x[d] + std::tanh(x[(d + 1) % x.size()] * inverse + physical * 0.003) *
                 (physical % 4 == 3 ? 0.125 : -0.0625);
    return out;
}
void pipeline_contract() {
    const std::vector<double> input{1, -1, 0, 0.125, -0.25, 2, -2, 0.5};
    auto expected = input;
    for (int layer = 0; layer < 64; ++layer) expected = block(expected, layer);
    for (int cut : {1, 3, 4, 31, 36, 63}) {
        auto stage0 = input;
        for (int layer = 0; layer < cut; ++layer) stage0 = block(stage0, layer);
        auto stage1 = stage0;
        for (int layer = cut; layer < 64; ++layer) stage1 = block(stage1, layer);
        require(stage1 == expected, "contiguous PP physical blocks must compose exactly");
    }
    auto wrong = input;
    for (int layer = 63; layer >= 0; --layer) wrong = block(wrong, layer);
    require(wrong != expected, "PP order negative control must be sensitive");
}
} // namespace
int main() {
    try {
        codec_contract(); projection_contract(); gqa_contract(); gdn_contract(); pipeline_contract();
        std::cout << "PASS: independent CPU codec, TP2 projection/SUM/residual, GQA, GDN, PP contracts\n"
                     "Scope: reference mathematics only; no production GPU or distributed route exercised.\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
