#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/tensor_projection.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
constexpr int kHidden = 5120, kIntermediate = 8704;
// Predeclared authorities: linear_swiglu_test_common.cpp A8 and tensor_projection.
// A16 authority is {.0033,.005,.0063}; this test does NOT label private A8 as A16.
constexpr ReductionCriterion kGateCriterion{.08, .01, .12};
constexpr ReductionCriterion kDownCriterion{.04, .02, .05};
// A16 decode profile (T<=32): the registered A16 LinearSwiGLU authority and tensor_projection A16.
constexpr ReductionCriterion kGateA16Criterion{.0033, .005, .0063};
constexpr ReductionCriterion kDownA16Criterion{1.e-5, 1.e-4, 1.e-5};

struct Packed {
    int n, k, bits;
    std::size_t high, scales;
    std::vector<std::uint8_t> bytes;

    Packed(int rows, int width, int precision)
        : n(rows), k(width), bits(precision), high(std::size_t(n) * k / 2),
          scales(high + (bits == 5 ? std::size_t(n) * k / 8 : 0)),
          bytes(scales + std::size_t(n) * k / 64 * 2) {
        for (int r = 0; r < n; ++r) {
            for (int g = 0; g < k / 64; ++g) {
                const auto group = std::size_t(r) * (k / 64) + g;
                const std::uint16_t scale =
                    g % 29 == 0 ? 3 : std::uint16_t(0x2000 + ((r * 31 + g * 17) & 0x7ff));
                bytes[scales + 2 * group]     = scale & 255;
                bytes[scales + 2 * group + 1] = scale >> 8;
                for (int lane = 0; lane < 64; ++lane) {
                    std::uint32_t z = std::uint32_t(r * k + g * 64 + lane) + 0x9e3779b9U;
                    z ^= z >> 16;
                    z *= 0x7feb352dU;
                    z ^= z >> 15;
                    unsigned code = z & ((1U << bits) - 1);
                    bytes[group * 32 + lane / 2] |= (code & 15) << (4 * (lane % 2));
                    if (bits == 5) bytes[high + group * 8 + lane / 8] |= (code >> 4) << (lane % 8);
                }
            }
        }
    }

    Weight weight(void* p) const {
        Weight w{};
        w.payload = w.qdata = p;
        w.payload_bytes     = bytes.size();
        w.qhigh             = bits == 5 ? static_cast<std::uint8_t*>(p) + high : nullptr;
        w.high_plane_bytes  = scales - high;
        w.scales            = static_cast<std::uint8_t*>(p) + scales;
        w.qtype             = bits == 4 ? QType::Q4G64_F16S : QType::Q5G64_F16S;
        w.layout            = QuantLayout::RowSplit;
        w.scale_dtype       = DType::FP16;
        w.group_size = w.group = 64;
        w.ndim                 = 2;
        w.n = w.shape[0] = w.padded_shape[0] = n;
        w.k = w.shape[1] = w.padded_shape[1] = k;
        return w;
    }

    double decode(int r, int c) const {
        // Independently decode stored words; never call a production codec or quantize A8.
        const auto g   = std::size_t(r) * (k / 64) + c / 64;
        const int lane = c % 64;
        unsigned code  = (bytes[g * 32 + lane / 2] >> (4 * (lane % 2))) & 15;
        if (bits == 5) code |= ((bytes[high + g * 8 + lane / 8] >> (lane % 8)) & 1) << 4;
        int signed_code   = code >= (1U << (bits - 1)) ? int(code) - (1 << bits) : int(code);
        unsigned word     = bytes[scales + 2 * g] | (unsigned(bytes[scales + 2 * g + 1]) << 8);
        unsigned exponent = (word >> 10) & 31, fraction = word & 1023;
        double scale = exponent == 0 ? std::ldexp(double(fraction), -24)
                                     : std::ldexp(double(1024 + fraction), int(exponent) - 25);
        return signed_code * ((word & 32768) ? -scale : scale);
    }
};

std::vector<int> samples(int n, bool full) {
    std::vector<int> rows;
    if (full) {
        rows.resize(n);
        std::iota(rows.begin(), rows.end(), 0);
        return rows;
    }
    rows = {0, n - 1};
    for (int r : {1, 3, 7, 31, 32, 63, 64, 127, 128, 255, 256, 4095, 4096})
        if (r < n) rows.push_back(r);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

int run_case(const Packed& gate, const Packed& down, int t, bool a16 = false) {
    const std::string label = std::string(a16 ? "tensor_mlp A16 T=" : "tensor_mlp T=") + std::to_string(t);
    const auto policy = a16 ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA8;
    // Existing SwiGLU policy: token zero dense, later tokens sparse. Token values are
    // independent of T. Exact sparse zeros accelerate only the independent host oracle.
    std::vector<std::uint16_t> input(std::size_t(kHidden) * t);
    for (int k = 0; k < kHidden; ++k) input[k] = f32_to_bf16(float((k * 37 % 63) - 31) * .000125f);
    for (int c = 1; c < t; ++c)
        for (int lane = 0; lane < 4; ++lane)
            input[std::size_t(c) * kHidden + (c * 67 + lane * 127) % kHidden] =
                f32_to_bf16(float(((c * 13 + lane * 7) % 31) - 15) * .015f);
    GuardedDeviceBuffer dx(input.size() * 2), dg(gate.bytes.size()), dd(down.bytes.size()),
        dz(std::size_t(kIntermediate) * t * 2), dy(std::size_t(kHidden) * t * 4);
    dx.copy_from_host(input.data(), dx.bytes());
    dg.copy_from_host(gate.bytes.data(), dg.bytes());
    dd.copy_from_host(down.bytes.data(), dd.bytes());
    dz.fill(255);
    dy.fill(255);
    auto capacity = std::max(
        ops::linear_swiglu_workspace_capacity_bytes(QType::Q4G64_F16S, 2 * kIntermediate, kHidden,
                                                    policy, 1, t),
        ops::tensor_projection_workspace_capacity_bytes(kIntermediate, 1, t));
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena ws(DeviceSpan{scratch.data(), capacity});
    Tensor x(dx.data(), DType::BF16, {kHidden, t}), z(dz.data(), DType::BF16, {kIntermediate, t}),
        y(dy.data(), DType::FP32, {kHidden, t});
    auto gw = gate.weight(dg.data()), dw = down.weight(dd.data());
    int failures = 0;
    if (a16 && t == 1) {
        try {
            (void)ops::linear_swiglu_workspace_capacity_bytes(
                QType::Q4G64_F16S, 2 * kIntermediate, kHidden, ops::LinearPolicy::A16Only, 1, 33);
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    if (!a16 && t == 1) {
        auto rejected = [&](auto operation) {
            try {
                operation();
                ++failures;
            } catch (const std::invalid_argument&) {}
        };
        rejected([&] {
            (void)ops::linear_swiglu_workspace_capacity_bytes(QType::Q4G64_F16S,
                                                              2 * kIntermediate - 128, kHidden,
                                                              ops::LinearPolicy::AllowA8, 1, 1);
        });
    }
    ops::linear_swiglu(x, gw, z, policy, ws, nullptr);
    cuda_synchronize();
    auto gated = from_device<std::uint16_t>(dz.data(), std::size_t(kIntermediate) * t);
    failures += ws.used() != 0;
    ops::tensor_projection(z, dw, y, policy, ws, nullptr);
    cuda_synchronize();
    auto output = from_device<float>(dy.data(), std::size_t(kHidden) * t);
    failures += ws.used() != 0;
    for (auto v : gated)
        if (!std::isfinite(bf16_to_f32(v))) {
            ++failures;
            break;
        }
    for (auto v : output)
        if (!std::isfinite(v)) {
            ++failures;
            break;
        }
    const auto cols = samples(t, t <= 4), rows = samples(kHidden, t <= 4);
    std::vector<double> gate_actual, gate_ideal, down_actual, down_ideal, composed_ideal;
    for (int c : cols) {
        std::vector<double> ideal_gated(kIntermediate);
        std::vector<std::pair<int, double>> active;
        for (int k = 0; k < kHidden; ++k) {
            const double v = bf16_to_f32(input[std::size_t(c) * kHidden + k]);
            if (v != 0) active.emplace_back(k, v);
        }
        // Full intermediate rows are needed even when down output is sampled.
        for (int r = 0; r < kIntermediate; ++r) {
            double g = 0, u = 0;
            for (auto [k, v] : active) {
                g += gate.decode(r, k) * v;
                u += gate.decode(kIntermediate + r, k) * v;
            }
            const double e     = std::exp(-std::abs(g));
            const double ideal = (g >= 0 ? g / (1 + e) : g * e / (1 + e)) * u;
            gate_actual.push_back(bf16_to_f32(gated[std::size_t(c) * kIntermediate + r]));
            gate_ideal.push_back(ideal);
            // This is the observable gated BF16 boundary, not private A8 rounding.
            ideal_gated[r] = bf16_to_f32(f32_to_bf16(float(ideal)));
        }
        for (int r : rows) {
            double staged = 0, composed = 0;
            for (int k = 0; k < kIntermediate; ++k) {
                const double weight = down.decode(r, k);
                staged += weight * double(bf16_to_f32(gated[std::size_t(c) * kIntermediate + k]));
                composed += weight * ideal_gated[k];
            }
            down_actual.push_back(output[std::size_t(c) * kHidden + r]);
            down_ideal.push_back(staged);
            composed_ideal.push_back(composed);
        }
    }
    failures += verify_reduction(label + " gate", gate_actual, gate_ideal, a16 ? kGateA16Criterion : kGateCriterion);
    failures += verify_reduction(label + " down", down_actual, down_ideal, a16 ? kDownA16Criterion : kDownCriterion);
    // Diagnostic only: no aggregate tolerance is invented from the two Op criteria.
    const auto stats = compute_reduction_stats(down_actual.data(), composed_ideal.data(),
                                               static_cast<std::int64_t>(down_actual.size()));
    std::cout << label << " composed diagnostic relative_l2=" << stats.relative_l2
              << " max_abs=" << stats.maximum_absolute_error << '\n';
    if (from_device<std::uint16_t>(dx.data(), input.size()) != input) ++failures;
    if (from_device<std::uint16_t>(dz.data(), gated.size()) != gated) ++failures;
    if (from_device<std::uint8_t>(dg.data(), gate.bytes.size()) != gate.bytes) ++failures;
    if (from_device<std::uint8_t>(dd.data(), down.bytes.size()) != down.bytes) ++failures;
    failures += dx.verify_guards(label) + dg.verify_guards(label) + dd.verify_guards(label) +
                dz.verify_guards(label) + dy.verify_guards(label) + scratch.verify_guards(label);
    if (t <= 4) {
        dz.fill(0);
        dy.fill(0);
        ops::linear_swiglu(x, gw, z, policy, ws, nullptr);
        ops::tensor_projection(z, dw, y, policy, ws, nullptr);
        cuda_synchronize();
        if (from_device<std::uint16_t>(dz.data(), gated.size()) != gated) ++failures;
        if (from_device<float>(dy.data(), output.size()) != output) ++failures;
        failures += ws.used() != 0;
    }
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        Packed gate(2 * kIntermediate, kHidden, 4), down(kHidden, kIntermediate, 5);
        int failures = 0;
        for (int t : {1, 4, 127, 128, 129, 4095, 4096, 4097}) failures += run_case(gate, down, t);
        for (int t : {1, 2, 3, 8, 9, 17, 32}) failures += run_case(gate, down, t, true);
        std::cout << (failures ? "FAIL" : "OK") << " tensor_mlp stage qualification\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
