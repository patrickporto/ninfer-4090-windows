#include "ninfer/ops/tensor_projection.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
constexpr int kRows = 5120;
// Registered before GPU observations; unchanged from Q5 LinearAdd A8.
constexpr ReductionCriterion kCriterion{.04, .02, .05};
constexpr ReductionCriterion kA16Criterion{1.e-5, 1.e-4, 1.e-5}; // FP32 accumulation of exact operands

struct Fixture {
    int k;
    std::size_t high, scales;
    std::vector<std::uint8_t> bytes;

    explicit Fixture(int width)
        : k(width), high(std::size_t(kRows) * k / 2), scales(high + std::size_t(kRows) * k / 8),
          bytes(scales + std::size_t(kRows) * (k / 64) * 2) {
        for (int r = 0; r < kRows; ++r) {
            for (int g = 0; g < k / 64; ++g) {
                const auto group = std::size_t(r) * (k / 64) + g;
                // Vary stored FP16 exponent and mantissa, including subnormals.
                const std::uint16_t scale =
                    g % 29 == 0 ? 0x0003 : std::uint16_t(0x2000 + ((r * 31 + g * 17) & 0x7ff));
                bytes[scales + 2 * group]     = scale & 255;
                bytes[scales + 2 * group + 1] = scale >> 8;
                for (int lane = 0; lane < 64; ++lane) {
                    std::uint32_t z = std::uint32_t(r * 8704 + g * 64 + lane) + 0x9e3779b9U;
                    z ^= z >> 16;
                    z *= 0x7feb352dU;
                    z ^= z >> 15;
                    const unsigned code = z & 31;
                    bytes[group * 32 + lane / 2] |= (code & 15) << (4 * (lane % 2));
                    bytes[high + group * 8 + lane / 8] |= (code >> 4) << (lane % 8);
                }
            }
        }
    }

    Weight weight(void* p) const {
        Weight w{};
        w.payload          = p;
        w.payload_bytes    = bytes.size();
        w.qdata            = p;
        w.qhigh            = static_cast<std::uint8_t*>(p) + high;
        w.scales           = static_cast<std::uint8_t*>(p) + scales;
        w.high_plane_bytes = scales - high;
        w.qtype            = QType::Q5G64_F16S;
        w.layout           = QuantLayout::RowSplit;
        w.scale_dtype      = DType::FP16;
        w.group_size = w.group = 64;
        w.ndim                 = 2;
        w.n = w.shape[0] = w.padded_shape[0] = kRows;
        w.k = w.shape[1] = w.padded_shape[1] = k;
        return w;
    }

    double decode(int row, int column) const {
        // Independent host decoder: no CUDA codec or production layout helpers.
        const auto g   = std::size_t(row) * (k / 64) + column / 64;
        const int lane = column % 64;
        unsigned code  = (bytes[g * 32 + lane / 2] >> (4 * (lane % 2))) & 15;
        code |= ((bytes[high + g * 8 + lane / 8] >> (lane % 8)) & 1) << 4;
        const int signed_code = code >= 16 ? int(code) - 32 : int(code);
        const unsigned word   = bytes[scales + 2 * g] | (unsigned(bytes[scales + 2 * g + 1]) << 8);
        const unsigned exponent = (word >> 10) & 31, fraction = word & 1023;
        const double scale = exponent == 0
                                 ? std::ldexp(double(fraction), -24)
                                 : std::ldexp(double(1024 + fraction), int(exponent) - 25);
        return signed_code * ((word & 32768) ? -scale : scale);
    }
};

std::vector<int> samples(int extent) {
    std::vector<int> result{0, extent - 1};
    for (int i : {1, 3, 7, 31, 32, 63, 64, 127, 128, 255, 256, 4095, 4096})
        if (i < extent) result.push_back(i);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int run_case(const Fixture& f, int t, bool witness = false, bool a16 = false) {
    const std::string label = std::string(a16 ? "tensor_projection A16" : "tensor_projection A8") +
                              " K=" + std::to_string(f.k) + " T=" + std::to_string(t);
    const auto policy = a16 ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA8;
    std::vector<float> input(std::size_t(f.k) * t);
    fill_uniform(input, 641, -1.f, 1.f);
    round_to_bf16(input);
    if (witness) std::fill(input.begin(), input.end(), 1.f);
    std::vector<std::uint16_t> bits(input.size());
    std::transform(input.begin(), input.end(), bits.begin(), f32_to_bf16);
    GuardedDeviceBuffer dx(bits.size() * 2), dw(f.bytes.size()), dy(std::size_t(kRows) * t * 4);
    dx.copy_from_host(bits.data(), dx.bytes());
    dw.copy_from_host(f.bytes.data(), dw.bytes());
    dy.fill(255); // NaN poisoning proves full overwrite and absence of residual reads.
    const auto capacity = ops::tensor_projection_workspace_capacity_bytes(f.k, 1, t);
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena ws(DeviceSpan{scratch.data(), capacity});
    Tensor x(dx.data(), DType::BF16, {f.k, t}), y(dy.data(), DType::FP32, {kRows, t});
    const auto w = f.weight(dw.data());
    int failures = 0;
    if (t == 1 && !witness) {
        auto rejected = [&](auto operation) {
            try {
                operation();
                ++failures;
            } catch (const std::invalid_argument&) {}
        };
        rejected([&] { (void)ops::tensor_projection_workspace_capacity_bytes(6144, 1, 1); });
        rejected([&] { (void)ops::tensor_projection_workspace_capacity_bytes(f.k, 0, 1); });
        rejected([&] { (void)ops::tensor_projection_workspace_capacity_bytes(f.k, 4, 1); });
        auto wrong = w;
        wrong.padded_shape[1] += 128;
        rejected([&] { ops::tensor_projection(x, wrong, y, ws, nullptr); });
        wrong       = w;
        wrong.qtype = QType::Q4G64_F16S;
        rejected([&] { ops::tensor_projection(x, wrong, y, ws, nullptr); });
        Tensor wrong_y(y.data, DType::BF16, {kRows, t});
        rejected([&] { ops::tensor_projection(x, w, wrong_y, ws, nullptr); });
        Tensor aliased(x.data, DType::FP32, {kRows, t});
        rejected([&] { ops::tensor_projection(x, w, aliased, ws, nullptr); });
    }
    ops::tensor_projection(x, w, y, policy, ws, nullptr);
    cuda_synchronize();
    auto output = from_device<float>(dy.data(), std::size_t(kRows) * t);
    failures += ws.used() != 0;
    for (float value : output)
        if (!std::isfinite(value)) {
            ++failures;
            break;
        }
    auto rows = samples(kRows), cols = samples(t);
    if (t <= 4) {
        rows.resize(kRows);
        std::iota(rows.begin(), rows.end(), 0);
        cols.resize(t);
        std::iota(cols.begin(), cols.end(), 0);
    }
    std::vector<double> actual, ideal;
    for (int c : cols)
        for (int r : rows) {
            double sum = 0;
            for (int k = 0; k < f.k; ++k)
                sum += f.decode(r, k) * double(input[std::size_t(c) * f.k + k]);
            actual.push_back(output[std::size_t(c) * kRows + r]);
            ideal.push_back(sum);
        }
    failures += verify_reduction(label, actual, ideal, a16 ? kA16Criterion : kCriterion);
    if (witness) {
        // Private BF16 materialization would round 1+1/512 to 1 and erase the difference.
        if (std::abs(double(output[0]) - 1.001953125) > 1.e-6 ||
            std::abs(double(output[0]) + double(output[1]) - .001953125) > 1.e-6)
            ++failures;
    }
    if (from_device<std::uint16_t>(dx.data(), bits.size()) != bits) ++failures;
    if (from_device<std::uint8_t>(dw.data(), f.bytes.size()) != f.bytes) ++failures;
    failures += dx.verify_guards(label) + dw.verify_guards(label) + dy.verify_guards(label) +
                scratch.verify_guards(label);
    // Repeated invocation overwrites arbitrary output and restores transient state.
    dy.fill(0);
    ops::tensor_projection(x, w, y, policy, ws, nullptr);
    cuda_synchronize();
    if (from_device<float>(dy.data(), output.size()) != output || ws.used() != 0) ++failures;
    return failures;
}
// Exact oracle for the declared TP2 residual boundary: one FP32 RNE addition of the complete
// all-reduced sum, then one BF16 RNE rounding. Rank partials are never rounded separately.
int residual_add_case(int t) {
    const std::size_t n = std::size_t(kRows) * t;
    std::vector<std::uint16_t> residual(n), expected(n);
    std::vector<float> sum(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t z = std::uint32_t(i * 2654435761u + t * 97u);
        z ^= z >> 15; z *= 0x2c1b3c6du; z ^= z >> 12;
        float r = float(int(z % 2001) - 1000) / 64.f;
        float s = float(int((z >> 11) % 4001) - 2000) / 4096.f;
        switch (i % 7) {
        case 0: s = -r; break;                         // exact cancellation to signed zero
        case 1: r = 3.3895313892515355e38f; s = 1.f; break; // maximal finite BF16 residual
        case 2: s = r * -0.99609375f; break;           // near-cancellation, large relative error
        case 3: r = 7.f; s = 2.f + 3.f; break;         // residual added once: 7 + (2+3) = 12
        default: break;
        }
        residual[i] = f32_to_bf16(r);
        sum[i]      = s;
        expected[i] = f32_to_bf16(bf16_to_f32(residual[i]) + s);
    }
    GuardedDeviceBuffer dr(n * 2), ds(n * 4);
    dr.copy_from_host(residual.data(), dr.bytes());
    ds.copy_from_host(sum.data(), ds.bytes());
    Tensor tr(dr.data(), DType::BF16, {kRows, t}), ts(ds.data(), DType::FP32, {kRows, t});
    ops::tensor_residual_add(ts, tr, nullptr);
    cuda_synchronize();
    int failures = from_device<std::uint16_t>(dr.data(), n) != expected;
    failures += from_device<float>(ds.data(), n) != sum;
    failures += dr.verify_guards("tensor_residual_add") + ds.verify_guards("tensor_residual_add");
    auto rejected = [&](auto&& f) {
        try { f(); ++failures; } catch (const std::invalid_argument&) {}
    };
    Tensor bf16_sum(ds.data(), DType::BF16, {kRows, t});
    rejected([&] { ops::tensor_residual_add(bf16_sum, tr, nullptr); });
    Tensor narrow(dr.data(), DType::BF16, {kRows - 1, t});
    rejected([&] { ops::tensor_residual_add(ts, narrow, nullptr); });
    Tensor alias(ds.data(), DType::BF16, {kRows, t});
    rejected([&] { ops::tensor_residual_add(ts, alias, nullptr); });
    if (failures) std::cerr << "tensor_residual_add T=" << t << " failed\n";
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        int failures = 0;
        for (int k : {3072, 8704}) {
            Fixture f(k);
            for (int t : {1, 4, 127, 128, 129, 4095, 4096, 4097}) failures += run_case(f, t);
            for (int t : {1, 2, 3, 4, 5, 8, 9, 17, 32}) failures += run_case(f, t, false, true);
            {
                // A16 admits T<=32 only.
                GuardedDeviceBuffer dx(std::size_t(f.k) * 33 * 2), dy(std::size_t(kRows) * 33 * 4);
                GuardedDeviceBuffer dw(f.bytes.size());
                GuardedDeviceBuffer scratch(4096);
                WorkspaceArena ws(DeviceSpan{scratch.data(), 4096});
                Tensor x(dx.data(), DType::BF16, {f.k, 33}), y(dy.data(), DType::FP32, {kRows, 33});
                try { ops::tensor_projection(x, f.weight(dw.data()), y, ops::LinearPolicy::A16Only, ws, nullptr); ++failures; }
                catch (const std::invalid_argument&) {}
            }
            std::fill(f.bytes.begin(), f.bytes.end(), 0);
            f.bytes[0]                            = 1;
            f.bytes[32]                           = 1;
            f.bytes[f.scales]                     = 0;
            f.bytes[f.scales + 1]                 = 0x3c;
            f.bytes[f.scales + 2]                 = 0;
            f.bytes[f.scales + 3]                 = 0x18;
            const auto row_group                  = std::size_t(k / 64);
            f.bytes[row_group * 32]               = 15;
            f.bytes[f.high + row_group * 8]       = 1;
            f.bytes[f.scales + row_group * 2 + 1] = 0x3c;
            failures += run_case(f, 1, true);
            failures += run_case(f, 1, true, true);
        }
        for (int t : {1, 8, 129, 1024}) failures += residual_add_case(t);
        std::cout << (failures ? "FAIL" : "OK") << " tensor_projection\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
