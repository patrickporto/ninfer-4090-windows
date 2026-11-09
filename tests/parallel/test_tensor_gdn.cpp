#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ops/gdn_ref.h"
#include "ops/input_projection_test_common.h"

#include <array>
#include <chrono>
#include <iostream>
#include <string>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
// Inherited named Op profiles, declared before any local-shape GPU observations.
constexpr ReductionCriterion kInputA8{.04, .02, .05};
// A16 decode profile (T<=15): unchanged registered A16 criterion (tests/ops/test_gdn_input_proj.cpp).
constexpr ReductionCriterion kInputA16{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kControl{1.4e-6, 5.e-7, 2.5e-6};
constexpr ReductionCriterion kNormControl{8.e-4, 1.5e-4, 2.e-3};
constexpr ReductionCriterion kNormOutput{1.75e-3, 1.e-4, 4.e-3};
constexpr ReductionCriterion kConv{1.85e-3, 1.e-3, 3.7e-3};
constexpr ReductionCriterion kRecurrence{4.1e-3, 5.e-6, 5.5e-3};
constexpr ReductionCriterion kState{2.7e-3, 1.e-5, 3.9e-3};
constexpr ReductionCriterion kGatedNorm{1.85e-3, 4.5e-5, 2.8e-3};

std::vector<double> read_f32(const void* p, std::size_t n) {
    auto f = from_device<float>(p, n);
    return {f.begin(), f.end()};
}

std::vector<float> fixture(int rows, int t, unsigned seed, float lo, float hi) {
    std::vector<float> x(std::size_t(rows) * t);
    fill_uniform(x, seed, lo, hi);
    round_to_bf16(x);
    return x;
}

Weight control_weight(void* p) {
    Weight w{};
    w.qtype   = QType::BF16_CTRL;
    w.layout  = QuantLayout::Contiguous;
    w.payload = w.qdata = p;
    w.payload_bytes     = 24 * 5120 * 2;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = 24;
    w.k = w.shape[1] = w.padded_shape[1] = 5120;
    return w;
}

int projection_case(DevicePackedWeight& qk, DevicePackedWeight& vz, int t, bool a16 = false) {
    const std::string label = std::string(a16 ? "local input A16 T=" : "local input T=") + std::to_string(t);
    const auto policy = a16 ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA8;
    auto x                  = fixture(5120, t, 411 + t, -.01f, .01f);
    auto dx                 = to_device_bf16(x);
    GuardedBf16Tensor qkv(5120, t), z(3072, t);
    Tensor tx(dx.p, DType::BF16, {5120, t}), tqkv = qkv.tensor(), tz = z.tensor();
    const auto bytes =
        ops::gdn_input_proj_workspace_capacity_bytes(5120, policy, t, t);
    WorkspaceArena ws(std::max<std::size_t>(bytes, 256));
    ops::gdn_input_proj(tx, qk.view(), vz.view(), tqkv, tz, policy, ws, nullptr);
    cuda_synchronize();
    int failures = qkv.verify_guards(label) + z.verify_guards(label) +
                   qkv.verify_fully_written(label) + z.verify_fully_written(label);
    auto check   = [&](const GuardedBf16Tensor& out, int ld, int begin, int rows,
                       const quantized_weight::PackedWeight& weight, int weight_begin) {
        const int samples = t <= 4 ? rows : 31;
        return compare(label, gather_rows(out.values(), ld, begin, rows, t, samples),
                       projection_oracle(weight, weight_begin, rows, x, 5120, t, samples),
                       a16 ? kInputA16 : kInputA8);
    };
    failures += check(qkv, 5120, 0, 2048, qk.host, 0);
    failures += check(qkv, 5120, 2048, 3072, vz.host, 0);
    failures += check(z, 3072, 0, 3072, vz.host, 3072);
    failures += verify_preserved(label, dx, bf16_bits(x));
    failures += qk.verify_preserved(label) + vz.verify_preserved(label);
    return failures + (ws.used() != 0);
}

int control_case(int t, bool norm) {
    const std::string label =
        std::string(norm ? "local norm-control T=" : "local control T=") + std::to_string(t);
    auto x  = fixture(5120, t, 713, -1.f, 1.f);
    // Ranges match the frozen authority fixture in tests/ops/test_gdn_gating_proj.cpp. The norm
    // budget includes propagated BF16 staging of h, calibrated for |w| <= 0.015; the composed
    // 48-head production route shares that staging, so wider weights are not a local-shard test.
    auto a  = fixture(24, 5120, 719, -.015f, .015f);
    auto b  = fixture(24, 5120, 727, -.015f, .015f);
    auto nw = fixture(5120, 1, 731, -.2f, .2f);
    std::vector<float> alog(24), dt(24);
    fill_uniform(alog, 733, -2.f, 1.f);
    fill_uniform(dt, 739, -1.f, 1.f);
    auto dx = to_device_bf16(x), da = to_device_bf16(a), db = to_device_bf16(b);
    auto dn = to_device_bf16(nw), dl = to_device_f32(alog), dd = to_device_f32(dt);
    GuardedDeviceBuffer dg(std::size_t(24) * t * 4), dbe(std::size_t(24) * t * 4);
    GuardedBf16Tensor dh(5120, t);
    dg.fill(255);
    dbe.fill(255);
    Tensor tx(dx.p, DType::BF16, {5120, t}), tn(dn.p, DType::BF16, {5120});
    Tensor tl(dl.p, DType::FP32, {24}), td(dd.p, DType::FP32, {24});
    Tensor tg(dg.data(), DType::FP32, {24, t}), tb(dbe.data(), DType::FP32, {24, t});
    Tensor th        = dh.tensor();
    const auto bytes = norm ? ops::gdn_norm_gating_proj_workspace_capacity_bytes(24, 5120, t, t)
                            : ops::gdn_gating_proj_workspace_capacity_bytes(24, 5120, t, t);
    WorkspaceArena ws(std::max<std::size_t>(bytes, 256));
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    const DeviceExecutionView execution{nullptr, prop.multiProcessorCount};
    if (norm)
        ops::gdn_norm_gating_proj(tx, tn, 1.e-6f, control_weight(da.p), control_weight(db.p), tl,
                                  td, ws, th, tg, tb, execution);
    else
        ops::gdn_gating_proj(tx, control_weight(da.p), control_weight(db.p), tl, td, ws, tg, tb,
                             execution);
    cuda_synchronize();
    std::vector<double> gr(24 * t), br(24 * t), nr(x.size());
    for (int c = 0; c < t; ++c) {
        double inv = 1.;
        if (norm) {
            double s = 0;
            for (int k = 0; k < 5120; ++k) {
                double v = x[c * 5120 + k];
                s += v * v;
            }
            inv = 1 / std::sqrt(s / 5120 + double(1.e-6f));
        }
        for (int k = 0; k < 5120; ++k)
            nr[c * 5120 + k] = double(x[c * 5120 + k]) * inv * (norm ? 1. + double(nw[k]) : 1.);
        for (int h = 0; h < 24; ++h) {
            double av = 0, bv = 0;
            for (int k = 0; k < 5120; ++k) {
                av += double(a[h * 5120 + k]) * nr[c * 5120 + k];
                bv += double(b[h * 5120 + k]) * nr[c * 5120 + k];
            }
            const double u = av + dt[h];
            gr[c * 24 + h] =
                -std::exp(double(alog[h])) * (std::max(u, 0.) + std::log1p(std::exp(-std::abs(u))));
            br[c * 24 + h] = 1. / (1. + std::exp(-bv));
        }
    }
    int failures =
        compare(label, read_f32(dg.data(), gr.size()), gr, norm ? kNormControl : kControl) +
        compare(label, read_f32(dbe.data(), br.size()), br, norm ? kNormControl : kControl);
    if (norm) failures += compare(label, dh.values(), nr, kNormOutput) + dh.verify_guards(label);
    failures += dg.verify_guards(label) + dbe.verify_guards(label);
    failures += verify_preserved(label, dx, bf16_bits(x)) +
                verify_preserved(label, da, bf16_bits(a)) +
                verify_preserved(label, db, bf16_bits(b));
    return failures + (ws.used() != 0);
}

// Cross-architecture/local-geometry contract (user-approved 2026-11-09): the frozen relative-L2
// budget is mandatory, while the frozen worst-element bound tolerates isolated outliers. A seed sweep
// showed the already-qualified 16QK/48V chunked route exceeds the worst-element bound by about one
// BF16 ULP on 2/10 random seeds (8QK/24V: 4/10), always on a single element, with relative L2 at ~92%
// of budget. At most 1e-5 of elements may exceed the bound, and none may exceed twice the bound.
int compare_with_isolated_outliers(const std::string& label, const std::vector<double>& actual,
                                   const std::vector<double>& expected, const ReductionCriterion& c) {
    if (actual.empty() || actual.size() != expected.size()) return compare(label, actual, expected, c);
    const auto stats = compute_reduction_stats(actual.data(), expected.data(),
                                               static_cast<std::int64_t>(actual.size()));
    report_reduction_stats(label, static_cast<std::int64_t>(actual.size()), stats, c);
    if (stats.first_non_finite >= 0 || stats.relative_l2 > c.relative_l2)
        return compare(label, actual, expected, c);
    const double limit = gross_error_limit(stats, c);
    std::size_t outliers = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::abs(actual[i] - expected[i]) > limit) ++outliers;
    const bool ok = stats.maximum_absolute_error <= 2 * limit &&
                    double(outliers) <= 1.e-5 * double(actual.size());
    if (outliers)
        std::cout << label << ": isolated outliers=" << outliers << "/" << actual.size()
                  << " max_abs=" << stats.maximum_absolute_error << " limit=" << limit
                  << (ok ? " (within contract)" : " (REJECTED)") << '\n';
    if (!ok) std::cerr << label << ": outlier contract failed\n";
    return ok ? 0 : 1;
}

int recurrence_case(int t, bool timing = false) {
    const std::string label = "local recurrence T=" + std::to_string(t);
    gdn_ref::Inputs in;
    in.head_dim    = 128;
    in.qk_heads    = 8;
    in.value_heads = 24;
    in.tokens      = t;
    in.q           = fixture(1024, t, 101 + t, -1.f, 1.f);
    in.k           = fixture(1024, t, 103 + t, -1.f, 1.f);
    in.v           = fixture(3072, t, 107 + t, -.5f, .5f);
    in.g.resize(24 * t);
    in.beta.resize(24 * t);
    in.state.resize(128 * 128 * 24);
    fill_uniform(in.g, 109 + t, -.1f, -.005f);
    fill_uniform(in.beta, 113 + t, .05f, .95f);
    fill_uniform(in.state, 127 + t, -.02f, .02f);
    const float scale = 1.f / std::sqrt(128.f);
    auto ref          = gdn_ref::evaluate(in, double(scale), true);
    auto dq = to_device_bf16(in.q), dk = to_device_bf16(in.k), dv = to_device_bf16(in.v);
    auto dg = to_device_f32(in.g), db = to_device_f32(in.beta);
    // Three-slot pool verifies that only the selected local state region changes.
    std::vector<float> pool(in.state.size() * 3, .125f);
    std::copy(in.state.begin(), in.state.end(), pool.begin() + in.state.size());
    GuardedDeviceBuffer state(pool.size() * 4), out(in.v.size() * 2);
    state.copy_from_host(pool.data(), state.bytes());
    out.fill(255);
    Tensor q(dq.p, DType::BF16, {128, 8, t}), k(dk.p, DType::BF16, {128, 8, t});
    Tensor v(dv.p, DType::BF16, {128, 24, t}), g(dg.p, DType::FP32, {24, t}),
        b(db.p, DType::FP32, {24, t});
    Tensor s(static_cast<float*>(state.data()) + in.state.size(), DType::FP32, {128, 128, 24});
    Tensor o(out.data(), DType::BF16, {128, 24, t});
    const auto bytes = ops::gated_delta_net_workspace_capacity_bytes(8, 24, true, t, t);
    WorkspaceArena ws(std::max<std::size_t>(bytes, 256));
    ops::gated_delta_net(q, k, v, g, b, scale, true, ws, s, o, nullptr);
    cuda_synchronize();
    int failures =
        compare_with_isolated_outliers(label, from_device_bf16(out.data(), in.v.size()), ref.out,
                                       kRecurrence) +
        compare_with_isolated_outliers(label, read_f32(s.data, in.state.size()), ref.final_state,
                                       kState);
    auto after   = from_device<float>(state.data(), pool.size());
    for (std::size_t i = 0; i < pool.size(); ++i)
        if ((i < in.state.size() || i >= 2 * in.state.size()) && after[i] != pool[i]) {
            ++failures;
            break;
        }
    failures += state.verify_guards(label) + out.verify_guards(label);
    failures += verify_preserved(label, dq, bf16_bits(in.q)) +
                verify_preserved(label, dk, bf16_bits(in.k)) +
                verify_preserved(label, dv, bf16_bits(in.v));
    if (timing && failures == 0) {
        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<float> elapsed;
        for (int repetition = 0; repetition < 12; ++repetition) {
            // Reset outside the timing interval: every trial sees the qualified state.
            state.copy_from_host(pool.data(), state.bytes());
            CUDA_CHECK(cudaEventRecord(start));
            ops::gated_delta_net(q, k, v, g, b, scale, true, ws, s, o, nullptr);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (repetition >= 2) elapsed.push_back(ms * 1000);
        }
        std::sort(elapsed.begin(), elapsed.end());
        std::cout << label << " warm public-Op median_us=" << (elapsed[4] + elapsed[5]) * .5f
                  << " workspace_bytes=" << bytes << '\n';
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    return failures + (ws.used() != 0);
}

int conv_case(int t, bool split) {
    const std::string label = "local conv T=" + std::to_string(t) + (split ? " split" : "");
    auto x = fixture(5120, t, 811 + t, -.5f, .5f), w = fixture(5120, 4, 821, -.5f, .5f);
    auto history = fixture(5120, 3, 823, -.5f, .5f);
    auto dx = to_device_bf16(x), dw = to_device_bf16(w), ds = to_device_bf16(history);
    GuardedBf16Tensor next(5120, 3), out(5120, t), q(1024, t), k(1024, t), v(3072, t);
    Tensor tx(dx.p, DType::BF16, {5120, t}), tw(dw.p, DType::BF16, {5120, 4}),
        ts(ds.p, DType::BF16, {5120, 3});
    Tensor tn = next.tensor(), to = out.tensor(), tq = q.tensor(), tk = k.tensor(), tv = v.tensor();
    if (split)
        ops::causal_conv1d_silu_split(tx, tw, ts, tn, tq, tk, tv, nullptr);
    else
        ops::causal_conv1d_silu(tx, tw, ts, tn, to, nullptr);
    cuda_synchronize();
    auto u = [&](int c, int token) -> double {
        return token < 0 ? history[(token + 3) * 5120 + c] : x[token * 5120 + c];
    };
    std::vector<double> ideal(x.size());
    std::vector<std::uint16_t> expected_state(5120 * 3);
    for (int token = 0; token < t; ++token)
        for (int c = 0; c < 5120; ++c) {
            double sum = 0;
            for (int j = 0; j < 4; ++j) sum += double(w[j * 5120 + c]) * u(c, token - 3 + j);
            ideal[token * 5120 + c] = sum / (1 + std::exp(-sum));
        }
    for (int j = 0; j < 3; ++j)
        for (int c = 0; c < 5120; ++c)
            expected_state[j * 5120 + c] = f32_to_bf16(float(u(c, t - 3 + j)));
    auto actual = out.values();
    if (split) {
        const auto aq = q.values(), ak = k.values(), av = v.values();
        for (int token = 0; token < t; ++token) {
            std::copy_n(aq.begin() + token * 1024, 1024, actual.begin() + token * 5120);
            std::copy_n(ak.begin() + token * 1024, 1024, actual.begin() + token * 5120 + 1024);
            std::copy_n(av.begin() + token * 3072, 3072, actual.begin() + token * 5120 + 2048);
        }
    }
    int failures = compare(label, actual, ideal, kConv) + (next.bits() != expected_state);
    failures += next.verify_guards(label) + out.verify_guards(label) + q.verify_guards(label) +
                k.verify_guards(label) + v.verify_guards(label);
    failures += verify_preserved(label, dx, bf16_bits(x)) +
                verify_preserved(label, dw, bf16_bits(w)) +
                verify_preserved(label, ds, bf16_bits(history));
    return failures;
}

int gated_norm_case(int t) {
    const std::string label = "local gated norm T=" + std::to_string(t);
    auto x = fixture(3072, t, 911 + t, -.5f, .5f), z = fixture(3072, t, 919 + t, -5.f, 5.f),
         w  = fixture(128, 1, 929, .25f, 1.75f); // authority test_gated_rmsnorm.cpp ranges
    auto dx = to_device_bf16(x), dz = to_device_bf16(z), dw = to_device_bf16(w);
    GuardedBf16Tensor out(3072, t);
    Tensor tx(dx.p, DType::BF16, {128, 24, t}), tz(dz.p, DType::BF16, {128, 24, t}),
        tw(dw.p, DType::BF16, {128});
    Tensor to(out.data(), DType::BF16, {128, 24, t});
    ops::gated_rmsnorm(tx, tw, tz, 1.e-6f, to, nullptr);
    cuda_synchronize();
    std::vector<double> ideal(x.size());
    for (int row = 0; row < 24 * t; ++row) {
        double sum = 0;
        for (int d = 0; d < 128; ++d) {
            double a = x[row * 128 + d];
            sum += a * a;
        }
        const double inv = 1 / std::sqrt(sum / 128 + double(1.e-6f));
        for (int d = 0; d < 128; ++d) {
            const auto i = row * 128 + d;
            double gate  = z[i];
            ideal[i]     = double(x[i]) * inv * double(w[d]) * gate / (1 + std::exp(-gate));
        }
    }
    return compare(label, out.values(), ideal, kGatedNorm) + out.verify_guards(label) +
           verify_preserved(label, dx, bf16_bits(x)) + verify_preserved(label, dz, bf16_bits(z)) +
           verify_preserved(label, dw, bf16_bits(w));
}

int conv_snapshot_case() {
    constexpr int c = 5120, width = 4, batch = 2, slots = 12;
    const std::string label = "local conv snapshots mixed prefixes";
    auto x = fixture(c, width * batch, 1009, -.5f, .5f), w = fixture(c, 4, 1013, -.5f, .5f);
    auto states = fixture(c, 3 * slots, 1019, -.5f, .5f);
    const std::vector<std::int32_t> initial{0, 1}, base{3, 8}, valid{1, 4};
    auto expected = bf16_bits(states);
    auto dx = to_device_bf16(x), dw = to_device_bf16(w), di = to_device(initial),
         db = to_device(base), dvalid = to_device(valid);
    GuardedDeviceBuffer ds(states.size() * 2);
    ds.copy_from_host(expected.data(), ds.bytes());
    GuardedBf16Tensor out(c, width * batch);
    Tensor tx(dx.p, DType::BF16, {c, width, batch}), tw(dw.p, DType::BF16, {c, 4}),
        ts(ds.data(), DType::BF16, {c, 3, slots});
    Tensor ti(di.p, DType::I32, {batch}), tb(db.p, DType::I32, {batch}),
        tv(dvalid.p, DType::I32, {batch});
    Tensor to(out.data(), DType::BF16, {c, width, batch});
    ops::causal_conv1d_silu_snapshot(tx, tw, ts, tv, ti, tb, to, nullptr);
    cuda_synchronize();
    std::vector<double> ideal(x.size(), 0.);
    for (int row = 0; row < batch; ++row) {
        const auto u = [&](int channel, int token) -> float {
            return token < 0 ? states[(initial[row] * 3 + token + 3) * c + channel]
                             : x[(row * width + token) * c + channel];
        };
        for (int t = 0; t < valid[row]; ++t)
            for (int channel = 0; channel < c; ++channel) {
                double sum = 0;
                for (int j = 0; j < 4; ++j)
                    sum += double(w[j * c + channel]) * double(u(channel, t - 3 + j));
                ideal[(row * width + t) * c + channel] = sum / (1 + std::exp(-sum));
                for (int j = 0; j < 3; ++j)
                    expected[((base[row] + t) * 3 + j) * c + channel] =
                        f32_to_bf16(u(channel, t - 2 + j));
            }
    }
    int failures = compare(label, out.values(), ideal, kConv);
    failures += from_device<std::uint16_t>(ds.data(), expected.size()) != expected;
    const auto bits = out.bits();
    for (int t = valid[0]; t < width; ++t)
        for (int channel = 0; channel < c; ++channel)
            if (bits[t * c + channel] != 0) {
                ++failures;
                break;
            }
    failures += ds.verify_guards(label) + out.verify_guards(label);
    failures +=
        verify_preserved(label, dx, bf16_bits(x)) + verify_preserved(label, dw, bf16_bits(w));
    failures += verify_preserved(label, di, initial) + verify_preserved(label, db, base) +
                verify_preserved(label, dvalid, valid);
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) return 77;
    const std::string mode = argc > 1 ? argv[1] : "all";
    int failures           = 0;
    try {
        if (mode == "all" || mode == "recurrence" || mode == "recurrence-perf")
            for (int t : {1, 4, 63, 64, 65, 128, 1024})
                failures += recurrence_case(t, mode == "recurrence-perf");
        if (mode == "all" || mode == "conv")
            for (int t : {1, 4, 63, 64, 65, 128, 1024}) failures += conv_case(t, false);
        if (mode == "all" || mode == "conv") failures += conv_snapshot_case();
        if (mode == "all" || mode == "split")
            for (int t : {1, 4, 63, 64, 65, 128, 1024}) failures += conv_case(t, true);
        if (mode == "all" || mode == "gating")
            for (int t : {1, 4, 63, 64, 65, 128, 1024}) failures += gated_norm_case(t);
        if (mode == "all" || mode == "control")
            for (int t : {1, 4, 63, 64, 65, 128, 1024}) {
                failures += control_case(t, false);
                failures += control_case(t, true);
            }
        if (mode == "all" || mode == "projection") {
            DevicePackedWeight qk(
                quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 2048, 5120, 409));
            DevicePackedWeight vz(
                quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 6144, 5120, 419));
            for (int t : {1, 4, 63, 64, 65, 128, 1024}) failures += projection_case(qk, vz, t);
            for (int t : {1, 2, 3, 4, 5, 6, 7, 8, 9, 15}) failures += projection_case(qk, vz, t, true);
        }
        std::cout << (failures ? "FAIL" : "OK") << " tensor local GDN " << mode << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
