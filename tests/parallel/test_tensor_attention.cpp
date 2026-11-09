#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"
#include "tensor_attention_codec.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace tensor_attention_test;
constexpr int D = 256, Page = 64;
// Frozen before execution. BF16 is the existing attention criterion; E8 uses the existing
// rotated low-bit value budget (kAttentionNvfp4Criterion/kAttentionK8V4Criterion in
// tests/ops/softmax_attention/causal_cache.cpp). E8 had no prior criterion: this new budget
// is predeclared for both old Q24/KV4 and local Q12/KV2. The oracle never quantizes Q or
// rounds probabilities/partial reductions to imitate private kernel arithmetic.
constexpr ReductionCriterion Bf16Criterion{.0028, .001, .0027};
constexpr ReductionCriterion E8Criterion{.015, .005, .011};

std::size_t index(int heads, int d, int h, int t, int b, int width) {
    return d + std::size_t(D) * (h + heads * (t + std::size_t(width) * b));
}

float value(int salt, int row, int head, int token, int d) {
    std::uint32_t z = std::uint32_t(salt * 1031 + row * 170003 + head * 739 + token * 1117 + d);
    z ^= z >> 16; z *= 0x7feb352du; z ^= z >> 15; z *= 0x846ca68bu; z ^= z >> 16;
    // Bounded dyadic inputs keep the H64 storage transform exact in FP32. Distinct signed
    // head/row signals expose head aliasing, swapped ranks, and incorrect table-row selection.
    return bf16_to_f32(f32_to_bf16(float(int(z % 257) - 128) / 128.f +
                                   float((row + head + salt) % 5 - 2) / 16.f));
}

struct Cache {
    bool e8;
    int kv_heads, logical_pages, rows, pages, row_bytes;
    std::vector<int> table;
    std::vector<std::uint8_t> k, v, ks, vs;

    Cache(bool quantized, int heads, int batch, int extent)
        : e8(quantized), kv_heads(heads), logical_pages((extent + Page - 1) / Page), rows(batch + 2),
          pages(rows * logical_pages + 3), row_bytes(e8 ? 128 : 512),
          table(rows * logical_pages), k(std::size_t(pages) * kv_heads * Page * row_bytes, 0xa5),
          v(k.size(), 0x5a), ks(e8 ? std::size_t(pages) * kv_heads * Page * 8 : 0, 0x69),
          vs(ks.size(), 0x96) {
        // Fragmented/reversed physical pages; extra pages and two unselected table rows remain
        // byte canaries. This map is independent of the production address helpers.
        for (int i = 0; i < int(table.size()); ++i) table[i] = int(table.size()) - i;
    }
    std::size_t address(int row, int pos, int head, int bytes) const {
        const int page = table[row * logical_pages + pos / Page];
        return std::size_t(bytes) * (pos % Page + Page * (head + kv_heads * page));
    }
    static void word(std::vector<std::uint8_t>& plane, std::size_t offset, std::uint16_t x) {
        plane[offset] = x & 255; plane[offset + 1] = x >> 8;
    }
    static std::uint16_t word(const std::vector<std::uint8_t>& plane, std::size_t offset) {
        return plane[offset] | (std::uint16_t(plane[offset + 1]) << 8);
    }
    void write(int row, int pos, int head, const float* key, const float* val) {
        const auto at = address(row, pos, head, row_bytes);
        if (!e8) {
            for (int d = 0; d < D; ++d) {
                word(k, at + d * 2, f32_to_bf16(key[d]));
                word(v, at + d * 2, half_bits(val[d]));
            }
            return;
        }
        const auto ek = encode_e8(key, true), ev = encode_e8(val, false);
        std::copy(ek.bytes.begin(), ek.bytes.end(), k.begin() + at);
        std::copy(ev.bytes.begin(), ev.bytes.end(), v.begin() + at);
        const auto scale = address(row, pos, head, 8);
        for (int g = 0; g < 4; ++g) {
            word(ks, scale + 2 * g, ek.scales[g]); word(vs, scale + 2 * g, ev.scales[g]);
        }
    }
    double read(bool key, int row, int pos, int head, int d) const {
        const auto& plane = key ? k : v;
        const auto at = address(row, pos, head, row_bytes);
        if (!e8) return key ? double(bf16_to_f32(word(plane, at + 2 * d)))
                            : half_value(word(plane, at + 2 * d));
        const unsigned nibble = (plane[at + d / 2] >> ((d & 1) * 4)) & 15;
        const int code = int(nibble ^ 8u) - 8;
        return code * half_value(word(key ? ks : vs, address(row, pos, head, 8) + 2 * (d / 64)));
    }
};

struct DeviceCache {
    GuardedDeviceBuffer k, v, ks, vs, table;
    PagedKVBatchLayerView view;
    explicit DeviceCache(const Cache& c)
        : k(c.k.size()), v(c.v.size()), ks(std::max<std::size_t>(1, c.ks.size())),
          vs(std::max<std::size_t>(1, c.vs.size())), table(c.table.size() * sizeof(int)) {
        const auto storage = c.e8 ? KvCacheStorage::RK4V4E8 : KvCacheStorage::BFloat16;
        view = {Tensor(k.data(), c.e8 ? DType::U8 : DType::BF16,
                       {c.e8 ? 128 : D, Page, c.kv_heads, c.pages}),
                Tensor(v.data(), c.e8 ? DType::U8 : DType::FP16,
                       {c.e8 ? 128 : D, Page, c.kv_heads, c.pages}),
                c.e8 ? Tensor(ks.data(), DType::FP16, {4, Page, c.kv_heads, c.pages}) : Tensor{},
                c.e8 ? Tensor(vs.data(), DType::FP16, {4, Page, c.kv_heads, c.pages}) : Tensor{},
                Tensor(table.data(), DType::I32, {c.logical_pages, c.rows}), D, c.kv_heads, storage};
        upload(c);
    }
    void upload(const Cache& c) {
        k.copy_from_host(c.k.data(), c.k.size()); v.copy_from_host(c.v.data(), c.v.size());
        if (c.e8) {
            ks.copy_from_host(c.ks.data(), c.ks.size()); vs.copy_from_host(c.vs.data(), c.vs.size());
        }
        table.copy_from_host(c.table.data(), c.table.size() * sizeof(int));
    }
    PagedKVLayerView single(const Cache& c, int row) {
        return {view.k_pages, view.v_pages, view.k_scale_pages, view.v_scale_pages,
                Tensor(static_cast<int*>(table.data()) + row * c.logical_pages,
                       DType::I32, {c.logical_pages}), D, c.kv_heads, view.storage};
    }
    int verify(const Cache& c, const std::string& label) {
        int failures = 0;
        auto check = [&](const GuardedDeviceBuffer& device, const auto& expected, const char* name) {
            const auto actual = from_device<std::uint8_t>(device.data(), expected.size());
            if (actual != expected) {
                auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
                const auto at = mismatch.first - actual.begin();
                std::cerr << label << ' ' << name << " byte " << at << " actual="
                          << int(actual[at]) << " expected=" << int(expected[at]) << '\n';
                ++failures;
            }
            failures += device.verify_guards(label);
        };
        check(k, c.k, "K"); check(v, c.v, "V");
        if (c.e8) { check(ks, c.ks, "K scales"); check(vs, c.vs, "V scales"); }
        failures += from_device<int>(table.data(), c.table.size()) != c.table;
        return failures + table.verify_guards(label);
    }
};

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), f32_to_bf16);
    return result;
}

std::vector<double> ideal(const Cache& c, const std::vector<float>& q, int width, int batch,
                           const std::vector<int>& base, const std::vector<int>& live,
                           const std::vector<int>& rows) {
    const int Q = int(q.size() / (std::size_t(D) * width * batch));
    const ops::AttentionHeadGeometry Geometry{D, Q, c.kv_heads};
    std::vector<double> transformed(q.begin(), q.end()), out(q.size());
    if (c.e8)
        for (std::size_t begin = 0; begin < transformed.size(); begin += 64) {
            const auto r = rotate(transformed.data() + begin);
            std::copy(r.begin(), r.end(), transformed.begin() + begin);
        }
    for (int b = 0; b < batch; ++b) {
        if (live[b] == 0) continue;
        naive_dense_softmax_attention(
            Geometry, live[b], base[b] + live[b], .0625,
            [&](int d, int h, int t) { return transformed[index(Q, d, h, t, b, width)]; },
            [&](int d, int h, int p) { return c.read(true, rows[b], p, h, d); },
            [&](int d, int h, int p) { return c.read(false, rows[b], p, h, d); },
            [&](int t, int p) { return p <= base[b] + t; },
            [&](int d, int h, int t, double x) { out[index(Q, d, h, t, b, width)] = x; });
    }
    if (c.e8)
        for (std::size_t begin = 0; begin < out.size(); begin += 64) {
            const auto r = rotate(out.data() + begin);
            std::copy(r.begin(), r.end(), out.begin() + begin);
        }
    return out;
}

int run_case(bool e8, int width, int batch, int start, bool masked, bool zero_query = false,
             int query_heads = 12) {
    const int Q = query_heads, KV = query_heads / 6;
    const ops::AttentionHeadGeometry Geometry{D, Q, KV};
    const std::string label = std::string(e8 ? "E8" : "BF16") + " Q" + std::to_string(Q) +
        "/KV" + std::to_string(KV) + " W=" +
        std::to_string(width) + " B=" + std::to_string(batch) + " start=" + std::to_string(start) +
        (masked ? " masked" : " dense") + (zero_query ? " zero-query" : "");
    std::vector<int> base(batch), live(batch, width), rows(batch), positions(width * batch);
    for (int b = 0; b < batch; ++b) {
        base[b] = start + (batch > 1 ? b % 3 : 0);
        rows[b] = batch - b;
        if (masked) live[b] = b == batch - 1 ? 0 : std::max(1, width - b);
        for (int t = 0; t < width; ++t)
            positions[b * width + t] = live[b] ? base[b] + std::min(t, live[b] - 1) : 0;
    }
    Cache before(e8, KV, batch, start + width + 4);
    for (int b = 0; b < batch; ++b)
        for (int p = 0; p < base[b]; ++p)
            for (int h = 0; h < KV; ++h) {
                std::array<float, D> k{}, v{};
                for (int d = 0; d < D; ++d) {
                    k[d] = value(3, rows[b], h, p, d); v[d] = value(5, rows[b], h, p, d);
                }
                before.write(rows[b], p, h, k.data(), v.data());
            }
    std::vector<float> q(std::size_t(D) * Q * width * batch),
        k(std::size_t(D) * KV * width * batch), v(k.size());
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < width; ++t) {
            for (int h = 0; h < Q; ++h)
                for (int d = 0; d < D; ++d)
                    q[index(Q, d, h, t, b, width)] = zero_query ? 0 : value(7, rows[b], h, t, d);
            for (int h = 0; h < KV; ++h)
                for (int d = 0; d < D; ++d) {
                    k[index(KV, d, h, t, b, width)] = value(11, rows[b], h, t, d);
                    v[index(KV, d, h, t, b, width)] = value(13, rows[b], h, t, d);
                    // Exact zero codec groups alongside nonzero groups and distinct heads.
                    if ((t + h + b) % 11 == 0 && d < 64) {
                        k[index(KV, d, h, t, b, width)] = 0;
                        v[index(KV, d, h, t, b, width)] = 0;
                    }
                }
        }
    Cache expected = before;
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < live[b]; ++t)
            for (int h = 0; h < KV; ++h)
                expected.write(rows[b], base[b] + t, h, &k[index(KV, 0, h, t, b, width)],
                               &v[index(KV, 0, h, t, b, width)]);
    const auto reference = ideal(expected, q, width, batch, base, live, rows);
    const auto qb = bits(q), kb = bits(k), vb = bits(v);
    GuardedDeviceBuffer dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2),
        dp(positions.size() * sizeof(int)), dl(batch * sizeof(int)), dr(batch * sizeof(int)),
        output(qb.size() * 2);
    dq.copy_from_host(qb.data(), dq.bytes()); dk.copy_from_host(kb.data(), dk.bytes());
    dv.copy_from_host(vb.data(), dv.bytes()); dp.copy_from_host(positions.data(), dp.bytes());
    dl.copy_from_host(live.data(), dl.bytes()); dr.copy_from_host(rows.data(), dr.bytes());
    output.fill(255);
    DeviceCache cache(before);
    Tensor qt(dq.data(), DType::BF16, {D, Q, width, batch}),
        kt(dk.data(), DType::BF16, {D, KV, width, batch}),
        vt(dv.data(), DType::BF16, {D, KV, width, batch}),
        pt(dp.data(), DType::I32, {width, batch}), lt(dl.data(), DType::I32, {batch}),
        rt(dr.data(), DType::I32, {batch}), ot(output.data(), DType::BF16, {D, Q, width, batch});
    const auto max_visible = std::uint32_t(start + width + 3);
    const ops::CausalAttentionExecutionEnvelope envelope{1, max_visible};
    // This is intentionally the first local attention call: the old wrapper rejects Q12.
    const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
        Geometry, cache.view.storage, envelope, batch, width, width);
    // A borrowed arena requires nonempty backing, including zero-workspace prompt routes.
    // Keep the Op's reported capacity as the usage bound rather than inflating that contract.
    GuardedDeviceBuffer scratch(std::max<std::size_t>(1, capacity));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    int failures = 0;
    if (batch == 1 && !masked) {
        ops::kv_cache_append(kt, vt, pt, cache.single(before, rows[0]), nullptr);
        cuda_synchronize();
        failures += cache.verify(expected, label + " standalone append");
        cache.upload(before);
    }
    ops::causal_softmax_attention(qt, kt, vt, pt, masked ? lt : Tensor{}, rt, Geometry, .0625f,
                                   cache.view, envelope, workspace, ot, nullptr);
    cuda_synchronize();
    const auto actual = from_device_bf16(output.data(), qb.size());
    const auto actual_bits = from_device<std::uint16_t>(output.data(), qb.size());
    const auto criterion = e8 ? E8Criterion : Bf16Criterion;
    failures += verify_reduction(label, actual, reference, criterion);
    failures += cache.verify(expected, label + " fused append");
    for (int b = 0; b < batch; ++b) {
        // Check each head/row separately so a wrong low-energy group cannot hide in an aggregate.
        for (int h = 0; h < Q; ++h) {
            std::vector<double> a, r;
            for (int t = 0; t < live[b]; ++t)
                for (int d = 0; d < D; ++d) {
                    const auto i = index(Q, d, h, t, b, width);
                    a.push_back(actual[i]); r.push_back(reference[i]);
                }
            if (!a.empty()) failures += verify_reduction(label + " row=" + std::to_string(b) +
                " head=" + std::to_string(h), a, r, criterion);
        }
        for (int t = live[b]; t < width; ++t)
            for (int h = 0; h < Q; ++h)
                for (int d = 0; d < D; ++d)
                    if (actual_bits[index(Q, d, h, t, b, width)] != 0) ++failures;
        if (live[b] == 0) continue;
        Tensor cq(static_cast<std::uint16_t*>(dq.data()) + index(Q, 0, 0, 0, b, width),
                  DType::BF16, {D, Q, live[b]});
        Tensor cp(static_cast<int*>(dp.data()) + width * b, DType::I32, {live[b]});
        GuardedDeviceBuffer co(std::size_t(D) * Q * live[b] * 2);
        co.fill(255);
        Tensor cot(co.data(), DType::BF16, {D, Q, live[b]});
        const auto cc = ops::causal_softmax_attention_workspace_capacity_bytes(
            Geometry, cache.view.storage, envelope, 1, live[b], live[b]);
        GuardedDeviceBuffer cw(std::max<std::size_t>(1, cc));
        WorkspaceArena cws(DeviceSpan{cw.data(), cw.bytes()});
        ops::causal_softmax_attention_cached(cq, cp, Geometry, .0625f,
            cache.single(expected, rows[b]), envelope, cws, cot, nullptr);
        cuda_synchronize();
        const std::vector<double> cr(reference.begin() + index(Q, 0, 0, 0, b, width),
            reference.begin() + index(Q, 0, 0, live[b], b, width));
        failures += verify_reduction(label + " cached row=" + std::to_string(b),
            from_device_bf16(co.data(), std::size_t(D) * Q * live[b]), cr, criterion);
        failures += co.verify_guards(label) + cw.verify_guards(label) + (cws.used() != 0) +
                    (cws.peak_used() > cc);
    }
    failures += cache.verify(expected, label + " cached read-only");
    failures += from_device<std::uint16_t>(dq.data(), qb.size()) != qb;
    failures += from_device<std::uint16_t>(dk.data(), kb.size()) != kb;
    failures += from_device<std::uint16_t>(dv.data(), vb.size()) != vb;
    failures += from_device<int>(dp.data(), positions.size()) != positions;
    failures += from_device<int>(dl.data(), live.size()) != live;
    failures += from_device<int>(dr.data(), rows.size()) != rows;
    failures += dq.verify_guards(label) + dk.verify_guards(label) + dv.verify_guards(label) +
        dp.verify_guards(label) + dl.verify_guards(label) + dr.verify_guards(label) +
        output.verify_guards(label) + scratch.verify_guards(label) + (workspace.used() != 0) +
        (workspace.peak_used() > capacity);
    std::cout << (failures ? "FAIL " : "OK ") << label << '\n';
    return failures;
}
// TP2 rank-local Q4/Q5 attention input projection [3584,5120] (Q12/KV2). The criterion is the
// unchanged registered A8 profile from tests/ops/test_attn_input_proj.cpp.
constexpr ReductionCriterion kAttnInputA8{0.04, 1.0 / 256.0, 0.06};
// A16 decode profile (T<=12): unchanged registered A16 criterion of the same file.
constexpr ReductionCriterion kAttnInputA16{2.9e-3, 4.0e-3, 4.5e-3};

int projection_case(input_projection::DevicePackedWeight& qk,
                    input_projection::DevicePackedWeight& gv, int t, bool a16 = false) {
    namespace ip = input_projection;
    const std::string label =
        std::string(a16 ? "local attention input A16 T=" : "local attention input T=") + std::to_string(t);
    const auto policy = a16 ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA8;
    const auto x = ip::make_bf16_activation(5120, t, 431 + t);
    auto dx      = to_device_bf16(x);
    ip::GuardedBf16Tensor q(3072, t), gate(3072, t), k(512, t), v(512, t);
    Tensor tx(dx.p, DType::BF16, {5120, t});
    Tensor tq = q.tensor(), tg = gate.tensor(), tk = k.tensor(), tv = v.tensor();
    const auto bytes =
        ops::attn_input_proj_workspace_capacity_bytes(5120, policy, t, t);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    WorkspaceArena ws(DeviceSpan{scratch.data(), scratch.bytes()});
    int failures = 0;
    if (!a16 && t > 12) try {
        ops::attn_input_proj(tx, qk.view(), gv.view(), tq, tg, tk, tv, ops::LinearPolicy::A16Only,
                             ws, nullptr);
        std::cerr << label << ": local A16 profile was admitted above T=12\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    ops::attn_input_proj(tx, qk.view(), gv.view(), tq, tg, tk, tv, policy, ws, nullptr);
    cuda_synchronize();
    const auto check = [&](const ip::GuardedBf16Tensor& out, int rows,
                           const quantized_weight::PackedWeight& weight, int weight_begin) {
        const int samples = t <= 4 ? rows : 31;
        return ip::compare(label, ip::gather_rows(out.values(), rows, 0, rows, t, samples),
                           ip::projection_oracle(weight, weight_begin, rows, x, 5120, t, samples),
                           a16 ? kAttnInputA16 : kAttnInputA8) +
               out.verify_guards(label) + out.verify_fully_written(label);
    };
    failures += check(q, 3072, qk.host, 0) + check(k, 512, qk.host, 3072);
    failures += check(gate, 3072, gv.host, 0) + check(v, 512, gv.host, 3072);
    failures += ip::verify_preserved(label, dx, ip::bf16_bits(x));
    failures += qk.verify_preserved(label) + gv.verify_preserved(label);
    failures += scratch.verify_guards(label) + (ws.used() != 0) + (ws.peak_used() > bytes);
    std::cout << (failures ? "FAIL " : "OK ") << label << '\n';
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        std::fesetround(FE_TONEAREST);
        int failures = 0;
        // Same independent E8 oracle and fixed budgets on the original global geometry.
        // Run before local registration so a preexisting E8 problem is identified separately.
        for (int width : {1, 9, 65})
            failures += run_case(true, width, 1, 63, false, false, 24);
        failures += run_case(true, 16, 3, 63, true, false, 24);
        for (bool e8 : {false, true}) {
            for (int width : {1, 6, 8, 9, 16}) failures += run_case(e8, width, 1, 63, false);
            failures += run_case(e8, 1, 1, 513, false);
            failures += run_case(e8, 8, 1, 1023, false);
            failures += run_case(e8, 16, 1, 1023, false);
            failures += run_case(e8, 65, 1, 0, false);
            failures += run_case(e8, 129, 1, 63, false);
            failures += run_case(e8, 9, 1, 63, false, true);
            for (int batch : {2, 3, 8}) {
                failures += run_case(e8, 1, batch, 63, false);
                failures += run_case(e8, 16, batch, 63, true);
            }
        }
        for (auto storage : {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                             KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value,
                             KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                             KvCacheStorage::RotatedInt4KeyInt4ValueGroup64,
                             KvCacheStorage::RK2V4E8}) {
            try {
                (void)ops::causal_softmax_attention_workspace_capacity_bytes(
                    {D, 12, 2}, storage, {1, 128}, 1, 1, 16);
                std::cerr << "unregistered local cache profile accepted\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
        }
        {
            input_projection::DevicePackedWeight qk(
                quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 3584, 5120, 443));
            input_projection::DevicePackedWeight gv(
                quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 3584, 5120, 449));
            for (int t : {1, 4, 8, 63, 64, 65, 128, 1024}) failures += projection_case(qk, gv, t);
            for (int t : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12}) failures += projection_case(qk, gv, t, true);
        }
        std::cout << (failures ? "FAIL" : "OK") << " tensor_attention\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "tensor_attention: " << e.what() << '\n';
        return 1;
    }
}
