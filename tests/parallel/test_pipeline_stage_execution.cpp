// Opt-in closed 27B TextContext stage qualification. No model artifact is required.
// Run with NINFER_TEST_PIPELINE_STAGE_EXECUTION=1 on a compiled SM89 + SM120 pair.
// This exercises eager leaves and physical state, not the unfinished public Engine coordinator.
#include "targets/qwen3_6_27b/impl/variant.h"
#include "core/cuda_error_scope.h"
#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {
using namespace ninfer;
namespace q36      = ninfer::targets::qwen3_6;
namespace q27      = ninfer::targets::qwen3_6_27b;
namespace detail27 = q27::detail;
namespace rt       = q36::detail::qwen3_6_27b_runtime;
using Config       = detail27::TextConfig;
using TextContext  = rt::schedule::TextContext;

void check(bool ok, std::string_view message) {
    if (!ok) { throw std::runtime_error(std::string(message)); }
}

template <class F>
void rejects_tail(F&& fn) {
    try {
        fn();
    } catch (const std::logic_error& e) {
        check(std::string(e.what()).find("stage 0") != std::string::npos,
              "tail rejection must identify stage 0 ownership");
        return;
    }
    throw std::runtime_error("tail executed an owner-only leaf");
}

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

double number(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

template <class T>
std::vector<T> read(const Tensor& tensor) {
    check(tensor.is_contiguous(), "fixture read must be contiguous");
    std::vector<T> result(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return result;
}

template <class T>
void write(const Tensor& tensor, const std::vector<T>& data) {
    check(tensor.bytes() == data.size() * sizeof(T), "fixture upload size");
    CUDA_CHECK(cudaMemcpy(tensor.data, data.data(), tensor.bytes(), cudaMemcpyHostToDevice));
}

// Construct legal row-split storage directly. All packed signed codes and FP16 scales are zero,
// hence their independent mathematical decode is exactly zero for every logical element.
// No source-sized host matrix or production quantizer is used by this fixture.
class FixtureWeights {
public:
    detail27::RuntimeModelView model{};

    explicit FixtureWeights(PhysicalLayerRange range) {
        const Tensor input_norm  = tensor(DType::BF16, {Config::hidden}, 0.5F);
        const Tensor qk_norm     = tensor(DType::BF16, {Config::head_dim}, 0.25F);
        const Tensor gdn_norm    = tensor(DType::BF16, {Config::gdn_value_head_dim}, 1.0F);
        const Tensor convolution = tensor(DType::BF16, {Config::convolution_dim, 4}, 0.0F);
        const Tensor control     = tensor(DType::FP32, {Config::gdn_value_heads}, 0.0F);
        const Weight ab          = dense_zero(Config::gdn_value_heads, Config::hidden);
        const detail27::DensePostMixerPayload mlp{
            zero(QType::Q4G64_F16S, 2 * Config::intermediate, Config::hidden),
            zero(QType::Q5G64_F16S, Config::hidden, Config::intermediate)};
        const Weight output = zero(QType::Q5G64_F16S, Config::hidden, Config::value_dim);
        for (unsigned layer = range.begin; layer < range.end; ++layer) {
            if (Config::is_full_attention(layer)) {
                auto& w      = model.full_layers[Config::full_attention_index(layer)];
                w.input_norm = input_norm;
                w.projection = detail27::SplitAttentionProjectionPayload{
                    zero(QType::Q4G64_F16S, 7168, Config::hidden),
                    zero(QType::Q5G64_F16S, 7168, Config::hidden)};
                w.query_norm          = qk_norm;
                w.key_norm            = qk_norm;
                w.output              = output;
                w.post_attention_norm = input_norm;
                w.post_mixer          = mlp;
            } else {
                auto& w              = model.gdn_layers[Config::gdn_index(layer)];
                w.input_norm         = input_norm;
                w.projection.a_log   = control;
                w.projection.dt_bias = control;
                w.projection.control_projection =
                    detail27::SplitGdnControlProjectionPayload{ab, ab};
                w.projection.input_projection = detail27::SplitGdnInputProjectionPayload{
                    zero(QType::Q4G64_F16S, 4096, Config::hidden),
                    zero(QType::Q5G64_F16S, 12288, Config::hidden)};
                w.convolution         = convolution;
                w.norm                = gdn_norm;
                w.output              = output;
                w.post_attention_norm = input_norm;
                w.post_mixer          = mlp;
            }
        }
        // Unowned global layer entries remain null. Any full-model loop or incorrect global
        // indexing into local state fails instead of accidentally using replicated weights.
        if (range.begin == 0) {
            model.final_norm  = tensor(DType::BF16, {Config::hidden}, 0.25F);
            model.output_head = zero(QType::W8G32_F16S, Config::output_rows, Config::hidden);
            // Separate storage: the cached zero head must not alias the nonzero embedding.
            model.token_embedding =
                packed_zero(QType::W8G32_F16S, Config::output_rows, Config::hidden);
            for (int token : {2, 7, 11, 12}) {
                std::vector<std::int8_t> codes(Config::hidden);
                for (int column = 0; column < Config::hidden; ++column) {
                    codes[column] = static_cast<std::int8_t>((column + token) % 7 - 3);
                }
                std::vector<std::uint16_t> scales(Config::hidden / 32, 0x3400U); // FP16 0.25
                CUDA_CHECK(cudaMemcpy(
                    static_cast<std::int8_t*>(const_cast<void*>(model.token_embedding.qdata)) +
                        static_cast<std::size_t>(token) * Config::hidden,
                    codes.data(), codes.size(), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(
                    static_cast<std::uint16_t*>(const_cast<void*>(model.token_embedding.scales)) +
                        static_cast<std::size_t>(token) * scales.size(),
                    scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
            }
        }
    }

private:
    std::vector<std::unique_ptr<DeviceBuffer>> buffers_;
    std::map<std::tuple<QType, int, int>, Weight> zero_weights_;

    void* allocate(std::size_t bytes) {
        auto& buffer = buffers_.emplace_back(std::make_unique<DeviceBuffer>(bytes));
        buffer->fill(0);
        return buffer->p;
    }

    Tensor tensor(DType dtype, std::initializer_list<std::int32_t> shape, float value) {
        Tensor out(nullptr, dtype, shape);
        out.data = allocate(out.bytes());
        if (dtype == DType::BF16) {
            write(out, std::vector<std::uint16_t>(out.numel(), bf16(value)));
        } else {
            write(out, std::vector<float>(out.numel(), value));
        }
        return out;
    }

    Weight dense_zero(int n, int k) {
        Weight w{};
        w.qtype  = QType::BF16_CTRL;
        w.layout = QuantLayout::Contiguous;
        w.ndim   = 2;
        w.n = w.shape[0] = w.padded_shape[0] = n;
        w.k = w.shape[1] = w.padded_shape[1] = k;
        w.payload_bytes                      = static_cast<std::uint64_t>(n) * k * 2;
        w.payload = w.qdata = allocate(w.payload_bytes);
        return w;
    }

    Weight zero(QType type, int n, int k) {
        auto key = std::tuple{type, n, k};
        if (auto it = zero_weights_.find(key); it != zero_weights_.end()) { return it->second; }
        Weight w = packed_zero(type, n, k);
        zero_weights_.emplace(key, w);
        return w;
    }

    Weight packed_zero(QType type, int n, int k) {
        const int group         = type == QType::W8G32_F16S ? 32 : 64;
        const int bits          = type == QType::W8G32_F16S ? 8 : type == QType::Q5G64_F16S ? 5 : 4;
        const std::size_t count = static_cast<std::size_t>(n) * k;
        const std::size_t low_bytes    = bits == 8 ? count : count / 2;
        const std::size_t high_bytes   = bits == 5 ? count / 8 : 0;
        const auto align               = [](std::size_t x) { return (x + 255) / 256 * 256; };
        const std::size_t scale_offset = align(low_bytes) + align(high_bytes);
        Weight w{};
        w.qtype       = type;
        w.layout      = QuantLayout::RowSplit;
        w.scale_dtype = DType::FP16;
        w.group = w.group_size = group;
        w.ndim                 = 2;
        w.n = w.shape[0] = w.padded_shape[0] = n;
        w.k = w.shape[1] = w.padded_shape[1] = k;
        w.payload_bytes                      = scale_offset + count / group * 2;
        auto* data                           = static_cast<std::byte*>(allocate(w.payload_bytes));
        w.payload = w.qdata = data;
        w.high_plane_bytes  = high_bytes;
        w.qhigh             = high_bytes ? data + align(low_bytes) : nullptr;
        w.scales            = data + scale_offset;
        return w;
    }
};

q27::Package::SequencePlan plan_stage(DeviceContext& device, PhysicalLayerRange range) {
    EngineOptions options;
    options.device                           = device.device;
    options.max_context                      = 64;
    options.max_concurrency                  = 4;
    // Four active lanes require four 64-token physical page groups, even at context 64.
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(256);
    options.prefill_chunk                    = 128;
    options.use_cuda_graph                   = false;
    options.enable_vision                    = false;
    options.context_cache.enabled            = false;
    options.context_cache.device_state_slots = 0;
    auto planner                             = q27::Package::make_sequence_planner(
        device, options, q27::Package::WeightsProfile::Qwen38GroupwiseInt, range);
    const auto pages = planner.capacity_curve().minimum_main_page_groups;
    return std::move(planner).finalize(pages);
}

struct Stage {
    DeviceContext& device;
    int rank;
    int source_slot;
    int destination_slot;
    FixtureWeights weights;
    q27::Package::SequencePlan plan;
    DeviceBuffer persistent;
    WorkspaceArena work;
    q36::DecoderState decoder;
    q36::StateImageDevicePool images;
    q36::RoundState io;
    Tensor prefill_hidden;
    DeviceBuffer residual_store{Config::hidden * 2 * 2};
    DeviceBuffer hidden_store{Config::hidden * 2 * 2};
    DeviceBuffer logits_store{Config::output_rows * 2 * 2};
    DeviceBuffer metadata{8 * sizeof(std::int32_t)};
    std::vector<DeviceKVPageLease> pages;
    KVExecutionRowLease row;
    KVExecutionRowLease guard_row;
    std::unique_ptr<TextContext> text;
    int active_page = -1;
    int guard_page  = -1;

    Stage(DeviceContext& ctx, unsigned stage_rank, PhysicalLayerRange range)
        : device(ctx), rank(stage_rank), source_slot(stage_rank), destination_slot(stage_rank + 2),
          weights(range), plan(plan_stage(ctx, range)), persistent(plan.impl_->persistent.bytes),
          work(plan.workspace_capacity_bytes()),
          decoder({persistent.p, persistent.bytes}, plan.impl_->persistent.decoder),
          images({persistent.p, persistent.bytes}, plan.impl_->persistent.state_images),
          io(range.begin == 0 ? q36::RoundState({persistent.p, persistent.bytes}, plan.impl_->persistent.round)
                              : q36::RoundState()),
          prefill_hidden(range.begin == 0
              ? plan.impl_->persistent.prefill_hidden.bind({persistent.p, persistent.bytes}) : Tensor{}) {
        auto& cache               = decoder.text_kv;
        row                       = cache.execution_tables().acquire(rank + 1);
        guard_row                 = cache.execution_tables().acquire(0);
        const unsigned page_count = rank + 2;
        auto reservation          = cache.page_pool().reserve(page_count);
        check(reservation.has_value(), "fixture KV-page reservation");
        pages.reserve(page_count);
        cache.page_pool().materialize(*reservation, page_count, pages);
        cache.execution_tables().publish(
            row.handle(), 0, std::span<const DeviceKVPageLease>(pages).subspan(rank + 1, 1),
            device.stream);
        cache.execution_tables().publish(guard_row.handle(), 0,
                                         std::span<const DeviceKVPageLease>(pages).first(1),
                                         device.stream);
        device.synchronize();
        active_page = read<std::int32_t>(cache.execution_tables().row(row.handle()))[0];
        guard_page  = read<std::int32_t>(cache.execution_tables().row(guard_row.handle()))[0];
        check(active_page != guard_page, "fixture requires distinct physical KV pages");
        for (std::size_t plane = 0; plane < cache.page_pool().plane_count(); ++plane) {
            const Tensor& storage = cache.page_pool().plane(plane);
            write(storage, std::vector<std::uint16_t>(storage.numel(), bf16(1.0F)));
        }
        for (unsigned layer = 0; layer < images.linear().layer_count(); ++layer) {
            for (int slot = 0; slot < images.slot_count(); ++slot) {
                const Tensor recurrent = images.linear().recurrent_slot(layer, slot);
                write(recurrent, std::vector<float>(recurrent.numel(), initial(layer, slot)));
                const Tensor conv = images.linear().conv_slot(layer, slot);
                write(conv, std::vector<std::uint16_t>(conv.numel(), bf16(initial(layer, slot))));
            }
        }
        for (int slot = 0; slot < images.slot_count(); ++slot) {
            const Tensor hidden = images.continuation_hidden_slot(slot);
            write(hidden, std::vector<std::uint16_t>(hidden.numel(), bf16(initial(0, slot))));
        }
        device.synchronize();
        text = std::make_unique<TextContext>(device, weights.model, work, cache.execution_view(row),
                                             images.linear(), io, prefill_hidden, 64, 0,
                                             q36::PagedKVCacheView{}, &cache, nullptr, range);
        text->set_linear_state_slots(source_slot, destination_slot);
        const auto geometry = q36::runtime_support::StageLayerLayout::create(range, 64);
        check(images.linear().layer_count() == geometry.gdn_count(), "local GDN allocation");
        check(cache.layers() == geometry.full_attention_count(), "local attention allocation");
    }

    double initial(unsigned layer, int slot) const {
        return (layer + 1) / 64.0 + (slot + 1) / 8.0 + rank / 4.0;
    }

    Tensor residual(int tokens) {
        return Tensor(residual_store.p, DType::BF16, {Config::hidden, tokens});
    }

    Tensor hidden(int tokens) {
        return Tensor(hidden_store.p, DType::BF16, {Config::hidden, tokens});
    }

    Tensor logits(int tokens) {
        return Tensor(logits_store.p, DType::BF16, {Config::output_rows, tokens});
    }

    Tensor ints(int offset, std::initializer_list<std::int32_t> values) {
        Tensor out(static_cast<std::int32_t*>(metadata.p) + offset, DType::I32,
                   {static_cast<int>(values.size())});
        write(out, std::vector<std::int32_t>(values));
        return out;
    }

    void check_state(unsigned tokens, unsigned cut, int step) {
        device.synchronize();
        for (unsigned layer = 0; layer < images.linear().layer_count(); ++layer) {
            for (int slot = 0; slot < images.slot_count(); ++slot) {
                // a=b=a_log=dt_bias=0 => exp(g)=exp(-softplus(0))=1/2. Q/K/V
                // are exactly zero, so no outer-product update survives: S_t=S_0*2^-t.
                const double expected = slot == destination_slot
                                            ? initial(layer, source_slot) * std::pow(0.5, tokens)
                                            : initial(layer, slot);
                for (float value : read<float>(images.linear().recurrent_slot(layer, slot))) {
                    const double tolerance =
                        slot == destination_slot ? 2e-5 * std::abs(expected) + 1e-6 : 0;
                    if (!std::isfinite(value) || std::abs(value - expected) > tolerance) {
                        throw std::runtime_error("rank " + std::to_string(rank) + " GDN layer " +
                                                 std::to_string(layer) + " slot " +
                                                 std::to_string(slot) + " decay or guard mismatch");
                    }
                }
                const auto conv = read<std::uint16_t>(images.linear().conv_slot(layer, slot));
                for (std::size_t index = 0; index < conv.size(); ++index) {
                    const unsigned column    = index / Config::convolution_dim;
                    const bool kept          = slot != destination_slot || column + tokens < 3;
                    const int initial_slot   = slot == destination_slot ? source_slot : slot;
                    const auto expected_conv = kept ? bf16(initial(layer, initial_slot)) : 0;
                    check(conv[index] == expected_conv,
                          "convolution shift or untouched slot changed");
                }
            }
        }
        for (int slot = 0; slot < images.slot_count(); ++slot) {
            for (auto value : read<std::uint16_t>(images.continuation_hidden_slot(slot))) {
                check(value == bf16(initial(0, slot)), "stage overwrote continuation hidden state");
            }
        }
        auto& pool = decoder.text_kv.page_pool();
        check(pool.geometry().device_plane_order == PagedKVPlaneOrder::PageMajor,
              "fixture KV oracle expects page-major physical storage");
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            const Tensor& storage = pool.plane(plane);
            const auto data       = read<std::uint16_t>(storage);
            for (int page = 0; page < storage.ne[3]; ++page) {
                for (int head = 0; head < storage.ne[2]; ++head) {
                    for (int position = 0; position < storage.ne[1]; ++position) {
                        const bool written =
                            page == active_page && position < static_cast<int>(tokens);
                        const auto expected = written ? 0 : bf16(1.0F);
                        for (int channel = 0; channel < storage.ne[0]; ++channel) {
                            const std::size_t index =
                                channel +
                                storage.ne[0] *
                                    (position + storage.ne[1] * (head + storage.ne[2] * page));
                            // Projection/RoPE may produce either IEEE zero sign; this private
                            // boundary specifies mathematical zero, not its sign bit. Guards
                            // remain bit-exact 1.0; every nonzero value and NaN still fails.
                            const bool matches = written ? (data[index] & 0x7fffU) == 0
                                                         : data[index] == expected;
                            if (!matches) {
                                std::ostringstream message;
                                message << "KV write escaped selected row/page/position: cut=" << cut
                                        << " phase=" << (step == 0 ? "prefill" : "decode")
                                        << " step=" << step << " rank=" << rank
                                        << " device=" << device.device << " plane=" << plane
                                        << " page=" << page << " active_page=" << active_page
                                        << " row=" << row.row_index() << " head=" << head
                                        << " position=" << position << " channel=" << channel
                                        << " tokens=" << tokens << " actual_u16=" << data[index]
                                        << " expected_u16=" << expected << " actual_hex=0x"
                                        << std::hex << data[index] << " expected_hex=0x" << expected;
                                throw std::runtime_error(message.str());
                            }
                        }
                    }
                }
            }
        }
        const auto active_table =
            read<std::int32_t>(decoder.text_kv.execution_tables().row(row.handle()));
        const auto guard_table =
            read<std::int32_t>(decoder.text_kv.execution_tables().row(guard_row.handle()));
        check(active_table[0] == active_page && guard_table[0] == guard_page,
              "execution-table row changed during stage evaluation");
    }
};

std::vector<std::uint16_t> embedding_oracle(std::span<const int> ids) {
    std::vector<std::uint16_t> result(ids.size() * Config::hidden);
    for (std::size_t token = 0; token < ids.size(); ++token) {
        for (int column = 0; column < Config::hidden; ++column) {
            result[token * Config::hidden + column] = bf16(((column + ids[token]) % 7 - 3) * 0.25F);
        }
    }
    return result;
}

void check_head(Stage& owner, std::span<const std::uint16_t> expected) {
    owner.device.bind_to_current_thread();
    const int count = expected.size() / Config::hidden;
    Tensor residual = owner.residual(count), hidden = owner.hidden(count),
           logits = owner.logits(count);
    owner.text->pipeline_final_norm(residual, hidden);
    owner.text->pipeline_head(hidden, logits);
    owner.device.synchronize();
    const auto result = read<std::uint16_t>(hidden);
    for (int token = 0; token < count; ++token) {
        double sumsq = 0;
        for (int column = 0; column < Config::hidden; ++column) {
            const double x = number(expected[token * Config::hidden + column]);
            sumsq += x * x;
        }
        const double denominator =
            std::sqrt(sumsq / Config::hidden + static_cast<double>(Config::rms_epsilon));
        for (int column = 0; column < Config::hidden; ++column) {
            const auto index = token * Config::hidden + column;
            // RMSNorm's registered additive weight convention is 1 + 0.25.
            const double oracle = number(expected[index]) * 1.25 / denominator;
            // One BF16 ULP relative allowance covers the final cast and FP32 reduction order.
            check(std::abs(number(result[index]) - oracle) <= std::abs(oracle) / 128.0 + 1e-5,
                  "final norm differs from independent FP64 oracle");
        }
    }
    for (auto value : read<std::uint16_t>(logits)) {
        check(number(value) == 0.0, "zero decoded output-head matrix must produce zero logits");
    }
    check(read<std::uint16_t>(residual) ==
              std::vector<std::uint16_t>(expected.begin(), expected.end()),
          "final norm/head modified caller-owned residual");
}

void run_cut(DeviceContext& owner_device, DeviceContext& tail_device, unsigned cut) {
    owner_device.bind_to_current_thread();
    Stage owner(owner_device, 0, {0, cut});
    tail_device.bind_to_current_thread();
    Stage tail(tail_device, 1, {cut, 64});
    check(owner.row.row_index() != tail.row.row_index() && owner.source_slot != tail.source_slot &&
              owner.destination_slot != tail.destination_slot &&
              owner.active_page != tail.active_page,
          "fixture must use independent rank-local selectors");
    const auto owner_address = owner.residual_store.p;
    const auto tail_address  = tail.residual_store.p;
    for (int step = 0; step < 3; ++step) {
        const int tokens = step == 0 ? 2 : 1;
        const std::array<int, 2> ids =
            step == 0 ? std::array<int, 2>{2, 7} : std::array<int, 2>{10 + step, 0};
        const auto expected = embedding_oracle(std::span<const int>(ids).first(tokens));
        owner_device.bind_to_current_thread();
        Tensor owner_residual = owner.residual(tokens);
        Tensor token_ids      = step == 0 ? owner.ints(0, {2, 7}) : owner.ints(0, {10 + step});
        owner.text->pipeline_embedding(token_ids, owner_residual);
        owner_device.synchronize();
        check(read<std::uint16_t>(owner_residual) == expected, "packed embedding decode oracle");
        std::vector<std::uint16_t> handoff;
        for (Stage* stage : {&owner, &tail}) {
            stage->device.bind_to_current_thread();
            Tensor residual = stage->residual(tokens);
            // Explicit host transport is test-only; each stage receives its own physical buffer.
            // The production peer-transfer lifetime contract is qualified in its separate test.
            if (stage == &tail) { write(residual, handoff); }
            Tensor positions = step == 0 ? stage->ints(2, {0, 1}) : stage->ints(2, {1 + step});
            Tensor row       = stage->ints(4, {stage->row.row_index()});
            if (step == 0) {
                stage->text->pipeline_prefill_stage(residual, positions, positions, row, {2, 2});
            } else {
                Tensor source      = stage->ints(5, {stage->destination_slot});
                Tensor destination = stage->ints(6, {stage->destination_slot});
                const auto visible = static_cast<std::uint32_t>(2 + step);
                stage->text->pipeline_decode_stage(residual, positions, positions, row, source,
                                                   destination, {visible, visible});
            }
            stage->device.synchronize();
            check(read<std::uint16_t>(residual) == expected,
                  "zero output projections must preserve residual exactly");
            stage->work.reset();
            check(read<std::uint16_t>(residual) == expected,
                  "workspace reset clobbered caller-owned residual");
            stage->check_state(2 + step, cut, step);
            handoff = read<std::uint16_t>(residual);
        }
        tail_device.bind_to_current_thread();
        const auto tail_result = read<std::uint16_t>(tail.residual(tokens));
        owner_device.bind_to_current_thread();
        write(owner_residual, tail_result);
        check_head(owner, expected);
        check(owner.residual_store.p == owner_address && tail.residual_store.p == tail_address,
              "stable hidden address changed between calls");
    }
    tail_device.bind_to_current_thread();
    Tensor residual = tail.residual(1), hidden = tail.hidden(1), logits = tail.logits(1);
    Tensor ids = tail.ints(0, {2});
    rejects_tail([&] { tail.text->pipeline_embedding(ids, residual); });
    rejects_tail([&] { tail.text->pipeline_final_norm(residual, hidden); });
    rejects_tail([&] { tail.text->pipeline_head(hidden, logits); });
    tail_device.synchronize();
    owner_device.synchronize();
    std::cout << "pipeline stage cut " << cut << " passed prefill/decode/state FP64 oracles\n";
}
} // namespace

int main() {
    const char* enabled = std::getenv("NINFER_TEST_PIPELINE_STAGE_EXECUTION");
    if (enabled == nullptr || std::string(enabled) != "1") {
        std::cout
            << "SKIP: set NINFER_TEST_PIPELINE_STAGE_EXECUTION=1 for the two-GPU stage oracle\n";
        return 77;
    }
    int count         = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count < 2)) {
        std::cout << "SKIP: two CUDA devices required\n";
        return 77;
    }
    try {
        ScopedCudaErrors errors;
        CUDA_CHECK(status);
        int sm89 = -1, sm120 = -1;
        for (int id = 0; id < count; ++id) {
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            const int sm = prop.major * 10 + prop.minor;
            if (!q36::runtime_support::compiled_groupwise_device(sm, NINFER_COMPILED_SM89,
                                                                 NINFER_COMPILED_SM120)) {
                continue;
            }
            if (sm == 89 && sm89 < 0) { sm89 = id; }
            if (sm == 120 && sm120 < 0) { sm120 = id; }
        }
        if (sm89 < 0 || sm120 < 0) {
            std::cout << "SKIP: compiled SM89 + SM120 device pair required\n";
            return 77;
        }
        DeviceContext owner(sm89), tail(sm120);
        run_cut(owner, tail, 37);
        run_cut(owner, tail, 5);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
