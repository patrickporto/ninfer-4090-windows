#pragma once
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/rank_executor.h"
#include "core/linear_attention_state.h"
#include "core/tensor_collective.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/tensor_projection.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

// TP2 eager dataflow per decoder layer, identical on both ranks (hidden x is replicated BF16):
//   mixer: h = RMSNorm(x) -> column-parallel local heads -> local attention/GDN with rank-local
//          KV/GDN state -> row-parallel output projection -> FP32 partial p_r
//          -> FP32 SUM all-reduce (sequence 2*layer) -> x = BF16(x + SUM p_r)
//   MLP:   h = RMSNorm(x) -> local gate/up SwiGLU (8704) -> row-parallel down -> FP32 partial
//          -> FP32 SUM all-reduce (sequence 2*layer+1) -> x = BF16(x + SUM p_r)
// Residual is added exactly once, after the complete FP32 reduction; partials are never BF16.
// Exactly 128 reductions per pass, in fixed order; one collective unit per pass.
struct TensorLayerCall {
    DeviceContext* device = nullptr;
    WorkspaceArena* work  = nullptr;
    Tensor residual; // BF16 [hidden,T], replicated on both ranks at every sublayer boundary.
    Tensor partial;  // FP32 [hidden,T], rank partial then in-place all-reduced sum.
    const Tensor* cache_positions = nullptr;
    const Tensor* rope_positions  = nullptr;
    const Tensor* kv_rows         = nullptr; // Prefill [1]; Verify [B]. Rank-local rows.
    const qwen3_6::PagedKVCache* kv           = nullptr; // Rank-local 2-KV-head cache.
    const LinearAttentionStatePool* state     = nullptr; // Rank-local 8QK/24V state images.
    qwen3_6::TextPhase phase                  = qwen3_6::TextPhase::Prefill;
    std::int32_t batch                        = 0; // Verify rows (width one); zero for Prefill.
    std::int32_t source_slot                  = 0; // Prefill state selectors.
    std::int32_t destination_slot             = 0;
    const Tensor* source_slots                = nullptr; // Verify state selectors [B].
    const Tensor* destination_slots           = nullptr;
    ops::CausalAttentionExecutionEnvelope envelope{};

    // Mirrors Single's groupwise policy: A8 for prefill at every extent; A16 for decode (Verify)
    // batches that every local A16 route admits (attention input T<=12, GDN input T<=15,
    // SwiGLU/row-parallel T<=32). Larger decode batches keep the A8 profile.
    static constexpr std::int32_t kA16MaxDecodeRows = 8;
    [[nodiscard]] ops::LinearPolicy policy() const noexcept {
        return phase == qwen3_6::TextPhase::Verify && residual.ne[1] <= kA16MaxDecodeRows
                   ? ops::LinearPolicy::A16Only
                   : ops::LinearPolicy::AllowA8;
    }
};

class TensorRankLayers {
public:
    virtual ~TensorRankLayers() = default;
    // Computes the rank partial of reduction `sequence` into call.partial. Runs on the rank owner.
    virtual void sublayer(std::uint32_t sequence, const TensorLayerCall& call) const = 0;
};

namespace tensor_detail {
inline constexpr std::uint32_t kReductionsPerPass = 2U * static_cast<std::uint32_t>(TextConfig::layers);

template <class Allocator>
Tensor matrix(Allocator& allocator, DType dtype, std::int32_t rows, std::int32_t tokens) {
    return allocator.alloc(dtype, {rows, tokens});
}

inline void scratch(WorkspaceLayoutBuilder& layout, std::size_t bytes) {
    if (bytes == 0) return;
    auto scope = layout.scope();
    (void)layout.alloc_bytes(bytes);
}

// Splits the snapshot-convolution output [conv,B] into contiguous Q/K/V operands.
inline void split_rows(const Tensor& source, std::int32_t row_begin, Tensor& destination,
                       cudaStream_t stream) {
    const std::size_t element = dtype_size(source.dtype);
    CUDA_CHECK(cudaMemcpy2DAsync(destination.data, static_cast<std::size_t>(destination.ne[0]) * element,
        static_cast<const std::byte*>(source.data) + static_cast<std::size_t>(row_begin) * element,
        static_cast<std::size_t>(source.ne[0]) * element,
        static_cast<std::size_t>(destination.ne[0]) * element,
        static_cast<std::size_t>(source.ne[1]), cudaMemcpyDeviceToDevice, stream));
}
} // namespace tensor_detail

// Rank-local mathematics over one package ModelView holding the TP2 local shards. Only exact
// variants that register supports_tensor_parallel instantiate it.
template <class V>
class TensorRankRunner final : public TensorRankLayers {
public:
    using Local = TensorLocalConfig;
    explicit TensorRankRunner(const typename V::ModelView& model) : model_(model) {}

    void sublayer(std::uint32_t sequence, const TensorLayerCall& call) const override {
        if (sequence >= tensor_detail::kReductionsPerPass || !call.device || !call.work ||
            !call.kv || !call.state || !call.cache_positions || !call.rope_positions ||
            !call.kv_rows || call.residual.ne[1] != call.partial.ne[1]) {
            throw std::logic_error("tensor sublayer call is incomplete");
        }
        const int layer = static_cast<int>(sequence / 2);
        auto scope      = call.work->scope();
        if (TextConfig::is_full_attention(layer)) {
            const auto index = TextConfig::full_attention_index(layer);
            const auto& w    = model_.full_layers.at(static_cast<std::size_t>(index));
            if (sequence % 2 == 0) attention(w, static_cast<std::uint32_t>(index), call);
            else mlp(w.post_attention_norm, w.post_mixer, call);
        } else {
            const auto index = TextConfig::gdn_index(layer);
            const auto& w    = model_.gdn_layers.at(static_cast<std::size_t>(index));
            if (sequence % 2 == 0) gdn(w, static_cast<std::uint32_t>(index), call);
            else mlp(w.post_attention_norm, w.post_mixer, call);
        }
    }

    // Startup simulation of exactly the allocations made by sublayer() for one phase interval.
    static std::size_t workspace_bytes(qwen3_6::TextPhase phase, std::int32_t batch,
                                       std::int32_t first, std::int32_t last,
                                       KvCacheStorage storage,
                                       ops::CausalAttentionExecutionEnvelope envelope) {
        using tensor_detail::matrix;
        using tensor_detail::scratch;
        const bool verify = phase == qwen3_6::TextPhase::Verify;
        const std::int32_t min_width = verify ? 1 : first;
        const std::int32_t max_width = verify ? 1 : last;
        const std::int32_t rows      = verify ? batch : 1;
        std::size_t peak             = 0;
        {
            WorkspaceLayoutBuilder layout;
            matrix(layout, DType::BF16, Local::hidden, last);
            matrix(layout, DType::BF16, Local::query_size, last);
            matrix(layout, DType::BF16, Local::query_size, last);
            matrix(layout, DType::BF16, Local::kv_size, last);
            matrix(layout, DType::BF16, Local::kv_size, last);
            scratch(layout, V::tensor_attention_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, Local::query_size, last);
            matrix(layout, DType::BF16, Local::kv_size, last);
            matrix(layout, DType::BF16, Local::query_size, last);
            scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                {Local::head_dim, Local::query_heads, Local::kv_heads}, storage,
                                envelope, rows, min_width, max_width));
            scratch(layout, ops::tensor_projection_workspace_capacity_bytes(Local::query_size,
                                                                            first, last));
            peak = std::max(peak, layout.peak_bytes(1));
        }
        {
            WorkspaceLayoutBuilder layout;
            matrix(layout, DType::BF16, Local::hidden, last);
            matrix(layout, DType::FP32, Local::gdn_value_heads, last);
            matrix(layout, DType::FP32, Local::gdn_value_heads, last);
            scratch(layout, ops::gdn_norm_gating_proj_workspace_capacity_bytes(
                                Local::gdn_value_heads, Local::hidden, first, last));
            matrix(layout, DType::BF16, Local::convolution_dim, last);
            matrix(layout, DType::BF16, Local::value_dim, last);
            scratch(layout, V::tensor_gdn_input_projection_workspace_capacity_bytes(first, last));
            if (verify) matrix(layout, DType::BF16, Local::convolution_dim, last);
            matrix(layout, DType::BF16, Local::key_dim, last);
            matrix(layout, DType::BF16, Local::key_dim, last);
            matrix(layout, DType::BF16, Local::value_dim, last);
            matrix(layout, DType::BF16, Local::value_dim, last);
            if (!verify)
                scratch(layout, ops::gated_delta_net_workspace_capacity_bytes(
                                    Local::gdn_key_heads, Local::gdn_value_heads, true, first, last));
            matrix(layout, DType::BF16, Local::value_dim, last);
            scratch(layout, ops::tensor_projection_workspace_capacity_bytes(Local::value_dim,
                                                                            first, last));
            peak = std::max(peak, layout.peak_bytes(1));
        }
        {
            WorkspaceLayoutBuilder layout;
            matrix(layout, DType::BF16, Local::hidden, last);
            scratch(layout, V::tensor_post_mixer_workspace_capacity_bytes(first, last));
            peak = std::max(peak, layout.peak_bytes(1));
        }
        return peak;
    }

private:
    template <class Weights>
    void attention(const Weights& w, std::uint32_t index, const TensorLayerCall& call) const {
        auto& work         = *call.work;
        const Tensor& x    = call.residual;
        cudaStream_t s     = call.device->stream;
        const int T        = x.ne[1];
        const bool verify  = call.phase == qwen3_6::TextPhase::Verify;
        Tensor h           = work.alloc(DType::BF16, {Local::hidden, T});
        ops::rmsnorm(x, w.input_norm, TextConfig::rms_epsilon, true, h, s);
        Tensor q    = work.alloc(DType::BF16, {Local::query_size, T});
        Tensor gate = work.alloc(DType::BF16, {Local::query_size, T});
        Tensor k    = work.alloc(DType::BF16, {Local::kv_size, T});
        Tensor v    = work.alloc(DType::BF16, {Local::kv_size, T});
        V::tensor_attention_projection(h, w.projection, q, gate, k, v, call.policy(), work, s);
        Tensor qn = work.alloc(DType::BF16, {Local::query_size, T});
        Tensor kn = work.alloc(DType::BF16, {Local::kv_size, T});
        Tensor a  = work.alloc(DType::BF16, {Local::query_size, T});
        Tensor q3 = q.view({Local::head_dim, Local::query_heads, T});
        Tensor k3 = k.view({Local::head_dim, Local::kv_heads, T});
        Tensor v3 = v.view({Local::head_dim, Local::kv_heads, T});
        Tensor qn3 = qn.view({Local::head_dim, Local::query_heads, T});
        Tensor kn3 = kn.view({Local::head_dim, Local::kv_heads, T});
        Tensor a3  = a.view({Local::head_dim, Local::query_heads, T});
        Tensor gate3 = gate.view({Local::head_dim, Local::query_heads, T});
        ops::rmsnorm(q3, w.query_norm, TextConfig::rms_epsilon, true, qn3, s);
        ops::rmsnorm(k3, w.key_norm, TextConfig::rms_epsilon, true, kn3, s);
        Tensor rope = verify ? call.rope_positions->view({T}) : *call.rope_positions;
        ops::rope(rope, TextConfig::rotary_dim, TextConfig::rope_theta, qn3, kn3, s);
        const ops::AttentionHeadGeometry geometry{Local::head_dim, Local::query_heads,
                                                  Local::kv_heads};
        if (verify) {
            const std::int32_t batch = call.batch;
            if (batch <= 0 || batch != T) throw std::logic_error("tensor decode batch mismatch");
            Tensor q_batch = qn.view({Local::head_dim, Local::query_heads, 1, batch});
            Tensor k_batch = kn.view({Local::head_dim, Local::kv_heads, 1, batch});
            Tensor v_batch = v.view({Local::head_dim, Local::kv_heads, 1, batch});
            Tensor a_batch = a.view({Local::head_dim, Local::query_heads, 1, batch});
            Tensor positions = call.cache_positions->view({1, batch});
            ops::causal_softmax_attention(q_batch, k_batch, v_batch, positions, Tensor{},
                                          *call.kv_rows, geometry, V::attention_scale,
                                          call.kv->batch_layer_view(index), call.envelope, work,
                                          a_batch, s);
        } else {
            ops::causal_softmax_attention(qn3, kn3, v3, *call.cache_positions, Tensor{},
                                          *call.kv_rows, geometry, V::attention_scale,
                                          call.kv->batch_layer_view(index), call.envelope, work,
                                          a3, s);
        }
        ops::sigmoid_mul(gate3, a3, s);
        Tensor partial = call.partial;
        ops::tensor_projection(a, w.output, partial, call.policy(), work, s);
    }

    template <class Weights>
    void gdn(const Weights& w, std::uint32_t index, const TensorLayerCall& call) const {
        auto& work        = *call.work;
        const Tensor& x   = call.residual;
        cudaStream_t s    = call.device->stream;
        const int T       = x.ne[1];
        const bool verify = call.phase == qwen3_6::TextPhase::Verify;
        Tensor h    = work.alloc(DType::BF16, {Local::hidden, T});
        Tensor g    = work.alloc(DType::FP32, {Local::gdn_value_heads, T});
        Tensor beta = work.alloc(DType::FP32, {Local::gdn_value_heads, T});
        V::gdn_norm_control_projection(x, w.input_norm, TextConfig::rms_epsilon, w.projection, h,
                                       g, beta, work, call.device->execution_view());
        Tensor qkv = work.alloc(DType::BF16, {Local::convolution_dim, T});
        Tensor z   = work.alloc(DType::BF16, {Local::value_dim, T});
        V::tensor_gdn_input_projection(h, w.projection, qkv, z, call.policy(), work, s);
        Tensor conv;
        if (verify) conv = work.alloc(DType::BF16, {Local::convolution_dim, T});
        Tensor q = work.alloc(DType::BF16, {Local::key_dim, T});
        Tensor k = work.alloc(DType::BF16, {Local::key_dim, T});
        Tensor v = work.alloc(DType::BF16, {Local::value_dim, T});
        Tensor o = work.alloc(DType::BF16, {Local::value_dim, T});
        if (verify) {
            const std::int32_t batch = call.batch;
            if (batch <= 0 || batch != T || !call.source_slots || !call.destination_slots)
                throw std::logic_error("tensor GDN decode binding is incomplete");
            const auto layer = call.state->layer_view(index);
            Tensor conv_states = layer.conv;
            Tensor recurrent   = layer.recurrent;
            Tensor conv_input  = qkv.view({Local::convolution_dim, 1, batch});
            Tensor conv_output = conv.view({Local::convolution_dim, 1, batch});
            ops::causal_conv1d_silu_snapshot(conv_input, w.convolution, conv_states, Tensor{},
                                             *call.source_slots, *call.destination_slots,
                                             conv_output, s);
            tensor_detail::split_rows(conv, 0, q, s);
            tensor_detail::split_rows(conv, Local::key_dim, k, s);
            tensor_detail::split_rows(conv, 2 * Local::key_dim, v, s);
            Tensor q_batch = q.view({Local::gdn_key_head_dim, Local::gdn_key_heads, 1, batch});
            Tensor k_batch = k.view({Local::gdn_key_head_dim, Local::gdn_key_heads, 1, batch});
            Tensor v_batch = v.view({Local::gdn_value_head_dim, Local::gdn_value_heads, 1, batch});
            Tensor o_batch = o.view({Local::gdn_value_head_dim, Local::gdn_value_heads, 1, batch});
            Tensor g_batch    = g.view({Local::gdn_value_heads, 1, batch});
            Tensor beta_batch = beta.view({Local::gdn_value_heads, 1, batch});
            ops::gated_delta_net_batch_update(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                              V::gdn_scale, /*normalize_qk=*/true, recurrent,
                                              *call.source_slots, *call.destination_slots,
                                              o_batch, s);
        } else {
            Tensor conv_in  = call.state->conv_slot(index, call.source_slot);
            Tensor conv_out = call.state->conv_slot(index, call.destination_slot);
            ops::causal_conv1d_silu_split(qkv, w.convolution, conv_in, conv_out, q, k, v, s);
            Tensor state_in  = call.state->recurrent_slot(index, call.source_slot);
            Tensor state_out = call.state->recurrent_slot(index, call.destination_slot);
            Tensor q3 = q.view({Local::gdn_key_head_dim, Local::gdn_key_heads, T});
            Tensor k3 = k.view({Local::gdn_key_head_dim, Local::gdn_key_heads, T});
            Tensor v3 = v.view({Local::gdn_value_head_dim, Local::gdn_value_heads, T});
            Tensor o3 = o.view({Local::gdn_value_head_dim, Local::gdn_value_heads, T});
            ops::gated_delta_net(q3, k3, v3, g, beta, V::gdn_scale, /*normalize_qk=*/true, work,
                                 state_in, state_out, o3, s);
        }
        Tensor on  = work.alloc(DType::BF16, {Local::value_dim, T});
        Tensor o3  = o.view({Local::gdn_value_head_dim, Local::gdn_value_heads, T});
        Tensor z3  = z.view({Local::gdn_value_head_dim, Local::gdn_value_heads, T});
        Tensor on3 = on.view({Local::gdn_value_head_dim, Local::gdn_value_heads, T});
        ops::gated_rmsnorm(o3, w.norm, z3, TextConfig::rms_epsilon, on3, s);
        Tensor partial = call.partial;
        ops::tensor_projection(on, w.output, partial, call.policy(), work, s);
    }

    template <class PostMixer>
    void mlp(const Tensor& norm, const PostMixer& weights, const TensorLayerCall& call) const {
        auto& work      = *call.work;
        const Tensor& x = call.residual;
        cudaStream_t s  = call.device->stream;
        Tensor h        = work.alloc(DType::BF16, {Local::hidden, x.ne[1]});
        ops::rmsnorm(x, norm, TextConfig::rms_epsilon, true, h, s);
        Tensor partial = call.partial;
        V::tensor_post_mixer(h, weights, partial, call.policy(), work, s);
    }

    const typename V::ModelView& model_;
};

template <class V>
concept TensorParallelVariant = requires { requires V::supports_tensor_parallel; };

template <class V>
std::unique_ptr<TensorRankLayers> make_tensor_rank_layers(const typename V::ModelView& model) {
    if constexpr (TensorParallelVariant<V>) {
        return std::make_unique<TensorRankRunner<V>>(model);
    } else {
        (void)model;
        throw std::invalid_argument("tensor-parallel execution is not registered for this target");
    }
}

template <class V>
std::size_t tensor_layer_workspace_bytes(qwen3_6::TextPhase phase, std::int32_t batch,
                                         std::int32_t first, std::int32_t last,
                                         KvCacheStorage storage,
                                         ops::CausalAttentionExecutionEnvelope envelope) {
    if constexpr (TensorParallelVariant<V>) {
        return TensorRankRunner<V>::workspace_bytes(phase, batch, first, last, storage, envelope);
    } else {
        (void)phase; (void)batch; (void)first; (void)last; (void)storage; (void)envelope;
        throw std::invalid_argument("tensor-parallel planning is not registered for this target");
    }
}

// Owns the two-rank collective and the rank-0 owner thread. Rank 1 reuses its physical stage
// executor. Construction initializes NCCL on both owners; destruction closes both on their owners.
// Each sublayer is one coordinator-prepared ticket executed concurrently by both rank owners;
// any rank failure quarantines the collective so its peer leaves the reduction promptly.
class TensorParallelUnit {
public:
    TensorParallelUnit(const LoadedModelData& owner_model, const LoadedModelData& tail_model,
                       DeviceContext& owner, DeviceContext& tail,
                       runtime_support::RankExecutor& tail_executor, std::uint64_t& tail_command,
                       std::uint32_t max_columns)
        : tail_executor_(tail_executor), tail_command_(tail_command),
          owner_executor_(std::make_unique<runtime_support::RankExecutor>(owner)),
          collective_({owner.device, tail.device},
                      static_cast<std::size_t>(TextConfig::hidden) * max_columns,
                      std::chrono::seconds(30), tensor_detail::kReductionsPerPass),
          layers_{make_tensor_rank_layers<Variant>(owner_model),
                  make_tensor_rank_layers<Variant>(tail_model)} {
        if (!TensorCollective::available())
            throw std::invalid_argument("TP2 requires a build with NINFER_ENABLE_NCCL");
        both([&](std::size_t rank, DeviceContext& context) {
            collective_.initialize_rank(rank, context);
        });
    }
    ~TensorParallelUnit() noexcept {
        try {
            both([&](std::size_t rank, DeviceContext&) { collective_.close_rank(rank); });
        } catch (...) {
            // Owner-thread guards abort any remaining communicator when the executor exits.
            collective_.quarantine();
        }
        owner_executor_.reset();
    }
    TensorParallelUnit(const TensorParallelUnit&)            = delete;
    TensorParallelUnit& operator=(const TensorParallelUnit&) = delete;

    [[nodiscard]] bool failed() const noexcept { return collective_.failed(); }

    // Executes all 2*layers reductions for one pass. The caller has already broadcast the
    // owner-produced embedding into both residuals and uploaded rank-local metadata.
    void pass(const std::array<TensorLayerCall, 2>& calls) {
        const std::int32_t columns = calls[0].residual.ne[1];
        if (columns <= 0 || calls[1].residual.ne[1] != columns)
            throw std::logic_error("tensor pass ranks disagree on column count");
        const auto count = static_cast<std::size_t>(TextConfig::hidden) * columns;
        const std::uint64_t unit = next_unit_++;
        // One ticket per pass: each rank owner enqueues all sublayers and their stream-ordered
        // all-reduces back to back, then waits once for both ranks' streams.
        const auto ticket = collective_.prepare_unit(unit, tensor_detail::kReductionsPerPass, count);
        both([&, ticket](std::size_t rank, DeviceContext&) {
            const TensorLayerCall& call = calls[rank];
            collective_.begin_unit(ticket, rank);
            Tensor residual = call.residual;
            auto& prof = profile_[rank];
            const bool profiling = prof.enabled();
            const auto host_start = std::chrono::steady_clock::now();
            if (profiling) prof.record(0, call.device->stream);
            for (std::uint32_t sequence = 0; sequence < tensor_detail::kReductionsPerPass; ++sequence) {
                layers_[rank]->sublayer(sequence, call);
                if (profiling) prof.record(2 * sequence + 1, call.device->stream);
                collective_.enqueue_sum(ticket, rank, sequence, static_cast<float*>(call.partial.data));
                ops::tensor_residual_add(call.partial, residual, call.device->stream);
                if (profiling) prof.record(2 * sequence + 2, call.device->stream);
            }
            const auto host_enqueued = std::chrono::steady_clock::now();
            collective_.finish_unit(ticket, rank);
            if (profiling) prof.accumulate(rank, columns, host_start, host_enqueued);
        });
    }

private:
    // Submits both owner commands before waiting for either, then waits for both. A failure on
    // either side latches the collective so the peer's blocked reduction returns.
    template <class Function>
    void both(Function&& function) {
        std::array<std::future<void>, 2> done;
        std::exception_ptr error;
        try {
            done[0] = owner_executor_->submit(owner_command_++, [&](DeviceContext& context) {
                function(std::size_t{0}, context);
            });
            done[1] = tail_executor_.submit(tail_command_++, [&](DeviceContext& context) {
                function(std::size_t{1}, context);
            });
        } catch (...) {
            error = std::current_exception();
            collective_.quarantine();
        }
        for (auto& future : done) {
            if (!future.valid()) continue;
            try {
                future.get();
            } catch (...) {
                if (!error) error = std::current_exception();
                collective_.quarantine();
            }
        }
        if (error) std::rethrow_exception(error);
    }

    runtime_support::RankExecutor& tail_executor_;
    std::uint64_t& tail_command_;
    std::unique_ptr<runtime_support::RankExecutor> owner_executor_;
    std::uint64_t owner_command_ = 1;
    TensorCollective collective_;
    std::array<std::unique_ptr<TensorRankLayers>, 2> layers_;
    std::uint64_t next_unit_ = 1;

    // Opt-in (NINFER_TP_PROFILE=1) per-rank breakdown of one pass: GPU time inside sublayers vs
    // inside all-reduce+residual, and host enqueue time. Prints an average every 64 passes.
    struct PassProfile {
        static constexpr int kEvents = 2 * tensor_detail::kReductionsPerPass + 1;
        std::array<cudaEvent_t, kEvents> events{};
        bool created = false;
        double compute_ms = 0, reduce_ms = 0, host_ms = 0, gpu_ms = 0;
        std::uint32_t passes = 0;
        static bool env() {
            static const bool on = [] { const char* v = std::getenv("NINFER_TP_PROFILE"); return v && v[0] == '1'; }();
            return on;
        }
        bool enabled() const { return env(); }
        void record(int index, cudaStream_t stream) {
            if (!created) {
                for (auto& e : events) CUDA_CHECK(cudaEventCreate(&e));
                created = true;
            }
            CUDA_CHECK(cudaEventRecord(events[static_cast<std::size_t>(index)], stream));
        }
        void accumulate(std::size_t rank, std::int32_t columns, std::chrono::steady_clock::time_point h0,
                        std::chrono::steady_clock::time_point h1) {
            for (int i = 0; i < static_cast<int>(tensor_detail::kReductionsPerPass); ++i) {
                float a = 0, b = 0;
                CUDA_CHECK(cudaEventElapsedTime(&a, events[2 * i], events[2 * i + 1]));
                CUDA_CHECK(cudaEventElapsedTime(&b, events[2 * i + 1], events[2 * i + 2]));
                compute_ms += a; reduce_ms += b;
            }
            float total = 0;
            CUDA_CHECK(cudaEventElapsedTime(&total, events[0], events[kEvents - 1]));
            gpu_ms += total;
            host_ms += std::chrono::duration<double, std::milli>(h1 - h0).count();
            if (++passes == 64) {
                std::fprintf(stderr, "TP_PROFILE rank=%zu cols=%d avg/pass gpu=%.3fms compute=%.3fms reduce=%.3fms host_enqueue=%.3fms\n",
                             rank, columns, gpu_ms / passes, compute_ms / passes, reduce_ms / passes, host_ms / passes);
                compute_ms = reduce_ms = host_ms = gpu_ms = 0; passes = 0;
            }
        }
    };
    std::array<PassProfile, 2> profile_{};
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
