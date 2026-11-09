#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "targets/qwen3_6/impl/runtime/pipeline_schedule.h"
#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/state_image.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    std::optional<qwen3_6::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

// Fixed TP2 rank-local geometry: complete GQA6 and GDN3 head groups and K128 MLP channels.
// Only the registered TP2 package plan uses it; Single and PP keep TextConfig geometry.
struct TensorLocalConfig {
    static constexpr int degree             = 2;
    static constexpr int hidden             = TextConfig::hidden;
    static constexpr int head_dim           = TextConfig::head_dim;
    static constexpr int query_heads        = TextConfig::query_heads / degree;
    static constexpr int kv_heads           = TextConfig::kv_heads / degree;
    static constexpr int query_size         = head_dim * query_heads;
    static constexpr int kv_size            = head_dim * kv_heads;
    static constexpr int gdn_key_heads      = TextConfig::gdn_key_heads / degree;
    static constexpr int gdn_value_heads    = TextConfig::gdn_value_heads / degree;
    static constexpr int gdn_key_head_dim   = TextConfig::gdn_key_head_dim;
    static constexpr int gdn_value_head_dim = TextConfig::gdn_value_head_dim;
    static constexpr int key_dim            = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim          = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim    = 2 * key_dim + value_dim;
    static constexpr int intermediate       = TextConfig::intermediate / degree;
};

// NCCL allocates its communicator buffers outside NInfer arenas during Program construction.
// Reserve them explicitly in each TP rank's vector budget rather than relying on slack.
inline constexpr std::size_t kTensorCommunicationAllowanceBytes = 256ULL * 1024ULL * 1024ULL;

struct PipelineBufferLayout {
    TensorRegion residual, ids, positions, rope_positions, rows, sources, destinations, partial;
    std::size_t bytes;
    // `tensor` adds the TP2 FP32 row-parallel partial/all-reduce buffer; PP layouts are unchanged.
    static PipelineBufferLayout make(std::uint32_t chunk, bool tensor = false) {
        LayoutBuilder builder;
        const auto columns = static_cast<std::int32_t>(std::max(chunk, kMaximumConcurrency));
        PipelineBufferLayout out;
        out.residual = builder.add_tensor(DType::BF16, {TextConfig::hidden, columns}, 256,
                                          "pipeline residual");
        auto metadata = [&] (const char* name) {
            return builder.add_tensor(DType::I32, {columns}, 256, name);
        };
        out.ids = metadata("pipeline ids");
        out.positions = metadata("pipeline cache positions");
        out.rope_positions = metadata("pipeline RoPE positions");
        out.rows = metadata("pipeline KV rows");
        out.sources = metadata("pipeline source slots");
        out.destinations = metadata("pipeline destination slots");
        if (tensor)
            out.partial = builder.add_tensor(DType::FP32, {TextConfig::hidden, columns}, 256,
                                             "tensor FP32 partial sum");
        out.bytes = builder.finish(256);
        return out;
    }
};

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    qwen3_6::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    TensorLayout token_masks;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct VisionWorkspacePlan {
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    std::uint32_t turn_checkpoint_ring     = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    std::optional<PhysicalLayerRange> pipeline_layers;
    std::optional<std::uint32_t> tensor_rank; // TP2 rank-local geometry; rank 0 owns endpoints.
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN36_VARIANT> {
    typename NINFER_QWEN36_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    std::uint32_t turn_checkpoint_ring     = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    std::optional<PhysicalLayerRange> pipeline_layers;
    std::optional<std::uint32_t> tensor_rank;
    NINFER_QWEN36_RUNTIME_NS::PersistentLayout persistent;
    NINFER_QWEN36_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

template <>
struct SequencePlannerImpl<NINFER_QWEN36_VARIANT> {
    NINFER_QWEN36_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl<NINFER_QWEN36_VARIANT>> minimum;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using SequencePlanImpl = qwen3_6::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile,
                           std::optional<PhysicalLayerRange> pipeline_layers = std::nullopt,
                           std::optional<std::uint32_t> tensor_rank = std::nullopt);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
