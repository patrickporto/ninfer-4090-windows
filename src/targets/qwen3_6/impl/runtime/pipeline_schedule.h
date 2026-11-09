#pragma once

#include <ninfer/types.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::targets::qwen3_6::runtime_support {

// A physical stage retains global weight indices, but owns only its local KV/GDN inventory.
// Deliberately independent of loaded weight views: placement is a runtime ownership fact.
class StageLayerLayout final {
public:
    [[nodiscard]] static StageLayerLayout create(PhysicalLayerRange range,
                                                 std::uint32_t total_layers) {
        if (range.begin >= range.end || range.end > total_layers) {
            throw std::invalid_argument("pipeline physical layer range is invalid");
        }
        return StageLayerLayout(range);
    }
    [[nodiscard]] PhysicalLayerRange range() const noexcept { return range_; }
    [[nodiscard]] std::uint32_t full_attention_count() const noexcept {
        return range_.end / 4 - range_.begin / 4;
    }
    [[nodiscard]] std::uint32_t gdn_count() const noexcept {
        return range_.end - range_.begin - full_attention_count();
    }
    [[nodiscard]] std::uint32_t local_full_attention_index(std::uint32_t layer) const {
        require_layer(layer);
        if (!is_full_attention_layer(layer)) {
            throw std::invalid_argument("pipeline layer is not full attention");
        }
        return static_cast<std::uint32_t>(full_attention_index(layer)) - range_.begin / 4;
    }
    [[nodiscard]] std::uint32_t local_gdn_index(std::uint32_t layer) const {
        require_layer(layer);
        if (is_full_attention_layer(layer)) {
            throw std::invalid_argument("pipeline layer is not GDN");
        }
        return static_cast<std::uint32_t>(gdn_index(layer)) - (range_.begin - range_.begin / 4);
    }
    void require_pipeline_pools() const {
        if (full_attention_count() == 0) {
            throw std::invalid_argument("M2 pipeline stage requires at least one attention layer");
        }
        if (gdn_count() == 0) {
            throw std::invalid_argument("M2 pipeline stage requires at least one GDN layer");
        }
    }
private:
    explicit StageLayerLayout(PhysicalLayerRange range) : range_(range) {}
    void require_layer(std::uint32_t layer) const {
        if (layer < range_.begin || layer >= range_.end) {
            throw std::invalid_argument("layer is outside pipeline stage");
        }
    }
    PhysicalLayerRange range_;
};

// Preparation writes each acquired handle immediately into its rank's bundle. Abort must
// release even a partially prepared bundle, in reverse acquisition order, and must not throw.
// No logical sequence is published until both physical preparations have succeeded. CUDA
// operations queued during preparation must be completed by abort before freeing resources.
// The concrete Program supplies its StateImage/KV handle bundle; ranks never share slot indices.
template<class StageHandles, class Prepare, class Abort>
[[nodiscard]] std::optional<std::array<StageHandles, 2>>
prepare_pipeline_active(Prepare&& prepare, Abort&& abort) {
    static_assert(std::is_nothrow_invocable_v<Abort&, unsigned, StageHandles&>);
    static_assert(std::is_nothrow_move_constructible_v<StageHandles>);
    std::array<StageHandles, 2> handles{};
    unsigned attempted = 0;
    const auto rollback = [&]() noexcept {
        while (attempted != 0) {
            --attempted;
            abort(attempted, handles[attempted]);
        }
    };
    try {
        for (unsigned rank = 0; rank < 2; ++rank) {
            ++attempted;
            if (!prepare(rank, handles[rank])) {
                rollback();
                return std::nullopt;
            }
        }
    } catch (...) {
        rollback();
        throw;
    }
    return std::optional<std::array<StageHandles, 2>>(std::move(handles));
}

[[nodiscard]] constexpr bool compiled_groupwise_device(int capability, bool sm89,
                                                       bool sm120) noexcept {
    return (capability == 89 && sm89) || (capability == 120 && sm120);
}

inline void require_pipeline_eager_options(const EngineOptions& options) {
    if (options.turn_checkpoint_ring != 0 || options.auto_save_evicted) {
        throw std::invalid_argument(
            "M2 pipeline does not support turn checkpoints or automatic checkpoint spill");
    }
    if (options.use_cuda_graph || options.enable_vision ||
        options.speculative.backend != SpeculativeBackend::None || options.context_cache.enabled) {
        throw std::invalid_argument(
            "M2 pipeline requires eager text execution with speculation and context cache disabled");
    }
}

// TP2 eager MVP: speculation (MTP), CUDA graphs, Vision, context reuse and checkpoints are
// rejected explicitly. Local attention is qualified for BF16 and E8 rk4v4 KV only, and one FP32
// all-reduce covers at most 1024 prefill columns.
inline void require_tensor_eager_options(const EngineOptions& options) {
    if (options.turn_checkpoint_ring != 0 || options.auto_save_evicted) {
        throw std::invalid_argument(
            "TP2 does not support turn checkpoints or automatic checkpoint spill");
    }
    if (options.use_cuda_graph || options.enable_vision ||
        options.speculative.backend != SpeculativeBackend::None || options.context_cache.enabled) {
        throw std::invalid_argument(
            "TP2 requires eager text execution with CUDA graphs, Vision, speculation (MTP) and "
            "context cache disabled");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::RK4V4E8) {
        throw std::invalid_argument("TP2 local attention supports only bf16 and E8 rk4v4 KV storage");
    }
    if (options.prefill_chunk > 1024) {
        throw std::invalid_argument("TP2 prefill_chunk must be at most 1024");
    }
}

// Geometry and registered identity validation follow through the exact package ParallelPlan
// before CUDA initialization; this gate closes unavailable execution modes immediately.
inline void require_runtime_placement(const EngineOptions& options) {
    switch (options.placement.mode) {
    case ParallelMode::Single:
        if (!options.placement.devices.empty() || !options.placement.pipeline_layers.empty())
            throw std::invalid_argument("Single placement requires empty rank and layer vectors");
        return;
    case ParallelMode::Pipeline:
    case ParallelMode::Tensor:
        return;
    default:
        throw std::invalid_argument("Engine parallel execution mode is invalid");
    }
}

} // namespace ninfer::targets::qwen3_6::runtime_support
