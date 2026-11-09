#pragma once

#include <ninfer/types.h>

namespace ninfer::targets::qwen3_6_27b::detail {

// Distinct types prevent accidentally indexing a rank-local buffer by a global head.
struct LocalHeadIndex {
    std::uint32_t value;
};

struct GlobalHeadIndex {
    std::uint32_t value;
};

struct HeadMapping {
    LocalHeadIndex local;
    GlobalHeadIndex global;
};

struct ChannelRange {
    std::uint32_t begin;
    std::uint32_t end;
};

struct LayerMapping {
    std::uint32_t local;  // Index in the rank-local attention or GDN store.
    std::uint32_t global; // Index in the global attention or GDN inventory.
    std::uint32_t physical;
};

struct HeadShard final {
    const std::vector<HeadMapping> query;
    const std::vector<HeadMapping> kv;
    const std::vector<HeadMapping> gdn_qk;
    const std::vector<HeadMapping> gdn_value;
    const ChannelRange mlp;
    // Validates complete GQA6/GDN3 groups and K128-aligned MLP channels.
    [[nodiscard]] static HeadShard create(ChannelRange query, ChannelRange kv, ChannelRange gdn_qk,
                                          ChannelRange gdn_value, ChannelRange mlp);
private:
    HeadShard(std::vector<HeadMapping> q, std::vector<HeadMapping> k, std::vector<HeadMapping> gq,
              std::vector<HeadMapping> gv, ChannelRange m);
};

struct ParallelRank final {
    const std::uint32_t rank;
    const int device;
    const PhysicalLayerRange layers;
    const std::vector<LayerMapping> full_attention_layers;
    const std::vector<LayerMapping> gdn_layers;
    const HeadShard heads;
};

// CPU-only closed geometry contract. Construct after artifact identity inspection and before
// any CUDA context, arena or weight allocation. This does not qualify runtime capabilities.
class ParallelPlan final {
public:
    [[nodiscard]] static ParallelPlan create(const EngineOptions& options, std::string_view target,
                                             std::string_view storage_profile);

    [[nodiscard]] ParallelMode mode() const noexcept { return mode_; }

    [[nodiscard]] const std::vector<ParallelRank>& ranks() const noexcept { return ranks_; }
private:
    ParallelPlan(ParallelMode mode, std::vector<ParallelRank> ranks)
        : mode_(mode), ranks_(std::move(ranks)) {}

    const ParallelMode mode_;
    const std::vector<ParallelRank> ranks_;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
