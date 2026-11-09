#include "parallel_plan.h"

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

bool bounded(ChannelRange r, std::uint32_t limit) { return r.begin < r.end && r.end <= limit; }

std::vector<HeadMapping> heads(ChannelRange r) {
    std::vector<HeadMapping> result;
    for (auto i = r.begin; i < r.end; ++i) result.push_back({{i - r.begin}, {i}});
    return result;
}

ParallelRank rank(std::uint32_t index, int device, PhysicalLayerRange range, HeadShard h) {
    std::vector<LayerMapping> full, gdn;
    for (auto layer = range.begin; layer < range.end; ++layer) {
        if (layer % 4 == 3)
            full.push_back({static_cast<std::uint32_t>(full.size()), layer / 4, layer});
        else
            gdn.push_back({static_cast<std::uint32_t>(gdn.size()), layer - layer / 4, layer});
    }
    return {index, device, range, std::move(full), std::move(gdn), std::move(h)};
}

HeadShard whole_heads() { return HeadShard::create({0, 24}, {0, 4}, {0, 16}, {0, 48}, {0, 17408}); }
} // namespace

HeadShard::HeadShard(std::vector<HeadMapping> q, std::vector<HeadMapping> k,
                     std::vector<HeadMapping> gq, std::vector<HeadMapping> gv, ChannelRange m)
    : query(std::move(q)), kv(std::move(k)), gdn_qk(std::move(gq)), gdn_value(std::move(gv)),
      mlp(m) {}

HeadShard HeadShard::create(ChannelRange q, ChannelRange k, ChannelRange gq, ChannelRange gv,
                            ChannelRange m) {
    require(bounded(q, 24) && bounded(k, 4) && q.begin == 6 * k.begin && q.end == 6 * k.end,
            "GQA partition must preserve complete groups of six query heads per KV head");
    require(bounded(gq, 16) && bounded(gv, 48) && gv.begin == 3 * gq.begin && gv.end == 3 * gq.end,
            "GDN partition must preserve complete groups of three value heads per Q/K head");
    require(bounded(m, 17408) && m.begin % 128 == 0 && m.end % 128 == 0,
            "MLP channel range must be nonempty, bounded and K128 aligned");
    return {heads(q), heads(k), heads(gq), heads(gv), m};
}

ParallelPlan ParallelPlan::create(const EngineOptions& options, std::string_view target,
                                  std::string_view profile) {
    const auto& p = options.placement;
    require(options.device >= 0, "device ordinals must be nonnegative");
    require(p.mode == ParallelMode::Single || p.mode == ParallelMode::Pipeline ||
                p.mode == ParallelMode::Tensor,
            "unsupported parallel mode");
    if (p.mode == ParallelMode::Single) {
        require(target == "qwen3.6-27b" || target == "qwen3.8-27b",
                "27B package placement cannot describe another target; use its registered package");
        require(p.devices.empty() && p.pipeline_layers.empty(),
                "Single placement uses only EngineOptions::device; rank devices and layer ranges "
                "must be empty");
        return {p.mode, {rank(0, options.device, {0, 64}, whole_heads())}};
    }
    require(p.devices.size() == 2,
            "multi-device placement requires exactly two devices (PP2 or TP2)");
    require(p.devices[0] >= 0 && p.devices[1] >= 0, "device ordinals must be nonnegative");
    require(p.devices[0] != p.devices[1],
            "duplicate devices are not allowed in multi-device placement");
    require(p.devices[0] == options.device,
            "placement rank zero must equal the EngineOptions::device coordinator");
    require(
        target == "qwen3.8-27b",
        "multi-device placement supports only qwen3.8-27b; MoE and other targets are unsupported");
    // TP2 executes the offline-sharded groupwise-int-tp2 artifact of the same weights.
    require(profile == "groupwise-int" ||
                (p.mode == ParallelMode::Tensor && profile == "groupwise-int-tp2"),
            "multi-device placement requires groupwise-int (TP2: groupwise-int-tp2); NVFP4 is "
            "unsupported");
    require(options.speculative.backend != SpeculativeBackend::DFlash &&
                options.speculative.backend != SpeculativeBackend::DFlash2,
            "DFlash and DFlash2 are unsupported with multi-device placement");
    std::vector<ParallelRank> ranks;
    ranks.reserve(2);
    if (p.mode == ParallelMode::Pipeline) {
        const auto& r = p.pipeline_layers;
        require(r.size() == 2 && r[0].begin == 0 && r[0].end > 0 && r[0].end < 64 &&
                    r[1].begin == r[0].end && r[1].end == 64,
                "Pipeline layer ranges must exactly cover [0,64) as [0,cut), [cut,64), with cut in "
                "1..63");
        for (std::uint32_t i = 0; i < 2; ++i)
            ranks.push_back(rank(i, p.devices[i], r[i], whole_heads()));
    } else {
        require(p.pipeline_layers.empty(),
                "Tensor placement must not specify pipeline layer ranges");
        for (std::uint32_t i = 0; i < 2; ++i) {
            ranks.push_back(rank(i, p.devices[i], {0, 64},
                                 HeadShard::create({i * 12, (i + 1) * 12}, {i * 2, (i + 1) * 2},
                                                   {i * 8, (i + 1) * 8}, {i * 24, (i + 1) * 24},
                                                   {i * 8704, (i + 1) * 8704})));
        }
    }
    return {p.mode, std::move(ranks)};
}
} // namespace ninfer::targets::qwen3_6_27b::detail
