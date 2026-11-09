#include "targets/qwen3_6_27b/impl/load/pipeline_weights.h"
#include <type_traits>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
std::array<artifact::MaterializationPlan, 2>
pipeline_materialization_plans(const ArtifactLoadPlan& full, std::size_t cut) {
    if (cut == 0 || cut >= kTextLayers)
        throw artifact::ArtifactError("pipeline cut must lie inside the decoder");
    if (full.bindings.owner.features.masked_draft())
        throw artifact::ArtifactError("pipeline DFlash/DFlash2 is unsupported");
    std::vector<int> owner(full.materialization.object_count, -1);
    std::vector<bool> device_selected(owner.size());
    for (auto p : full.materialization.device_objects) {
        if (p.object.index >= owner.size())
            throw artifact::ArtifactError("device handle out of range");
        device_selected[p.object.index] = true;
    }
    for (std::size_t l = 0; l < kTextLayers; ++l) {
        const auto& b = full.bindings.text_layers[l];
        if (b.is_full_attention != (l % 4 == 3))
            throw artifact::ArtifactError("invalid physical decoder topology");
        auto add = [&](artifact::ObjectHandle h) {
            if (h.index >= owner.size() || !device_selected[h.index] || owner[h.index] != -1)
                throw artifact::ArtifactError("decoder handle absent or assigned more than once");
            owner[h.index] = l < cut ? 0 : 1;
        };
        auto weight = [&](const WeightPlan& w) {
            if (w.format == artifact::NumericFormat::NVFP4)
                throw artifact::ArtifactError("pipeline NVFP4 is unsupported");
            add(w.object);
        };
        add(b.input_norm);
        add(b.post_attention_norm);
        weight(b.mlp.gate_up);
        weight(b.mlp.down);
        if (b.is_full_attention) {
            const auto& a = b.attention;
            add(a.query_norm);
            add(a.key_norm);
            weight(a.output);
            std::visit(
                [&](const auto& p) {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, SplitAttentionProjectionPlan>) {
                        weight(p.query_key);
                        weight(p.gate_value);
                    } else
                        weight(p.query_key_gate_value);
                },
                a.projection);
        } else {
            const auto& g = b.gdn;
            add(g.a_log);
            add(g.dt_bias);
            add(g.convolution);
            add(g.norm);
            weight(g.output);
            std::visit(
                [&](const auto& p) {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, SplitGdnInputProjectionPlan>) {
                        weight(p.query_key);
                        weight(p.value_z);
                    } else
                        weight(p.query_key_value_z);
                },
                g.input_projection);
            std::visit(
                [&](const auto& p) {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, SplitGdnControlProjectionPlan>) {
                        weight(p.a_projection);
                        weight(p.b_projection);
                    } else
                        weight(p.a_b_projection);
                },
                g.control_projection);
        }
    }
    std::array<std::vector<artifact::ObjectHandle>, 2> selected;
    for (auto p : full.materialization.device_objects)
        selected[owner[p.object.index] == 1 ? 1 : 0].push_back(p.object);
    for (auto p : full.materialization.host_objects) selected[0].push_back(p.object);
    return {artifact::filter_materialization_plan(full.materialization, selected[0]),
            artifact::filter_materialization_plan(full.materialization, selected[1])};
}
} // namespace ninfer::targets::qwen3_6_27b::detail
