#include "artifact/binder.h"
#include "targets/qwen3_6_27b/impl/load/pipeline_weights.h"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace ninfer::artifact;

void check(bool b) {
    if (!b) throw std::runtime_error("materialization assertion failed");
}

template <class F>
void rejects(F f) {
    try {
        f();
    } catch (const ArtifactError&) { return; }
    throw std::runtime_error("expected rejection");
}

void typed_partition(bool split) {
    using namespace ninfer::targets::qwen3_6_27b::detail;
    ArtifactLoadPlan p;
    std::size_t next = 0;
    int physical     = -1;
    std::vector<int> physical_owner;
    auto handle = [&] {
        auto h = ObjectHandle{next++};
        physical_owner.push_back(physical);
        p.materialization.device_objects.push_back({h, (next - 1) * 256, 7, 256});
        return h;
    };
    p.bindings.owner.token_embedding = {handle(), NumericFormat::Q6G64_F16S};
    for (std::size_t l = 0; l < 64; ++l) {
        physical              = static_cast<int>(l);
        auto& b               = p.bindings.text_layers[l];
        b.is_full_attention   = l % 4 == 3;
        b.input_norm          = handle();
        b.post_attention_norm = handle();
        b.mlp.gate_up.object  = handle();
        b.mlp.down.object     = handle();
        b.mlp.gate_up.format  = NumericFormat::Q4G64_F16S;
        b.mlp.down.format     = NumericFormat::Q5G64_F16S;
        if (b.is_full_attention) {
            b.attention.query_norm    = handle();
            b.attention.key_norm      = handle();
            b.attention.output.object = handle();
            if (split)
                b.attention.projection =
                    SplitAttentionProjectionPlan{WeightPlan{handle(), NumericFormat::W8G32_F16S},
                                                 WeightPlan{handle(), NumericFormat::W8G32_F16S}};
            else
                b.attention.projection =
                    FusedAttentionProjectionPlan{WeightPlan{handle(), NumericFormat::W8G32_F16S}};
        } else {
            b.gdn.a_log         = handle();
            b.gdn.dt_bias       = handle();
            b.gdn.convolution   = handle();
            b.gdn.norm          = handle();
            b.gdn.output.object = handle();
            if (split) {
                b.gdn.input_projection =
                    SplitGdnInputProjectionPlan{WeightPlan{handle(), NumericFormat::W8G32_F16S},
                                                WeightPlan{handle(), NumericFormat::W8G32_F16S}};
                b.gdn.control_projection =
                    SplitGdnControlProjectionPlan{WeightPlan{handle(), NumericFormat::W8G32_F16S},
                                                  WeightPlan{handle(), NumericFormat::W8G32_F16S}};
            } else {
                b.gdn.input_projection =
                    FusedGdnInputProjectionPlan{WeightPlan{handle(), NumericFormat::W8G32_F16S}};
                b.gdn.control_projection =
                    FusedGdnControlProjectionPlan{WeightPlan{handle(), NumericFormat::W8G32_F16S}};
            }
        }
    }
    physical                                = -1;
    p.bindings.owner.final_norm                   = handle();
    p.bindings.owner.output_head                  = {handle(), NumericFormat::Q6G64_F16S};
    p.bindings.owner.draft_head                   = handle();
    p.bindings.owner.draft_head_token_ids         = handle();
    p.bindings.owner.mtp.input_projection         = handle();
    p.bindings.owner.vision_merger_fc2            = handle();
    p.bindings.owner.features.vision              = true;
    p.bindings.owner.features.speculative         = ninfer::SpeculativeBackend::Mtp;
    p.bindings.owner.features.proposal_head       = ninfer::ProposalHead::Optimized;
    p.materialization.device_capacity_bytes = (next - 1) * 256 + 7;
    p.materialization.host_objects.push_back({ObjectHandle{next++}});
    p.materialization.object_count = next;
    for (std::size_t cut = 1; cut < 64; ++cut) {
        auto stages = pipeline_materialization_plans(p, cut);
        std::vector<unsigned> counts(next);
        for (std::size_t rank = 0; rank < stages.size(); ++rank) {
            const auto& stage = stages[rank];
            check(stage.object_count == next);
            for (auto d : stage.device_objects) {
                ++counts[d.object.index];
                const int physical_layer        = physical_owner.at(d.object.index);
                const std::size_t expected_rank = physical_layer >= static_cast<int>(cut) ? 1 : 0;
                check(rank == expected_rank);
            }
            for (auto h : stage.host_objects) ++counts[h.object.index];
        }
        for (auto count : counts) check(count == 1);
        check(stages[0].host_objects.size() == 1 && stages[1].host_objects.empty());
        check(stages[0].device_objects.front().object.index == 0);
        const auto full = (64 / 4) - (cut / 4), gdn = (64 - cut) - full;
        check(stages[1].device_objects.size() == full * (split ? 9 : 8) + gdn * (split ? 13 : 11));
        if (cut == 36) check(full == 7 && gdn == 21);
    }
    rejects([&] { pipeline_materialization_plans(p, 0); });
    rejects([&] { pipeline_materialization_plans(p, 64); });
    auto bad                               = p;
    bad.bindings.text_layers[0].input_norm = {next};
    rejects([&] { pipeline_materialization_plans(bad, 36); });
    bad                                    = p;
    bad.bindings.text_layers[0].input_norm = bad.bindings.text_layers[1].input_norm;
    rejects([&] { pipeline_materialization_plans(bad, 36); });
    bad = p;
    bad.materialization.device_objects.erase(bad.materialization.device_objects.begin() + 1);
    rejects([&] { pipeline_materialization_plans(bad, 36); });
    bad                                           = p;
    bad.bindings.text_layers[0].is_full_attention = true;
    rejects([&] { pipeline_materialization_plans(bad, 36); });
    bad                                         = p;
    bad.bindings.text_layers[0].mlp.down.format = NumericFormat::NVFP4;
    rejects([&] { pipeline_materialization_plans(bad, 36); });
    bad = p;
    bad.materialization.host_objects.push_back({bad.bindings.text_layers[0].input_norm});
    rejects([&] { pipeline_materialization_plans(bad, 36); });
    for (auto backend : {ninfer::SpeculativeBackend::DFlash, ninfer::SpeculativeBackend::DFlash2}) {
        bad                               = p;
        bad.bindings.owner.features.speculative = backend;
        rejects([&] { pipeline_materialization_plans(bad, 36); });
    }
}

int main() {
    typed_partition(false);
    typed_partition(true);
    MaterializationPlan p{
        8, 529, {{{1}, 0, 3, 16}, {{3}, 256, 5, 256}, {{6}, 512, 17, 256}}, {{{7}}}};
    std::array<ObjectHandle, 3> selected{{{3}, {6}, {7}}};
    auto q = filter_materialization_plan(p, selected);
    check(q.object_count == 8 && q.device_capacity_bytes == 273 && q.device_objects.size() == 2);
    check(q.device_objects[0].object.index == 3 && q.device_objects[0].offset == 0);
    check(q.device_objects[1].object.index == 6 && q.device_objects[1].offset == 256);
    check(q.host_objects.size() == 1 && q.host_objects[0].object.index == 7);
    check(filter_materialization_plan(p, {}).device_capacity_bytes == 0);
    rejects([&] { filter_materialization_plan(p, std::array<ObjectHandle, 1>{{{2}}}); });
    rejects([&] { filter_materialization_plan(p, std::array<ObjectHandle, 2>{{{1}, {1}}}); });
    auto bad = p;
    bad.device_objects.push_back(bad.device_objects[0]);
    rejects([&] { filter_materialization_plan(bad, selected); });
    bad                             = p;
    bad.device_objects[0].alignment = 3;
    rejects([&] { filter_materialization_plan(bad, selected); });
    bad                          = p;
    bad.device_objects[0].offset = 528;
    rejects([&] { filter_materialization_plan(bad, selected); });
    bad                              = p;
    bad.host_objects[0].object.index = 8;
    rejects([&] { filter_materialization_plan(bad, selected); });
    bad                          = p;
    bad.device_objects[1].offset = 0;
    rejects([&] { filter_materialization_plan(bad, selected); });
    bad                         = p;
    bad.device_objects[0].bytes = std::numeric_limits<std::uint64_t>::max();
    rejects([&] { filter_materialization_plan(bad, selected); });
    // Retaining a high-alignment object must preserve alignment, not its former gap.
    MaterializationPlan high{3, 8193, {{{0}, 0, 1, 1}, {{1}, 8192, 1, 4096}}, {{{2}}}};
    auto dense = filter_materialization_plan(high, std::array<ObjectHandle, 2>{{{0}, {1}}});
    check(dense.device_objects[1].offset == 4096 && dense.device_capacity_bytes == 4097);
    std::cout << "pipeline materialization passed\n";
}
