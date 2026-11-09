#include "targets/qwen3_6_27b/impl/parallel_plan.h"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace ninfer;
using namespace ninfer::targets::qwen3_6_27b::detail;

void check(bool value) {
    if (!value) throw std::runtime_error("placement assertion failed");
}

template <class F>
void rejects(F f, const char* message) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        check(std::string(e.what()).find(message) != std::string::npos);
        return;
    }
    throw std::runtime_error(std::string("expected rejection: ") + message);
}

auto plan(const EngineOptions& o, std::string_view target = "qwen3.8-27b",
          std::string_view profile = "groupwise-int") {
    return ParallelPlan::create(o, target, profile);
}

int main() {
    EngineOptions o;
    o.device    = 3;
    auto single = plan(o);
    check(single.mode() == ParallelMode::Single && single.ranks().size() == 1);
    check(single.ranks()[0].device == 3);
    check(single.ranks()[0].full_attention_layers.size() == 16);
    check(single.ranks()[0].gdn_layers.size() == 48);
    // The package factory cannot emit 27B geometry for another target. Engine keeps those
    // single-device routes in their existing target packages.
    check(plan(o, "qwen3.6-27b", "nvfp4").ranks().size() == 1);
    rejects([&] { plan(o, "qwen3.6-35b-a3b"); }, "27B package");
    o.placement       = {ParallelMode::Pipeline, {3, 1}, {{0, 36}, {36, 64}}};
    auto pp           = plan(o);
    const auto& stage = pp.ranks()[1];
    check(stage.layers.begin == 36 && stage.layers.end == 64);
    check(stage.full_attention_layers.size() == 7 && stage.gdn_layers.size() == 21);
    check(stage.full_attention_layers[0].physical == 39 &&
          stage.full_attention_layers[0].global == 9);
    check(stage.gdn_layers[0].physical == 36 && stage.gdn_layers[0].global == 27);
    for (unsigned cut = 1; cut < 64; ++cut) {
        o.placement.pipeline_layers = {{0, cut}, {cut, 64}};
        auto p                      = plan(o);
        check(p.ranks()[0].gdn_layers.size() + p.ranks()[1].gdn_layers.size() == 48);
    }
    for (auto ranges : {std::vector<PhysicalLayerRange>{{0, 35}, {36, 64}},
                        {{0, 37}, {36, 64}},
                        {{0, 0}, {0, 64}},
                        {{0, 65}, {65, 64}},
                        {{1, 36}, {36, 64}},
                        {{0, 36}, {36, 63}},
                        {{0, 64}}}) {
        o.placement.pipeline_layers = ranges;
        rejects([&] { plan(o); }, "layer ranges");
    }
    o.placement = {ParallelMode::Tensor, {3, 1}, {}};
    auto tp     = plan(o);
    for (unsigned r = 0; r < 2; ++r) {
        const auto& h = tp.ranks()[r].heads;
        check(h.query.size() == 12 && h.kv.size() == 2 && h.gdn_qk.size() == 8 &&
              h.gdn_value.size() == 24);
        check(h.mlp.end - h.mlp.begin == 8704);
        for (unsigned q = 0; q < 12; ++q) {
            check(h.query[q].local.value == q && h.query[q].global.value == r * 12 + q);
            check(h.query[q].global.value / 6 == h.kv[q / 6].global.value);
        }
        for (unsigned v = 0; v < 24; ++v)
            check(h.gdn_value[v].global.value / 3 == h.gdn_qk[v / 3].global.value);
    }
    rejects([&] { (void)HeadShard::create({1, 13}, {0, 2}, {0, 8}, {0, 24}, {0, 8704}); }, "GQA");
    rejects([&] { (void)HeadShard::create({0, 12}, {0, 2}, {0, 8}, {1, 25}, {0, 8704}); }, "GDN");
    rejects([&] { (void)HeadShard::create({0, 12}, {0, 2}, {0, 8}, {0, 24}, {1, 8705}); }, "MLP");
    o.placement.devices = {3, 3};
    rejects([&] { plan(o); }, "duplicate");
    o.placement.devices = {3, 1, 2};
    rejects([&] { plan(o); }, "exactly two");
    o.placement.devices = {1, 3};
    rejects([&] { plan(o); }, "coordinator");
    o.placement.devices = {3, -1};
    rejects([&] { plan(o); }, "nonnegative");
    o.placement.devices = {3, 1};
    rejects([&] { plan(o, "qwen3.6-35b-a3b"); }, "qwen3.8-27b");
    rejects([&] { plan(o, "qwen3.8-27b", "nvfp4"); }, "groupwise-int");
    for (auto backend : {SpeculativeBackend::DFlash, SpeculativeBackend::DFlash2}) {
        o.speculative.backend = backend;
        rejects([&] { plan(o); }, "DFlash");
    }
    o.speculative.backend = SpeculativeBackend::Mtp;
    check(plan(o).ranks().size() == 2);
    o.placement.pipeline_layers = {{0, 64}};
    rejects([&] { plan(o); }, "Tensor");
    o.placement = {ParallelMode::Single, {3}, {}};
    rejects([&] { plan(o); }, "Single");
    o.placement = {static_cast<ParallelMode>(255), {}, {}};
    rejects([&] { plan(o); }, "mode");
    o.placement = {};
    o.device    = -1;
    rejects([&] { plan(o); }, "nonnegative");
    std::cout << "parallel placement contract passed\n";
}
