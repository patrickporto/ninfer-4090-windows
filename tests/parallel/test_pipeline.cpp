#include "targets/qwen3_6/impl/runtime/pipeline_schedule.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6::runtime_support;

void check(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template<class F> void rejects(F&& fn, const char* message) {
    try { fn(); } catch (const std::invalid_argument& e) {
        check(std::string(e.what()).find(message) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("expected invalid argument");
}

int main() {
    const auto full = StageLayerLayout::create({0, 64}, 64);
    check(full.full_attention_count() == 16 && full.gdn_count() == 48, "full geometry");
    const auto tail = StageLayerLayout::create({36, 64}, 64);
    check(tail.full_attention_count() == 7 && tail.gdn_count() == 21, "tail geometry");
    check(tail.local_full_attention_index(39) == 0, "first tail attention local index");
    check(tail.local_gdn_index(36) == 0, "first tail GDN local index");
    check(tail.local_full_attention_index(63) == 6, "last tail attention local index");
    check(tail.local_gdn_index(62) == 20, "last tail GDN local index");
    for (unsigned cut = 1; cut < 64; ++cut) {
        const auto a = StageLayerLayout::create({0, cut}, 64);
        const auto b = StageLayerLayout::create({cut, 64}, 64);
        check(a.full_attention_count() + b.full_attention_count() == 16, "attention conservation");
        check(a.gdn_count() + b.gdn_count() == 48, "GDN conservation");
        unsigned attention = 0, gdn = 0;
        for (unsigned layer = cut; layer < 64; ++layer) {
            if ((layer + 1) % 4 == 0) {
                check(b.local_full_attention_index(layer) == attention++, "unaligned attention mapping");
            } else {
                check(b.local_gdn_index(layer) == gdn++, "unaligned GDN mapping");
            }
        }
    }
    rejects([&] { (void)tail.local_gdn_index(35); }, "outside");
    rejects([&] { (void)tail.local_gdn_index(39); }, "GDN");
    rejects([&] { (void)tail.local_full_attention_index(36); }, "attention");
    rejects([] { (void)StageLayerLayout::create({36, 65}, 64); }, "range");
    rejects([] { (void)StageLayerLayout::create({36, 36}, 64); }, "range");
    rejects([] { StageLayerLayout::create({0, 3}, 64).require_pipeline_pools(); }, "attention");
    rejects([] { StageLayerLayout::create({63, 64}, 64).require_pipeline_pools(); }, "GDN");
    tail.require_pipeline_pools();

    // A device-1 OOM after allocating its StateImage must release that partial allocation and
    // device 0's KV/StateImage too. Different handles deliberately rule out shared slot indices.
    struct Handles { int state = -1; int kv = -1; };
    for (int fail_step : {0, 1, 2, 3, 4}) {
        std::array<int, 2> occupied{};
        std::vector<int> released;
        released.reserve(2);
        int step = 0;
        auto prepared = prepare_pipeline_active<Handles>([&](unsigned rank, Handles& h) {
            if (step++ == fail_step) { return false; }
            h.state = rank == 0 ? 17 : 41;
            ++occupied[rank];
            if (step++ == fail_step) { return false; }
            h.kv = rank == 0 ? 9 : 23;
            ++occupied[rank];
            return true;
        }, [&](unsigned rank, Handles& h) noexcept {
            if (h.kv != -1) { --occupied[rank]; h.kv = -1; }
            if (h.state != -1) { --occupied[rank]; h.state = -1; }
            released.push_back(rank);
        });
        if (fail_step < 4) {
            check(!prepared && occupied == std::array<int, 2>{0, 0}, "joint rollback leaked");
            check(released.back() == 0, "rollback order");
        } else {
            check(prepared && (*prepared)[0].state == 17 && (*prepared)[1].state == 41,
                  "rank-local handles lost");
            check(occupied == std::array<int, 2>{2, 2} && released.empty(), "early publication/release");
        }
    }
    std::array<int, 2> occupied{};
    try {
        (void)prepare_pipeline_active<Handles>([&](unsigned rank, Handles& h) {
            h.state = rank + 10;
            ++occupied[rank];
            if (rank == 1) { throw std::bad_alloc(); }
            return true;
        }, [&](unsigned rank, Handles& h) noexcept {
            if (h.state != -1) { --occupied[rank]; h.state = -1; }
        });
        throw std::runtime_error("exception swallowed");
    } catch (const std::bad_alloc&) {
        check(occupied == std::array<int, 2>{0, 0}, "exception rollback leaked");
    }
    check(compiled_groupwise_device(89, true, false), "SM89 compiled rejection");
    check(compiled_groupwise_device(120, false, true), "SM120 compiled rejection");
    check(!compiled_groupwise_device(120, true, false), "uncompiled SM120 accepted");
    check(!compiled_groupwise_device(89, false, true), "uncompiled SM89 accepted");
    check(!compiled_groupwise_device(90, true, true), "unknown architecture accepted");
    EngineOptions eager;
    eager.use_cuda_graph = false;
    eager.enable_vision = false;
    eager.context_cache.enabled = false;
    require_pipeline_eager_options(eager);
    eager.use_cuda_graph = true;
    rejects([&] { require_pipeline_eager_options(eager); }, "eager text");
    eager.use_cuda_graph = false;
    eager.enable_vision = true;
    rejects([&] { require_pipeline_eager_options(eager); }, "eager text");
    eager.enable_vision = false;
    eager.context_cache.enabled = true;
    rejects([&] { require_pipeline_eager_options(eager); }, "eager text");
    eager.context_cache.enabled = false;
    eager.speculative.backend = SpeculativeBackend::Mtp;
    rejects([&] { require_pipeline_eager_options(eager); }, "eager text");
    eager.speculative.backend = SpeculativeBackend::None;
    eager.turn_checkpoint_ring = 1;
    rejects([&] { require_pipeline_eager_options(eager); }, "checkpoint");
    eager.turn_checkpoint_ring = 0;
    eager.auto_save_evicted = true;
    rejects([&] { require_pipeline_eager_options(eager); }, "checkpoint");
    EngineOptions options;
    require_runtime_placement(options);
    options.placement.mode = ParallelMode::Pipeline;
    require_runtime_placement(options);
    options.placement.mode = ParallelMode::Tensor;
    rejects([&] { require_runtime_placement(options); }, "Tensor");
    options.placement.mode = ParallelMode::Single;
    options.placement.devices = {0};
    rejects([&] { require_runtime_placement(options); }, "Single");
    std::cout << "pipeline stage geometry and joint prepare passed\n";
}
