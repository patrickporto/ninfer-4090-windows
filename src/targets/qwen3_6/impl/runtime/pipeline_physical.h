#pragma once
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"
#include "targets/qwen3_6/impl/runtime/rank_executor.h"
#include "core/peer_transfer.h"
#include <algorithm>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

inline void drain_pipeline_device(DeviceContext& device) noexcept {
    ScopedDevice bind(device.device, std::nothrow);
    if (device.transfer_stream) (void)cudaStreamSynchronize(device.transfer_stream);
    if (device.stream) (void)cudaStreamSynchronize(device.stream);
}

// These capabilities never cross stores. A transaction owns them until both ranks are ready;
// the continuation generation, rather than the reusable control lane, owns them thereafter.
struct PipelineActivePreparation {
    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    std::optional<KVActivationReservation> activation;
};
struct PipelineSequenceBinding {
    std::uint64_t generation = 0;
    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
};

struct PipelineBuffers {
    DeviceArena storage;
    Tensor residual, ids, positions, rope_positions, rows, sources, destinations;
    Tensor partial; // TP2 only: FP32 row-parallel partial and in-place all-reduced sum.
    explicit PipelineBuffers(std::uint32_t chunk, bool tensor = false)
        : PipelineBuffers(PipelineBufferLayout::make(chunk, tensor)) {}
private:
    explicit PipelineBuffers(const PipelineBufferLayout& layout) : storage(layout.bytes) {
        const auto backing = storage.alloc_bytes(layout.bytes);
        residual = layout.residual.bind(backing);
        ids = layout.ids.bind(backing);
        positions = layout.positions.bind(backing);
        rope_positions = layout.rope_positions.bind(backing);
        rows = layout.rows.bind(backing);
        sources = layout.sources.bind(backing);
        destinations = layout.destinations.bind(backing);
        if (layout.partial.region.bytes != 0) partial = layout.partial.bind(backing);
    }
};

// Physical storage only: no requests, frontend, RNG, scheduler or output publication policy.
// Construction occurs with device bound; callers mutate the stores on the coordinator thread.
class PipelinePhysicalStage {
public:
    const LoadedModelData& model;
    DeviceContext& device;
    const PhysicalLayerRange layers;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::unique_ptr<qwen3_6::StateImageDevicePool> state_images;
    std::unique_ptr<LogicalKVPageStore> pages;
    std::unique_ptr<KVAddressSpaceStore> addresses;
    std::unique_ptr<StateImageStore> states;
    PipelineBuffers buffers;
    // TextContext stage leaves borrow this empty descriptor; no sampling buffers are allocated.
    qwen3_6::RoundState unused_round;
    Tensor unused_hidden;
    std::unique_ptr<runtime_support::RankExecutor> executor;
    std::uint64_t next_command = 1;

    PipelinePhysicalStage(const LoadedModelData& weights, const SequencePlanImpl& plan,
                          DeviceContext& context)
        : model(weights), device(context),
          layers(plan.pipeline_layers.value_or(PhysicalLayerRange{0, TextConfig::layers})),
          persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
          work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
          buffers(plan.prefill_chunk, plan.tensor_rank.has_value()) {
        // A PP tail owns [cut,64) at full geometry; a TP rank-1 stage owns every layer with
        // rank-local heads. Neither owns embedding, head, sampling or speculative state.
        const bool tail_stage = plan.tensor_rank ? *plan.tensor_rank == 1 && !plan.pipeline_layers
                                                 : plan.pipeline_layers && layers.begin != 0;
        if (!tail_stage || plan.context_cache.enabled || plan.use_cuda_graph ||
            plan.speculative_backend != SpeculativeBackend::None || plan.features.vision) {
            throw std::invalid_argument("pipeline tail requires a closed eager text stage");
        }
        const auto backing = persistent.alloc_bytes(plan.persistent.bytes);
        decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
        state_images = std::make_unique<qwen3_6::StateImageDevicePool>(backing, plan.persistent.state_images);
        pages = std::make_unique<LogicalKVPageStore>(decoder->text_kv.page_pool(),
                                                    decoder->text_kv.page_pool().capacity_pages());
        addresses = std::make_unique<KVAddressSpaceStore>(*pages,
            decoder->text_kv.execution_tables(), plan.max_concurrency + 1,
            decoder->text_kv.execution_tables().logical_page_capacity());
        states = std::make_unique<StateImageStore>(*state_images, nullptr, state_images->slot_count());
        executor = std::make_unique<runtime_support::RankExecutor>(device);
    }
    ~PipelinePhysicalStage() noexcept {
        executor.reset(); // Host enqueue completion precedes GPU drain and destruction of arenas.
        drain_pipeline_device(device);
    }
    bool prepare(PipelineActivePreparation& pending, std::uint32_t entitlement, std::int32_t row) {
        ScopedDevice bind(device.device);
        pending.state = states->reserve_destination();
        if (!pending.state) return false;
        states->activate_reset(*pending.state, device.stream);
        pending.address = addresses->create_inactive();
        if (!pending.address) return false;
        pending.activation.emplace(addresses->prepare_activation(*pending.address, entitlement, row, std::nullopt));
        return pending.activation.has_value();
    }
    void rollback(PipelineActivePreparation& pending) noexcept {
        drain_pipeline_device(device);
        pending.activation.reset();
        if (pending.address) {
            try {
                if (addresses->active(*pending.address)) addresses->deactivate(*pending.address);
                (void)addresses->release(*pending.address);
            } catch (...) {}
            pending.address.reset();
        }
        if (pending.state) { (void)states->release(*pending.state); pending.state.reset(); }
    }
    void release(PipelineSequenceBinding& binding) noexcept {
        drain_pipeline_device(device);
        if (binding.address) {
            try {
                if (addresses->active(*binding.address)) addresses->deactivate(*binding.address);
                (void)addresses->release(*binding.address);
            } catch (...) {}
            binding.address.reset();
        }
        if (binding.state) { (void)states->release(*binding.state); binding.state.reset(); }
        binding.generation = 0;
    }
};
} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
