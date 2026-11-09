#pragma once
#include "targets/qwen3_6/impl/runtime/pipeline_physical.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/tensor_execution.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/argmax.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct TensorPlacementTag {};

// Two-rank physical execution unit below the single Program coordinator. PP runs contiguous
// layer stages; TP2 (tensor != nullptr) runs every layer on both ranks with rank-local heads.
// Both modes share the rank-1 physical stage and the paired transaction/lifecycle hooks.
class PipelineExecution {
public:
    const PhysicalLayerRange owner_layers;
    PipelineBuffers owner_buffers;
    std::unique_ptr<PipelinePhysicalStage> tail;
    // PP: hidden owner->tail and tail->owner. TP2: owner embedding broadcast only (inbound null).
    std::unique_ptr<PeerTransfer> outbound, inbound;
    std::vector<PipelineSequenceBinding> sequences;
    bool failed = false;
    // Declared after `tail`: closes both NCCL ranks on their owners before the stage executor.
    std::unique_ptr<TensorParallelUnit> tensor;

    PipelineExecution(const LoadedModelData& tail_model, const SequencePlanImpl& owner_plan,
                      const SequencePlanImpl& tail_plan, DeviceContext& owner_device,
                      DeviceContext& tail_device)
        : owner_layers(owner_plan.pipeline_layers.value()), owner_buffers(owner_plan.prefill_chunk),
          sequences(owner_plan.max_concurrency) {
        {
            ScopedDevice bind(tail_device.device);
            tail = std::make_unique<PipelinePhysicalStage>(tail_model, tail_plan, tail_device);
        }
        outbound = std::make_unique<PeerTransfer>(owner_device, tail_device, owner_buffers.residual.bytes());
        inbound = std::make_unique<PeerTransfer>(tail_device, owner_device, owner_buffers.residual.bytes());
    }
    PipelineExecution(TensorPlacementTag, const LoadedModelData& owner_model,
                      const LoadedModelData& tail_model, const SequencePlanImpl& owner_plan,
                      const SequencePlanImpl& tail_plan, DeviceContext& owner_device,
                      DeviceContext& tail_device)
        : owner_layers{0, static_cast<std::uint32_t>(TextConfig::layers)},
          owner_buffers(owner_plan.prefill_chunk, true), sequences(owner_plan.max_concurrency) {
        if (owner_plan.tensor_rank != 0U || tail_plan.tensor_rank != 1U)
            throw std::invalid_argument("TP2 execution requires rank-0 and rank-1 local plans");
        {
            ScopedDevice bind(tail_device.device);
            tail = std::make_unique<PipelinePhysicalStage>(tail_model, tail_plan, tail_device);
        }
        outbound = std::make_unique<PeerTransfer>(owner_device, tail_device, owner_buffers.residual.bytes());
        const auto columns = std::max(owner_plan.prefill_chunk, kMaximumConcurrency);
        tensor = std::make_unique<TensorParallelUnit>(owner_model, tail_model, owner_device,
            tail_device, *tail->executor, tail->next_command, columns);
    }
    void drain(DeviceContext& owner) noexcept {
        drain_pipeline_device(tail->device);
        drain_pipeline_device(owner);
    }

    schedule::PrefillChunkResult prefill(schedule::PrefillContext& state,
        StateImageHandle remote_state, KVAddressSpaceHandle remote_address,
        std::int32_t owner_row_index, std::span<const TokenId> tokens, std::uint32_t count, bool finalize, std::optional<std::uint32_t> split = std::nullopt) {
        if (failed) throw std::runtime_error("pipeline execution unit is quarantined");
        if (tensor)
            return tensor_prefill(state, remote_state, remote_address, owner_row_index, tokens,
                                  count, finalize, split);
        const auto begin = state.text_kv_base;
        if (count == 0 || count > state.execution.prefill_chunk || begin > tokens.size() ||
            count > tokens.size() - begin) throw std::invalid_argument("pipeline prefill range");
        if (split && *split > begin && *split < begin + count) {
            count = *split - begin;
            finalize = false;
        }
        runtime::ExecutionTimingRecorder timing;
        auto& owner = state.execution;
        const auto columns = static_cast<std::int32_t>(count);
        Tensor residual = owner_buffers.residual.slice(1, 0, columns);
        Tensor ids = owner_buffers.ids.slice(0, 0, columns);
        Tensor positions = owner_buffers.positions.slice(0, 0, columns);
        Tensor row = owner_buffers.rows.slice(0, 0, 1);
        Tensor remote_residual = tail->buffers.residual.slice(1, 0, columns);
        Tensor remote_positions = tail->buffers.positions.slice(0, 0, columns);
        Tensor remote_row = tail->buffers.rows.slice(0, 0, 1);
        Tensor hidden = owner.prefill_hidden.slice(1, 0, columns);
        const auto remote_slot = tail->states->physical_slot(remote_state);
        const auto remote_row_index = tail->addresses->bound_row(remote_address);
        const ops::CausalAttentionExecutionEnvelope envelope{begin + count, begin + count};
        try {
            CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + begin, ids.bytes(),
                                       cudaMemcpyHostToDevice, owner.device.stream));
            ops::fill_i32_positions(positions, static_cast<std::int32_t>(begin), owner.device.stream);
            ops::set_i32_scalar(row, owner_row_index, owner.device.stream);
            schedule::TextContext card(owner.device, owner.model, owner.work, state.text_kv,
                owner.linear_attention, owner.io, owner.prefill_hidden, owner.prefill_chunk,
                begin, {}, &state.text_cache, nullptr, owner_layers);
            card.set_linear_state_slots(state.state_source_slot, state.state_destination_slot);
            card.pipeline_embedding(ids, residual);
            card.pipeline_prefill_stage(residual, positions, positions, row, envelope);
            outbound->enqueue(remote_residual.data, residual.data, residual.bytes(),
                               owner.device.stream, tail->device.stream);
            auto command = tail->executor->submit(tail->next_command++, [&, remote_slot,
                remote_row_index, envelope, begin](DeviceContext& device) {
                ops::fill_i32_positions(remote_positions, static_cast<std::int32_t>(begin), device.stream);
                ops::set_i32_scalar(remote_row, remote_row_index, device.stream);
                schedule::TextContext remote(device, tail->model, tail->work, {},
                    tail->state_images->linear(), tail->unused_round, tail->unused_hidden,
                    owner.prefill_chunk, begin, {}, &tail->decoder->text_kv, nullptr, tail->layers);
                remote.set_linear_state_slots(remote_slot, remote_slot);
                remote.pipeline_prefill_stage(remote_residual, remote_positions, remote_positions,
                                               remote_row, envelope);
            });
            command.get();
            inbound->enqueue(residual.data, remote_residual.data, residual.bytes(),
                              tail->device.stream, owner.device.stream);
            card.pipeline_final_norm(residual, hidden);
            if (finalize) {
                Tensor last = hidden.slice(1, columns - 1, 1);
                Tensor logits = owner.io.logits.slice(1, 0, 1);
                card.pipeline_head(last, logits);
                ops::set_i32_scalar(owner.io.pos, static_cast<std::int32_t>(begin + count), owner.device.stream);
                ops::set_i32_scalar(owner.io.rope_pos, static_cast<std::int32_t>(begin + count), owner.device.stream);
                if (state.sampling) {
                    ops::sample(logits, owner.io.token, TextConfig::token_domain, state.sampling,
                        owner.io.pos, ops::kSamplePurposePrefill, owner.work, owner.device.stream);
                } else {
                    ops::argmax(logits, owner.io.token, TextConfig::token_domain, owner.device.stream);
                }
            }
            timing.begin_wait();
            tail->device.synchronize();
            owner.device.synchronize();
            timing.end_wait();
            owner.work.reset();
            return {.processed_tokens = count, .finalized = finalize, .timing = timing.finish()};
        } catch (...) {
            failed = true;
            drain(owner.device);
            throw;
        }
    }

    void decode(schedule::OrdinaryBatchContext& state, std::span<const std::uint32_t> indices,
                ops::CausalAttentionExecutionEnvelope envelope) {
        if (failed) throw std::runtime_error("pipeline execution unit is quarantined");
        if (tensor) return tensor_decode(state, indices, envelope);
        auto& owner = state.execution;
        auto& frame = state.frame;
        const auto batch = static_cast<std::int32_t>(indices.size());
        std::array<std::int32_t, kMaximumConcurrency> remote_rows{}, remote_slots{};
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto& binding = sequences.at(indices[i]);
            remote_rows[i] = tail->addresses->bound_row(binding.address.value());
            remote_slots[i] = tail->states->physical_slot(binding.state.value());
        }
        Tensor residual = owner_buffers.residual.slice(1, 0, batch);
        Tensor remote_residual = tail->buffers.residual.slice(1, 0, batch);
        Tensor tokens = frame.tokens.slice(0, 0, batch);
        Tensor positions = frame.cache_positions.slice(0, 0, batch);
        Tensor rope = frame.rope_positions.slice(0, 0, batch);
        Tensor rows = frame.text_kv_table_rows.slice(0, 0, batch);
        Tensor sources = frame.state_source_slots.slice(0, 0, batch);
        Tensor destinations = frame.state_destination_slots.slice(0, 0, batch);
        Tensor hidden = frame.hidden.slice(1, 0, batch);
        Tensor logits = frame.logits.slice(1, 0, batch);
        Tensor sampled = frame.sampled_tokens.slice(0, 0, batch);
        try {
            CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice, owner.device.stream));
            schedule::TextContext card(owner.device, owner.model, owner.work, {},
                owner.linear_attention, owner.io, owner.prefill_hidden, owner.prefill_chunk,
                0, {}, &state.text_cache, nullptr, owner_layers);
            card.pipeline_embedding(tokens, residual);
            card.pipeline_decode_stage(residual, positions, rope, rows, sources, destinations, envelope);
            outbound->enqueue(remote_residual.data, residual.data, residual.bytes(),
                               owner.device.stream, tail->device.stream);
            tail->executor->submit(tail->next_command++, [&, batch, envelope](DeviceContext& device) {
                Tensor p = tail->buffers.positions.slice(0, 0, batch);
                Tensor r = tail->buffers.rope_positions.slice(0, 0, batch);
                Tensor kv = tail->buffers.rows.slice(0, 0, batch);
                Tensor slots = tail->buffers.sources.slice(0, 0, batch);
                const auto upload = [&](Tensor& target, const void* source) {
                    CUDA_CHECK(cudaMemcpyAsync(target.data, source, target.bytes(),
                                               cudaMemcpyHostToDevice, device.stream));
                };
                upload(p, state.host_ingress.cache_positions.data());
                upload(r, state.host_ingress.rope_positions.data());
                upload(kv, remote_rows.data());
                upload(slots, remote_slots.data());
                schedule::TextContext remote(device, tail->model, tail->work, {},
                    tail->state_images->linear(), tail->unused_round, tail->unused_hidden,
                    owner.prefill_chunk, 0, {}, &tail->decoder->text_kv, nullptr, tail->layers);
                remote.pipeline_decode_stage(remote_residual, p, r, kv, slots, slots, envelope);
            }).get();
            inbound->enqueue(residual.data, remote_residual.data, residual.bytes(),
                              tail->device.stream, owner.device.stream);
            card.pipeline_final_norm(residual, hidden);
            card.pipeline_head(hidden, logits);
            ops::scatter(hidden, destinations, state.continuation_hidden_store, owner.device.stream);
            ops::sample(logits, sampled, TextConfig::token_domain, frame.sampling, positions,
                         ops::kSamplePurposeDecode, owner.work, owner.device.stream);
            CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost, owner.device.stream));
            tail->device.synchronize();
            owner.device.synchronize();
        } catch (...) {
            failed = true;
            drain(owner.device);
            throw;
        }
    }

private:
    // Owner embedding on rank 0, then a BF16 broadcast into the rank-1 residual. Both ranks then
    // hold the identical replicated hidden that every sublayer boundary requires.
    void broadcast_embedding(DeviceContext& owner, const LoadedModelData& model, const Tensor& ids,
                             Tensor& residual, Tensor& remote_residual) {
        ops::embedding(ids, model.token_embedding, residual, owner.stream);
        outbound->enqueue(remote_residual.data, residual.data, residual.bytes(), owner.stream,
                          tail->device.stream);
    }

    TensorLayerCall tail_call(Tensor residual, Tensor partial, qwen3_6::TextPhase phase,
                              ops::CausalAttentionExecutionEnvelope envelope) {
        TensorLayerCall call;
        call.device   = &tail->device;
        call.work     = &tail->work;
        call.residual = residual;
        call.partial  = partial;
        call.kv       = &tail->decoder->text_kv;
        call.state    = &tail->state_images->linear();
        call.phase    = phase;
        call.envelope = envelope;
        return call;
    }

    schedule::PrefillChunkResult tensor_prefill(schedule::PrefillContext& state,
        StateImageHandle remote_state, KVAddressSpaceHandle remote_address,
        std::int32_t owner_row_index, std::span<const TokenId> tokens, std::uint32_t count,
        bool finalize, std::optional<std::uint32_t> split) {
        const auto begin = state.text_kv_base;
        if (count == 0 || count > state.execution.prefill_chunk || begin > tokens.size() ||
            count > tokens.size() - begin) throw std::invalid_argument("tensor prefill range");
        if (split && *split > begin && *split < begin + count) {
            count = *split - begin;
            finalize = false;
        }
        runtime::ExecutionTimingRecorder timing;
        auto& owner = state.execution;
        const auto columns = static_cast<std::int32_t>(count);
        Tensor residual = owner_buffers.residual.slice(1, 0, columns);
        Tensor partial = owner_buffers.partial.slice(1, 0, columns);
        Tensor ids = owner_buffers.ids.slice(0, 0, columns);
        Tensor positions = owner_buffers.positions.slice(0, 0, columns);
        Tensor row = owner_buffers.rows.slice(0, 0, 1);
        Tensor remote_residual = tail->buffers.residual.slice(1, 0, columns);
        Tensor remote_partial = tail->buffers.partial.slice(1, 0, columns);
        Tensor remote_positions = tail->buffers.positions.slice(0, 0, columns);
        Tensor remote_row = tail->buffers.rows.slice(0, 0, 1);
        Tensor hidden = owner.prefill_hidden.slice(1, 0, columns);
        const auto remote_slot = tail->states->physical_slot(remote_state);
        const auto remote_row_index = tail->addresses->bound_row(remote_address);
        const ops::CausalAttentionExecutionEnvelope envelope{begin + count, begin + count};
        try {
            CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + begin, ids.bytes(),
                                       cudaMemcpyHostToDevice, owner.device.stream));
            ops::fill_i32_positions(positions, static_cast<std::int32_t>(begin), owner.device.stream);
            ops::set_i32_scalar(row, owner_row_index, owner.device.stream);
            broadcast_embedding(owner.device, owner.model, ids, residual, remote_residual);
            tail->executor->submit(tail->next_command++, [&, remote_row_index, begin](DeviceContext& device) {
                ops::fill_i32_positions(remote_positions, static_cast<std::int32_t>(begin), device.stream);
                ops::set_i32_scalar(remote_row, remote_row_index, device.stream);
            }).get();
            std::array<TensorLayerCall, 2> calls{};
            calls[0].device = &owner.device;
            calls[0].work = &owner.work;
            calls[0].residual = residual;
            calls[0].partial = partial;
            calls[0].kv = &state.text_cache;
            calls[0].state = &owner.linear_attention;
            calls[0].phase = qwen3_6::TextPhase::Prefill;
            calls[0].envelope = envelope;
            calls[1] = tail_call(remote_residual, remote_partial, qwen3_6::TextPhase::Prefill, envelope);
            calls[0].cache_positions = calls[0].rope_positions = &positions;
            calls[0].kv_rows = &row;
            calls[0].source_slot = state.state_source_slot;
            calls[0].destination_slot = state.state_destination_slot;
            calls[1].cache_positions = calls[1].rope_positions = &remote_positions;
            calls[1].kv_rows = &remote_row;
            calls[1].source_slot = calls[1].destination_slot = remote_slot;
            owner.work.reset();
            tensor->pass(calls);
            ops::rmsnorm(residual, owner.model.final_norm, TextConfig::rms_epsilon, true, hidden,
                         owner.device.stream);
            if (finalize) {
                Tensor last = hidden.slice(1, columns - 1, 1);
                Tensor logits = owner.io.logits.slice(1, 0, 1);
                ops::linear(last, owner.model.output_head, logits, owner.device.stream);
                ops::set_i32_scalar(owner.io.pos, static_cast<std::int32_t>(begin + count), owner.device.stream);
                ops::set_i32_scalar(owner.io.rope_pos, static_cast<std::int32_t>(begin + count), owner.device.stream);
                if (state.sampling) {
                    ops::sample(logits, owner.io.token, TextConfig::token_domain, state.sampling,
                        owner.io.pos, ops::kSamplePurposePrefill, owner.work, owner.device.stream);
                } else {
                    ops::argmax(logits, owner.io.token, TextConfig::token_domain, owner.device.stream);
                }
            }
            timing.begin_wait();
            tail->device.synchronize();
            owner.device.synchronize();
            timing.end_wait();
            owner.work.reset();
            tail->work.reset();
            return {.processed_tokens = count, .finalized = finalize, .timing = timing.finish()};
        } catch (...) {
            failed = true;
            drain(owner.device);
            throw;
        }
    }

    void tensor_decode(schedule::OrdinaryBatchContext& state, std::span<const std::uint32_t> indices,
                       ops::CausalAttentionExecutionEnvelope envelope) {
        auto& owner = state.execution;
        auto& frame = state.frame;
        const auto batch = static_cast<std::int32_t>(indices.size());
        if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency))
            throw std::invalid_argument("tensor decode batch size must be in [1,8]");
        std::array<std::int32_t, kMaximumConcurrency> remote_rows{}, remote_slots{};
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto& binding = sequences.at(indices[i]);
            remote_rows[i] = tail->addresses->bound_row(binding.address.value());
            remote_slots[i] = tail->states->physical_slot(binding.state.value());
        }
        Tensor residual = owner_buffers.residual.slice(1, 0, batch);
        Tensor partial = owner_buffers.partial.slice(1, 0, batch);
        Tensor remote_residual = tail->buffers.residual.slice(1, 0, batch);
        Tensor remote_partial = tail->buffers.partial.slice(1, 0, batch);
        Tensor tokens = frame.tokens.slice(0, 0, batch);
        Tensor positions = frame.cache_positions.slice(0, 0, batch);
        Tensor rope = frame.rope_positions.slice(0, 0, batch);
        Tensor rows = frame.text_kv_table_rows.slice(0, 0, batch);
        Tensor sources = frame.state_source_slots.slice(0, 0, batch);
        Tensor destinations = frame.state_destination_slots.slice(0, 0, batch);
        Tensor hidden = frame.hidden.slice(1, 0, batch);
        Tensor logits = frame.logits.slice(1, 0, batch);
        Tensor sampled = frame.sampled_tokens.slice(0, 0, batch);
        Tensor remote_positions = tail->buffers.positions.slice(0, 0, batch);
        Tensor remote_rope = tail->buffers.rope_positions.slice(0, 0, batch);
        Tensor remote_kv = tail->buffers.rows.slice(0, 0, batch);
        Tensor remote_sources = tail->buffers.sources.slice(0, 0, batch);
        try {
            CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice, owner.device.stream));
            broadcast_embedding(owner.device, owner.model, tokens, residual, remote_residual);
            tail->executor->submit(tail->next_command++, [&](DeviceContext& device) {
                const auto upload = [&](Tensor& target, const void* source) {
                    CUDA_CHECK(cudaMemcpyAsync(target.data, source, target.bytes(),
                                               cudaMemcpyHostToDevice, device.stream));
                };
                upload(remote_positions, state.host_ingress.cache_positions.data());
                upload(remote_rope, state.host_ingress.rope_positions.data());
                upload(remote_kv, remote_rows.data());
                upload(remote_sources, remote_slots.data());
            }).get();
            std::array<TensorLayerCall, 2> calls{};
            calls[0].device = &owner.device;
            calls[0].work = &owner.work;
            calls[0].residual = residual;
            calls[0].partial = partial;
            calls[0].kv = &state.text_cache;
            calls[0].state = &owner.linear_attention;
            calls[0].phase = qwen3_6::TextPhase::Verify;
            calls[0].envelope = envelope;
            calls[0].batch = batch;
            calls[0].cache_positions = &positions;
            calls[0].rope_positions = &rope;
            calls[0].kv_rows = &rows;
            calls[0].source_slots = &sources;
            calls[0].destination_slots = &destinations;
            calls[1] = tail_call(remote_residual, remote_partial, qwen3_6::TextPhase::Verify, envelope);
            calls[1].batch = batch;
            calls[1].cache_positions = &remote_positions;
            calls[1].rope_positions = &remote_rope;
            calls[1].kv_rows = &remote_kv;
            // Ordinary rank-1 decode updates each row's own state image in place.
            calls[1].source_slots = calls[1].destination_slots = &remote_sources;
            owner.work.reset();
            tensor->pass(calls);
            ops::rmsnorm(residual, owner.model.final_norm, TextConfig::rms_epsilon, true, hidden,
                         owner.device.stream);
            ops::linear(hidden, owner.model.output_head, logits, owner.device.stream);
            ops::scatter(hidden, destinations, state.continuation_hidden_store, owner.device.stream);
            ops::sample(logits, sampled, TextConfig::token_domain, frame.sampling, positions,
                         ops::kSamplePurposeDecode, owner.work, owner.device.stream);
            CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost, owner.device.stream));
            tail->device.synchronize();
            owner.device.synchronize();
            owner.work.reset();
            tail->work.reset();
        } catch (...) {
            failed = true;
            drain(owner.device);
            throw;
        }
    }
};
} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
