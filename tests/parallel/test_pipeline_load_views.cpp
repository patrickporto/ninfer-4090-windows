#include "targets/qwen3_6_27b/impl/load/pipeline_weights.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::targets::qwen3_6_27b::detail;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class F>
void rejects(F f, const char* message) {
    try {
        f();
    } catch (const ArtifactError& error) {
        check(std::string(error.what()).find(message) != std::string::npos,
              "unexpected artifact rejection");
        return;
    }
    throw std::runtime_error("expected artifact rejection");
}

void validate_views(const LoadedModelData& data, const BindingPlan& plan, PhysicalLayerRange range,
                    bool owner, int device) {
    auto pointer = [&](const void* actual, ObjectHandle handle, bool present) {
        if (!present) {
            check(actual == nullptr, "unowned runtime slot must remain empty");
            return;
        }
        check(actual == data.backing.device_data(handle),
              "runtime view lost global handle mapping");
        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, actual));
        check(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
              "runtime pointer belongs to wrong device");
    };
    pointer(data.runtime.token_embedding.payload, plan.owner.token_embedding.object, owner);
    pointer(data.runtime.output_head.payload, plan.owner.output_head.object, owner);
    pointer(data.runtime.final_norm.data, plan.owner.final_norm, owner);
    const auto& frontend = data.frontend;
    for (const auto* resource :
         {&frontend.tokenizer_json, &frontend.tokenizer_config_json, &frontend.chat_template_jinja,
          &frontend.generation_config_json, &frontend.preprocessor_config_json,
          &frontend.video_preprocessor_config_json})
        check(resource->empty() != owner, "frontend ownership mismatch");
    check(!data.runtime.mtp && !data.runtime.vision && !data.runtime.dflash &&
              !data.runtime.optimized_proposal,
          "disabled feature unexpectedly materialized");
    std::size_t full = 0, gdn = 0, owned_full = 0, owned_gdn = 0;
    for (std::size_t physical = 0; physical < 64; ++physical) {
        const auto& source = plan.text_layers[physical];
        const bool present = physical >= range.begin && physical < range.end;
        if (physical % 4 == 3) {
            const auto& target = data.runtime.full_layers.at(full++);
            owned_full += present;
            pointer(target.input_norm.data, source.input_norm, present);
            pointer(target.post_attention_norm.data, source.post_attention_norm, present);
            pointer(target.post_mixer.gate_up.payload, source.mlp.gate_up.object, present);
            pointer(target.post_mixer.down.payload, source.mlp.down.object, present);
            pointer(target.query_norm.data, source.attention.query_norm, present);
            pointer(target.key_norm.data, source.attention.key_norm, present);
            pointer(target.output.payload, source.attention.output.object, present);
            std::visit(
                [&](const auto& payload) {
                    using T = std::decay_t<decltype(payload)>;
                    if constexpr (std::is_same_v<T, SplitAttentionProjectionPayload>) {
                        if (present) {
                            const auto& expected =
                                std::get<SplitAttentionProjectionPlan>(source.attention.projection);
                            pointer(payload.query_key.payload, expected.query_key.object, true);
                            pointer(payload.gate_value.payload, expected.gate_value.object, true);
                        } else {
                            check(!payload.query_key.payload && !payload.gate_value.payload,
                                  "foreign projection present");
                        }
                    } else {
                        const auto& expected =
                            std::get<FusedAttentionProjectionPlan>(source.attention.projection);
                        pointer(payload.query_key_gate_value.payload,
                                expected.query_key_gate_value.object, present);
                    }
                },
                target.projection);
        } else {
            const auto& target = data.runtime.gdn_layers.at(gdn++);
            owned_gdn += present;
            pointer(target.input_norm.data, source.input_norm, present);
            pointer(target.post_attention_norm.data, source.post_attention_norm, present);
            pointer(target.post_mixer.gate_up.payload, source.mlp.gate_up.object, present);
            pointer(target.post_mixer.down.payload, source.mlp.down.object, present);
            pointer(target.projection.a_log.data, source.gdn.a_log, present);
            pointer(target.projection.dt_bias.data, source.gdn.dt_bias, present);
            pointer(target.convolution.data, source.gdn.convolution, present);
            pointer(target.norm.data, source.gdn.norm, present);
            pointer(target.output.payload, source.gdn.output.object, present);
            std::visit(
                [&](const auto& payload) {
                    using T = std::decay_t<decltype(payload)>;
                    if constexpr (std::is_same_v<T, SplitGdnInputProjectionPayload>) {
                        if (present) {
                            const auto& expected =
                                std::get<SplitGdnInputProjectionPlan>(source.gdn.input_projection);
                            pointer(payload.query_key.payload, expected.query_key.object, true);
                            pointer(payload.value_z.payload, expected.value_z.object, true);
                        } else
                            check(!payload.query_key.payload && !payload.value_z.payload,
                                  "foreign GDN projection present");
                    } else {
                        const auto& expected =
                            std::get<FusedGdnInputProjectionPlan>(source.gdn.input_projection);
                        pointer(payload.query_key_value_z.payload,
                                expected.query_key_value_z.object, present);
                    }
                },
                target.projection.input_projection);
            std::visit(
                [&](const auto& payload) {
                    using T = std::decay_t<decltype(payload)>;
                    if constexpr (std::is_same_v<T, SplitGdnControlProjectionPayload>) {
                        if (present) {
                            const auto& expected = std::get<SplitGdnControlProjectionPlan>(
                                source.gdn.control_projection);
                            pointer(payload.a_projection.payload, expected.a_projection.object,
                                    true);
                            pointer(payload.b_projection.payload, expected.b_projection.object,
                                    true);
                        } else
                            check(!payload.a_projection.payload && !payload.b_projection.payload,
                                  "foreign GDN control present");
                    } else {
                        const auto& expected =
                            std::get<FusedGdnControlProjectionPlan>(source.gdn.control_projection);
                        pointer(payload.a_b_projection.payload, expected.a_b_projection.object,
                                present);
                    }
                },
                target.projection.control_projection);
        }
    }
    check(full == 16 && gdn == 48, "global runtime indices incomplete");
    if (range.begin == 36)
        check(owned_full == 7 && owned_gdn == 21, "cut36 stage counts incorrect");
}
} // namespace

int main() {
    const char* path = std::getenv("NINFER_PARALLEL_TEST_ARTIFACT");
    if (!path || !*path) {
        std::cout << "NINFER_PARALLEL_TEST_ARTIFACT unset; skipping\n";
        return 77;
    }
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        std::cout << "two CUDA devices required; skipping\n";
        return 77;
    }
    Reader reader(path);
    check(reader.identity().weights_id == "groupwise-int", "test requires groupwise-int artifact");
    const auto profile = reader.identity().model_id == "qwen3.8-27b"
                             ? WeightsProfile::Qwen38GroupwiseInt
                             : WeightsProfile::Qwen36GroupwiseInt;
    check(reader.identity().model_id == "qwen3.8-27b" ||
              reader.identity().model_id == "qwen3.6-27b",
          "test requires registered 27B artifact");
    Binder binder(reader);
    auto full   = bind_artifact(binder, profile, {});
    auto stages = pipeline_materialization_plans(full, 36);
    DeviceContext devices[]{DeviceContext(0), DeviceContext(1)};
    rejects([&] { LoadedModelData invalid(full.bindings, {}, PipelineStage{{36, 36}, false}); },
            "physical layer range");
    rejects([&] { LoadedModelData invalid(full.bindings, {}, PipelineStage{{0, 65}, true}); },
            "physical layer range");
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(1 - rank));
        auto backing = materialize(reader, stages[rank], devices[rank]);
        int current  = -1;
        CUDA_CHECK(cudaGetDevice(&current));
        check(current == 1 - rank, "materialization did not restore caller device");
        const PhysicalLayerRange range =
            rank == 0 ? PhysicalLayerRange{0, 36} : PhysicalLayerRange{36, 64};
        LoadedModelData loaded(full.bindings, std::move(backing), PipelineStage{range, rank == 0});
        check(loaded.pipeline_stage.has_value(), "stage metadata absent");
        validate_views(loaded, full.bindings, range, rank == 0, rank);
        check(loaded.backing.stats().device_capacity_bytes == stages[rank].device_capacity_bytes,
              "stage retained full model capacity");
        rejects(
            [&] {
                loaded.backing.device_data(
                    full.bindings.text_layers[rank == 0 ? 36 : 0].input_norm);
            },
            "materialized tensor");
    } // Both stage allocations are released before the full-model allocation.
    {
        std::vector<ObjectHandle> selected;
        const auto missing = full.bindings.text_layers[36].input_norm.index;
        for (auto p : stages[1].device_objects)
            if (p.object.index != missing) selected.push_back(p.object);
        const auto incomplete = filter_materialization_plan(stages[1], selected);
        auto backing          = materialize(reader, incomplete, devices[1]);
        rejects(
            [&] {
                LoadedModelData invalid(full.bindings, std::move(backing),
                                        PipelineStage{{36, 64}, false});
            },
            "materialized tensor");
    }
    {
        auto backing = materialize(reader, full.materialization, devices[0]);
        LoadedModelData loaded(full.bindings, std::move(backing));
        check(!loaded.pipeline_stage, "default constructor gained stage restriction");
        validate_views(loaded, full.bindings, {0, 64}, true, 0);
    }
    std::cout << "pipeline stage runtime views and default full views passed\n";
}
