#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/pipeline_weights.h"
#include "targets/qwen3_6_27b/impl/load/tensor_weights.h"
#include "targets/qwen3_6_27b/impl/parallel_plan.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_27b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
    std::optional<PipelineStage> stage;
    // TP2 rank load: plan.materialization holds only this rank's allocation.
    struct TensorRank {
        OwnerBindingPlan owner;
        TensorRankBindingPlan rank;
    };
    std::optional<TensorRank> tensor;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_27b::detail

namespace ninfer::targets::qwen3_6_27b {
namespace {

// General-task presets published with each exact model. Keep the registrations separate even
// while their values agree so an upstream model-specific change has one obvious owner.
constexpr ModelSamplingDefaults kQwen3_6Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6Defaults; }
    if (model == qwen3_8_model_id) { return kQwen3_8Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen36GroupwiseInt;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen38GroupwiseInt;
    }
    if (identity.model_id == model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Qwen36Nvfp4;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Qwen38Nvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options))));
}

std::array<Package::LoadPlan, 2> Package::plan_pipeline_load(
    artifact::Binder& binder, const EngineOptions& options, WeightsProfile weights_profile) {
    const auto cut = options.placement.pipeline_layers.at(0).end;
    auto full = detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options));
    auto materializations = detail::pipeline_materialization_plans(full, cut);
    auto first = std::make_unique<LoadPlan::Impl>(weights_profile,
        detail::ArtifactLoadPlan{full.bindings, std::move(materializations[0])});
    auto second = std::make_unique<LoadPlan::Impl>(weights_profile,
        detail::ArtifactLoadPlan{std::move(full.bindings), std::move(materializations[1])});
    first->stage = detail::PipelineStage{{0, cut}, true};
    second->stage = detail::PipelineStage{{cut, 64}, false};
    return {LoadPlan(std::move(first)), LoadPlan(std::move(second))};
}

std::array<Package::LoadPlan, 2> Package::plan_tensor_load(const artifact::Reader& reader,
                                                          const EngineOptions& options) {
    auto tensor = detail::bind_tensor_artifact(reader, qwen3_6::startup_features(options));
    std::array<std::unique_ptr<LoadPlan::Impl>, 2> ranks;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        ranks[rank] = std::make_unique<LoadPlan::Impl>(
            WeightsProfile::Qwen38GroupwiseInt,
            detail::ArtifactLoadPlan{{}, std::move(tensor.materialization[rank])});
        ranks[rank]->tensor.emplace(LoadPlan::Impl::TensorRank{tensor.owner, tensor.ranks[rank]});
    }
    return {LoadPlan(std::move(ranks[0])), LoadPlan(std::move(ranks[1]))};
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    if (plan.impl_->tensor) {
        auto impl = std::make_unique<LoadedModel::Impl>(plan.impl_->weights_profile,
            plan.impl_->tensor->owner, plan.impl_->tensor->rank, std::move(materialized));
        plan.impl_.reset();
        return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
    }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized), plan.impl_->stage);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                      .vision_max_tokens        = model.impl_->data.runtime.features.vision_max_tokens,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile,
                                                        std::optional<PhysicalLayerRange> pipeline_layers,
                                                        std::optional<std::uint32_t> tensor_rank) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile,
                                                           pipeline_layers, tensor_rank);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model,
                                                          SequencePlan&& plan,
                                                          DeviceContext& device,
                                                          const StartupObserver& startup_observer) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (model.impl_->data.pipeline_stage)
        throw std::invalid_argument("pipeline stage weights require the pipeline Program constructor");
    if (model.impl_->data.tensor_rank)
        throw std::invalid_argument("TP2 rank weights require the tensor Program constructor");
    return qwen3_6::create_program<detail::Variant>(model.impl_->data.runtime,
                                                    model.impl_->weights_profile, std::move(plan),
                                                    device, startup_observer);
}

std::unique_ptr<Package::Program> Package::create_pipeline_program(
    const LoadedModel& owner_model, const LoadedModel& tail_model,
    SequencePlan&& owner_plan, SequencePlan&& tail_plan,
    DeviceContext& owner_device, DeviceContext& tail_device,
    const StartupObserver& observer) {
    if (!owner_model.impl_ || !tail_model.impl_ ||
        !owner_model.impl_->data.pipeline_stage || !tail_model.impl_->data.pipeline_stage) {
        throw std::invalid_argument("pipeline loaded models require stage ownership");
    }
    return qwen3_6::create_pipeline_program<detail::Variant>(
        owner_model.impl_->data.runtime, tail_model.impl_->data.runtime,
        owner_model.impl_->weights_profile, std::move(owner_plan), std::move(tail_plan),
        owner_device, tail_device, observer);
}

std::unique_ptr<Package::Program> Package::create_tensor_program(
    const LoadedModel& owner_model, const LoadedModel& tail_model,
    SequencePlan&& owner_plan, SequencePlan&& tail_plan,
    DeviceContext& owner_device, DeviceContext& tail_device,
    const StartupObserver& observer) {
    if (!owner_model.impl_ || !tail_model.impl_ || owner_model.impl_->data.tensor_rank != 0U ||
        tail_model.impl_->data.tensor_rank != 1U) {
        throw std::invalid_argument("TP2 loaded models require rank-0 and rank-1 ownership");
    }
    return qwen3_6::create_tensor_program<detail::Variant>(
        owner_model.impl_->data.runtime, tail_model.impl_->data.runtime,
        owner_model.impl_->weights_profile, std::move(owner_plan), std::move(tail_plan),
        owner_device, tail_device, observer);
}

} // namespace ninfer::targets::qwen3_6_27b
