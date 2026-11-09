#pragma once

#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6_27b::detail {

// These are descriptor plans, not the single-device RuntimeModelView. A future TP
// Variant must consume their local dimensions explicitly.
struct TensorAttentionPlan {
    static constexpr int query_heads = 12, kv_heads = 2, head_dim = 256;
    WeightPlan query_key, gate_value, output;
    artifact::ObjectHandle query_norm, key_norm;
};

struct TensorGdnPlan {
    static constexpr int qk_heads = 8, value_heads = 24, head_dim = 128;
    WeightPlan query_key, value_z, a_projection, b_projection, output;
    artifact::ObjectHandle a_log, dt_bias, convolution, norm;
    // Logical channel-major convolution reuses convolution, with axes [1,0].
};

struct TensorMlpPlan {
    static constexpr int intermediate = 8704;
    WeightPlan gate_up, down;
};

struct TensorLayerBindingPlan {
    artifact::ObjectHandle input_norm, post_attention_norm;
    std::variant<TensorGdnPlan, TensorAttentionPlan> mixer;
    TensorMlpPlan mlp;
};

struct TensorRankBindingPlan {
    std::size_t rank = 0;
    std::array<TensorLayerBindingPlan, kTextLayers> layers;
};

enum class TensorObjectPlacement { Rank0Device, Rank1Device, OwnerHost, ValidateOnly };

struct TensorObjectAssignment {
    artifact::ObjectHandle object;
    TensorObjectPlacement placement;
};

struct TensorArtifactLoadPlan {
    OwnerBindingPlan owner;
    std::array<TensorRankBindingPlan, 2> ranks;
    // No union allocation is exposed to a materializer.
    std::array<artifact::MaterializationPlan, 2> materialization;
};

// Closed, ordered inventory recipe; useful for CPU-only contract fixtures.
std::string tensor_inventory_json(bool dflash2);
void validate_tensor_inventory(std::string_view json, bool dflash2);

// Every global handle, including validate-only objects, requires one assignment.
std::array<artifact::MaterializationPlan, 2>
tensor_materialization_plans(const artifact::MaterializationPlan& full,
                             std::span<const TensorObjectAssignment> assignments);

TensorArtifactLoadPlan bind_tensor_artifact(const artifact::Reader& reader,
                                            qwen3_6::StartupFeatures features);

} // namespace ninfer::targets::qwen3_6_27b::detail
