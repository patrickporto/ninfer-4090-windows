#pragma once
#include "targets/qwen3_6_27b/impl/load/bindings.h"

namespace ninfer::targets::qwen3_6_27b::detail {
// Full binding/validation precedes partitioning. Decoder handles follow physical layers;
// every selected non-decoder object (including host resources) belongs to stage zero.
std::array<artifact::MaterializationPlan, 2>
pipeline_materialization_plans(const ArtifactLoadPlan& full, std::size_t cut);
} // namespace ninfer::targets::qwen3_6_27b::detail
