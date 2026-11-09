#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {
inline constexpr std::int32_t kTensorProjectionTokenTile = 4096;
// A16 decode profile: columns accumulated per weight pass; larger T loops over chunks.
inline constexpr std::int32_t kTensorProjectionA16ColsPerPass = 8;
inline constexpr std::int32_t kTensorProjectionA16MaxCols     = 32;
void tensor_projection_gemv_launch(const Tensor& input, const Weight& weight, Tensor& output,
                                   cudaStream_t stream);
void tensor_projection_launch(const Tensor& input, const Weight& weight, Tensor& output,
                              std::int8_t* codes, float* scales, cudaStream_t stream);
void tensor_residual_add_launch(const Tensor& sum, Tensor& residual, cudaStream_t stream);
} // namespace ninfer::ops::detail
