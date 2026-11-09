#include "ops/tensor_parallel/tensor_projection_launch.h"
#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// One FP32 RNE addition followed by one BF16 RNE rounding per element; exact by contract.
__global__ void tensor_residual_add_kernel(const float* sum, __nv_bfloat16* residual,
                                           std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        residual[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(residual[i]), sum[i]));
    }
}

} // namespace

void tensor_residual_add_launch(const Tensor& sum, Tensor& residual, cudaStream_t stream) {
    const std::int64_t n   = residual.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 4096;
    const int grid         = static_cast<int>(
        std::min<std::int64_t>(kMaxGrid, std::max<std::int64_t>(1, (n + kBlock - 1) / kBlock)));
    tensor_residual_add_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const float*>(sum.data), static_cast<__nv_bfloat16*>(residual.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
