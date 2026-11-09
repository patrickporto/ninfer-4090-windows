#include "ops/tensor_parallel/tensor_projection_launch.h"
#include "ops/common/act_quant_g64.h"
#include "ops/common/int8_rowsplit_gemm.cuh"
#include "core/cuda_function_config.h"
#include "core/device.h"
#include <algorithm>

namespace ninfer::ops::detail {
namespace {
using Schedule        = Int8RowSplitGemmSchedule<Int8GemmCodec::Q5, 64, 128, 16, 64, 3, 1>;
constexpr auto kStore = Int8GemmEpilogue::StoreFp32;

template <bool Full>
void launch(const Weight& w, Tensor& y, const std::int8_t* codes, const float* scales, int tokens,
            cudaStream_t stream) {
    configure_cuda_shared_memory(int8_rowsplit_gemm_kernel<Schedule, Full, kStore>,
                                 Schedule::kSharedBytes);
    const dim3 grid(5120 / Schedule::kBlockRows,
                    (tokens + Schedule::kBlockCols - 1) / Schedule::kBlockCols);
    int8_rowsplit_gemm_kernel<Schedule, Full, kStore>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            codes, scales, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<float*>(y.data), 5120, tokens, w.k, 5120);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void tensor_projection_launch(const Tensor& x, const Weight& w, Tensor& y, std::int8_t* codes,
                              float* scales, cudaStream_t stream) {
    for (int offset = 0; offset < x.ne[1];) {
        const int count    = std::min(kTensorProjectionTokenTile, x.ne[1] - offset);
        const Tensor input = x.slice(1, offset, count);
        Tensor output      = y.slice(1, offset, count);
        act_quant_g64_launch(static_cast<const __nv_bfloat16*>(input.data), codes, scales, w.k,
                             count, w.k, stream);
        if (count % Schedule::kBlockCols == 0)
            launch<true>(w, output, codes, scales, count, stream);
        else
            launch<false>(w, output, codes, scales, count, stream);
        offset += count;
    }
}
} // namespace ninfer::ops::detail
