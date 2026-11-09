// TP2 rank-local Q5 row-parallel projection, A16 decode profile: FP32 output [5120, T] for small T.
// Same weight streaming and per-lane FP32 accumulation order as the registered Q5 row-split GEMV
// (q5_rowsplit_gemv.cuh), generalized to K = 3072 | 8704 (8704 = 8.5 tiles of 16 groups: the last
// tile carries 8 groups) and to C <= 8 columns per weight pass, so a decode batch streams the
// weights once. Partials are stored in FP32 (never BF16) for the subsequent all-reduce.
#include "ops/tensor_parallel/tensor_projection_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows          = 5120;
constexpr int kRowsPerBlock  = 16;
constexpr int kStages        = 3;
constexpr int kGroupK        = 64;
constexpr int kGroupsPerTile = 16;

template <int NG>
__device__ __forceinline__ void issue_tile(uint4* __restrict__ s_nib, uint4* __restrict__ s_hi,
                                           uint4* __restrict__ s_sc,
                                           const std::uint8_t* __restrict__ code_row,
                                           const std::uint8_t* __restrict__ high_row,
                                           const std::uint8_t* __restrict__ scale_row, int g0,
                                           int lane) {
    if (lane < NG * 2)
        pipe_copy<16>(&s_nib[lane], reinterpret_cast<const uint4*>(code_row + g0 * 32) + lane);
    if (lane < NG / 2)
        pipe_copy<16>(&s_hi[lane], reinterpret_cast<const uint4*>(high_row + g0 * 8) + lane);
    if (lane < NG / 8)
        pipe_copy<16>(&s_sc[lane], reinterpret_cast<const uint4*>(scale_row + g0 * 2) + lane);
    pipe_commit();
}

template <int NG, int C>
__device__ __forceinline__ void consume_tile(const __nv_bfloat16* __restrict__ x, int k_rows,
                                             int cols, const uint4* __restrict__ s_nib,
                                             const uint4* __restrict__ s_hi,
                                             const uint4* __restrict__ s_sc, int g0, int lane,
                                             float (&acc)[C]) {
    const auto* tn  = reinterpret_cast<const std::uint8_t*>(s_nib);
    const auto* th  = reinterpret_cast<const std::uint8_t*>(s_hi);
    const auto* tsc = reinterpret_cast<const std::uint16_t*>(s_sc);
#pragma unroll
    for (int tg = 0; tg < NG; ++tg) {
        const float scale       = __half2float(__ushort_as_half(tsc[tg]));
        const std::uint8_t low  = tn[tg * 32 + lane];
        const std::uint8_t high = th[tg * 8 + (lane >> 2)] >> ((lane & 3) * 2);
        const int q0 = sign_extend<5>(static_cast<int>((low & 0x0fu) | ((high & 0x01u) << 4)));
        const int q1 = sign_extend<5>(static_cast<int>((low >> 4) | ((high & 0x02u) << 3)));
        const float w0 = static_cast<float>(q0) * scale;
        const float w1 = static_cast<float>(q1) * scale;
        const int k0   = (g0 + tg) * kGroupK + lane * 2;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < cols) {
                const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                    x + static_cast<std::int64_t>(c) * k_rows + k0));
                acc[c] = fmaf(w0, xv.x, acc[c]);
                acc[c] = fmaf(w1, xv.y, acc[c]);
            }
        }
    }
}

template <int K, int C>
__global__ void __launch_bounds__(kRowsPerBlock * 32)
tensor_projection_gemv_kernel(const __nv_bfloat16* __restrict__ x, int cols,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high_bits,
                              const std::uint8_t* __restrict__ scales, float* __restrict__ y) {
    constexpr int kGroups    = K / kGroupK;
    constexpr int kFullTiles = kGroups / kGroupsPerTile;
    constexpr int kTail      = kGroups % kGroupsPerTile;
    constexpr int kTiles     = kFullTiles + (kTail ? 1 : 0);
    constexpr int kPrefetch  = kStages - 1;
    static_assert(kTail == 0 || kTail == 8, "tail tile must hold 0 or 8 groups");
    __shared__ uint4 s_nib[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_hi[kRowsPerBlock][kStages][8];
    __shared__ uint4 s_sc[kRowsPerBlock][kStages][2];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kGroups * 32;
    const std::uint8_t* high_row  = high_bits + static_cast<std::int64_t>(row) * kGroups * 8;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;

    auto issue = [&](int tile) {
        const int buf = tile % kStages;
        if (tile < kFullTiles)
            issue_tile<kGroupsPerTile>(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row,
                                       high_row, scale_row, tile * kGroupsPerTile, lane);
        else if constexpr (kTail != 0)
            issue_tile<kTail>(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row,
                              high_row, scale_row, tile * kGroupsPerTile, lane);
    };
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) issue(p);
        else pipe_commit();
    }
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) issue(fetch);
        else pipe_commit();
        pipe_wait<kPrefetch>();
        __syncwarp();
        const int buf = tile % kStages;
        if (tile < kFullTiles)
            consume_tile<kGroupsPerTile, C>(x, K, cols, s_nib[warp][buf], s_hi[warp][buf],
                                            s_sc[warp][buf], tile * kGroupsPerTile, lane, acc);
        else if constexpr (kTail != 0)
            consume_tile<kTail, C>(x, K, cols, s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf],
                                   tile * kGroupsPerTile, lane, acc);
        __syncwarp();
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        const float sum = warp_reduce_sum(acc[c]);
        if (lane == 0 && c < cols) y[static_cast<std::int64_t>(c) * kRows + row] = sum;
    }
}

template <int K, int C>
void launch_chunk(const __nv_bfloat16* x, int cols, const Weight& w, float* y, cudaStream_t stream) {
    tensor_projection_gemv_kernel<K, C><<<kRows / kRowsPerBlock, kRowsPerBlock * 32, 0, stream>>>(
        x, cols, static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), y);
    CUDA_CHECK(cudaGetLastError());
}

template <int K>
void launch_k(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    auto* yp       = static_cast<float*>(y.data);
    for (int offset = 0; offset < x.ne[1]; offset += kTensorProjectionA16ColsPerPass) {
        const int cols = std::min(kTensorProjectionA16ColsPerPass, x.ne[1] - offset);
        const auto* xo = xp + static_cast<std::int64_t>(offset) * K;
        auto* yo       = yp + static_cast<std::int64_t>(offset) * kRows;
        if (cols == 1) launch_chunk<K, 1>(xo, cols, w, yo, stream);
        else if (cols == 2) launch_chunk<K, 2>(xo, cols, w, yo, stream);
        else if (cols <= 4) launch_chunk<K, 4>(xo, cols, w, yo, stream);
        else launch_chunk<K, 8>(xo, cols, w, yo, stream);
    }
}
} // namespace

void tensor_projection_gemv_launch(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    if (w.k == 3072) launch_k<3072>(x, w, y, stream);
    else if (w.k == 8704) launch_k<8704>(x, w, y, stream);
    else throw std::invalid_argument("tensor_projection A16: K must be 3072 or 8704");
}
} // namespace ninfer::ops::detail
