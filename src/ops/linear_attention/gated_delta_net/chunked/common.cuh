#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::chunked {

// FP32 intermediates must be converted to the TF32 operand representation before
// MMA. Round to nearest, ties away from zero; raw FP32 register bits are not the
// conversion contract. BF16 values expanded to FP32 are already exact TF32 inputs.
__device__ __forceinline__ unsigned tf32_operand_bits(float value) {
    unsigned bits;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
    return bits;
}

__device__ __forceinline__ void mma_tf32_rounded(float& c0, float& c1, float& c2, float& c3,
                                                 float a0, float a1, float a2, float a3,
                                                 float b0, float b1) {
    ninfer::ops::mma_tf32_bits(c0, c1, c2, c3, tf32_operand_bits(a0), tf32_operand_bits(a1),
                              tf32_operand_bits(a2), tf32_operand_bits(a3),
                              tf32_operand_bits(b0), tf32_operand_bits(b1));
}

inline constexpr int BT    = kChunkSize;
inline constexpr int BC    = 16;
inline constexpr int MMA_M = 16;
inline constexpr int MMA_N = 8;
inline constexpr int MMA_K = 8;

static_assert(BT % BC == 0, "BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "BT must be a multiple of MMA_M");

} // namespace ninfer::ops::detail::gated_delta_net::chunked
