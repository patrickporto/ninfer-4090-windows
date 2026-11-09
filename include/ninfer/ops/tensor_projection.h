#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: tensor_projection
 * Math: output[n,t] = sum_k decode(weight[n,k]) * input[k,t].
 * Shapes: contiguous BF16 input [K,T], contiguous FP32 output [5120,T];
 * Q5G64_F16S RowSplit weight [5120,K], K exactly 3072 or 8704; every positive T.
 * Numeric: represented BF16 inputs and signed Q5 codes with stored FP16 scales
 * define the naive FP64 oracle. Private symmetric group64 A8 quantization and
 * integer contraction are admitted, as in the registered Q5 LinearAdd A8 profile;
 * FP32 cross-group accumulation is stored directly, with no BF16 partial cast.
 * Qualification uses that profile's unchanged reduction criterion: relative L2
 * .04, gross absolute .02, gross relative-to-maximum-reference .05.
 * Effects: overwrites output; never reads residual or output, changes inputs,
 * or mutates persistent state. No collective or residual addition is performed.
 * All inputs/outputs/workspace are disjoint; pointers are 16-byte aligned.
 * Workspace: caller-owned transient capacity reported below; restored on return.
 * Finite represented inputs and scales are required, as for the parent profile.
 */
void tensor_projection(const Tensor& input, const Weight& weight, Tensor& output,
                       WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Policy form. AllowA8: the A8 profile above for every T (prefill). A16Only: the decode profile,
 * T in [1,32]: BF16 activations are consumed directly (no activation quantization), each output
 * row accumulated in FP32 in the registered Q5 row-split GEMV order, stored FP32; no workspace.
 * A16 qualification: relative L2 1e-5, gross absolute 1e-4, gross relative-to-maximum 1e-5.
 */
void tensor_projection(const Tensor& input, const Weight& weight, Tensor& output,
                       LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t tensor_projection_workspace_capacity_bytes(std::int32_t input_rows,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);

/**
 * Op: tensor_residual_add
 * Math: residual[i] = round_bf16_rne(fp32(residual[i]) +_fp32 sum[i]) for every element i.
 * Shapes: contiguous FP32 sum and BF16 residual with identical [5120,T] shape; every positive T.
 * Numeric: the declared TP2 BF16 residual boundary. `sum` is the complete FP32 all-reduced
 * row-parallel output (never a rank partial); one FP32 RNE addition and one BF16 RNE rounding
 * define the exact oracle, so qualification is bitwise.
 * Effects: updates residual in place; sum is read-only. The tensors must not overlap. No
 * workspace and no persistent state.
 */
void tensor_residual_add(const Tensor& sum, Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops
