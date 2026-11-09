// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "ops/common/int8_proj_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Both parents store the query/gate half first and the key/value half second: [7168, 5120]
// with halves 6144/1024, or the TP2 rank-local [3584, 5120] with halves 3072/512. Each half
// writes its own contiguous output, so the four jobs share the quantized activation only.

Int8ProjJob make_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                     Tensor& out) {
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const bool q5             = weight.qtype == QType::Q5G64_F16S;
    return Int8ProjJob{
        static_cast<const std::uint8_t*>(weight.qdata) +
            static_cast<std::int64_t>(row_begin) * groups * 32,
        q5 ? static_cast<const std::uint8_t*>(weight.qhigh) +
                 static_cast<std::int64_t>(row_begin) * groups * 8
           : nullptr,
        static_cast<const std::uint8_t*>(weight.scales) +
            static_cast<std::int64_t>(row_begin) * groups * 2,
        static_cast<__nv_bfloat16*>(out.data),
        row_count,
        out.ne[0],
        q5,
    };
}

} // namespace

void q4_q5_attn_input_int8_launch(const Tensor& x, const Weight& query_key_weight,
                                  const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, const Int8ProjWorkspace& scratch,
                                  cudaStream_t stream) {
    const std::int32_t query_rows = q.ne[0];
    const std::int32_t kv_rows    = k.ne[0];
    const Int8ProjJob jobs[4] = {
        make_job(query_key_weight, 0, query_rows, q),
        make_job(gate_value_weight, 0, query_rows, gate),
        make_job(query_key_weight, query_rows, kv_rows, k),
        make_job(gate_value_weight, query_rows, kv_rows, v),
    };
    int8_proj_launch(x, jobs, 4, scratch, stream);
}

} // namespace ninfer::ops::detail
