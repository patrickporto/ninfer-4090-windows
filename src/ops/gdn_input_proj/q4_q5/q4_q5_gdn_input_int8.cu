// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "ops/common/int8_proj_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Both registered geometries store equal value/z halves and compact local outputs.

Int8ProjJob make_job(const Weight& weight, std::int32_t weight_row_begin, std::int32_t row_count,
                     Tensor& out, std::int32_t out_row_begin) {
    const std::int64_t groups = weight.padded_shape[1] / weight.group;
    const bool q5             = weight.qtype == QType::Q5G64_F16S;
    return Int8ProjJob{
        static_cast<const std::uint8_t*>(weight.qdata) +
            static_cast<std::int64_t>(weight_row_begin) * groups * 32,
        q5 ? static_cast<const std::uint8_t*>(weight.qhigh) +
                 static_cast<std::int64_t>(weight_row_begin) * groups * 8
           : nullptr,
        static_cast<const std::uint8_t*>(weight.scales) +
            static_cast<std::int64_t>(weight_row_begin) * groups * 2,
        static_cast<__nv_bfloat16*>(out.data) + out_row_begin,
        row_count,
        out.ne[0],
        q5,
    };
}

} // namespace

void q4_q5_gdn_input_int8_launch(const Tensor& x, const Weight& qk_weight,
                                 const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                 const Int8ProjWorkspace& scratch, cudaStream_t stream) {
    const std::int32_t value_rows = value_z_weight.n / 2;
    const Int8ProjJob jobs[3] = {
        make_job(qk_weight, 0, qk_weight.n, qkv, 0),
        make_job(value_z_weight, 0, value_rows, qkv, qk_weight.n),
        make_job(value_z_weight, value_rows, value_rows, z, 0),
    };
    int8_proj_launch(x, jobs, 3, scratch, stream);
}

} // namespace ninfer::ops::detail
