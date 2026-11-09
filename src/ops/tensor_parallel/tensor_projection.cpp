#include "ninfer/ops/tensor_projection.h"
#include "ops/tensor_parallel/tensor_projection_launch.h"
#include "core/layout.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::ops {
namespace {
void require_shape(std::int32_t k, std::int32_t first, std::int32_t last) {
    if ((k != 3072 && k != 8704) || first <= 0 || last < first)
        throw std::invalid_argument("tensor_projection: unsupported shape or token interval");
}

bool aligned(const void* p) { return p && (reinterpret_cast<std::uintptr_t>(p) % 16 == 0); }

bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    if (an == 0 || bn == 0) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

void require_tensor(const Tensor& t, DType dtype, int rows, int cols) {
    if (t.dtype != dtype || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned(t.data))
        throw std::invalid_argument("tensor_projection: invalid tensor");
}

template <class Allocator>
auto allocate(Allocator& a, int k, int t) {
    const int tile      = std::min(t, detail::kTensorProjectionTokenTile);
    const Tensor codes  = a.alloc(DType::I8, {k, tile});
    const Tensor scales = a.alloc(DType::FP32, {k / 64, tile});
    return std::pair{static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}
} // namespace

std::size_t tensor_projection_workspace_capacity_bytes(int k, int first, int last) {
    require_shape(k, first, last);
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, k, last);
    return layout.peak_bytes(1);
}

void tensor_projection(const Tensor& x, const Weight& w, Tensor& y, WorkspaceArena& ws,
                       cudaStream_t stream) {
    tensor_projection(x, w, y, LinearPolicy::AllowA8, ws, stream);
}

void tensor_projection(const Tensor& x, const Weight& w, Tensor& y, LinearPolicy policy,
                       WorkspaceArena& ws, cudaStream_t stream) {
    if (policy != LinearPolicy::AllowA8 &&
        (policy != LinearPolicy::A16Only || x.ne[1] > detail::kTensorProjectionA16MaxCols))
        throw std::invalid_argument("tensor_projection: policy admits AllowA8, or A16Only for T<=32");
    require_shape(w.k, x.ne[1], x.ne[1]);
    require_tensor(x, DType::BF16, w.k, x.ne[1]);
    require_tensor(y, DType::FP32, 5120, x.ne[1]);
    if (w.n != 5120 || w.qtype != QType::Q5G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 64 || w.group_size != 64 || w.ndim != 2 ||
        w.shape[0] != w.n || w.shape[1] != w.k || w.shape[2] != 1 || w.shape[3] != 1 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.padded_shape[2] != 1 ||
        w.padded_shape[3] != 1 || !aligned(w.qdata) || !aligned(w.qhigh) || !aligned(w.scales))
        throw std::invalid_argument("tensor_projection: invalid Q5G64 RowSplit weight");
    const std::size_t groups                          = std::size_t(w.n) * (w.k / 64);
    const std::pair<const void*, std::size_t> reads[] = {
        {x.data, x.bytes()}, {w.qdata, groups * 32}, {w.qhigh, groups * 8}, {w.scales, groups * 2}};
    for (const auto& [p, size] : reads) {
        if (overlap(p, size, y.data, y.bytes()))
            throw std::invalid_argument("tensor_projection: overlapping storage");
    }
    if (policy == LinearPolicy::A16Only) {
        detail::tensor_projection_gemv_launch(x, w, y, stream);
        return;
    }
    // Inputs may live in the caller's workspace scope; only the staging this
    // call allocates must stay disjoint from what it reads and writes.
    auto scope                 = ws.scope();
    const auto [codes, scales] = allocate(ws, w.k, x.ne[1]);
    const std::size_t tile     = std::min(x.ne[1], detail::kTensorProjectionTokenTile);
    const std::pair<const void*, std::size_t> staging[] = {
        {codes, std::size_t(w.k) * tile}, {scales, std::size_t(w.k / 64) * tile * sizeof(float)}};
    for (const auto& [sp, ssize] : staging) {
        if (overlap(sp, ssize, y.data, y.bytes()))
            throw std::invalid_argument("tensor_projection: overlapping workspace/output");
        for (const auto& [p, size] : reads)
            if (overlap(p, size, sp, ssize))
                throw std::invalid_argument("tensor_projection: overlapping storage");
    }
    detail::tensor_projection_launch(x, w, y, codes, scales, stream);
}

void tensor_residual_add(const Tensor& sum, Tensor& residual, cudaStream_t stream) {
    const int columns = residual.ne[1];
    if (columns <= 0) throw std::invalid_argument("tensor_residual_add: T must be positive");
    if (sum.dtype != DType::FP32 || sum.ne[0] != 5120 || sum.ne[1] != columns || sum.ne[2] != 1 ||
        sum.ne[3] != 1 || !sum.is_contiguous() || !aligned(sum.data))
        throw std::invalid_argument("tensor_residual_add: invalid FP32 sum");
    if (residual.dtype != DType::BF16 || residual.ne[0] != 5120 || residual.ne[2] != 1 ||
        residual.ne[3] != 1 || !residual.is_contiguous() || !aligned(residual.data))
        throw std::invalid_argument("tensor_residual_add: invalid BF16 residual");
    if (overlap(sum.data, sum.bytes(), residual.data, residual.bytes()))
        throw std::invalid_argument("tensor_residual_add: overlapping storage");
    detail::tensor_residual_add_launch(sum, residual, stream);
}
} // namespace ninfer::ops
