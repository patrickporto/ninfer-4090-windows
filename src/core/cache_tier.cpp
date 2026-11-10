#include "core/cache_tier.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer {

CacheTierBuffer::CacheTierBuffer(std::size_t size_bytes, int device) : device_(device < 0 ? -1 : device) {
    if (size_bytes == 0) { throw std::invalid_argument("CacheTierBuffer size must be nonzero"); }
    if (device_ < 0) {
        host_.emplace(size_bytes);
        return;
    }
    ScopedDevice bind(device_);
    vram_.emplace(size_bytes);
}

void* CacheTierBuffer::data() const noexcept { return host_ ? host_->data() : vram_ ? vram_->p : nullptr; }

std::size_t CacheTierBuffer::size() const noexcept {
    return host_ ? host_->size() : vram_ ? vram_->bytes : 0;
}

void copy_tier_bytes(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0) { return; }
    cuda_require(cudaMemcpy(destination, source, bytes, cudaMemcpyDefault), "copy_tier_bytes");
}

} // namespace ninfer
