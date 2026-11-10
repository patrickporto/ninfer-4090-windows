#pragma once

#include "core/arena.h"

#include <cstddef>
#include <optional>

namespace ninfer {

// Backing for the context-cache demotion tier (Host StateImages and Host KV): pinned host
// memory, or the VRAM of a device other than the compute device. Every transfer into or out of
// the tier uses cudaMemcpyDefault, so UVA resolves both residencies; a device-resident tier is
// not CPU addressable, so CPU readers go through copy_tier_bytes().
class CacheTierBuffer {
public:
    // A negative device selects pinned host memory.
    CacheTierBuffer(std::size_t size_bytes, int device);

    CacheTierBuffer(const CacheTierBuffer&)            = delete;
    CacheTierBuffer& operator=(const CacheTierBuffer&) = delete;
    CacheTierBuffer(CacheTierBuffer&&) noexcept        = default;
    CacheTierBuffer& operator=(CacheTierBuffer&&)      = default;

    [[nodiscard]] void* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    [[nodiscard]] bool device_resident() const noexcept { return device_ >= 0; }

    [[nodiscard]] int device() const noexcept { return device_; }

private:
    int device_ = -1;
    std::optional<PinnedHostBuffer> host_;
    std::optional<DeviceBuffer> vram_;
};

// Synchronous byte copy between any pair of host or cache-tier pointers (UVA cudaMemcpyDefault).
void copy_tier_bytes(void* destination, const void* source, std::size_t bytes);

} // namespace ninfer
