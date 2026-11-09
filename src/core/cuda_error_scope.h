#pragma once

namespace ninfer {

// CUDA_CHECK keeps its fatal Single default. A parallel execution boundary opts
// into exceptions so its coordinator can quarantine and drain every rank. This
// host-thread policy is not inherited by newly created executor threads.
class ScopedCudaErrors final {
public:
    explicit ScopedCudaErrors(bool enable_throw = true) noexcept;
    ~ScopedCudaErrors() noexcept;
    ScopedCudaErrors(const ScopedCudaErrors&) = delete;
    ScopedCudaErrors& operator=(const ScopedCudaErrors&) = delete;
private:
    bool previous_;
};

} // namespace ninfer
