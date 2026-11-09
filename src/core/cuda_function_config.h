#pragma once

namespace ninfer {

// Opt in on the calling thread's current CUDA device. Successful requests retain the
// largest configured size for that device and kernel for the lifetime of the process.
// Primary contexts must not be reset while the engine is alive.
void configure_cuda_shared_memory(const void* kernel, int bytes);

template <class Return, class... Args>
void configure_cuda_shared_memory(Return (*kernel)(Args...), int bytes) {
    configure_cuda_shared_memory(reinterpret_cast<const void*>(kernel), bytes);
}

} // namespace ninfer
