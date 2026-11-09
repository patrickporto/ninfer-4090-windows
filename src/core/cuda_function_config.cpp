#include "core/cuda_function_config.h"

#include "core/device.h"

#include <map>
#include <mutex>

namespace ninfer {

void configure_cuda_shared_memory(const void* kernel, int bytes) {
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    static std::mutex mutex;
    static std::map<int, std::map<const void*, int>> configured;
    std::lock_guard<std::mutex> lock(mutex);
    auto& kernels    = configured[device];
    const auto found = kernels.find(kernel);
    if (found != kernels.end() && found->second >= bytes) return;

    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernels[kernel] = bytes;
}

} // namespace ninfer
