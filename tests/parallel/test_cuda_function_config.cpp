#include "core/cuda_function_config.h"
#include "core/cuda_error_scope.h"

#include <cuda_runtime_api.h>

#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
thread_local int current_device = 0;
std::mutex calls_mutex;
using Key = std::pair<int, const void*>;
std::map<Key, std::vector<int>> calls;
bool fail_next = false;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void first_kernel() {}

void second_kernel() {}

void retry_kernel() {}

void concurrent_kernel() {}

template <class Kernel>
std::vector<int> requests(int device, Kernel kernel) {
    std::lock_guard<std::mutex> lock(calls_mutex);
    return calls[{device, reinterpret_cast<const void*>(kernel)}];
}
} // namespace

extern "C" cudaError_t __wrap_cudaGetDevice(int* device) {
    *device = current_device;
    return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaFuncSetAttribute(const void* kernel, cudaFuncAttribute attribute,
                                                   int bytes) {
    require(attribute == cudaFuncAttributeMaxDynamicSharedMemorySize, "wrong attribute");
    std::lock_guard<std::mutex> lock(calls_mutex);
    calls[{current_device, kernel}].push_back(bytes);
    if (fail_next) {
        fail_next = false;
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

int main() {
    try {
        using ninfer::configure_cuda_shared_memory;
        current_device = 0;
        configure_cuda_shared_memory(first_kernel, 65536);
        configure_cuda_shared_memory(first_kernel, 65536);
        current_device = 1;
        configure_cuda_shared_memory(first_kernel, 65536);
        require(requests(0, first_kernel) == std::vector<int>{65536} &&
                    requests(1, first_kernel) == std::vector<int>{65536},
                "shared memory opt-in must be applied independently on each device");

        current_device = 0;
        configure_cuda_shared_memory(first_kernel, 49152);
        configure_cuda_shared_memory(first_kernel, 81920);
        configure_cuda_shared_memory(first_kernel, 65536);
        configure_cuda_shared_memory(first_kernel, 81920);
        require(requests(0, first_kernel) == std::vector<int>({65536, 81920}),
                "smaller requests must not reduce the applied maximum");
        configure_cuda_shared_memory(second_kernel, 65536);
        require(requests(0, second_kernel) == std::vector<int>{65536},
                "distinct kernels must receive independent opt-ins");

        fail_next   = true;
        bool failed = false;
        try {
            ninfer::ScopedCudaErrors errors;
            configure_cuda_shared_memory(retry_kernel, 65536);
        } catch (const std::runtime_error&) { failed = true; }
        require(failed, "attribute failure did not propagate");
        configure_cuda_shared_memory(retry_kernel, 65536);
        require(requests(0, retry_kernel) == std::vector<int>({65536, 65536}),
                "failed attribute requests must be retried");

        fail_next = true;
        failed    = false;
        try {
            ninfer::ScopedCudaErrors errors;
            configure_cuda_shared_memory(retry_kernel, 81920);
        } catch (const std::runtime_error&) { failed = true; }
        require(failed, "attribute increase failure did not propagate");
        configure_cuda_shared_memory(retry_kernel, 65536);
        configure_cuda_shared_memory(retry_kernel, 81920);
        require(requests(0, retry_kernel) == std::vector<int>({65536, 65536, 81920, 81920}),
                "failed increases must preserve the earlier maximum and retry the increase");

        std::promise<void> start;
        auto gate = start.get_future().share();
        std::vector<std::future<void>> ranks;
        for (int rank = 0; rank < 16; ++rank) {
            ranks.push_back(std::async(std::launch::async, [gate, rank] {
                current_device = rank % 2;
                gate.wait();
                for (int i = 0; i < 32; ++i)
                    ninfer::configure_cuda_shared_memory(concurrent_kernel, 65536);
            }));
        }
        start.set_value();
        for (auto& rank : ranks) rank.get();
        require(requests(0, concurrent_kernel) == std::vector<int>{65536} &&
                    requests(1, concurrent_kernel) == std::vector<int>{65536},
                "concurrent ranks must configure once per device and kernel");
        std::cout << "PASS per-device CUDA shared memory configuration and retry\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
