#include "core/device_group.h"
#include <stdexcept>
#include <unordered_set>
namespace ninfer {
DeviceGroup::DeviceGroup(std::vector<int> devices) {
    if (devices.empty()) throw std::invalid_argument("device group is empty");
    std::unordered_set<int> distinct;
    for (int device : devices) {
        if (device < 0 || !distinct.insert(device).second)
            throw std::invalid_argument("device group requires distinct nonnegative devices");
    }
    int previous = 0;
    cuda_require(cudaGetDevice(&previous), "device group current device");
    ScopedDevice restore(previous);
    contexts_.reserve(devices.size());
    for (int device : devices) contexts_.push_back(std::make_unique<DeviceContext>(device));
}
void DeviceGroup::synchronize() const {
    for (const auto& ctx : contexts_) {
        ScopedDevice binding(ctx->device);
        cuda_require(cudaStreamSynchronize(ctx->stream), "group compute synchronization");
        cuda_require(cudaStreamSynchronize(ctx->transfer_stream), "group transfer synchronization");
    }
}
}
