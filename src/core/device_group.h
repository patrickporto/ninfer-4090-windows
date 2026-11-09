#pragma once
#include "core/device.h"
#include <memory>
#include <vector>
namespace ninfer {
// Fixed, distinct physical contexts. Rank executors borrow these contexts and must
// stop before destruction. No binding or stream state is shared implicitly.
class DeviceGroup {
public:
    explicit DeviceGroup(std::vector<int> devices);
    DeviceGroup(const DeviceGroup&) = delete;
    DeviceGroup& operator=(const DeviceGroup&) = delete;
    std::size_t size() const noexcept { return contexts_.size(); }
    DeviceContext& context(std::size_t rank) { return *contexts_.at(rank); }
    const DeviceContext& context(std::size_t rank) const { return *contexts_.at(rank); }
    ScopedDevice bind(std::size_t rank) const { return ScopedDevice(context(rank).device); }
    void synchronize() const;
private:
    std::vector<std::unique_ptr<DeviceContext>> contexts_;
};
}
