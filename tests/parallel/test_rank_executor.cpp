#include "core/device_group.h"
#include "targets/qwen3_6/impl/runtime/rank_executor.h"

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

using ninfer::targets::qwen3_6::runtime_support::RankExecutor;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void failure_during_backpressure(ninfer::DeviceContext& context) {
    using namespace std::chrono_literals;
    RankExecutor executor(context, 1);
    std::promise<void> active, release, producer_entered;
    auto gate = release.get_future().share();
    auto running = executor.submit(1, [&](ninfer::DeviceContext&) {
        active.set_value();
        gate.wait();
        throw std::runtime_error("injected backpressure failure");
    });
    active.get_future().get();
    std::atomic<bool> queued_ran = false;
    auto queued = executor.submit(2, [&](ninfer::DeviceContext&) { queued_ran = true; });
    auto producer = std::async(std::launch::async, [&] {
        producer_entered.set_value();
        auto rejected = executor.submit(3, [&](ninfer::DeviceContext&) { queued_ran = true; });
        rejected.get();
    });
    producer_entered.get_future().get();
    require(producer.wait_for(20ms) == std::future_status::timeout, "queue did not backpressure");
    release.set_value();
    require(producer.wait_for(2s) == std::future_status::ready, "failed blocked producer hung");
    bool active_failed = false, queued_failed = false, producer_failed = false;
    try { running.get(); } catch (const std::runtime_error&) { active_failed = true; }
    try { queued.get(); } catch (const std::runtime_error&) { queued_failed = true; }
    try { producer.get(); } catch (const std::runtime_error&) { producer_failed = true; }
    require(active_failed && queued_failed && producer_failed && !queued_ran,
            "backpressured failure published more work");
}

void destructor_drains_fifo(ninfer::DeviceContext& context) {
    using namespace std::chrono_literals;
    auto executor = std::make_unique<RankExecutor>(context, 3);
    std::promise<void> active, release, destroying;
    auto gate = release.get_future().share();
    std::vector<int> order;
    auto first = executor->submit(1, [&](ninfer::DeviceContext&) {
        active.set_value(); gate.wait(); order.push_back(1);
    });
    active.get_future().get();
    auto second = executor->submit(2, [&](ninfer::DeviceContext&) { order.push_back(2); });
    auto third = executor->submit(3, [&](ninfer::DeviceContext&) { order.push_back(3); });
    auto destroyed = std::async(std::launch::async,
        [executor = std::move(executor), &destroying]() mutable {
            destroying.set_value(); executor.reset();
        });
    destroying.get_future().get();
    require(destroyed.wait_for(20ms) == std::future_status::timeout,
            "destructor released active captures");
    release.set_value();
    require(destroyed.wait_for(2s) == std::future_status::ready, "destructor failed to drain");
    destroyed.get(); first.get(); second.get(); third.get();
    require(order == std::vector<int>({1, 2, 3}), "healthy queued work lost FIFO order");
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    try {
        ninfer::DeviceGroup group({0, 1});
        RankExecutor first(group.context(0), 2);
        RankExecutor second(group.context(1), 2);
        auto a = first.submit(1, [](ninfer::DeviceContext& context) {
            int current = -1;
            CUDA_CHECK(cudaGetDevice(&current));
            if (current != context.device) throw std::runtime_error("rank binding");
            return current;
        });
        auto b = second.submit(1, [](ninfer::DeviceContext& context) {
            int current = -1;
            CUDA_CHECK(cudaGetDevice(&current));
            if (current != context.device) throw std::runtime_error("rank binding");
            return current;
        });
        require(a.get() == 0 && b.get() == 1, "rank tasks ran on wrong device");
        bool duplicate = false;
        try { (void)first.submit(1, [](ninfer::DeviceContext&) {}); }
        catch (const std::invalid_argument&) { duplicate = true; }
        require(duplicate, "duplicate command sequence accepted");

        std::promise<void> release;
        auto gate = release.get_future().share();
        std::atomic<bool> ran_after_failure = false;
        auto bad = first.submit(2, [gate](ninfer::DeviceContext&) {
            gate.wait();
            throw std::runtime_error("injected rank failure");
        });
        auto queued = first.submit(3, [&](ninfer::DeviceContext&) { ran_after_failure = true; });
        release.set_value();
        bool failed = false, cancelled = false;
        try { bad.get(); } catch (const std::runtime_error&) { failed = true; }
        try { queued.get(); } catch (const std::runtime_error&) { cancelled = true; }
        require(failed && cancelled && !ran_after_failure, "failure did not quarantine queued work");
        require(first.failed(), "failed rank reports healthy");
        bool rejected = false;
        try { (void)first.submit(4, [](ninfer::DeviceContext&) {}); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "failed rank accepts new work");
        require(second.submit(4, [](ninfer::DeviceContext&) { return 42; }).get() == 42,
                "unrelated worker was corrupted");
        failure_during_backpressure(group.context(0));
        destructor_drains_fifo(group.context(1));
        std::cout << "PASS rank executor ordering, binding and quarantine\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
