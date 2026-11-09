#pragma once

#include "core/device.h"
#include "core/cuda_error_scope.h"
#include "core/host_worker_pool.h"
#include "core/tensor_collective.h"

#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <utility>

namespace ninfer::targets::qwen3_6::runtime_support {

// One bounded FIFO and host worker for one physical rank. The DeviceGroup and
// command captures outlive this executor. Completion of a future means host
// enqueue has finished; GPU completion remains an explicit event dependency.
// A command failure quarantines the rank and rejects all queued/new work.
class RankExecutor final {
public:
    explicit RankExecutor(DeviceContext& context, std::size_t queue_capacity = 4);
    ~RankExecutor();
    RankExecutor(const RankExecutor&) = delete;
    RankExecutor& operator=(const RankExecutor&) = delete;

    template <class Function>
    [[nodiscard]] auto submit(std::uint64_t sequence, Function&& function)
        -> std::future<std::invoke_result_t<std::decay_t<Function>&, DeviceContext&>> {
        using Result = std::invoke_result_t<std::decay_t<Function>&, DeviceContext&>;
        auto task = std::make_shared<std::packaged_task<Result()>>(
            [state = state_, context = context_, fn = std::forward<Function>(function)]() mutable
                -> Result {
                rethrow_failure(state);
                try {
                    ScopedCudaErrors errors;
                    ScopedDevice binding(context->device);
                    return std::invoke(fn, *context);
                } catch (...) {
                    remember_failure(state, std::current_exception());
                    TensorCollective::fail_current_thread();
                    throw;
                }
            });
        auto result = task->get_future();
        enqueue(sequence, [task = std::move(task)] { (*task)(); });
        return result;
    }

    [[nodiscard]] bool failed() const;
    [[nodiscard]] std::exception_ptr failure() const;

private:
    struct State;
    static void rethrow_failure(const std::shared_ptr<State>& state);
    static void remember_failure(const std::shared_ptr<State>& state, std::exception_ptr error);
    void enqueue(std::uint64_t sequence, std::function<void()> task);

    DeviceContext* context_;
    std::shared_ptr<State> state_;
    std::unique_ptr<HostWorkerPool> worker_;
};

} // namespace ninfer::targets::qwen3_6::runtime_support
