#include "targets/qwen3_6/impl/runtime/rank_executor.h"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace ninfer::targets::qwen3_6::runtime_support {

struct RankExecutor::State {
    std::mutex submission_mutex;
    std::mutex failure_mutex;
    std::exception_ptr error;
    std::optional<std::uint64_t> last_sequence;
    std::thread::id worker_thread;
    bool stopping = false;
};

RankExecutor::RankExecutor(DeviceContext& context, std::size_t queue_capacity)
    : context_(&context), state_(std::make_shared<State>()),
      worker_(std::make_unique<HostWorkerPool>(1, queue_capacity)) {
    worker_->submit([context = context_, state = state_] {
        context->bind_to_current_thread();
        state->worker_thread = std::this_thread::get_id();
    }).get();
}

RankExecutor::~RankExecutor() {
    std::unique_ptr<HostWorkerPool> worker;
    {
        std::lock_guard lock(state_->submission_mutex);
        state_->stopping = true;
        worker = std::move(worker_);
    }
    // Drain host tasks before captures can be released. The coordinator drains
    // GPU work (and aborts failed collectives) before freeing rank allocations.
    worker.reset();
}

void RankExecutor::rethrow_failure(const std::shared_ptr<State>& state) {
    std::exception_ptr error;
    {
        std::lock_guard lock(state->failure_mutex);
        error = state->error;
    }
    if (error) std::rethrow_exception(error);
}

void RankExecutor::remember_failure(const std::shared_ptr<State>& state, std::exception_ptr error) {
    std::lock_guard lock(state->failure_mutex);
    if (!state->error) state->error = std::move(error);
}

void RankExecutor::enqueue(std::uint64_t sequence, std::function<void()> task) {
    if (std::this_thread::get_id() == state_->worker_thread)
        throw std::logic_error("rank executor cannot submit to itself");
    // Serialize producers across backpressure, independently of the failure
    // mutex, so a worker can quarantine a rank while a submitter waits for space.
    std::lock_guard lock(state_->submission_mutex);
    rethrow_failure(state_);
    if (state_->stopping) throw std::runtime_error("rank executor is stopping");
    if (state_->last_sequence && sequence <= *state_->last_sequence)
        throw std::invalid_argument("rank command sequence must increase");
    (void)worker_->submit(std::move(task), [state = state_] { rethrow_failure(state); });
    state_->last_sequence = sequence;
}

std::exception_ptr RankExecutor::failure() const {
    std::lock_guard lock(state_->failure_mutex);
    return state_->error;
}

bool RankExecutor::failed() const { return static_cast<bool>(failure()); }

} // namespace ninfer::targets::qwen3_6::runtime_support
