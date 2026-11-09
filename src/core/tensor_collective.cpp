#include "core/tensor_collective.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if NINFER_HAS_NCCL
#include <nccl.h>
#endif

namespace ninfer {
namespace {
using Clock = std::chrono::steady_clock;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
#if NINFER_HAS_NCCL
void nccl_require(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess && result != ncclInProgress)
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(result));
}
#endif
}

struct TensorCollective::Impl {
    struct Rank {
        std::thread::id owner;
        const DeviceContext* context = nullptr;
        cudaEvent_t done = nullptr;
#if NINFER_HAS_NCCL
        ncclComm_t comm = nullptr;
#endif
    };
    std::array<int, 2> devices;
    std::size_t capacity;
    std::chrono::milliseconds timeout;
    std::uint32_t sequences;
    std::array<Rank, 2> ranks;
    std::atomic<bool> unhealthy{false};
    std::mutex mutex;
    std::condition_variable changed;
    std::uint64_t generation = 0, last_unit = 0;
    std::uint32_t last_sequence = 0;
    std::size_t count = 0, initialized = 0, completed = 0, departed = 0;
    std::array<bool, 2> entered{};
    std::size_t dispatch_ready = 0;
    bool active = false, has_order = false;
    bool unit_mode = false;
    std::uint32_t unit_sequences = 0;
    std::array<std::uint32_t, 2> enqueued{};
#if NINFER_HAS_NCCL
    std::once_flag unique_once;
    ncclUniqueId unique{};
#endif
    Impl(std::array<int, 2> d, std::size_t c, std::chrono::milliseconds t, std::uint32_t s)
        : devices(d), capacity(c), timeout(t), sequences(s) {}

    void check(Clock::time_point deadline) {
        if (unhealthy.load()) throw std::runtime_error("tensor collective quarantined");
        if (Clock::now() >= deadline) throw std::runtime_error("tensor collective timeout; process may require restart");
    }
    void poison() noexcept { unhealthy.store(true); changed.notify_all(); }
    Rank& owner(std::size_t rank) {
        require(rank < 2, "tensor collective rank must be 0 or 1");
        auto& r = ranks[rank];
        require(r.context && r.owner == std::this_thread::get_id(),
                "tensor collective must run on its initialized rank owner");
        return r;
    }
#if NINFER_HAS_NCCL
    void poll(Rank& r, Clock::time_point deadline) {
        for (;;) {
            check(deadline);
            ncclResult_t state = ncclSuccess;
            nccl_require(ncclCommGetAsyncError(r.comm, &state), "ncclCommGetAsyncError");
            nccl_require(state, "NCCL asynchronous operation");
            if (state == ncclSuccess) return;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    void cleanup(Rank& r) {
        // Only the communicator owner reaches here, after its last NCCL call.
        // Abort itself is not guaranteed to have a bounded wall-clock duration.
        if (r.comm) {
            const auto result = ncclCommAbort(r.comm);
            r.comm = nullptr;
            nccl_require(result, "ncclCommAbort");
        }
        if (r.done) {
            const auto result = cudaEventDestroy(r.done);
            r.done = nullptr;
            cuda_require(result, "cudaEventDestroy collective completion");
        }
    }
    void cleanup_failure(Rank& r) noexcept {
        poison();
        // Rebind on the owning thread; preserve the original failure even when
        // the device is already lost. NCCL abort still gets its cleanup attempt.
        if (r.context) (void)cudaSetDevice(r.context->device);
        try { cleanup(r); } catch (...) { /* original failure remains authoritative */ }
    }
    struct OwnerCleanup {
        std::shared_ptr<Impl> state;
        std::size_t rank;
        ~OwnerCleanup() noexcept {
            auto& r = state->ranks[rank];
            if (r.comm || r.done) state->cleanup_failure(r);
        }
    };
    static auto& owner_guards() {
        static thread_local std::vector<std::unique_ptr<OwnerCleanup>> guards;
        return guards;
    }
    static void register_owner(const std::shared_ptr<Impl>& state, std::size_t rank) {
        auto guard = std::make_unique<OwnerCleanup>();
        guard->state = state;
        guard->rank = rank;
        owner_guards().push_back(std::move(guard));
    }
    void release_owner(std::size_t rank) {
        auto& guards = owner_guards();
        std::erase_if(guards, [&](const auto& guard) {
            return guard->state.get() == this && guard->rank == rank;
        });
    }
#endif
};

TensorCollective::TensorCollective(std::array<int, 2> devices, std::size_t max_count,
                                 std::chrono::milliseconds timeout, std::uint32_t sequences) {
    require(devices[0] >= 0 && devices[1] >= 0 && devices[0] != devices[1],
            "tensor collective requires two distinct physical devices");
    require(max_count > 0 && max_count <= (20 * 1024 * 1024) / sizeof(float),
            "invalid tensor collective capacity");
    require(timeout.count() > 0 && sequences > 0, "invalid tensor collective timeout or sequence bound");
    impl_ = std::make_shared<Impl>(devices, max_count, timeout, sequences);
}

TensorCollective::~TensorCollective() noexcept {
    // Owner guards retain physical state until explicit close or worker exit.
    // No NCCL calls or device-resource destruction occur on the coordinator.
    impl_->poison();
}

TensorCollective::Ticket TensorCollective::prepare(
    const std::array<TensorCollectiveDescriptor, 2>& descriptors) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->unhealthy.load()) throw std::runtime_error("tensor collective quarantined");
    require(!impl_->active, "previous tensor collective has not completed on both ranks");
    const auto& a = descriptors[0];
    const auto& b = descriptors[1];
    require(a.rank == 0 && b.rank == 1, "tensor collective descriptors must name ranks 0 and 1");
    require(a.unit == b.unit && a.sequence == b.sequence && a.count == b.count,
            "tensor collective descriptor mismatch before dispatch");
    require(a.count > 0 && a.count <= impl_->capacity, "tensor collective count exceeds fixed capacity");
    require(a.sequence < impl_->sequences, "tensor collective sequence exceeds fixed bound");
    if (!impl_->has_order) {
        require(a.sequence == 0, "first tensor collective sequence must be zero");
    } else if (impl_->last_sequence + 1 < impl_->sequences) {
        require(a.unit == impl_->last_unit && a.sequence == impl_->last_sequence + 1,
                "tensor collective unit or sequence diverged");
    } else {
        require(a.unit > impl_->last_unit && a.sequence == 0,
                "tensor collective next unit must increase and begin at sequence zero");
    }
    require(impl_->generation < std::numeric_limits<std::uint64_t>::max(), "tensor collective ticket exhausted");
    impl_->last_unit = a.unit;
    impl_->last_sequence = a.sequence;
    impl_->has_order = true;
    impl_->count = a.count;
    impl_->completed = impl_->departed = impl_->dispatch_ready = 0;
    impl_->entered = {};
    impl_->unit_mode = false;
    impl_->active = true;
    return Ticket{++impl_->generation};
}

TensorCollective::Ticket TensorCollective::prepare_unit(std::uint64_t unit, std::uint32_t sequences,
                                                        std::size_t count) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->unhealthy.load()) throw std::runtime_error("tensor collective quarantined");
    require(!impl_->active, "previous tensor collective has not completed on both ranks");
    require(sequences > 0 && sequences <= impl_->sequences, "tensor collective unit length exceeds bound");
    require(count > 0 && count <= impl_->capacity, "tensor collective count exceeds fixed capacity");
    require(!impl_->has_order || (impl_->last_sequence + 1 == impl_->sequences && unit > impl_->last_unit),
            "tensor collective next unit must increase after a complete unit");
    require(impl_->generation < std::numeric_limits<std::uint64_t>::max(), "tensor collective ticket exhausted");
    impl_->last_unit = unit;
    impl_->last_sequence = impl_->sequences - 1;
    impl_->has_order = true;
    impl_->count = count;
    impl_->completed = impl_->departed = impl_->dispatch_ready = 0;
    impl_->entered = {};
    impl_->enqueued = {};
    impl_->unit_sequences = sequences;
    impl_->unit_mode = true;
    impl_->active = true;
    return Ticket{++impl_->generation};
}

void TensorCollective::initialize_rank(std::size_t rank, const DeviceContext& context) {
    require(rank < 2, "tensor collective rank must be 0 or 1");
#if !NINFER_HAS_NCCL
    (void)context;
    impl_->poison();
    throw std::runtime_error("tensor collective backend requires NINFER_ENABLE_NCCL");
#else
    auto& r = impl_->ranks[rank];
    require(!r.context, "tensor collective rank already initialized");
    r.owner = std::this_thread::get_id();
    r.context = &context;
    const auto deadline = Clock::now() + impl_->timeout;
    try {
        Impl::register_owner(impl_, rank);
        impl_->check(deadline);
        require(context.device == impl_->devices[rank], "tensor collective context device mismatch");
        const char* blocking = std::getenv("NCCL_COMM_BLOCKING");
        require(!blocking || std::string(blocking) == "0", "NCCL_COMM_BLOCKING must be unset or 0");
        cuda_require(cudaSetDevice(context.device), "cudaSetDevice collective initialization");
        require(cuda_stream_device(context.stream) == context.device, "tensor collective stream device mismatch");
        cuda_require(cudaEventCreateWithFlags(&r.done, cudaEventDisableTiming), "cudaEventCreate collective completion");
        std::call_once(impl_->unique_once, [&] {
            nccl_require(ncclGetUniqueId(&impl_->unique), "ncclGetUniqueId");
        });
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 0;
        nccl_require(ncclCommInitRankConfig(&r.comm, 2, impl_->unique, static_cast<int>(rank), &config),
                     "ncclCommInitRankConfig");
        impl_->poll(r, deadline);
        std::unique_lock lock(impl_->mutex);
        impl_->check(deadline);
        ++impl_->initialized;
        impl_->changed.notify_all();
        while (impl_->initialized != 2) {
            impl_->check(deadline);
            impl_->changed.wait_for(lock, std::chrono::microseconds(100));
        }
    } catch (...) {
        impl_->cleanup_failure(r);
        throw;
    }
#endif
}

void TensorCollective::sum_rank(Ticket ticket, std::size_t rank, float* in_place) {
#if !NINFER_HAS_NCCL
    (void)ticket; (void)rank; (void)in_place;
    impl_->poison();
    throw std::runtime_error("tensor collective backend requires NINFER_ENABLE_NCCL");
#else
    auto& r = impl_->owner(rank);
    const auto deadline = Clock::now() + impl_->timeout;
    try {
        std::size_t count;
        {
            std::lock_guard lock(impl_->mutex);
            impl_->check(deadline);
            require(impl_->initialized == 2 && r.comm, "tensor collective ranks are not initialized");
            require(impl_->active && ticket.value == impl_->generation && !impl_->entered[rank],
                    "invalid or duplicate tensor collective ticket");
            require(in_place != nullptr, "tensor collective requires an in-place FP32 buffer");
            impl_->entered[rank] = true;
            count = impl_->count;
        }
        cuda_require(cudaSetDevice(r.context->device), "cudaSetDevice collective dispatch");
        cudaStreamCaptureStatus capture;
        cuda_require(cudaStreamIsCapturing(r.context->stream, &capture), "cudaStreamIsCapturing collective");
        require(capture == cudaStreamCaptureStatusNone, "NCCL graph capture is not supported");
        {
            // Both local validations finish before either rank posts GPU work.
            std::unique_lock lock(impl_->mutex);
            ++impl_->dispatch_ready;
            impl_->changed.notify_all();
            while (impl_->dispatch_ready != 2) {
                impl_->check(deadline);
                impl_->changed.wait_for(lock, std::chrono::microseconds(100));
            }
            impl_->check(deadline);
        }
        nccl_require(ncclAllReduce(in_place, in_place, count, ncclFloat32, ncclSum, r.comm,
                                  r.context->stream), "ncclAllReduce FP32 SUM");
        // ncclSuccess means posted, not GPU-complete. InProgress must first finish
        // posting before recording an event on the stream.
        impl_->poll(r, deadline);
        cuda_require(cudaEventRecord(r.done, r.context->stream), "cudaEventRecord collective completion");
        for (;;) {
            impl_->poll(r, deadline);
            const auto result = cudaEventQuery(r.done);
            if (result == cudaSuccess) break;
            if (result != cudaErrorNotReady) cuda_require(result, "cudaEventQuery collective completion");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        std::unique_lock lock(impl_->mutex);
        impl_->check(deadline);
        ++impl_->completed;
        impl_->changed.notify_all();
        while (impl_->completed != 2) {
            impl_->check(deadline);
            impl_->changed.wait_for(lock, std::chrono::microseconds(100));
        }
        if (++impl_->departed == 2) impl_->active = false;
    } catch (...) {
        impl_->cleanup_failure(r);
        throw;
    }
#endif
}

#if NINFER_HAS_NCCL
namespace {
// Posting is normally immediate; spin briefly before falling back to the sleeping poll.
template <class Impl, class Rank>
void finish_posting(Impl& impl, Rank& r, Clock::time_point deadline) {
    for (int spin = 0; spin < 256; ++spin) {
        ncclResult_t state = ncclSuccess;
        nccl_require(ncclCommGetAsyncError(r.comm, &state), "ncclCommGetAsyncError");
        nccl_require(state, "NCCL asynchronous operation");
        if (state == ncclSuccess) return;
        std::this_thread::yield();
    }
    impl.poll(r, deadline);
}
} // namespace
#endif

void TensorCollective::begin_unit(Ticket ticket, std::size_t rank) {
#if !NINFER_HAS_NCCL
    (void)ticket; (void)rank;
    impl_->poison();
    throw std::runtime_error("tensor collective backend requires NINFER_ENABLE_NCCL");
#else
    auto& r = impl_->owner(rank);
    const auto deadline = Clock::now() + impl_->timeout;
    try {
        {
            std::lock_guard lock(impl_->mutex);
            impl_->check(deadline);
            require(impl_->initialized == 2 && r.comm, "tensor collective ranks are not initialized");
            require(impl_->active && impl_->unit_mode && ticket.value == impl_->generation &&
                        !impl_->entered[rank],
                    "invalid or duplicate tensor collective unit ticket");
            impl_->entered[rank] = true;
        }
        cuda_require(cudaSetDevice(r.context->device), "cudaSetDevice collective dispatch");
        cudaStreamCaptureStatus capture;
        cuda_require(cudaStreamIsCapturing(r.context->stream, &capture), "cudaStreamIsCapturing collective");
        require(capture == cudaStreamCaptureStatusNone, "NCCL graph capture is not supported");
        std::unique_lock lock(impl_->mutex);
        ++impl_->dispatch_ready;
        impl_->changed.notify_all();
        while (impl_->dispatch_ready != 2) {
            impl_->check(deadline);
            impl_->changed.wait_for(lock, std::chrono::microseconds(20));
        }
        impl_->check(deadline);
    } catch (...) {
        impl_->cleanup_failure(r);
        throw;
    }
#endif
}

void TensorCollective::enqueue_sum(Ticket ticket, std::size_t rank, std::uint32_t sequence,
                                   float* in_place) {
#if !NINFER_HAS_NCCL
    (void)ticket; (void)rank; (void)sequence; (void)in_place;
    impl_->poison();
    throw std::runtime_error("tensor collective backend requires NINFER_ENABLE_NCCL");
#else
    auto& r = impl_->owner(rank);
    const auto deadline = Clock::now() + impl_->timeout;
    try {
        std::size_t count;
        {
            std::lock_guard lock(impl_->mutex);
            impl_->check(deadline);
            require(impl_->active && impl_->unit_mode && ticket.value == impl_->generation &&
                        impl_->entered[rank] && impl_->dispatch_ready == 2,
                    "tensor collective unit was not begun on both ranks");
            require(sequence == impl_->enqueued[rank] && sequence < impl_->unit_sequences,
                    "tensor collective unit sequence diverged");
            require(in_place != nullptr, "tensor collective requires an in-place FP32 buffer");
            ++impl_->enqueued[rank];
            count = impl_->count;
        }
        nccl_require(ncclAllReduce(in_place, in_place, count, ncclFloat32, ncclSum, r.comm,
                                  r.context->stream), "ncclAllReduce FP32 SUM");
        finish_posting(*impl_, r, deadline);
    } catch (...) {
        impl_->cleanup_failure(r);
        throw;
    }
#endif
}

void TensorCollective::finish_unit(Ticket ticket, std::size_t rank) {
#if !NINFER_HAS_NCCL
    (void)ticket; (void)rank;
    impl_->poison();
    throw std::runtime_error("tensor collective backend requires NINFER_ENABLE_NCCL");
#else
    auto& r = impl_->owner(rank);
    const auto deadline = Clock::now() + impl_->timeout;
    try {
        {
            std::lock_guard lock(impl_->mutex);
            impl_->check(deadline);
            require(impl_->active && impl_->unit_mode && ticket.value == impl_->generation &&
                        impl_->enqueued[rank] == impl_->unit_sequences,
                    "tensor collective unit finished before every reduction was posted");
        }
        cuda_require(cudaEventRecord(r.done, r.context->stream), "cudaEventRecord collective completion");
        for (;;) {
            impl_->check(deadline);
            ncclResult_t state = ncclSuccess;
            nccl_require(ncclCommGetAsyncError(r.comm, &state), "ncclCommGetAsyncError");
            nccl_require(state, "NCCL asynchronous operation");
            const auto result = cudaEventQuery(r.done);
            if (result == cudaSuccess) break;
            if (result != cudaErrorNotReady) cuda_require(result, "cudaEventQuery collective completion");
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        std::unique_lock lock(impl_->mutex);
        impl_->check(deadline);
        ++impl_->completed;
        impl_->changed.notify_all();
        while (impl_->completed != 2) {
            impl_->check(deadline);
            impl_->changed.wait_for(lock, std::chrono::microseconds(20));
        }
        if (++impl_->departed == 2) impl_->active = false;
    } catch (...) {
        impl_->cleanup_failure(r);
        throw;
    }
#endif
}

void TensorCollective::close_rank(std::size_t rank) {
    auto& r = impl_->owner(rank);
#if NINFER_HAS_NCCL
    cuda_require(cudaSetDevice(r.context->device), "cudaSetDevice collective close");
    try { impl_->cleanup(r); impl_->release_owner(rank); }
    catch (...) { impl_->poison(); throw; }
#else
    (void)r;
#endif
}
void TensorCollective::fail_current_thread() noexcept {
#if NINFER_HAS_NCCL
    auto& guards = Impl::owner_guards();
    // Publish every CPU failure before the first potentially unbounded abort.
    for (const auto& guard : guards) guard->state->poison();
    for (const auto& guard : guards)
        guard->state->cleanup_failure(guard->state->ranks[guard->rank]);
#endif
}
void TensorCollective::quarantine() noexcept { impl_->poison(); }
bool TensorCollective::failed() const noexcept { return impl_->unhealthy.load(); }
bool TensorCollective::available() noexcept {
#if NINFER_HAS_NCCL
    return true;
#else
    return false;
#endif
}
} // namespace ninfer
