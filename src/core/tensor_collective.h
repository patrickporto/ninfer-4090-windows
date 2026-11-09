#pragma once

#include "core/device.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer {

struct TensorCollectiveDescriptor {
    std::size_t rank;
    std::uint64_t unit;
    std::uint32_t sequence;
    std::size_t count;
};

// Closed two-rank, in-place FP32 SUM. No threads, buffers, or DeviceContexts are
// owned here. prepare() is coordinator-only and checks both descriptors before
// dispatch. initialize_rank(), sum_rank(), and close_rank() run exclusively on
// the same fixed owner thread per rank. Submit BOTH rank callbacks before waiting
// for either. Each sum callback returns only after BOTH GPU endpoints complete.
//
// Construct before workers; normally close both ranks on their owners before destruction.
// A failed initialize/sum callback quarantines this object and aborts its local
// communicator before throwing; its peer does likewise. The owner executor calls
// fail_current_thread() from its common exception boundary for unrelated failures.
// Thread-exit RAII is the fallback; destruction only quarantines shared state. Never destroy
// contexts or buffers while callbacks remain active. Abort and CUDA driver calls
// cannot be given a hard in-process deadline: if those stall, restart the process.
class TensorCollective final {
public:
    struct Ticket {
        std::uint64_t value = 0;
    };
    TensorCollective(std::array<int, 2> devices, std::size_t max_count,
                     std::chrono::milliseconds timeout = std::chrono::seconds(30),
                     std::uint32_t sequences_per_unit = 128);
    ~TensorCollective() noexcept;
    TensorCollective(const TensorCollective&) = delete;
    TensorCollective& operator=(const TensorCollective&) = delete;

    // CPU-only, including when NCCL is disabled. First sequence is zero; units
    // strictly increase only after the preceding complete unit. Rejected
    // descriptors do not consume an order slot. Only one ticket may be active.
    Ticket prepare(const std::array<TensorCollectiveDescriptor, 2>& descriptors);
    void initialize_rank(std::size_t rank, const DeviceContext& context);
    void sum_rank(Ticket ticket, std::size_t rank, float* in_place);
    void close_rank(std::size_t rank);

    // Stream-ordered unit mode: one ticket covers `sequences` consecutive reductions of one pass.
    // Each owner calls begin_unit() once (one CPU validation barrier before either rank posts GPU
    // work), then enqueue_sum() for sequence 0..sequences-1 in order (posts the NCCL all-reduce on
    // the rank stream and returns without waiting), then finish_unit(), which returns only after
    // BOTH ranks' streams completed every reduction of the unit. Host failures between calls must
    // quarantine (the executor boundary does); a waiting peer then aborts its communicator.
    Ticket prepare_unit(std::uint64_t unit, std::uint32_t sequences, std::size_t count);
    void begin_unit(Ticket ticket, std::size_t rank);
    void enqueue_sum(Ticket ticket, std::size_t rank, std::uint32_t sequence, float* in_place);
    void finish_unit(Ticket ticket, std::size_t rank);
    // Coordinator cancellation/error before dispatch, or external failure while
    // both callbacks are running. This only latches failure; owners do cleanup.
    void quarantine() noexcept;
    bool failed() const noexcept;
    static bool available() noexcept;
    // Common owner-executor exception boundary: first latch CPU failure for ALL
    // registered collectives on this thread, then abort their local resources.
    // Call after publishing executor failure. The latch is bounded CPU work;
    // cleanup may stall in NCCL/CUDA, so coordinator must observe failed()/its
    // own deadline without waiting for this call or releasing physical storage.
    static void fail_current_thread() noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
