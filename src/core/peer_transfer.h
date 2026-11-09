#pragma once
#include "core/device.h"
#include "core/arena.h"
#include <memory>
namespace ninfer {
enum class TransferPolicy { automatic, pinned_host };
enum class TransferTransport { peer, pinned_host };
// One bounded transfer slot. Construct before dispatch; no allocations in enqueue.
// Each object and both contexts must outlive queued work. Host submissions are serial.
// Queue source writes on producer, then enqueue; destination consumers may be queued
// immediately on consumer. Before overwriting source on any other stream, wait on
// source_complete(). All destination consumers must be queued on consumer before
// the next enqueue, which orders slot reuse after those consumers. Use two objects
// for double buffering. Completion events are reused and refer to the last enqueue.
class PeerTransfer {
public:
    PeerTransfer(const DeviceContext& source, const DeviceContext& destination,
                 std::size_t max_bytes, TransferPolicy policy = TransferPolicy::automatic);
    ~PeerTransfer() noexcept;
    PeerTransfer(const PeerTransfer&) = delete;
    PeerTransfer& operator=(const PeerTransfer&) = delete;
    void enqueue(void* destination, const void* source, std::size_t bytes,
                 cudaStream_t producer, cudaStream_t consumer);
    void synchronize() const;
    TransferTransport transport() const noexcept { return transport_; }
    bool source_can_access_destination() const noexcept { return source_access_; }
    bool destination_can_access_source() const noexcept { return destination_access_; }
    const CudaCompletionEvent& source_complete() const noexcept { return source_done_; }
    const CudaCompletionEvent& destination_complete() const noexcept { return destination_done_; }
private:
    const DeviceContext& source_;
    const DeviceContext& destination_;
    std::size_t capacity_;
    bool source_access_ = false, destination_access_ = false;
    bool submitted_ = false, failed_ = false;
    TransferTransport transport_ = TransferTransport::pinned_host;
    std::unique_ptr<PinnedHostBuffer> staging_;
    CudaCompletionEvent source_ready_, destination_released_, source_done_, destination_done_;
};
}
