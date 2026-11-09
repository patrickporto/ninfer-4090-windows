#include "core/peer_transfer.h"
#include <cstdio>
#include <stdexcept>
namespace ninfer {
namespace {
void validate_pointer(const void* pointer, int device) {
    if (!pointer) throw std::invalid_argument("transfer endpoint is null");
    cudaPointerAttributes attr{};
    cuda_require(cudaPointerGetAttributes(&attr, pointer), "transfer pointer query");
    if (attr.type != cudaMemoryTypeDevice || attr.device != device)
        throw std::invalid_argument("transfer endpoint device mismatch");
}
void drain_noexcept(const DeviceContext& ctx) noexcept {
    ScopedDevice binding(ctx.device, std::nothrow);
    for (auto stream : {ctx.stream, ctx.transfer_stream}) {
        auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) std::fprintf(stderr, "transfer cleanup: %s\n", cudaGetErrorString(status));
    }
}
}
PeerTransfer::PeerTransfer(const DeviceContext& source, const DeviceContext& destination,
                           std::size_t max_bytes, TransferPolicy policy)
    : source_(source), destination_(destination), capacity_(max_bytes),
      source_ready_(source), destination_released_(destination),
      source_done_(source), destination_done_(destination) {
    if (source.device == destination.device) throw std::invalid_argument("transfer requires distinct devices");
    if (!capacity_) throw std::invalid_argument("transfer capacity must be nonzero");
    int forward=0, reverse=0;
    cuda_require(cudaDeviceCanAccessPeer(&forward, source.device, destination.device), "source peer capability");
    cuda_require(cudaDeviceCanAccessPeer(&reverse, destination.device, source.device), "destination peer capability");
    source_access_=forward; destination_access_=reverse;
    if (policy == TransferPolicy::automatic && source_access_) {
        ScopedDevice binding(source.device);
        auto status=cudaDeviceEnablePeerAccess(destination.device, 0);
        if (status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled) {
            if (status == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
            transport_=TransferTransport::peer;
        } else if (status == cudaErrorTooManyPeers || status == cudaErrorPeerAccessUnsupported) {
            (void)cudaGetLastError();
        } else cuda_require(status, "enable directional peer access");
    }
    if (transport_ == TransferTransport::pinned_host) staging_=std::make_unique<PinnedHostBuffer>(capacity_);
    // Enablement belongs to the CUDA primary context. Never disable another user's access.
}
PeerTransfer::~PeerTransfer() noexcept {
    if (submitted_) { drain_noexcept(source_); drain_noexcept(destination_); }
}
void PeerTransfer::enqueue(void* destination, const void* source, std::size_t bytes,
                           cudaStream_t producer, cudaStream_t consumer) {
    if (failed_) throw std::logic_error("transfer endpoint failed; recreate transfer");
    if (producer == nullptr || consumer == nullptr || producer == cudaStreamLegacy ||
        consumer == cudaStreamLegacy || producer == cudaStreamPerThread || consumer == cudaStreamPerThread)
        throw std::invalid_argument("transfer requires explicit endpoint streams");
    if (bytes > capacity_) throw std::out_of_range("transfer exceeds reserved capacity");
    if (cuda_stream_device(producer) != source_.device || cuda_stream_device(consumer) != destination_.device)
        throw std::invalid_argument("transfer stream device mismatch");
    validate_pointer(source, source_.device); validate_pointer(destination, destination_.device);
    // All validation precedes submission, so rejected calls do not poison a reusable slot.
    try {
        destination_released_.record(consumer);
        source_ready_.record(producer);
        {
            ScopedDevice binding(source_.device);
            source_ready_.wait(source_.transfer_stream);
            if (submitted_) destination_done_.wait(source_.transfer_stream);
            // The direct path writes destination, so also wait for its previous consumers.
            if (transport_ == TransferTransport::peer) {
                destination_released_.wait(source_.transfer_stream);
                submitted_=true;
                cuda_require(cudaMemcpyPeerAsync(destination, destination_.device, source, source_.device,
                                                 bytes, source_.transfer_stream), "peer copy");
            } else {
                submitted_=true;
                cuda_require(cudaMemcpyAsync(staging_->data(), source, bytes, cudaMemcpyDeviceToHost,
                                             source_.transfer_stream), "staged device-to-host copy");
            }
            source_done_.record(source_.transfer_stream);
        }
        {
            ScopedDevice binding(destination_.device);
            destination_released_.wait(destination_.transfer_stream);
            source_done_.wait(destination_.transfer_stream);
            if (transport_ == TransferTransport::pinned_host)
                cuda_require(cudaMemcpyAsync(destination, staging_->data(), bytes, cudaMemcpyHostToDevice,
                                             destination_.transfer_stream), "staged host-to-device copy");
            destination_done_.record(destination_.transfer_stream);
            destination_done_.wait(consumer);
        }
    } catch (...) {
        failed_=true;
        // Settle both endpoints before callers can release either external buffer.
        drain_noexcept(source_); drain_noexcept(destination_);
        throw;
    }
}
void PeerTransfer::synchronize() const {
    if (!submitted_) return;
    source_done_.synchronize(); destination_done_.synchronize();
}
}
