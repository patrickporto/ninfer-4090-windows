#include "core/device.h"
#include "core/cuda_error_scope.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

thread_local bool throw_cuda_errors = false;

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(cuda_error_message(operation, status));
}
int cuda_stream_device(cudaStream_t stream) {
    int device = -1;
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        cuda_require(cudaGetDevice(&device), "default stream current device");
        return device;
    }
    cuda_require(cudaStreamGetDevice(stream, &device), "cudaStreamGetDevice");
    return device;
}
ScopedDevice::ScopedDevice(int device) {
    cuda_require(cudaGetDevice(&previous_), "cudaGetDevice");
    cuda_require(cudaSetDevice(device), "cudaSetDevice");
}
ScopedDevice::ScopedDevice(int device, std::nothrow_t) noexcept {
    log_cuda_error("cudaGetDevice", cudaGetDevice(&previous_));
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}
ScopedDevice::~ScopedDevice() noexcept {
    if (previous_ >= 0) log_cuda_error("cudaSetDevice(restore)", cudaSetDevice(previous_));
}

ScopedCudaErrors::ScopedCudaErrors(bool enable_throw) noexcept
    : previous_(throw_cuda_errors) {
    throw_cuda_errors = previous_ || enable_throw;
}

ScopedCudaErrors::~ScopedCudaErrors() noexcept { throw_cuda_errors = previous_; }

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    if (throw_cuda_errors) {
        const std::string operation = std::string(file) + ":" + std::to_string(line) +
                                      ": CUDA_CHECK(" + expr + ") failed";
        throw std::runtime_error(cuda_error_message(operation.c_str(), err));
    }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream          = compute;
    transfer_stream = load;
}

DeviceContext::~DeviceContext() {
    ScopedDevice binding(device, std::nothrow);
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props) {
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    ScopedDevice binding(device, std::nothrow);
    destroy_stream(transfer_stream);
    destroy_stream(stream);

    device          = other.device;
    props           = other.props;
    stream          = other.stream;
    transfer_stream = other.transfer_stream;

    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

void DeviceContext::synchronize() const { ScopedDevice binding(device); CUDA_CHECK(cudaStreamSynchronize(stream)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : device_(ctx.device), stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ScopedDevice binding(ctx.device);

    if (cuda_stream_device(stream) != device_) throw std::invalid_argument("timer stream device mismatch");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    ScopedDevice binding(device_, std::nothrow);
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : device_(other.device_), stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    ScopedDevice binding(device_, std::nothrow);
    destroy_event(stop_);
    destroy_event(start_);

    device_ = other.device_;
    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { ScopedDevice binding(device_); CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { ScopedDevice binding(device_); CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    ScopedDevice binding(device_);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    ScopedDevice binding(device_);
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ScopedDevice binding(ctx.device);
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { ScopedDevice binding(device_, std::nothrow); destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    ScopedDevice binding(device_, std::nothrow);
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    if (cuda_stream_device(stream) != device_) throw std::invalid_argument("event record stream device mismatch");
    ScopedDevice binding(device_);
    cuda_require(cudaEventRecord(event_, stream), "cudaEventRecord");
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    ScopedDevice binding(cuda_stream_device(stream));
    cuda_require(cudaStreamWaitEvent(stream, event_, 0), "cudaStreamWaitEvent");
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    ScopedDevice binding(device_);
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    cuda_require(status, "cudaEventQuery");
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    ScopedDevice binding(device_);
    cuda_require(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

} // namespace ninfer
