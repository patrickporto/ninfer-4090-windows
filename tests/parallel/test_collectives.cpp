#include "core/device_group.h"
#include "core/tensor_collective.h"
#include "targets/qwen3_6/impl/runtime/rank_executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using ninfer::TensorCollective;
using ninfer::TensorCollectiveDescriptor;
using ninfer::targets::qwen3_6::runtime_support::RankExecutor;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid collective protocol accepted");
}
std::array<TensorCollectiveDescriptor, 2> descriptors(std::uint64_t unit = 1,
                                                    std::uint32_t sequence = 0,
                                                    std::size_t count = 5120) {
    return {{{0, unit, sequence, count}, {1, unit, sequence, count}}};
}
void cpu() {
    rejects([] { TensorCollective c({0, 0}, 5120); });
    rejects([] { TensorCollective c({-1, 1}, 5120); });
    rejects([] { TensorCollective c({0, 1}, 0); });
    rejects([] { TensorCollective c({0, 1}, 5 * 1024 * 1024 + 1); });
    TensorCollective::fail_current_thread();
    rejects([] { TensorCollective c({0, 1}, 5120, std::chrono::milliseconds(0)); });
    rejects([] { TensorCollective c({0, 1}, 5120, std::chrono::seconds(1), 0); });
    TensorCollective c({0, 1}, 5120);
    auto d = descriptors();
    d[1].count = 5119; rejects([&] { c.prepare(d); });
    d = descriptors(); d[1].rank = 0; rejects([&] { c.prepare(d); });
    d = descriptors(); d[1].unit = 2; rejects([&] { c.prepare(d); });
    d = descriptors(); d[1].sequence = 1; rejects([&] { c.prepare(d); });
    rejects([&] { c.prepare(descriptors(1, 1)); });
    rejects([&] { c.prepare(descriptors(1, 128)); });
    rejects([&] { c.prepare(descriptors(1, 0, 0)); });
    rejects([&] { c.prepare(descriptors(1, 0, 5121)); });
    const auto ticket = c.prepare(descriptors());
    require(ticket.value != 0, "rejected descriptors consumed first valid order slot");
    rejects([&] { c.prepare(descriptors()); });
    c.quarantine();
    require(c.failed(), "pre-dispatch fault did not quarantine group");
    rejects([&] { c.prepare(descriptors(2)); });
}

// The independent oracle adds the represented FP32 inputs in FP64 and rounds
// once to FP32. Acceptance is finite and <= one ULP at that rounded result.
// No engine/KV/transaction or end-to-end performance claim is made here.
float partial(std::size_t rank, std::size_t i, std::uint32_t sequence) {
    switch ((i + sequence) % 8) {
    case 0: return 0.0f;
    case 1: return rank ? -1.0e10f : 1.0e10f;
    case 2: return rank ? -4095.75f : 4096.0f;
    case 3: return rank ? 0.000244140625f : 1.0f;
    case 4: return rank ? -0.125f : -0.25f;
    case 5: return rank ? 65536.0f : -65535.0f;
    case 6: return rank ? 1.0f : 1.0e-7f;
    default: return rank ? -16.0f : 16.125f;
    }
}
void check_sum(const std::vector<float>& values, std::uint32_t sequence) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float expected = static_cast<float>(static_cast<double>(partial(0, i, sequence)) +
                                                   static_cast<double>(partial(1, i, sequence)));
        const double upper = static_cast<double>(std::nextafter(expected, std::numeric_limits<float>::infinity())) - expected;
        const double lower = static_cast<double>(expected) - std::nextafter(expected, -std::numeric_limits<float>::infinity());
        require(std::isfinite(values[i]) && std::abs(static_cast<double>(values[i]) - expected) <= std::max(upper, lower),
                "FP32 SUM differs from independent FP64 oracle by more than one ULP");
    }
}
void owner_exit_cleanup() {
    ninfer::DeviceGroup devices({0, 1});
    auto collective = std::make_unique<TensorCollective>(std::array<int, 2>{0, 1}, 5120);
    {
        RankExecutor a(devices.context(0)), b(devices.context(1));
        auto first = a.submit(1, [&](ninfer::DeviceContext& context) {
            collective->initialize_rank(0, context);
        });
        auto second = b.submit(1, [&](ninfer::DeviceContext& context) {
            collective->initialize_rank(1, context);
        });
        std::exception_ptr error;
        try { first.get(); } catch (...) { error = std::current_exception(); }
        try { second.get(); } catch (...) { if (!error) error = std::current_exception(); }
        if (error) std::rethrow_exception(error);
        // A callback outside the collective API fails. The common executor
        // boundary invokes the Core owner hook; it is spelled out here so this
        // primitive test remains usable before family integration is registered.
        auto failed = a.submit(2, [](ninfer::DeviceContext&) {
            throw std::runtime_error("injected unrelated callback failure");
        });
        rejects([&] { failed.get(); });
        require(collective->failed(), "unrelated owner failure did not quarantine collective");
        // No manual close is possible on the quarantined owner. Destroying the
        // facade is CPU-only; guards retain state until both worker exits.
        collective.reset();
    }
}
void gpu() {
    ninfer::DeviceGroup devices({0, 1});
    TensorCollective collective({0, 1}, 5 * 1024 * 1024);
    RankExecutor a(devices.context(0)), b(devices.context(1));
    std::array<RankExecutor*, 2> executors{&a, &b};
    std::array<float*, 2> buffers{};
    std::uint64_t command = 0;
    auto pair = [&](auto fn) {
        std::array<std::future<std::exception_ptr>, 2> results;
        ++command;
        for (std::size_t rank = 0; rank < 2; ++rank)
            results[rank] = executors[rank]->submit(command, [&, rank](ninfer::DeviceContext& context) {
                // Preserve the worker for test cleanup after an injected fault.
                try { fn(rank, context); return std::exception_ptr{}; }
                catch (...) { return std::current_exception(); }
            });
        std::exception_ptr error;
        for (auto& result : results) {
            auto current = result.get();
            if (current && !error) error = current;
        }
        if (error) std::rethrow_exception(error);
    };
    auto cleanup = [&] {
        pair([&](std::size_t rank, ninfer::DeviceContext&) {
            collective.close_rank(rank);
            if (buffers[rank]) {
                ninfer::cuda_require(cudaFree(buffers[rank]), "cudaFree test collective buffer");
                buffers[rank] = nullptr;
            }
        });
    };
    try {
        pair([&](std::size_t rank, ninfer::DeviceContext& context) {
            collective.initialize_rank(rank, context);
        });
        pair([&](std::size_t rank, ninfer::DeviceContext&) {
            ninfer::cuda_require(cudaMalloc(&buffers[rank], 20 * 1024 * 1024), "cudaMalloc test collective buffer");
        });
        for (std::uint32_t sequence = 0; sequence < 128; ++sequence) {
            // Representative payloads: 10KiB, 20KiB, 160KiB, 10MiB, 20MiB.
            constexpr std::array<std::size_t, 5> counts{2560, 5120, 40960, 2621440, 5242880};
            const std::size_t count = sequence < counts.size() ? counts[sequence] : 5120;
            const auto ticket = collective.prepare(descriptors(1, sequence, count));
            pair([&](std::size_t rank, ninfer::DeviceContext& context) {
                std::vector<float> host(count);
                for (std::size_t i = 0; i < count; ++i) host[i] = partial(rank, i, sequence);
                ninfer::cuda_require(cudaMemcpyAsync(buffers[rank], host.data(), count * sizeof(float),
                    cudaMemcpyHostToDevice, context.stream), "copy collective inputs");
                collective.sum_rank(ticket, rank, buffers[rank]);
                ninfer::cuda_require(cudaMemcpyAsync(host.data(), buffers[rank], count * sizeof(float),
                    cudaMemcpyDeviceToHost, context.stream), "copy collective result");
                ninfer::cuda_require(cudaStreamSynchronize(context.stream), "read collective result");
                check_sum(host, sequence);
            });
        }
        rejects([&] { collective.prepare(descriptors(1, 0)); });
        rejects([&] { collective.prepare(descriptors(2, 1)); });
        const auto ticket = collective.prepare(descriptors(2));
        // Fault before NCCL posting: both owners abort and report quarantine.
        collective.quarantine();
        rejects([&] {
            pair([&](std::size_t rank, ninfer::DeviceContext&) {
                collective.sum_rank(ticket, rank, buffers[rank]);
            });
        });
        require(collective.failed(), "fault did not quarantine collective");
        cleanup();
    } catch (...) {
        const auto error = std::current_exception();
        try { cleanup(); } catch (...) {}
        std::rethrow_exception(error);
    }
}
}
int main(int argc, char** argv) {
    try {
        cpu();
        if (argc == 2 && std::string(argv[1]) == "--gpu") {
            if (!TensorCollective::available()) return 77;
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
            gpu();
            owner_exit_cleanup();
        } else if (argc != 1) {
            throw std::invalid_argument("usage: test_collectives [--gpu]");
        }
        std::cout << "PASS tensor collective CPU preflight" << (argc == 2 ? " and two-GPU FP32 SUM" : "") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
