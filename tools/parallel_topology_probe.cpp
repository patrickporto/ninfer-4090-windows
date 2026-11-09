#include "core/arena.h"
#include "core/device_group.h"
#include "core/peer_transfer.h"
#include <nlohmann/json.hpp>
#if NINFER_HAS_NCCL
#include <nccl.h>
#endif
#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

std::vector<int> parse_devices(const std::string& text) {
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        auto end = text.find(',', begin);
        if (end == std::string::npos) end = text.size();
        int value = -1;
        auto parsed = std::from_chars(text.data() + begin, text.data() + end, value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + end || value < 0)
            throw std::invalid_argument("--devices requires two distinct nonnegative ordinals");
        result.push_back(value);
        begin = end + 1;
        if (end + 1 == text.size()) throw std::invalid_argument("trailing device separator");
    }
    if (result.size() != 2 || result[0] == result[1])
        throw std::invalid_argument("--devices requires two distinct nonnegative ordinals");
    return result;
}

Json sysfs_value(const std::string& bus, const char* field) {
#ifdef __linux__
    std::ifstream file("/sys/bus/pci/devices/" + bus + "/" + field);
    std::string value;
    if (std::getline(file, value)) return value;
#endif
    return nullptr;
}

Json describe(const ninfer::DeviceContext& ctx) {
    ninfer::ScopedDevice binding(ctx.device);
    std::size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    std::array<char, 32> bus{};
    CUDA_CHECK(cudaDeviceGetPCIBusId(bus.data(), bus.size(), ctx.device));
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
        uuid << std::setw(2) << static_cast<unsigned>(
            static_cast<unsigned char>(ctx.props.uuid.bytes[i]));
    }
    std::string pci = bus.data();
    // CUDA and Linux may print the PCI domain with different widths.
    if (auto colon = pci.find(':'); colon != std::string::npos && colon > 4)
        pci.erase(0, colon - 4);
    std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return {{"ordinal", ctx.device}, {"name", ctx.props.name}, {"uuid", uuid.str()},
            {"sm", ctx.compute_capability()}, {"pci_bus_id", pci},
            {"free_bytes", free}, {"total_bytes", total},
            {"pcie_current_link_speed", sysfs_value(pci, "current_link_speed")},
            {"pcie_current_link_width", sysfs_value(pci, "current_link_width")},
            {"pcie_max_link_speed", sysfs_value(pci, "max_link_speed")},
            {"pcie_max_link_width", sysfs_value(pci, "max_link_width")}};
}

Json measure(const ninfer::DeviceContext& source, const ninfer::DeviceContext& destination,
             ninfer::TransferPolicy policy, std::size_t bytes) {
    constexpr int warmups = 5, repeats = 30;
    std::vector<unsigned char> pattern(bytes), observed(bytes);
    for (std::size_t i = 0; i < bytes; ++i) pattern[i] = (i * 131 + i / 251 + bytes) & 255;
    auto source_binding = ninfer::ScopedDevice(source.device);
    ninfer::DeviceBuffer input(bytes);
    input.copy_from_host(pattern.data(), bytes);
    auto output = [&] {
        ninfer::ScopedDevice binding(destination.device);
        ninfer::DeviceBuffer allocation(bytes);
        allocation.fill(0);
        return allocation;
    }();
    ninfer::PeerTransfer transfer(source, destination, bytes, policy);
    std::vector<double> samples;
    for (int i = -warmups; i < repeats; ++i) {
        const auto start = Clock::now();
        transfer.enqueue(output.p, input.p, bytes, source.stream, destination.stream);
        transfer.synchronize();
        const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        if (i >= 0) samples.push_back(us);
    }
    output.copy_to_host(observed.data(), bytes);
    if (pattern != observed) throw std::runtime_error("transfer byte verification failed");
    std::sort(samples.begin(), samples.end());
    const double median = (samples[14] + samples[15]) / 2;
    return {{"source", source.device}, {"destination", destination.device},
            {"policy", policy == ninfer::TransferPolicy::automatic ? "automatic" : "pinned_host"},
            {"transport", transfer.transport() == ninfer::TransferTransport::peer ? "peer" : "pinned_host"},
            {"source_can_access_destination", transfer.source_can_access_destination()},
            {"destination_can_access_source", transfer.destination_can_access_source()},
            {"bytes", bytes}, {"warmups", warmups}, {"repeats", repeats},
            {"median_us", median}, {"p95_us", samples[28]},
            {"payload_gbytes_per_second", bytes / (median * 1000.0)},
            {"verified_exact_bytes", true}, {"samples_us", samples}};
}
}

int main(int argc, char** argv) {
    try {
        std::vector<int> devices;
        std::string output;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: ninfer_parallel_topology_probe --devices 0,1 [--json FILE]\n";
                return 0;
            }
            if ((arg != "--devices" && arg != "--json") || i + 1 == argc)
                throw std::invalid_argument("expected --devices LIST or --json FILE");
            if (arg == "--devices") devices = parse_devices(argv[++i]);
            else output = argv[++i];
        }
        if (devices.empty()) throw std::invalid_argument("explicit --devices is required");
        ninfer::DeviceGroup group(devices);
        int runtime = 0, driver = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        Json report = {{"schema_version", 1}, {"cuda_runtime_version", runtime},
                       {"cuda_driver_version", driver}, {"nccl_version", nullptr},
                       {"latency_scope", "host enqueue through both endpoint completion events; includes dependency and synchronization overhead"},
                       {"gbytes_scope", "one-way payload bytes divided by end-to-end latency; not PCIe wire bandwidth"},
                       {"devices", Json::array()}, {"devices_after_transfers", Json::array()},
                       {"transfers", Json::array()}};
#if NINFER_HAS_NCCL
        int nccl_version = 0;
        if (ncclGetVersion(&nccl_version) != ncclSuccess)
            throw std::runtime_error("NCCL version query failed");
        report["nccl_version"] = nccl_version;
#endif
        for (std::size_t rank = 0; rank < 2; ++rank)
            report["devices"].push_back(describe(group.context(rank)));
        for (std::size_t rank = 0; rank < 2; ++rank)
            for (auto policy : {ninfer::TransferPolicy::automatic, ninfer::TransferPolicy::pinned_host})
                for (auto bytes : {10ULL*1024, 20ULL*1024, 160ULL*1024, 10ULL*1024*1024, 20ULL*1024*1024})
                    report["transfers"].push_back(measure(group.context(rank), group.context(1-rank), policy, bytes));
        for (std::size_t rank = 0; rank < 2; ++rank)
            report["devices_after_transfers"].push_back(describe(group.context(rank)));
        if (!output.empty()) {
            std::ofstream file(output);
            file << report.dump(2) << '\n';
            file.close();
            if (!file) throw std::runtime_error("could not write topology report");
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topology probe: " << error.what() << '\n';
        return 1;
    }
}
