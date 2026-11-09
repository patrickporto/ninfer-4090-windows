#include "targets/qwen3_6_27b/impl/load/tensor_weights.h"
#include "core/cuda_error_scope.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <string_view>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ninfer::artifact;
using namespace ninfer::targets::qwen3_6_27b::detail;

namespace {
void check(bool condition) {
    if (!condition) throw std::runtime_error("tensor materialization assertion failed");
}

template <class F>
void rejects(F f) {
    try {
        f();
    } catch (const ArtifactError&) { return; }
    throw std::runtime_error("expected ArtifactError");
}

void inventory_contract() {
    for (bool optional : {false, true}) {
        const auto text = tensor_inventory_json(optional);
        validate_tensor_inventory(text, optional);
        auto baseline = nlohmann::json::parse(text);
        auto bad      = baseline;
        bad["extra"]  = 1;
        rejects([&] { validate_tensor_inventory(bad.dump(), optional); });
        bad               = baseline;
        bad["world_size"] = 2.0;
        rejects([&] { validate_tensor_inventory(bad.dump(), optional); });
        bad               = baseline;
        bad["owner_rank"] = false;
        rejects([&] { validate_tensor_inventory(bad.dump(), optional); });
        bad                            = baseline;
        bad["rank_objects"][1]["rank"] = 0;
        rejects([&] { validate_tensor_inventory(bad.dump(), optional); });
        bad                                  = baseline;
        bad["rank_objects"][0]["objects"][0] = "rank1/text/layers/0/input_norm";
        rejects([&] { validate_tensor_inventory(bad.dump(), optional); });
        bad = baseline;
        bad["owner_objects"].erase(bad["owner_objects"].begin());
        rejects([&] { validate_tensor_inventory(bad.dump(), optional); });
        bad                             = baseline;
        bad["aliases"][0]["objects"][0] = "text/output_head";
        rejects([&] { validate_tensor_inventory(bad.dump(), optional); });
        rejects([&] { validate_tensor_inventory(text, !optional); });
        rejects(
            [&] { validate_tensor_inventory("{\"owner_rank\":0," + text.substr(1), optional); });
    }
}

void ownership_contract() {
    MaterializationPlan p{
        5, 529, {{{0}, 0, 3, 16}, {{2}, 256, 5, 256}, {{3}, 512, 17, 256}}, {{{1}}}};
    using P = TensorObjectPlacement;
    std::vector<TensorObjectAssignment> a{{{0}, P::Rank0Device},
                                          {{1}, P::OwnerHost},
                                          {{2}, P::Rank1Device},
                                          {{3}, P::Rank1Device},
                                          {{4}, P::ValidateOnly}};
    const auto plans = tensor_materialization_plans(p, a);
    check(plans[0].object_count == 5 && plans[1].object_count == 5);
    check(plans[0].device_capacity_bytes == 3 && plans[1].device_capacity_bytes == 273);
    check(plans[0].host_objects.size() == 1 && plans[1].host_objects.empty());
    check(plans[1].device_objects[0].object.index == 2);
    check(plans[1].device_objects[1].object.index == 3);
    auto bad = a;
    bad.pop_back();
    rejects([&] { tensor_materialization_plans(p, bad); });
    bad           = a;
    bad[4].object = {3};
    rejects([&] { tensor_materialization_plans(p, bad); });
    bad           = a;
    bad[4].object = {5};
    rejects([&] { tensor_materialization_plans(p, bad); });
    bad              = a;
    bad[2].placement = P::ValidateOnly;
    rejects([&] { tensor_materialization_plans(p, bad); });
    bad              = a;
    bad[4].placement = P::Rank0Device;
    rejects([&] { tensor_materialization_plans(p, bad); });
    bad              = a;
    bad[1].placement = P::Rank1Device;
    rejects([&] { tensor_materialization_plans(p, bad); });
    bad              = a;
    bad[4].placement = static_cast<P>(99);
    rejects([&] { tensor_materialization_plans(p, bad); });
    auto broken                    = p;
    broken.device_objects[0].bytes = std::numeric_limits<std::uint64_t>::max();
    rejects([&] { tensor_materialization_plans(broken, a); });
}

// Backing qualification only: no runtime model views or TP arithmetic are constructed.
void gpu_backing(const Reader& reader, const TensorArtifactLoadPlan& plan) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) throw std::runtime_error("TP2 load views require two visible CUDA devices");
    ninfer::DeviceContext devices[]{ninfer::DeviceContext(0), ninfer::DeviceContext(1)};
    for (int rank = 0; rank < 2; ++rank) {
        const ninfer::ScopedDevice scope(rank);
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto needed = plan.materialization[rank].device_capacity_bytes;
        std::cout << "TP2 preflight rank=" << rank << " device=" << devices[rank].props.name
                  << " required_bytes=" << needed << " free_bytes=" << free
                  << " total_bytes=" << total << std::endl;
        if (needed > free)
            throw std::runtime_error("insufficient free VRAM for TP2 rank " + std::to_string(rank));
    }
    // Both allocations stay alive through every ownership and byte check.
    std::array<MaterializedArtifact, 2> resident;
    for (int rank = 0; rank < 2; ++rank) {
        const ninfer::ScopedDevice caller(1 - rank);
        resident[rank] = materialize(reader, plan.materialization[rank], devices[rank]);
        int current    = -1;
        CUDA_CHECK(cudaGetDevice(&current));
        check(current == 1 - rank);
    }
    std::vector<unsigned> global_coverage(reader.objects().size());
    for (int rank = 0; rank < 2; ++rank) {
        const ninfer::ScopedDevice scope(rank);
        const auto& local   = plan.materialization[rank];
        const auto& backing = resident[rank];
        std::vector<bool> device_selected(reader.objects().size()),
            host_selected(reader.objects().size());
        std::uint64_t expected_h2d = 0, expected_host = 0;
        for (const auto& selected : local.device_objects) {
            const auto handle = selected.object;
            check(++global_coverage.at(handle.index) == 1);
            device_selected.at(handle.index) = true;
            const auto& descriptor           = reader.objects().at(handle.index);
            const std::string label =
                "rank=" + std::to_string(rank) + " object=" + std::string(object_name(descriptor));
            const auto* pointer = static_cast<const std::byte*>(backing.device_data(handle));
            cudaPointerAttributes attributes{};
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
            if (attributes.type != cudaMemoryTypeDevice || attributes.device != rank)
                throw std::runtime_error("wrong CUDA backing owner: " + label);
            const auto payload = reader.payload(descriptor).data;
            check(payload.size() == selected.bytes);
            const auto sample_size = std::min<std::size_t>(16, payload.size());
            check(sample_size > 0);
            const std::array<std::size_t, 3> offsets{0, (payload.size() - sample_size) / 2,
                                                     payload.size() - sample_size};
            for (const auto offset : offsets) {
                std::array<std::byte, 16> actual{};
                CUDA_CHECK(cudaMemcpy(actual.data(), pointer + offset, sample_size,
                                      cudaMemcpyDeviceToHost));
                if (!std::equal(actual.begin(), actual.begin() + sample_size,
                                payload.begin() + offset))
                    throw std::runtime_error("GPU payload mismatch: " + label +
                                             " offset=" + std::to_string(offset));
            }
            expected_h2d += selected.bytes;
        }
        for (const auto& selected : local.host_objects) {
            const auto handle = selected.object;
            check(rank == 0 && ++global_coverage.at(handle.index) == 1);
            host_selected.at(handle.index) = true;
            const auto actual              = backing.resource_bytes(handle);
            const auto expected            = reader.payload(reader.objects().at(handle.index)).data;
            check(actual.size() == expected.size());
            if (!std::equal(actual.begin(), actual.end(), expected.begin()))
                throw std::runtime_error("host resource payload mismatch: " +
                                         std::string(object_name(reader.objects()[handle.index])));
            expected_host += expected.size();
        }
        for (std::size_t index = 0; index < reader.objects().size(); ++index) {
            if (!device_selected[index]) rejects([&] { backing.device_data({index}); });
            if (!host_selected[index]) rejects([&] { backing.resource_bytes({index}); });
        }
        const auto& stats = backing.stats();
        check(stats.device_capacity_bytes == local.device_capacity_bytes);
        check(stats.h2d_bytes == expected_h2d && stats.retained_resource_bytes == expected_host);
        check(stats.tensor_count == local.device_objects.size() &&
              stats.resource_count == local.host_objects.size());
        check(stats.file_bytes >= expected_h2d + expected_host);
        check(stats.peak_staging_bytes > 0);
        std::cout << "TP2 backing rank=" << rank << " tensors=" << stats.tensor_count
                  << " resources=" << stats.resource_count
                  << " device_bytes=" << stats.device_capacity_bytes
                  << " h2d_bytes=" << stats.h2d_bytes
                  << " host_bytes=" << stats.retained_resource_bytes
                  << " peak_staging_bytes=" << stats.peak_staging_bytes
                  << " file_bytes=" << stats.file_bytes
                  << " upload_seconds=" << stats.upload_seconds << std::endl;
    }
    std::size_t covered = 0, validated = 0;
    for (std::size_t index = 0; index < reader.objects().size(); ++index) {
        const auto name = object_name(reader.objects()[index]);
        const bool validate_only =
            name == "parallel/tp2_inventory.json" || name.starts_with("dflash2/");
        check(global_coverage[index] == (validate_only ? 0U : 1U));
        if (validate_only)
            ++validated;
        else
            ++covered;
    }
    std::cout << "TP2 simultaneous backing passed: covered=" << covered
              << " validate_only=" << validated
              << " disjoint_global_handles=true samples_per_tensor=3 sample_bytes=16\n";
}

void real_artifact(const char* path, bool load_views) {
    Reader reader(path);
    auto plan = bind_tensor_artifact(reader, {});
    check(plan.ranks[0].rank == 0 && plan.ranks[1].rank == 1);
    std::vector<int> seen(reader.objects().size());
    for (std::size_t rank = 0; rank < 2; ++rank) {
        const auto& local = plan.materialization[rank];
        check(local.object_count == reader.objects().size());
        check(local.device_capacity_bytes < reader.file_bytes());
        for (const auto& d : local.device_objects) {
            check(++seen[d.object.index] == 1);
            const auto name          = object_name(reader.objects()[d.object.index]);
            const std::string prefix = "rank" + std::to_string(rank) + "/";
            check(name.starts_with(prefix) || (rank == 0 && name.starts_with("text/")));
        }
        for (const auto& h : local.host_objects) check(++seen[h.object.index] == 1);
        for (std::size_t l = 0; l < 64; ++l) {
            const auto& layer = plan.ranks[rank].layers[l];
            check(std::holds_alternative<TensorAttentionPlan>(layer.mixer) == (l % 4 == 3));
            const auto& desc =
                std::get<TensorDescriptor>(reader.objects()[layer.mlp.down.object.index]);
            check(desc.shape == std::vector<std::uint64_t>({5120, 8704}));
        }
    }
    check(plan.materialization[0].host_objects.size() == 6);
    check(plan.materialization[1].host_objects.empty());
    check(plan.materialization[0].device_objects.size() == 771);
    check(plan.materialization[1].device_objects.size() == 768);
    for (std::size_t l = 0; l < 64; ++l)
        check(plan.ranks[0].layers[l].input_norm.index != plan.ranks[1].layers[l].input_norm.index);
    const ninfer::targets::qwen3_6::StartupFeatures enabled{
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized};
    const auto complete = bind_tensor_artifact(reader, enabled);
    check(complete.materialization[1].device_capacity_bytes ==
          plan.materialization[1].device_capacity_bytes);
    check(complete.materialization[1].device_objects.size() == 768);
    std::size_t disabled = 1; // inventory is validation-only
    for (const auto& object : reader.objects())
        if (object_name(object).starts_with("dflash2/")) ++disabled;
    check(complete.materialization[0].device_objects.size() +
              complete.materialization[1].device_objects.size() + 6 + disabled ==
          reader.objects().size());
    for (const auto& object : complete.materialization[0].device_objects)
        check(!object_name(reader.objects()[object.object.index]).starts_with("dflash2/"));
    for (auto backend : {ninfer::SpeculativeBackend::DFlash, ninfer::SpeculativeBackend::DFlash2}) {
        auto unsupported        = enabled;
        unsupported.speculative = backend;
        rejects([&] { bind_tensor_artifact(reader, unsupported); });
    }
    if (load_views) gpu_backing(reader, complete);
    std::cout << "real TP2 strict binding passed: " << reader.objects().size() << " objects\n";
}
} // namespace

int main() {
    try {
        ninfer::ScopedCudaErrors errors;
        inventory_contract();
        ownership_contract();
        const char* load      = std::getenv("NINFER_TEST_TP2_LOAD_VIEWS");
        const bool load_views = load && std::string_view(load) == "1";
        const char* path      = std::getenv("NINFER_TEST_TP2_ARTIFACT");
        if (load_views && (!path || !*path))
            throw std::runtime_error(
                "NINFER_TEST_TP2_LOAD_VIEWS=1 requires NINFER_TEST_TP2_ARTIFACT");
        if (path && *path) real_artifact(path, load_views);
        std::cout << "tensor inventory and ownership passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor materialization failed: " << error.what() << '\n';
        return 1;
    }
}
