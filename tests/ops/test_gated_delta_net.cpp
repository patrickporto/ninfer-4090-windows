#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/l2norm.h"

#include "ops/gdn_ref.h"
#include "ops/op_tester.h"
// Task-local capture diagnosis only; ordinary qualification uses the public Op.
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateDim = 128;

constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

constexpr ReductionCriterion gated_delta_net_state_fp32_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/3.9e-3};
}

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

constexpr Case kLongCase{"27b measured direct prefill fused-qk-norm", 16, 48, 1024, true};
constexpr std::uint32_t kLongSeed = 13024u;

struct CapturedInputs {
    gdn_ref::Inputs represented;
    std::vector<std::uint16_t> q_bits, k_bits, v_bits;
};

template <typename T>
std::vector<T> read_capture_tensor(const std::filesystem::path& path, std::size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(T));
    if (!file || file.tellg() != bytes) {
        throw std::runtime_error("capture missing or unexpected byte count: " + path.string());
    }
    file.seekg(0);
    std::vector<T> values(count);
    if (!file.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("capture read failed: " + path.string());
    }
    return values;
}

CapturedInputs load_capture(const std::filesystem::path& directory) {
    if (!directory.is_absolute() || std::endian::native != std::endian::little) {
        throw std::runtime_error("capture requires an absolute directory and little-endian host");
    }
    std::ifstream manifest_file(directory / "manifest.json");
    if (!manifest_file) throw std::runtime_error("capture manifest missing");
    const auto manifest = nlohmann::json::parse(manifest_file);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const nlohmann::json expected = {
        {"tokens", 1024}, {"qk_heads", 16}, {"value_heads", 48}, {"head_dim", 128},
        {"normalize_qk", true}, {"initial_state", "zero"},
        {"layout", "contiguous fastest dimension first"}, {"byte_order", "little-endian"},
        {"q_shape", {128, 16, 1024}}, {"k_shape", {128, 16, 1024}},
        {"v_shape", {128, 48, 1024}}, {"g_shape", {48, 1024}}, {"beta_shape", {48, 1024}},
        {"output_shape", {128, 48, 1024}}, {"state_shape", {128, 128, 48}}};
    for (const auto& [key, value] : expected.items()) {
        if (!manifest.contains(key) || manifest.at(key) != value) {
            throw std::runtime_error("unsupported capture manifest field: " + key);
        }
    }
    if (manifest.at("scale").get<float>() != scale) {
        throw std::runtime_error("capture scale differs from represented 1/sqrt(128)");
    }
    CapturedInputs capture;
    auto& in = capture.represented;
    in.head_dim = 128; in.qk_heads = 16; in.value_heads = 48; in.tokens = 1024;
    capture.q_bits = read_capture_tensor<std::uint16_t>(directory / "q.bf16", 128 * 16 * 1024);
    capture.k_bits = read_capture_tensor<std::uint16_t>(directory / "k.bf16", 128 * 16 * 1024);
    capture.v_bits = read_capture_tensor<std::uint16_t>(directory / "v.bf16", 128 * 48 * 1024);
    const auto decode = [](const std::vector<std::uint16_t>& bits) {
        std::vector<float> values(bits.size());
        std::transform(bits.begin(), bits.end(), values.begin(), bf16_to_f32);
        return values;
    };
    in.q = decode(capture.q_bits);
    in.k = decode(capture.k_bits);
    in.v = decode(capture.v_bits);
    in.g = read_capture_tensor<float>(directory / "g.fp32", 48 * 1024);
    in.beta = read_capture_tensor<float>(directory / "beta.fp32", 48 * 1024);
    in.state.assign(128 * 128 * 48, 0.0f);
    // Only represented inputs are consumed. Saved device outputs/state never
    // participate in the independent FP64 mathematical reference.
    std::cout << "GDN capture=" << directory << " T=1024 Hqk=16 Hv=48 D=128"
              << " normalize_qk=true initial_state=zero scale=" << std::hexfloat << scale
              << std::defaultfloat << std::endl;
    return capture;
}

// Fingerprint represented host inputs so separate device runs can confirm they
// received identical Q/K/V, gates, beta, and initial state before recurrence.
std::uint64_t input_fingerprint(const std::vector<float>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (float value : values) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (bits >> (8 * byte)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    if (test_case.tokens == kLongCase.tokens) {
        std::cout << test_case.name << " fixture seed=" << seed
                  << " T=" << in.tokens << " Hqk=" << in.qk_heads
                  << " Hv=" << in.value_heads << " D=" << in.head_dim
                  << " represented-input-fnv1a64 q=" << std::hex << input_fingerprint(in.q)
                  << " k=" << input_fingerprint(in.k) << " v=" << input_fingerprint(in.v)
                  << " g=" << input_fingerprint(in.g) << " beta=" << input_fingerprint(in.beta)
                  << " state=" << input_fingerprint(in.state) << std::dec << std::endl;
    }
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return ninfer::test::verify_exact(label.c_str(), got, expected);
}

int verify_recurrence(const std::string& label, const std::vector<double>& got,
                      const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), got, expected, criterion);
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in, const CapturedInputs* capture = nullptr)
        : q(capture ? to_device(capture->q_bits) : to_device_bf16(in.q)),
          k(capture ? to_device(capture->k_bits) : to_device_bf16(in.k)),
          v(capture ? to_device(capture->v_bits) : to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

int inplace_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_tensor(state.data(), DType::FP32, {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace, state_tensor,
                         out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " inplace";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += state.verify_guards((label + " state").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int distinct_state_case(const Case& test_case, std::uint32_t seed,
                        const CapturedInputs* capture = nullptr) {
    const gdn_ref::Inputs in = capture ? capture->represented : make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in, capture);
    GuardedDeviceBuffer state_in(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(in.state.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_in_tensor(state_in.data(), DType::FP32,
                           {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_out_tensor(state_out.data(), DType::FP32,
                            {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace,
                         state_in_tensor, state_out_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct-state";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state_out.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += verify_exact(label + " state-in unchanged",
                             from_device<float>(state_in.data(), in.state.size()), in.state);
    failures += state_in.verify_guards((label + " state-in").c_str());
    failures += state_out.verify_guards((label + " state-out").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    const char* direct_diagnostic = std::getenv("NINFER_TEST_GDN_DIRECT_DIAGNOSTIC");
    if (capture != nullptr && direct_diagnostic != nullptr &&
        std::string(direct_diagnostic) == "1") {
        // Diagnostic comparisons retain the ORIGINAL represented-input FP64
        // reference. In particular, route B's private BF16 normalization is
        // measured as implementation error, never incorporated in the oracle.
        DeviceBuffer normalized_q(in.q.size() * sizeof(std::uint16_t));
        DeviceBuffer normalized_k(in.k.size() * sizeof(std::uint16_t));
        Tensor q_normalized(normalized_q.p, DType::BF16,
                            {kStateDim, test_case.qk_heads, test_case.tokens});
        Tensor k_normalized(normalized_k.p, DType::BF16,
                            {kStateDim, test_case.qk_heads, test_case.tokens});
        const auto fingerprint = [](const auto& values) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
            std::uint64_t hash = 14695981039346656037ULL;
            for (std::size_t i = 0; i < values.size() * sizeof(values[0]); ++i) {
                hash ^= bytes[i];
                hash *= 1099511628211ULL;
            }
            return hash;
        };
        const auto run_route = [&](const char* name, auto&& launch) {
            cudaEvent_t start{}, stop{};
            cuda_check(cudaEventCreate(&start), "diagnostic event create start");
            cuda_check(cudaEventCreate(&stop), "diagnostic event create stop");
            float total_ms = 0.0f, minimum_ms = 0.0f, maximum_ms = 0.0f;
            for (int iteration = 0; iteration < 8; ++iteration) {
                // Same stream: initial zero state and output poisoning complete
                // before the start event, outside every measured interval.
                cuda_check(cudaMemsetAsync(state_in.data(), 0, state_in.bytes(), nullptr),
                           "diagnostic reset state-in");
                cuda_check(cudaMemsetAsync(state_out.data(), 0xff, state_out.bytes(), nullptr),
                           "diagnostic reset state-out");
                cuda_check(cudaMemsetAsync(out.data(), 0xff, out.bytes(), nullptr),
                           "diagnostic reset output");
                cuda_check(cudaEventRecord(start, nullptr), "diagnostic record start");
                launch();
                cuda_check(cudaEventRecord(stop, nullptr), "diagnostic record stop");
                cuda_check(cudaEventSynchronize(stop), "diagnostic synchronize stop");
                if (iteration >= 3) {
                    float ms;
                    cuda_check(cudaEventElapsedTime(&ms, start, stop), "diagnostic elapsed");
                    total_ms += ms;
                    minimum_ms = iteration == 3 ? ms : std::min(minimum_ms, ms);
                    maximum_ms = iteration == 3 ? ms : std::max(maximum_ms, ms);
                }
            }
            cuda_check(cudaEventDestroy(start), "diagnostic destroy start");
            cuda_check(cudaEventDestroy(stop), "diagnostic destroy stop");
            const auto output_bits = from_device<std::uint16_t>(out.data(), in.v.size());
            const auto final_state = from_device<float>(state_out.data(), in.state.size());
            std::vector<double> output_values(output_bits.size());
            std::transform(output_bits.begin(), output_bits.end(), output_values.begin(),
                           [](std::uint16_t value) { return double(bf16_to_f32(value)); });
            std::cout << "GDN_DIRECT_DIAGNOSTIC route=" << name
                      << " warmup=3 timed=5 gpu_mean_ms=" << total_ms / 5.0f
                      << " gpu_min_ms=" << minimum_ms << " gpu_max_ms=" << maximum_ms
                      << " output_fnv1a64=" << std::hex << fingerprint(output_bits)
                      << " state_fnv1a64=" << fingerprint(final_state) << std::dec << '\n';
            const std::string route_label = std::string("captured diagnostic ") + name;
            int route_failures = verify_recurrence(route_label + " out", output_values,
                                                   ref.out, gated_delta_net_output_bf16_criterion());
            route_failures += verify_recurrence(route_label + " state", doubles(final_state),
                                                ref.final_state, gated_delta_net_state_fp32_criterion());
            route_failures += verify_exact(route_label + " state-in unchanged",
                                            from_device<float>(state_in.data(), in.state.size()), in.state);
            route_failures += state_in.verify_guards((route_label + " state-in").c_str());
            route_failures += state_out.verify_guards((route_label + " state-out").c_str());
            route_failures += out.verify_guards((route_label + " out").c_str());
            return route_failures;
        };
        failures += run_route("public-chunk", [&] {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace,
                                 state_in_tensor, state_out_tensor, out_tensor, nullptr);
        });
        failures += run_route("A-direct-raw-qk-fp32-normalization", [&] {
            ops::detail::gated_delta_net::launch_recurrent_inout(
                q, k, v, g, beta, scale, true, state_in_tensor, state_out_tensor, out_tensor, nullptr);
        });
        // Route B timing includes both public l2norm calls on every invocation.
        failures += run_route("B-direct-public-bf16-normalization", [&] {
            ops::l2norm(q, 1.0e-6f, q_normalized, nullptr);
            ops::l2norm(k, 1.0e-6f, k_normalized, nullptr);
            ops::detail::gated_delta_net::launch_recurrent_inout(
                q_normalized, k_normalized, v, g, beta, scale, false,
                state_in_tensor, state_out_tensor, out_tensor, nullptr);
        });
        failures += verify_common_inputs_unchanged("captured direct diagnostics", in, device.q,
                                                   device.k, device.v, device.g, device.beta);
    }
    return failures;
}

int batch_update_case(const Case& test_case, const std::vector<int>& source_slots,
                      const std::vector<int>& destination_slots, int slots, std::uint32_t seed) {
    if (test_case.tokens != 1) { throw std::logic_error("batch_update_case requires W=1"); }
    if (source_slots.size() != destination_slots.size()) {
        throw std::logic_error("batch_update_case selector sizes differ");
    }
    const int batch   = static_cast<int>(source_slots.size());
    const int width   = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t qk_row_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * width);
    const std::size_t value_row_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * width);
    const std::size_t gate_row_size = static_cast<std::size_t>(test_case.value_heads * width);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);

    gdn_ref::Inputs aggregate;
    aggregate.head_dim    = kStateDim;
    aggregate.qk_heads    = test_case.qk_heads;
    aggregate.value_heads = test_case.value_heads;
    aggregate.tokens      = static_cast<std::int64_t>(width) * batch;
    aggregate.q.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(value_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(gate_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(gate_row_size * static_cast<std::size_t>(batch));

    std::vector<gdn_ref::Inputs> rows;
    rows.reserve(static_cast<std::size_t>(batch));
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 0.125f);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() +
                      static_cast<std::size_t>(source_slots[static_cast<std::size_t>(row)]) *
                          state_size);
        rows.push_back(std::move(input));
    }

    std::vector<gdn_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<double> expected_output(value_row_size * static_cast<std::size_t>(batch), 0.0);
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Result reference =
            gdn_ref::evaluate(rows[static_cast<std::size_t>(row)], static_cast<double>(scale),
                              test_case.normalize_qk);
        std::copy(reference.out.begin(), reference.out.end(),
                  expected_output.begin() + static_cast<std::size_t>(row) * value_row_size);
        written_slots[static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)])] =
            true;
        references.push_back(std::move(reference));
    }

    DeviceInputs device(aggregate);
    GuardedDeviceBuffer states(initial_states.size() * sizeof(float));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_states.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_source_slots      = to_device(source_slots);
    DeviceBuffer device_destination_slots = to_device(destination_slots);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor states_tensor(states.data(), DType::FP32,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor source_slots_tensor(device_source_slots.p, DType::I32, {batch});
    Tensor destination_slots_tensor(device_destination_slots.p, DType::I32, {batch});
    Tensor out_tensor(out.data(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    ops::gated_delta_net_batch_update(q, k, v, g, beta, scale, test_case.normalize_qk,
                                      states_tensor, source_slots_tensor, destination_slots_tensor,
                                      out_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        std::string(test_case.name) + " batch update B=" + std::to_string(batch);
    int failures                         = 0;
    const std::vector<double> got_output = from_device_bf16(out.data(), aggregate.v.size());
    failures += verify_recurrence(label + " out", got_output, expected_output,
                                  gated_delta_net_output_bf16_criterion());

    const std::vector<float> got_states = from_device<float>(states.data(), initial_states.size());
    for (int row = 0; row < batch; ++row) {
        const std::size_t begin =
            static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)]) * state_size;
        failures +=
            verify_recurrence(label + " row " + std::to_string(row) + " final state",
                              doubles(std::vector<float>(got_states.begin() + begin,
                                                         got_states.begin() + begin + state_size)),
                              references[static_cast<std::size_t>(row)].final_state,
                              gated_delta_net_state_fp32_criterion());
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot),
            std::vector<float>(got_states.begin() + begin, got_states.begin() + begin + state_size),
            std::vector<float>(initial_states.begin() + begin,
                               initial_states.begin() + begin + state_size));
    }
    failures +=
        verify_exact(label + " source selectors unchanged",
                     from_device_i32(device_source_slots, source_slots.size()), source_slots);
    failures += verify_exact(label + " destination selectors unchanged",
                             from_device_i32(device_destination_slots, destination_slots.size()),
                             destination_slots);
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, aggregate, device.q, device.k, device.v,
                                               device.g, device.beta);
    return failures;
}

int contract_rejection_cases() {
    DeviceBuffer q_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer k_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer v_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer g_buffer(8 * sizeof(float));
    DeviceBuffer beta_buffer(8 * sizeof(float));
    DeviceBuffer state_buffer(kStateDim * kStateDim * 8 * sizeof(float));
    DeviceBuffer out_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    WorkspaceArena workspace(256);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    auto is_rejected = [&](int activation_dim, int state_dim, int qk_heads, int value_heads) {
        Tensor q(q_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor k(k_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor v(v_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        Tensor g(g_buffer.p, DType::FP32, {value_heads, 1});
        Tensor beta(beta_buffer.p, DType::FP32, {value_heads, 1});
        Tensor state(state_buffer.p, DType::FP32, {state_dim, state_dim, value_heads});
        Tensor out(out_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        try {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state, out, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!is_rejected(64, kStateDim, 4, 8)) {
        std::cerr << "gated_delta_net accepted Q/K/V head dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, 64, 4, 8)) {
        std::cerr << "gated_delta_net accepted state dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, kStateDim, 4, 6)) {
        std::cerr << "gated_delta_net accepted a non-divisible head map\n";
        ++failures;
    }
    return failures;
}

int direct_prefill_capacity_cases() {
    const auto capacity = [](int heads, bool normalize, int first, int last) {
        return ops::gated_delta_net_workspace_capacity_bytes(16, heads, normalize, first, last);
    };
    int failures = 0;
    if (capacity(48, true, 1024, 1024) != 0 ||
        capacity(48, true, 1023, 1023) == 0 || capacity(48, true, 1025, 1025) == 0 ||
        capacity(48, false, 1024, 1024) == 0 || capacity(32, true, 1024, 1024) == 0) {
        std::cerr << "gated_delta_net direct prefill capacity route is not closed to its measured profile\n";
        ++failures;
    }
    const auto interval = capacity(48, true, 1, 1024);
    const auto last_chunked = capacity(48, true, 1023, 1023);
    if (interval != last_chunked || capacity(48, true, 1023, 1024) != last_chunked) {
        std::cerr << "gated_delta_net interval lost the shorter chunk workspace witness\n";
        ++failures;
    }
    for (const bool normalize : {false, true}) {
        const auto enclosing = capacity(48, normalize, 1, 1024);
        for (int tokens = 1; tokens <= 1024; ++tokens) {
            if (capacity(48, normalize, tokens, tokens) > enclosing) {
                std::cerr << "gated_delta_net interval under-reserved at T=" << tokens << '\n';
                ++failures;
                break;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    const int capacity_failures = direct_prefill_capacity_cases();
    if (capacity_failures != 0) return 1;
    if (const char* capacity_only = std::getenv("NINFER_TEST_GDN_CAPACITY_ONLY");
        capacity_only != nullptr && std::string(capacity_only) == "1") {
        std::cout << "OK gated_delta_net CPU interval capacity\n";
        return 0;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    const char* capture_directory = std::getenv("NINFER_TEST_GDN_CAPTURE_DIR");
    if (capture_directory != nullptr && capture_directory[0] != '\0') {
        try {
            // Capture qualification always reports both frozen criteria and
            // measured errors, including successful comparisons.
            if (::setenv("NINFER_OP_REPORT_STATS", "1", 1) != 0) {
                throw std::runtime_error("could not enable capture error statistics");
            }
            const CapturedInputs capture = load_capture(capture_directory);
            const int failures = distinct_state_case(
                {"captured 27b recurrence", 16, 48, 1024, true}, 0, &capture);
            std::cout << (failures == 0 ? "OK" : "FAIL")
                      << " gated_delta_net captured FP64 recurrence correctness\n";
            return failures == 0 ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "gated_delta_net capture: " << error.what() << '\n';
            return 1;
        }
    }

    // Select the same long qualification case alone for bounded cross-device
    // diagnosis. It uses the ordinary independent FP64 oracle and frozen suite
    // criteria; no route, staging cast, or tolerance is changed.
    const char* long_diagnostic = std::getenv("NINFER_TEST_GDN_LONG_DIAGNOSTIC");
    if (long_diagnostic != nullptr && std::string(long_diagnostic) == "1") {
        const int failures = distinct_state_case(kLongCase, kLongSeed);
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " gated_delta_net long recurrence correctness\n";
        return failures == 0 ? 0 : 1;
    }

    int failures = 0;

    for (const bool normalize_qk : {false, true}) {
        const std::size_t interval =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 63, 65);
        const std::size_t witness =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 65, 65);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the chunk boundary\n";
            ++failures;
        }
    }
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 0, 65);
        std::cerr << "gated_delta_net accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(4, 6, true, 1, 65);
        std::cerr << "gated_delta_net workspace accepted a non-divisible head map\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += contract_rejection_cases();

    // Registered 27B/35B-A3B geometries, public state forms, and the recurrent/chunk/tail route
    // boundary are all qualified directly against the same complete FP64 recurrence.
    failures += inplace_case({"27b decode fused-qk-norm", 16, 48, 1, true}, 12001u);
    failures += distinct_state_case({"27b raw-qk small-T", 16, 48, 7, false}, 12007u);
    failures += distinct_state_case({"35b pre-chunk fused-qk-norm", 16, 32, 63, true}, 12063u);
    failures += distinct_state_case({"27b exact chunk fused-qk-norm", 16, 48, 64, true}, 12064u);
    failures += distinct_state_case({"27b exact chunk raw-qk", 16, 48, 64, false}, 12164u);
    failures += inplace_case({"35b chunk-tail fused-qk-norm", 16, 32, 65, true}, 12065u);
    failures += distinct_state_case({"generic grouped-map chunk-tail", 3, 12, 65, true}, 12365u);
    failures += distinct_state_case({"27b two-chunk fused-qk-norm", 16, 48, 128, true}, 12128u);
    failures += inplace_case({"35b two-chunk raw-qk", 16, 32, 128, false}, 12228u);
    failures += distinct_state_case(kLongCase, kLongSeed);

    // The production decode path updates selected state-pool slots in place at width one.
    failures += batch_update_case({"27b selected-slot fused-qk-norm", 16, 48, 1, true}, {7}, {7}, 8,
                                  12101u);
    failures += batch_update_case({"35b selected-slot near-zero", 16, 32, 1, true, true}, {6}, {6},
                                  8, 12201u);
    failures += batch_update_case({"35b ordinary", 16, 32, 1, true}, {8, 9, 10, 11, 12, 13, 14, 15},
                                  {8, 9, 10, 11, 12, 13, 14, 15}, 16, 13001u);
    failures += batch_update_case({"35b mixed fork destinations", 16, 32, 1, true}, {0, 2, 4, 6},
                                  {1, 3, 5, 7}, 8, 13101u);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net correctness\n";
    return failures == 0 ? 0 : 1;
}
