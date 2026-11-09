// Private diagnostic: same-GPU physical partition versus Single, then identical-input tail replay.
// This is not a public parallel qualification or a performance benchmark.
#include "ninfer/engine.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#include "targets/qwen3_6_27b/impl/load/pipeline_weights.h"
#include "core/cuda_error_scope.h"
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/l2norm.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include <type_traits>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace {
using namespace ninfer;
namespace q36 = ninfer::targets::qwen3_6;
namespace q27 = ninfer::targets::qwen3_6_27b;
namespace detail27 = q27::detail;
namespace rt = q36::detail::qwen3_6_27b_runtime;
using Config = detail27::TextConfig;
constexpr PhysicalLayerRange ranges[]{{0, 36}, {36, 64}};
constexpr unsigned chunk = 1024;

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
EngineOptions options(const char* path) {
    EngineOptions result;
    result.artifact_path = path;
    result.purpose = EnginePurpose::CausalScoring;
    result.max_context = 2048;
    result.kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    result.max_concurrency = 1;
    result.prefill_chunk = chunk;
    result.use_cuda_graph = false;
    result.context_cache.enabled = false;
    return result;
}
template <typename T> std::vector<T> download(const Tensor& tensor) {
    check(tensor.is_contiguous(), "diagnostic download must be contiguous");
    std::vector<T> result(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return result;
}
void save_bf16(const char* name, const std::vector<std::uint16_t>& values) {
    const char* directory = std::getenv("NINFER_TEST_PIPELINE_REPORT_DIR");
    if (!directory || !*directory) return;
    std::filesystem::create_directories(directory);
    const auto path = std::filesystem::path(directory) / name;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(values[0]));
    check(static_cast<bool>(file), "cannot write BF16 diagnostic dump");
    std::cout << "dump=" << path.string() << " shape=[" << Config::hidden << ",1024] dtype=BF16\n";
}

q27::Package::SequencePlan make_plan(DeviceContext& device, EngineOptions options,
                                    q27::Package::WeightsProfile profile, PhysicalLayerRange range) {
    options.device = device.device;
    auto planner = q27::Package::make_sequence_planner(device, options, profile, range);
    const auto pages = planner.capacity_curve().minimum_main_page_groups;
    return std::move(planner).finalize(pages);
}
struct Stage {
    DeviceContext& device;
    PhysicalLayerRange range;
    detail27::LoadedModelData model;
    q27::Package::SequencePlan plan;
    DeviceBuffer persistent;
    WorkspaceArena work;
    q36::DecoderState decoder;
    q36::StateImageDevicePool images;
    q36::RoundState io;
    Tensor hidden;
    std::vector<DeviceKVPageLease> pages;
    KVExecutionRowLease row;
    DeviceBuffer metadata{(chunk + 1) * sizeof(std::int32_t)};

    Stage(DeviceContext& device, PhysicalLayerRange range, detail27::BindingPlan bindings,
          artifact::MaterializedArtifact backing, EngineOptions options,
          q27::Package::WeightsProfile profile)
        : device(device), range(range),
          model(std::move(bindings), std::move(backing), detail27::PipelineStage{range, range.begin == 0}),
          plan(make_plan(device, options, profile, range)), persistent(plan.impl_->persistent.bytes),
          work(plan.workspace_capacity_bytes()),
          decoder({persistent.p, persistent.bytes}, plan.impl_->persistent.decoder),
          images({persistent.p, persistent.bytes}, plan.impl_->persistent.state_images),
          io(range.begin == 0 ? q36::RoundState({persistent.p, persistent.bytes}, plan.impl_->persistent.round)
                             : q36::RoundState()),
          hidden(range.begin == 0 ? plan.impl_->persistent.prefill_hidden.bind({persistent.p, persistent.bytes})
                                 : Tensor{}) {
        auto& cache = decoder.text_kv;
        row = cache.execution_tables().acquire(0);
        const auto count = (1099U + cache.page_pool().geometry().page_tokens - 1) /
                           cache.page_pool().geometry().page_tokens;
        auto reservation = cache.page_pool().reserve(count);
        check(reservation.has_value(), "diagnostic KV reservation failed");
        pages.reserve(count);
        cache.page_pool().materialize(*reservation, count, pages);
        cache.execution_tables().publish(row.handle(), 0, pages, device.stream);
        images.zero_slot(0, device.stream);
        device.synchronize();
        std::cout << "stage device=" << device.device << " layers=[" << range.begin << ',' << range.end
                  << ") weights=" << model.backing.stats().device_capacity_bytes
                  << " persistent=" << persistent.bytes << " workspace=" << plan.workspace_capacity_bytes()
                  << " kv_pages=" << count << " row=" << row.row_index() << " state_slot=0\n";
    }
    ~Stage() {
        // Preserve the initiating exception if a CUDA operation failed.
        (void)cudaStreamSynchronize(device.stream);
    }
    std::unique_ptr<rt::schedule::TextContext> context(unsigned begin) {
        auto result = std::make_unique<rt::schedule::TextContext>(device, model.runtime, work,
            decoder.text_kv.execution_view(row), images.linear(), io, hidden, chunk, begin,
            q36::PagedKVCacheView{}, &decoder.text_kv, nullptr, range);
        result->set_linear_state_slots(0, 0);
        return result;
    }
    void prefill(Tensor& residual, unsigned begin, unsigned count) {
        Tensor positions(metadata.p, DType::I32, {static_cast<int>(count)});
        Tensor row_tensor(static_cast<std::int32_t*>(metadata.p) + chunk, DType::I32, {1});
        ops::fill_i32_positions(positions, static_cast<std::int32_t>(begin), device.stream);
        ops::set_i32_scalar(row_tensor, row.row_index(), device.stream);
        auto text = context(begin);
        text->pipeline_prefill_stage(residual, positions, positions, row_tensor, {begin + count, begin + count});
        device.synchronize();
    }
};

struct RecurrenceCapture {
    std::vector<std::uint16_t> normalized_q, normalized_k, w, u, v_new, h_chunk, output;
    std::vector<float> g_cumsum, state;
};
struct Layer36Capture {
    RecurrenceCapture internal;
    std::vector<std::uint16_t> h, projected_qkv, z, q, k, v, output;
    std::vector<float> g, beta, state;
};
struct ProbeTensor {
    Tensor tensor;
    DeviceBuffer storage;
    ProbeTensor(DType type, std::initializer_list<std::int32_t> shape)
        : tensor(nullptr, type, shape), storage(tensor.bytes()) { tensor.data = storage.p; }
};
template <class T>
void upload_probe(const Tensor& tensor, const std::vector<T>& data, cudaStream_t stream) {
    check(tensor.bytes() == data.size() * sizeof(T), "probe upload extent");
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, data.data(), tensor.bytes(), cudaMemcpyHostToDevice, stream));
}
template <class T>
void compare_boundary(const char* label, const char* shape,
                      const std::vector<T>& expected, const std::vector<T>& actual) {
    check(expected.size() == actual.size(), "boundary capture extent");
    std::size_t differences = 0, first = actual.size(), worst = 0;
    double maximum = 0, squares = 0;
    const auto value = [](T bits) -> double {
        if constexpr (std::is_same_v<T, std::uint16_t>)
            return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
        else return bits;
    };
    for (std::size_t i = 0; i < actual.size(); ++i) {
        bool equal;
        if constexpr (std::is_same_v<T, float>)
            equal = std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]);
        else equal = actual[i] == expected[i];
        if (!equal) { ++differences; first = std::min(first, i); }
        const double a = value(actual[i]), b = value(expected[i]);
        check(std::isfinite(a) && std::isfinite(b), "nonfinite layer36 boundary");
        const double error = std::abs(a - b);
        if (error > maximum) { maximum = error; worst = i; }
        squares += error * error;
    }
    std::cout << "layer36 boundary=" << label << " dtype="
              << (std::is_same_v<T, float> ? "FP32" : "BF16") << " shape=" << shape
              << " bit_differences=" << differences << '/' << actual.size()
              << " first_index=" << (first == actual.size() ? -1LL : static_cast<long long>(first))
              << " max_abs=" << maximum << " worst_index=" << worst
              << " rms=" << std::sqrt(squares / actual.size());
    if (first < actual.size()) std::cout << " first_reference=" << value(expected[first]) << " first_actual=" << value(actual[first]);
    std::cout << '\n' << std::flush;
}
RecurrenceCapture probe_recurrence_stages(DeviceContext& device, const Tensor& q, const Tensor& k,
    const Tensor& v, const Tensor& g, const Tensor& beta, const std::vector<std::uint16_t>& public_output,
    const std::vector<float>& public_state, const RecurrenceCapture* reference) {
    namespace kernels = ops::detail::gated_delta_net::chunked;
    ProbeTensor normalized_q(DType::BF16, {128, 16, chunk});
    ProbeTensor normalized_k(DType::BF16, {128, 16, chunk});
    ProbeTensor state(DType::FP32, {128, 128, 48});
    ProbeTensor output(DType::BF16, {128, 48, chunk});
    const auto layout = kernels::compute_workspace_layout(48, chunk);
    DeviceBuffer scratch(layout.total_bytes);
    const DeviceSpan backing{scratch.p, scratch.bytes};
    Tensor cumulative = layout.g_cumsum.bind(backing);
    Tensor w = layout.W.bind(backing), u = layout.U.bind(backing);
    Tensor v_new = layout.v_new.bind(backing), h_chunk = layout.h_chunk.bind(backing);
    Tensor qr = q.view({128, 16, chunk}), kr = k.view({128, 16, chunk});
    CUDA_CHECK(cudaMemsetAsync(state.tensor.data, 0, state.tensor.bytes(), device.stream));
    ops::l2norm(qr, 1.0e-6F, normalized_q.tensor, device.stream);
    ops::l2norm(kr, 1.0e-6F, normalized_k.tensor, device.stream);
    device.synchronize();
    RecurrenceCapture result;
    result.normalized_q = download<std::uint16_t>(normalized_q.tensor);
    result.normalized_k = download<std::uint16_t>(normalized_k.tensor);
    if (reference) {
        compare_boundary("internal_normalized_Q", "[128,16,1024]", reference->normalized_q, result.normalized_q);
        compare_boundary("internal_normalized_K", "[128,16,1024]", reference->normalized_k, result.normalized_k);
    }
    kernels::prepare_wy_wu_config prepare{};
    prepare.H_qk = 16; prepare.H_v = 48; prepare.L = chunk;
    prepare.k = static_cast<const __nv_bfloat16*>(normalized_k.tensor.data);
    prepare.v = static_cast<const __nv_bfloat16*>(v.data);
    prepare.g_in = static_cast<const float*>(g.data);
    prepare.beta = static_cast<const float*>(beta.data);
    prepare.W = static_cast<__nv_bfloat16*>(w.data);
    prepare.U = static_cast<__nv_bfloat16*>(u.data);
    prepare.g_cumsum_out = static_cast<float*>(cumulative.data);
    prepare.stream = device.stream;
    CUDA_CHECK(kernels::launch_prepare_wy_wu(prepare));
    device.synchronize();
    result.g_cumsum = download<float>(cumulative);
    result.w = download<std::uint16_t>(w);
    result.u = download<std::uint16_t>(u);
    if (reference) {
        compare_boundary("internal_g_cumsum", "[48,1024]", reference->g_cumsum, result.g_cumsum);
        compare_boundary("internal_W", "[128,48,1024]", reference->w, result.w);
        compare_boundary("internal_U", "[128,48,1024]", reference->u, result.u);
    }
    kernels::state_passing_config passing{};
    passing.H_qk = 16; passing.H_v = 48; passing.L = chunk;
    passing.W = static_cast<const __nv_bfloat16*>(w.data);
    passing.U = static_cast<const __nv_bfloat16*>(u.data);
    passing.k = static_cast<const __nv_bfloat16*>(normalized_k.tensor.data);
    passing.g_cumsum = static_cast<const float*>(cumulative.data);
    passing.state_in = static_cast<const float*>(state.tensor.data);
    passing.v_new = static_cast<__nv_bfloat16*>(v_new.data);
    passing.h_chunk = static_cast<__nv_bfloat16*>(h_chunk.data);
    passing.state_out = static_cast<float*>(state.tensor.data);
    passing.stream = device.stream;
    CUDA_CHECK(kernels::launch_state_passing(passing));
    device.synchronize();
    result.v_new = download<std::uint16_t>(v_new);
    result.h_chunk = download<std::uint16_t>(h_chunk);
    result.state = download<float>(state.tensor);
    if (reference) {
        compare_boundary("internal_v_new", "[128,48,1024]", reference->v_new, result.v_new);
        compare_boundary("internal_h_chunk", "[128,128,48,16]", reference->h_chunk, result.h_chunk);
        compare_boundary("internal_state_out", "[128,128,48]", reference->state, result.state);
    }
    kernels::chunk_output_config final{};
    final.H_qk = 16; final.H_v = 48; final.L = chunk;
    final.q = static_cast<const __nv_bfloat16*>(normalized_q.tensor.data);
    final.k = static_cast<const __nv_bfloat16*>(normalized_k.tensor.data);
    final.v_new = static_cast<const __nv_bfloat16*>(v_new.data);
    final.g_cumsum = static_cast<const float*>(cumulative.data);
    final.h_chunk = static_cast<const __nv_bfloat16*>(h_chunk.data);
    final.attn_out = static_cast<__nv_bfloat16*>(output.tensor.data);
    final.scale = detail27::Variant::gdn_scale;
    final.stream = device.stream;
    CUDA_CHECK(kernels::launch_output(final));
    device.synchronize();
    result.output = download<std::uint16_t>(output.tensor);
    if (reference)
        compare_boundary("internal_output", "[128,48,1024]", reference->output, result.output);
    std::cout << "internal_probe_device=" << device.device << '\n';
    compare_boundary("internal_vs_public_output", "[128,48,1024]", public_output, result.output);
    compare_boundary("internal_vs_public_state", "[128,128,48]", public_state, result.state);
    check(public_output == result.output && public_state == result.state,
          "internal recurrence launch probe did not reproduce public Op");
    return result;
}

template <class T>
void save_layer36_tensor(const std::filesystem::path& directory, const char* name,
                         const std::vector<T>& values) {
    const auto path = directory / name;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    check(static_cast<bool>(file), "cannot write layer36 tensor capture");
}
void save_layer36(const Layer36Capture& capture) {
    const char* path = std::getenv("NINFER_TEST_PIPELINE_REPORT_DIR");
    if (!path || !*path) return;
    const auto directory = std::filesystem::path(path) / "layer36-recurrence";
    std::filesystem::create_directories(directory);
    save_layer36_tensor(directory, "q.bf16", capture.q);
    save_layer36_tensor(directory, "k.bf16", capture.k);
    save_layer36_tensor(directory, "v.bf16", capture.v);
    save_layer36_tensor(directory, "g.fp32", capture.g);
    save_layer36_tensor(directory, "beta.fp32", capture.beta);
    save_layer36_tensor(directory, "sm89-output.bf16", capture.output);
    save_layer36_tensor(directory, "sm89-state.fp32", capture.state);
    std::ofstream manifest(directory / "manifest.json");
    manifest << std::setprecision(9)
             << "{\"layer\":36,\"tokens\":1024,\"qk_heads\":16,\"value_heads\":48,\"head_dim\":128,"
                "\"normalize_qk\":true,\"initial_state\":\"zero\",\"scale\":" << detail27::Variant::gdn_scale
             << ",\"layout\":\"contiguous fastest dimension first\",\"byte_order\":\"little-endian\","
                "\"q_shape\":[128,16,1024],\"k_shape\":[128,16,1024],\"v_shape\":[128,48,1024],"
                "\"g_shape\":[48,1024],\"beta_shape\":[48,1024],\"output_shape\":[128,48,1024],"
                "\"state_shape\":[128,128,48]}\n";
    check(static_cast<bool>(manifest), "cannot write layer36 capture manifest");
    std::cout << "layer36 recurrence captures=" << directory.string() << '\n' << std::flush;
}

void save_control_layer36(Stage& stage, const Layer36Capture& capture) {
    const char* path = std::getenv("NINFER_TEST_PIPELINE_REPORT_DIR");
    if (!path || !*path) return;
    auto directory = std::filesystem::path(path) / "control-layer36";
    // Root is the fixed SM89 input set for both-device standalone probes. Preserve
    // native SM120 control outputs separately instead of overwriting that input set.
    if (stage.device.device != 0) directory /= "sm120";
    std::filesystem::create_directories(directory);
    const auto& projection = stage.model.runtime.gdn_layers[Config::gdn_index(36)].projection;
    const Weight* a = nullptr;
    const Weight* b = nullptr;
    unsigned b_row_offset = 0;
    if (const auto* split = std::get_if<detail27::SplitGdnControlProjectionPayload>(&projection.control_projection)) {
        a = &split->a_projection;
        b = &split->b_projection;
        check(a->n == 48 && b->n == 48, "control capture split weight rows");
    } else {
        const auto& parent = std::get<detail27::FusedGdnControlProjectionPayload>(projection.control_projection).a_b_projection;
        a = b = &parent;
        b_row_offset = 48;
        check(parent.n == 96, "control capture fused weight rows");
    }
    const auto read_weight = [](const Weight& weight, unsigned first_row) {
        check(weight.qtype == QType::BF16_CTRL && weight.layout == QuantLayout::Contiguous &&
              weight.k == 5120 && weight.qdata != nullptr &&
              weight.payload_bytes >= static_cast<std::size_t>(first_row + 48) * 5120 * 2,
              "control capture requires represented contiguous BF16 weights");
        std::vector<std::uint16_t> values(48 * 5120);
        const auto* source = static_cast<const std::uint16_t*>(weight.qdata) + first_row * 5120;
        CUDA_CHECK(cudaMemcpy(values.data(), source, values.size() * sizeof(values[0]), cudaMemcpyDeviceToHost));
        return values;
    };
    save_layer36_tensor(directory, "h.bf16", capture.h);
    save_layer36_tensor(directory, "a.bf16", read_weight(*a, 0));
    save_layer36_tensor(directory, "b.bf16", read_weight(*b, b_row_offset));
    save_layer36_tensor(directory, "a_log.fp32", download<float>(projection.a_log));
    save_layer36_tensor(directory, "dt_bias.fp32", download<float>(projection.dt_bias));
    save_layer36_tensor(directory, "g.fp32", capture.g);
    save_layer36_tensor(directory, "beta.fp32", capture.beta);
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\"layer\":36,\"tokens\":1024,\"hidden\":5120,\"heads\":48,\"device\":" << stage.device.device
             << ",\"byte_order\":\"little-endian\",\"h_shape\":[5120,1024],\"h_dtype\":\"BF16\","
                "\"h_layout\":\"fastest dimension first\",\"a_shape\":[48,5120],\"b_shape\":[48,5120],"
                "\"a_dtype\":\"BF16\",\"b_dtype\":\"BF16\",\"weight_layout\":\"row-major; K contiguous\","
                "\"a_log_shape\":[48],\"dt_bias_shape\":[48],\"a_log_dtype\":\"FP32\",\"dt_bias_dtype\":\"FP32\","
                "\"g_shape\":[48,1024],\"beta_shape\":[48,1024],\"g_dtype\":\"FP32\",\"beta_dtype\":\"FP32\","
                "\"output_layout\":\"fastest dimension first\"}\n";
    check(static_cast<bool>(manifest), "cannot write control capture manifest");
    std::cout << "layer36 control captures=" << directory.string() << '\n' << std::flush;
}

Layer36Capture probe_layer36(Stage& stage, const std::vector<std::uint16_t>& input,
                             const Layer36Capture* reference = nullptr) {
    auto& device = stage.device;
    const auto& weights = stage.model.runtime.gdn_layers[Config::gdn_index(36)];
    ProbeTensor x(DType::BF16, {Config::hidden, chunk});
    ProbeTensor h(DType::BF16, {Config::hidden, chunk});
    ProbeTensor g(DType::FP32, {Config::gdn_value_heads, chunk});
    ProbeTensor beta(DType::FP32, {Config::gdn_value_heads, chunk});
    ProbeTensor qkv(DType::BF16, {Config::convolution_dim, chunk});
    ProbeTensor z(DType::BF16, {Config::value_dim, chunk});
    ProbeTensor q(DType::BF16, {Config::key_dim, chunk});
    ProbeTensor k(DType::BF16, {Config::key_dim, chunk});
    ProbeTensor v(DType::BF16, {Config::value_dim, chunk});
    ProbeTensor conv(DType::BF16, {Config::convolution_dim, Config::gdn_conv_state_width});
    ProbeTensor state(DType::FP32, {Config::gdn_key_head_dim, Config::gdn_value_head_dim, Config::gdn_value_heads});
    ProbeTensor output(DType::BF16, {Config::gdn_value_head_dim, Config::gdn_value_heads, chunk});
    upload_probe(x.tensor, input, device.stream);
    CUDA_CHECK(cudaMemsetAsync(conv.tensor.data, 0, conv.tensor.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(state.tensor.data, 0, state.tensor.bytes(), device.stream));
    stage.work.reset();
    detail27::Variant::gdn_norm_control_projection(x.tensor, weights.input_norm, Config::rms_epsilon,
        weights.projection, h.tensor, g.tensor, beta.tensor, stage.work, device.execution_view());
    detail27::Variant::gdn_input_projection(h.tensor, weights.projection, qkv.tensor, z.tensor,
        q36::TextPhase::Prefill, stage.work, device.stream);
    ops::causal_conv1d_silu_split(qkv.tensor, weights.convolution, conv.tensor, conv.tensor,
        q.tensor, k.tensor, v.tensor, device.stream);
    device.synchronize();
    Layer36Capture result;
    result.h = download<std::uint16_t>(h.tensor);
    result.g = download<float>(g.tensor);
    result.beta = download<float>(beta.tensor);
    result.projected_qkv = download<std::uint16_t>(qkv.tensor);
    result.z = download<std::uint16_t>(z.tensor);
    result.q = download<std::uint16_t>(q.tensor);
    result.k = download<std::uint16_t>(k.tensor);
    result.v = download<std::uint16_t>(v.tensor);
    const auto recurrent = [&] {
        Tensor qr = q.tensor.view({Config::gdn_key_head_dim, Config::gdn_key_heads, chunk});
        Tensor kr = k.tensor.view({Config::gdn_key_head_dim, Config::gdn_key_heads, chunk});
        Tensor vr = v.tensor.view({Config::gdn_value_head_dim, Config::gdn_value_heads, chunk});
        ops::gated_delta_net(qr, kr, vr, g.tensor, beta.tensor, detail27::Variant::gdn_scale,
                            true, stage.work, state.tensor, output.tensor, device.stream);
        device.synchronize();
    };
    recurrent();
    result.output = download<std::uint16_t>(output.tensor);
    result.state = download<float>(state.tensor);
    save_control_layer36(stage, result);
    const bool public_chunked = ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, chunk, chunk) != 0;
    if (!reference) {
        save_layer36(result);
        if (public_chunked) {
            result.internal = probe_recurrence_stages(device, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
                result.output, result.state, nullptr);
        } else {
            std::cout << "public direct1024 route: skipping obsolete chunked reproduction probe\n";
        }
    }
    if (reference) {
        compare_boundary("normalized_h", "[5120,1024]", reference->h, result.h);
        compare_boundary("projected_QKV", "[10240,1024]", reference->projected_qkv, result.projected_qkv);
        compare_boundary("projected_Z", "[6144,1024]", reference->z, result.z);
        compare_boundary("convolved_Q", "[2048,1024]", reference->q, result.q);
        compare_boundary("convolved_K", "[2048,1024]", reference->k, result.k);
        compare_boundary("convolved_V", "[6144,1024]", reference->v, result.v);
        compare_boundary("g", "[48,1024]", reference->g, result.g);
        compare_boundary("beta", "[48,1024]", reference->beta, result.beta);
        compare_boundary("native_recurrent_output", "[128,48,1024]", reference->output, result.output);
        compare_boundary("native_recurrent_state", "[128,128,48]", reference->state, result.state);
        // Isolate recurrence: overwrite EVERY input with captured GPU0 representations and reset S.
        upload_probe(q.tensor, reference->q, device.stream);
        upload_probe(k.tensor, reference->k, device.stream);
        upload_probe(v.tensor, reference->v, device.stream);
        upload_probe(g.tensor, reference->g, device.stream);
        upload_probe(beta.tensor, reference->beta, device.stream);
        CUDA_CHECK(cudaMemsetAsync(state.tensor.data, 0, state.tensor.bytes(), device.stream));
        recurrent();
        const auto identical_output = download<std::uint16_t>(output.tensor);
        const auto identical_state = download<float>(state.tensor);
        compare_boundary("identical_input_recurrent_output", "[128,48,1024]", reference->output, identical_output);
        compare_boundary("identical_input_recurrent_state", "[128,128,48]", reference->state, identical_state);
        if (public_chunked) {
            (void)probe_recurrence_stages(device, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
                identical_output, identical_state, &reference->internal);
        } else {
            std::cout << "public direct1024 route: identical-input direct replay reported above\n";
        }
    }
    stage.work.reset();
    return result;
}

struct GdnSnapshot {
    std::vector<std::vector<float>> recurrent;
    std::vector<std::vector<std::uint16_t>> convolution;
};
GdnSnapshot snapshot_gdn(Stage& stage) {
    stage.device.synchronize();
    GdnSnapshot result;
    for (unsigned layer = 0; layer < stage.images.linear().layer_count(); ++layer) {
        result.recurrent.push_back(download<float>(stage.images.linear().recurrent_slot(layer, 0)));
        result.convolution.push_back(download<std::uint16_t>(stage.images.linear().conv_slot(layer, 0)));
    }
    return result;
}
void compare_gdn(const GdnSnapshot& reference, const GdnSnapshot& actual) {
    check(reference.recurrent.size() == actual.recurrent.size(), "GDN snapshot layer count");
    unsigned local = 0;
    for (unsigned global = 36; global < 64; ++global) {
        if (Config::is_full_attention(global)) continue;
        std::size_t recurrent_differences = 0, conv_differences = 0;
        double maximum = 0;
        for (std::size_t i = 0; i < reference.recurrent[local].size(); ++i) {
            const float a = actual.recurrent[local][i], b = reference.recurrent[local][i];
            check(std::isfinite(a) && std::isfinite(b), "nonfinite GDN replay state");
            recurrent_differences += std::bit_cast<std::uint32_t>(a) != std::bit_cast<std::uint32_t>(b);
            maximum = std::max(maximum, std::abs(static_cast<double>(a) - b));
        }
        for (std::size_t i = 0; i < reference.convolution[local].size(); ++i)
            conv_differences += actual.convolution[local][i] != reference.convolution[local][i];
        std::cout << "tail_state global_layer=" << global << " local_gdn=" << local
                  << " recurrent_bit_differences=" << recurrent_differences
                  << " recurrent_max_abs=" << maximum << " convolution_bit_differences=" << conv_differences << '\n';
        ++local;
    }
}

float report_scores(const std::vector<TokenId>& tokens, const std::vector<float>& reference,
                    const std::vector<float>& actual) {
    check(reference.size() == actual.size() && actual.size() == 1099, "diagnostic score shape");
    std::size_t worst = 0;
    double sum = 0, squares = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]), "nonfinite same-GPU partition score");
        const auto error = std::abs(actual[i] - reference[i]);
        if (error > std::abs(actual[worst] - reference[worst])) worst = i;
        sum += error;
        squares += static_cast<double>(error) * error;
    }
    const auto maximum = std::abs(actual[worst] - reference[worst]);
    std::cout << std::setprecision(9) << "same_gpu_partition max_error=" << maximum
              << " target=" << worst + 1 << " token=" << tokens[worst + 1]
              << " single=" << reference[worst] << " partition=" << actual[worst]
              << " mean_abs=" << sum / actual.size() << " rms=" << std::sqrt(squares / actual.size()) << '\n';
    if (const char* directory = std::getenv("NINFER_TEST_PIPELINE_REPORT_DIR"); directory && *directory) {
        std::filesystem::create_directories(directory);
        std::ofstream file(std::filesystem::path(directory) / "pipeline-partition-score-pairs.jsonl");
        file << std::setprecision(9);
        for (std::size_t i = 0; i < actual.size(); ++i)
            file << "{\"target_position\":" << i + 1 << ",\"token\":" << tokens[i + 1]
                 << ",\"single\":" << reference[i] << ",\"partition\":" << actual[i]
                 << ",\"difference\":" << actual[i] - reference[i] << "}\n";
        check(static_cast<bool>(file), "cannot write partition score report");
    }
    std::cout << std::flush;
    return maximum;
}

std::vector<float> run_partition(artifact::Reader& reader, const detail27::ArtifactLoadPlan& full,
                                const std::array<artifact::MaterializationPlan, 2>& materialization,
                                EngineOptions options, q27::Package::WeightsProfile profile,
                                const std::vector<TokenId>& tokens,
                                std::vector<std::uint16_t>& tail_input,
                                std::vector<std::uint16_t>& tail_output, GdnSnapshot& tail_state, Layer36Capture& layer36) {
    DeviceContext device(0);
    device.bind_to_current_thread();
    Stage owner(device, ranges[0], full.bindings, artifact::materialize(reader, materialization[0], device), options, profile);
    Stage tail(device, ranges[1], full.bindings, artifact::materialize(reader, materialization[1], device), options, profile);
    DeviceBuffer residual_storage(static_cast<std::size_t>(Config::hidden) * chunk * 2);
    DeviceBuffer ids_storage(chunk * sizeof(TokenId));
    std::vector<float> output;
    // 1024 predictor columns followed by 75: identical to public scoring's 1024-column head tiles.
    check(rt::kCausalScoreTile == chunk, "diagnostic must preserve the public score tile");
    for (unsigned begin = 0; begin < tokens.size() - 1;) {
        const unsigned count = std::min<unsigned>(chunk, tokens.size() - 1 - begin);
        Tensor residual(residual_storage.p, DType::BF16, {Config::hidden, static_cast<int>(count)});
        Tensor ids(ids_storage.p, DType::I32, {static_cast<int>(count)});
        CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + begin, ids.bytes(), cudaMemcpyHostToDevice, device.stream));
        auto owner_text = owner.context(begin);
        owner_text->pipeline_embedding(ids, residual);
        owner.prefill(residual, begin, count);
        if (begin == 0) {
            tail_input = download<std::uint16_t>(residual);
            layer36 = probe_layer36(tail, tail_input);
        }
        tail.prefill(residual, begin, count);
        if (begin == 0) {
            tail_output = download<std::uint16_t>(residual);
            tail_state = snapshot_gdn(tail);
            compare_boundary("sm89_probe_vs_stage_state", "[128,128,48]", layer36.state, tail_state.recurrent[0]);
            check(layer36.state == tail_state.recurrent[0], "SM89 probe did not reproduce actual stage recurrence");
        }
        Tensor hidden = owner.hidden.slice(1, 0, static_cast<int>(count));
        owner_text->pipeline_final_norm(residual, hidden);
        owner.work.reset();
        Tensor logits = owner.work.alloc(DType::BF16, {Config::output_rows, static_cast<int>(count)});
        Tensor target_ids = owner.work.alloc(DType::I32, {static_cast<int>(count)});
        Tensor logprobs = owner.work.alloc(DType::FP32, {static_cast<int>(count)});
        ops::linear(hidden, owner.model.runtime.output_head, logits, device.stream);
        CUDA_CHECK(cudaMemcpyAsync(target_ids.data, tokens.data() + begin + 1, target_ids.bytes(), cudaMemcpyHostToDevice, device.stream));
        ops::target_logprobs(logits, target_ids, Config::token_domain, logprobs, device.stream);
        device.synchronize();
        auto scores = download<float>(logprobs);
        output.insert(output.end(), scores.begin(), scores.end());
        owner.work.reset();
        begin += count;
    }
    return output;
}

void replay_tail(artifact::Reader& reader, const detail27::ArtifactLoadPlan& full,
                 const artifact::MaterializationPlan& materialization, EngineOptions options,
                 q27::Package::WeightsProfile profile, const std::vector<std::uint16_t>& input,
                 const std::vector<std::uint16_t>& expected, const GdnSnapshot& reference_state, const Layer36Capture& layer36) {
    DeviceContext device(1);
    device.bind_to_current_thread();
    Stage tail(device, ranges[1], full.bindings, artifact::materialize(reader, materialization, device), options, profile);
    const auto native = probe_layer36(tail, input, &layer36);
    DeviceBuffer storage(input.size() * sizeof(input[0]));
    Tensor residual(storage.p, DType::BF16, {Config::hidden, static_cast<int>(chunk)});
    CUDA_CHECK(cudaMemcpy(residual.data, input.data(), residual.bytes(), cudaMemcpyHostToDevice));
    tail.prefill(residual, 0, chunk);
    const auto actual = download<std::uint16_t>(residual);
    save_bf16("pipeline-tail-sm120-output.bf16", actual);
    const auto replay_state = snapshot_gdn(tail);
    compare_gdn(reference_state, replay_state);
    compare_boundary("sm120_probe_vs_stage_state", "[128,128,48]", native.state, replay_state.recurrent[0]);
    check(native.state == replay_state.recurrent[0], "SM120 probe did not reproduce actual stage recurrence");
    std::size_t mismatches = 0, worst = 0;
    double maximum = 0, squares = 0, reference_squares = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i];
        const double a = std::bit_cast<float>(static_cast<std::uint32_t>(actual[i]) << 16);
        const double b = std::bit_cast<float>(static_cast<std::uint32_t>(expected[i]) << 16);
        check(std::isfinite(a) && std::isfinite(b), "nonfinite tail replay residual");
        const double error = std::abs(a - b);
        if (error > maximum) { maximum = error; worst = i; }
        squares += error * error;
        reference_squares += b * b;
    }
    std::cout << "tail_identical_input sm89_vs_sm120 bit_mismatches=" << mismatches
              << '/' << actual.size() << " max_abs=" << maximum << " worst_token=" << worst / Config::hidden
              << " worst_channel=" << worst % Config::hidden << " rms=" << std::sqrt(squares / actual.size())
              << " relative_l2=" << std::sqrt(squares / std::max(reference_squares, 1e-30)) << '\n' << std::flush;
}
} // namespace

int main() {
    const char* enabled = std::getenv("NINFER_TEST_PIPELINE_PARTITION_DIAGNOSTIC");
    const char* path = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    int devices = 0;
    if (!enabled || std::string(enabled) != "1" || !path || !*path ||
        cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::cout << "SKIP: opt-in artifact and CUDA device required\n";
        return 77;
    }
    try {
        ScopedCudaErrors errors;
        cudaDeviceProp owner_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&owner_properties, 0));
        if (!NINFER_COMPILED_SM89 || owner_properties.major * 10 + owner_properties.minor != 89) {
            std::cout << "SKIP: compiled SM89 device0 required\n";
            return 77;
        }
        auto normalized = options(path);
        std::vector<TokenId> tokens;
        std::vector<float> reference;
        {
            Engine single(normalized);
            normalized = single.options();
            std::string text;
            while (tokens.size() < 1100) {
                text += "Each score window begins with fresh state. Pipeline execution preserves causal token probabilities.\n";
                tokens = single.tokenize_text(text);
            }
            tokens.resize(1100);
            reference = single.score_tokens(tokens, 1);
        }
        check(normalized.prefill_chunk == chunk, "normalized scoring chunk changed");
        artifact::Reader reader(path);
        check(reader.identity().weights_id == "groupwise-int", "diagnostic requires groupwise-int artifact");
        const auto profile = q27::Package::resolve_weights(reader.identity());
        artifact::Binder binder(reader);
        auto full = detail27::bind_artifact(binder, profile, {});
        auto materialization = detail27::pipeline_materialization_plans(full, 36);
        std::vector<std::uint16_t> tail_input, tail_output;
        GdnSnapshot tail_state;
        Layer36Capture layer36;
        const auto partition = run_partition(reader, full, materialization, normalized, profile, tokens, tail_input, tail_output, tail_state, layer36);
        const float error = report_scores(tokens, reference, partition);
        save_bf16("pipeline-tail-sm89-input.bf16", tail_input);
        save_bf16("pipeline-tail-sm89-output.bf16", tail_output);
        // GPU0 models and both stage stores have been destroyed before GPU1 replay allocation.
        if (devices >= 2 && NINFER_COMPILED_SM120) {
            cudaDeviceProp tail_properties{};
            CUDA_CHECK(cudaGetDeviceProperties(&tail_properties, 1));
            if (tail_properties.major * 10 + tail_properties.minor == 120)
                replay_tail(reader, full, materialization[1], normalized, profile, tail_input, tail_output, tail_state, layer36);
            else std::cout << "SKIP tail replay: device1 is not SM120\n";
        } else std::cout << "SKIP tail replay: compiled SM120 device1 unavailable\n";
        check(error < 0.25F, "same-GPU partition score differs from Single beyond unchanged tolerance");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
