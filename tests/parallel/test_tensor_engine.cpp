// Opt-in public Engine qualification of eager TP2 (rank 0 SM89, rank 1 SM120) against Single.
// Single uses the groupwise-int artifact and TP2 its offline groupwise-int-tp2 shards; engines run
// sequentially to avoid duplicate weights. Same aggregate contract as the PP Engine test.
#include "ninfer/engine.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#if defined(NINFER_TEST_TENSOR_FAULT_WRAP)
namespace {
std::atomic<bool> fail_tail_reset{false};
std::atomic<unsigned> tail_reset_failures{0};
}
extern "C" cudaError_t __real_cudaMemsetAsync(void*, int, std::size_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemsetAsync(void* pointer, int value, std::size_t bytes,
                                           cudaStream_t stream) {
    int device = -1;
    if (fail_tail_reset.load() && cudaGetDevice(&device) == cudaSuccess && device == 1 &&
        fail_tail_reset.exchange(false)) {
        ++tail_reset_failures;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaMemsetAsync(pointer, value, bytes, stream);
}
#endif

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct NumericalChecks {
    bool diagnostic = [] {
        const char* enabled = std::getenv("NINFER_TEST_TENSOR_BEHAVIOR_DIAGNOSTIC");
        return enabled && std::string(enabled) == "1";
    }();
    unsigned failures = 0;
    const char* phase = "scoring";
    void expect(bool condition, const std::string& message) {
        if (condition) return;
        if (!diagnostic) throw std::runtime_error(message);
        ++failures;
        std::cout << "NUMERICAL FAIL: " << message << '\n' << std::flush;
    }
    void passed(const std::string& label) const {
        if (diagnostic) std::cout << "BEHAVIOR PASS: " << label << '\n' << std::flush;
    }
};
// Cross-architecture precision contract (user-approved 2026-11-09): per-Op routes stay bound to their
// frozen independent FP64 budgets; model-level PP-vs-Single agreement is judged in aggregate NLL plus
// greedy generation on margin-certified fixtures. Isolated tokens may differ because BF16/TF32 tensor-core
// accumulation order is implementation-defined across SM89/SM120. A hard cap still rejects gross errors.
struct ScoreAgreement {
    double mean_abs = 0, mean_signed = 0, rms = 0, maximum = 0;
    std::size_t outliers = 0, count = 0;
};
constexpr double kMeanAbsLimit = 0.01;       // nats/token
constexpr double kMeanSignedLimit = 0.005;   // aggregate NLL drift; perplexity ratio within ~0.5%
constexpr double kRmsLimit = 0.05;
constexpr double kOutlierThreshold = 0.25;   // former per-token ceiling, now an outlier counter
constexpr double kOutlierFractionLimit = 0.005;
constexpr double kHardMaximum = 1.0;
ScoreAgreement score_agreement(const std::vector<float>& reference, const std::vector<float>& candidate,
                               std::size_t reference_offset = 0) {
    ScoreAgreement result;
    result.count = candidate.size();
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        check(std::isfinite(candidate[i]), "nonfinite TP2 score");
        const double difference = double(candidate[i]) - double(reference.at(i + reference_offset));
        result.mean_abs += std::abs(difference);
        result.mean_signed += difference;
        result.rms += difference * difference;
        result.maximum = std::max(result.maximum, std::abs(difference));
        if (std::abs(difference) > kOutlierThreshold) ++result.outliers;
    }
    if (result.count) {
        result.mean_abs /= double(result.count);
        result.mean_signed /= double(result.count);
        result.rms = std::sqrt(result.rms / double(result.count));
    }
    return result;
}
// TP2 changes the kernel route of almost every op from layer 0 (local shapes), unlike PP which keeps
// Single's routes. Its reference envelope is Single's own route-change noise, measured on this fixture
// 2026-11-09 as Single A16 (NINFER_A16_PREFILL=1) vs Single A8: mean|d|=0.0347 rms=0.399
// outliers>0.25=1.18% max=10.33 (local-test/parallel/tensor-engine-real-single-a16.log). TP2 must be no
// further from Single A8 than that, and its aggregate NLL must not be worse than Single by > 0.005 nats.
struct AgreementLimits {
    double mean_abs, rms, outlier_fraction, maximum;
    bool one_sided_signed; // reject only aggregate NLL regressions (candidate log-prob lower)
};
constexpr AgreementLimits kSameRouteLimits{kMeanAbsLimit, kRmsLimit, kOutlierFractionLimit, kHardMaximum, false};
constexpr AgreementLimits kRouteChangeEnvelope{0.035, 0.40, 0.012, std::numeric_limits<double>::infinity(), true};
void expect_score_agreement(NumericalChecks& numerical, const ScoreAgreement& a, const std::string& label,
                            const AgreementLimits& limits = kSameRouteLimits) {
    std::cout << label << " agreement: count=" << a.count << " mean_abs=" << a.mean_abs
              << " mean_signed=" << a.mean_signed << " rms=" << a.rms << " max=" << a.maximum
              << " outliers>" << kOutlierThreshold << "=" << a.outliers << '\n' << std::flush;
    numerical.expect(a.mean_abs <= limits.mean_abs, label + ": mean |dNLL| above limit");
    numerical.expect(limits.one_sided_signed ? a.mean_signed >= -kMeanSignedLimit
                                             : std::abs(a.mean_signed) <= kMeanSignedLimit,
                     label + ": aggregate NLL drift above limit");
    numerical.expect(a.rms <= limits.rms, label + ": RMS dNLL above limit");
    numerical.expect(double(a.outliers) <= limits.outlier_fraction * double(a.count),
                     label + ": too many tokens beyond outlier threshold");
    numerical.expect(a.maximum <= limits.maximum, label + ": token dNLL above hard maximum");
}
void report_score_windows(const std::vector<float>& reference, const std::vector<float>& all,
                          const std::vector<float>& suffix, const std::vector<float>& repeated) {
    if (const char* directory = std::getenv("NINFER_TEST_TENSOR_REPORT_DIR"); directory && *directory) {
        std::filesystem::create_directories(directory);
        const auto path = std::filesystem::path(directory) / "tensor-score-windows.jsonl";
        std::ofstream report(path);
        report << std::setprecision(9);
        check(static_cast<bool>(report), "cannot open score window report");
        for (std::size_t i = 0; i < suffix.size(); ++i)
            report << "{\"target_position\":" << i + 1025 << ",\"single\":" << reference.at(i + 1024)
                   << ",\"tensor_full\":" << all.at(i + 1024) << ",\"tensor_suffix\":" << suffix[i]
                   << ",\"tensor_repeated\":" << repeated.at(i) << "}\n";
        check(static_cast<bool>(report), "cannot write score window report");
        std::cout << "score window report=" << path.string() << '\n' << std::flush;
    }
}
void print_score_configuration(const char* label, const ninfer::Engine& engine,
                               std::size_t token_count) {
    const auto& options = engine.options();
    const auto memory = engine.memory_summary();
    std::cout << label << " score configuration: tokens=" << token_count
              << " prefill_chunk=" << options.prefill_chunk
              << " nominal_chunks=" << (token_count + options.prefill_chunk - 1) / options.prefill_chunk
              << " max_context=" << options.max_context << " concurrency=" << options.max_concurrency
              << " kv_capacity=" << memory.kv_capacity
              << " physical_devices=" << memory.physical_device_count << "\n";
    for (unsigned rank = 0; rank < memory.physical_device_count; ++rank) {
        const auto& physical = memory.physical_devices[rank];
        std::cout << label << " rank=" << rank << " device=" << physical.device
                  << " weights_bytes=" << physical.weights.used_bytes
                  << " state_capacity=" << physical.state_capacity_slots
                  << " kv_pages=" << physical.kv_capacity_pages
                  << " runtime_reservation_bytes=" << physical.runtime_reservation_bytes << "\n";
    }
}
void report_score_pairs(const std::vector<ninfer::TokenId>& tokens,
                        const std::vector<float>& reference, const std::vector<float>& candidate) {
    std::size_t worst = 0;
    double absolute_sum = 0, squared_sum = 0, signed_sum = 0;
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        const double difference = static_cast<double>(candidate[i]) - reference[i];
        absolute_sum += std::abs(difference);
        squared_sum += difference * difference;
        signed_sum += difference;
        if (std::abs(difference) > std::abs(static_cast<double>(candidate[worst]) - reference[worst]))
            worst = i;
    }
    std::cout << std::setprecision(9)
              << "score diagnostics: pairs=" << candidate.size() << " max_index=" << worst
              << " target_position=" << worst + 1 << " token=" << tokens[worst + 1]
              << " single=" << reference[worst] << " candidate=" << candidate[worst]
              << " max_absolute_error=" << std::abs(static_cast<double>(candidate[worst]) - reference[worst])
              << " mean_absolute_error=" << absolute_sum / candidate.size()
              << " mean_signed_error=" << signed_sum / candidate.size()
              << " rms_error=" << std::sqrt(squared_sum / candidate.size()) << "\n";
    auto print_pair = [&](std::size_t target) {
        const auto i = target - 1;
        if (i >= candidate.size()) return;
        std::cout << "score pair target_position=" << target << " token=" << tokens[target]
                  << " single=" << reference[i] << " candidate=" << candidate[i]
                  << " difference=" << static_cast<double>(candidate[i]) - reference[i] << "\n";
    };
    for (std::size_t target = 1; target <= 12; ++target) print_pair(target);
    for (std::size_t target = 1022; target <= 1027; ++target) print_pair(target);
    if (const char* directory = std::getenv("NINFER_TEST_TENSOR_REPORT_DIR"); directory && *directory) {
        std::filesystem::create_directories(directory);
        const auto path = std::filesystem::path(directory) / "tensor-score-pairs.jsonl";
        std::ofstream report(path);
        report << std::setprecision(9);
        check(static_cast<bool>(report), "cannot open score diagnostic report");
        for (std::size_t i = 0; i < candidate.size(); ++i)
            report << "{\"index\":" << i << ",\"target_position\":" << i + 1
                   << ",\"token\":" << tokens[i + 1] << ",\"single\":" << reference[i]
                   << ",\"tp2\":" << candidate[i] << ",\"difference\":"
                   << static_cast<double>(candidate[i]) - reference[i] << "}\n";
        check(static_cast<bool>(report), "cannot write score diagnostic report");
        std::cout << "score paired report=" << path.string() << "\n";
    }
    std::cout << std::flush;
}
void check_empty_physical(const ninfer::Engine& engine) {
    const auto memory = engine.memory_summary();
    check(memory.physical_device_count == 2, "TP2 has no per-device inventory");
    for (unsigned rank = 0; rank < 2; ++rank) {
        const auto& device = memory.physical_devices[rank];
        check(device.device == static_cast<int>(rank), "physical device identity");
        check(device.state_occupied_slots == 0 && device.kv_occupied_pages == 0,
              "TP2 cleanup leaked physical StateImage or KV reservations");
        check(device.runtime_reservation_bytes != 0, "missing per-rank startup budget");
    }
}
#if defined(NINFER_TEST_TENSOR_FAULT_WRAP)
template <class Function>
void check_tail_prepare_failure(ninfer::Engine& engine, Function&& execute) {
    const auto previous = tail_reset_failures.load();
    fail_tail_reset = true;
    bool rejected = false;
    try { execute(); } catch (const std::exception&) { rejected = true; }
    fail_tail_reset = false;
    check(rejected, "second-rank reset failure published a result");
    check(tail_reset_failures == previous + 1, "second-rank prepare fault was not reached");
    check(!engine.healthy(), "stateful TP2 failure did not quarantine Engine");
    check_empty_physical(engine);
}
#endif
template <class Predicate>
void wait_until(Predicate&& predicate, const char* message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!predicate()) {
        check(std::chrono::steady_clock::now() < deadline, message);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
bool both_active(const ninfer::MemorySummary& memory) {
    return memory.physical_device_count == 2 &&
        memory.physical_devices[0].state_occupied_slots != 0 &&
        memory.physical_devices[1].state_occupied_slots != 0 &&
        memory.physical_devices[0].kv_occupied_pages != 0 &&
        memory.physical_devices[1].kv_occupied_pages != 0;
}
bool both_empty(const ninfer::Engine& engine) {
    const auto memory = engine.memory_summary();
    return memory.physical_device_count == 2 &&
        memory.physical_devices[0].state_occupied_slots == 0 &&
        memory.physical_devices[1].state_occupied_slots == 0 &&
        memory.physical_devices[0].kv_occupied_pages == 0 &&
        memory.physical_devices[1].kv_occupied_pages == 0;
}
ninfer::EngineOptions options(const char* artifact, bool tensor, bool scoring) {
    ninfer::EngineOptions value;
    value.artifact_path = artifact;
    value.purpose = scoring ? ninfer::EnginePurpose::CausalScoring : ninfer::EnginePurpose::Generation;
    value.max_context = 2048;
    value.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    value.prefill_chunk = 128;
    value.max_concurrency = scoring ? 1 : 8;
    value.use_cuda_graph = false;
    value.context_cache.enabled = false;
    if (tensor) {
        value.placement.mode = ninfer::ParallelMode::Tensor;
        value.placement.devices = {0, 1};
    }
    return value;
}
// Unsupported TP2 features fail before any artifact read or CUDA context, never silently.
void reject_unsupported_startup() {
    for (bool scoring : {true, false}) {
        for (unsigned feature = 0; feature < 7; ++feature) {
            auto invalid = options("/missing-tensor-gate-fixture.ninfer", true, scoring);
            const char* expected = "eager text";
            switch (feature) {
            case 0: invalid.use_cuda_graph = true; break;
            case 1: invalid.context_cache.enabled = true; break;
            case 2: invalid.enable_vision = true; break;
            case 3: invalid.speculative.backend = ninfer::SpeculativeBackend::Mtp; break;
            case 4: invalid.turn_checkpoint_ring = 1; expected = "checkpoint"; break;
            case 5: invalid.kv_cache = ninfer::KvCacheStorage::Int8Group64; expected = "KV storage"; break;
            case 6: invalid.prefill_chunk = 2048; invalid.max_context = 4096;
                    invalid.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
                    expected = "prefill_chunk"; break;
            }
            bool rejected = false;
            try { ninfer::Engine engine(std::move(invalid)); }
            catch (const std::invalid_argument& error) {
                rejected = std::string(error.what()).find(expected) != std::string::npos;
            }
            check(rejected, "TP2 normalized an explicitly unsupported startup feature");
        }
    }
}
// Placement identity gates precede CUDA initialization: the unsharded artifact cannot run TP2.
void reject_unsharded_artifact(const char* single_artifact) {
    bool rejected = false;
    try { ninfer::Engine engine(options(single_artifact, true, true)); }
    catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("groupwise-int-tp2") != std::string::npos;
    }
    check(rejected, "TP2 accepted the unsharded groupwise-int artifact");
}
// Rank-local state is half of Single: no full KV/GDN copy is hidden on either rank.
void check_local_state(const ninfer::MemorySummary& single, const ninfer::MemorySummary& tensor) {
    std::cout << "memory single text_kv=" << single.text_kv_bytes << " gdn=" << single.gdn_state_bytes
              << " weights=" << single.physical_devices[0].weights.used_bytes
              << " | TP2 rank0 text_kv=" << tensor.text_kv_bytes << " gdn=" << tensor.gdn_state_bytes;
    for (unsigned rank = 0; rank < tensor.physical_device_count; ++rank) {
        const auto& physical = tensor.physical_devices[rank];
        std::cout << " rank" << rank << "{weights=" << physical.weights.used_bytes
                  << " sequence=" << physical.sequence.capacity_bytes
                  << " workspace=" << physical.workspace.capacity_bytes
                  << " communication=" << physical.communication.capacity_bytes
                  << " state_slots=" << physical.state_capacity_slots
                  << " kv_pages=" << physical.kv_capacity_pages << "}";
    }
    std::cout << '\n' << std::flush;
    check(tensor.physical_device_count == 2, "TP2 has no per-device inventory");
    check(2 * tensor.text_kv_bytes <= single.text_kv_bytes + single.text_kv_bytes / 100,
          "TP2 rank KV is not head-local");
    check(2 * tensor.gdn_state_bytes <= single.gdn_state_bytes + single.gdn_state_bytes / 100,
          "TP2 rank GDN state is not head-local");
    for (unsigned rank = 0; rank < 2; ++rank)
        check(tensor.physical_devices[rank].weights.used_bytes <
                  single.physical_devices[0].weights.used_bytes,
              "TP2 rank holds unsharded decoder weights");
    check(tensor.physical_devices[0].kv_capacity_pages == tensor.physical_devices[1].kv_capacity_pages &&
          tensor.physical_devices[0].state_capacity_slots == tensor.physical_devices[1].state_capacity_slots,
          "TP2 ranks disagree on logical page or state capacity");
}
}
int main() {
    try { reject_unsupported_startup(); }
    catch (const std::exception& error) { std::cerr << error.what() << "\n"; return 1; }
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* tensor_artifact = std::getenv("NINFER_TEST_TP2_ARTIFACT");
    const char* enabled = std::getenv("NINFER_TEST_TENSOR_ENGINE");
    int count = 0;
    if (!artifact || !*artifact || !tensor_artifact || !*tensor_artifact || !enabled ||
        std::string(enabled) != "1" || cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::cout << "SKIP: set NINFER_TEST_TENSOR_ENGINE=1, NINFER_QWEN3_6_27B_WEIGHTS (groupwise-int) "
                     "and NINFER_TEST_TP2_ARTIFACT (groupwise-int-tp2) with two GPUs\n";
        return 77;
    }
    cudaDeviceProp properties[2]{};
    if (!NINFER_COMPILED_SM89 || !NINFER_COMPILED_SM120 ||
        cudaGetDeviceProperties(&properties[0], 0) != cudaSuccess ||
        cudaGetDeviceProperties(&properties[1], 1) != cudaSuccess ||
        properties[0].major * 10 + properties[0].minor != 89 ||
        properties[1].major * 10 + properties[1].minor != 120) {
        std::cout << "SKIP: public TP2 fixture requires the ordered compiled SM89/SM120 pair\n";
        return 77;
    }
    NumericalChecks numerical;
    if (numerical.diagnostic)
        std::cout << "BEHAVIOR DIAGNOSTIC: score tolerance failures will be collected; any failure keeps final exit nonzero. "
                     "Generation, batches, cancellation, forced suffix and faults are pending.\n" << std::flush;
    try {
        reject_unsharded_artifact(artifact);
        std::vector<ninfer::TokenId> tokens;
        std::vector<float> reference;
        ninfer::MemorySummary single_memory;
        {
            ninfer::Engine engine(options(artifact, false, true));
            single_memory = engine.memory_summary();
            std::string text;
            while (tokens.size() < 1100) {
                text += "Each score window begins with fresh state. Pipeline execution preserves causal token probabilities.\n";
                tokens = engine.tokenize_text(text);
            }
            tokens.resize(1100);
            print_score_configuration("Single", engine, tokens.size());
            reference = engine.score_tokens(tokens, 1);
        }
        {
            ninfer::Engine engine(options(tensor_artifact, true, true));
            print_score_configuration("TP2", engine, tokens.size());
            check_local_state(single_memory, engine.memory_summary());
            auto all = engine.score_tokens(tokens, 1);
            auto suffix = engine.score_tokens(tokens, 1025);
            auto repeated = engine.score_tokens(tokens, 1025);
            check(all.size() == reference.size() && suffix.size() == 75, "score shape");
            const ScoreAgreement full = score_agreement(reference, all);
            report_score_pairs(tokens, reference, all);
            expect_score_agreement(numerical, full, "TP2 full window vs Single", kRouteChangeEnvelope);
            check(suffix == repeated, "TP2 score window inherited state");
            if (numerical.diagnostic) report_score_windows(reference, all, suffix, repeated);
            expect_score_agreement(numerical, score_agreement(all, suffix, 1024), "TP2 suffix window vs TP2 full");
            numerical.passed("repeated score window is bit-exact");
            std::cout << "score maximum error=" << full.maximum << "\n";
            bool invalid_rejected = false;
            try { (void)engine.score_tokens(tokens, 0); }
            catch (const std::invalid_argument&) { invalid_rejected = true; }
            check(invalid_rejected && engine.healthy(), "invalid score input poisoned Engine");
            check_empty_physical(engine);
            numerical.passed("score validation preserves health; both physical stores released");
#if defined(NINFER_TEST_TENSOR_FAULT_WRAP)
            numerical.phase = "scoring second-rank fault rollback";
            check_tail_prepare_failure(engine, [&] { (void)engine.score_tokens(tokens, 1); });
            numerical.passed("scoring second-rank fault quarantines and releases both stores");
#endif
        }
        numerical.phase = "Single greedy fixture and margin certification";
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 2;
        request.execution.allow_prefix_reuse = false;
        request.execution.sampling.temperature = 0;
        request.execution.sampling.presence_penalty = 0;
        request.execution.sampling.frequency_penalty = 0;
        request.stop.include_model_defaults = false;
        std::vector<std::vector<ninfer::TokenId>> histories(8), greedy(8);
        {
            ninfer::Engine engine(options(artifact, false, false));
            for (unsigned row = 0; row < 8; ++row) {
                std::string text;
                for (unsigned repeat = 0; repeat < row; ++repeat)
                    text += "Numbers increase by one. ";
                text += "Count upward by one:";
                for (unsigned number = row + 1; number <= row + 9; ++number)
                    text += " " + std::to_string(number) + ",";
                histories[row] = engine.tokenize_text(text);
                check(histories[row].size() + 127 <= 256,
                      "batch fixture entitlement exceeds the shared KV pool");
                greedy[row] = engine.generate(engine.prepare_tokens(histories[row], false), request).generated_token_ids;
                check(greedy[row].size() == 2, "greedy fixture did not reach ordinary decode");
            }
        }
        tokens = histories[0];
        const auto& single_generation = greedy[0];
        check(single_generation.size() == 2, "greedy fixture did not reach ordinary decode");
        {
            ninfer::Engine engine(options(artifact, false, true));
            for (unsigned row = 0; row < histories.size(); ++row) {
                auto scored = histories[row];
                scored.insert(scored.end(), greedy[row].begin(), greedy[row].end());
                const auto probabilities = engine.score_tokens(scored, static_cast<std::uint32_t>(histories[row].size()));
                for (float log_probability : probabilities) {
                    // Every alternative has probability <= 1-p, so this certifies the global
                    // argmax margin without a private logits API or a chosen competitor subset.
                    // Reserve a 0.01 logprob buffer for the FP32 reduction before certification.
                    const double conservative = static_cast<double>(log_probability) - 0.01;
                    const double probability = std::exp(conservative);
                    const double margin = conservative - std::log1p(-probability);
                    std::cout << "certified greedy logit margin >= " << margin << "\n";
                    check(std::isfinite(log_probability) && log_probability <= 0.001F && probability > 0.5 && margin >= 0.5,
                          "greedy fixture lacks a certified clear logit margin");
                }
            }
        }
        numerical.passed("Single distinct-history greedy margins certified");
        numerical.phase = "TP2 generation and lane reuse";
        // Exercise publication, cancellation through handle destruction, and physical slot reuse.
        {
            ninfer::Engine engine(options(tensor_artifact, true, false));
            auto first = engine.generate(engine.prepare_tokens(tokens, false), request);
            auto again = engine.generate(engine.prepare_tokens(tokens, false), request);
            check(!first.generated_token_ids.empty(), "TP2 produced no tokens");
            check(first.generated_token_ids == single_generation, "Single/TP2 greedy sequence differs");
            check(first.generated_token_ids == again.generated_token_ids, "TP2 lane reuse inherited state");
            numerical.passed("TP2 greedy publication and repeated lane reuse");
            numerical.phase = "active cancellation and lane reuse";
            auto long_request = request;
            long_request.execution.requested_output_tokens = 1024;
            const auto before_cancel = engine.runtime_stats().committed_decode_tokens;
            auto cancelled = engine.submit(engine.prepare_tokens(tokens, false), long_request);
            const auto cancellation_started = std::chrono::steady_clock::now();
            std::uint64_t cancellation_polls = 0;
            wait_until([&] {
                ++cancellation_polls;
                check(engine.healthy(), "Engine failed before active cancellation");
                const auto before_snapshot = engine.runtime_stats();
                if (before_snapshot.committed_decode_tokens <= before_cancel) return false;
                const auto snapshot_started = std::chrono::steady_clock::now();
                std::cout << "cancel poll before memory: poll=" << cancellation_polls
                          << " elapsed_seconds=" << std::chrono::duration<double>(snapshot_started - cancellation_started).count()
                          << " baseline_committed=" << before_cancel
                          << " committed=" << before_snapshot.committed_decode_tokens
                          << " running=" << before_snapshot.running_requests
                          << " decode_ready=" << before_snapshot.decode_ready_requests
                          << " terminal_pending=" << before_snapshot.terminal_pending_requests << '\n' << std::flush;
                const auto memory = engine.memory_summary();
                const auto snapshot_finished = std::chrono::steady_clock::now();
                const auto after_snapshot = engine.runtime_stats();
                std::cout << "cancel poll after memory: poll=" << cancellation_polls
                          << " blocked_seconds=" << std::chrono::duration<double>(snapshot_finished - snapshot_started).count()
                          << " elapsed_seconds=" << std::chrono::duration<double>(snapshot_finished - cancellation_started).count()
                          << " committed=" << after_snapshot.committed_decode_tokens
                          << " running=" << after_snapshot.running_requests
                          << " decode_ready=" << after_snapshot.decode_ready_requests
                          << " terminal_pending=" << after_snapshot.terminal_pending_requests
                          << " physical_devices=" << memory.physical_device_count;
                bool empty = memory.physical_device_count == 2;
                for (unsigned rank = 0; rank < memory.physical_device_count; ++rank) {
                    const auto& physical = memory.physical_devices[rank];
                    std::cout << " rank" << rank << "_state=" << physical.state_occupied_slots
                              << " rank" << rank << "_kv=" << physical.kv_occupied_pages;
                    empty = empty && physical.state_occupied_slots == 0 && physical.kv_occupied_pages == 0;
                }
                std::cout << '\n' << std::flush;
                const auto completed_decode_budget = long_request.execution.requested_output_tokens - 1;
                check(!(empty && after_snapshot.running_requests == 0 && after_snapshot.decode_ready_requests == 0 &&
                        after_snapshot.committed_decode_tokens - before_cancel >= completed_decode_budget),
                      "cancellation fixture missed active window: full decode budget completed before memory observation");
                return both_active(memory);
            }, "cancellation fixture never reached two-rank active decode");
            cancelled = ninfer::GenerationHandle{};
            wait_until([&] { return both_empty(engine); }, "active cancellation did not drain both ranks");
            check_empty_physical(engine);
            auto after = engine.generate(engine.prepare_tokens(tokens, false), request);
            check(first.generated_token_ids == after.generated_token_ids, "cancelled TP2 lane leaked state");
            numerical.passed("active two-rank cancellation drains and lane reuse matches");
            numerical.phase = "compact batches B2 through B8";
            auto batch_request = request;
            batch_request.execution.requested_output_tokens = 128;
            for (unsigned batch = 2; batch <= 8; ++batch) {
                const auto before = engine.runtime_stats();
                std::vector<ninfer::PreparedPrompt> prompts;
                for (unsigned row = 0; row < batch; ++row)
                    prompts.push_back(engine.prepare_tokens(histories[row], false));
                std::vector<ninfer::GenerationHandle> handles;
                for (auto& prompt : prompts)
                    handles.push_back(engine.submit(std::move(prompt), batch_request));
                for (unsigned row = 0; row < batch; ++row) {
                    const auto result = handles[row].wait();
                    check(result.generated_token_ids.size() == 128,
                          "batch fixture did not reach its long decode budget");
                    check(std::equal(greedy[row].begin(), greedy[row].end(), result.generated_token_ids.begin()),
                          "compact batch mapped a distinct history to the wrong clear-margin output");
                    check(std::all_of(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                                     [](ninfer::TokenId token) { return token >= 0 && token < 248077; }),
                          "batch returned a token outside the closed artifact domain");
                }
                const auto after_batch = engine.runtime_stats();
                const auto rounds = after_batch.decode_rounds - before.decode_rounds;
                const auto rows = after_batch.decode_row_rounds - before.decode_row_rounds;
                std::cout << "batch " << batch << " rounds=" << rounds << " rows=" << rows << "\n";
                check(rounds != 0 && rows > (batch - 1) * rounds,
                      "submitted requests never formed the required actual compact batch size");
                check_empty_physical(engine);
                numerical.passed("compact batch B" + std::to_string(batch) + " outputs, actual membership and cleanup");
            }
            check_empty_physical(engine);
            numerical.phase = "forced thinking suffix";
            ninfer::PromptInput chat;
            ninfer::ChatMessage message;
            message.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                .text = "Solve 1234567 times 7654321 step by step."});
            chat.messages.push_back(std::move(message));
            chat.options.enable_thinking = true;
            request.execution.requested_output_tokens = 32;
            request.execution.thinking.budget = 1;
            auto forced = engine.generate(engine.prepare(chat), request);
            check(forced.thinking.applied && forced.thinking.injected_tokens != 0,
                  "fixture did not exercise forced thinking suffix");
            auto forced_again = engine.generate(engine.prepare(std::move(chat)), request);
            check(forced.generated_token_ids == forced_again.generated_token_ids,
                  "forced suffix corrupted reusable physical lane");
            check(engine.healthy(), "TP2 Engine became unhealthy");
            check_empty_physical(engine);
            numerical.passed("forced thinking suffix and subsequent lane reuse");
#if defined(NINFER_TEST_TENSOR_FAULT_WRAP)
            numerical.phase = "generation second-rank fault rollback";
            request.execution.thinking = {};
            request.execution.requested_output_tokens = 2;
            check_tail_prepare_failure(engine, [&] {
                (void)engine.generate(engine.prepare_tokens(tokens, false), request);
            });
            numerical.passed("generation second-rank fault quarantines and releases both stores");
#else
            if (numerical.diagnostic) std::cout << "BEHAVIOR NOT RUN: fault injection unavailable on this platform\n";
#endif
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        if (numerical.diagnostic)
            std::cerr << "BEHAVIOR FAIL during " << numerical.phase << "; later cases remain pending. NUMERICAL "
                      << (numerical.failures ? "FAIL" : "no collected tolerance failures")
                      << " count=" << numerical.failures << "\n";
        return 1;
    }
    if (numerical.diagnostic) {
        std::cout << "BEHAVIOR PASS: all available cases completed. NUMERICAL "
                  << (numerical.failures ? "FAIL" : "PASS") << " count=" << numerical.failures << "\n";
        if (numerical.failures) return 1;
    }
    std::cout << "OK public TP2 Engine\n";
}
