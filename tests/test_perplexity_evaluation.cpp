#include "evaluation.h"
#include "kv_storage.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

int check_partition(std::size_t tokens, std::uint32_t context, std::uint32_t stride) {
    const auto windows   = ninfer::perplexity::plan_windows(tokens, context, stride);
    int failures         = 0;
    std::size_t frontier = 1;
    for (const auto& window : windows) {
        failures += require(window.input_begin < window.input_end, "window input is nonempty");
        failures +=
            require(window.input_end - window.input_begin <= context, "window respects context");
        failures += require(window.target_begin == frontier, "target ranges are contiguous");
        failures +=
            require(window.target_end <= window.input_end, "targets are inside input window");
        failures += require(window.first_target == window.target_begin - window.input_begin,
                            "local target matches global range");
        frontier = window.target_end;
    }
    failures += require(frontier == tokens, "targets cover every token after the first");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    using ninfer::KvCacheStorage;
    const std::array kv_modes{
        std::pair{"bf16", KvCacheStorage::BFloat16},
        std::pair{"int8", KvCacheStorage::Int8Group64},
        std::pair{"fp8", KvCacheStorage::Fp8E4M3Row256},
        std::pair{"nvfp4", KvCacheStorage::Nvfp4Group16},
        std::pair{"k8v4", KvCacheStorage::Fp8KeyNvfp4Value},
        std::pair{"rk8v4", KvCacheStorage::RotatedInt8KeyInt4ValueGroup64},
        std::pair{"rk4v4", KvCacheStorage::RotatedInt4KeyInt4ValueGroup64},
        std::pair{"rk4v4-e8", KvCacheStorage::RK4V4E8},
        std::pair{"rk2v4-e8", KvCacheStorage::RK2V4E8},
    };
    for (const auto& [name, storage] : kv_modes) {
        failures += require(ninfer::perplexity::parse_kv_storage(name) == storage,
                            "perplexity KV dtype selects the matching engine storage");
        const std::string expected_name = std::string(name) == "int8"  ? "int8-g64"
                                          : std::string(name) == "fp8" ? "fp8-e4m3-r256"
                                                                       : name;
        failures += require(ninfer::perplexity::kv_name(storage) == expected_name,
                            "perplexity report preserves the selected KV dtype");
    }
    bool rejected_kv = false;
    try {
        (void)ninfer::perplexity::parse_kv_storage("rk3v4-e8");
    } catch (const std::invalid_argument&) { rejected_kv = true; }
    failures += require(rejected_kv, "perplexity rejects unknown KV dtypes");
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("ninfer-token-nll-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(directory)) {
        throw std::runtime_error("cannot create token NLL test directory");
    }

    struct RemoveDirectory {
        std::filesystem::path path;

        ~RemoveDirectory() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } remove_directory{directory};

    std::filesystem::create_directory(directory / "reports");
    const std::vector<std::filesystem::path> protected_paths{directory / "model.ninfer",
                                                             directory / "corpus.json",
                                                             directory / "text.txt",
                                                             directory / "corpus-stream.txt",
                                                             directory / "reports/report.json",
                                                             directory / "reports/report.json.tmp"};
    for (std::size_t index = 0; index < 4; ++index) {
        std::ofstream(protected_paths[index]) << "preserve input";
    }
    const auto rejects_destination = [&](const std::filesystem::path& path) {
        try {
            ninfer::perplexity::validate_token_nll_destination(path, protected_paths);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    for (const auto& path : protected_paths) {
        failures +=
            require(rejects_destination(path), "token NLL protects inputs and report paths");
        failures += require(rejects_destination(path.parent_path() / ".." /
                                                path.parent_path().filename() / path.filename()),
                            "token NLL rejects normalized aliases");
    }
    std::filesystem::create_symlink(protected_paths.front(), directory / "model-symlink");
    std::filesystem::create_hard_link(protected_paths.front(), directory / "model-hardlink");
    std::filesystem::create_directory_symlink(directory / "reports", directory / "report-alias");
    failures += require(rejects_destination(directory / "model-symlink"),
                        "token NLL protects symlink aliases");
    failures += require(rejects_destination(directory / "model-hardlink"),
                        "token NLL protects hardlink aliases");
    failures += require(rejects_destination(directory / "report-alias/report.json"),
                        "token NLL protects report directory aliases");
    ninfer::perplexity::validate_token_nll_destination(directory / "tokens.jsonl", protected_paths);
    failures += check_partition(2, 4096, 2048);
    failures += check_partition(4096, 4096, 2048);
    failures += check_partition(4097, 4096, 2048);
    failures += check_partition(12001, 4096, 2048);

    const std::array<ninfer::perplexity::WindowPlan, 3> expected{{
        {.input_begin = 0, .input_end = 6, .target_begin = 1, .target_end = 6, .first_target = 1},
        {.input_begin = 2, .input_end = 8, .target_begin = 6, .target_end = 8, .first_target = 4},
        {.input_begin = 4, .input_end = 10, .target_begin = 8, .target_end = 10, .first_target = 4},
    }};
    const auto planned = ninfer::perplexity::plan_windows(10, 6, 2);
    failures += require(planned.size() == expected.size(), "window count matches the protocol");
    for (std::size_t index = 0; index < std::min(planned.size(), expected.size()); ++index) {
        const auto& actual = planned[index];
        const auto& wanted = expected[index];
        failures += require(actual.input_begin == wanted.input_begin &&
                                actual.input_end == wanted.input_end &&
                                actual.target_begin == wanted.target_begin &&
                                actual.target_end == wanted.target_end &&
                                actual.first_target == wanted.first_target,
                            "window boundaries match the protocol");
    }

    const std::vector<float> first{-1.0F, -2.0F};
    const std::vector<float> second{-3.0F};
    ninfer::perplexity::ScoreAggregate a;
    ninfer::perplexity::ScoreAggregate b;
    a.add(first);
    b.add(second);
    a.add(b);
    failures += require(a.scored_tokens == 3, "aggregate counts target tokens");
    failures += require(std::abs(a.total_nll - 6.0) < 1e-12, "aggregate accumulates FP64 NLL");
    failures += require(std::abs(a.mean_nll() - 2.0) < 1e-12, "aggregate computes mean NLL");
    failures += require(std::abs(a.ppl() - std::exp(2.0)) < 1e-12, "aggregate computes perplexity");

    // Overlapping windows must export each absolute target once, including IDs with whitespace.
    std::ostringstream token_output;
    const std::string stream_id = "notes with spaces\nsecond line";
    for (const auto& window : planned) {
        std::vector<float> logprobs;
        for (std::size_t i = window.target_begin; i < window.target_end; ++i) {
            logprobs.push_back(-static_cast<float>(i) / 7.0F);
        }
        ninfer::perplexity::write_token_nll(token_output, stream_id, window, logprobs);
    }
    std::istringstream rows(token_output.str());
    std::size_t target = 1;
    std::string row;
    while (std::getline(rows, row)) {
        const auto record = nlohmann::json::parse(row);
        failures += require(record.size() == 3, "token NLL rows have three fields");
        failures += require(record.at("stream_id") == stream_id, "token NLL preserves stream IDs");
        failures +=
            require(record.at("target_index") == target, "token NLL uses absolute positions");
        const double nll = record.at("nll").get<double>();
        failures += require(std::isfinite(nll) &&
                                nll == static_cast<double>(static_cast<float>(target) / 7.0F),
                            "token NLL preserves finite scorer precision and sign");
        ++target;
    }
    failures += require(target == 10, "token NLL exports every scored token exactly once");

    const auto rejects_export = [&](std::ostream& output, std::span<const float> logprobs) {
        try {
            ninfer::perplexity::write_token_nll(output, "stream", planned.front(), logprobs);
        } catch (const std::exception&) { return true; }
        return false;
    };
    std::ostringstream invalid_output;
    failures +=
        require(rejects_export(invalid_output, first), "token NLL rejects target count mismatch");
    std::vector<float> invalid_logprobs(5, -1.0F);
    invalid_logprobs.back() = std::numeric_limits<float>::infinity();
    failures += require(rejects_export(invalid_output, invalid_logprobs),
                        "token NLL rejects non-finite values");
    failures += require(invalid_output.str().empty(), "invalid token NLL window writes no rows");
    std::ostringstream failed_output;
    failed_output.setstate(std::ios::badbit);
    const std::vector<float> valid_logprobs(5, -1.0F);
    failures +=
        require(rejects_export(failed_output, valid_logprobs), "token NLL reports write failures");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_evaluation\n";
    return failures == 0 ? 0 : 1;
}
