#include "evaluation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace ninfer::perplexity {

std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                     std::uint32_t stride) {
    if (tokens < 2) { throw std::invalid_argument("perplexity stream must contain two tokens"); }
    if (context < 2 || stride == 0 || stride >= context) {
        throw std::invalid_argument("perplexity requires context>=2 and 1<=stride<context");
    }

    std::vector<WindowPlan> windows;
    std::size_t previous_end = std::min<std::size_t>(tokens, context);
    windows.push_back(WindowPlan{.input_begin  = 0,
                                 .input_end    = previous_end,
                                 .target_begin = 1,
                                 .target_end   = previous_end,
                                 .first_target = 1});
    while (previous_end < tokens) {
        const std::size_t end          = std::min(tokens, previous_end + stride);
        const std::size_t begin        = end > context ? end - context : 0;
        const std::size_t local_target = previous_end - begin;
        if (local_target == 0 || local_target >= end - begin ||
            local_target > std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("perplexity window has an invalid target suffix");
        }
        windows.push_back(WindowPlan{
            .input_begin  = begin,
            .input_end    = end,
            .target_begin = previous_end,
            .target_end   = end,
            .first_target = static_cast<std::uint32_t>(local_target),
        });
        previous_end = end;
    }
    return windows;
}

void write_token_nll(std::ostream& output, std::string_view stream_id, const WindowPlan& window,
                     std::span<const float> logprobs) {
    if (window.target_end < window.target_begin ||
        logprobs.size() != window.target_end - window.target_begin) {
        throw std::invalid_argument("token NLL target count differs from the scoring window");
    }
    for (const float logprob : logprobs) {
        if (!std::isfinite(logprob)) {
            throw std::runtime_error("causal scoring returned a non-finite logprob");
        }
    }
    for (std::size_t index = 0; index < logprobs.size(); ++index) {
        output << nlohmann::json{{"stream_id", stream_id},
                                 {"target_index", window.target_begin + index},
                                 {"nll", -static_cast<double>(logprobs[index])}}
                      .dump()
               << '\n';
    }
    if (!output) { throw std::runtime_error("cannot write per-token NLL output"); }
}

void validate_token_nll_destination(const std::filesystem::path& destination,
                                    std::span<const std::filesystem::path> protected_paths) {
    const auto normalized =
        std::filesystem::weakly_canonical(std::filesystem::absolute(destination));
    for (const auto& protected_path : protected_paths) {
        const auto protected_normalized =
            std::filesystem::weakly_canonical(std::filesystem::absolute(protected_path));
        if (normalized == protected_normalized ||
            (std::filesystem::exists(normalized) && std::filesystem::exists(protected_normalized) &&
             std::filesystem::equivalent(normalized, protected_normalized))) {
            throw std::invalid_argument(
                "per-token NLL output aliases an input or reserved report: " +
                destination.string());
        }
    }
}

void ScoreAggregate::add(std::span<const float> logprobs) {
    for (const float logprob : logprobs) {
        if (!std::isfinite(logprob)) {
            throw std::runtime_error("causal scoring returned a non-finite logprob");
        }
        total_nll -= static_cast<double>(logprob);
    }
    if (logprobs.size() > std::numeric_limits<std::uint64_t>::max() - scored_tokens) {
        throw std::overflow_error("perplexity scored-token count overflowed");
    }
    scored_tokens += static_cast<std::uint64_t>(logprobs.size());
}

void ScoreAggregate::add(const ScoreAggregate& other) noexcept {
    scored_tokens += other.scored_tokens;
    total_nll += other.total_nll;
}

double ScoreAggregate::mean_nll() const {
    if (scored_tokens == 0) { throw std::logic_error("perplexity aggregate is empty"); }
    return total_nll / static_cast<double>(scored_tokens);
}

double ScoreAggregate::ppl() const { return std::exp(mean_nll()); }

} // namespace ninfer::perplexity
