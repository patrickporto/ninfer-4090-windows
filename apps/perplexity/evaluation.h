#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::perplexity {

struct WindowPlan {
    std::size_t input_begin    = 0;
    std::size_t input_end      = 0;
    std::size_t target_begin   = 0;
    std::size_t target_end     = 0;
    std::uint32_t first_target = 0;
};

[[nodiscard]] std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                                   std::uint32_t stride);

// One JSON line per scored target, using its zero-based position in the original stream.
void write_token_nll(std::ostream& output, std::string_view stream_id, const WindowPlan& window,
                     std::span<const float> logprobs);

// Reject aliases of any input or reserved report file before opening the export for truncation.
void validate_token_nll_destination(const std::filesystem::path& destination,
                                    std::span<const std::filesystem::path> protected_paths);

struct ScoreAggregate {
    std::uint64_t scored_tokens = 0;
    double total_nll            = 0.0;

    void add(std::span<const float> logprobs);
    void add(const ScoreAggregate& other) noexcept;
    [[nodiscard]] double mean_nll() const;
    [[nodiscard]] double ppl() const;
};

} // namespace ninfer::perplexity
