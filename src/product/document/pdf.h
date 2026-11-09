#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::product::document {

struct PdfPage {
    std::string text;
    std::vector<std::uint8_t> png;
};

// Text is extracted without a model. Pages without extractable text require Vision.
// No temporary files are written; the caller retains acquisition/deadline ownership.
std::vector<PdfPage> read_pdf(
    std::span<const std::uint8_t> bytes, bool vision, std::size_t max_pages = 64,
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max(),
    const std::function<bool()>& cancelled         = {});

} // namespace ninfer::product::document
