// Build without poppler (NINFER_ENABLE_PDF=OFF, the Windows default): PDF parts are rejected
// as a client error instead of failing the build.
#include "product/document/pdf.h"

#include <stdexcept>

namespace ninfer::product::document {

std::vector<PdfPage> read_pdf(std::span<const std::uint8_t>, bool, std::size_t,
                              std::chrono::steady_clock::time_point,
                              const std::function<bool()>&) {
    throw std::invalid_argument("PDF input is not available in this build (built without poppler)");
}

} // namespace ninfer::product::document
