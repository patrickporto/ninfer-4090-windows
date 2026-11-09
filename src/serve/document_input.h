#pragma once
#include "serve/request.h"
#include "serve/request_json.h"
#include <chrono>
#include <functional>

namespace ninfer::serve {
product::media_acquire::Source parse_pdf_source(const RequestJson& file, const char* param);
product::media_acquire::Source parse_anthropic_pdf_source(const RequestJson& block);
using DocumentAcquirer = std::function<std::vector<std::uint8_t>(const ContentPart&)>;
GenerationRequest expand_pdf_documents(const GenerationRequest& request, bool vision,
                                       const DocumentAcquirer& acquire,
                                       std::chrono::steady_clock::time_point deadline,
                                       const std::function<bool()>& cancelled);
} // namespace ninfer::serve
