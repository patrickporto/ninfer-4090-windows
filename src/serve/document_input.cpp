#include "serve/document_input.h"
#include "serve/request_validation.h"
#include "product/document/pdf.h"
#include <algorithm>
#include <iterator>

namespace ninfer::serve {
product::media_acquire::Source parse_pdf_source(const RequestJson& file, const char* param) {
    using namespace product::media_acquire;
    if (!file.is_object()) { bad_request("PDF file must be an object", param); }
    if (file.contains("file_id") && !file.at("file_id").is_null()) {
        bad_request("file_id requires a Files API; send PDF file_data or file_url instead", param,
                    "file_id_not_supported");
    }
    const bool data = file.contains("file_data") && !file.at("file_data").is_null();
    const bool url  = file.contains("file_url") && !file.at("file_url").is_null();
    if (data == url) { bad_request("PDF requires exactly one of file_data or file_url", param); }
    const auto& value = file.at(data ? "file_data" : "file_url");
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        bad_request("PDF source must be a non-empty string", param);
    }
    Source source;
    source.media_type = "application/pdf";
    source.value      = value.get<std::string>();
    if (data) {
        source.kind = SourceKind::Data;
        if (!source.value.starts_with("data:")) {
            source.value.insert(0, "data:application/pdf;base64,");
        } else if (!source.value.starts_with("data:application/pdf;base64,")) {
            bad_request("file_data must be a base64 application/pdf data URL", param);
        }
    } else {
        source.kind = SourceKind::Url;
        if (!source.value.starts_with("https://") && !source.value.starts_with("http://")) {
            bad_request("file_url must use http or https", param);
        }
    }
    return source;
}

product::media_acquire::Source parse_anthropic_pdf_source(const RequestJson& block) {
    if (block.contains("citations") && !block.at("citations").is_null()) {
        const auto& c = block.at("citations");
        if (!c.is_object() || (c.contains("enabled") && c.at("enabled") != false)) {
            bad_request("PDF citations are not supported", "messages", "citations_not_supported");
        }
    }
    if (!block.contains("source") || !block.at("source").is_object()) {
        bad_request("document requires a PDF source", "messages");
    }
    const auto& s = block.at("source");
    if (!s.contains("type") || !s.at("type").is_string()) {
        bad_request("document source requires a string type", "messages", "invalid_document");
    }
    if (s.at("type") == "base64") {
        if (!s.contains("media_type") || s.at("media_type") != "application/pdf" ||
            !s.contains("data")) {
            bad_request("document source must contain application/pdf base64 data", "messages");
        }
        return parse_pdf_source(RequestJson{{"file_data", s.at("data")}}, "messages");
    }
    if (s.at("type") == "url" && s.contains("url")) {
        return parse_pdf_source(RequestJson{{"file_url", s.at("url")}}, "messages");
    }
    bad_request("document source must be base64 or url", "messages");
}

GenerationRequest expand_pdf_documents(const GenerationRequest& request, bool vision,
                                       const DocumentAcquirer& acquire,
                                       std::chrono::steady_clock::time_point deadline,
                                       const std::function<bool()>& cancelled) {
    GenerationRequest result     = request;
    std::size_t remaining_pages  = 64;
    std::size_t remaining_output = 16ULL << 20;
    for (auto& message : result.messages) {
        std::vector<ContentPart> expanded;
        for (auto& part : message.content) {
            if (part.kind != ContentKind::Document) {
                expanded.push_back(std::move(part));
                continue;
            }
            if (!part.text.empty()) {
                if (part.text.size() > remaining_output) {
                    bad_request("PDF metadata exceeds document output budget", "messages",
                                "document_budget_exceeded");
                }
                remaining_output -= part.text.size();
                ContentPart metadata;
                metadata.kind     = ContentKind::Text;
                metadata.type_raw = "text";
                metadata.text     = part.text;
                expanded.push_back(std::move(metadata));
            }
            const auto bytes = acquire(part);
            auto pages =
                product::document::read_pdf(bytes, vision, remaining_pages, deadline, cancelled);
            remaining_pages -= pages.size();
            for (std::size_t index = 0; index < pages.size(); ++index) {
                auto& page      = pages[index];
                const auto size = page.text.size() + page.png.size();
                if (size > remaining_output) {
                    bad_request("PDFs exceed the aggregate 16 MiB output budget", "messages",
                                "document_budget_exceeded");
                }
                remaining_output -= size;
                ContentPart text;
                text.kind     = ContentKind::Text;
                text.type_raw = "text";
                text.text     = "\n[PDF page " + std::to_string(index + 1) + "]\n" + page.text;
                expanded.push_back(std::move(text));
                if (!page.png.empty()) {
                    ContentPart image;
                    image.kind              = ContentKind::Image;
                    image.type_raw          = "image";
                    image.source.kind       = product::media_acquire::SourceKind::Bytes;
                    image.source.media_type = "image/png";
                    image.source.bytes      = std::move(page.png);
                    expanded.push_back(std::move(image));
                }
            }
            expanded.back().cache_boundary_after = part.cache_boundary_after;
        }
        message.content = std::move(expanded);
    }
    return result;
}
} // namespace ninfer::serve
