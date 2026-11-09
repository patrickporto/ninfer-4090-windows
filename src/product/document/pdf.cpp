#include "product/document/pdf.h"
#include "product/media_acquire/acquire.h"

#include <poppler.h>
#include <cairo.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace ninfer::product::document {
namespace {
constexpr std::size_t kMaxPdfBytes    = 32ULL << 20;
constexpr std::size_t kMaxOutputBytes = 16ULL << 20;

struct ObjectDelete {
    void operator()(gpointer object) const {
        if (object) { g_object_unref(object); }
    }
};

cairo_status_t write_png(void* target, const unsigned char* data, unsigned int length) {
    auto& bytes = *static_cast<std::vector<std::uint8_t>*>(target);
    if (length > kMaxOutputBytes - bytes.size()) { return CAIRO_STATUS_WRITE_ERROR; }
    try {
        bytes.insert(bytes.end(), data, data + length);
    } catch (...) { return CAIRO_STATUS_NO_MEMORY; }
    return CAIRO_STATUS_SUCCESS;
}
} // namespace

std::vector<PdfPage> read_pdf(std::span<const std::uint8_t> bytes, bool vision,
                              std::size_t max_pages, std::chrono::steady_clock::time_point deadline,
                              const std::function<bool()>& cancelled) {
    if (bytes.size() < 5 || bytes.size() > kMaxPdfBytes ||
        std::string_view(reinterpret_cast<const char*>(bytes.data()), 5) != "%PDF-") {
        throw std::invalid_argument("PDF must be a valid PDF document of at most 32 MiB");
    }
    GBytes* input = g_bytes_new_static(bytes.data(), bytes.size());
    GError* error = nullptr;
    std::unique_ptr<PopplerDocument, ObjectDelete> document(
        poppler_document_new_from_bytes(input, nullptr, &error));
    g_bytes_unref(input);
    if (!document) {
        const std::string message = error ? error->message : "invalid PDF";
        if (error) { g_error_free(error); }
        throw std::invalid_argument("Cannot read PDF (encrypted PDFs need unlocking): " + message);
    }
    const int pages = poppler_document_get_n_pages(document.get());
    if (pages < 1 || static_cast<std::size_t>(pages) > max_pages) {
        throw std::invalid_argument("PDF page count exceeds the supported limit of " +
                                    std::to_string(max_pages));
    }
    std::size_t output_bytes = 0;
    std::vector<PdfPage> result;
    for (int index = 0; index < pages; ++index) {
        if (cancelled && cancelled()) {
            throw media_acquire::Error(media_acquire::ErrorKind::Cancelled,
                                       "PDF preparation cancelled");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw media_acquire::Error(media_acquire::ErrorKind::DeadlineExceeded,
                                       "PDF preparation exceeded request deadline");
        }
        std::unique_ptr<PopplerPage, ObjectDelete> page(
            poppler_document_get_page(document.get(), index));
        if (!page) { throw std::invalid_argument("Cannot load PDF page"); }
        PdfPage out;
        gchar* text = poppler_page_get_text(page.get());
        if (text) {
            out.text = text;
            g_free(text);
        }
        const bool has_text = out.text.find_first_not_of(" \t\r\n\f") != std::string::npos;
        if (!has_text && !vision) {
            throw std::invalid_argument("PDF page has no extractable text; start the server with "
                                        "--vision for scanned PDFs");
        }
        if (vision) {
            double width = 0, height = 0;
            poppler_page_get_size(page.get(), &width, &height);
            if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0) {
                throw std::invalid_argument("Invalid PDF page dimensions");
            }
            const double scale = 1280.0 / std::max(width, height);
            const int w        = std::max(1, static_cast<int>(std::ceil(width * scale)));
            const int h        = std::max(1, static_cast<int>(std::ceil(height * scale)));
            std::unique_ptr<cairo_surface_t, decltype(&cairo_surface_destroy)> surface(
                cairo_image_surface_create(CAIRO_FORMAT_RGB24, w, h), cairo_surface_destroy);
            std::unique_ptr<cairo_t, decltype(&cairo_destroy)> context(cairo_create(surface.get()),
                                                                       cairo_destroy);
            cairo_set_source_rgb(context.get(), 1, 1, 1);
            cairo_paint(context.get());
            cairo_scale(context.get(), scale, scale);
            poppler_page_render(page.get(), context.get());
            if (cairo_status(context.get()) != CAIRO_STATUS_SUCCESS ||
                cairo_surface_write_to_png_stream(surface.get(), write_png, &out.png) !=
                    CAIRO_STATUS_SUCCESS) {
                throw std::invalid_argument("Cannot render PDF page within output limits");
            }
            if (!has_text) { out.text.clear(); }
        }
        if (out.text.size() > kMaxOutputBytes - output_bytes) {
            throw std::invalid_argument("Extracted PDF exceeds 16 MiB");
        }
        output_bytes += out.text.size();
        if (out.png.size() > kMaxOutputBytes - output_bytes) {
            throw std::invalid_argument("Rendered PDF exceeds 16 MiB");
        }
        output_bytes += out.png.size();
        result.push_back(std::move(out));
    }
    return result;
}
} // namespace ninfer::product::document
