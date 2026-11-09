#include "product/document/pdf.h"
#include "product/media_acquire/acquire.h"
#include <cairo-pdf.h>
#include <cairo.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
using Bytes = std::vector<std::uint8_t>;

static cairo_status_t write_bytes(void* closure, const unsigned char* data, unsigned int size) {
    auto& out = *static_cast<Bytes*>(closure);
    out.insert(out.end(), data, data + size);
    return CAIRO_STATUS_SUCCESS;
}

static Bytes fixture(bool scan) {
    Bytes bytes;
    auto* surface = cairo_pdf_surface_create_for_stream(write_bytes, &bytes, 612, 792);
    auto* cr      = cairo_create(surface);
    cairo_set_source_rgb(cr, 0, 0, 0);
    if (scan) {
        cairo_move_to(cr, 72, 100);
        cairo_set_font_size(cr, 28);
        cairo_text_path(cr, "Invoice 7391 total 42.50");
        cairo_fill(cr);
    } else {
        cairo_move_to(cr, 72, 100);
        cairo_set_font_size(cr, 20);
        cairo_show_text(cr, "Invoice 7391 total 42.50");
    }
    cairo_show_page(cr);
    cairo_destroy(cr);
    cairo_surface_finish(surface);
    cairo_surface_destroy(surface);
    return bytes;
}

int main(int argc, char** argv) {
    if (argc == 2) {
        auto bytes = fixture(true);
        std::ofstream file(argv[1], std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        return file ? 0 : 1;
    }
    try {
        auto text = ninfer::product::document::read_pdf(fixture(false), false);
        if (text.size() != 1 || text[0].text.find("7391") == std::string::npos ||
            !text[0].png.empty())
            throw std::runtime_error("PDF text extraction failed");
        bool rejected = false;
        try {
            (void)ninfer::product::document::read_pdf(Bytes{1, 2, 3}, false);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("malformed PDF accepted");
        rejected = false;
        try {
            (void)ninfer::product::document::read_pdf(fixture(true), false);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("scanned PDF silently accepted without vision");
        auto scan = ninfer::product::document::read_pdf(fixture(true), true);
        if (scan.size() != 1 || scan[0].png.size() < 8 || scan[0].png[0] != 137 ||
            scan[0].png[1] != 'P')
            throw std::runtime_error("scanned page was not rendered to PNG");
        rejected = false;
        try {
            (void)ninfer::product::document::read_pdf(fixture(false), false, 0);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("page limit ignored");
        using namespace ninfer::product::media_acquire;
        rejected = false;
        try {
            (void)ninfer::product::document::read_pdf(fixture(false), false, 64,
                                                      std::chrono::steady_clock::time_point::max(),
                                                      [] { return true; });
        } catch (const Error& e) { rejected = e.kind() == ErrorKind::Cancelled; }
        if (!rejected) throw std::runtime_error("PDF cancellation lost its error kind");
        rejected = false;
        try {
            (void)ninfer::product::document::read_pdf(fixture(false), false, 64,
                                                      std::chrono::steady_clock::now() -
                                                          std::chrono::seconds(1));
        } catch (const Error& e) { rejected = e.kind() == ErrorKind::DeadlineExceeded; }
        if (!rejected) throw std::runtime_error("PDF deadline lost its error kind");
        std::cout << "PDF text, malformed input, scan rendering and page limits passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
