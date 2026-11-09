#include "serve/document_input.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/anthropic_messages.h"
#include <iostream>
#include <stdexcept>
using namespace ninfer::serve;
using J = RequestJson;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <class F>
static void invalid(F f) {
    try {
        f();
    } catch (const ApiException& e) {
        require(e.error().status == 400, "wrong PDF validation status");
        return;
    }
    throw std::runtime_error("invalid PDF source accepted");
}

int main() {
    try {
        const J file{{"file_data", "data:application/pdf;base64,JVBERi0="},
                     {"filename", "test.pdf"}};
        J chat{
            {"model", "qwen"},
            {"messages",
             J::array({J{{"role", "user"},
                         {"content", J::array({J{{"type", "file"}, {"file", file}},
                                               J{{"type", "text"}, {"text", "Summarize"}}})}}})}};
        const auto c = parse_chat_completion_request(chat, {}).generation;
        require(c.messages[0].content[0].kind == ContentKind::Document, "Chat PDF not preserved");
        require(c.messages[0].content[1].text == "Summarize", "Chat part order changed");
        J input       = file;
        input["type"] = "input_file";
        J responses{{"model", "qwen"},
                    {"input", J::array({J{{"role", "user"}, {"content", J::array({input})}}})}};
        const auto r = parse_openai_responses_create_request(responses, {});
        require(r.prompt.input_turns[0].content[0].kind == ContentKind::Document,
                "Responses PDF not preserved");
        J doc{{"type", "document"},
              {"title", "Report A"},
              {"context", "Use fiscal year 2026"},
              {"source",
               J{{"type", "base64"}, {"media_type", "application/pdf"}, {"data", "JVBERi0="}}}};
        J anthropic{{"model", "qwen"},
                    {"max_tokens", 32},
                    {"messages", J::array({J{{"role", "user"}, {"content", J::array({doc})}}})}};
        auto a = parse_anthropic_messages_request(anthropic, {}).generation;
        require(a.messages[0].content[0].text.find("Report A") != std::string::npos,
                "PDF title lost");
        require(a.messages[0].content[0].text.find("2026") != std::string::npos,
                "PDF context lost");
        doc["context"]                         = 7;
        anthropic["messages"][0]["content"][0] = doc;
        invalid([&] { (void)parse_anthropic_messages_request(anthropic, {}); });
        invalid([&] { (void)parse_pdf_source(J{{"file_id", "file_123"}}, "input"); });
        invalid([&] {
            (void)parse_pdf_source(
                J{{"file_data", "abc"}, {"file_url", "https://example.com/a.pdf"}}, "input");
        });
        invalid([&] { (void)parse_pdf_source(J{{"file_url", "file:///etc/passwd"}}, "input"); });
        invalid([&] { (void)parse_pdf_source(J{{"file_data", false}}, "input"); });
        invalid([&] {
            (void)parse_pdf_source(J{{"file_data", "data:image/png;base64,aA=="}}, "input");
        });
        require(parse_pdf_source(J{{"file_data", "JVBERi0="}}, "input")
                    .value.starts_with("data:application/pdf;base64,"),
                "raw base64 not normalized");
        std::cout << "PDF wire contracts, metadata preservation and invalid sources passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
