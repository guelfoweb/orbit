#include "chat.h"
#include "json.h"
#include "llama.h"

#include "nlohmann/json.hpp"

#include <cstddef>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#if defined(_WIN32)
#define ORBIT_EXPORT __declspec(dllexport)
#else
#define ORBIT_EXPORT __attribute__((visibility("default")))
#endif

namespace {

using json = nlohmann::ordered_json;

thread_local std::string last_error;

struct orbit_chat_context {
    common_chat_templates_ptr templates;
    common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string generation_prompt;
    std::string parser;
    std::string reasoning_start_tag;
    std::string reasoning_end_tag;
    bool render_ready = false;
};

int copy_result(const std::string & value, char * output, size_t output_size) {
    if (output == nullptr || output_size == 0) {
        return static_cast<int>(value.size());
    }
    if (value.size() + 1 > output_size) {
        return static_cast<int>(value.size());
    }
    std::memcpy(output, value.data(), value.size());
    output[value.size()] = '\0';
    return static_cast<int>(value.size());
}

json parsed_message_json(const common_chat_msg & message) {
    json result = {
        {"content", message.content},
        {"reasoning_content", message.reasoning_content},
        {"tool_calls", json::array()},
    };
    for (const auto & call : message.tool_calls) {
        result["tool_calls"].push_back({
            {"id", call.id},
            {"type", "function"},
            {"function", {
                {"name", call.name},
                {"arguments", call.arguments},
            }},
        });
    }
    return result;
}

}  // namespace

extern "C" {

ORBIT_EXPORT uint32_t orbit_chat_bridge_api_version() {
    return 1;
}

ORBIT_EXPORT const char * orbit_chat_bridge_last_error() {
    return last_error.c_str();
}

ORBIT_EXPORT void * orbit_chat_bridge_create(const llama_model * model) {
    last_error.clear();
    if (model == nullptr) {
        last_error = "model handle is null";
        return nullptr;
    }
    try {
        auto context = std::make_unique<orbit_chat_context>();
        context->templates = common_chat_templates_init(model, "");
        if (!context->templates) {
            last_error = "failed to initialize chat templates";
            return nullptr;
        }
        return context.release();
    } catch (const std::exception & exc) {
        last_error = exc.what();
        return nullptr;
    }
}

ORBIT_EXPORT void orbit_chat_bridge_free(void * opaque) {
    delete static_cast<orbit_chat_context *>(opaque);
}

static int render_contract(
    void * opaque,
    const char * messages_json,
    const char * tools_json,
    bool enable_thinking,
    bool required,
    char * output,
    size_t output_size
) {
    last_error.clear();
    auto * context = static_cast<orbit_chat_context *>(opaque);
    if (context == nullptr || messages_json == nullptr || tools_json == nullptr) {
        last_error = "invalid render arguments";
        return -1;
    }
    try {
        // upstream chat.h takes its own common_json wrapper (41abbfd); nlohmann
        // stays for the bridge's own output document only.
        const common_json messages = common_json::parse(messages_json);
        const common_json tools = common_json::parse(tools_json);
        if (!messages.is_array() || !tools.is_array()) {
            throw std::invalid_argument("messages and tools must be arrays");
        }

        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools = common_chat_tools_parse_oaicompat(tools);
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.tool_choice = required ? COMMON_CHAT_TOOL_CHOICE_REQUIRED : COMMON_CHAT_TOOL_CHOICE_AUTO;
        inputs.parallel_tool_calls = false;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.enable_thinking = enable_thinking;
        inputs.chat_template_kwargs["enable_thinking"] = enable_thinking ? "true" : "false";

        const common_chat_params params = common_chat_templates_apply(context->templates.get(), inputs);
        if (params.prompt.empty()) {
            throw std::runtime_error("chat template produced an empty prompt");
        }
        if (required && (inputs.tools.empty() || params.grammar.empty() || params.grammar_lazy)) {
            throw std::runtime_error("required tool grammar unavailable or lazy");
        }
        context->format = params.format;
        context->generation_prompt = params.generation_prompt;
        context->parser = params.parser;
        context->reasoning_start_tag = params.thinking_start_tag;
        context->reasoning_end_tag = params.thinking_end_tags.empty() ? std::string() : params.thinking_end_tags.front();
        context->render_ready = true;

        json result = {
            {"prompt", params.prompt},
            {"generation_prompt", params.generation_prompt},
            {"format", common_chat_format_name(params.format)},
            {"supports_thinking", params.supports_thinking},
            {"thinking_start_tag", params.thinking_start_tag},
            // upstream now carries a list of end tags; keep the scalar key (first
            // tag, or empty) for existing readers and expose the full list beside it.
            {"thinking_end_tag", params.thinking_end_tags.empty() ? std::string() : params.thinking_end_tags.front()},
            {"thinking_end_tags", params.thinking_end_tags},
            {"additional_stops", params.additional_stops},
        };
        if (required) {
            result["grammar"] = params.grammar;
            result["grammar_lazy"] = params.grammar_lazy;
            result["tool_choice"] = "required";
        }
        return copy_result(result.dump(), output, output_size);
    } catch (const std::exception & exc) {
        context->render_ready = false;
        last_error = exc.what();
        return -1;
    }
}

ORBIT_EXPORT int orbit_chat_bridge_render(
    void * opaque, const char * messages_json, const char * tools_json,
    bool enable_thinking, char * output, size_t output_size
) {
    return render_contract(opaque, messages_json, tools_json, enable_thinking,
                           false, output, output_size);
}

ORBIT_EXPORT int orbit_chat_bridge_render_required(
    void * opaque, const char * messages_json, const char * tools_json,
    bool enable_thinking, char * output, size_t output_size
) {
    return render_contract(opaque, messages_json, tools_json, enable_thinking,
                           true, output, output_size);
}

// Same generation-prompt prefill as common/sampling.cpp. Catch native errors
// at the bridge boundary; never expose a half-initialized sampler to Python.
ORBIT_EXPORT llama_sampler * orbit_chat_bridge_required_sampler(
    const llama_vocab * vocab, const char * grammar, const char * generation_prompt
) {
    last_error.clear();
    llama_sampler * sampler = nullptr;
    try {
        if (!vocab || !grammar || !*grammar || !generation_prompt) {
            throw std::invalid_argument("invalid required grammar inputs");
        }
        auto params = llama_sampler_chain_default_params();
        params.no_perf = false;
        sampler = llama_sampler_chain_init(params);
        if (!sampler) throw std::runtime_error("required sampler allocation failed");
        auto * constraint = llama_sampler_init_grammar(vocab, grammar, "root");
        if (!constraint) throw std::runtime_error("required grammar compilation failed");
        llama_sampler_chain_add(sampler, constraint);
        const std::string prefill(generation_prompt);
        const auto tokens = common_tokenize(vocab, prefill, false, true);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && !piece.empty() && !prefill.empty() &&
                std::isspace(static_cast<unsigned char>(piece[0])) &&
                !std::isspace(static_cast<unsigned char>(prefill[0]))) continue;
            llama_token_data token{tokens[i], 0.0f, 0.0f};
            llama_token_data_array candidates{&token, 1, -1, false};
            llama_sampler_apply(constraint, &candidates);
            if (!std::isfinite(token.logit)) throw std::runtime_error("grammar rejects generation prompt");
            llama_sampler_accept(constraint, tokens[i]);
        }
        auto * greedy = llama_sampler_init_greedy();
        if (!greedy) throw std::runtime_error("required greedy sampler allocation failed");
        llama_sampler_chain_add(sampler, greedy);
        return sampler;
    } catch (const std::exception & exc) {
        if (sampler) llama_sampler_free(sampler);
        last_error = exc.what();
        return nullptr;
    }
}

ORBIT_EXPORT int orbit_chat_bridge_parse(
    void * opaque,
    const char * generated_text,
    bool is_partial,
    char * output,
    size_t output_size
) {
    last_error.clear();
    auto * context = static_cast<orbit_chat_context *>(opaque);
    if (context == nullptr || generated_text == nullptr || !context->render_ready) {
        last_error = "chat parser is not initialized";
        return -1;
    }
    try {
        common_chat_parser_params params;
        params.format = context->format;
        params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        params.reasoning_in_content = false;
        params.generation_prompt = context->generation_prompt;
        params.reasoning_start_tag = context->reasoning_start_tag;
        params.reasoning_end_tag = context->reasoning_end_tag;
        params.parse_tool_calls = true;
        if (!context->parser.empty()) {
            params.parser.load(context->parser);
        }
        const common_chat_msg message = common_chat_parse(generated_text, is_partial, params);
        return copy_result(parsed_message_json(message).dump(), output, output_size);
    } catch (const std::exception & exc) {
        last_error = exc.what();
        return -1;
    }
}

}  // extern "C"
