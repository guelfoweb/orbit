// Offline real renderer/PEG parser probe. No model, context or inference.
#include "chat.h"
#include "json.h"
#include <iostream>
#include <string>
int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        common_json output;
        try {
            const auto request = common_json::parse(line);
            auto tmpl = common_chat_templates_init(nullptr, request.at("template").get<std::string>());
            common_chat_templates_inputs inputs;
            inputs.messages = common_chat_msgs_parse_oaicompat(request.at("messages"));
            inputs.tools = common_chat_tools_parse_oaicompat(request.at("tools"));
            inputs.use_jinja = true;
            inputs.add_generation_prompt = true;
            inputs.parallel_tool_calls = false;
            inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            inputs.enable_thinking = request.value("thinking", false);
            inputs.chat_template_kwargs["enable_thinking"] = inputs.enable_thinking ? "true" : "false";
            auto rendered = common_chat_templates_apply(tmpl.get(), inputs);
            output["prompt"] = rendered.prompt;
            output["grammar"] = rendered.grammar;
            output["parser"] = rendered.parser;
            output["format"] = common_chat_format_name(rendered.format);
            common_chat_parser_params params(rendered);
            params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            params.parser.load(rendered.parser);
            auto message = common_chat_parse(request.at("text"), request.value("partial", false), params);
            output["content"] = message.content;
            output["reasoning"] = message.reasoning_content;
            output["calls"] = common_json::array();
            for (const auto & call : message.tool_calls) {
                output["calls"].push_back({{"name",call.name},{"arguments",call.arguments}});
            }
        } catch (const std::exception & error) { output["error"] = error.what(); }
        std::cout << output.dump() << std::endl;
    }
}
