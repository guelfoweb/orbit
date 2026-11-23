// Offline real renderer/PEG parser probe. No model, context or inference.
#include "chat.h"
#include "orbit-reasoning-control.h"
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
            output["effort_none_applied"] = orbit_apply_reasoning_control(tmpl.get(), inputs, request.value("thinking", false));
            auto rendered = common_chat_templates_apply(tmpl.get(), inputs);
            output["prompt"] = rendered.prompt.str();
            output["grammar"] = rendered.grammar;
            output["parser_defined"] = !rendered.parser.empty();
            output["format"] = common_chat_format_name(rendered.format);
            common_chat_parser_params params(rendered);
            params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            params.parser = rendered.parser;
            auto message = common_chat_parse(common_chat_input(request.at("text").get<std::string>()), request.value("partial", false), params);
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
