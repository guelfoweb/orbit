#pragma once
#include "chat.h"

// Map Orbit's thinking-off request to the template's declared effort control,
// when (and only when) it explicitly supports the standard "none" level.
// This is rendering metadata, never user-intent classification or a forced
// reasoning delimiter. Templates without this capability retain their behavior.
inline bool orbit_apply_reasoning_control(const common_chat_templates * templates,
                                         common_chat_templates_inputs & inputs,
                                         bool enabled) {
    inputs.enable_thinking = enabled;
    inputs.chat_template_kwargs["enable_thinking"] = enabled ? "true" : "false";
    if (enabled) {
        return false; // retain the template's default effort
    }
    const auto caps = common_chat_templates_get_caps(templates);
    if (!caps.value("supports_reasoning_effort", false) ||
        !caps.contains("reasoning_effort_levels") || !caps.at("reasoning_effort_levels").is_array()) {
        return false;
    }
    for (const auto & level : caps.at("reasoning_effort_levels")) {
        if (level.is_string() && level.get<std::string>() == "none") {
            inputs.chat_template_kwargs["reasoning_effort"] = "\"none\"";
            return true;
        }
    }
    return false;
}
