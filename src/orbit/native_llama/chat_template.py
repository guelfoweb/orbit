from __future__ import annotations

import json
from typing import Any


NativeMessage = dict[str, Any]


def render_gemma4_chat(
    messages: list[NativeMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    chunks: list[str] = ["<bos>"]
    start = 0
    if messages and messages[0].get("role") in {"system", "developer"}:
        chunks.append("<|turn>system\n")
        chunks.append(_content_text(messages[0]).strip())
        tool_text = _render_tools(tools or [])
        if tool_text:
            chunks.append("\n\n")
            chunks.append(tool_text)
        chunks.append("<turn|>\n")
        start = 1
    elif tools:
        chunks.append("<|turn>system\n")
        chunks.append(_render_tools(tools))
        chunks.append("<turn|>\n")

    for message in messages[start:]:
        role = message.get("role")
        if role == "tool":
            chunks.append(_render_tool_response(message))
            continue
        gemma_role = "model" if role == "assistant" else str(role or "user")
        chunks.append(f"<|turn>{gemma_role}\n")
        content = _strip_thinking(_content_text(message)).strip()
        if content:
            chunks.append(content)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    chunks.append(_render_tool_call(tool_call))
        chunks.append("<turn|>\n")

    if add_generation_prompt:
        chunks.append("<|turn>model\n")
        chunks.append("<|channel>thought\n<channel|>")
    return "".join(chunks)


def _content_text(message: NativeMessage) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return " ".join(parts)
    return ""


def _render_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return ""
    arguments = _arguments_mapping(function.get("arguments"))
    return f"<|tool_call>call:{name}{{{_format_mapping(arguments)}}}<tool_call|>"


def _render_tool_response(message: NativeMessage) -> str:
    name = message.get("name")
    if not isinstance(name, str) or not name:
        name = "tool"
    content = _content_text(message)
    return f"<|tool_response>response:{name}{{value:{_format_argument(content)}}}<tool_response|>"


def _render_tools(tools: list[dict[str, Any]]) -> str:
    functions: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            functions.append(function)
    if not functions:
        return ""
    return "".join(f"<|tool>{_format_function_declaration(function)}<tool|>" for function in functions)


def _format_function_declaration(function: dict[str, Any]) -> str:
    name = function["name"]
    description = function.get("description", "")
    params = function.get("parameters")
    result = f'declaration:{name}{{description:{_format_argument(str(description))}'
    if isinstance(params, dict):
        result += ",parameters:{" + _format_parameters_object(params) + "}"
    return result + "}"


def _format_parameters_object(params: dict[str, Any]) -> str:
    parts: list[str] = []
    properties = params.get("properties")
    if isinstance(properties, dict) and properties:
        parts.append("properties:{" + _format_parameter_properties(properties) + "}")
    required = params.get("required")
    if isinstance(required, list) and required:
        parts.append("required:[" + ",".join(_format_argument(item) for item in required if isinstance(item, str)) + "]")
    param_type = params.get("type")
    if isinstance(param_type, str) and param_type:
        parts.append("type:" + _format_argument(param_type.upper()))
    return ",".join(parts)


def _format_parameter_properties(properties: dict[str, Any]) -> str:
    rendered: list[str] = []
    for key in sorted(properties):
        value = properties[key]
        if not isinstance(value, dict):
            continue
        fields: list[str] = []
        description = value.get("description")
        if isinstance(description, str) and description:
            fields.append("description:" + _format_argument(description))
        value_type = value.get("type")
        if isinstance(value_type, str) and value_type:
            fields.append("type:" + _format_argument(value_type.upper()))
        if fields:
            rendered.append(f"{key}:{{{','.join(fields)}}}")
    return ",".join(rendered)


def _arguments_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def _format_mapping(mapping: dict[str, Any]) -> str:
    return ",".join(f"{key}:{_format_argument(mapping[key])}" for key in sorted(mapping))


def _format_argument(value: object) -> str:
    if isinstance(value, str):
        return f'<|"|>{value}<|"|>'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "{" + _format_mapping(value) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_format_argument(item) for item in value) + "]"
    if value is None:
        return "null"
    return str(value)


def _strip_thinking(text: str) -> str:
    result = ""
    for part in text.split("<channel|>"):
        if "<|channel>" in part:
            result += part.split("<|channel>", 1)[0]
        else:
            result += part
    return result
