from __future__ import annotations

import json
from typing import Any

from orbit.backend.base import Message
from orbit.runtime.tool_calls import tool_call_id
from orbit.runtime.tools import ToolResult


def assistant_tool_call_message(content: str, tool_calls: list[dict[str, object]]) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [_safe_tool_call_for_history(tool_call) for tool_call in tool_calls]
    return message


def tool_result_message(tool_call: dict[str, object], tool_result: ToolResult) -> Message:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id(tool_call),
        "name": tool_result.name,
        "content": tool_result.content,
    }


def _safe_tool_call_for_history(tool_call: dict[str, object]) -> dict[str, Any]:
    sanitized = dict(tool_call)
    function = sanitized.get("function")
    if not isinstance(function, dict):
        return sanitized
    safe_function = dict(function)
    safe_function["arguments"] = _safe_arguments_json(safe_function.get("arguments"))
    sanitized["function"] = safe_function
    return sanitized


def _safe_arguments_json(arguments: object) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"invalid_arguments": arguments[:200]}
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        parsed = {}
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
