from __future__ import annotations

from orbit.backend.base import Message
from orbit.runtime.tool_calls import tool_call_id
from orbit.runtime.tools import ToolResult


def assistant_tool_call_message(content: str, tool_calls: list[dict[str, object]]) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def tool_result_message(tool_call: dict[str, object], tool_result: ToolResult) -> Message:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id(tool_call),
        "name": tool_result.name,
        "content": tool_result.content,
    }
