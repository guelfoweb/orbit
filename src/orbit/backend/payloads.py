from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Message

@dataclass(frozen=True)
class ChatPayloadOptions:
    model: str
    messages: list[Message]
    temperature: float
    max_tokens: int
    thinking: bool = False
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    cache_prompt: bool = True
    route_prefix_anchor: bool = False
    allow_mtp_experimental: bool | None = None


def build_chat_payload(options: ChatPayloadOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": options.model,
        "messages": [_message_payload(message) for message in options.messages],
        "temperature": options.temperature,
        "max_tokens": options.max_tokens,
        "cache_prompt": options.cache_prompt,
        "thinking": options.thinking,
        "chat_template_kwargs": {"enable_thinking": options.thinking},
    }
    if options.stream:
        payload["stream"] = True
    if options.route_prefix_anchor:
        payload["route_prefix_anchor"] = True
    if options.allow_mtp_experimental is not None:
        payload["allow_mtp_experimental"] = options.allow_mtp_experimental
    if options.tools:
        payload["tools"] = options.tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
        payload["parse_tool_calls"] = True
    return payload


def _message_payload(message: Message) -> Message:
    role = message.get("role")
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.get("tool_call_id", ""),
            "name": message.get("name", ""),
            "content": message.get("content", ""),
        }
    payload: Message = {
        "role": role,
        "content": message.get("content", ""),
    }
    if role == "assistant" and isinstance(message.get("tool_calls"), list):
        payload["tool_calls"] = message["tool_calls"]
    return payload
