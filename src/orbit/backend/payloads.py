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
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    cache_prompt: bool = True


def build_chat_payload(options: ChatPayloadOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": options.model,
        "messages": options.messages,
        "temperature": options.temperature,
        "max_tokens": options.max_tokens,
        "cache_prompt": options.cache_prompt,
    }
    if options.stream:
        payload["stream"] = True
    if options.tools:
        payload["tools"] = options.tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
        payload["parse_tool_calls"] = True
    return payload
