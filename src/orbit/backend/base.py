from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


Message = dict[str, Any]


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str | None
    finish_reason: str | None
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    prompt_tokens_per_second: float | None
    generation_tokens_per_second: float | None


@dataclass(frozen=True)
class StreamProgress:
    phase: str
    current: int
    total: int
    percent: int


@dataclass(frozen=True)
class ModelInfo:
    id: str | None
    capabilities: tuple[str, ...]
    context_length: int | None
    parameter_count: int | None
    size_bytes: int | None


class ChatBackend(Protocol):
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        ...

    def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None],
        on_progress: Callable[[StreamProgress], None] | None = None,
    ) -> ChatResult:
        ...
