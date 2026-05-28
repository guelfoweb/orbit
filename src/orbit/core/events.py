from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class OrbitEvent(Protocol):
    pass


class EventSink(Protocol):
    def __call__(self, event: OrbitEvent) -> None:
        ...


@dataclass(frozen=True)
class ModelRequestEvent:
    loop: int


@dataclass(frozen=True)
class ToolRouteEvent:
    loop: int
    intent: str
    categories: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ThinkingStartEvent:
    loop: int


@dataclass(frozen=True)
class ThinkingChunkEvent:
    loop: int
    text: str


@dataclass(frozen=True)
class ThinkingEndEvent:
    loop: int


@dataclass(frozen=True)
class ThinkingUnavailableEvent:
    loop: int
    detail: str


@dataclass(frozen=True)
class EmptyReplyRetryEvent:
    loop: int


@dataclass(frozen=True)
class RepeatedToolRetryEvent:
    loop: int
    detail: str | None = None


@dataclass(frozen=True)
class ToolCallEvent:
    loop: int
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultEvent:
    loop: int
    name: str
    ok: bool
    error: str | None = None
    returncode: int | None = None
    stderr: str | None = None
    stdout: str | None = None
    elapsed_ms: float | None = None


@dataclass(frozen=True)
class SessionAutoCompactEvent:
    level: str | None
    score: float
    reason: str | None
    session_messages: int
    estimated_prompt_tokens: int


@dataclass(frozen=True)
class ToolResultCompactEvent:
    level: str | None
    reason: str | None
    session_messages: int
    estimated_prompt_tokens: int
    tool_name: str | None = None


@dataclass(frozen=True)
class DebugTimingEvent:
    phase: str
    elapsed_ms: float
    detail: str | None = None
