from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


Message = dict[str, Any]


@dataclass(frozen=True)
class StreamPromptMetrics:
    """What prefill measured, known before any token is generated."""

    prompt_tokens: int | None = None
    evaluated_tokens: int | None = None
    cached_tokens: int | None = None


class RecoverableBackendError(RuntimeError):
    """A backend failure that ends the current request, not the session.

    The distinction matters because the two are handled oppositely: a
    recoverable failure leaves the analyst their session and evidence, while an
    unexpected `RuntimeError` is a bug and must propagate so the process tears
    down and releases its workspace. Naming the recoverable case here lets a
    runtime catch exactly it without importing upward from the backend layer,
    and without widening to bare `RuntimeError` and swallowing real crashes.
    """


class StreamConsumerAbort(Exception):
    """A stream consumer stopped its own call on purpose; the backend is fine.

    A delta callback may decide mid-stream that it has seen enough and raise to
    stop generating early. That exception travels out through the backend call
    exactly like a transport error, but it means the opposite thing: the
    request reached the model and the model was answering. Backends recognise
    this marker so they do not report a healthy call as a failed attempt. The
    exception still propagates -- only the accounting treats it differently.

    `prompt_metrics` carries whatever prefill had already measured when the
    consumer stopped. Prefill finishes before the first token, so its token
    counts are final by then and are simply lost if the terminal metrics event
    is never read. The streaming layer attaches them on the way out; nothing
    reads them for control flow.
    """

    prompt_metrics: "StreamPromptMetrics | None" = None


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
    reasoning_content: str = ""
    backend_ttft_ms: float | None = None
    stream_ttft_ms: float | None = None


@dataclass(frozen=True)
class StreamProgress:
    phase: str
    current: int
    total: int
    percent: int
    evaluated_current: int | None = None
    evaluated_total: int | None = None
    cached_tokens: int | None = None
    elapsed_seconds: float | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str | None
    capabilities: tuple[str, ...]
    context_length: int | None
    parameter_count: int | None
    size_bytes: int | None


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    context_tokens: int | None
    rendered_hash: str | None = None
    token_hash: str | None = None


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

    def artifact_content_stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_delta: Callable[[str], None],
        on_progress: Callable[[StreamProgress], None] | None = None,
    ) -> ChatResult:
        ...
