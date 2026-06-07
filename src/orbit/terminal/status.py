from __future__ import annotations

from orbit.backend.base import ChatResult
from orbit.runtime.session_memory import MemoryRefresh, estimate_message_tokens
from orbit.terminal.streaming import format_elapsed


def format_turn_status(
    result: ChatResult,
    *,
    elapsed_seconds: float | None = None,
    estimated_context_tokens: int | None = None,
    context_tokens: int | None = None,
) -> str:
    parts = []
    if result.model:
        parts.append(f"model: {result.model}")
    if estimated_context_tokens is not None and context_tokens is not None and context_tokens > 0:
        parts.append(f"ctx: {context_tokens} ({(estimated_context_tokens / context_tokens) * 100:.0f}%)")
    if result.prompt_tokens is not None or result.completion_tokens is not None:
        cached = f", cached {result.cached_tokens}" if result.cached_tokens is not None else ""
        parts.append(f"tks: {result.prompt_tokens}->{result.completion_tokens}{cached}")
    speeds = []
    if result.prompt_tokens_per_second is not None:
        speeds.append(f"pf {result.prompt_tokens_per_second:.1f}/s")
    if result.generation_tokens_per_second is not None:
        speeds.append(f"gen {result.generation_tokens_per_second:.1f}/s")
    if speeds:
        parts.append(" | ".join(speeds))
    if result.finish_reason:
        parts.append(f"stop: {result.finish_reason}")
    if elapsed_seconds is not None:
        parts.append(f"time: {format_elapsed(elapsed_seconds)}")
    return " | ".join(parts) if parts else "no metrics"


def estimate_context_status_tokens(messages: list[dict[str, object]]) -> int:
    return estimate_message_tokens(messages)


def format_memory_refresh(refresh: MemoryRefresh) -> str:
    saved = max(0, refresh.estimated_tokens_before - refresh.estimated_tokens_after)
    ratio = _saved_ratio(refresh.estimated_tokens_before, saved)
    parts = [
        f"memory: {refresh.estimated_tokens_before}->{refresh.estimated_tokens_after} est. tokens",
        f"saved {saved} ({ratio:.0f}%)",
        f"{refresh.elapsed_seconds:.1f}s",
    ]
    if refresh.threshold_tokens is not None and refresh.context_tokens is not None:
        parts.append(f"threshold {refresh.threshold_tokens}/{refresh.context_tokens}")
    return " | ".join(parts)


def _saved_ratio(before: int, saved: int) -> float:
    if before <= 0:
        return 0.0
    return (saved / before) * 100.0
