from __future__ import annotations

from dataclasses import dataclass
from shutil import get_terminal_size
from typing import Sequence

from orbit.backend.base import ChatResult
from orbit.runtime.kv_diag import emit_footer_metrics
from orbit.runtime.session_memory import MemoryRefresh, estimate_message_tokens
from orbit.runtime.turn_trace import ModelStepMetrics
from orbit.terminal.streaming import format_elapsed


@dataclass(frozen=True)
class TurnTokenUsage:
    model_calls: int
    prompt_tokens: int | None
    evaluated_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None


@dataclass
class TokenUsageAccumulator:
    model_calls: int = 0
    prompt_tokens: int | None = 0
    evaluated_tokens: int | None = 0
    cached_tokens: int | None = 0
    completion_tokens: int | None = 0

    def add(self, step: ModelStepMetrics) -> None:
        self._add_metrics(
            prompt_tokens=step.prompt_tokens,
            completion_tokens=step.completion_tokens,
            cached_tokens=step.cached_tokens,
        )

    def add_result(self, result: ChatResult) -> None:
        self._add_metrics(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
        )

    def _add_metrics(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
    ) -> None:
        self.model_calls += 1
        self.prompt_tokens = _add_optional_metric(self.prompt_tokens, prompt_tokens)
        self.completion_tokens = _add_optional_metric(self.completion_tokens, completion_tokens)
        if (
            self.cached_tokens is None
            or self.evaluated_tokens is None
            or prompt_tokens is None
            or cached_tokens is None
            or not 0 <= cached_tokens <= prompt_tokens
        ):
            self.cached_tokens = None
            self.evaluated_tokens = None
        else:
            self.cached_tokens += cached_tokens
            self.evaluated_tokens += prompt_tokens - cached_tokens

    def snapshot(self) -> TurnTokenUsage:
        return TurnTokenUsage(
            model_calls=self.model_calls,
            prompt_tokens=self.prompt_tokens,
            evaluated_tokens=self.evaluated_tokens,
            cached_tokens=self.cached_tokens,
            completion_tokens=self.completion_tokens,
        )


def summarize_turn_token_usage(model_steps: Sequence[ModelStepMetrics]) -> TurnTokenUsage | None:
    if not model_steps:
        return None

    prompt_values = [step.prompt_tokens for step in model_steps]
    completion_values = [step.completion_tokens for step in model_steps]
    cache_pairs = [(step.prompt_tokens, step.cached_tokens) for step in model_steps]

    prompt_tokens = sum(prompt_values) if all(value is not None for value in prompt_values) else None
    completion_tokens = sum(completion_values) if all(value is not None for value in completion_values) else None
    cache_metrics_complete = all(
        prompt is not None and cached is not None and 0 <= cached <= prompt
        for prompt, cached in cache_pairs
    )
    cached_tokens = sum(cached for _, cached in cache_pairs if cached is not None) if cache_metrics_complete else None
    evaluated_tokens = (
        sum(prompt - cached for prompt, cached in cache_pairs if prompt is not None and cached is not None)
        if cache_metrics_complete
        else None
    )
    return TurnTokenUsage(
        model_calls=len(model_steps),
        prompt_tokens=prompt_tokens,
        evaluated_tokens=evaluated_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )


def format_session_token_usage(usage: TurnTokenUsage) -> str:
    return " | ".join(_token_usage_parts(usage, prefix="session tks"))


def format_turn_status(
    result: ChatResult,
    *,
    elapsed_seconds: float | None = None,
    estimated_context_tokens: int | None = None,
    context_tokens: int | None = None,
    turn_token_usage: TurnTokenUsage | None = None,
    terminal_columns: int | None = None,
) -> str:
    emit_footer_metrics(
        result,
        elapsed_seconds=elapsed_seconds,
        estimated_context_tokens=estimated_context_tokens,
        context_tokens=context_tokens,
    )
    parts = []
    if estimated_context_tokens is not None and context_tokens is not None and context_tokens > 0:
        pressure = _context_pressure(estimated_context_tokens, context_tokens)
        parts.append(f"ctx: {estimated_context_tokens}/{context_tokens} ({(estimated_context_tokens / context_tokens) * 100:.0f}%)")
        if pressure:
            parts.append(f"pressure: {pressure}")
    prompt_tokens = result.prompt_tokens
    completion_tokens = result.completion_tokens
    cached_tokens = result.cached_tokens
    if turn_token_usage is not None:
        prompt_tokens = turn_token_usage.prompt_tokens
        completion_tokens = turn_token_usage.completion_tokens
        cached_tokens = turn_token_usage.cached_tokens
        parts.extend(_token_usage_parts(turn_token_usage, prefix="tks"))
    elif prompt_tokens is not None or completion_tokens is not None:
        cached = f", cached {cached_tokens}" if cached_tokens is not None else ""
        parts.append(f"tks: {prompt_tokens}->{completion_tokens}{cached}")
    if turn_token_usage is None and prompt_tokens and cached_tokens is not None:
        parts.append(f"cache: {(cached_tokens / prompt_tokens) * 100:.0f}%")
    header = []
    if elapsed_seconds is not None:
        header.append(f"time elapsed: {format_elapsed(elapsed_seconds)}")
    if result.prompt_tokens_per_second is not None:
        header.append(f"pf {result.prompt_tokens_per_second:.1f}/s")
    if result.generation_tokens_per_second is not None:
        header.append(f"gen {result.generation_tokens_per_second:.1f}/s")
    if result.finish_reason:
        parts.append(f"stop: {result.finish_reason}")
    details = " | ".join(parts) if parts else "no metrics"
    if not header:
        return details
    columns = terminal_columns or get_terminal_size((80, 20)).columns
    heading = f"__ {' | '.join(header)} "
    separator = heading + ("_" * max(0, columns - len(heading)))
    return f"{separator}\n{details}"


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


def _add_optional_metric(total: int | None, value: int | None) -> int | None:
    if total is None or value is None:
        return None
    return total + value


def _token_usage_parts(usage: TurnTokenUsage, *, prefix: str) -> list[str]:
    parts = []
    if usage.prompt_tokens is not None and usage.completion_tokens is not None:
        total = usage.prompt_tokens + usage.completion_tokens
        parts.append(f"{prefix}: {total} total ({usage.prompt_tokens} in + {usage.completion_tokens} out)")
    elif usage.prompt_tokens is not None:
        parts.append(f"{prefix}: input {usage.prompt_tokens}")
    elif usage.completion_tokens is not None:
        parts.append(f"{prefix}: output {usage.completion_tokens}")
    else:
        parts.append(f"{prefix}: unavailable")
    if usage.evaluated_tokens is not None and usage.completion_tokens is not None:
        work = usage.evaluated_tokens + usage.completion_tokens
        parts.append(f"work: {work} ({usage.evaluated_tokens} prefill + {usage.completion_tokens} decode)")
    if usage.prompt_tokens and usage.cached_tokens is not None:
        ratio = (usage.cached_tokens / usage.prompt_tokens) * 100
        parts.append(f"cache: {usage.cached_tokens} ({ratio:.0f}%)")
    parts.append(f"calls: {usage.model_calls}")
    return parts


def _context_pressure(estimated_context_tokens: int, context_tokens: int) -> str | None:
    ratio = estimated_context_tokens / context_tokens
    if ratio >= 0.85:
        return "memory refresh"
    if ratio >= 0.70:
        return "high | consider /compact tools"
    if ratio >= 0.50:
        return "moderate"
    return None
