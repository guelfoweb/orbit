from __future__ import annotations

from dataclasses import dataclass
from shutil import get_terminal_size
from typing import Sequence

from orbit.backend.base import ChatResult, StreamPromptMetrics
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
    failed_calls: int = 0
    # True when an attempt was made whose token metrics never reached us, so
    # the sums below are real but not the whole story. Kept separate from the
    # counters themselves: a number we do know stays knowable even when the
    # total it belongs to is incomplete.
    usage_incomplete: bool = False


@dataclass
class TokenUsageAccumulator:
    model_calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int | None = 0
    evaluated_tokens: int | None = 0
    cached_tokens: int | None = 0
    completion_tokens: int | None = 0
    usage_incomplete: bool = False

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

    def add_failed_call(self) -> None:
        """Record an attempt that raised, without discarding what is known.

        The attempt reached the backend, so it counts; but its token metrics
        never arrived, so the running totals are now short by an unknown
        amount. Zeroing them would invent a number, and clearing them threw
        away figures that had already been measured -- a recovered failure
        used to erase an entire session's accounting. The totals are kept and
        marked incomplete instead, which is the only honest description.
        """
        self.model_calls += 1
        self.failed_calls += 1
        self.usage_incomplete = True

    def add_aborted_call(self, prompt_metrics: StreamPromptMetrics | None = None) -> None:
        """Record an attempt the client stopped on purpose.

        The request reached the model and real work happened, so it counts as
        a model call. What this must not do is claim a failure: nothing went
        wrong, and reporting one sends a reader looking for a fault that does
        not exist.

        Prefill finishes before the first token, so when the stream carried
        those counts they are final and get accumulated here. Decode never
        reported, so the totals stay marked incomplete either way -- known
        numbers are worth keeping even when the whole picture is not.
        """
        self.model_calls += 1
        self.usage_incomplete = True
        if prompt_metrics is None:
            return
        prompt = getattr(prompt_metrics, "prompt_tokens", None)
        cached = getattr(prompt_metrics, "cached_tokens", None)
        if prompt is None:
            return
        self.prompt_tokens = _add_optional_metric(self.prompt_tokens, prompt)
        if (
            self.cached_tokens is None
            or self.evaluated_tokens is None
            or cached is None
            or not 0 <= cached <= prompt
        ):
            self.cached_tokens = None
            self.evaluated_tokens = None
        else:
            self.cached_tokens += cached
            self.evaluated_tokens += prompt - cached

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
            failed_calls=self.failed_calls,
            usage_incomplete=self.usage_incomplete,
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
        failed_calls=0,
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
    usage = turn_token_usage or _single_call_usage(result)
    columns = max(20, terminal_columns or get_terminal_size((80, 20)).columns)
    summary = []
    if elapsed_seconds is not None:
        summary.append(format_elapsed(elapsed_seconds))
    if usage is not None:
        summary.append(f"{usage.model_calls} calls")
    if result.finish_reason:
        summary.append(result.finish_reason)

    lines = _pack_metric_parts(summary or ["no turn metrics"], columns=columns)
    lines.extend(_token_metric_lines(usage, columns=columns))

    rates = []
    if result.prompt_tokens_per_second is not None:
        rates.append(f"{result.prompt_tokens_per_second:.1f} tok/s prefill")
    if result.generation_tokens_per_second is not None:
        rates.append(f"{result.generation_tokens_per_second:.1f} tok/s decode")
    if rates:
        lines.extend(_pack_metric_parts(rates, columns=columns, prefix="last call: "))

    if estimated_context_tokens is not None and context_tokens is not None and context_tokens > 0:
        context = f"{estimated_context_tokens}/{context_tokens} ({_context_percentage(estimated_context_tokens, context_tokens)})"
        pressure = _context_pressure(estimated_context_tokens, context_tokens)
        context_parts = [context]
        if pressure:
            context_parts.append(f"pressure {pressure}")
        lines.extend(_pack_metric_parts(context_parts, columns=columns, prefix="context: "))
    if usage is not None and usage.failed_calls:
        lines.append(
            f"warning: {usage.failed_calls} failed attempt(s); token totals exclude them"
        )
    return "\n".join(lines)


def _context_percentage(used: int, total: int) -> str:
    if used == 0:
        return "0%"
    percentage = (used / total) * 100
    if 0 < percentage < 1:
        return "<1%"
    return f"{percentage:.0f}%"


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


def _single_call_usage(result: ChatResult) -> TurnTokenUsage:
    evaluated = None
    cached = result.cached_tokens
    if (
        result.prompt_tokens is not None
        and cached is not None
        and 0 <= cached <= result.prompt_tokens
    ):
        evaluated = result.prompt_tokens - cached
    else:
        cached = None
    return TurnTokenUsage(
        model_calls=1,
        prompt_tokens=result.prompt_tokens,
        evaluated_tokens=evaluated,
        cached_tokens=cached,
        completion_tokens=result.completion_tokens,
    )


def _token_metric_lines(usage: TurnTokenUsage | None, *, columns: int) -> list[str]:
    if usage is None:
        return []
    parts = []
    for value, label in (
        (usage.prompt_tokens, "in"),
        (usage.evaluated_tokens, "eval"),
        (usage.cached_tokens, "cache"),
        (usage.completion_tokens, "out"),
    ):
        if value is not None:
            parts.append(f"{value:,} {label}")
    if parts and usage.usage_incomplete:
        # The figures are real; they just are not the whole turn.
        parts.append("(partial)")
    return _pack_metric_parts(parts or ["unavailable"], columns=columns, prefix="tokens: ")


def _pack_metric_parts(parts: list[str], *, columns: int, prefix: str = "") -> list[str]:
    lines: list[str] = []
    current = prefix
    for part in parts:
        candidate = f"{current} · {part}" if current and current != prefix else f"{current}{part}"
        if current and current != prefix and len(candidate) > columns:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


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
    if usage.usage_incomplete:
        parts.append("totals: partial")
    if usage.evaluated_tokens is not None and usage.completion_tokens is not None:
        work = usage.evaluated_tokens + usage.completion_tokens
        parts.append(f"work: {work} ({usage.evaluated_tokens} prefill + {usage.completion_tokens} decode)")
    if usage.prompt_tokens and usage.cached_tokens is not None:
        ratio = (usage.cached_tokens / usage.prompt_tokens) * 100
        parts.append(f"cache: {usage.cached_tokens} ({ratio:.0f}%)")
    parts.append(f"calls: {usage.model_calls}")
    if usage.failed_calls:
        parts.append(f"failed attempts: {usage.failed_calls}")
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
