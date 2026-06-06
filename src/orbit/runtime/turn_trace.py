from __future__ import annotations

from dataclasses import dataclass

from orbit.backend import ChatResult


@dataclass(frozen=True)
class ModelStepMetrics:
    loop: int
    phase: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    prompt_tokens_per_second: float | None
    generation_tokens_per_second: float | None
    tool_calls: int

    @classmethod
    def from_result(cls, *, loop: int, result: ChatResult) -> ModelStepMetrics:
        tool_calls = len(result.tool_calls)
        return cls(
            loop=loop,
            phase="tool_call" if tool_calls else "final",
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            prompt_tokens_per_second=result.prompt_tokens_per_second,
            generation_tokens_per_second=result.generation_tokens_per_second,
            tool_calls=tool_calls,
        )


def cache_ratio(metrics: ModelStepMetrics) -> float | None:
    if metrics.cached_tokens is None or not metrics.prompt_tokens:
        return None
    return metrics.cached_tokens / metrics.prompt_tokens
