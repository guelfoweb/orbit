from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


NativePhase = Literal["load", "prefill", "generation"]


@dataclass(frozen=True)
class NativeProgress:
    phase: NativePhase
    current: int
    total: int
    evaluated_current: int | None = None
    evaluated_total: int | None = None
    cached_tokens: int | None = None
    elapsed_seconds: float | None = None
    tokens_per_second: float | None = None

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return min(100, max(0, int((self.current / self.total) * 100)))


@dataclass(frozen=True)
class NativeTimings:
    prompt_tokens: int
    output_tokens: int
    reused_prompt_tokens: int
    evaluated_prompt_tokens: int
    prefill_ms: float
    generation_ms: float
    cancelled: bool = False


@dataclass(frozen=True)
class NativeCompletion:
    content: str
    timings: NativeTimings
    stopped_by_stop: bool = False
    completed_after_thought: bool = False
    reasoning_content: str = ""
    reasoning_tokens: int = 0
    tool_calls: tuple[dict[str, Any], ...] = ()
