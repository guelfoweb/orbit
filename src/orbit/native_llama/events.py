from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NativePhase = Literal["load", "prefill", "generation"]


@dataclass(frozen=True)
class NativeProgress:
    phase: NativePhase
    current: int
    total: int

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
