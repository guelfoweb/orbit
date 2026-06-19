from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from orbit.backend import ChatBackend, ChatResult
from orbit.backend.base import StreamProgress
from orbit.runtime.thinking_mode import ThinkingMode


@dataclass(frozen=True)
class ContinueController:
    backend: ChatBackend
    thinking: ThinkingMode
    merge_results: Callable[[ChatResult, ChatResult], ChatResult]

    def continue_until_settled(
        self,
        *,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        max_passes: int = 3,
    ) -> ChatResult:
        merged = self._continue_once(
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
        )
        passes = 1
        while passes < max_passes:
            continuation_kind = self.thinking.continuation_kind_for(
                content=merged.content,
                finish_reason=merged.finish_reason,
            )
            if continuation_kind is None:
                break
            continuation = self._continue_once(
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
            )
            merged = self.merge_results(merged, continuation)
            passes += 1
            if not continuation.content:
                break
        return merged

    def _continue_once(
        self,
        *,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
    ) -> ChatResult:
        if on_final_delta is None:
            return self.backend.continue_current(max_tokens=max_tokens)
        return self.backend.continue_current(
            max_tokens=max_tokens,
            on_delta=on_final_delta,
            on_progress=on_progress,
        )
