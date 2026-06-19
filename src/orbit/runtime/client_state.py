from __future__ import annotations

from dataclasses import dataclass

from orbit.backend import ChatResult
from orbit.runtime.thinking_mode import ThinkingMode


@dataclass
class ClientState:
    last_finish_reason: str | None = None
    continuation_kind: str | None = None
    last_content: str = ""

    @property
    def can_continue(self) -> bool:
        return self.continuation_kind is not None

    def update_from_result(self, result: ChatResult, *, thinking: ThinkingMode) -> None:
        self.last_finish_reason = result.finish_reason
        self.last_content = result.content
        self.continuation_kind = thinking.continuation_kind_for(
            content=result.content,
            finish_reason=result.finish_reason,
        )

    def reset(self) -> None:
        self.last_finish_reason = None
        self.continuation_kind = None
        self.last_content = ""
