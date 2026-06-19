from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionBudget:
    requested_max_tokens: int

    def internal(self, cap: int) -> int:
        return max(1, min(self.requested_max_tokens, cap))

    def user_visible(self) -> int:
        return max(1, self.requested_max_tokens)
