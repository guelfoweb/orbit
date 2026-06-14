from __future__ import annotations

from dataclasses import dataclass


FALLBACK_PREFILL_TOKENS_PER_SECOND = 12.0
MIN_PREFILL_TOKENS_PER_SECOND = 4.0
MAX_PREFILL_TOKENS_PER_SECOND = 80.0
EMA_PREVIOUS_WEIGHT = 0.8
EMA_OBSERVED_WEIGHT = 0.2


@dataclass
class PrefillEstimator:
    """In-memory UI-only prefill rate estimator."""

    rate: float = FALLBACK_PREFILL_TOKENS_PER_SECOND

    def estimate_seconds(self, tokens: int) -> float | None:
        if tokens <= 0 or self.rate <= 0:
            return None
        return tokens / self.rate

    def update(self, *, prompt_tokens: int | None, prompt_tokens_per_second: float | None) -> None:
        del prompt_tokens
        if prompt_tokens_per_second is None or prompt_tokens_per_second <= 0:
            return
        observed = _clamp_rate(prompt_tokens_per_second)
        self.rate = _clamp_rate((self.rate * EMA_PREVIOUS_WEIGHT) + (observed * EMA_OBSERVED_WEIGHT))


def _clamp_rate(rate: float) -> float:
    return max(MIN_PREFILL_TOKENS_PER_SECOND, min(rate, MAX_PREFILL_TOKENS_PER_SECOND))
