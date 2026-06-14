from __future__ import annotations

from dataclasses import dataclass


FALLBACK_PREFILL_TOKENS_PER_SECOND = 12.0
MIN_PREFILL_TOKENS_PER_SECOND = 4.0
MAX_PREFILL_TOKENS_PER_SECOND = 80.0
EMA_PREVIOUS_WEIGHT = 0.8
EMA_OBSERVED_WEIGHT = 0.2
DEFAULT_PREFILL_PROFILE = "default"
CHAT_PREFILL_PROFILE = "chat"
TOOL_PREFILL_PROFILE = "tool"
FINAL_FROM_TOOL_PREFILL_PROFILE = "final_from_tool"


def prefill_profile_for_phase(phase: str | None) -> str:
    if phase in {"chat_final", "chat_final_retry"}:
        return CHAT_PREFILL_PROFILE
    if phase in {"final_from_tool", "final_from_tool_retry"}:
        return FINAL_FROM_TOOL_PREFILL_PROFILE
    if phase in {"route", "tool_call", "tool_call_retry"}:
        return TOOL_PREFILL_PROFILE
    return DEFAULT_PREFILL_PROFILE


@dataclass
class PrefillEstimator:
    """In-memory UI-only prefill rate estimator."""

    rate: float = FALLBACK_PREFILL_TOKENS_PER_SECOND

    def __post_init__(self) -> None:
        self._rates: dict[str, float] = {DEFAULT_PREFILL_PROFILE: _clamp_rate(self.rate)}

    def estimate_seconds(self, tokens: int, *, profile: str = DEFAULT_PREFILL_PROFILE) -> float | None:
        rate = self.rate_for(profile)
        if tokens <= 0 or rate <= 0:
            return None
        return tokens / rate

    def update(
        self,
        *,
        prompt_tokens: int | None,
        prompt_tokens_per_second: float | None,
        profile: str = DEFAULT_PREFILL_PROFILE,
    ) -> None:
        del prompt_tokens
        if prompt_tokens_per_second is None or prompt_tokens_per_second <= 0:
            return
        observed = _clamp_rate(prompt_tokens_per_second)
        previous = self.rate_for(profile)
        self._rates[profile] = _clamp_rate((previous * EMA_PREVIOUS_WEIGHT) + (observed * EMA_OBSERVED_WEIGHT))
        if profile == DEFAULT_PREFILL_PROFILE:
            self.rate = self._rates[profile]

    def rate_for(self, profile: str) -> float:
        return self._rates.get(profile) or self._rates.get(DEFAULT_PREFILL_PROFILE) or FALLBACK_PREFILL_TOKENS_PER_SECOND


def _clamp_rate(rate: float) -> float:
    return max(MIN_PREFILL_TOKENS_PER_SECOND, min(rate, MAX_PREFILL_TOKENS_PER_SECOND))
