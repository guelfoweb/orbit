from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetProfile:
    soft_messages: int
    soft_tokens: int
    hard_messages: int
    hard_tokens: int
    soft_ctx_ratio: float = 0.55
    hard_ctx_ratio: float = 0.72


@dataclass(frozen=True)
class BudgetPressure:
    session_messages: int
    estimated_prompt_tokens: int
    score: float
    level: str | None
    should_compact: bool
    reason: str | None
    overflow_tokens: int


DEFAULT_BUDGET_PROFILE = BudgetProfile(
    soft_messages=28,
    soft_tokens=9_000,
    hard_messages=42,
    hard_tokens=14_000,
)


def profile_for_model(model_name: str | None) -> BudgetProfile:
    return DEFAULT_BUDGET_PROFILE


def evaluate_budget_pressure(
    *,
    model_name: str | None,
    session_messages: int,
    estimated_prompt_tokens: int,
    context_window: int | None = None,
) -> BudgetPressure:
    profile = profile_for_model(model_name)
    soft_tokens = profile.soft_tokens
    hard_tokens = profile.hard_tokens
    if isinstance(context_window, int) and context_window > 0:
        soft_tokens = min(soft_tokens, max(2_000, int(context_window * profile.soft_ctx_ratio)))
        hard_tokens = min(hard_tokens, max(3_000, int(context_window * profile.hard_ctx_ratio)))
    overflow_tokens = max(0, estimated_prompt_tokens - soft_tokens)
    score = max(
        session_messages / profile.soft_messages,
        estimated_prompt_tokens / soft_tokens,
    )
    level = None
    if (
        session_messages >= profile.hard_messages
        or estimated_prompt_tokens >= hard_tokens
        or score >= 1.5
    ):
        level = "hard"
    elif (
        session_messages >= profile.soft_messages
        or estimated_prompt_tokens >= soft_tokens
        or score >= 1.0
    ):
        level = "soft"
    should_compact = level is not None
    reason = None
    if should_compact:
        reason = (
            f"{level} pressure: score={score:.2f}, msg={session_messages}, "
            f"est_tokens={estimated_prompt_tokens}, "
            f"budget={profile.soft_messages}/{soft_tokens}->{profile.hard_messages}/{hard_tokens}, "
            f"overflow={overflow_tokens}"
        )
    return BudgetPressure(
        session_messages=session_messages,
        estimated_prompt_tokens=estimated_prompt_tokens,
        score=score,
        level=level,
        should_compact=should_compact,
        reason=reason,
        overflow_tokens=overflow_tokens,
    )
