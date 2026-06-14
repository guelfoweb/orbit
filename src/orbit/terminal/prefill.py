from __future__ import annotations

from orbit.backend.base import Message
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT
from orbit.runtime.session_memory import estimate_message_tokens, estimate_text_tokens
from orbit.terminal.prefill_estimator import FALLBACK_PREFILL_TOKENS_PER_SECOND


DEFAULT_PREFILL_TOKENS_PER_SECOND = FALLBACK_PREFILL_TOKENS_PER_SECOND
MIN_PREFILL_ESTIMATE_SECONDS = 2.0


def estimate_prefill_seconds(
    messages: list[Message],
    prompt: str,
    *,
    prompt_tokens_per_second: float | None = None,
    system_prompt: str = ROUTE_SYSTEM_PROMPT,
) -> float | None:
    tokens = estimate_prefill_tokens(messages, prompt, system_prompt=system_prompt)
    rate = prompt_tokens_per_second or DEFAULT_PREFILL_TOKENS_PER_SECOND
    return estimate_prefill_seconds_for_tokens(tokens, rate=rate)


def estimate_prefill_seconds_for_tokens(tokens: int, *, rate: float) -> float | None:
    if rate <= 0:
        return None
    seconds = tokens / rate
    if seconds < MIN_PREFILL_ESTIMATE_SECONDS:
        return None
    return seconds


def estimate_prefill_tokens(messages: list[Message], prompt: str, *, system_prompt: str = ROUTE_SYSTEM_PROMPT) -> int:
    return (
        estimate_text_tokens(system_prompt)
        + estimate_message_tokens(messages)
        + estimate_text_tokens(prompt)
        + 8
    )


def estimate_prefill_tokens_after_tool_result(messages: list[Message], tool_result_content: str) -> int:
    return (
        estimate_prefill_tokens(messages, "", system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)
        + estimate_text_tokens(tool_result_content)
        + 4
    )
