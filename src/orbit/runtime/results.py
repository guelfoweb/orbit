from __future__ import annotations

from orbit.backend import ChatResult


def error_result(content: str, previous: ChatResult) -> ChatResult:
    return ChatResult(
        content=content,
        model=previous.model,
        finish_reason="error",
        tool_calls=[],
        prompt_tokens=previous.prompt_tokens,
        completion_tokens=previous.completion_tokens,
        cached_tokens=previous.cached_tokens,
        prompt_tokens_per_second=previous.prompt_tokens_per_second,
        generation_tokens_per_second=previous.generation_tokens_per_second,
    )
