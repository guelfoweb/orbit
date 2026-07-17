from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from orbit.backend.base import Message
from orbit.runtime.final_policy import classify_final_answer_completeness, contains_raw_tool_call
from orbit.runtime.thinking_mode import contains_control_channel_markup


_TECHNICAL_TAG_RE = re.compile(r"</?(?:tool_call|function|arguments?)\b", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*[{[]", re.IGNORECASE)


@dataclass(frozen=True)
class PostToolFinalReuseDecision:
    eligible: bool
    reason: str


def evaluate_post_tool_final_reuse(
    *,
    content: str,
    finish_reason: str | None,
    tool_calls: Sequence[Mapping[str, object]],
    phase: str | None,
    messages: Sequence[Message],
    tool_rounds: int,
    output_was_suppressed: bool,
    pending_internal_request: bool,
    executed_internal_tool_prompt: bool,
    shell_error_pending: bool,
    shadow_attempt_detected: bool,
) -> PostToolFinalReuseDecision:
    """Apply structural eligibility only; never interpret or rewrite model prose."""
    if phase != "post_tool_route":
        return PostToolFinalReuseDecision(False, "not_post_tool_route")
    if finish_reason != "stop":
        return PostToolFinalReuseDecision(False, "finish_reason")
    if tool_calls:
        return PostToolFinalReuseDecision(False, "tool_call_present")
    if tool_rounds < 1 or not _latest_message_is_tool(messages):
        return PostToolFinalReuseDecision(False, "no_terminal_tool_result")
    if not output_was_suppressed:
        return PostToolFinalReuseDecision(False, "output_not_suppressed")
    if pending_internal_request:
        return PostToolFinalReuseDecision(False, "pending_internal_request")
    if executed_internal_tool_prompt:
        return PostToolFinalReuseDecision(False, "internal_guard_result")
    if shell_error_pending:
        return PostToolFinalReuseDecision(False, "tool_error_pending")
    stripped = content.strip()
    if not stripped:
        return PostToolFinalReuseDecision(False, "empty_prose")
    if shadow_attempt_detected:
        return PostToolFinalReuseDecision(False, "tool_attempt_detected")
    if contains_raw_tool_call(content) or contains_control_channel_markup(content):
        return PostToolFinalReuseDecision(False, "technical_markup")
    if "<|" in content or "|>" in content:
        return PostToolFinalReuseDecision(False, "technical_markup")
    if _TECHNICAL_TAG_RE.search(content) or _looks_like_technical_json(stripped):
        return PostToolFinalReuseDecision(False, "technical_payload")
    completeness = classify_final_answer_completeness(content, messages=list(messages))
    if not completeness.is_complete:
        return PostToolFinalReuseDecision(False, f"incomplete_{completeness.status}")
    return PostToolFinalReuseDecision(True, "complete_post_tool_prose")


def _latest_message_is_tool(messages: Sequence[Message]) -> bool:
    return bool(messages) and messages[-1].get("role") == "tool"


def _looks_like_technical_json(content: str) -> bool:
    if "{" in content or "}" in content:
        return True
    return bool(content.startswith("[") or _JSON_FENCE_RE.match(content))
