from __future__ import annotations

from dataclasses import dataclass
import re


FINAL_MARKERS = (
    "**final answer:**",
    "final answer:",
    "the final answer is:",
    "the final answer:",
)
REASONING_PREFIXES = (
    "### reasoning",
    "## reasoning",
    "# reasoning",
    "reasoning:",
    "plan:",
)

_REASONING_META_KEYWORDS = (
    "constraint",
    "constraints",
    "scenario",
    "user's request",
    "user request",
    "looking at the prompt",
    "drafting the response",
    "let's assume",
)


def contains_control_channel_markup(content: str) -> bool:
    return "<|channel>" in content or "<channel|>" in content


def has_open_thought_channel(content: str) -> bool:
    if "<|channel>thought" not in content:
        return False
    tail = content.split("<|channel>thought", 1)[1]
    return "<channel|>" not in tail


def last_assistant_has_open_reasoning(messages: list[dict[str, object]]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False
        return has_open_thought_channel(content) or looks_like_reasoning_without_final(content, thinking_enabled=True)
    return False


def looks_like_reasoning_without_final(content: str, *, thinking_enabled: bool) -> bool:
    if not thinking_enabled:
        return has_open_thought_channel(content)
    stripped = content.strip().lower()
    if not stripped:
        return False
    if has_open_thought_channel(content):
        return True
    if any(marker in stripped for marker in FINAL_MARKERS):
        return False
    return stripped.startswith(REASONING_PREFIXES)


def looks_like_truncated_reasoning_prelude(content: str) -> bool:
    stripped = content.strip().lower()
    if not stripped:
        return False
    if any(marker in stripped for marker in FINAL_MARKERS):
        return False
    if stripped.startswith(REASONING_PREFIXES):
        return True
    preview = stripped[:600]
    if any(keyword in preview for keyword in _REASONING_META_KEYWORDS):
        return True
    lines = [line.strip() for line in preview.splitlines() if line.strip()]
    bullet_lines = sum(1 for line in lines[:8] if re.match(r"(?:[-*•]\s|\d+[.)]\s)", line))
    return bullet_lines >= 2


def looks_like_incomplete_tool_answer(content: str) -> bool:
    text = content.strip()
    if len(text) < 24:
        return False
    if text.endswith((".", "!", "?", "`", "\"", "'", ")", "]")):
        return False
    return True


@dataclass(frozen=True)
class ThinkingMode:
    enabled: bool = False

    def should_stream_tool_plan(self, *, has_delta_sink: bool, backend_supports_streaming: bool) -> bool:
        return self.enabled and has_delta_sink and backend_supports_streaming

    def continuation_kind_for(self, *, content: str, finish_reason: str | None) -> str | None:
        if looks_like_reasoning_without_final(content, thinking_enabled=self.enabled):
            return "thinking"
        if self.enabled and finish_reason == "length" and looks_like_truncated_reasoning_prelude(content):
            return "thinking"
        if finish_reason == "length":
            return "final_answer"
        return None
