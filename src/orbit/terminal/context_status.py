from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbit.runtime.session_memory import MEMORY_MARKER, DEFAULT_CONTEXT_TOKENS, estimate_text_tokens


MESSAGE_OVERHEAD_TOKENS = 4


@dataclass(frozen=True)
class ContextBreakdown:
    system: int = 0
    memory: int = 0
    user: int = 0
    assistant: int = 0
    tool_result: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.system + self.memory + self.user + self.assistant + self.tool_result + self.other


@dataclass(frozen=True)
class MessageBreakdown:
    system: int = 0
    memory: int = 0
    user: int = 0
    assistant: int = 0
    tool_result: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.system + self.memory + self.user + self.assistant + self.tool_result + self.other


def context_status_text(messages: list[dict[str, object]], *, context_tokens: int | None) -> str:
    window = context_tokens or DEFAULT_CONTEXT_TOKENS
    breakdown = estimate_context_breakdown(messages)
    message_breakdown = count_messages_by_type(messages)
    total = breakdown.total
    lines = [
        "Context",
        "-------",
        f"window: {window}",
        f"estimated_total: {total}",
        f"usage: {_percent(total, window)}",
        f"messages: {message_breakdown.total}",
        "",
        "Token estimate",
        "--------------",
        _line("system", breakdown.system, total),
        _line("memory", breakdown.memory, total),
        _line("user", breakdown.user, total),
        _line("assistant", breakdown.assistant, total),
        _line("tool_result", breakdown.tool_result, total),
        _line("other", breakdown.other, total),
        "",
        "Message count",
        "-------------",
        f"system: {message_breakdown.system}",
        f"memory: {message_breakdown.memory}",
        f"user: {message_breakdown.user}",
        f"assistant: {message_breakdown.assistant}",
        f"tool_result: {message_breakdown.tool_result}",
        f"other: {message_breakdown.other}",
        "",
        "Recommendation",
        "--------------",
        context_recommendation(breakdown, window=window),
    ]
    return "\n".join(lines)


def estimate_context_breakdown(messages: list[dict[str, object]]) -> ContextBreakdown:
    buckets = {
        "system": 0,
        "memory": 0,
        "user": 0,
        "assistant": 0,
        "tool_result": 0,
        "other": 0,
    }
    for message in messages:
        bucket = _message_bucket(message)
        buckets[bucket] += _message_cost(message)
    return ContextBreakdown(**buckets)


def count_messages_by_type(messages: list[dict[str, object]]) -> MessageBreakdown:
    buckets = {
        "system": 0,
        "memory": 0,
        "user": 0,
        "assistant": 0,
        "tool_result": 0,
        "other": 0,
    }
    for message in messages:
        buckets[_message_bucket(message)] += 1
    return MessageBreakdown(**buckets)


def _message_bucket(message: dict[str, object]) -> str:
    role = message.get("role")
    content = message.get("content")
    if role == "system" and isinstance(content, str) and content.startswith(f"{MEMORY_MARKER} "):
        return "memory"
    if role == "system":
        return "system"
    if role == "user":
        return "user"
    if role == "assistant":
        return "assistant"
    if role == "tool":
        return "tool_result"
    return "other"


def _message_cost(message: dict[str, object]) -> int:
    return estimate_text_tokens(_message_to_text(message)) + MESSAGE_OVERHEAD_TOKENS


def _message_to_text(message: dict[str, object]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _content_list_to_text(content)
    if content is None:
        return ""
    return str(content)


def _content_list_to_text(content: list[Any]) -> str:
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif item.get("type") == "image_url":
            parts.append("[image]")
        elif item.get("type") == "input_audio":
            parts.append("[audio]")
    return "\n".join(parts)


def _line(label: str, value: int, total: int) -> str:
    return f"{label}: {value} ({_percent(value, total)})"


def _percent(value: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{(value / total) * 100:.0f}%"


def context_recommendation(breakdown: ContextBreakdown, *, window: int) -> str:
    total = breakdown.total
    if total <= 0 or window <= 0:
        return "context is empty"
    usage = total / window
    tool_ratio = breakdown.tool_result / total if total else 0.0
    if usage >= 0.85:
        if tool_ratio >= 0.50:
            return "memory refresh is near; consider /compact tools because tool results dominate"
        return "memory refresh is near; consider /compact or /reset before starting a different task"
    if usage >= 0.70:
        if tool_ratio >= 0.50:
            return "context pressure is high; consider /compact tools"
        return "context pressure is high; consider /reset for a new task"
    if usage >= 0.50:
        if tool_ratio >= 0.60:
            return "context pressure is moderate; consider /compact tools if the task is changing"
        return "context pressure is moderate; continue if this is the same task"
    return "context is healthy"
