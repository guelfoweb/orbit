from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from orbit.backend.base import ChatBackend, Message
from orbit.runtime.session_memory import estimate_text_tokens


COMPACTION_MARKER = "[Compacted tool result]"
ORIGINAL_TOOL_CONTENT_KEY = "_orbit_original_tool_content"
COMPACTION_META_KEY = "_orbit_tool_compaction"
MIN_TOOL_RESULT_TOKENS = 200
MIN_TOOL_RESULT_AGE_MESSAGES = 2
MAX_TOOL_RESULTS_PER_RUN = 2
SUMMARY_MAX_TOKENS = 256


@dataclass(frozen=True)
class ToolResultCandidate:
    index: int
    tool: str
    age_messages: int
    estimated_tokens: int


@dataclass(frozen=True)
class ToolResultCompactionItem:
    tool: str
    age_messages: int
    before_tokens: int
    after_tokens: int
    saved_tokens: int
    changed: bool
    reason: str


@dataclass(frozen=True)
class ToolResultCompactionReport:
    candidates: list[ToolResultCandidate]
    items: list[ToolResultCompactionItem]

    @property
    def changed(self) -> bool:
        return any(item.changed for item in self.items)

    @property
    def before_tokens(self) -> int:
        return sum(item.before_tokens for item in self.items)

    @property
    def after_tokens(self) -> int:
        return sum(item.after_tokens for item in self.items)

    @property
    def saved_tokens(self) -> int:
        return sum(item.saved_tokens for item in self.items)


def compact_tool_results(
    messages: list[Message],
    *,
    backend: ChatBackend,
    temperature: float,
) -> ToolResultCompactionReport:
    candidates = find_tool_result_candidates(messages)
    items: list[ToolResultCompactionItem] = []
    for candidate in candidates[:MAX_TOOL_RESULTS_PER_RUN]:
        message = messages[candidate.index]
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            items.append(_unchanged(candidate, "empty-tool-result"))
            continue
        summary = _generate_tool_result_summary(
            backend=backend,
            temperature=temperature,
            tool=candidate.tool,
            content=content,
        )
        if not summary:
            items.append(_unchanged(candidate, "summary-empty"))
            continue
        compacted = _compacted_content(candidate.tool, summary)
        after_tokens = estimate_text_tokens(compacted)
        if after_tokens >= candidate.estimated_tokens:
            items.append(_unchanged(candidate, "summary-not-smaller", after_tokens=after_tokens))
            continue
        message[ORIGINAL_TOOL_CONTENT_KEY] = content
        message[COMPACTION_META_KEY] = {
            "tool": candidate.tool,
            "age_messages": candidate.age_messages,
            "before_tokens": candidate.estimated_tokens,
            "after_tokens": after_tokens,
            "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16],
        }
        message["content"] = compacted
        items.append(
            ToolResultCompactionItem(
                tool=candidate.tool,
                age_messages=candidate.age_messages,
                before_tokens=candidate.estimated_tokens,
                after_tokens=after_tokens,
                saved_tokens=max(0, candidate.estimated_tokens - after_tokens),
                changed=True,
                reason="compacted",
            )
        )
    return ToolResultCompactionReport(candidates=candidates, items=items)


def find_tool_result_candidates(messages: list[Message]) -> list[ToolResultCandidate]:
    candidates: list[ToolResultCandidate] = []
    last_index = len(messages) - 1
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or ORIGINAL_TOOL_CONTENT_KEY in message:
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        tokens = estimate_text_tokens(content)
        age = max(0, last_index - index)
        if tokens < MIN_TOOL_RESULT_TOKENS or age < MIN_TOOL_RESULT_AGE_MESSAGES:
            continue
        tool = message.get("name")
        candidates.append(
            ToolResultCandidate(
                index=index,
                tool=tool if isinstance(tool, str) and tool else "unknown",
                age_messages=age,
                estimated_tokens=tokens,
            )
        )
    return sorted(candidates, key=lambda item: item.estimated_tokens, reverse=True)


def persistent_messages(messages: list[Message]) -> list[Message]:
    expanded: list[Message] = []
    for message in messages:
        item = dict(message)
        original = item.pop(ORIGINAL_TOOL_CONTENT_KEY, None)
        item.pop(COMPACTION_META_KEY, None)
        if item.get("role") == "tool" and isinstance(original, str):
            item["content"] = original
        expanded.append(item)
    return expanded


def _generate_tool_result_summary(
    *,
    backend: ChatBackend,
    temperature: float,
    tool: str,
    content: str,
) -> str:
    prompt = "\n".join(
        [
            "Create a durable compact representation of this tool result.",
            "Preserve facts, numbers, paths, URLs, filenames, errors, and conclusions.",
            "Do not invent.",
            "Do not solve the user task.",
            "Return only the compact representation.",
            "",
            f"Tool: {tool}",
            "Tool result:",
            content,
        ]
    )
    try:
        result = backend.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=SUMMARY_MAX_TOKENS,
            tools=None,
        )
    except Exception:
        return ""
    if result.tool_calls:
        return ""
    return result.content.strip()


def _compacted_content(tool: str, summary: str) -> str:
    return "\n".join(
        [
            f"{COMPACTION_MARKER}: {tool}",
            "Model-generated durable summary:",
            summary.strip(),
            "Original verbatim tool result is preserved in session storage.",
        ]
    )


def _unchanged(candidate: ToolResultCandidate, reason: str, *, after_tokens: int | None = None) -> ToolResultCompactionItem:
    return ToolResultCompactionItem(
        tool=candidate.tool,
        age_messages=candidate.age_messages,
        before_tokens=candidate.estimated_tokens,
        after_tokens=after_tokens if after_tokens is not None else candidate.estimated_tokens,
        saved_tokens=0,
        changed=False,
        reason=reason,
    )
