from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


SUMMARY_MARKER = "SESSION MEMORY SUMMARY"
KEEP_RECENT_MESSAGES = 10
MIN_RECENT_MESSAGES = 4
KEEP_RECENT_CHARS = 12_000
MIN_RECENT_CHARS = 4_000
MAX_SECTION_ITEMS = 8
MAX_ITEM_CHARS = 240
MODEL_SUMMARY_MAX_CHARS = 4_000

HYBRID_COMPACT_SYSTEM_PROMPT = """Rewrite session memory into a compact operational summary.
Return plain text only.
Preserve durable facts, important tool findings, constraints, and the next likely step.
Do not invent facts.
Use this structure:
Working memory:
- Current objective
- Open thread
- Next step

Durable memory:
- Confirmed facts
- Touched files and artifacts
- Useful tool findings
- Constraints

Keep the same factual meaning as the fallback summary.
Keep it concise and actionable.
"""


@dataclass(frozen=True)
class CompactionPlan:
    system_message: dict[str, Any]
    recent_messages: list[dict[str, Any]]
    fallback_summary: str


def plan_compaction(messages: list[dict[str, Any]], *, overflow_tokens: int = 0) -> CompactionPlan | None:
    if not messages:
        return None
    system_message = messages[0]
    body = messages[1:]
    if not _should_compact(body, overflow_tokens=overflow_tokens):
        return None

    older, recent = _split_messages(body, overflow_tokens=overflow_tokens)
    if not older:
        return None
    summary = _build_summary(older)
    if not summary:
        return None
    return CompactionPlan(
        system_message=system_message,
        recent_messages=recent,
        fallback_summary=summary,
    )


def apply_compaction(plan: CompactionPlan, summary: str | None = None) -> list[dict[str, Any]]:
    final_summary = _normalize_summary(summary or plan.fallback_summary)
    return [
        plan.system_message,
        {"role": "system", "content": final_summary},
        *plan.recent_messages,
    ]


def compact_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    plan = plan_compaction(messages)
    if plan is None:
        return messages, False
    return apply_compaction(plan), True


def build_hybrid_refinement_messages(plan: CompactionPlan) -> list[dict[str, Any]]:
    recent_snapshot = _render_recent_snapshot(plan.recent_messages)
    user_content = (
        "Rewrite this local fallback summary into a tighter operational memory.\n\n"
        f"Fallback summary:\n{plan.fallback_summary}\n\n"
        f"Recent raw context:\n{recent_snapshot}"
    )
    return [
        {"role": "system", "content": HYBRID_COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def normalize_model_summary(content: str) -> str | None:
    normalized = _normalize_summary(content)
    if not normalized or normalized == SUMMARY_MARKER:
        return None
    return normalized


def _should_compact(body: list[dict[str, Any]], *, overflow_tokens: int = 0) -> bool:
    if overflow_tokens > 0:
        return True
    if len(body) > KEEP_RECENT_MESSAGES:
        return True
    return _estimate_messages_chars(body) > KEEP_RECENT_CHARS


def _split_messages(body: list[dict[str, Any]], *, overflow_tokens: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recent_message_limit, recent_char_limit = _recent_limits(overflow_tokens)
    recent: list[dict[str, Any]] = []
    recent_chars = 0
    for message in reversed(body):
        message_chars = _estimate_message_chars(message)
        if len(recent) < MIN_RECENT_MESSAGES:
            recent.append(message)
            recent_chars += message_chars
            continue
        if len(recent) >= recent_message_limit:
            break
        if recent_chars + message_chars > recent_char_limit:
            break
        recent.append(message)
        recent_chars += message_chars
    recent.reverse()
    older_count = len(body) - len(recent)
    return body[:older_count], recent


def _recent_limits(overflow_tokens: int) -> tuple[int, int]:
    if overflow_tokens <= 0:
        return KEEP_RECENT_MESSAGES, KEEP_RECENT_CHARS
    # Use the projected overflow to shrink the "recent" window aggressively
    # instead of retrying with a still-near-full prompt.
    reduced_messages = max(MIN_RECENT_MESSAGES, KEEP_RECENT_MESSAGES - max(1, overflow_tokens // 800))
    reduced_chars = max(MIN_RECENT_CHARS, KEEP_RECENT_CHARS - (overflow_tokens * 6))
    return reduced_messages, reduced_chars


def _build_summary(messages: list[dict[str, Any]]) -> str:
    previous_sections: dict[str, list[str]] = {
        "current_objective": [],
        "open_thread": [],
        "next_step": [],
        "confirmed_facts": [],
        "touched_files": [],
        "tool_findings": [],
        "constraints": [],
    }
    user_items: list[str] = []
    assistant_items: list[str] = []
    tool_items: list[str] = []
    touched_files: list[str] = []
    constraints: list[str] = []

    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if not content and role != "tool":
            continue
        if role == "system" and SUMMARY_MARKER in content:
            parsed = _parse_existing_summary(content)
            for key, items in parsed.items():
                previous_sections[key].extend(items)
            continue
        if role == "user":
            user_items.append(_trim_line(content))
            touched_files.extend(_extract_artifact_markers(content))
            constraints.extend(_extract_constraint_hints(content))
            continue
        if role == "assistant":
            assistant_items.append(_trim_line(content))
            touched_files.extend(_extract_artifact_markers(content))
            constraints.extend(_extract_constraint_hints(content))
            continue
        if role == "tool":
            summary = _tool_summary(message)
            tool_items.append(summary)
            touched_files.extend(_extract_artifact_markers(summary))

    sections: list[str] = [SUMMARY_MARKER]

    sections.append("Working memory:")
    working_sections = {
        "Current objective:": _dedupe(previous_sections["current_objective"] + user_items[-2:])[:2],
        "Open thread:": _dedupe(previous_sections["open_thread"] + assistant_items[-2:] + user_items[-1:])[:3],
        "Next step:": _dedupe(previous_sections["next_step"] + user_items[-1:])[:2],
    }
    for heading, items in working_sections.items():
        if not items:
            continue
        sections.append(heading)
        sections.extend(f"- {item}" for item in items)

    sections.append("")
    sections.append("Durable memory:")
    durable_sections = {
        "Confirmed facts:": _dedupe(previous_sections["confirmed_facts"] + assistant_items + tool_items[:2])[:MAX_SECTION_ITEMS],
        "Touched files and artifacts:": _dedupe(previous_sections["touched_files"] + touched_files)[: MAX_SECTION_ITEMS * 2],
        "Useful tool findings:": _dedupe(previous_sections["tool_findings"] + tool_items)[: MAX_SECTION_ITEMS * 2],
        "Constraints:": _dedupe(previous_sections["constraints"] + constraints)[:MAX_SECTION_ITEMS],
    }
    for heading, items in durable_sections.items():
        if not items:
            continue
        sections.append(heading)
        sections.extend(f"- {item}" for item in items)
    return "\n".join(sections).strip()


def _render_recent_snapshot(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "- none"
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip() or "unknown"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"- {role}: {_trim_line(content)}")
            continue
        if role == "tool":
            lines.append(f"- tool: {_tool_summary(message)}")
            continue
        lines.append(f"- {role}: [no content]")
    return "\n".join(lines)


def _tool_summary(message: dict[str, Any]) -> str:
    tool_name = str(message.get("tool_name") or message.get("name") or "tool")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return f"{tool_name}: no content"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return f"{tool_name}: {_trim_line(content)}"

    ok = payload.get("ok")
    status = "ok" if ok else "error"
    fields: list[str] = []
    for key in ("path", "url", "final_url", "command", "title", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            fields.append(f"{key}={_trim_line(value)}")
            if len(fields) >= 2:
                break
    suffix = f" ({', '.join(fields)})" if fields else ""
    return f"{tool_name}: {status}{suffix}"


def _trim_line(value: str) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= MAX_ITEM_CHARS:
        return collapsed
    return collapsed[: MAX_ITEM_CHARS - 3] + "..."


def _normalize_summary(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if SUMMARY_MARKER not in text:
        text = f"{SUMMARY_MARKER}\n{text}"
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    if len(normalized) <= MODEL_SUMMARY_MAX_CHARS:
        return normalized
    return normalized[: MODEL_SUMMARY_MAX_CHARS - 3].rstrip() + "..."


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


SUMMARY_SECTION_ALIASES = {
    "current objective": "current_objective",
    "objective": "current_objective",
    "open thread": "open_thread",
    "open questions": "open_thread",
    "next step": "next_step",
    "confirmed facts": "confirmed_facts",
    "durable facts": "confirmed_facts",
    "files and artifacts": "touched_files",
    "touched files and artifacts": "touched_files",
    "tool findings": "tool_findings",
    "useful tool findings": "tool_findings",
    "tool activity": "tool_findings",
    "constraints": "constraints",
}


def _parse_existing_summary(content: str) -> dict[str, list[str]]:
    parsed = {value: [] for value in SUMMARY_SECTION_ALIASES.values()}
    current_key: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line == SUMMARY_MARKER:
            continue
        normalized = line.rstrip(":").strip().lower()
        if normalized in {"working memory", "durable memory"}:
            current_key = None
            continue
        alias = SUMMARY_SECTION_ALIASES.get(normalized)
        if alias is not None:
            current_key = alias
            continue
        if current_key is None:
            continue
        item = line[2:].strip() if line.startswith("- ") else line
        item = _trim_line(item)
        if item:
            parsed[current_key].append(item)
    return parsed


def _extract_artifact_markers(value: str) -> list[str]:
    markers: list[str] = []
    for match in re.findall(r"(?:path=)?([A-Za-z0-9_./-]+\.[A-Za-z0-9_+-]+)", value):
        markers.append(match)
    return [_trim_line(marker) for marker in markers]


def _extract_constraint_hints(value: str) -> list[str]:
    hints: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", value):
        lowered = sentence.lower()
        if any(token in lowered for token in ("must", "should", "avoid", "do not", "don't", "deve", "non deve", "evita", "mai")):
            trimmed = _trim_line(sentence)
            if trimmed:
                hints.append(trimmed)
    return hints


def _estimate_messages_chars(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_message_chars(message) for message in messages)


def _estimate_message_chars(message: dict[str, Any]) -> int:
    total = len(str(message.get("role", "")))
    total += len(str(message.get("content", "")))
    tool_name = message.get("tool_name")
    if tool_name:
        total += len(str(tool_name))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += len(json.dumps(tool_calls, ensure_ascii=False))
    return total
