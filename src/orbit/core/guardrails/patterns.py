from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable
import re
import shlex
import shutil

from .documents import extract_explicit_text_path
from ..messages import has_recent_tool_result


OPEN_CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[\s\]\s+")
HEADING_RE = re.compile(r"^\s*#+\s+")
DATE_RE = re.compile(r"📅\s*(\d{4}-\d{2}-\d{2})")
RG_MARKDOWN_CHECKBOX_PATTERN = r"^(#+ |[-*] \[ \])"
PRIORITY_SCORES = {"🔺": 0, "⏫": 1, "🔼": 2, "🔽": 4}


@dataclass(frozen=True)
class MarkdownTask:
    section: str
    text: str
    due: date | None
    priority_score: int
    recurring: bool
    inconsistent: bool


def seed_markdown_checkbox_extraction(
    *,
    skill: Any,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> None:
    if not markdown_task_skill_enabled(skill):
        return
    path = markdown_checkbox_extraction_path(user_input)
    if path is None:
        return
    if has_recent_tool_result(messages, "bash"):
        return
    command = markdown_checkbox_extraction_command(path)
    run_guardrail_tool(
        name="bash",
        arguments={"command": command},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )
    if markdown_checkbox_needs_semantic_analysis(user_input):
        messages.append(
            {
                "role": "system",
                "content": (
                    "The previous bash stdout is the complete extracted evidence for open Markdown checkbox tasks and headings. "
                    "Do not call read_file for the same file. Do not include completed tasks. "
                    "Answer the user's semantic task using only those extracted open-task lines."
                ),
            }
        )


def local_markdown_checkbox_extraction_result(
    *,
    skill: Any,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if not markdown_task_skill_enabled(skill):
        return None
    if markdown_checkbox_extraction_path(user_input) is None:
        return None
    payload = latest_markdown_checkbox_bash_result(messages)
    if payload is None:
        return None
    stdout = payload.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return "No open Markdown checkbox tasks found."
    if markdown_checkbox_needs_semantic_analysis(user_input):
        return None
    return format_markdown_checkbox_matches(stdout)


def markdown_checkbox_redundant_read_prompt(
    *,
    skill: Any,
    user_input: str,
    name: str,
    arguments: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str | None:
    if not markdown_task_skill_enabled(skill):
        return None
    target_path = markdown_checkbox_extraction_path(user_input)
    if target_path is None or not markdown_checkbox_needs_semantic_analysis(user_input):
        return None
    if name != "read_file":
        return None
    requested_path = arguments.get("path")
    if not isinstance(requested_path, str) or requested_path.strip() != target_path:
        return None
    if latest_markdown_checkbox_bash_result(messages) is None:
        return None
    return (
        "Skipped read_file because open Markdown tasks were already extracted with a bounded pattern search. "
        "Use only the previous bash stdout as evidence, ignore completed tasks, and answer the requested semantic analysis now."
    )


def markdown_task_skill_enabled(skill: Any) -> bool:
    name = getattr(skill, "name", "")
    content = getattr(skill, "content", "")
    if isinstance(name, str) and name in {"task-notes", "daily-tasks", "obsidian-daily"}:
        return True
    if isinstance(content, str) and ("Markdown Task Review" in content or "Obsidian Daily Task Review" in content):
        return True
    return False


def markdown_checkbox_extraction_path(user_input: str) -> str | None:
    lowered = user_input.lower()
    if "- [ ]" not in user_input and "open task" not in lowered and "task aperti" not in lowered:
        return None
    if not any(hint in lowered for hint in ("extract", "estrai", "find", "show", "return", "list", "open tasks", "task aperti")):
        return None
    path = extract_explicit_text_path(user_input)
    if path is None or not path.lower().endswith((".md", ".markdown")):
        return None
    return path


def markdown_checkbox_needs_semantic_analysis(user_input: str) -> bool:
    lowered = user_input.lower()
    semantic_hints = (
        "analyze",
        "analyse",
        "analysis",
        "semantically",
        "semantic",
        "priorities",
        "priority",
        "priorità",
        "priorita",
        "overdue",
        "expired",
        "scaduti",
        "scadute",
        "ricorrenti",
        "recurring",
        "suggest",
        "suggerisci",
        "group",
        "grouped",
        "raggruppa",
        "bloccati",
        "blocked",
    )
    return any(hint in lowered for hint in semantic_hints)


def markdown_checkbox_extraction_command(path: str) -> str:
    quoted_path = shlex.quote(path)
    quoted_pattern = shlex.quote(RG_MARKDOWN_CHECKBOX_PATTERN)
    if shutil.which("rg") is not None:
        return f"rg -n {quoted_pattern} {quoted_path}"
    return f"grep -nE {quoted_pattern} {quoted_path}"


def latest_markdown_checkbox_bash_result(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return None
        if message.get("role") != "tool" or message.get("tool_name") != "bash":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            import json

            payload = json.loads(content)
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        command = payload.get("command")
        if isinstance(command, str) and RG_MARKDOWN_CHECKBOX_PATTERN in command:
            return payload
    return None


def format_markdown_checkbox_matches(stdout: str) -> str:
    tasks = parse_markdown_checkbox_tasks(stdout)
    notes: list[str] = []
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for task in tasks:
        groups.setdefault(task.section, []).append(task.text)
        if task.inconsistent:
            notes.append(f"`{task.text}` has inconsistent metadata: open checkbox with a completion marker.")
    rendered: list[str] = []
    for section, section_tasks in groups.items():
        if not section_tasks:
            continue
        if rendered:
            rendered.append("")
        rendered.append(section)
        rendered.extend(section_tasks)
    if notes:
        rendered.append("")
        rendered.append("Notes")
        rendered.extend(f"- {note}" for note in notes)
    return "\n".join(rendered) if rendered else "No open Markdown checkbox tasks found."


def format_markdown_checkbox_semantic_analysis(stdout: str, *, today: date | None = None) -> str:
    current_date = today or date.today()
    tasks = parse_markdown_checkbox_tasks(stdout)
    if not tasks:
        return "No open Markdown checkbox tasks found."
    groups: OrderedDict[str, list[MarkdownTask]] = OrderedDict()
    for task in tasks:
        groups.setdefault(task.section, []).append(task)
    overdue = [task for task in tasks if task.due is not None and task.due < current_date]
    recurring = [task for task in tasks if task.recurring]
    priorities = sorted(
        tasks,
        key=lambda task: (
            0 if task.due is not None and task.due < current_date else 1,
            task.priority_score,
            task.due or date.max,
            task.text.lower(),
        ),
    )[:3]
    rendered: list[str] = ["Open tasks by section"]
    for section, section_tasks in groups.items():
        rendered.append("")
        rendered.append(section)
        rendered.extend(task.text for task in section_tasks)
    rendered.append("")
    rendered.append("Overdue tasks")
    if overdue:
        rendered.extend(task.text for task in overdue)
    else:
        rendered.append("- None found from explicit due dates.")
    rendered.append("")
    rendered.append("Recurring tasks")
    if recurring:
        rendered.extend(task.text for task in recurring)
    else:
        rendered.append("- None found.")
    rendered.append("")
    rendered.append("Top 3 priorities for today")
    for index, task in enumerate(priorities, start=1):
        rendered.append(f"{index}. {task.text}")
    notes = [
        f"`{task.text}` has inconsistent metadata: open checkbox with a completion marker."
        for task in tasks
        if task.inconsistent
    ]
    if notes:
        rendered.append("")
        rendered.append("Notes")
        rendered.extend(f"- {note}" for note in notes)
    return "\n".join(rendered)


def parse_markdown_checkbox_tasks(stdout: str) -> list[MarkdownTask]:
    tasks: list[MarkdownTask] = []
    current_section = "Tasks"
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        _, sep, content = line.partition(":")
        if not sep:
            content = line
        content = content.strip()
        if HEADING_RE.match(content):
            current_section = HEADING_RE.sub("", content).strip() or "Tasks"
            continue
        if not OPEN_CHECKBOX_RE.match(content):
            continue
        tasks.append(
            MarkdownTask(
                section=current_section,
                text=content,
                due=_extract_due_date(content),
                priority_score=min((score for marker, score in PRIORITY_SCORES.items() if marker in content), default=3),
                recurring="🔁" in content,
                inconsistent="✅" in content,
            )
        )
    return tasks


def _extract_due_date(text: str) -> date | None:
    match = DATE_RE.search(text)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None
