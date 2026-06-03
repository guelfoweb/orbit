from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from typing import Any


TOOL_HISTORY_LOOKBACK = 6


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    signature: str
    detail: str


def register_tool_calls(history: list[ToolCallRecord], tool_calls: list[dict[str, Any]]) -> None:
    for tool_call in tool_calls:
        history.append(tool_record(tool_call))


def repeated_tool_records(
    *,
    tool_calls: list[dict[str, Any]],
    history: list[ToolCallRecord],
) -> list[ToolCallRecord]:
    if not history:
        return []
    current_records = [tool_record(call) for call in tool_calls]
    recent_history = history[-TOOL_HISTORY_LOOKBACK:]
    return [record for record in current_records if find_matching_record(record, recent_history) is not None]


def repeated_read_path_record(
    *,
    tool_calls: list[dict[str, Any]],
    history: list[ToolCallRecord],
    min_unique_paths: int = 5,
) -> ToolCallRecord | None:
    if len(sampled_read_paths(history)) < min_unique_paths:
        return None
    recent_history = history[-TOOL_HISTORY_LOOKBACK:]
    for record in [tool_record(call) for call in tool_calls]:
        if record.name != "read_file":
            continue
        current_path = extract_read_path(record)
        if not current_path:
            continue
        matches = 0
        for item in recent_history:
            if item.name != "read_file":
                continue
            if extract_read_path(item) == current_path:
                matches += 1
        if matches >= 1:
            return record
    return None


def repeated_write_path_record(
    *,
    tool_calls: list[dict[str, Any]],
    history: list[ToolCallRecord],
) -> ToolCallRecord | None:
    recent_history = history[-TOOL_HISTORY_LOOKBACK:]
    for record in [tool_record(call) for call in tool_calls]:
        if record.name not in {"write_file", "append_file", "replace_in_file", "make_directory", "delete_path"}:
            continue
        current_path = extract_path(record, prefix=f"{record.name}.path=")
        if not current_path:
            continue
        for item in recent_history:
            if item.name != record.name:
                continue
            if extract_path(item, prefix=f"{item.name}.path=") == current_path:
                return record
    return None


def repeated_tool_count(record: ToolCallRecord, history: list[ToolCallRecord]) -> int:
    return sum(1 for item in history if signatures_match(record, item)) + 1


def tool_record(tool_call: dict[str, Any]) -> ToolCallRecord:
    fn = tool_call.get("function", {})
    name = str(fn.get("name") or "tool")
    arguments = fn.get("arguments")
    if isinstance(arguments, str):
        rendered = arguments
    else:
        rendered = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    return ToolCallRecord(name=name, signature=f"{name}:{rendered}", detail=render_detail(name, arguments))


def find_matching_record(record: ToolCallRecord, history: list[ToolCallRecord]) -> ToolCallRecord | None:
    for item in reversed(history):
        if signatures_match(record, item):
            return item
    return None


def signatures_match(left: ToolCallRecord, right: ToolCallRecord) -> bool:
    if left.name != right.name:
        return False
    return normalize_signature(left.signature) == normalize_signature(right.signature)


def normalize_signature(value: str) -> str:
    return " ".join(value.lower().split())


def render_detail(name: str, arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return ""
    for key in ("command", "path", "url", "query", "title"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return f"{name}.{key}={trim(value)}"
    if arguments:
        return f"{name}.args={trim(json.dumps(arguments, ensure_ascii=False, sort_keys=True))}"
    return ""


def trim(value: str, limit: int = 120) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def is_trivial_python_entrypoint(detail: str) -> bool:
    marker = "read_file.path="
    if marker not in detail:
        return False
    raw_path = detail.split(marker, 1)[1].strip()
    try:
        name = PurePosixPath(raw_path).name
    except Exception:
        return False
    return name in {"__init__.py", "__main__.py"}


def sampled_read_paths(history: list[ToolCallRecord], limit: int = 8) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for item in history:
        if item.name != "read_file":
            continue
        path = extract_read_path(item)
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def extract_read_path(record: ToolCallRecord) -> str:
    return extract_path(record, prefix="read_file.path=")


def extract_path(record: ToolCallRecord, prefix: str) -> str:
    marker = "read_file.path="
    marker = prefix
    if marker not in record.detail:
        return ""
    return record.detail.split(marker, 1)[1].strip()
