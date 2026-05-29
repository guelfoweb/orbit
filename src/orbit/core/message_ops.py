from __future__ import annotations

import json
import re
from typing import Any

LIKELY_BINARY_EXTENSIONS = {
    ".pdf",
    ".apk",
    ".dex",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".dylib",
    ".a",
    ".o",
    ".jar",
    ".zip",
    ".aar",
    ".ipa",
    ".doc",
    ".docm",
    ".docx",
    ".xls",
    ".xlsm",
    ".xlsx",
    ".ppt",
    ".pptm",
    ".pptx",
    ".rtf",
    ".js",
    ".jse",
    ".vbs",
    ".vbe",
    ".wsf",
    ".hta",
    ".ps1",
    ".bat",
    ".cmd",
    ".sh",
    ".py",
    ".php",
    ".pl",
    ".rb",
    ".lua",
    ".html",
    ".htm",
    ".mhtml",
    ".svg",
    ".xml",
    ".xsl",
}
ARCHIVE_CONTAINER_EXTENSIONS = {".apk", ".zip", ".jar", ".aar", ".ipa", ".docx", ".docm", ".xlsx", ".xlsm", ".pptx", ".pptm"}
ARCHIVE_PATH_RE = re.compile(r"([^\s'\"]+\.(?:apk|zip|jar|aar|ipa))", re.IGNORECASE)


def assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content", ""),
    }
    if message.get("tool_calls"):
        record["tool_calls"] = message["tool_calls"]
    return record


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("role", "")))
        total_chars += len(str(message.get("content", "")))
        tool_name = message.get("tool_name")
        if tool_name:
            total_chars += len(str(tool_name))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    estimated = (total_chars + 3) // 4
    return max(1, estimated)


def last_read_file_result(messages: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != "read_file":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") is True and payload.get("path") == path:
            return payload
    return None


def normalize_relative_path(path: str) -> str:
    value = path.replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return value


def was_listed_by_list_files(messages: list[dict[str, Any]], path: str) -> bool:
    return listed_entry_type(messages, path) is not None


def listed_entry_type(messages: list[dict[str, Any]], path: str) -> str | None:
    normalized_target = normalize_relative_path(path)
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != "list_files":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("entries", []):
            if not isinstance(entry, dict):
                continue
            listed_path = entry.get("path")
            if isinstance(listed_path, str) and normalize_relative_path(listed_path) == normalized_target:
                entry_type = entry.get("type")
                return entry_type if isinstance(entry_type, str) else ""
    return None


def has_recent_tool_result(messages: list[dict[str, Any]], tool_name: str) -> bool:
    for message in reversed(messages):
        if message.get("role") == "user":
            return False
        if message.get("role") == "tool" and message.get("tool_name") == tool_name:
            return True
    return False


def likely_binary_candidates_from_recent_listing(messages: list[dict[str, Any]], limit: int = 6) -> list[str]:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != "list_files":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            continue
        explicit: list[str] = []
        fallback: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            path = entry.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            normalized = normalize_relative_path(path)
            lowered = normalized.lower()
            if lowered.startswith("."):
                continue
            if any(lowered.endswith(ext) for ext in LIKELY_BINARY_EXTENSIONS):
                explicit.append(normalized)
            else:
                fallback.append(normalized)
        if explicit:
            return explicit[:limit]
        return fallback[:limit]
    return []


def recent_archive_container_for_member(messages: list[dict[str, Any]], member_path: str) -> str | None:
    normalized_member = normalize_relative_path(member_path)
    if not normalized_member:
        return None
    lowered_member = normalized_member.lower()
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != "bash":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        command = payload.get("command")
        stdout = payload.get("stdout")
        if not isinstance(command, str):
            continue
        match = ARCHIVE_PATH_RE.search(command)
        if match is None:
            continue
        container = normalize_relative_path(match.group(1))
        if not any(container.lower().endswith(ext) for ext in ARCHIVE_CONTAINER_EXTENSIONS):
            continue
        if lowered_member in command.lower():
            return container
        if isinstance(stdout, str) and lowered_member in stdout.lower():
            return container
    return None


def successful_read_results_in_current_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") != "tool" or message.get("tool_name") != "read_file":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            results.append(payload)
    results.reverse()
    return results


def latest_successful_read_result_in_current_turn(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    results = successful_read_results_in_current_turn(messages)
    if not results:
        return None
    return results[-1]


def merged_read_file_result_in_current_turn(messages: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    normalized_target = normalize_relative_path(path)
    chunks = [
        item
        for item in successful_read_results_in_current_turn(messages)
        if normalize_relative_path(str(item.get("path", ""))) == normalized_target
    ]
    if not chunks:
        return None
    chunks.sort(key=lambda item: int(item.get("start_line", 1)))
    merged_parts: list[str] = []
    last_end_line = 0
    last_chunk = chunks[-1]
    for chunk in chunks:
        content = chunk.get("content")
        start_line = chunk.get("start_line")
        returned_lines = chunk.get("returned_lines")
        if not isinstance(content, str) or not isinstance(start_line, int) or not isinstance(returned_lines, int):
            continue
        if start_line <= last_end_line:
            continue
        merged_parts.append(content.rstrip("\n"))
        last_end_line = start_line + max(0, returned_lines) - 1
    if not merged_parts:
        return None
    return {
        "ok": True,
        "path": str(last_chunk.get("path", path)),
        "content": "\n".join(part for part in merged_parts if part),
        "truncated": bool(last_chunk.get("truncated")),
        "has_more": bool(last_chunk.get("has_more")),
        "next_start_line": last_chunk.get("next_start_line"),
        "total_lines": last_chunk.get("total_lines"),
    }


def successful_write_results_in_current_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") != "tool" or message.get("tool_name") not in {"write_file", "append_file", "replace_in_file", "make_directory", "delete_path"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            results.append(payload)
    results.reverse()
    return results


def successful_bash_results_in_current_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") != "tool" or message.get("tool_name") != "bash":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            results.append(payload)
    results.reverse()
    return results


def last_fetch_url_result(messages: list[dict[str, Any]], url: str) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != "fetch_url":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") is True and payload.get("url") == url:
            return payload
    return None


def latest_fetch_url_result_in_current_turn(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") != "tool" or message.get("tool_name") != "fetch_url":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            return payload
    return None


def latest_search_web_result_in_current_turn(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") != "tool" or message.get("tool_name") != "search_web":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            return payload
    return None


def recent_listed_paths_by_type(messages: list[dict[str, Any]], *, entry_type: str, limit: int = 12) -> list[str]:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != "list_files":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        entries = payload.get("entries")
        if not isinstance(entries, list):
            continue
        out: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != entry_type:
                continue
            path = entry.get("path")
            if isinstance(path, str) and path.strip():
                out.append(path.strip())
                if len(out) >= limit:
                    return out
        return out
    return []


def recent_listed_file_paths(messages: list[dict[str, Any]], limit: int = 12) -> list[str]:
    return recent_listed_paths_by_type(messages, entry_type="file", limit=limit)


def recent_listed_directory_paths(messages: list[dict[str, Any]], limit: int = 12) -> list[str]:
    return recent_listed_paths_by_type(messages, entry_type="dir", limit=limit)


def recent_listed_paths(messages: list[dict[str, Any]], limit: int = 12) -> list[str]:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != "list_files":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        entries = payload.get("entries")
        if not isinstance(entries, list):
            continue
        out: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if isinstance(path, str) and path.strip():
                out.append(path.strip())
                if len(out) >= limit:
                    return out
        return out
    return []
