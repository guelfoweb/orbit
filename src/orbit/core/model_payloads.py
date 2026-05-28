from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

MODEL_FIRST_TOOL_TEXT_LIMIT = 1600
MODEL_FIRST_FETCH_URL_TEXT_LIMIT = 4000
MODEL_FIRST_TOOL_TEXT_SOFT_LIMIT = 600
MODEL_FIRST_TOOL_ENTRIES_LIMIT = 12
MODEL_FIRST_TOOL_LINKS_LIMIT = 6

COMPACT_TOOL_DESCRIPTIONS = {
    "read_file": "Read a known text file in bounded chunks.",
    "list_files": "List files or directories under the workdir.",
    "stat_path": "Inspect bounded file or directory metadata.",
    "make_directory": "Create one directory inside the workdir.",
    "delete_path": "Delete one file or directory inside the workdir.",
    "replace_in_file": "Replace text inside an existing UTF-8 file.",
    "write_file": "Write one small UTF-8 text file.",
    "append_file": "Append one small UTF-8 text chunk to a file.",
    "bash": "Run one bounded safe command in the workdir.",
    "search_web": "Search the web and return bounded results.",
    "fetch_url": "Fetch one explicit URL and return bounded page content.",
}


class ModelPayloadCompactor:
    def __init__(self) -> None:
        self._tool_definition_cache: dict[str, dict[str, Any]] = {}
        self._tool_message_cache: dict[tuple[str, str], str] = {}

    def clear_message_cache(self) -> None:
        self._tool_message_cache.clear()

    def compact_tool_definition(self, tool: dict[str, Any]) -> dict[str, Any]:
        cache_key = json.dumps(tool, ensure_ascii=False, sort_keys=True)
        cached = self._tool_definition_cache.get(cache_key)
        if cached is not None:
            return cached
        compact = compact_tool_definition(tool)
        self._tool_definition_cache[cache_key] = compact
        return compact

    def compact_message_for_model(self, message: dict[str, Any]) -> dict[str, Any]:
        compact = dict(message)
        compact.pop("thinking", None)
        if compact.get("role") != "tool":
            return compact
        tool_name = compact.get("tool_name")
        content = compact.get("content")
        if not isinstance(tool_name, str) or not isinstance(content, str):
            return compact
        cache_key = (tool_name, content)
        cached_content = self._tool_message_cache.get(cache_key)
        if cached_content is None:
            cached_content = compact_tool_message_content(tool_name, content)
            self._tool_message_cache[cache_key] = cached_content
        compact["content"] = cached_content
        return compact


def compact_tool_definition(tool: dict[str, Any]) -> dict[str, Any]:
    compact = dict(tool)
    function = dict(compact.get("function") or {})
    name = str(function.get("name") or "")
    if name in COMPACT_TOOL_DESCRIPTIONS:
        function["description"] = COMPACT_TOOL_DESCRIPTIONS[name]
    parameters = function.get("parameters")
    if isinstance(parameters, dict):
        compact_parameters = {"type": parameters.get("type", "object")}
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            compact_properties: dict[str, Any] = {}
            for key, value in properties.items():
                if not isinstance(value, dict):
                    continue
                entry: dict[str, Any] = {"type": value.get("type", "string")}
                if "enum" in value:
                    entry["enum"] = value["enum"]
                compact_properties[key] = entry
            if compact_properties:
                compact_parameters["properties"] = compact_properties
        required = parameters.get("required")
        if isinstance(required, list) and required:
            compact_parameters["required"] = required
        function["parameters"] = compact_parameters
    compact["function"] = function
    return compact


def compact_message_for_model(message: dict[str, Any]) -> dict[str, Any]:
    compact = dict(message)
    compact.pop("thinking", None)
    if compact.get("role") != "tool":
        return compact
    tool_name = compact.get("tool_name")
    content = compact.get("content")
    if not isinstance(tool_name, str) or not isinstance(content, str):
        return compact
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return compact
    if not isinstance(payload, dict):
        return compact
    compact["content"] = json.dumps(compact_tool_payload(tool_name, payload), ensure_ascii=False)
    return compact


@lru_cache(maxsize=512)
def compact_tool_message_content(tool_name: str, content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(payload, dict):
        return content
    return json.dumps(compact_tool_payload(tool_name, payload), ensure_ascii=False)


def compact_tool_payload(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "list_files":
        result = {
            "ok": payload.get("ok"),
            "path": payload.get("path"),
            "count": payload.get("count"),
            "dir_count": payload.get("dir_count"),
            "file_count": payload.get("file_count"),
            "truncated": payload.get("truncated"),
            "summary": payload.get("summary"),
        }
        entries = payload.get("entries")
        if isinstance(entries, list):
            trimmed_entries: list[dict[str, Any]] = []
            for item in entries[:MODEL_FIRST_TOOL_ENTRIES_LIMIT]:
                if not isinstance(item, dict):
                    continue
                trimmed_entries.append({"path": item.get("path"), "type": item.get("type")})
            result["entries"] = trimmed_entries
        return result
    if tool_name == "read_file":
        summary_read = payload.get("summary_read")
        result = {
            "ok": payload.get("ok"),
            "path": payload.get("path"),
            "summary_read": summary_read,
            "sampled_chunks": payload.get("sampled_chunks"),
            "sampled_start_lines": payload.get("sampled_start_lines"),
            "total_lines": payload.get("total_lines"),
            "returned_lines": payload.get("returned_lines"),
            "next_start_line": payload.get("next_start_line"),
            "has_more": payload.get("has_more"),
            "truncated": payload.get("truncated"),
            "notice": payload.get("notice"),
        }
        chunk_notes = payload.get("chunk_notes")
        if isinstance(chunk_notes, list):
            result["chunk_notes"] = chunk_notes[:8]
        content = payload.get("content")
        if isinstance(content, str) and not summary_read:
            result["content"] = content[:MODEL_FIRST_TOOL_TEXT_LIMIT]
        return result
    if tool_name == "stat_path":
        result = {
            "ok": payload.get("ok"),
            "path": payload.get("path"),
            "type": payload.get("type"),
            "size_bytes": payload.get("size_bytes"),
            "modified_at": payload.get("modified_at"),
            "mode": payload.get("mode"),
            "recursive": payload.get("recursive"),
            "count": payload.get("count"),
            "total_entries": payload.get("total_entries"),
            "file_count": payload.get("file_count"),
            "dir_count": payload.get("dir_count"),
            "truncated": payload.get("truncated"),
        }
        entries = payload.get("entries")
        if isinstance(entries, list):
            compact_entries: list[dict[str, Any]] = []
            for item in entries[:MODEL_FIRST_TOOL_ENTRIES_LIMIT]:
                if not isinstance(item, dict):
                    continue
                compact_entries.append(
                    {
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "size_bytes": item.get("size_bytes"),
                        "modified_at": item.get("modified_at"),
                    }
                )
            result["entries"] = compact_entries
        return result
    if tool_name == "bash":
        result = {
            "ok": payload.get("ok"),
            "command": payload.get("command"),
            "returncode": payload.get("returncode"),
            "truncated": payload.get("truncated"),
        }
        stdout = payload.get("stdout")
        stderr = payload.get("stderr")
        if isinstance(stdout, str):
            result["stdout"] = stdout[:MODEL_FIRST_TOOL_TEXT_SOFT_LIMIT]
        if isinstance(stderr, str) and stderr:
            result["stderr"] = stderr[:MODEL_FIRST_TOOL_TEXT_SOFT_LIMIT]
        return result
    if tool_name == "search_web":
        result = {
            "ok": payload.get("ok"),
            "query": payload.get("query"),
            "provider": payload.get("provider"),
        }
        results = payload.get("results")
        if isinstance(results, list):
            trimmed: list[dict[str, Any]] = []
            for item in results[:3]:
                if not isinstance(item, dict):
                    continue
                trimmed.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "snippet": str(item.get("snippet", ""))[:140],
                    }
                )
            result["results"] = trimmed
        return result
    if tool_name == "fetch_url":
        result = {
            "ok": payload.get("ok"),
            "url": payload.get("url"),
            "final_url": payload.get("final_url"),
            "status_code": payload.get("status_code"),
            "title": payload.get("title"),
            "start_char": payload.get("start_char"),
            "end_char": payload.get("end_char"),
            "total_chars": payload.get("total_chars"),
            "chunk_index": payload.get("chunk_index"),
            "chunk_count": payload.get("chunk_count"),
            "next_start_char": payload.get("next_start_char"),
            "has_more": payload.get("has_more"),
            "truncated": payload.get("truncated"),
        }
        highlights = payload.get("highlights")
        if isinstance(highlights, list):
            result["highlights"] = highlights[:3]
        text = payload.get("text")
        if isinstance(text, str):
            result["text"] = text[:MODEL_FIRST_FETCH_URL_TEXT_LIMIT]
        links = payload.get("links")
        if isinstance(links, list):
            result["links"] = links[:MODEL_FIRST_TOOL_LINKS_LIMIT]
        return result
    return payload
