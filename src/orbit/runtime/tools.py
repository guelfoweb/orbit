from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit.runtime.file_tools import (
    DEFAULT_CHUNK_CHARS,
    MAX_APPEND_CHARS,
    MAX_CHUNK_CHARS,
    MAX_READ_BYTES,
    MAX_REPLACE_CHARS,
    MAX_TEXT_FILE_BYTES_AFTER_APPEND,
    MAX_TEXT_FILE_BYTES_AFTER_REPLACE,
    MAX_WRITE_CHARS,
    append_file,
    delete_path,
    list_files,
    make_directory,
    read_file,
    replace_in_file,
    stat_path,
    write_file,
)
from orbit.runtime.web import (
    DEFAULT_FETCH_CHUNK_CHARS,
    DEFAULT_SEARCH_RESULTS,
    MAX_FETCH_CHUNK_CALLS_PER_TURN,
    MAX_FETCH_CHUNK_CHARS,
    MAX_SEARCH_RESULTS,
    fetch_url,
    search_web,
)
from orbit.runtime.edit_guardrails import apply_diff_definition, apply_local_edit_file, edit_file_definition
from orbit.runtime.shell_guardrails import exec_shell_definition


MAX_CHUNK_CALLS_PER_TURN = 3


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str


TOOL_NAMES = (
    "list_files",
    "read_file",
    "stat_path",
    "make_directory",
    "delete_path",
    "fetch_url",
    "search_web",
    "exec_shell_command",
    "write_file",
    "append_file",
    "replace_in_file",
    "edit_file",
    "apply_diff",
)


def tool_names() -> tuple[str, ...]:
    return TOOL_NAMES


def tool_definitions(names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List a local directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read UTF-8 text/code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        },
                        "chunk_index": {
                            "type": "integer",
                        },
                        "chunk_chars": {
                            "type": "integer",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stat_path",
                "description": "Stat a local path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "make_directory",
                "description": "Create a local directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_path",
                "description": "Delete local file/directory only for explicit delete/remove requests, not shell commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        },
                        "recursive": {
                            "type": "boolean",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch an http/https URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                        },
                        "chunk_index": {
                            "type": "integer",
                        },
                        "chunk_chars": {
                            "type": "integer",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                        },
                        "max_results": {
                            "type": "integer",
                        },
                        "site": {
                            "type": "string",
                        },
                        "timelimit": {
                            "type": "string",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        exec_shell_definition(),
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a UTF-8 file. No overwrite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        },
                        "content": {
                            "type": "string",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append UTF-8 text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        },
                        "content": {
                            "type": "string",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": "Replace unique UTF-8 text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        },
                        "old": {
                            "type": "string",
                        },
                        "new": {
                            "type": "string",
                        },
                    },
                    "required": ["path", "old", "new"],
                },
            },
        },
        edit_file_definition(),
        apply_diff_definition(),
    ]
    if names is None:
        return definitions
    allowed = set(names)
    return [tool for tool in definitions if tool["function"]["name"] in allowed]


def execute_tool(
    name: str,
    arguments: str | dict[str, Any],
    *,
    workdir: Path,
    chunk_budget: dict[str, int] | None = None,
) -> ToolResult:
    if name not in TOOL_NAMES:
        return ToolResult(name=name, content=f"error: unknown tool: {name}")
    parsed = _parse_arguments(arguments)
    if isinstance(parsed, str):
        return ToolResult(name=name, content=parsed)
    if name == "list_files":
        return ToolResult(name=name, content=list_files(parsed.get("path", "."), workdir=workdir))
    if name == "stat_path":
        return ToolResult(name=name, content=stat_path(parsed.get("path"), workdir=workdir))
    if name == "make_directory":
        return ToolResult(name=name, content=make_directory(parsed.get("path"), workdir=workdir))
    if name == "delete_path":
        return ToolResult(name=name, content=delete_path(parsed.get("path"), parsed.get("recursive", False), workdir=workdir))
    if name == "fetch_url":
        if "chunk_index" in parsed and chunk_budget is not None:
            used = chunk_budget.get("fetch_url_chunks", 0)
            if used >= MAX_FETCH_CHUNK_CALLS_PER_TURN:
                return ToolResult(
                    name=name,
                    content=(
                        f"error: fetch_url chunk budget exceeded for this turn: max {MAX_FETCH_CHUNK_CALLS_PER_TURN} chunks. "
                        "Stop and explain that complete analysis requires continuing in later turns."
                    ),
                )
            chunk_budget["fetch_url_chunks"] = used + 1
        return ToolResult(
            name=name,
            content=fetch_url(
                parsed.get("url"),
                chunk_index=parsed.get("chunk_index"),
                chunk_chars=parsed.get("chunk_chars", DEFAULT_FETCH_CHUNK_CHARS),
            ),
        )
    if name == "search_web":
        return ToolResult(
            name=name,
            content=search_web(
                parsed.get("query"),
                max_results=parsed.get("max_results", DEFAULT_SEARCH_RESULTS),
                site=parsed.get("site"),
                timelimit=parsed.get("timelimit"),
            ),
        )
    if name == "exec_shell_command":
        return ToolResult(name=name, content="error: exec_shell_command requires llama-server built-in tool support")
    if name == "write_file":
        return ToolResult(name=name, content=write_file(parsed.get("path"), parsed.get("content"), workdir=workdir))
    if name == "append_file":
        return ToolResult(name=name, content=append_file(parsed.get("path"), parsed.get("content"), workdir=workdir))
    if name == "replace_in_file":
        return ToolResult(
            name=name,
            content=replace_in_file(parsed.get("path"), parsed.get("old"), parsed.get("new"), workdir=workdir),
        )
    if name == "edit_file":
        return ToolResult(name=name, content=apply_local_edit_file(parsed, workdir=workdir))
    if name == "apply_diff":
        return ToolResult(name=name, content="error: apply_diff requires llama-server built-in tool support")
    if "chunk_index" in parsed and chunk_budget is not None:
        used = chunk_budget.get("read_file_chunks", 0)
        if used >= MAX_CHUNK_CALLS_PER_TURN:
            return ToolResult(
                name=name,
                content=(
                    f"error: chunk budget exceeded for this turn: max {MAX_CHUNK_CALLS_PER_TURN} chunks. "
                    "Stop and explain that complete analysis requires continuing in later turns."
                ),
            )
        chunk_budget["read_file_chunks"] = used + 1
    return ToolResult(name=name, content=read_file(parsed.get("path"), arguments=parsed, workdir=workdir))


def _parse_arguments(arguments: str | dict[str, Any]) -> dict[str, Any] | str:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return f"error: invalid JSON tool arguments: {exc}"
    if not isinstance(parsed, dict):
        return "error: tool arguments must be a JSON object"
    return parsed
