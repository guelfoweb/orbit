from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from orbit.backend.llama_server import LlamaServerError
from orbit.runtime.edit_guardrails import prepare_apply_diff, prepare_edit_file
from orbit.runtime.shell_guardrails import prepare_exec_shell_command
from orbit.runtime.tools import ToolResult, execute_tool, tool_definitions


LLAMA_SERVER_SAFE_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "file_glob_search",
        "grep_search",
        "get_datetime",
        "exec_shell_command",
        "edit_file",
        "apply_diff",
    }
)

MAX_SERVER_READ_FILE_BYTES = 4 * 1024
MAX_SERVER_TOOL_RESULT_CHARS = 1_000


class ServerToolBackend(Protocol):
    def server_tools(self) -> list[dict[str, Any]]:
        ...

    def execute_server_tool(self, name: str, arguments: dict[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class ToolExecution:
    result: ToolResult
    source: str


class HybridToolExecutor:
    def __init__(
        self,
        *,
        backend: ServerToolBackend | None,
        workdir: Path,
        allowed_tool_names: tuple[str, ...],
        prefer_server: bool = True,
        server_allowlist: frozenset[str] = LLAMA_SERVER_SAFE_TOOLS,
    ) -> None:
        self.backend = backend
        self.workdir = workdir
        self.allowed_tool_names = allowed_tool_names
        self.prefer_server = prefer_server
        self.server_allowlist = server_allowlist
        self._server_tools: dict[str, dict[str, Any]] | None = None

    def tool_definitions(self) -> list[dict[str, Any]]:
        orbit_definitions = tool_definitions(self.allowed_tool_names)
        orbit_by_name = {_definition_name(definition): definition for definition in orbit_definitions}
        if not self.prefer_server:
            return orbit_definitions
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name, item in self._available_server_tools().items():
            if name not in self.allowed_tool_names or name not in self.server_allowlist:
                continue
            definition = _compact_server_definition(name) or (
                orbit_by_name.get(name) if name in {"exec_shell_command", "edit_file", "apply_diff"} else item.get("definition")
            )
            if isinstance(definition, dict):
                merged.append(definition)
                seen.add(name)
        for definition in orbit_definitions:
            name = _definition_name(definition)
            if name and name not in seen:
                merged.append(definition)
        return merged

    def execute(
        self,
        name: str,
        arguments: str | dict[str, Any],
        *,
        chunk_budget: dict[str, int] | None,
    ) -> ToolExecution:
        if name not in self.allowed_tool_names:
            return ToolExecution(ToolResult(name=name, content=f"error: tool not available for this turn: {name}"), "orbit")
        parsed = _parse_arguments(arguments)
        if isinstance(parsed, str):
            return ToolExecution(ToolResult(name=name, content=parsed), "orbit")
        if _prefer_orbit_tool(name, parsed, workdir=self.workdir):
            return ToolExecution(execute_tool(name, parsed, workdir=self.workdir, chunk_budget=chunk_budget), "orbit")
        if self._should_use_server(name):
            try:
                server_args = _server_arguments(name, parsed, workdir=self.workdir)
                if isinstance(server_args, str):
                    return ToolExecution(ToolResult(name=name, content=server_args), "orbit")
                content = self.backend.execute_server_tool(name, server_args)
                return ToolExecution(ToolResult(name=name, content=_bounded_server_result(content)), "llama-server")
            except (LlamaServerError, OSError, TimeoutError) as exc:
                fallback = execute_tool(name, parsed, workdir=self.workdir, chunk_budget=chunk_budget)
                if not fallback.content.startswith("error:"):
                    return ToolExecution(fallback, "orbit")
                return ToolExecution(ToolResult(name=name, content=f"error: llama-server tool failed: {exc}"), "llama-server")
        return ToolExecution(execute_tool(name, parsed, workdir=self.workdir, chunk_budget=chunk_budget), "orbit")

    def _should_use_server(self, name: str) -> bool:
        return (
            self.prefer_server
            and self.backend is not None
            and name in self.server_allowlist
            and name in self._available_server_tools()
        )

    def _available_server_tools(self) -> dict[str, dict[str, Any]]:
        if self._server_tools is not None:
            return self._server_tools
        if self.backend is None:
            self._server_tools = {}
            return self._server_tools
        items = self.backend.server_tools()
        tools: dict[str, dict[str, Any]] = {}
        for item in items:
            name = item.get("tool")
            if isinstance(name, str) and name:
                tools[name] = item
        self._server_tools = tools
        return tools


def _definition_name(definition: dict[str, Any]) -> str | None:
    function = definition.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def _compact_server_definition(name: str) -> dict[str, Any] | None:
    if name == "read_file":
        return _tool_definition(
            "read_file",
            "Read file text, optionally by 1-based lines.",
            {
                "path": _schema("string"),
                "start_line": _schema("integer"),
                "end_line": _schema("integer"),
                "append_loc": _schema("boolean"),
            },
            ["path"],
        )
    if name == "file_glob_search":
        return _tool_definition(
            "file_glob_search",
            "Find files by glob under path.",
            {
                "path": _schema("string"),
                "include": _schema("string"),
                "exclude": _schema("string"),
            },
            ["path"],
        )
    if name == "grep_search":
        return _tool_definition(
            "grep_search",
            "Search regex in files.",
            {
                "path": _schema("string"),
                "pattern": _schema("string"),
                "include": _schema("string"),
                "exclude": _schema("string"),
                "return_line_numbers": _schema("boolean"),
            },
            ["path", "pattern"],
        )
    if name == "write_file":
        return _tool_definition(
            "write_file",
            "Write file content.",
            {
                "path": _schema("string"),
                "content": _schema("string"),
            },
            ["path", "content"],
        )
    return None


def _tool_definition(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _schema(schema_type: str) -> dict[str, str]:
    return {"type": schema_type}


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


def _server_arguments(name: str, arguments: dict[str, Any], *, workdir: Path) -> dict[str, Any] | str:
    if name == "exec_shell_command":
        return prepare_exec_shell_command(arguments, workdir=workdir)
    if name == "edit_file":
        return prepare_edit_file(arguments, workdir=workdir)
    if name == "apply_diff":
        return prepare_apply_diff(arguments, workdir=workdir)
    if name not in {"read_file", "write_file", "file_glob_search", "grep_search"}:
        return dict(arguments)
    updated = dict(arguments)
    path = updated.get("path")
    if name in {"file_glob_search", "grep_search"} and not path:
        updated["path"] = str(workdir.resolve())
        return updated
    if isinstance(path, str) and path and not Path(path).is_absolute():
        updated["path"] = str((workdir / path).resolve())
    return updated


def _prefer_orbit_tool(name: str, arguments: dict[str, Any], *, workdir: Path) -> bool:
    if name == "edit_file":
        return True
    if name != "read_file":
        return False
    if "chunk_index" in arguments:
        return True
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return False
    target = Path(path)
    if not target.is_absolute():
        target = workdir / target
    try:
        resolved = target.resolve()
        root = workdir.expanduser().resolve()
        resolved.relative_to(root)
        return resolved.is_file() and resolved.stat().st_size > MAX_SERVER_READ_FILE_BYTES
    except (OSError, ValueError):
        return False


def _bounded_server_result(content: str) -> str:
    if len(content) <= MAX_SERVER_TOOL_RESULT_CHARS:
        return content
    omitted = len(content) - MAX_SERVER_TOOL_RESULT_CHARS
    return (
        content[:MAX_SERVER_TOOL_RESULT_CHARS].rstrip()
        + f"\n\n[server tool result truncated: {omitted} chars omitted]"
    )
