from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit.runtime.directory_listing import execute_list_directory, list_directory_definition
from orbit.runtime.shell_guardrails import exec_shell_full_definition, execute_exec_shell_full_command
from orbit.runtime.system_info import execute_system_info, system_info_definition
from orbit.runtime.tool_arguments import parse_tool_arguments
from orbit.runtime.web import execute_fetch_url, fetch_url_definition


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str


TOOL_NAMES = ("exec_shell_full_command", "fetch_url", "list_directory", "system_info")
DEFAULT_TOOL_NAMES = TOOL_NAMES


def tool_names() -> tuple[str, ...]:
    return TOOL_NAMES


def default_tool_names() -> tuple[str, ...]:
    return DEFAULT_TOOL_NAMES


def tool_definitions(names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    definitions = [exec_shell_full_definition(), fetch_url_definition(), list_directory_definition(), system_info_definition()]
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
    user_prompt: str | None = None,
) -> ToolResult:
    del chunk_budget
    if name not in TOOL_NAMES:
        return ToolResult(name=name, content=f"error: unknown tool: {name}")
    parsed = parse_tool_arguments(arguments)
    if isinstance(parsed, str):
        return ToolResult(name=name, content=parsed)
    if name == "fetch_url":
        return ToolResult(name=name, content=execute_fetch_url(parsed))
    if name == "list_directory":
        return ToolResult(name=name, content=execute_list_directory(parsed, workdir=workdir))
    if name == "system_info":
        return ToolResult(name=name, content=execute_system_info(parsed))
    return ToolResult(name=name, content=execute_exec_shell_full_command(parsed, workdir=workdir, user_prompt=user_prompt))
