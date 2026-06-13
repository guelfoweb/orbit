from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from orbit.runtime.shell_guardrails import validate_shell_full_contract
from orbit.runtime.tool_arguments import parse_tool_arguments
from orbit.runtime.tools import ToolResult, execute_tool, tool_definitions


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
    """Compatibility facade for the current shell-only tool runtime."""

    def __init__(
        self,
        *,
        backend: ServerToolBackend | None,
        workdir: Path,
        allowed_tool_names: tuple[str, ...],
        user_prompt: str | None = None,
        prefer_server: bool = True,
    ) -> None:
        del backend, prefer_server
        self.workdir = workdir
        self.allowed_tool_names = allowed_tool_names
        self.user_prompt = user_prompt

    def tool_definitions(self) -> list[dict[str, Any]]:
        return tool_definitions(self.allowed_tool_names)

    def execute(
        self,
        name: str,
        arguments: str | dict[str, Any],
        *,
        chunk_budget: dict[str, int] | None,
    ) -> ToolExecution:
        if name not in self.allowed_tool_names:
            return ToolExecution(ToolResult(name=name, content=f"error: tool not available for this turn: {name}"), "orbit")
        if name != "exec_shell_full_command":
            return ToolExecution(ToolResult(name=name, content=f"error: unknown tool: {name}"), "orbit")
        parsed = parse_tool_arguments(arguments)
        if isinstance(parsed, str):
            return ToolExecution(ToolResult(name=name, content=parsed), "orbit")
        contract_error = validate_shell_full_contract(parsed, user_prompt=self.user_prompt)
        if contract_error:
            return ToolExecution(ToolResult(name=name, content=contract_error), "orbit")
        return ToolExecution(execute_tool(name, parsed, workdir=self.workdir, chunk_budget=chunk_budget), "orbit")
