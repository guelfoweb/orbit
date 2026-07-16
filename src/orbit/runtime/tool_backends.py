from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from orbit.runtime.shell_guardrails import validate_read_only_shell_mutation, validate_shell_full_contract
from orbit.runtime.tool_contract import (
    CanonicalToolDecision,
    canonical_rejection_content,
    validate_canonical_tool_call,
)
from orbit.runtime.tool_arguments import parse_tool_arguments
from orbit.runtime.tools import ToolResult, execute_tool, tool_definitions
from orbit.runtime.web import fetch_url_result_status
from orbit.tool_contract_config import resolve_tool_call_canonical_gate


class ServerToolBackend(Protocol):
    def server_tools(self) -> list[dict[str, Any]]:
        ...

    def execute_server_tool(self, name: str, arguments: dict[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class ToolExecution:
    result: ToolResult
    source: str
    terminal_outcome: str = "executed"
    terminal_reason: str | None = None


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
        canonical_decision: CanonicalToolDecision | None = None,
    ) -> ToolExecution:
        gate = resolve_tool_call_canonical_gate()
        canonical_validated = False
        decision = canonical_decision
        if decision is None and gate.enabled:
            decision = validate_canonical_tool_call(
                name,
                arguments,
                tool_definitions=tool_definitions(),
                allowed_tool_names=self.allowed_tool_names,
                workdir=self.workdir,
                user_prompt=self.user_prompt,
            )
        if decision is not None:
            if not decision.accepted:
                content = canonical_rejection_content(name, decision)
                return ToolExecution(
                    ToolResult(name=name, content=content),
                    "orbit",
                    decision.terminal_decision,
                    decision.rejection_code,
                )
            assert decision.normalized_call is not None
            assert name == decision.normalized_call.name
            name = decision.normalized_call.name
            arguments = decision.normalized_call.arguments
            canonical_validated = True
        elif name not in self.allowed_tool_names:
            # Legacy rollback keeps the historical permission boundary. The
            # default canonical path receives this decision from the runtime.
            return ToolExecution(
                ToolResult(name=name, content=f"error: tool not available for this turn: {name}"),
                "orbit",
                "rejected_permission",
                "tool_not_enabled",
            )
        if name not in {"exec_shell_full_command", "fetch_url", "list_directory", "system_info"}:
            return ToolExecution(
                ToolResult(name=name, content=f"error: unknown tool: {name}"),
                "orbit",
                "rejected_schema",
                "unknown_tool",
            )
        parsed = parse_tool_arguments(arguments)
        if isinstance(parsed, str):
            return ToolExecution(ToolResult(name=name, content=parsed), "orbit", "rejected_parse", "invalid_arguments")
        if name == "exec_shell_full_command" and not canonical_validated:
            read_only_mutation_error = validate_read_only_shell_mutation(parsed, user_prompt=self.user_prompt)
            if read_only_mutation_error:
                return ToolExecution(
                    ToolResult(name=name, content=read_only_mutation_error),
                    "orbit",
                    "rejected_policy",
                    "read_only_mutation",
                )
            contract_error = validate_shell_full_contract(parsed, user_prompt=self.user_prompt)
            if contract_error:
                return ToolExecution(
                    ToolResult(name=name, content=contract_error),
                    "orbit",
                    "rejected_guardrail",
                    "shell_contract",
                )
        result = execute_tool(name, parsed, workdir=self.workdir, chunk_budget=chunk_budget, user_prompt=self.user_prompt)
        outcome = "runtime_error" if _tool_result_is_error(name, result.content) else "executed"
        return ToolExecution(result, "orbit", outcome, "tool_error" if outcome == "runtime_error" else None)


def _tool_result_is_error(name: str, content: str) -> bool:
    if content.startswith("error:") or "\nerror:" in content:
        return True
    if content.startswith("shell_command_failed: true"):
        return True
    if name == "fetch_url":
        status = fetch_url_result_status(content)
        return status is not None and status != "ok"
    return False
