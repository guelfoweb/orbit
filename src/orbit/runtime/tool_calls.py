from __future__ import annotations

from orbit.runtime.tool_backends import HybridToolExecutor, ToolExecution
from orbit.runtime.tool_contract import CanonicalToolDecision
from orbit.runtime.tools import ToolResult


def execute_tool_call(
    tool_call: dict[str, object],
    *,
    chunk_budget: dict[str, int] | None = None,
    executor: HybridToolExecutor,
    canonical_decision: CanonicalToolDecision | None = None,
) -> ToolExecution:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ToolExecution(
            ToolResult(name="unknown", content="error: invalid tool call"),
            "orbit",
            "rejected_schema",
            "invalid_tool_call",
        )
    name = function.get("name")
    arguments = function.get("arguments", {})
    tool_name = name if isinstance(name, str) else "unknown"
    return executor.execute(
        tool_name,
        arguments,
        chunk_budget=chunk_budget,
        canonical_decision=canonical_decision,
    )


def tool_call_id(tool_call: dict[str, object]) -> str:
    value = tool_call.get("id")
    return value if isinstance(value, str) and value else "tool-call"


def tool_call_signature(tool_call: dict[str, object]) -> tuple[str, str]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ("unknown", "{}")
    name = function.get("name")
    arguments = function.get("arguments", "")
    return (name if isinstance(name, str) else "unknown", arguments if isinstance(arguments, str) else str(arguments))
