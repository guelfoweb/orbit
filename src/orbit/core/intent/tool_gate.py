from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .gate import parse_intent_gate_reply
from ..tools.router import ToolRoute


TOOL_GATE_SYSTEM_PROMPT = (
    "You are a tool-call gate for a local CLI. "
    "Answer only YES or NO. "
    "YES means the proposed tool call is appropriate for the user's request and should be executed. "
    "NO means the tool call is unnecessary, contradicted by the prompt, too risky for the request, "
    "or should be handled as normal conversation instead."
)


@dataclass(frozen=True)
class ToolGateDecision:
    confirm: bool
    reason: str


def tool_gate_decision(
    *,
    user_input: str,
    route: ToolRoute | None,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolGateDecision:
    if route is None:
        return ToolGateDecision(confirm=True, reason="no route available")
    if not tool_name:
        return ToolGateDecision(confirm=True, reason="missing tool name")
    lowered = user_input.lower()
    if _forbids_web_access(lowered) and tool_name in {"search_web", "fetch_url"}:
        return ToolGateDecision(confirm=True, reason="prompt forbids web access")
    if _forbids_local_inspection(lowered) and tool_name in {"read_file", "list_files", "stat_path", "bash"}:
        return ToolGateDecision(confirm=True, reason="prompt forbids local inspection")
    if tool_name in {"write_file", "append_file", "replace_in_file", "delete_path", "make_directory"}:
        if _forbids_file_changes(lowered):
            return ToolGateDecision(confirm=True, reason="prompt forbids file changes")
    if tool_name == "fetch_url" and not _has_explicit_url(user_input):
        return ToolGateDecision(confirm=True, reason="fetch_url without explicit user URL")
    if tool_name == "bash" and route.intent_class not in {"shell_task", "machine_inspection", "binary_analysis", "pdf_analysis"}:
        return ToolGateDecision(confirm=True, reason="bash outside shell-like intent")
    return ToolGateDecision(confirm=False, reason="clear tool call")


def tool_gate_messages(
    *,
    user_input: str,
    route: ToolRoute,
    tool_name: str,
    arguments: dict[str, Any],
    reason: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TOOL_GATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User prompt:\n{user_input}\n\n"
                f"Route: {route.intent} / {route.intent_class}\n"
                f"Proposed tool call: {tool_name}\n"
                f"Arguments: {arguments}\n"
                f"Concern: {reason}\n\n"
                "Should this exact tool call be executed? Answer only YES or NO."
            ),
        },
    ]


def parse_tool_gate_reply(content: object) -> bool | None:
    return parse_intent_gate_reply(content)


def _forbids_web_access(lowered: str) -> bool:
    blockers = (
        "without browsing",
        "without searching",
        "without web search",
        "do not browse",
        "don't browse",
        "do not search",
        "don't search",
        "no browsing",
        "no web search",
        "senza navigare",
        "senza cercare online",
        "non cercare online",
        "non usare il web",
    )
    return any(blocker in lowered for blocker in blockers)


def _forbids_local_inspection(lowered: str) -> bool:
    blockers = (
        "without checking local files",
        "without checking files",
        "without inspecting local files",
        "without inspecting the workspace",
        "do not inspect the workspace",
        "don't inspect the workspace",
        "do not check local files",
        "don't check local files",
        "do not read local files",
        "don't read local files",
        "senza leggere file",
        "senza ispezionare",
        "non leggere file",
        "non ispezionare",
    )
    return any(blocker in lowered for blocker in blockers)


def _forbids_file_changes(lowered: str) -> bool:
    blockers = (
        "do not create",
        "don't create",
        "do not write",
        "don't write",
        "do not save",
        "don't save",
        "do not modify",
        "don't modify",
        "without creating",
        "without writing",
        "without saving",
        "non creare",
        "non scrivere",
        "non salvare",
        "senza creare",
        "senza scrivere",
        "senza salvare",
    )
    return any(blocker in lowered for blocker in blockers)


def _has_explicit_url(user_input: str) -> bool:
    return bool(re.search(r"https?://[^\s)>\]\"']+", user_input))
