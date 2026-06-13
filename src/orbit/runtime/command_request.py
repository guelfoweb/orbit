from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from orbit.runtime.tool_arguments import parse_tool_arguments_or_empty

SHELL_TOOL_ALIASES = {"exec_shell_full_command", "shell"}


class ToolRoute(StrEnum):
    CHAT = "CHAT"
    FILESYSTEM = "FILESYSTEM"
    FILE_EDIT = "FILE_EDIT"
    WEB = "WEB"
    MEDIA = "MEDIA"


@dataclass(frozen=True)
class RouteDecision:
    route: ToolRoute
    tool_names: tuple[str, ...] = ()


def parse_tool_command(content: str) -> ToolRoute | None:
    decision = parse_command_decision(content)
    return decision.route if decision is not None else None


def parse_command_decision_from_tool_calls(tool_calls: list[dict[str, Any]]) -> RouteDecision | None:
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        args = parse_tool_arguments_or_empty(function.get("arguments"))
        if _has_command(args) or name in SHELL_TOOL_ALIASES:
            return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    return None


def command_tool_call_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    allowed_tool_names: tuple[str, ...],
) -> dict[str, Any] | None:
    if "exec_shell_full_command" not in set(allowed_tool_names):
        return None
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        args = parse_tool_arguments_or_empty(function.get("arguments"))
        if _has_command(args):
            return _tool_call("exec_shell_full_command", {"command": args["command"]})
    return None


def command_tool_call_from_content(
    content: str,
    allowed_tool_names: tuple[str, ...],
) -> dict[str, Any] | None:
    if "exec_shell_full_command" not in set(allowed_tool_names):
        return None
    value = _parse_command_json_object(content) or _extract_last_json_object(content) or _extract_loose_command_object(content)
    if not _has_command(value):
        return None
    return _tool_call("exec_shell_full_command", {"command": value["command"]})


def parse_command_decision(content: str) -> RouteDecision | None:
    text = _strip_json_fence(content.strip())
    if not text:
        return None
    if _parse_raw_tool_call_decision(text) is not None:
        return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    value = _parse_command_json_object(text)
    if _has_command(value):
        return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    if _has_command(_extract_loose_command_object(text)):
        return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    for line in text.splitlines():
        value = _parse_command_json_object(_strip_json_fence(line.strip()))
        if _has_command(value):
            return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    return None


ROUTE_STREAM_PREFIX_LIMIT = 512


def command_stream_state(text: str, *, max_prefix_chars: int = ROUTE_STREAM_PREFIX_LIMIT) -> str:
    stripped = text.lstrip()
    if len(stripped) > max_prefix_chars:
        return "not_command"
    if not stripped:
        return "pending"
    if _is_partial_prefix(stripped, "<|tool_call>") or _is_partial_prefix(stripped, '{"command"'):
        return "pending"
    if stripped.startswith("<|tool_call>"):
        if "<tool_call|>" in stripped:
            return "route" if parse_tool_command(stripped) is not None else "not_command"
        return "pending"
    if stripped.startswith('{"command"'):
        if _looks_like_complete_json_object(stripped):
            return "route" if parse_tool_command(stripped) is not None else "not_command"
        return "pending"
    return "not_command"


def _is_partial_prefix(text: str, full_prefix: str) -> bool:
    return len(text) < len(full_prefix) and full_prefix.startswith(text)


class CommandStreamFilter:
    def __init__(self, on_delta, *, max_prefix_chars: int = ROUTE_STREAM_PREFIX_LIMIT) -> None:
        self._on_delta = on_delta
        self._max_prefix_chars = max_prefix_chars
        self._buffer = ""
        self._released = False
        self._command_detected = False

    @property
    def command_detected(self) -> bool:
        return self._command_detected

    @property
    def content(self) -> str:
        return self._buffer

    def write(self, text: str) -> None:
        if not text:
            return
        if self._released:
            self._on_delta(text)
            return
        self._buffer += text
        state = command_stream_state(self._buffer, max_prefix_chars=self._max_prefix_chars)
        if state == "pending":
            return
        if state == "route":
            self._command_detected = True
            return
        self._released = True
        self._on_delta(self._buffer)
        self._buffer = ""

    def finish(self) -> None:
        if self._released or self._command_detected or not self._buffer:
            return
        self._on_delta(self._buffer)
        self._released = True
        self._buffer = ""


def tool_names_for_decision(route: ToolRoute, prompt: str | None = None) -> tuple[str, ...]:
    del prompt
    if route == ToolRoute.FILESYSTEM:
        return ("exec_shell_full_command",)
    return ()


def decision_tool_names(decision: RouteDecision, prompt: str | None = None) -> tuple[str, ...]:
    del prompt
    if decision.tool_names:
        return tuple(name for name in decision.tool_names if name == "exec_shell_full_command")
    return tool_names_for_decision(decision.route)


def default_tool_names_for_decision(route: ToolRoute, prompt: str | None = None) -> tuple[str, ...]:
    return tool_names_for_decision(route, prompt)


def refine_decision_for_prompt(decision: RouteDecision, prompt: str | None) -> RouteDecision:
    del prompt
    return decision


def command_like_tool_call(content: str, allowed_tool_names: tuple[str, ...]) -> dict[str, Any] | None:
    return command_tool_call_from_content(content, allowed_tool_names)


def _parse_raw_tool_call_decision(text: str) -> ToolRoute | None:
    if "<|tool_call>" not in text:
        return None
    if "exec_shell_full_command" in text or "call:shell" in text or "command" in text:
        return ToolRoute.FILESYSTEM
    return None


def _parse_command_json_object(content: str) -> dict[str, Any] | None:
    text = _strip_json_fence(content.strip())
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_last_json_object(content: str) -> dict[str, Any] | None:
    matches = re.findall(r"\{[^{}]*\}", content.replace('<|"|>', '"'), flags=re.DOTALL)
    for candidate in reversed(matches):
        normalized = re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', candidate)
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_loose_command_object(content: str) -> dict[str, Any] | None:
    text = _strip_json_fence(content.strip())
    match = re.search(r'"command"\s*:\s*"(?P<command>.*)"\s*\}', text, flags=re.DOTALL)
    if not match:
        return None
    command = _decode_loose_json_string(match.group("command"))
    return {"command": command} if command.strip() else None


def _decode_loose_json_string(value: str) -> str:
    normalized = value.replace("\n", "\\n")
    try:
        decoded = json.loads(f'"{normalized}"')
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, str) else value


def _has_command(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    command = value.get("command")
    return isinstance(command, str) and bool(command.strip())


def _looks_like_complete_json_object(text: str) -> bool:
    in_string = False
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return not text[index + 1 :].strip()
            if depth < 0:
                return False
    return False


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```") or not text.endswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    first = lines[0].strip().lower()
    if first not in {"```", "```json"}:
        return text
    return "\n".join(lines[1:-1]).strip()


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "command-tool-call-1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }
