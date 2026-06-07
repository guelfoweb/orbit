from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


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


def parse_tool_route(content: str) -> ToolRoute | None:
    decision = parse_route_decision(content)
    return decision.route if decision is not None else None


def parse_route_decision_from_tool_calls(tool_calls: list[dict[str, Any]]) -> RouteDecision | None:
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        args = _parse_tool_call_arguments(function.get("arguments"))
        route = _route_from_name(name)
        if route is None:
            route = _route_for_tool_name(name)
        if route is None:
            decision = _parse_json_decision(json.dumps(args, ensure_ascii=False))
            if decision is None:
                continue
            return decision
        selected = (name,) if name in route_tool_names(route) else _extract_requested_tool_names(args)
        return RouteDecision(route, _valid_route_tools(route, selected))
    return None


def route_tool_call_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    allowed_tool_names: tuple[str, ...],
) -> dict[str, Any] | None:
    allowed = set(allowed_tool_names)
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        args = _parse_tool_call_arguments(function.get("arguments"))
        if name in allowed:
            return _tool_call(name, args)
        route = _route_from_name(name)
        if route is None:
            decision = _parse_json_decision(json.dumps(args, ensure_ascii=False))
            if decision is None:
                continue
            route = decision.route
        selected = _valid_route_tools(route, _extract_requested_tool_names(args))
        if len(selected) != 1 or selected[0] not in allowed:
            continue
        tool_args = {key: value for key, value in args.items() if key not in {"tool", "tools", "route", "_route"}}
        if tool_args or selected[0] == "list_files":
            return _tool_call(selected[0], tool_args)
    return None


def route_tool_call_from_content(
    content: str,
    allowed_tool_names: tuple[str, ...],
) -> dict[str, Any] | None:
    value = _parse_route_json_object(content)
    if value is None:
        return None
    decision = _parse_json_decision(json.dumps(value, ensure_ascii=False))
    if decision is None:
        return None
    selected = _valid_route_tools(decision.route, _extract_requested_tool_names(value))
    if len(selected) != 1 or selected[0] not in set(allowed_tool_names):
        return None
    explicit_args = value.get("args", value.get("arguments"))
    if isinstance(explicit_args, dict):
        tool_args = explicit_args
    else:
        tool_args = {key: item for key, item in value.items() if key not in {"_route", "route", "tool", "tools"}}
    if not tool_args:
        return None
    return _tool_call(selected[0], tool_args)


def parse_route_decision(content: str) -> RouteDecision | None:
    text = _strip_json_fence(content.strip())
    if not text:
        return None
    raw_route = _parse_raw_tool_call_route(text)
    if raw_route is not None:
        return RouteDecision(raw_route)
    decision = _parse_json_decision(text)
    if decision is not None:
        return decision
    malformed_route = _parse_malformed_route(text)
    if malformed_route is not None:
        return malformed_route
    for line in text.splitlines():
        decision = _parse_json_decision(_strip_json_fence(line.strip()))
        if decision is not None:
            return decision
    return None


def _route_from_name(name: str) -> ToolRoute | None:
    normalized = name.strip().upper()
    for candidate in ToolRoute:
        if normalized == candidate.value:
            return candidate
    return None


def _route_for_tool_name(name: str) -> ToolRoute | None:
    for route in ToolRoute:
        if name in route_tool_names(route):
            return route
    return None


def _parse_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_route_json_object(content: str) -> dict[str, Any] | None:
    text = _strip_json_fence(content.strip())
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


ROUTE_STREAM_PREFIX_LIMIT = 512


def route_stream_state(text: str, *, max_prefix_chars: int = ROUTE_STREAM_PREFIX_LIMIT) -> str:
    stripped = text.lstrip()
    if len(stripped) > max_prefix_chars:
        return "not_route"
    if not stripped:
        return "pending"
    if (
        _is_partial_prefix(stripped, "<|tool_call>")
        or _is_partial_prefix(stripped, '{"_route"')
    ):
        return "pending"
    if stripped.startswith("<|tool_call>"):
        if "<tool_call|>" in stripped:
            return "route" if parse_tool_route(stripped) is not None else "not_route"
        return "pending"
    if stripped.startswith('{"_route"'):
        if _looks_like_complete_json_object(stripped):
            return "route" if parse_tool_route(stripped) is not None else "not_route"
        return "pending"
    return "not_route"


def _is_partial_prefix(text: str, full_prefix: str) -> bool:
    return len(text) < len(full_prefix) and full_prefix.startswith(text)


class RouteStreamFilter:
    def __init__(self, on_delta, *, max_prefix_chars: int = ROUTE_STREAM_PREFIX_LIMIT) -> None:
        self._on_delta = on_delta
        self._max_prefix_chars = max_prefix_chars
        self._buffer = ""
        self._released = False
        self._route_detected = False

    @property
    def route_detected(self) -> bool:
        return self._route_detected

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
        state = route_stream_state(self._buffer, max_prefix_chars=self._max_prefix_chars)
        if state == "pending":
            return
        if state == "route":
            self._route_detected = True
            return
        self._released = True
        self._on_delta(self._buffer)
        self._buffer = ""

    def finish(self) -> None:
        if self._released or self._route_detected or not self._buffer:
            return
        self._on_delta(self._buffer)
        self._released = True
        self._buffer = ""


def _parse_raw_tool_call_route(text: str) -> ToolRoute | None:
    if "<|tool_call>" not in text:
        return None
    normalized = text.upper()
    if "FILESYSTEM" in normalized:
        return ToolRoute.FILESYSTEM
    if "FILE_EDIT" in normalized:
        return ToolRoute.FILE_EDIT
    if "WEB" in normalized:
        return ToolRoute.WEB
    if "MEDIA" in normalized:
        return ToolRoute.MEDIA
    if any(name in text for name in ("list_files", "read_file", "stat_path", "file_glob_search", "grep_search", "exec_shell_command")):
        return ToolRoute.FILESYSTEM
    if any(name in text for name in ("write_file", "append_file", "replace_in_file", "make_directory", "delete_path")):
        return ToolRoute.FILE_EDIT
    if any(name in text for name in ("search_web", "fetch_url")):
        return ToolRoute.WEB
    return None


def _parse_malformed_route(text: str) -> RouteDecision | None:
    stripped = text.strip()
    if not stripped.startswith('{"_route"'):
        return None
    normalized = stripped.upper()
    for candidate in ToolRoute:
        if f'"{candidate.value}"' in normalized:
            return RouteDecision(candidate)
    return None


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


def _parse_json_decision(text: str) -> RouteDecision | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    route = value.get("_route", value.get("route", value.get("tool")))
    if not isinstance(route, str):
        return None
    normalized = route.strip().upper()
    for candidate in ToolRoute:
        if normalized == candidate.value:
            return RouteDecision(candidate, _valid_route_tools(candidate, _extract_requested_tool_names(value)))
    return None


def _extract_requested_tool_names(value: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    single = value.get("tool")
    if isinstance(single, str):
        names.append(single)
    multiple = value.get("tools")
    if isinstance(multiple, list):
        names.extend(item for item in multiple if isinstance(item, str))
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in names:
        stripped = name.strip()
        if stripped and stripped not in seen:
            cleaned.append(stripped)
            seen.add(stripped)
    return tuple(cleaned)


def _valid_route_tools(route: ToolRoute, names: tuple[str, ...]) -> tuple[str, ...]:
    allowed = set(route_tool_names(route))
    return tuple(name for name in names if name in allowed)


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


def route_tool_names(route: ToolRoute, prompt: str | None = None) -> tuple[str, ...]:
    if route == ToolRoute.CHAT:
        return ()
    if route == ToolRoute.FILESYSTEM:
        return ("list_files", "read_file", "file_glob_search", "grep_search", "exec_shell_command")
    if route == ToolRoute.FILE_EDIT:
        return (
            "read_file",
            "make_directory",
            "delete_path",
            "write_file",
            "edit_file",
            "apply_diff",
        )
    if route == ToolRoute.WEB:
        return ("fetch_url", "search_web")
    return ()


def decision_tool_names(decision: RouteDecision, prompt: str | None = None) -> tuple[str, ...]:
    decision = refine_decision_for_prompt(decision, prompt)
    if decision.tool_names:
        return decision.tool_names
    return route_tool_names(decision.route, prompt)


def refine_decision_for_prompt(decision: RouteDecision, prompt: str | None) -> RouteDecision:
    if not prompt:
        return decision
    text = prompt.lower()
    if _contains_actual_diff(text):
        return RouteDecision(ToolRoute.FILE_EDIT, ("apply_diff",))
    if "replace line" in text or "sostituisci riga" in text:
        return RouteDecision(ToolRoute.FILE_EDIT, ("edit_file",))
    if "changes " in text and " appends " in text:
        return RouteDecision(ToolRoute.FILE_EDIT, ("read_file", "edit_file"))
    if "search inside" in text or "cerca dentro" in text:
        return RouteDecision(ToolRoute.FILESYSTEM, ("grep_search",))
    return decision


def _contains_actual_diff(text: str) -> bool:
    return ("diff --git " in text or "\n--- " in text) and "\n+++ " in text and "\n@@" in text


def route_like_tool_call(content: str, allowed_tool_names: tuple[str, ...]) -> dict[str, Any] | None:
    if parse_tool_route(content) != ToolRoute.WEB:
        return None
    args = _extract_last_json_object(content)
    if not args:
        return None
    if "search_web" in allowed_tool_names and isinstance(args.get("query"), str):
        return _tool_call("search_web", {"query": args["query"]})
    if "fetch_url" in allowed_tool_names and isinstance(args.get("url"), str):
        return _tool_call("fetch_url", {"url": args["url"]})
    return None


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


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "route-like-tool-call-1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def route_json(route: ToolRoute) -> dict[str, Any]:
    return {"_route": route.value}
