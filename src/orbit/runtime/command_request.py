from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from orbit.runtime.tool_arguments import parse_tool_arguments_or_empty

SHELL_TOOL_ALIASES = {"exec_shell_full_command", "shell"}
WEB_SEARCH_TOOL_ALIASES = {"orbit-web-search", "orbit_web_search"}
FETCH_URL_TOOL_ALIASES = {"fetch_url"}
LIST_DIRECTORY_TOOL_ALIASES = {"list_directory"}
LIST_DIRECTORY_KEYS = ("path", "recursive", "max_depth", "max_entries", "include_hidden", "dirs_first", "files_only", "dirs_only")
SYSTEM_INFO_TOOL_ALIASES = {"system_info"}
SYSTEM_INFO_KEYS = ("include_disks", "include_cpu", "include_memory", "include_os", "include_runtime", "include_gpu", "human_readable")


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


class RouteOutputClass(StrEnum):
    CANONICAL = "canonical"
    LEGACY_TOLERATED = "legacy_tolerated"
    DIRECT_PROSE = "direct_prose"
    MALFORMED = "malformed"
    CONTROL_LOOP = "control_loop"


@dataclass(frozen=True)
class RouteOutputDiagnostic:
    classification: RouteOutputClass
    reason: str
    canonical: bool
    parser_accepted: bool


_CONTROL_TOKEN_CYCLE = (100, 45518, 107, 101)
_CONTROL_ONLY_RE = re.compile(r"(?:\s*<\|channel>thought\s*<channel\|>\s*)+")


def classify_route_output(
    content: str,
    *,
    parsed_decision: RouteDecision | None,
    parser_source: str | None,
    direct_prose: bool,
    finish_reason: str | None,
    output_tokens: int | None,
    raw_token_ids: list[int] | tuple[int, ...] | None = None,
) -> RouteOutputDiagnostic:
    """Classify a completed route output without changing its interpretation."""

    parser_accepted = parsed_decision is not None
    if _has_control_token_cycle(raw_token_ids):
        return RouteOutputDiagnostic(RouteOutputClass.CONTROL_LOOP, "repeated_control_token_cycle", False, parser_accepted)
    text = content.strip()
    if text and _CONTROL_ONLY_RE.fullmatch(text):
        return RouteOutputDiagnostic(RouteOutputClass.CONTROL_LOOP, "control_only_text", False, parser_accepted)
    if not text and finish_reason == "length" and isinstance(output_tokens, int) and output_tokens >= len(_CONTROL_TOKEN_CYCLE) * 2:
        return RouteOutputDiagnostic(RouteOutputClass.CONTROL_LOOP, "empty_visible_control_output", False, parser_accepted)

    canonical_reason = _canonical_route_reason(text)
    if canonical_reason is not None and parser_accepted and parser_source == "content":
        return RouteOutputDiagnostic(RouteOutputClass.CANONICAL, canonical_reason, True, True)
    if parser_accepted:
        reason = "normalized_backend_tool_calls" if parser_source == "tool_calls" else "parser_tolerated_noncanonical"
        return RouteOutputDiagnostic(RouteOutputClass.LEGACY_TOLERATED, reason, False, True)
    if direct_prose and text and not _looks_like_route_syntax(text):
        return RouteOutputDiagnostic(RouteOutputClass.DIRECT_PROSE, "accepted_direct_answer", False, False)
    reason = "empty_output" if not text else "unaccepted_route_syntax" if _looks_like_route_syntax(text) else "unaccepted_output"
    return RouteOutputDiagnostic(RouteOutputClass.MALFORMED, reason, False, False)


def _has_control_token_cycle(raw_token_ids: list[int] | tuple[int, ...] | None) -> bool:
    if raw_token_ids is None or len(raw_token_ids) < len(_CONTROL_TOKEN_CYCLE) * 2:
        return False
    return all(token == _CONTROL_TOKEN_CYCLE[index % len(_CONTROL_TOKEN_CYCLE)] for index, token in enumerate(raw_token_ids))


def _canonical_route_reason(text: str) -> str | None:
    if not text:
        return None
    try:
        value = json.loads(text, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, _DuplicateJsonKey):
        return None
    if not isinstance(value, dict):
        return None
    keys = set(value)
    if keys == {"route"} and value.get("route") == ToolRoute.CHAT.value:
        return "canonical_chat"
    if keys == {"command"} and _nonempty_string(value.get("command")):
        return "canonical_command"
    if keys == {"url"} and _nonempty_string(value.get("url")):
        return "canonical_url"
    if keys and keys <= set(LIST_DIRECTORY_KEYS) and _canonical_list_directory(value):
        return "canonical_list_directory"
    if keys and keys <= set(SYSTEM_INFO_KEYS) and all(isinstance(arg, bool) for arg in value.values()):
        return "canonical_system_info"
    return None


class _DuplicateJsonKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKey(key)
        value[key] = item
    return value


def _canonical_list_directory(value: dict[str, Any]) -> bool:
    for key, item in value.items():
        if key == "path" and not isinstance(item, str):
            return False
        if key in {"recursive", "include_hidden", "dirs_first", "files_only", "dirs_only"} and not isinstance(item, bool):
            return False
        if key == "max_depth" and item is not None and (isinstance(item, bool) or not isinstance(item, int)):
            return False
        if key == "max_entries" and (isinstance(item, bool) or not isinstance(item, int)):
            return False
    return True


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _looks_like_route_syntax(text: str) -> bool:
    if text.startswith(("{", "[", "```", "<|tool_call>")):
        return True
    return any(marker in text for marker in ('{"route"', '{"command"', '{"url"', '{"path"', '{"include_'))


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
        if name in WEB_SEARCH_TOOL_ALIASES and _web_search_command_from_args(args):
            return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
        if name in FETCH_URL_TOOL_ALIASES and _has_url(args):
            return RouteDecision(ToolRoute.FILESYSTEM, ("fetch_url",))
        if name in LIST_DIRECTORY_TOOL_ALIASES and _has_list_directory_args(args):
            return RouteDecision(ToolRoute.FILESYSTEM, ("list_directory",))
        if name in SYSTEM_INFO_TOOL_ALIASES and _valid_system_info_args(args):
            return RouteDecision(ToolRoute.FILESYSTEM, ("system_info",))
    return None


def command_tool_call_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    allowed_tool_names: tuple[str, ...],
) -> dict[str, Any] | None:
    allowed = set(allowed_tool_names)
    if not ({"exec_shell_full_command", "fetch_url", "list_directory", "system_info"} & allowed):
        return None
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        raw_arguments = function.get("arguments")
        args = parse_tool_arguments_or_empty(raw_arguments)
        parsed_valid_command = _has_command(args)
        if not parsed_valid_command and isinstance(raw_arguments, str):
            args = _extract_loose_command_object(raw_arguments) or _extract_last_json_object(raw_arguments) or {}
        if _has_command(args):
            if name == "exec_shell_full_command" and isinstance(raw_arguments, str) and parsed_valid_command:
                return tool_call
            return _tool_call("exec_shell_full_command", {"command": args["command"]})
        if name in FETCH_URL_TOOL_ALIASES and _has_url(args):
            if "fetch_url" not in allowed:
                return None
            if name == "fetch_url" and isinstance(raw_arguments, str):
                return tool_call
            return _tool_call("fetch_url", {"url": args["url"]})
        if name in LIST_DIRECTORY_TOOL_ALIASES and _has_list_directory_args(args):
            if "list_directory" not in allowed:
                return None
            if name == "list_directory" and isinstance(raw_arguments, str):
                return tool_call
            return _tool_call("list_directory", _list_directory_args(args))
        if name in SYSTEM_INFO_TOOL_ALIASES and _valid_system_info_args(args):
            if "system_info" not in allowed:
                return None
            if name == "system_info" and isinstance(raw_arguments, str):
                return tool_call
            return _tool_call("system_info", _system_info_args(args))
        if name in WEB_SEARCH_TOOL_ALIASES:
            command = _web_search_command_from_args(args)
            if command and "exec_shell_full_command" in allowed:
                return _tool_call("exec_shell_full_command", {"command": command})
    return None


def command_tool_call_from_content(
    content: str,
    allowed_tool_names: tuple[str, ...],
) -> dict[str, Any] | None:
    allowed = set(allowed_tool_names)
    if not ({"exec_shell_full_command", "fetch_url", "list_directory", "system_info"} & allowed):
        return None
    raw_command = _extract_raw_tool_call_command(content)
    if raw_command:
        if "exec_shell_full_command" not in allowed:
            return None
        return _tool_call("exec_shell_full_command", {"command": raw_command})
    raw_url = _extract_raw_tool_call_url(content)
    if raw_url:
        if "fetch_url" not in allowed:
            return None
        return _tool_call("fetch_url", {"url": raw_url})
    raw_listing = _extract_raw_tool_call_list_directory(content)
    if raw_listing:
        if "list_directory" not in allowed:
            return None
        return _tool_call("list_directory", raw_listing)
    raw_system_info = _extract_raw_tool_call_system_info(content)
    if raw_system_info is not None:
        if "system_info" not in allowed:
            return None
        return _tool_call("system_info", raw_system_info)
    value = _parse_command_json_object(content) or _extract_last_json_object(content) or _extract_loose_command_object(content)
    if not _has_command(value):
        if _has_url(value) and "fetch_url" in allowed:
            return _tool_call("fetch_url", {"url": value["url"]})
        if _has_list_directory_args(value) and "list_directory" in allowed:
            return _tool_call("list_directory", _list_directory_args(value))
        if _has_system_info_args(value) and "system_info" in allowed:
            return _tool_call("system_info", _system_info_args(value))
        return None
    return _tool_call("exec_shell_full_command", {"command": value["command"]})


def parse_command_decision(content: str) -> RouteDecision | None:
    text = _strip_json_fence(content.strip())
    if not text:
        return None
    if _extract_raw_tool_call_command(text):
        return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    if _extract_raw_tool_call_url(text):
        return RouteDecision(ToolRoute.FILESYSTEM, ("fetch_url",))
    if _extract_raw_tool_call_list_directory(text):
        return RouteDecision(ToolRoute.FILESYSTEM, ("list_directory",))
    if _extract_raw_tool_call_system_info(text) is not None:
        return RouteDecision(ToolRoute.FILESYSTEM, ("system_info",))
    if _parse_raw_tool_call_decision(text) is not None:
        return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command", "fetch_url", "list_directory", "system_info"))
    value = _parse_command_json_object(text)
    if _has_command(value):
        return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    if _has_url(value):
        return RouteDecision(ToolRoute.FILESYSTEM, ("fetch_url",))
    if _has_list_directory_args(value):
        return RouteDecision(ToolRoute.FILESYSTEM, ("list_directory",))
    if _has_system_info_args(value):
        return RouteDecision(ToolRoute.FILESYSTEM, ("system_info",))
    if _has_chat_route(value):
        return RouteDecision(ToolRoute.CHAT)
    if _has_command(_extract_loose_command_object(text)):
        return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
    for line in text.splitlines():
        value = _parse_command_json_object(_strip_json_fence(line.strip()))
        if _has_command(value):
            return RouteDecision(ToolRoute.FILESYSTEM, ("exec_shell_full_command",))
        if _has_url(value):
            return RouteDecision(ToolRoute.FILESYSTEM, ("fetch_url",))
        if _has_list_directory_args(value):
            return RouteDecision(ToolRoute.FILESYSTEM, ("list_directory",))
        if _has_system_info_args(value):
            return RouteDecision(ToolRoute.FILESYSTEM, ("system_info",))
        if _has_chat_route(value):
            return RouteDecision(ToolRoute.CHAT)
    return None


ROUTE_STREAM_PREFIX_LIMIT = 512


def command_stream_state(text: str, *, max_prefix_chars: int = ROUTE_STREAM_PREFIX_LIMIT) -> str:
    stripped = text.lstrip()
    if len(stripped) > max_prefix_chars:
        return "not_command"
    if not stripped:
        return "pending"
    if (
        _is_partial_prefix(stripped, "<|tool_call>")
        or _is_partial_prefix(stripped, '{"command"')
        or _is_partial_prefix(stripped, '{"url"')
        or _is_partial_prefix(stripped, '{"path"')
        or _is_partial_prefix(stripped, '{"include_')
        or _is_partial_prefix(stripped, '{"route"')
    ):
        return "pending"
    if stripped.startswith("<|tool_call>"):
        if "<tool_call|>" in stripped:
            return "route" if parse_tool_command(stripped) is not None else "not_command"
        return "pending"
    if (
        stripped.startswith('{"command"')
        or stripped.startswith('{"url"')
        or stripped.startswith('{"path"')
        or stripped.startswith('{"include_')
        or stripped.startswith('{"route"')
    ):
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
        return ("exec_shell_full_command", "fetch_url", "list_directory", "system_info")
    return ()


def decision_tool_names(decision: RouteDecision, prompt: str | None = None) -> tuple[str, ...]:
    del prompt
    if decision.tool_names:
        return tuple(name for name in decision.tool_names if name in {"exec_shell_full_command", "fetch_url", "list_directory", "system_info"})
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
    if _extract_raw_tool_call_command(text):
        return ToolRoute.FILESYSTEM
    if _extract_raw_tool_call_url(text):
        return ToolRoute.FILESYSTEM
    if _extract_raw_tool_call_list_directory(text):
        return ToolRoute.FILESYSTEM
    if _extract_raw_tool_call_system_info(text) is not None:
        return ToolRoute.FILESYSTEM
    if "exec_shell_full_command" in text or "call:shell" in text or "command" in text or "list_directory" in text or "system_info" in text:
        return ToolRoute.FILESYSTEM
    return None


def _extract_raw_tool_call_command(content: str) -> str | None:
    text = content.replace('<|"|>', '"')
    match = re.search(
        r"<\|tool_call\>\s*call:(?P<name>[A-Za-z0-9_-]+)\s*(?P<args>\{.*?\})\s*<tool_call\|>",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return _extract_loose_raw_shell_command(text)
    name = match.group("name")
    raw_args = match.group("args")
    value = _parse_command_json_object(raw_args) or _extract_last_json_object(raw_args) or _extract_loose_key_value_object(raw_args)
    if name in SHELL_TOOL_ALIASES and _has_command(value):
        return value["command"]
    if name in WEB_SEARCH_TOOL_ALIASES:
        return _web_search_command_from_args(value)
    return None


def _extract_raw_tool_call_url(content: str) -> str | None:
    text = content.replace('<|"|>', '"')
    match = re.search(
        r"<\|tool_call\>\s*call:(?P<name>[A-Za-z0-9_-]+)\s*(?P<args>\{.*?\})\s*<tool_call\|>",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    name = match.group("name")
    if name not in FETCH_URL_TOOL_ALIASES:
        return None
    raw_args = match.group("args")
    value = _parse_command_json_object(raw_args) or _extract_last_json_object(raw_args) or _extract_loose_key_value_object(raw_args)
    if _has_url(value):
        return value["url"]
    return None


def _extract_raw_tool_call_list_directory(content: str) -> dict[str, Any] | None:
    text = content.replace('<|"|>', '"')
    match = re.search(
        r"<\|tool_call\>\s*call:(?P<name>[A-Za-z0-9_-]+)\s*(?P<args>\{.*?\})\s*<tool_call\|>",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    name = match.group("name")
    if name not in LIST_DIRECTORY_TOOL_ALIASES:
        return None
    raw_args = match.group("args")
    value = _parse_command_json_object(raw_args) or _extract_last_json_object(raw_args) or _extract_loose_key_value_object(raw_args)
    if _has_list_directory_args(value):
        return _list_directory_args(value)
    return None


def _extract_raw_tool_call_system_info(content: str) -> dict[str, Any] | None:
    text = content.replace('<|"|>', '"')
    match = re.search(
        r"<\|tool_call\>\s*call:(?P<name>[A-Za-z0-9_-]+)\s*(?P<args>\{.*?\})\s*<tool_call\|>",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    name = match.group("name")
    if name not in SYSTEM_INFO_TOOL_ALIASES:
        return None
    raw_args = match.group("args")
    value = _parse_command_json_object(raw_args)
    if value is None:
        value = _extract_last_json_object(raw_args) or _extract_loose_key_value_object(raw_args)
    if _valid_system_info_args(value):
        return _system_info_args(value)
    return None


def _web_search_command_from_args(value: dict[str, Any] | None) -> str | None:
    if not isinstance(value, dict):
        return None
    query = value.get("query") or value.get("q")
    if not isinstance(query, str) or not query.strip():
        return None
    return f"orbit-web-search {shlex.quote(query.strip())}"


def _has_list_directory_args(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict) or _has_command(value) or _has_url(value):
        return False
    if not any(key in value for key in LIST_DIRECTORY_KEYS):
        return False
    if "path" in value and not isinstance(value.get("path"), str):
        return False
    return True


def _list_directory_args(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in LIST_DIRECTORY_KEYS if key in value}


def _has_system_info_args(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict) or _has_command(value) or _has_url(value) or _has_list_directory_args(value):
        return False
    if not any(key in value for key in SYSTEM_INFO_KEYS):
        return False
    return _valid_system_info_args(value)


def _valid_system_info_args(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    return all(key in SYSTEM_INFO_KEYS and isinstance(arg, bool) for key, arg in value.items())


def _system_info_args(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in SYSTEM_INFO_KEYS if key in value}


def _has_chat_route(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    if _has_command(value) or _has_url(value) or _has_list_directory_args(value) or _has_system_info_args(value):
        return False
    route = value.get("route")
    return isinstance(route, str) and route.strip().upper() == ToolRoute.CHAT.value


def _extract_loose_key_value_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if not text.startswith("{") or not text.endswith("}"):
        return None
    body = text[1:-1].strip()
    if not body:
        return None
    match = re.fullmatch(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"(?P<value>[^\"]+)\"", body, flags=re.DOTALL)
    if not match:
        return None
    return {match.group("key"): match.group("value")}


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


def _extract_loose_raw_shell_command(text: str) -> str | None:
    call_match = re.search(
        r'<\|tool_call\>\s*call\s*\(\s*(?P<name>[A-Za-z0-9_-]+)\s*,\s*"(?P<command>(?:[^"\\]|\\.)*)"\s*\)\s*(?=<\|tool_call\>|<tool_call\|>|$)',
        text,
        flags=re.DOTALL,
    )
    if call_match:
        name = call_match.group("name")
        if name not in SHELL_TOOL_ALIASES:
            return None
        command = _decode_loose_json_string(call_match.group("command")).strip()
        return command or None
    match = re.search(
        r"<\|tool_call\>\s*call\s+shell\s+(?P<command>.*?)(?=<\|tool_call\>|<tool_call\|>|$)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    command = match.group("command").strip()
    if not command or "{" in command or "}" in command:
        return None
    return command


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


def _has_url(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    url = value.get("url")
    return isinstance(url, str) and bool(url.strip())


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
