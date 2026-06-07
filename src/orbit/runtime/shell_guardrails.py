from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


DEFAULT_SHELL_TIMEOUT = 10
MAX_SHELL_TIMEOUT = 15
DEFAULT_SHELL_OUTPUT_BYTES = 12_000
MAX_SHELL_OUTPUT_BYTES = 12_000

_SHELL_META_CHARS = frozenset("|&;<>`$\\\n\r")
_ALLOWED_COMMANDS = frozenset({"pwd", "ls", "find", "du", "df", "wc", "head", "tail", "file", "stat", "cat"})
_ALLOWED_SIMPLE_FLAGS = {
    "ls": frozenset({"-1", "-a", "-l", "-h", "-F", "-la", "-al", "-lh", "-hl", "-lah", "-lha", "-alh", "-ahl", "-hal", "-hla"}),
    "du": frozenset({"-s", "-h", "-sh", "-hs"}),
    "df": frozenset({"-h", "-k", "-m"}),
    "wc": frozenset({"-l", "-w", "-c", "-m"}),
    "file": frozenset({"-b"}),
    "stat": frozenset({"-c"}),
    "cat": frozenset(),
}


def exec_shell_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "exec_shell_command",
            "description": (
                "Run one bounded read-only command in workdir. "
                "Allowed: pwd, ls, find, du, df, wc, head, tail, file, stat, cat. "
                "Use ls -F for listing. No ls -R, shell operators, or outside paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                    },
                    "timeout": {
                        "type": "integer",
                    },
                    "max_output_size": {
                        "type": "integer",
                    },
                },
                "required": ["command"],
            },
        },
    }


def prepare_exec_shell_command(arguments: dict[str, Any], *, workdir: Path) -> dict[str, Any] | str:
    raw_command = arguments.get("command")
    if not isinstance(raw_command, str) or not raw_command.strip():
        return "error: exec_shell_command requires a non-empty command string"
    if any(char in raw_command for char in _SHELL_META_CHARS):
        return "error: shell operators, redirects, variables, escapes, and multi-line commands are not allowed"
    try:
        tokens = shlex.split(raw_command)
    except ValueError as exc:
        return f"error: invalid shell command syntax: {exc}"
    if not tokens:
        return "error: exec_shell_command requires a command"
    command = tokens[0]
    if command not in _ALLOWED_COMMANDS:
        return f"error: command not allowed: {command}"
    validation_error = _validate_command_tokens(command, tokens[1:])
    if validation_error:
        return validation_error
    timeout = _bounded_int(arguments.get("timeout"), default=DEFAULT_SHELL_TIMEOUT, maximum=MAX_SHELL_TIMEOUT)
    output_size = _bounded_int(arguments.get("max_output_size"), default=DEFAULT_SHELL_OUTPUT_BYTES, maximum=MAX_SHELL_OUTPUT_BYTES)
    safe_command = shlex.join(tokens)
    return {
        "command": f"cd {shlex.quote(str(workdir.resolve()))} && {safe_command}",
        "timeout": timeout,
        "max_output_size": output_size,
    }


def _validate_command_tokens(command: str, args: list[str]) -> str | None:
    if command == "pwd":
        return None if not args else "error: pwd does not accept arguments here"
    if command == "find":
        return _validate_find(args)
    if command in {"head", "tail"}:
        return _validate_head_tail(args)
    if command in _ALLOWED_SIMPLE_FLAGS:
        return _validate_simple_command(command, args)
    return "error: command not configured"


def _validate_simple_command(command: str, args: list[str]) -> str | None:
    allowed_flags = _ALLOWED_SIMPLE_FLAGS[command]
    skip_next_path_validation = False
    for token in args:
        if skip_next_path_validation:
            skip_next_path_validation = False
            continue
        if command == "stat" and token == "-c":
            skip_next_path_validation = True
            continue
        if token.startswith("-"):
            if token.startswith("--max-depth=") and command == "du":
                if _int_suffix(token.removeprefix("--max-depth="), maximum=3) is None:
                    return "error: du --max-depth must be between 0 and 3"
                continue
            if token not in allowed_flags:
                return f"error: flag not allowed for {command}: {token}"
            continue
        path_error = _validate_relative_path_token(token)
        if path_error:
            return path_error
    return None


def _validate_find(args: list[str]) -> str | None:
    if not args:
        return None
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-maxdepth", "-mindepth"}:
            if index + 1 >= len(args) or _int_suffix(args[index + 1], maximum=3) is None:
                return f"error: {token} must be between 0 and 3"
            index += 2
            continue
        if token == "-type":
            if index + 1 >= len(args) or args[index + 1] not in {"f", "d"}:
                return "error: find -type allows only f or d"
            index += 2
            continue
        if token == "-name":
            if index + 1 >= len(args) or _unsafe_pattern(args[index + 1]):
                return "error: unsafe find -name pattern"
            index += 2
            continue
        if token == "-print":
            index += 1
            continue
        if token.startswith("-"):
            return f"error: find option not allowed: {token}"
        path_error = _validate_relative_path_token(token)
        if path_error:
            return path_error
        index += 1
    return None


def _validate_head_tail(args: list[str]) -> str | None:
    index = 0
    while index < len(args):
        token = args[index]
        if token == "-n":
            if index + 1 >= len(args) or _int_suffix(args[index + 1], maximum=200) is None:
                return "error: -n must be between 0 and 200"
            index += 2
            continue
        if token.startswith("-n") and len(token) > 2:
            if _int_suffix(token[2:], maximum=200) is None:
                return "error: -n must be between 0 and 200"
            index += 1
            continue
        if token.startswith("-"):
            return f"error: flag not allowed: {token}"
        path_error = _validate_relative_path_token(token)
        if path_error:
            return path_error
        index += 1
    return None


def _validate_relative_path_token(token: str) -> str | None:
    if not token or token.startswith("~"):
        return f"error: unsafe path token: {token}"
    path = Path(token)
    if path.is_absolute() or ".." in path.parts:
        return f"error: path escapes workdir: {token}"
    return None


def _unsafe_pattern(pattern: str) -> bool:
    return not pattern or "/" in pattern or "\\" in pattern or ".." in pattern


def _int_suffix(value: str, *, maximum: int) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 0 or parsed > maximum:
        return None
    return parsed


def _bounded_int(value: Any, *, default: int, maximum: int) -> int:
    if not isinstance(value, int):
        return default
    return max(1, min(value, maximum))
