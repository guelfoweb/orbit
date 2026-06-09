from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_SHELL_TIMEOUT = 10
MAX_SHELL_TIMEOUT = 15
DEFAULT_SHELL_OUTPUT_BYTES = 12_000
MAX_SHELL_OUTPUT_BYTES = 12_000
MAX_CHAINED_COMMANDS = 4

_SHELL_META_CHARS = frozenset("|;<>`$\\\n\r")
_FILE_COMMANDS = frozenset({"pwd", "ls", "find", "du", "df", "wc", "head", "tail", "file", "stat", "cat"})
_SYSTEM_INFO_COMMANDS = frozenset({"uname", "hostname", "uptime", "whoami", "id", "date"})
_HARDWARE_INFO_COMMANDS = frozenset({"free", "lscpu", "lsblk"})
_PROCESS_INFO_COMMANDS = frozenset({"ps", "pgrep"})
_NETWORK_INFO_COMMANDS = frozenset({"ip", "ss"})
_ALLOWED_COMMANDS = (
    _FILE_COMMANDS | _SYSTEM_INFO_COMMANDS | _HARDWARE_INFO_COMMANDS | _PROCESS_INFO_COMMANDS | _NETWORK_INFO_COMMANDS
)
_ALLOWED_SIMPLE_FLAGS = {
    "ls": frozenset({"-1", "-a", "-l", "-h", "-F", "-la", "-al", "-lh", "-hl", "-lah", "-lha", "-alh", "-ahl", "-hal", "-hla"}),
    "du": frozenset({"-s", "-h", "-sh", "-hs"}),
    "df": frozenset({"-h", "-k", "-m"}),
    "free": frozenset({"-h", "-b", "-k", "-m", "-g"}),
    "lscpu": frozenset(),
    "lsblk": frozenset({"-f", "-J", "-p"}),
    "uname": frozenset({"-a", "-s", "-r", "-m", "-n", "-p", "-o"}),
    "hostname": frozenset(),
    "uptime": frozenset({"-p", "-s"}),
    "whoami": frozenset(),
    "id": frozenset({"-u", "-g"}),
    "date": frozenset({"-I", "-R", "-u"}),
    "ss": frozenset({"-t", "-u", "-l", "-n", "-p", "-a", "-e", "-r", "-tulpen", "-tulpn", "-tunlp", "-ltnp", "-lntu"}),
    "wc": frozenset({"-l", "-w", "-c", "-m"}),
    "file": frozenset({"-b"}),
    "stat": frozenset({"-c"}),
    "cat": frozenset(),
}
_NO_PATH_ARGUMENT_COMMANDS = frozenset(
    {
        "free",
        "lscpu",
        "lsblk",
        "uname",
        "hostname",
        "uptime",
        "whoami",
        "id",
        "date",
        "ss",
    }
)


def exec_shell_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "exec_shell_command",
            "description": (
                "Run one bounded read-only command in workdir. "
                "Allowed: pwd, ls, find, du, df, free, lscpu, lsblk, uname, hostname, uptime, "
                "whoami, id, date, ps, pgrep, ip, ss, wc, head, tail, file, stat, cat. "
                "Use ls -F for listing. Only && may chain allowed commands. No ls -R or outside paths."
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
    command_tokens, error = _validated_command_tokens(raw_command)
    if error:
        return error
    safe_command = _join_command_tokens(command_tokens)
    timeout = _bounded_int(arguments.get("timeout"), default=DEFAULT_SHELL_TIMEOUT, maximum=MAX_SHELL_TIMEOUT)
    output_size = _bounded_int(arguments.get("max_output_size"), default=DEFAULT_SHELL_OUTPUT_BYTES, maximum=MAX_SHELL_OUTPUT_BYTES)
    return {
        "command": f"cd {shlex.quote(str(workdir.resolve()))} && {safe_command}",
        "timeout": timeout,
        "max_output_size": output_size,
    }


def execute_exec_shell_command(arguments: dict[str, Any], *, workdir: Path) -> str:
    raw_command = arguments.get("command")
    if not isinstance(raw_command, str) or not raw_command.strip():
        return "error: exec_shell_command requires a non-empty command string"
    command_tokens, error = _validated_command_tokens(raw_command)
    if error:
        return error
    timeout = _bounded_int(arguments.get("timeout"), default=DEFAULT_SHELL_TIMEOUT, maximum=MAX_SHELL_TIMEOUT)
    output_size = _bounded_int(arguments.get("max_output_size"), default=DEFAULT_SHELL_OUTPUT_BYTES, maximum=MAX_SHELL_OUTPUT_BYTES)
    output_parts: list[str] = []
    for tokens in command_tokens:
        try:
            completed = subprocess.run(
                tokens,
                cwd=workdir,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"error: exec_shell_command failed: {exc}"
        if completed.stdout:
            output_parts.append(completed.stdout.rstrip())
        if completed.stderr:
            output_parts.append(completed.stderr.rstrip())
        if completed.returncode != 0:
            output_parts.append(f"error: command exited with status {completed.returncode}: {shlex.join(tokens)}")
            break
    content = "\n".join(part for part in output_parts if part)
    if not content:
        return ""
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) <= output_size:
        return content
    return encoded[:output_size].decode("utf-8", errors="replace") + "\n[truncated]"


def exec_shell_should_run_locally(arguments: dict[str, Any]) -> bool:
    raw_command = arguments.get("command")
    return isinstance(raw_command, str) and "&&" in raw_command


def _validated_command_tokens(raw_command: str) -> tuple[list[list[str]], str | None]:
    if any(char in raw_command for char in _SHELL_META_CHARS):
        return [], "error: shell operators, redirects, variables, escapes, and multi-line commands are not allowed"
    if "&" in raw_command.replace("&&", ""):
        return [], "error: only && is allowed for chaining shell commands"
    parts = [part.strip() for part in raw_command.split("&&")]
    if any(not part for part in parts):
        return [], "error: empty command in shell command chain"
    if len(parts) > MAX_CHAINED_COMMANDS:
        return [], f"error: too many chained shell commands: max {MAX_CHAINED_COMMANDS}"
    prepared: list[list[str]] = []
    for part in parts:
        try:
            tokens = shlex.split(part)
        except ValueError as exc:
            return [], f"error: invalid shell command syntax: {exc}"
        if not tokens:
            return [], "error: exec_shell_command requires a command"
        command = tokens[0]
        if command not in _ALLOWED_COMMANDS:
            return [], f"error: command not allowed: {command}"
        validation_error = _validate_command_tokens(command, tokens[1:])
        if validation_error:
            return [], validation_error
        prepared.append(tokens)
    return prepared, None


def _join_command_tokens(command_tokens: list[list[str]]) -> str:
    return " && ".join(shlex.join(tokens) for tokens in command_tokens)


def _validate_command_tokens(command: str, args: list[str]) -> str | None:
    if command == "pwd":
        return None if not args else "error: pwd does not accept arguments here"
    if command == "find":
        return _validate_find(args)
    if command in {"head", "tail"}:
        return _validate_head_tail(args)
    if command == "ip":
        return _validate_ip(args)
    if command == "ps":
        return _validate_ps(args)
    if command == "pgrep":
        return _validate_pgrep(args)
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
        if command in _NO_PATH_ARGUMENT_COMMANDS:
            return f"error: arguments not allowed for {command}: {token}"
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


def _validate_ip(args: list[str]) -> str | None:
    allowed_forms = {
        ("addr",),
        ("addr", "show"),
        ("address",),
        ("address", "show"),
        ("route",),
        ("route", "show"),
        ("-brief", "addr"),
        ("-brief", "addr", "show"),
        ("-brief", "address"),
        ("-brief", "address", "show"),
        ("-br", "addr"),
        ("-br", "addr", "show"),
        ("-br", "address"),
        ("-br", "address", "show"),
    }
    if tuple(args) in allowed_forms:
        return None
    return "error: ip allows only local addr/address/route diagnostics"


def _validate_ps(args: list[str]) -> str | None:
    if not args or args in (["-e"], ["-ef"], ["aux"]):
        return None
    return "error: ps allows only no arguments, -e, -ef, or aux"


def _validate_pgrep(args: list[str]) -> str | None:
    if not args:
        return "error: pgrep requires one pattern"
    allowed_flags = frozenset({"-a", "-f", "-l"})
    pattern_count = 0
    for token in args:
        if token.startswith("-"):
            if token not in allowed_flags:
                return f"error: flag not allowed for pgrep: {token}"
            continue
        if _unsafe_process_pattern(token):
            return "error: unsafe pgrep pattern"
        pattern_count += 1
    if pattern_count != 1:
        return "error: pgrep requires exactly one pattern"
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


def _unsafe_process_pattern(pattern: str) -> bool:
    return len(pattern) > 80 or _unsafe_pattern(pattern)


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
