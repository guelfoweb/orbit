from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from .common import ToolError, coerce_int, resolve_path


MAX_BASH_OUTPUT = 12_000
DEFAULT_BASH_TIMEOUT = 30

BLOCKED_COMMANDS = {
    "sudo",
    "su",
    "dd",
    "mkfs",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "kill",
    "killall",
    "pkill",
    "chown",
}
BLOCKED_SHELL_EXECUTORS = {"sh", "bash", "zsh", "fish", "ksh", "csh", "tcsh"}
BLOCKED_TOKENS = {"||", "&", "&&", ";", "|&"}
PIPE_FILTER_COMMANDS = {"head", "tail", "grep", "sed", "cut", "sort", "uniq", "wc", "tr", "base64"}
REDIRECT_TOKEN_RE = re.compile(r"(^\d*(>>?|<<))|(^\d*[<>]&)|([<>]{1,2})")
SIGPIPE_RETURN_CODE = -13


def shell_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run one bounded shell-like command in the workdir. Safe pipelines only; no redirects, chaining, or destructive commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
                    },
                    "required": ["command"],
                },
            },
        }
    ]


@dataclass
class ShellTools:
    workdir: Path

    def bash(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command is required")
        timeout = min(120, max(1, coerce_int(arguments.get("timeout"), DEFAULT_BASH_TIMEOUT)))
        command = self._normalize_workspace_df_command(command)
        segments = self._parse_bash_segments(command)
        completed = self._run_bash_segments(segments, timeout=timeout)
        stdout = completed.stdout[:MAX_BASH_OUTPUT]
        stderr = completed.stderr[:MAX_BASH_OUTPUT]
        ok = completed.returncode == 0 or self._is_acceptable_pipe_termination(completed)
        return {
            "ok": ok,
            "command": command,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": len(completed.stdout) > MAX_BASH_OUTPUT or len(completed.stderr) > MAX_BASH_OUTPUT,
        }

    def _parse_bash_segments(self, command: str) -> list[list[str]]:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"invalid shell syntax: {exc}") from exc
        if not tokens:
            raise ToolError("command is empty")
        self._validate_bash_tokens(tokens)
        segments: list[list[str]] = [[]]
        for token in tokens:
            if token == "|":
                if not segments[-1]:
                    raise ToolError("invalid pipe syntax")
                if len(segments) >= 3:
                    raise ToolError("too many pipe segments")
                segments.append([])
                continue
            segments[-1].append(token)
        if not segments[-1]:
            raise ToolError("invalid pipe syntax")
        for index, parts in enumerate(segments):
            self._validate_bash_segment(parts, segment_index=index)
        return segments

    @staticmethod
    def _normalize_workspace_df_command(command: str) -> str:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return command
        if len(tokens) < 2 or tokens[0] != "df":
            return command
        if any(token == "|" for token in tokens):
            return command
        if any(not token.startswith("-") for token in tokens[1:]):
            return command
        return f"{command} ."

    @staticmethod
    def _validate_bash_tokens(tokens: list[str]) -> None:
        for token in tokens:
            if token in BLOCKED_TOKENS or "$(" in token or "`" in token:
                raise ToolError("shell operators are not allowed")
            if token != "|" and REDIRECT_TOKEN_RE.search(token):
                raise ToolError("shell redirection is not allowed")

    def _validate_bash_segment(self, parts: list[str], *, segment_index: int) -> None:
        if not parts:
            raise ToolError("command is empty")
        command = parts[0]
        command_name = Path(command).name
        if command in BLOCKED_COMMANDS or command_name in BLOCKED_COMMANDS:
            raise ToolError(f"blocked command: {command}")
        if command_name in BLOCKED_SHELL_EXECUTORS:
            raise ToolError(f"blocked shell executor: {command}")
        if command_name in {"env", "command", "builtin"} and len(parts) > 1:
            next_name = Path(parts[1]).name
            if next_name in BLOCKED_SHELL_EXECUTORS:
                raise ToolError(f"blocked shell executor: {parts[1]}")
        if command_name == "busybox" and len(parts) > 1 and Path(parts[1]).name in BLOCKED_SHELL_EXECUTORS:
            raise ToolError(f"blocked shell executor: {parts[1]}")
        if parts[:2] == ["git", "reset"] or parts[:2] == ["git", "checkout"]:
            raise ToolError("blocked git command")
        if command == "rm":
            self._validate_rm_targets(parts)
        if segment_index > 0 and command not in PIPE_FILTER_COMMANDS:
            raise ToolError(f"unsupported pipe filter: {command}")

    def _validate_rm_targets(self, parts: list[str]) -> None:
        targets = [part for part in parts[1:] if not part.startswith("-")]
        if not targets:
            raise ToolError("rm requires at least one target")
        for target in targets:
            if not target.strip():
                raise ToolError("rm target is empty")
            resolve_path(self.workdir, target)

    def _run_bash_segments(self, segments: list[list[str]], *, timeout: int) -> subprocess.CompletedProcess[str]:
        if len(segments) == 1:
            return subprocess.run(
                segments[0],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                check=False,
            )
        processes: list[subprocess.Popen[str]] = []
        previous_stdout = None
        try:
            for parts in segments:
                process = subprocess.Popen(
                    parts,
                    cwd=self.workdir,
                    stdin=previous_stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=False,
                )
                if previous_stdout is not None:
                    previous_stdout.close()
                previous_stdout = process.stdout
                processes.append(process)
            stdout, stderr = processes[-1].communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            for process in processes:
                process.kill()
            raise exc
        finally:
            for process in processes[:-1]:
                if process.stdout is not None:
                    process.stdout.close()
        upstream_stderr: list[str] = []
        for process in processes[:-1]:
            process.wait(timeout=1)
            if process.stderr is not None:
                upstream_stderr.append(process.stderr.read())
                process.stderr.close()
        returncode = 0
        for process in processes:
            if process.returncode not in (0, None):
                returncode = process.returncode
                break
        if returncode == 0:
            returncode = processes[-1].returncode or 0
        return subprocess.CompletedProcess(
            args=" | ".join(" ".join(parts) for parts in segments),
            returncode=returncode,
            stdout=stdout,
            stderr="".join(filter(None, upstream_stderr + [stderr])),
        )

    @staticmethod
    def _is_acceptable_pipe_termination(completed: subprocess.CompletedProcess[str]) -> bool:
        if completed.returncode != SIGPIPE_RETURN_CODE:
            return False
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        return bool(stdout.strip()) and not stderr.strip()
