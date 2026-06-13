from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from orbit.runtime.file_tools import read_file
from orbit.runtime.path_guardrails import resolve_inside_workdir
from orbit.runtime.web import html_to_text


DEFAULT_SHELL_TIMEOUT = 10
MAX_SHELL_TIMEOUT = 15
DEFAULT_SHELL_OUTPUT_BYTES = 12_000
MAX_SHELL_OUTPUT_BYTES = 12_000
SEARCH_SHELL_OUTPUT_BYTES = 800
SHELL_READ_FILE_THRESHOLD_BYTES = 8 * 1024
SHELL_FULL_CONTRACT_ERROR_PREFIX = "error: shell-full analysis requests require content/source/string evidence"
SHELL_FULL_CONTRACT_RETRY_PROMPT = (
    "The previous shell-full command was rejected because it only listed metadata. "
    "Use the available exec_shell_full_command tool now to inspect source/content/string evidence. "
    "Return only the tool call."
)

_ANALYSIS_PROMPT_RE = re.compile(
    r"\b(analy[sz]e|analysis|review|inspect|vulnerab|exploit|malware|dropper|c2|ioc|reverse|decompil|static)\b",
    re.IGNORECASE,
)
_METADATA_ONLY_RE = re.compile(r"^\s*(?:ls|file|stat)(?:\s|$)", re.IGNORECASE)
_CONTENT_EVIDENCE_RE = re.compile(
    r"\b(?:cat|sed|head|tail|grep|rg|strings|python3?|node|jq|awk|xxd|hexdump|readelf|objdump|jadx|apktool)\b",
    re.IGNORECASE,
)


def exec_shell_full_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "exec_shell_full_command",
            "description": (
                "Local shell confined to the current workdir. May read, write, delete, execute, and access network. "
                "Use whatever commands are needed to complete the task. "
                "For analysis, prefer direct evidence from content, source, binaries, strings, logs, archives, and fetched data, not only metadata. "
                "For URLs, use curl when content is needed. "
                "Quote paths containing spaces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "max_output_size": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    }


def execute_exec_shell_full_command(arguments: dict[str, Any], *, workdir: Path) -> str:
    raw_command = arguments.get("command")
    if not isinstance(raw_command, str) or not raw_command.strip():
        return "error: exec_shell_full_command requires a non-empty command string"
    confinement_error = _validate_workdir_confined_command(raw_command)
    if confinement_error:
        return confinement_error
    timeout = _bounded_int(arguments.get("timeout"), default=DEFAULT_SHELL_TIMEOUT, maximum=MAX_SHELL_TIMEOUT)
    output_size = _bounded_int(arguments.get("max_output_size"), default=DEFAULT_SHELL_OUTPUT_BYTES, maximum=MAX_SHELL_OUTPUT_BYTES)
    resolved_workdir = workdir.expanduser().resolve()
    env = dict(os.environ)
    env["HOME"] = str(resolved_workdir)
    env["PWD"] = str(resolved_workdir)
    try:
        completed = subprocess.run(
            raw_command,
            cwd=resolved_workdir,
            env=env,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: exec_shell_full_command failed: {exc}"
    output_parts = []
    if completed.stdout:
        output_parts.append(completed.stdout.rstrip())
    if completed.stderr:
        output_parts.append(completed.stderr.rstrip())
    if completed.returncode != 0:
        output_parts.append(f"error: command exited with status {completed.returncode}")
    content = "\n".join(part for part in output_parts if part)
    if not content:
        return ""
    processed = _postprocess_shell_full_output(raw_command, content, workdir=workdir, output_size=output_size)
    if processed is not None:
        return processed
    if _is_search_command(raw_command):
        return _bounded_text(content, min(output_size, SEARCH_SHELL_OUTPUT_BYTES))
    return _bounded_text(content, output_size)


def validate_shell_full_contract(arguments: dict[str, Any], *, user_prompt: str | None) -> str | None:
    raw_command = arguments.get("command")
    if not isinstance(raw_command, str) or not user_prompt:
        return None
    if not _ANALYSIS_PROMPT_RE.search(user_prompt):
        return None
    if _CONTENT_EVIDENCE_RE.search(raw_command):
        return None
    if _METADATA_ONLY_RE.search(raw_command):
        return (
            f"{SHELL_FULL_CONTRACT_ERROR_PREFIX}, not only metadata/listing. "
            "Use a bounded command such as sed/head/grep/strings on the target file."
        )
    return None


def is_shell_full_contract_error(content: str) -> bool:
    return content.startswith(SHELL_FULL_CONTRACT_ERROR_PREFIX)


def _postprocess_shell_full_output(raw_command: str, content: str, *, workdir: Path, output_size: int) -> str | None:
    cat_result = _read_large_cat_target(raw_command, workdir=workdir)
    if cat_result is not None:
        return cat_result
    if _looks_like_html_or_fragment(content):
        text = html_to_text(content)
        if not text:
            text = _strip_html_tags(content)
        if not text:
            return "shell_output_html_cleaned: true\ntext:\n[no readable text extracted]"
        return _bounded_text("\n".join(["shell_output_html_cleaned: true", "text:", text]), min(output_size, 4_000))
    return None


def _read_large_cat_target(raw_command: str, *, workdir: Path) -> str | None:
    try:
        tokens = shlex.split(raw_command)
    except ValueError:
        return None
    if len(tokens) != 2 or tokens[0] != "cat":
        return None
    path = tokens[1]
    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return None
    target = target_or_error
    if not target.is_file():
        return None
    try:
        size = target.stat().st_size
    except OSError:
        return None
    if size <= SHELL_READ_FILE_THRESHOLD_BYTES:
        return None
    result = read_file(path, arguments={}, workdir=workdir)
    return "\n".join(
        [
            "shell_output_read_file: true",
            f"original_command: {raw_command}",
            f"threshold_bytes: {SHELL_READ_FILE_THRESHOLD_BYTES}",
            result,
        ]
    )


def _is_search_command(raw_command: str) -> bool:
    try:
        tokens = shlex.split(raw_command)
    except ValueError:
        return False
    return bool(tokens and tokens[0] in {"ag", "grep", "rg"})


def _looks_like_html(content: str) -> bool:
    prefix = content.lstrip()[:4096].lower()
    return (
        prefix.startswith("<!doctype html")
        or prefix.startswith("<html")
        or "<html" in prefix
        or "</html>" in prefix
        or ("<body" in prefix and "</body>" in prefix)
    )


def _looks_like_html_or_fragment(content: str) -> bool:
    if _looks_like_html(content):
        return True
    prefix = content.lstrip()[:4096].lower()
    return bool(
        re.search(
            r"</?(?:a|article|body|div|h[1-6]|html|li|main|meta|p|script|section|span|style|table|td|tr|ul)\b",
            prefix,
        )
        or "&lt;" in prefix
        or "&amp;" in prefix
    )


def _bounded_text(content: str, output_size: int) -> str:
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) <= output_size:
        return content
    return encoded[:output_size].decode("utf-8", errors="replace") + "\n[truncated]"


def _strip_html_tags(content: str) -> str:
    without_scripts = re.sub(r"<(?:script|style|noscript)\b.*?(?:</(?:script|style|noscript)>|$)", " ", content, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", text).strip()


def _bounded_int(value: Any, *, default: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(1, min(value, maximum))
    return default


def _validate_workdir_confined_command(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"error: invalid shell command: {exc}"
    if not tokens:
        return "error: exec_shell_full_command requires a non-empty command string"
    for token in tokens:
        if token in {"cd", "pushd", "popd"}:
            return "error: shell command must stay inside workdir; directory-changing commands are not allowed"
        if "$HOME" in token or "${HOME}" in token or token == "~" or token.startswith("~/"):
            return "error: shell command must stay inside workdir; home-directory paths are not allowed"
        if _token_escapes_workdir(token):
            return "error: shell command must stay inside workdir; absolute paths and parent traversal are not allowed"
    return None


def _token_escapes_workdir(token: str) -> bool:
    if token.startswith(("http://", "https://")):
        return False
    if token.startswith("/"):
        return True
    return token == ".." or token.startswith("../") or "/../" in token or token.endswith("/..")
