from __future__ import annotations

import os
import re
import signal
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit.runtime.file_tools import MAX_CHUNK_FILE_BYTES, MAX_READ_CHARS, read_file
from orbit.runtime.path_guardrails import resolve_inside_workdir
from orbit.runtime.web import html_to_text, search_web


DEFAULT_SHELL_TIMEOUT = 10
MAX_SHELL_TIMEOUT = 15
DEFAULT_SHELL_OUTPUT_BYTES = 12_000
MAX_SHELL_OUTPUT_BYTES = 12_000
SEARCH_SHELL_OUTPUT_BYTES = 800
SHELL_FAILURE_STREAM_CHARS = 1200
PDF_CHUNK_CHARS = 3_000
SHELL_READ_FILE_THRESHOLD_BYTES = 8 * 1024
SHELL_FULL_CONTRACT_ERROR_PREFIX = "error: shell-full analysis requests require content/source/string evidence"
SHELL_FULL_CONTRACT_RETRY_PROMPT = (
    "The previous shell-full command was rejected because it only listed metadata. "
    "Use the available exec_shell_full_command tool now to inspect source/content/string evidence. "
    "Return only the tool call."
)
SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT = (
    "Your previous command inspected only metadata or listings.\n\n"
    "For this coding task, inspect real file or test content before continuing.\n\n"
    "Use commands such as cat, sed -n, grep/rg on file contents, or test output.\n\n"
    "If file names are unknown, use grep/rg recursively over file contents.\n\n"
    "Do not use ls, find, tree, file, or stat.\n\n"
    "Return only JSON:\n\n"
    '{"command":"..."}'
)
SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT = (
    "The command succeeded but produced no output.\n\n"
    "Verify that the requested change actually occurred.\n\n"
    "The verification command must print direct evidence of the requested value or state, "
    "not only metadata, paths, tags, field names, or key names.\n\n"
    "Return only JSON:\n\n"
    '{"command":"..."}'
)
SHELL_FULL_COMPLETION_GUARD_PROMPT = (
    "You identified the target but did not perform the requested modification.\n\n"
    "Continue the task.\n\n"
    "Prefer short robust commands; avoid fragile quoting and long heredocs. "
    "Prefer minimal edits over rewriting entire files.\n\n"
    "Return only JSON:\n\n"
    '{"command":"..."}'
)
SHELL_FULL_MINIMAL_PATCH_GUARD_PROMPT = (
    "Your previous command tried to rewrite too much and was too long or incomplete.\n\n"
    "Use a minimal local patch to modify only the necessary lines.\n\n"
    "Do not use heredocs, cat > file, tee, or full-file rewrites for existing files.\n\n"
    "Use a short command that changes only the needed lines.\n\n"
    "Return only JSON:\n\n"
    '{"command":"..."}'
)
SHELL_FULL_SEMANTIC_REPAIR_PROMPT = (
    "The previous modification was applied, and the verification output is in context.\n\n"
    "Check whether all requested changes are satisfied.\n\n"
    "If any requested change is missing, return a minimal follow-up command to complete it.\n\n"
    "If all requested changes are satisfied, return only: OK\n\n"
    "Return command JSON only when a follow-up command is needed:\n\n"
    '{"command":"..."}'
)


@dataclass(frozen=True)
class ShellFailure:
    exit_code: int
    stdout: str
    stderr: str

_ANALYSIS_PROMPT_RE = re.compile(
    r"\b(analy[sz]e|analysis|review|inspect|vulnerab|exploit|malware|dropper|c2|ioc|reverse|decompil|static)\b",
    re.IGNORECASE,
)
_METADATA_ONLY_RE = re.compile(r"^\s*(?:ls|file|stat)(?:\s|$)", re.IGNORECASE)
_CONTENT_EVIDENCE_RE = re.compile(
    r"\b(?:cat|sed|head|tail|grep|rg|strings|pdftotext|python3?|node|jq|awk|xxd|hexdump|readelf|objdump|jadx|apktool|orbit-web-search)\b",
    re.IGNORECASE,
)
_HTML_SOURCE_PROMPT_RE = re.compile(
    r"\b(?:html\s+source|source\s+html|page\s+source|source\s+code|sorgente|codice\s+html|html\s+code)\b",
    re.IGNORECASE,
)
_READ_ONLY_PROMPT_RE = re.compile(
    r"^\s*(?:read|show|list|tell|display|print|count|search|find|grep|summari[sz]e|explain|describe)\b",
    re.IGNORECASE,
)
_MUTATION_PROMPT_RE = re.compile(
    r"\b(?:add|change|create|fix|harden|improve|write|edit|modify|replace|append|delete|remove|rename|refactor|move|copy|install|commit|update|insert|drop|alter|set|enable|disable|configure)\b",
    re.IGNORECASE,
)
_NEGATED_MUTATION_PROMPT_RE = re.compile(
    r"\b(?:do\s+not|don't|without)\s+(?:add|change|create|fix|harden|improve|write|edit|modify|replace|append|delete|remove|rename|refactor|move|copy|install|commit|update|insert|drop|alter|set|enable|disable|configure)\b",
    re.IGNORECASE,
)


def exec_shell_full_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "exec_shell_full_command",
            "description": (
                "Unrestricted local shell launched from the current workdir. May read, write, delete, execute, access network, and access paths outside workdir. "
                "Use whatever commands are needed to complete the task. "
                "For analysis, prefer direct evidence from content, source, binaries, strings, logs, archives, and fetched data, not only metadata. "
                'For generic web search, use orbit-web-search "query"; for explicit URLs, use curl. '
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


def execute_exec_shell_full_command(arguments: dict[str, Any], *, workdir: Path, user_prompt: str | None = None) -> str:
    raw_command = arguments.get("command")
    if not isinstance(raw_command, str) or not raw_command.strip():
        return "error: exec_shell_full_command requires a non-empty command string"
    timeout = _bounded_int(arguments.get("timeout"), default=DEFAULT_SHELL_TIMEOUT, maximum=MAX_SHELL_TIMEOUT)
    output_size = _bounded_int(arguments.get("max_output_size"), default=DEFAULT_SHELL_OUTPUT_BYTES, maximum=MAX_SHELL_OUTPUT_BYTES)
    resolved_workdir = workdir.expanduser().resolve()
    web_search_result = _run_orbit_web_search(raw_command)
    if web_search_result is not None:
        return _bounded_text(web_search_result, output_size)
    pdf_result = _read_pdf_target(raw_command, workdir=workdir)
    if pdf_result is not None:
        return _bounded_text(pdf_result, output_size)
    env = dict(os.environ)
    env["HOME"] = str(resolved_workdir)
    env["PWD"] = str(resolved_workdir)
    try:
        completed = _run_shell_command(raw_command, cwd=resolved_workdir, env=env, timeout=timeout)
    except OSError as exc:
        return f"error: exec_shell_full_command failed: {exc}"
    except subprocess.TimeoutExpired as exc:
        return f"error: exec_shell_full_command timed out after {timeout}s"
    if completed.returncode != 0:
        return _format_shell_failure(completed.returncode, completed.stdout, completed.stderr)
    output_parts = []
    if completed.stdout:
        output_parts.append(completed.stdout.rstrip())
    if completed.stderr:
        output_parts.append(completed.stderr.rstrip())
    content = "\n".join(part for part in output_parts if part)
    if not content:
        return ""
    processed = _postprocess_shell_full_output(raw_command, content, workdir=workdir, output_size=output_size, user_prompt=user_prompt)
    if processed is not None:
        return processed
    if _is_search_command(raw_command):
        return _bounded_text(content, min(output_size, SEARCH_SHELL_OUTPUT_BYTES))
    return _bounded_text(content, output_size)


def _run_shell_command(raw_command: str, *, cwd: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        raw_command,
        cwd=cwd,
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        stdout, stderr = process.communicate()
        exc.stdout = stdout
        exc.stderr = stderr
        raise exc
    return subprocess.CompletedProcess(raw_command, process.returncode, stdout, stderr)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        return
    process.kill()


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


def shell_failure_from_output(content: str) -> ShellFailure | None:
    lines = content.splitlines()
    if not lines or lines[0] != "shell_command_failed: true":
        return None
    exit_code: int | None = None
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    section: str | None = None
    for line in lines[1:]:
        if line.startswith("exit_code: "):
            try:
                exit_code = int(line.removeprefix("exit_code: ").strip())
            except ValueError:
                return None
            continue
        if line == "STDOUT:":
            section = "stdout"
            continue
        if line == "STDERR:":
            section = "stderr"
            continue
        if section == "stdout":
            stdout_lines.append(line)
        elif section == "stderr":
            stderr_lines.append(line)
    if exit_code is None:
        return None
    return ShellFailure(exit_code=exit_code, stdout="\n".join(stdout_lines).strip(), stderr="\n".join(stderr_lines).strip())


def is_shell_full_execution_error(content: str) -> bool:
    return shell_failure_from_output(content) is not None


def is_repairable_shell_error(content: str) -> bool:
    failure = shell_failure_from_output(content)
    if failure is None:
        return False
    combined = f"{failure.stdout}\n{failure.stderr}".lower()
    non_repairable = (
        "permission denied",
        "operation not permitted",
        "read-only file system",
        "no space left on device",
        "resource temporarily unavailable",
        "killed",
        "out of memory",
        "network is unreachable",
        "temporary failure in name resolution",
        "could not resolve host",
        "connection timed out",
    )
    if any(marker in combined for marker in non_repairable):
        return False
    return True


def should_verify_shell_mutation(command: str, *, user_prompt: str | None) -> bool:
    if user_prompt and _READ_ONLY_PROMPT_RE.search(user_prompt) and not _MUTATION_PROMPT_RE.search(user_prompt):
        return False
    return is_mutating_shell_command(command)


def is_mutative_user_request(user_prompt: str | None) -> bool:
    if not user_prompt:
        return False
    if _NEGATED_MUTATION_PROMPT_RE.search(user_prompt) and not re.search(
        r"\b(?:fix|update|change|create|write|rename|refactor|edit)\b.*\b(?:file|code|implementation|config|test)\b",
        user_prompt,
        re.IGNORECASE,
    ):
        return False
    if _READ_ONLY_PROMPT_RE.search(user_prompt) and not _MUTATION_PROMPT_RE.search(user_prompt):
        return False
    return _MUTATION_PROMPT_RE.search(user_prompt) is not None


def is_mutating_shell_command(command: str) -> bool:
    return _is_mutating_shell_command(command)


def is_metadata_only_shell_command(command: str | None) -> bool:
    if not command:
        return False
    return _METADATA_ONLY_RE.search(command) is not None


def is_content_evidence_shell_command(command: str | None) -> bool:
    if not command:
        return False
    return _CONTENT_EVIDENCE_RE.search(command) is not None


def looks_like_broad_file_rewrite(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if re.search(r"\bcat\s+<<\s*['\"]?\w+['\"]?\s*>\s*[^\s]+", text):
        return True
    if re.search(r"\bcat\s*>\s*[^\s]+\s*<<\s*['\"]?\w+['\"]?", text):
        return True
    if re.search(r"\b(?:tee|dd)\b.*\b(?:of=|>\s*)", text):
        return True
    if re.search(r"\bwrite_(?:text|bytes)\s*\(", text):
        return True
    if re.search(r"\bopen\s*\([^)]*,\s*['\"]w", text):
        return True
    if "heredoc" in lowered:
        return True
    return False


def is_incomplete_shell_json_or_command_error(content: str) -> bool:
    lowered = content.lower()
    return (
        "invalid json tool arguments" in lowered
        or "unterminated string" in lowered
        or "missing closing quote" in lowered
        or "unexpected eof" in lowered
    )


def shell_repair_prompt(content: str) -> str:
    failure = shell_failure_from_output(content)
    if failure is None:
        failure = ShellFailure(exit_code=1, stdout="", stderr=_bounded_stream(content))
    return "\n".join(
        [
            "The previous shell command failed.",
            "",
            f"Exit code: {failure.exit_code}",
            "",
            "STDOUT:",
            _bounded_stream(failure.stdout) or "(empty)",
            "",
            "STDERR:",
            _bounded_stream(failure.stderr) or "(empty)",
            "",
            "Prefer short robust commands; avoid fragile quoting and long heredocs. Prefer minimal edits over rewriting entire files.",
            "",
            "Return only corrected JSON:",
            "",
            '{"command":"..."}',
        ]
    )


def _run_orbit_web_search(raw_command: str) -> str | None:
    try:
        tokens = shlex.split(raw_command)
    except ValueError as exc:
        if raw_command.strip().startswith("orbit-web-search"):
            return f"error: invalid orbit-web-search command: {exc}"
        return None
    if not tokens or tokens[0] != "orbit-web-search":
        return None
    query = " ".join(tokens[1:]).strip()
    if not query:
        return "error: orbit-web-search requires a query"
    return search_web(query)


def _postprocess_shell_full_output(
    raw_command: str,
    content: str,
    *,
    workdir: Path,
    output_size: int,
    user_prompt: str | None = None,
) -> str | None:
    cat_result = _read_large_cat_target(raw_command, workdir=workdir)
    if cat_result is not None:
        return cat_result
    if _looks_like_html_or_fragment(content) and not _wants_html_source(user_prompt):
        text = html_to_text(content)
        if not text:
            text = _strip_html_tags(content)
        if not text:
            return "shell_output_html_cleaned: true\ntext:\n[no readable text extracted]"
        return _bounded_text("\n".join(["shell_output_html_cleaned: true", "text:", text]), min(output_size, 4_000))
    return None


def _read_pdf_target(raw_command: str, *, workdir: Path) -> str | None:
    try:
        tokens = shlex.split(raw_command)
    except ValueError:
        return None
    if not tokens or not _command_intends_pdf_text(tokens):
        return None
    target = _first_pdf_target(tokens, workdir=workdir)
    if target is None:
        return None
    text, method = _extract_pdf_text(target)
    if not text.strip():
        return f"error: no text extracted from PDF: {target.name}"
    return _format_extracted_pdf(target, text, method=method)


def _command_intends_pdf_text(tokens: list[str]) -> bool:
    command = Path(tokens[0]).name
    if command in {"cat", "grep", "head", "pdftotext", "rg", "sed", "strings", "tail"}:
        return True
    return any(Path(token).name in {"cat", "grep", "head", "pdftotext", "rg", "sed", "strings", "tail"} for token in tokens)


def _first_pdf_target(tokens: list[str], *, workdir: Path) -> Path | None:
    for token in tokens:
        cleaned = token.strip("'\"")
        if not cleaned.lower().endswith(".pdf"):
            continue
        target_or_error = resolve_inside_workdir(cleaned, workdir=workdir)
        if isinstance(target_or_error, str) or not target_or_error.is_file():
            continue
        return target_or_error
    return None


def _extract_pdf_text(target: Path) -> tuple[str, str]:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        completed = subprocess.run(
            [pdftotext, "-layout", str(target), "-"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=MAX_SHELL_TIMEOUT,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout, "pdftotext"
    strings = shutil.which("strings")
    if not strings:
        return "", "unavailable"
    completed = subprocess.run(
        [strings, "-a", "-n", "8", str(target)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=MAX_SHELL_TIMEOUT,
        check=False,
    )
    if completed.returncode == 0:
        return _clean_pdf_strings_output(completed.stdout), "strings"
    return "", "strings"


def _format_extracted_pdf(target: Path, text: str, *, method: str) -> str:
    if len(text.encode("utf-8", errors="replace")) > MAX_CHUNK_FILE_BYTES:
        return f"error: extracted PDF text too large: max {MAX_CHUNK_FILE_BYTES} bytes"
    if len(text) <= MAX_READ_CHARS:
        return "\n".join(
            [
                "shell_output_pdf_text: true",
                f"path: {target.name}",
                f"extractor: {method}",
                "content:",
                text.strip() or "(empty PDF text)",
            ]
        )
    end = min(PDF_CHUNK_CHARS, len(text))
    total_chunks = max(1, (len(text) + PDF_CHUNK_CHARS - 1) // PDF_CHUNK_CHARS)
    return "\n".join(
        [
            "shell_output_pdf_text: true",
            f"path: {target.name}",
            f"extractor: {method}",
            "chunk_index: 0",
            f"total_chunks: {total_chunks}",
            f"chars: 0-{end} of {len(text)}",
            "content:",
            text[:end],
        ]
    )


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


def _is_mutating_shell_command(command: str) -> bool:
    if _has_shell_write_operator(command):
        return True
    try:
        tokens = shlex.split(command)
    except ValueError:
        return any(marker in command for marker in (">", ">>", "| tee", " -i", "--in-place"))
    if not tokens:
        return False
    words = [Path(token).name for token in tokens if token not in {"sudo", "env", "command", "xargs"}]
    if not words:
        return False
    primary = words[0]
    if primary in {"cp", "install", "ln", "mkdir", "mv", "rm", "rmdir", "tee", "touch", "truncate"}:
        return True
    if primary in {"chmod", "chgrp", "chown", "setfacl"}:
        return True
    if primary == "sed" and any(token == "-i" or token.startswith("-i") or token == "--in-place" for token in tokens[1:]):
        return True
    if primary == "perl" and any("i" in token and token.startswith("-") for token in tokens[1:]):
        return True
    if primary == "git" and len(words) > 1 and words[1] in {"add", "am", "apply", "checkout", "clean", "commit", "merge", "mv", "rebase", "reset", "restore", "rm", "stash", "switch"}:
        return True
    if primary == "sqlite3" and re.search(r"\b(?:alter|create|delete|drop|insert|replace|update)\b", command, re.IGNORECASE):
        return True
    if primary in {"bash", "dash", "fish", "sh", "zsh", "python", "python3", "node", "ruby"}:
        return _has_shell_write_operator(command) or re.search(r"\b(?:open|write_text|write_bytes|remove|rename|unlink|mkdir)\b", command) is not None
    if len(words) > 1 and words[1] in {"install", "update", "add", "remove"} and primary in {"apt", "apt-get", "brew", "cargo", "npm", "pip", "pip3", "uv"}:
        return True
    return False


def _has_shell_write_operator(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ">" in command
    return any(token in {">", ">>", "2>", "2>>", "&>", "&>>"} for token in tokens) or bool(re.search(r"(^|[^<>])>{1,2}[^&]", command))


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


def _wants_html_source(user_prompt: str | None) -> bool:
    return bool(user_prompt and _HTML_SOURCE_PROMPT_RE.search(user_prompt))


def _bounded_text(content: str, output_size: int) -> str:
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) <= output_size:
        return content
    return encoded[:output_size].decode("utf-8", errors="replace") + "\n[truncated]"


def _bounded_stream(content: str) -> str:
    return _bounded_text(content.strip(), SHELL_FAILURE_STREAM_CHARS)


def _format_shell_failure(exit_code: int, stdout: str, stderr: str) -> str:
    return "\n".join(
        [
            "shell_command_failed: true",
            f"exit_code: {exit_code}",
            "STDOUT:",
            _bounded_stream(stdout) or "(empty)",
            "STDERR:",
            _bounded_stream(stderr) or "(empty)",
        ]
    )


def _strip_html_tags(content: str) -> str:
    without_scripts = re.sub(r"<(?:script|style|noscript)\b.*?(?:</(?:script|style|noscript)>|$)", " ", content, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", text).strip()


def _clean_pdf_strings_output(content: str) -> str:
    lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or len(line) < 4:
            continue
        lowered = line.lower()
        if lowered in {"xref", "trailer", "stream", "endstream", "startxref", "endobj"}:
            continue
        if line.startswith(("%PDF", "%%EOF", "<<", ">>", "/")):
            continue
        if re.fullmatch(r"\d+(?:\s+\d+)?\s+(?:obj|r)?", lowered):
            continue
        if not re.search(r"[A-Za-zÀ-ÿ]", line):
            continue
        letters = len(re.findall(r"[A-Za-zÀ-ÿ]", line))
        if letters / max(len(line), 1) < 0.45:
            continue
        words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", line)
        if len(words) < 2 or " " not in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def _bounded_int(value: Any, *, default: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(1, min(value, maximum))
    return default
