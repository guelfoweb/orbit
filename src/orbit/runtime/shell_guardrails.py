from __future__ import annotations

import os
import re
import signal
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit.runtime.file_tools import (
    PDF_CHUNK_CHARS,
    extract_pdf_text,
    format_pdf_result,
    read_file,
    read_pdf,
)
from orbit.runtime.path_guardrails import resolve_inside_workdir
from orbit.runtime.web import html_to_text, search_web


DEFAULT_SHELL_TIMEOUT = 10
MAX_SHELL_TIMEOUT = 15
DEFAULT_SHELL_OUTPUT_BYTES = 12_000
MAX_SHELL_OUTPUT_BYTES = 12_000
SEARCH_SHELL_OUTPUT_BYTES = 800
SHELL_FAILURE_STREAM_CHARS = 1200
SHELL_READ_FILE_THRESHOLD_BYTES = 8 * 1024
SHELL_FULL_CONTRACT_ERROR_PREFIX = "error: shell-full analysis requests require content/source/string evidence"
SHELL_FULL_READ_ONLY_MUTATION_ERROR_PREFIX = "error: read-only request rejected mutating shell command"
SHELL_FULL_CONTRACT_RETRY_PROMPT = (
    "The previous shell-full command was rejected because it only listed metadata. "
    "Use the available exec_shell_full_command tool now to inspect source/content/string evidence. "
    "Return only the tool call."
)
SHELL_FULL_READ_ONLY_MUTATION_RETRY_PROMPT = (
    "The previous shell-full command was rejected because the latest user request only asks to read, inspect, search, "
    "summarize, or explain content, but the command would modify local state.\n\n"
    "Do not write, edit, replace, delete, move, rename, install, commit, or otherwise mutate files or system state for this turn.\n\n"
    "Either answer from existing valid evidence, or return exactly one non-mutating tool call that gathers the missing evidence:\n\n"
    '{"command":"..."}'
)
SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT = (
    "Your previous command inspected only metadata or listings.\n\n"
    "For this task, inspect real file, document, source, string, or test content before continuing.\n\n"
    "Use commands such as cat, sed -n, grep/rg on file contents, or test output.\n"
    "For PDFs, prefer pdftotext <file> - piped to sed, head, tail, or grep.\n\n"
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
SHELL_FULL_ANALYSIS_COMPLETION_GUARD_PROMPT = (
    "You already have direct content/source evidence from previous tool results.\n\n"
    "If that evidence is sufficient to answer the user, stop calling tools and answer now.\n\n"
    "Only request one more tool if a specific missing fact is still required. "
    "If you need another tool, prefer a direct content-reading command over more discovery or listings.\n\n"
    "Do not use broad directory discovery unless the missing fact is explicitly about directory contents.\n\n"
    "Either answer directly in plain prose, or return exactly one JSON tool call:\n\n"
    '{"command":"..."}'
)
SHELL_FULL_FILE_RECOVERY_GUARD_PROMPT_PREFIX = (
    "Requested file not read yet.\n\n"
    "Use the real evidence below.\n"
    "If a candidate path exists, prefer one direct content-reading command on it.\n"
    "Otherwise use one targeted discovery step.\n"
    "If the file is unavailable, answer clearly.\n\n"
    "Either answer briefly in plain prose, or return exactly one JSON tool call:\n\n"
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
    r"\b("
    r"analy[sz]e|analysis|review|inspect|"
    r"analizz|analisi|ispezion|esamina|esamin|"
    r"vulnerab|exploit|malware|dropper|c2|ioc|reverse|decompil|static|"
    r"summar(?:y|ize)|riassunt\w*|riassum\w*|sintesi"
    r")\b",
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
_USER_PATH_RE = re.compile(
    r"(?:[\"'`])([^\"'`\n]+?\.[A-Za-z0-9]{1,8})(?:[\"'`])|(?:^|[\s(])([A-Za-z0-9_./-]+?\.[A-Za-z0-9]{1,8})(?=$|[\s),:;])"
)
_URL_RE = re.compile(r"https?://[^\s<>'\")]+", re.IGNORECASE)
_METADATA_REQUEST_RE = re.compile(
    r"\b(?:metadata|meta-data|stat|stats|size|permission|permissions|owner|group|mtime|ctime|timestamp|timestamps|modified|file\s+info|file\s+details)\b",
    re.IGNORECASE,
)
_URL_CONTENT_REQUEST_RE = re.compile(
    r"\b(?:fetch|read|open|explain|summari[sz]e|analy[sz]e|review|inspect|translate|describe|thesis|central\s+thesis|riassum\w*|sintesi|spiega|leggi|apri|analizz\w*|descrivi|traduci)\b",
    re.IGNORECASE,
)
_TRIVIAL_URL_NONFETCH_RE = re.compile(r"^\s*(?:echo|printf|true|false|pwd)(?:\s|$)", re.IGNORECASE)
_TRANSFER_PROGRESS_HEADER_RE = re.compile(
    r"^\s*% Total\s+% Received\s+% Xferd\s+Average Speed\s+Time\s+Time\s+Time\s+Current\s*$",
    re.IGNORECASE,
)
_TRANSFER_PROGRESS_COLUMNS_RE = re.compile(
    r"^\s*Dload\s+Upload\s+Total\s+Spent\s+Left\s+Speed\s*$",
    re.IGNORECASE,
)
_TRANSFER_PROGRESS_ROW_RE = re.compile(
    r"^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+[-:\d]+\s+[-:\d]+\s+[-:\d]+\s+\d+\s*$"
)
_URL_FETCH_FAILURE_RE = re.compile(
    r"\b(?:"
    r"HTTP/\d(?:\.\d)?\s+[45]\d{2}\b|"
    r"[45]\d{2}\s+(?:Not Found|Forbidden|Unauthorized|Internal Server Error|Bad Gateway|Service Unavailable)\b|"
    r"Could not resolve host|"
    r"Name or service not known|"
    r"Temporary failure in name resolution|"
    r"Connection refused|"
    r"Failed to connect|"
    r"Operation timed out|"
    r"timed out|"
    r"TLS|"
    r"SSL|"
    r"certificate|"
    r"Proxy CONNECT aborted"
    r")",
    re.IGNORECASE,
)
_HTTP_STATUS_LINE_RE = re.compile(r"^\s*HTTP/\d(?:\.\d)?\s+\d{3}\b", re.IGNORECASE)
_HTTP_HEADER_LINE_RE = re.compile(r"^[A-Za-z0-9-]+:\s+.+$")
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
                'For generic web search, use orbit-web-search "query"; for explicit URL content requests, prefer fetch_url and use shell fetch commands only when needed. '
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
    requested_url = extract_requested_user_url(user_prompt)
    if requested_url and requires_url_content_evidence(user_prompt):
        if raw_command.strip().startswith("orbit-web-search"):
            return (
                f"{SHELL_FULL_CONTRACT_ERROR_PREFIX}, explicit URL requests require direct URL fetch or a real HTTP/network failure, "
                "not generic web search."
            )
        if not is_explicit_url_fetch_shell_command(raw_command, requested_url=requested_url):
            return (
                f"{SHELL_FULL_CONTRACT_ERROR_PREFIX}, explicit URL requests require direct URL content/failure evidence from the requested URL."
            )
    if not _requires_direct_content_evidence(user_prompt):
        return None
    if _CONTENT_EVIDENCE_RE.search(raw_command):
        return None
    if _METADATA_ONLY_RE.search(raw_command):
        return (
            f"{SHELL_FULL_CONTRACT_ERROR_PREFIX}, not only metadata/listing. "
            "Use a bounded command such as sed/head/grep/strings on the target file."
    )
    return None


def validate_read_only_shell_mutation(arguments: dict[str, Any], *, user_prompt: str | None) -> str | None:
    raw_command = arguments.get("command")
    if not isinstance(raw_command, str) or not raw_command.strip():
        return None
    if not is_read_only_user_request(user_prompt):
        return None
    if not is_mutating_shell_command(raw_command):
        return None
    return (
        f"{SHELL_FULL_READ_ONLY_MUTATION_ERROR_PREFIX}. "
        "Use a non-mutating command or answer from existing evidence unless the latest user request explicitly asks for a change."
    )


def _requires_direct_content_evidence(user_prompt: str) -> bool:
    if _ANALYSIS_PROMPT_RE.search(user_prompt):
        return True
    if _METADATA_REQUEST_RE.search(user_prompt):
        return False
    return _READ_ONLY_PROMPT_RE.search(user_prompt) is not None and _USER_PATH_RE.search(user_prompt) is not None


def extract_requested_user_url(prompt: str | None) -> str | None:
    if not prompt:
        return None
    match = _URL_RE.search(prompt)
    if not match:
        return None
    return match.group(0).rstrip(".,;:!?")


def requires_url_content_evidence(prompt: str | None) -> bool:
    if not prompt:
        return False
    if not extract_requested_user_url(prompt):
        return False
    return _URL_CONTENT_REQUEST_RE.search(prompt) is not None


def is_explicit_url_fetch_shell_command(command: str | None, *, requested_url: str) -> bool:
    if not command:
        return False
    if requested_url not in command:
        return False
    if is_metadata_only_shell_command(command):
        return False
    return _TRIVIAL_URL_NONFETCH_RE.search(command) is None


def looks_like_transfer_progress_only(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    cleaned = _strip_transfer_progress_noise(text)
    return not bool(cleaned.strip())


def looks_like_url_fetch_failure(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    if shell_failure_from_output(text) is not None:
        return True
    cleaned = _strip_transfer_progress_noise(text)
    if not cleaned.strip():
        return False
    if _URL_FETCH_FAILURE_RE.search(cleaned):
        return True
    if _looks_like_http_headers_only(cleaned) and re.search(r"^\s*HTTP/\d(?:\.\d)?\s+[45]\d{2}\b", cleaned, re.IGNORECASE | re.MULTILINE):
        return True
    return False


def looks_like_url_content_evidence(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    if shell_failure_from_output(text) is not None:
        return False
    if looks_like_transfer_progress_only(text):
        return False
    cleaned = _strip_transfer_progress_noise(text)
    if not cleaned.strip():
        return False
    if looks_like_url_fetch_failure(cleaned):
        return False
    if _looks_like_http_headers_only(cleaned):
        return False
    if "shell_output_html_cleaned: true" in cleaned and "[no readable text extracted]" not in cleaned:
        return True
    if _looks_like_html_or_fragment(cleaned):
        return True
    compact = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return len(compact) >= 40 or len(compact.split()) >= 6


def is_shell_full_contract_error(content: str) -> bool:
    return content.startswith(SHELL_FULL_CONTRACT_ERROR_PREFIX)


def is_shell_full_read_only_mutation_error(content: str) -> bool:
    return content.startswith(SHELL_FULL_READ_ONLY_MUTATION_ERROR_PREFIX)


def build_shell_full_file_recovery_guard_prompt(
    *,
    requested_path: str,
    last_error: str | None,
    candidate_paths: list[str],
    requested_path_exists: bool = False,
    last_command: str | None = None,
    last_failure_content: str | None = None,
) -> str:
    prompt_prefix = SHELL_FULL_FILE_RECOVERY_GUARD_PROMPT_PREFIX
    if requested_path_exists:
        prompt_prefix = (
            "Requested file not read yet.\n\n"
            "Use the real evidence below.\n"
            "The requested file exists, but no usable content has been extracted from it yet.\n"
            "Before answering, use one appropriate direct content-reading command on the file or a confirmed candidate path.\n"
            "Use commands such as cat, sed -n, head, tail, or grep/rg on the file contents.\n\n"
            "Return only JSON:\n\n"
            '{"command":"..."}'
        )
    details = [f"Requested file: {requested_path}"]
    if requested_path_exists:
        details.append("Requested file currently exists in the workdir: yes")
    if last_error:
        details.append(f"Direct read failure: {last_error}")
    failure = shell_failure_from_output(last_failure_content or "")
    if last_command:
        details.append(f"Last failed command: {last_command}")
    if failure is not None:
        details.append(f"Last exit code: {failure.exit_code}")
        if failure.stderr:
            details.append(f"Last stderr: {_bounded_stream(failure.stderr)}")
        elif failure.stdout:
            details.append(f"Last stdout: {_bounded_stream(failure.stdout)}")
    if candidate_paths:
        details.append("Candidate paths from prior discovery:")
        details.extend(f"- {path}" for path in candidate_paths[:5])
    else:
        details.append("Candidate paths from prior discovery: none")
    if requested_path_exists and requested_path.lower().endswith(".pdf"):
        details.append("No usable content has been extracted from the requested PDF yet.")
        details.append("Do not conclude that the file is missing or unreadable yet.")
        details.append("Verify the file path in the workdir, then use a simple PDF text extractor if available, preferably pdftotext.")
        details.append("If pdftotext is unavailable or still fails, try a minimal text fallback such as strings.")
        details.append("Avoid specialist PDF utilities when the goal is only to read text and summarize the document.")
        details.append("Use only text actually extracted from the file in the final answer.")
    elif requested_path_exists:
        details.append("No usable content has been extracted from the requested file yet.")
        details.append("Do not conclude that the file is missing or unreadable yet.")
    return f"{prompt_prefix}\n\n" + "\n".join(details)


def build_shell_full_url_recovery_guard_prompt(
    *,
    requested_url: str,
    last_command: str | None = None,
    last_failure_content: str | None = None,
    fetch_attempted: bool = False,
) -> str:
    details = [
        "Requested URL content not fetched yet.",
        f"Requested URL: {requested_url}",
    ]
    if fetch_attempted:
        details.append("A previous command still did not provide usable URL content or a real HTTP/network failure for that URL.")
    details.extend(
        [
            "Before answering, use fetch_url or one other appropriate tool to retrieve the URL content or verify the real HTTP/network failure.",
            "Do not speculate about the page contents or existence from the URL alone.",
            "Do not replace the direct fetch with generic web search unless you already have direct fetch failure evidence.",
        ]
    )
    failure = shell_failure_from_output(last_failure_content or "")
    if last_command:
        details.append(f"Last command: {last_command}")
    if failure is not None:
        details.append(f"Last exit code: {failure.exit_code}")
        if failure.stderr:
            details.append(f"Last stderr: {_bounded_stream(failure.stderr)}")
        elif failure.stdout:
            details.append(f"Last stdout: {_bounded_stream(failure.stdout)}")
    details.extend(
        [
            "",
            "Return only the tool call.",
        ]
    )
    return "\n".join(details)


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
    if is_read_only_user_request(user_prompt):
        return False
    return is_mutating_shell_command(command)


def is_read_only_user_request(user_prompt: str | None) -> bool:
    if not user_prompt:
        return False
    if _NEGATED_MUTATION_PROMPT_RE.search(user_prompt):
        return True
    intent_text = _without_user_paths_and_urls(user_prompt)
    return _READ_ONLY_PROMPT_RE.search(user_prompt) is not None and _MUTATION_PROMPT_RE.search(intent_text) is None


def is_mutative_user_request(user_prompt: str | None) -> bool:
    if not user_prompt:
        return False
    intent_text = _without_user_paths_and_urls(user_prompt)
    if _NEGATED_MUTATION_PROMPT_RE.search(user_prompt) and not re.search(
        r"\b(?:fix|update|change|create|write|rename|refactor|edit)\b.*\b(?:file|code|implementation|config|test)\b",
        intent_text,
        re.IGNORECASE,
    ):
        return False
    if _READ_ONLY_PROMPT_RE.search(user_prompt) and not _MUTATION_PROMPT_RE.search(intent_text):
        return False
    return _MUTATION_PROMPT_RE.search(intent_text) is not None


def _without_user_paths_and_urls(text: str) -> str:
    without_urls = _URL_RE.sub(" ", text)
    return _USER_PATH_RE.sub(" ", without_urls)


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


def _strip_transfer_progress_noise(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _TRANSFER_PROGRESS_HEADER_RE.match(stripped):
            continue
        if _TRANSFER_PROGRESS_COLUMNS_RE.match(stripped):
            continue
        if _TRANSFER_PROGRESS_ROW_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _looks_like_http_headers_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    header_like = 0
    for index, line in enumerate(lines):
        if index == 0 and _HTTP_STATUS_LINE_RE.match(line):
            header_like += 1
            continue
        if _HTTP_HEADER_LINE_RE.match(line):
            header_like += 1
            continue
    return header_like == len(lines)


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
    text, method = extract_pdf_text(target)
    if not text.strip():
        return f"error: no text extracted from PDF: {target.name}"
    filtered = _apply_pdf_text_filters(raw_command, target=target, text=text)
    if filtered is None:
        result = read_pdf(str(target.relative_to(workdir)), arguments={}, workdir=workdir)
        if result.startswith("error:"):
            return result
        return "\n".join(
            [
                "shell_output_pdf_text: true",
                result.removeprefix("pdf_text: true\n"),
            ]
        )
    if len(filtered) <= PDF_CHUNK_CHARS:
        formatted = format_pdf_result(target, filtered, extractor=method)
    else:
        formatted = format_pdf_result(
            target,
            filtered[:PDF_CHUNK_CHARS],
            extractor=method,
            chunk_index=0,
            total_chunks=max(1, (len(filtered) + PDF_CHUNK_CHARS - 1) // PDF_CHUNK_CHARS),
            chars_start=0,
            chars_end=PDF_CHUNK_CHARS,
            total_length=len(filtered),
        )
    return "\n".join(["shell_output_pdf_text: true", formatted.removeprefix("pdf_text: true\n")])


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


def _apply_pdf_text_filters(raw_command: str, *, target: Path, text: str) -> str | None:
    try:
        stages = [shlex.split(stage) for stage in _split_shell_pipeline(raw_command)]
    except ValueError:
        return None
    if not stages:
        return None
    pdf_stage_index = -1
    target_name = target.name
    target_path = str(target)
    for index, stage in enumerate(stages):
        if any(token.strip("'\"") in {target_name, target_path} or token.strip("'\"").endswith(f"/{target_name}") for token in stage):
            pdf_stage_index = index
            break
    if pdf_stage_index < 0:
        return None
    current = text
    for stage in stages[pdf_stage_index + 1 :]:
        current = _apply_pdf_text_filter_stage(current, stage)
        if current is None:
            return None
    return current


def _split_shell_pipeline(raw_command: str) -> list[str]:
    stages: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape = False
    for char in raw_command:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            continue
        if char == "|":
            stage = "".join(current).strip()
            if stage:
                stages.append(stage)
            current = []
            continue
        current.append(char)
    stage = "".join(current).strip()
    if stage:
        stages.append(stage)
    return stages


def _apply_pdf_text_filter_stage(text: str, stage: list[str]) -> str | None:
    if not stage:
        return text
    command = Path(stage[0]).name
    lines = text.splitlines()
    if command == "head":
        count = _read_line_count_option(stage[1:], default=10)
        return "\n".join(lines[:count])
    if command == "tail":
        count = _read_line_count_option(stage[1:], default=10)
        return "\n".join(lines[-count:] if count > 0 else [])
    if command == "sed":
        return _apply_pdf_sed_filter(lines, stage[1:])
    if command in {"grep", "rg"}:
        return _apply_pdf_grep_filter(lines, stage)
    return text


def _read_line_count_option(tokens: list[str], *, default: int) -> int:
    for index, token in enumerate(tokens):
        if token == "-n" and index + 1 < len(tokens):
            return max(0, _safe_int(tokens[index + 1], default))
        if token.startswith("-n") and len(token) > 2:
            return max(0, _safe_int(token[2:], default))
    return default


def _apply_pdf_sed_filter(lines: list[str], tokens: list[str]) -> str:
    if "-n" not in tokens:
        return "\n".join(lines)
    for token in tokens:
        match = re.fullmatch(r"(\d+),(\d+)p", token)
        if match:
            start = max(1, int(match.group(1)))
            end = max(start, int(match.group(2)))
            return "\n".join(lines[start - 1 : end])
    return "\n".join(lines)


def _apply_pdf_grep_filter(lines: list[str], stage: list[str]) -> str:
    flags = 0
    invert = False
    patterns: list[str] = []
    tokens = stage[1:]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-v":
            invert = True
        elif token == "-i":
            flags |= re.IGNORECASE
        elif token == "-E":
            pass
        elif token in {"-e", "--regexp"} and index + 1 < len(tokens):
            index += 1
            patterns.append(tokens[index])
        elif token.startswith("-") and set(token[1:]).issubset({"i", "v", "E"}):
            if "i" in token:
                flags |= re.IGNORECASE
            if "v" in token:
                invert = True
        elif not patterns:
            patterns.append(token)
        index += 1
    if not patterns:
        return "\n".join(lines)
    regex = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), flags)
    filtered = [line for line in lines if bool(regex.search(line)) != invert]
    return "\n".join(filtered)


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


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
        return bool(re.search(r"(^|[^<>])>{1,2}[^&]", _without_shell_quoted_strings(command)))
    return any(token in {">", ">>", "2>", "2>>", "&>", "&>>"} for token in tokens) or bool(
        re.search(r"(^|[^<>])>{1,2}[^&]", _without_shell_quoted_strings(command))
    )


def _without_shell_quoted_strings(command: str) -> str:
    chars: list[str] = []
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            chars.append(" " if quote else char)
            escaped = False
            continue
        if char == "\\":
            chars.append(" " if quote else char)
            escaped = bool(quote)
            continue
        if quote:
            if char == quote:
                quote = None
                chars.append(char)
            else:
                chars.append(" ")
            continue
        if char in {"'", '"'}:
            quote = char
            chars.append(char)
            continue
        chars.append(char)
    return "".join(chars)


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
