from __future__ import annotations

import json
import re
import shlex


LARGE_TOOL_RESULT_CHARS = 10_000
SHELL_FULL_CONTRACT_ERROR_PREFIX = "error: shell-full analysis requests require content/source/string evidence"
PREVIEW_LINE_LIMIT = 3
PREVIEW_INLINE_LIMIT = 120
COMMAND_PREVIEW_LIMIT = 96
DISPLAY_TOOL_NAMES = {
    "exec_shell_full_command": "exec",
    "fetch_url": "fetch_url",
}


def display_tool_name(name: str) -> str:
    return DISPLAY_TOOL_NAMES.get(name, name)


def format_tool_call_event(name: str, args: str) -> str:
    if name == "exec_shell_full_command":
        command = _command_from_args(args)
        if command:
            return _format_shell_command_call(command)
    if name == "fetch_url":
        url = _url_from_args(args)
        if url:
            return f"Fetch: {_truncate_inline(url, limit=COMMAND_PREVIEW_LIMIT)}"
    return f"{display_tool_name(name)} {args}"


def format_tool_result_event(name: str, chars: int, source: str | None = None, content: str | None = None) -> str:
    del source
    preview = _tool_result_preview(content)
    suffix_parts: list[str] = []
    if _is_rejected_contract_result(content):
        suffix_parts.append("rejected")
    if chars >= LARGE_TOOL_RESULT_CHARS:
        suffix_parts.append("large context")
    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    chunk = _chunk_label(content)
    prefix = f"{chunk} " if chunk else ""
    preview_text = f"{_truncate_inline(preview, limit=PREVIEW_INLINE_LIMIT)} | " if preview else ""
    return f" └ {prefix}{preview_text}{chars} chars -> model{suffix}"


def _command_from_args(args: str) -> str | None:
    try:
        parsed = json.loads(args)
    except Exception:
        return None
    if isinstance(parsed, dict):
        command = parsed.get("command")
        if isinstance(command, str) and command.strip():
            return command.strip()
    return None


def _url_from_args(args: str) -> str | None:
    try:
        parsed = json.loads(args)
    except Exception:
        return None
    if isinstance(parsed, dict):
        url = parsed.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _format_shell_command_call(command: str) -> str:
    category = _shell_command_category(command)
    return f"{category}: {_truncate_inline(command, limit=COMMAND_PREVIEW_LIMIT)}"


def _shell_command_category(command: str) -> str:
    primary, tokens = _shell_command_tokens(command)
    lowered = command.lower()
    if primary in {"curl", "wget", "lynx", "links"} or "orbit-web-search" in lowered or "http://" in lowered or "https://" in lowered:
        return "Web"
    if primary in {"rg", "grep", "ag", "ack"}:
        return "Search"
    if primary == "find":
        if any(flag in tokens for flag in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex")):
            return "Search"
        return "List"
    if primary in {"ls", "tree", "du"}:
        return "List"
    if primary in {"cat", "head", "tail", "sed", "awk", "python", "python3", "perl", "strings", "pdftotext"}:
        if primary in {"sed", "perl"} and "-i" in tokens:
            return "Edit"
        if primary in {"python", "python3", "perl"} and any(operator in command for operator in (">", ">>", ".write(", "write_text(", "write_bytes(")):
            return "Write"
        return "Read"
    if primary in {"tee", "cp", "mv", "mkdir", "touch", "install", "ln", "truncate"}:
        return "Write"
    if primary in {"rm", "rmdir"}:
        return "Edit"
    if any(operator in command for operator in (">", ">>")):
        return "Write"
    return "Exec"


def _shell_command_tokens(command: str) -> tuple[str, tuple[str, ...]]:
    try:
        tokens = tuple(shlex.split(command))
    except ValueError:
        tokens = tuple(command.split())
    primary = tokens[0] if tokens else ""
    return primary, tokens


def _tool_result_preview(content: str | None) -> str | None:
    if not content:
        return None
    stripped = content.strip()
    if not stripped:
        return None
    if _is_rejected_contract_result(content):
        return "rejected metadata-only output"
    if stripped.startswith("error:"):
        return stripped.splitlines()[0]
    if "url_fetch: true" in content:
        title = _metadata_value(content, "title")
        if title and title != "null":
            return title
        preview = _body_preview(content, marker="text:")
        if preview:
            return preview
        error = _metadata_value(content, "error")
        status = _metadata_value(content, "status")
        if error and error != "null":
            return f"{status or 'fetch'}: {error}"
        if status and status != "null":
            return status
    path = _metadata_value(content, "path")
    if "shell_output_pdf_text: true" in content:
        preview = _body_preview(content, marker="content:")
        return _prefix_path_preview(path, preview or "PDF text extracted")
    if "shell_output_read_file: true" in content:
        preview = _body_preview(content, marker="content:")
        return _prefix_path_preview(path, preview or "file content loaded")
    if "shell_output_html_cleaned: true" in content:
        preview = _body_preview(content, marker="text:")
        return preview or "page text extracted"
    if "content:\n" in content:
        preview = _body_preview(content, marker="content:")
        return _prefix_path_preview(path, preview)
    preview = _lines_preview(content)
    return preview


def _chunk_label(content: str | None) -> str | None:
    if not content or "shell_output_read_file: true" not in content:
        return None
    chunk_match = re.search(r"^chunk_index:\s*(\d+)$", content, flags=re.MULTILINE)
    total_match = re.search(r"^total_chunks:\s*(\d+)$", content, flags=re.MULTILINE)
    if not chunk_match or not total_match:
        return None
    chunk_index = int(chunk_match.group(1)) + 1
    total_chunks = int(total_match.group(1))
    return f"chunk {chunk_index}/{total_chunks}"


def _is_rejected_contract_result(content: str | None) -> bool:
    return bool(content and content.startswith(SHELL_FULL_CONTRACT_ERROR_PREFIX))


def _metadata_value(content: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", content, flags=re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _body_preview(content: str, *, marker: str) -> str | None:
    if f"{marker}\n" not in content:
        return None
    _prefix, body = content.split(f"{marker}\n", 1)
    return _lines_preview(body)


def _lines_preview(content: str) -> str | None:
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("shell_output_", "path:", "extractor:", "chunk_index:", "total_chunks:", "chars:", "large_file_excerpt:", "url_fetch:", "url:", "final_url:", "http_status:", "content_type:", "encoding:", "title:", "text_truncated:", "status:", "error:")):
            continue
        if line == "[truncated]":
            continue
        lines.append(_truncate_inline(line, limit=48))
        if len(lines) >= PREVIEW_LINE_LIMIT:
            break
    if not lines:
        return None
    return " | ".join(lines)


def _prefix_path_preview(path: str | None, preview: str | None) -> str | None:
    if path and preview:
        return f"{path}: {preview}"
    return path or preview


def _truncate_inline(text: str | None, *, limit: int) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"
