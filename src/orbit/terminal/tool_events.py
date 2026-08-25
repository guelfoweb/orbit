from __future__ import annotations

import json
import re
import shlex

from orbit.terminal.theme import sanitize_terminal_text


LARGE_TOOL_RESULT_CHARS = 10_000
SHELL_FULL_CONTRACT_ERROR_PREFIX = "error: shell-full analysis requests require content/source/string evidence"
PREVIEW_LINE_LIMIT = 3
PREVIEW_INLINE_LIMIT = 120
# A streamed result is a progress signal, not the evidence record: it shows
# enough of what an action produced to follow along, while the full text
# stays one evidence id away. Sized below the guided step preview, which is
# what the analyst reads once control is handed back.
BLOCK_LINE_LIMIT = 12
BLOCK_LINE_CHARS = 160
# Total rows the block may occupy, blank separators included. The line
# budget counts content, so without this a body that is mostly blank space
# still costs a row per gap: 32000 empty lines around three statements
# scrolls the surrounding status out of view to show nothing.
BLOCK_ROW_LIMIT = 24
BLOCK_INDENT = "    "
DISPLAY_TOOL_NAMES = {
    "exec_shell_full_command": "Exec",
    "fetch_url": "Web",
    "list_directory": "Read",
    "system_info": "Read",
    "write_artifact": "Artifact",
    "verify_artifact": "Artifact",
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
            return f"› Web  {_normalize_inline(url)}"
    if name == "list_directory":
        path, recursive = _list_directory_from_args(args)
        if path:
            suffix = " · recursive" if recursive else ""
            return f"› Read  {_normalize_inline(path)}{suffix}"
    if name == "system_info":
        return "› Read  system information"
    if name == "write_artifact":
        try:
            parsed = json.loads(args)
        except (TypeError, ValueError):
            parsed = {}
        path = parsed.get("path") if isinstance(parsed, dict) else None
        if isinstance(path, str) and path:
            return f"› Artifact  {_normalize_inline(path)}"
    if name == "verify_artifact":
        try:
            parsed = json.loads(args)
        except (TypeError, ValueError):
            parsed = {}
        check = parsed.get("check") if isinstance(parsed, dict) else None
        if isinstance(check, str):
            return f"› Artifact  verify published artifact · {check}"
    detail = f"  {args}" if args else ""
    return f"› {display_tool_name(name)}{detail}"


def format_tool_activity_label(name: str, args: str) -> str:
    if name == "exec_shell_full_command":
        return _command_from_args(args) or name
    if name == "fetch_url":
        return _url_from_args(args) or name
    if name == "list_directory":
        path, _recursive = _list_directory_from_args(args)
        return path or name
    return display_tool_name(name)


def format_tool_result_event(name: str, chars: int, source: str | None = None, content: str | None = None) -> str:
    del source
    preview = _tool_result_preview(content)
    suffix_parts: list[str] = []
    if _is_rejected_contract_result(content):
        suffix_parts.append("rejected")
    if _is_truncated_result(content):
        suffix_parts.append("truncated")
    if chars >= LARGE_TOOL_RESULT_CHARS:
        suffix_parts.append("large context")
    suffix = f" · {' · '.join(suffix_parts)}" if suffix_parts else ""
    chunk = _chunk_label(content)
    prefix = f"{chunk} " if chunk else ""
    block = _result_block(content)
    if block:
        # A multi-line result gets a header line carrying the same metadata a
        # one-line result carries, then the output itself indented beneath it,
        # so code and structured text stay readable instead of running into
        # the status line beside them.
        header = f"└ {prefix}{chars} chars{suffix}"
        return "\n".join([header, *block])
    preview_text = _truncate_inline(preview, limit=PREVIEW_INLINE_LIMIT) if preview else None
    if preview_text:
        return f"└ {prefix}{preview_text}{suffix}"
    return f"└ {prefix}{chars} chars{suffix}"


def _result_block(content: str | None) -> list[str]:
    """The action's own output as an indented block, when a block helps.

    Structural only: the decision is the shape of the text -- how many lines,
    whether they carry indentation -- never what the text says. There is no
    language detection, no Markdown parsing and no syntax guessing.

    Three kinds of result deliberately keep the compact one-line form they
    already had, because a block would cost several lines to say what one line
    said just as clearly:

    - errors and contract refusals, whose first line is the whole point and
      which must stay visually distinct from output that succeeded;
    - metadata-only envelopes, which carry no action output at all;
    - flat short output such as a directory listing, where `a | b | c` reads
      better than the same three words stacked.

    What is left is what a block is for: the body of a real result, long
    enough or structured enough that its line breaks and indentation are part
    of the information.
    """
    if not content or not content.strip():
        return []
    if content.strip().startswith("error:"):
        # An error's first line is its subject, and the compact form keeps it
        # beside the failure marker rather than under a `chars` header. A
        # contract refusal is one of these: it opens with `error:` too, so this
        # covers refusals without a second check that could drift apart.
        return []
    body = _result_body(content)
    if body is None:
        return []
    stripped = body.strip("\n")
    if not stripped.strip():
        return []
    lines = [line for line in _display_lines(stripped) if line.strip()]
    if len(lines) < 2:
        return []
    if not _is_structured(lines):
        return []
    return _block_lines(stripped)


# Flat lines carry no shape of their own, so they only earn a block once
# there are too many to read as an inline join. A directory listing of a few
# entries is the common case and stays compact; a screen of them does not.
MIN_FLAT_BLOCK_LINES = 8


def _is_structured(lines: list[str]) -> bool:
    """Whether these lines need their own shape to stay readable.

    Indentation is the signal that line breaks carry meaning -- it is what
    distinguishes a source file or a nested structure from a flat list of
    names. Failing that, enough lines to no longer fit an inline join.

    Only space and tab count as indentation. `str.isspace` is also true of
    carriage return, vertical tab, form feed and the Unicode separators, which
    would let a single control byte in tool output decide which shape Orbit
    renders -- harmless to the terminal, since the text is escaped either way,
    but the choice should be Orbit's.
    """
    if any(line[:1] in (" ", "\t") for line in lines):
        return True
    return len(lines) >= MIN_FLAT_BLOCK_LINES


def _result_body(content: str) -> str | None:
    """The action's output within a transport envelope, or the content itself."""
    for marker in ("content:\n", "text:\n"):
        if marker in content:
            return content.split(marker, 1)[1]
    if _METADATA_HEAD.match(content):
        # A recognized envelope with no body marker carries no action output
        # to show: everything it has is metadata the header line covers.
        return None
    return content


# An envelope opens with a transport token Orbit itself wrote, anchored at the
# start of the result.
#
# Only these distinctive tokens count. `path:`, `status:` and `chars:` are
# fields *inside* an envelope, never its opening line, and treating them as
# envelope markers would misread real command output that happens to begin
# with one -- `path: /etc/hosts` from a config dump, say -- as metadata, and
# silently drop its body from the display.
_METADATA_HEAD = re.compile(
    r"^(shell_output_\w+:|url_fetch:|directory_listing:|system_info:)"
)


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


def _list_directory_from_args(args: str) -> tuple[str | None, bool]:
    try:
        parsed = json.loads(args)
    except Exception:
        return None, False
    if isinstance(parsed, dict):
        path = parsed.get("path")
        recursive = parsed.get("recursive")
        return (path if isinstance(path, str) and path.strip() else ".", bool(recursive) if isinstance(recursive, bool) else False)
    return None, False


def _format_shell_command_call(command: str) -> str:
    category = _shell_command_category(command)
    return f"› {category}  {_normalize_inline(command)}"


def _shell_command_category(command: str) -> str:
    primary, tokens = _shell_command_tokens(command)
    lowered = command.lower()
    if primary in {"curl", "wget", "lynx", "links"} or "orbit-web-search" in lowered or "http://" in lowered or "https://" in lowered:
        return "Web"
    if primary in {"rg", "grep", "ag", "ack"}:
        return "Read"
    if primary == "find":
        if any(flag in tokens for flag in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex")):
            return "Read"
        return "Read"
    if primary in {"ls", "tree", "du"}:
        return "Read"
    if primary in {"cat", "head", "tail", "sed", "awk", "python", "python3", "perl", "strings", "pdftotext"}:
        if primary in {"sed", "perl"} and "-i" in tokens:
            return "Exec"
        if primary in {"python", "python3", "perl"} and any(operator in command for operator in (">", ">>", ".write(", "write_text(", "write_bytes(")):
            return "Exec"
        return "Read"
    if primary in {"tee", "cp", "mv", "mkdir", "touch", "install", "ln", "truncate"}:
        return "Exec"
    if primary in {"rm", "rmdir"}:
        return "Exec"
    if any(operator in command for operator in (">", ">>")):
        return "Exec"
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
    if content.startswith("directory_listing:"):
        if "error=true" in content:
            status = _metadata_inline_value(content, "status")
            return f"directory listing {status or 'error'}"
        preview = _lines_preview(content)
        return preview or "directory listing"
    if content.startswith("system_info:"):
        preview = _lines_preview(content)
        return preview or "system info"
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


def _is_truncated_result(content: str | None) -> bool:
    return bool(
        content
        and (
            content.strip() == "[truncated]"
            or "\n[truncated]" in content
            or "text_truncated: true" in content
            or "result_truncated: true" in content
            or "results_truncated: true" in content
            or "truncated=true" in content
            or "large_file_excerpt: true" in content
        )
    )


def _metadata_value(content: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", content, flags=re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _metadata_inline_value(content: str, key: str) -> str | None:
    first = content.splitlines()[0] if content else ""
    match = re.search(rf"\b{re.escape(key)}=([^\s]+)", first)
    return match.group(1) if match else None


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
        if line.startswith(("shell_output_", "path:", "extractor:", "chunk_index:", "total_chunks:", "chars:", "large_file_excerpt:", "url_fetch:", "url:", "final_url:", "http_status:", "content_type:", "encoding:", "title:", "text_truncated:", "status:", "directory_listing:", "error:")):
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
    normalized = _normalize_inline(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _normalize_inline(text: str) -> str:
    """One terminal-safe line, for text that shares a line with metadata.

    Collapsing whitespace removes newlines and tabs but leaves the rest of the
    control range intact, and an escape sequence is not whitespace: without
    sanitizing, tool output could move the cursor or erase the line it is
    printed on. `sanitize_terminal_text` is the shared boundary, applied
    before collapsing so an escaped sequence becomes ordinary visible text
    rather than a control character that survives as-is.
    """
    return " ".join(sanitize_terminal_text(text).split())


def _display_lines(text: str) -> list[str]:
    """Split on newline only, so no other control character can forge a line."""
    return text.split("\n")


def _block_lines(text: str) -> list[str]:
    """`text` as indented display lines, bounded and marked when shortened.

    Line breaks and leading indentation are what make code and structured
    output readable, so they are preserved rather than collapsed; each line is
    sanitized on its own, which escapes any control character without touching
    the printable text around it. Nothing here inspects what the text is: no
    language detection, no Markdown, no syntax guessing -- only how much of it
    fits on a terminal.

    Only `\n` starts a new line here. `str.splitlines` also breaks on carriage
    return, vertical tab, form feed and the Unicode separators, which would let
    tool output invent lines inside the block -- the same line forgery the
    sanitizer exists to stop. Split on `\n` alone and every one of those stays
    an ordinary character that sanitizing turns into visible text.
    """
    out: list[str] = []
    shortened = False
    kept = 0
    for index, line in enumerate(_display_lines(text)):
        if not line.strip():
            # A blank line separates what is around it; it is not itself a
            # line of output, so it does not spend the line budget. Emitted
            # empty rather than indented: indenting nothing only adds trailing
            # whitespace the renderer invented. It still costs a row, so that
            # a long run of them cannot scroll the terminal on its own.
            if len(out) >= BLOCK_ROW_LIMIT:
                shortened = True
                break
            out.append("")
            continue
        if kept == BLOCK_LINE_LIMIT or len(out) >= BLOCK_ROW_LIMIT:
            shortened = True
            break
        safe = sanitize_terminal_text(line)
        if len(safe) > BLOCK_LINE_CHARS:
            safe = safe[:BLOCK_LINE_CHARS] + "…"
            shortened = True
        out.append(f"{BLOCK_INDENT}{safe}")
        kept += 1
    del index
    while out and not out[-1]:
        # Trailing separators have nothing left to separate.
        out.pop()
    if shortened:
        # No char count here: it would be the length of this body, while the
        # header line already reports the result's own size. Two different
        # numbers for "the same" output on adjacent lines is worse than one.
        out.append(f"{BLOCK_INDENT}[preview truncated; full output in evidence]")
    return out
