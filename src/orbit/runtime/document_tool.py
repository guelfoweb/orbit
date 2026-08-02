from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from orbit.runtime.file_tools import MAX_FULL_DOCUMENT_BYTES, load_text_file


DISPLAY_MAX_LINES = 200
DISPLAY_MAX_BYTES = 64 * 1024
FILE_DISPLAY_MARKER = "file_display_result: true"


def execute_read_file(arguments: dict[str, Any], *, workdir: Path) -> str:
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return "error: read_file requires a non-empty path"
    loaded = load_text_file(path, workdir=workdir, max_bytes=MAX_FULL_DOCUMENT_BYTES)
    if isinstance(loaded, str):
        return loaded
    target, text = loaded
    return _display(target, text, arguments=arguments, workdir=workdir)


def _display(target: Path, text: str, *, arguments: dict[str, Any], workdir: Path) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cursor = arguments.get("cursor")
    start_line = arguments.get("start_line", 1)
    if cursor is not None:
        if "start_line" in arguments:
            return "error: cursor and start_line cannot be used together"
        parsed = _parse_cursor(cursor)
        if parsed is None:
            return "error: invalid read_file cursor"
        expected_digest, start_line = parsed
        if expected_digest != digest:
            return "error: file changed since the previous page; restart from the first page"
    line_count = arguments.get("line_count", 100)
    explicit_selection = any(key in arguments for key in ("start_line", "line_count", "cursor"))
    if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
        return "error: start_line must be an integer >= 1"
    if not isinstance(line_count, int) or isinstance(line_count, bool) or not 1 <= line_count <= DISPLAY_MAX_LINES:
        return f"error: line_count must be between 1 and {DISPLAY_MAX_LINES}"
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return "\n".join(
            [
                FILE_DISPLAY_MARKER,
                f"path: {_relative(target, workdir)}",
                "bytes: 0",
                "lines: 0",
                f"sha256: {digest}",
                "coverage: complete",
                f"selection: {'explicit' if explicit_selection else 'default'}",
                "line_range: none",
                "next_cursor: none",
                "content:",
                "",
            ]
        )
    if start_line > max(1, total_lines):
        return f"error: start_line out of range: {start_line}, total lines {total_lines}"
    selected = "".join(lines[start_line - 1 : start_line - 1 + line_count]) if lines else ""
    selected_bytes = len(selected.encode("utf-8"))
    if selected_bytes > DISPLAY_MAX_BYTES:
        return (
            f"error: selected display page is {selected_bytes} bytes, max {DISPLAY_MAX_BYTES}; "
            "request fewer lines"
        )
    shown = min(line_count, max(0, total_lines - start_line + 1))
    end_line = start_line + shown - 1 if shown else start_line
    complete = start_line == 1 and end_line == total_lines
    return "\n".join(
        [
            FILE_DISPLAY_MARKER,
            f"path: {_relative(target, workdir)}",
            f"bytes: {len(text.encode('utf-8'))}",
            f"lines: {total_lines}",
            f"sha256: {digest}",
            f"coverage: {'complete' if complete else 'partial'}",
            f"selection: {'explicit' if explicit_selection else 'default'}",
            f"line_range: {start_line}-{end_line}",
            f"next_cursor: {f'v1:{digest}:{end_line + 1}' if end_line < total_lines else 'none'}",
            "content:",
            selected,
        ]
    )


def _parse_cursor(value: object) -> tuple[str, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"v1:([0-9a-f]{64}):([1-9][0-9]*)", value)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _relative(target: Path, workdir: Path) -> str:
    return str(target.relative_to(workdir.expanduser().resolve()))
