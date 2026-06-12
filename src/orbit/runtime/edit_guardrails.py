from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from orbit.runtime.path_guardrails import (
    BINARY_OR_SPECIAL_EXTENSIONS,
    TEXT_EXTENSIONS,
    resolve_inside_workdir,
    validate_existing_file_path,
)
from orbit.runtime.file_tools import MAX_REPLACE_CHARS, MAX_TEXT_FILE_BYTES_AFTER_REPLACE


MAX_EDIT_CHANGES = 20
MAX_APPLY_DIFF_CHARS = 48_000
MAX_APPLY_DIFF_FILES = 12


def edit_file_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Apply small line edits to one existing UTF-8 file in workdir. "
                "Modes: replace, delete, append. Lines are 1-based. "
                "For appending at end of file, use line_start=-1 and line_end=-1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "changes": {
                        "type": "array",
                        "description": f"Line edits, max {MAX_EDIT_CHANGES}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string"},
                                "line_start": {"type": "integer"},
                                "line_end": {"type": "integer"},
                                "content": {"type": "string"},
                            },
                            "required": ["mode", "line_start", "line_end", "content"],
                        },
                    },
                },
                "required": ["path", "changes"],
            },
        },
    }


def apply_diff_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "apply_diff",
            "description": (
                "Apply a small valid git/unified diff to UTF-8 files in workdir. "
                "Use only for actual diff text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diff": {
                        "type": "string",
                    }
                },
                "required": ["diff"],
            },
        },
    }


def prepare_edit_file(arguments: dict[str, Any], *, workdir: Path) -> dict[str, Any] | str:
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return "error: edit_file requires a relative path"
    target_or_error = _validate_existing_text_file(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    changes = arguments.get("changes")
    if not isinstance(changes, list) or not changes:
        return "error: edit_file requires a non-empty changes list"
    if len(changes) > MAX_EDIT_CHANGES:
        return f"error: too many edit_file changes: {len(changes)}, max {MAX_EDIT_CHANGES}"

    line_count = _line_count(target)
    normalized: list[dict[str, Any]] = []
    ranges: list[tuple[int, int]] = []
    for item in changes:
        if not isinstance(item, dict):
            return "error: each edit_file change must be an object"
        mode = item.get("mode")
        line_start = item.get("line_start")
        line_end = item.get("line_end")
        content = item.get("content")
        if mode not in {"replace", "delete", "append"}:
            return f"error: unsupported edit_file mode: {mode}"
        if mode == "append" and isinstance(line_start, int) and line_end is None:
            line_end = line_start
        if mode == "append" and line_start == -1:
            line_end = 0
        if mode == "append" and line_start == line_count + 1 and line_end == line_count + 1:
            line_start = -1
            line_end = 0
        if not isinstance(line_start, int) or not isinstance(line_end, int):
            return "error: edit_file line_start and line_end must be integers"
        if not isinstance(content, str):
            return "error: edit_file content must be a string"
        if len(content) > MAX_REPLACE_CHARS:
            return f"error: edit_file content too large: {len(content)} chars, max {MAX_REPLACE_CHARS}"
        if "\x00" in content:
            return "error: edit_file content appears to be binary"
        range_error = _validate_line_range(mode, line_start, line_end, line_count)
        if range_error:
            return range_error
        if mode == "delete" and content:
            return "error: delete changes must have empty content"
        if line_start != -1:
            ranges.append((line_start, line_end if mode != "append" else line_start))
        normalized.append({"mode": mode, "line_start": line_start, "line_end": line_end, "content": content})

    overlap_error = _validate_non_overlapping(ranges)
    if overlap_error:
        return overlap_error
    return {"path": str(target), "changes": normalized}


def apply_local_edit_file(arguments: dict[str, Any], *, workdir: Path) -> str:
    prepared = prepare_edit_file(arguments, workdir=workdir)
    if isinstance(prepared, str):
        return prepared
    target = Path(prepared["path"])
    text = target.read_text(encoding="utf-8")
    lines = text.splitlines()
    had_final_newline = text.endswith("\n")
    changes = list(prepared["changes"])

    for change in sorted(changes, key=_edit_sort_key, reverse=True):
        mode = change["mode"]
        line_start = change["line_start"]
        line_end = change["line_end"]
        content_lines = change["content"].splitlines()
        if mode == "replace":
            lines[line_start - 1 : line_end] = content_lines
        elif mode == "delete":
            del lines[line_start - 1 : line_end]
        elif mode == "append":
            if line_start == -1:
                lines.extend(content_lines)
            else:
                lines[line_start:line_start] = content_lines

    updated = "\n".join(lines)
    if lines and (had_final_newline or any(change["mode"] == "append" for change in changes)):
        updated += "\n"
    if len(updated.encode("utf-8")) > MAX_TEXT_FILE_BYTES_AFTER_REPLACE:
        return f"error: edited file too large: max {MAX_TEXT_FILE_BYTES_AFTER_REPLACE} bytes"
    write_error = _atomic_write_text(target, updated)
    if write_error:
        return write_error
    return f"edited {target.name}: {len(changes)} change(s)"


def _atomic_write_text(target: Path, content: str) -> str | None:
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            tmp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        tmp_name = None
        _fsync_directory(target.parent)
    except OSError as exc:
        if tmp_name:
            try:
                Path(tmp_name).unlink()
            except OSError:
                pass
        return f"error: cannot write edited file atomically: {exc}"
    return None


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _edit_sort_key(change: dict[str, Any]) -> int:
    line_start = change["line_start"]
    if line_start == -1:
        return 10**12
    return line_start


def prepare_apply_diff(arguments: dict[str, Any], *, workdir: Path, server_cwd: Path | None = None) -> dict[str, Any] | str:
    diff = arguments.get("diff")
    if not isinstance(diff, str) or not diff.strip():
        return "error: apply_diff requires a non-empty diff string"
    if len(diff) > MAX_APPLY_DIFF_CHARS:
        return f"error: apply_diff too large: {len(diff)} chars, max {MAX_APPLY_DIFF_CHARS}"
    if "\x00" in diff:
        return "error: apply_diff appears to contain binary data"
    if _has_forbidden_diff_metadata(diff):
        return "error: apply_diff rejects delete/rename/mode-change patches"

    server_root = (server_cwd or Path.cwd()).resolve()
    path_map: dict[str, str] = {}
    for raw_path in _diff_paths(diff):
        if raw_path == "/dev/null":
            continue
        rewritten_path = _validate_diff_path(raw_path, workdir=workdir, server_cwd=server_root)
        if rewritten_path.startswith("error:"):
            return rewritten_path
        path_map[raw_path] = rewritten_path
    if not path_map:
        return "error: apply_diff did not contain editable file paths"
    if len(set(path_map.values())) > MAX_APPLY_DIFF_FILES:
        return f"error: apply_diff touches too many files: max {MAX_APPLY_DIFF_FILES}"
    return {"diff": _rewrite_diff_paths(diff, path_map)}


def _validate_existing_text_file(path: str, *, workdir: Path) -> Path | str:
    target_or_error = validate_existing_file_path(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    return _validate_text_target(target_or_error)


def _validate_text_target(target: Path) -> Path | str:
    suffix = target.suffix.lower()
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: edit tools support UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for edit tools: {suffix}"
    if target.exists() and target.stat().st_size > MAX_TEXT_FILE_BYTES_AFTER_REPLACE:
        return f"error: file too large for edit tools: {target.stat().st_size} bytes, max {MAX_TEXT_FILE_BYTES_AFTER_REPLACE}"
    if target.exists():
        raw = target.read_bytes()
        if b"\x00" in raw:
            return "error: existing file appears to be binary and cannot be edited"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            return "error: existing file is not valid UTF-8 text"
    return target


def _line_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if not text:
        return 0
    return len(text.splitlines())


def _validate_line_range(mode: str, line_start: int, line_end: int, line_count: int) -> str | None:
    if line_start == -1:
        return None if mode == "append" else "error: line_start -1 is allowed only for append"
    if line_start < 1 or line_end < line_start:
        return "error: invalid edit_file line range"
    if mode == "append":
        if line_start > line_count:
            return f"error: append line_start out of range: {line_start}, lines {line_count}"
        return None
    if line_end > line_count:
        return f"error: edit_file line range out of range: {line_start}-{line_end}, lines {line_count}"
    return None


def _validate_non_overlapping(ranges: list[tuple[int, int]]) -> str | None:
    previous_end = 0
    for start, end in sorted(ranges):
        if start <= previous_end:
            return "error: edit_file changes must not overlap"
        previous_end = end
    return None


def _has_forbidden_diff_metadata(diff: str) -> bool:
    forbidden_prefixes = (
        "deleted file mode ",
        "rename from ",
        "rename to ",
        "similarity index ",
        "dissimilarity index ",
        "old mode ",
        "new mode ",
    )
    return any(line.startswith(forbidden_prefixes) for line in diff.splitlines())


def _diff_paths(diff: str) -> list[str]:
    paths: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) == 4:
                paths.extend([_strip_diff_prefix(parts[2]), _strip_diff_prefix(parts[3])])
        elif line.startswith("--- ") or line.startswith("+++ "):
            token = line.split(maxsplit=1)[1].split("\t", maxsplit=1)[0]
            paths.append(_strip_diff_prefix(token))
    return list(dict.fromkeys(paths))


def _strip_diff_prefix(path: str) -> str:
    if path in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _validate_diff_path(raw_path: str, *, workdir: Path, server_cwd: Path) -> str:
    if not raw_path or raw_path.startswith("/") or raw_path.startswith("~") or ".." in Path(raw_path).parts:
        return f"error: unsafe diff path: {raw_path}"
    target_or_error = resolve_inside_workdir(raw_path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if target.exists():
        validation = _validate_text_target(target)
        if isinstance(validation, str):
            return validation
    else:
        if not target.parent.exists():
            return f"error: parent directory does not exist for diff path: {raw_path}"
        suffix = target.suffix.lower()
        if suffix in BINARY_OR_SPECIAL_EXTENSIONS or (suffix and suffix not in TEXT_EXTENSIONS):
            return f"error: unsupported text/code file extension for diff path: {suffix}"
    try:
        server_relative = target.resolve().relative_to(server_cwd)
    except ValueError:
        return "error: apply_diff requires workdir to be under the llama-server working directory"
    return server_relative.as_posix()


def _rewrite_diff_paths(diff: str, path_map: dict[str, str]) -> str:
    lines: list[str] = []
    for line in diff.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        if body.startswith("diff --git "):
            parts = body.split()
            if len(parts) == 4:
                old = _strip_diff_prefix(parts[2])
                new = _strip_diff_prefix(parts[3])
                if old in path_map and new in path_map:
                    body = f"diff --git a/{path_map[old]} b/{path_map[new]}"
        elif body.startswith("--- "):
            body = _rewrite_diff_header(body, "---", "a", path_map)
        elif body.startswith("+++ "):
            body = _rewrite_diff_header(body, "+++", "b", path_map)
        lines.append(body + newline)
    return "".join(lines)


def _rewrite_diff_header(line: str, marker: str, prefix: str, path_map: dict[str, str]) -> str:
    rest = line[len(marker) + 1 :]
    path_part, separator, suffix = rest.partition("\t")
    stripped = _strip_diff_prefix(path_part)
    if stripped == "/dev/null":
        return line
    replacement = path_map.get(stripped)
    if not replacement:
        return line
    return f"{marker} {prefix}/{replacement}{separator}{suffix}"
