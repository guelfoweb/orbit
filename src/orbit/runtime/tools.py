from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from orbit.runtime.web import (
    DEFAULT_FETCH_CHUNK_CHARS,
    DEFAULT_SEARCH_RESULTS,
    MAX_FETCH_CHUNK_CALLS_PER_TURN,
    MAX_FETCH_CHUNK_CHARS,
    MAX_SEARCH_RESULTS,
    fetch_url,
    search_web,
)


MAX_LIST_ITEMS = 200
MAX_READ_BYTES = 256 * 1024
MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 64 * 1024
MAX_APPEND_CHARS = 16 * 1024
MAX_REPLACE_CHARS = 16 * 1024
MAX_TEXT_FILE_BYTES_AFTER_APPEND = 512 * 1024
MAX_TEXT_FILE_BYTES_AFTER_REPLACE = 512 * 1024
MAX_CHUNK_FILE_BYTES = 1024 * 1024
DEFAULT_CHUNK_CHARS = 12_000
MAX_CHUNK_CHARS = 24_000
MAX_CHUNK_CALLS_PER_TURN = 3

TEXT_EXTENSIONS = {
    ".bat",
    ".bib",
    ".c",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".dart",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".log",
    ".lua",
    ".md",
    ".php",
    ".properties",
    ".ps1",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

BINARY_OR_SPECIAL_EXTENSIONS = {
    ".7z",
    ".bmp",
    ".doc",
    ".docx",
    ".flac",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".ogg",
    ".pdf",
    ".png",
    ".tar",
    ".wav",
    ".webp",
    ".zip",
}


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str


TOOL_NAMES = (
    "list_files",
    "read_file",
    "stat_path",
    "make_directory",
    "delete_path",
    "fetch_url",
    "search_web",
    "write_file",
    "append_file",
    "replace_in_file",
)


def tool_names() -> tuple[str, ...]:
    return TOOL_NAMES


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List workdir files/directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory. Defaults to '.'.",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text/source file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file.",
                        },
                        "chunk_index": {
                            "type": "integer",
                            "description": "Zero-based chunk for large files.",
                        },
                        "chunk_chars": {
                            "type": "integer",
                            "description": f"Chunk size. Default {DEFAULT_CHUNK_CHARS}, max {MAX_CHUNK_CHARS}.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stat_path",
                "description": "Inspect bounded metadata for a workdir path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "make_directory",
                "description": "Create one directory inside the workdir, including missing parent directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative target directory path.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_path",
                "description": (
                    "Delete one file or directory inside the workdir. "
                    "Non-empty directories require recursive=true."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative target file or directory path.",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Required only to delete a non-empty directory.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch an explicit http/https URL and return bounded readable text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Explicit http/https URL.",
                        },
                        "chunk_index": {
                            "type": "integer",
                            "description": "Zero-based chunk for long fetched pages.",
                        },
                        "chunk_chars": {
                            "type": "integer",
                            "description": f"Chunk size. Default {DEFAULT_FETCH_CHUNK_CHARS}, max {MAX_FETCH_CHUNK_CHARS}.",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web and return bounded structured results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": f"Number of results. Default {DEFAULT_SEARCH_RESULTS}, max {MAX_SEARCH_RESULTS}.",
                        },
                        "site": {
                            "type": "string",
                            "description": "Optional bare domain filter, for example example.com. Do not pass full URLs.",
                        },
                        "timelimit": {
                            "type": "string",
                            "description": "Optional time filter: d, w, m, or y.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a new UTF-8 text/source file at an explicit workdir path. Never overwrites.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative target file path. Parent directory must already exist.",
                        },
                        "content": {
                            "type": "string",
                            "description": f"UTF-8 text content, max {MAX_WRITE_CHARS} characters.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append UTF-8 text to an existing text/source file at an explicit workdir path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative existing file path.",
                        },
                        "content": {
                            "type": "string",
                            "description": f"UTF-8 text to append, max {MAX_APPEND_CHARS} characters.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": "Replace one unique exact UTF-8 text fragment in an existing text/source file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative existing file path.",
                        },
                        "old": {
                            "type": "string",
                            "description": f"Exact text to replace, max {MAX_REPLACE_CHARS} characters.",
                        },
                        "new": {
                            "type": "string",
                            "description": f"Replacement UTF-8 text, max {MAX_REPLACE_CHARS} characters.",
                        },
                    },
                    "required": ["path", "old", "new"],
                },
            },
        },
    ]


def execute_tool(
    name: str,
    arguments: str | dict[str, Any],
    *,
    workdir: Path,
    chunk_budget: dict[str, int] | None = None,
) -> ToolResult:
    if name not in TOOL_NAMES:
        return ToolResult(name=name, content=f"error: unknown tool: {name}")
    parsed = _parse_arguments(arguments)
    if isinstance(parsed, str):
        return ToolResult(name=name, content=parsed)
    if name == "list_files":
        return ToolResult(name=name, content=_list_files(parsed.get("path", "."), workdir=workdir))
    if name == "stat_path":
        return ToolResult(name=name, content=_stat_path(parsed.get("path"), workdir=workdir))
    if name == "make_directory":
        return ToolResult(name=name, content=_make_directory(parsed.get("path"), workdir=workdir))
    if name == "delete_path":
        return ToolResult(name=name, content=_delete_path(parsed.get("path"), parsed.get("recursive", False), workdir=workdir))
    if name == "fetch_url":
        if "chunk_index" in parsed and chunk_budget is not None:
            used = chunk_budget.get("fetch_url_chunks", 0)
            if used >= MAX_FETCH_CHUNK_CALLS_PER_TURN:
                return ToolResult(
                    name=name,
                    content=(
                        f"error: fetch_url chunk budget exceeded for this turn: max {MAX_FETCH_CHUNK_CALLS_PER_TURN} chunks. "
                        "Stop and explain that complete analysis requires continuing in later turns."
                    ),
                )
            chunk_budget["fetch_url_chunks"] = used + 1
        return ToolResult(
            name=name,
            content=fetch_url(
                parsed.get("url"),
                chunk_index=parsed.get("chunk_index"),
                chunk_chars=parsed.get("chunk_chars", DEFAULT_FETCH_CHUNK_CHARS),
            ),
        )
    if name == "search_web":
        return ToolResult(
            name=name,
            content=search_web(
                parsed.get("query"),
                max_results=parsed.get("max_results", DEFAULT_SEARCH_RESULTS),
                site=parsed.get("site"),
                timelimit=parsed.get("timelimit"),
            ),
        )
    if name == "write_file":
        return ToolResult(name=name, content=_write_file(parsed.get("path"), parsed.get("content"), workdir=workdir))
    if name == "append_file":
        return ToolResult(name=name, content=_append_file(parsed.get("path"), parsed.get("content"), workdir=workdir))
    if name == "replace_in_file":
        return ToolResult(
            name=name,
            content=_replace_in_file(parsed.get("path"), parsed.get("old"), parsed.get("new"), workdir=workdir),
        )
    if "chunk_index" in parsed and chunk_budget is not None:
        used = chunk_budget.get("read_file_chunks", 0)
        if used >= MAX_CHUNK_CALLS_PER_TURN:
            return ToolResult(
                name=name,
                content=(
                    f"error: chunk budget exceeded for this turn: max {MAX_CHUNK_CALLS_PER_TURN} chunks. "
                    "Stop and explain that complete analysis requires continuing in later turns."
                ),
            )
        chunk_budget["read_file_chunks"] = used + 1
    return ToolResult(name=name, content=_read_file(parsed.get("path"), arguments=parsed, workdir=workdir))


def _parse_arguments(arguments: str | dict[str, Any]) -> dict[str, Any] | str:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return f"error: invalid JSON tool arguments: {exc}"
    if not isinstance(parsed, dict):
        return "error: tool arguments must be a JSON object"
    return parsed


def _list_files(path: Any, *, workdir: Path) -> str:
    if not isinstance(path, str):
        return "error: path must be a string"
    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_dir():
        return f"error: path is not a directory: {path}"

    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    names = [f"{item.name}/" if item.is_dir() else item.name for item in entries[:MAX_LIST_ITEMS]]
    if len(entries) > MAX_LIST_ITEMS:
        names.append(f"... truncated, {len(entries) - MAX_LIST_ITEMS} more entries")
    return "\n".join(names) if names else "(empty directory)"


def _stat_path(path: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return "\n".join(
            [
                f"path: {path}",
                "exists: false",
            ]
        )
    try:
        stat = target.stat()
    except OSError as exc:
        return f"error: cannot stat path: {exc}"

    if target.is_dir():
        path_type = "directory"
    elif target.is_file():
        path_type = "file"
    else:
        path_type = "other"

    modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    return "\n".join(
        [
            f"path: {path}",
            "exists: true",
            f"type: {path_type}",
            f"size_bytes: {stat.st_size}",
            f"modified: {modified}",
        ]
    )


def _make_directory(path: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    root = workdir.expanduser().resolve()
    if target == root:
        return "error: refusing to create the workdir root"
    if target.exists():
        if target.is_dir():
            return f"error: directory already exists: {path}"
        return f"error: path already exists and is not a directory: {path}"
    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        return f"error: cannot create directory: {exc}"
    return "\n".join(
        [
            f"path: {path}",
            "created: true",
            "type: directory",
        ]
    )


def _delete_path(path: Any, recursive: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(recursive, bool):
        return "error: recursive must be a boolean"
    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    root = workdir.expanduser().resolve()
    if target == root:
        return "error: refusing to delete the workdir root"
    if not target.exists() and not target.is_symlink():
        return f"error: path not found: {path}"
    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
            path_type = "file"
        elif target.is_dir():
            if any(target.iterdir()) and not recursive:
                return f"error: directory is not empty: {path}. Use recursive=true only if you really want to remove it."
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
            path_type = "directory"
        else:
            return f"error: unsupported path type: {path}"
    except OSError as exc:
        return f"error: cannot delete path: {exc}"
    return "\n".join(
        [
            f"path: {path}",
            "deleted: true",
            f"type: {path_type}",
        ]
    )


def _read_file(path: Any, *, arguments: dict[str, Any], workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_file():
        return f"error: path is not a file: {path}"

    suffix = target.suffix.lower()
    if suffix == ".pdf":
        return "error: read_file supports UTF-8 text/code files only; PDF requires read_pdf, which is not available yet"
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: read_file supports UTF-8 text/code files only; unsupported file type: {suffix}"

    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        if "chunk_index" in arguments:
            return _read_chunk(
                path,
                chunk_index=arguments.get("chunk_index"),
                chunk_chars=arguments.get("chunk_chars", DEFAULT_CHUNK_CHARS),
                workdir=workdir,
            )
        return (
            f"error: file too large for complete read_file: {size} bytes, max {MAX_READ_BYTES}. "
            "Use read_file with chunk_index for explicit chunked reading."
        )
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for read_file: {suffix}"

    raw = target.read_bytes()
    if b"\x00" in raw:
        return "error: file appears to be binary and cannot be read as UTF-8 text"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: file is not valid UTF-8 text"

    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n... truncated, {len(text) - MAX_READ_CHARS} more characters"
    return text if text else "(empty file)"


def _write_file(path: Any, content: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(content, str):
        return "error: content must be a string"
    if len(content) > MAX_WRITE_CHARS:
        return f"error: content too large for write_file: {len(content)} chars, max {MAX_WRITE_CHARS}"
    if "\x00" in content:
        return "error: content appears to be binary and cannot be written as UTF-8 text"

    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if target.exists():
        return f"error: refusing to overwrite existing path: {path}"
    if not target.parent.exists():
        return f"error: parent directory does not exist: {target.parent.relative_to(workdir.expanduser().resolve())}"
    if not target.parent.is_dir():
        return f"error: parent path is not a directory: {target.parent.relative_to(workdir.expanduser().resolve())}"

    suffix = target.suffix.lower()
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: write_file supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for write_file: {suffix}"

    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"error: cannot write file: {exc}"
    return "\n".join(
        [
            f"path: {path}",
            "created: true",
            f"chars: {len(content)}",
            f"bytes: {len(content.encode('utf-8'))}",
        ]
    )


def _append_file(path: Any, content: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(content, str):
        return "error: content must be a string"
    if len(content) > MAX_APPEND_CHARS:
        return f"error: content too large for append_file: {len(content)} chars, max {MAX_APPEND_CHARS}"
    if "\x00" in content:
        return "error: content appears to be binary and cannot be appended as UTF-8 text"

    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}. Use write_file to create a new file."
    if not target.is_file():
        return f"error: path is not a file: {path}"

    suffix = target.suffix.lower()
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: append_file supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for append_file: {suffix}"

    current_size = target.stat().st_size
    append_bytes = len(content.encode("utf-8"))
    if current_size + append_bytes > MAX_TEXT_FILE_BYTES_AFTER_APPEND:
        return (
            f"error: append would make file too large: {current_size + append_bytes} bytes, "
            f"max {MAX_TEXT_FILE_BYTES_AFTER_APPEND}"
        )
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"error: cannot read existing file before append: {exc}"
    if b"\x00" in raw:
        return "error: existing file appears to be binary and cannot be appended as UTF-8 text"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: existing file is not valid UTF-8 text"

    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        return f"error: cannot append file: {exc}"
    return "\n".join(
        [
            f"path: {path}",
            "appended: true",
            f"chars_added: {len(content)}",
            f"bytes_added: {append_bytes}",
            f"bytes_total: {current_size + append_bytes}",
        ]
    )


def _replace_in_file(path: Any, old: Any, new: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(old, str) or not old:
        return "error: old must be a non-empty string"
    if not isinstance(new, str):
        return "error: new must be a string"
    if len(old) > MAX_REPLACE_CHARS:
        return f"error: old text too large for replace_in_file: {len(old)} chars, max {MAX_REPLACE_CHARS}"
    if len(new) > MAX_REPLACE_CHARS:
        return f"error: new text too large for replace_in_file: {len(new)} chars, max {MAX_REPLACE_CHARS}"
    if "\x00" in old or "\x00" in new:
        return "error: replacement text appears to be binary and cannot be used as UTF-8 text"

    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_file():
        return f"error: path is not a file: {path}"

    suffix = target.suffix.lower()
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: replace_in_file supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for replace_in_file: {suffix}"
    if target.stat().st_size > MAX_TEXT_FILE_BYTES_AFTER_REPLACE:
        return f"error: file too large for replace_in_file: {target.stat().st_size} bytes, max {MAX_TEXT_FILE_BYTES_AFTER_REPLACE}"

    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"error: cannot read file before replacement: {exc}"
    if b"\x00" in raw:
        return "error: existing file appears to be binary and cannot be edited as UTF-8 text"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: existing file is not valid UTF-8 text"

    matches = text.count(old)
    if matches == 0:
        return "error: old text not found"
    if matches > 1:
        return f"error: old text is ambiguous: {matches} matches"
    updated = text.replace(old, new, 1)
    updated_bytes = len(updated.encode("utf-8"))
    if updated_bytes > MAX_TEXT_FILE_BYTES_AFTER_REPLACE:
        return f"error: replacement would make file too large: {updated_bytes} bytes, max {MAX_TEXT_FILE_BYTES_AFTER_REPLACE}"
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return f"error: cannot replace in file: {exc}"
    return "\n".join(
        [
            f"path: {path}",
            "replaced: true",
            "matches: 1",
            f"bytes_total: {updated_bytes}",
        ]
    )


def _read_chunk(path: Any, *, chunk_index: Any, chunk_chars: Any, workdir: Path) -> str:
    if not isinstance(chunk_index, int) or chunk_index < 0:
        return "error: chunk_index must be a non-negative integer"
    if not isinstance(chunk_chars, int) or chunk_chars <= 0:
        return "error: chunk_chars must be a positive integer"
    if chunk_chars > MAX_CHUNK_CHARS:
        return f"error: chunk_chars too large: {chunk_chars}, max {MAX_CHUNK_CHARS}"

    validation = _load_text_file(path, workdir=workdir, max_bytes=MAX_CHUNK_FILE_BYTES)
    if isinstance(validation, str):
        return validation
    target, text = validation

    total_chunks = max(1, (len(text) + chunk_chars - 1) // chunk_chars)
    if chunk_index >= total_chunks:
        return f"error: chunk_index out of range: {chunk_index}, total_chunks {total_chunks}"

    start = chunk_index * chunk_chars
    end = min(start + chunk_chars, len(text))
    chunk = text[start:end]
    return "\n".join(
        [
            f"path: {target.name}",
            f"chunk_index: {chunk_index}",
            f"total_chunks: {total_chunks}",
            f"chars: {start}-{end} of {len(text)}",
            "content:",
            chunk,
        ]
    )


def _load_text_file(path: Any, *, workdir: Path, max_bytes: int) -> tuple[Path, str] | str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = _resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_file():
        return f"error: path is not a file: {path}"

    suffix = target.suffix.lower()
    if suffix == ".pdf":
        return "error: read_file chunk mode supports UTF-8 text/code files only; PDF requires read_pdf, which is not available yet"
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: read_file chunk mode supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for read_file chunk mode: {suffix}"

    size = target.stat().st_size
    if size > max_bytes:
        return f"error: file too large for read_file chunk mode: {size} bytes, max {max_bytes}"

    raw = target.read_bytes()
    if b"\x00" in raw:
        return "error: file appears to be binary and cannot be read as UTF-8 text"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: file is not valid UTF-8 text"
    return target, text


def _resolve_inside_workdir(path: str, *, workdir: Path) -> Path | str:
    root = workdir.expanduser().resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return "error: path escapes workdir"
    return target
