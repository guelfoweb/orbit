from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_MAX_ENTRIES = 200
DEFAULT_RECURSIVE_MAX_DEPTH = 2
MAX_ENTRIES_LIMIT = 1000
MAX_DEPTH_LIMIT = 20


def list_directory_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "Return a compact, structured directory listing. Use this for directory/file listing requests "
                "instead of noisy shell commands like ls -R, find, or tree. This tool does not read file contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list, relative to the current workdir unless absolute.",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list nested entries.",
                        "default": False,
                    },
                    "max_depth": {
                        "type": ["integer", "null"],
                        "description": "Maximum recursive depth. Defaults to 2 when recursive is true.",
                        "default": None,
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum entries to return before truncating.",
                        "default": DEFAULT_MAX_ENTRIES,
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include dotfiles and dot-directories.",
                        "default": False,
                    },
                    "dirs_first": {
                        "type": "boolean",
                        "description": "Sort directories before files at each level.",
                        "default": True,
                    },
                    "files_only": {
                        "type": "boolean",
                        "description": "Return only files.",
                        "default": False,
                    },
                    "dirs_only": {
                        "type": "boolean",
                        "description": "Return only directories.",
                        "default": False,
                    },
                },
                "additionalProperties": False,
            },
        },
    }


def execute_list_directory(arguments: dict[str, Any], *, workdir: Path) -> str:
    path = _string_arg(arguments.get("path"), default=".")
    recursive = _bool_arg(arguments.get("recursive"), default=False)
    max_depth = _max_depth_arg(arguments.get("max_depth"), recursive=recursive)
    max_entries = _int_arg(arguments.get("max_entries"), default=DEFAULT_MAX_ENTRIES, minimum=1, maximum=MAX_ENTRIES_LIMIT)
    include_hidden = _bool_arg(arguments.get("include_hidden"), default=False)
    dirs_first = _bool_arg(arguments.get("dirs_first"), default=True)
    files_only = _bool_arg(arguments.get("files_only"), default=False)
    dirs_only = _bool_arg(arguments.get("dirs_only"), default=False)
    if files_only and dirs_only:
        return _error_result(path=path, status="invalid_arguments", message="files_only and dirs_only cannot both be true")

    root = workdir.resolve()
    target = _resolve_target(path, root=root)
    if target is None:
        return _error_result(path=path, status="path_outside_workdir", message="path is outside the current workdir")
    if not target.exists():
        return _error_result(path=path, status="not_found", message="path does not exist")
    if not target.is_dir():
        return _error_result(path=path, status="not_directory", message="path is not a directory")

    entries: list[str] = []
    total_seen = 0
    truncated = False
    for entry in _iter_entries(
        target,
        root=root,
        recursive=recursive,
        max_depth=max_depth,
        include_hidden=include_hidden,
        dirs_first=dirs_first,
        files_only=files_only,
        dirs_only=dirs_only,
    ):
        total_seen += 1
        if len(entries) >= max_entries:
            truncated = True
            continue
        entries.append(entry)

    rel_path = _display_path(target, root=root)
    lines = [
        (
            f"directory_listing: path={rel_path} recursive={_bool_text(recursive)} "
            f"max_depth={max_depth if recursive else 'null'} shown={len(entries)} "
            f"total_seen={total_seen} truncated={_bool_text(truncated)}"
        )
    ]
    lines.extend(entries)
    return "\n".join(lines)


def _iter_entries(
    directory: Path,
    *,
    root: Path,
    recursive: bool,
    max_depth: int,
    include_hidden: bool,
    dirs_first: bool,
    files_only: bool,
    dirs_only: bool,
):
    def walk(current: Path, depth: int):
        children = _sorted_children(current, dirs_first=dirs_first)
        for child in children:
            if not include_hidden and child.name.startswith("."):
                continue
            entry_type = _entry_type(child)
            include = not ((files_only and entry_type != "file") or (dirs_only and entry_type != "dir"))
            if include:
                yield _format_entry(child, root=root, entry_type=entry_type)
            if recursive and depth < max_depth and entry_type == "dir" and not child.is_symlink():
                yield from walk(child, depth + 1)

    yield from walk(directory, 1)


def _sorted_children(path: Path, *, dirs_first: bool) -> list[Path]:
    try:
        children = list(path.iterdir())
    except OSError:
        return []
    return sorted(children, key=lambda item: (_type_sort(item, dirs_first=dirs_first), item.name.lower(), item.name))


def _type_sort(path: Path, *, dirs_first: bool) -> int:
    if not dirs_first:
        return 0
    return 0 if path.is_dir() and not path.is_symlink() else 1


def _format_entry(path: Path, *, root: Path, entry_type: str) -> str:
    rel = _display_path(path, root=root)
    suffix = "/" if entry_type == "dir" and not rel.endswith("/") else ""
    line = f"[{entry_type}] {rel}{suffix}"
    if entry_type == "symlink":
        try:
            target = os.readlink(path)
        except OSError:
            target = "unreadable"
        line += f" -> {target}"
    elif entry_type == "file":
        try:
            line += f" ({path.stat().st_size} B)"
        except OSError:
            pass
    return line


def _entry_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "dir"
    if path.is_file():
        return "file"
    return "other"


def _resolve_target(path: str, *, root: Path) -> Path | None:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _display_path(path: Path, *, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        try:
            rel = path.resolve(strict=False).relative_to(root)
        except ValueError:
            return str(path)
    value = rel.as_posix()
    return "." if value == "." else value


def _error_result(*, path: str, status: str, message: str) -> str:
    return f"directory_listing: error=true status={status} path={path}\nerror: {message}"


def _string_arg(value: object, *, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _bool_arg(value: object, *, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _int_arg(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _max_depth_arg(value: object, *, recursive: bool) -> int:
    if not recursive:
        return 1
    return _int_arg(value, default=DEFAULT_RECURSIVE_MAX_DEPTH, minimum=1, maximum=MAX_DEPTH_LIMIT)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
