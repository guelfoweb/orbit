from __future__ import annotations

from .loop import ToolCallRecord, extract_path


def edited_paths(history: list[ToolCallRecord]) -> set[str]:
    paths: set[str] = set()
    for item in history:
        if item.name not in {"write_file", "append_file", "replace_in_file", "make_directory", "delete_path"}:
            continue
        path = extract_path(item, prefix=f"{item.name}.path=")
        if path:
            paths.add(path)
    return paths


def file_edit_completion_message(paths: set[str]) -> str:
    ordered = sorted(paths)
    if not ordered:
        return "Completed the requested file update."
    if len(ordered) == 1:
        return f"Updated `{ordered[0]}` and added the requested follow-up section."
    joined = ", ".join(f"`{path}`" for path in ordered)
    return f"Updated these files: {joined}."


def completed_edit_paths(history: list[ToolCallRecord]) -> set[str]:
    writes: set[str] = set()
    appends: set[str] = set()
    replacements: set[str] = set()
    directories: set[str] = set()
    deletions: set[str] = set()
    for item in history:
        if item.name == "write_file":
            path = extract_path(item, prefix="write_file.path=")
            if path:
                writes.add(path)
        elif item.name == "append_file":
            path = extract_path(item, prefix="append_file.path=")
            if path:
                appends.add(path)
        elif item.name == "replace_in_file":
            path = extract_path(item, prefix="replace_in_file.path=")
            if path:
                replacements.add(path)
        elif item.name == "make_directory":
            path = extract_path(item, prefix="make_directory.path=")
            if path:
                directories.add(path)
        elif item.name == "delete_path":
            path = extract_path(item, prefix="delete_path.path=")
            if path:
                deletions.add(path)
    return (writes & appends) | replacements | directories | deletions


def target_edit_paths(tool_calls: list[dict[str, object]]) -> set[str]:
    paths: set[str] = set()
    for call in tool_calls:
        fn = call.get("function", {}) or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        arguments = fn.get("arguments")
        if not isinstance(arguments, dict):
            continue
        if name not in {"write_file", "append_file", "replace_in_file", "make_directory", "delete_path"}:
            continue
        path = arguments.get("path")
        if isinstance(path, str) and path.strip():
            paths.add(path.strip())
    return paths


def read_target_paths(tool_calls: list[dict[str, object]]) -> set[str]:
    paths: set[str] = set()
    for call in tool_calls:
        fn = call.get("function", {}) or {}
        if not isinstance(fn, dict):
            continue
        if fn.get("name") != "read_file":
            continue
        arguments = fn.get("arguments")
        if not isinstance(arguments, dict):
            continue
        path = arguments.get("path")
        if isinstance(path, str) and path.strip():
            paths.add(path.strip())
    return paths
