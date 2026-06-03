from __future__ import annotations

from .loop import (
    ToolCallRecord,
    extract_path,
    extract_read_path,
    is_trivial_python_entrypoint,
    sampled_read_paths,
)


def repeated_tool_retry_prompt(record: ToolCallRecord, state: object) -> str:
    base = (
        "You just repeated a tool call that was already attempted. "
        "Do not repeat the same tool call with the same arguments. "
        "Use the existing tool result or answer directly."
    )
    tool_history = getattr(state, "tool_history", [])
    if record.name == "list_files":
        return (
            f"{base} Repeated call: {record.detail or record.name}. "
            "Use the paths already returned by list_files to choose the next file or directory. "
            "For project analysis, do one recursive listing and then reuse those exact relative paths. "
            "Do not call list_files again for the same location. Pick one concrete returned path for read_file, or answer from the current evidence if there is no clear candidate."
        )
    if record.name == "read_file":
        sampled_paths = sampled_read_paths(tool_history)
        if is_trivial_python_entrypoint(record.detail):
            return (
                f"{base} Repeated call: {record.detail or record.name}. "
                "For project analysis, stop spending turns on __init__.py or __main__.py unless they contain real logic. "
                "Use the paths you already discovered to open substantive implementation files such as agent.py, runtime.py, cli.py, registry.py, filesystem.py, shell.py, web.py, or the matching tests."
            )
        if sampled_paths:
            repeated_path = extract_read_path(record)
            return (
                f"{base} Repeated call: {record.detail or record.name}. "
                f"You already sampled these files: {', '.join(sampled_paths)}. "
                "Stop expanding the file list and synthesize the architecture or the findings from the evidence already collected. "
                f"Do not keep drilling into the same file ({repeated_path}) once you already have a representative sample, unless the user explicitly asked for a deep dive into that file."
            )
        return (
            f"{base} Repeated call: {record.detail or record.name}. "
            "If the previous read_file path was guessed or failed, call list_files first to discover the real path instead of guessing again. "
            "Reuse the exact relative path already discovered from list_files. "
            "If you need more file content, continue with a new bounded chunk using next_start_line instead. "
            "If you already sampled several key project files, stop exploring exhaustively and answer with the current architectural summary or findings."
        )
    if record.name == "fetch_url":
        return (
            f"{base} Repeated call: {record.detail or record.name}. "
            "fetch_url is only for a concrete known page URL. "
            "Do not guess Google, Bing, Wikipedia, or other search/result URLs from a name alone. "
            "If the user asked for a general online search, use search_web first and only then open promising result URLs with fetch_url."
        )
    if record.name == "search_web":
        return (
            f"{base} Repeated call: {record.detail or record.name}. "
            "Reuse the structured search results you already have. "
            "Open one or two promising result URLs with fetch_url, refine the query meaningfully, or answer from the evidence already collected."
        )
    if record.name in {"write_file", "append_file", "replace_in_file"}:
        repeated_path = extract_path(record, prefix=f"{record.name}.path=")
        return (
            f"{base} Repeated call: {record.detail or record.name}. "
            f"You already updated {repeated_path}. "
            "Do not keep editing the same file unless the user explicitly asked for another change. "
            "Answer now with a short confirmation of the completed edit."
        )
    return f"{base} Repeated call: {record.detail or record.name}."
