from __future__ import annotations

import re


LARGE_TOOL_RESULT_CHARS = 10_000
SHELL_FULL_CONTRACT_ERROR_PREFIX = "error: shell-full analysis requests require content/source/string evidence"
DISPLAY_TOOL_NAMES = {
    "exec_shell_full_command": "exec",
}


def display_tool_name(name: str) -> str:
    return DISPLAY_TOOL_NAMES.get(name, name)


def format_tool_call_event(name: str, args: str) -> str:
    return f"{display_tool_name(name)} {args}"


def format_tool_result_event(name: str, chars: int, source: str | None = None, content: str | None = None) -> str:
    del source
    suffix_parts: list[str] = []
    if _is_rejected_contract_result(content):
        suffix_parts.append("rejected")
    if chars >= LARGE_TOOL_RESULT_CHARS:
        suffix_parts.append("large context")
    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    chunk = _chunk_label(content)
    if chunk:
        return f" └ {chunk} {chars} chars -> model{suffix}"
    return f" └ {chars} chars -> model{suffix}"


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
