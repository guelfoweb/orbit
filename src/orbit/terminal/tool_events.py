from __future__ import annotations


LARGE_TOOL_RESULT_CHARS = 10_000


def format_tool_result_event(name: str, chars: int, source: str | None = None) -> str:
    suffix = " | large context" if chars >= LARGE_TOOL_RESULT_CHARS else ""
    source_part = f" | src: {source}" if source else ""
    return f" └ {name} {chars} chars{source_part}{suffix}"
