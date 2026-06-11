from __future__ import annotations

import json
from typing import Any


def parse_tool_arguments(arguments: str | dict[str, Any]) -> dict[str, Any] | str:
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


def parse_tool_arguments_or_empty(arguments: Any) -> dict[str, Any]:
    parsed = parse_tool_arguments(arguments)
    return parsed if isinstance(parsed, dict) else {}
