from __future__ import annotations

import json
from typing import Any


class _DuplicateToolArgument(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateToolArgument(key)
        result[key] = value
    return result


def parse_tool_arguments(
    arguments: str | dict[str, Any],
    *,
    reject_duplicate_keys: bool = False,
) -> dict[str, Any] | str:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(
            arguments,
            object_pairs_hook=_unique_json_object if reject_duplicate_keys else None,
        )
    except _DuplicateToolArgument as exc:
        return f"error: duplicate JSON tool argument: {exc}"
    except json.JSONDecodeError as exc:
        return f"error: invalid JSON tool arguments: {exc}"
    if not isinstance(parsed, dict):
        return "error: tool arguments must be a JSON object"
    return parsed


def parse_tool_arguments_or_empty(arguments: Any) -> dict[str, Any]:
    parsed = parse_tool_arguments(arguments)
    return parsed if isinstance(parsed, dict) else {}
