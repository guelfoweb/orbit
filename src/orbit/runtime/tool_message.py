from __future__ import annotations

import json
from typing import Any

from orbit.backend.base import Message
from orbit.runtime.evidence import EvidenceStore, tool_evidence_ref
from orbit.runtime.tool_calls import tool_call_id
from orbit.runtime.tools import ToolResult


def assistant_tool_call_message(content: str, tool_calls: list[dict[str, object]]) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = _safe_tool_calls_for_history(tool_calls)
    return message


def tool_result_message(
    tool_call: dict[str, object],
    tool_result: ToolResult,
    *,
    evidence_store: EvidenceStore | None = None,
    metadata: dict[str, object] | None = None,
) -> Message:
    content = tool_result.content
    evidence_id = None
    evidence_user_turn_id = None
    if evidence_store is not None and tool_result.content:
        evidence_metadata = dict(metadata or _tool_call_metadata(tool_call))
        evidence_metadata.setdefault("tool_call_id", tool_call_id(tool_call))
        record = evidence_store.add(tool_result.name, tool_result.content, metadata=evidence_metadata)
        content = tool_evidence_ref(record)
        evidence_id = record.evidence_id
        evidence_user_turn_id = record.user_turn_id
    return {
        "role": "tool",
        "tool_call_id": tool_call_id(tool_call),
        "name": tool_result.name,
        "content": content,
        **({"evidence_id": evidence_id} if evidence_id else {}),
        **({"user_turn_id": evidence_user_turn_id} if evidence_user_turn_id else {}),
    }


def _safe_tool_calls_for_history(tool_calls: list[dict[str, object]]) -> list[dict[str, Any]]:
    correlation_ids = [tool_call_id(tool_call) for tool_call in tool_calls]
    if len(tool_calls) > 1:
        explicit_ids = [tool_call.get("id") for tool_call in tool_calls]
        if any(not isinstance(value, str) or not value for value in explicit_ids):
            raise ValueError("multiple tool calls require explicit identifiers")
        if len(set(correlation_ids)) != len(correlation_ids):
            raise ValueError("multiple tool calls require unique identifiers")
    return [
        _safe_tool_call_for_history(tool_call, correlation_id=correlation_id)
        for tool_call, correlation_id in zip(tool_calls, correlation_ids, strict=True)
    ]


def _safe_tool_call_for_history(tool_call: dict[str, object], *, correlation_id: str) -> dict[str, Any]:
    sanitized = dict(tool_call)
    sanitized["id"] = correlation_id
    function = sanitized.get("function")
    if not isinstance(function, dict):
        return sanitized
    safe_function = dict(function)
    safe_function["arguments"] = _safe_arguments_json(safe_function.get("arguments"))
    sanitized["function"] = safe_function
    return sanitized


def _safe_arguments_json(arguments: object) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"invalid_arguments": arguments[:200]}
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        parsed = {}
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _tool_call_metadata(tool_call: dict[str, object]) -> dict[str, object]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        return {}
    command = parsed.get("command")
    query = parsed.get("query")
    metadata: dict[str, object] = {}
    if isinstance(command, str):
        metadata["command"] = command
    if isinstance(query, str):
        metadata["query"] = query
    return metadata
