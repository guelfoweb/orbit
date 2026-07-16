from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Any, Iterable
from urllib.parse import urlparse

from orbit.runtime.directory_listing import MAX_DEPTH_LIMIT, MAX_ENTRIES_LIMIT
from orbit.runtime.path_guardrails import resolve_inside_workdir
from orbit.runtime.shell_guardrails import (
    MAX_SHELL_OUTPUT_BYTES,
    MAX_SHELL_TIMEOUT,
    validate_read_only_shell_mutation,
    validate_shell_full_contract,
)
from orbit.runtime.web import MAX_FETCH_MAX_BYTES, MAX_FETCH_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CanonicalToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ContractStageOutcome:
    accepted: bool
    code: str | None = None
    path: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class CanonicalToolDecision:
    normalized_call: CanonicalToolCall | None
    schema_outcome: ContractStageOutcome
    policy_outcome: ContractStageOutcome
    permission_outcome: ContractStageOutcome
    operational_limit_outcome: ContractStageOutcome
    terminal_decision: str
    rejection_code: str | None

    @property
    def accepted(self) -> bool:
        return self.terminal_decision == "accepted"


class DuplicateToolArgumentKey(ValueError):
    pass


def validate_canonical_tool_call(
    name: object,
    arguments: object,
    *,
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
) -> CanonicalToolDecision:
    definitions = _definitions_by_name(tool_definitions)
    allowed = frozenset(allowed_tool_names)
    neutral = ContractStageOutcome(True)

    if not isinstance(name, str) or not name:
        return _rejected("rejected_schema", "invalid_tool_name", schema=ContractStageOutcome(False, "invalid_tool_name", "name"))
    permission = ContractStageOutcome(name in allowed, None if name in allowed else "tool_not_enabled", None if name in allowed else "name")
    if not permission.accepted:
        return CanonicalToolDecision(None, neutral, neutral, permission, neutral, "rejected_permission", permission.code)
    if name not in definitions:
        return _rejected("rejected_schema", "unknown_tool", schema=ContractStageOutcome(False, "unknown_tool", "name"))
    parsed, parse_error = _parse_arguments(arguments)
    if parse_error is not None:
        terminal = "rejected_parse" if parse_error in {"invalid_json", "duplicate_key"} else "rejected_schema"
        return _rejected(terminal, parse_error, schema=ContractStageOutcome(False, parse_error, "arguments"))
    assert parsed is not None
    call = CanonicalToolCall(name, parsed)
    schema_error = validate_schema(parsed, definitions[name], path="arguments")
    if schema_error is not None:
        return CanonicalToolDecision(
            call,
            schema_error,
            neutral,
            neutral,
            neutral,
            "rejected_schema",
            schema_error.code,
        )
    policy, operational = _policy_and_operational(call, workdir=workdir, user_prompt=user_prompt)
    if not policy.accepted:
        terminal = "rejected_policy" if policy.code == "policy_read_only_mutation" else "rejected_guardrail"
        return CanonicalToolDecision(call, neutral, policy, permission, operational, terminal, policy.code)
    if not operational.accepted:
        return CanonicalToolDecision(call, neutral, policy, permission, operational, "rejected_guardrail", operational.code)
    return CanonicalToolDecision(call, neutral, policy, permission, operational, "accepted", None)


def validate_canonical_tool_call_payload(
    tool_call: object,
    *,
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
) -> CanonicalToolDecision:
    if not isinstance(tool_call, dict):
        return _rejected(
            "rejected_schema",
            "invalid_tool_call",
            schema=ContractStageOutcome(False, "invalid_tool_call", "tool_call"),
        )
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return _rejected(
            "rejected_schema",
            "invalid_tool_call",
            schema=ContractStageOutcome(False, "invalid_tool_call", "function"),
        )
    return validate_canonical_tool_call(
        function.get("name"),
        function.get("arguments", {}),
        tool_definitions=tool_definitions,
        allowed_tool_names=allowed_tool_names,
        workdir=workdir,
        user_prompt=user_prompt,
    )


def canonical_rejection_content(name: object, decision: CanonicalToolDecision) -> str:
    if decision.policy_outcome.message:
        return decision.policy_outcome.message
    label = name if isinstance(name, str) and name else "unknown"
    if decision.rejection_code == "tool_not_enabled":
        return f"error: tool not available for this turn: {label}"
    if decision.rejection_code == "unknown_tool":
        return f"error: unknown tool: {label}"
    return f"error: canonical tool call rejected: {decision.rejection_code}"


def validate_schema(value: Any, schema: dict[str, Any], *, path: str) -> ContractStageOutcome | None:
    expected = schema.get("type")
    if expected is not None and not _matches_type(value, expected):
        return ContractStageOutcome(False, "type_mismatch", path)
    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        required = required if isinstance(required, list) else []
        for key in required:
            if key not in value:
                return ContractStageOutcome(False, "missing_required", f"{path}.{key}")
        for key in value:
            if key not in properties:
                return ContractStageOutcome(False, "additional_property", f"{path}.<unknown>")
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                error = validate_schema(item, child_schema, path=f"{path}.{key}")
                if error is not None:
                    return error
    return None


def _parse_arguments(arguments: object) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(arguments, dict):
        return arguments, None
    if not isinstance(arguments, str) or not arguments.strip():
        return None, "arguments_not_object"
    try:
        parsed = json.loads(arguments, object_pairs_hook=_unique_object)
    except DuplicateToolArgumentKey:
        return None, "duplicate_key"
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "arguments_not_object"
    return parsed, None


def _policy_and_operational(
    call: CanonicalToolCall,
    *,
    workdir: Path,
    user_prompt: str | None,
) -> tuple[ContractStageOutcome, ContractStageOutcome]:
    neutral = ContractStageOutcome(True)
    name = call.name
    arguments = call.arguments
    if name == "exec_shell_full_command":
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return neutral, ContractStageOutcome(False, "empty_required_value", "arguments.command")
        if _has_unsafe_control(command):
            return neutral, ContractStageOutcome(False, "unsafe_control_character", "arguments.command")
        try:
            shlex.split(command)
        except ValueError:
            return neutral, ContractStageOutcome(False, "invalid_shell_syntax", "arguments.command")
        policy_error = validate_read_only_shell_mutation(arguments, user_prompt=user_prompt)
        if policy_error:
            return ContractStageOutcome(
                False,
                "policy_read_only_mutation",
                "arguments.command",
                policy_error,
            ), neutral
        contract_error = validate_shell_full_contract(arguments, user_prompt=user_prompt)
        if contract_error:
            return ContractStageOutcome(
                False,
                "policy_shell_contract",
                "arguments.command",
                contract_error,
            ), neutral
        limit = _integer_limit(arguments, "timeout", 1, MAX_SHELL_TIMEOUT)
        if limit is None:
            limit = _integer_limit(arguments, "max_output_size", 1, MAX_SHELL_OUTPUT_BYTES)
        return neutral, limit or neutral
    if name == "fetch_url":
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            return neutral, ContractStageOutcome(False, "empty_required_value", "arguments.url")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return neutral, ContractStageOutcome(False, "unsupported_url", "arguments.url")
        limit = _integer_limit(arguments, "timeout", 1, MAX_FETCH_TIMEOUT_SECONDS)
        if limit is None:
            limit = _integer_limit(arguments, "max_bytes", 1, MAX_FETCH_MAX_BYTES)
        return neutral, limit or neutral
    if name == "list_directory":
        path = arguments.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            return neutral, ContractStageOutcome(False, "empty_required_value", "arguments.path")
        if isinstance(resolve_inside_workdir(path, workdir=workdir), str):
            return neutral, ContractStageOutcome(False, "path_outside_workdir", "arguments.path")
        if arguments.get("files_only") is True and arguments.get("dirs_only") is True:
            return neutral, ContractStageOutcome(False, "contradictory_flags", "arguments")
        limit = _integer_limit(arguments, "max_depth", 1, MAX_DEPTH_LIMIT, allow_none=True)
        if limit is None:
            limit = _integer_limit(arguments, "max_entries", 1, MAX_ENTRIES_LIMIT)
        return neutral, limit or neutral
    return neutral, neutral


def _integer_limit(
    arguments: dict[str, Any],
    key: str,
    minimum: int,
    maximum: int,
    *,
    allow_none: bool = False,
) -> ContractStageOutcome | None:
    if key not in arguments or (allow_none and arguments[key] is None):
        return None
    value = arguments[key]
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        return ContractStageOutcome(False, "limit_out_of_range", f"arguments.{key}")
    return None


def _matches_type(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    return any(
        (kind == "object" and isinstance(value, dict))
        or (kind == "array" and isinstance(value, list))
        or (kind == "string" and isinstance(value, str))
        or (kind == "boolean" and isinstance(value, bool))
        or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
        or (kind == "null" and value is None)
        for kind in types
    )


def _definitions_by_name(tool_definitions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for definition in tool_definitions:
        function = definition.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, dict):
            definitions[name] = parameters
    return definitions


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateToolArgumentKey(key)
        result[key] = value
    return result


def _has_unsafe_control(value: str) -> bool:
    return any(ord(character) < 32 and character not in {"\n", "\t"} for character in value)


def _rejected(
    terminal: str,
    code: str,
    *,
    schema: ContractStageOutcome,
) -> CanonicalToolDecision:
    neutral = ContractStageOutcome(True)
    return CanonicalToolDecision(None, schema, neutral, neutral, neutral, terminal, code)
