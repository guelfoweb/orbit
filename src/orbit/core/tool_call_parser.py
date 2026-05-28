from __future__ import annotations

import ast
import json
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_OPEN_RE = re.compile(r"```(?:json)?\s*(\{.*\})", re.DOTALL | re.IGNORECASE)
UNQUOTED_KEY_RE = re.compile(r'([{\s,])([A-Za-z_][A-Za-z0-9_-]*)\s*:')
UNQUOTED_NAME_VALUE_RE = re.compile(r'("name"\s*:\s*)([A-Za-z_][A-Za-z0-9_-]*)\b')
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
PLACEHOLDER_DIRECT_RE = re.compile(r'("(?:content|new)"\s*:\s*)"<tool_response\.content>"')
PLACEHOLDER_SUFFIX_RE = re.compile(
    r'("(?:content|new)"\s*:\s*)"(?P<prefix>(?:\\.|[^"\\])*)"\s*\+\s*"(?P<placeholder><[^">]*content[^">]*>)"',
    re.IGNORECASE,
)
PLACEHOLDER_PREFIX_RE = re.compile(
    r'("(?:content|new)"\s*:\s*)"(?P<placeholder><[^">]*content[^">]*>)"\s*\+\s*"(?P<suffix>(?:\\.|[^"\\])*)"',
    re.IGNORECASE,
)
PLACEHOLDER_DIRECT_GENERIC_RE = re.compile(r'("(?:content|new)"\s*:\s*)"(?P<placeholder><[^">]*content[^">]*>)"', re.IGNORECASE)
LOOSE_NAME_RE = re.compile(
    r"""["']?name["']?\s*:\s*["']?(?P<name>write_file|append_file|replace_in_file|make_directory|delete_path)["']?""",
    re.IGNORECASE,
)
LOOSE_PATH_RE = re.compile(r"""["']?path["']?\s*:\s*["'](?P<path>[^"']+)["']""", re.IGNORECASE)
LOOSE_DIRECT_RE = re.compile(r"""["']?(?:content|new)["']?\s*:\s*["'](?P<placeholder><[^"'>]*content[^"'>]*>)["']""", re.IGNORECASE)
LOOSE_SUFFIX_RE = re.compile(
    r"""["']?(?:content|new)["']?\s*:\s*(?P<prefix>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')\s*\+\s*["'](?P<placeholder><[^"'>]*content[^"'>]*>)["']""",
    re.IGNORECASE,
)
LOOSE_PREFIX_RE = re.compile(
    r"""["']?(?:content|new)["']?\s*:\s*["'](?P<placeholder><[^"'>]*content[^"'>]*>)["']\s*\+\s*(?P<suffix>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""",
    re.IGNORECASE,
)


def parse_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            loaded = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(loaded, dict):
            return loaded
    return {}


def fallback_tool_calls(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, str):
        return []
    payload = extract_tool_payload(content)
    if payload is None:
        return []
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str):
        return []
    return [{"function": {"name": name, "arguments": arguments}}]


def extract_tool_payload(content: str) -> dict[str, Any] | None:
    candidates = [content.strip()]
    match = JSON_BLOCK_RE.search(content)
    if match:
        candidates.insert(0, match.group(1).strip())
    else:
        open_match = JSON_BLOCK_OPEN_RE.search(content)
        if open_match:
            candidates.insert(0, open_match.group(1).strip())
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                payload = json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                payload = parse_relaxed_tool_payload(candidate)
        if isinstance(payload, dict):
            return payload
    return None


def parse_relaxed_tool_payload(candidate: str) -> dict[str, Any] | None:
    normalized = candidate.strip()
    normalized = UNQUOTED_KEY_RE.sub(r'\1"\2":', normalized)
    normalized = UNQUOTED_NAME_VALUE_RE.sub(r'\1"\2"', normalized)
    normalized = re.sub(r"\bTrue\b", "true", normalized)
    normalized = re.sub(r"\bFalse\b", "false", normalized)
    normalized = re.sub(r"\bNone\b", "null", normalized)
    normalized = TRAILING_COMMA_RE.sub(r"\1", normalized)
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        try:
            payload = json.loads(normalized, strict=False)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("name"), str):
        return None
    arguments = payload.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        return None
    return payload


def repair_placeholder_write_payload(content: str, replacement_text: str) -> dict[str, Any] | None:
    candidates = [content.strip()]
    match = JSON_BLOCK_RE.search(content)
    if match:
        candidates.insert(0, match.group(1).strip())
    else:
        open_match = JSON_BLOCK_OPEN_RE.search(content)
        if open_match:
            candidates.insert(0, open_match.group(1).strip())
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        if (
            '"name": "write_file"' not in candidate
            and '"name": "append_file"' not in candidate
            and '"name": "replace_in_file"' not in candidate
            and '"name": "make_directory"' not in candidate
            and '"name": "delete_path"' not in candidate
        ):
            continue
        repaired = candidate
        repaired = PLACEHOLDER_DIRECT_RE.sub(
            lambda m: f'{m.group(1)}{json.dumps(replacement_text)}',
            repaired,
        )
        repaired = PLACEHOLDER_DIRECT_GENERIC_RE.sub(
            lambda m: f'{m.group(1)}{json.dumps(replacement_text)}',
            repaired,
        )
        repaired = PLACEHOLDER_SUFFIX_RE.sub(
            lambda m: f'{m.group(1)}{json.dumps(_decode_json_fragment(m.group("prefix")) + replacement_text)}',
            repaired,
        )
        repaired = PLACEHOLDER_PREFIX_RE.sub(
            lambda m: f'{m.group(1)}{json.dumps(replacement_text + _decode_json_fragment(m.group("suffix")))}',
            repaired,
        )
        try:
            payload = json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            payload = _repair_placeholder_write_payload_loose(repaired, replacement_text)
            if payload is None:
                continue
        if not isinstance(payload, dict):
            continue
        if not isinstance(payload.get("name"), str):
            continue
        arguments = payload.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            continue
        return payload
    return None


def _decode_json_fragment(value: str) -> str:
    return json.loads(f'"{value}"')


def _repair_placeholder_write_payload_loose(candidate: str, replacement_text: str) -> dict[str, Any] | None:
    name_match = LOOSE_NAME_RE.search(candidate)
    path_match = LOOSE_PATH_RE.search(candidate)
    if name_match is None or path_match is None:
        return None
    payload_key = "content"
    value: str | None = None
    if LOOSE_DIRECT_RE.search(candidate):
        value = replacement_text
    else:
        suffix_match = LOOSE_SUFFIX_RE.search(candidate)
        if suffix_match is not None:
            value = _decode_loose_fragment(suffix_match.group("prefix")) + replacement_text
        else:
            prefix_match = LOOSE_PREFIX_RE.search(candidate)
            if prefix_match is not None:
                value = replacement_text + _decode_loose_fragment(prefix_match.group("suffix"))
    if value is None:
        return None
    return {
        "name": name_match.group("name"),
        "arguments": {
            "path": path_match.group("path"),
            payload_key: value,
        },
    }


def _decode_loose_fragment(value: str) -> str:
    try:
        decoded = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")
    return decoded if isinstance(decoded, str) else value.strip("\"'")
