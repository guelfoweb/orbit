"""Lossless FINISH wire-order compatibility for the native Qwen parser.

The template permits parameters in any order; the current native parser only
accepts required parameters before optional ones. Never infer a call from prose:
recognize a single complete envelope and move whole, unchanged parameter blocks.
"""
from __future__ import annotations

import json


FINISH_NAME = "finish_analysis_question"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


def strict_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def reorder_finish_parameters(raw: str, tools: list[dict] | None):
    if not tools or len(tools) != 1:
        return None
    function = tools[0].get("function", {})
    if function.get("name") != FINISH_NAME:
        return None
    schema = function.get("parameters", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not required or not all(isinstance(k, str) and k in properties for k in required):
        return None
    prefix = f"<tool_call>\n<function={FINISH_NAME}>\n"
    suffix = "</function>\n</tool_call>"
    if not raw.startswith(prefix) or not raw.endswith(suffix):
        return None
    body = raw[len(prefix):-len(suffix)]
    blocks = {}
    expected = {}
    while body:
        if not body.startswith("<parameter="):
            return None
        end = body.find(">\n")
        if end < 0:
            return None
        name = body[len("<parameter="):end]
        if name not in properties or name in blocks:
            return None
        close = "\n</parameter>\n"
        stop = body.find(close, end + 2)
        if stop < 0:
            return None
        value = body[end + 2:stop]
        if any(tag in value for tag in (
            "<parameter", "</parameter", "<function", "</function",
            "<tool_call", "</tool_call",
        )):
            return None
        kind = properties[name].get("type")
        if kind == "string":
            expected[name] = value
        elif kind in {"array", "object"}:
            try:
                expected[name] = strict_json(value)
            except (ValueError, TypeError):
                return None
            if not isinstance(expected[name], list if kind == "array" else dict):
                return None
        else:
            return None
        boundary = stop + len(close)
        blocks[name] = body[:boundary]
        body = body[boundary:]
    if not all(name in blocks for name in required):
        return None
    ordered = [k for k in blocks if k in required] + [k for k in blocks if k not in required]
    if ordered == list(blocks):
        return None
    return prefix + "".join(blocks[k] for k in ordered) + suffix, expected


def exact_finish_arguments(calls, expected: dict) -> bool:
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != FINISH_NAME:
        return False
    try:
        actual = strict_json(calls[0]["function"]["arguments"])
    except (ValueError, TypeError, KeyError):
        return False
    # Canonical JSON comparison preserves scalar types (True != 1 here).
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
