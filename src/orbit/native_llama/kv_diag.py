from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from itertools import count
from typing import Any, Iterator


_REQUEST_IDS = count(1)
_CURRENT_REQUEST: contextvars.ContextVar["NativeDiagRequest | None"] = contextvars.ContextVar(
    "orbit_native_kv_diag_request",
    default=None,
)


@dataclass(frozen=True)
class NativeDiagRequest:
    backend_request_id: str
    endpoint: str | None
    stream: bool | None
    cache_prompt: bool | None
    session_id_hash: str | None
    tools_parameter_present: bool
    tool_count: int
    message_count: int
    role_sequence: list[str]


def enabled() -> bool:
    value = os.environ.get("ORBIT_KV_DIAG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def reset_diagnostics_for_tests() -> None:
    global _REQUEST_IDS
    _REQUEST_IDS = count(1)


@contextlib.contextmanager
def request_context(*, endpoint: str | None, payload: dict[str, Any]) -> Iterator[None]:
    if not enabled() or _CURRENT_REQUEST.get() is not None:
        yield
        return
    request = NativeDiagRequest(
        backend_request_id=f"native_req_{next(_REQUEST_IDS):06d}",
        endpoint=endpoint,
        stream=payload.get("stream") is True,
        cache_prompt=payload.get("cache_prompt") if isinstance(payload.get("cache_prompt"), bool) else None,
        session_id_hash=_hash(str(payload.get("session_id") or "default")),
        tools_parameter_present=isinstance(payload.get("tools"), list) and bool(payload.get("tools")),
        tool_count=len(payload.get("tools")) if isinstance(payload.get("tools"), list) else 0,
        message_count=len(payload.get("messages")) if isinstance(payload.get("messages"), list) else 0,
        role_sequence=_role_sequence(payload.get("messages")),
    )
    token = _CURRENT_REQUEST.set(request)
    try:
        yield
    finally:
        _CURRENT_REQUEST.reset(token)


def emit_prompt_cache_event(
    *,
    prompt_tokens: list[int],
    previous_prompt_tokens: list[int],
    reused_prompt_tokens: int,
    output_tokens: int,
    cancelled: bool,
    slot_id: str,
) -> None:
    if not enabled():
        return
    request = _CURRENT_REQUEST.get()
    common = _longest_common_prefix(prompt_tokens, previous_prompt_tokens)
    prompt_count = len(prompt_tokens)
    previous_count = len(previous_prompt_tokens)
    first_mismatch_token = common if common < min(prompt_count, previous_count) else None
    evaluated = max(0, prompt_count - reused_prompt_tokens)
    event = {
        "event": "kv_diag_native_cache",
        "backend_request_id": request.backend_request_id if request else None,
        "model_call_id": None,
        "phase": None,
        "endpoint": request.endpoint if request else None,
        "stream": request.stream if request else None,
        "cache_prompt": request.cache_prompt if request else None,
        "slot_id": slot_id,
        "session_cache_key_hash": request.session_id_hash if request else None,
        "session_id_hash": request.session_id_hash if request else None,
        "message_count": request.message_count if request else None,
        "role_sequence": request.role_sequence if request else [],
        "tools_parameter_present": request.tools_parameter_present if request else False,
        "tool_count": request.tool_count if request else 0,
        "prompt_tokens": prompt_count,
        "previous_prompt_tokens": previous_count,
        "tokenized_prompt_hash": _hash_tokens(prompt_tokens),
        "tokenized_prefix_hash": _hash_tokens(prompt_tokens[:common]) if common else None,
        "tokenized_prefix_length": common,
        "longest_common_prefix_tokens": common,
        "first_mismatch_token": first_mismatch_token,
        "cached_tokens": reused_prompt_tokens,
        "evaluated_tokens": evaluated,
        "output_tokens": output_tokens,
        "cancelled": cancelled,
        "cache_miss_reason": _cache_miss_reason(
            cache_prompt=request.cache_prompt if request else None,
            previous_prompt_tokens=previous_count,
            common_tokens=common,
            reused_prompt_tokens=reused_prompt_tokens,
        ),
    }
    _emit(event)


def emit_route_prefix_anchor_event(metadata: dict[str, Any]) -> None:
    if not enabled():
        return
    request = _CURRENT_REQUEST.get()
    event = {
        "event": "kv_diag_route_prefix_anchor",
        "phase": _safe_str(metadata.get("phase")) or "route",
        "backend_request_id": request.backend_request_id if request else None,
        "endpoint": request.endpoint if request else None,
        "stream": request.stream if request else None,
        "cache_prompt": request.cache_prompt if request else None,
        "session_id_hash": request.session_id_hash if request else None,
        "message_count": request.message_count if request else None,
        "role_sequence": request.role_sequence if request else [],
        "tools_parameter_present": request.tools_parameter_present if request else False,
        "tool_count": request.tool_count if request else 0,
        "route_anchor_enabled": bool(metadata.get("route_anchor_enabled")),
        "route_anchor_attempted": bool(metadata.get("route_anchor_attempted")),
        "route_anchor_hit": bool(metadata.get("route_anchor_hit")),
        "route_anchor_miss": bool(metadata.get("route_anchor_miss")),
        "capture_attempted": bool(metadata.get("capture_attempted")),
        "restore_attempted": bool(metadata.get("restore_attempted")),
        "restore_used": bool(metadata.get("restore_used")),
        "fallback_reason": _safe_str(metadata.get("fallback_reason")),
        "prefix_hash": _safe_str(metadata.get("prefix_hash")),
        "prefix_token_count": _safe_int(metadata.get("prefix_token_count")),
        "checkpoint_size": _safe_int(metadata.get("checkpoint_size")),
        "checkpoint_size_bytes": _safe_int(metadata.get("checkpoint_size_bytes")) or _safe_int(metadata.get("checkpoint_size")),
        "checkpoint_age_ms": _safe_int(metadata.get("checkpoint_age_ms")),
        "anchor_invalidated": bool(metadata.get("anchor_invalidated")),
        "invalidation_reason": _safe_str(metadata.get("invalidation_reason")),
        "cached_tokens": _safe_int(metadata.get("cached_tokens")),
        "evaluated_tokens": _safe_int(metadata.get("evaluated_tokens")),
        "lcp_tokens": _safe_int(metadata.get("lcp_tokens")),
    }
    _emit(event)


def _cache_miss_reason(
    *,
    cache_prompt: bool | None,
    previous_prompt_tokens: int,
    common_tokens: int,
    reused_prompt_tokens: int,
) -> str | None:
    if cache_prompt is False:
        return "cache_disabled"
    if previous_prompt_tokens <= 0:
        return "no_previous_prompt"
    if reused_prompt_tokens > 0:
        return None
    if common_tokens <= 0:
        return "prefix_mismatch_at_token_0"
    return f"prefix_mismatch_after_token_{common_tokens}"


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    common = 0
    limit = min(len(left), len(right))
    while common < limit and left[common] == right[common]:
        common += 1
    return common


def _role_sequence(messages: object) -> list[str]:
    if not isinstance(messages, list):
        return []
    roles: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            roles.append("unknown")
            continue
        role = message.get("role")
        roles.append(role if isinstance(role, str) else "unknown")
    return roles


def _hash_tokens(tokens: list[int]) -> str:
    return _hash(",".join(str(token) for token in tokens))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _safe_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _safe_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _emit(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path = os.environ.get("ORBIT_KV_DIAG_FILE")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return
    print(line, file=sys.stderr, flush=True)
