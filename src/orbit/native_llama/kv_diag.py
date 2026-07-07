from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Iterator


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
    phase: str | None
    tools_mode: str | None
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
        phase=_safe_str(payload.get("_orbit_kv_phase")),
        tools_mode=_safe_str(payload.get("_orbit_kv_tools_mode")),
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
    component_tokens: dict[str, Any] | None = None,
) -> None:
    if not enabled():
        return
    request = _CURRENT_REQUEST.get()
    common = _longest_common_prefix(prompt_tokens, previous_prompt_tokens)
    prompt_count = len(prompt_tokens)
    previous_count = len(previous_prompt_tokens)
    first_mismatch_token = common if common < min(prompt_count, previous_count) else None
    previous_token_at_mismatch = (
        previous_prompt_tokens[common] if first_mismatch_token is not None and common < previous_count else None
    )
    current_token_at_mismatch = prompt_tokens[common] if first_mismatch_token is not None and common < prompt_count else None
    evaluated = max(0, prompt_count - reused_prompt_tokens)
    event = {
        "event": "kv_diag_native_cache",
        "backend_request_id": request.backend_request_id if request else None,
        "model_call_id": None,
        "phase": request.phase if request else None,
        "tools_mode": request.tools_mode if request else None,
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
        "current_tokenized_prompt_hash": _hash_tokens(prompt_tokens),
        "previous_tokenized_prompt_hash": _hash_tokens(previous_prompt_tokens) if previous_prompt_tokens else None,
        "tokenized_prompt_hash": _hash_tokens(prompt_tokens),
        "tokenized_prefix_hash": _hash_tokens(prompt_tokens[:common]) if common else None,
        "tokenized_prefix_length": common,
        "longest_common_prefix_tokens": common,
        "first_mismatch_index": first_mismatch_token,
        "first_mismatch_token": first_mismatch_token,
        "previous_token_at_mismatch": previous_token_at_mismatch,
        "current_token_at_mismatch": current_token_at_mismatch,
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
    if component_tokens is not None and _is_final_or_retry_phase(request.phase if request else None):
        _emit_prompt_component_tokens_event(
            request=request,
            component_tokens=component_tokens,
            prompt_tokens=prompt_count,
            cached_tokens=reused_prompt_tokens,
            evaluated_tokens=evaluated,
        )


def build_prompt_component_tokens(
    *,
    messages: list[dict[str, Any]],
    prompt_tokens_total: int,
    token_count: Callable[[str], int],
) -> dict[str, Any]:
    components = {
        "system_prompt_tokens": 0,
        "conversation_window_tokens": 0,
        "assistant_history_tokens": 0,
        "user_message_tokens": 0,
        "evidence_total_tokens": 0,
        "evidence_wrapper_tokens": 0,
        "evidence_metadata_tokens": 0,
        "evidence_raw_excerpt_tokens": 0,
        "evidence_summary_tokens": 0,
        "tool_result_tokens": 0,
        "other_message_tokens": 0,
        "template_overhead_tokens": 0,
        "unknown_tokens": 0,
    }
    evidence_kinds: list[str] = []
    evidence_card_count = 0
    evidence_cards: list[dict[str, Any]] = []
    content_total = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = _content_text(message.get("content"))
        tokens = token_count(content) if content else 0
        if role == "system" and index == 0:
            components["system_prompt_tokens"] += tokens
            content_total += tokens
            continue
        if role == "system" and _looks_like_evidence_context(content):
            evidence = _evidence_component_tokens(content, token_count=token_count)
            for key in (
                "evidence_wrapper_tokens",
                "evidence_metadata_tokens",
                "evidence_raw_excerpt_tokens",
                "evidence_summary_tokens",
            ):
                components[key] += evidence[key]
            components["evidence_total_tokens"] += tokens
            content_total += tokens
            evidence_card_count += evidence["evidence_card_count"]
            evidence_cards.extend(evidence.get("cards", []))
            for kind in evidence["evidence_kinds"]:
                if kind not in evidence_kinds:
                    evidence_kinds.append(kind)
            continue
        if role == "assistant":
            components["assistant_history_tokens"] += tokens
            components["conversation_window_tokens"] += tokens
        elif role == "user":
            components["user_message_tokens"] += tokens
            components["conversation_window_tokens"] += tokens
        elif role == "tool":
            components["tool_result_tokens"] += tokens
            components["conversation_window_tokens"] += tokens
        else:
            components["other_message_tokens"] += tokens
        content_total += tokens
    remainder = prompt_tokens_total - content_total
    if remainder >= 0:
        components["template_overhead_tokens"] = remainder
    else:
        components["unknown_tokens"] = -remainder
    return {
        "components": components,
        "evidence_card_count": evidence_card_count,
        "evidence_kinds": evidence_kinds,
        "content_tokens_total": content_total,
        "evidence_cards": evidence_cards,
    }


def _emit_prompt_component_tokens_event(
    *,
    request: NativeDiagRequest | None,
    component_tokens: dict[str, Any],
    prompt_tokens: int,
    cached_tokens: int,
    evaluated_tokens: int,
) -> None:
    event = {
        "event": "kv_diag_prompt_component_tokens",
        "backend_request_id": request.backend_request_id if request else None,
        "model_call_id": None,
        "phase": request.phase if request else None,
        "tools_mode": request.tools_mode if request else None,
        "prompt_tokens_total": prompt_tokens,
        "cached_tokens": cached_tokens,
        "evaluated_tokens": evaluated_tokens,
        "role_sequence": request.role_sequence if request else [],
        "components": component_tokens.get("components", {}),
        "evidence_card_count": _safe_int(component_tokens.get("evidence_card_count")) or 0,
        "evidence_kinds": component_tokens.get("evidence_kinds") if isinstance(component_tokens.get("evidence_kinds"), list) else [],
        "content_tokens_total": _safe_int(component_tokens.get("content_tokens_total")),
    }
    _emit(event)
    for card in component_tokens.get("evidence_cards", []):
        if isinstance(card, dict):
            _emit_evidence_card_tokens_event(
                request=request,
                card=card,
            )


def _emit_evidence_card_tokens_event(
    *,
    request: NativeDiagRequest | None,
    card: dict[str, Any],
) -> None:
    event = {
        "event": "kv_diag_evidence_card_tokens",
        "backend_request_id": request.backend_request_id if request else None,
        "model_call_id": None,
        "phase": request.phase if request else None,
        "tools_mode": request.tools_mode if request else None,
        "role_sequence": request.role_sequence if request else [],
        "card_index": _safe_int(card.get("card_index")),
        "evidence_id_hash": _safe_str(card.get("evidence_id_hash")),
        "kind": _safe_str(card.get("kind")),
        "status": _safe_str(card.get("status")),
        "command_hash": _safe_str(card.get("command_hash")),
        "path_hash": _safe_str(card.get("path_hash")),
        "metadata_tokens": _safe_int(card.get("metadata_tokens")) or 0,
        "raw_excerpt_tokens": _safe_int(card.get("raw_excerpt_tokens")) or 0,
        "summary_tokens": _safe_int(card.get("summary_tokens")) or 0,
        "wrapper_tokens": _safe_int(card.get("wrapper_tokens")) or 0,
        "unknown_tokens": _safe_int(card.get("unknown_tokens")) or 0,
        "total_tokens": _safe_int(card.get("total_tokens")) or 0,
        "has_raw_excerpt": bool(card.get("has_raw_excerpt")),
        "has_summary": bool(card.get("has_summary")),
        "is_error_status": bool(card.get("is_error_status")),
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


def emit_route_prefix_prewarm_event(metadata: dict[str, Any]) -> None:
    if not enabled():
        return
    event = {
        "event": "kv_diag_route_prefix_prewarm",
        "tools_default_enabled": bool(metadata.get("tools_default_enabled")),
        "tools_startup_enabled": bool(metadata.get("tools_startup_enabled")),
        "prewarm_enabled": bool(metadata.get("prewarm_enabled")),
        "prewarm_mode": _safe_str(metadata.get("prewarm_mode")),
        "prewarm_attempted": bool(metadata.get("prewarm_attempted")),
        "prewarm_succeeded": bool(metadata.get("prewarm_succeeded")),
        "prewarm_skipped_reason": _safe_str(metadata.get("prewarm_skipped_reason")),
        "prewarm_failed_reason": _safe_str(metadata.get("prewarm_failed_reason")),
        "prewarm_prefix_token_count": _safe_int(metadata.get("prewarm_prefix_token_count")),
        "prewarm_checkpoint_size_bytes": _safe_int(metadata.get("prewarm_checkpoint_size_bytes")),
        "prewarm_ms": _safe_int(metadata.get("prewarm_ms")),
        "decode_calls": _safe_int(metadata.get("decode_calls")),
        "sampled_tokens": _safe_int(metadata.get("sampled_tokens")),
        "generated_tokens": _safe_int(metadata.get("generated_tokens")),
        "sampler_touched": bool(metadata.get("sampler_touched")),
        "session_history_touched": bool(metadata.get("session_history_touched")),
        "restore_ready": bool(metadata.get("restore_ready")),
    }
    _emit(event)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return " ".join(parts)
    return ""


def _looks_like_evidence_context(content: str) -> bool:
    return content.startswith("evidence_context:")


def _is_final_or_retry_phase(phase: str | None) -> bool:
    if phase is None:
        return False
    return phase == "chat_final" or phase == "chat_final_retry" or phase.startswith("final_from_tool")


def _evidence_component_tokens(content: str, *, token_count: Callable[[str], int]) -> dict[str, Any]:
    wrapper_lines: list[str] = []
    metadata_lines: list[str] = []
    raw_lines: list[str] = []
    summary_lines: list[str] = []
    evidence_kinds: list[str] = []
    in_raw_excerpt = False
    evidence_card_count = 0
    cards: list[dict[str, Any]] = []
    current_card: dict[str, Any] | None = None
    summary_prefixes = (
        "stdout_excerpt:",
        "stderr_excerpt:",
        "first_matches:",
        "top_snippets:",
        "compat_excerpt:",
    )
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "bounded_raw_excerpt:":
            wrapper_lines.append(line)
            if current_card is not None:
                current_card["wrapper_lines"].append(line)
            in_raw_excerpt = True
            continue
        if stripped.startswith("- evidence "):
            wrapper_lines.append(line)
            evidence_card_count += 1
            current_card = {
                "card_index": evidence_card_count,
                "wrapper_lines": [line],
                "metadata_lines": [],
                "raw_lines": [],
                "summary_lines": [],
                "kind": None,
                "status": None,
                "raw_ref": None,
                "command": None,
                "path": None,
            }
            cards.append(current_card)
            in_raw_excerpt = False
            continue
        if stripped == "evidence_context:":
            wrapper_lines.append(line)
            in_raw_excerpt = False
            continue
        if in_raw_excerpt:
            raw_lines.append(line)
            if current_card is not None:
                current_card["raw_lines"].append(line)
            continue
        if stripped.startswith("kind:"):
            kind = stripped.split(":", 1)[1].strip()
            if kind and kind not in evidence_kinds:
                evidence_kinds.append(kind)
            if current_card is not None:
                current_card["kind"] = kind or None
        if stripped.startswith("status:") and current_card is not None:
            current_card["status"] = stripped.split(":", 1)[1].strip() or None
        if stripped.startswith("raw_ref:") and current_card is not None:
            current_card["raw_ref"] = stripped.split(":", 1)[1].strip() or None
        if stripped.startswith("command:") and current_card is not None:
            current_card["command"] = stripped.split(":", 1)[1].strip() or None
        if stripped.startswith(("file_paths:", "path:")) and current_card is not None:
            current_card["path"] = stripped.split(":", 1)[1].strip() or None
        if stripped.startswith(summary_prefixes):
            summary_lines.append(line)
            if current_card is not None:
                current_card["summary_lines"].append(line)
            continue
        metadata_lines.append(line)
        if current_card is not None:
            current_card["metadata_lines"].append(line)
    return {
        "evidence_wrapper_tokens": _joined_token_count(wrapper_lines, token_count=token_count),
        "evidence_metadata_tokens": _joined_token_count(metadata_lines, token_count=token_count),
        "evidence_raw_excerpt_tokens": _joined_token_count(raw_lines, token_count=token_count),
        "evidence_summary_tokens": _joined_token_count(summary_lines, token_count=token_count),
        "evidence_card_count": evidence_card_count,
        "evidence_kinds": evidence_kinds,
        "cards": [_evidence_card_component_tokens(card, token_count=token_count) for card in cards],
    }


def _evidence_card_component_tokens(card: dict[str, Any], *, token_count: Callable[[str], int]) -> dict[str, Any]:
    wrapper_tokens = _joined_token_count(card["wrapper_lines"], token_count=token_count)
    metadata_tokens = _joined_token_count(card["metadata_lines"], token_count=token_count)
    raw_tokens = _joined_token_count(card["raw_lines"], token_count=token_count)
    summary_tokens = _joined_token_count(card["summary_lines"], token_count=token_count)
    total = wrapper_tokens + metadata_tokens + raw_tokens + summary_tokens
    raw_ref = card.get("raw_ref")
    command = card.get("command")
    path = card.get("path")
    status = _safe_str(card.get("status"))
    return {
        "card_index": _safe_int(card.get("card_index")) or 0,
        "evidence_id_hash": _hash(raw_ref) if isinstance(raw_ref, str) and raw_ref else None,
        "kind": _safe_str(card.get("kind")),
        "status": status,
        "command_hash": _hash(command) if isinstance(command, str) and command else None,
        "path_hash": _hash(path) if isinstance(path, str) and path else None,
        "metadata_tokens": metadata_tokens,
        "raw_excerpt_tokens": raw_tokens,
        "summary_tokens": summary_tokens,
        "wrapper_tokens": wrapper_tokens,
        "unknown_tokens": 0,
        "total_tokens": total,
        "has_raw_excerpt": raw_tokens > 0,
        "has_summary": summary_tokens > 0,
        "is_error_status": status == "error",
    }


def _joined_token_count(lines: list[str], *, token_count: Callable[[str], int]) -> int:
    if not lines:
        return 0
    return token_count("\n".join(lines))


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
