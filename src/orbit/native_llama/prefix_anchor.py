from __future__ import annotations

from ctypes import c_ubyte
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import time
from typing import Any, Mapping


def prefix_anchor_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    configured = env.get("ORBIT_KV_PREFIX_ANCHOR")
    if configured is not None:
        value = configured.strip().lower()
        if value == "auto":
            return "auto"
        if value == "off":
            return "off"
        return "off"
    legacy = env.get("ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT", "")
    if legacy.strip().lower() in {"1", "true", "yes", "on"}:
        return "auto"
    return "auto"


def prefix_anchor_enabled() -> bool:
    return prefix_anchor_mode() == "auto"


@dataclass(frozen=True)
class PrefixAnchorState:
    prefix_hash: str | None = None
    token_count: int = 0
    model_id: str | None = None
    template_id: str | None = None
    tool_schema_hash: str | None = None
    capability_summary_hash: str | None = None
    runtime_policy_hash: str | None = None
    route_contract_hash: str | None = None
    backend_version: str | None = None
    native_version: str | None = None
    tools_mode: str | None = None
    checkpoint_size: int = 0
    checkpoint_created_at_monotonic: float | None = None
    valid: bool = False
    invalidation_reason: str | None = None
    checkpoint_data: bytes | None = field(default=None, repr=False, compare=False)


def compute_prefix_anchor_key(
    *,
    model_id: str | None,
    template_id: str | None,
    tool_schema_hash: str | None,
    capability_summary_hash: str | None,
    runtime_policy_hash: str | None,
    route_contract_hash: str | None,
    backend_version: str | None,
    native_version: str | None,
    tools_mode: str | None,
) -> str:
    payload = {
        "model_id": model_id,
        "template_id": template_id,
        "tool_schema_hash": tool_schema_hash,
        "capability_summary_hash": capability_summary_hash,
        "runtime_policy_hash": runtime_policy_hash,
        "route_contract_hash": route_contract_hash,
        "backend_version": backend_version,
        "native_version": native_version,
        "tools_mode": tools_mode,
    }
    return _hash(payload)


def can_use_prefix_anchor(
    state: PrefixAnchorState,
    *,
    enabled: bool | None = None,
    prefix_hash: str,
    token_count: int | None = None,
    model_id: str | None,
    template_id: str | None,
    tool_schema_hash: str | None,
    capability_summary_hash: str | None,
    runtime_policy_hash: str | None,
    route_contract_hash: str | None,
    backend_version: str | None,
    native_version: str | None,
    tools_mode: str | None,
) -> tuple[bool, str | None]:
    if enabled is None:
        enabled = prefix_anchor_enabled()
    if not enabled:
        return False, "anchor_disabled"
    if not state.valid:
        return False, state.invalidation_reason or "anchor_invalid"
    if state.prefix_hash != prefix_hash:
        return False, "prefix_hash_changed"
    if token_count is not None and state.token_count != token_count:
        return False, "token_count_changed"
    if state.model_id != model_id:
        return False, "model_id_changed"
    if state.template_id != template_id:
        return False, "template_id_changed"
    if state.tool_schema_hash != tool_schema_hash:
        return False, "tool_schema_changed"
    if state.capability_summary_hash != capability_summary_hash:
        return False, "capability_summary_changed"
    if state.runtime_policy_hash != runtime_policy_hash:
        return False, "runtime_policy_changed"
    if state.route_contract_hash != route_contract_hash:
        return False, "route_contract_changed"
    if state.backend_version != backend_version:
        return False, "backend_version_changed"
    if state.native_version != native_version:
        return False, "native_version_changed"
    if state.tools_mode != tools_mode:
        return False, "tools_mode_changed"
    if not state.checkpoint_data or state.checkpoint_size <= 0:
        return False, "checkpoint_missing"
    return True, None


def invalidate_prefix_anchor(state: PrefixAnchorState, reason: str) -> PrefixAnchorState:
    return replace(
        state,
        valid=False,
        invalidation_reason=reason,
        checkpoint_data=None,
        checkpoint_size=0,
        checkpoint_created_at_monotonic=None,
        token_count=0,
    )


def capture_prefix_anchor(
    *,
    lib: Any | None,
    ctx: Any | None,
    seq_id: int = 0,
    prefix_hash: str,
    token_count: int,
    model_id: str | None,
    template_id: str | None,
    tool_schema_hash: str | None,
    capability_summary_hash: str | None,
    runtime_policy_hash: str | None,
    route_contract_hash: str | None,
    backend_version: str | None,
    native_version: str | None,
    tools_mode: str | None,
    enabled: bool | None = None,
) -> tuple[PrefixAnchorState, dict[str, Any]]:
    if enabled is None:
        enabled = prefix_anchor_enabled()
    metadata = _metadata(
        enabled=enabled,
        key_hash=prefix_hash,
        capture_attempted=False,
        restore_attempted=False,
        restore_used=False,
        checkpoint_size=0,
        checkpoint_age_ms=None,
        token_count=token_count,
    )
    base_state = PrefixAnchorState(
        prefix_hash=prefix_hash,
        token_count=token_count,
        model_id=model_id,
        template_id=template_id,
        tool_schema_hash=tool_schema_hash,
        capability_summary_hash=capability_summary_hash,
        runtime_policy_hash=runtime_policy_hash,
        route_contract_hash=route_contract_hash,
        backend_version=backend_version,
        native_version=native_version,
        tools_mode=tools_mode,
        valid=False,
        invalidation_reason="capture_not_attempted",
    )
    if not enabled:
        metadata["fallback_reason"] = "anchor_disabled"
        metadata["invalidation_reason"] = "anchor_disabled"
        return invalidate_prefix_anchor(base_state, "anchor_disabled"), metadata
    if lib is None or ctx is None:
        metadata["fallback_reason"] = "native_handles_missing"
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "native_handles_missing"
        return invalidate_prefix_anchor(base_state, "native_handles_missing"), metadata
    if token_count <= 0:
        metadata["fallback_reason"] = "invalid_token_count"
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "invalid_token_count"
        return invalidate_prefix_anchor(base_state, "invalid_token_count"), metadata
    metadata["capture_attempted"] = True
    try:
        size = int(lib.llama_state_seq_get_size(ctx, seq_id))
    except Exception:
        metadata["fallback_reason"] = "checkpoint_size_failed"
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "checkpoint_size_failed"
        return invalidate_prefix_anchor(base_state, "checkpoint_size_failed"), metadata
    if size <= 0:
        metadata["fallback_reason"] = "empty_checkpoint"
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "empty_checkpoint"
        return invalidate_prefix_anchor(base_state, "empty_checkpoint"), metadata
    try:
        buffer = (c_ubyte * size)()
        written = int(lib.llama_state_seq_get_data(ctx, buffer, size, seq_id))
    except Exception:
        metadata["fallback_reason"] = "checkpoint_capture_failed"
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "checkpoint_capture_failed"
        return invalidate_prefix_anchor(base_state, "checkpoint_capture_failed"), metadata
    if written != size:
        metadata["fallback_reason"] = "checkpoint_size_mismatch"
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "checkpoint_size_mismatch"
        return invalidate_prefix_anchor(base_state, "checkpoint_size_mismatch"), metadata
    checkpoint = bytes(buffer)
    state = replace(
        base_state,
        checkpoint_size=size,
        checkpoint_created_at_monotonic=time.monotonic(),
        checkpoint_data=checkpoint,
        valid=True,
        invalidation_reason=None,
    )
    metadata["anchor_valid"] = True
    metadata["checkpoint_size"] = size
    metadata["checkpoint_size_bytes"] = size
    metadata["checkpoint_age_ms"] = 0
    metadata["anchor_hit"] = False
    metadata["anchor_miss"] = False
    return state, metadata


def restore_prefix_anchor(
    state: PrefixAnchorState,
    *,
    lib: Any | None,
    ctx: Any | None,
    seq_id: int = 0,
    prefix_hash: str,
    token_count: int | None = None,
    model_id: str | None,
    template_id: str | None,
    tool_schema_hash: str | None,
    capability_summary_hash: str | None,
    runtime_policy_hash: str | None,
    route_contract_hash: str | None,
    backend_version: str | None,
    native_version: str | None,
    tools_mode: str | None,
    enabled: bool | None = None,
) -> tuple[bool, PrefixAnchorState, dict[str, Any]]:
    if enabled is None:
        enabled = prefix_anchor_enabled()
    ok, reason = can_use_prefix_anchor(
        state,
        enabled=enabled,
        prefix_hash=prefix_hash,
        token_count=token_count,
        model_id=model_id,
        template_id=template_id,
        tool_schema_hash=tool_schema_hash,
        capability_summary_hash=capability_summary_hash,
        runtime_policy_hash=runtime_policy_hash,
        route_contract_hash=route_contract_hash,
        backend_version=backend_version,
        native_version=native_version,
        tools_mode=tools_mode,
    )
    metadata = _metadata(
        enabled=enabled,
        key_hash=prefix_hash,
        capture_attempted=False,
        restore_attempted=True,
        restore_used=False,
        checkpoint_size=state.checkpoint_size,
        checkpoint_age_ms=_checkpoint_age_ms(state),
        token_count=state.token_count,
    )
    if not ok:
        metadata["fallback_reason"] = reason
        metadata["anchor_miss"] = True
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = reason
        return False, invalidate_prefix_anchor(state, reason or "anchor_invalid"), metadata
    if lib is None or ctx is None:
        metadata["fallback_reason"] = "native_handles_missing"
        metadata["anchor_miss"] = True
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "native_handles_missing"
        return False, invalidate_prefix_anchor(state, "native_handles_missing"), metadata
    try:
        assert state.checkpoint_data is not None
        buffer = (c_ubyte * len(state.checkpoint_data)).from_buffer_copy(state.checkpoint_data)
        written = int(lib.llama_state_seq_set_data(ctx, buffer, len(state.checkpoint_data), seq_id))
    except Exception:
        metadata["fallback_reason"] = "checkpoint_restore_failed"
        metadata["anchor_miss"] = True
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "checkpoint_restore_failed"
        return False, invalidate_prefix_anchor(state, "checkpoint_restore_failed"), metadata
    if written != len(state.checkpoint_data):
        metadata["fallback_reason"] = "checkpoint_restore_size_mismatch"
        metadata["anchor_miss"] = True
        metadata["anchor_invalidated"] = True
        metadata["invalidation_reason"] = "checkpoint_restore_size_mismatch"
        return False, invalidate_prefix_anchor(state, "checkpoint_restore_size_mismatch"), metadata
    metadata["anchor_hit"] = True
    metadata["restore_used"] = True
    return True, state, metadata


def anchor_metadata(
    state: PrefixAnchorState,
    *,
    enabled: bool | None = None,
    anchor_hit: bool = False,
    anchor_miss: bool = False,
    capture_attempted: bool = False,
    restore_attempted: bool = False,
    restore_used: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    if enabled is None:
        enabled = prefix_anchor_enabled()
    metadata = _metadata(
        enabled=enabled,
        key_hash=state.prefix_hash,
        capture_attempted=capture_attempted,
        restore_attempted=restore_attempted,
        restore_used=restore_used,
        checkpoint_size=state.checkpoint_size,
        checkpoint_age_ms=_checkpoint_age_ms(state),
        token_count=state.token_count,
    )
    metadata["anchor_valid"] = state.valid
    metadata["anchor_invalidated"] = not state.valid
    metadata["anchor_hit"] = anchor_hit
    metadata["anchor_miss"] = anchor_miss
    metadata["fallback_reason"] = fallback_reason or state.invalidation_reason
    metadata["invalidation_reason"] = state.invalidation_reason
    return metadata


def _metadata(
    *,
    enabled: bool,
    key_hash: str | None,
    capture_attempted: bool,
    restore_attempted: bool,
    restore_used: bool,
    checkpoint_size: int,
    checkpoint_age_ms: int | None,
    token_count: int,
) -> dict[str, Any]:
    anchor_invalidated = False
    return {
        "anchor_enabled": enabled,
        "anchor_key_hash": key_hash,
        "anchor_valid": False,
        "anchor_invalidated": anchor_invalidated,
        "anchor_hit": False,
        "anchor_miss": False,
        "capture_attempted": capture_attempted,
        "restore_attempted": restore_attempted,
        "restore_used": restore_used,
        "fallback_reason": None,
        "checkpoint_size": checkpoint_size,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_age_ms": checkpoint_age_ms,
        "invalidation_reason": None,
        "token_count": token_count,
    }


def _checkpoint_age_ms(state: PrefixAnchorState) -> int | None:
    if state.checkpoint_created_at_monotonic is None:
        return None
    return max(0, int((time.monotonic() - state.checkpoint_created_at_monotonic) * 1000))


def _hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
