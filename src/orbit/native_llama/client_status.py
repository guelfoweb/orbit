"""Read-only status / metrics serialisation for NativeLlamaClient.

RUNTIME-DECOMPOSITION-CAMPAIGN-1 R1. These are the observability accessors that
serialise the client's own runtime state into the dicts `/props` and diagnostics
consume. They are pure and read-only -- they never mutate client state -- so this
is an ownership move, not a behaviour change: `NativeLlamaClient` keeps thin
delegates with identical signatures and identical dict values/ordering.

Each function takes the client instance and reads its attributes; this module
never imports client.py, so there is no runtime import cycle. State stays owned by
the client -- nothing here is duplicated or cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .expert_usage import summarize_expert_usage
from .model_profiles import ORNITH15_PROFILE_ID, QWEN36_PROFILE_ID, QWEN3_CODER_PROFILE_ID, QWEN38_FLASH_NEXT_PROFILE_ID
from .ornith_route_prefix import ORNITH_ROUTE_TOKENIZER_IDENTITY
from .qwen36_shell_tool_prefix import (
    QWEN36_SHELL_TOOL_PREFIX_FORMAT_VERSION,
    QWEN36_SHELL_TOOL_TOKENIZER_IDENTITY,
)
from .qwen3_coder_route_prefix import QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY
from .qwen_route_prefix import QWEN_ROUTE_TOKENIZER_IDENTITY, hash_text

if TYPE_CHECKING:  # pragma: no cover - typing only, never a runtime import
    from .client import NativeLlamaClient


def final_prefix_experiment_status(client: "NativeLlamaClient") -> dict[str, object]:
    status = client._final_prefix_status
    profile = getattr(client, "model_profile", None)
    profile_eligible = profile is None or profile.gemma_prefix_reuse_supported
    return {
        "enabled": client.config.final_prefix_experiment_enabled and profile_eligible,
        "initialized": status.initialized,
        "prefix_tokens": status.prefix_tokens,
        "capture_count": status.capture_count,
        "restore_count": status.restore_count,
        "fallback_count": status.fallback_count,
        "failure_reason": status.failure_reason,
        "last_used": status.last_used,
        "checkpoint_size_bytes": client._final_prefix_anchor_state.checkpoint_size,
    }


def qwen_route_prefix_reuse_status(client: "NativeLlamaClient") -> dict[str, object]:
    status = client._qwen_route_prefix_status
    profile = getattr(client, "model_profile", None)
    profile_eligible = (
        getattr(profile, "profile_id", None) == QWEN36_PROFILE_ID
        and getattr(profile, "verified", False)
        and client._model_metadata_identity.get("general.file_type") == "15"
    )
    if getattr(profile, "profile_id", None) == QWEN38_FLASH_NEXT_PROFILE_ID:
        profile_eligible = (
            getattr(profile, "verified", False)
            and getattr(profile, "route_prefix_reuse_supported", False)
            and client._model_metadata_identity.get("general.file_type") == "31"
            and client._qwen38_route_prefix_config_eligible()
        )
    spec = client._qwen_route_prefix_spec
    return {
        "enabled": client.config.qwen_route_prefix_reuse_enabled and profile_eligible,
        "source": client.config.qwen_route_prefix_reuse_source,
        "config_error": client.config.qwen_route_prefix_reuse_config_error,
        "initialized": status.initialized,
        "prefix_tokens": status.prefix_tokens,
        "capture_count": status.capture_count,
        "restore_count": status.restore_count,
        "fallback_count": status.fallback_count,
        "invalidation_count": status.invalidation_count,
        "failure_reason": status.failure_reason,
        "last_used": status.last_used,
        "checkpoint_size_bytes": client._qwen_route_prefix_anchor_state.checkpoint_size,
        "profile_identity": getattr(profile, "profile_id", None),
        "template_identity": getattr(profile, "template_sha256", None),
        "tokenizer_identity": hash_text(QWEN_ROUTE_TOKENIZER_IDENTITY),
        "prefix_token_hash": spec.prefix_token_hash if spec is not None else None,
        "prefix_text_hash": spec.invariant_text_hash if spec is not None else None,
    }


def qwen3_coder_route_prefix_reuse_status(client: "NativeLlamaClient") -> dict[str, object]:
    status = client._qwen3_coder_route_prefix_status
    profile = getattr(client, "model_profile", None)
    profile_eligible = (
        getattr(profile, "profile_id", None) == QWEN3_CODER_PROFILE_ID
        and getattr(profile, "verified", False)
        and client._model_metadata_identity.get("general.file_type") == "15"
    )
    spec = client._qwen3_coder_route_prefix_spec
    return {
        "enabled": client.config.qwen3_coder_route_prefix_reuse_enabled and profile_eligible,
        "source": client.config.qwen3_coder_route_prefix_reuse_source,
        "config_error": client.config.qwen3_coder_route_prefix_reuse_config_error,
        "initialized": status.initialized,
        "prefix_tokens": status.prefix_tokens,
        "capture_count": status.capture_count,
        "restore_count": status.restore_count,
        "fallback_count": status.fallback_count,
        "invalidation_count": status.invalidation_count,
        "failure_reason": status.failure_reason,
        "last_used": status.last_used,
        "checkpoint_size_bytes": client._qwen3_coder_route_prefix_anchor_state.checkpoint_size,
        "profile_identity": getattr(profile, "profile_id", None),
        "template_identity": getattr(profile, "template_sha256", None),
        "tokenizer_identity": hash_text(QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY),
        "prefix_token_hash": spec.prefix_token_hash if spec is not None else None,
        "prefix_text_hash": spec.invariant_text_hash if spec is not None else None,
    }


def ornith_route_prefix_reuse_status(client: "NativeLlamaClient") -> dict[str, object]:
    status = client._ornith_route_prefix_status
    profile = getattr(client, "model_profile", None)
    profile_eligible = (
        getattr(profile, "profile_id", None) == ORNITH15_PROFILE_ID
        and getattr(profile, "verified", False)
        and client._model_metadata_identity.get("general.file_type") == "15"
    )
    spec = client._ornith_route_prefix_spec
    return {
        "enabled": client.config.ornith_route_prefix_reuse_enabled and profile_eligible,
        "source": client.config.ornith_route_prefix_reuse_source,
        "config_error": client.config.ornith_route_prefix_reuse_config_error,
        "initialized": status.initialized,
        "prefix_tokens": status.prefix_tokens,
        "capture_count": status.capture_count,
        "restore_count": status.restore_count,
        "fallback_count": status.fallback_count,
        "invalidation_count": status.invalidation_count,
        "failure_reason": status.failure_reason,
        "last_used": status.last_used,
        "checkpoint_size_bytes": client._ornith_route_prefix_anchor_state.checkpoint_size,
        "profile_identity": getattr(profile, "profile_id", None),
        "template_identity": getattr(profile, "template_sha256", None),
        "tokenizer_identity": hash_text(ORNITH_ROUTE_TOKENIZER_IDENTITY),
        "prefix_token_hash": spec.prefix_token_hash if spec is not None else None,
        "prefix_text_hash": spec.invariant_text_hash if spec is not None else None,
    }


def qwen36_shell_tool_prefix_reuse_status(client: "NativeLlamaClient") -> dict[str, object]:
    with client._qwen36_shell_tool_prefix_lock:
        status = client._qwen36_shell_tool_prefix_status
        profile = getattr(client, "model_profile", None)
        profile_eligible = (
            getattr(profile, "profile_id", None) == QWEN36_PROFILE_ID
            and getattr(profile, "verified", False)
            and client._model_metadata_identity.get("general.file_type") == "15"
        )
        spec = client._qwen36_shell_tool_prefix_spec
        return {
            "enabled": client.config.qwen36_shell_tool_prefix_reuse_enabled and profile_eligible,
            "source": client.config.qwen36_shell_tool_prefix_reuse_source,
            "config_error": client.config.qwen36_shell_tool_prefix_reuse_config_error,
            "initialized": status.initialized,
            "prefix_tokens": status.prefix_tokens,
            "capture_count": status.capture_count,
            "restore_count": status.restore_count,
            "fallback_count": status.fallback_count,
            "invalidation_count": status.invalidation_count,
            "failure_reason": status.failure_reason,
            "last_used": status.last_used,
            "checkpoint_size_bytes": client._qwen36_shell_tool_prefix_anchor_state.checkpoint_size,
            "checkpoint_identity": QWEN36_SHELL_TOOL_PREFIX_FORMAT_VERSION,
            "profile_identity": getattr(profile, "profile_id", None),
            "template_identity": getattr(profile, "template_sha256", None),
            "tokenizer_identity": hash_text(QWEN36_SHELL_TOOL_TOKENIZER_IDENTITY),
            "prefix_token_hash": spec.prefix_token_hash if spec is not None else None,
            "prefix_text_hash": spec.invariant_text_hash if spec is not None else None,
            "tool_schema_hash": spec.tool_schema_hash if spec is not None else None,
        }


def compatibility_diagnostics(client: "NativeLlamaClient") -> dict[str, object]:
    profile = getattr(client, "model_profile", None)
    if profile is None:
        return {
            "model_family": "unknown",
            "compatibility_profile": "uninitialized",
            "verified": False,
            "failure_reason": "model_profile_uninitialized",
        }
    diagnostics = profile.diagnostics(thinking_enabled=client.config.thinking)
    diagnostics["chat_bridge_loaded"] = client.chat_bridge is not None
    diagnostics["chat_bridge_revision_bound"] = bool(
        client.chat_bridge is not None and client.chat_bridge.build_identity
    )
    return diagnostics


def model_load_status(client: "NativeLlamaClient") -> dict[str, bool | int | None]:
    semantics = getattr(client, "_model_load_semantics", None) or {}
    return {
        "low_memory": client.config.low_memory,
        "cpu_repack": client._cpu_repack_enabled,
        # llama_model_params values the model was (or will be) loaded with.
        "load_mode": semantics.get("load_mode"),
        "lazy_mode": semantics.get("lazy_mode"),
        "load_mtp": semantics.get("load_mtp"),
    }


def moe_expert_usage_status(client: "NativeLlamaClient") -> dict[str, object]:
    base = {
        "enabled": client.config.moe_expert_usage_enabled,
        "available": client.lib.expert_usage_available,
        "counter_storage_bytes": client.lib.expert_usage_storage_size(),
        "scope": "process_cpu_backend",
    }
    if not client.config.moe_expert_usage_enabled:
        return base
    shape = client._moe_expert_usage_shape()
    if shape is None:
        return {**base, "error": "model_moe_metadata_unavailable"}
    counts, tokens = client.lib.expert_usage_snapshot()
    return {
        **base, "architecture": shape[0],
        **summarize_expert_usage(counts, tokens, layers=shape[1], experts=shape[2], active=shape[3]),
    }
