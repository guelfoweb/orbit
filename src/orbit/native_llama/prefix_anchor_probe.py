from __future__ import annotations

from ctypes import POINTER, c_float, cast, pointer
from dataclasses import dataclass
import hashlib
from typing import Any, Callable

from .bindings import llama_pos, llama_seq_id, llama_token
from .chat_template import RoutePromptSegments
from .prefix_anchor import (
    capture_prefix_anchor,
    compute_prefix_anchor_key,
    restore_prefix_anchor,
)


TokenizeFn = Callable[[str], list[int]]


@dataclass(frozen=True)
class PrefixAnchorProbeResult:
    ok: bool
    reason: str | None
    prefix_token_count: int
    suffix_token_count: int
    full_token_count: int
    checkpoint_size: int
    restore_used: bool
    baseline_next_token: int | None
    restored_next_token: int | None
    logits_hash_baseline: str | None
    logits_hash_restored: str | None
    logits_match: bool | None
    seq_id: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "probe_ok": self.ok,
            "reason": self.reason,
            "prefix_token_count": self.prefix_token_count,
            "suffix_token_count": self.suffix_token_count,
            "full_token_count": self.full_token_count,
            "checkpoint_size": self.checkpoint_size,
            "restore_used": self.restore_used,
            "baseline_next_token": self.baseline_next_token,
            "restored_next_token": self.restored_next_token,
            "logits_hash_baseline": self.logits_hash_baseline,
            "logits_hash_restored": self.logits_hash_restored,
            "logits_match": self.logits_match,
            "seq_id": self.seq_id,
        }


@dataclass(frozen=True)
class RouteBoundaryTokenProbeResult:
    token_prefix_ok: bool
    reason: str | None
    stable_prefix_token_count: int
    full_prompt_token_count: int
    token_lcp_with_stable_prefix: int
    divergence_index: int | None
    stable_prefix_hash: str
    full_prompt_hash: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "route_boundary_token_prefix_ok": self.token_prefix_ok,
            "reason": self.reason,
            "stable_prefix_token_count": self.stable_prefix_token_count,
            "full_prompt_token_count": self.full_prompt_token_count,
            "token_lcp_with_stable_prefix": self.token_lcp_with_stable_prefix,
            "divergence_index": self.divergence_index,
            "stable_prefix_hash": self.stable_prefix_hash,
            "full_prompt_hash": self.full_prompt_hash,
        }


@dataclass(frozen=True)
class PrefixPrefillOnlyProbeResult:
    ok: bool
    reason: str | None
    prefix_hash: str
    prefix_token_count: int
    checkpoint_size: int
    prefill_ms: float
    decode_calls: int
    sampled_tokens: int = 0
    generated_tokens: int = 0
    sampler_touched: bool = False
    session_history_touched: bool = False
    seq_id: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "probe_ok": self.ok,
            "reason": self.reason,
            "prefix_hash": self.prefix_hash,
            "prefix_token_count": self.prefix_token_count,
            "checkpoint_size_bytes": self.checkpoint_size,
            "prefill_ms": self.prefill_ms,
            "decode_calls": self.decode_calls,
            "sampled_tokens": self.sampled_tokens,
            "generated_tokens": self.generated_tokens,
            "sampler_touched": self.sampler_touched,
            "session_history_touched": self.session_history_touched,
            "seq_id": self.seq_id,
        }


def probe_route_boundary_token_prefix(
    *,
    segments: RoutePromptSegments,
    tokenize: TokenizeFn,
) -> RouteBoundaryTokenProbeResult:
    stable_tokens = list(tokenize(segments.stable_prefix_text))
    full_tokens = list(tokenize(segments.full_prompt_text))
    lcp = _longest_common_prefix(stable_tokens, full_tokens)
    reason = None
    if not stable_tokens:
        reason = "empty_stable_prefix_tokens"
    elif len(full_tokens) < len(stable_tokens):
        reason = "full_shorter_than_stable_prefix"
    elif lcp != len(stable_tokens):
        reason = "stable_prefix_not_token_prefix"
    return RouteBoundaryTokenProbeResult(
        token_prefix_ok=reason is None,
        reason=reason,
        stable_prefix_token_count=len(stable_tokens),
        full_prompt_token_count=len(full_tokens),
        token_lcp_with_stable_prefix=lcp,
        divergence_index=None if reason is None else lcp,
        stable_prefix_hash=segments.stable_prefix_hash,
        full_prompt_hash=segments.full_prompt_hash,
    )


def split_prompt_by_token_prefix(
    *,
    tokenize: TokenizeFn,
    prefix_text: str,
    full_text: str,
) -> tuple[list[int], list[int], list[int], str | None]:
    prefix_tokens = list(tokenize(prefix_text))
    full_tokens = list(tokenize(full_text))
    if not prefix_tokens:
        return prefix_tokens, [], full_tokens, "empty_prefix_tokens"
    if len(full_tokens) < len(prefix_tokens):
        return prefix_tokens, [], full_tokens, "full_shorter_than_prefix"
    if full_tokens[: len(prefix_tokens)] != prefix_tokens:
        return prefix_tokens, [], full_tokens, "prefix_not_token_prefix"
    suffix_tokens = full_tokens[len(prefix_tokens) :]
    return prefix_tokens, suffix_tokens, full_tokens, None


def probe_route_prefix_prefill_only(
    *,
    lib: Any,
    ctx: Any,
    tokenize: TokenizeFn,
    prefix_text: str,
    prefix_hash: str,
    seq_id: int = 0,
    model_id: str | None = None,
    template_id: str | None = None,
    tool_schema_hash: str | None = None,
    capability_summary_hash: str | None = None,
    runtime_policy_hash: str | None = "probe-runtime-policy",
    route_contract_hash: str | None = "probe-route-contract",
    backend_version: str | None = None,
    native_version: str | None = None,
    tools_mode: str | None = "tools-on-route",
    decode_step: int = 256,
) -> PrefixPrefillOnlyProbeResult:
    prefix_tokens = list(tokenize(prefix_text))
    if not prefix_tokens:
        return PrefixPrefillOnlyProbeResult(
            ok=False,
            reason="empty_prefix_tokens",
            prefix_hash=prefix_hash,
            prefix_token_count=0,
            checkpoint_size=0,
            prefill_ms=0.0,
            decode_calls=0,
            seq_id=seq_id,
        )

    anchor_key = compute_prefix_anchor_key(
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
    anchor_kwargs = {
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

    _clear_context(lib, ctx)
    start_us = int(lib.llama_time_us()) if hasattr(lib, "llama_time_us") else 0
    decode_calls = 0
    step = max(1, decode_step)
    try:
        for offset in range(0, len(prefix_tokens), step):
            chunk = prefix_tokens[offset : offset + step]
            _decode_tokens(lib, ctx, chunk, seq_id=seq_id, start_pos=offset)
            decode_calls += 1
    except Exception as exc:
        return PrefixPrefillOnlyProbeResult(
            ok=False,
            reason=f"prefix_decode_failed:{type(exc).__name__}",
            prefix_hash=prefix_hash,
            prefix_token_count=len(prefix_tokens),
            checkpoint_size=0,
            prefill_ms=0.0,
            decode_calls=max(1, decode_calls),
            seq_id=seq_id,
        )
    end_us = int(lib.llama_time_us()) if hasattr(lib, "llama_time_us") else start_us
    prefill_ms = max(0.0, (end_us - start_us) / 1000.0)

    state, capture_meta = capture_prefix_anchor(
        lib=lib,
        ctx=ctx,
        seq_id=seq_id,
        prefix_hash=anchor_key,
        token_count=len(prefix_tokens),
        enabled=True,
        **anchor_kwargs,
    )
    if not state.valid:
        return PrefixPrefillOnlyProbeResult(
            ok=False,
            reason=str(capture_meta.get("fallback_reason") or state.invalidation_reason or "capture_failed"),
            prefix_hash=prefix_hash,
            prefix_token_count=len(prefix_tokens),
            checkpoint_size=0,
            prefill_ms=prefill_ms,
            decode_calls=decode_calls,
            seq_id=seq_id,
        )

    return PrefixPrefillOnlyProbeResult(
        ok=True,
        reason=None,
        prefix_hash=prefix_hash,
        prefix_token_count=len(prefix_tokens),
        checkpoint_size=state.checkpoint_size,
        prefill_ms=prefill_ms,
        decode_calls=decode_calls,
        sampled_tokens=0,
        generated_tokens=0,
        sampler_touched=False,
        session_history_touched=False,
        seq_id=seq_id,
    )


def _longest_common_prefix(first: list[int], second: list[int]) -> int:
    common = 0
    max_common = min(len(first), len(second))
    while common < max_common and first[common] == second[common]:
        common += 1
    return common


def probe_prefix_anchor_equivalence(
    *,
    lib: Any,
    ctx: Any,
    vocab: Any,
    sampler: Any,
    tokenize: TokenizeFn,
    prefix_text: str,
    full_text: str,
    seq_id: int = 0,
    model_id: str | None = None,
    template_id: str | None = None,
    tool_schema_hash: str | None = None,
    capability_summary_hash: str | None = None,
    runtime_policy_hash: str | None = "probe-runtime-policy",
    route_contract_hash: str | None = "probe-route-contract",
    backend_version: str | None = None,
    native_version: str | None = None,
    tools_mode: str | None = "tools-on-route",
) -> PrefixAnchorProbeResult:
    prefix_tokens, suffix_tokens, full_tokens, split_reason = split_prompt_by_token_prefix(
        tokenize=tokenize,
        prefix_text=prefix_text,
        full_text=full_text,
    )
    if split_reason is not None:
        return PrefixAnchorProbeResult(
            ok=False,
            reason=split_reason,
            prefix_token_count=len(prefix_tokens),
            suffix_token_count=len(suffix_tokens),
            full_token_count=len(full_tokens),
            checkpoint_size=0,
            restore_used=False,
            baseline_next_token=None,
            restored_next_token=None,
            logits_hash_baseline=None,
            logits_hash_restored=None,
            logits_match=None,
            seq_id=seq_id,
        )
    if not suffix_tokens:
        return PrefixAnchorProbeResult(
            ok=False,
            reason="empty_suffix_tokens",
            prefix_token_count=len(prefix_tokens),
            suffix_token_count=0,
            full_token_count=len(full_tokens),
            checkpoint_size=0,
            restore_used=False,
            baseline_next_token=None,
            restored_next_token=None,
            logits_hash_baseline=None,
            logits_hash_restored=None,
            logits_match=None,
            seq_id=seq_id,
        )

    prefix_hash = compute_prefix_anchor_key(
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
    anchor_kwargs = {
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

    _clear_context(lib, ctx)
    _decode_tokens(lib, ctx, prefix_tokens, seq_id=seq_id, start_pos=0)
    state, capture_meta = capture_prefix_anchor(
        lib=lib,
        ctx=ctx,
        seq_id=seq_id,
        prefix_hash=prefix_hash,
        token_count=len(prefix_tokens),
        enabled=True,
        **anchor_kwargs,
    )
    if not state.valid:
        return PrefixAnchorProbeResult(
            ok=False,
            reason=capture_meta.get("fallback_reason") or state.invalidation_reason or "capture_failed",
            prefix_token_count=len(prefix_tokens),
            suffix_token_count=len(suffix_tokens),
            full_token_count=len(full_tokens),
            checkpoint_size=0,
            restore_used=False,
            baseline_next_token=None,
            restored_next_token=None,
            logits_hash_baseline=None,
            logits_hash_restored=None,
            logits_match=None,
            seq_id=seq_id,
        )

    _decode_tokens(lib, ctx, suffix_tokens, seq_id=seq_id, start_pos=len(prefix_tokens))
    baseline_next_token = _sample_next_token(lib, sampler, ctx)
    logits_hash_baseline = _logits_hash(lib, ctx, vocab)

    _clear_context(lib, ctx)
    ok, restored_state, restore_meta = restore_prefix_anchor(
        state,
        lib=lib,
        ctx=ctx,
        seq_id=seq_id,
        prefix_hash=prefix_hash,
        enabled=True,
        **anchor_kwargs,
    )
    if not ok:
        return PrefixAnchorProbeResult(
            ok=False,
            reason=restore_meta.get("fallback_reason") or restored_state.invalidation_reason or "restore_failed",
            prefix_token_count=len(prefix_tokens),
            suffix_token_count=len(suffix_tokens),
            full_token_count=len(full_tokens),
            checkpoint_size=state.checkpoint_size,
            restore_used=False,
            baseline_next_token=baseline_next_token,
            restored_next_token=None,
            logits_hash_baseline=logits_hash_baseline,
            logits_hash_restored=None,
            logits_match=None,
            seq_id=seq_id,
        )

    _decode_tokens(lib, ctx, suffix_tokens, seq_id=seq_id, start_pos=len(prefix_tokens))
    restored_next_token = _sample_next_token(lib, sampler, ctx)
    logits_hash_restored = _logits_hash(lib, ctx, vocab)
    logits_match = logits_hash_baseline == logits_hash_restored
    return PrefixAnchorProbeResult(
        ok=(baseline_next_token == restored_next_token and logits_match is not False),
        reason=None if (baseline_next_token == restored_next_token and logits_match is not False) else "equivalence_mismatch",
        prefix_token_count=len(prefix_tokens),
        suffix_token_count=len(suffix_tokens),
        full_token_count=len(full_tokens),
        checkpoint_size=state.checkpoint_size,
        restore_used=bool(restore_meta.get("restore_used")),
        baseline_next_token=baseline_next_token,
        restored_next_token=restored_next_token,
        logits_hash_baseline=logits_hash_baseline,
        logits_hash_restored=logits_hash_restored,
        logits_match=logits_match,
        seq_id=seq_id,
    )


def _clear_context(lib: Any, ctx: Any) -> None:
    mem = lib.llama_get_memory(ctx)
    if mem:
        lib.llama_memory_clear(mem, True)


def _decode_tokens(
    lib: Any,
    ctx: Any,
    tokens: list[int],
    *,
    seq_id: int,
    start_pos: int,
) -> None:
    if not tokens:
        return
    if seq_id == 0 and hasattr(lib, "llama_batch_get_one"):
        token_array = (llama_token * len(tokens))(*tokens)
        batch = lib.llama_batch_get_one(token_array, len(tokens))
        rc = int(lib.llama_decode(ctx, batch))
        if rc != 0:
            raise RuntimeError(f"llama_decode failed in prefix-anchor probe: {rc}")
        if hasattr(lib, "llama_synchronize"):
            lib.llama_synchronize(ctx)
        return
    batch = lib.llama_batch_init(len(tokens), 0, 1)
    batch.n_tokens = len(tokens)
    seq_value = llama_seq_id(seq_id)
    seq_ptr = cast(pointer(seq_value), POINTER(llama_seq_id))
    try:
        for index, token in enumerate(tokens):
            batch.token[index] = llama_token(token)
            batch.pos[index] = llama_pos(start_pos + index)
            batch.n_seq_id[index] = 1
            batch.seq_id[index] = seq_ptr
            batch.logits[index] = 1 if index == (len(tokens) - 1) else 0
        rc = int(lib.llama_decode(ctx, batch))
        if rc != 0:
            raise RuntimeError(f"llama_decode failed in prefix-anchor probe: {rc}")
        if hasattr(lib, "llama_synchronize"):
            lib.llama_synchronize(ctx)
    finally:
        lib.llama_batch_free(batch)


def _sample_next_token(lib: Any, sampler: Any, ctx: Any) -> int:
    lib.llama_sampler_reset(sampler)
    token = int(lib.llama_sampler_sample(sampler, ctx, -1))
    return token


def _logits_hash(lib: Any, ctx: Any, vocab: Any) -> str | None:
    if not hasattr(lib, "llama_get_logits_ith") or not hasattr(lib, "llama_vocab_n_tokens"):
        return None
    vocab_size = int(lib.llama_vocab_n_tokens(vocab))
    if vocab_size <= 0:
        return None
    ptr = lib.llama_get_logits_ith(ctx, -1)
    if not ptr:
        return None
    row_type = c_float * vocab_size
    row = cast(ptr, POINTER(row_type)).contents
    return hashlib.sha256(bytes(row)).hexdigest()[:32]
