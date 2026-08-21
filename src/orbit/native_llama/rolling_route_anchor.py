"""A route checkpoint that rolls forward with the conversation.

The existing prefix anchors cache a prefix whose content never changes, so they
authorize reuse by comparing hashes for equality. A conversational route prompt
is different: every turn appends to the previous one, so the useful checkpoint
is the whole prompt as it stood at the end of the last route prefill, and reuse
is authorized precisely when the *new* prompt differs -- by extending it.

That inverted condition is why this state lives apart from `PrefixAnchorState`
rather than inside it. Equality of hashes is the wrong question here, so the
saved tokens are kept in full and compared exactly. A hash may reject a
candidate early, but nothing is ever authorized on a hash alone: a collision
would restore a KV sequence that does not match the prompt, and the model would
answer confidently from someone else's context.

The checkpoint holds the prompt only. Generated tokens are never serialized --
a snapshot containing them is not a prefix of the next route prompt, so it
would be useless at best and wrong at worst.
"""

from __future__ import annotations

import time
from ctypes import c_ubyte
from dataclasses import dataclass, field, replace
from typing import Any

ROLLING_ROUTE_STRATEGY_ID = "ornith15-rolling-route-v1"


@dataclass(frozen=True)
class RollingRouteIdentity:
    """What must be unchanged for saved tokens to still mean the same thing.

    Content identity lives in the token sequence itself, not here; these are
    the surrounding facts that make those tokens interpretable.
    """

    strategy_id: str
    session_id: str
    profile_id: str
    model_id: str
    template_id: str
    tool_schema_hash: str
    capability_summary_hash: str
    runtime_policy_hash: str
    native_version: str
    tools_mode: str
    reset_generation: int


@dataclass
class RollingRouteAnchorState:
    """KV as it stood at the end of a route prefill, with its exact tokens."""

    identity: RollingRouteIdentity | None = None
    tokens: list[int] = field(default_factory=list)
    checkpoint_data: bytes | None = field(default=None, repr=False, compare=False)
    created_at_monotonic: float | None = None
    invalidation_reason: str | None = None

    @property
    def valid(self) -> bool:
        return bool(self.identity is not None and self.tokens and self.checkpoint_data)

    @property
    def checkpoint_size(self) -> int:
        return len(self.checkpoint_data or b"")


def invalidate_rolling_route_anchor(
    state: RollingRouteAnchorState, reason: str
) -> RollingRouteAnchorState:
    """Drop the checkpoint. Cheap to rebuild, unsafe to keep when in doubt."""
    return RollingRouteAnchorState(invalidation_reason=reason)


def rolling_route_reuse_start(
    state: RollingRouteAnchorState,
    prompt_tokens: list[int],
    identity: RollingRouteIdentity,
) -> int | None:
    """Tokens already resident if this checkpoint serves this prompt, else None.

    Exact prefix only, and the prompt must extend the saved tokens: an
    equal-length prompt leaves nothing to evaluate, and the final prompt token
    still has to be decoded to produce fresh logits.
    """
    if not state.valid or state.identity != identity:
        return None
    saved = state.tokens
    if len(prompt_tokens) <= len(saved):
        return None
    if prompt_tokens[: len(saved)] != saved:
        return None
    return len(saved)


def rolling_route_should_replace(
    state: RollingRouteAnchorState,
    prompt_tokens: list[int],
    identity: RollingRouteIdentity,
) -> bool:
    """Whether this prefill should become the new checkpoint.

    Only a prompt continuing the tracked chain replaces it. This is what lets
    the checkpoint survive the final call that wipes live KV: the final prompt
    shares almost nothing with the route chain, so it must not evict the state
    the next route depends on. Keeping the older useful checkpoint beats
    keeping the most recent one.
    """
    if not state.valid:
        return True
    if state.identity != identity:
        return True
    return rolling_route_reuse_start(state, prompt_tokens, identity) is not None


def capture_rolling_route_anchor(
    lib: Any,
    ctx: Any,
    *,
    prompt_tokens: list[int],
    identity: RollingRouteIdentity,
    seq_id: int = 0,
) -> tuple[RollingRouteAnchorState, dict[str, Any]]:
    """Snapshot the sequence exactly as the route prefill left it.

    Called at the prefill boundary before a single token is generated, so the
    saved tokens are exactly `prompt_tokens`.
    """
    metadata: dict[str, Any] = {"capture_attempted": True}
    if not ctx or not prompt_tokens:
        metadata["fallback_reason"] = "no_context_or_tokens"
        return invalidate_rolling_route_anchor(
            RollingRouteAnchorState(), "no_context_or_tokens"
        ), metadata
    try:
        size = int(lib.llama_state_seq_get_size(ctx, seq_id))
    except Exception:
        metadata["fallback_reason"] = "checkpoint_size_failed"
        return invalidate_rolling_route_anchor(
            RollingRouteAnchorState(), "checkpoint_size_failed"
        ), metadata
    if size <= 0:
        metadata["fallback_reason"] = "empty_checkpoint"
        return invalidate_rolling_route_anchor(
            RollingRouteAnchorState(), "empty_checkpoint"
        ), metadata
    try:
        buffer = (c_ubyte * size)()
        written = int(lib.llama_state_seq_get_data(ctx, buffer, size, seq_id))
    except Exception:
        metadata["fallback_reason"] = "checkpoint_capture_failed"
        return invalidate_rolling_route_anchor(
            RollingRouteAnchorState(), "checkpoint_capture_failed"
        ), metadata
    if written != size:
        metadata["fallback_reason"] = "checkpoint_size_mismatch"
        return invalidate_rolling_route_anchor(
            RollingRouteAnchorState(), "checkpoint_size_mismatch"
        ), metadata
    metadata["checkpoint_size_bytes"] = size
    metadata["checkpoint_tokens"] = len(prompt_tokens)
    return (
        RollingRouteAnchorState(
            identity=identity,
            tokens=list(prompt_tokens),
            checkpoint_data=bytes(buffer),
            created_at_monotonic=time.monotonic(),
        ),
        metadata,
    )


def restore_rolling_route_anchor(
    lib: Any,
    ctx: Any,
    state: RollingRouteAnchorState,
    *,
    seq_id: int = 0,
) -> tuple[bool, RollingRouteAnchorState, dict[str, Any]]:
    """Put the saved sequence back. Any doubt reports failure and drops it.

    A failed restore may have written part of the sequence, so False means "KV
    is unknown" and the caller must clear before prefilling.
    """
    metadata: dict[str, Any] = {"restore_attempted": True}
    if not state.valid or not ctx:
        metadata["fallback_reason"] = "no_checkpoint"
        return False, invalidate_rolling_route_anchor(state, "no_checkpoint"), metadata
    data = state.checkpoint_data or b""
    try:
        buffer = (c_ubyte * len(data)).from_buffer_copy(data)
        read = int(lib.llama_state_seq_set_data(ctx, buffer, len(data), seq_id))
    except Exception:
        metadata["fallback_reason"] = "checkpoint_restore_failed"
        return False, invalidate_rolling_route_anchor(
            state, "checkpoint_restore_failed"
        ), metadata
    if read != len(data):
        metadata["fallback_reason"] = "checkpoint_restore_size_mismatch"
        return False, invalidate_rolling_route_anchor(
            state, "checkpoint_restore_size_mismatch"
        ), metadata
    metadata["restored_tokens"] = len(state.tokens)
    return True, replace(state), metadata
