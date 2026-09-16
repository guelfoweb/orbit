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
from typing import Any, Callable

ROLLING_ROUTE_STRATEGY_ID = "ornith15-rolling-route-v1"

# The analysis lineage reuses every primitive below unchanged -- the checkpoint
# format, the exact-prefix rule, the capture precondition -- and differs only in
# which conversation the saved tokens belong to. Keeping that difference in the
# identity's `strategy_id` is what makes a CHAT checkpoint and an ANALYSIS
# checkpoint mutually unusable: identities are compared whole, so neither can
# ever satisfy the other's reuse check, without a second cache or a mode flag
# reaching the backend.
ROLLING_ANALYSIS_STRATEGY_ID = "ornith15-rolling-analysis-v1"

# The analysis STEP turn keeps a checkpoint of its own. The structured
# controller renders a STEP as the committed history followed by one transient
# user turn -- the per-question guidance -- and a FINISH control turn runs
# between any two STEPs. Under one shared analysis slot the FINISH evicted the
# STEP checkpoint every time, so no STEP ever restored: measured on the
# retained traces as every STEP after the first prefilling 1.0-1.6k tokens
# its predecessor had already decoded. A distinct strategy id gives the STEP
# its own slot through the same store, and -- because identities are compared
# whole -- makes a STEP checkpoint and a control checkpoint mutually unusable.
ROLLING_STEP_STRATEGY_ID = "ornith15-rolling-analysis-step-v1"

# The control lineage's second checkpoint: the history BEFORE a control turn's
# own transient user turn(s). Stage A's control checkpoint sits after the
# completion message and before the assistant opener, which is exactly what
# a repair extends -- and exactly what the NEXT FINISH does not: measured on
# the normalized Fattura replay, every later FINISH shares its predecessor's
# tokens only up to the previous completion message (1581 of 2542, 2337 of
# 2751), so each one prefilled cold, 42% of all prefill. A checkpoint taken
# before that message is a strict prefix of the next FINISH whatever question
# it closes, because the history between them is append-only. It lives in its
# own slot so the repair keeps Stage A's longer checkpoint untouched.
ROLLING_CONTROL_HISTORY_STRATEGY_ID = "ornith15-rolling-control-history-v1"

# The CHAT route lineage's second checkpoint: the route checkpoint advanced
# past the assistant reply the final call just committed. A turn is two calls
# with different fixed heads -- the route prompt and the tools-free final --
# so the final's own state can never serve the next route prompt, and the
# next route prompt re-evaluated the whole reply (measured: 498 of 524
# evaluated tokens on a ~2.6k-character answer). The shadow is that reply
# decoded in ROUTE context on top of the restored route checkpoint, after the
# final's output has already been delivered. It lives in its own slot so the
# route checkpoint it extends stays available: a next route prompt that does
# not extend the shadow (the runtime committed different text, the history
# was compacted) still meets the checkpoint it would have met without it.
ROLLING_ROUTE_SHADOW_STRATEGY_ID = "ornith15-rolling-route-shadow-v1"


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
    # Consecutive same-identity route prompts that neither extended this
    # checkpoint nor replaced it, and the tokens of the latest one. See
    # `rolling_route_should_replace`.
    non_extending_misses: int = field(default=0, compare=False)
    last_miss_tokens: list[int] = field(default_factory=list, compare=False, repr=False)
    # The route lineage only: the messages and tools the checkpointed prompt
    # was rendered from, so the post-final shadow can render the same history
    # extended by the committed reply through the same renderer. None on
    # every other lineage and on a checkpoint taken without them, which means
    # no shadow can be built from it.
    render_messages: list[dict[str, Any]] | None = field(default=None, compare=False, repr=False)
    render_tools: list[dict[str, Any]] | None = field(default=None, compare=False, repr=False)

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


def rolling_capture_boundary(
    prompt: str,
    prompt_tokens: list[int],
    *,
    generation_prompt: str | None,
    tokenize: Callable[[str], list[int]],
) -> int | None:
    """How many leading tokens the NEXT turn can still extend, or None.

    A checkpoint is only worth keeping if a later prompt can be a strict
    extension of it. The rendered prompt ends with the generation prompt -- the
    text that opens the assistant turn the model is about to write -- and that
    tail is precisely what the next prompt does NOT repeat: a repair appends a
    user turn in its place, and a continuation renders the assistant's actual
    reply there. Measured on the retained ANALYSIS traces, every same-phase
    miss diverged at exactly that point, with the saved tokens continuing into
    the assistant role opener and the new prompt into the user's. A checkpoint
    that includes the generation prompt can therefore never be restored by a
    same-phase successor, whatever else matches.

    The boundary is the prompt with that tail removed, re-tokenized, and
    accepted only if those tokens are a strict prefix of the full prompt's --
    a boundary that lands inside a token is refused rather than rounded. Every
    refusal returns None, and None means the caller keeps its existing
    behaviour; nothing here can make a checkpoint LESS safe, only shorter.
    """
    if not generation_prompt or not prompt.endswith(generation_prompt):
        return None
    head = prompt[: len(prompt) - len(generation_prompt)]
    if not head:
        return None
    head_tokens = tokenize(head)
    n = len(head_tokens)
    if n <= 0 or n >= len(prompt_tokens):
        return None
    if prompt_tokens[:n] != head_tokens:
        return None
    return n


def rolling_step_boundary(
    prompt: str,
    prompt_tokens: list[int],
    *,
    head: str | None,
    tokenize: Callable[[str], list[int]],
) -> int | None:
    """How many leading tokens the NEXT step can still extend, or None.

    `head` is the rendered prompt up to -- not including -- the transient
    user turn that closes a STEP prompt: the controller's per-question
    guidance, which the next STEP replaces rather than repeats. Everything
    before it is the committed history, which is append-only, so the next
    STEP prompt begins with exactly this text followed by whatever the turn
    in between appended. The caller renders that head through the same
    renderer as the prompt; this only decides whether it is a boundary.

    Accepted only when the head is literally how the prompt begins, and its
    tokens are a strict prefix of the prompt's tokens: a head whose render
    differs from the prompt's opening -- a template that treats its last turn
    specially, a rewrite of history -- and a boundary that lands inside a
    token are both refused rather than rounded. Every refusal returns None,
    and None means no STEP checkpoint is taken on this call; it can never
    make a checkpoint less safe, only absent.
    """
    if not head or not prompt.startswith(head) or len(head) >= len(prompt):
        return None
    head_tokens = tokenize(head)
    n = len(head_tokens)
    if n <= 0 or n >= len(prompt_tokens):
        return None
    if prompt_tokens[:n] != head_tokens:
        return None
    return n


# Two user turns that differ in their first character. The head shared by
# their renders is, by construction, everything the template emits before the
# user's text -- whatever the template is.
SHADOW_SENTINEL_USER_TURNS = ("a", "9")


def rolling_shadow_head(
    checkpoint_tokens: list[int],
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    assistant_content: str,
    render: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], str],
    tokenize: Callable[[str], list[int]],
) -> tuple[list[int] | None, str]:
    """The tokens every next route prompt starts with, or (None, why not).

    The next route prompt renders `messages` + the committed assistant reply
    + the user's next turn + the generation prompt. Its head -- everything
    before the user's text -- is what can be decoded now, and it is found
    without knowing the template: the same history is rendered with two
    sentinel user turns that differ in their first character, and the text
    both renders share is exactly that head. Nothing is guessed about how
    the template opens a user turn or closes an assistant one.

    Accepted only when its tokens (1) still begin with the checkpoint's own
    tokens -- a template that renders history differently once another turn
    follows would break the chain, and then no shadow is taken -- (2) extend
    them, and (3) are a strict token-prefix of BOTH sentinel renders'
    tokenizations, so the boundary is not inside a token and does not depend
    on what the user types next. Every refusal returns None; None means the
    route checkpoint stays exactly as it is.

    Note that (3) is checked against the sentinels, not against every
    possible user turn: a turn whose first characters could merge with the
    opener's last token under the tokenizer would make the next prompt miss
    the shadow and fall back to the route checkpoint -- a cost, never a
    wrong state. On the qualified template user content is trimmed before
    it is rendered, so no user turn starts with whitespace that could merge
    with the opener's trailing newline.
    """
    if not checkpoint_tokens:
        return None, "no_checkpoint"
    if not assistant_content or not assistant_content.strip():
        return None, "empty_assistant_content"
    history = [*[dict(message) for message in messages], {"role": "assistant", "content": assistant_content}]
    try:
        renders = [
            render([*history, {"role": "user", "content": sentinel}], tools)
            for sentinel in SHADOW_SENTINEL_USER_TURNS
        ]
    except Exception:
        return None, "render_failed"
    head_text = _common_text_prefix(renders[0], renders[1])
    if not head_text:
        return None, "empty_head"
    try:
        head_tokens = tokenize(head_text)
        full_tokens = [tokenize(text) for text in renders]
    except Exception:
        return None, "tokenize_failed"
    n_checkpoint = len(checkpoint_tokens)
    if len(head_tokens) <= n_checkpoint:
        return None, "head_does_not_extend_checkpoint"
    if head_tokens[:n_checkpoint] != checkpoint_tokens:
        return None, "checkpoint_not_a_prefix_of_head"
    for tokens in full_tokens:
        if len(tokens) <= len(head_tokens) or tokens[: len(head_tokens)] != head_tokens:
            return None, "head_not_a_token_prefix"
    return head_tokens, "ok"


def _common_text_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def rolling_route_should_replace(
    state: RollingRouteAnchorState,
    prompt_tokens: list[int],
    identity: RollingRouteIdentity,
) -> bool:
    """Whether this prefill should become the new checkpoint.

    A prompt continuing the tracked chain replaces it. A prompt that does not
    (a post-tool route rendering that diverges before the end, a conversation
    reset or compacted on the same session) does not evict it on the first
    miss: the next turn's route prompt usually still extends the saved chain,
    and keeping the older useful checkpoint beats keeping the most recent
    one. It does on a later miss when that prompt extends the latest missed
    one (`last_miss_tokens`, recorded by the capture site on every miss;
    `non_extending_misses` counts them): two route prompts in a row that
    build on each other but not on the checkpoint are a new chain, and
    holding on to the old one would leave that conversation cold for the
    rest of the server's life.
    Misses that do not build on each other (the post-tool window regime,
    where every route prompt is `[system, latest user, evidence]`) never
    replace: nothing they capture could be restored. Nothing here authorizes
    reuse -- that stays with `rolling_route_reuse_start`.
    """
    if not state.valid:
        return True
    if state.identity != identity:
        return True
    if rolling_route_reuse_start(state, prompt_tokens, identity) is not None:
        return True
    return state.non_extending_misses >= 1 and _strictly_extends(prompt_tokens, state.last_miss_tokens)


def _strictly_extends(prompt_tokens: list[int], head: list[int]) -> bool:
    return bool(head) and len(prompt_tokens) > len(head) and prompt_tokens[: len(head)] == head


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
