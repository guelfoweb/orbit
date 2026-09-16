"""QWEN38-POST-FINAL-ROUTE-CACHE-23: the route checkpoint advances past the
reply the final call committed, while the model is idle.

A CHAT turn is two native calls with different fixed heads: the route prompt
(command system prompt + history) and the tools-free final (chat system prompt
+ history). Measured on the real Qwen3.8 session: the turn-3 route prompt
reused the turn-2 route checkpoint (1107 tokens) and evaluated 524 -- 498 of
them the previous assistant reply, token-identical to what the final call had
generated, plus 26 tokens of turn framing and the user's 12-token question.
The final's own state can never serve the next route prompt (it diverges at
token 3, inside the system prompt), so the reply is decoded in ROUTE context
after the final has been delivered: restore the route checkpoint, prefill the
reply and the template's user-turn opener, capture into the shadow slot.

What is pinned here, over the same hybrid fake memory as the #358 tests (the
recurrent state is a function of the whole history, partial `seq_rm` is
refused, one blob carries KV + recurrent state):

* the head is found template-agnostically (two sentinel user turns) and is
  refused unless the checkpoint is still its exact token prefix and it is a
  token prefix of both sentinel renders;
* a short and a 400+ token reply both advance; the next short user turn then
  reuses the shadow exactly, evaluates only the new turn, and leaves the
  live KV and recurrent state identical to a clean cold decode;
* a preempted shadow is captured at the exact batch boundary it reached and
  the next turn finishes it; nothing decoded is ever mis-recorded;
* a reset, a session change, a failed restore or decode, and a next prompt
  the shadow does not match all fall back to the route checkpoint or to a
  cold prefill -- never to a partial `seq_rm`, never to stale state;
* the server schedules a shadow only for a declared, naturally stopped,
  non-empty, tool-free reply, after the response, and yields to any request.
"""

from __future__ import annotations

import ctypes
import json
import re
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbit.native_llama.client import (
    POST_FINAL_ROUTE_SHADOW_PROFILE_IDS,
    ROLLING_ROUTE_PROFILE_IDS,
    NativeLlamaClient,
)
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID, QWEN38_FLASH_NEXT_PROFILE_ID
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ROUTE_SHADOW_STRATEGY_ID,
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    rolling_shadow_head,
)
from orbit.native_server.app import OrbitNativeServer
from orbit.native_server.protocol import parse_chat_request
from tests.test_qwen38_rolling_route_reuse import HybridLib, _LibHolder, _Profile, fold

SYSTEM = "route system prompt"
TOOLS: list[dict] = []


# --- a ChatML-shaped fake template and a word tokenizer ---------------------

def render(messages: list[dict], tools=None, *, generation_prompt: str = "<A>") -> str:
    tag = {"system": "S", "user": "U", "assistant": "A"}
    text = "".join(f"<{tag[m['role']]}>{m['content']}</{tag[m['role']]}>" for m in messages)
    return text + generation_prompt


_VOCAB: dict[str, int] = {}


def tokenize(text: str) -> list[int]:
    pieces = re.findall(r"</?[A-Z]>|[^\s<]+|<|\s+", text)
    return [_VOCAB.setdefault(piece, 1000 + len(_VOCAB)) for piece in pieces]


class ShadowLib(HybridLib):
    """The #358 hybrid fake plus the prompt-decode entry points the shadow uses."""

    def __init__(self) -> None:
        super().__init__()
        self.decode_calls: list[list[int]] = []
        self.fail_decode = False

    def llama_batch_get_one(self, token_ptr, n):
        return [int(token_ptr[i]) for i in range(n)]

    def llama_decode(self, ctx, batch):
        if self.fail_decode:
            return 1
        self.decode_calls.append(list(batch))
        self.decode(list(batch))
        return 0


class _Config:
    use_mtp_experimental = False
    context_tokens = 4096
    thinking = False
    progress_step = 64
    batch_size = 256
    moe_expert_usage_enabled = False


class _Session:
    def __init__(self) -> None:
        self.ctx_tgt = object()
        self.session_id = "default"
        self.cached_prompt_tokens: list[int] = []
        self.committed_sequence_tokens: list[int] = []
        self.mtp_enabled = False
        self.in_flight = False
        self.continuation_ready = True
        self.cancel_requested = False
        self.prompt_cache_mode: str | None = None


def shadow_client(profile_id: str = QWEN38_FLASH_NEXT_PROFILE_ID) -> tuple[NativeLlamaClient, ShadowLib]:
    lib = ShadowLib()
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    client.lib = _LibHolder(lib)
    client.config = _Config()
    client._session = _Session()
    client._vocab = object()
    client.paths = SimpleNamespace(model="/models/qwen38-flash-next.gguf")
    client._model_metadata_identity = {}
    client._reset_generation = 0
    client.cancel_event = threading.Event()
    client.model_profile = _Profile(profile_id)
    client.apply_chat_template = lambda messages, tools=None, thinking=None: render(messages, tools)  # type: ignore[method-assign]
    client.tokenize = tokenize  # type: ignore[method-assign]
    client._invalidate_committed_sequence = (  # type: ignore[method-assign]
        lambda: client._session.committed_sequence_tokens.clear()
    )
    return client, lib


def route_prompt_tokens(history: list[dict]) -> list[int]:
    return tokenize(render([{"role": "system", "content": SYSTEM}, *history]))


def route_turn(client: NativeLlamaClient, lib: ShadowLib, history: list[dict]) -> list[int]:
    """A route call as the client runs it: restore-or-cold, prefill, capture."""
    messages = [{"role": "system", "content": SYSTEM}, *history]
    prompt = route_prompt_tokens(history)
    identity = client._rolling_route_identity(tools=TOOLS)
    client._rolling_route_identity_cache = identity
    reused = client._prepare_memory_with_ornith_rolling_route_anchor(prompt)
    lib.decode(prompt[reused:])
    client._session.cached_prompt_tokens = list(prompt)
    if client._rolling_route_capture_allowed(prompt, identity):
        client._capture_whole_prompt_checkpoint(prompt, identity, render_messages=messages, render_tools=TOOLS)
    client._session.committed_sequence_tokens = list(prompt)
    return prompt


def final_turn(lib: ShadowLib, history: list[dict], reply: str) -> None:
    """The tools-free final: a different fixed head, so it prefills cold and
    leaves the live state belonging to the final prompt plus the reply."""
    lib.llama_memory_clear("mem", True)
    lib.decode(tokenize(render([{"role": "system", "content": "chat system"}, *history])))
    lib.decode(tokenize(reply))


LONG_REPLY = " ".join(f"word{i}" for i in range(450))


class ShadowHeadTests(unittest.TestCase):
    def test_the_head_is_everything_before_the_next_user_text(self) -> None:
        history = [{"role": "user", "content": "hi"}]
        checkpoint = route_prompt_tokens(history)
        head, reason = rolling_shadow_head(
            checkpoint,
            messages=[{"role": "system", "content": SYSTEM}, *history],
            tools=TOOLS,
            assistant_content="Hello there",
            render=render,
            tokenize=tokenize,
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(head, checkpoint + tokenize("Hello there</A><U>"))
        # ... and every next route prompt starts with it, whatever the user types
        for user in ("What is the capital of France?", "9", "  spaced", "\n\nnewlines", "东京"):
            nxt = route_prompt_tokens([*history, {"role": "assistant", "content": "Hello there"}, {"role": "user", "content": user}])
            self.assertEqual(nxt[: len(head)], head, user)
            self.assertGreater(len(nxt), len(head))

    def test_empty_content_and_missing_checkpoint_are_refused(self) -> None:
        checkpoint = route_prompt_tokens([{"role": "user", "content": "hi"}])
        common = dict(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}], tools=TOOLS, render=render, tokenize=tokenize)
        self.assertEqual(rolling_shadow_head(checkpoint, assistant_content="   ", **common), (None, "empty_assistant_content"))
        self.assertEqual(rolling_shadow_head([], assistant_content="x", **common), (None, "no_checkpoint"))

    def test_a_template_that_rewrites_history_breaks_the_chain_and_is_refused(self) -> None:
        # A renderer that emits a different system line once the history is
        # longer: the checkpoint is no longer a prefix of the head.
        def drifting(messages, tools=None):
            if len(messages) > 3:
                messages = [{"role": "system", "content": "another system"}, *messages[1:]]
            return render(messages, tools)

        history = [{"role": "user", "content": "hi"}]
        checkpoint = tokenize(drifting([{"role": "system", "content": SYSTEM}, *history]))
        head, reason = rolling_shadow_head(
            checkpoint, messages=[{"role": "system", "content": SYSTEM}, *history], tools=TOOLS,
            assistant_content="Hello", render=drifting, tokenize=tokenize,
        )
        self.assertIsNone(head)
        self.assertEqual(reason, "checkpoint_not_a_prefix_of_head")

    def test_a_boundary_inside_a_token_is_refused(self) -> None:
        # A tokenizer that fuses the user opener with the first character of
        # the user's text: the shared text head tokenizes to something that
        # is not a prefix of either full prompt's tokens.
        def fusing(text: str) -> list[int]:
            pieces = re.findall(r"<U>.|</?[A-Z]>|[^\s<]+|<|\s+", text)
            return [_VOCAB.setdefault(piece, 1000 + len(_VOCAB)) for piece in pieces]

        history = [{"role": "user", "content": "hi"}]
        checkpoint = fusing(render([{"role": "system", "content": SYSTEM}, *history]))
        head, reason = rolling_shadow_head(
            checkpoint, messages=[{"role": "system", "content": SYSTEM}, *history], tools=TOOLS,
            assistant_content="Hello", render=render, tokenize=fusing,
        )
        self.assertIsNone(head)
        self.assertEqual(reason, "head_not_a_token_prefix")

    def test_a_render_failure_is_a_refusal_not_an_error(self) -> None:
        def boom(messages, tools=None):
            raise RuntimeError("bridge unavailable")

        head, reason = rolling_shadow_head(
            [1, 2, 3], messages=[], tools=TOOLS, assistant_content="x", render=boom, tokenize=tokenize
        )
        self.assertEqual((head, reason), (None, "render_failed"))


class WiringTests(unittest.TestCase):
    """`complete_chat` hands the render inputs to `complete_prompt`, which
    forwards them to the standard prefill: all three signatures must agree
    (the live server hit a TypeError when only the innermost one did)."""

    def test_the_render_inputs_travel_through_every_layer(self) -> None:
        import inspect

        for method in (NativeLlamaClient.complete_prompt, NativeLlamaClient._complete_prompt_standard):
            parameters = inspect.signature(method).parameters
            with self.subTest(method=method.__name__):
                self.assertIn("rolling_route_messages", parameters)
                self.assertIn("rolling_route_tools", parameters)
        source = inspect.getsource(NativeLlamaClient.complete_prompt)
        self.assertIn("rolling_route_messages=rolling_route_messages", source)
        self.assertIn("rolling_route_tools=rolling_route_tools", source)
        source = inspect.getsource(NativeLlamaClient.complete_chat)
        self.assertIn("rolling_route_messages=", source)

    def test_complete_prompt_forwards_the_render_inputs_to_the_standard_prefill(self) -> None:
        client, _ = shadow_client()
        client._persistent_mtp_runtime = None
        client._thinking_enabled = lambda thinking: False  # type: ignore[method-assign]
        seen: dict = {}

        def standard(prompt, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(prompt_tokens=1, output_tokens=0, reused_prompt_tokens=0, evaluated_prompt_tokens=1, cancelled=False)

        client._complete_prompt_standard = standard  # type: ignore[method-assign]
        client.complete_prompt(
            "<S>s</S><U>hi</U><A>",
            allow_mtp_experimental=False,
            rolling_route_messages=[{"role": "user", "content": "hi"}],
            rolling_route_tools=[],
        )
        self.assertEqual(seen["rolling_route_messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(seen["rolling_route_tools"], [])


class AdvanceTests(unittest.TestCase):
    """The client method, over the hybrid fake, through the real restore,
    prefill and capture primitives."""

    def _one_turn(self, reply: str):
        client, lib = shadow_client()
        history = [{"role": "user", "content": "hi"}]
        route1 = route_turn(client, lib, history)
        final_turn(lib, history, reply)
        self.assertEqual(lib.recurrent, fold(lib.kv), "the live state is the final's")
        return client, lib, history, route1

    def test_a_short_reply_advances_the_checkpoint_into_the_shadow_slot(self) -> None:
        client, lib, history, route1 = self._one_turn("Hello there")

        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)

        expected = route1 + tokenize("Hello there</A><U>")
        self.assertEqual(result["status"], "advanced", result)
        self.assertEqual(result["delta_tokens"], len(expected) - len(route1))
        self.assertEqual(client._rolling_anchor_store().route_shadow_state.tokens, expected)
        self.assertEqual(client._rolling_route_anchor_state.tokens, route1, "the route checkpoint is untouched")
        self.assertEqual(lib.kv, expected, "the live sequence is exactly the shadow")
        self.assertEqual(lib.recurrent, fold(expected))
        self.assertEqual(client._session.committed_sequence_tokens, expected)
        self.assertEqual(lib.seq_rm_calls, [], "no partial seq_rm on the hybrid memory")
        self.assertFalse(client._session.continuation_ready, "the final's context is gone")
        self.assertEqual(lib.set_data_blobs, [client._rolling_route_anchor_state.checkpoint_data], "restored the route blob, once")

    def test_a_long_reply_and_the_next_short_turn_reuse_the_shadow_exactly(self) -> None:
        client, lib, history, route1 = self._one_turn(LONG_REPLY)
        result = client.advance_route_checkpoint_after_final(LONG_REPLY, should_cancel=lambda: False)
        self.assertEqual(result["status"], "advanced")
        self.assertGreaterEqual(result["delta_tokens"], 450 + 2)
        shadow = list(client._rolling_anchor_store().route_shadow_state.tokens)
        shadow_blob = client._rolling_anchor_store().route_shadow_state.checkpoint_data
        lib.set_data_blobs.clear()

        history2 = [*history, {"role": "assistant", "content": LONG_REPLY}, {"role": "user", "content": "What is the capital of France?"}]
        route2 = route_prompt_tokens(history2)
        reused = client._prepare_memory_with_ornith_rolling_route_anchor(route2)

        self.assertEqual(reused, len(shadow), "exactly the shadow is reused")
        self.assertEqual(route2[:reused], shadow)
        self.assertEqual(len(route2) - reused, len(tokenize("What is the capital of France?</U><A>")), "only the new turn is evaluated")
        self.assertEqual(lib.set_data_blobs, [], "the live sequence already is the shadow: no restore at all")
        self.assertIsNotNone(shadow_blob)
        self.assertEqual(lib.seq_rm_calls, [])
        # ... finish the turn as the client would, and compare with a clean decode
        lib.decode(route2[reused:])
        self.assertEqual(lib.kv, route2)
        clean = ShadowLib(); clean.decode(route2)
        self.assertEqual(lib.recurrent, clean.recurrent, "state identical to a cold decode of the whole prompt")

    def test_the_next_turns_whole_prompt_capture_supersedes_the_shadow(self) -> None:
        client, lib, history, _ = self._one_turn("Hello there")
        client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        history2 = [*history, {"role": "assistant", "content": "Hello there"}, {"role": "user", "content": "ok"}]
        route2 = route_turn(client, lib, history2)
        store = client._rolling_anchor_store()
        self.assertEqual(store.route_state.tokens, route2)
        self.assertFalse(store.route_shadow_state.valid, "the shadow it extended is superseded")
        self.assertEqual(store.route_state.render_messages, [{"role": "system", "content": SYSTEM}, *history2])

    def test_a_preempted_shadow_is_captured_at_the_batch_boundary_it_reached(self) -> None:
        client, lib, history, route1 = self._one_turn(LONG_REPLY)
        client.config.progress_step = 100
        calls = {"n": 0}

        def should_cancel() -> bool:  # a request arrives after the first batch
            calls["n"] += 1
            return calls["n"] > 2

        result = client.advance_route_checkpoint_after_final(LONG_REPLY, should_cancel=should_cancel)

        self.assertEqual(result["status"], "partial", result)
        self.assertTrue(result["preempted"])
        self.assertEqual(result["decoded_tokens"], 100)
        shadow = client._rolling_anchor_store().route_shadow_state.tokens
        self.assertEqual(len(shadow), len(route1) + 100)
        self.assertEqual(lib.kv, shadow, "exactly the tokens resident are the tokens recorded")
        self.assertEqual(client._session.committed_sequence_tokens, shadow)
        # the next route prompt finishes what the shadow started
        history2 = [*history, {"role": "assistant", "content": LONG_REPLY}, {"role": "user", "content": "ok"}]
        route2 = route_prompt_tokens(history2)
        reused = client._prepare_memory_with_ornith_rolling_route_anchor(route2)
        self.assertEqual(reused, len(shadow))
        lib.decode(route2[reused:])
        clean = ShadowLib(); clean.decode(route2)
        self.assertEqual((lib.kv, lib.recurrent), (route2, clean.recurrent))

    def test_a_request_already_waiting_skips_before_any_render_or_restore(self) -> None:
        client, lib, history, route1 = self._one_turn("Hello there")
        live = list(lib.kv)
        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: True)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "request_waiting")
        self.assertFalse(client._rolling_anchor_store().route_shadow_state.valid)
        self.assertEqual(lib.set_data_blobs, [], "nothing restored")
        self.assertEqual(lib.kv, live, "the final's live sequence is left exactly as it was")
        self.assertTrue(client._session.continuation_ready, "and it can still be continued")
        self.assertTrue(client._rolling_route_anchor_state.valid)

    def test_a_checkpoint_this_turn_missed_is_never_advanced(self) -> None:
        # A compacted or reset conversation: the route prompt did not extend
        # the stored checkpoint and the keep-older rule kept it. A reply
        # decoded on top of it could never be restored.
        client, lib, history, route1 = self._one_turn("Hello there")
        client._rolling_route_capture_allowed(route_prompt_tokens([{"role": "user", "content": "elsewhere"}]), client._rolling_route_identity(tools=TOOLS))
        self.assertEqual(client._rolling_route_anchor_state.non_extending_misses, 1)
        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "checkpoint_missed_this_turn")
        self.assertEqual(lib.set_data_blobs, [])

    def test_a_head_past_the_context_budget_is_refused_before_touching_the_model(self) -> None:
        client, lib = shadow_client()
        client.config.context_tokens = 100          # part of the identity: set before the route turn
        history = [{"role": "user", "content": "hi"}]
        route_turn(client, lib, history)
        final_turn(lib, history, LONG_REPLY)
        result = client.advance_route_checkpoint_after_final(LONG_REPLY, should_cancel=lambda: False)
        self.assertEqual(result["reason"], "context_budget")
        self.assertEqual(lib.set_data_blobs, [])
        self.assertTrue(client._session.continuation_ready)

    def test_a_reset_invalidates_the_shadow_and_a_stale_checkpoint_is_never_advanced(self) -> None:
        client, lib, history, _ = self._one_turn("Hello there")
        client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        self.assertTrue(client._rolling_anchor_store().route_shadow_state.valid)

        client._reset_generation += 1                       # what reset_session_state does ...
        client._invalidate_rolling_route_anchor("session_reset")

        store = client._rolling_anchor_store()
        self.assertFalse(store.route_shadow_state.valid)
        self.assertFalse(store.route_state.valid)
        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "no_route_checkpoint")
        # ... and a checkpoint captured under the previous generation is stale
        client, lib, history, _ = self._one_turn("Hello there")
        client._reset_generation += 1
        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "checkpoint_identity_stale")
        self.assertFalse(client._rolling_anchor_store().route_shadow_state.valid)
        self.assertEqual(lib.set_data_blobs, [], "nothing restored")

    def test_a_session_change_never_advances(self) -> None:
        client, lib, _, _ = self._one_turn("Hello there")
        client._session.session_id = "other"
        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "checkpoint_identity_stale")
        self.assertEqual(lib.set_data_blobs, [])

    def test_a_failed_restore_leaves_no_stale_state(self) -> None:
        client, lib, _, _ = self._one_turn("Hello there")
        lib.llama_state_seq_set_data = lambda ctx, buffer, size, seq_id: 0  # type: ignore[method-assign]
        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "checkpoint_restore_failed")
        self.assertEqual(lib.kv, [], "memory cleared: a partial restore is unknown state")
        self.assertEqual(client._session.committed_sequence_tokens, [])
        self.assertFalse(client._rolling_anchor_store().route_shadow_state.valid)

    def test_an_external_cancel_during_a_batch_discards_the_unknown_state(self) -> None:
        # `/cancel` with nothing in flight reaches the abort callback inside
        # llama_decode: the batch returns 2 and is counted as processed by
        # the range decoder. That state is unknown and must not be captured.
        client, lib, _, route1 = self._one_turn(LONG_REPLY)
        client.config.progress_step = 100
        original = lib.llama_decode

        def aborted_second_batch(ctx, batch):
            if len(lib.decode_calls) == 1:
                client.cancel_event.set()
                return 2
            return original(ctx, batch)

        lib.llama_decode = aborted_second_batch  # type: ignore[method-assign]
        result = client.advance_route_checkpoint_after_final(LONG_REPLY, should_cancel=lambda: False)
        self.assertEqual(result["reason"], "cancelled")
        self.assertFalse(client._rolling_anchor_store().route_shadow_state.valid)
        self.assertEqual((lib.kv, lib.recurrent), ([], 0), "memory cleared")
        self.assertEqual(client._session.committed_sequence_tokens, [])
        self.assertFalse(client.cancel_event.is_set(), "the cancel state is reset for the next request")
        self.assertEqual(client._rolling_route_anchor_state.tokens, route1)

    def test_an_external_cancel_between_batches_keeps_the_exactly_known_partial(self) -> None:
        # `/cancel` landing between two batches: nothing has run since the
        # last batch completed, so the partial is exact and is kept.
        client, lib, history, route1 = self._one_turn(LONG_REPLY)
        client.config.progress_step = 100
        seen = {"n": 0}

        def observe() -> bool:                       # never preempts, but the cancel lands after batch 1
            seen["n"] += 1
            if seen["n"] == 3:
                client.cancel_event.set()
            return False

        result = client.advance_route_checkpoint_after_final(LONG_REPLY, should_cancel=observe)
        self.assertEqual(result["status"], "partial", result)
        self.assertEqual(result["decoded_tokens"], 100)
        shadow = client._rolling_anchor_store().route_shadow_state.tokens
        self.assertEqual(lib.kv, shadow)
        self.assertFalse(client.cancel_event.is_set())

    def test_a_preempted_shadow_resets_the_cancel_state(self) -> None:
        client, lib, _, _ = self._one_turn(LONG_REPLY)
        client.config.progress_step = 100
        seen = {"n": 0}

        def should_cancel() -> bool:
            seen["n"] += 1
            return seen["n"] > 2

        result = client.advance_route_checkpoint_after_final(LONG_REPLY, should_cancel=should_cancel)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(client.cancel_event.is_set())
        self.assertFalse(client._session.cancel_requested)

    def test_a_failed_decode_leaves_no_stale_state_and_keeps_the_route_checkpoint(self) -> None:
        client, lib, _, route1 = self._one_turn("Hello there")
        lib.fail_decode = True
        result = client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "decode_failed")
        self.assertEqual(lib.kv, [])
        self.assertEqual(client._session.committed_sequence_tokens, [])
        self.assertFalse(client._rolling_anchor_store().route_shadow_state.valid)
        self.assertEqual(client._rolling_route_anchor_state.tokens, route1, "the route checkpoint still serves the next turn")

    def test_a_next_prompt_the_shadow_does_not_match_falls_back_to_the_route_checkpoint(self) -> None:
        client, lib, history, route1 = self._one_turn("Hello there")
        client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        route_blob = client._rolling_route_anchor_state.checkpoint_data
        lib.set_data_blobs.clear()
        # the runtime committed something else than what the shadow assumed
        history2 = [*history, {"role": "assistant", "content": "Hello there!"}, {"role": "user", "content": "ok"}]
        route2 = route_prompt_tokens(history2)
        reused = client._prepare_memory_with_ornith_rolling_route_anchor(route2)
        self.assertEqual(reused, len(route1), "exactly the route checkpoint, as without any shadow")
        self.assertEqual(lib.set_data_blobs, [route_blob])
        self.assertEqual(lib.seq_rm_calls, [])
        lib.decode(route2[reused:])
        clean = ShadowLib(); clean.decode(route2)
        self.assertEqual((lib.kv, lib.recurrent), (route2, clean.recurrent))

    def test_a_prompt_extending_neither_checkpoint_falls_cold(self) -> None:
        client, lib, history, _ = self._one_turn("Hello there")
        client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        lib.set_data_blobs.clear()
        other = route_prompt_tokens([{"role": "user", "content": "a new conversation"}])
        self.assertEqual(client._prepare_memory_with_ornith_rolling_route_anchor(other), 0)
        self.assertEqual(lib.set_data_blobs, [])
        self.assertEqual((lib.kv, lib.recurrent), ([], 0))

    def test_an_equal_length_shadow_match_is_served_by_the_route_checkpoint(self) -> None:
        # An equal-length prompt leaves nothing to decode for fresh logits,
        # so the shadow is not chosen; the shorter route checkpoint is.
        client, lib, history, route1 = self._one_turn("Hello there")
        client.advance_route_checkpoint_after_final("Hello there", should_cancel=lambda: False)
        shadow = list(client._rolling_anchor_store().route_shadow_state.tokens)
        route_blob = client._rolling_route_anchor_state.checkpoint_data
        lib.set_data_blobs.clear()
        self.assertEqual(client._prepare_memory_with_ornith_rolling_route_anchor(shadow), len(route1))
        self.assertEqual(lib.set_data_blobs, [route_blob])

    def test_only_flash_next_is_admitted(self) -> None:
        self.assertEqual(POST_FINAL_ROUTE_SHADOW_PROFILE_IDS, {QWEN38_FLASH_NEXT_PROFILE_ID})
        self.assertTrue(POST_FINAL_ROUTE_SHADOW_PROFILE_IDS <= ROLLING_ROUTE_PROFILE_IDS)
        client, lib = shadow_client(ORNITH15_PROFILE_ID)
        history = [{"role": "user", "content": "hi"}]
        route1 = route_turn(client, lib, history)
        final_turn(lib, history, "Hello")
        result = client.advance_route_checkpoint_after_final("Hello", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "model_profile_ineligible")
        self.assertEqual(client._rolling_route_anchor_state.tokens, route1, "Ornith's rolling checkpoint is exactly as before")
        self.assertFalse(client._rolling_anchor_store().route_shadow_state.valid)
        self.assertEqual(lib.set_data_blobs, [])

    def test_thinking_mtp_and_in_flight_block_the_shadow(self) -> None:
        for change in ("thinking", "mtp", "in_flight"):
            client, lib, _, _ = self._one_turn("Hello")
            if change == "thinking":
                client.config.thinking = True
            elif change == "mtp":
                client._session.mtp_enabled = True
            else:
                client._session.in_flight = True
            with self.subTest(change=change):
                result = client.advance_route_checkpoint_after_final("Hello", should_cancel=lambda: False)
                self.assertEqual(result["status"], "skipped")
                self.assertEqual(result["reason"], "native_request_in_flight" if change == "in_flight" else "model_profile_ineligible")
                self.assertEqual(lib.set_data_blobs, [])

    def test_a_checkpoint_without_render_inputs_cannot_be_advanced(self) -> None:
        client, lib, _, route1 = self._one_turn("Hello")
        state = client._rolling_route_anchor_state
        client._rolling_route_anchor_state = RollingRouteAnchorState(
            identity=state.identity, tokens=list(state.tokens), checkpoint_data=state.checkpoint_data
        )
        result = client.advance_route_checkpoint_after_final("Hello", should_cancel=lambda: False)
        self.assertEqual(result["reason"], "checkpoint_without_render_inputs")


# --- the server side ---------------------------------------------------------

class _Timings:
    def __init__(self, *, cancelled: bool = False, output_tokens: int = 5) -> None:
        self.cancelled = cancelled
        self.output_tokens = output_tokens
        self.prompt_tokens = 10
        self.reused_prompt_tokens = 0
        self.evaluated_prompt_tokens = 10
        self.prefill_ms = 1.0
        self.generation_ms = 1.0


class _Completion:
    def __init__(self, content: str, *, cancelled=False, output_tokens=5, tool_calls=(), stopped_by_stop=False) -> None:
        self.content = content
        self.timings = _Timings(cancelled=cancelled, output_tokens=output_tokens)
        self.stopped_by_stop = stopped_by_stop
        self.completed_after_thought = False
        self.tool_calls = tuple(tool_calls)
        self.reasoning_content = ""
        self.reasoning_tokens = 0


class _ServerClient:
    """A client whose final returns a scripted completion and whose shadow
    records what it was asked, optionally blocking so preemption can be seen."""

    supports_vision = False
    supports_audio = False

    def __init__(self, completion: _Completion) -> None:
        self.completion = completion
        self.config = SimpleNamespace(thinking=False)
        self.model_profile = _Profile(QWEN38_FLASH_NEXT_PROFILE_ID)
        self.advance_calls: list[str] = []
        self.should_cancel = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def complete_chat_text(self, messages, **kwargs):
        return self.completion

    def advance_route_checkpoint_after_final(self, content: str, *, should_cancel=None):
        self.advance_calls.append(content)
        self.should_cancel = should_cancel
        self.started.set()
        self.release.wait(5)
        return {"status": "advanced", "preempted": bool(should_cancel and should_cancel())}


def _server(completion: _Completion) -> tuple[OrbitNativeServer, _ServerClient]:
    client = _ServerClient(completion)
    with mock.patch("orbit.native_server.app.safe_native_capability_manifest", return_value={}):
        server = OrbitNativeServer(client=client, model_alias="m")  # type: ignore[arg-type]
    return server, client


def _payload(**extra) -> dict:
    payload = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}], "max_tokens": 64}
    payload.update(extra)
    return payload


class ServerSchedulingTests(unittest.TestCase):
    def test_a_declared_naturally_stopped_reply_schedules_the_shadow_after_the_response(self) -> None:
        server, client = _server(_Completion("Hello there"))
        client.release.clear()
        result = server.chat(_payload(route_history_continuation=True))
        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(client.started.wait(2), "the shadow runs on its own thread once the response is built")
        self.assertEqual(client.advance_calls, ["Hello there"])
        client.release.set()
        server.wait_for_route_shadow(2)
        self.assertEqual(server.last_route_shadow, {"status": "advanced", "preempted": False})

    def test_without_the_declaration_nothing_is_scheduled(self) -> None:
        server, client = _server(_Completion("Hello there"))
        server.chat(_payload())
        server.wait_for_route_shadow(1)
        self.assertEqual(client.advance_calls, [])
        self.assertFalse(parse_chat_request(_payload()).route_history_continuation)
        self.assertTrue(parse_chat_request(_payload(route_history_continuation=True)).route_history_continuation)

    def test_cancelled_truncated_empty_tool_and_stop_sequence_replies_never_schedule(self) -> None:
        cases = {
            "cancelled": _Completion("partial", cancelled=True),
            "length": _Completion("cut off", output_tokens=64),
            "empty": _Completion("   "),
            "tool_calls": _Completion("", tool_calls=({"id": "", "type": "function", "function": {"name": "f", "arguments": "{}"}},)),
            "stop_sequence": _Completion("until here", stopped_by_stop=True),
        }
        for name, completion in cases.items():
            server, client = _server(completion)
            with self.subTest(case=name):
                server.chat(_payload(route_history_continuation=True, stop=["here"] if name == "stop_sequence" else []))
                server.wait_for_route_shadow(1)
                self.assertEqual(client.advance_calls, [], name)

    def test_a_request_that_arrives_preempts_the_running_shadow(self) -> None:
        server, client = _server(_Completion("Hello there"))
        client.release.clear()
        server.chat(_payload(route_history_continuation=True))
        self.assertTrue(client.started.wait(2))
        self.assertFalse(client.should_cancel(), "no request waiting: the shadow keeps going")
        entered = threading.Event()

        def request() -> None:
            with server._model_lock():
                entered.set()

        waiter = threading.Thread(target=request, daemon=True)
        waiter.start()
        deadline = time.monotonic() + 2
        while not client.should_cancel() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(client.should_cancel(), "a waiting request is what stops the shadow")
        self.assertFalse(entered.is_set(), "the request waits for the shadow to yield at a batch boundary")
        client.release.set()
        self.assertTrue(entered.wait(2))
        waiter.join(2)
        server.wait_for_route_shadow(2)
        self.assertEqual(server.last_route_shadow, {"status": "advanced", "preempted": True})
        self.assertFalse(server._requests_waiting())

    def test_a_shadow_never_starts_while_a_request_is_waiting(self) -> None:
        server, client = _server(_Completion("Hello there"))
        with server._waiters_lock:
            server._waiters += 1                # a request announced before the shadow thread ran
        try:
            server.chat(_payload(route_history_continuation=True))
            server.wait_for_route_shadow(2)
        finally:
            with server._waiters_lock:
                server._waiters -= 1
        self.assertEqual(client.advance_calls, [])
        self.assertEqual(server.last_route_shadow, {"status": "skipped", "reason": "request_waiting"})

    def test_a_later_final_supersedes_a_shadow_that_has_not_run_yet(self) -> None:
        # A final followed by its retry (both declared) schedules two shadows;
        # whichever runs first, only the latest reply's shadow is built.
        server, client = _server(_Completion("first reply"))
        client.release.clear()
        server.chat(_payload(route_history_continuation=True))
        self.assertTrue(client.started.wait(2))          # shadow 1 is running, holding the lock
        client.completion = _Completion("retry reply")
        first_started = client.started
        client.started = threading.Event()
        done = threading.Event()

        def second_request() -> None:
            server.chat(_payload(route_history_continuation=True))   # waits for the lock, preempts shadow 1
            done.set()

        threading.Thread(target=second_request, daemon=True).start()
        deadline = time.monotonic() + 2
        while not client.should_cancel() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(client.should_cancel(), "the retry request preempts the first shadow")
        client.release.set()                              # shadow 1 yields
        self.assertTrue(done.wait(2))
        self.assertTrue(client.started.wait(2), "the retry's shadow runs")
        server.wait_for_route_shadow(2)
        self.assertEqual(client.advance_calls, ["first reply", "retry reply"])
        self.assertEqual(server.last_route_shadow, {"status": "advanced", "preempted": False})

    def test_a_stale_shadow_thread_is_skipped_as_superseded(self) -> None:
        server, client = _server(_Completion("first reply"))
        # schedule one, then bump the serial as a later final would, then let it run
        with server.lock:
            server._schedule_route_shadow(parse_chat_request(_payload(route_history_continuation=True)), content="first reply", finish_reason="stop", stopped=False, tool_calls=())
            server._route_shadow_serial += 1
        server.wait_for_route_shadow(2)
        self.assertEqual(client.advance_calls, [])
        self.assertEqual(server.last_route_shadow, {"status": "skipped", "reason": "superseded"})

    def test_shutdown_cancels_and_joins_a_running_shadow(self) -> None:
        server, client = _server(_Completion("Hello there"))
        client.release.clear()
        cancelled = threading.Event()

        def cancel() -> None:
            cancelled.set()
            client.release.set()

        client.cancel = cancel  # type: ignore[attr-defined]
        server.chat(_payload(route_history_continuation=True))
        self.assertTrue(client.started.wait(2))
        server.stop_route_shadow(timeout=2)
        self.assertTrue(cancelled.is_set(), "the shadow is told to stop")
        self.assertFalse(server._route_shadow_thread.is_alive(), "and is joined before the context goes")
        server.stop_route_shadow(timeout=1)               # idempotent with nothing running

    def test_shutdown_stops_a_shadow_that_is_still_waiting_for_the_lock(self) -> None:
        server, client = _server(_Completion("Hello there"))
        held = threading.Event(); release = threading.Event()

        def request() -> None:                       # an in-flight request holding the model
            with server._model_lock():
                held.set(); release.wait(5)

        threading.Thread(target=request, daemon=True).start()
        self.assertTrue(held.wait(2))
        server._schedule_route_shadow(parse_chat_request(_payload(route_history_continuation=True)), content="Hello there", finish_reason="stop", stopped=False, tool_calls=())
        stopper = threading.Thread(target=server.stop_route_shadow, kwargs={"timeout": 5}, daemon=True)
        stopper.start()
        time.sleep(0.05)
        release.set()                                # the request drains; the shadow gets the lock ...
        stopper.join(5)
        server.wait_for_route_shadow(2)
        self.assertEqual(client.advance_calls, [], "... and must not start decoding into a context about to be freed")
        self.assertEqual(server.last_route_shadow, {"status": "skipped", "reason": "stopping"})

    def test_a_shadow_error_is_recorded_and_never_raised(self) -> None:
        server, client = _server(_Completion("Hello there"))

        def boom(content, *, should_cancel=None):
            raise RuntimeError("bridge exploded")

        client.advance_route_checkpoint_after_final = boom  # type: ignore[method-assign]
        server.chat(_payload(route_history_continuation=True))
        server.wait_for_route_shadow(2)
        self.assertEqual(server.last_route_shadow, {"status": "skipped", "reason": "error", "error": "bridge exploded"})


# --- the runtime's declaration and the backend's request flag -----------------

from orbit.backend.base import ChatResult
from orbit.backend.llama_server import _route_history_continuation_requested
from orbit.backend.payloads import ChatPayloadOptions, build_chat_payload
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.kv_diag import (
    current_phase,
    current_route_history_continuation,
    model_call_context,
    route_history_continuation_context,
)


def _result(content: str, finish_reason: str = "stop") -> ChatResult:
    return ChatResult(
        content=content, model="fake", finish_reason=finish_reason, tool_calls=[], prompt_tokens=None,
        completion_tokens=None, cached_tokens=None, prompt_tokens_per_second=None, generation_tokens_per_second=None,
    )


class _DeclaringBackend:
    """Records, per model call, the phase and whether the runtime declared the
    reply as extending the route history -- what the real backend reads."""

    def __init__(self, results: list[ChatResult]) -> None:
        self.results = list(results)
        self.seen: list[tuple[str | None, bool]] = []

    def chat(self, messages, *, temperature, max_tokens, tools=None) -> ChatResult:
        self.seen.append((current_phase(), current_route_history_continuation()))
        return self.results.pop(0)


class DeclarationTests(unittest.TestCase):
    def test_the_final_of_a_plain_chat_turn_is_declared_and_the_route_is_not(self) -> None:
        backend = _DeclaringBackend([_result('{"route":"CHAT"}'), _result("Hello there")])
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        result = runtime.ask_auto("hi", temperature=0, max_tokens=64, workdir=Path("."))
        self.assertEqual(result.content, "Hello there")
        self.assertEqual(backend.seen, [("route", False), ("chat_final", True)])
        self.assertEqual(runtime.messages[-1], {"role": "assistant", "content": "Hello there"}, "what the next route prompt shows")

    def test_a_length_route_falls_to_a_declared_final_retry(self) -> None:
        backend = _DeclaringBackend([_result("I am starting to answer directly", "length"), _result("Paris")])
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        result = runtime.ask_auto("What is the capital of France?", temperature=0, max_tokens=64, workdir=Path("."))
        self.assertEqual(result.content, "Paris")
        self.assertEqual(backend.seen, [("route", False), ("chat_final_retry", True)])

    def test_with_evidence_in_the_session_nothing_is_declared(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence")
            store.add(
                "exec_shell_full_command", "recorded output " + "p" * 120,
                metadata={"tool_call_id": "call-1", "user_turn_id": "turn-1", "produced_by_phase": "tool_call"},
            )
            backend = _DeclaringBackend([_result('{"route":"CHAT"}'), _result("Hello there")])
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages.append({"role": "user", "content": "run it"})
            runtime.messages.append({"role": "assistant", "content": "done"})
            runtime.ask_auto("and now?", temperature=0, max_tokens=64, workdir=Path(tmp))
        self.assertEqual([declared for _phase, declared in backend.seen], [False, False])

    def test_the_backend_requests_the_flag_only_for_declared_final_phases(self) -> None:
        with mock.patch("orbit.backend.llama_server.prefix_anchor_enabled", return_value=True):
            with route_history_continuation_context(True):
                for phase, expected in (("chat_final", True), ("chat_final_retry", True), ("chat_final_completion_repair", True), ("route", False), ("final_from_tool", False)):
                    with model_call_context(phase=phase, tools_mode="on"), self.subTest(phase=phase):
                        self.assertEqual(_route_history_continuation_requested(native_backend=True), expected)
                        self.assertFalse(_route_history_continuation_requested(native_backend=False))
            with model_call_context(phase="chat_final", tools_mode="on"):
                self.assertFalse(_route_history_continuation_requested(native_backend=True), "undeclared")
                with route_history_continuation_context(False):
                    self.assertFalse(_route_history_continuation_requested(native_backend=True))
        with mock.patch("orbit.backend.llama_server.prefix_anchor_enabled", return_value=False):
            with route_history_continuation_context(True), model_call_context(phase="chat_final", tools_mode="on"):
                self.assertFalse(_route_history_continuation_requested(native_backend=True), "anchors off: no shadow either")

    def test_a_native_continuation_refused_after_a_shadow_falls_back_to_the_prompt(self) -> None:
        from orbit.backend.base import RecoverableBackendError

        class _Backend(_DeclaringBackend):
            def continue_current(self, **kwargs):
                raise RecoverableBackendError("backend server HTTP 500: no active continuation state")

        backend = _Backend([_result("...and the rest of the answer.")])
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        runtime.messages.append({"role": "user", "content": "explain"})
        runtime.messages.append({"role": "assistant", "content": "Plan: outline the answer first"})
        runtime.client_state.update_from_result(_result("Plan: outline the answer first", "stop"), thinking=runtime._thinking())
        self.assertTrue(runtime.can_continue_last_response(), "the runtime heuristic offers a continuation after this stop")
        outcome = runtime._continue_environment().continue_last_response(
            temperature=0, max_tokens=64, on_final_delta=None, on_progress=None, on_model_step=None, on_phase_start=None
        )
        self.assertTrue(outcome.used_prompt_fallback)
        self.assertFalse(outcome.used_native_continue)
        self.assertEqual(outcome.result.content, "...and the rest of the answer.")

    def test_the_payload_carries_the_flag_only_when_set(self) -> None:
        base = dict(model="m", messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=8)
        self.assertNotIn("route_history_continuation", build_chat_payload(ChatPayloadOptions(**base)))
        self.assertTrue(build_chat_payload(ChatPayloadOptions(**base, route_history_continuation=True))["route_history_continuation"])


if __name__ == "__main__":
    unittest.main()
