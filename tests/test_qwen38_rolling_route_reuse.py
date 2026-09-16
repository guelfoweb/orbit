"""QWEN38-PROMPT-CACHE-REUSE-22: Qwen3.8 Flash Next (qwen4exp) takes the rolling
route checkpoint, and the restore carries the hybrid model's whole state.

Production symptom: every chat turn re-evaluated the full ~1100-token route
prompt (`cached=0`) because the rolling route strategy -- the only mechanism
that carries a conversation's route prompt across the route -> final -> route
alternation -- was gated on the Ornith profile id alone. The state itself was
never the problem: on the vendored llama.cpp 41abbfd the qwen4exp memory is
`llama_memory_hybrid_idx`, whose `state_write`/`state_read` serialize the
attention KV, the DeltaNet recurrent state and the indexer cache in one seq
blob, which is exactly what the rolling checkpoint captures and restores.

The fake library below models what matters about that memory: the recurrent
state only ever reflects the latest position (so a partial `seq_rm` is refused,
as `llama_memory_recurrent::seq_rm` does), and the seq-state blob carries the
recurrent state alongside the KV cells. What is pinned is the client's
contract with that memory: the whole captured blob is restored through the seq
state API, no partial `seq_rm` is ever attempted, a non-extending prompt falls
cold with no stale state, and the live recurrent state after a restore is the
checkpoint's. The proof that the real blob carries the recurrent half is the
real-model probe recorded in AGENTS.md (bit-identical logits after restore,
also over a contaminated live state); it cannot be a unit test.
"""

from __future__ import annotations

import ctypes
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.client import ROLLING_ROUTE_PROFILE_IDS, NativeLlamaClient
from orbit.native_llama.model_profiles import (
    GEMMA4_PROFILE_ID,
    ORNITH15_PROFILE_ID,
    QWEN36_PROFILE_ID,
    QWEN38_FLASH_NEXT_PROFILE_ID,
    QWEN38_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
)
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
    capture_rolling_route_anchor,
    rolling_route_should_replace,
)

ROUTE1 = [100, 101, 102, 103, 104, 105]          # system + "hi" + assistant header
ROUTE2 = ROUTE1 + [106, 107, 108, 109]           # ... + reply + "hello again" + header
FINAL = [900, 901, 902]                           # the tools-free final prompt: shares nothing
DIVERGENT = ROUTE1[:3] + [777, 778, 779, 780]     # same head, different tail


def identity(**overrides) -> RollingRouteIdentity:
    base = dict(
        strategy_id=ROLLING_ROUTE_STRATEGY_ID,
        session_id="default",
        profile_id=QWEN38_FLASH_NEXT_PROFILE_ID,
        model_id="/models/qwen38-flash-next.gguf",
        template_id="tpl",
        tool_schema_hash="tools",
        capability_summary_hash="caps",
        runtime_policy_hash="policy",
        native_version="libllama.so",
        tools_mode="on",
        reset_generation=0,
    )
    base.update(overrides)
    return RollingRouteIdentity(**base)


def fold(tokens: list[int]) -> int:
    """The recurrent state after these tokens: a function of the whole history."""
    state = 0
    for token in tokens:
        state = (state * 1_000_003 + token) % (2**61 - 1)
    return state


class HybridLib:
    """attention KV cells + one DeltaNet-style recurrent state, one seq blob for both."""

    def __init__(self) -> None:
        self.kv: list[int] = []
        self.recurrent: int = 0
        self.cleared = 0
        self.seq_rm_calls: list[tuple[int, int]] = []
        self.set_data_blobs: list[bytes] = []

    # -- what decoding does to the live state -------------------------------
    def decode(self, tokens: list[int]) -> None:
        self.kv.extend(tokens)
        self.recurrent = fold(self.kv)

    # -- llama.cpp memory API as the client uses it ---------------------------
    def llama_get_memory(self, ctx):
        return "mem"

    def llama_memory_clear(self, mem, data):
        self.cleared += 1
        self.kv = []
        self.recurrent = 0

    def llama_memory_seq_rm(self, mem, seq_id, p0, p1):
        self.seq_rm_calls.append((p0, p1))
        if p0 <= 0:
            self.kv = []
            self.recurrent = 0
            return True
        # the recurrent state is not preserved for previous tokens: refuse
        return False

    def llama_state_seq_get_size(self, ctx, seq_id):
        return len(self._blob())

    def llama_state_seq_get_data(self, ctx, buffer, size, seq_id):
        data = self._blob()
        ctypes.memmove(buffer, data, len(data))
        return len(data)

    def llama_state_seq_set_data(self, ctx, buffer, size, seq_id):
        raw = ctypes.string_at(buffer, size)
        self.set_data_blobs.append(raw)
        state = json.loads(raw.decode())
        self.kv = list(state["kv"])
        self.recurrent = int(state["recurrent"])
        return size

    def _blob(self) -> bytes:
        return json.dumps({"kv": self.kv, "recurrent": self.recurrent}).encode()


class _LibHolder:
    def __init__(self, lib) -> None:
        self.lib = lib


class _Profile:
    def __init__(self, profile_id: str, verified: bool = True) -> None:
        self.profile_id = profile_id
        self.verified = verified


class _Config:
    use_mtp_experimental = False
    context_tokens = 4096
    thinking = False


class _Session:
    def __init__(self) -> None:
        self.ctx_tgt = object()
        self.session_id = "default"
        self.cached_prompt_tokens: list[int] = []
        self.committed_sequence_tokens: list[int] = []
        self.mtp_enabled = False
        self.prompt_cache_mode: str | None = None


def flash_next_client(lib: HybridLib | None = None) -> tuple[NativeLlamaClient, HybridLib]:
    """A client reaching the real strict-append gate over the fake hybrid memory."""
    lib = lib or HybridLib()
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    client.lib = _LibHolder(lib)
    client.config = _Config()
    client._session = _Session()
    client.model_profile = _Profile(QWEN38_FLASH_NEXT_PROFILE_ID)
    client._rolling_route_anchor_state = RollingRouteAnchorState()
    client._rolling_route_identity_cache = identity()
    client._invalidate_committed_sequence = (  # type: ignore[method-assign]
        lambda: client._session.committed_sequence_tokens.clear()
    )
    return client, lib


class EligibilityTest(unittest.TestCase):
    def test_flash_next_route_calls_are_eligible(self) -> None:
        client, _ = flash_next_client()
        self.assertTrue(
            client._ornith_rolling_route_eligible(route_prefix_anchor=True, tools=None, thinking=False)
        )

    def test_the_qualified_set_is_exactly_ornith_and_flash_next(self) -> None:
        self.assertEqual(ROLLING_ROUTE_PROFILE_IDS, {ORNITH15_PROFILE_ID, QWEN38_FLASH_NEXT_PROFILE_ID})
        for profile_id in (GEMMA4_PROFILE_ID, QWEN36_PROFILE_ID, QWEN38_PROFILE_ID, QWEN3_CODER_PROFILE_ID):
            client, _ = flash_next_client()
            client.model_profile = _Profile(profile_id)
            with self.subTest(profile=profile_id):
                self.assertFalse(
                    client._ornith_rolling_route_eligible(route_prefix_anchor=True, tools=None, thinking=False)
                )

    def test_the_usual_blockers_still_apply(self) -> None:
        client, _ = flash_next_client()
        self.assertFalse(client._ornith_rolling_route_eligible(route_prefix_anchor=False, tools=None, thinking=False))
        self.assertFalse(client._ornith_rolling_route_eligible(route_prefix_anchor=True, tools=None, thinking=True))
        client._session.mtp_enabled = True
        self.assertFalse(client._ornith_rolling_route_eligible(route_prefix_anchor=True, tools=None, thinking=False))
        client, _ = flash_next_client()
        client.model_profile = _Profile(QWEN38_FLASH_NEXT_PROFILE_ID, verified=False)
        self.assertFalse(client._ornith_rolling_route_eligible(route_prefix_anchor=True, tools=None, thinking=False))

    def test_analysis_lineage_stays_ornith_only(self) -> None:
        client, _ = flash_next_client()
        self.assertFalse(client._ornith_rolling_analysis_eligible(analysis_rolling_anchor=True, thinking=False))


class ModeTransitionTest(unittest.TestCase):
    """route (tools) -> final (chat) -> route (tools): the internal switch must
    not destroy the checkpoint for Flash Next, exactly as for Ornith."""

    def _client(self, profile_id: str, mode: str):
        client, _ = flash_next_client()
        client.model_profile = _Profile(profile_id)
        client._session.prompt_cache_mode = mode
        recorded: list[dict] = []
        client.reset_session_state = lambda **kwargs: recorded.append(dict(kwargs))  # type: ignore[method-assign]
        return client, recorded

    def test_flash_next_preserves_the_checkpoint_across_the_turn(self) -> None:
        client, recorded = self._client(QWEN38_FLASH_NEXT_PROFILE_ID, "tools:thinking=off")
        client._ensure_prompt_cache_mode("chat:thinking=off")
        client._ensure_prompt_cache_mode("tools:thinking=off")
        self.assertEqual([r["preserve_ornith_rolling_route_checkpoint"] for r in recorded], [True, True])

    def test_thinking_or_multimodal_transitions_still_destroy(self) -> None:
        client, recorded = self._client(QWEN38_FLASH_NEXT_PROFILE_ID, "tools:thinking=off")
        client._ensure_prompt_cache_mode("chat:thinking=on")
        client._session.prompt_cache_mode = "tools:thinking=off"
        client._ensure_prompt_cache_mode("multimodal:thinking=off")
        self.assertEqual([r["preserve_ornith_rolling_route_checkpoint"] for r in recorded], [False, False])

    def test_other_profiles_are_not_preserved(self) -> None:
        for profile_id in (QWEN36_PROFILE_ID, QWEN38_PROFILE_ID, GEMMA4_PROFILE_ID):
            client, recorded = self._client(profile_id, "tools:thinking=off")
            client._ensure_prompt_cache_mode("chat:thinking=off")
            with self.subTest(profile=profile_id):
                self.assertFalse(recorded[0]["preserve_ornith_rolling_route_checkpoint"])


class HybridStateRoundTripTest(unittest.TestCase):
    """The recurrent state has to come back with the checkpoint, not be
    rebuilt from the KV cells or rolled back with seq_rm."""

    def _turn_one(self) -> tuple[NativeLlamaClient, HybridLib, RollingRouteAnchorState]:
        client, lib = flash_next_client()
        lib.decode(ROUTE1)                                   # route-1 prefill
        state, _meta = capture_rolling_route_anchor(lib, client._session.ctx_tgt, prompt_tokens=ROUTE1, identity=identity())
        self.assertTrue(state.valid)
        client._rolling_route_anchor_state = state
        lib.decode([555, 556])                               # the generated reply moves the recurrent state on
        lib.llama_memory_clear("mem", True)                  # final call: session reset + its own prefill
        lib.decode(FINAL)
        self.assertEqual(lib.recurrent, fold(FINAL), "the live recurrent state now belongs to the final prompt")
        return client, lib, state

    def test_route_two_restores_kv_and_recurrent_state_and_reuses_the_exact_prefix(self) -> None:
        client, lib, state = self._turn_one()

        reused = client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2)

        self.assertEqual(reused, len(ROUTE1), "exactly the committed route-1 prefix is reused")
        self.assertEqual(lib.kv, ROUTE1, "the attention cells are the route-1 prefill")
        self.assertEqual(lib.recurrent, fold(ROUTE1), "the recurrent state is route-1's, not the final prompt's")
        self.assertNotEqual(lib.recurrent, fold(FINAL))
        self.assertEqual(lib.set_data_blobs, [state.checkpoint_data], "the whole captured blob is restored, once")
        self.assertEqual(lib.seq_rm_calls, [], "no partial seq_rm is ever attempted on the hybrid memory")
        self.assertEqual(client._session.committed_sequence_tokens, ROUTE1)
        self.assertEqual(client._session.cached_prompt_tokens, ROUTE2)
        self.assertEqual(len(ROUTE2) - reused, 4, "only the new suffix is left to prefill")

    def test_the_recurrent_assertion_is_sensitive_to_a_kv_only_restore(self) -> None:
        # A self-check of the fake, so the recurrent assertion above cannot
        # pass vacuously: a library that restored only the KV cells leaves
        # the final prompt's recurrent state in place, and that is detected.
        client, lib, _state = self._turn_one()

        def kv_only_set_data(ctx, buffer, size, seq_id):
            raw = ctypes.string_at(buffer, size)
            lib.kv = list(json.loads(raw.decode())["kv"])
            return size

        lib.llama_state_seq_set_data = kv_only_set_data  # type: ignore[method-assign]
        client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2)
        self.assertEqual(lib.kv, ROUTE1)
        self.assertNotEqual(lib.recurrent, fold(ROUTE1), "a KV-only restore is detectable: stale recurrent state")

    def test_a_divergent_route_falls_cold_instead_of_rolling_the_recurrent_state_back(self) -> None:
        client, lib, _state = self._turn_one()

        reused = client._prepare_memory_with_ornith_rolling_route_anchor(DIVERGENT)

        self.assertEqual(reused, 0)
        self.assertEqual(lib.set_data_blobs, [], "a prompt that does not extend the checkpoint restores nothing")
        self.assertEqual(lib.kv, [], "the memory is cleared for a cold prefill")
        self.assertEqual(lib.recurrent, 0, "no stale recurrent state survives into the new prompt")
        self.assertEqual(client._session.committed_sequence_tokens, [])

    def test_an_equal_length_prompt_is_not_served_from_the_checkpoint(self) -> None:
        client, lib, _state = self._turn_one()
        reused = client._prepare_memory_with_ornith_rolling_route_anchor(list(ROUTE1))
        self.assertEqual(reused, 0)
        self.assertEqual(lib.set_data_blobs, [])

    def test_a_different_session_or_reset_generation_never_reuses(self) -> None:
        for override in ({"session_id": "other"}, {"reset_generation": 1}, {"tool_schema_hash": "other-tools"}):
            client, lib, _state = self._turn_one()
            client._rolling_route_identity_cache = identity(**override)
            with self.subTest(override=override):
                self.assertEqual(client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2), 0)
                self.assertEqual(lib.set_data_blobs, [])


class ChainBreakRecoveryTest(unittest.TestCase):
    """Review follow-up: a conversation reset or compacted on the same session
    produces route prompts that never extend the stored checkpoint. The first
    such prompt keeps the older checkpoint (a post-tool route rendering is a
    one-off miss); the second consecutive one takes the slot, so reuse resumes
    instead of staying off for the server's life."""

    CONVERSATION_A = ROUTE1
    B1 = [300, 301, 302, 303]          # a new conversation on the same session
    B2 = B1 + [304, 305]
    B3 = B2 + [306]

    def _client_with_a(self):
        client, lib = flash_next_client()
        lib.decode(self.CONVERSATION_A)
        state, _ = capture_rolling_route_anchor(lib, client._session.ctx_tgt, prompt_tokens=self.CONVERSATION_A, identity=identity())
        client._rolling_route_anchor_state = state
        return client, lib

    def test_one_miss_keeps_the_older_checkpoint(self) -> None:
        client, _ = self._client_with_a()
        self.assertFalse(client._rolling_route_capture_allowed(self.B1, identity()))
        self.assertEqual(client._rolling_route_anchor_state.tokens, self.CONVERSATION_A)
        self.assertEqual(client._rolling_route_anchor_state.non_extending_misses, 1)
        self.assertTrue(client._rolling_route_anchor_state.valid, "the miss does not invalidate anything")

    def test_an_extending_prompt_after_one_miss_still_reuses_and_captures(self) -> None:
        client, _ = self._client_with_a()
        client._rolling_route_capture_allowed(self.B1, identity())          # the transient miss
        self.assertEqual(client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2), len(ROUTE1))
        self.assertTrue(client._rolling_route_capture_allowed(ROUTE2, identity()), "the chain continues: capture")

    def test_the_second_consecutive_miss_takes_the_slot(self) -> None:
        client, lib = self._client_with_a()
        self.assertFalse(client._rolling_route_capture_allowed(self.B1, identity()))
        self.assertTrue(client._rolling_route_capture_allowed(self.B2, identity()), "second miss: replace")
        # ... as the capture site would now do:
        lib.llama_memory_clear("mem", True); lib.decode(self.B2)
        state, _ = capture_rolling_route_anchor(lib, client._session.ctx_tgt, prompt_tokens=self.B2, identity=identity())
        client._rolling_route_anchor_state = state
        self.assertEqual(client._rolling_route_anchor_state.non_extending_misses, 0, "a fresh capture starts clean")
        lib.llama_memory_clear("mem", True); lib.decode(FINAL)
        self.assertEqual(client._prepare_memory_with_ornith_rolling_route_anchor(self.B3), len(self.B2), "reuse resumes")
        self.assertEqual(lib.recurrent, fold(self.B2))

    def test_the_miss_counter_never_authorizes_reuse(self) -> None:
        client, lib = self._client_with_a()
        client._rolling_route_capture_allowed(self.B1, identity())
        client._rolling_route_capture_allowed(self.B2, identity())
        self.assertEqual(client._prepare_memory_with_ornith_rolling_route_anchor(self.B3), 0)
        self.assertEqual(lib.set_data_blobs, [], "a checkpoint the prompt does not extend is never restored")

    def test_two_misses_that_do_not_build_on_each_other_never_replace(self) -> None:
        # The post-tool window regime: every route prompt is [system, latest
        # user, evidence], so consecutive prompts share only the head. Nothing
        # captured there could ever be restored, so nothing is captured.
        client, lib = self._client_with_a()
        W1 = ROUTE1[:2] + [401, 402]
        W2 = ROUTE1[:2] + [403, 404, 405]
        W3 = ROUTE1[:2] + [406]
        for window in (W1, W2, W3):
            self.assertFalse(client._rolling_route_capture_allowed(window, identity()))
        self.assertEqual(client._rolling_route_anchor_state.tokens, self.CONVERSATION_A, "the chain checkpoint is kept")
        self.assertEqual(client._rolling_route_anchor_state.non_extending_misses, 3)
        self.assertEqual(client._rolling_route_anchor_state.last_miss_tokens, W3)
        # ... and the original chain still reuses when it comes back
        self.assertEqual(client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2), len(ROUTE1))

    def test_identity_drift_and_invalid_state_replace_regardless(self) -> None:
        state = RollingRouteAnchorState()
        self.assertTrue(rolling_route_should_replace(state, self.B1, identity()))
        client, _ = self._client_with_a()
        self.assertTrue(client._rolling_route_capture_allowed(self.B1, identity(reset_generation=1)))


if __name__ == "__main__":
    unittest.main()
