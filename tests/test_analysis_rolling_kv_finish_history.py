"""Rolling ANALYSIS KV for consecutive FINISH turns, through the real wiring.

ANALYSIS-FINISH-PREFILL-1. Stage A checkpoints a control turn before its
assistant opener, which its repair extends; the next FINISH does not -- it
closes another question with another completion message, and the retained
replay shows every later FINISH's tokens equal its predecessor's exactly up to
the previous completion message and no further. So a control turn now takes a
second, earlier checkpoint: the history before its own trailing user turn(s),
in a slot of its own (`_control_history`), which the next FINISH extends.
Stage A's checkpoint, slot and repair reuse are untouched; the STEP slot is
untouched; the CHAT route is untouched.

Behaviour-first, as the Stage A/B suites: the capture block is lifted from
`_complete_prompt_standard` and executed against recorded tokens; the restore
goes through `_prepare_memory_with_ornith_rolling_route_anchor` and asserts
WHICH bytes were restored; the head is rendered by `_control_history_head`
over a recording renderer; `complete_chat` is driven to the prefill.
"""
from __future__ import annotations

import inspect
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.native_llama.client as client_module
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID
from orbit.native_llama.rolling_anchor_store import RollingAnchorStore
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ANALYSIS_STRATEGY_ID,
    ROLLING_CONTROL_HISTORY_STRATEGY_ID,
    ROLLING_ROUTE_STRATEGY_ID,
    ROLLING_STEP_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
    rolling_capture_boundary,
    rolling_route_reuse_start,
    rolling_step_boundary,
)

from tests.test_analysis_rolling_kv import identity
from tests.test_analysis_rolling_kv_control_repair import _boundary_block_source
from tests.test_analysis_rolling_kv_step_slot import (
    GEN_TEXT,
    _RecordingLib,
    _Session,
    _render_messages,
    int_tokenize,
    tokens_as_text,
)

# A control call as tokens: [control system, history..., completion user turn,
# opener]. The next FINISH repeats the history (grown by one action) under
# ANOTHER completion message; the repair repeats everything up to the opener.
GEN = [900, 901]
HISTORY1 = [10, 11, 12, 13, 14]                    # control system + history through action 1
FINISH_USER1 = [30, 31, 32]                        # "The question was: Q1 ... produced ... Call finish"
FINISH1 = HISTORY1 + FINISH_USER1 + GEN
REPAIR_USER = [40, 41]
REPAIR1 = HISTORY1 + FINISH_USER1 + REPAIR_USER + GEN
HISTORY2 = HISTORY1 + [15, 16, 17]                 # + analyst turn, assistant call, tool result
FINISH_USER2 = [50, 51, 52, 53]                    # Q2's completion message
FINISH2 = HISTORY2 + FINISH_USER2 + GEN
HISTORY_BYTES = b"control-history-checkpoint"
CONTROL_BYTES = b"control-stage-a-checkpoint-bytes"
STEP_BYTES = b"step-checkpoint"


def control_identity(**overrides) -> RollingRouteIdentity:
    return identity(ROLLING_ANALYSIS_STRATEGY_ID, **{"tool_schema_hash": "finish-tools", **overrides})


def history_identity(**overrides) -> RollingRouteIdentity:
    return identity(ROLLING_CONTROL_HISTORY_STRATEGY_ID, **{"tool_schema_hash": "finish-tools", **overrides})


def state(tokens, ident, data) -> RollingRouteAnchorState:
    return RollingRouteAnchorState(identity=ident, tokens=list(tokens), checkpoint_data=data, created_at_monotonic=1.0)


# --------------------------------------------------------------------------
class HeadTests(unittest.TestCase):
    """`_control_history_head`: before the trailing run of user turns."""

    FINISH = [
        {"role": "system", "content": "CONTROL"},
        {"role": "user", "content": "artifact"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "The question was: Q1. Call finish_analysis_question."},
    ]

    def _client(self, *, bridge=True, generation_prompt=GEN_TEXT):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.model_profile = mock.Mock(uses_native_chat_bridge=bridge, profile_id=ORNITH15_PROFILE_ID)
        rendered: list[list] = []

        def render(messages, *, tools=None, thinking=None):
            rendered.append(list(messages))
            prompt = _render_messages(messages, tools=tools, thinking=thinking)
            client._active_profile_render = {"prompt": prompt, "generation_prompt": generation_prompt}
            return prompt

        client.apply_chat_template = render  # type: ignore[method-assign]
        return client, rendered

    def test_a_finish_is_cut_before_its_completion_message(self) -> None:
        client, rendered = self._client()
        head = client._control_history_head(self.FINISH, tools=[{"f": 1}], thinking=False)
        self.assertEqual(rendered, [self.FINISH[:-1]])
        self.assertTrue(head.endswith("result<|im_end|>\n"), "the history ends with the tool result")
        self.assertNotIn("The question was", head)
        self.assertTrue(_render_messages(self.FINISH, tools=[{"f": 1}], thinking=False).startswith(head))

    def test_a_repair_is_cut_before_both_trailing_user_turns(self) -> None:
        """M7 witness: the repair's boundary is the FINISH's, not after it."""
        client, rendered = self._client()
        repair = [*self.FINISH, {"role": "user", "content": "The previous control response could not be parsed"}]
        head = client._control_history_head(repair, tools=None, thinking=False)
        self.assertEqual(rendered, [self.FINISH[:-1]])
        self.assertNotIn("The question was", head)
        self.assertNotIn("could not be parsed", head)

    def test_a_plan_shaped_call_has_no_history_to_checkpoint(self) -> None:
        """PLAN has no tool turn yet: only the system turn precedes its user
        run, and a later FINISH cannot extend a PLAN's system turn (its tool
        schema differs). No head, no render, no ~75 MB snapshot for nothing."""
        client, rendered = self._client()
        plan = [{"role": "system", "content": "CONTROL"}, {"role": "user", "content": "artifact"},
                {"role": "user", "content": "plan please"}]
        self.assertIsNone(client._control_history_head(plan, tools=None, thinking=False))
        self.assertEqual(rendered, [])

    def test_fail_closed(self) -> None:
        client, _ = self._client(bridge=False)
        self.assertIsNone(client._control_history_head(self.FINISH, tools=None, thinking=False))
        client, _ = self._client()
        self.assertIsNone(client._control_history_head([{"role": "user", "content": "only"}], tools=None, thinking=False))
        self.assertIsNone(client._control_history_head([{"role": "system", "content": "s"}, {"role": "tool", "content": "t"}], tools=None, thinking=False),
                          "no trailing user turn: nothing is transient here")
        client, _ = self._client(generation_prompt=None)
        self.assertIsNone(client._control_history_head(self.FINISH, tools=None, thinking=False))


# --------------------------------------------------------------------------
class CaptureBlockTests(unittest.TestCase):
    """The control lineage takes two checkpoints; the STEP and route none extra."""

    def _run(self, *, ident, head, suffix, prompt_tokens, processed, existing=(), cancelled=False):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        store = client._rolling_anchor_store()
        for st in existing:
            store.store(st.identity, st)
        client.cancel_event = threading.Event()
        if cancelled:
            client.cancel_event.set()
        client._session = _Session()
        client.tokenize = int_tokenize  # type: ignore[method-assign]
        decoded: list[tuple[int, int]] = []

        def fake_decode(token_array, *, processed, end, step, total, **_kw):
            decoded.append((processed, end))
            return end

        client._decode_prompt_range = fake_decode  # type: ignore[method-assign]
        captured: list[tuple[str, list[int]]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            captured.append((identity.strategy_id, list(prompt_tokens)))
            return state(prompt_tokens, identity, b"cap-" + identity.strategy_id.encode()), {}

        g = {
            "rolling_route_eligible": True, "rolling_route_identity": ident,
            "ROLLING_ANALYSIS_STRATEGY_ID": ROLLING_ANALYSIS_STRATEGY_ID,
            "ROLLING_STEP_STRATEGY_ID": ROLLING_STEP_STRATEGY_ID,
            "ROLLING_CONTROL_HISTORY_STRATEGY_ID": ROLLING_CONTROL_HISTORY_STRATEGY_ID,
            "replace": client_module.replace,
            "rolling_capture_boundary": rolling_capture_boundary, "rolling_step_boundary": rolling_step_boundary,
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
            "prompt": tokens_as_text(prompt_tokens), "prompt_tokens": list(prompt_tokens),
            "rolling_boundary_suffix": suffix, "rolling_boundary_head": head,
            "processed": processed, "reused": processed, "token_array": object(), "step": 64,
            "n_prompt": len(prompt_tokens), "on_progress": None, "should_cancel": None,
            "pf_start": 0, "lib": object(), "self": client,
        }
        exec(_boundary_block_source(), g)
        return client, g, decoded, captured

    HEAD1 = tokens_as_text(HISTORY1) + " "
    HEAD2 = tokens_as_text(HISTORY2) + " "
    SUFFIX = tokens_as_text(GEN)

    def test_a_finish_checkpoints_its_history_then_its_stage_a_boundary(self) -> None:
        client, g, decoded, captured = self._run(ident=control_identity(), head=self.HEAD1, suffix=self.SUFFIX,
                                                 prompt_tokens=FINISH1, processed=0)
        self.assertEqual(decoded, [(0, len(HISTORY1)), (len(HISTORY1), len(HISTORY1 + FINISH_USER1))])
        self.assertEqual(captured, [(ROLLING_CONTROL_HISTORY_STRATEGY_ID, HISTORY1),
                                    (ROLLING_ANALYSIS_STRATEGY_ID, HISTORY1 + FINISH_USER1)])
        self.assertEqual(client._rolling_control_history_anchor_state.tokens, HISTORY1)
        self.assertEqual(client._rolling_analysis_anchor_state.tokens, HISTORY1 + FINISH_USER1, "Stage A unchanged")
        self.assertFalse(client._rolling_step_anchor_state.valid)
        self.assertFalse(set(FINISH_USER1) & set(client._rolling_control_history_anchor_state.tokens),
                         "M7: no completion-message token in the history checkpoint")

    def test_the_history_checkpoint_serves_the_next_finish_and_the_stage_a_one_the_repair(self) -> None:
        client, *_ = self._run(ident=control_identity(), head=self.HEAD1, suffix=self.SUFFIX, prompt_tokens=FINISH1, processed=0)
        history = client._rolling_control_history_anchor_state
        control = client._rolling_analysis_anchor_state
        self.assertEqual(rolling_route_reuse_start(history, FINISH2, history_identity()), len(HISTORY1), "T1/T4")
        self.assertIsNone(rolling_route_reuse_start(control, FINISH2, control_identity()), "Stage A's cannot serve FINISH 2")
        self.assertEqual(rolling_route_reuse_start(control, REPAIR1, control_identity()), len(HISTORY1 + FINISH_USER1), "T11")
        self.assertEqual(rolling_route_reuse_start(history, REPAIR1, history_identity()), len(HISTORY1), "shorter, so not chosen")

    def test_a_repair_does_not_move_the_history_checkpoint(self) -> None:
        """After the Stage A restore the history boundary is already resident."""
        existing = (state(HISTORY1, history_identity(), HISTORY_BYTES), state(HISTORY1 + FINISH_USER1, control_identity(), CONTROL_BYTES))
        client, g, decoded, captured = self._run(ident=control_identity(), head=self.HEAD1, suffix=self.SUFFIX,
                                                 prompt_tokens=REPAIR1, processed=len(HISTORY1 + FINISH_USER1), existing=existing)
        self.assertEqual(captured, [(ROLLING_ANALYSIS_STRATEGY_ID, HISTORY1 + FINISH_USER1 + REPAIR_USER)])
        self.assertEqual(client._rolling_control_history_anchor_state.tokens, HISTORY1)
        self.assertEqual(client._rolling_control_history_anchor_state.checkpoint_data, HISTORY_BYTES)

    def test_the_next_finish_advances_the_history_checkpoint_after_restoring_it(self) -> None:
        existing = (state(HISTORY1, history_identity(), HISTORY_BYTES),)
        client, g, decoded, captured = self._run(ident=control_identity(), head=self.HEAD2, suffix=self.SUFFIX,
                                                 prompt_tokens=FINISH2, processed=len(HISTORY1), existing=existing)
        self.assertEqual(decoded[0], (len(HISTORY1), len(HISTORY2)))
        self.assertEqual(captured[0], (ROLLING_CONTROL_HISTORY_STRATEGY_ID, HISTORY2))
        self.assertEqual(client._rolling_control_history_anchor_state.tokens, HISTORY2)
        self.assertEqual(client._rolling_analysis_anchor_state.tokens, HISTORY2 + FINISH_USER2)

    def test_no_head_means_no_history_checkpoint_and_stage_a_unchanged(self) -> None:
        client, g, decoded, captured = self._run(ident=control_identity(), head=None, suffix=self.SUFFIX, prompt_tokens=FINISH1, processed=0)
        self.assertEqual(captured, [(ROLLING_ANALYSIS_STRATEGY_ID, HISTORY1 + FINISH_USER1)])
        self.assertFalse(client._rolling_control_history_anchor_state.valid)

    def test_a_head_that_does_not_open_the_prompt_captures_no_history(self) -> None:
        client, g, decoded, captured = self._run(ident=control_identity(), head="10 11 99 ", suffix=self.SUFFIX, prompt_tokens=FINISH1, processed=0)
        self.assertEqual([c[0] for c in captured], [ROLLING_ANALYSIS_STRATEGY_ID])

    def test_a_cancellation_during_the_history_prefill_captures_nothing(self) -> None:
        client, g, decoded, captured = self._run(ident=control_identity(), head=self.HEAD1, suffix=self.SUFFIX, prompt_tokens=FINISH1, processed=0, cancelled=True)
        self.assertEqual(captured, [])
        self.assertFalse(client._rolling_control_history_anchor_state.valid)

    def test_the_step_lineage_takes_no_history_checkpoint(self) -> None:
        """T12: a STEP still captures exactly one checkpoint, into the STEP slot."""
        client, g, decoded, captured = self._run(ident=identity(ROLLING_STEP_STRATEGY_ID, tool_schema_hash="step-tools"),
                                                 head=self.HEAD1, suffix=None, prompt_tokens=HISTORY1 + [77, 78] + GEN, processed=0)
        self.assertEqual(captured, [(ROLLING_STEP_STRATEGY_ID, HISTORY1)])
        self.assertFalse(client._rolling_control_history_anchor_state.valid)

    def test_the_route_lineage_is_untouched(self) -> None:
        client, g, decoded, captured = self._run(ident=identity(ROLLING_ROUTE_STRATEGY_ID), head=self.HEAD1, suffix=self.SUFFIX, prompt_tokens=FINISH1, processed=0)
        self.assertEqual(captured, [])


# --------------------------------------------------------------------------
class RestoreTests(unittest.TestCase):
    """Through the production strategy method: which bytes, for which prompt."""

    def _client(self, ident, *, history=None, control=None, step=None):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        lib = _RecordingLib()
        client.lib = mock.Mock(lib=lib)
        client._session = _Session()
        store = client._rolling_anchor_store()
        for st in (history, control, step):
            if st is not None:
                store.store(st.identity, st)
        client._rolling_route_identity_cache = ident
        prepared: list[list[int]] = []
        client._prepare_memory_for_prompt = lambda tokens: prepared.append(list(tokens)) or 0  # type: ignore[method-assign]
        client._invalidate_committed_sequence = lambda: client._session.committed_sequence_tokens.clear()  # type: ignore[method-assign]
        return client, lib, prepared

    def _both(self, ident=None):
        return self._client(ident or control_identity(),
                            history=state(HISTORY1, history_identity(), HISTORY_BYTES),
                            control=state(HISTORY1 + FINISH_USER1, control_identity(), CONTROL_BYTES),
                            step=state(HISTORY1, identity(ROLLING_STEP_STRATEGY_ID, tool_schema_hash="step-tools"), STEP_BYTES))

    def test_t1_t4_the_next_finish_restores_the_history_bytes(self) -> None:
        client, lib, prepared = self._both()
        client._prepare_memory_with_ornith_rolling_route_anchor(FINISH2)
        self.assertEqual(lib.restored, [HISTORY_BYTES])
        self.assertEqual(client._session.committed_sequence_tokens, HISTORY1)
        self.assertEqual(prepared, [FINISH2], "strict append still judges the whole prompt")

    def test_t11_the_repair_still_restores_the_longer_stage_a_bytes(self) -> None:
        client, lib, prepared = self._both()
        client._prepare_memory_with_ornith_rolling_route_anchor(REPAIR1)
        self.assertEqual(lib.restored, [CONTROL_BYTES], "the longer exact match wins")
        self.assertEqual(client._session.committed_sequence_tokens, HISTORY1 + FINISH_USER1)

    def test_t2_a_one_token_mutation_before_the_boundary_refuses(self) -> None:
        for position in (0, 2, len(HISTORY1) - 1):
            mutated = list(FINISH2); mutated[position] = 99
            client, lib, prepared = self._both()
            client._prepare_memory_with_ornith_rolling_route_anchor(mutated)
            self.assertEqual(lib.restored, [], f"position {position}")
            self.assertEqual(prepared, [mutated])

    def test_t3_an_incompatible_question_lineage_refuses(self) -> None:
        """A FINISH whose history was rewritten before the boundary (T8) or
        that belongs to another lineage: nothing exact, nothing restored."""
        rewritten = HISTORY1[:3] + [66, 67] + [15, 16, 17] + FINISH_USER2 + GEN
        client, lib, _ = self._both()
        client._prepare_memory_with_ornith_rolling_route_anchor(rewritten)
        self.assertEqual(lib.restored, [])

    def test_t5_t6_t7_identity_dimensions_refuse(self) -> None:
        for field, value in (("tool_schema_hash", "plan-tools"), ("runtime_policy_hash", "ctx-4096"), ("session_id", "other"),
                             ("model_id", "/models/other.gguf"), ("template_id", "tpl-2"), ("reset_generation", 1),
                             ("native_version", "new.so"), ("profile_id", "other"), ("capability_summary_hash", "x")):
            client, lib, _ = self._both(control_identity(**{field: value}))
            client._prepare_memory_with_ornith_rolling_route_anchor(FINISH2)
            self.assertEqual(lib.restored, [], f"{field}={value!r}")

    def test_a_finish_never_restores_the_step_bytes(self) -> None:
        """M2: the STEP slot holds the same tokens; the control lineage must not read it."""
        client, lib, _ = self._client(control_identity(),
                                      step=state(HISTORY1, identity(ROLLING_STEP_STRATEGY_ID, tool_schema_hash="step-tools"), STEP_BYTES))
        client._prepare_memory_with_ornith_rolling_route_anchor(FINISH2)
        self.assertEqual(lib.restored, [])

    def test_a_step_never_restores_the_history_bytes(self) -> None:
        client, lib, _ = self._client(identity(ROLLING_STEP_STRATEGY_ID, tool_schema_hash="step-tools"),
                                      history=state(HISTORY1, history_identity(), HISTORY_BYTES))
        client._prepare_memory_with_ornith_rolling_route_anchor(HISTORY2 + [77] + GEN)
        self.assertEqual(lib.restored, [])

    def test_the_longer_exact_match_wins_whichever_slot_holds_it(self) -> None:
        client, lib, _ = self._client(control_identity(), history=state(HISTORY2, history_identity(), HISTORY_BYTES),
                                      control=state(HISTORY1 + FINISH_USER1, control_identity(), CONTROL_BYTES))
        client._prepare_memory_with_ornith_rolling_route_anchor(FINISH2)
        self.assertEqual(lib.restored, [HISTORY_BYTES])
        self.assertEqual(client._session.committed_sequence_tokens, HISTORY2)


# --------------------------------------------------------------------------
class WiringAndLifecycleTests(unittest.TestCase):
    def test_the_store_isolates_the_fourth_slot(self) -> None:
        store = RollingAnchorStore()
        self.assertEqual(RollingAnchorStore.slot_for(history_identity()), "control_history")
        store.store(history_identity(), state(HISTORY1, history_identity(), HISTORY_BYTES))
        for other in (control_identity(), identity(ROLLING_STEP_STRATEGY_ID), identity(ROLLING_ROUTE_STRATEGY_ID)):
            self.assertFalse(store.state_for(other).valid)
        store.store(control_identity(), state(HISTORY1 + FINISH_USER1, control_identity(), CONTROL_BYTES))
        self.assertIs(store.control_history_state.checkpoint_data, HISTORY_BYTES, "a control capture does not evict the history")
        store.invalidate("session_reset"); store.invalidate("later")
        self.assertFalse(store.control_history_state.valid)
        self.assertEqual(store.control_history_state.invalidation_reason, "session_reset")

    def test_t9_reset_session_state_destroys_the_history_checkpoint(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._session = _Session(); client._session.cancel_requested = False
        client._session.prompt_cache_mode = "tools:thinking=off"; client._session.continuation_ready = True; client._session.last_metrics = object()
        client.cancel_event = threading.Event()
        client.lib = mock.Mock(lib=mock.Mock(llama_get_memory=lambda ctx: object(), llama_memory_clear=lambda mem, flag: None))
        client.config = mock.Mock(use_mtp_experimental=False); client._reset_generation = 0; client._persistent_mtp_runtime = None
        client._store_rolling_anchor_state(history_identity(), state(HISTORY1, history_identity(), HISTORY_BYTES))
        client._invalidate_committed_sequence = lambda: None  # type: ignore[method-assign]
        client._invalidate_final_prefix = lambda reason: None  # type: ignore[method-assign]
        client._invalidate_qwen_route_prefix = lambda reason, profile_id=None: None  # type: ignore[method-assign]
        client._invalidate_qwen36_shell_tool_prefix = lambda reason: None  # type: ignore[method-assign]
        client.reset_session_state()
        st = client._rolling_control_history_anchor_state
        self.assertFalse(st.valid); self.assertEqual(st.invalidation_reason, "session_reset")
        self.assertIsNone(rolling_route_reuse_start(st, FINISH2, history_identity(reset_generation=1)))

    def test_complete_chat_hands_the_history_head_to_a_control_turn_only(self) -> None:
        from tests.test_analysis_rolling_kv_step_slot import CompleteChatStepWiringTest
        harness = CompleteChatStepWiringTest("test_a_step_turn_takes_the_step_strategy_and_hands_over_its_head")
        client, seen, renders = harness._client()
        finish = HeadTests.FINISH
        with mock.patch.object(client_module, "prepare_multimodal_messages", lambda *a, **k: None):
            client.complete_chat(finish, tools=[{"f": 1}], analysis_rolling_anchor=True)
        call = seen[0]
        self.assertEqual(call["rolling_route_identity"].strategy_id, ROLLING_ANALYSIS_STRATEGY_ID)
        self.assertEqual(call["rolling_boundary_head"], _render_messages(finish[:-1], tools=[{"f": 1}], thinking=False)[: -len(GEN_TEXT)])
        self.assertEqual(call["rolling_boundary_suffix"], GEN_TEXT)
        self.assertEqual(renders, [finish[:-1], finish], "head first, production render last")
        seen.clear(); renders.clear()
        with mock.patch.object(client_module, "prepare_multimodal_messages", lambda *a, **k: None):
            client.complete_chat(finish, tools=[{"f": 1}])
        self.assertIsNone(seen[0]["rolling_boundary_head"])
        self.assertEqual(renders, [finish])

    def test_t10_the_lifted_block_still_captures_the_step_slot_exactly_as_stage_b(self) -> None:
        source = inspect.getsource(NativeLlamaClient._complete_prompt_standard)
        self.assertIn("elif step_lineage:", source)
        self.assertLess(source.index("history_boundary = rolling_step_boundary("), source.index("elif step_lineage:"))


if __name__ == "__main__":
    unittest.main()
