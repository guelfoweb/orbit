"""Rolling ANALYSIS KV for a control turn and its repair, through the real wiring.

The rolling ANALYSIS anchor shipped in #224 and measured 82-88% reuse on
consecutive steps. The structured controller then added PLAN and FINISH
phases, and two things silently excluded them from the lineage:

  * the backend's eligibility gate was an equality test against the step
    phase, written when that was the only analysis model call;
  * the checkpoint was taken at the END of the prompt, so it carried the text
    that opens the assistant turn -- and no same-phase successor ever repeats
    that text. A repair puts a user turn there; a continuation puts the
    reply. Measured on the retained traces, every same-phase miss diverged at
    exactly that point.

These tests drive the production methods, not the anchor module in
isolation: the boundary block inside `_complete_prompt_standard` is lifted
from its source and executed against recorded inputs, the restore goes
through `_prepare_memory_with_ornith_rolling_route_anchor`, and the gate is
the backend's own. Each asserts that the hook it depends on was actually
reached; a test that checked only a return value would pass just as happily
if the wiring were deleted.
"""

from __future__ import annotations

import inspect
import sys
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.backend.llama_server as llama_server
import orbit.native_llama.client as client_module
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ANALYSIS_STRATEGY_ID,
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
    rolling_capture_boundary,
    rolling_route_reuse_start,
)
from orbit.runtime.analysis_runtime import (
    ANALYSIS_COVER_PHASE,
    ANALYSIS_FINISH_PHASE,
    ANALYSIS_PLAN_PHASE,
    ANALYSIS_REPORT_PHASE,
    ANALYSIS_STEP_PHASE,
)
from orbit.runtime.kv_diag import model_call_context

from tests.test_analysis_rolling_kv import identity, strategy_client

# A control prompt as tokens: the messages, then the generation prompt that
# opens the assistant turn. The repair is the same messages plus a user turn,
# then its own generation prompt. Only the messages are shared.
GEN = [900, 901]                  # the generation-prompt tokens
MESSAGES = [10, 11, 12, 13, 14]   # everything before the generation prompt
CONTROL = MESSAGES + GEN
REPAIR = MESSAGES + [20, 21, 22] + GEN
CHECKPOINT = b"control-checkpoint"


def analysis_identity(**overrides) -> RollingRouteIdentity:
    return identity(ROLLING_ANALYSIS_STRATEGY_ID, **overrides)


def valid_state(tokens, ident=None) -> RollingRouteAnchorState:
    return RollingRouteAnchorState(
        identity=ident or analysis_identity(),
        tokens=list(tokens),
        checkpoint_data=CHECKPOINT,
        created_at_monotonic=1.0,
    )


# --------------------------------------------------------------------------
# A tokenizer over a tiny language, so a boundary can land inside a token.
# --------------------------------------------------------------------------
def word_tokenize(text: str) -> list[int]:
    """One token per whitespace-separated word, stable across calls."""
    return [abs(hash(("tok", w))) % 100000 + 1 for w in text.split()]


class BoundaryHelperTest(unittest.TestCase):
    """`rolling_capture_boundary`: where the next turn can still extend."""

    def test_the_boundary_is_the_prompt_without_its_generation_prompt(self) -> None:
        prompt = "sys user tool <GEN>"
        tokens = word_tokenize(prompt)
        boundary = rolling_capture_boundary(
            prompt, tokens, generation_prompt="<GEN>", tokenize=word_tokenize
        )
        self.assertEqual(boundary, 3)
        self.assertEqual(tokens[:boundary], word_tokenize("sys user tool"))

    def test_the_boundary_is_a_prefix_of_the_repair_prompt(self) -> None:
        """The property the whole change rests on, on the shape a repair has."""
        control = "sys user tool <GEN>"
        repair = "sys user tool repair <GEN>"
        boundary = rolling_capture_boundary(
            control, word_tokenize(control), generation_prompt="<GEN>",
            tokenize=word_tokenize,
        )
        self.assertIsNotNone(boundary)
        self.assertEqual(
            word_tokenize(repair)[:boundary], word_tokenize(control)[:boundary]
        )
        self.assertNotEqual(
            word_tokenize(repair)[: len(word_tokenize(control))],
            word_tokenize(control),
            "the whole control prompt is NOT a prefix of the repair: that is the miss",
        )

    def test_no_generation_prompt_means_no_boundary(self) -> None:
        prompt = "sys user tool <GEN>"
        tokens = word_tokenize(prompt)
        self.assertIsNone(rolling_capture_boundary(
            prompt, tokens, generation_prompt=None, tokenize=word_tokenize))
        self.assertIsNone(rolling_capture_boundary(
            prompt, tokens, generation_prompt="", tokenize=word_tokenize))

    def test_a_prompt_that_does_not_end_with_it_is_refused(self) -> None:
        prompt = "sys user tool <GEN> trailing"
        self.assertIsNone(rolling_capture_boundary(
            prompt, word_tokenize(prompt), generation_prompt="<GEN>",
            tokenize=word_tokenize))

    def test_a_boundary_inside_a_token_is_refused_not_rounded(self) -> None:
        prompt = "sys user tool <GEN>"
        tokens = word_tokenize(prompt)
        # "EN>" is a suffix of the text but not of any token: the head ends
        # mid-word and re-tokenizes to something that is not a prefix.
        self.assertIsNone(rolling_capture_boundary(
            prompt, tokens, generation_prompt="EN>", tokenize=word_tokenize))

    def test_a_head_that_differs_at_any_single_position_is_refused(self) -> None:
        """The prefix check is exact at EVERY position of the head.

        A head can re-tokenize to the same length as the prompt's prefix and
        differ at one position only -- a word that tokenizes one way mid-text
        and another way at end-of-text is the usual cause at the last
        position, but a comparison that skipped the first token, or any
        middle one, would be just as wrong. The cost of accepting such a head
        is not a wrong restore (the snapshot is still exactly the resident
        tokens) but a boundary no successor can ever extend, i.e. a
        checkpoint that is captured and never used.

        None of these cases is masked by the whole-prompt guard: the head is
        strictly shorter than the prompt. The fake tokenizer RAISES on any
        text it was not told about, so a change to how the head is derived
        cannot make the test pass by never reaching the comparison.
        """
        full = [101, 102, 103, 900]           # sys user tool <GEN>
        head_text = "sys user tool "

        for position in range(3):
            with self.subTest(position=position):
                retokenized = full[:3]
                retokenized[position] += 1     # same length, one token differs

                def tokenize(text: str, _head=retokenized) -> list[int]:
                    if text == head_text:
                        return list(_head)
                    raise AssertionError(f"unexpected tokenize input: {text!r}")

                self.assertIsNone(rolling_capture_boundary(
                    "sys user tool <GEN>", full, generation_prompt="<GEN>",
                    tokenize=tokenize,
                ))

        # And the unmutated head, through the same strict fake, is accepted.
        def exact(text: str) -> list[int]:
            if text == head_text:
                return full[:3]
            raise AssertionError(f"unexpected tokenize input: {text!r}")

        self.assertEqual(rolling_capture_boundary(
            "sys user tool <GEN>", full, generation_prompt="<GEN>", tokenize=exact,
        ), 3)

    def test_an_empty_head_or_whole_prompt_boundary_is_refused(self) -> None:
        self.assertIsNone(rolling_capture_boundary(
            "<GEN>", word_tokenize("<GEN>"), generation_prompt="<GEN>",
            tokenize=word_tokenize))
        # A tokenizer that returns the full prompt for the head leaves nothing
        # to evaluate; the final token must be decoded for fresh logits.
        prompt = "a b <GEN>"
        self.assertIsNone(rolling_capture_boundary(
            prompt, word_tokenize(prompt), generation_prompt="<GEN>",
            tokenize=lambda _text: word_tokenize(prompt)))

    def test_the_precondition_test_case_now_yields_a_usable_boundary(self) -> None:
        """The counter-case the historical suite keeps, turned into a boundary.

        `test_a_template_that_replaces_its_generation_prompt_breaks_reuse`
        shows a renderer whose assistant turn does not re-emit `<GEN>`; the
        second prompt starts with the first minus `<GEN>`. That is exactly the
        boundary this helper finds.
        """
        first = "[system:s][user:u]<GEN>"
        second = "[system:s][user:u][assistant:a][user:v]<GEN>"
        tokenize = lambda text: [ord(c) for c in text]  # noqa: E731
        boundary = rolling_capture_boundary(
            first, tokenize(first), generation_prompt="<GEN>", tokenize=tokenize
        )
        self.assertEqual(boundary, len(first) - len("<GEN>"))
        self.assertEqual(tokenize(second)[:boundary], tokenize(first)[:boundary])


class EligibilityGateTest(unittest.TestCase):
    """The control phases join the lineage; REPORT and COVER do not."""

    def _requested(self, phase: str | None, *, native: bool = True) -> bool:
        with mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: True):
            if phase is None:
                return llama_server._analysis_rolling_anchor_requested(native_backend=native)
            with model_call_context(phase=phase, tools_mode="on"):
                return llama_server._analysis_rolling_anchor_requested(native_backend=native)

    def test_plan_and_finish_join_the_lineage(self) -> None:
        self.assertTrue(self._requested(ANALYSIS_PLAN_PHASE))
        self.assertTrue(self._requested(ANALYSIS_FINISH_PHASE))
        self.assertTrue(self._requested(ANALYSIS_STEP_PHASE))

    def test_a_finish_phase_carries_its_question_as_a_suffix(self) -> None:
        self.assertTrue(self._requested(f"{ANALYSIS_FINISH_PHASE}:Q1"))
        self.assertTrue(self._requested(f"{ANALYSIS_FINISH_PHASE}:Q12"))

    def test_report_and_cover_stay_out(self) -> None:
        self.assertFalse(self._requested(ANALYSIS_REPORT_PHASE))
        self.assertFalse(self._requested(ANALYSIS_COVER_PHASE))

    def test_a_lookalike_phase_is_not_a_member(self) -> None:
        # Prefix membership is on `<phase>:`, not on any string that merely
        # starts with the phase text.
        self.assertFalse(self._requested(f"{ANALYSIS_STEP_PHASE}_extra"))
        self.assertFalse(self._requested(f"{ANALYSIS_PLAN_PHASE}x"))

    def test_route_and_no_phase_and_non_native_stay_out(self) -> None:
        self.assertFalse(self._requested("route"))
        self.assertFalse(self._requested(None))
        self.assertFalse(self._requested(ANALYSIS_PLAN_PHASE, native=False))

    def test_anchor_disabled_refuses_every_phase(self) -> None:
        with mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: False):
            for phase in (ANALYSIS_STEP_PHASE, ANALYSIS_PLAN_PHASE, ANALYSIS_FINISH_PHASE):
                with model_call_context(phase=phase, tools_mode="on"):
                    self.assertFalse(
                        llama_server._analysis_rolling_anchor_requested(native_backend=True)
                    )


class GenerationPromptSuffixTest(unittest.TestCase):
    """The boundary comes from the renderer's own report, or not at all."""

    def _client(self, *, bridge: bool, render):
        client = NativeLlamaClient.__new__(NativeLlamaClient)

        class Profile:
            uses_native_chat_bridge = bridge

        client.model_profile = Profile()
        if render is not None:
            client._active_profile_render = render
        return client

    def test_the_bridge_render_supplies_the_suffix(self) -> None:
        client = self._client(bridge=True, render={"prompt": "x", "generation_prompt": "<|turn>model\n"})
        self.assertEqual(client._generation_prompt_suffix(), "<|turn>model\n")

    def test_a_non_bridge_profile_never_reads_a_stale_render(self) -> None:
        client = self._client(bridge=False, render={"generation_prompt": "<|turn>model\n"})
        self.assertIsNone(client._generation_prompt_suffix())

    def test_missing_or_empty_render_yields_none(self) -> None:
        self.assertIsNone(self._client(bridge=True, render=None)._generation_prompt_suffix())
        self.assertIsNone(self._client(bridge=True, render={})._generation_prompt_suffix())
        self.assertIsNone(
            self._client(bridge=True, render={"generation_prompt": ""})._generation_prompt_suffix()
        )


# --------------------------------------------------------------------------
# The boundary block inside `_complete_prompt_standard`, lifted verbatim.
# --------------------------------------------------------------------------
def _boundary_block_source() -> str:
    """From the `capture_at` decision through the boundary capture.

    Read from the installed source so the test cannot drift from the code it
    protects: if the block changes, this text changes with it.
    """
    source = inspect.getsource(NativeLlamaClient._complete_prompt_standard)
    start = source.index("        capture_at: int | None = None")
    end = source.index("        while processed < n_prompt and not self.cancel_event.is_set():", start)
    return textwrap.dedent(source[start:end])


class _Session:
    def __init__(self) -> None:
        self.ctx_tgt = object()
        self.session_id = "default"
        self.cached_prompt_tokens: list[int] = []
        self.committed_sequence_tokens: list[int] = []
        self.mtp_enabled = False


class BoundaryCaptureBlockTest(unittest.TestCase):
    """What the ANALYSIS lineage snapshots, and when it does not."""

    def _run(
        self,
        *,
        ident: RollingRouteIdentity,
        suffix: str | None,
        prompt_tokens: list[int],
        processed: int,
        existing: RollingRouteAnchorState | None = None,
        cancelled: bool = False,
        capture_result: RollingRouteAnchorState | None = None,
        eligible: bool = True,
    ):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._rolling_analysis_anchor_state = existing or RollingRouteAnchorState()
        client._rolling_route_anchor_state = RollingRouteAnchorState()
        client.cancel_event = threading.Event()
        if cancelled:
            client.cancel_event.set()
        client._session = _Session()
        # The "prompt" is its tokens joined as words, so the real helper can
        # re-tokenize the head and the boundary lands where the suffix says.
        words = [str(t) for t in prompt_tokens]
        prompt = " ".join(words)
        client.tokenize = lambda text: [int(w) for w in text.split()]  # type: ignore[method-assign]

        decoded: list[tuple[int, int]] = []

        def fake_decode(token_array, *, processed, end, step, total, **_kw):
            decoded.append((processed, end))
            return end

        client._decode_prompt_range = fake_decode  # type: ignore[method-assign]
        captured: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            captured.append(list(prompt_tokens))
            return (capture_result or valid_state(prompt_tokens, ident)), {}

        exec_globals = {
            "rolling_route_eligible": eligible,
            "rolling_route_identity": ident,
            "ROLLING_ANALYSIS_STRATEGY_ID": ROLLING_ANALYSIS_STRATEGY_ID,
            "ROLLING_STEP_STRATEGY_ID": client_module.ROLLING_STEP_STRATEGY_ID,
            "ROLLING_CONTROL_HISTORY_STRATEGY_ID": client_module.ROLLING_CONTROL_HISTORY_STRATEGY_ID,
            "replace": client_module.replace,
            "rolling_capture_boundary": rolling_capture_boundary,
            "rolling_step_boundary": client_module.rolling_step_boundary,
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
            "prompt": prompt,
            "prompt_tokens": list(prompt_tokens),
            "rolling_boundary_suffix": suffix,
            "rolling_boundary_head": None,
            "processed": processed,
            "reused": processed,
            "token_array": object(),
            "step": 64,
            "n_prompt": len(prompt_tokens),
            "on_progress": None,
            "should_cancel": None,
            "pf_start": 0,
            "lib": object(),
            "self": client,
        }
        exec(_boundary_block_source(), exec_globals)
        return client, exec_globals, decoded, captured

    def test_a_control_prompt_is_snapshotted_at_the_boundary_not_the_end(self) -> None:
        """The M7 witness: the checkpoint is the messages, never the opener."""
        client, g, decoded, captured = self._run(
            ident=analysis_identity(), suffix=" ".join(str(t) for t in GEN),
            prompt_tokens=CONTROL, processed=0,
        )
        self.assertEqual(g["capture_at"], len(MESSAGES))
        self.assertEqual(decoded, [(0, len(MESSAGES))], "prefill stops at the boundary first")
        self.assertEqual(g["processed"], len(MESSAGES))
        self.assertEqual(captured, [MESSAGES], "the snapshot is the messages, and only them")
        self.assertEqual(client._rolling_analysis_anchor_state.tokens, MESSAGES)
        self.assertNotEqual(client._rolling_analysis_anchor_state.tokens, CONTROL)

    def test_the_boundary_checkpoint_serves_the_repair(self) -> None:
        """End to end on tokens: what the block stored, the repair restores."""
        client, _g, _d, _c = self._run(
            ident=analysis_identity(), suffix=" ".join(str(t) for t in GEN),
            prompt_tokens=CONTROL, processed=0,
        )
        state = client._rolling_analysis_anchor_state
        self.assertEqual(
            rolling_route_reuse_start(state, REPAIR, analysis_identity()), len(MESSAGES)
        )
        self.assertIsNone(
            rolling_route_reuse_start(valid_state(CONTROL), REPAIR, analysis_identity()),
            "a whole-prompt checkpoint could never have served it",
        )

    def test_the_route_lineage_keeps_its_whole_prompt_capture(self) -> None:
        client, g, decoded, captured = self._run(
            ident=identity(ROLLING_ROUTE_STRATEGY_ID),
            suffix=" ".join(str(t) for t in GEN), prompt_tokens=CONTROL, processed=0,
        )
        self.assertIsNone(g["capture_at"])
        self.assertEqual(decoded, [])
        self.assertEqual(captured, [])

    def test_no_suffix_means_no_boundary_capture(self) -> None:
        client, g, decoded, captured = self._run(
            ident=analysis_identity(), suffix=None, prompt_tokens=CONTROL, processed=0,
        )
        self.assertIsNone(g["capture_at"])
        self.assertEqual(captured, [])

    def test_a_boundary_already_resident_is_not_recaptured(self) -> None:
        # Restored up to the boundary: the block must not snapshot state it
        # did not decode, and must not re-snapshot what is already held.
        existing = valid_state(MESSAGES)
        client, g, decoded, captured = self._run(
            ident=analysis_identity(), suffix=" ".join(str(t) for t in GEN),
            prompt_tokens=CONTROL, processed=len(MESSAGES), existing=existing,
        )
        self.assertIsNone(g["capture_at"])
        self.assertEqual(captured, [])
        self.assertEqual(client._rolling_analysis_anchor_state.tokens, MESSAGES)

    def test_a_longer_boundary_rolls_the_checkpoint_forward(self) -> None:
        # Restored to the control boundary, now prefilling the repair: the
        # repair's own boundary is longer and continues the chain.
        existing = valid_state(MESSAGES)
        client, g, decoded, captured = self._run(
            ident=analysis_identity(), suffix=" ".join(str(t) for t in GEN),
            prompt_tokens=REPAIR, processed=len(MESSAGES), existing=existing,
        )
        boundary = len(REPAIR) - len(GEN)
        self.assertEqual(g["capture_at"], boundary)
        self.assertEqual(decoded, [(len(MESSAGES), boundary)])
        self.assertEqual(captured, [REPAIR[:boundary]])

    def test_a_cancelled_prefill_never_captures(self) -> None:
        client, g, decoded, captured = self._run(
            ident=analysis_identity(), suffix=" ".join(str(t) for t in GEN),
            prompt_tokens=CONTROL, processed=0, cancelled=True,
        )
        self.assertEqual(g["capture_at"], len(MESSAGES))
        self.assertEqual(decoded, [], "cancelled: nothing decoded")
        self.assertEqual(captured, [])

    def test_a_cancellation_during_the_boundary_prefill_never_captures(self) -> None:
        """The guard after the loop, not just the one inside it.

        `_decode_prompt_range` advances `processed` even when the decode was
        interrupted, so a cancellation that lands during the last batch
        leaves `processed == capture_at` with a KV that was not fully written.
        The capture must still be refused: it checks the cancel flag again,
        after the loop. Cancelling BEFORE the loop (the test above) never
        reaches that check.
        """
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._rolling_analysis_anchor_state = RollingRouteAnchorState()
        client._rolling_route_anchor_state = RollingRouteAnchorState()
        client.cancel_event = threading.Event()
        client._session = _Session()
        client.tokenize = lambda text: [int(w) for w in text.split()]  # type: ignore[method-assign]

        def cancelling_decode(token_array, *, processed, end, step, total, **_kw):
            client.cancel_event.set()   # the interrupt lands mid-batch
            return end                  # ...and `processed` still advances

        client._decode_prompt_range = cancelling_decode  # type: ignore[method-assign]
        captured: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            captured.append(list(prompt_tokens))
            return valid_state(prompt_tokens), {}

        g = {
            "rolling_route_eligible": True,
            "rolling_route_identity": analysis_identity(),
            "ROLLING_ANALYSIS_STRATEGY_ID": ROLLING_ANALYSIS_STRATEGY_ID,
            "ROLLING_STEP_STRATEGY_ID": client_module.ROLLING_STEP_STRATEGY_ID,
            "ROLLING_CONTROL_HISTORY_STRATEGY_ID": client_module.ROLLING_CONTROL_HISTORY_STRATEGY_ID,
            "replace": client_module.replace,
            "rolling_capture_boundary": rolling_capture_boundary,
            "rolling_step_boundary": client_module.rolling_step_boundary,
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
            "prompt": " ".join(str(t) for t in CONTROL),
            "prompt_tokens": list(CONTROL),
            "rolling_boundary_suffix": " ".join(str(t) for t in GEN),
            "rolling_boundary_head": None,
            "processed": 0, "reused": 0, "token_array": object(), "step": 64,
            "n_prompt": len(CONTROL), "on_progress": None, "should_cancel": None,
            "pf_start": 0, "lib": object(), "self": client,
        }
        exec(_boundary_block_source(), g)
        self.assertEqual(g["processed"], len(MESSAGES), "the loop did run to the boundary")
        self.assertEqual(captured, [], "a cancelled prefill must not be snapshotted")
        self.assertFalse(client._rolling_analysis_anchor_state.valid)

    def test_a_failed_capture_preserves_the_previous_checkpoint(self) -> None:
        existing = valid_state(MESSAGES[:3])
        failed = RollingRouteAnchorState(invalidation_reason="checkpoint_capture_failed")
        client, _g, _d, captured = self._run(
            ident=analysis_identity(), suffix=" ".join(str(t) for t in GEN),
            prompt_tokens=CONTROL, processed=0, existing=existing, capture_result=failed,
        )
        self.assertEqual(captured, [MESSAGES], "capture was attempted")
        self.assertEqual(client._rolling_analysis_anchor_state.tokens, MESSAGES[:3])
        self.assertTrue(client._rolling_analysis_anchor_state.valid)

    def test_ineligible_calls_never_enter_the_block(self) -> None:
        client, g, decoded, captured = self._run(
            ident=analysis_identity(), suffix=" ".join(str(t) for t in GEN),
            prompt_tokens=CONTROL, processed=0, eligible=False,
        )
        self.assertIsNone(g["capture_at"])
        self.assertEqual(captured, [])


class WholePromptGuardSkippedTest(unittest.TestCase):
    """A boundary checkpoint must not be overwritten by the whole prompt."""

    def test_end_capture_is_skipped_after_a_boundary_capture(self) -> None:
        source = inspect.getsource(NativeLlamaClient._complete_prompt_standard)
        start = source.index("        if (\n            capture_at is None")
        end = source.index("        self.last_committed_generated_tokens = []", start)
        block = textwrap.dedent(source[start:end])

        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._rolling_analysis_anchor_state = valid_state(MESSAGES)
        client.cancel_event = threading.Event()
        client._session = _Session()
        captured: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            captured.append(list(prompt_tokens))
            return valid_state(prompt_tokens), {}

        exec(block, {
            "capture_at": len(MESSAGES),
            "step_lineage": False,
            "rolling_route_eligible": True,
            "rolling_route_identity": analysis_identity(),
            "processed": len(CONTROL),
            "n_prompt": len(CONTROL),
            "prompt_tokens": CONTROL,
            "self": client,
            "lib": object(),
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
        })
        self.assertEqual(captured, [], "the whole prompt must not replace the boundary checkpoint")
        self.assertEqual(client._rolling_analysis_anchor_state.tokens, MESSAGES)


class _RecordingLib:
    """A lib that remembers WHICH bytes were restored, not just that some were."""

    def __init__(self) -> None:
        self.restored: list[bytes] = []
        self.cleared = 0

    def llama_state_seq_set_data(self, ctx, buffer, size, seq_id):
        self.restored.append(bytes(buffer))
        return size

    def llama_get_memory(self, ctx):
        return object()

    def llama_memory_clear(self, mem, flag):
        self.cleared += 1


class RestoredBytesAreTheAnalysisCheckpointTest(unittest.TestCase):
    """Counting a restore is not enough; the bytes have to be the right ones.

    A restore that wrote the CHAT route's checkpoint while keeping the
    ANALYSIS slot's token bookkeeping would pass every call-count assertion
    and put the wrong KV under the right committed tokens -- the one failure
    the whole design exists to prevent.
    """

    ROUTE_BYTES = b"route-checkpoint-different-bytes"

    def _client(self):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        lib = _RecordingLib()
        client.lib = mock.Mock(lib=lib)
        client._session = _Session()
        client._rolling_route_anchor_state = RollingRouteAnchorState(
            identity=identity(ROLLING_ROUTE_STRATEGY_ID), tokens=list(MESSAGES),
            checkpoint_data=self.ROUTE_BYTES, created_at_monotonic=1.0,
        )
        client._rolling_analysis_anchor_state = valid_state(MESSAGES)
        client._rolling_route_identity_cache = analysis_identity()
        client._prepare_memory_for_prompt = lambda tokens: len(MESSAGES)  # type: ignore[method-assign]
        client._invalidate_committed_sequence = (  # type: ignore[method-assign]
            lambda: client._session.committed_sequence_tokens.clear()
        )
        return client, lib

    def test_the_repair_restores_the_analysis_bytes_not_the_route_bytes(self) -> None:
        client, lib = self._client()
        client._prepare_memory_with_ornith_rolling_route_anchor(REPAIR)
        self.assertEqual(lib.restored, [CHECKPOINT], "exactly the analysis checkpoint's bytes")
        self.assertNotIn(self.ROUTE_BYTES, lib.restored)
        self.assertEqual(client._session.committed_sequence_tokens, MESSAGES)

    def test_bookkeeping_and_bytes_come_from_the_same_slot(self) -> None:
        # Put DIFFERENT tokens in the route slot with the same bytes length so
        # a slot mix-up would be visible on either axis.
        client, lib = self._client()
        client._rolling_route_anchor_state = RollingRouteAnchorState(
            identity=identity(ROLLING_ROUTE_STRATEGY_ID), tokens=[1, 2, 3],
            checkpoint_data=self.ROUTE_BYTES, created_at_monotonic=1.0,
        )
        client._prepare_memory_with_ornith_rolling_route_anchor(REPAIR)
        self.assertEqual(lib.restored, [CHECKPOINT])
        self.assertEqual(client._session.committed_sequence_tokens, MESSAGES)


class CompleteChatWiringTest(unittest.TestCase):
    """The boundary suffix actually travels from the render to the prefill.

    Every boundary test above injects `rolling_boundary_suffix` straight into
    the lifted block. If `complete_chat` stopped passing it, all of those would
    still pass and the feature would be silently off. This drives the real
    `complete_chat` up to `complete_prompt` and asserts what it was handed.
    """

    GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def _client(self, *, bridge: bool = True):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.model_profile = mock.Mock(
            profile_id=ORNITH15_PROFILE_ID, verified=True,
            uses_native_chat_bridge=bridge, template_sha256="tpl",
        )
        client.config = mock.Mock(
            use_mtp_experimental=False, context_tokens=8192, thinking=False,
            progress_step=64, batch_size=256,
        )
        client.paths = mock.Mock(model="/models/ornith.gguf")
        client._model_metadata_identity = {}
        client._reset_generation = 0
        client._session = _Session()
        client._persistent_mtp_runtime = None
        client._media_marker = None
        client._final_prefix_store = lambda: mock.Mock()  # type: ignore[method-assign]
        client._thinking_enabled = lambda thinking: False  # type: ignore[method-assign]
        client._ensure_prompt_cache_mode = lambda mode: None  # type: ignore[method-assign]
        client._qwen_route_anchor_plan_for_prompt = lambda *a, **k: None  # type: ignore[method-assign]
        client._final_prefix_experiment_eligible = lambda flag: False  # type: ignore[method-assign]
        client._ornith_rolling_route_eligible = lambda **k: False  # type: ignore[method-assign]

        gen = self.GEN

        def render(messages, *, tools=None, thinking=None):
            prompt = "PROMPT " + gen
            client._active_profile_render = {"prompt": prompt, "generation_prompt": gen}
            return prompt

        client.apply_chat_template = render  # type: ignore[method-assign]
        seen: list[dict] = []

        def fake_complete_prompt(prompt, **kwargs):
            seen.append(dict(kwargs, prompt=prompt))
            return mock.Mock(prompt_tokens=1, output_tokens=1,
                             reused_prompt_tokens=0, evaluated_prompt_tokens=1)

        client.complete_prompt = fake_complete_prompt  # type: ignore[method-assign]
        return client, seen

    def test_an_analysis_turn_hands_the_boundary_suffix_to_the_prefill(self) -> None:
        client, seen = self._client()
        with mock.patch.object(client_module, "prepare_multimodal_messages", lambda *a, **k: None):
            client.complete_chat(
                [{"role": "user", "content": "x"}], tools=[{"a": 1}],
                analysis_rolling_anchor=True,
            )
        self.assertEqual(len(seen), 1)
        call = seen[0]
        self.assertTrue(call["rolling_route_eligible"])
        self.assertEqual(call["rolling_route_identity"].strategy_id, ROLLING_ANALYSIS_STRATEGY_ID)
        self.assertEqual(call["rolling_boundary_suffix"], self.GEN,
                         "the bridge's generation prompt must reach the prefill")

    def test_without_the_anchor_flag_no_suffix_is_passed(self) -> None:
        client, seen = self._client()
        with mock.patch.object(client_module, "prepare_multimodal_messages", lambda *a, **k: None):
            client.complete_chat([{"role": "user", "content": "x"}], tools=[{"a": 1}])
        self.assertIsNone(seen[0]["rolling_boundary_suffix"])
        self.assertFalse(seen[0]["rolling_route_eligible"])

    def test_a_non_bridge_profile_passes_no_suffix_even_when_eligible(self) -> None:
        client, seen = self._client(bridge=False)
        with mock.patch.object(client_module, "prepare_multimodal_messages", lambda *a, **k: None):
            client.complete_chat(
                [{"role": "user", "content": "x"}], tools=[{"a": 1}],
                analysis_rolling_anchor=True,
            )
        self.assertTrue(seen[0]["rolling_route_eligible"])
        self.assertIsNone(seen[0]["rolling_boundary_suffix"],
                          "a stale bridge render must never supply a boundary")


class RepairRestoreTest(unittest.TestCase):
    """The restore, through the production strategy method."""

    def _restore(self, state, prompt, ident):
        client, lib, calls = strategy_client(analysis_state=state, prepare_result=len(state.tokens))
        client._rolling_route_identity_cache = ident
        reused = client._prepare_memory_with_ornith_rolling_route_anchor(prompt)
        return client, lib, calls, reused

    def test_the_repair_restores_the_control_boundary(self) -> None:
        client, lib, calls, reused = self._restore(valid_state(MESSAGES), REPAIR, analysis_identity())
        self.assertEqual(lib.set_data_calls, 1, "the checkpoint was restored")
        self.assertEqual(client._session.committed_sequence_tokens, MESSAGES)
        self.assertEqual(calls, [REPAIR], "strict append still decides")
        self.assertEqual(reused, len(MESSAGES))

    def test_one_token_mutated_inside_the_boundary_is_refused(self) -> None:
        """Every position, and the LAST one in particular.

        A comparison that stopped one token short would accept a prompt whose
        final committed token differs -- the exact off-by-one that turns a
        strict prefix into a fuzzy one -- so the last position is asserted
        on its own, not left to be covered by an earlier mismatch.
        """
        for position in (0, 2, len(MESSAGES) - 1):
            with self.subTest(position=position):
                mutated = list(REPAIR)
                mutated[position] += 1
                _c, lib, calls, _r = self._restore(
                    valid_state(MESSAGES), mutated, analysis_identity()
                )
                self.assertEqual(lib.set_data_calls, 0, "a mismatch must never restore")
                self.assertEqual(calls, [mutated], "falls to the cold gate")

    def test_same_meaning_different_tokens_is_refused(self) -> None:
        # A retokenization that spells the same messages with different ids.
        respelled = [t + 1000 for t in MESSAGES] + REPAIR[len(MESSAGES):]
        _c, lib, _calls, _r = self._restore(valid_state(MESSAGES), respelled, analysis_identity())
        self.assertEqual(lib.set_data_calls, 0)

    def test_an_incompatible_tool_schema_is_refused(self) -> None:
        _c, lib, _calls, _r = self._restore(
            valid_state(MESSAGES, analysis_identity(tool_schema_hash="plan")),
            REPAIR, analysis_identity(tool_schema_hash="finish"),
        )
        self.assertEqual(lib.set_data_calls, 0, "a PLAN checkpoint must not serve a FINISH")

    def test_an_incompatible_runtime_policy_is_refused(self) -> None:
        _c, lib, _calls, _r = self._restore(
            valid_state(MESSAGES, analysis_identity(runtime_policy_hash="ctx8192")),
            REPAIR, analysis_identity(runtime_policy_hash="ctx4096"),
        )
        self.assertEqual(lib.set_data_calls, 0)

    def test_a_stale_session_anchor_is_refused(self) -> None:
        _c, lib, _calls, _r = self._restore(
            valid_state(MESSAGES, analysis_identity(session_id="old")),
            REPAIR, analysis_identity(session_id="new"),
        )
        self.assertEqual(lib.set_data_calls, 0)

    def test_another_model_template_or_tokenizer_identity_is_refused(self) -> None:
        for field, other in (("model_id", "/models/other.gguf"), ("template_id", "tpl2"),
                             ("native_version", "libllama.so.2"), ("profile_id", "other")):
            with self.subTest(field=field):
                _c, lib, _calls, _r = self._restore(
                    valid_state(MESSAGES), REPAIR, analysis_identity(**{field: other})
                )
                self.assertEqual(lib.set_data_calls, 0)

    def test_a_reset_generation_bump_is_refused(self) -> None:
        _c, lib, _calls, _r = self._restore(
            valid_state(MESSAGES, analysis_identity(reset_generation=0)),
            REPAIR, analysis_identity(reset_generation=1),
        )
        self.assertEqual(lib.set_data_calls, 0)

    def test_a_checkpoint_rolled_forward_still_restores(self) -> None:
        # After the repair was captured at ITS boundary, a further extension
        # restores that longer state.
        longer = REPAIR[: len(REPAIR) - len(GEN)]
        further = longer + [30, 31] + GEN
        client, lib, _calls, reused = self._restore(valid_state(longer), further, analysis_identity())
        self.assertEqual(lib.set_data_calls, 1)
        self.assertEqual(reused, len(longer))
        self.assertEqual(client._session.committed_sequence_tokens, longer)

    def test_a_history_rewrite_that_changes_the_prefix_is_refused(self) -> None:
        # Compaction rewrote a message in the middle: the committed identity
        # the checkpoint describes no longer exists in this prompt.
        rewritten = MESSAGES[:2] + [77, 78] + MESSAGES[4:] + [20, 21, 22] + GEN
        _c, lib, calls, _r = self._restore(valid_state(MESSAGES), rewritten, analysis_identity())
        self.assertEqual(lib.set_data_calls, 0)
        self.assertEqual(calls, [rewritten])

    def test_a_chat_checkpoint_never_serves_a_control_repair(self) -> None:
        # Same tokens, other lineage: the slot is the route's, the analysis
        # slot is empty, and nothing restores.
        client, lib, calls = strategy_client(
            route_state=RollingRouteAnchorState(
                identity=identity(ROLLING_ROUTE_STRATEGY_ID), tokens=list(MESSAGES),
                checkpoint_data=CHECKPOINT, created_at_monotonic=1.0),
            prepare_result=0,
        )
        client._rolling_route_identity_cache = analysis_identity()
        client._prepare_memory_with_ornith_rolling_route_anchor(REPAIR)
        self.assertEqual(lib.set_data_calls, 0)


class IdentityCarriesTheSafetyDimensionsTest(unittest.TestCase):
    """The hashes the restore gate compares must actually depend on their inputs.

    `RepairRestoreTest` refuses a mismatched identity, but it builds identities
    by hand. If the builder stopped folding the tool schema or the context
    configuration into the identity, every phase would hash the same and the
    refusals above would never fire in production. These pin the builder.
    """

    def _client(self, *, context_tokens: int = 8192):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.model_profile = mock.Mock(
            profile_id=ORNITH15_PROFILE_ID, template_sha256="tpl", verified=True
        )
        client.paths = mock.Mock(model="/models/ornith.gguf")
        client._model_metadata_identity = {}
        client._reset_generation = 0
        client.config = mock.Mock(
            use_mtp_experimental=False, context_tokens=context_tokens, thinking=False
        )
        client._session = _Session()
        return client

    def test_a_different_tool_schema_is_a_different_identity(self) -> None:
        client = self._client()
        plan = client._rolling_route_identity(
            tools=[{"function": {"name": "submit_analysis_plan"}}],
            strategy_id=ROLLING_ANALYSIS_STRATEGY_ID,
        )
        finish = client._rolling_route_identity(
            tools=[{"function": {"name": "finish_analysis_question"}}],
            strategy_id=ROLLING_ANALYSIS_STRATEGY_ID,
        )
        self.assertNotEqual(plan.tool_schema_hash, finish.tool_schema_hash)
        self.assertNotEqual(plan, finish)
        # And the same schema is the same identity, which is what lets a
        # repair meet the checkpoint of the call it repairs.
        again = client._rolling_route_identity(
            tools=[{"function": {"name": "submit_analysis_plan"}}],
            strategy_id=ROLLING_ANALYSIS_STRATEGY_ID,
        )
        self.assertEqual(plan, again)

    def test_a_different_context_configuration_is_a_different_identity(self) -> None:
        tools = [{"function": {"name": "submit_analysis_plan"}}]
        wide = self._client(context_tokens=8192)._rolling_route_identity(
            tools=tools, strategy_id=ROLLING_ANALYSIS_STRATEGY_ID
        )
        narrow = self._client(context_tokens=4096)._rolling_route_identity(
            tools=tools, strategy_id=ROLLING_ANALYSIS_STRATEGY_ID
        )
        self.assertNotEqual(wide.runtime_policy_hash, narrow.runtime_policy_hash)
        self.assertNotEqual(wide, narrow)


class LifecycleTest(unittest.TestCase):
    """A checkpoint cannot outlive the session that produced it."""

    def test_invalidation_clears_the_analysis_slot(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._rolling_analysis_anchor_state = valid_state(MESSAGES)
        self.assertTrue(client._rolling_analysis_anchor_state.valid)
        client._invalidate_rolling_route_anchor("session_close")
        state = client._rolling_analysis_anchor_state
        self.assertFalse(state.valid)
        self.assertEqual(state.invalidation_reason, "session_close")
        self.assertIsNone(
            rolling_route_reuse_start(state, REPAIR, analysis_identity()),
            "an invalidated checkpoint restores nothing",
        )

    def test_reset_session_state_destroys_the_analysis_checkpoint(self) -> None:
        """Behavioural, not a source grep: run the real reset and look.

        A grep for the call's name would pass with the call commented out.
        This drives `reset_session_state` itself with the handful of
        collaborators it touches faked, and asserts the slot is gone and the
        generation bumped -- both of which a later restore checks.
        """
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._session = _Session()
        client._session.cancel_requested = False
        client._session.prompt_cache_mode = "tools:thinking=off"
        client._session.continuation_ready = True
        client._session.last_metrics = object()
        client.cancel_event = threading.Event()
        client.lib = mock.Mock(lib=mock.Mock(llama_get_memory=lambda ctx: object(),
                                             llama_memory_clear=lambda mem, flag: None))
        client.config = mock.Mock(use_mtp_experimental=False)
        client._reset_generation = 0
        client._persistent_mtp_runtime = None
        client._rolling_analysis_anchor_state = valid_state(MESSAGES)
        client._invalidate_committed_sequence = lambda: None  # type: ignore[method-assign]
        client._invalidate_final_prefix = lambda reason: None  # type: ignore[method-assign]
        client._invalidate_qwen_route_prefix = lambda reason, profile_id=None: None  # type: ignore[method-assign]
        client._invalidate_qwen36_shell_tool_prefix = lambda reason: None  # type: ignore[method-assign]

        self.assertTrue(client._rolling_analysis_anchor_state.valid)
        client.reset_session_state()

        state = client._rolling_analysis_anchor_state
        self.assertFalse(state.valid, "a reset must destroy the analysis checkpoint")
        self.assertEqual(state.invalidation_reason, "session_reset")
        self.assertEqual(client._reset_generation, 1)
        self.assertIsNone(
            rolling_route_reuse_start(state, REPAIR, analysis_identity(reset_generation=1))
        )


if __name__ == "__main__":
    unittest.main()
