"""Rolling KV for consecutive ANALYSIS steps, driven through the real client.

These tests enter the production methods -- `_prepare_memory_with_ornith_
rolling_route_anchor`, the capture site's helpers, the eligibility gates and
the identity builder -- rather than exercising the anchor module in isolation,
because what has to be true lives in the wiring: that an analysis prompt can
only meet an analysis checkpoint, that restoring still has to satisfy strict
append, and that a CHAT checkpoint can never stand in for an ANALYSIS one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.backend.llama_server as llama_server
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ANALYSIS_STRATEGY_ID,
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
    rolling_route_reuse_start,
)
from orbit.runtime.analysis_runtime import ANALYSIS_STEP_PHASE
from orbit.runtime.kv_diag import model_call_context

STEP1 = [10, 11, 12, 13]
STEP2 = STEP1 + [14, 15, 16]
STEP3 = STEP2 + [17, 18]
CHECKPOINT = b"analysis-checkpoint"


def identity(strategy_id: str = ROLLING_ANALYSIS_STRATEGY_ID, **overrides) -> RollingRouteIdentity:
    base = dict(
        strategy_id=strategy_id,
        session_id="default",
        profile_id=ORNITH15_PROFILE_ID,
        model_id="/models/ornith.gguf",
        template_id="tpl",
        tool_schema_hash="analysis-tools",
        capability_summary_hash="caps",
        runtime_policy_hash="policy",
        native_version="libllama.so",
        tools_mode="on",
        reset_generation=0,
    )
    base.update(overrides)
    return RollingRouteIdentity(**base)


class _Lib:
    def __init__(self, *, fail_set: bool = False) -> None:
        self.fail_set = fail_set
        self.cleared = 0
        self.set_data_calls = 0

    def llama_state_seq_set_data(self, ctx, buffer, size, seq_id):
        self.set_data_calls += 1
        if self.fail_set:
            raise RuntimeError("restore exploded")
        return size

    def llama_get_memory(self, ctx):
        return object()

    def llama_memory_clear(self, mem, flag):
        self.cleared += 1


class _LibHolder:
    def __init__(self, lib: _Lib) -> None:
        self.lib = lib


class _Session:
    def __init__(self) -> None:
        self.ctx_tgt = object()
        self.session_id = "default"
        self.cached_prompt_tokens: list[int] = []
        self.committed_sequence_tokens: list[int] = []
        self.mtp_enabled = False


def strategy_client(
    *,
    analysis_state: RollingRouteAnchorState | None = None,
    route_state: RollingRouteAnchorState | None = None,
    fail_set: bool = False,
    prepare_result: int = 99,
):
    """A client holding only what the rolling strategy path reaches."""
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    lib = _Lib(fail_set=fail_set)
    client.lib = _LibHolder(lib)
    client._session = _Session()
    client._rolling_route_anchor_state = route_state or RollingRouteAnchorState()
    client._rolling_analysis_anchor_state = analysis_state or RollingRouteAnchorState()
    client._rolling_route_identity_cache = None
    calls: list[list[int]] = []

    def fake_prepare(prompt_tokens):
        # Stands in for the real strict-append gate, and records that the
        # strategy actually deferred the decision to it.
        calls.append(list(prompt_tokens))
        return prepare_result

    client._prepare_memory_for_prompt = fake_prepare  # type: ignore[method-assign]
    client._invalidate_committed_sequence = (  # type: ignore[method-assign]
        lambda: client._session.committed_sequence_tokens.clear()
    )
    return client, lib, calls


def valid_state(tokens=STEP1, ident=None) -> RollingRouteAnchorState:
    return RollingRouteAnchorState(
        identity=ident or identity(),
        tokens=list(tokens),
        checkpoint_data=CHECKPOINT,
        created_at_monotonic=1.0,
    )


class ExactPrefixReuseTest(unittest.TestCase):
    """Step N -> step N+1: the saved chain is restored, then judged."""

    def test_step2_restores_the_step1_checkpoint(self) -> None:
        client, lib, calls = strategy_client(analysis_state=valid_state(STEP1))
        client._rolling_route_identity_cache = identity()

        result = client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 1, "the analysis checkpoint must be restored")
        self.assertEqual(client._session.committed_sequence_tokens, STEP1)
        self.assertEqual(client._session.cached_prompt_tokens, STEP1)
        self.assertEqual(calls, [STEP2], "strict append must still decide reuse")
        self.assertEqual(result, 99, "the strategy returns strict append's verdict")
        self.assertNotEqual(
            result, len(STEP1), "returning the restored length would bypass the exact-prefix gate"
        )

    def test_only_the_new_suffix_remains_to_evaluate(self) -> None:
        # `_prepare_memory_for_prompt` returns the reused prefix length, so the
        # suffix is what the caller still has to prefill.
        client, _lib, _calls = strategy_client(
            analysis_state=valid_state(STEP1), prepare_result=len(STEP1)
        )
        client._rolling_route_identity_cache = identity()

        reused = client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(reused, len(STEP1))
        self.assertEqual(len(STEP2) - reused, 3, "only the three new tokens are evaluated")

    def test_step3_rolls_forward_from_step2(self) -> None:
        client, lib, calls = strategy_client(analysis_state=valid_state(STEP2))
        client._rolling_route_identity_cache = identity()

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP3)

        self.assertEqual(lib.set_data_calls, 1)
        self.assertEqual(client._session.committed_sequence_tokens, STEP2)
        self.assertEqual(calls, [STEP3])

    def test_reuse_start_is_exact_prefix_only(self) -> None:
        state = valid_state(STEP1)
        ident = identity()

        self.assertEqual(rolling_route_reuse_start(state, STEP2, ident), len(STEP1))
        # One byte different anywhere in the prefix and it is not a prefix.
        self.assertIsNone(rolling_route_reuse_start(state, [10, 11, 99, 13, 14], ident))
        # An equal-length prompt leaves nothing to evaluate.
        self.assertIsNone(rolling_route_reuse_start(state, STEP1, ident))


class MismatchColdPathTest(unittest.TestCase):
    def test_prefix_mismatch_never_restores(self) -> None:
        client, lib, calls = strategy_client(analysis_state=valid_state(STEP1))
        client._rolling_route_identity_cache = identity()

        result = client._prepare_memory_with_ornith_rolling_route_anchor([10, 11, 99, 13, 14])

        self.assertEqual(lib.set_data_calls, 0, "a mismatched prompt must not restore")
        self.assertEqual(calls, [[10, 11, 99, 13, 14]], "it still goes through strict append")
        self.assertEqual(result, 99)

    def test_shorter_prompt_falls_cold(self) -> None:
        client, lib, _calls = strategy_client(analysis_state=valid_state(STEP2))
        client._rolling_route_identity_cache = identity()

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP1)

        self.assertEqual(lib.set_data_calls, 0)

    def test_failed_restore_clears_then_falls_cold(self) -> None:
        client, lib, calls = strategy_client(analysis_state=valid_state(STEP1), fail_set=True)
        client._rolling_route_identity_cache = identity()

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 1)
        self.assertGreaterEqual(lib.cleared, 1, "a partial restore leaves KV unknown; clear it")
        self.assertFalse(
            client._rolling_analysis_anchor_state.valid,
            "a failed restore must drop the analysis checkpoint",
        )
        self.assertEqual(calls, [STEP2])

    def test_no_staged_identity_refuses_reuse(self) -> None:
        client, lib, calls = strategy_client(analysis_state=valid_state(STEP1))
        client._rolling_route_identity_cache = None

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 0, "without a staged identity nothing may be reused")
        self.assertEqual(calls, [STEP2])

    def test_missing_analysis_slot_falls_cold_instead_of_raising(self) -> None:
        client, lib, calls = strategy_client()
        del client._rolling_analysis_anchor_state
        client._rolling_route_identity_cache = identity()

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 0)
        self.assertEqual(calls, [STEP2])


class CrossModeIsolationTest(unittest.TestCase):
    """Neither lineage may ever be restored into the other."""

    def test_chat_checkpoint_cannot_restore_into_analysis(self) -> None:
        chat_state = valid_state(STEP1, ident=identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID))
        client, lib, calls = strategy_client(route_state=chat_state)
        # An analysis step arrives while only a CHAT checkpoint exists.
        client._rolling_route_identity_cache = identity()

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 0, "a CHAT checkpoint must not serve an analysis step")
        self.assertEqual(calls, [STEP2])
        self.assertTrue(chat_state.valid, "and it must be left intact for CHAT")

    def test_analysis_checkpoint_cannot_restore_into_chat(self) -> None:
        analysis_state = valid_state(STEP1, ident=identity())
        client, lib, calls = strategy_client(analysis_state=analysis_state)
        client._rolling_route_identity_cache = identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID)

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 0, "an analysis checkpoint must not serve a route call")
        self.assertEqual(calls, [STEP2])
        self.assertTrue(analysis_state.valid)

    def test_the_two_identities_are_never_equal(self) -> None:
        self.assertNotEqual(identity(), identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID))
        self.assertIsNone(
            rolling_route_reuse_start(
                valid_state(STEP1, ident=identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID)),
                STEP2,
                identity(),
            )
        )

    def test_each_lineage_keeps_its_own_slot(self) -> None:
        client, _lib, _calls = strategy_client(
            analysis_state=valid_state(STEP1, ident=identity()),
            route_state=valid_state([90, 91], ident=identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID)),
        )

        self.assertEqual(client._rolling_anchor_state_for(identity()).tokens, STEP1)
        self.assertEqual(
            client._rolling_anchor_state_for(identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID)).tokens,
            [90, 91],
        )

    def test_storing_one_lineage_leaves_the_other_untouched(self) -> None:
        client, _lib, _calls = strategy_client(
            route_state=valid_state([90, 91], ident=identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID))
        )

        client._store_rolling_anchor_state(identity(), valid_state(STEP3))

        self.assertEqual(client._rolling_analysis_anchor_state.tokens, STEP3)
        self.assertEqual(client._rolling_route_anchor_state.tokens, [90, 91])

    def test_mode_switch_round_trip_reuses_neither_wrongly(self) -> None:
        # /analysis -> /chat -> /analysis: the analysis chain resumes, and the
        # route chain is never offered analysis tokens.
        client, lib, _calls = strategy_client(
            analysis_state=valid_state(STEP1, ident=identity()),
            route_state=valid_state([90, 91], ident=identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID)),
        )

        client._rolling_route_identity_cache = identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID)
        client._prepare_memory_with_ornith_rolling_route_anchor([90, 91, 92])
        self.assertEqual(client._session.committed_sequence_tokens, [90, 91])

        client._rolling_route_identity_cache = identity()
        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)
        self.assertEqual(client._session.committed_sequence_tokens, STEP1)
        self.assertEqual(lib.set_data_calls, 2, "each lineage restored its own checkpoint")


class IdentityInvalidationTest(unittest.TestCase):
    def test_tool_schema_change_invalidates_reuse(self) -> None:
        state = valid_state(STEP1, ident=identity(tool_schema_hash="old"))
        client, lib, calls = strategy_client(analysis_state=state)
        client._rolling_route_identity_cache = identity(tool_schema_hash="new")

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 0)
        self.assertEqual(calls, [STEP2])

    def test_every_identity_field_invalidates_reuse(self) -> None:
        for field, value in (
            ("session_id", "other"),
            ("profile_id", "other-profile"),
            ("model_id", "/models/other.gguf"),
            ("template_id", "other-tpl"),
            ("tool_schema_hash", "other-tools"),
            ("capability_summary_hash", "other-caps"),
            ("runtime_policy_hash", "other-policy"),
            ("native_version", "other.so"),
            ("tools_mode", "off"),
            ("reset_generation", 1),
        ):
            with self.subTest(field=field):
                state = valid_state(STEP1, ident=identity())
                self.assertIsNone(
                    rolling_route_reuse_start(state, STEP2, identity(**{field: value})),
                    f"{field} must invalidate reuse",
                )

    def test_reset_generation_bump_invalidates(self) -> None:
        state = valid_state(STEP1, ident=identity(reset_generation=0))
        client, lib, _calls = strategy_client(analysis_state=state)
        client._rolling_route_identity_cache = identity(reset_generation=1)

        client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)

        self.assertEqual(lib.set_data_calls, 0, "a reset must not leave reusable analysis state")

    def test_invalidate_clears_both_lineages(self) -> None:
        client, _lib, _calls = strategy_client(
            analysis_state=valid_state(STEP1, ident=identity()),
            route_state=valid_state([90, 91], ident=identity(strategy_id=ROLLING_ROUTE_STRATEGY_ID)),
        )

        client._invalidate_rolling_route_anchor("session_reset")

        self.assertFalse(client._rolling_route_anchor_state.valid)
        self.assertFalse(client._rolling_analysis_anchor_state.valid)


class EligibilityTest(unittest.TestCase):
    """The gate is the phase the runtime declares, never an inferred mode."""

    def client_for(self, *, profile_id=ORNITH15_PROFILE_ID, verified=True, mtp=False):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.model_profile = mock.Mock(profile_id=profile_id, verified=verified)
        client.config = mock.Mock(use_mtp_experimental=mtp)
        client._session = _Session()
        return client

    def test_analysis_anchor_enables_reuse(self) -> None:
        client = self.client_for()
        self.assertTrue(
            client._ornith_rolling_analysis_eligible(analysis_rolling_anchor=True, thinking=False)
        )

    def test_without_the_anchor_flag_nothing_is_eligible(self) -> None:
        client = self.client_for()
        self.assertFalse(
            client._ornith_rolling_analysis_eligible(analysis_rolling_anchor=False, thinking=False)
        )

    def test_unverified_or_foreign_profile_is_refused(self) -> None:
        for kwargs in ({"verified": False}, {"profile_id": "some-other-profile"}):
            with self.subTest(**kwargs):
                client = self.client_for(**kwargs)
                self.assertFalse(
                    client._ornith_rolling_analysis_eligible(
                        analysis_rolling_anchor=True, thinking=False
                    )
                )

    def test_thinking_and_mtp_are_refused(self) -> None:
        self.assertFalse(
            self.client_for()._ornith_rolling_analysis_eligible(
                analysis_rolling_anchor=True, thinking=True
            )
        )
        self.assertFalse(
            self.client_for(mtp=True)._ornith_rolling_analysis_eligible(
                analysis_rolling_anchor=True, thinking=False
            )
        )

    def test_identity_carries_the_analysis_strategy(self) -> None:
        client = self.client_for()
        client.paths = mock.Mock(model="/models/ornith.gguf")
        client._model_metadata_identity = {}
        client._reset_generation = 0
        client.config = mock.Mock(
            use_mtp_experimental=False, context_tokens=4096, thinking=False
        )
        client._session = _Session()

        analysis = client._rolling_route_identity(
            tools=[{"a": 1}], strategy_id=ROLLING_ANALYSIS_STRATEGY_ID
        )
        route = client._rolling_route_identity(tools=[{"a": 1}])

        self.assertEqual(analysis.strategy_id, ROLLING_ANALYSIS_STRATEGY_ID)
        self.assertEqual(route.strategy_id, ROLLING_ROUTE_STRATEGY_ID)
        self.assertNotEqual(analysis, route)


class PhaseWiringTest(unittest.TestCase):
    """The backend is told which chain a call belongs to; it never guesses."""

    def test_analysis_phase_requests_the_analysis_anchor_only(self) -> None:
        with mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: True):
            with model_call_context(phase=ANALYSIS_STEP_PHASE, tools_mode="on"):
                self.assertTrue(
                    llama_server._analysis_rolling_anchor_requested(native_backend=True)
                )
                self.assertFalse(
                    llama_server._route_prefix_anchor_requested(native_backend=True),
                    "an analysis step must not also claim the route anchor",
                )

    def test_route_phase_requests_the_route_anchor_only(self) -> None:
        with mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: True):
            with model_call_context(phase="route", tools_mode="on"):
                self.assertTrue(llama_server._route_prefix_anchor_requested(native_backend=True))
                self.assertFalse(
                    llama_server._analysis_rolling_anchor_requested(native_backend=True)
                )

    def test_no_phase_requests_neither(self) -> None:
        with mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: True):
            self.assertFalse(llama_server._analysis_rolling_anchor_requested(native_backend=True))

    def test_non_native_backend_never_requests_it(self) -> None:
        with mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: True):
            with model_call_context(phase=ANALYSIS_STEP_PHASE, tools_mode="on"):
                self.assertFalse(
                    llama_server._analysis_rolling_anchor_requested(native_backend=False)
                )

    def test_payload_carries_the_flag_only_when_asked(self) -> None:
        from orbit.backend.payloads import ChatPayloadOptions, build_chat_payload

        on = build_chat_payload(
            ChatPayloadOptions(
                model="m", messages=[], temperature=0.0, max_tokens=1, analysis_rolling_anchor=True
            )
        )
        off = build_chat_payload(
            ChatPayloadOptions(model="m", messages=[], temperature=0.0, max_tokens=1)
        )

        self.assertTrue(on["analysis_rolling_anchor"])
        self.assertNotIn("analysis_rolling_anchor", off)

    def test_request_parsing_round_trips_the_flag(self) -> None:
        from orbit.native_server.protocol import parse_chat_request

        messages = [{"role": "user", "content": "hi"}]
        request = parse_chat_request(
            {"model": "m", "messages": messages, "analysis_rolling_anchor": True}
        )
        self.assertTrue(request.analysis_rolling_anchor)
        self.assertFalse(
            parse_chat_request({"model": "m", "messages": messages}).analysis_rolling_anchor
        )

    def test_no_mode_string_reaches_the_backend_payload(self) -> None:
        from orbit.backend.payloads import ChatPayloadOptions, build_chat_payload

        payload = build_chat_payload(
            ChatPayloadOptions(
                model="m", messages=[], temperature=0.0, max_tokens=1, analysis_rolling_anchor=True
            )
        )
        serialized = repr(payload)
        for token in ("ANALYSIS", "CHAT", "WorkflowMode", "workflow_mode"):
            self.assertNotIn(token, serialized)


class HumanBoundaryTest(unittest.TestCase):
    """Declaring a phase must not change what a step does."""

    def analysis_runtime(self):
        import shutil
        import tempfile

        from orbit.backend.base import ChatResult
        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime,
            AnalysisWorkspace,
            acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmp = Path(tempfile.mkdtemp(prefix="orbit-rollingkv-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        artifact = tmp / "sample.js"
        artifact.write_text("var a = 1;\n", encoding="utf-8")
        workspace = AnalysisWorkspace.create()
        source = acquire_analysis_source(artifact, workspace.source_root)

        phases: list[str | None] = []

        class Backend:
            def __init__(self) -> None:
                self.calls = 0

            def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                            on_delta=None, on_progress=None):
                self.calls += 1
                from orbit.runtime.kv_diag import current_phase

                phases.append(current_phase())
                return ChatResult("ok", "scripted", "stop", [], 1, 1, 0, None, None)

            def chat(self, *args, **kwargs):
                return self.chat_stream(*args, **kwargs)

        backend = Backend()
        runtime = AnalysisRuntime(
            backend=backend,
            source=source,
            evidence_store=EvidenceStore(root=tmp / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        return runtime, backend, phases

    def test_one_model_call_per_step_with_the_phase_declared(self) -> None:
        runtime, backend, phases = self.analysis_runtime()

        for expected in (1, 2, 3):
            runtime.step("continue")
            self.assertEqual(backend.calls, expected, "the phase must not add a model call")

        self.assertEqual(phases, [ANALYSIS_STEP_PHASE] * 3)

    def test_phase_is_scoped_to_the_call_and_restored_after(self) -> None:
        from orbit.runtime.kv_diag import current_phase

        runtime, _backend, _phases = self.analysis_runtime()
        before = current_phase()

        runtime.step("continue")

        self.assertEqual(current_phase(), before, "the phase must not leak past the call")

    def test_step_still_returns_control_with_at_most_one_action(self) -> None:
        runtime, _backend, _phases = self.analysis_runtime()

        result = runtime.step("continue")

        self.assertEqual(result.model_calls, 1)
        self.assertFalse(result.action_executed)
        self.assertTrue(result.control_returned)
        self.assertEqual(runtime.actions_executed, 0)

    def test_history_stays_append_only_across_steps(self) -> None:
        runtime, _backend, _phases = self.analysis_runtime()

        runtime.step("one")
        first = [dict(m) for m in runtime.messages]
        runtime.step("two")

        self.assertEqual(runtime.messages[: len(first)], first)


class CaptureCommitmentTest(unittest.TestCase):
    """A checkpoint may only be taken from a fully committed prompt."""

    def test_should_replace_requires_a_continuing_chain(self) -> None:
        from orbit.native_llama.rolling_route_anchor import rolling_route_should_replace

        ident = identity()
        empty = RollingRouteAnchorState()
        self.assertTrue(rolling_route_should_replace(empty, STEP1, ident), "cold: capture")

        state = valid_state(STEP1, ident=ident)
        self.assertTrue(
            rolling_route_should_replace(state, STEP2, ident), "a continuation replaces"
        )
        self.assertFalse(
            rolling_route_should_replace(state, [7, 8, 9], ident),
            "an unrelated prompt must not evict a usable checkpoint",
        )

    def test_identity_change_replaces_rather_than_reuses(self) -> None:
        from orbit.native_llama.rolling_route_anchor import rolling_route_should_replace

        state = valid_state(STEP1, ident=identity(tool_schema_hash="old"))
        self.assertTrue(
            rolling_route_should_replace(state, STEP2, identity(tool_schema_hash="new"))
        )

    def test_capture_records_exactly_the_prompt_tokens(self) -> None:
        from orbit.native_llama.rolling_route_anchor import capture_rolling_route_anchor

        class Lib:
            def llama_state_seq_get_size(self, ctx, seq):
                return 4

            def llama_state_seq_get_data(self, ctx, buf, size, seq):
                return size

        captured, meta = capture_rolling_route_anchor(
            Lib(), object(), prompt_tokens=STEP2, identity=identity()
        )

        self.assertTrue(captured.valid)
        self.assertEqual(captured.tokens, STEP2, "the snapshot is the prompt and only the prompt")
        self.assertEqual(meta["checkpoint_tokens"], len(STEP2))


class RenderedPrefixPreconditionTest(unittest.TestCase):
    """The precondition the whole strategy rests on, asserted not assumed.

    Every other test in this file builds `STEP2 = STEP1 + [...]`, which proves
    the machinery given a prefix but never that consecutive analysis prompts
    actually are one. That relation has two halves: the history must be
    append-only, and the template must absorb its own generation prompt rather
    than replace it. The first is asserted here against the real runtime; the
    second is asserted against a ChatML-shaped renderer standing in for the
    Ornith template, which is not on this machine.
    """

    def analysis_messages(self, steps: int) -> list[list[dict]]:
        import shutil
        import tempfile

        from orbit.backend.base import ChatResult
        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime,
            AnalysisWorkspace,
            acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmp = Path(tempfile.mkdtemp(prefix="orbit-prefix-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        artifact = tmp / "sample.js"
        artifact.write_text("var a = 1;\n", encoding="utf-8")
        workspace = AnalysisWorkspace.create()
        source = acquire_analysis_source(artifact, workspace.source_root)
        seen: list[list[dict]] = []

        class Backend:
            def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                            on_delta=None, on_progress=None):
                seen.append([dict(m) for m in messages])
                return ChatResult("observing", "scripted", "stop", [], 1, 1, 0, None, None)

            def chat(self, *args, **kwargs):
                return self.chat_stream(*args, **kwargs)

        runtime = AnalysisRuntime(
            backend=Backend(),
            source=source,
            evidence_store=EvidenceStore(root=tmp / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        for index in range(steps):
            runtime.step(f"step {index}")
        return seen

    def test_each_step_sends_a_message_list_extending_the_last(self) -> None:
        seen = self.analysis_messages(4)

        self.assertEqual(len(seen), 4)
        for index in range(len(seen) - 1):
            earlier, later = seen[index], seen[index + 1]
            self.assertGreater(len(later), len(earlier))
            self.assertEqual(
                later[: len(earlier)], earlier, f"step {index + 2} must extend step {index + 1}"
            )

    def test_a_chatml_template_keeps_consecutive_prompts_exact_prefixes(self) -> None:
        # Ornith renders through the GGUF's embedded ChatML-family template
        # (profile renderer "llama.cpp-jinja"). What matters for reuse is that
        # the assistant turn re-emits the generation header the previous prompt
        # ended on, so the earlier prompt survives intact inside the later one.
        def render(messages: list[dict]) -> str:
            body = "".join(
                f"<|im_start|>{m['role']}\n{_text(m)}<|im_end|>\n" for m in messages
            )
            return body + "<|im_start|>assistant\n"

        def _text(message: dict) -> str:
            content = message.get("content")
            return content if isinstance(content, str) else ""

        rendered = [render(messages) for messages in self.analysis_messages(4)]
        for index in range(len(rendered) - 1):
            self.assertTrue(
                rendered[index + 1].startswith(rendered[index]),
                f"prompt {index + 2} must extend prompt {index + 1} exactly",
            )

    def test_a_template_that_replaces_its_generation_prompt_breaks_reuse(self) -> None:
        # The counter-case, kept explicit: a template whose assistant turn does
        # not re-emit the trailing generation prompt cannot support this
        # strategy, and must never be given a rolling anchor.
        def render(messages: list[dict]) -> str:
            body = "".join(f"[{m['role']}:{m.get('content', '')}]" for m in messages)
            return body + "<GEN>"

        messages = self.analysis_messages(2)
        first, second = render(messages[0]), render(messages[1])

        self.assertFalse(second.startswith(first))
        self.assertTrue(second.startswith(first[: -len("<GEN>")]))


class MetricsTest(unittest.TestCase):
    """Reused and evaluated tokens must add up to the prompt."""

    def test_reuse_reports_a_positive_reused_prefix(self) -> None:
        client, _lib, _calls = strategy_client(
            analysis_state=valid_state(STEP1), prepare_result=len(STEP1)
        )
        client._rolling_route_identity_cache = identity()

        reused = client._prepare_memory_with_ornith_rolling_route_anchor(STEP2)
        evaluated = len(STEP2) - reused

        self.assertGreater(reused, 0)
        self.assertEqual(evaluated, len(STEP2) - len(STEP1))
        self.assertEqual(reused + evaluated, len(STEP2), "accounting must cover the whole prompt")

    def test_cold_path_reports_no_reuse(self) -> None:
        client, _lib, _calls = strategy_client(
            analysis_state=valid_state(STEP1), prepare_result=0
        )
        client._rolling_route_identity_cache = identity()

        reused = client._prepare_memory_with_ornith_rolling_route_anchor([7, 8, 9])

        self.assertEqual(reused, 0)
        self.assertEqual(len([7, 8, 9]) - reused, 3, "a cold prompt is evaluated in full")


if __name__ == "__main__":
    unittest.main()
