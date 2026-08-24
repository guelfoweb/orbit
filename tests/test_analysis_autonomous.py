"""Bounded autonomous continuation: Orbit continues only while state changes.

The property under test is causal. A step that adds verifiable state -- a new
evidence content hash, a new or changed artifact digest -- earns another model
call; a step that adds nothing does not. These tests drive the real production
entry point (`AnalysisRuntime.run_autonomous`) through the real sandbox and the
real EvidenceStore, with only the model scripted, because the mechanism being
tested is exactly the decision to invoke the model again.

The scripted backend explodes when invoked more times than scripted, so a loop
that continued past its bound fails by raising rather than by returning a value
a lenient assertion might accept.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime.analysis_progress import (
    COMPLETE,
    ERROR,
    NEW_CONTENT,
    NO_PROGRESS,
    ProgressLedger,
)
from orbit.runtime.analysis_runtime import (
    ANALYSIS_AUTONOMY_ENV,
    ANALYSIS_TOOL_NAME,
    AUTONOMOUS_CONTINUATION_MESSAGE,
    MAX_AUTONOMOUS_ACTIONS,
    MAX_AUTONOMOUS_MODEL_CALLS,
    MAX_CONSECUTIVE_ERRORS,
    MAX_CONSECUTIVE_NO_PROGRESS,
    STOP_BACKEND_ERROR,
    STOP_CANCELLED,
    STOP_COMPLETE,
    STOP_ERROR,
    STOP_MAX_ACTIONS,
    STOP_MAX_MODEL_CALLS,
    STOP_NO_PROGRESS,
    AnalysisRuntime,
    acquire_analysis_source,
    analysis_autonomy_enabled,
)
from orbit.runtime.evidence import EvidenceStore

from tests.test_analysis_runtime import (
    ExhaustedBackend,
    ScriptedBackend,
    prose_response,
    tool_response,
)

FIXTURE = "alpha\nbeta\ngamma\n"


def emit(text: str) -> str:
    """A program whose output is fixed, so its evidence hash is fixed."""
    return f"print({text!r}, end='')"


def write_artifact(name: str, body: str) -> str:
    """A program that materialises a durable artifact under the work mount."""
    return (
        "import pathlib\n"
        f"p = pathlib.Path('/workspace/work/{name}')\n"
        f"p.write_text({body!r})\n"
        f"print('wrote {name}', end='')"
    )


def _is_docstring(node) -> bool:
    import ast

    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )


def _without_docstrings(source: str) -> str:
    """Drop every docstring and comment, leaving only executable text."""
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and _is_docstring(body[0]):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


class AutonomousTestBase(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._dir = tempfile.TemporaryDirectory(prefix="orbit-autonomous-")
        self.addCleanup(self._dir.cleanup)
        self.tmp = Path(self._dir.name)
        original = self.tmp / "artifact.txt"
        original.write_text(FIXTURE, encoding="utf-8")
        self.source = acquire_analysis_source(original, self.tmp / "owned")
        self.store = EvidenceStore(root=self.tmp / "evidence")

    def runtime(self, backend) -> AnalysisRuntime:
        built = AnalysisRuntime(
            backend=backend, source=self.source, evidence_store=self.store
        )
        self.addCleanup(built.close)
        return built


class ContinuationIsDrivenByNewContentTests(AutonomousTestBase):
    def test_new_content_causes_the_next_model_call(self) -> None:
        backend = ScriptedBackend(
            tool_response(emit("first")),
            prose_response("nothing further"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        # Two calls: the action earned the second one.
        self.assertEqual(backend.calls, 2)
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(
            [r.classification for r in run.progress], [NEW_CONTENT, COMPLETE]
        )
        self.assertEqual(run.stop_reason, STOP_COMPLETE)

    def test_the_continuation_message_is_the_generic_one(self) -> None:
        """Orbit must not steer. The follow-up carries no direction."""
        backend = ScriptedBackend(
            tool_response(emit("first")), prose_response("done")
        )
        self.runtime(backend).run_autonomous("inspect it")

        second_prompt = backend.seen_messages[1]
        analyst_lines = [m for m in second_prompt if m.get("role") == "user"]
        self.assertEqual(analyst_lines[-1]["content"], AUTONOMOUS_CONTINUATION_MESSAGE)
        self.assertEqual(AUTONOMOUS_CONTINUATION_MESSAGE, "continue")

    def test_several_new_content_steps_continue(self) -> None:
        backend = ScriptedBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            tool_response(emit("three")),
            prose_response("that is all"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(run.actions_executed, 3)
        self.assertEqual(run.model_calls, 4)
        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NEW_CONTENT, NEW_CONTENT, COMPLETE],
        )

    def test_prose_with_no_action_stops_immediately(self) -> None:
        backend = ScriptedBackend(prose_response("I have nothing to run"))
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(backend.calls, 1)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_COMPLETE)
        self.assertEqual([r.classification for r in run.progress], [COMPLETE])

    def test_model_prose_is_never_progress(self) -> None:
        """A confident claim in prose must not buy another step."""
        backend = ScriptedBackend(
            prose_response("I have decoded stage two and will continue shortly")
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(backend.calls, 1)
        self.assertNotIn(NEW_CONTENT, [r.classification for r in run.progress])

    def test_a_changed_artifact_sha_counts_as_new_content(self) -> None:
        """Same handle, different bytes: a real state transition."""
        backend = ScriptedBackend(
            tool_response(write_artifact("stage.bin", "aaa")),
            tool_response(write_artifact("stage.bin", "bbb")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("transform it")

        self.assertEqual(run.actions_executed, 2)
        second = run.progress[1]
        self.assertEqual(second.classification, NEW_CONTENT)
        self.assertEqual(second.changed_artifacts, ("/workspace/work/stage.bin",))

    def test_evidence_persists_across_autonomous_steps(self) -> None:
        backend = ScriptedBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        recorded = [s.evidence for s in run.steps if s.evidence is not None]
        self.assertEqual(len(recorded), 2)
        for record in recorded:
            # Still durable and re-attestable after the run ended.
            self.assertTrue(self.store.reattest_exact(record.evidence_id))
        self.assertEqual(len({r.raw_sha256 for r in recorded}), 2)

    def test_one_action_maximum_per_model_call(self) -> None:
        """Two calls in one response is rejected, exactly as in one-step mode."""
        backend = ScriptedBackend(
            tool_response(emit("x"), count=2),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        first = run.steps[0]
        self.assertTrue(first.action_attempted)
        self.assertFalse(first.action_executed)
        self.assertIsNotNone(first.rejection)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.progress[0].classification, ERROR)


class DuplicateWorkIsNotProgressTests(AutonomousTestBase):
    def test_repeated_identical_action_is_no_progress(self) -> None:
        code = emit("same")
        backend = ScriptedBackend(
            tool_response(code), tool_response(code), tool_response(code)
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS],
        )
        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS), run.stop_reason)
        # It stopped rather than exhausting the script.
        self.assertEqual(backend.calls, 3)
        self.assertTrue(run.progress[-1].repeated_action)

    def test_different_code_with_identical_output_is_not_progress(self) -> None:
        """Novelty is judged on evidence, not on the program that produced it."""
        backend = ScriptedBackend(
            tool_response(emit("same")),
            tool_response("x = 1\n" + emit("same")),
            tool_response("y = 2\n" + emit("same")),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS],
        )
        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS), run.stop_reason)

    def test_rewriting_an_artifact_with_identical_bytes_is_not_progress(self) -> None:
        """Re-writing the same bytes yields no artifact delta, so it stagnates.

        The first two steps differ in raw output -- the sandbox reports a new
        artifact only on the step that changed it -- so stagnation is judged
        from the third step, once the observation itself repeats. What is being
        asserted is that an unchanged digest buys nothing.
        """
        code = write_artifact("same.bin", "identical")
        backend = ScriptedBackend(*[tool_response(code)] * 5)
        run = self.runtime(backend).run_autonomous("transform it")

        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS), run.stop_reason)
        # No step after the first ever reports a changed digest for the handle.
        for record in run.progress[1:]:
            self.assertEqual(record.changed_artifacts, ())
        self.assertEqual(
            [r.classification for r in run.progress[-2:]], [NO_PROGRESS, NO_PROGRESS]
        )

    def test_stagnation_bound_stops_the_run(self) -> None:
        code = emit("same")
        backend = ScriptedBackend(*[tool_response(code)] * 6)
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS), run.stop_reason)
        self.assertEqual(backend.calls, 1 + MAX_CONSECUTIVE_NO_PROGRESS)

    def test_progress_resets_the_stagnation_counter(self) -> None:
        backend = ScriptedBackend(
            tool_response(emit("a")),
            tool_response(emit("a")),
            tool_response(emit("b")),
            tool_response(emit("b")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NEW_CONTENT, NO_PROGRESS, COMPLETE],
        )
        self.assertEqual(run.stop_reason, STOP_COMPLETE)


class ErrorsAreBoundedTests(AutonomousTestBase):
    def test_consecutive_errors_stop_the_run(self) -> None:
        backend = ScriptedBackend(*[tool_response(emit("x"), count=2)] * 5)
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertTrue(run.stop_reason.startswith(STOP_ERROR), run.stop_reason)
        self.assertEqual(backend.calls, MAX_CONSECUTIVE_ERRORS)
        self.assertEqual(run.actions_executed, 0)

    def test_malformed_tool_output_does_not_auto_repair(self) -> None:
        """A rejected call is never re-run; the loop only ever takes new steps."""
        broken = ChatResult(
            content="", model="m", finish_reason="stop",
            tool_calls=[{"id": "c", "type": "function",
                         "function": {"name": ANALYSIS_TOOL_NAME,
                                      "arguments": "{\"code\": \"print(1)"}}],
            prompt_tokens=1, completion_tokens=1, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )
        backend = ScriptedBackend(broken, broken, broken)
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertTrue(run.stop_reason.startswith(STOP_ERROR), run.stop_reason)
        self.assertEqual(backend.calls, MAX_CONSECUTIVE_ERRORS)
        self.assertEqual(run.actions_executed, 0)
        for step in run.steps:
            self.assertIsNotNone(step.rejection)

    def test_progress_resets_the_error_counter(self) -> None:
        backend = ScriptedBackend(
            tool_response(emit("x"), count=2),
            tool_response(emit("real")),
            tool_response(emit("x"), count=2),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(
            [r.classification for r in run.progress],
            [ERROR, NEW_CONTENT, ERROR, COMPLETE],
        )
        self.assertEqual(run.stop_reason, STOP_COMPLETE)


class BoundsAreEnforcedTests(AutonomousTestBase):
    def test_max_actions_stops_the_run(self) -> None:
        backend = ScriptedBackend(
            *[tool_response(emit(f"v{i}")) for i in range(MAX_AUTONOMOUS_ACTIONS + 3)]
        )
        run = self.runtime(backend).run_autonomous(
            "inspect it", max_model_calls=MAX_AUTONOMOUS_ACTIONS + 3
        )

        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)
        self.assertEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)
        self.assertEqual(backend.calls, MAX_AUTONOMOUS_ACTIONS)

    def test_max_model_calls_stops_the_run(self) -> None:
        backend = ScriptedBackend(
            *[tool_response(emit(f"v{i}")) for i in range(12)]
        )
        run = self.runtime(backend).run_autonomous(
            "inspect it", max_model_calls=3, max_actions=99
        )

        self.assertEqual(run.stop_reason, STOP_MAX_MODEL_CALLS)
        self.assertEqual(run.model_calls, 3)
        self.assertEqual(backend.calls, 3)

    def test_bounds_are_small_and_explicit(self) -> None:
        self.assertEqual(MAX_AUTONOMOUS_ACTIONS, 8)
        self.assertEqual(MAX_AUTONOMOUS_MODEL_CALLS, 10)
        self.assertEqual(MAX_CONSECUTIVE_NO_PROGRESS, 2)
        self.assertEqual(MAX_CONSECUTIVE_ERRORS, 2)
        # The call bound must leave room for calls that execute nothing.
        self.assertGreater(MAX_AUTONOMOUS_MODEL_CALLS, MAX_AUTONOMOUS_ACTIONS)

    def test_alternating_error_and_stagnation_is_still_bounded(self) -> None:
        """Alternation defeats both consecutive counters; the call bound holds.

        A run that alternates ERROR and NO_PROGRESS resets each consecutive
        counter before it can trip, so neither stagnation nor error alone ends
        it. That is by design -- a counter that survived an intervening failure
        would abort runs that were merely recovering -- and it is exactly why
        the total model-call bound exists as the outer backstop. Worst case is
        therefore MAX_AUTONOMOUS_MODEL_CALLS, never unbounded.
        """
        same = emit("same")
        script = []
        for _ in range(20):
            script.append(tool_response(same, count=2))  # ERROR
            script.append(tool_response(same))           # NO_PROGRESS
        run = self.runtime(ScriptedBackend(*script)).run_autonomous("inspect it")

        classifications = [r.classification for r in run.progress]
        self.assertIn(ERROR, classifications)
        self.assertIn(NO_PROGRESS, classifications)
        self.assertEqual(run.stop_reason, STOP_MAX_MODEL_CALLS)
        self.assertEqual(run.model_calls, MAX_AUTONOMOUS_MODEL_CALLS)

    def test_no_second_finalization_or_classifier_call(self) -> None:
        """Every model call is an ordinary step with the ordinary tool surface."""
        backend = ScriptedBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(backend.calls, run.model_calls)
        for offered in backend.seen_tools:
            self.assertEqual(offered, [ANALYSIS_TOOL_NAME])


class HumanControlTests(AutonomousTestBase):
    def test_cancellation_stops_the_loop_and_keeps_evidence(self) -> None:
        class CancellingBackend(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                if self.calls >= 1:
                    raise KeyboardInterrupt
                return super().chat_stream(messages, **kwargs)

        backend = CancellingBackend(tool_response(emit("one")))
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertTrue(run.cancelled)
        self.assertEqual(run.stop_reason, STOP_CANCELLED)
        self.assertEqual(run.actions_executed, 1)
        self.assertIsNotNone(run.steps[0].evidence)
        self.assertTrue(self.store.reattest_exact(run.steps[0].evidence.evidence_id))

    def test_cancelling_a_multi_step_run_keeps_the_completed_steps(self) -> None:
        """Cancellation must not discard history from steps that succeeded.

        The terminal rewinds to a pre-run checkpoint only when a run produced
        nothing at all. A multi-step run that is interrupted has already
        committed history and evidence for its completed steps, and truncating
        back past those would destroy the append-only record the rolling KV
        lineage and evidence provenance both depend on.
        """

        class CancelOnThird(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                if self.calls >= 2:
                    raise KeyboardInterrupt
                return super().chat_stream(messages, **kwargs)

        backend = CancelOnThird(
            tool_response(emit("one")), tool_response(emit("two"))
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("inspect it")

        self.assertTrue(run.cancelled)
        self.assertEqual(run.actions_executed, 2)
        # Not None, so the terminal's rewind path is not taken.
        self.assertIsNotNone(run.last_step)
        self.assertGreater(len(runtime.messages), 2)
        for step in run.steps:
            self.assertIsNotNone(step.evidence)
            self.assertTrue(self.store.reattest_exact(step.evidence.evidence_id))

    def test_a_run_cancelled_before_any_step_reports_nothing(self) -> None:
        """The only case where the terminal rewinds: no step ever completed."""

        class CancelImmediately(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                raise KeyboardInterrupt

        run = self.runtime(CancelImmediately()).run_autonomous("inspect it")

        self.assertTrue(run.cancelled)
        self.assertEqual(run.steps, ())
        self.assertIsNone(run.last_step)
        self.assertEqual(run.actions_executed, 0)

    def test_the_analyst_can_steer_after_an_autonomous_stop(self) -> None:
        """The same session and history continue; nothing is rebuilt."""
        backend = ScriptedBackend(
            tool_response(emit("one")),
            prose_response("stopping"),
            tool_response(emit("steered")),
            prose_response("done"),
        )
        runtime = self.runtime(backend)
        first = runtime.run_autonomous("inspect it")
        history_after_first = len(runtime.messages)

        second = runtime.run_autonomous("now look at the header instead")

        self.assertEqual(first.stop_reason, STOP_COMPLETE)
        self.assertEqual(second.actions_executed, 1)
        # Append-only: the second run extended the same history.
        self.assertGreater(len(runtime.messages), history_after_first)
        steering = [m for m in backend.seen_messages[2] if m.get("role") == "user"]
        self.assertEqual(steering[-1]["content"], "now look at the header instead")


class RecoverableBackendFailureTests(AutonomousTestBase):
    """A mid-run backend failure ends the run without undoing it."""

    def _failing(self, exc: Exception):
        class FailAfterTwo(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                if self.calls >= 2:
                    raise exc
                return super().chat_stream(messages, **kwargs)

        return FailAfterTwo(tool_response(emit("one")), tool_response(emit("two")))

    def test_a_backend_error_does_not_escape_the_run(self) -> None:
        """It must not propagate to a caller holding a pre-run checkpoint.

        A caller that rewinds to that checkpoint would delete the history and
        provenance of every step that already succeeded, orphaning their
        evidence on disk and re-issuing turn ids that are already in use.
        """
        from orbit.backend.llama_server import LlamaServerError

        runtime = self.runtime(self._failing(LlamaServerError("backend died")))
        run = runtime.run_autonomous("inspect it")  # must not raise

        self.assertTrue(run.stop_reason.startswith(STOP_BACKEND_ERROR))
        self.assertIn("LlamaServerError", run.stop_reason)
        self.assertEqual(run.actions_executed, 2)
        self.assertIsNotNone(run.last_step)

    def test_completed_steps_survive_a_backend_error(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        runtime = self.runtime(self._failing(LlamaServerError("backend died")))
        run = runtime.run_autonomous("inspect it")

        for step in run.steps:
            self.assertTrue(self.store.reattest_exact(step.evidence.evidence_id))
        turn_ids = [
            step.evidence.user_turn_id for step in run.steps if step.evidence
        ]
        self.assertEqual(len(set(turn_ids)), len(turn_ids), "turn ids must be unique")

    def test_a_context_admission_error_is_handled_the_same_way(self) -> None:
        from orbit.runtime.context_manager import ContextAdmissionError

        run = self.runtime(self._failing(ContextAdmissionError("too big"))).run_autonomous(
            "inspect it"
        )

        self.assertTrue(run.stop_reason.startswith(STOP_BACKEND_ERROR))
        self.assertEqual(run.actions_executed, 2)

    def test_no_unanswered_analyst_turn_is_left_behind(self) -> None:
        """`step()` appends the analyst line before calling the model.

        A step that never returned leaves that line unanswered at the end of an
        append-only history, and the next analyst message would then sit
        directly after it -- two consecutive user turns, permanently.
        """
        from orbit.backend.llama_server import LlamaServerError

        for exc in (LlamaServerError("died"), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__):
                runtime = self.runtime(self._failing(exc))
                runtime.run_autonomous("inspect it")

                roles = [m["role"] for m in runtime.messages]
                self.assertNotEqual(roles[-1], "user")
                # No run of unanswered user turns after the opening preamble.
                tail = roles[2:]
                self.assertFalse(
                    any(a == "user" and b == "user" for a, b in zip(tail, tail[1:])),
                    f"consecutive user turns in {roles}",
                )

    def test_only_a_trailing_user_turn_is_removed(self) -> None:
        """The guard must not eat an answered turn or an assistant reply."""
        backend = ScriptedBackend(
            tool_response(emit("one")), prose_response("done")
        )
        runtime = self.runtime(backend)
        runtime.run_autonomous("inspect it")
        before = [dict(m) for m in runtime.messages]

        # History ends with an assistant turn; the guard must do nothing.
        runtime._close_incomplete_turn()

        self.assertEqual(runtime.messages, before)

    def test_the_turn_counter_is_decremented_with_the_message(self) -> None:
        """A removed analyst turn must not leave its number consumed.

        `user_turn_id` in evidence provenance is derived from this counter, so
        a turn that is removed but still counted would make the next real turn
        skip a number and misname every record it produces.
        """
        backend = ScriptedBackend(tool_response(emit("one")), prose_response("done"))
        runtime = self.runtime(backend)
        runtime.run_autonomous("inspect it")

        turns_before = runtime.analyst_turns
        runtime.messages.append({"role": "user", "content": "unanswered"})
        runtime.analyst_turns += 1
        runtime._close_incomplete_turn()

        self.assertEqual(runtime.analyst_turns, turns_before)
        self.assertNotEqual(runtime.messages[-1]["role"], "user")

    def test_an_unexpected_runtime_error_still_propagates(self) -> None:
        """Only recoverable backend failures are absorbed.

        This repo overloads a bare `RuntimeError` to mean "a bug: tear the
        session down and release its workspace", and an existing lifecycle test
        asserts exactly that. Widening the catch to `RuntimeError` would both
        swallow real crashes and leak the temporary workspace, so the loop
        names `RecoverableBackendError` instead.
        """

        class Exploding(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                if self.calls >= 1:
                    raise RuntimeError("backend exploded")
                return super().chat_stream(messages, **kwargs)

        runtime = self.runtime(Exploding(tool_response(emit("one"))))

        with self.assertRaises(RuntimeError) as caught:
            runtime.run_autonomous("inspect it")
        self.assertEqual(str(caught.exception), "backend exploded")

    def test_a_recoverable_error_is_absorbed_but_a_bug_is_not(self) -> None:
        """The two are distinguished by type, not by message or timing."""
        from orbit.backend.base import RecoverableBackendError
        from orbit.backend.llama_server import LlamaServerError

        self.assertTrue(issubclass(LlamaServerError, RecoverableBackendError))
        self.assertTrue(issubclass(RecoverableBackendError, RuntimeError))
        # A plain RuntimeError must not satisfy the recoverable contract.
        self.assertFalse(isinstance(RuntimeError("x"), RecoverableBackendError))

    def test_the_analyst_can_steer_after_a_backend_error(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        class FailOnceThenWork(ScriptedBackend):
            def __init__(self, *responses):
                super().__init__(*responses)
                self.failed = False

            def chat_stream(self, messages, **kwargs):
                if self.calls >= 2 and not self.failed:
                    self.failed = True
                    raise LlamaServerError("transient")
                return super().chat_stream(messages, **kwargs)

        backend = FailOnceThenWork(
            tool_response(emit("one")),
            tool_response(emit("two")),
            tool_response(emit("steered")),
            prose_response("done"),
        )
        runtime = self.runtime(backend)
        first = runtime.run_autonomous("inspect it")
        second = runtime.run_autonomous("look at the header instead")

        self.assertTrue(first.stop_reason.startswith(STOP_BACKEND_ERROR))
        self.assertEqual(second.actions_executed, 1)
        self.assertEqual(second.stop_reason, STOP_COMPLETE)


class AutonomyIsOptInTests(unittest.TestCase):
    def test_the_gate_variable_is_named_exactly(self) -> None:
        """Pinned literal: a renamed constant makes the feature unreachable.

        Fail-closed, so a typo is not a safety problem -- but it would be
        silently undiscoverable, and every other test here refers to the
        symbol rather than the name an operator actually types.
        """
        self.assertEqual(ANALYSIS_AUTONOMY_ENV, "ORBIT_ANALYSIS_AUTONOMOUS")

    @mock.patch.dict(os.environ, {"ORBIT_ANALYSIS_AUTONOMOUS": "1"}, clear=True)
    def test_the_literal_variable_enables_autonomy(self) -> None:
        self.assertTrue(analysis_autonomy_enabled())

    def test_the_gate_defaults_off_and_fails_closed(self) -> None:
        self.assertFalse(analysis_autonomy_enabled({}))
        self.assertTrue(analysis_autonomy_enabled({ANALYSIS_AUTONOMY_ENV: "1"}))
        self.assertFalse(analysis_autonomy_enabled({ANALYSIS_AUTONOMY_ENV: "0"}))
        self.assertFalse(analysis_autonomy_enabled({ANALYSIS_AUTONOMY_ENV: "yes"}))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_the_terminal_uses_one_step_when_the_gate_is_off(self) -> None:
        from orbit.terminal import repl as repl_module

        import inspect

        source = inspect.getsource(repl_module.Repl._ask_analysis)
        self.assertIn("analysis_autonomy_enabled()", source)
        self.assertIn("self.analysis.step(", source)
        self.assertFalse(analysis_autonomy_enabled())


class RollingLineageTests(AutonomousTestBase):
    """Autonomous steps stay on the existing ANALYSIS rolling lineage."""

    def test_every_autonomous_step_declares_the_analysis_step_phase(self) -> None:
        from orbit.backend.llama_server import _analysis_rolling_anchor_requested
        from orbit.runtime.analysis_runtime import ANALYSIS_STEP_PHASE
        from orbit.runtime.kv_diag import current_phase, model_call_context

        seen: list[str | None] = []

        class PhaseObservingBackend(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                seen.append(current_phase())
                return super().chat_stream(messages, **kwargs)

        backend = PhaseObservingBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(len(seen), 3)
        self.assertEqual(set(seen), {ANALYSIS_STEP_PHASE})
        # And that phase is the one the backend joins the rolling lineage for.
        with model_call_context(phase=ANALYSIS_STEP_PHASE, tools_mode="on"):
            self.assertTrue(_analysis_rolling_anchor_requested(native_backend=True))
        self.assertEqual(run.model_calls, 3)

    def test_each_step_prompt_extends_the_previous_one(self) -> None:
        """History is append-only, so step N's messages prefix step N+1's.

        This is the runtime-side precondition for exact-prefix KV reuse: the
        backend can only reuse a committed prefix if the prompt it is asked to
        serve still begins with it. Nothing here normalises or approximates --
        the earlier messages must be present, unchanged, in order.
        """
        backend = ScriptedBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            prose_response("done"),
        )
        self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(len(backend.seen_messages), 3)
        for earlier, later in zip(backend.seen_messages, backend.seen_messages[1:]):
            self.assertLess(len(earlier), len(later))
            self.assertEqual(later[: len(earlier)], earlier)

    def test_autonomous_steps_never_declare_a_chat_phase(self) -> None:
        from orbit.runtime.analysis_runtime import ANALYSIS_STEP_PHASE
        from orbit.runtime.kv_diag import current_phase

        seen: list[str | None] = []

        class PhaseObservingBackend(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                seen.append(current_phase())
                return super().chat_stream(messages, **kwargs)

        backend = PhaseObservingBackend(
            tool_response(emit("one")), prose_response("done")
        )
        self.runtime(backend).run_autonomous("inspect it")

        for phase in seen:
            self.assertEqual(phase, ANALYSIS_STEP_PHASE)
            self.assertNotIn("chat", str(phase).lower())


class IntermediateEvidenceIsShownTests(AutonomousTestBase):
    """Every step of an autonomous run reaches the analyst, not just the last."""

    def test_the_terminal_renders_each_completed_step(self) -> None:
        rendered: list[object] = []

        backend = ScriptedBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            prose_response("done"),
        )
        self.runtime(backend).run_autonomous(
            "inspect it", on_step=lambda step, record: rendered.append(step)
        )

        # The hook the terminal renders through fires once per completed step.
        self.assertEqual(len(rendered), 3)
        with_evidence = [s for s in rendered if s.evidence is not None]
        self.assertEqual(len(with_evidence), 2)

    def test_a_silent_step_after_a_talkative_one_is_not_told_prose_was_shown(
        self,
    ) -> None:
        """One renderer serves every step, so its prose flag must be per-step.

        `format_analysis_step(prose_already_shown=True)` suppresses the prose
        it would otherwise reprint. Without a reset between steps the flag
        stays set from the first step that streamed anything, so a later step
        that streamed nothing would be told its prose was already on screen and
        would silently drop it.
        """
        from orbit.terminal.streaming import StreamRenderer

        backend = ScriptedBackend(
            tool_response(emit("a"), text="thinking out loud"),
            tool_response(emit("b"), text=""),
            prose_response("final answer"),
        )
        renderer = StreamRenderer(thinking=False, render_markdown_mode="plain")
        renderer.start()
        flags: list[bool] = []

        def show(step, record) -> None:
            flags.append(renderer.rendered_visible_text)
            renderer.reset_visible_text()

        try:
            self.runtime(backend).run_autonomous(
                "inspect it", on_delta=renderer.write, on_step=show
            )
        finally:
            renderer.finish()

        # The silent middle step must not inherit the first step's flag.
        self.assertEqual(flags, [True, False, True])

    def test_the_repl_names_a_first_step_failure_truthfully(self) -> None:
        """A backend that died must not be reported as an analyst interrupt.

        When no step completes there is nothing to render, and that branch used
        to print "cancelled" unconditionally -- so a first-step backend failure
        told the analyst they had interrupted their own run, and discarded the
        diagnostic. The run already carries both facts.
        """
        import inspect

        from orbit.terminal.repl import Repl

        source = inspect.getsource(Repl._ask_analysis)
        head, _, tail = source.partition("if result is None:")
        self.assertTrue(tail, "the no-step branch must exist")
        branch = tail.split("return")[0]
        self.assertIn("run.cancelled", branch)
        self.assertIn("run.stop_reason", branch)

    def test_a_first_step_backend_error_is_not_labelled_cancelled(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        class DeadBackend(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                raise LlamaServerError("upstream is down")

        run = self.runtime(DeadBackend()).run_autonomous("inspect it")

        self.assertIsNone(run.last_step)
        self.assertFalse(run.cancelled, "a dead backend is not a cancellation")
        self.assertIn("upstream is down", run.stop_reason)

    def test_a_first_step_cancellation_is_labelled_cancelled(self) -> None:
        class Interrupting(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                raise KeyboardInterrupt

        run = self.runtime(Interrupting()).run_autonomous("inspect it")

        self.assertIsNone(run.last_step)
        self.assertTrue(run.cancelled)
        self.assertEqual(run.stop_reason, STOP_CANCELLED)

    def test_the_repl_wires_the_on_step_hook(self) -> None:
        """A run whose last step is prose would otherwise show no evidence.

        `STOP_COMPLETE` is the ordinary ending, and its final step carries no
        action, so rendering only `run.last_step` would tell the analyst that
        actions ran while showing none of what they produced.
        """
        import inspect

        from orbit.terminal.repl import Repl

        source = inspect.getsource(Repl._ask_analysis)
        self.assertIn("on_step=", source)
        self.assertIn("format_analysis_step(", source)


class ProgressLedgerUnitTests(unittest.TestCase):
    """The classifier alone, on hand-built step shapes."""

    class Step:
        def __init__(self, **kw):
            self.action_attempted = kw.get("attempted", True)
            self.action_executed = kw.get("executed", True)
            self.rejection = kw.get("rejection")
            self.result = None
            self.evidence = kw.get("evidence")

    class Evidence:
        def __init__(self, sha, artifacts=()):
            self.raw_sha256 = sha
            self.metadata = {"artifacts": list(artifacts), "code_sha256": "c"}

    def test_duplicate_evidence_is_not_progress(self) -> None:
        ledger = ProgressLedger()
        a = self.Step(evidence=self.Evidence("sha-1"))
        b = self.Step(evidence=self.Evidence("sha-1"))
        self.assertEqual(ledger.classify(1, a).classification, NEW_CONTENT)
        self.assertEqual(ledger.classify(2, b).classification, NO_PROGRESS)

    def test_a_new_artifact_is_progress_without_new_evidence(self) -> None:
        ledger = ProgressLedger()
        art = [{"handle": "/workspace/work/a.bin", "sha256": "d1"}]
        first = self.Step(evidence=self.Evidence("sha-1"))
        second = self.Step(evidence=self.Evidence("sha-1", art))
        ledger.classify(1, first)
        record = ledger.classify(2, second)
        self.assertEqual(record.classification, NEW_CONTENT)
        self.assertEqual(record.new_artifacts, ("/workspace/work/a.bin",))

    def test_prose_is_complete_and_a_refusal_is_error(self) -> None:
        ledger = ProgressLedger()
        self.assertEqual(
            ledger.classify(1, self.Step(attempted=False, executed=False)).classification,
            COMPLETE,
        )
        self.assertEqual(
            ledger.classify(
                2, self.Step(executed=False, rejection="bad call")
            ).classification,
            ERROR,
        )

    def test_the_classifier_carries_no_domain_vocabulary(self) -> None:
        """A generic mechanism must not mention what it might be looking at."""
        from orbit.runtime import analysis_progress

        # Executable code only: the module docstring names what it refuses to
        # know, and a check that failed on its own disclaimer would push that
        # explanation out of the file.
        import ast

        tree = ast.parse(Path(analysis_progress.__file__).read_text())
        stripped = ast.unparse(
            ast.Module(
                body=[n for n in tree.body if not _is_docstring(n)], type_ignores=[]
            )
        )
        lowered = _without_docstrings(stripped).lower()
        for term in (
            "malware", "xor", "powershell", "ioc", "payload", "obfusc",
            "base64", "decode", "url", "shellcode", "virus", "indicator",
        ):
            self.assertNotIn(term, lowered, f"domain term {term!r} leaked in")


if __name__ == "__main__":
    unittest.main()
