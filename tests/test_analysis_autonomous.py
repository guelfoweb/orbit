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
    AUTONOMOUS_REPLAN_MESSAGE,
    ANALYSIS_TOOL_NAME,
    AUTONOMOUS_CONTINUATION_MESSAGE,
    FINISH_TOOL_NAME,
    PLAN_TOOL_NAME,
    MAX_AUTONOMOUS_ACTIONS,
    MAX_AUTONOMOUS_MODEL_CALLS,
    MAX_CONSECUTIVE_ERRORS,
    MAX_CONSECUTIVE_NO_PROGRESS,
    MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS,
    SOFT_MAX_AUTONOMOUS_ACTIONS,
    STOP_SOFT_MAX_ACTIONS,
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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        # Two calls: the action earned the second one.
        self.assertEqual(backend.calls, 2)
        # The count is the controller's 2N+1, not one call per action; what
        # this test is about is that new content CAUSED a further call.
        self.assertGreater(run.model_calls, 2)
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
        self.runtime(backend).run_autonomous("inspect it", finalize=False)

        second_prompt = backend.seen_messages[1]
        analyst_lines = [m for m in second_prompt if m.get("role") == "user"]
        # The active instruction is the controller's per-question directive;
        # asserting the continuation text was the last line encoded the
        # free-form prompt geometry and is retired. The text itself is still
        # required to be generic, which is the half worth keeping.
        self.assertIn("Work on this question and nothing else:",
                      analyst_lines[-1]["content"])
        self.assertIn("new useful", AUTONOMOUS_CONTINUATION_MESSAGE)
        self.assertIn("Do not repeat", AUTONOMOUS_CONTINUATION_MESSAGE)

    def test_several_new_content_steps_continue(self) -> None:
        backend = ScriptedBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            tool_response(emit("three")),
            prose_response("that is all"),
        )
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, 3)
        # 2N+1 for the actions and their finish decisions, plus the closing
        # prose call that ends the run -- measured, not derived.
        self.assertEqual(run.model_calls, 8)
        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NEW_CONTENT, NEW_CONTENT, COMPLETE],
        )

    def test_prose_with_no_action_stops_immediately(self) -> None:
        backend = ScriptedBackend(prose_response("I have nothing to run"))
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertEqual(backend.calls, 1)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_COMPLETE)
        self.assertEqual([r.classification for r in run.progress], [COMPLETE])

    def test_model_prose_is_never_progress(self) -> None:
        """A confident claim in prose must not buy another step."""
        backend = ScriptedBackend(
            prose_response("I have decoded stage two and will continue shortly")
        )
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertEqual(backend.calls, 1)
        self.assertNotIn(NEW_CONTENT, [r.classification for r in run.progress])

    def test_a_changed_artifact_sha_counts_as_new_content(self) -> None:
        """Same handle, different bytes: a real state transition."""
        backend = ScriptedBackend(
            tool_response(write_artifact("stage.bin", "aaa")),
            tool_response(write_artifact("stage.bin", "bbb")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("transform it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("transform it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NEW_CONTENT, NO_PROGRESS, COMPLETE],
        )
        self.assertEqual(run.stop_reason, STOP_COMPLETE)


class ErrorsAreBoundedTests(AutonomousTestBase):
    def test_consecutive_errors_stop_the_run(self) -> None:
        backend = ScriptedBackend(*[tool_response(emit("x"), count=2)] * 5)
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        # This test is about the ACTION bound, so the call ceiling is only
        # headroom. Under the structured controller each action costs two
        # calls -- itself and its finish decision -- plus one for the plan,
        # so the headroom is 2N+1 rather than N+3. The subject of the test
        # is unchanged; only the budget that lets it be reached is.
        run = self.runtime(backend).run_autonomous(
            "inspect it",
            max_model_calls=2 * MAX_AUTONOMOUS_ACTIONS + 1,
            finalize=False,
        )

        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)
        self.assertEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)
        self.assertEqual(backend.calls, MAX_AUTONOMOUS_ACTIONS)

    def test_max_model_calls_stops_the_run(self) -> None:
        backend = ScriptedBackend(
            *[tool_response(emit(f"v{i}")) for i in range(12)]
        )
        run = self.runtime(backend).run_autonomous(
            "inspect it", max_model_calls=3, max_actions=99, finalize=False
        )

        self.assertEqual(run.stop_reason, STOP_MAX_MODEL_CALLS)
        self.assertEqual(run.model_calls, 3)
        # The ceiling counts EVERY model invocation, so under the structured
        # controller it is spent across the plan, the action and the finish
        # decision rather than on three actions. `backend.calls` counts only
        # the scripted action slots, so the three counters are summed here:
        # the bound is on invocations, and this proves all three were it.
        self.assertEqual(
            backend.calls + backend.plan_calls + backend.finish_calls, 3
        )

    def test_bounds_are_small_and_explicit(self) -> None:
        self.assertEqual(SOFT_MAX_AUTONOMOUS_ACTIONS, 8)
        self.assertEqual(MAX_AUTONOMOUS_ACTIONS, 12)
        # 18, not 15: a question-ledger action costs two calls rather than one
        # -- the step and the call classifying its question -- plus coverage
        # and planning up front. Without a term for that overhead the action
        # ceiling is unreachable and runs end on arithmetic instead of on the
        # action policy, which is the defect that raised this from 13 to 15 in
        # the first place. The action bounds themselves are unchanged.
        self.assertEqual(MAX_AUTONOMOUS_MODEL_CALLS, 18)
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
        run = self.runtime(ScriptedBackend(*script)).run_autonomous("inspect it", finalize=False)

        classifications = [r.classification for r in run.progress]
        self.assertIn(ERROR, classifications)
        self.assertIn(NO_PROGRESS, classifications)
        self.assertEqual(run.stop_reason, STOP_MAX_MODEL_CALLS)
        self.assertEqual(run.model_calls, MAX_AUTONOMOUS_MODEL_CALLS)

    def test_no_second_finalization_or_classifier_call(self) -> None:
        """No call is spent on finalization or classification.

        The old form of this test asserted that EVERY call was an ordinary
        step offering only `execute_analysis`. That was the free-form
        contract: with the structured controller a run legitimately spends
        calls on the plan and on each finish decision, and every one of them
        is a control call rather than a hidden classifier.

        What the test is really about survives intact -- the runtime never
        adds a finalization or classification call of its own -- and is now
        asserted against the three surfaces that may legitimately appear.
        """
        backend = ScriptedBackend(
            tool_response(emit("one")),
            tool_response(emit("two")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        # Every invocation is accounted for by exactly one of the three
        # surfaces: nothing was spent on anything else.
        self.assertEqual(
            backend.calls + backend.plan_calls + backend.finish_calls,
            run.model_calls,
        )
        for offered in backend.seen_tools:
            self.assertIn(
                offered,
                ([ANALYSIS_TOOL_NAME], [PLAN_TOOL_NAME], [FINISH_TOOL_NAME]),
            )


class HumanControlTests(AutonomousTestBase):
    def test_cancellation_stops_the_loop_and_keeps_evidence(self) -> None:
        class CancellingBackend(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                if self.calls >= 1:
                    raise KeyboardInterrupt
                return super().chat_stream(messages, **kwargs)

        backend = CancellingBackend(tool_response(emit("one")))
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
        run = runtime.run_autonomous("inspect it", finalize=False)

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

        run = self.runtime(CancelImmediately()).run_autonomous("inspect it", finalize=False)

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
        first = runtime.run_autonomous("inspect it", finalize=False)
        history_after_first = len(runtime.messages)

        second = runtime.run_autonomous("now look at the header instead", finalize=False)

        self.assertEqual(first.stop_reason, STOP_COMPLETE)
        self.assertEqual(second.actions_executed, 1)
        # Append-only: the second run extended the same history.
        self.assertGreater(len(runtime.messages), history_after_first)
        # The analyst's line is carried into the second run's context. It is
        # no longer the LAST user message -- the per-question guidance is --
        # so presence is asserted rather than position.
        delivered = [
            str(m.get("content", ""))
            for conversation in backend.seen_messages
            for m in conversation
        ]
        self.assertTrue(
            any("now look at the header instead" in text for text in delivered)
        )


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
        run = runtime.run_autonomous("inspect it", finalize=False)  # must not raise

        self.assertTrue(run.stop_reason.startswith(STOP_BACKEND_ERROR))
        self.assertIn("LlamaServerError", run.stop_reason)
        self.assertEqual(run.actions_executed, 2)
        self.assertIsNotNone(run.last_step)

    def test_completed_steps_survive_a_backend_error(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        runtime = self.runtime(self._failing(LlamaServerError("backend died")))
        run = runtime.run_autonomous("inspect it", finalize=False)

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
                runtime.run_autonomous("inspect it", finalize=False)

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
        runtime.run_autonomous("inspect it", finalize=False)
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
        runtime.run_autonomous("inspect it", finalize=False)

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
            runtime.run_autonomous("inspect it", finalize=False)
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
        first = runtime.run_autonomous("inspect it", finalize=False)
        second = runtime.run_autonomous("look at the header instead", finalize=False)

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
    def test_the_terminal_uses_one_step_when_autonomy_is_off(self) -> None:
        """The terminal branches on its session state, not on the environment.

        It used to read `analysis_autonomy_enabled()` on every analysis turn.
        That state is now resolved once at startup and can be overridden with
        `/autonomous`, so re-reading the environment per turn would let an
        exported value silently outrank what the analyst last chose. The gate
        itself is unchanged and still supplies the startup default.
        """
        from orbit.terminal import repl as repl_module

        import inspect

        source = inspect.getsource(repl_module.Repl._ask_analysis)
        self.assertIn("self.autonomous_analysis", source)
        self.assertNotIn("analysis_autonomy_enabled()", source)
        self.assertIn("self.analysis.step(", source)
        # The runtime gate still decides the startup default, and is still off.
        self.assertFalse(analysis_autonomy_enabled())
        init = inspect.getsource(repl_module.Repl.__post_init__)
        self.assertIn("analysis_autonomy_enabled()", init)


class RollingLineageTests(AutonomousTestBase):
    def runtime(self, backend):
        """Call headroom for the structured controller's 2N+1 cost.

        No test in this class names `max_model_calls`: their subject is
        how each step's prompt extends the previous one. An action now costs two calls -- itself and its finish
        decision -- plus one for the plan, so the default ceiling would stop
        these runs on CALLS and hide what they check. Only the headroom
        changes; every assertion is the original one.

        This cannot mask the call-ceiling tests, which live in
        `BoundsAreEnforcedTests`, set `max_model_calls` explicitly and assert
        on `run.model_calls`.
        """
        built = super().runtime(backend)
        original = built.run_autonomous

        def with_headroom(*args, **kwargs):
            kwargs.setdefault(
                "max_model_calls", 2 * MAX_AUTONOMOUS_ACTIONS + 1
            )
            return original(*args, **kwargs)

        built.run_autonomous = with_headroom
        return built

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
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        # Every call the run makes declares the analysis-step phase, control
        # calls included: the count is now the controller's 2N+1 rather than
        # one per action, and the phase is what this test is about.
        self.assertEqual(len(seen), run.model_calls)
        self.assertEqual(set(seen), {ANALYSIS_STEP_PHASE})
        # And that phase is the one the backend joins the rolling lineage for.
        with model_call_context(phase=ANALYSIS_STEP_PHASE, tools_mode="on"):
            self.assertTrue(_analysis_rolling_anchor_requested(native_backend=True))
        self.assertEqual(run.actions_executed, 2)

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
        self.runtime(backend).run_autonomous("inspect it", finalize=False)

        # Compared across ACTION calls, and on the committed history rather
        # than the whole prompt. Each call ends with a transient directive --
        # the per-question guidance, which `_resolve_messages` builds fresh
        # and never commits -- so the prompts differ in their last line by
        # design. What KV reuse needs, and what this test is for, is that
        # everything BEFORE it extends without rewriting.
        steps = [
            messages
            for messages, offered in zip(backend.seen_messages, backend.seen_tools)
            if offered == [ANALYSIS_TOOL_NAME]
        ]
        self.assertEqual(len(steps), 3)
        for earlier, later in zip(steps, steps[1:]):
            committed_earlier, committed_later = earlier[:-1], later[:-1]
            self.assertLess(len(committed_earlier), len(committed_later))
            self.assertEqual(
                committed_later[: len(committed_earlier)], committed_earlier
            )

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
        self.runtime(backend).run_autonomous("inspect it", finalize=False)

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
            "inspect it", on_step=lambda step, record: rendered.append(step), finalize=False
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
                "inspect it", on_delta=renderer.write, on_step=show, finalize=False
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

        run = self.runtime(DeadBackend()).run_autonomous("inspect it", finalize=False)

        self.assertIsNone(run.last_step)
        self.assertFalse(run.cancelled, "a dead backend is not a cancellation")
        self.assertIn("upstream is down", run.stop_reason)

    def test_a_first_step_cancellation_is_labelled_cancelled(self) -> None:
        class Interrupting(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                raise KeyboardInterrupt

        run = self.runtime(Interrupting()).run_autonomous("inspect it", finalize=False)

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


# --- progress reliability: strategy fingerprint, replan, grounded ending ----


def nondet(tag: str) -> str:
    """A program whose stdout differs every run but whose experiment does not."""
    return (
        "import os, random, time\n"
        f"print({tag!r}, random.random(), time.time(), os.getpid(), end='')"
    )


class StrategyFingerprintTests(AutonomousTestBase):
    def runtime(self, backend):
        """Call headroom for the structured controller's 2N+1 cost.

        No test in this class names `max_model_calls`: their subject is
        which strategies count as genuinely new work. An action now costs two calls -- itself and its finish
        decision -- plus one for the plan, so the default ceiling would stop
        these runs on CALLS and hide what they check. Only the headroom
        changes; every assertion is the original one.

        This cannot mask the call-ceiling tests, which live in
        `BoundsAreEnforcedTests`, set `max_model_calls` explicitly and assert
        on `run.model_calls`.
        """
        built = super().runtime(backend)
        original = built.run_autonomous

        def with_headroom(*args, **kwargs):
            kwargs.setdefault(
                "max_model_calls", 2 * MAX_AUTONOMOUS_ACTIONS + 1
            )
            return original(*args, **kwargs)

        built.run_autonomous = with_headroom
        return built

    """Re-running one experiment is not discovery, whatever it prints.

    Novelty by evidence hash alone cannot see this: a program that prints a
    timestamp, a pid or a random value produces a different hash every time and
    looked like progress on every repetition. The fingerprint asks the other
    question -- has this strategy already been tried against this state -- and
    only an unseen strategy that also added attested state counts.
    """

    def test_repeated_action_with_random_output_is_not_progress(self) -> None:
        run = self.runtime(
            ScriptedBackend(*[tool_response(nondet("scan"))] * 8)
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS],
        )
        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS))
        self.assertTrue(run.progress[-1].repeated_strategy)
        # It stops far short of the action bound it used to run to -- and the
        # two repeats no longer reach the sandbox at all, so one useful action
        # is spent where three used to be. The classification sequence above is
        # unchanged: what moved is the cost of reaching it.
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.suppressed_duplicates, 2)

    def test_repeated_action_with_timestamp_output_is_not_progress(self) -> None:
        code = "import time\nprint('t', time.time(), end='')"
        run = self.runtime(ScriptedBackend(*[tool_response(code)] * 8)).run_autonomous(
            "inspect it", finalize=False
        )

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS],
        )

    def test_repeated_action_with_pid_output_is_not_progress(self) -> None:
        code = "import os\nprint('pid', os.getpid(), os.urandom(8).hex(), end='')"
        run = self.runtime(ScriptedBackend(*[tool_response(code)] * 8)).run_autonomous(
            "inspect it", finalize=False
        )

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS],
        )

    def test_a_new_action_producing_new_evidence_is_still_progress(self) -> None:
        """The fingerprint must not suppress genuine work."""
        run = self.runtime(
            ScriptedBackend(
                tool_response(emit("a")),
                tool_response(emit("b")),
                tool_response(emit("c")),
                prose_response("done"),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NEW_CONTENT, NEW_CONTENT, COMPLETE],
        )

    def test_duplicate_evidence_from_a_new_action_is_not_progress(self) -> None:
        """Both questions must pass: an unseen strategy that adds nothing fails."""
        run = self.runtime(
            ScriptedBackend(
                tool_response(emit("same")),
                tool_response("x = 1\n" + emit("same")),
                tool_response("y = 2\n" + emit("same")),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS],
        )

    def test_the_same_code_after_the_workspace_changed_is_a_new_experiment(
        self,
    ) -> None:
        """Re-running a program over changed inputs is not a repeat.

        Without the workspace component the fingerprint would be (code, source)
        alone, and a program legitimately re-run after an earlier step
        materialised a new artifact would be suppressed as a repeat -- exactly
        the multi-stage case autonomous analysis exists for.
        """
        read_work = (
            "import pathlib\n"
            "p = pathlib.Path('/workspace/work/stage.txt')\n"
            "print(p.read_text() if p.exists() else 'absent', end='')"
        )
        run = self.runtime(
            ScriptedBackend(
                tool_response(read_work),                       # 'absent'
                tool_response(write_artifact("stage.txt", "one")),
                tool_response(read_work),                       # same code, new state
                prose_response("done"),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NEW_CONTENT, NEW_CONTENT, COMPLETE],
        )
        self.assertFalse(run.progress[2].repeated_strategy)

    def test_an_iterative_carver_is_never_suppressed(self) -> None:
        """The case that matters most: one program, real progress every run.

        A carver that extracts the next embedded object each time reuses one
        program and writes on every run. It must keep going. An earlier version
        of this classifier bounded how often one program could claim artifact
        progress, and it halted this exact trajectory after two objects while
        three more remained -- a confidently incomplete analysis, which is
        worse than a slow one.
        """
        carver = (
            "import pathlib\n"
            "w = pathlib.Path('/workspace/work')\n"
            "n = len(list(w.glob('obj*.bin')))\n"
            "(w / f'obj{n}.bin').write_text('object-%d' % n)\n"
            "print('carved object', n, end='')"
        )
        run = self.runtime(
            ScriptedBackend(*[tool_response(carver)] * MAX_AUTONOMOUS_ACTIONS)
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT] * MAX_AUTONOMOUS_ACTIONS,
        )
        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)

    def test_a_program_that_only_rewrites_random_bytes_is_still_bounded(self) -> None:
        """The honest limit of the classifier, recorded rather than papered over.

        A program rewriting random bytes into an artifact is indistinguishable
        here from one carving the next stage: same program, writes every run,
        and a new evidence hash each time because the observation names the
        digest just written. No deterministic signal available to the ledger
        separates them, so this case is not caught by classification at all --
        it is ended by the action bound. That is a bounded, safe stop, and the
        alternative was suppressing genuine unpacking.
        """
        code = (
            "import random, pathlib\n"
            "pathlib.Path('/workspace/work/a.bin').write_text(str(random.random()))\n"
            "print('wrote', end='')"
        )
        run = self.runtime(ScriptedBackend(*[tool_response(code)] * 12)).run_autonomous(
            "inspect it", finalize=False
        )

        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)
        self.assertEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)

    def test_genuine_multi_stage_work_is_not_suppressed(self) -> None:
        """Distinct programs building on each other must run to the bound.

        This is the case autonomous analysis exists for, and it is what the
        artifact credit must not break: each stage is a different program, so
        none of them exhausts its own credit.
        """
        stages = [
            tool_response(
                "import pathlib\n"
                f"pathlib.Path('/workspace/work/s{i}.bin').write_text('{i}')\n"
                f"print('stage{i}', end='')"
            )
            for i in range(1, MAX_AUTONOMOUS_ACTIONS + 1)
        ]
        run = self.runtime(ScriptedBackend(*stages)).run_autonomous(
            "inspect it", finalize=False
        )

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT] * MAX_AUTONOMOUS_ACTIONS,
        )
        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)

    def test_the_fingerprint_covers_code_source_and_workspace(self) -> None:
        """All three components are load-bearing, checked directly."""
        from orbit.runtime.analysis_progress import ProgressLedger

        ledger = ProgressLedger()

        class R:
            def __init__(self, code, src):
                self.code_sha256 = code
                self.input_sha256 = src

        class S:
            def __init__(self, code, src):
                self.result = R(code, src)

        base = ledger._strategy_fingerprint(S("c1", "s1"), None, "c1")
        self.assertNotEqual(base, ledger._strategy_fingerprint(S("c2", "s1"), None, "c2"))
        self.assertNotEqual(base, ledger._strategy_fingerprint(S("c1", "s2"), None, "c1"))
        # Workspace state participates too.
        ledger.artifacts["/workspace/work/a"] = "d1"
        self.assertNotEqual(base, ledger._strategy_fingerprint(S("c1", "s1"), None, "c1"))

    def test_the_fingerprint_does_not_inspect_code_or_output(self) -> None:
        """It is three hashes of state, not an interpreter."""
        import inspect

        from orbit.runtime import analysis_progress

        # Executable code only: the docstring says the method does not
        # normalise, and a check that tripped on its own disclaimer would push
        # that explanation out of the file.
        import textwrap

        source = _without_docstrings(
            textwrap.dedent(
                inspect.getsource(analysis_progress.ProgressLedger._strategy_fingerprint)
            )
        )
        for banned in ("ast.", "re.", "normali", ".lower()", ".split(", ".strip()"):
            self.assertNotIn(banned, source, f"{banned!r} suggests inspection")
        self.assertIn("sha256", source)


class BoundedReplanTests(AutonomousTestBase):
    """Exactly one replan, and only on the first unproductive step."""

    def _messages(self, backend) -> list[str]:
        """Every user message the model was sent, across all calls.

        This used to read only the LAST user message of each call, which was
        the whole prompt when the free-form loop drove the run. Under the
        structured controller the last message is the per-question guidance
        and the replan sits behind it in the carried history -- still sent,
        still exactly once, but no longer the final line.

        Scanning the whole context keeps what this class is about: how MANY
        replans the model receives and after which step. Counting the same
        message once per call it survives in would inflate that, so the
        de-duplication below preserves the original meaning of "count".
        """
        seen: list[str] = []
        for msgs in backend.seen_messages:
            for m in msgs:
                if m.get("role") == "user" and m["content"] not in seen:
                    seen.append(m["content"])
        return seen

    def test_first_no_progress_inserts_exactly_one_replan(self) -> None:
        same = emit("x")
        backend = ScriptedBackend(
            tool_response(same), tool_response(same),
            tool_response(emit("new")), prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        sent = self._messages(backend)
        # Sent exactly once, and the run agrees it armed exactly once. The
        # POSITION it used to assert is retired: under the structured
        # controller the replan is carried context, and the per-question
        # guidance is the active instruction. Being present is the contract;
        # being last is prompt geometry the controller legitimately owns.
        self.assertEqual(sent.count(AUTONOMOUS_REPLAN_MESSAGE), 1)
        self.assertEqual(run.replans, 1)
        # Independent witness for "after the unproductive step": the stall is
        # the second classification, established by evidence hashing rather
        # than by the branch that arms the replan.
        self.assertEqual(
            [r.classification for r in run.progress][:2],
            [NEW_CONTENT, NO_PROGRESS],
        )

    def test_the_replan_message_is_model_visible(self) -> None:
        """It reaches the model -- as context, not necessarily as the last line.

        The old form asserted it was the final user message of the third
        call. That was true of the free-form loop; the controller puts its
        per-question guidance there instead. What must remain true, and is
        what this test is for, is that the model actually SEES the replan.
        """
        same = emit("x")
        backend = ScriptedBackend(
            tool_response(same), tool_response(same),
            tool_response(emit("new")), tool_response(emit("newer")),
            prose_response("done"),
        )
        self.runtime(backend).run_autonomous(
            "inspect it", finalize=False, max_model_calls=25
        )

        delivered = [
            str(m.get("content", ""))
            for conversation in backend.seen_messages
            for m in conversation
        ]
        self.assertTrue(
            any(AUTONOMOUS_REPLAN_MESSAGE in text for text in delivered)
        )

    def test_the_question_guidance_is_the_active_instruction(self) -> None:
        """What the controller puts in the place the replan used to hold.

        The counterpart to the assertion retired above: the final user
        message of an action call is the per-question directive, which is
        how every accepted action is bound to one question without any
        prose tag.
        """
        same = emit("x")
        backend = ScriptedBackend(
            tool_response(same), tool_response(same),
            tool_response(emit("new")), prose_response("done"),
        )
        self.runtime(backend).run_autonomous(
            "inspect it", finalize=False, max_model_calls=25
        )

        action_prompts = [
            conversation[-1]["content"]
            for conversation, offered in zip(
                backend.seen_messages, backend.seen_tools
            )
            if offered == [ANALYSIS_TOOL_NAME]
        ]
        self.assertTrue(action_prompts)
        for prompt in action_prompts:
            self.assertIn("Work on this question and nothing else:", prompt)

    def test_a_productive_run_arms_no_replan(self) -> None:
        """The other side of the bound: nothing unproductive, nothing armed."""
        backend = ScriptedBackend(
            *[tool_response(emit(f"v{i}")) for i in range(4)],
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous(
            "inspect it", finalize=False, max_model_calls=25
        )

        # Independent witness: every step produced new evidence.
        self.assertNotIn(
            NO_PROGRESS, [r.classification for r in run.progress]
        )
        self.assertEqual(run.replans, 0)
        self.assertNotIn(
            AUTONOMOUS_REPLAN_MESSAGE, self._messages(backend)
        )

    def test_replan_followed_by_new_content_resumes_the_normal_loop(self) -> None:
        same = emit("x")
        backend = ScriptedBackend(
            tool_response(same), tool_response(same),
            tool_response(emit("new")), tool_response(emit("newer")),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NEW_CONTENT, NEW_CONTENT, COMPLETE],
        )
        self.assertEqual(run.stop_reason, STOP_COMPLETE)
        # Only the one replan, even though the loop continued afterwards.
        self.assertEqual(self._messages(backend).count(AUTONOMOUS_REPLAN_MESSAGE), 1)

    def test_the_replan_is_once_per_episode_not_once_per_run(self) -> None:
        """A recovered run that stalls again is told again.

        `consecutive_no_progress` resets on progress, so an alternating
        trajectory re-arms the replan: each stall is a new situation and gets
        its own message. What is never repeated is asking twice about the SAME
        stall -- a second consecutive unproductive step ends the run.

        The earlier tests all used a single stall, so this property went
        untested while the code comment claimed "exactly one is ever sent".
        The total stays bounded: every replan follows a step that consumed an
        action, so replans <= actions <= the hard ceiling.
        """
        script = []
        # Longer than the trajectory needs. Duplicates are now answered from
        # evidence instead of re-executed, so they no longer spend the action
        # budget and the run reaches further into the script than it used to;
        # a script sized to the old cost would run out mid-run.
        for i in range(12):
            script.append(tool_response(emit(f"v{i}")))
            script.append(tool_response(emit(f"v{i}")))  # duplicate -> NO_PROGRESS
        backend = ScriptedBackend(*script)

        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        classifications = [r.classification for r in run.progress]
        # Strictly alternating. Asserted as the property rather than as an
        # exact even-length list: the run now ends wherever the model-call
        # ceiling falls, which may be on either phase of the alternation.
        expected = [
            NEW_CONTENT if i % 2 == 0 else NO_PROGRESS
            for i in range(len(classifications))
        ]
        self.assertEqual(classifications, expected)
        stalls = classifications.count(NO_PROGRESS)
        # Every stall was a fresh one -- none followed another -- so each got
        # its own replan. This is the property; the exact count depends on
        # where the action budget cuts the trajectory off.
        self.assertGreater(run.replans, 1, "each fresh stall earns its own replan")
        # Every stall that completed earned a replan. The run can end on the
        # call ceiling mid-stall, in which case the final classification is
        # recorded but its replan was never sent -- so the counts differ by at
        # most the one stall in flight.
        self.assertIn(run.replans, (stalls, stalls - 1))
        # Each stall here is a suppressed duplicate: a model call, no action
        # slot. Both bounds still hold and are asserted at their tightest --
        # a suppressed stall follows a step that did consume an action, so
        # replans cannot outnumber actions any more than before.
        self.assertEqual(run.suppressed_duplicates, stalls)
        self.assertLessEqual(
            run.replans, run.actions_executed, "replans cannot outnumber actions"
        )
        # Every stall episode armed its own replan, and no episode armed two.
        # The old form additionally required each to be the final user line
        # of a call and that the runtime send ONLY its two generic messages;
        # both are retired, because the controller's per-question guidance is
        # now the active instruction and is neither of them.
        self.assertEqual(
            run.replans,
            [r.classification for r in run.progress].count(NO_PROGRESS),
        )
        # And the run really did stall more than once, so the equality above
        # is a per-episode claim rather than a coincidence at zero or one.
        self.assertGreater(
            [r.classification for r in run.progress].count(NO_PROGRESS), 1
        )

    def test_the_same_stall_is_never_replanned_twice(self) -> None:
        """Two consecutive unproductive steps end the run after one replan."""
        same = emit("x")
        backend = ScriptedBackend(*[tool_response(same)] * 6)

        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.replans, 1)
        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS))

    def test_second_consecutive_no_progress_stops(self) -> None:
        same = emit("x")
        backend = ScriptedBackend(*[tool_response(same)] * 6)
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS))
        self.assertEqual(run.replans, 1, "no second replan")
        # Armed once, and the run ended on the second stall before another
        # call could carry it. Delivery is asserted where a call follows --
        # `test_the_replan_message_is_model_visible` -- rather than here,
        # because requiring it of a run that stops is requiring a message
        # after the last word.
        self.assertEqual(
            [r.classification for r in run.progress].count(NO_PROGRESS), 2
        )

    def test_the_continuation_message_asks_for_one_new_useful_step(self) -> None:
        """Generic: it names no artifact, technique or direction."""
        text = AUTONOMOUS_CONTINUATION_MESSAGE.lower()
        self.assertIn("new", text)
        self.assertIn("do not repeat", text)
        for domain in ("xor", "powershell", "url", "base64", "decode", "malware", "ioc"):
            self.assertNotIn(domain, text)
            self.assertNotIn(domain, AUTONOMOUS_REPLAN_MESSAGE.lower())

    def test_the_continuation_text_stays_generic(self) -> None:
        """The constant survives; its position as the active line does not.

        This replaces a test asserting the continuation message was the last
        user line of the second call. Under the structured controller that
        place belongs to the per-question guidance -- asserted directly in
        `test_the_question_guidance_is_the_active_instruction`. The
        continuation text remains in the module and remains generic, which
        is what the sibling test above checks in detail.
        """
        self.assertTrue(AUTONOMOUS_CONTINUATION_MESSAGE.strip())
        self.assertNotIn("question", AUTONOMOUS_CONTINUATION_MESSAGE.lower())


class GroundedFinalizationTests(AutonomousTestBase):
    """Every ending that is not a cancellation produces one grounded answer."""

    def _run(self, *responses, **kw):
        backend = ScriptedBackend(*responses)
        run = self.runtime(backend).run_autonomous("inspect it", **kw)
        return run, backend

    def test_natural_completion_produces_a_grounded_report(self) -> None:
        run, backend = self._run(
            tool_response(emit("a")), prose_response("done"), prose_response("REPORT")
        )

        self.assertEqual(run.stop_reason, STOP_COMPLETE)
        self.assertIsNotNone(run.final_report)
        self.assertEqual(run.final_report.text, "REPORT")

    def test_a_protective_stop_produces_a_grounded_report(self) -> None:
        same = emit("x")
        run, backend = self._run(
            tool_response(same), tool_response(same), tool_response(same),
            prose_response("REPORT"),
        )

        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS))
        self.assertIsNotNone(run.final_report)
        self.assertEqual(run.final_report.text, "REPORT")

    def test_the_report_is_told_the_run_stopped_early(self) -> None:
        """A reader must not mistake a bounded stop for a finished analysis."""
        same = emit("x")
        run, backend = self._run(
            tool_response(same), tool_response(same), tool_response(same),
            prose_response("REPORT"),
        )

        final_prompt = backend.seen_messages[-1]
        text = " ".join(str(m.get("content", "")) for m in final_prompt)
        self.assertIn("stopped before the model chose to finish", text)
        self.assertIn(STOP_NO_PROGRESS, text)

    def test_a_natural_ending_is_not_labelled_as_stopped_early(self) -> None:
        run, backend = self._run(
            tool_response(emit("a")), prose_response("done"), prose_response("REPORT")
        )

        final_prompt = backend.seen_messages[-1]
        text = " ".join(str(m.get("content", "")) for m in final_prompt)
        self.assertNotIn("stopped before the model chose to finish", text)

    def test_finalization_is_offered_no_tools(self) -> None:
        run, backend = self._run(
            tool_response(emit("a")), prose_response("done"), prose_response("REPORT")
        )

        # The closing report carries no tools -- the point of the test. The
        # first call is now PLAN rather than an action, so the opening
        # assertion checks that an analysis step happened at all rather than
        # that it came first.
        self.assertIn([ANALYSIS_TOOL_NAME], backend.seen_tools)
        self.assertEqual(backend.seen_tools[-1], [])

    def test_finalization_cannot_restart_the_analysis(self) -> None:
        """A tool call in the closing report must execute nothing."""
        run, backend = self._run(
            tool_response(emit("a")),
            prose_response("done"),
            tool_response(emit("should never run"), text="REPORT"),
        )

        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(len(run.steps), 2, "the report is not a step")
        # Exactly one closing call: no loop back into stepping.
        self.assertEqual(backend.calls, 3)

    def test_finalization_appends_nothing_to_history(self) -> None:
        backend = ScriptedBackend(
            tool_response(emit("a")), prose_response("done"), prose_response("REPORT")
        )
        runtime = self.runtime(backend)
        runtime.run_autonomous("inspect it")

        self.assertNotIn("REPORT", [m.get("content") for m in runtime.messages])
        self.assertEqual(runtime.messages[-1]["role"], "assistant")

    def test_cancellation_does_not_finalise(self) -> None:
        """The analyst asked it to stop; another model call is not stopping."""

        class CancelStepsOnly(ScriptedBackend):
            """Interrupts the loop, but would happily serve a report.

            A backend that raised on every call could not tell a run that
            skipped finalization from one that attempted it and was itself
            interrupted -- both end with no report. This one answers a
            no-tools call normally, so an unwanted finalization is visible.
            """

            def chat_stream(self, messages, *, tools=None, **kwargs):
                if tools != [] and self.calls >= 1:
                    raise KeyboardInterrupt
                return super().chat_stream(messages, tools=tools, **kwargs)

        backend = CancelStepsOnly(tool_response(emit("a")), prose_response("REPORT"))
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertTrue(run.cancelled)
        self.assertIsNone(run.final_report)
        # No closing call was spent: the analyst asked for the run to stop.
        # Asserted on the tool surface too, because a finalization attempt that
        # was itself interrupted would also leave `final_report` None -- only
        # the absence of a no-tools call proves none was attempted.
        self.assertEqual(backend.calls, 1)
        self.assertNotIn([], backend.seen_tools)

    def test_a_backend_failure_during_finalization_keeps_the_run(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        class FailAtReport(ScriptedBackend):
            def chat_stream(self, messages, *, tools=None, **kwargs):
                if tools == []:
                    raise LlamaServerError("died at report")
                return super().chat_stream(messages, tools=tools, **kwargs)

        backend = FailAtReport(tool_response(emit("a")), prose_response("done"))
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(run.stop_reason, STOP_COMPLETE)
        self.assertIsNone(run.final_report)
        self.assertEqual(run.actions_executed, 1)
        self.assertTrue(
            self.store.reattest_exact(run.steps[0].evidence.evidence_id)
        )

    def test_cancelling_the_closing_report_does_not_destroy_the_run(self) -> None:
        """Ctrl-C during the report is the likeliest interrupt of all.

        It is the longest single generation in a run, and by then the analyst
        has already read every step. If it escaped `run_autonomous` the caller
        would unwind to a pre-run checkpoint and rewind past every completed
        step, leaving their evidence durable on disk with nothing referring to
        it.
        """
        class CancelAtReport(ScriptedBackend):
            def chat_stream(self, messages, *, tools=None, **kwargs):
                if tools == []:
                    raise KeyboardInterrupt
                return super().chat_stream(messages, tools=tools, **kwargs)

        backend = CancelAtReport(
            tool_response(emit("one")), tool_response(emit("two")), prose_response("done")
        )
        runtime = self.runtime(backend)

        run = runtime.run_autonomous("inspect it")  # must not raise

        self.assertIsNone(run.final_report)
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(run.stop_reason, STOP_COMPLETE)
        self.assertGreater(len(runtime.messages), 2)
        for step in run.steps:
            if step.evidence is not None:
                self.assertTrue(self.store.reattest_exact(step.evidence.evidence_id))

    def test_a_run_with_no_steps_produces_no_report(self) -> None:
        class Dead(ScriptedBackend):
            def chat_stream(self, messages, **kwargs):
                raise KeyboardInterrupt

        run = self.runtime(Dead()).run_autonomous("inspect it")
        self.assertIsNone(run.final_report)


# --- soft action budget with a hard ceiling ---------------------------------


class SoftActionBudgetTests(AutonomousTestBase):
    """Past the budget, continuing must be earned by verifiable progress.

    8 was a budget inherited from the research harness, not a boundary: a
    measured full-sample run ended on it with all eight steps still producing
    new evidence and the report naming the next deterministic step. A budget
    that cuts off work is a different thing from a policy that declines it, so
    the budget now yields to demonstrated progress -- and only to that.
    """

    def _productive(self, n: int) -> list:
        return [tool_response(emit(f"v{i}")) for i in range(n)]

    def runtime(self, backend):
        """Same runtime, with a call ceiling that lets the ACTION policy run.

        Every test in this class is about the action budget: none names
        `max_model_calls`, which was mere headroom when an action cost one
        call. Under the structured controller an action costs two -- itself
        and its finish decision -- plus one for the plan, so the default
        ceiling would stop these runs on CALLS and hide the action policy
        they exist to check. The ceiling is raised to the hard action bound's
        2N+1, and nothing else about them changes.

        This does not mask the call-bound tests: those live in
        `BoundsAreEnforcedTests`, set `max_model_calls` explicitly, and
        assert on `run.model_calls` -- which this override never touches.
        """
        built = super().runtime(backend)
        original = built.run_autonomous

        def with_headroom(*args, **kwargs):
            kwargs.setdefault(
                "max_model_calls", 2 * MAX_AUTONOMOUS_ACTIONS + 1
            )
            return original(*args, **kwargs)

        built.run_autonomous = with_headroom
        return built

    def test_new_content_at_the_soft_limit_may_continue(self) -> None:
        run = self.runtime(
            ScriptedBackend(
                *self._productive(SOFT_MAX_AUTONOMOUS_ACTIONS + 2),
                prose_response("done"),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertGreater(run.actions_executed, SOFT_MAX_AUTONOMOUS_ACTIONS)
        self.assertNotEqual(run.stop_reason, STOP_SOFT_MAX_ACTIONS)

    def test_no_progress_at_the_soft_limit_stops(self) -> None:
        same = emit("repeat")
        run = self.runtime(
            ScriptedBackend(
                *self._productive(SOFT_MAX_AUTONOMOUS_ACTIONS),
                tool_response(same),
                tool_response(same),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.stop_reason, STOP_SOFT_MAX_ACTIONS)
        self.assertEqual(run.progress[-1].classification, NO_PROGRESS)
        self.assertLess(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)

    def test_an_error_at_the_soft_limit_stops_the_run(self) -> None:
        """At the budget an error is a non-productive step, so the run ends.

        Below the budget one error is tolerated and only a second consecutive
        one stops the run. At or past it, continuing must be earned, and a
        refused action has earned nothing -- so the soft gate ends the run
        before the error counter can reach two. The error policy itself is
        unchanged; it is simply not the bound that fires first here.
        """
        run = self.runtime(
            ScriptedBackend(
                *self._productive(SOFT_MAX_AUTONOMOUS_ACTIONS),
                tool_response(emit("x"), count=2),
                tool_response(emit("x"), count=2),
                prose_response("unreached"),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.stop_reason, STOP_SOFT_MAX_ACTIONS)
        self.assertEqual(run.progress[-1].classification, ERROR)
        self.assertEqual(run.actions_executed, SOFT_MAX_AUTONOMOUS_ACTIONS)

    def test_a_single_error_below_the_budget_is_still_tolerated(self) -> None:
        """The unchanged error policy, exercised where the budget is not in play."""
        run = self.runtime(
            ScriptedBackend(
                tool_response(emit("a")),
                tool_response(emit("x"), count=2),
                tool_response(emit("b")),
                prose_response("done"),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, ERROR, NEW_CONTENT, COMPLETE],
        )
        self.assertEqual(run.stop_reason, STOP_COMPLETE)

    def test_actions_beyond_the_soft_limit_keep_running_while_productive(self) -> None:
        run = self.runtime(
            ScriptedBackend(*self._productive(MAX_AUTONOMOUS_ACTIONS - 1), prose_response("done"))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS - 1)
        self.assertEqual(run.stop_reason, STOP_COMPLETE)

    def test_the_hard_ceiling_stops_even_while_still_productive(self) -> None:
        """The ceiling is not conditional. Progress does not extend it."""
        run = self.runtime(
            ScriptedBackend(*self._productive(MAX_AUTONOMOUS_ACTIONS + 4))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)
        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)
        self.assertEqual(
            [r.classification for r in run.progress], [NEW_CONTENT] * MAX_AUTONOMOUS_ACTIONS
        )

    def test_the_ceiling_is_exact_not_off_by_one(self) -> None:
        run = self.runtime(
            ScriptedBackend(*self._productive(MAX_AUTONOMOUS_ACTIONS + 4))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, 12)
        self.assertNotEqual(run.actions_executed, 11)
        self.assertNotEqual(run.actions_executed, 13)

    def test_natural_completion_before_the_ceiling_stops_normally(self) -> None:
        run = self.runtime(
            ScriptedBackend(*self._productive(3), prose_response("done"))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.stop_reason, STOP_COMPLETE)
        self.assertEqual(run.actions_executed, 3)

    def test_tolerated_failures_do_not_consume_the_action_budget(self) -> None:
        """A tolerated failure costs a call, never an action.

        The error policy tolerates one failure between productive steps --
        progress resets the counter -- so a run can legitimately interleave
        them, and those failures must not eat into the action ceiling.

        The test was originally framed as "the call ceiling must not truncate
        a trajectory the policy allows", written when a tight ceiling made the
        action bound unreachable for any run containing a mis-formed call.
        That framing is retired: under the structured controller the two
        ceilings are independent and the call ceiling legitimately dominates
        at the production defaults. What survives -- and is what the test was
        always really about -- is that the interleaved failures cost actions
        nothing, which is asserted by reaching the full action bound here with
        the calls to afford it.
        """
        script = []
        for i in range(MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS):
            script.append(tool_response(emit(f"p{i}")))
            script.append(tool_response(emit("bad"), count=2))
        script += [
            tool_response(emit(f"q{i}"))
            for i in range(MAX_AUTONOMOUS_ACTIONS - MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS)
        ]
        script.append(prose_response("unreached"))

        # 2N+1 for the actions and their finish decisions, plus one call per
        # tolerated failure -- which is the whole point: those calls are what
        # the failures cost, and the action budget is untouched by them.
        run = self.runtime(ScriptedBackend(*script)).run_autonomous(
            "inspect it",
            finalize=False,
            max_model_calls=(
                2 * MAX_AUTONOMOUS_ACTIONS + 1 + MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS
            ),
        )

        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)
        # The point: every tolerated failure in the script cost a call and
        # not an action, so the full action budget was still spent on work.
        self.assertEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)

    def test_the_chosen_terms_in_the_budget_are_explicit(self) -> None:
        """Every chosen term is named, not folded into a number.

        Two of them now. The non-productive allowance is what a run may spend
        on calls that execute nothing; the ledger overhead is what the
        question-ledger path costs beyond one call per action -- coverage,
        planning, and a classification per action. Both are stated so the
        ceiling stays derived rather than picked.
        """
        from orbit.runtime.analysis_runtime import MAX_LEDGER_OVERHEAD_CALLS

        self.assertEqual(MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS, 2)
        self.assertEqual(MAX_LEDGER_OVERHEAD_CALLS, 3)
        self.assertEqual(
            MAX_AUTONOMOUS_MODEL_CALLS,
            MAX_AUTONOMOUS_ACTIONS
            + 1
            + MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS
            + MAX_LEDGER_OVERHEAD_CALLS,
        )

    def test_the_budget_relation_is_derived_not_guessed(self) -> None:
        """The call ceiling must not be able to truncate the action policy.

        Every loop iteration spends exactly one model call, so a ceiling below
        hard actions + one closing prose call could end a run for want of
        budget while the action policy still allowed it -- making the hard
        ceiling unreachable and its test vacuous.
        """
        self.assertEqual(SOFT_MAX_AUTONOMOUS_ACTIONS, 8)
        self.assertEqual(MAX_AUTONOMOUS_ACTIONS, 12)
        self.assertGreater(MAX_AUTONOMOUS_ACTIONS, SOFT_MAX_AUTONOMOUS_ACTIONS)
        self.assertGreaterEqual(
            MAX_AUTONOMOUS_MODEL_CALLS,
            MAX_AUTONOMOUS_ACTIONS + 1 + MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS,
        )

    def test_the_two_ceilings_are_independent(self) -> None:
        """Whichever ceiling is tighter stops the run, and it stops cleanly.

        This replaces a test that asserted the call ceiling could never cut
        the action policy short. That was true when an action cost one model
        call. The structured controller spends two per action -- the action
        and its finish decision -- plus one for the plan, so at the
        production defaults (12 actions, 18 calls) the CALL ceiling is now
        the tighter of the two and dominates. Exploration is still bounded,
        by the controller's own 6 questions x 2 actions.

        Retired here is the derived assumption that the two ceilings are
        reachable together, never the ceilings themselves: both are still
        asserted below, and `test_the_hard_action_bound_is_not_dead` proves
        the action bound still enforces when the calls are there to reach it.
        """
        run = self.runtime(
            ScriptedBackend(
                *self._productive(MAX_AUTONOMOUS_ACTIONS), prose_response("done")
            )
        ).run_autonomous(
            "inspect it",
            finalize=False,
            max_model_calls=MAX_AUTONOMOUS_MODEL_CALLS,
        )

        # Both ceilings still hold, and the run stopped on one of them
        # honestly rather than running past either.
        self.assertLessEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)
        self.assertLessEqual(run.model_calls, MAX_AUTONOMOUS_MODEL_CALLS)
        self.assertIn(run.stop_reason, (STOP_MAX_ACTIONS, STOP_MAX_MODEL_CALLS))
        # At today's defaults it is specifically the call ceiling, which is
        # the measured consequence this test now documents.
        self.assertEqual(run.stop_reason, STOP_MAX_MODEL_CALLS)
        self.assertEqual(run.model_calls, MAX_AUTONOMOUS_MODEL_CALLS)

    def test_the_hard_action_bound_is_not_dead(self) -> None:
        """The action ceiling still enforces when the calls exist to reach it.

        Independent witness that retiring the reachability contract above did
        not retire the bound: given 2N+1 calls the run reaches exactly
        `MAX_AUTONOMOUS_ACTIONS` and refuses the thirteenth.
        """
        backend = ScriptedBackend(
            *self._productive(MAX_AUTONOMOUS_ACTIONS + 3), prose_response("done")
        )
        run = self.runtime(backend).run_autonomous(
            "inspect it",
            finalize=False,
            max_model_calls=2 * MAX_AUTONOMOUS_ACTIONS + 1,
        )

        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)
        self.assertEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)
        # Action 13 was scripted and available; the runtime did not run it.
        self.assertEqual(backend.calls, MAX_AUTONOMOUS_ACTIONS)

    def test_replan_semantics_are_unchanged_by_the_budget(self) -> None:
        same = emit("x")
        backend = ScriptedBackend(
            tool_response(same), tool_response(same),
            tool_response(emit("new")), prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.replans, 1)
        self.assertEqual(
            [r.classification for r in run.progress],
            [NEW_CONTENT, NO_PROGRESS, NEW_CONTENT, COMPLETE],
        )

    def test_consecutive_limits_are_unchanged_below_the_budget(self) -> None:
        same = emit("x")
        run = self.runtime(ScriptedBackend(*[tool_response(same)] * 6)).run_autonomous(
            "inspect it", finalize=False
        )

        self.assertTrue(run.stop_reason.startswith(STOP_NO_PROGRESS))
        # The consecutive no-progress bound still ends the run at the same
        # point; the repeated observations behind it are suppressed rather than
        # re-executed, so they cost model calls but not action budget.
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.suppressed_duplicates, 2)

    def test_the_report_runs_exactly_once_after_a_soft_budget_stop(self) -> None:
        same = emit("repeat")
        backend = ScriptedBackend(
            *self._productive(SOFT_MAX_AUTONOMOUS_ACTIONS),
            tool_response(same), tool_response(same),
            prose_response("REPORT"),
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(run.stop_reason, STOP_SOFT_MAX_ACTIONS)
        self.assertIsNotNone(run.final_report)
        self.assertEqual(run.final_report.text, "REPORT")
        self.assertEqual(backend.seen_tools.count([]), 1)

    def test_the_report_runs_exactly_once_after_the_hard_ceiling(self) -> None:
        backend = ScriptedBackend(
            *self._productive(MAX_AUTONOMOUS_ACTIONS), prose_response("REPORT")
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(run.stop_reason, STOP_MAX_ACTIONS)
        self.assertIsNotNone(run.final_report)
        self.assertEqual(backend.seen_tools.count([]), 1)

    def test_the_report_is_not_counted_as_an_action(self) -> None:
        backend = ScriptedBackend(
            *self._productive(3), prose_response("done"), prose_response("REPORT")
        )
        run = self.runtime(backend).run_autonomous("inspect it")

        self.assertEqual(run.actions_executed, 3)
        self.assertEqual(len(run.steps), 4, "the report is not a step")
