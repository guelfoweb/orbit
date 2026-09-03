"""The controller driving real runs: association without prose.

The case that matters most is C -- a tool call with an empty assistant message.
That is what the previous control plane could not handle, and it is the normal
shape of a native call. Here it simply works, because the question was already
active when the call arrived.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_controller import (  # noqa: E402
    BLOCKED,
    MAX_ACTIONS_PER_QUESTION,
    RESOLVED,
)
from orbit.runtime.analysis_runtime import (  # noqa: E402
    ANALYSIS_TOOL_NAME,
    FINISH_TOOL_NAME,
    PLAN_TOOL_NAME,
    STOP_CONTROL_UNSUPPORTED,
    MAX_CONTROL_ERROR_CHARS,
    STOP_LEDGER_EXHAUSTED,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
from orbit.backend.llama_server import (  # noqa: E402
    LlamaServerToolCallParseError,
    LlamaServerBackend,
    LlamaServerError,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

CTX = 8192
SOURCE = (
    "import os\n"
    "import pickle\n"
    "\n"
    "\n"
    "def load(raw):\n"
    "    return pickle.loads(raw)\n"
)


class _Model:
    """A scripted model that answers by CALL TYPE, as a real one does.

    Deliberately emits **empty assistant content** with every tool call: that
    is the shape the previous control plane could not carry a question id
    through, and the shape this design has to work with.
    """

    def __init__(self, plan, decisions=None, code="print(1)") -> None:
        self.plan = plan
        self.decisions = list(decisions or [])
        self.code = code
        self.calls: list[dict] = []
        self.plans_submitted = 0
        self.actions = 0
        self.prose = ""

    def reply(self, tools, last: str):
        names = [t["function"]["name"] for t in (tools or [])]
        if PLAN_TOOL_NAME in names:
            self.plans_submitted += 1
            return self._call(PLAN_TOOL_NAME, {"questions": self.plan})
        if FINISH_TOOL_NAME in names:
            decision = (
                self.decisions.pop(0) if self.decisions
                else {"status": "resolved", "answer_summary": "done"}
            )
            return self._call(FINISH_TOOL_NAME, decision)
        if ANALYSIS_TOOL_NAME in names:
            # Distinct code each time, as a real model working different
            # questions would send. Identical code is suppressed by the
            # pre-existing duplicate-action guard, which would mask what these
            # tests are about.
            self.actions += 1
            return self._call(
                ANALYSIS_TOOL_NAME,
                {"code": f"# action {self.actions}\n{self.code}"},
            )
        return None

    def _call(self, name, arguments):
        return [{
            "id": f"c{len(self.calls)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]


class _Backend:
    thinking = False

    def __init__(self, model: _Model) -> None:
        self.model = model
        self.chat_calls: list[dict] = []

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * 0.25), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        tools = kwargs.get("tools")
        self.chat_calls.append(
            {"tools": tools, "messages": list(messages), "kwargs": dict(kwargs)}
        )
        last = str(messages[-1].get("content", ""))
        calls = self.model.reply(tools, last) or []
        self.model.calls.extend(calls)
        return ChatResult(
            # Empty content on purpose: the association must not need prose.
            content=self.model.prose, model="m", finish_reason="stop",
            tool_calls=calls, prompt_tokens=1, completion_tokens=1,
            cached_tokens=0, prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


def _question(text: str) -> dict:
    return {"question": text, "missing_fact": "needs execution"}


class _Case(unittest.TestCase):
    def _runtime(self, model: _Model, data: bytes = None) -> AnalysisRuntime:
        data = SOURCE.encode() if data is None else data
        self.backend = _Backend(model)
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / "artifact.py"
        path.write_bytes(data)
        runtime = AnalysisRuntime(
            backend=self.backend,
            source=AnalysisSource(
                snapshot_path=path, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(path),
            ),
            evidence_store=EvidenceStore(root=workspace.root / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        return runtime

    def _run(self, runtime, **kwargs):
        counter = {"n": 0}

        def distinct(**_kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"FINDING {counter['n']}",
                stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            return runtime.run_autonomous("Analyse it.", finalize=False, **kwargs)


class EmptyPlanTests(_Case):
    """A. No questions, no actions, straight to report."""

    def test_an_empty_plan_runs_nothing(self) -> None:
        runtime = self._runtime(_Model(plan=[]))
        run = self._run(runtime)
        self.assertEqual(run.cover_calls, 1)
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.initial_questions, 0)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)


class AssociationWithoutProseTests(_Case):
    """C/D. The invariant the previous design could not carry."""

    def test_a_tool_call_with_empty_prose_is_associated(self) -> None:
        model = _Model(plan=[_question("is pickle reachable?")])
        model.prose = ""          # exactly what killed the old protocol
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(model.prose, "")
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.resolved_questions, ("Q1",))

    def test_no_question_tag_is_required_anywhere(self) -> None:
        model = _Model(plan=[_question("a"), _question("b")])
        model.prose = "here is some unrelated commentary"
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(set(run.resolved_questions), {"Q1", "Q2"})

    def test_the_analysis_tool_schema_has_no_question_field(self) -> None:
        """Association is positional, so nothing was added to the call."""
        properties = (
            module.ANALYSIS_TOOL_SCHEMA["function"]["parameters"]["properties"]
        )
        self.assertEqual(set(properties), {"code"})


class SequencingTests(_Case):
    """B/E. One question at a time, in runtime order."""

    def test_three_questions_are_worked_one_at_a_time(self) -> None:
        model = _Model(plan=[_question("a"), _question("b"), _question("c")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(run.initial_questions, 3)
        self.assertEqual(run.actions_executed, 3)
        self.assertEqual(list(run.resolved_questions), ["Q1", "Q2", "Q3"])
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_runtime_assigns_the_ids(self) -> None:
        model = _Model(plan=[_question("a"), _question("b")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(list(run.resolved_questions), ["Q1", "Q2"])


class BudgetTests(_Case):
    """F/G. A question gets a second attempt, then is blocked."""

    def test_still_open_allows_another_action(self) -> None:
        model = _Model(
            plan=[_question("a")],
            decisions=[
                {"status": "still_open", "answer_summary": "not yet"},
                {"status": "resolved", "answer_summary": "now"},
            ],
        )
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(run.resolved_questions, ("Q1",))

    def test_the_limit_blocks_and_the_run_continues(self) -> None:
        model = _Model(
            plan=[_question("a"), _question("b")],
            decisions=[{"status": "still_open"}] * MAX_ACTIONS_PER_QUESTION
            + [{"status": "resolved", "answer_summary": "done"}],
        )
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertIn("Q1", run.open_questions)      # blocked, not resolved
        self.assertIn("Q2", run.resolved_questions)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)


class MalformedControlTests(_Case):
    """K/L/M. One repair, then a safe outcome -- never the old loop."""

    def test_a_model_that_never_plans_is_unsupported(self) -> None:
        class _Silent(_Model):
            def reply(self, tools, last):
                return None      # ordinary prose, no control call

        runtime = self._runtime(_Silent(plan=[]))
        run = self._run(runtime)
        self.assertEqual(run.stop_reason, STOP_CONTROL_UNSUPPORTED)
        self.assertEqual(run.actions_executed, 0)

    def test_a_malformed_plan_is_repaired_once_then_unsupported(self) -> None:
        class _Bad(_Model):
            def reply(self, tools, last):
                names = [t["function"]["name"] for t in (tools or [])]
                if PLAN_TOOL_NAME in names:
                    self.plans_submitted += 1
                    return self._call(PLAN_TOOL_NAME, {"questions": "not a list"})
                return None

        model = _Bad(plan=[])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(model.plans_submitted, 2)   # one repair, no more
        self.assertEqual(run.stop_reason, STOP_CONTROL_UNSUPPORTED)

    def test_a_malformed_completion_blocks_the_question(self) -> None:
        class _BadFinish(_Model):
            def reply(self, tools, last):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    return self._call(FINISH_TOOL_NAME, {"status": "nonsense"})
                return super().reply(tools, last)

        runtime = self._runtime(_BadFinish(plan=[_question("a")]))
        run = self._run(runtime)
        self.assertEqual(run.resolved_questions, ())
        self.assertIn("Q1", run.open_questions)

    def test_no_run_ever_falls_back_to_the_free_form_loop(self) -> None:
        """N/11. The load-bearing negative: no unbounded path exists."""
        source = (
            ROOT / "src" / "orbit" / "runtime" / "analysis_runtime.py"
        ).read_text()
        for gone in ("_named_question", "_QUESTION_TAG", "_ledger_instruction",
                     "_resolution_instruction", "plan_questions",
                     "_classify_question", "_pending_question"):
            self.assertNotIn(gone, source, gone)


class TransientHistoryTests(_Case):
    """N. Control bookkeeping must not accumulate permanently."""

    def test_history_grows_by_the_action_turn_only(self) -> None:
        """§8, measured: control exchanges cost the permanent record nothing.

        Each question spends a plan share, an action, and a completion call.
        Only the action turn is appended; the control prompts and replies are
        built transiently and discarded. So the growth per question is
        constant, and adding questions cannot make the history grow with
        everything the protocol needed to say.
        """
        sizes = {}
        for count in (1, 2, 3, 4):
            runtime = self._runtime(
                _Model(plan=[_question(f"q{i}") for i in range(count)])
            )
            run = self._run(runtime)
            self.assertEqual(run.actions_executed, count)
            sizes[count] = (
                len(runtime.messages),
                sum(len(str(m.get("content") or "")) for m in runtime.messages),
            )
        # Every increment is the same: one action turn, nothing else.
        deltas = {
            n: (sizes[n][0] - sizes[n - 1][0], sizes[n][1] - sizes[n - 1][1])
            for n in (2, 3, 4)
        }
        self.assertEqual(len(set(deltas.values())), 1, deltas)
        messages_added, _chars = next(iter(deltas.values()))
        # An action turn is the analyst line, the assistant reply and the tool
        # result. If control prompting were appended this would be larger.
        self.assertLessEqual(messages_added, 3)

    def test_the_permanent_history_holds_no_control_prompts(self) -> None:
        runtime = self._runtime(_Model(plan=[_question("a")]))
        self._run(runtime)
        history = "\n".join(str(m.get("content")) for m in runtime.messages)
        for marker in ("Work on this question", "call submit_analysis_plan",
                       "Call finish_analysis_question", "could not be used"):
            self.assertNotIn(marker, history, marker)


class ReportSeamTests(_Case):
    """P. What the closing report is told about questions nobody answered.

    The seam between the controller and the report was untested, and it hid a
    real defect: the prompt was given `open_ids`, which counts only questions
    still OPEN and excludes BLOCKED ones. A run where every question was
    blocked therefore produced a report that read as a completed analysis.

    Blocked is the common case, not an edge: it is what the action limit
    produces, what an unreadable completion produces, and what the model itself
    can report.
    """

    def _report_prompt(self, run, runtime) -> str:
        """The question the closing report was actually asked."""
        from orbit.runtime.analysis_controller import RESOLVED

        controller_order = getattr(self, "_last_controller", None)
        return module.AnalysisRuntime._final_question(
            run.stop_reason,
            tuple(run.open_questions),
            dossier=self._dossier,
        )

    def _run_and_capture(self, model, **kwargs):
        """Run, and keep the dossier the report would have been given."""
        runtime = self._runtime(model)
        real = module.AnalysisRuntime._final_question
        captured = {}

        def spy(stop_reason, open_questions=(), dossier=""):
            captured["prompt"] = real(stop_reason, open_questions, dossier)
            return captured["prompt"]

        with mock.patch.object(
            module.AnalysisRuntime, "_final_question", staticmethod(spy)
        ):
            counter = {"n": 0}

            def distinct(**_kwargs):
                counter["n"] += 1
                return AnalysisResult(
                    status="ok", code_sha256=f"{counter['n']:064d}",
                    input_sha256="i" * 64, stdout=f"FINDING {counter['n']}",
                    stderr="", exit_status=0, duration_seconds=0.1,
                )

            with mock.patch.object(module, "execute_analysis", distinct):
                run = runtime.run_autonomous(
                    "Analyse it.", finalize=True, **kwargs
                )
        return run, captured.get("prompt", "")

    def test_a_question_blocked_by_the_action_limit_is_named(self) -> None:
        model = _Model(
            plan=[_question("unanswerable")],
            decisions=[{"status": "still_open"}] * (MAX_ACTIONS_PER_QUESTION + 2),
        )
        _run, prompt = self._run_and_capture(model)
        self.assertIn("Q1", prompt)
        self.assertIn("unanswerable", prompt)
        self.assertIn("not resolved", prompt)

    def test_a_question_the_model_blocked_is_named(self) -> None:
        model = _Model(
            plan=[_question("cannot be settled")],
            decisions=[{"status": "blocked", "answer_summary": "no way to test"}],
        )
        _run, prompt = self._run_and_capture(model)
        self.assertIn("Q1", prompt)
        self.assertIn("cannot be settled", prompt)

    def test_a_question_blocked_by_an_unreadable_completion_is_named(self) -> None:
        class _BadFinish(_Model):
            def reply(self, tools, last):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    return self._call(FINISH_TOOL_NAME, {"status": "nonsense"})
                return super().reply(tools, last)

        _run, prompt = self._run_and_capture(
            _BadFinish(plan=[_question("never classified")])
        )
        self.assertIn("Q1", prompt)
        self.assertIn("never classified", prompt)

    def test_a_fully_resolved_run_names_nothing_unresolved(self) -> None:
        """The control: the clause appears only when something is unanswered."""
        _run, prompt = self._run_and_capture(_Model(plan=[_question("a")]))
        self.assertNotIn("not resolved", prompt)

    def test_a_controller_run_always_reaches_the_report(self) -> None:
        """A run that answered nothing must still produce a report."""
        model = _Model(
            plan=[_question("a")],
            decisions=[{"status": "blocked", "answer_summary": "no"}],
        )
        runtime = self._runtime(model)
        counter = {"n": 0}

        def distinct(**_kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"FINDING {counter['n']}",
                stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertIsNotNone(run.final_report)

    def test_open_questions_reports_everything_not_resolved(self) -> None:
        """The predicate the defect turned on, pinned directly."""
        model = _Model(
            plan=[_question("a"), _question("b")],
            decisions=[
                {"status": "resolved", "answer_summary": "done"},
                {"status": "blocked", "answer_summary": "no"},
            ],
        )
        run, _prompt = self._run_and_capture(model)
        self.assertEqual(run.resolved_questions, ("Q1",))
        self.assertEqual(run.open_questions, ("Q2",))


class PreservedBehaviourTests(_Case):
    """Q/R/S/K. Everything outside the controller is untouched."""

    def test_a_binary_artifact_still_runs(self) -> None:
        """Q. No COVER, so no plan -- and the run still works."""
        runtime = self._runtime(_Model(plan=[]), data=b"\x00\xffbinary")
        run = self._run(runtime, max_model_calls=3)
        self.assertEqual(run.cover_calls, 0)
        self.assertEqual(run.plan_calls, 0)

    def test_a_guided_step_needs_no_controller(self) -> None:
        """R. `step()` without a controller offers the analysis tool."""
        runtime = self._runtime(_Model(plan=[]))
        with mock.patch.object(
            module, "execute_analysis",
            lambda **kw: AnalysisResult(
                status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
                stdout="ok", stderr="", exit_status=0, duration_seconds=0.1,
            ),
        ):
            step = runtime.step("look at it")
        self.assertTrue(step.action_executed)
        names = [
            t["function"]["name"] for t in self.backend.chat_calls[0]["tools"]
        ]
        self.assertEqual(names, [ANALYSIS_TOOL_NAME])

    def test_global_ceilings_are_unchanged(self) -> None:
        self.assertEqual(module.MAX_AUTONOMOUS_ACTIONS, 12)
        self.assertEqual(module.SOFT_MAX_AUTONOMOUS_ACTIONS, 8)
        self.assertEqual(module.MAX_AUTONOMOUS_MODEL_CALLS, 18)
        self.assertEqual(module.MAX_CONSECUTIVE_NO_PROGRESS, 2)

    def test_planning_never_offers_the_analysis_tool(self) -> None:
        runtime = self._runtime(_Model(plan=[_question("a")]))
        run = self._run(runtime)
        planning = self.backend.chat_calls[
            run.cover_calls: run.cover_calls + run.plan_calls
        ]
        for call in planning:
            names = [t["function"]["name"] for t in (call["tools"] or [])]
            self.assertEqual(names, [PLAN_TOOL_NAME])


if __name__ == "__main__":
    unittest.main()


class BackendSignatureTests(_Case):
    """S. Control calls satisfy the real backend signature, not just the double.

    `_Backend.chat_stream` takes `**kwargs`, so a control call that omitted a
    required keyword-only argument still passed every scripted test here and
    then raised TypeError against the live backend -- which is exactly how the
    missing `on_delta` in `_control_call` reached a live run. These tests bind
    the arguments actually passed against `LlamaServerBackend.chat_stream`.
    """

    def test_every_call_binds_against_the_real_backend_signature(self) -> None:
        signature = inspect.signature(LlamaServerBackend.chat_stream)
        model = _Model(plan=[_question("a"), _question("b")])
        runtime = self._runtime(model)
        self._run(runtime)

        self.assertTrue(self.backend.chat_calls)
        for call in self.backend.chat_calls:
            # Raises TypeError if a required keyword-only argument is missing.
            signature.bind(runtime.backend, call["messages"], **call["kwargs"])

    def test_control_calls_pass_on_delta(self) -> None:
        model = _Model(plan=[_question("a")])
        runtime = self._runtime(model)
        self._run(runtime)
        control = [c for c in self.backend.chat_calls if c["tools"]]
        self.assertTrue(control)
        for call in control:
            self.assertIn("on_delta", call["kwargs"])

    def test_the_binding_check_would_catch_a_missing_on_delta(self) -> None:
        # The guard itself must fail when the keyword is absent, or it proves
        # nothing about the calls above.
        signature = inspect.signature(LlamaServerBackend.chat_stream)
        with self.assertRaises(TypeError):
            signature.bind(
                object(), [], temperature=0.0, max_tokens=16, tools=[],
                on_progress=None,
            )


class PlanStateTests(_Case):
    """T. PLAN_NOT_STARTED and a valid empty plan are different states.

    Proven necessary by a live Ornith run: the COVER reply carried malformed
    tool-call syntax, the backend raised `LlamaServerError`, and PLAN -- which
    shared COVER's `try` block -- was skipped with it. The run reached the
    loop with no controller, which looked exactly like a plan that had asked
    nothing, and reported as though the analysis were complete. Observed:
    `cover_calls=1`, `plan_calls=0`, `_control_call` never invoked,
    `stop_reason="no open question requires an action"`.
    """

    def _cover_then_raise(self, runtime):
        """COVER succeeds, then its model call raises -- the live shape.

        The source is appended and coverage is real; the failure lands after
        it. Nesting PLAN inside COVER's `try` loses planning to this.
        """
        real = type(runtime).cover_source

        def wrapper(self_, coverage, **kwargs):
            # The live shape: coverage completed -- the source was appended
            # and the call counted -- and the failure landed after it.
            calls = real(self_, coverage, **kwargs)
            self_.model_calls += 0
            wrapper.calls = calls
            raise LlamaServerError("Failed to parse input at pos 118")

        return mock.patch.object(type(runtime), "cover_source", wrapper)

    def test_a_failure_after_cover_still_lets_planning_run(self) -> None:
        model = _Model(plan=[_question("a")])
        runtime = self._runtime(model)
        with self._cover_then_raise(runtime):
            run = self._run(runtime)
        # The regression: PLAN must not be collateral damage of COVER's
        # failure domain. Either it planned, or it failed closed -- never a
        # silent unplanned run reported as a finished one.
        self.assertNotEqual(
            (run.plan_calls, run.stop_reason), (0, STOP_LEDGER_EXHAUSTED)
        )

    def test_a_valid_empty_plan_reports(self) -> None:
        runtime = self._runtime(_Model(plan=[]))
        run = self._run(runtime)
        # The model was asked and answered "nothing to investigate". That is
        # a real answer, and reporting is the correct transition.
        self.assertGreater(run.plan_calls, 0)
        self.assertEqual(run.initial_questions, 0)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_a_valid_plan_with_questions_activates_the_first(self) -> None:
        runtime = self._runtime(_Model(plan=[_question("a"), _question("b")]))
        run = self._run(runtime)
        self.assertGreater(run.plan_calls, 0)
        self.assertEqual(run.initial_questions, 2)
        self.assertEqual(list(run.resolved_questions), ["Q1", "Q2"])

    def test_zero_questions_is_only_ever_a_planned_answer(self) -> None:
        # `initial_questions == 0` may only be reached by asking. A run that
        # ends with no questions because PLAN never happened is the state this
        # whole distinction exists to prevent, so the assertion is on
        # `plan_calls`, not on the question count.
        asked = self._runtime(_Model(plan=[]))
        asked_run = self._run(asked)
        self.assertEqual(asked_run.initial_questions, 0)
        self.assertGreater(asked_run.plan_calls, 0)

        # Even with COVER failing, the run that had a question to ask asks it
        # -- planning is no longer collateral damage.
        after_failure = self._runtime(_Model(plan=[_question("a")]))
        with self._cover_then_raise(after_failure):
            failed_run = self._run(after_failure)
        self.assertGreater(failed_run.plan_calls, 0)
        self.assertEqual(failed_run.initial_questions, 1)

    def test_an_unplanned_covered_run_never_enters_the_free_form_loop(self) -> None:
        # No controller after a covered run means nothing bounds the loop.
        # That is the unbounded autonomy the controller replaced, so it must
        # fail closed rather than execute actions. `plan_analysis` is replaced
        # by one that returns without building a controller -- the exact
        # residue a skipped or half-run planning step leaves behind.
        runtime = self._runtime(_Model(plan=[_question("a")]))

        # A controller that never adopted a plan and never reported failure:
        # no questions, not unsupported. The loop must still refuse to run
        # free-form actions against it.
        def no_controller(self_, controller, message, **kwargs):
            return 0

        with mock.patch.object(type(runtime), "plan_analysis", no_controller):
            run = self._run(runtime)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_CONTROL_UNSUPPORTED)

    def test_a_backend_failure_during_planning_fails_closed(self) -> None:
        # PLAN raising is not a reason to run unplanned: the controller is
        # marked unsupported and the run reports, with no action executed.
        runtime = self._runtime(_Model(plan=[_question("a")]))

        def raising(self_, controller, message, **kwargs):
            raise LlamaServerError("planning call failed")

        with mock.patch.object(type(runtime), "plan_analysis", raising):
            run = self._run(runtime)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_CONTROL_UNSUPPORTED)


class PlanBackendErrorTests(_Case):
    """U. A backend error during PLAN is a planning failure, not a skipped plan.

    The live Ornith cause, reproduced without inference. The server raised

        LlamaServerError: Failed to parse input at pos 118: <tool_call>
        <function=execute_analysis> ...

    from `chat.cpp`'s tool-call grammar parser: PLAN offers only
    `submit_analysis_plan`, the model answered with an `execute_analysis` call
    in a format the grammar could not accept, and the native server reported a
    parse error rather than a message. The exception surfaced inside
    `_control_call`, on the first and only control call PLAN makes.

    What is pinned here is the runtime's response to it, not the model's
    output: `_control_call` is reached, the failure is attributed to planning,
    and the run fails closed instead of proceeding unplanned.
    """

    def _raising_backend(self, runtime, exc):
        real = type(runtime.backend).chat_stream

        def wrapper(self_, messages, **kwargs):
            names = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
            if PLAN_TOOL_NAME in names:
                raise exc
            return real(self_, messages, **kwargs)

        return mock.patch.object(type(runtime.backend), "chat_stream", wrapper)

    def test_the_plan_call_is_reached_before_the_failure(self) -> None:
        runtime = self._runtime(_Model(plan=[_question("a")]))
        seen: list[str] = []
        real_control = type(runtime)._control_call

        def counting(self_, messages, schema, **kwargs):
            seen.append(schema["function"]["name"])
            return real_control(self_, messages, schema, **kwargs)

        with mock.patch.object(type(runtime), "_control_call", counting):
            with self._raising_backend(
                runtime, LlamaServerError("Failed to parse input at pos 118")
            ):
                self._run(runtime)
        # PLAN is not skipped: it is attempted, once, and fails there.
        self.assertEqual(seen, [PLAN_TOOL_NAME])

    def test_a_parse_error_during_planning_fails_closed(self) -> None:
        runtime = self._runtime(_Model(plan=[_question("a")]))
        with self._raising_backend(
            runtime, LlamaServerError("Failed to parse input at pos 118")
        ):
            run = self._run(runtime)
        self.assertEqual(run.stop_reason, STOP_CONTROL_UNSUPPORTED)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.initial_questions, 0)

    def test_a_healthy_plan_call_is_made_exactly_once(self) -> None:
        # The invariant the failing runs are measured against:
        # PLAN_NOT_STARTED -> plan_analysis -> one control call -> plan_calls 1.
        runtime = self._runtime(_Model(plan=[_question("a")]))
        seen: list[str] = []
        real_control = type(runtime)._control_call

        def counting(self_, messages, schema, **kwargs):
            seen.append(schema["function"]["name"])
            return real_control(self_, messages, schema, **kwargs)

        with mock.patch.object(type(runtime), "_control_call", counting):
            run = self._run(runtime)
        self.assertEqual(seen.count(PLAN_TOOL_NAME), 1)
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.initial_questions, 1)


class _StrictBackend(_Backend):
    """A double that refuses what the real backend would refuse.

    `_Backend` takes `**kwargs`, so a call missing a required keyword-only
    argument passes here and fails only against a live server -- which is how
    a missing `on_delta` reached production. This one binds every call against
    `LlamaServerBackend.chat_stream` before answering it, and can be told to
    raise a chosen exception on the phase under test.
    """

    def __init__(self, model, failures=None) -> None:
        super().__init__(model)
        # One entry per *control* call, popped in order: an exception to
        # raise, or None to answer normally. Tools-free calls -- COVER and the
        # report -- are never affected, so a test can fail exactly the phase
        # it is about.
        self.failures = list(failures or [])
        self.attempts: list[str] = []

    def chat_stream(self, messages, **kwargs):
        inspect.signature(LlamaServerBackend.chat_stream).bind(
            self, messages, **kwargs
        )
        names = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
        control = [n for n in names if n in (PLAN_TOOL_NAME, FINISH_TOOL_NAME)]
        if control:
            self.attempts.append(control[0])
            if self.failures:
                failure = self.failures.pop(0)
                if failure is not None:
                    raise failure
        return super().chat_stream(messages, **kwargs)


class ControlRepairTests(_Case):
    """V. One bounded repair when the model's output could not be parsed.

    The live cause: PLAN offered only `submit_analysis_plan` while the system
    prompt still told the model to run `execute_analysis`. The model followed
    the prose, and the server's tool-call grammar refused to parse the reply
    into a message at all -- so there was no assistant message to reject
    structurally, only a `ToolCallParseError`.
    """

    def _strict(self, model, failures=None, data: bytes = None):
        data = SOURCE.encode() if data is None else data
        self.backend = _StrictBackend(model, failures)
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / "artifact.py"
        path.write_bytes(data)
        runtime = AnalysisRuntime(
            backend=self.backend,
            source=AnalysisSource(
                snapshot_path=path, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(path),
            ),
            evidence_store=EvidenceStore(root=workspace.root / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        return runtime

    # A. healthy PLAN: one attempt, one adopted plan, no repair.
    def test_a_healthy_plan_needs_no_repair(self) -> None:
        runtime = self._strict(_Model(plan=[_question("a")]))
        run = self._run(runtime)
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.control_repairs, 0)
        self.assertEqual(run.initial_questions, 1)
        self.assertGreater(run.control_attempts, 0)

    # B. first call parse-fails, the repair succeeds, the plan is adopted.
    def test_a_parse_failure_is_repaired_once(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[LlamaServerToolCallParseError("Failed to parse input at pos 118")],
        )
        run = self._run(runtime)
        self.assertEqual(run.control_repairs, 1)
        self.assertEqual(run.initial_questions, 1)
        self.assertEqual(list(run.resolved_questions), ["Q1"])

    # C. both the call and its repair parse-fail: unsupported, no fallback.
    def test_a_second_parse_failure_is_unsupported(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[
                LlamaServerToolCallParseError("Failed to parse input at pos 118"),
                LlamaServerToolCallParseError("Failed to parse input at pos 118"),
            ],
        )
        run = self._run(runtime)
        self.assertEqual(run.stop_reason, STOP_CONTROL_UNSUPPORTED)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.control_repairs, 1)

    # D. a generic backend error is not a format repair.
    def test_a_generic_backend_error_is_not_repaired(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[LlamaServerError("llama_decode failed")],
        )
        run = self._run(runtime)
        self.assertEqual(run.control_repairs, 0)
        self.assertEqual(run.stop_reason, STOP_CONTROL_UNSUPPORTED)
        self.assertEqual(run.actions_executed, 0)

    # E. the PLAN request carries no instruction to run an analysis action.
    def test_the_plan_request_offers_no_analysis_instruction(self) -> None:
        runtime = self._strict(_Model(plan=[_question("a")]))
        seen: list[list[dict]] = []
        real = type(runtime.backend).chat_stream

        def spy(self_, messages, **kwargs):
            names = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
            if PLAN_TOOL_NAME in names:
                seen.append([dict(m) for m in messages])
            return real(self_, messages, **kwargs)

        with mock.patch.object(type(runtime.backend), "chat_stream", spy):
            self._run(runtime)
        self.assertTrue(seen)
        for context in seen:
            body = "\n".join(str(m.get("content") or "") for m in context)
            self.assertNotIn(ANALYSIS_TOOL_NAME, body)

    # F. a valid control call with empty prose is still accepted.
    def test_empty_prose_with_a_valid_call_is_accepted(self) -> None:
        model = _Model(plan=[_question("a")])
        model.prose = ""
        runtime = self._strict(model)
        run = self._run(runtime)
        self.assertEqual(run.initial_questions, 1)
        self.assertEqual(list(run.resolved_questions), ["Q1"])

    # G. the same repair covers a question completion, not only PLAN.
    def test_a_completion_parse_failure_is_repaired_once(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            # PLAN answers normally; the failure lands on the finish call,
            # which is the other structured control phase.
            failures=[None, LlamaServerToolCallParseError("Failed to parse input at pos 4")],
        )
        run = self._run(runtime)
        self.assertEqual(run.control_repairs, 1)
        self.assertEqual(run.initial_questions, 1)

    # H. every dispatched call satisfies the real backend signature.
    def test_every_control_call_binds_against_the_real_signature(self) -> None:
        # `_StrictBackend` binds on every call, so reaching the end is the
        # assertion; this pins that control calls were actually made.
        runtime = self._strict(_Model(plan=[_question("a")]))
        run = self._run(runtime)
        self.assertIn(PLAN_TOOL_NAME, self.backend.attempts)
        self.assertGreater(run.control_attempts, 0)

    # I. a failed PLAN does not erase the coverage that already happened.
    def test_cover_is_still_recorded_when_planning_fails(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[
                LlamaServerToolCallParseError("Failed to parse input at pos 1"),
                LlamaServerToolCallParseError("Failed to parse input at pos 1"),
            ],
        )
        run = self._run(runtime)
        self.assertEqual(run.cover_calls, 1)
        self.assertTrue(runtime.source_covered)

    # J. an attempt that raises is still an attempt.
    def test_attempts_count_calls_that_raise(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[
                LlamaServerToolCallParseError("Failed to parse input at pos 1"),
                LlamaServerToolCallParseError("Failed to parse input at pos 1"),
            ],
        )
        run = self._run(runtime)
        # Both the first call and its repair raised before returning; both
        # reached the model and both must be visible.
        self.assertEqual(run.control_attempts, 2)
        self.assertEqual(run.plan_calls, 0)


class ControlRepairPromptTests(ControlRepairTests):
    """W. What the repair turn actually says, and what survives a failure."""

    def _repair_prompt(self, runtime) -> str:
        seen: list[str] = []
        real = type(runtime.backend).chat_stream

        def spy(self_, messages, **kwargs):
            names = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
            if any(n in (PLAN_TOOL_NAME, FINISH_TOOL_NAME) for n in names):
                seen.append(str(messages[-1].get("content") or ""))
            return real(self_, messages, **kwargs)

        with mock.patch.object(type(runtime.backend), "chat_stream", spy):
            self._run(runtime)
        return "\n".join(seen)

    def test_the_repair_names_the_allowed_tool(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[LlamaServerToolCallParseError("Failed to parse input at pos 118")],
        )
        prompt = self._repair_prompt(runtime)
        # The model has to be told which call is accepted, by name: the whole
        # failure was it reaching for a tool this phase does not offer.
        #
        # Asserted on the repair sentence itself, not the whole prompt: the
        # original planning instruction already names the tool, so a prompt
        # -wide check passes even when the repair says nothing useful.
        repair = [
            line for line in prompt.splitlines()
            if "This phase accepts only" in line
        ]
        self.assertTrue(repair, "the repair turn was not issued")
        self.assertIn(PLAN_TOOL_NAME, repair[0])

    def test_the_repair_never_offers_the_analysis_tool(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[LlamaServerToolCallParseError("Failed to parse input at pos 118")],
        )
        seen: list[list[str]] = []
        real = type(runtime.backend).chat_stream

        def spy(self_, messages, **kwargs):
            names = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
            # Control phases only: a RESOLVE step legitimately offers the
            # analysis tool, and that is not what this is about.
            if any(n in (PLAN_TOOL_NAME, FINISH_TOOL_NAME) for n in names):
                seen.append(names)
            return real(self_, messages, **kwargs)

        with mock.patch.object(type(runtime.backend), "chat_stream", spy):
            self._run(runtime)
        self.assertTrue(seen)
        for offered in seen:
            self.assertNotIn(ANALYSIS_TOOL_NAME, offered)
            self.assertEqual(len(offered), 1)

    def test_the_quoted_failure_is_bounded_and_single_line(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[
                LlamaServerToolCallParseError(
                    "Failed to parse input at pos 118: " + "<tool_call>\n" * 400
                )
            ],
        )
        prompt = self._repair_prompt(runtime)
        marker = "could not be parsed: "
        self.assertIn(marker, prompt)
        # Take everything after the marker up to the instruction that
        # follows, NOT up to the first newline -- splitting on the newline
        # would hide exactly the failure this is meant to catch.
        rest = prompt.split(marker, 1)[1]
        quoted = rest.split("This phase accepts only", 1)[0].rstrip("\n")
        # Untrusted model output, quoted back: bounded, and on one line so it
        # cannot forge the instruction that follows it.
        self.assertLessEqual(len(quoted), MAX_CONTROL_ERROR_CHARS + 3)
        self.assertNotIn("\n", quoted)

    def test_coverage_survives_a_cancelled_run(self) -> None:
        # `covered` is read from the history, so it reports what the model
        # actually holds -- not whether this run happened to finish.
        runtime = self._strict(_Model(plan=[_question("a")]))
        self._run(runtime)
        self.assertTrue(runtime.source_covered)
        # Still true on a second look: it is derived, not a one-shot flag.
        self.assertTrue(runtime.source_covered)


class ControlContextTests(unittest.TestCase):
    """X. The transient control context states one contract, whatever it wraps."""

    def _roles(self, messages):
        return [m["role"] for m in module._control_context(list(messages))]

    def test_the_system_turn_is_replaced(self) -> None:
        out = module._control_context(
            [{"role": "system", "content": "run execute_analysis"},
             {"role": "user", "content": "u"}]
        )
        self.assertIn("control turn", str(out[0]["content"]))
        self.assertNotIn("execute_analysis", str(out[0]["content"]))

    def test_a_context_without_a_system_turn_gains_one(self) -> None:
        out = module._control_context([{"role": "user", "content": "u"}])
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("control turn", str(out[0]["content"]))

    def test_every_system_turn_is_replaced_not_only_the_first(self) -> None:
        # Production appends exactly one, but a second would be another place
        # the analysis instruction could reappear.
        out = module._control_context(
            [{"role": "system", "content": "first"},
             {"role": "user", "content": "u"},
             {"role": "system", "content": "run execute_analysis"}]
        )
        systems = [m for m in out if m["role"] == "system"]
        self.assertEqual(len(systems), 1)
        self.assertNotIn("execute_analysis", str(systems[0]["content"]))

    def test_the_caller_list_is_not_mutated(self) -> None:
        original = [{"role": "system", "content": "S"},
                    {"role": "user", "content": "u"}]
        module._control_context(original)
        self.assertEqual(original[0]["content"], "S")


class ControlIsolationTests(ControlRepairTests):
    """Y. What a repair costs, and where it is not allowed to appear."""

    def test_the_repair_never_reaches_the_permanent_history(self) -> None:
        runtime = self._strict(
            _Model(plan=[_question("a")]),
            failures=[LlamaServerToolCallParseError("Failed to parse input at pos 118")],
        )
        run = self._run(runtime)
        self.assertEqual(run.control_repairs, 1)
        history = "\n".join(
            str(m.get("content") or "") for m in runtime.messages
        )
        # The repair is transient. Control bookkeeping in the append-only
        # record would make the permanent history grow with every prompt the
        # protocol needed.
        for marker in (
            "could not be parsed",
            "This phase accepts only",
            "This is a control turn",
            PLAN_TOOL_NAME,
            FINISH_TOOL_NAME,
        ):
            self.assertNotIn(marker, history)

    def test_a_repair_is_charged_against_the_model_call_ceiling(self) -> None:
        # The repair is a real model call, counted inside `_control_dispatch`.
        # If it were not, a phase that repairs could spend more of the budget
        # than the ceiling allows -- so the property to pin is the ceiling
        # itself, under repeated repairs, not a fixed call count.
        for cap in (2, 3, 5, 8):
            with self.subTest(cap=cap):
                runtime = self._strict(
                    _Model(plan=[_question("a"), _question("b")]),
                    failures=[
                        LlamaServerToolCallParseError("Failed to parse at pos 1"),
                        None,
                        None,
                        LlamaServerToolCallParseError("Failed to parse at pos 2"),
                        None,
                        None,
                    ],
                )
                counter = {"n": 0}

                def distinct(**_kwargs):
                    counter["n"] += 1
                    return AnalysisResult(
                        status="ok", code_sha256=f"{counter['n']:064d}",
                        input_sha256="i" * 64, stdout="x", stderr="",
                        exit_status=0, duration_seconds=0.1,
                    )

                with mock.patch.object(module, "execute_analysis", distinct):
                    run = runtime.run_autonomous(
                        "Analyse it.", finalize=False, max_model_calls=cap
                    )
                # The documented allowance is the ceiling plus one closing
                # report call. A repair must not buy a call beyond it.
                self.assertLessEqual(run.model_calls, cap + 1)
                # And every control call is charged: `model_calls` must
                # account for at least the control attempts this run made.
                # Without that, an uncounted control call is free budget --
                # the ceiling stops bounding the phase that repairs.
                self.assertGreaterEqual(run.model_calls, run.control_attempts)

    def test_the_counters_report_one_run_not_a_session(self) -> None:
        # The counters live on the runtime, which a REPL session reuses. Every
        # other field on the result is per-run; these must be too.
        runtime = self._strict(_Model(plan=[_question("a")]))
        first = self._run(runtime)
        second = self._run(runtime)
        self.assertEqual(second.control_attempts, first.control_attempts)
        self.assertEqual(second.control_repairs, first.control_repairs)

    def test_one_control_dispatch_charges_exactly_one_model_call(self) -> None:
        # Measured at the dispatch itself, not through a whole run: other
        # phases also charge `model_calls`, so a run-level inequality stays
        # true even when control calls stop being counted. An uncounted
        # control call is free budget, and the ceiling stops bounding the
        # phase that repairs.
        runtime = self._strict(_Model(plan=[_question("a")]))
        before_calls = runtime.model_calls
        before_attempts = runtime.control_attempts
        runtime._control_call(
            [{"role": "user", "content": "x"}], module.PLAN_TOOL_SCHEMA
        )
        self.assertEqual(runtime.model_calls - before_calls, 1)
        self.assertEqual(runtime.control_attempts - before_attempts, 1)

    def test_the_sanitiser_collapses_whitespace(self) -> None:
        text = module._sanitise_control_error(
            RuntimeError("line one\nline two\r\n\tline three")
        )
        self.assertNotIn("\n", text)
        self.assertEqual(text, "line one line two line three")


class _InterruptingModel(_Model):
    """Cancels the run at the completion call, not at the action.

    The interesting instant is the one *between* an action and its outcome:
    the execution has already run and its evidence is already durable, but
    nothing has yet recorded what the question made of it.
    """

    def __init__(self, plan, *, interrupt_on_finish: int = 1, **kwargs) -> None:
        super().__init__(plan, **kwargs)
        self.interrupt_on_finish = interrupt_on_finish
        self.finish_calls = 0

    def reply(self, tools, last: str):
        names = [t["function"]["name"] for t in (tools or [])]
        if FINISH_TOOL_NAME in names:
            self.finish_calls += 1
            if self.finish_calls == self.interrupt_on_finish:
                raise KeyboardInterrupt
        return super().reply(tools, last)


class CancellationDuringCompletionTests(_Case):
    """Cancelling between an action and its outcome must not escape the run.

    Every other model call in `run_autonomous` -- COVER, the step, the closing
    report -- catches `KeyboardInterrupt` and ends the run, because the caller
    holds only a pre-run checkpoint and rewinding to it would delete the
    history and provenance of every step that already succeeded. The
    completion call was the one omission from that policy.
    """

    def test_cancelling_the_completion_does_not_escape_the_run(self) -> None:
        model = _InterruptingModel(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertTrue(run.cancelled)
        self.assertEqual(run.stop_reason, module.STOP_CANCELLED)

    def test_the_interrupted_question_is_not_resolved(self) -> None:
        """Nothing recorded an outcome, so nothing may claim one."""
        model = _InterruptingModel(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(run.resolved_questions, ())
        self.assertEqual(run.open_questions, ("Q1",))

    def test_a_cancelled_run_writes_no_closing_report(self) -> None:
        """The analyst asked it to stop; spending minutes summarising is not stopping.

        This is pre-existing policy for every other cancellation path, and the
        containment here must not quietly buy an exception to it.
        """
        model = _InterruptingModel(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        with mock.patch.object(
            module, "execute_analysis",
            lambda **kw: AnalysisResult(
                status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
                stdout="FINDING", stderr="", exit_status=0,
                duration_seconds=0.1,
            ),
        ):
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertTrue(run.cancelled)
        self.assertIsNone(run.final_report)
        # Unresolved and reported as such -- the run does not go quiet about
        # the question it abandoned.
        self.assertEqual(run.open_questions, ("Q1",))
        self.assertEqual(run.resolved_questions, ())

    def test_the_runtime_blocks_the_question_with_the_exact_reason(self) -> None:
        """Production's own `exhaust_active` call, asserted on production's state.

        The controller is run-local, so it is captured as the runtime builds
        it. Asserting a reason the test itself wrote would prove nothing about
        the shipped path.
        """
        model = _InterruptingModel(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        built: list = []
        real = module.AnalysisController

        def capture(*args, **kwargs):
            controller = real(*args, **kwargs)
            built.append(controller)
            return controller

        with mock.patch.object(module, "AnalysisController", capture):
            run = self._run(runtime)

        self.assertTrue(run.cancelled)
        self.assertEqual(len(built), 1)
        state = built[0].states["Q1"]
        self.assertEqual(state.status, BLOCKED)
        # Compared against an independent literal, not against the constant
        # production wrote: reading the constant back proves only that the
        # assignment happened, never that the sentence says what it should.
        self.assertEqual(
            state.reason,
            "the analyst stopped the run before this question was closed",
        )
        self.assertIn(
            "the analyst stopped the run before this question was closed",
            built[0].dossier(),
        )
        self.assertIn("Q1 [BLOCKED]", built[0].dossier())

    def test_the_cancellation_reason_says_the_question_was_not_closed(self) -> None:
        """Pinned exactly, because word ORDER carries the meaning here.

        A vocabulary check cannot hold this alone. Reviewing one showed that
        reordering the shipped words into "this question was closed before the
        analyst stopped the run" passes every vocabulary test and tells the
        next analyst the question was settled -- the exact inversion the
        constraint exists to prevent. A set of words discards the only thing
        that distinguishes the two sentences.

        So the string is pinned. A change to it is then a deliberate act that
        must come here and be read as a sentence, which is the point: this
        text is rendered verbatim under `unresolved:` in the dossier and
        embedded in the guided follow-up.
        """
        self.assertEqual(
            module.CANCELLED_QUESTION_REASON,
            "the analyst stopped the run before this question was closed",
        )

    def test_the_backend_failure_reason_says_the_same_of_itself(self) -> None:
        """The cancellation's twin, which was previously unasserted.

        Mutating this string killed no test, while its cancellation twin had
        two. The asymmetry is what matters: both are rendered to the analyst
        in the same place, so both are pinned.
        """
        class _Boom(_Model):
            def __init__(self, plan, **kwargs) -> None:
                super().__init__(plan, **kwargs)
                self.finish_calls = 0

            def reply(self, tools, last: str):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    self.finish_calls += 1
                    raise module.RecoverableBackendError("upstream is down")
                return _Model.reply(self, tools, last)

        model = _Boom(plan=[_question("q")])
        runtime = self._runtime(model)
        built: list = []
        real = module.AnalysisController

        def capture(*args, **kwargs):
            controller = real(*args, **kwargs)
            built.append(controller)
            return controller

        # Production's own string, read off production's own state.
        with mock.patch.object(module, "AnalysisController", capture):
            self._run(runtime)
        controller = built[0]
        reason = controller.states["Q1"].reason
        self.assertEqual(
            reason,
            "the run stopped on a backend error before this question was "
            "closed",
        )
        # And it reaches the analyst, in the dossier the report is built from.
        self.assertIn(reason, controller.dossier())
        self.assertIn("Q1 [BLOCKED]", controller.dossier())

    def test_calls_spent_before_the_interrupt_are_still_counted(self) -> None:
        """A cancelled run must not be reported as cheaper than it was.

        `finish_question` reports its spend by returning it, and an interrupt
        leaves by a path that returns nothing -- so calls that reached the
        model vanished from the total the analyst reads. The case that shows
        it is an interrupt on the RETRY: a first completion attempt was
        already spent and repaired before the analyst stopped the run.
        """
        class _Retry(_InterruptingModel):
            def reply(self, tools, last: str):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    self.finish_calls += 1
                    if self.finish_calls == 1:
                        # Unusable: earns a repair, and costs a call.
                        return self._call(FINISH_TOOL_NAME, {"bogus": "x"})
                    raise KeyboardInterrupt
                return _Model.reply(self, tools, last)

        model = _Retry(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)

        self.assertTrue(run.cancelled)
        self.assertEqual(model.finish_calls, 2)
        # Counted against the backend, the only independent witness.
        self.assertEqual(run.model_calls, len(self.backend.chat_calls))

    def test_the_first_completion_call_is_counted_when_interrupted(self) -> None:
        """Counted against the BACKEND, not against another Orbit counter.

        `run.model_calls == runtime.model_calls` is a tautology: both are fed
        by the same increment, so both are wrong together whenever that
        increment is misplaced. The only independent witness of how many calls
        were made is the backend that served them.

        COVER is excluded from the comparison deliberately. It is an
        optimisation whose own handler swallows a `TimeoutError` and leaves
        the run uncovered, so under load it may not run at all -- an equality
        against the raw call total is then flaky, passing or failing on
        machine load rather than on the accounting this asserts.
        """
        model = _InterruptingModel(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        # The interrupted call reached the model and cost a turn.
        self.assertEqual(run.model_calls, len(self.backend.chat_calls))

    def test_a_backend_failure_during_completion_ends_the_run(self) -> None:
        """The other half of the same omission.

        Cancellation was not the only unguarded exit from the completion
        call. A backend that fails there unwound `run_autonomous` exactly as
        an interrupt did, and for the same cost: the caller holds only a
        pre-run checkpoint.
        """
        class _Boom(_Model):
            def __init__(self, plan, **kwargs) -> None:
                super().__init__(plan, **kwargs)
                self.finish_calls = 0

            def reply(self, tools, last: str):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    self.finish_calls += 1
                    raise module.RecoverableBackendError("upstream is down")
                return _Model.reply(self, tools, last)

        model = _Boom(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)

        # The run ended, rather than escaping to a caller that would rewind it.
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(len(run.steps), 1)
        # The analyst is told the cause, not left to infer one.
        self.assertIn("upstream is down", run.stop_reason)
        self.assertIn(module.STOP_BACKEND_ERROR, run.stop_reason)
        # An outage is not a decision: this is not a cancellation.
        self.assertFalse(run.cancelled)
        self.assertNotEqual(run.stop_reason, module.STOP_CANCELLED)
        # Nothing recorded an answer, so nothing claims one.
        self.assertEqual(run.resolved_questions, ())
        self.assertEqual(run.open_questions, ("Q1",))
        # And the spend is counted here too, against the backend.
        self.assertEqual(run.model_calls, len(self.backend.chat_calls))

    def test_a_backend_failure_on_the_retry_still_counts_the_first_call(self) -> None:
        """The discriminating case for the backend path's accounting.

        A failure on the FIRST completion call raises before the dispatch
        counter moves, so nothing is spent and any accounting looks correct.
        Only a failure on the retry -- after a call was spent and repaired --
        can show whether the spend survives the exception.
        """
        class _BoomOnRetry(_Model):
            def __init__(self, plan, **kwargs) -> None:
                super().__init__(plan, **kwargs)
                self.finish_calls = 0

            def reply(self, tools, last: str):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    self.finish_calls += 1
                    if self.finish_calls == 1:
                        return self._call(FINISH_TOOL_NAME, {"bogus": "x"})
                    raise module.RecoverableBackendError("upstream is down")
                return _Model.reply(self, tools, last)

        model = _BoomOnRetry(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)

        self.assertEqual(model.finish_calls, 2)
        self.assertIn("upstream is down", run.stop_reason)
        self.assertEqual(run.model_calls, len(self.backend.chat_calls))

    def test_the_last_action_is_still_rendered_to_the_analyst(self) -> None:
        """A step that ran must reach `on_step`, however the run then ends.

        The REPL renders autonomous steps ONLY through `on_step` -- it forces
        an empty step block for autonomous runs precisely because every step
        was already shown. Leaving the loop before classifying meant the last
        executed action was in `run.steps` but had never been rendered, and a
        cancelled run produces no closing report either, so its evidence id
        was displayed nowhere at all.
        """
        class _Late(_InterruptingModel):
            def reply(self, tools, last: str):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    self.finish_calls += 1
                    if self.finish_calls == 3:
                        raise KeyboardInterrupt
                return _Model.reply(self, tools, last)

        model = _Late(plan=[_question("a"), _question("b"), _question("c")])
        runtime = self._runtime(model)
        rendered: list = []
        counter = {"n": 0}

        def distinct(**_kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"F{counter['n']}", stderr="",
                exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            run = runtime.run_autonomous(
                "Analyse it.", finalize=False,
                on_step=lambda step, _record: rendered.append(step),
            )

        self.assertTrue(run.cancelled)
        # Every step that ran was rendered, and the ledger agrees with it.
        self.assertEqual(len(rendered), len(run.steps))
        self.assertEqual(len(run.progress), len(run.steps))
        # Specifically the last one, which is the one that used to vanish.
        self.assertIs(rendered[-1], run.steps[-1])
        self.assertIsNotNone(run.steps[-1].evidence)

    def test_a_backend_failure_also_renders_its_last_action(self) -> None:
        """Same invariant on the other new exit."""
        class _Boom(_Model):
            def __init__(self, plan, **kwargs) -> None:
                super().__init__(plan, **kwargs)
                self.finish_calls = 0

            def reply(self, tools, last: str):
                names = [t["function"]["name"] for t in (tools or [])]
                if FINISH_TOOL_NAME in names:
                    self.finish_calls += 1
                    raise module.RecoverableBackendError("upstream is down")
                return _Model.reply(self, tools, last)

        model = _Boom(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        rendered: list = []
        run = self._run(runtime, on_step=lambda s, _r: rendered.append(s))

        self.assertEqual(len(rendered), len(run.steps))
        self.assertEqual(len(run.progress), len(run.steps))
        self.assertIn("upstream is down", run.stop_reason)

    def test_every_failure_path_counts_the_calls_it_spent(self) -> None:
        """One table, because the defect was the same at four call sites.

        COVER, PLAN, the step and the completion each report their spend by
        RETURNING it, and each has handlers that leave without returning. A
        call that reached the model and then failed was therefore spent and
        never counted, and the analyst reads that total. The witness is the
        backend, the only counter not fed by the code under test.
        """
        def failing(tool, exc):
            class _M(_Model):
                def __init__(self, plan, **kwargs) -> None:
                    super().__init__(plan, **kwargs)
                    self.hits = 0

                def reply(self, tools, last: str):
                    names = [t["function"]["name"] for t in (tools or [])]
                    # COVER is the call that offers no tools at all.
                    if (tool is None and not names) or (tool and tool in names):
                        self.hits += 1
                        if self.hits == 1:
                            raise exc
                    return _Model.reply(self, tools, last)
            return _M

        interrupt = KeyboardInterrupt
        outage = module.RecoverableBackendError
        # The success path first: a run where nothing fails must also count
        # exactly, or the failure rows below prove only that two wrongs match.
        model = _Model(plan=[_question("q")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(run.model_calls, len(self.backend.chat_calls),
                         "a run with no failure must count exactly")

        # And the boundary the counter must sit on, at EVERY site: admission
        # refuses before anything is sent, so a refused request reached no
        # model and must not be billed. Patching `_admit` globally only ever
        # exercises the first site a run reaches, which is why each is
        # refused in turn here by the phase it belongs to.
        real_admit = module.AnalysisRuntime._admit
        # Through 6, not 4: with `finalize=True` the run reaches a report
        # dispatch too, and its admission boundary is otherwise never
        # exercised -- a context exhausted at the closing report is exactly
        # when that path runs in production.
        for nth in range(1, 7):
            with self.subTest(refuse_admission_number=nth):
                model = _Model(plan=[_question("q")])
                runtime = self._runtime(model)
                seen = {"n": 0}

                def refusing(self, *args, **kwargs):
                    seen["n"] += 1
                    if seen["n"] == nth:
                        raise module.ContextAdmissionError("nothing fits")
                    return real_admit(self, *args, **kwargs)

                with mock.patch.object(
                    module.AnalysisRuntime, "_admit", refusing
                ):
                    run = runtime.run_autonomous("Analyse it.", finalize=True)
                # The refused admission sent nothing, so it is not a call --
                # every other admitted request is.
                self.assertEqual(
                    run.model_calls, len(self.backend.chat_calls),
                    f"refusal #{nth} was billed as a call",
                )

        sites = [
            ("COVER", None), ("PLAN", PLAN_TOOL_NAME),
            ("STEP", ANALYSIS_TOOL_NAME), ("FINISH", FINISH_TOOL_NAME),
        ]
        for label, tool in sites:
            for kind, exc in (("interrupt", interrupt("stop")),
                              ("outage", outage("upstream is down"))):
                with self.subTest(site=label, failure=kind):
                    model = failing(tool, exc)(plan=[_question("q")])
                    runtime = self._runtime(model)
                    run = self._run(runtime)
                    self.assertEqual(
                        run.model_calls, len(self.backend.chat_calls),
                        f"{label} on {kind}: calls spent were not counted",
                    )

    def test_cancelling_spends_no_shadow_observation(self) -> None:
        """The cancellation flag, tested where it actually bites.

        `while not cancelled` only ends the loop at the NEXT iteration, so
        without the flag the rest of this one still runs. When the completion
        shadow is due that costs a real generation -- spent summarising a run
        the analyst just asked to stop.

        The schedule fires after four actions, so the interrupt must land ON
        the fourth completion: a run that stops earlier never reaches the
        checkpoint and the assertion would hold for the wrong reason.
        """
        with mock.patch.dict(
            os.environ, {"ORBIT_ANALYSIS_COMPLETION_SHADOW": "1"}, clear=False
        ):
            model = _InterruptingModel(
                plan=[_question(str(i)) for i in range(4)],
                interrupt_on_finish=4,
            )
            runtime = self._runtime(model)
            counter = {"n": 0}

            def distinct(**_kwargs):
                counter["n"] += 1
                return AnalysisResult(
                    status="ok", code_sha256=f"{counter['n']:064d}",
                    input_sha256="i" * 64, stdout=f"F{counter['n']}",
                    stderr="", exit_status=0, duration_seconds=0.1,
                )

            with mock.patch.object(module, "execute_analysis", distinct):
                run = runtime.run_autonomous("Analyse it.", finalize=False)

        self.assertTrue(run.cancelled)
        self.assertEqual(run.actions_executed, 4, "the schedule must be reached")
        self.assertEqual(
            len(run.completion_shadow.observations), 0,
            "a cancelled run must not spend a generation on a diagnostic",
        )

    def test_a_closing_report_is_counted_like_every_other_call(self) -> None:
        """The fifth dispatch site. It was the one left unconverted.

        `report()` reached the model without touching the counter at all, so
        a report that died mid-generation was spent and never counted, and
        even a successful one left the runtime's lifetime total short.
        """
        model = _Model(plan=[_question("q")])
        runtime = self._runtime(model)
        reached = {"n": 0}
        served = self.backend.chat_stream

        def counting(messages, **kwargs):
            # Counted before dispatch, as production does, so a call that
            # raises is still witnessed.
            reached["n"] += 1
            return served(messages, **kwargs)

        self.backend.chat_stream = counting
        with mock.patch.object(
            module, "execute_analysis",
            lambda **kw: AnalysisResult(
                status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
                stdout="F", stderr="", exit_status=0, duration_seconds=0.1,
            ),
        ):
            run = runtime.run_autonomous("Analyse it.", finalize=True)

        self.assertIsNotNone(run.final_report)
        self.assertEqual(run.model_calls, reached["n"])
        # And the lifetime counter agrees, which it did not before.
        self.assertEqual(runtime.model_calls, reached["n"])

    def test_a_coverage_report_is_counted_too(self) -> None:
        """The other report path -- the one with no evidence to report on.

        A covered run whose plan was empty takes no step at all: the model
        said the source answered everything, which the planning instruction
        calls a correct reply. That run still reports, through a second
        dispatch site, and that site was the one left uncounted -- so an
        ordinary successful analysis under-reported what it cost.
        """
        model = _Model(plan=[])
        runtime = self._runtime(model)
        run = runtime.run_autonomous("Analyse it.", finalize=True)

        # No steps, but a real report, served by a real call.
        self.assertEqual(run.actions_executed, 0)
        self.assertIsNotNone(run.final_report)
        self.assertEqual(run.model_calls, len(self.backend.chat_calls))
        self.assertEqual(runtime.model_calls, len(self.backend.chat_calls))

    def test_a_report_that_fails_still_counts_the_call_it_made(self) -> None:
        """The report's failure path, which returns nothing to add.

        `report()` reports its spend in the object it returns; the handler
        that catches a failing report returns None instead, so the call it
        already made vanished from the total the analyst reads.
        """
        model = _Model(plan=[_question("q")])
        runtime = self._runtime(model)
        reached = {"n": 0}
        served = self.backend.chat_stream

        def failing(messages, **kwargs):
            reached["n"] += 1
            # The first toolless call is COVER; the second is the report.
            if not kwargs.get("tools") and reached["n"] > 1:
                raise module.RecoverableBackendError("upstream is down")
            return served(messages, **kwargs)

        self.backend.chat_stream = failing
        with mock.patch.object(
            module, "execute_analysis",
            lambda **kw: AnalysisResult(
                status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
                stdout="F", stderr="", exit_status=0, duration_seconds=0.1,
            ),
        ):
            run = runtime.run_autonomous("Analyse it.", finalize=True)

        # The run survives a report it could not compose...
        self.assertIsNone(run.final_report)
        # ...and still says what it cost.
        self.assertEqual(run.model_calls, reached["n"])

    def test_the_action_and_its_evidence_survive(self) -> None:
        """What ran, ran. The interrupt withholds an outcome, not a result."""
        model = _InterruptingModel(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(len(run.steps), 1)
        self.assertIsNotNone(run.steps[0].evidence)

    def test_cancelling_stops_the_run_rather_than_continuing(self) -> None:
        """Leaving the loop is load-bearing -- but not because of Q2.

        `while not cancelled` already stops the next iteration, so Q2 never
        starts either way. What leaving actually protects is the REST of this
        iteration: a COMPLETE classification below would overwrite the stop
        reason that says the analyst cancelled, and the completion shadow
        would spend a model call on a run that is already over.
        """
        model = _InterruptingModel(plan=[_question("first"), _question("second")])
        runtime = self._runtime(model)
        built: list = []
        real = module.AnalysisController

        def capture(*args, **kwargs):
            controller = real(*args, **kwargs)
            built.append(controller)
            return controller

        with mock.patch.object(module, "AnalysisController", capture):
            run = self._run(runtime)

        self.assertTrue(run.cancelled)
        self.assertEqual(run.stop_reason, module.STOP_CANCELLED)
        # Exactly one action ran and exactly one completion was attempted.
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(model.finish_calls, 1)
        self.assertEqual(len(run.steps), 1)
        # Q2 was never activated, never acted on, and is still open.
        self.assertEqual(built[0].states["Q2"].actions, 0)
        self.assertEqual(run.resolved_questions, ())
        self.assertEqual(run.open_questions, ("Q1", "Q2"))
        # The step that ran IS classified and rendered -- it is real work the
        # analyst must see -- but nothing after it runs, so the cancellation
        # survives as the stop reason.
        self.assertEqual(len(run.progress), len(run.steps))
        self.assertEqual(run.stop_reason, module.STOP_CANCELLED)

    def test_normal_completion_is_untouched(self) -> None:
        """The guard catches an interrupt; it changes nothing otherwise."""
        model = _Model(plan=[_question("is pickle reachable?")])
        runtime = self._runtime(model)
        run = self._run(runtime)
        self.assertFalse(run.cancelled)
        self.assertEqual(run.resolved_questions, ("Q1",))
        self.assertEqual(run.open_questions, ())
