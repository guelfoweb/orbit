"""The controller driving real runs: association without prose.

The case that matters most is C -- a tool call with an empty assistant message.
That is what the previous control plane could not handle, and it is the normal
shape of a native call. Here it simply works, because the question was already
active when the call arrived.
"""
from __future__ import annotations

import hashlib
import json
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
    STOP_LEDGER_EXHAUSTED,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
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
        self.chat_calls.append({"tools": tools, "messages": list(messages)})
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
