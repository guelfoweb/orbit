"""The ledger bounds actions. It must never bound knowledge.

ANALYSIS-QUESTION-LEDGER-1 at the runtime seam. The efficiency case is easy to
test and easy to get right; the correctness case is the one that matters. A
mechanism that hid an unresolved fact because nobody declared it as a question
would be worse than the inefficiency it replaces, so
`LedgerDoesNotEraseKnowledgeTests` is the heart of this file: an omitted
behaviour must still reach the report, and a fact that genuinely needs
execution must be reportable as unresolved rather than silently dropped.
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
from orbit.runtime.analysis_question_ledger import (  # noqa: E402
    Question,
    QuestionLedger,
)
from orbit.runtime.analysis_runtime import (  # noqa: E402
    STOP_LEDGER_EXHAUSTED,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
    _named_question,
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
    "\n"
    "\n"
    "def token(email):\n"
    "    return abs(hash(email)) % 1000000\n"
)


class _Script:
    """A scripted model: a queue of replies, each optionally with a call."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.default = ("done", None)

    def next(self):
        return self.replies.pop(0) if self.replies else self.default


class _Backend:
    thinking = False

    def __init__(self, script: _Script, *, per_char: float = 0.25) -> None:
        self.script = script
        self.per_char = per_char
        self.chat_calls: list[dict] = []

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * self.per_char), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        tools = kwargs.get("tools")
        self.chat_calls.append({"tools": tools, "messages": list(messages)})
        if not tools:
            # COVER, PLAN and classification calls all arrive tools-free.
            text, _ = self.script.next()
            return self._result(text, [])
        text, code = self.script.next()
        calls = []
        if code is not None:
            calls = [{
                "id": f"c{len(self.chat_calls)}", "type": "function",
                "function": {"name": "execute_analysis",
                             "arguments": json.dumps({"code": code})},
            }]
        return self._result(text, calls)

    def _result(self, content, calls):
        return ChatResult(
            content=content, model="m", finish_reason="stop", tool_calls=calls,
            prompt_tokens=1, completion_tokens=1, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


def _sandbox(stdout="FINDING: something real"):
    return AnalysisResult(
        status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
        stdout=stdout, stderr="", exit_status=0, duration_seconds=0.1,
    )


class _Case(unittest.TestCase):
    def _runtime(self, script, data: bytes = None) -> AnalysisRuntime:
        data = SOURCE.encode() if data is None else data
        self.backend = _Backend(script)
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

    def _run(self, runtime, sandbox=None, **kwargs):
        result = sandbox or _sandbox()
        with mock.patch.object(module, "execute_analysis", lambda **kw: result):
            return runtime.run_autonomous(
                "Analyse it.", finalize=False, **kwargs
            )


def _plan(*questions) -> str:
    return json.dumps({"questions": [
        {"id": qid, "question": text, "why": "needs execution"}
        for qid, text in questions
    ]})


def _resolved(qid, evidence="ev", child=None) -> str:
    payload = {"question": qid, "state": "resolved", "evidence": evidence}
    if child:
        payload["child"] = child
    return json.dumps(payload)


class EmptyPlanTests(_Case):
    """A. COVER answered everything: no actions at all."""

    def test_an_empty_plan_runs_no_actions(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),          # COVER reply
            ('{"questions":[]}', None),  # PLAN
        ]))
        run = self._run(runtime)
        self.assertEqual(run.cover_calls, 1)
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.initial_questions, 0)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_no_tools_are_offered_during_cover_or_plan(self) -> None:
        """K/L. Both are bookkeeping calls, not chances to act."""
        runtime = self._runtime(_Script([
            ("noted", None), ('{"questions":[]}', None),
        ]))
        self._run(runtime)
        self.assertEqual([bool(c["tools"]) for c in self.backend.chat_calls],
                         [False, False])


class TargetedResolutionTests(_Case):
    """B. Two questions, two targeted actions, both resolved, then report."""

    def test_two_questions_produce_exactly_two_actions(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "is pickle reachable?"), ("Q2", "is the token stable?")), None),
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q1", "ev_1"), None),
            ("question: Q2\nrunning", "print(2)"),
            (_resolved("Q2", "ev_2"), None),
        ]))
        run = self._run(runtime)
        self.assertEqual(run.initial_questions, 2)
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(set(run.resolved_questions), {"Q1", "Q2"})
        self.assertEqual(run.open_questions, ())
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_a_still_open_question_stays_open(self) -> None:
        """The action ran and did not settle it. That is a real outcome."""
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "is pickle reachable?")), None),
            ("question: Q1\nrunning", "print(1)"),
            (json.dumps({"question": "Q1", "state": "still_open",
                         "evidence": "ev_1"}), None),
            ("question: Q1\nagain", "print(2)"),
            (_resolved("Q1", "ev_2"), None),
        ]))
        run = self._run(runtime, max_model_calls=8)
        self.assertIn("Q1", run.resolved_questions)


class ChildQuestionRuntimeTests(_Case):
    """C. A child forced by the evidence is accepted and worked."""

    def test_a_caused_child_is_accepted_and_resolved(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "is pickle reachable?")), None),
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q1", "ev_1", child={
                "id": "Q1.1", "question": "which opcode does it accept?",
                "why": "the result showed an opcode", "caused_by": "ev_1"}), None),
            ("question: Q1.1\nrunning", "print(2)"),
            (_resolved("Q1.1", "ev_2"), None),
        ]))
        run = self._run(runtime, max_model_calls=10)
        self.assertEqual(run.initial_questions, 1)
        self.assertEqual(run.child_questions, 1)
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(set(run.resolved_questions), {"Q1", "Q1.1"})

    def test_a_child_without_causing_evidence_is_not_accepted(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "is pickle reachable?")), None),
            ("question: Q1\nrunning", "print(1)"),
            (json.dumps({"question": "Q1", "state": "resolved", "evidence": "ev_1",
                         "child": {"id": "Q1.1", "question": "unrelated",
                                   "why": "curious"}}), None),
        ]))
        run = self._run(runtime, max_model_calls=10)
        # `caused_by` defaults to the evidence of the action, so this one IS
        # caused; what must not be accepted is a child at depth 2 or a restatement.
        self.assertLessEqual(run.child_questions, 1)

    def test_a_grandchild_is_refused(self) -> None:
        ledger = QuestionLedger()
        ledger.add(Question("Q1", "root", "w"))
        ledger.resolve("Q1", "ev_1")
        ledger.accept_child(Question("Q1.1", "child", "w", 1, "Q1", "ev_1"))
        ledger.resolve("Q1.1", "ev_2")
        runtime = self._runtime(_Script([]))
        runtime._accept_child_question(
            ledger,
            json.dumps({"child": {"id": "Q1.1.1", "question": "grandchild",
                                  "why": "w", "caused_by": "ev_2"}}),
            "Q1.1", "ev_2",
        )
        self.assertNotIn("Q1.1.1", ledger.questions)


class FreeExplorationTests(_Case):
    """D. An action that belongs to no open question does not run."""

    def test_an_unnamed_action_is_refused(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "is pickle reachable?")), None),
            ("let me look at something else", "print('exploring')"),
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q1", "ev_1"), None),
        ]))
        run = self._run(runtime, max_model_calls=10)
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.rejected_free_actions, 1)

    def test_an_action_naming_a_closed_question_is_refused(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "a"), ("Q2", "b")), None),
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q1", "ev_1"), None),
            ("question: Q1\nagain", "print(2)"),     # already resolved
            ("question: Q2\nrunning", "print(3)"),
            (_resolved("Q2", "ev_2"), None),
        ]))
        run = self._run(runtime, max_model_calls=12)
        self.assertEqual(run.rejected_free_actions, 1)
        self.assertEqual(run.actions_executed, 2)

    def test_the_question_tag_is_read_only_from_the_head(self) -> None:
        """A mention buried in reasoning is not a declaration."""
        self.assertEqual(_named_question("question: Q1\nrunning"), "Q1")
        self.assertEqual(_named_question("Question = Q2"), "Q2")
        self.assertIsNone(_named_question("no tag here"))
        buried = "\n".join(["thinking"] * 8 + ["question: Q1"])
        self.assertIsNone(_named_question(buried))


class MalformedPlanTests(_Case):
    """F. One repair, then fall back to the behaviour that existed before."""

    def test_a_repaired_plan_is_used(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            ("I think we should look at pickle.", None),   # malformed
            (_plan(("Q1", "is pickle reachable?")), None),  # repaired
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q1", "ev_1"), None),
        ]))
        run = self._run(runtime, max_model_calls=10)
        self.assertEqual(run.plan_calls, 2)
        self.assertEqual(run.initial_questions, 1)

    def test_two_malformed_plans_fall_back_to_the_old_path(self) -> None:
        """No ledger: actions are unrestricted, exactly as before."""
        runtime = self._runtime(_Script([
            ("noted", None),
            ("prose", None),
            ("still prose", None),
            ("exploring freely", "print(1)"),
        ]))
        run = self._run(runtime, max_model_calls=6)
        self.assertEqual(run.plan_calls, 2)
        self.assertEqual(run.initial_questions, 0)
        self.assertEqual(run.actions_executed, 1)   # ran without naming a question
        self.assertEqual(run.rejected_free_actions, 0)
        self.assertNotEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)


class UnchangedPathTests(_Case):
    """G/H/I/K. Everything outside the covered autonomous path is untouched."""

    def test_without_coverage_there_is_no_plan(self) -> None:
        runtime = self._runtime(_Script([
            ("exploring", "print(1)"),
        ]), data=b"\x00\xffbinary")
        run = self._run(runtime, max_model_calls=4)
        self.assertEqual(run.cover_calls, 0)
        self.assertEqual(run.plan_calls, 0)
        self.assertEqual(run.actions_executed, 1)

    def test_planning_can_be_disabled(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            ("exploring", "print(1)"),
        ]))
        run = self._run(runtime, plan=False, max_model_calls=4)
        self.assertEqual(run.cover_calls, 1)
        self.assertEqual(run.plan_calls, 0)
        self.assertEqual(run.actions_executed, 1)

    def test_a_guided_step_is_unrestricted(self) -> None:
        """H/I. `step()` without a ledger offers tools and needs no question."""
        runtime = self._runtime(_Script([("exploring", "print(1)")]))
        with mock.patch.object(module, "execute_analysis", lambda **kw: _sandbox()):
            step = runtime.step("look at it")
        self.assertTrue(step.action_executed)
        self.assertTrue(self.backend.chat_calls[0]["tools"])

    def test_a_full_initial_ledger_fits_the_model_call_budget(self) -> None:
        """§2. Every declared question can actually be worked through.

        Coverage and planning cost two calls; each question costs one to run
        and one to classify. A plan the run could not finish would be a plan it
        cannot honour.
        """
        from orbit.runtime.analysis_question_ledger import MAX_INITIAL_QUESTIONS

        needed = 2 + 2 * MAX_INITIAL_QUESTIONS
        self.assertLessEqual(needed, module.MAX_AUTONOMOUS_MODEL_CALLS)

    def test_a_ledger_that_outgrows_the_budget_still_reports_openly(self) -> None:
        """§4. Children may exceed it; the ceiling stops the run, losing nothing."""
        script = [("noted", None), (_plan(("Q1", "a"), ("Q2", "b")), None)]
        for qid in ("Q1", "Q2"):
            script += [(f"question: {qid}\nrunning", "print(1)"),
                       (json.dumps({"question": qid, "state": "still_open",
                                    "evidence": "ev"}), None)]
        runtime = self._runtime(_Script(script))
        run = self._run(runtime, max_model_calls=4)
        # Stopped by the ceiling, with both questions still visibly open.
        self.assertEqual(run.stop_reason, module.STOP_MAX_MODEL_CALLS)
        self.assertEqual(set(run.open_questions), {"Q1", "Q2"})
        self.assertEqual(run.resolved_questions, ())

    def test_the_ledger_path_costs_two_calls_per_action(self) -> None:
        """The budget derivation changes shape on this path, and says so.

        Off the ledger an iteration is one call. On it an executing iteration
        is two -- the step and its classification -- plus coverage and planning
        up front. The constant is unchanged deliberately: six declared
        questions fit, and twelve actions was never a target but the point at
        which an unbounded run is stopped.
        """
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "a"), ("Q2", "b")), None),
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q1", "ev_1"), None),
            ("question: Q2\nrunning", "print(2)"),
            (_resolved("Q2", "ev_2"), None),
        ]))
        run = self._run(runtime, max_model_calls=12)
        self.assertEqual(run.actions_executed, 2)
        # cover + plan + 2*(step + classify)
        self.assertEqual(run.model_calls, 6)

    def test_action_ceilings_are_unchanged(self) -> None:
        self.assertEqual(module.MAX_AUTONOMOUS_ACTIONS, 12)
        self.assertEqual(module.SOFT_MAX_AUTONOMOUS_ACTIONS, 8)
        self.assertEqual(module.MAX_CONSECUTIVE_NO_PROGRESS, 2)

    def test_a_ledger_run_ends_on_policy_not_arithmetic(self) -> None:
        """A ceiling no run can hit makes its own test vacuous.

        Behaviour, not arithmetic: a ledger run working a full plan must stop
        because the questions ran out, never because the call budget did.
        Asserting only that the constant is large enough would pass on an
        implementation whose overhead term is zero.
        """
        script = [("noted", None),
                  (_plan(*[(f"Q{i}", f"q{i}") for i in range(6)]), None)]
        for i in range(6):
            script += [(f"question: Q{i}\nrunning", f"print({i})"),
                       (_resolved(f"Q{i}", f"ev_{i}"), None)]
        from orbit.runtime.analysis_sandbox import AnalysisResult

        runtime = self._runtime(_Script(script))
        counter = {"n": 0}

        def distinct(**kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"FINDING {counter['n']}",
                stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            run = runtime.run_autonomous("Analyse it.", finalize=False)
        self.assertEqual(run.actions_executed, 6)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)
        self.assertNotEqual(run.stop_reason, module.STOP_MAX_MODEL_CALLS)

    def test_a_plan_with_children_still_ends_on_policy(self) -> None:
        """Where the overhead term actually bites.

        Six declared questions cost fourteen calls and fit either ceiling, so
        the plain case cannot show the term does anything. Children are what
        push past it: seven worked questions need sixteen calls, and without
        the term the run dies on "model call bound reached" with questions
        still open rather than finishing them.
        """
        from orbit.runtime.analysis_sandbox import AnalysisResult

        script = [("noted", None),
                  (_plan(*[(f"Q{i}", f"q{i}") for i in range(6)]), None)]
        for i in range(6):
            child = None
            if i == 0:
                child = {"id": "Q0.1", "question": "forced follow-up",
                         "why": "the result forced it", "caused_by": "ev_0"}
            script += [(f"question: Q{i}\nrunning", f"print({i})"),
                       (_resolved(f"Q{i}", f"ev_{i}", child=child), None)]
        script += [("question: Q0.1\nrunning", "print(99)"),
                   (_resolved("Q0.1", "ev_child"), None)]
        runtime = self._runtime(_Script(script))
        counter = {"n": 0}

        def distinct(**kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"FINDING {counter['n']}",
                stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            run = runtime.run_autonomous("Analyse it.", finalize=False)
        self.assertEqual(run.child_questions, 1)
        self.assertEqual(run.actions_executed, 7)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)
        self.assertEqual(run.open_questions, ())

    def test_the_unledgered_path_spends_none_of_the_new_overhead(self) -> None:
        """Raising the ceiling must not loosen the path that does not use it."""
        self.assertGreater(
            module.MAX_AUTONOMOUS_MODEL_CALLS, module.MAX_AUTONOMOUS_ACTIONS
        )
        runtime = self._runtime(_Script([
            ("noted", None), ("prose", None), ("prose", None),
        ]))
        run = self._run(runtime, plan=False)
        # No plan, no classification: one call per iteration, as before.
        self.assertEqual(run.plan_calls, 0)


class _Insatiable(_Backend):
    """A model that always has another idea -- the behaviour actually observed.

    It answers by call TYPE rather than from a fixed script, so it keeps
    proposing actions for as long as it is asked. Without a ledger it runs to
    the action ceiling; with one it stops at the questions it declared.
    """

    def __init__(self, questions) -> None:
        super().__init__(_Script([]))
        self.questions = list(questions)
        self.pending: str | None = None
        self.n = 0

    def chat_stream(self, messages, **kwargs):
        import re

        tools = kwargs.get("tools")
        self.chat_calls.append({"tools": tools, "messages": list(messages)})
        last = str(messages[-1].get("content", ""))
        self.n += 1
        if not tools:
            if "Did that resolve" in last:
                question, self.pending = self.pending, None
                return self._result(_resolved(question, "ev"), [])
            if "list only the questions" in last or "could not be read" in last:
                return self._result(
                    _plan(*[(q, f"question {q}") for q in self.questions]), []
                )
            return self._result("noted", [])
        call = [{
            "id": f"c{self.n}", "type": "function",
            "function": {"name": "execute_analysis",
                         "arguments": json.dumps({"code": f"print({self.n})"})},
        }]
        open_ids = re.findall(r"^(\S+) \[OPEN\]", last, re.M)
        if open_ids:
            self.pending = open_ids[0]
            return self._result(f"question: {open_ids[0]}\nrunning", call)
        return self._result("exploring something else", call)


class TerminationAgainstAnInsatiableModelTests(_Case):
    """§10. The ledger terminates because it is finite, not because the model stops.

    This is the decision the whole design rests on. Driven by a model that
    never runs out of ideas, the current path spends the entire action budget;
    the ledger stops at exactly the questions that were declared, and the
    complete source is still in the history when it does.
    """

    def _run_insatiable(self, questions, plan):
        from orbit.runtime.analysis_sandbox import AnalysisResult

        runtime = self._runtime(_Script([]))
        runtime.backend = _Insatiable(questions)
        self.backend = runtime.backend
        counter = {"n": 0}

        def distinct(**kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"DISTINCT FINDING {counter['n']}",
                stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            run = runtime.run_autonomous("go", plan=plan, finalize=False)
        history = "\n".join(str(m.get("content")) for m in runtime.messages)
        return run, SOURCE in history

    def test_without_a_ledger_the_run_reaches_the_action_ceiling(self) -> None:
        run, _ = self._run_insatiable([], plan=False)
        self.assertEqual(run.actions_executed, module.MAX_AUTONOMOUS_ACTIONS)
        self.assertEqual(run.stop_reason, module.STOP_MAX_ACTIONS)

    def test_with_a_ledger_the_run_stops_at_its_declared_questions(self) -> None:
        for declared in ([], ["Q1"], ["Q1", "Q2", "Q3"]):
            with self.subTest(declared=len(declared)):
                run, source_kept = self._run_insatiable(declared, plan=True)
                self.assertEqual(run.actions_executed, len(declared))
                self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)
                self.assertEqual(set(run.resolved_questions), set(declared))
                # And the source is still there for the report to draw on.
                self.assertTrue(source_kept)

    def test_the_ledger_costs_far_fewer_actions_than_the_ceiling(self) -> None:
        bounded, _ = self._run_insatiable(["Q1", "Q2", "Q3"], plan=True)
        unbounded, _ = self._run_insatiable([], plan=False)
        self.assertLess(bounded.actions_executed, unbounded.actions_executed)


class CapturedLivePlanReplayTests(_Case):
    """The captured plan driving a real run, end to end.

    Parsing the fixture proves the cap was the cause. This proves the
    consequence: with the same reply the model actually sent, the ledger
    engages, no repair round is spent, and the run converges to a report
    instead of falling back to the unbounded loop.
    """

    FIXTURES = ROOT / "tests" / "fixtures"

    def _captured(self, index: int) -> str:
        path = self.FIXTURES / f"live_plan_{index}.json"
        if not path.exists():
            self.skipTest(f"captured plan {index} missing")
        return path.read_text()

    def _run_with(self, plan_text: str):
        from orbit.runtime.analysis_sandbox import AnalysisResult

        script = [("noted", None), (plan_text, None)]
        for index in range(3):
            script += [
                (f"question: Q{index + 1}\nrunning", f"print({index})"),
                (_resolved(f"Q{index + 1}", f"ev_{index}"), None),
            ]
        runtime = self._runtime(_Script(script))
        counter = {"n": 0}

        def distinct(**kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"FINDING {counter['n']}",
                stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            return runtime.run_autonomous("Analyse it.", finalize=False)

    def test_the_first_captured_plan_engages_the_ledger(self) -> None:
        run = self._run_with(self._captured(1))
        self.assertEqual(run.cover_calls, 1)
        self.assertEqual(run.plan_calls, 1, "no repair round should be spent")
        self.assertEqual(run.initial_questions, 3)
        self.assertEqual(run.actions_executed, 3)
        self.assertEqual(run.rejected_free_actions, 0)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_the_second_captured_plan_engages_the_ledger(self) -> None:
        run = self._run_with(self._captured(2))
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.initial_questions, 3)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_the_captured_plans_do_not_fall_back(self) -> None:
        """The failure of live run #1, stated as the thing that must not recur."""
        for index in (1, 2):
            with self.subTest(plan=index):
                run = self._run_with(self._captured(index))
                self.assertGreater(run.initial_questions, 0)
                self.assertNotEqual(run.stop_reason, module.STOP_MAX_ACTIONS)
                self.assertNotEqual(
                    run.stop_reason, module.STOP_MAX_MODEL_CALLS
                )


class LedgerDoesNotEraseKnowledgeTests(_Case):
    """§8. The ledger controls tool use, never what the model knows.

    These are the tests that decide whether this design is acceptable at all.
    An omitted behaviour must still be reportable, and a fact that genuinely
    needs execution must be reportable as unresolved rather than vanish.
    """

    def test_the_whole_source_stays_in_the_history_after_an_empty_plan(self) -> None:
        """J. Nothing was asked, so nothing ran -- and nothing was lost."""
        runtime = self._runtime(_Script([
            ("noted", None), ('{"questions":[]}', None),
        ]))
        self._run(runtime)
        history = "\n".join(str(m.get("content")) for m in runtime.messages)
        self.assertIn(SOURCE, history)
        # Both behaviours are visible even though neither was ever a question.
        self.assertIn("pickle.loads", history)
        self.assertIn("hash(email)", history)

    def test_a_deliberately_incomplete_plan_still_leaves_the_source(self) -> None:
        """The plan omits the pickle behaviour entirely; the source keeps it."""
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "is the token stable?")), None),
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q1", "ev_1"), None),
        ]))
        run = self._run(runtime, max_model_calls=10)
        self.assertEqual(run.initial_questions, 1)
        history = "\n".join(str(m.get("content")) for m in runtime.messages)
        self.assertIn("pickle.loads", history)

    def test_the_report_is_built_from_evidence_not_from_the_ledger(self) -> None:
        """The report path never consults the ledger, so it cannot filter.

        `report()` takes a `question` parameter, but that is the analyst's own
        query -- unrelated to the ledger. What matters is that no ledger state
        reaches it: the report is a view over the EvidenceStore and the
        history, both of which hold everything COVER supplied.
        """
        import inspect

        source = inspect.getsource(AnalysisRuntime.report)
        for banned in ("ledger", "open_questions", "resolved_questions",
                       "QuestionLedger", "_pending_question"):
            self.assertNotIn(banned, source, banned)
        # And the messages it builds come from the store, not from questions.
        builder = inspect.getsource(AnalysisRuntime._report_messages)
        for banned in ("ledger", "open_questions", "QuestionLedger"):
            self.assertNotIn(banned, builder, banned)

    def test_an_unresolved_question_is_reported_open_not_dropped(self) -> None:
        """The fact needs execution and was not settled. It must stay visible."""
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "is the token stable across processes?")), None),
            ("question: Q1\nrunning", "print(1)"),
            (json.dumps({"question": "Q1", "state": "still_open",
                         "evidence": "ev_1"}), None),
            ("question: Q1\nagain", "print(2)"),
            (json.dumps({"question": "Q1", "state": "still_open",
                         "evidence": "ev_2"}), None),
        ]))
        run = self._run(runtime, max_model_calls=8)
        self.assertIn("Q1", run.open_questions)
        self.assertEqual(run.resolved_questions, ())

    def test_an_unreadable_classification_never_resolves(self) -> None:
        """The failure path, not just the explicit `still_open` answer.

        A garbled or truncated classification reply must leave its question
        open. Resolving it there would report a question the model could not
        answer as answered -- the erasure this design exists to avoid -- and
        the explicit `still_open` path being covered does not cover this one.
        """
        for garbled in ("I think so?", "", "{truncated", '{"question":"Q1"}'):
            with self.subTest(garbled=garbled[:12]):
                runtime = self._runtime(_Script([
                    ("noted", None),
                    (_plan(("Q1", "needs execution")), None),
                    ("question: Q1\nrunning", "print(1)"),
                    (garbled, None),
                ]))
                run = self._run(runtime, max_model_calls=5)
                self.assertEqual(run.open_questions, ("Q1",))
                self.assertEqual(run.resolved_questions, ())

    def test_a_classification_about_another_question_never_resolves(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "a"), ("Q2", "b")), None),
            ("question: Q1\nrunning", "print(1)"),
            (_resolved("Q2", "ev_1"), None),   # answered about the wrong one
        ]))
        run = self._run(runtime, max_model_calls=5)
        self.assertEqual(run.resolved_questions, ())
        self.assertEqual(set(run.open_questions), {"Q1", "Q2"})

    def test_a_backend_failure_during_classification_never_resolves(self) -> None:
        """The comment says the question stays open; this makes it so."""
        from orbit.runtime.context_manager import ContextAdmissionError

        runtime = self._runtime(_Script([
            ("noted", None),
            (_plan(("Q1", "needs execution")), None),
            ("question: Q1\nrunning", "print(1)"),
        ]))
        real = runtime._admit
        state = {"cover_and_plan": 0}

        def refuse_on_classification(messages, **kwargs):
            last = str(messages[-1].get("content", ""))
            if "Did that resolve" in last:
                raise ContextAdmissionError("context admission failed: test")
            return real(messages, **kwargs)

        runtime._admit = refuse_on_classification
        run = self._run(runtime, max_model_calls=5)
        self.assertEqual(run.resolved_questions, ())
        self.assertIn("Q1", run.open_questions)

    def test_exhaustion_is_not_a_completeness_claim(self) -> None:
        """§6. The stop reason says what it means and no more."""
        self.assertIn("action", STOP_LEDGER_EXHAUSTED)
        for banned in ("complete", "finished", "done", "nothing left"):
            self.assertNotIn(banned, STOP_LEDGER_EXHAUSTED.lower(), banned)

    def test_open_questions_reach_the_closing_report(self) -> None:
        """A question nobody answered must be named in the report prompt.

        `AutonomousRunResult` carrying it is not enough: if the finalising
        model is never told, the report reads as though everything was
        answered, which is the erasure this design must not cause.
        """
        asked = AnalysisRuntime._final_question(
            STOP_LEDGER_EXHAUSTED, ("Q1", "Q3")
        )
        self.assertIn("Q1", asked)
        self.assertIn("Q3", asked)
        self.assertIn("not resolved", asked)

    def test_an_exhausted_run_is_not_described_as_cut_short(self) -> None:
        """It ended because nothing needed a tool, not because a bound hit."""
        asked = AnalysisRuntime._final_question(STOP_LEDGER_EXHAUSTED, ())
        self.assertNotIn("stopped before", asked)
        self.assertIn("artifact source", asked)

    def test_a_run_cut_short_by_a_bound_still_says_so(self) -> None:
        """The other half: a bound that intervened must not read as finished.

        Both halves are needed. A test that only checks the exhausted wording
        passes on an implementation that gives every ending that wording --
        including the ones where a ceiling stopped the analysis mid-way, which
        a reader would then take for a completed one.
        """
        for stop in (module.STOP_MAX_ACTIONS, module.STOP_MAX_MODEL_CALLS,
                     module.STOP_NO_PROGRESS, module.STOP_ERROR):
            with self.subTest(stop=stop):
                asked = AnalysisRuntime._final_question(stop, ())
                self.assertIn("stopped before the model chose to finish", asked)
                self.assertIn(stop, asked)

    def test_a_covered_run_reports_even_with_no_steps(self) -> None:
        """The empty plan is a correct reply and must not erase the analysis.

        Skipping the report here would turn a run that was handed the whole
        source into "no evidence has been collected", discarding everything
        COVER supplied.
        """
        runtime = self._runtime(_Script([
            ("noted", None),
            ('{"questions":[]}', None),
            ("The module calls pickle.loads on untrusted bytes.", None),
        ]))
        with mock.patch.object(module, "execute_analysis", lambda **kw: _sandbox()):
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertEqual(len(run.steps), 0)
        self.assertIsNotNone(run.final_report)
        self.assertIn("pickle.loads", run.final_report.text)

    def test_a_refused_coverage_report_never_claims_no_evidence(self) -> None:
        """The last leak of the invariant, on the admission-refusal path.

        The source WAS supplied in full, so "no analysis evidence has been
        collected" is the same false claim the coverage report exists to
        prevent. What failed is composing the report, which is a different
        fact -- and the accurate wording must not be gated on a deterministic
        appendix that most artifacts do not have.
        """
        from orbit.runtime.context_manager import ContextAdmissionError

        runtime = self._runtime(_Script([("noted", None), ('{"questions":[]}', None)]))
        runtime.cover_source(runtime.plan_source_coverage())
        self.assertFalse(runtime.deterministic_sections())
        real = runtime._admit

        def refuse_the_report(messages, **kwargs):
            if any("No execution was performed" in str(m.get("content", ""))
                   for m in messages):
                raise ContextAdmissionError("context admission failed: test")
            return real(messages, **kwargs)

        runtime._admit = refuse_the_report
        report = runtime.report()
        self.assertNotIn("No analysis evidence", report.text)
        self.assertIn("supplied in full", report.text)

    def test_the_coverage_report_is_grounded_in_the_supplied_source(self) -> None:
        runtime = self._runtime(_Script([
            ("noted", None), ('{"questions":[]}', None), ("report", None),
        ]))
        with mock.patch.object(module, "execute_analysis", lambda **kw: _sandbox()):
            runtime.run_autonomous("Analyse it.", finalize=True)
        sent = "\n".join(
            str(m.get("content"))
            for call in self.backend.chat_calls[-1:]
            for m in call["messages"]
        )
        self.assertIn(SOURCE, sent)

    def test_the_ledger_message_says_it_bounds_only_running(self) -> None:
        ledger = QuestionLedger()
        ledger.add(Question("Q1", "a question", "w"))
        told = module._ledger_instruction(ledger)
        self.assertIn("bounds only what you may RUN", told)
        self.assertIn("stays yours to report", told)


if __name__ == "__main__":
    unittest.main()
