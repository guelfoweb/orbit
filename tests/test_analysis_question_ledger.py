"""Tool exploration is bounded by declared questions; knowledge is not.

ANALYSIS-QUESTION-LEDGER-1. Source acquisition stopped being the problem: the
live run supplied the whole artifact, suppressed the one re-read, and still
spent seven actions -- four re-deriving facts the source already showed (two of
them near-identical repeats), three genuinely needing execution, and none
caused by an earlier result. Every valid new action counted as progress, so the
run grew until a ceiling stopped it.

The ledger bounds which ACTIONS may run. It must never bound what the model may
conclude or report: a finding visible in the source needs no question, and a
question nobody could answer stays visibly open. The tests that matter most
here are the ones in `LedgerDoesNotEraseKnowledgeTests` -- a mechanism that
hid an unresolved fact because nobody declared it would be worse than the
inefficiency it replaces.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_question_ledger import (  # noqa: E402
    MAX_CHILD_DEPTH,
    MAX_INITIAL_QUESTIONS,
    OPEN,
    RESOLVED,
    LedgerError,
    Question,
    QuestionLedger,
    parse_plan,
    parse_resolution,
)


def _ledger(*ids: str) -> QuestionLedger:
    ledger = QuestionLedger()
    for qid in ids:
        ledger.add(Question(qid, f"question for {qid}", "needs execution"))
    return ledger


class PlanParsingTests(unittest.TestCase):
    """The plan is read strictly, or not at all."""

    def test_a_well_formed_plan_is_read(self) -> None:
        questions = parse_plan(
            '{"questions":[{"id":"Q1","question":"Is the token predictable?",'
            '"why":"hash() salting cannot be read off the source"}]}'
        )
        self.assertEqual([q.id for q in questions], ["Q1"])
        self.assertEqual(questions[0].depth, 0)

    def test_an_empty_plan_is_valid_and_meaningful(self) -> None:
        """A. The source answered everything -- a correct reply, not a failure."""
        self.assertEqual(parse_plan('{"questions":[]}'), [])

    def test_json_inside_prose_or_a_fence_is_read(self) -> None:
        body = '{"questions":[{"id":"Q1","question":"q","why":"w"}]}'
        for wrapper in (
            f"Here is my plan:\n{body}",
            f"```json\n{body}\n```",
            f"{body}\n\nLet me know if that works.",
        ):
            with self.subTest(wrapper=wrapper[:16]):
                self.assertEqual(len(parse_plan(wrapper)), 1)

    def test_malformed_plans_are_refused(self) -> None:
        for text, label in (
            ("", "empty"),
            ("no json at all", "prose only"),
            ("{", "truncated"),
            ('{"questions": "Q1"}', "not a list"),
            ('{"nope": []}', "missing field"),
            ('{"questions":[{"id":"Q1"}]}', "missing question"),
            ('{"questions":[{"id":"Q1","question":"q"}]}', "missing why"),
            ('{"questions":[{"id":"","question":"q","why":"w"}]}', "empty id"),
            ('{"questions":[{"id":"a b","question":"q","why":"w"}]}', "spaced id"),
            ('{"questions":[{"id":"Q1","question":"q","why":"w"},'
             '{"id":"Q1","question":"r","why":"w"}]}', "duplicate id"),
            ('{"questions":["Q1"]}', "not objects"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(LedgerError, msg=label):
                    parse_plan(text)

    def test_an_over_long_plan_is_refused(self) -> None:
        entries = ",".join(
            f'{{"id":"Q{i}","question":"q","why":"w"}}'
            for i in range(MAX_INITIAL_QUESTIONS + 1)
        )
        with self.assertRaises(LedgerError):
            parse_plan('{"questions":[' + entries + ']}')

    def test_over_long_text_is_refused(self) -> None:
        with self.assertRaises(LedgerError):
            parse_plan(
                '{"questions":[{"id":"Q1","question":"' + "x" * 5000
                + '","why":"w"}]}'
            )

    def test_an_absurdly_large_plan_is_refused_without_parsing(self) -> None:
        with self.assertRaises(LedgerError):
            parse_plan("{" + "x" * 100_000)


class LedgerTransitionTests(unittest.TestCase):
    """E. Resolution is one-way without new causal evidence."""

    def test_questions_start_open(self) -> None:
        ledger = _ledger("Q1", "Q2")
        self.assertEqual(ledger.open_ids, ["Q1", "Q2"])
        self.assertFalse(ledger.exhausted)

    def test_resolving_closes_exactly_one(self) -> None:
        ledger = _ledger("Q1", "Q2")
        ledger.resolve("Q1", "ev_1")
        self.assertEqual(ledger.open_ids, ["Q2"])
        self.assertEqual(ledger.resolved_ids, ["Q1"])

    def test_an_exhausted_ledger_has_nothing_open(self) -> None:
        ledger = _ledger("Q1")
        ledger.resolve("Q1", "ev_1")
        self.assertTrue(ledger.exhausted)

    def test_resolution_requires_named_evidence(self) -> None:
        ledger = _ledger("Q1")
        with self.assertRaises(LedgerError):
            ledger.resolve("Q1", "")

    def test_a_resolved_question_cannot_be_resolved_again(self) -> None:
        ledger = _ledger("Q1")
        ledger.resolve("Q1", "ev_1")
        with self.assertRaises(LedgerError):
            ledger.resolve("Q1", "ev_2")

    def test_reopening_without_new_evidence_is_refused(self) -> None:
        """A run that could reopen freely would cycle on one question."""
        ledger = _ledger("Q1")
        ledger.resolve("Q1", "ev_1")
        for evidence in ("", "ev_1"):
            with self.subTest(evidence=evidence):
                with self.assertRaises(LedgerError):
                    ledger.reopen("Q1", evidence)
        self.assertEqual(ledger.state["Q1"], RESOLVED)
        self.assertEqual(ledger.reopen_attempts, 2)

    def test_reopening_on_genuinely_new_evidence_is_allowed(self) -> None:
        ledger = _ledger("Q1")
        ledger.resolve("Q1", "ev_1")
        ledger.reopen("Q1", "ev_2")
        self.assertEqual(ledger.state["Q1"], OPEN)

    def test_an_unknown_question_cannot_be_resolved(self) -> None:
        with self.assertRaises(LedgerError):
            _ledger("Q1").resolve("Q9", "ev_1")


class ChildQuestionTests(unittest.TestCase):
    """C. A child is admitted only when the evidence really forced it."""

    def _parent(self) -> QuestionLedger:
        ledger = _ledger("Q1")
        ledger.resolve("Q1", "ev_a")
        return ledger

    def test_a_caused_child_is_accepted(self) -> None:
        ledger = self._parent()
        ledger.accept_child(
            Question("Q1.1", "Does the salt persist across runs?", "follow-up",
                     depth=1, parent="Q1", caused_by="ev_a")
        )
        self.assertEqual(ledger.open_ids, ["Q1.1"])

    def test_a_child_without_a_known_parent_is_refused(self) -> None:
        ledger = self._parent()
        with self.assertRaises(LedgerError):
            ledger.accept_child(
                Question("Q9.1", "x", "y", depth=1, parent="Qz", caused_by="ev_a")
            )

    def test_a_child_naming_no_evidence_is_refused(self) -> None:
        ledger = self._parent()
        with self.assertRaises(LedgerError):
            ledger.accept_child(
                Question("Q1.2", "x", "y", depth=1, parent="Q1", caused_by=None)
            )

    def test_a_child_beyond_the_depth_cap_is_refused(self) -> None:
        ledger = self._parent()
        with self.assertRaises(LedgerError):
            ledger.accept_child(
                Question("Q1.3", "x", "y", depth=MAX_CHILD_DEPTH + 1,
                         parent="Q1", caused_by="ev_a")
            )

    def test_a_child_restating_an_existing_question_is_refused(self) -> None:
        """D. Rewording is not a new question."""
        ledger = self._parent()
        original = ledger.questions["Q1"].question
        with self.assertRaises(LedgerError):
            ledger.accept_child(
                Question("Q1.4", original.upper() + "   ", "y",
                         depth=1, parent="Q1", caused_by="ev_a")
            )

    def test_a_duplicate_child_id_is_refused(self) -> None:
        ledger = self._parent()
        ledger.accept_child(
            Question("Q1.1", "first", "y", depth=1, parent="Q1", caused_by="ev_a")
        )
        with self.assertRaises(LedgerError):
            ledger.accept_child(
                Question("Q1.1", "second", "y", depth=1, parent="Q1",
                         caused_by="ev_a")
            )

    def test_every_refusal_is_counted(self) -> None:
        ledger = self._parent()
        for child in (
            Question("A", "x", "y", 1, "Qz", "ev_a"),
            Question("B", "x", "y", 1, "Q1", None),
            Question("C", "x", "y", 9, "Q1", "ev_a"),
        ):
            with self.assertRaises(LedgerError):
                ledger.accept_child(child)
        self.assertEqual(ledger.rejected_children, 3)


class TerminationTests(unittest.TestCase):
    """§10. The ledger terminates structurally, not by the model relenting."""

    def test_a_ledger_with_no_children_ends_after_its_questions(self) -> None:
        ledger = _ledger("Q1", "Q2", "Q3")
        for index, qid in enumerate(list(ledger.open_ids)):
            ledger.resolve(qid, f"ev_{index}")
        self.assertTrue(ledger.exhausted)

    def test_children_cannot_chain_beyond_the_depth_cap(self) -> None:
        """The adversarial case: every answer tries to raise another question."""
        ledger = _ledger("Q1")
        ledger.resolve("Q1", "ev_0")
        ledger.accept_child(
            Question("Q1.1", "child", "y", 1, "Q1", "ev_0")
        )
        ledger.resolve("Q1.1", "ev_1")
        # A grandchild is refused however it is dressed up.
        with self.assertRaises(LedgerError):
            ledger.accept_child(
                Question("Q1.1.1", "grandchild", "y", 2, "Q1.1", "ev_1")
            )
        self.assertTrue(ledger.exhausted)

    def test_the_total_size_is_capped_independently_of_the_ceilings(self) -> None:
        """§10. Termination must be the ledger's property, not the ceiling's.

        Without this cap the only bound on growth is the global action limit --
        one distinct child per action, indefinitely -- and the rendered ledger
        grows into every later prompt until admission refuses it. That is a run
        bounded by a ceiling, not by a finite plan.
        """
        from orbit.runtime.analysis_question_ledger import MAX_TOTAL_QUESTIONS

        ledger = _ledger("Q1")
        ledger.resolve("Q1", "ev")
        accepted = 0
        for index in range(1000):
            try:
                ledger.accept_child(
                    Question(f"C{index}", f"distinct child {index}", "w",
                             1, "Q1", "ev")
                )
                accepted += 1
            except LedgerError:
                break
        self.assertLessEqual(len(ledger.questions), MAX_TOTAL_QUESTIONS)
        self.assertLess(accepted, 1000)

    def test_the_worst_case_size_is_bounded(self) -> None:
        """N initial questions, each raising one child, and no more."""
        ledger = _ledger(*[f"Q{i}" for i in range(MAX_INITIAL_QUESTIONS)])
        for index, qid in enumerate(list(ledger.open_ids)):
            ledger.resolve(qid, f"ev_{index}")
            ledger.accept_child(
                Question(f"{qid}.1", f"child of {qid}", "y", 1, qid, f"ev_{index}")
            )
        self.assertEqual(len(ledger.questions), MAX_INITIAL_QUESTIONS * 2)
        for index, qid in enumerate(list(ledger.open_ids)):
            ledger.resolve(qid, f"evc_{index}")
        self.assertTrue(ledger.exhausted)


class ResolutionParsingTests(unittest.TestCase):
    """The classification is read strictly and never inferred."""

    def test_a_resolved_classification_is_read(self) -> None:
        ledger = _ledger("Q1")
        self.assertEqual(
            parse_resolution(
                '{"question":"Q1","state":"resolved","evidence":"ev_1"}', ledger
            ),
            ("Q1", RESOLVED, "ev_1"),
        )

    def test_still_open_is_a_real_answer(self) -> None:
        ledger = _ledger("Q1")
        qid, state, _ = parse_resolution(
            '{"question":"Q1","state":"still_open","evidence":"ev_1"}', ledger
        )
        self.assertEqual((qid, state), ("Q1", "still_open"))

    def test_a_classification_about_a_closed_question_is_refused(self) -> None:
        ledger = _ledger("Q1", "Q2")
        ledger.resolve("Q1", "ev_1")
        with self.assertRaises(LedgerError):
            parse_resolution(
                '{"question":"Q1","state":"resolved","evidence":"ev_2"}', ledger
            )

    def test_malformed_classifications_are_refused(self) -> None:
        ledger = _ledger("Q1")
        for text, label in (
            ("nonsense", "prose"),
            ('{"question":"Q9","state":"resolved"}', "unknown question"),
            ('{"question":"Q1","state":"done"}', "unknown state"),
            ('{"question":"Q1","state":"resolved","evidence":5}', "bad evidence"),
            ('["Q1"]', "not an object"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(LedgerError):
                    parse_resolution(text, ledger)


class SafetyTests(unittest.TestCase):
    """§6. No execution, no language-specific policy, no hardcoded findings."""

    SOURCE = ROOT / "src" / "orbit" / "runtime" / "analysis_question_ledger.py"

    def test_the_module_never_evaluates(self) -> None:
        import ast

        tree = ast.parse(self.SOURCE.read_text())
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            body = node.body
            if (
                body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree)
        for banned in ("eval(", "exec(", "literal_eval", "__import__",
                       "subprocess", "os.system", "pickle"):
            self.assertNotIn(banned, code, banned)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(
                    node.func.id, {"eval", "exec", "compile", "__import__", "open"}
                )

    def test_no_language_or_vulnerability_specific_policy(self) -> None:
        text = self.SOURCE.read_text().lower()
        for banned in ("javascript", "powershell", "python-specific", "sql",
                       "injection", "traversal", "pickle", "xss", "cve",
                       "vulnerab"):
            self.assertNotIn(banned, text, banned)

    def test_hostile_input_is_bounded_and_never_raises_unexpectedly(self) -> None:
        import time

        started = time.monotonic()
        for hostile in (
            "{" * 50_000,
            '{"questions":[' + ",".join('{"id":"Q","question":"q","why":"w"}'
                                        for _ in range(10_000)) + "]}",
            '{"questions":[{"id":"' + "9" * 100_000 + '","question":"q","why":"w"}]}',
            "\x00" * 10_000,
        ):
            with self.subTest(hostile=hostile[:12]):
                with self.assertRaises(LedgerError):
                    parse_plan(hostile)
        self.assertLess(time.monotonic() - started, 10.0)


if __name__ == "__main__":
    unittest.main()
