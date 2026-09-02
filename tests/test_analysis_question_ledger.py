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


class CapturedLivePlanTests(unittest.TestCase):
    """The two plans Ornith actually produced, replayed byte for byte.

    Live run #1 never exercised the ledger. Both replies were syntactically
    valid and each carried three useful questions; both were refused because
    one shared 300-character cap covered the `why` field, and the observed
    `why` fields ran 295 to 357. The run fell back to the unbounded path
    exactly as designed -- nothing broke -- but the hypothesis went untested.

    These fixtures are the captured replies verbatim, so a future change to the
    bounds is measured against what the model really writes rather than against
    what a test author imagines it writes.
    """

    FIXTURES = ROOT / "tests" / "fixtures"

    def _plan(self, index: int) -> str:
        path = self.FIXTURES / f"live_plan_{index}.json"
        if not path.exists():
            self.skipTest(f"captured plan {index} missing")
        return path.read_text()

    def test_the_first_captured_plan_parses_into_three_questions(self) -> None:
        questions = parse_plan(self._plan(1))
        self.assertEqual([q.id for q in questions], ["Q1", "Q2", "Q3"])

    def test_the_second_captured_plan_parses_into_three_questions(self) -> None:
        questions = parse_plan(self._plan(2))
        self.assertEqual([q.id for q in questions], ["Q1", "Q2", "Q3"])

    def test_the_captured_text_is_preserved_exactly(self) -> None:
        """No truncation, no normalisation -- what was written is what is kept."""
        import json as _json

        for index in (1, 2):
            with self.subTest(plan=index):
                raw = self._plan(index)
                payload = _json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
                parsed = parse_plan(raw)
                for original, question in zip(payload["questions"], parsed):
                    self.assertEqual(question.id, original["id"])
                    self.assertEqual(question.question, original["question"].strip())
                    self.assertEqual(question.why, original["why"].strip())

    def test_the_captured_plans_need_no_repair(self) -> None:
        """Each parses on the first attempt, so no repair round is spent."""
        for index in (1, 2):
            with self.subTest(plan=index):
                parse_plan(self._plan(index))  # would raise if a repair were needed

    def test_each_captured_plan_still_exceeds_the_old_cap(self) -> None:
        """Pins the measurement the fix rests on, per fixture.

        Pooling both fixtures into one maximum let either be shortened
        unnoticed -- the other's length still satisfied the assertion. Each is
        now checked in its own right, so a fixture that stops exercising the
        old bound fails loudly instead of quietly testing nothing.
        """
        import json as _json

        for index in (1, 2):
            with self.subTest(plan=index):
                raw = self._plan(index)
                payload = _json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
                whys = [len(q["why"]) for q in payload["questions"]]
                questions = [len(q["question"]) for q in payload["questions"]]
                self.assertGreater(
                    max(whys), 300,
                    f"fixture {index} no longer exceeds the old 300 cap",
                )
                self.assertLessEqual(max(whys), 512)
                # And every question stayed inside the unchanged bound, which
                # is why only the `why` cap needed to move.
                self.assertLess(max(questions), 300)

    def test_every_captured_why_would_have_failed_the_old_cap(self) -> None:
        """Not just the longest: the plans were unusable, not marginal."""
        import json as _json

        over = 0
        for index in (1, 2):
            raw = self._plan(index)
            payload = _json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            over += sum(1 for q in payload["questions"] if len(q["why"]) > 300)
        self.assertGreaterEqual(over, 4, "most captured whys must exceed 300")

class TextBoundaryTests(unittest.TestCase):
    """The bounds, probed at the boundary with literal numbers.

    Deliberately not written in terms of `MAX_WHY_CHARS`: a test that reuses
    the production constant passes whatever that constant becomes, which is
    exactly how a 300-character cap survived until a live run hit it.
    """

    def _plan_with(self, question: str, why: str) -> str:
        import json as _json

        return _json.dumps(
            {"questions": [{"id": "Q1", "question": question, "why": why}]}
        )

    def test_a_why_of_511_characters_is_accepted(self) -> None:
        parse_plan(self._plan_with("q", "w" * 511))

    def test_a_why_of_512_characters_is_accepted(self) -> None:
        parse_plan(self._plan_with("q", "w" * 512))

    def test_a_why_of_513_characters_is_refused(self) -> None:
        with self.assertRaises(LedgerError):
            parse_plan(self._plan_with("q", "w" * 513))

    def test_the_question_bound_is_unchanged_at_300(self) -> None:
        parse_plan(self._plan_with("q" * 300, "w"))
        with self.assertRaises(LedgerError):
            parse_plan(self._plan_with("q" * 301, "w"))

    def test_a_long_why_does_not_licence_a_long_question(self) -> None:
        """The caps are separate; widening one must not widen the other."""
        with self.assertRaises(LedgerError):
            parse_plan(self._plan_with("q" * 400, "w" * 400))

    def test_the_two_bounds_are_distinct(self) -> None:
        from orbit.runtime.analysis_question_ledger import (
            MAX_QUESTION_CHARS,
            MAX_WHY_CHARS,
        )

        self.assertEqual(MAX_QUESTION_CHARS, 300)
        self.assertEqual(MAX_WHY_CHARS, 512)


class WhyCostsNothingPerStepTests(unittest.TestCase):
    """Why a wider `why` bound is safe: it never enters a later prompt.

    The whole ledger is rendered into every step prompt, so a bound that grows
    the rendering would compound across a run and could push a context
    admission that the tighter bound survived. `render()` prints the question
    and the state, never the `why` -- that text appears once, in the model's own
    PLAN reply, and is not repeated. This pins that, because the argument for
    512 rests on it.
    """

    def test_the_rendered_ledger_omits_the_why(self) -> None:
        ledger = QuestionLedger()
        ledger.add(Question("Q1", "the question text", "UNIQUE-WHY-MARKER"))
        rendered = ledger.render()
        self.assertIn("the question text", rendered)
        self.assertNotIn("UNIQUE-WHY-MARKER", rendered)

    def test_the_worst_case_rendering_is_bounded_by_the_question_cap(self) -> None:
        from orbit.runtime.analysis_question_ledger import (
            MAX_QUESTION_CHARS,
            MAX_TOTAL_QUESTIONS,
            MAX_WHY_CHARS,
        )

        ledger = QuestionLedger()
        for index in range(MAX_TOTAL_QUESTIONS):
            ledger.add(
                Question(f"Q{index}", "q" * MAX_QUESTION_CHARS, "w" * MAX_WHY_CHARS)
            )
        rendered = ledger.render()
        # Every `why` is excluded, so the rendering cannot exceed the question
        # text plus a short marker per line.
        self.assertLess(
            len(rendered), MAX_TOTAL_QUESTIONS * (MAX_QUESTION_CHARS + 64)
        )
        self.assertNotIn("w" * 64, rendered)

    def test_a_full_plan_at_both_caps_still_fits_the_document_bound(self) -> None:
        import json as _json

        from orbit.runtime.analysis_question_ledger import (
            MAX_INITIAL_QUESTIONS,
            MAX_PLAN_CHARS,
            MAX_QUESTION_CHARS,
            MAX_WHY_CHARS,
        )

        plan = _json.dumps({"questions": [
            {"id": f"Q{i}", "question": "q" * MAX_QUESTION_CHARS,
             "why": "w" * MAX_WHY_CHARS}
            for i in range(MAX_INITIAL_QUESTIONS)
        ]})
        self.assertLess(len(plan), MAX_PLAN_CHARS)
        self.assertEqual(len(parse_plan(plan)), MAX_INITIAL_QUESTIONS)

    def test_the_cap_counts_characters_and_that_stays_bounded(self) -> None:
        """Multi-byte text costs more bytes; the document bound still holds."""
        import json as _json

        for filler in ("w", "é", "\U0001f600"):
            with self.subTest(filler=filler):
                plan = _json.dumps(
                    {"questions": [{"id": "Q1", "question": "q",
                                    "why": filler * 512}]}
                )
                self.assertEqual(len(parse_plan(plan)), 1)
                with self.assertRaises(LedgerError):
                    parse_plan(_json.dumps(
                        {"questions": [{"id": "Q1", "question": "q",
                                        "why": filler * 513}]}
                    ))


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
