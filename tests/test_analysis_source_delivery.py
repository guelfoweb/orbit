"""Source delivery on a run without COVER: the first complete read is the anchor.

ANALYSIS-SOURCE-CHURN-1. The exact recognizers (`analysis_source_identity`,
`analysis_source_dominance`) and the suppression seam in `step()` exist since
#314/#315, but they were gated on `covered_source_text`, which only a COVER
turn establishes. Every retained Fattura run ran without COVER, so 13 of its
135 actions -- byte-exact or reversibly exact re-reads of a source an earlier
action had already produced in full -- were treated as new work and each cost
a STEP, a FINISH and usually a repair.

These tests are behaviour-first: they drive `step()` and `run_autonomous()`
with a scripted backend and a faked sandbox, and witness what happened through
the sandbox dispatch count, `actions_executed`, the evidence store's record
count, the ledger, and the delivery state itself -- never only a flag.

The invariant under test: an action is suppressed ONLY when its output is
provably the source the session already holds authoritatively (covered, or
delivered by an earlier action of the same session) plus nothing but exactly
recomputable properties. Everything else runs and counts.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_progress import NO_PROGRESS, ProgressLedger  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    ANALYSIS_TOOL_NAME,
    FINISH_TOOL_NAME,
    PLAN_TOOL_NAME,
    SOURCE_DOMINATED,
    SOURCE_REACQUISITION,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
    SourceDelivery,
)
from orbit.runtime.analysis_sandbox import AnalysisResult, DerivedArtifact  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

from tests.test_analysis_reacquisition_runtime import (  # noqa: E402
    SOURCE,
    _Backend,
    _replaced_result,
    _result,
    _truncated_result,
)

CTX = 8192
SHA = hashlib.sha256(SOURCE.encode()).hexdigest()
NUMBERED = "\n".join(f"{i:3}: {line}" for i, line in enumerate(SOURCE.splitlines()))


class _Case(unittest.TestCase):
    def _runtime(self, data: bytes = None, name: str = "artifact.py") -> AnalysisRuntime:
        data = SOURCE.encode() if data is None else data
        self.backend = _Backend()
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / name
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
        self.dispatched = 0
        return runtime

    def _step(self, runtime, result, code: str | None = None):
        """One step with a distinct program, so the pre-execution duplicate
        guard (identical code) cannot mask what the recognizers decide."""
        self.backend.code = code or f"print({len(self.backend.chat_calls)})"

        def sandbox(**kw):
            self.dispatched += 1
            return result

        with mock.patch.object(module, "execute_analysis", sandbox):
            return runtime.step("go")

    def _records(self, runtime) -> int:
        return len(runtime.evidence_store.records)


# --------------------------------------------------------------------------
class FirstReadTests(_Case):
    """Case 1 / 8 / 10: a session with no authority reads the source as work."""

    def test_case1_the_first_complete_read_executes_and_counts(self) -> None:
        runtime = self._runtime()
        self.assertIsNone(runtime.source_delivery)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertEqual(self.dispatched, 1)
        self.assertTrue(step.action_executed)
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertEqual(runtime.actions_executed, 1)
        delivery = runtime.source_delivery
        self.assertIsInstance(delivery, SourceDelivery)
        self.assertEqual(delivery.evidence_id, step.evidence.evidence_id)
        self.assertEqual(delivery.representation, "raw")
        self.assertEqual(delivery.sha256, SHA)

    def test_case8_an_excerpt_is_not_a_delivery_and_the_full_read_after_it_runs(self) -> None:
        runtime = self._runtime()
        head = self._step(runtime, _result(stdout=SOURCE[:40] + "\n"))
        self.assertTrue(head.action_executed)
        self.assertIsNone(runtime.source_delivery, "a bounded excerpt delivers nothing")
        full = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertTrue(full.action_executed, "the first COMPLETE read must run")
        self.assertIsNone(full.suppressed_duplicate_of)
        self.assertEqual(runtime.actions_executed, 2)
        self.assertEqual(runtime.source_delivery.evidence_id, full.evidence.evidence_id)
        again = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertFalse(again.action_executed)
        self.assertEqual(runtime.actions_executed, 2)

    def test_a_tail_excerpt_and_a_head_excerpt_together_deliver_nothing(self) -> None:
        runtime = self._runtime()
        self._step(runtime, _result(stdout=SOURCE[: len(SOURCE) // 2]))
        self._step(runtime, _result(stdout=SOURCE[len(SOURCE) // 2 :]))
        self.assertIsNone(runtime.source_delivery, "pieces are never reassembled")
        full = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertTrue(full.action_executed)

    def test_naming_the_file_is_not_delivering_it(self) -> None:
        runtime = self._runtime(name="Fattura981033956.js")
        step = self._step(runtime, _result(stdout=f"read Fattura981033956.js: {len(SOURCE)} bytes\n"))
        self.assertTrue(step.action_executed)
        self.assertIsNone(runtime.source_delivery)
        full = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertTrue(full.action_executed)
        self.assertEqual(runtime.actions_executed, 2)

    def test_case10_a_new_session_inherits_nothing(self) -> None:
        first = self._runtime()
        self._step(first, _result(stdout=SOURCE + "\n"))
        self.assertIsNotNone(first.source_delivery)
        second = self._runtime()
        self.assertIsNone(second.source_delivery)
        step = self._step(second, _result(stdout=SOURCE + "\n"))
        self.assertTrue(step.action_executed, "a new session must read the source itself")
        self.assertEqual(second.actions_executed, 1)

    def test_case10b_a_different_artifact_inherits_nothing(self) -> None:
        other = (SOURCE + "# changed\n").encode()
        runtime = self._runtime(data=other)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertTrue(step.action_executed)
        self.assertIsNone(runtime.source_delivery, "the OLD source is not this artifact")

    def test_a_truncated_or_substituted_first_read_delivers_nothing(self) -> None:
        for bad in (_truncated_result(SOURCE + "\n"), _replaced_result(SOURCE + "\n")):
            runtime = self._runtime()
            step = self._step(runtime, bad)
            self.assertTrue(step.action_executed)
            self.assertIsNone(runtime.source_delivery)

    def test_a_failed_read_delivers_nothing(self) -> None:
        runtime = self._runtime()
        self._step(runtime, _result(stdout=SOURCE + "\n", status="error"))
        self.assertIsNone(runtime.source_delivery)


# --------------------------------------------------------------------------
class SuppressionAfterDeliveryTests(_Case):
    """Cases 2-4, 7, 9: identical exact forms after a delivery add nothing."""

    def _delivered(self) -> AnalysisRuntime:
        runtime = self._runtime()
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertIsNotNone(runtime.source_delivery)
        return runtime

    def _assert_suppressed(self, runtime, step, kind: str, recognizer: str) -> None:
        self.assertFalse(step.action_executed)
        self.assertIsNotNone(step.suppressed_duplicate_of)
        self.assertEqual(runtime.actions_executed, 1, "no action slot consumed")
        record = runtime.evidence_store.records[step.suppressed_duplicate_of]
        self.assertEqual(record.metadata["suppressed_as"], kind)
        self.assertEqual(record.metadata["suppression_recognizer"], recognizer)
        self.assertEqual(
            record.metadata["suppressed_against"],
            f"delivery:{runtime.source_delivery.evidence_id}",
        )

    def test_case2_a_second_identical_full_read_is_suppressed(self) -> None:
        runtime = self._delivered()
        before = self._records(runtime)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertEqual(self.dispatched, 2, "the program still runs: nothing is blocked")
        self._assert_suppressed(runtime, step, SOURCE_REACQUISITION, "raw")
        self.assertEqual(self._records(runtime), before + 2, "recorded like any execution (record + raw)")
        self.assertEqual(ProgressLedger().classify(2, step).classification, NO_PROGRESS)

    def test_case3_a_reversible_repr_is_suppressed(self) -> None:
        runtime = self._delivered()
        step = self._step(runtime, _result(stdout=repr(SOURCE) + "\n"))
        self._assert_suppressed(runtime, step, SOURCE_REACQUISITION, "repr")

    def test_case4_a_strict_numbered_listing_is_suppressed(self) -> None:
        runtime = self._delivered()
        step = self._step(runtime, _result(stdout=NUMBERED + "\n"))
        self._assert_suppressed(runtime, step, SOURCE_REACQUISITION, "numbered")

    def test_case7_source_plus_recomputable_properties_is_suppressed(self) -> None:
        runtime = self._delivered()
        out = f"{SOURCE}\nLEN: {len(SOURCE)}\nTOTAL_LINES: {len(SOURCE.splitlines())}\nSHA256: {SHA}\n"
        step = self._step(runtime, _result(stdout=out))
        self._assert_suppressed(runtime, step, SOURCE_DOMINATED, "raw")
        record = runtime.evidence_store.records[step.suppressed_duplicate_of]
        self.assertEqual(set(record.metadata["verified_properties"]),
                         {"text_length", "line_count", "sha256"})

    def test_case9_an_identical_work_artifact_is_suppressed_under_the_existing_contract(self) -> None:
        runtime = self._delivered()
        copy = DerivedArtifact(name="anything.bin", size_bytes=len(SOURCE.encode()), sha256=SHA)
        step = self._step(runtime, _result(stdout="", artifacts=[copy]))
        self._assert_suppressed(runtime, step, SOURCE_REACQUISITION, "artifact")
        self.assertEqual(len(step.artifact_handles), 1, "the file it wrote is still reported")

    def test_the_model_is_told_where_the_bytes_are(self) -> None:
        """The record named in the note must actually hold the complete source."""
        runtime = self._delivered()
        delivery = runtime.source_delivery
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        told = str(runtime.messages[-1]["content"])
        self.assertIn(SOURCE_REACQUISITION, told.lower())
        # The note is the suppressed record's own content (the tool message
        # wraps it in the canonical reference, whose header names that record).
        note = runtime.evidence_store.reattest_exact(step.suppressed_duplicate_of)
        named = re.findall(r"evidence:(ev_[0-9a-f_]+)", note)
        self.assertEqual(set(named), {delivery.raw_evidence_id},
                         "the note names the raw record and nothing else")
        self.assertNotIn(delivery.evidence_id, note, "the bounded record is not 'the same bytes'")
        raw = runtime.evidence_store.reattest_exact(delivery.raw_evidence_id)
        self.assertIn(SOURCE, raw, "the record the note names holds the complete source")
        self.assertNotIn("earlier in this conversation", told, "no COVER turn exists to point at")
        for banned in ("analysis is complete", "you are finished", "you are done",
                       "stop now", "report now", "no further", "conclude"):
            self.assertNotIn(banned, told.lower(), banned)

    def test_the_delivery_does_not_move_to_the_suppressed_read(self) -> None:
        runtime = self._delivered()
        anchor = runtime.source_delivery.evidence_id
        self._step(runtime, _result(stdout=repr(SOURCE) + "\n"))
        self.assertEqual(runtime.source_delivery.evidence_id, anchor)


# --------------------------------------------------------------------------
class FailClosedAfterDeliveryTests(_Case):
    """Cases 5, 6, 7b: anything not proven exact still runs and counts."""

    def _delivered(self) -> AnalysisRuntime:
        runtime = self._runtime()
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        return runtime

    def _assert_useful(self, runtime, step) -> None:
        self.assertTrue(step.action_executed)
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertEqual(runtime.actions_executed, 2)

    def test_case5_a_listing_with_one_line_changed_runs(self) -> None:
        runtime = self._delivered()
        lines = NUMBERED.split("\n")
        lines[2] = lines[2] + " # patched"
        self._assert_useful(runtime, self._step(runtime, _result(stdout="\n".join(lines) + "\n")))

    def test_case5b_a_partial_listing_runs(self) -> None:
        runtime = self._delivered()
        partial = "\n".join(NUMBERED.split("\n")[:-1])
        self._assert_useful(runtime, self._step(runtime, _result(stdout=partial + "\n")))

    def test_case6_source_plus_a_semantic_fact_runs(self) -> None:
        runtime = self._delivered()
        for extra in ("DEF_COUNT: 1", "IMPORTS: os", "calls: environ.get", "===END==="):
            step = self._step(runtime, _result(stdout=f"{SOURCE}\n{extra}\n"))
            self.assertTrue(step.action_executed, extra)
            self.assertIsNone(step.suppressed_duplicate_of, extra)

    def test_case7b_a_wrong_property_value_runs(self) -> None:
        runtime = self._delivered()
        self._assert_useful(runtime, self._step(runtime, _result(stdout=f"{SOURCE}\nLEN: {len(SOURCE) + 1}\n")))

    def test_a_one_byte_difference_runs(self) -> None:
        runtime = self._delivered()
        self._assert_useful(runtime, self._step(runtime, _result(stdout=SOURCE.replace("os", "0s", 1) + "\n")))

    def test_a_non_reversible_repr_runs(self) -> None:
        runtime = self._delivered()
        self._assert_useful(runtime, self._step(runtime, _result(stdout=repr(SOURCE)[:-3] + "X'\n")))

    def test_the_first_read_shape_with_a_marker_runs_and_is_not_suppressed_later(self) -> None:
        """Source plus an unexplained marker: useful every time (F on the traces)."""
        runtime = self._delivered()
        out = f"LENGTH: {len(SOURCE)}\nREPR START:\n{repr(SOURCE)}\nREPR END\n"
        self._assert_useful(runtime, self._step(runtime, _result(stdout=out)))

    def test_stderr_and_truncation_defeat_the_proof(self) -> None:
        runtime = self._delivered()
        for bad in (_result(stdout=SOURCE + "\n", stderr="warning"), _truncated_result(SOURCE + "\n")):
            step = self._step(runtime, bad)
            self.assertTrue(step.action_executed)

    def test_a_semantic_action_after_a_suppression_runs(self) -> None:
        runtime = self._delivered()
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        step = self._step(runtime, _result(stdout="FINDING: os.environ read\n"))
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, 2)


# --------------------------------------------------------------------------
# A source larger than MAX_EVIDENCE_CHARS: the model-facing record of a read
# is then a truncated observation, which is the live situation (7.7 KB
# artifact, 3200-char bound) and the one a 63-byte fixture cannot reach.
LARGE_SOURCE = "".join(
    f"function f{i}(a, b) {{ return decode(\"{hashlib.sha256(str(i).encode()).hexdigest()[:24]}\", {i % 97}); }}\n"
    for i in range(120)
)
assert len(LARGE_SOURCE) > 2 * module.MAX_EVIDENCE_CHARS


class RealSizeTests(_Case):
    """At the live size: the bounded record is not the bytes; the raw one is."""

    def _large(self) -> AnalysisRuntime:
        return self._runtime(data=LARGE_SOURCE.encode())

    def test_delivery_by_containment_at_the_live_size(self) -> None:
        runtime = self._large()
        out = f"LENGTH: {len(LARGE_SOURCE)}\nREPR START:\n{repr(LARGE_SOURCE)}\nREPR END\n"
        step = self._step(runtime, _result(stdout=out))
        self.assertTrue(step.action_executed)
        delivery = runtime.source_delivery
        self.assertIsNotNone(delivery, "a 8 KB first read must deliver like a 63-byte one")
        self.assertEqual(delivery.representation, "repr_contained")
        record = runtime.evidence_store.records[delivery.evidence_id]
        self.assertTrue(record.metadata.get("observation_truncated"), "the model-facing record is bounded")
        raw = runtime.evidence_store.reattest_exact(delivery.raw_evidence_id)
        self.assertIn(repr(LARGE_SOURCE), raw, "the raw record holds the complete output")

    def test_the_note_names_a_record_that_holds_the_complete_source(self) -> None:
        runtime = self._large()
        self._step(runtime, _result(stdout=LARGE_SOURCE + "\n"))
        step = self._step(runtime, _result(stdout=LARGE_SOURCE + "\n"))
        self.assertFalse(step.action_executed)
        note = runtime.evidence_store.reattest_exact(step.suppressed_duplicate_of)
        named = re.findall(r"evidence:(ev_[0-9a-f_]+)", note)
        self.assertTrue(named)
        delivery = runtime.source_delivery
        self.assertNotIn(delivery.evidence_id, note, "the bounded record is a truncated view, never 'the same bytes'")
        for evidence_id in named:
            raw = runtime.evidence_store.reattest_exact(evidence_id)
            self.assertIsNotNone(raw, evidence_id)
            self.assertIn(LARGE_SOURCE, raw, "a named record that does not hold the bytes is a false pointer")
            self.assertNotIn("truncated for prompt", raw)

    def test_an_identical_large_read_is_suppressed_and_a_range_is_not(self) -> None:
        runtime = self._large()
        self._step(runtime, _result(stdout=LARGE_SOURCE + "\n"))
        self.assertFalse(self._step(runtime, _result(stdout=LARGE_SOURCE + "\n")).action_executed)
        self.assertTrue(self._step(runtime, _result(stdout=LARGE_SOURCE[:4000] + "\n")).action_executed)
        self.assertTrue(self._step(runtime, _result(stdout=LARGE_SOURCE[4000:] + "\n")).action_executed)
        self.assertEqual(runtime.actions_executed, 3)

    def test_containment_is_byte_exact_not_whitespace_insensitive(self) -> None:
        """Leading and trailing whitespace are source bytes like any other."""
        padded = "  \n" + LARGE_SOURCE + "\n\n"
        runtime = self._runtime(data=padded.encode())
        step = self._step(runtime, _result(stdout=padded.strip() + "\n"))
        self.assertTrue(step.action_executed)
        self.assertIsNone(runtime.source_delivery, "the stripped text is not the whole source")
        exact = self._step(runtime, _result(stdout=padded))
        self.assertTrue(exact.action_executed)
        self.assertIsNotNone(runtime.source_delivery)


class DeliveryFormsTests(_Case):
    """Which first-read shapes establish the anchor -- exactly, never fuzzily."""

    def test_each_exact_form_delivers(self) -> None:
        for out, rep in (
            (SOURCE + "\n", "raw"), (repr(SOURCE) + "\n", "repr"), (NUMBERED + "\n", "numbered"),
            (f"{SOURCE}\nLEN: {len(SOURCE)}\n", "raw"),
        ):
            runtime = self._runtime()
            self._step(runtime, _result(stdout=out))
            self.assertEqual(runtime.source_delivery.representation, rep, out[:20])

    def test_the_live_first_read_shapes_deliver_by_containment(self) -> None:
        """`LENGTH:` + `REPR START:` around the literal, and the source printed twice."""
        for out, rep in (
            (f"LENGTH: {len(SOURCE)}\nREPR START:\n{repr(SOURCE)}\nREPR END\n", "repr_contained"),
            (f"--- BEGIN ---\n{SOURCE}\n--- END ---\n{SOURCE}\n", "raw_contained"),
        ):
            runtime = self._runtime()
            step = self._step(runtime, _result(stdout=out))
            self.assertTrue(step.action_executed, "the acquisition itself always counts")
            self.assertEqual(runtime.actions_executed, 1)
            self.assertEqual(runtime.source_delivery.representation, rep)
            again = self._step(runtime, _result(stdout=SOURCE + "\n"))
            self.assertFalse(again.action_executed)

    def test_containment_is_of_the_whole_source_only(self) -> None:
        for out in (SOURCE[:-5] + "\n...\n", SOURCE[3:] + "\n", repr(SOURCE)[:-4] + "'\n",
                    SOURCE.replace("\n", "\r\n")):
            runtime = self._runtime()
            self._step(runtime, _result(stdout=out))
            self.assertIsNone(runtime.source_delivery, out[:30])

    def test_a_numbered_listing_inside_other_output_does_not_deliver(self) -> None:
        """Only the two exact literal forms are searched for; nothing is extracted."""
        runtime = self._runtime()
        self._step(runtime, _result(stdout=f"LISTING:\n{NUMBERED}\nEND\n"))
        self.assertIsNone(runtime.source_delivery)

    def test_delivery_is_recorded_in_provenance(self) -> None:
        runtime = self._runtime()
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        record = runtime.evidence_store.records[step.evidence.evidence_id]
        self.assertEqual(record.metadata["source_delivery"], "raw")
        self.assertEqual(record.metadata["source_delivery_sha256"], SHA)
        self.assertTrue(runtime.messages[-1].get("source_delivered"))


# --------------------------------------------------------------------------
class LifecycleTests(_Case):
    """The anchor is a property of the history and of the pinned snapshot."""

    def test_rewinding_the_history_withdraws_the_delivery(self) -> None:
        runtime = self._runtime()
        checkpoint = len(runtime.messages)
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertIsNotNone(runtime.source_delivery)
        del runtime.messages[checkpoint:]
        self.assertIsNone(runtime.source_delivery)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertTrue(step.action_executed, "after a rewind the source must be read again")

    def test_a_changed_snapshot_withdraws_the_delivered_text(self) -> None:
        runtime = self._runtime()
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        runtime.source.snapshot_path.chmod(0o600)
        runtime.source.snapshot_path.write_bytes(b"different bytes\n")
        self.assertIsNone(runtime.delivered_source_text)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertTrue(step.action_executed, "nothing is suppressed on a file nobody delivered")

    def test_a_mark_for_another_digest_is_not_a_delivery(self) -> None:
        runtime = self._runtime()
        runtime.messages.append({"role": "tool", "tool_call_id": "x", "name": ANALYSIS_TOOL_NAME,
                                 "content": "ref", "source_delivered": {
                                     "evidence_id": "ev_other", "representation": "raw", "sha256": "0" * 64}})
        self.assertIsNone(runtime.source_delivery)

    def test_coverage_takes_precedence_and_is_unchanged(self) -> None:
        """A covered run behaves exactly as before: same wording, same provenance."""
        runtime = self._runtime()
        coverage = runtime.plan_source_coverage()
        self.assertTrue(coverage.covered)
        runtime.cover_source(coverage)
        self.backend.chat_calls.clear()
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertFalse(step.action_executed)
        record = runtime.evidence_store.records[step.suppressed_duplicate_of]
        self.assertEqual(record.metadata["suppressed_against"], "coverage")
        self.assertIn("earlier in this conversation", str(runtime.messages[-1]["content"]))
        self.assertIsNone(runtime.source_delivery, "coverage needs no delivery")


# --------------------------------------------------------------------------
class _ScriptedModel:
    """PLAN, then actions with distinct code, then FINISH decisions."""

    def __init__(self, questions, n_actions_per_question=3) -> None:
        self.questions = questions
        self.calls: list[dict] = []
        self.actions = 0

    def reply(self, tools):
        names = [t["function"]["name"] for t in (tools or [])]
        if PLAN_TOOL_NAME in names:
            return self._call(PLAN_TOOL_NAME, {"questions": self.questions})
        if FINISH_TOOL_NAME in names:
            return self._call(FINISH_TOOL_NAME, {"status": "still_open", "answer_summary": "more to do"})
        if ANALYSIS_TOOL_NAME in names:
            self.actions += 1
            return self._call(ANALYSIS_TOOL_NAME, {"code": f"# action {self.actions}\nprint({self.actions})"})
        return None

    def _call(self, name, arguments):
        return [{"id": f"c{len(self.calls)}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}]


class _ControllerBackend:
    thinking = False

    def __init__(self, model) -> None:
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

    def count_text_tokens(self, text: str):
        return TokenCount(tokens=len(text.split()), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        tools = kwargs.get("tools")
        self.chat_calls.append({"tools": tools, "messages": list(messages)})
        calls = self.model.reply(tools) or []
        self.model.calls.extend(calls)
        return ChatResult(content="", model="m", finish_reason="stop", tool_calls=calls,
                          prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                          prompt_tokens_per_second=None, generation_tokens_per_second=None)

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


class ControllerPathTests(unittest.TestCase):
    """M8: the recognizer runs on the structured controller's STEP too.

    `run_autonomous(cover=False)` is the live shape: PLAN, then STEP/FINISH per
    question. The sandbox is scripted to answer the live sequence -- full
    source, repr, numbered listing, then a semantic finding -- and the
    witnesses are the dispatch count, `actions_executed`, the FINISH calls the
    backend saw, and the evidence store.
    """

    def _run(self, outputs):
        # Two questions: each may run MAX_ACTIONS_PER_QUESTION (2) actions, so
        # the four-step live sequence fits within the controller's own bounds.
        model = _ScriptedModel([
            {"question": "What does it do?", "missing_fact": "behaviour"},
            {"question": "Where does it send data?", "missing_fact": "destination"},
        ])
        backend = _ControllerBackend(model)
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / "artifact.py"
        path.write_bytes(SOURCE.encode())
        runtime = AnalysisRuntime(
            backend=backend,
            source=AnalysisSource(snapshot_path=path, sha256=SHA, size_bytes=len(SOURCE.encode()),
                                  original_path=str(path)),
            evidence_store=EvidenceStore(root=workspace.root / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        queue = list(outputs)
        dispatched = []

        def sandbox(**kw):
            out = queue.pop(0) if queue else "FINDING: nothing more"
            dispatched.append(out)
            # A distinct program identity per action, as the real sandbox
            # reports: a constant one would make the ledger read every later
            # step as a repeated strategy, which is a fixture artefact.
            return AnalysisResult(
                status="ok", code_sha256=hashlib.sha256(kw["code"].encode()).hexdigest(),
                input_sha256="i" * 64, stdout=out, stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", sandbox):
            run = runtime.run_autonomous("Analyse it.", cover=False, finalize=False,
                                         max_model_calls=18, max_actions=6)
        finishes = sum(1 for c in backend.chat_calls
                       if any(t["function"]["name"] == FINISH_TOOL_NAME for t in (c["tools"] or [])))
        return runtime, run, dispatched, finishes

    def test_the_live_sequence_spends_one_action_on_three_reads(self) -> None:
        runtime, run, dispatched, finishes = self._run([
            SOURCE + "\n", repr(SOURCE) + "\n", NUMBERED + "\n", "FINDING: pickle.loads on input\n",
        ])
        self.assertEqual(run.cover_calls, 0)
        self.assertGreaterEqual(len(dispatched), 4, "every program still runs")
        self.assertEqual(run.suppressed_duplicates, 2, "the repr and the listing added nothing")
        # The first read and the finding count; the two re-reads do not.
        self.assertGreaterEqual(run.actions_executed, 2)
        self.assertLessEqual(run.actions_executed, len(dispatched) - 2)
        self.assertEqual(finishes, run.actions_executed, "a suppressed read costs no FINISH call")
        self.assertIsNotNone(runtime.source_delivery)

    def test_without_a_prior_delivery_every_distinct_read_runs(self) -> None:
        """Control: three semantic findings, nothing suppressed."""
        runtime, run, dispatched, finishes = self._run(["A: 1\n", "B: 2\n", "C: 3\n"])
        self.assertEqual(run.suppressed_duplicates, 0)
        self.assertEqual(run.actions_executed, len(dispatched))
        self.assertIsNone(runtime.source_delivery)


if __name__ == "__main__":
    unittest.main()
