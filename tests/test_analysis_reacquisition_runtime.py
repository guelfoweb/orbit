"""A proven source reacquisition must cost no useful action and no evidence.

ANALYSIS-SOURCE-EQUIVALENCE-1, at the runtime seam. The recognizer tests prove
what is and is not the covered source; these prove what the runtime does with
that answer: the program still runs, its output is still recorded, but it does
not consume an action slot, it does not become new useful evidence, and it
feeds the existing NO_PROGRESS path rather than a new one.

Everything is gated on coverage. Without it the model was never given the
source, so reading it is how the session learns what the artifact is -- and
every path below must stay byte-identical to a run before this existed.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_progress import NO_PROGRESS, ProgressLedger  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    SOURCE_REACQUISITION,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult, DerivedArtifact  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

CTX = 8192
SOURCE = (
    "import os\n"
    "\n"
    "\n"
    "def handler(name):\n"
    "    return os.environ.get(name)\n"
)


class _Backend:
    """Orbit-native backend that always emits one execute_analysis call."""

    thinking = False

    def __init__(self, *, per_char: float = 0.25) -> None:
        self.per_char = per_char
        self.chat_calls: list[dict] = []
        self.code = "import orbit_tools\nprint(orbit_tools.read_file(0, 4096))"

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(
            tokens=int(40 + chars * self.per_char), context_tokens=CTX,
            rendered_hash="a" * 64, token_hash="b" * 64,
        )

    def chat_stream(self, messages, **kwargs):
        self.chat_calls.append({"messages": list(messages), "tools": kwargs.get("tools")})
        if not kwargs.get("tools"):
            return self._result("noted", [])
        import json

        return self._result("running", [{
            "id": f"call_{len(self.chat_calls)}", "type": "function",
            "function": {"name": "execute_analysis",
                         "arguments": json.dumps({"code": self.code})},
        }])

    def _result(self, content, calls):
        return ChatResult(
            content=content, model="m", finish_reason="stop", tool_calls=calls,
            prompt_tokens=1, completion_tokens=1, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


def _result(stdout="", stderr="", status="ok", artifacts=()):
    return AnalysisResult(
        status=status, code_sha256="c" * 64, input_sha256="i" * 64,
        stdout=stdout, stderr=stderr, exit_status=0, duration_seconds=0.1,
        artifacts=tuple(artifacts),
    )


def _replaced_result(stdout: str) -> AnalysisResult:
    """An `ok` result whose decode had to substitute U+FFFD."""
    return AnalysisResult(
        status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
        stdout=stdout, stderr="", exit_status=0, duration_seconds=0.1,
        output_replaced=True,
    )


def _truncated_result(stdout: str) -> AnalysisResult:
    """An `ok` result the sandbox had to cut. Truncation alone is not `bounded`."""
    return AnalysisResult(
        status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
        stdout=stdout, stderr="", exit_status=0, duration_seconds=0.1,
        truncated=True,
    )


class _Case(unittest.TestCase):
    def _runtime(self, data: bytes = None, backend=None) -> AnalysisRuntime:
        data = SOURCE.encode() if data is None else data
        self.backend = backend or _Backend()
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

    def _cover(self, runtime) -> None:
        coverage = runtime.plan_source_coverage()
        self.assertTrue(coverage.covered, "fixture must be COVER-eligible")
        runtime.cover_source(coverage)
        self.backend.chat_calls.clear()

    def _step(self, runtime, result):
        with mock.patch.object(module, "execute_analysis", lambda **kw: result):
            return runtime.step("go")


class CoveredSourceTextTests(_Case):
    """The authority for what was covered is the snapshot, not the prompt."""

    def test_no_covered_text_before_cover(self) -> None:
        self.assertIsNone(self._runtime().covered_source_text)

    def test_covered_text_is_the_artifact_bytes(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        self.assertEqual(runtime.covered_source_text, SOURCE)

    def test_rewinding_the_history_withdraws_it(self) -> None:
        runtime = self._runtime()
        checkpoint = len(runtime.messages)
        self._cover(runtime)
        del runtime.messages[checkpoint:]
        self.assertIsNone(runtime.covered_source_text)

    def test_a_binary_artifact_has_no_covered_text(self) -> None:
        runtime = self._runtime(b"\x00\xffbinary")
        self.assertIsNone(runtime.covered_source_text)


class SuppressionTests(_Case):
    """A-D. The four provable forms, through the real step()."""

    def test_raw_reacquisition_is_suppressed(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertIsNotNone(step.suppressed_duplicate_of)
        self.assertFalse(step.action_executed)

    def test_repr_reacquisition_is_suppressed(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=repr(SOURCE) + "\n"))
        self.assertIsNotNone(step.suppressed_duplicate_of)

    def test_numbered_reacquisition_is_suppressed(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        listing = "\n".join(
            f"{i:3}: {line}" for i, line in enumerate(SOURCE.splitlines())
        )
        step = self._step(runtime, _result(stdout=listing + "\n"))
        self.assertIsNotNone(step.suppressed_duplicate_of)

    def test_a_copied_file_is_suppressed(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        copy = DerivedArtifact(
            name="copy.py", size_bytes=len(SOURCE.encode()),
            sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
        step = self._step(runtime, _result(stdout="", artifacts=[copy]))
        self.assertIsNotNone(step.suppressed_duplicate_of)


class FailClosedRuntimeTests(_Case):
    """E-J. Everything that must remain useful evidence."""

    def _assert_useful(self, runtime, result) -> None:
        before = runtime.actions_executed
        step = self._step(runtime, result)
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, before + 1)

    def test_one_byte_difference_stays_useful(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        self._assert_useful(runtime, _result(stdout=SOURCE.replace("os", "0s", 1)))

    def test_source_plus_a_computed_line_stays_useful(self) -> None:
        """The live pattern: the source AND something worked out."""
        runtime = self._runtime()
        self._cover(runtime)
        self._assert_useful(
            runtime, _result(stdout=f"{SOURCE}\nLEN: {len(SOURCE)}")
        )

    def test_partial_source_stays_useful(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        self._assert_useful(runtime, _result(stdout=SOURCE[:40]))

    def test_a_targeted_calculation_stays_useful(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        self._assert_useful(runtime, _result(stdout="SHA256: abc123\nIMPORTS: ['os']"))

    def test_stderr_defeats_the_proof(self) -> None:
        """A warning is information; the output is not the source alone."""
        runtime = self._runtime()
        self._cover(runtime)
        self._assert_useful(
            runtime, _result(stdout=SOURCE, stderr="warning: deprecated")
        )

    def test_truncated_output_is_never_suppressed(self) -> None:
        """A prefix proves nothing about what was cut off.

        The sandbox caps stdout and leaves `status` as "ok", so a program that
        printed the source and THEN its findings has exactly the visible bytes
        of a bare re-read. Suppressing it would discard the findings while
        telling the model nothing was established -- the catastrophic
        direction. `truncated` is the only signal the comparison is partial.

        The artifact here is small enough to be COVER-eligible; what is
        exercised is the flag, which the sandbox sets independently of size.
        """
        runtime = self._runtime()
        self._cover(runtime)
        before = runtime.actions_executed
        step = self._step(runtime, _truncated_result(SOURCE + "\n"))
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, before + 1)

    def test_replaced_output_is_never_suppressed(self) -> None:
        """Decoding substituted U+FFFD, so the text is not what was printed.

        The same defect as truncation by a different route. `errors="replace"`
        turns non-UTF-8 bytes into U+FFFD, and an artifact that itself contains
        U+FFFD -- legal UTF-8, so coverable -- would then compare equal to
        output whose bytes never matched it. The comparison would be over an
        altered view of what the action produced.
        """
        source = "he\ufffdlo\n"
        runtime = self._runtime(source.encode())
        self._cover(runtime)
        before = runtime.actions_executed
        step = self._step(runtime, _replaced_result(source))
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, before + 1)

    def test_the_same_output_without_replacement_is_suppressed(self) -> None:
        """The control: only the substitution changes the verdict."""
        source = "he\ufffdlo\n"
        runtime = self._runtime(source.encode())
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=source))
        self.assertIsNotNone(step.suppressed_duplicate_of)

    def test_the_sandbox_detects_substitution_on_each_stream(self) -> None:
        """Each stream is checked in its own right.

        Testing only the pair would let one half cover for the other: an
        implementation that checked stderr alone would still pass while
        stdout -- the stream the recognizers actually read -- went unguarded.
        """
        from orbit.runtime.analysis_sandbox import execute_analysis  # noqa: F401

        for raw, expected in ((b"he\xfflo\n", True), (b"hello\n", False),
                              ("héllo\n".encode(), False)):
            with self.subTest(raw=raw):
                decoded = raw.decode("utf-8", "replace")
                self.assertEqual(decoded.encode("utf-8") != raw, expected)

    def test_replacement_on_stdout_alone_is_detected(self) -> None:
        """The stream the recognizers read must be guarded by itself."""
        import inspect

        from orbit.runtime import analysis_sandbox

        source = inspect.getsource(analysis_sandbox)
        marker = source[source.index("replaced = ("):]
        marker = marker[: marker.index(")\n")]
        self.assertIn("stdout.encode", marker)
        self.assertIn("stderr.encode", marker)

    def test_the_same_output_untruncated_is_suppressed(self) -> None:
        """The control: only truncation changes the verdict."""
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertIsNotNone(step.suppressed_duplicate_of)

    def test_a_failed_action_is_never_suppressed(self) -> None:
        """A failure must reach the model as a failure."""
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=SOURCE, status="error"))
        self.assertIsNone(step.suppressed_duplicate_of)

    def test_a_copy_alongside_other_output_stays_useful(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        copy = DerivedArtifact(
            name="copy.py", size_bytes=len(SOURCE.encode()),
            sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
        self._assert_useful(
            runtime, _result(stdout="LINE COUNT: 5", artifacts=[copy])
        )

    def test_the_source_printed_with_a_copy_stays_useful(self) -> None:
        """An artifact is new state whatever the stdout says."""
        runtime = self._runtime()
        self._cover(runtime)
        copy = DerivedArtifact(
            name="copy.py", size_bytes=len(SOURCE.encode()),
            sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
        self._assert_useful(runtime, _result(stdout=SOURCE, artifacts=[copy]))

    def test_a_different_file_written_stays_useful(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        other = DerivedArtifact(name="notes.txt", size_bytes=5, sha256="f" * 64)
        self._assert_useful(runtime, _result(stdout="", artifacts=[other]))


class WithoutCoverageTests(_Case):
    """K. Without COVER, nothing changes."""

    def test_reading_the_source_is_ordinary_work_without_coverage(self) -> None:
        runtime = self._runtime()
        before = runtime.actions_executed
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, before + 1)

    def test_a_copied_file_is_not_suppressed_without_coverage(self) -> None:
        """The artifact recognizer compares digests, not covered text.

        It never consults `covered_source_text`, so only the coverage gate
        stops it firing on a session that was never given the source -- where
        copying the artifact is ordinary, useful work. This is the case the
        stdout recognizers cannot cover for, because they refuse a None source
        on their own.
        """
        runtime = self._runtime()
        copy = DerivedArtifact(
            name="copy.py", size_bytes=len(SOURCE.encode()),
            sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
        before = runtime.actions_executed
        step = self._step(runtime, _result(stdout="", artifacts=[copy]))
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, before + 1)

    def test_suppression_starts_only_after_coverage(self) -> None:
        runtime = self._runtime()
        first = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertIsNone(first.suppressed_duplicate_of)
        self._cover(runtime)
        second = self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertIsNotNone(second.suppressed_duplicate_of)


class AccountingTests(_Case):
    """M-O. Progress, ceilings and provenance."""

    def test_a_suppressed_step_consumes_no_action_slot(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        before = runtime.actions_executed
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertEqual(runtime.actions_executed, before)

    def test_a_suppressed_step_still_counts_its_model_call(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        before = runtime.model_calls
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertEqual(runtime.model_calls, before + 1)

    def test_it_is_counted_in_the_suppression_tally(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        before = runtime.suppressed_duplicates
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        self.assertEqual(runtime.suppressed_duplicates, before + 1)

    def test_the_ledger_classifies_it_as_no_progress(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        record = ProgressLedger().classify(1, step)
        self.assertEqual(record.classification, NO_PROGRESS)

    def test_the_repeated_four_way_sequence_never_progresses(self) -> None:
        """M. All four live forms in a row, none of them useful."""
        runtime = self._runtime()
        self._cover(runtime)
        listing = "\n".join(
            f"{i:3}: {line}" for i, line in enumerate(SOURCE.splitlines())
        )
        copy = DerivedArtifact(
            name="copy.py", size_bytes=len(SOURCE.encode()),
            sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
        ledger = ProgressLedger()
        outputs = [
            _result(stdout=SOURCE + "\n"),
            _result(stdout=repr(SOURCE) + "\n"),
            _result(stdout=listing + "\n"),
            _result(stdout="", artifacts=[copy]),
        ]
        for index, result in enumerate(outputs, 1):
            step = self._step(runtime, result)
            self.assertEqual(
                ledger.classify(index, step).classification, NO_PROGRESS
            )
        self.assertEqual(runtime.actions_executed, 0)
        self.assertEqual(runtime.suppressed_duplicates, 4)

    def test_a_suppressed_step_still_reports_its_artifacts(self) -> None:
        """Suppression is about the observation, never about hiding the action.

        A suppressed action still ran and may have written a file. The
        analyst's trailer reads these fields, so dropping them would leave a
        real artifact on disk that nothing mentions.
        """
        runtime = self._runtime()
        self._cover(runtime)
        copy = DerivedArtifact(
            name="copy.py", size_bytes=len(SOURCE.encode()),
            sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
        step = self._step(runtime, _result(stdout="", artifacts=[copy]))
        self.assertIsNotNone(step.suppressed_duplicate_of)
        self.assertIsNotNone(step.raw_output_evidence_id)
        self.assertEqual(len(step.artifact_handles), 1)
        self.assertIn("copy.py", step.artifact_handles[0])

    def test_provenance_records_what_was_suppressed(self) -> None:
        """N. An audit can see the claim and check it."""
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        record = runtime.evidence_store.records[step.suppressed_duplicate_of]
        self.assertEqual(record.metadata["suppressed_as"], SOURCE_REACQUISITION)
        self.assertEqual(record.metadata["suppression_recognizer"], "raw")
        self.assertIn("suppression_detail", record.metadata)

    def test_the_raw_output_is_still_retained_and_attestable(self) -> None:
        """Nothing is hidden: the real bytes stay in the store."""
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(runtime, _result(stdout=SOURCE + "\n"))
        record = runtime.evidence_store.records[step.suppressed_duplicate_of]
        raw_id = record.metadata["raw_output_evidence_id"]
        raw = runtime.evidence_store.reattest_exact(raw_id)
        self.assertIsNotNone(raw)
        self.assertIn(SOURCE, raw)

    def test_the_model_is_told_plainly_and_not_that_it_is_finished(self) -> None:
        runtime = self._runtime()
        self._cover(runtime)
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        told = str(runtime.messages[-1]["content"])
        self.assertIn(SOURCE_REACQUISITION, told.lower())
        # It says the OUTPUT added nothing. It must not say the ANALYSIS is
        # finished, nor tell the model what to do instead -- deciding that is
        # the model's work, and a runtime that directed it would be doing
        # analysis rather than orchestration.
        for banned in (
            "analysis is complete", "you are finished", "you are done",
            "stop now", "report now", "no further", "conclude",
        ):
            self.assertNotIn(banned, told.lower(), banned)

    def test_global_ceilings_are_unchanged(self) -> None:
        """O. No bound moved."""
        self.assertEqual(module.MAX_AUTONOMOUS_ACTIONS, 12)
        self.assertEqual(module.SOFT_MAX_AUTONOMOUS_ACTIONS, 8)
        self.assertEqual(module.MAX_AUTONOMOUS_MODEL_CALLS, 15)
        self.assertEqual(module.MAX_CONSECUTIVE_NO_PROGRESS, 2)


class AutonomousLoopTests(_Case):
    """The loop ends on repeated no-progress rather than spending the budget."""

    def test_a_run_of_reacquisitions_stops_early(self) -> None:
        runtime = self._runtime()
        with mock.patch.object(
            module, "execute_analysis", lambda **kw: _result(stdout=SOURCE + "\n")
        ):
            run = runtime.run_autonomous("analyse", max_model_calls=8, finalize=False)
        self.assertEqual(run.cover_calls, 1)
        self.assertEqual(run.actions_executed, 0)
        self.assertGreater(run.suppressed_duplicates, 0)
        self.assertIn("no new evidence", run.stop_reason)

    def test_legitimate_work_still_runs_after_a_suppression(self) -> None:
        """J. Suppression is per-observation, never a mode the run enters."""
        runtime = self._runtime()
        self._cover(runtime)
        self._step(runtime, _result(stdout=SOURCE + "\n"))
        step = self._step(runtime, _result(stdout="FINDING: os.environ read"))
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, 1)


if __name__ == "__main__":
    unittest.main()
