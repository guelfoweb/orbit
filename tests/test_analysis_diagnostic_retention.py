"""Opt-in diagnostics for one live ANALYSIS run.

A live report failure could not be diagnosed at all: `AnalysisWorkspace.close`
deletes the workspace, the EvidenceStore lives inside it, and per-call token
accounting was never persisted. These tests pin the retention path that makes
the next run reconstructible -- and, just as importantly, that an ordinary run
is unchanged by its existence.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.runtime.analysis_runtime import (
    ANALYSIS_RETENTION_ENV,
    retention_root,
)

from tests.test_analysis_controller_runtime import _Case, _Model, _question


class _DiagnosticTestBase(unittest.TestCase):
    def _runtime(self, questions=3):
        borrowed = _Case("run")
        borrowed.addCleanup = self.addCleanup
        return borrowed._runtime(
            _Model(plan=[_question(f"q{i}") for i in range(questions)])
        )

    def _retain_dir(self) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="orbit-diag-"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(directory, ignore_errors=True)
        )
        return directory


class RetentionIsOffByDefaultTests(_DiagnosticTestBase):
    def test_no_retention_directory_is_configured_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ANALYSIS_RETENTION_ENV, None)
            self.assertIsNone(retention_root())

    def test_the_workspace_is_still_deleted(self) -> None:
        """The whole point of the default: runs leave nothing behind."""
        runtime = self._runtime()
        root = runtime.workspace.root
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ANALYSIS_RETENTION_ENV, None)
            runtime.run_autonomous("Analyse it.", finalize=False, max_model_calls=20)
            runtime.close()
        self.assertFalse(root.exists())

    def test_no_report_trace_is_written(self) -> None:
        retained = self._retain_dir()
        runtime = self._runtime()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ANALYSIS_RETENTION_ENV, None)
            runtime.run_autonomous("Analyse it.", finalize=True, max_model_calls=20)
        self.assertEqual(list(retained.iterdir()), [])


class RetentionPreservesTheRunTests(_DiagnosticTestBase):
    def test_the_workspace_survives_with_its_evidence(self) -> None:
        retained = self._retain_dir()
        runtime = self._runtime()
        root = runtime.workspace.root
        with mock.patch.dict(
            os.environ, {ANALYSIS_RETENTION_ENV: str(retained)}, clear=False
        ):
            runtime.run_autonomous("Analyse it.", finalize=True, max_model_calls=20)
            runtime.close()

        # Moved, not copied: the original is gone and the bundle holds it.
        self.assertFalse(root.exists())
        bundle = retained / root.name
        self.assertTrue(bundle.is_dir())
        evidence = list((bundle / "evidence").glob("*.txt"))
        self.assertTrue(evidence, "the run's evidence bodies are retained")

    def test_retained_evidence_is_byte_exact(self) -> None:
        """Lossless: ids, bodies and the index all survive."""
        retained = self._retain_dir()
        runtime = self._runtime()
        root = runtime.workspace.root
        with mock.patch.dict(
            os.environ, {ANALYSIS_RETENTION_ENV: str(retained)}, clear=False
        ):
            runtime.run_autonomous("Analyse it.", finalize=True, max_model_calls=20)
            before = {
                record_id: runtime.evidence_store.load_raw(record_id)
                for record_id in runtime.evidence_store.records
            }
            runtime.close()

        bundle = retained / root.name / "evidence"
        for record_id, body in before.items():
            path = bundle / f"{record_id}.txt"
            self.assertTrue(path.exists(), f"{record_id} retained")
            self.assertEqual(path.read_text(encoding="utf-8"), body)

    def test_the_report_admission_is_recorded(self) -> None:
        """What a refused report needs, captured before admission decides."""
        retained = self._retain_dir()
        runtime = self._runtime()
        with mock.patch.dict(
            os.environ, {ANALYSIS_RETENTION_ENV: str(retained)}, clear=False
        ):
            runtime.run_autonomous("Analyse it.", finalize=True, max_model_calls=20)

        trace = retained / "report_admission.jsonl"
        self.assertTrue(trace.exists())
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertTrue(records)
        entry = records[0]
        self.assertEqual(entry["event"], "analysis_report_admission")
        # The exact prompt, and a size table that makes each component's
        # contribution reconstructible without re-running anything.
        self.assertTrue(entry["messages"])
        self.assertEqual(len(entry["message_chars"]), len(entry["messages"]))
        self.assertEqual(len(entry["record_ids"]), entry["record_count"])

    def test_a_cancelled_run_still_retains_its_bundle(self) -> None:
        """Diagnostics must survive the runs most worth diagnosing."""
        retained = self._retain_dir()
        runtime = self._runtime()
        root = runtime.workspace.root
        with mock.patch.dict(
            os.environ, {ANALYSIS_RETENTION_ENV: str(retained)}, clear=False
        ), mock.patch.object(type(runtime), "step", side_effect=KeyboardInterrupt):
            run = runtime.run_autonomous(
                "Analyse it.", finalize=False, max_model_calls=20
            )
            runtime.close()

        self.assertTrue(run.cancelled)
        self.assertTrue((retained / root.name).is_dir())


class RetentionDoesNotChangeBehaviourTests(_DiagnosticTestBase):
    def test_the_run_result_is_identical_either_way(self) -> None:
        """A diagnostic that changes the answer is not a diagnostic."""

        def observe(retain: Path | None):
            runtime = self._runtime()
            environment = (
                {ANALYSIS_RETENTION_ENV: str(retain)} if retain else {}
            )
            with mock.patch.dict(os.environ, environment, clear=False):
                if retain is None:
                    os.environ.pop(ANALYSIS_RETENTION_ENV, None)
                run = runtime.run_autonomous(
                    "Analyse it.", finalize=True, max_model_calls=20
                )
                runtime.close()
            return (
                run.stop_reason, run.model_calls, run.actions_executed,
                len(run.steps), run.plan_calls, run.initial_questions,
                list(run.resolved_questions), list(run.open_questions),
                run.final_report.text if run.final_report else None,
            )

        self.assertEqual(observe(None), observe(self._retain_dir()))

    def test_a_broken_retention_directory_does_not_fail_the_run(self) -> None:
        """Retention is a diagnostic; it may not turn a good run bad."""
        blocked = self._retain_dir() / "wall"
        blocked.write_text("not a directory", encoding="utf-8")
        runtime = self._runtime()
        root = runtime.workspace.root
        with mock.patch.dict(
            os.environ, {ANALYSIS_RETENTION_ENV: str(blocked / "under")},
            clear=False,
        ):
            run = runtime.run_autonomous(
                "Analyse it.", finalize=True, max_model_calls=20
            )
            runtime.close()

        self.assertIsNotNone(run.stop_reason)
        # Fell back to today's cleanup rather than leaving the workspace.
        self.assertFalse(root.exists())


class TheCallTraceDistinguishesPhasesTests(_DiagnosticTestBase):
    def _trace(self, runtime):
        from orbit.runtime.kv_diag import instrument_backend

        directory = Path(tempfile.mkdtemp(prefix="orbit-trace-"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(directory, ignore_errors=True)
        )
        path = directory / "trace.jsonl"
        # The environment is set BEFORE instrumenting: `instrument_backend`
        # reads `enabled()` at wrap time and hands the backend straight back
        # when diagnostics are off, so patching afterwards would leave an
        # uninstrumented backend and an empty trace.
        with mock.patch.dict(
            os.environ,
            {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(path)},
            clear=False,
        ):
            runtime.backend = instrument_backend(runtime.backend)
            runtime.run_autonomous("Analyse it.", finalize=True, max_model_calls=20)
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if json.loads(line).get("event") == "kv_diag_model_call"
        ]

    def test_plan_action_finish_and_report_are_distinguishable(self) -> None:
        """The first question an autopsy asks: which call was this?

        All four dispatched under `analysis_step` before, so a trace could
        not tell a plan from an action from a finish decision.
        """
        phases = [
            str(record.get("phase") or "") for record in self._trace(self._runtime())
        ]
        self.assertTrue(any(p.startswith("analysis_plan") for p in phases))
        self.assertTrue(any(p.startswith("analysis_finish") for p in phases))
        self.assertIn("analysis_step", phases)
        self.assertIn("analysis_report", phases)

    def test_a_finish_call_names_its_question(self) -> None:
        phases = [
            str(record.get("phase") or "") for record in self._trace(self._runtime())
        ]
        finishes = [p for p in phases if p.startswith("analysis_finish")]
        self.assertTrue(finishes)
        for phase in finishes:
            self.assertRegex(phase, r"^analysis_finish:Q\d+$")

    def test_backend_metrics_are_reported_not_invented(self) -> None:
        """Cache and eval come from the backend, or not at all."""
        records = self._trace(self._runtime())
        self.assertTrue(records)
        for record in records:
            for field in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                self.assertIn(field, record)
            prompt = record.get("prompt_tokens")
            cached = record.get("cached_tokens")
            evaluated = record.get("evaluated_tokens")
            if prompt is None or cached is None:
                # Never fabricated when the backend does not report it.
                self.assertIsNone(evaluated)
            else:
                self.assertEqual(evaluated, max(0, prompt - cached))


if __name__ == "__main__":
    unittest.main()
