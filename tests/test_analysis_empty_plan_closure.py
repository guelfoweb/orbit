"""Part A of ANALYSIS-LARGE-SOURCE-RESOLUTION-1: never terminate silently.

A finalizing, non-cancelled autonomous run must always produce a truthful
closing result -- including a run that did nothing at all (no step, no
coverage, an empty plan). Before this, the finalize guard `(steps or
covered_calls)` let an oversized-source + empty-plan run return no report, so
the analyst was handed a stop reason and silence. These tests pin the
invariant and its truthful cause attribution, with no model call and no
fabricated finding. They use scripted fakes only; no Ornith, no sandbox code
runs.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.backend.llama_server import LlamaServerError  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    NO_EVIDENCE_REPORT,
    PLAN_TOOL_NAME,
    FINISH_TOOL_NAME,
    SOURCE_TOO_LARGE_REPORT,
    STOP_BACKEND_ERROR,
    STOP_CANCELLED,
    STOP_LEDGER_EXHAUSTED,
    ANALYSIS_TOOL_NAME,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

CTX = 8192
# Just above the admissible budget once the cover reserve is added: the backend
# below counts 40 + chars*0.25 tokens, so ~40 KiB of source cannot be covered.
OVERSIZED = b"// padding line that is not a comment-marker test\n" * 900
# Comfortably coverable.
SMALL = b"print('hello world')\n"


class _ScriptedModel:
    """Answers by offered tool. `plan` is the question list PLAN returns."""

    def __init__(self, plan, *, prose="", fail_tools=None, fail=None):
        self.plan = plan
        self.prose = prose
        self.fail_tools = fail_tools  # e.g. [] to fail the no-tools report call
        self.fail = fail
        self.actions = 0

    def reply(self, tools):
        names = [t["function"]["name"] for t in (tools or [])]
        if self.fail is not None and self.fail_tools is not None:
            if (tools or []) == [] and self.fail_tools == []:
                raise self.fail
            if ANALYSIS_TOOL_NAME in names and ANALYSIS_TOOL_NAME in (
                t["function"]["name"] for t in self.fail_tools
            ):
                raise self.fail
        if PLAN_TOOL_NAME in names:
            return self._call(PLAN_TOOL_NAME, {"questions": self.plan})
        if FINISH_TOOL_NAME in names:
            return self._call(FINISH_TOOL_NAME,
                              {"status": "resolved", "answer_summary": "x"})
        if ANALYSIS_TOOL_NAME in names:
            self.actions += 1
            return self._call(ANALYSIS_TOOL_NAME,
                              {"code": f"print({self.actions})"})
        return []

    def _call(self, name, arguments):
        return [{"id": f"c{name}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}]


class _Backend:
    thinking = False

    def __init__(self, model):
        self.model = model
        self.seen_tools = []

    def supports_exact_context_admission(self):
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * 0.25), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def count_text_tokens(self, text):
        return TokenCount(tokens=len(text.split()), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        tools = kwargs.get("tools")
        self.seen_tools.append(tools)
        calls = self.model.reply(tools)
        return ChatResult(content=self.model.prose, model="m",
                          finish_reason="stop", tool_calls=calls,
                          prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                          prompt_tokens_per_second=None,
                          generation_tokens_per_second=None)

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


class _Case(unittest.TestCase):
    def _runtime(self, model, data):
        self.backend = _Backend(model)
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / "artifact.hta"
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

    def _no_action(self):
        # An action, if ever dispatched, would produce evidence; these tests
        # assert it never is on the empty-plan path, so the stub is a tripwire.
        def boom(**_kwargs):  # pragma: no cover - must never run
            raise AssertionError("no action should run on an empty plan")
        return mock.patch.object(module, "execute_analysis", boom)


class SilentTerminationTests(_Case):
    def test_a1_oversized_empty_plan_produces_a_closing_result(self) -> None:
        """A1: uncovered oversized source + empty PLAN -> a report exists."""
        runtime = self._runtime(_ScriptedModel(plan=[]), OVERSIZED)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertFalse(run.source_covered)
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.initial_questions, 0)
        self.assertIsNotNone(run.final_report)

    def test_a2_result_names_the_oversized_cause(self) -> None:
        """A2: the closing result states analysis was blocked, and why."""
        runtime = self._runtime(_ScriptedModel(plan=[]), OVERSIZED)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        opening = SOURCE_TOO_LARGE_REPORT.split("{", 1)[0].rstrip()
        self.assertTrue(run.final_report.text.lstrip().startswith(opening))
        # The artifact metadata is present and exact.
        self.assertIn(str(len(OVERSIZED)), run.final_report.text)
        self.assertIn(runtime.source.sha256, run.final_report.text)

    def test_a3_no_fabricated_findings_or_iocs(self) -> None:
        """A3: the closing result invents nothing about the artifact."""
        runtime = self._runtime(_ScriptedModel(plan=[]), OVERSIZED)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        text = run.final_report.text
        # No URL/host/behaviour claim; an oversized block asserts only the cause.
        self.assertNotIn("http", text)
        self.assertNotIn("CreateObject", text)
        self.assertIn("not a finding", text)
        # No model call was spent to produce it.
        self.assertEqual(run.final_report.model_calls, 0)

    def test_a4_zero_actions_stays_zero(self) -> None:
        """A4: the closing result did not cause an action to run."""
        runtime = self._runtime(_ScriptedModel(plan=[]), OVERSIZED)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_a5_covered_small_source_behaviour_unchanged(self) -> None:
        """A5: a normal coverable source still covers and reports as before."""
        runtime = self._runtime(_ScriptedModel(plan=[]), SMALL)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertTrue(run.source_covered)
        self.assertGreaterEqual(run.cover_calls, 1)
        self.assertIsNotNone(run.final_report)
        # Covered run is NOT an oversized block.
        opening = SOURCE_TOO_LARGE_REPORT.split("{", 1)[0].rstrip()
        self.assertFalse(run.final_report.text.lstrip().startswith(opening))

    def test_a6_non_empty_plan_path_unchanged(self) -> None:
        """A6: a run that adopts a question and acts still reports normally."""
        model = _ScriptedModel(plan=[{"question": "q", "missing_fact": "f"}])
        runtime = self._runtime(model, SMALL)

        def one_finding(**_kwargs):
            return AnalysisResult(status="ok", code_sha256="c" * 64,
                                  input_sha256="i" * 64, stdout="FINDING",
                                  stderr="", exit_status=0, duration_seconds=0.1)
        with mock.patch.object(module, "execute_analysis", one_finding):
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertGreaterEqual(run.actions_executed, 1)
        self.assertIsNotNone(run.final_report)

    def test_a7_cancellation_still_synthesises_no_report(self) -> None:
        """A7: a cancelled run produces no closing report."""
        class _Cancel(_ScriptedModel):
            def reply(self, tools):
                raise KeyboardInterrupt
        runtime = self._runtime(_Cancel(plan=[]), OVERSIZED)
        run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertTrue(run.cancelled)
        self.assertIsNone(run.final_report)
        self.assertEqual(run.stop_reason, STOP_CANCELLED)

    def test_a8_backend_failure_is_not_labelled_source_too_large(self) -> None:
        """A8: a backend error with no evidence reports 'no evidence', not
        'source too large' -- and a coverable source proves the distinction."""
        class _FailControl(_ScriptedModel):
            def reply(self, tools):
                raise LlamaServerError("upstream down")
        runtime = self._runtime(_FailControl(plan=[]), SMALL)
        run = runtime.run_autonomous("Analyse it.", finalize=True)
        # The control call failed, so the run ends on a backend error with no
        # evidence; the closing result, if any, must not claim oversize.
        if run.final_report is not None:
            opening = SOURCE_TOO_LARGE_REPORT.split("{", 1)[0].rstrip()
            self.assertFalse(
                run.final_report.text.lstrip().startswith(opening))

    def test_a9_not_eligible_source_is_not_called_oversized(self) -> None:
        """A9 (honesty boundary): a binary / non-UTF-8 artifact that cannot be
        covered is NOT_ELIGIBLE, not TOO_LARGE. The closing result must use the
        generic no-evidence line and must never claim the source overflowed the
        context -- coverage never measured it against the window."""
        # A large binary blob (embedded NUL -> decode_artifact returns None ->
        # COVERAGE_NOT_ELIGIBLE), big enough that size is not the point.
        binary = (b"\x00\x01\x02\x03" * 12000)
        runtime = self._runtime(_ScriptedModel(plan=[]), binary)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertFalse(run.source_covered)
        self.assertIsNotNone(run.final_report)
        opening = SOURCE_TOO_LARGE_REPORT.split("{", 1)[0].rstrip()
        self.assertFalse(run.final_report.text.lstrip().startswith(opening))
        self.assertTrue(
            run.final_report.text.lstrip().startswith(NO_EVIDENCE_REPORT))

    def test_a9b_non_attesting_backend_is_not_called_oversized(self) -> None:
        """A9 (honesty boundary): a backend that cannot attest exact tokens
        yields UNADMISSIBLE coverage -- the source was never measured against
        the window, so the closing result must not claim it was oversized."""
        class _NoAdmission(_Backend):
            def supports_exact_context_admission(self):
                return False
        model = _ScriptedModel(plan=[])
        # Build a runtime, then swap in a backend that refuses exact admission.
        runtime = self._runtime(model, OVERSIZED)
        runtime.backend = _NoAdmission(model)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        self.assertFalse(run.source_covered)
        self.assertIsNotNone(run.final_report)
        opening = SOURCE_TOO_LARGE_REPORT.split("{", 1)[0].rstrip()
        self.assertFalse(run.final_report.text.lstrip().startswith(opening))

    def test_a10_report_text_contract_valid(self) -> None:
        """A10: the closing result is a well-formed AnalysisReport."""
        runtime = self._runtime(_ScriptedModel(plan=[]), OVERSIZED)
        with self._no_action():
            run = runtime.run_autonomous("Analyse it.", finalize=True)
        report = run.final_report
        self.assertIsInstance(report.text, str)
        self.assertTrue(report.text.strip())
        self.assertEqual(report.evidence_ids, ())
        self.assertIsInstance(report.model_calls, int)


if __name__ == "__main__":
    unittest.main()
