"""Terminal presentation of canonical reports, through the real REPL/config."""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from orbit.runtime.analysis_runtime import (
    ANALYSIS_REPORT_PHASE, AnalysisProgressEvent, AnalysisReport, AutonomousRunResult,
)
from orbit.terminal.analysis_mode import AnalysisProgressDisplay
from orbit.runtime.workflow_mode import WorkflowMode
from orbit.terminal.config import add_config_arguments, load_app_config
from orbit.terminal.repl import Repl
from orbit.terminal.streaming import StreamRenderer
from tests.test_analysis_report_visibility import ReportVisibilityTestBase, _ReportBackend


REPORT = """# Analysis report

## Summary
One **attested** destination; café.

## Technical behaviour
Static evidence only.

## Limits
No execution observed.

## IoC / Evidence
- URL: https://example.invalid/a?one=1&two=2
- State: `RESOLVED_EXACT`
"""


class Terminal(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class RuntimeReportPresentationTests(ReportVisibilityTestBase):
    def test_report_starts_a_new_document_after_step_prose(self):
        for prose in ("Reading source now.", "Reading **source now", "```text\nreading source"):
            with self.subTest(prose=prose):
                analysis = self._analysis("var x = 1;", backend=_ReportBackend(prose))
                self._finding(analysis)
                repl = self._repl(analysis)
                repl.autonomous_analysis = True
                output = Terminal(True)
                with mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=True), \
                        contextlib.redirect_stdout(output), \
                        mock.patch.object(StreamRenderer, "start"), \
                        mock.patch("orbit.terminal.repl._print_orbit_summary"):
                    repl._ask_analysis("Analyse the artifact")
                self.assertIn("\033[1m\033[36mAnalysis report", output.getvalue())
                self.assertNotIn("# Analysis report", output.getvalue())
                self.assertEqual(output.getvalue().count("Analysis report"), 1)
                self.assertTrue(analysis.last_report.text.startswith("# Analysis report\n"))
                self.assertEqual(analysis.actions_executed, 0)

    def test_real_zero_call_runtime_publishes_rendered_view_and_canonical_session(self):
        for path in ("autonomous", "report"):
            with self.subTest(path=path):
                analysis = self._analysis("location.href='https://fixture.invalid/resource';")
                repl = self._repl(analysis)
                repl.autonomous_analysis = True
                output = Terminal(True)
                with mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=True), \
                        contextlib.redirect_stdout(output), \
                        mock.patch.object(StreamRenderer, "start"), \
                        mock.patch("orbit.terminal.repl._print_orbit_summary"), \
                        mock.patch.object(repl, "_save_session") as save:
                    if path == "autonomous":
                        repl._ask_analysis("Identify network destinations")
                    else:
                        repl._handle_report_command("Identify network destinations")
                report = analysis.last_report
                self.assertIn("\033[1m\033[36mAnalysis report", output.getvalue())
                self.assertNotIn("# Analysis report", output.getvalue())
                self.assertIn("# Analysis report", report.text)
                self.assertNotIn("\033", report.text)
                self.assertIn("https://fixture.invalid/resource", output.getvalue())
                self.assertEqual(output.getvalue().count("Analysis report"), 1)
                save.assert_called_once_with(analysis_report=asdict(report))
                self.assertEqual(analysis.model_calls, 0)
                self.assertEqual(analysis.actions_executed, 0)


class ReportPresentationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="orbit-report-markdown-")
        self.addCleanup(folder.cleanup)
        self.config_path = str(Path(folder.name) / "absent.json")

    def config(self):
        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        return load_app_config(parser.parse_args([
            "--config", self.config_path, "--think", "off", "--tools", "off",
        ]))

    def present(self, path, *, tty=True, env=None, stream=True, text=REPORT):
        report = AnalysisReport(text=text, model_calls=0, narrative_status="not_requested",
                                dossier_text="Complete retained dossier: unchanged.")
        original = asdict(report)

        def deliver(*args, on_delta=None, **kwargs):
            if stream:
                on_delta(report.text)
            return report

        def autonomous(*args, **kwargs):
            return AutonomousRunResult(steps=(), progress=(), stop_reason="fixture stop",
                                       model_calls=0, actions_executed=0,
                                       final_report=deliver(*args, **kwargs))

        analysis = SimpleNamespace(messages=[], analyst_turns=0,
                                   report=deliver, run_autonomous=autonomous)
        output = Terminal(tty)
        # Only the timer and footer are excluded: no background timing or
        # unrelated telemetry escapes in assertions about the report body.
        with mock.patch.dict(os.environ, {"TERM": "xterm", **(env or {})}, clear=True), \
                contextlib.redirect_stdout(output), \
                mock.patch.object(StreamRenderer, "start"), \
                mock.patch("orbit.terminal.repl._print_orbit_summary"):
            repl = Repl(runtime=SimpleNamespace(), backend=SimpleNamespace(),
                        config=self.config(), analysis=analysis,
                        workflow_mode=WorkflowMode.ANALYSIS, autonomous_analysis=True)
            with mock.patch.object(repl, "_save_session") as save:
                if path == "autonomous":
                    repl._ask_analysis("Original analyst request")
                else:
                    repl._handle_report_command("Original analyst request")
            save.assert_called_once_with(analysis_report=original)
        self.assertEqual(asdict(report), original)
        self.assertEqual(report.text.encode("utf-8"), text.encode("utf-8"))
        self.assertEqual(output.getvalue().count("Analysis report"), 1)
        return output.getvalue()

    def test_interactive_report_uses_live_markdown(self):
        for path in ("autonomous", "report"):
            with self.subTest(path=path):
                output = self.present(path)
                self.assertIn("\033[1m\033[36mAnalysis report", output)
                self.assertNotIn("# Analysis report", output)
                self.assertIn("https://example.invalid/a?one=1&two=2", output)

    def test_non_streaming_report_still_renders(self):
        for path in ("autonomous", "report"):
            with self.subTest(path=path):
                self.assertIn("\033[", self.present(path, stream=False))

    def test_redirect_is_raw_even_with_live_enabled(self):
        self.assert_raw(tty=False, env={"ORBIT_RENDER_MARKDOWN": "live"})

    def test_no_color_is_raw_including_empty_environment_value(self):
        self.assert_raw(env={"NO_COLOR": ""})

    def test_dumb_terminal_is_raw(self):
        self.assert_raw(env={"TERM": "dumb"})

    def test_all_explicit_plain_settings_are_raw(self):
        for value in ("plain", "off", "0", "false"):
            with self.subTest(value=value):
                self.assert_raw(env={"ORBIT_RENDER_MARKDOWN": value})

    def assert_raw(self, **kwargs):
        for path in ("autonomous", "report"):
            for stream in (True, False):
                with self.subTest(path=path, stream=stream):
                    output = self.present(path, stream=stream, **kwargs)
                    self.assertIn(REPORT, output)
                    self.assertNotIn("\033", output)

    def test_artifact_controls_remain_inert_and_canonical_text_is_untouched(self):
        hostile = REPORT + "\x1b[2J\x1b]52;c;secret\x07\rforged"
        for path in ("autonomous", "report"):
            for stream in (True, False):
                with self.subTest(path=path, stream=stream):
                    output = self.present(path, stream=stream, text=hostile)
                    for control in ("\x1b[2J", "\x1b]52;", "\x07", "\r"):
                        self.assertNotIn(control, output)

    def test_normal_chat_final_keeps_its_existing_renderer(self):
        text = "# Chat final\n\n**Supported** answer.\n"

        def answer(*args, on_final_delta, **kwargs):
            on_final_delta(text)
            return SimpleNamespace(content=text)

        for mode in ("live", "plain"):
            with self.subTest(mode=mode), \
                    mock.patch.dict(os.environ, {"TERM": "xterm", "ORBIT_RENDER_MARKDOWN": mode}, clear=True), \
                    mock.patch.object(StreamRenderer, "start"):
                output = Terminal(True)
                runtime = SimpleNamespace(messages=[], last_analysis_request=None, ask_chat=answer)
                repl = Repl(runtime=runtime, backend=SimpleNamespace(), config=self.config())
                with contextlib.redirect_stdout(output), \
                        mock.patch.object(repl, "_save_session"), \
                        mock.patch.object(repl, "_print_turn_footer"):
                    repl._ask("A normal chat question")
                if mode == "live":
                    self.assertIn("\033[1m\033[36mChat final", output.getvalue())
                    self.assertNotIn("# Chat final", output.getvalue())
                else:
                    self.assertIn(text, output.getvalue())
                    self.assertNotIn("\033", output.getvalue())

    def test_report_event_resets_document_without_completed_step(self):
        # A failed step need not reach on_step. The runtime-owned report
        # event must still isolate the report even with progress lines hidden.
        output = Terminal(True)
        with mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=True), \
                contextlib.redirect_stdout(output):
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("```text\nunfinished step")
            display = AnalysisProgressDisplay(renderer, interactive=False)
            display(AnalysisProgressEvent(ANALYSIS_REPORT_PHASE, "report"))
            self.assertFalse(renderer.rendered_visible_text)
            renderer.write(REPORT)
            renderer.finish()
        self.assertIn("\033[1m\033[36mAnalysis report", output.getvalue())
        self.assertNotIn("# Analysis report", output.getvalue())


if __name__ == "__main__":
    unittest.main()
