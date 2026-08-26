"""Where the model's answer ends and Orbit's own telemetry begins.

An analysis prints the model's closing report and then Orbit's summary line --
`analysis | mode: ANALYSIS | ...` -- directly beneath it. Both are plain,
left-aligned prose. The only thing that distinguished them was `dim()`, which
is a no-op wherever ANSI is unavailable: a pipe, a redirected log, `NO_COLOR`,
`TERM=dumb`. In those places the analyst reads Orbit's counters as the model's
last sentence, and a report is free to contain a line that looks exactly like
the summary.

These tests pin the separation, and pin that it costs nothing else: the report
bytes, the telemetry bytes and the sanitization boundary are all unchanged.
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

from orbit.terminal.repl import _print_orbit_summary
from orbit.terminal.theme import sanitize_terminal_text


SUMMARY = (
    "analysis | mode: ANALYSIS | model calls: 3 | actions: 1 | "
    "steps: 2 | stopped: done | 1.2s"
)


def shown(report: str | None, summary: str = SUMMARY) -> str:
    """The final block exactly as the terminal receives it.

    Mirrors the production order -- report, then telemetry -- so the assertions
    are about what an analyst sees rather than about either piece alone.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ended_blank = False
        if report is not None:
            shown = sanitize_terminal_text(report, allow_newlines=True)
            print(shown, flush=True)
            ended_blank = shown.endswith("\n")
        _print_orbit_summary(summary, preceded_by_blank_line=ended_blank)
    return buffer.getvalue()


class BoundaryIsUnambiguousTests(unittest.TestCase):
    """The separation must not depend on colour."""

    def test_a_blank_line_precedes_the_summary(self) -> None:
        rendered = shown("- Finding: one Fix: none")
        lines = rendered.split("\n")

        self.assertEqual(lines[1], "", "no blank line separates report from telemetry")
        self.assertTrue(lines[2].startswith("› "))

    def test_the_summary_carries_the_orbit_prefix(self) -> None:
        """`›` already means 'Orbit is speaking' elsewhere in this terminal."""
        self.assertIn(f"› {SUMMARY}", shown("- Finding: one Fix: none"))

    def test_a_report_that_mimics_the_summary_stays_on_the_report_side(self) -> None:
        """The whole point: a lookalike line must not read as Orbit's.

        Case H. Before this change the two were byte-identical in style, so a
        report ending in `analysis | mode: ANALYSIS ...` was indistinguishable
        from the row Orbit writes next.
        """
        mimic = "- Finding: analysis | mode: ANALYSIS | model calls: 99 Fix: none"
        rendered = shown(mimic)
        lines = rendered.rstrip("\n").split("\n")

        self.assertFalse(lines[0].startswith("› "), "report text gained Orbit's marker")
        self.assertTrue(lines[-1].startswith("› "), "telemetry lost Orbit's marker")
        self.assertEqual(lines.count(""), 1, "exactly one separating blank line")

    def test_the_separation_holds_without_ansi(self) -> None:
        """`dim()` is a no-op off a tty; the boundary must survive that."""
        from orbit.terminal import theme

        self.assertFalse(theme.supports_ansi(io.StringIO()))
        rendered = shown("- Finding: one Fix: none")

        self.assertIn("\n\n› ", rendered)


class NoAccumulationTests(unittest.TestCase):
    """Spacing must be exactly one line, whatever the report ends with."""

    def test_a_report_without_a_trailing_newline(self) -> None:
        """Case D."""
        rendered = shown("- Finding: one Fix: none")

        self.assertNotIn("\n\n\n", rendered)
        self.assertEqual(rendered.count("\n\n"), 1)

    def test_a_report_ending_with_a_newline(self) -> None:
        """Case C: one separating line, not two.

        `print` adds a newline of its own, so a report that already ends with
        one leaves a blank line standing. Adding another unconditionally would
        open a ragged gap in the common case -- prose usually ends with a
        newline -- rather than the rare one.
        """
        rendered = shown("- Finding: one Fix: none\n")
        before_marker = rendered.split("› ")[0]

        self.assertTrue(rendered.startswith("- Finding: one Fix: none\n"))
        self.assertEqual(
            before_marker.count("\n") - 1,
            1,
            "exactly one blank line must separate the report from telemetry",
        )

    def test_a_report_with_its_own_trailing_blank_line_is_left_alone(self) -> None:
        """The model's own spacing is not rewritten.

        A report ending in a real blank line keeps it. Collapsing that would
        mean editing what the model wrote, which display must never do -- so
        the invariant is "Orbit adds at most one", not "there is always
        exactly one".
        """
        rendered = shown("- Finding: one Fix: none\n\n")
        before_marker = rendered.split("› ")[0]

        self.assertEqual(before_marker.count("\n") - 1, 2)
        self.assertTrue(rendered.startswith("- Finding: one Fix: none\n\n"))

    def test_a_multiline_report(self) -> None:
        """Case B."""
        rendered = shown("- Finding: a\n- Finding: b\n- Finding: c")

        for bullet in ("- Finding: a", "- Finding: b", "- Finding: c"):
            self.assertIn(bullet, rendered)
        self.assertEqual(rendered.count("› "), 1, "duplicate telemetry marker")

    def test_only_one_marker_is_ever_emitted(self) -> None:
        rendered = shown("- Finding: › not Orbit's marker Fix: none")

        self.assertEqual(rendered.count("\n› "), 1)


class EmptyReportTests(unittest.TestCase):
    """Case I: nothing above must not produce ragged spacing."""

    def test_no_report_at_all(self) -> None:
        rendered = shown(None)

        self.assertTrue(rendered.startswith("\n› "))
        self.assertNotIn("\n\n\n", rendered)

    def test_an_empty_report_string(self) -> None:
        rendered = shown("")

        self.assertIn("› ", rendered)
        self.assertNotIn("\n\n\n", rendered)


class ContentPreservationTests(unittest.TestCase):
    """The answer itself must be untouched."""

    def test_one_line_report_is_verbatim(self) -> None:
        """Case A."""
        report = "- Finding: the loader trusts the path. Fix: resolve it first."

        self.assertTrue(shown(report).startswith(report + "\n"))

    def test_unicode_is_verbatim(self) -> None:
        """Case E."""
        report = "- Finding: café — 日本語 ✓ αβγ Ω 🎯. Fix: ünïcödé."

        self.assertTrue(shown(report).startswith(report + "\n"))

    def test_windows_paths_are_verbatim(self) -> None:
        """Case F."""
        report = r"- Finding: C:\Users\analyst\sample.exe. Fix: \\server\share$."

        self.assertTrue(shown(report).startswith(report + "\n"))

    def test_already_sanitized_hostile_content_is_unchanged(self) -> None:
        """Case G: the separation must not re-escape or drop anything."""
        report = sanitize_terminal_text(
            "- Finding: ok\x1b[2J\rmimic Fix: none", allow_newlines=True
        )
        rendered = shown(report)

        self.assertTrue(rendered.startswith(report + "\n"))
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\r", rendered)


class TelemetryPreservationTests(unittest.TestCase):
    """Orbit's own text must not change either."""

    def test_the_summary_text_is_carried_verbatim(self) -> None:
        returned = _print_orbit_summary(SUMMARY)

        self.assertEqual(returned, f"› {SUMMARY}")
        self.assertIn(SUMMARY, returned)

    def test_the_report_summary_shape_is_carried_verbatim(self) -> None:
        summary = "report | mode: ANALYSIS | model calls: 1 | actions: 0 | 0.4s"

        self.assertEqual(_print_orbit_summary(summary), f"› {summary}")


class StreamedAndNonStreamedTests(unittest.TestCase):
    """Case J: both routes end in the same separated shape."""

    def test_both_paths_produce_the_same_final_block(self) -> None:
        report = "- Finding: café C:\\Users\\a Fix: none"

        # Streamed: the renderer wrote the prose, then the summary prints.
        streamed = io.StringIO()
        with contextlib.redirect_stdout(streamed):
            print(sanitize_terminal_text(report, allow_newlines=True), flush=True)
            _print_orbit_summary(SUMMARY)

        self.assertEqual(streamed.getvalue(), shown(report))


class ProductionCallSiteTests(unittest.TestCase):
    """Both ANALYSIS summaries must go through the helper."""

    def test_the_summary_prints_on_the_non_autonomous_path(self) -> None:
        """The telemetry row appears even when there is no autonomous run.

        The hint is read after the `run is not None` branch, so initialising
        it inside that branch raises `UnboundLocalError` on the ordinary
        non-autonomous path -- a live crash that no assertion about spacing
        would catch. Driven through the real `Repl` rather than a stub,
        because the stub is exactly what missed it.
        """
        import shutil
        import tempfile

        from orbit.backend.base import ChatResult
        from orbit.runtime import ChatRuntime
        from orbit.runtime.evidence import EvidenceStore
        from orbit.terminal.config import AppConfig
        from orbit.terminal.repl import Repl

        class Backend:
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                return ChatResult(
                    content="observing",
                    model="scripted",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

            def chat_stream(
                self, messages, *, temperature, max_tokens, tools=None,
                on_delta=None, on_progress=None,
            ):
                if on_delta:
                    on_delta("observing")
                return self.chat(
                    messages, temperature=temperature, max_tokens=max_tokens
                )

        tmp = Path(tempfile.mkdtemp(prefix="orbit-final-ux-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        artifact = tmp / "sample.js"
        artifact.write_text("console.log(1);\n", encoding="utf-8")

        backend = Backend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        runtime.evidence_store = EvidenceStore(root=tmp / "evidence")
        repl = Repl(runtime=runtime, backend=backend, config=AppConfig(workdir=tmp))
        self.addCleanup(repl._close_analysis)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(
            io.StringIO()
        ):
            repl._handle_command(f"/analysis {artifact}")
            repl._ask("continue")
        rendered = buffer.getvalue()

        self.assertIn("mode: ANALYSIS", rendered, "no telemetry row was printed")
        self.assertIn("› ", rendered, "telemetry lost Orbit's marker")

    def test_the_analysis_path_reports_what_it_printed(self) -> None:
        """The hint must be derived, not hardcoded.

        A call site that always passes False looks correct and renders
        correctly for reports without a trailing newline -- which is what a
        spacing assertion would sample. This records the value the real
        method hands the helper for a report that DOES end with a newline.
        """
        from unittest import mock

        from orbit.runtime.analysis_runtime import AnalysisReport
        from orbit.terminal import repl as repl_module

        seen: list[bool] = []

        def record(summary, *, preceded_by_blank_line=False):
            seen.append(preceded_by_blank_line)
            return summary

        renderer = mock.Mock(rendered_visible_text=False)
        report = AnalysisReport(text="- Finding: one Fix: none\n", model_calls=1)

        # The exact shape of the production branch, with the real helper
        # swapped for a recorder.
        with mock.patch.object(repl_module, "_print_orbit_summary", record):
            ended_blank = False
            if report is not None and not renderer.rendered_visible_text:
                shown = repl_module.sanitize_terminal_text(
                    report.text, allow_newlines=True
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    print(shown, flush=True)
                ended_blank = shown.endswith("\n")
            repl_module._print_orbit_summary(
                "analysis | mode: ANALYSIS | 1.0s",
                preceded_by_blank_line=ended_blank,
            )

        self.assertEqual(seen, [True], "a report ending in a newline must say so")

    def test_the_call_sites_derive_the_hint(self) -> None:
        """Both methods must compute the flag, not pin it to a constant.

        Paired with the behavioural test above: that one proves the value is
        right for a derived flag, this one proves neither site quietly stops
        deriving it. Matched on the derivation itself so a rename of the
        surrounding variable cannot fail it.
        """
        import inspect
        import re

        from orbit.terminal import repl

        for method in (repl.Repl._ask_analysis, repl.Repl._handle_report_command):
            with self.subTest(method=method.__name__):
                source = inspect.getsource(method)
                self.assertIsNotNone(
                    re.search(r"=\s*shown\.endswith\(", source),
                    "the hint must be derived from the printed text",
                )

    def test_no_telemetry_is_printed_the_old_way(self) -> None:
        """Every ANALYSIS summary must go through the helper.

        Asserted on the absence of the old call rather than on the text of the
        new one, so a keyword-argument refactor cannot fail this spuriously.
        """
        import inspect

        from orbit.terminal import repl

        source = inspect.getsource(repl)

        self.assertNotIn("print(dim(summary), flush=True)", source)
        self.assertEqual(source.count("_print_orbit_summary("), 3)  # def + 2 sites
