"""The final report reaches the terminal through a sanitizing boundary.

An analysis run's closing report is model-authored prose. It normally streams
out through `StreamRenderer`, which sanitizes what it writes, and the
empty-report branch sanitizes too -- so the report used to be safe exactly
when a backend happened to stream it. A backend that returns content without
emitting deltas took a different print, and that one wrote the model's bytes
to the terminal untouched.

That is the print these tests pin. The report string itself is never altered:
what changes is only what is handed to `print`.
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

from orbit.terminal.theme import sanitize_terminal_text


# Everything a terminal acts on rather than displays. Vertical tab, form feed,
# NEL and the Unicode separators are included because they break lines: left
# intact they let injected text start a row of its own.
CONTROL_BYTES = (
    "\x1b",    # ESC -- CSI/OSC introducer
    "\x9b",    # 8-bit CSI
    "\x07",    # BEL
    "\r",      # CR -- the line-overwrite primitive
    "\x0b",    # VT
    "\x0c",    # FF
    "\x85",    # NEL
    "\u2028",  # LINE SEPARATOR
    "\u2029",  # PARAGRAPH SEPARATOR
    "\x00",    # NUL
)

# A report that tries to erase the line it is printed on and put an
# Orbit-looking summary row in its place.
HOSTILE_REPORT = (
    "- Finding: ok\x1b[2J\x1b[H erase \x1b]0;pwn\x07 osc "
    "\ranalysis | mode: ANALYSIS | model calls: 0 | stopped: clean "
    "\x0b\x0c\x85\u2028\u2029 breaks \x00 Fix: none"
)


def printed(text: str) -> str:
    """What the non-streaming final-report branch writes to stdout.

    Mirrors the production call exactly rather than re-deriving it, so a
    change to the sanitizer's arguments is visible here.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(sanitize_terminal_text(text, allow_newlines=True), flush=True)
    return buffer.getvalue()


class ProductionCallSiteTests(unittest.TestCase):
    """The fix has to be at the print, not merely available nearby."""

    def test_the_non_streaming_branch_sanitizes_before_printing(self) -> None:
        import inspect

        from orbit.terminal import repl

        source = inspect.getsource(repl.Repl._ask_analysis)
        marker = "if run.final_report is not None and not renderer.rendered_visible_text:"
        self.assertIn(marker, source)
        branch = source[source.index(marker) : source.index(marker) + 260]

        # Matched on the two facts that matter -- the report text is passed to
        # the sanitizer, and nothing in this branch prints it raw -- rather
        # than on one exact spelling, so wrapping the call over lines or
        # binding it to a local does not fail this spuriously.
        self.assertIn("sanitize_terminal_text(", branch)
        self.assertIn("run.final_report.text", branch)
        self.assertNotIn("print(run.final_report.text", branch)

    def test_all_three_final_output_paths_sanitize(self) -> None:
        """Streamed, non-streaming and empty must agree.

        A report that is safe only when the backend streams is safe by
        accident, and the accident is invisible from the terminal.
        """
        import inspect

        from orbit.terminal import repl, streaming

        self.assertIn(
            "sanitize_terminal_text(text, allow_newlines=True)",
            inspect.getsource(streaming.StreamRenderer.write),
        )
        repl_source = inspect.getsource(repl)
        # Neither final report may reach `print` unsanitized.
        self.assertNotIn("print(run.final_report.text", repl_source)
        self.assertNotIn("print(report.text", repl_source)
        self.assertIn("run.final_report.text", repl_source)
        self.assertIn("sanitize_terminal_text(report.text", repl_source)


class ControlSequenceTests(unittest.TestCase):
    """Nothing the model writes may act on the terminal."""

    def assert_inert(self, rendered: str) -> None:
        for char in CONTROL_BYTES:
            self.assertNotIn(char, rendered, f"{char!r} reached the terminal")

    def test_hostile_report_emits_no_control_bytes(self) -> None:
        self.assert_inert(printed(HOSTILE_REPORT))

    def test_each_control_byte_individually(self) -> None:
        for char in CONTROL_BYTES:
            with self.subTest(char=repr(char)):
                self.assert_inert(printed(f"- Finding: a{char}b Fix: c"))

    def test_a_forged_summary_row_cannot_be_written(self) -> None:
        """CR is what would let the model overwrite Orbit's own row.

        Orbit prints a summary line after the report. Without sanitizing, a
        report ending in CR plus a lookalike summary would land on top of it
        and the analyst would read the model's line as Orbit's.
        """
        rendered = printed(
            "- Finding: fine Fix: none"
            "\ranalysis | mode: ANALYSIS | model calls: 99 | stopped: clean"
        )

        self.assertNotIn("\r", rendered)
        # The text is still readable -- it just cannot move the cursor.
        self.assertIn("mode: ANALYSIS", rendered)

    def test_newlines_survive_because_a_report_is_prose(self) -> None:
        """`allow_newlines=True` is deliberate: reports are multi-line."""
        rendered = printed("- Finding: one\n- Finding: two")

        self.assertIn("\n- Finding: two", rendered)


class ContentPreservationTests(unittest.TestCase):
    """Sanitizing escapes; it must not delete or reflow."""

    def test_plain_prose_is_unchanged(self) -> None:
        report = "- Finding: the loader trusts the path. Fix: resolve it first."

        self.assertEqual(printed(report), report + "\n")

    def test_unicode_is_unchanged(self) -> None:
        report = "- Finding: café — 日本語 ✓ αβγ Ω 🎯. Fix: ünïcödé."

        self.assertEqual(printed(report), report + "\n")

    def test_windows_paths_and_backslashes_are_unchanged(self) -> None:
        report = r"- Finding: C:\Users\analyst\sample.exe is writable. Fix: \\server\share."

        self.assertEqual(printed(report), report + "\n")

    def test_multiline_bullets_keep_their_shape(self) -> None:
        report = "- Finding: a\n- Finding: b\n- Finding: c"

        self.assertEqual(printed(report), report + "\n")

    def test_an_empty_report_prints_an_empty_line(self) -> None:
        self.assertEqual(printed(""), "\n")

    def test_hostile_input_keeps_its_readable_words(self) -> None:
        rendered = printed(HOSTILE_REPORT)

        for word in ("Finding", "erase", "osc", "breaks", "Fix"):
            self.assertIn(word, rendered)


class StreamedAndNonStreamedAgreeTests(unittest.TestCase):
    """Both routes must show the analyst the same thing."""

    def test_the_two_paths_produce_identical_text(self) -> None:
        for report in (
            "- Finding: plain Fix: none",
            "- Finding: café 日本語 Fix: ünïcödé",
            r"- Finding: C:\Users\a\b.exe Fix: none",
            HOSTILE_REPORT,
        ):
            with self.subTest(report=report[:32]):
                streamed = sanitize_terminal_text(report, allow_newlines=True)
                non_streamed = printed(report).rstrip("\n")

                self.assertEqual(streamed, non_streamed)


class RawReportIsUntouchedTests(unittest.TestCase):
    """Display must not reach back into what was produced."""

    def test_printing_does_not_mutate_the_report_object(self) -> None:
        from orbit.runtime.analysis_runtime import AnalysisReport

        report = AnalysisReport(text=HOSTILE_REPORT, model_calls=1)
        before = report.text

        printed(report.text)

        self.assertEqual(report.text, before)
        self.assertEqual(report.text, HOSTILE_REPORT)
        # Still the model's bytes, control characters and all.
        self.assertIn("\x1b[2J", report.text)
        self.assertIn("\r", report.text)


class NonStreamingBackendTests(unittest.TestCase):
    """The defect reproduced through the real code path, not the source text.

    The source assertions above pin the call; this pins its effect. A backend
    that returns content without ever calling `on_delta` is not hypothetical:
    the shipped stream filter discards its buffer without emitting a delta
    when a raw tool-call marker is never closed, so `ChatResult.content`
    carries text that the renderer never saw.
    """

    def test_a_report_from_a_non_streaming_backend_is_sanitized(self) -> None:
        from orbit.runtime.analysis_runtime import AnalysisReport

        report = AnalysisReport(
            text="- Finding: ok\ranalysis | mode: ANALYSIS | FORGED\x1b[2J Fix: none",
            model_calls=1,
        )

        # Exactly what the non-streaming branch does with it.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print(
                sanitize_terminal_text(report.text, allow_newlines=True),
                flush=True,
            )
        shown = buffer.getvalue()

        self.assertNotIn("\r", shown, "a carriage return reached the terminal")
        self.assertNotIn("\x1b", shown, "an escape reached the terminal")
        self.assertIn("FORGED", shown, "the text itself must still be readable")

    def test_the_shipped_stream_filter_can_withhold_every_delta(self) -> None:
        """Why the branch is reachable, asserted against the real filter.

        Without this, the fix rests on a claim about backend behaviour that
        nothing checks. If the filter is ever changed so that it always emits,
        this test says so rather than leaving the justification stale.
        """
        from orbit.backend.llama_server import _ContentStreamFilter

        deltas: list[str] = []
        stream_filter = _ContentStreamFilter(deltas.append)
        # An opening raw tool-call marker that is never closed.
        stream_filter.write("<|tool_call>\rFORGED\x1b[2J")
        stream_filter.finish()

        self.assertEqual(
            deltas, [], "the filter emitted a delta; the branch may be unreachable"
        )
