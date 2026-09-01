"""Rendering a finished report changes how it looks and nothing else.

The canonical `report.text` is what APIs, saved sessions, files and pipes
read, so the two properties that matter most here are that it is never
touched, and that a terminal-bound reader gets structure without ever getting
an escape the artifact chose.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import unittest
from unittest import mock

from orbit.terminal.markdown_report import render_report

ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")

SAMPLE = """## Findings

**Bold claim** and `inline code`.

- first bullet
- second bullet

1. first step
2. second step

## Verified indicators

- uri: http://a.invalid/1.php?s=k
  evidence: ev_26e1592c3e90_6a892b624a2a237c
"""


def visible(text: str) -> str:
    """The text with every escape removed -- what a reader actually reads."""
    return ANSI.sub("", text)


class CanonicalTextTests(unittest.TestCase):
    """Rendering is a view. The report itself is never rewritten."""

    def test_the_input_string_is_not_mutated(self) -> None:
        original = SAMPLE
        render_report(SAMPLE, force_style=True)
        self.assertEqual(SAMPLE, original)

    def test_styling_preserves_every_visible_character(self) -> None:
        """Structure is shown, not edited: strip the escapes and it is the
        same text, marker for marker."""
        self.assertEqual(visible(render_report(SAMPLE, force_style=True)), SAMPLE)

    def test_plain_mode_returns_the_text_unchanged(self) -> None:
        self.assertEqual(render_report(SAMPLE, force_style=False), SAMPLE)

    def test_an_empty_report_stays_empty(self) -> None:
        self.assertEqual(render_report("", force_style=True), "")


class ByteIdentityPropertyTests(unittest.TestCase):
    """The invariant the whole design rests on, checked over a corpus.

    Rendering may add escapes and may add nothing else. Strip the escapes and
    the sanitised input must come back exactly -- otherwise the terminal shows
    an analyst a different byte sequence from the canonical `report.text`,
    which is the one thing this must never do.
    """

    CORPUS = (
        # Markers separated by more than one space, or by a tab. A renderer
        # that re-emits a hardcoded separator collapses these silently.
        "1.  Fetches a payload",
        "10. tenth item",
        "##   Findings",
        "-\tTabbed bullet",
        "*  star bullet",
        "+   plus bullet",
        "1)  paren numbered",
        # Four-space indentation is Markdown's literal code block: the
        # whitespace IS the content.
        "    -  literal code, not a bullet",
        "\t\tdeeply indented",
        # Inline spans in every awkward arrangement.
        "**a `b` c** and `d **e** f`",
        "a ** unmatched",
        "a ` unmatched",
        "***triple***",
        "****",
        "`` ``",
        r"escaped \*not bold\*",
        # Headings and fences at their edges.
        "#NoSpaceAfterHash",
        "####### seven hashes",
        "## Trailing hashes ##",
        "````\nfour tick fence\n````",
        "~~~\ntilde fence\n~~~",
        "    ```\n    indented fence\n    ```",
        "```\n  key: value inside a fence\n```",
        # Provenance-shaped lines and ids.
        "  authority: gibuzuy37v2v.top",
        "  Mixed_Case: value",
        "ev_26e1592c3e90_6a892b624a2a237c",
        "xev_26e1592c3e90_6a892b624a2a237c_extra",
        "ev_short_id",
        # Line endings and unusual text.
        "## A\r\n- b\r\n",
        "trailing spaces   \n",
        "\n\n\n",
        "- 🔥 emoji **bold**",
        "a\u200bzero width",
        "",
        SAMPLE,
    )

    def test_stripping_the_escapes_returns_the_sanitised_input(self) -> None:
        from orbit.terminal.theme import sanitize_terminal_text

        for text in self.CORPUS:
            with self.subTest(text=text[:40]):
                rendered = render_report(text, force_style=True)
                self.assertEqual(
                    visible(rendered),
                    sanitize_terminal_text(text, allow_newlines=True),
                )

    def test_plain_mode_matches_the_sanitiser_exactly(self) -> None:
        from orbit.terminal.theme import sanitize_terminal_text

        for text in self.CORPUS:
            with self.subTest(text=text[:40]):
                self.assertEqual(
                    render_report(text, force_style=False),
                    sanitize_terminal_text(text, allow_newlines=True),
                )

    def test_every_style_opened_is_closed(self) -> None:
        """A style left open bleeds into whatever the terminal prints next."""
        for text in self.CORPUS:
            with self.subTest(text=text[:40]):
                rendered = render_report(text, force_style=True)
                for line in rendered.split("\n"):
                    opens = len(re.findall(r"\033\[(?:1|2|33|36)m", line))
                    closes = line.count("\033[0m")
                    if not opens:
                        continue
                    # Every attribute is closed, and the last escape on the
                    # line is a reset -- so nothing bleeds into what the
                    # terminal prints next. Counts rather than a suffix check,
                    # because a line may legitimately end in unstyled text.
                    self.assertGreaterEqual(
                        closes, 1, f"a styled line must reset: {line!r}"
                    )
                    last = re.findall(r"\033\[[0-9;]*m", line)[-1]
                    self.assertEqual(
                        last, "\033[0m", f"style left open: {line!r}"
                    )


class StructureTests(unittest.TestCase):
    def _styled(self, text: str) -> str:
        return render_report(text, force_style=True)

    def test_a_heading_is_styled_and_keeps_its_hashes(self) -> None:
        rendered = self._styled("## Findings\n")
        self.assertIn("\033[1m", rendered)
        self.assertIn("## Findings", visible(rendered))

    def test_runtime_sections_are_styled_apart_from_narrative(self) -> None:
        """Same text, different voice: one half is written from evidence by
        the runtime, the other is the model's prose."""
        runtime = self._styled("## Verified indicators\n")
        narrative = self._styled("## Findings\n")

        self.assertIn("\033[33m", runtime)      # amber
        self.assertIn("\033[36m", narrative)    # cyan
        self.assertNotIn("\033[33m", narrative)
        self.assertIn("## Verified indicators", visible(runtime))

    def test_the_transformation_appendix_is_styled_the_same_way(self) -> None:
        rendered = self._styled("## Deterministic transformations\n")
        self.assertIn("\033[33m", rendered)

    def test_more_than_six_hashes_is_not_a_heading(self) -> None:
        """Markdown stops at six. Styling a seventh would format prose that
        merely begins with hashes as a section title."""
        rendered = self._styled("####### seven hashes\n")
        self.assertNotIn("\033[1m", rendered)
        self.assertEqual(visible(rendered), "####### seven hashes\n")

    def test_six_hashes_is_still_a_heading(self) -> None:
        self.assertIn("\033[1m", self._styled("###### six\n"))

    def test_top_level_prose_with_a_colon_is_not_a_provenance_field(self) -> None:
        """The dim `key:` styling belongs to the runtime's own indented
        provenance lines. Applying it to any sentence containing a colon would
        dim the first words of ordinary narrative."""
        rendered = self._styled("Note: the artifact deletes itself\n")
        self.assertNotIn("\033[2m", rendered)

    def test_an_indented_provenance_field_is_styled(self) -> None:
        rendered = self._styled("  authority: a.invalid\n")
        self.assertIn("\033[2m", rendered)
        self.assertEqual(visible(rendered), "  authority: a.invalid\n")

    def test_bold_is_styled_and_keeps_its_markers(self) -> None:
        rendered = self._styled("a **strong** claim")
        self.assertIn("\033[1m", rendered)
        self.assertEqual(visible(rendered), "a **strong** claim")

    def test_inline_code_is_styled(self) -> None:
        rendered = self._styled("run `Get-Date` now")
        self.assertIn("\033[36m", rendered)
        self.assertEqual(visible(rendered), "run `Get-Date` now")

    def test_bullets_keep_their_indentation(self) -> None:
        rendered = self._styled("- top\n  - nested\n")
        lines = visible(rendered).split("\n")
        self.assertEqual(lines[0], "- top")
        self.assertEqual(lines[1], "  - nested")

    def test_numbered_lists_are_styled(self) -> None:
        rendered = self._styled("1. first\n2. second\n")
        self.assertIn("\033[36m", rendered)
        self.assertEqual(visible(rendered), "1. first\n2. second\n")

    def test_a_report_without_markdown_is_left_readable(self) -> None:
        plain = "The artifact writes a file and deletes it.\n"
        self.assertEqual(visible(self._styled(plain)), plain)


class SpanBoundaryTests(unittest.TestCase):
    """Which characters a span covers, not merely that one was styled."""

    def _spans(self, text: str, opener: str) -> list[str]:
        """The visible text inside each span opened with `opener`."""
        rendered = render_report(text, force_style=True)
        return re.findall(
            re.escape(opener) + r"(.*?)\033\[0m", rendered, re.S
        )

    def test_bold_is_not_greedy_across_separate_spans(self) -> None:
        """A greedy match would style `a** x **b` as one run, silently
        emphasising text the author left plain."""
        spans = self._spans("**a** x **b**", "\033[1m")
        self.assertEqual(spans, ["**a**", "**b**"])

    def test_an_empty_code_span_is_not_matched(self) -> None:
        """```` `` ```` is two literal backticks, not an empty code span."""
        rendered = render_report("a `` b", force_style=True)
        self.assertNotIn("\033[36m", rendered)
        self.assertEqual(visible(rendered), "a `` b")

    def test_an_empty_bold_span_is_not_matched(self) -> None:
        rendered = render_report("a **** b", force_style=True)
        self.assertNotIn("\033[1m", rendered)
        self.assertEqual(visible(rendered), "a **** b")

    def test_only_a_correctly_shaped_evidence_id_is_tinted(self) -> None:
        """An id is `ev_` plus 12 and 16 hex digits. Anything else is text
        that merely resembles one, and tinting it would suggest a provenance
        it does not have."""
        real = "ev_26e1592c3e90_6a892b624a2a237c"
        self.assertIn("\033[2m", render_report(real, force_style=True))

        for lookalike in ("ev_short_id", "ev_26e1592c3e90", "ev_zzzzzzzzzzzz_6a892b624a2a237c"):
            with self.subTest(text=lookalike):
                rendered = render_report(lookalike, force_style=True)
                self.assertNotIn("\033[2m", rendered)

    def test_an_embedded_id_is_not_tinted(self) -> None:
        """Word boundaries: `xev_...` is not an id, and neither is one with
        extra hex glued to its tail."""
        for text in (
            "xev_26e1592c3e90_6a892b624a2a237c",
            "ev_26e1592c3e90_6a892b624a2a237cff",
        ):
            with self.subTest(text=text):
                self.assertNotIn("\033[2m", render_report(text, force_style=True))


class CodeFenceTests(unittest.TestCase):
    def test_a_fenced_block_is_preserved_verbatim(self) -> None:
        """Inside a fence, markers are code -- not formatting."""
        text = "```\nnot **bold** and not `code`\n```\n"
        rendered = render_report(text, force_style=True)

        self.assertEqual(visible(rendered), text)
        # Because the markers are preserved, "unchanged text" cannot by itself
        # prove the block was left alone -- the styling has to be checked. A
        # renderer that formatted inside the fence would wrap the asterisks in
        # a bold escape; the whole line carries one code colour instead.
        body = [ln for ln in rendered.split("\n") if "bold" in ln][0]
        self.assertNotIn("\033[1m", body)
        self.assertTrue(body.startswith("\033[36m"))
        self.assertTrue(body.endswith("\033[0m"))

    def test_formatting_resumes_after_the_fence_closes(self) -> None:
        text = "```\nraw\n```\n**after**\n"
        rendered = render_report(text, force_style=True)
        self.assertIn("\033[1m", rendered)
        self.assertEqual(visible(rendered), text)

    def test_a_tilde_fence_is_recognised(self) -> None:
        """Both fence syntaxes are Markdown; treating only one as a fence
        would format the contents of the other as prose."""
        text = "~~~\nnot **bold**\n~~~\n"
        rendered = render_report(text, force_style=True)
        body = [ln for ln in rendered.split("\n") if "bold" in ln][0]
        self.assertNotIn("\033[1m", body)
        self.assertEqual(visible(rendered), text)

    def test_an_indented_fence_is_recognised(self) -> None:
        text = "  ```\n  not **bold**\n  ```\n"
        rendered = render_report(text, force_style=True)
        body = [ln for ln in rendered.split("\n") if "bold" in ln][0]
        self.assertNotIn("\033[1m", body)

    def test_an_unclosed_fence_does_not_swallow_the_report(self) -> None:
        text = "```\nstill inside\n"
        self.assertEqual(visible(render_report(text, force_style=True)), text)


class TableTests(unittest.TestCase):
    """Tables are left as raw Markdown, deliberately."""

    TABLE = "| claim | supported |\n| --- | --- |\n| beacon | NO |\n"

    def test_a_table_is_readable_and_unaltered(self) -> None:
        rendered = render_report(self.TABLE, force_style=True)
        self.assertEqual(visible(rendered), self.TABLE)


class EvidenceIdTests(unittest.TestCase):
    EVIDENCE = "ev_26e1592c3e90_6a892b624a2a237c"

    def test_an_evidence_id_is_tinted_but_never_altered(self) -> None:
        """An id that came back changed would be a different fact."""
        rendered = render_report(f"see evidence: {self.EVIDENCE} here", force_style=True)

        self.assertIn(self.EVIDENCE, visible(rendered))
        self.assertEqual(visible(rendered), f"see evidence: {self.EVIDENCE} here")
        self.assertIn("\033[2m", rendered)

    def test_a_provenance_field_keeps_its_value(self) -> None:
        line = "  authority: gibuzuy37v2v.top\n"
        rendered = render_report(line, force_style=True)
        self.assertEqual(visible(rendered), line)
        self.assertIn("gibuzuy37v2v.top", visible(rendered))

    def test_an_indicator_section_survives_rendering_whole(self) -> None:
        section = (
            "## Verified indicators\n"
            "\n"
            "- uri: http://a.invalid/1.php?s=k,2\n"
            "  authority: a.invalid\n"
            "  path: /1.php\n"
            "  query: ?s=k,2\n"
            f"  evidence: {self.EVIDENCE}\n"
            "  sha256: " + "a" * 64 + "\n"
        )
        self.assertEqual(visible(render_report(section, force_style=True)), section)


class TerminalSafetyTests(unittest.TestCase):
    """Report text carries decoded artifact bytes: an attacker's characters."""

    HOSTILE = (
        "## \x1b[2J\x1b[H heading\n"
        "- \x1b]0;title\x07 bullet\n"
        "**\rforged**\n"
        "`\x1b]52;c;ZXZpbA==\x07`\n"
        "\x9b2J\n"
    )

    def _assert_only_our_escapes(self, rendered: str) -> None:
        """Every escape present must be one of ours, and nothing else."""
        for control in ("\x1b[2J", "\x1b[H", "\x1b]0;", "\x1b]52;", "\r", "\x07", "\x9b"):
            with self.subTest(control=repr(control)):
                self.assertNotIn(control, rendered)

    def test_styled_output_carries_no_artifact_escape(self) -> None:
        self._assert_only_our_escapes(render_report(self.HOSTILE, force_style=True))

    def test_plain_output_carries_no_artifact_escape(self) -> None:
        rendered = render_report(self.HOSTILE, force_style=False)
        self._assert_only_our_escapes(rendered)
        self.assertNotIn("\033", rendered)

    def test_sanitisation_happens_before_styling(self) -> None:
        """Styling sanitised text is safe; sanitising styled text would either
        destroy our escapes or leave a crafted one indistinguishable."""
        rendered = render_report("**\x1b[31mred**", force_style=True)
        self.assertNotIn("\x1b[31m", rendered)
        self.assertIn("\033[1m", rendered)

    def test_a_fence_cannot_smuggle_escapes(self) -> None:
        rendered = render_report("```\n\x1b[2Jcleared\n```\n", force_style=True)
        self.assertNotIn("\x1b[2J", rendered)


class NonStreamingReportTests(unittest.TestCase):
    """A backend that does not stream hands the terminal a whole report.

    That is the one case where the complete text -- prose and appendices --
    is printed in one call, so it is the case that must be rendered rather
    than printed raw.
    """

    def test_a_whole_report_reaches_the_renderer(self) -> None:
        import inspect

        from orbit.terminal import repl

        source = inspect.getsource(repl.Repl._ask_analysis)
        self.assertIn("render_report(run.final_report.text)", source)
        self.assertNotIn(
            "sanitize_terminal_text(\n                        run.final_report.text",
            source,
        )

    def test_every_report_print_site_renders(self) -> None:
        """No path may print report text unrendered: an unstyled branch is
        also an unsanitised-by-this-route branch."""
        import inspect

        from orbit.terminal import repl

        for method in (repl.Repl._ask_analysis, repl.Repl._handle_report_command):
            source = inspect.getsource(method)
            with self.subTest(method=method.__name__):
                self.assertNotIn("print(report.text", source)
                self.assertNotIn("print(run.final_report.text", source)


class OutputModeTests(unittest.TestCase):
    """Every reason colour may be disallowed produces raw Markdown."""

    def _rendered_under(self, **env) -> str:
        with mock.patch.dict(os.environ, env, clear=False):
            for key in ("NO_COLOR",):
                if key not in env:
                    os.environ.pop(key, None)
            return render_report(SAMPLE)

    def test_no_color_yields_raw_markdown(self) -> None:
        with mock.patch("orbit.terminal.theme.is_tty", return_value=True):
            rendered = self._rendered_under(NO_COLOR="1")
        self.assertEqual(rendered, SAMPLE)
        self.assertNotIn("\033", rendered)

    def test_a_dumb_terminal_yields_raw_markdown(self) -> None:
        with mock.patch("orbit.terminal.theme.is_tty", return_value=True):
            rendered = self._rendered_under(TERM="dumb")
        self.assertEqual(rendered, SAMPLE)

    def test_a_non_tty_yields_raw_markdown(self) -> None:
        with mock.patch("orbit.terminal.theme.is_tty", return_value=False):
            rendered = self._rendered_under(TERM="xterm")
        self.assertEqual(rendered, SAMPLE)

    def test_redirected_output_yields_raw_markdown(self) -> None:
        """The real path: stdout is a StringIO, not a terminal."""
        os.environ.pop("NO_COLOR", None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print(render_report(SAMPLE))
        self.assertNotIn("\033", out.getvalue())
        self.assertIn("## Verified indicators", out.getvalue())

    def test_a_colour_terminal_yields_styled_markdown(self) -> None:
        with mock.patch("orbit.terminal.theme.is_tty", return_value=True):
            rendered = self._rendered_under(TERM="xterm")
        self.assertIn("\033", rendered)
        self.assertEqual(visible(rendered), SAMPLE)


if __name__ == "__main__":
    unittest.main()
