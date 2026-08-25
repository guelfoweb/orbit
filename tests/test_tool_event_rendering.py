"""Rendering of ANALYSIS intermediate output.

These tests pin the presentation contract of `tool_events`: what the analyst
sees while a step is still running, before control is handed back. Two things
matter and neither is about taste. Content must survive rendering byte for
byte, because this is the only view of an action's output the analyst gets
without opening evidence by hand; and nothing in that output may act on the
terminal, because every byte of it is authored by a model or a tool.

Structural assertions are preferred over golden strings so that unrelated
spacing changes do not fail the suite. Where a golden string appears it is
because the exact shape is the contract.
"""

from __future__ import annotations

import json
import unittest

from orbit.terminal.tool_events import (
    BLOCK_INDENT,
    BLOCK_LINE_CHARS,
    BLOCK_LINE_LIMIT,
    BLOCK_ROW_LIMIT,
    format_tool_call_event,
    format_tool_result_event,
)


def envelope(path: str, body: str) -> str:
    """A read-file transport envelope wrapping `body`."""
    return f"shell_output_read_file: true\npath: {path}\ncontent:\n{body}"


def block_body(rendered: str) -> list[str]:
    """The payload lines of a rendered block, indent removed.

    A blank line in the body is emitted empty rather than indented, so it is
    kept as `""` here: dropping it would hide whether the renderer preserved
    the gaps in the original text.
    """
    body: list[str] = []
    for line in rendered.split("\n")[1:]:
        if line.startswith(BLOCK_INDENT):
            body.append(line[len(BLOCK_INDENT):])
        elif not line:
            body.append("")
    return body


CONTROL_CHARS = ("\x1b", "\r", "\x07", "\x08", "\x0b", "\x0c", "\x85")


class ContentPreservationTests(unittest.TestCase):
    """Rendering shows the output; it does not edit it."""

    def assert_no_control(self, rendered: str) -> None:
        for ch in CONTROL_CHARS:
            self.assertNotIn(ch, rendered, f"{ch!r} reached the terminal")

    def test_single_line_result_stays_compact(self) -> None:
        rendered = format_tool_result_event("exec_shell_full_command", 8, content="10:15:01")

        self.assertEqual(rendered, "└ 10:15:01")
        self.assertNotIn("\n", rendered)

    def test_multiline_code_keeps_every_line_and_its_indentation(self) -> None:
        source = (
            "def load(path):\n"
            "    with open(path) as fh:\n"
            "        data = fh.read()\n"
            "    return data"
        )
        content = envelope("loader.py", source)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), source.split("\n"))

    def test_json_like_output_keeps_its_shape(self) -> None:
        body = '{\n  "name": "orbit",\n  "nested": {\n    "deep": [1, 2, 3]\n  }\n}'
        content = envelope("config.json", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), body.split("\n"))
        self.assertEqual("\n".join(block_body(rendered)), body)

    def test_backslashes_and_windows_paths_are_not_escaped_away(self) -> None:
        body = "C:\\Users\\analyst\\sample.exe\n    regex \\d+\\.\\d+\\\\n literal\n\ttabbed\\path"
        content = envelope("notes.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), body.split("\n"))

    def test_unicode_survives_unchanged(self) -> None:
        body = "café — 日本語 ✓ αβγ Ω 🎯\n    ünïcödé indented\n\tروبوت"
        content = envelope("unicode.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), body.split("\n"))

    def test_blank_lines_inside_the_body_are_preserved(self) -> None:
        body = "first\n\n    indented after a gap\n\nlast"
        content = envelope("gaps.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), ["first", "", "    indented after a gap", "", "last"])

    def test_trailing_whitespace_on_a_line_is_kept(self) -> None:
        body = "    keep me:   \n    and me\t"
        content = envelope("ws.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), ["    keep me:   ", "    and me\t"])


class TruncationTests(unittest.TestCase):
    """Shortening is always announced."""

    def test_long_output_is_bounded_and_marked(self) -> None:
        body = "\n".join(f"    line {i:03d}" for i in range(40))
        content = envelope("big.log", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        lines = block_body(rendered)
        self.assertEqual(len(lines), BLOCK_LINE_LIMIT + 1)
        self.assertIn("truncated", lines[-1])
        self.assertIn("evidence", lines[-1], "the full text must remain findable")

    def test_an_overlong_single_line_is_marked_truncated(self) -> None:
        body = "    " + ("x" * 5000) + "\n    second"
        content = envelope("wide.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertIn("truncated", rendered)
        self.assertIn("…", rendered)

    def test_blank_lines_do_not_consume_the_line_budget(self) -> None:
        """Separators are not output.

        Counting blank lines against the budget halves how much of a
        double-spaced file the analyst sees: twelve rows of terminal spent to
        show six statements.
        """
        body = "\n\n".join(f"    stmt_{i}()" for i in range(20))
        content = envelope("spaced.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        shown = [line for line in rendered.split("\n") if "stmt_" in line]
        self.assertEqual(len(shown), BLOCK_LINE_LIMIT)

    def test_a_run_of_blank_lines_cannot_flood_the_terminal(self) -> None:
        """Separators are free of the line budget but not of the row budget.

        Blank lines do not spend the content budget, so without a row cap a
        body that is mostly empty space still costs a row per gap: thousands
        of rows to show a couple of statements, scrolling the surrounding
        status out of view. What is dropped must still be marked.
        """
        body = "    first\n" + "\n" * 32_000 + "    last"
        content = envelope("gap.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        rows = rendered.split("\n")
        self.assertLessEqual(len(rows), BLOCK_ROW_LIMIT + 2, "the block must stay bounded")
        self.assertIn("first", rendered)
        self.assertIn("truncated", rendered, "dropped output must say so")

    def test_a_short_gap_does_not_hide_later_output(self) -> None:
        """Ordinary spacing still shows everything around it."""
        body = "    first\n\n\n    last"
        content = envelope("gap.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertIn("first", rendered)
        self.assertIn("last", rendered)

    def test_the_renderer_invents_no_trailing_whitespace(self) -> None:
        """A blank line is empty, not an indent with nothing after it."""
        body = "    a\n\n    b\n\n    c"
        content = envelope("ws.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        for line in rendered.split("\n"):
            self.assertEqual(line, line.rstrip(), f"invented trailing whitespace: {line!r}")

    def test_trailing_blank_lines_are_not_rendered(self) -> None:
        """A separator at the end has nothing left to separate.

        Through the public path `_result_block` has already stripped trailing
        newlines, so this pins the rendered result rather than the guard in
        `_block_lines`, which is defensive for direct callers.
        """
        content = envelope("tail.py", "    a\n    b\n\n\n")

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), ["    a", "    b"])
        self.assertFalse(rendered.endswith("\n"))

    def test_the_truncation_marker_states_no_competing_size(self) -> None:
        """The header already carries the result's size.

        A second, different number on the next line describes the excerpt
        rather than the result, and reads as a contradiction.
        """
        body = "\n".join(f"    line {i}" for i in range(40))
        content = envelope("big.log", body)

        rendered = format_tool_result_event("exec_shell_full_command", 500_000, content=content)

        marker = rendered.split("\n")[-1]
        self.assertIn("truncated", marker)
        self.assertIn("evidence", marker)
        self.assertNotIn("chars", marker)

    def test_the_line_width_boundary_is_exact(self) -> None:
        """A line at the limit is untouched; one past it is cut and marked.

        Pins the constant itself: without this, narrowing the width would
        silently shorten output that used to fit.
        """
        exact = "    " + "x" * (BLOCK_LINE_CHARS - len(BLOCK_INDENT))
        content = envelope("w.txt", f"{exact}\n    second")

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertIn(exact, rendered)
        self.assertNotIn("…", rendered)
        self.assertNotIn("truncated", rendered)

        over = exact + "y"
        content = envelope("w.txt", f"{over}\n    second")

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertIn("…", rendered)
        self.assertIn("truncated", rendered)
        body = block_body(rendered)
        self.assertEqual(len(body[0]), BLOCK_LINE_CHARS + 1, "cut to the limit plus the ellipsis")

    def test_the_row_budget_binds_before_the_line_budget_on_sparse_output(self) -> None:
        """Pinned by observable behaviour, not by the constant.

        Every other bound here is derived from `BLOCK_ROW_LIMIT`, so raising
        the constant moves the goalposts and the assertion follows it. This
        input is sparse enough that the row budget stops the block while the
        line budget still had room: fewer content lines appear than the line
        budget allows, and that shortening is marked. Remove or raise the row
        cap and all twelve lines appear unmarked.
        """
        body = "\n\n\n".join(f"    c{i}" for i in range(12))
        content = envelope("sparse.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        shown = [line for line in rendered.split("\n") if line.strip().startswith("c")]
        self.assertLess(
            len(shown), BLOCK_LINE_LIMIT, "the row budget must stop this before the line budget"
        )
        self.assertIn("truncated", rendered, "what the row budget drops must be marked")

    def test_blank_rows_alone_cannot_exceed_the_row_budget(self) -> None:
        """The row cap binds on separators, not only on content lines."""
        body = "    only\n" + "\n" * 500
        content = envelope("b.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertLessEqual(len(rendered.split("\n")), BLOCK_ROW_LIMIT + 2)

    def test_transport_truncation_flag_still_surfaces(self) -> None:
        content = "directory_listing: path=. total_seen=500 truncated=true\na\nb"

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertIn("truncated", rendered)

    def test_untruncated_output_is_not_marked(self) -> None:
        body = "def f():\n    return 1"
        content = envelope("small.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertNotIn("truncated", rendered)


class ErrorAndRefusalTests(unittest.TestCase):
    """Failure stays visually distinct from output that succeeded."""

    def test_error_keeps_its_prefix_on_the_status_line(self) -> None:
        content = "error: exit 1\nFAILED test_example.py::test_value\nmore details"

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertTrue(rendered.startswith("└ error: exit 1"))

    def test_error_is_not_reshaped_into_an_output_block(self) -> None:
        """An error reads as a failure, not as a result worth previewing."""
        content = "error: exit 2\n    traceback line\n    another"

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertNotIn("\n", rendered)
        self.assertIn("error:", rendered)

    def test_refusal_stays_marked_rejected(self) -> None:
        content = (
            "error: shell-full analysis requests require content/source/string evidence, "
            "not only metadata/listing. Use a bounded command such as sed/head/grep/strings on the target file."
        )

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(rendered, "└ rejected metadata-only output · rejected")

    def test_a_successful_block_carries_no_failure_marker(self) -> None:
        content = envelope("ok.py", "def f():\n    return 1")

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertNotIn("error:", rendered)
        self.assertNotIn("rejected", rendered)


class TerminalSafetyTests(unittest.TestCase):
    """Model- and tool-authored bytes never act on the terminal."""

    def assert_inert(self, rendered: str) -> None:
        for ch in CONTROL_CHARS:
            self.assertNotIn(ch, rendered, f"{ch!r} reached the terminal")

    def test_ansi_colour_and_erase_cannot_execute_inline(self) -> None:
        content = "before \x1b[31mRED\x1b[0m \x1b[2J\x1b[H erased"

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assert_inert(rendered)
        self.assertIn("RED", rendered)

    def test_ansi_cannot_execute_inside_a_block(self) -> None:
        body = "    a = 1 \x1b[2J\x1b[H\n    b = 2 \x1b]0;pwned\x07\n    c = 3"
        content = envelope("evil.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assert_inert(rendered)
        self.assertIn("a = 1", rendered)

    def test_carriage_return_cannot_forge_a_line_in_a_block(self) -> None:
        """CR is the line-overwrite primitive and must not start a new line.

        `str.splitlines` breaks on CR, so splitting that way would let tool
        output invent an extra indented line -- a forged status line inside
        Orbit's own block.
        """
        body = "    real line\n    visible\rFORGED"
        content = envelope("forge.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assert_inert(rendered)
        self.assertEqual(len(block_body(rendered)), 2)
        self.assertIn("FORGED", rendered)

    def test_exotic_line_breaks_cannot_forge_lines(self) -> None:
        body = "    a\x0bV\x0cF\x85N\u2028S\u2029P\n    b"
        content = envelope("exotic.txt", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assert_inert(rendered)
        self.assertEqual(len(block_body(rendered)), 2)

    def test_hostile_bytes_in_a_tool_call_cannot_execute(self) -> None:
        args = json.dumps({"command": "echo \x1b[2J\x1b[H\x1b]0;pwned\x07"})

        rendered = format_tool_call_event("exec_shell_full_command", args)

        self.assert_inert(rendered)

    def test_hostile_bytes_in_a_url_cannot_execute(self) -> None:
        args = json.dumps({"url": "https://example.com/\x1b[31m\r\x07"})

        rendered = format_tool_call_event("fetch_url", args)

        self.assert_inert(rendered)

    def test_hostile_bytes_in_an_artifact_path_cannot_execute(self) -> None:
        args = json.dumps({"path": "out/\x1b[2Jreport.json"})

        rendered = format_tool_call_event("write_artifact", args)

        self.assert_inert(rendered)

    def test_the_renderer_emits_no_escape_of_its_own(self) -> None:
        """Styling is applied by the caller, not baked into the text."""
        content = envelope("plain.py", "def f():\n    return 1")

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertNotIn("\x1b", rendered)


class ShapeSelectionTests(unittest.TestCase):
    """Which form a result takes is decided by shape, never by content."""

    def test_indented_output_becomes_a_block(self) -> None:
        content = envelope("x.py", "def f():\n    return 1")

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertIn("\n", rendered)

    def test_a_short_flat_listing_stays_inline(self) -> None:
        content = ".\n./pdf\n./text\n./samples\n"

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertNotIn("\n", rendered)

    def test_a_control_character_cannot_decide_the_shape(self) -> None:
        """Only ordinary indentation makes output structured.

        `str.isspace` is true of vertical tab, form feed, carriage return and
        the Unicode separators, so testing it would let one control byte in
        tool output choose which shape Orbit renders.
        """
        for lead in ("\x0b", "\x0c", "\r", "\x85", "\u2028"):
            with self.subTest(lead=lead):
                content = envelope("c.txt", f"{lead}alpha\nbeta")

                rendered = format_tool_result_event(
                    "exec_shell_full_command", len(content), content=content
                )

                self.assertNotIn("\n", rendered, "a control byte forced the block form")

    def test_ordinary_indentation_still_selects_the_block(self) -> None:
        for lead in (" ", "\t"):
            with self.subTest(lead=lead):
                content = envelope("i.txt", f"{lead}alpha\nbeta")

                rendered = format_tool_result_event(
                    "exec_shell_full_command", len(content), content=content
                )

                self.assertIn("\n", rendered)

    def test_shape_choice_does_not_depend_on_the_words(self) -> None:
        """Identical structure renders identically whatever the text says."""
        shape = "    alpha\n    beta\n    gamma"
        other = "    MZ\x90\x00 header\n    suspicious\n    payload"

        first = format_tool_result_event(
            "exec_shell_full_command", 100, content=envelope("a.txt", shape)
        )
        second = format_tool_result_event(
            "exec_shell_full_command", 100, content=envelope("a.txt", other)
        )

        self.assertEqual(
            len(first.split("\n")),
            len(second.split("\n")),
            "the renderer must not treat one body as more interesting than another",
        )

    def test_a_single_indented_line_stays_inline(self) -> None:
        """One line is one line, indented or not.

        Indentation is what marks output as structured, so without an explicit
        line-count floor a single indented line would be wrapped in a block:
        two rows of terminal to show one row of text.
        """
        content = envelope("one.txt", "    only one line here")

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertNotIn("\n", rendered)

    def test_output_beginning_with_an_envelope_field_name_is_not_suppressed(self) -> None:
        """Real output can start with `path:` -- that does not make it metadata.

        `path:`, `status:` and `chars:` are fields inside a transport envelope,
        never its opening line. Treating them as envelope markers would drop
        the body of any command output that happens to begin with one, which
        is silent information loss rather than a formatting choice.
        """
        for first in ("path: /etc/hosts", "status: 200", "chars: 12"):
            with self.subTest(first=first):
                content = f"{first}\n    second line\n    third line"

                rendered = format_tool_result_event(
                    "exec_shell_full_command", len(content), content=content
                )

                self.assertEqual(
                    block_body(rendered), content.split("\n"), "the body must survive intact"
                )

    def test_a_payload_containing_a_body_marker_is_not_cut_short(self) -> None:
        """Only the envelope's own first marker separates head from body."""
        body = "def f():\n    content:\n        nested = 1\n    return nested"
        content = envelope("t.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertEqual(block_body(rendered), body.split("\n"))

    def test_metadata_only_envelope_has_no_block(self) -> None:
        content = "shell_output_read_file: true\npath: a.txt\nchars: 10"

        rendered = format_tool_result_event("exec_shell_full_command", len(content), content=content)

        self.assertNotIn("\n", rendered)

    def test_empty_content_falls_back_to_the_char_count(self) -> None:
        rendered = format_tool_result_event("exec_shell_full_command", 0, content="")

        self.assertEqual(rendered, "└ 0 chars")


class EvidenceVisibilityTests(unittest.TestCase):
    """Identifiers and metadata stay visible without dominating the view."""

    def test_chunk_label_survives_the_block_form(self) -> None:
        body = "def f():\n    return 1\n    # more"
        content = (
            "shell_output_read_file: true\npath: b.py\n"
            f"chunk_index: 0\ntotal_chunks: 3\ncontent:\n{body}"
        )

        rendered = format_tool_result_event("exec_shell_full_command", 200, content=content)

        self.assertIn("chunk 1/3", rendered.split("\n")[0])

    def test_large_context_marker_survives_the_block_form(self) -> None:
        body = "def f():\n    return 1"
        content = envelope("c.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", 20_000, content=content)

        self.assertIn("large context", rendered.split("\n")[0])

    def test_status_metadata_stays_on_one_header_line(self) -> None:
        body = "def f():\n    return 1"
        content = envelope("d.py", body)

        rendered = format_tool_result_event("exec_shell_full_command", 20_000, content=content)

        header = rendered.split("\n")[0]
        self.assertTrue(header.startswith("└ "))
        self.assertIn("chars", header)
