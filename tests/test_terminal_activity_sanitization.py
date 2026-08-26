"""Sanitization of the tool labels that share a line with Orbit's own output.

Three formatting paths returned model- or tool-authored text without passing
it through the terminal boundary: the waiting indicator's activity label, the
fallback used when no branch recognises a tool, and the `verify_artifact`
check field. Everything else in this module already went through
`_normalize_inline`.

The activity label is the sharpest of the three. It is printed on a line that
begins with a carriage return, so an escape sequence in a command or URL could
erase that line and write over Orbit's own status row: the terminal would show
a row that looks like Orbit's and says whatever the tool wanted.

These tests assert on the returned string. What reaches the terminal is that
string, and the printing site adds only styling.
"""

from __future__ import annotations

import json
import unittest

from orbit.terminal.tool_events import (
    format_tool_activity_label,
    format_tool_call_event,
)


# Anything in the control range that a terminal acts on rather than shows.
# Vertical tab, form feed, NEL and the Unicode separators are included because
# `str.splitlines` treats them as line breaks: left intact they can forge a row.
CONTROL_BYTES = (
    "\x1b",      # ESC -- CSI/OSC introducer
    "\x9b",      # 8-bit CSI
    "\x07",      # BEL
    "\r",        # CR -- the line-overwrite primitive
    "\x0b",      # VT
    "\x0c",      # FF
    "\x85",      # NEL
    "\u2028",    # LINE SEPARATOR
    "\u2029",    # PARAGRAPH SEPARATOR
    "\x00",      # NUL
    "\x08",      # BS
)

HOSTILE = (
    "start \x1b[2J\x1b[H erase \x1b]0;title\x07 osc \r overwrite "
    "\x0b\x0c\x85\u2028\u2029 breaks \x00\x08 end"
)


def rendered_paths(payload: str) -> dict[str, str]:
    """Every formatter branch that can carry untrusted text, keyed by name."""
    return {
        "activity/shell": format_tool_activity_label(
            "exec_shell_full_command", json.dumps({"command": payload})
        ),
        "activity/url": format_tool_activity_label(
            "fetch_url", json.dumps({"url": payload})
        ),
        "activity/list_directory": format_tool_activity_label(
            "list_directory", json.dumps({"path": payload})
        ),
        "activity/tool_name": format_tool_activity_label(payload, "{}"),
        "call/unknown_args": format_tool_call_event("some_unknown_tool", payload),
        "call/unknown_name": format_tool_call_event(payload, ""),
        "call/verify_check": format_tool_call_event(
            "verify_artifact", json.dumps({"check": payload})
        ),
    }


class ControlSequenceTests(unittest.TestCase):
    """No path may hand the terminal something it will act on."""

    def assert_inert(self, label: str, rendered: str) -> None:
        for char in CONTROL_BYTES:
            self.assertNotIn(
                char, rendered, f"{label}: {char!r} reached the terminal"
            )
        self.assertNotIn("\n", rendered, f"{label}: rendered more than one line")

    def test_no_path_emits_a_control_sequence(self) -> None:
        for label, rendered in rendered_paths(HOSTILE).items():
            with self.subTest(path=label):
                self.assert_inert(label, rendered)

    def test_each_control_byte_individually(self) -> None:
        """One at a time, so a failure names the byte that got through."""
        for char in CONTROL_BYTES:
            for label, rendered in rendered_paths(f"a{char}b").items():
                with self.subTest(path=label, char=repr(char)):
                    self.assert_inert(label, rendered)

    def test_a_forged_status_row_cannot_be_written(self) -> None:
        """CR is what lets injected text overwrite Orbit's own row.

        The waiting indicator prints its line starting with a carriage return.
        If a payload's own CR survived, everything after it would land at
        column zero and replace what Orbit had written there.
        """
        payload = "harmless\r› Exec  rm -rf /   "

        for label, rendered in rendered_paths(payload).items():
            with self.subTest(path=label):
                self.assertNotIn("\r", rendered)
                # The text is still shown -- it is just no longer a cursor move.
                self.assertIn("Exec", rendered)


class ContentPreservationTests(unittest.TestCase):
    """Sanitizing must not cost ordinary content."""

    def test_printable_text_survives(self) -> None:
        for label, rendered in rendered_paths("ls -la /tmp").items():
            with self.subTest(path=label):
                self.assertIn("ls -la /tmp", rendered)

    def test_unicode_survives(self) -> None:
        payload = "café — 日本語 ✓ αβγ Ω 🎯 ünïcödé"
        for label, rendered in rendered_paths(payload).items():
            with self.subTest(path=label):
                self.assertIn(payload, rendered)

    def test_windows_paths_and_backslashes_survive(self) -> None:
        payload = r"C:\Users\analyst\Desktop\sample.exe"
        for label, rendered in rendered_paths(payload).items():
            with self.subTest(path=label):
                self.assertIn(payload, rendered)

    def test_urls_survive_intact(self) -> None:
        payload = "https://example.com/a?b=1&c=2#frag"
        self.assertIn(
            payload,
            format_tool_activity_label("fetch_url", json.dumps({"url": payload})),
        )

    def test_hostile_input_keeps_its_readable_text(self) -> None:
        """Escaping is not deletion: the words are still there to read."""
        for label, rendered in rendered_paths(HOSTILE).items():
            with self.subTest(path=label):
                for word in ("start", "erase", "overwrite", "breaks", "end"):
                    self.assertIn(word, rendered)


class IdempotenceTests(unittest.TestCase):
    """An already-sanitized value must not be escaped a second time."""

    def test_a_sanitized_value_gains_no_second_layer_of_escapes(self) -> None:
        """Passing a value through twice must not escape it again.

        Whitespace still collapses on a second pass -- that is what these
        single-line labels are for -- so the property under test is the escape
        count, not string equality. A value that gained a layer each time it
        crossed the boundary would slowly turn into its own transcript.
        """
        for label, once in rendered_paths(HOSTILE).items():
            with self.subTest(path=label):
                twice = format_tool_activity_label(once, "{}")
                self.assertEqual(
                    once.count("\\x"),
                    twice.count("\\x"),
                    "a sanitized value was escaped again",
                )
                self.assertNotIn("\\\\x", twice)

    def test_the_boundary_helper_is_idempotent(self) -> None:
        from orbit.terminal.tool_events import _normalize_inline

        once = _normalize_inline(HOSTILE)

        self.assertEqual(once, _normalize_inline(once))


class UnrecognisedToolTests(unittest.TestCase):
    """The fallback carries two untrusted values, not one."""

    def test_both_the_name_and_the_arguments_are_sanitized(self) -> None:
        rendered = format_tool_call_event("evil\x1b[2Jtool", "arg\x07value")

        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x07", rendered)
        self.assertIn("tool", rendered)
        self.assertIn("value", rendered)

    def test_a_recognised_tool_still_renders_normally(self) -> None:
        rendered = format_tool_call_event(
            "exec_shell_full_command", json.dumps({"command": "ls -la"})
        )

        self.assertEqual(rendered, "› Read  ls -la")
