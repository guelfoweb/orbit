"""The prompt tells the truth about mode and autonomy, and colour obeys.

`/autonomous on` deliberately changes no mode, so before this the analyst
typed it, saw `chat>` unchanged, and had nothing to distinguish a setting that
took effect from one that did not. The prompt now carries both facts, and one
formatter renders every on/off Orbit shows.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.runtime.workflow_mode import WorkflowMode
from orbit.terminal.config import AppConfig
from orbit.terminal.repl import Repl
from orbit.terminal.repl_input import input_prompt
from orbit.terminal.theme import on_off


class _StubBackend:
    thinking = False


class _StubRuntime:
    def __init__(self) -> None:
        self.messages: list = []
        self.context_tokens = 8192
        self.evidence_store = None
        self.last_memory_refresh = None

    # Read by `collect_runtime_status`; zero is the truthful value for a
    # session that has done nothing.
    memory_refreshes = 0
    total_memory_tokens_saved = 0
    mutation_verifications = 0
    mutation_verification_repairs = 0
    mutation_verification_failures = 0

    def can_continue_last_response(self) -> bool:
        return False


class UXTestBase(unittest.TestCase):
    def _prompt(self, repl: Repl) -> str:
        """The prompt as a non-colour terminal renders it."""
        with mock.patch("orbit.terminal.repl_input.sys.stdout") as stdout:
            stdout.isatty.return_value = False
            return input_prompt(
                repl._prompt_label(), autonomous=bool(repl.autonomous_analysis)
            )

    def _repl(self) -> Repl:
        return Repl(
            runtime=_StubRuntime(),
            backend=_StubBackend(),
            config=AppConfig(workdir=Path(".")),
        )

    def _command(self, repl: Repl, line: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._handle_command(line)
        return out.getvalue()


class PromptStateTests(UXTestBase):
    def test_startup_prompt_shows_chat_and_autonomy_off(self) -> None:
        repl = self._repl()
        repl.autonomous_analysis = False
        self.assertEqual(self._prompt(repl), "chat [auto:off]> ")

    def test_toggling_autonomy_on_updates_the_prompt(self) -> None:
        repl = self._repl()
        repl.autonomous_analysis = False
        self._command(repl, "/autonomous on")
        self.assertEqual(self._prompt(repl), "chat [auto:on]> ")

    def test_toggling_autonomy_on_does_not_enter_analysis(self) -> None:
        """The whole reason the prompt has to say so: the mode is unchanged."""
        repl = self._repl()
        repl.autonomous_analysis = False

        self._command(repl, "/autonomous on")

        self.assertTrue(repl.autonomous_analysis)
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)

    def test_toggling_autonomy_off_updates_the_prompt(self) -> None:
        repl = self._repl()
        repl.autonomous_analysis = True
        self._command(repl, "/autonomous off")
        self.assertEqual(self._prompt(repl), "chat [auto:off]> ")

    def test_analysis_mode_keeps_the_autonomy_marker(self) -> None:
        repl = self._repl()
        repl.autonomous_analysis = True
        repl.workflow_mode = WorkflowMode.ANALYSIS
        self.assertEqual(self._prompt(repl), "analysis [auto:on]> ")

    def test_autonomy_survives_returning_to_chat(self) -> None:
        repl = self._repl()
        repl.autonomous_analysis = True
        repl.workflow_mode = WorkflowMode.ANALYSIS

        self._command(repl, "/chat")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertTrue(repl.autonomous_analysis, "the setting is not a mode")
        self.assertEqual(self._prompt(repl), "chat [auto:on]> ")

    def test_a_mistyped_argument_leaves_the_prompt_unchanged(self) -> None:
        repl = self._repl()
        repl.autonomous_analysis = False
        self._command(repl, "/autonomous maybe")
        self.assertEqual(self._prompt(repl), "chat [auto:off]> ")


class AnalysisStartTests(UXTestBase):
    """`/analysis <path>` with autonomy on begins; toggling never does."""

    ARTIFACT = "var x = 1;\nconsole.log(x);\n"

    def _artifact(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory(prefix="orbit-ux-")
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "artifact.js"
        path.write_text(self.ARTIFACT, encoding="utf-8")
        return path

    def _repl_in(self, workdir: Path) -> Repl:
        return Repl(
            runtime=_StubRuntime(),
            backend=_StubBackend(),
            config=AppConfig(workdir=workdir),
        )

    def test_autonomy_on_starts_the_first_step_without_a_second_instruction(self) -> None:
        artifact = self._artifact()
        repl = self._repl_in(artifact.parent)
        repl.autonomous_analysis = True

        with mock.patch.object(Repl, "_ask_analysis") as asked:
            self._command(repl, f"/analysis {artifact.name}")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        asked.assert_called_once()
        self.assertTrue(asked.call_args.args[0].strip(), "an opening line is sent")

    def test_autonomy_off_waits_for_the_analyst(self) -> None:
        """Guided mode is unchanged: entering a session runs nothing."""
        artifact = self._artifact()
        repl = self._repl_in(artifact.parent)
        repl.autonomous_analysis = False

        with mock.patch.object(Repl, "_ask_analysis") as asked:
            self._command(repl, f"/analysis {artifact.name}")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        asked.assert_not_called()

    def test_toggling_autonomy_alone_starts_nothing(self) -> None:
        repl = self._repl()
        with mock.patch.object(Repl, "_ask_analysis") as asked:
            self._command(repl, "/autonomous on")
        asked.assert_not_called()

    def test_a_refused_artifact_inside_a_session_starts_nothing(self) -> None:
        """The dangerous case: a typo while a session is already open.

        A refused path deliberately leaves the mode and the previous session
        untouched, so a guard on the mode is satisfied by the session that was
        already there -- and the mistyped command would launch an autonomous
        run against the wrong artifact while printing an error.
        """
        artifact = self._artifact()
        repl = self._repl_in(artifact.parent)
        repl.autonomous_analysis = True

        with mock.patch.object(Repl, "_ask_analysis"):
            self._command(repl, f"/analysis {artifact.name}")
        opened = repl.analysis
        self.assertIsNotNone(opened)

        with mock.patch.object(Repl, "_ask_analysis") as asked:
            output = self._command(repl, "/analysis no-such-file.js")

        self.assertIn("error", output)
        asked.assert_not_called()
        # The session that was open is still the one that is open.
        self.assertIs(repl.analysis, opened)
        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)

    def test_a_second_artifact_starts_its_own_run(self) -> None:
        """Replacing one session with another is a new instruction."""
        first = self._artifact()
        second = first.parent / "second.js"
        second.write_text(self.ARTIFACT, encoding="utf-8")
        repl = self._repl_in(first.parent)
        repl.autonomous_analysis = True

        with mock.patch.object(Repl, "_ask_analysis"):
            self._command(repl, f"/analysis {first.name}")
        with mock.patch.object(Repl, "_ask_analysis") as asked:
            self._command(repl, f"/analysis {second.name}")

        asked.assert_called_once()

    def test_a_refused_artifact_starts_nothing(self) -> None:
        """A typo must not launch an autonomous run against nothing."""
        repl = self._repl()
        repl.autonomous_analysis = True

        with mock.patch.object(Repl, "_ask_analysis") as asked:
            output = self._command(repl, "/analysis no-such-file.js")

        self.assertIn("error", output)
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        asked.assert_not_called()


class PromptEscapingTests(unittest.TestCase):
    """Readline counts the prompt's width; every escape must be hidden from it."""

    def _rendered(self, label: str, autonomous: bool, *, tty: bool) -> str:
        import orbit.terminal.repl_input as repl_input

        original = repl_input.sys.stdout
        repl_input.sys.stdout = mock.Mock(**{"isatty.return_value": tty})
        try:
            return input_prompt(label, autonomous=autonomous)
        finally:
            repl_input.sys.stdout = original

    def test_every_escape_sits_inside_the_ignore_markers(self) -> None:
        """An unwrapped escape makes readline mis-place the cursor on every
        edited line -- a defect that never shows up in captured output."""
        import re

        for label, autonomous in (("chat", False), ("chat", True), ("analysis", True)):
            with self.subTest(label=label, autonomous=autonomous):
                prompt = self._rendered(label, autonomous, tty=True)
                # Strip each ... span; no escape may remain outside.
                visible = re.sub("\001[^\002]*\002", "", prompt)
                self.assertNotIn("\033", visible)
                self.assertEqual(
                    visible,
                    f"{label} [auto:{'on' if autonomous else 'off'}]> ",
                )

    def test_analysis_keeps_its_amber_marker(self) -> None:
        """The mode colour is chosen from the mode, not from a label that now
        also carries the autonomy state."""
        prompt = self._rendered("analysis", True, tty=True)
        self.assertIn("\033[33m", prompt)
        self.assertNotIn("\033[36m", prompt)

    def test_chat_keeps_its_cyan_marker(self) -> None:
        prompt = self._rendered("chat", False, tty=True)
        self.assertIn("\033[36m", prompt)
        self.assertNotIn("\033[33m", prompt)

    def test_the_state_token_carries_its_own_colour(self) -> None:
        self.assertIn("\033[32m", self._rendered("chat", True, tty=True))
        self.assertIn("\033[31m", self._rendered("chat", False, tty=True))

    def test_a_non_tty_prompt_has_no_escapes_or_markers(self) -> None:
        prompt = self._rendered("analysis", True, tty=False)
        self.assertEqual(prompt, "analysis [auto:on]> ")
        for control in ("\033", "\001", "\002"):
            with self.subTest(control=repr(control)):
                self.assertNotIn(control, prompt)


class BooleanColourTests(unittest.TestCase):
    """One formatter, and it obeys every reason colour may be disallowed."""

    def _tty(self) -> mock._patch:
        return mock.patch("orbit.terminal.theme.is_tty", return_value=True)

    def test_on_is_green_and_off_is_red_on_a_colour_terminal(self) -> None:
        with self._tty(), mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            self.assertEqual(on_off(True), "\033[32mon\033[0m")
            self.assertEqual(on_off(False), "\033[31moff\033[0m")

    def test_no_colour_environment_yields_no_escapes(self) -> None:
        with self._tty(), mock.patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            self.assertEqual(on_off(True), "on")
            self.assertEqual(on_off(False), "off")
            self.assertNotIn("\033", on_off(True) + on_off(False))

    def test_a_dumb_terminal_yields_no_escapes(self) -> None:
        with self._tty(), mock.patch.dict(os.environ, {"TERM": "dumb"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            self.assertNotIn("\033", on_off(True) + on_off(False))

    def test_a_non_tty_yields_no_escapes(self) -> None:
        with mock.patch("orbit.terminal.theme.is_tty", return_value=False):
            os.environ.pop("NO_COLOR", None)
            self.assertEqual(on_off(True), "on")
            self.assertEqual(on_off(False), "off")

    def test_redirected_output_yields_no_escapes(self) -> None:
        """The real path: stdout is a StringIO, not a terminal."""
        os.environ.pop("NO_COLOR", None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print(on_off(True), on_off(False))
        self.assertNotIn("\033", out.getvalue())
        self.assertIn("on off", out.getvalue())


class BannerTests(unittest.TestCase):
    def _status(self, **overrides):
        import dataclasses

        from orbit.terminal.runtime_status import RuntimeStatus

        values = {field.name: "x" for field in dataclasses.fields(RuntimeStatus)}
        values.update(
            version="0.1", workdir="~/w", model="m", backend="native",
            banner_model="Ornith", context_window="8192",
            tools="on", think="off", autonomous="off",
        )
        values.update(overrides)
        return RuntimeStatus(**values)

    def test_the_banner_names_autonomy_between_think_and_ctx(self) -> None:
        from orbit.terminal.runtime_status import format_startup_banner

        banner = format_startup_banner(self._status())
        self.assertIn(
            "tools on · think off · autonomous off · ctx 8192", banner
        )

    def test_the_banner_colours_only_the_state_tokens(self) -> None:
        from orbit.terminal.runtime_status import format_startup_banner

        with mock.patch("orbit.terminal.theme.is_tty", return_value=True), \
                mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            banner = format_startup_banner(self._status())

        self.assertIn("tools \033[32mon\033[0m", banner)
        self.assertIn("think \033[31moff\033[0m", banner)
        self.assertIn("autonomous \033[31moff\033[0m", banner)
        # The labels and the workdir carry no state and stay plain.
        self.assertNotIn("\033[32mworkdir", banner)
        self.assertNotIn("\033[31mctx", banner)

    def test_the_banner_is_plain_without_colour(self) -> None:
        from orbit.terminal.runtime_status import format_startup_banner

        with mock.patch("orbit.terminal.theme.is_tty", return_value=False):
            banner = format_startup_banner(self._status())
        self.assertNotIn("\033", banner)

    def test_a_non_boolean_tools_mode_is_left_alone(self) -> None:
        """`tools` can read `restricted`, which has no green-or-red answer."""
        from orbit.terminal.runtime_status import format_startup_banner

        with mock.patch("orbit.terminal.theme.is_tty", return_value=True), \
                mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            banner = format_startup_banner(self._status(tools="restricted"))

        self.assertIn("tools restricted", banner)
        self.assertNotIn("\033[32mrestricted", banner)
        self.assertNotIn("\033[31mrestricted", banner)

    def test_the_banner_reports_the_session_setting(self) -> None:
        from orbit.terminal.runtime_status import format_startup_banner

        banner = format_startup_banner(self._status(autonomous="on"))
        self.assertIn("autonomous on", banner)


class BannerReflectsSessionTests(UXTestBase):
    """Built through the real collector, so the banner cannot claim a
    setting the session does not hold."""

    def _banner_for(self, autonomous: bool) -> str:
        from orbit.terminal.runtime_status import (
            collect_runtime_status, format_startup_banner,
        )

        repl = self._repl()
        repl.autonomous_analysis = autonomous
        status = collect_runtime_status(
            repl.runtime, repl.config, repl.backend,
            tools_mode=repl.tools_mode,
            autonomous=bool(repl.autonomous_analysis),
        )
        return format_startup_banner(status)

    def test_autonomy_on_is_reported_on_the_banner(self) -> None:
        self.assertIn("autonomous on", self._banner_for(True))

    def test_autonomy_off_is_reported_on_the_banner(self) -> None:
        banner = self._banner_for(False)
        self.assertIn("autonomous off", banner)
        self.assertNotIn("autonomous on", banner)


class EchoWidthTests(unittest.TestCase):
    """The rows erased must match the rows the prompt actually occupied."""

    def test_the_echo_marker_is_the_displayed_prompt(self) -> None:
        """A marker narrower than the real prompt erases one row too few once
        the difference pushes a typed line past the terminal edge."""
        from orbit.terminal.repl_input import input_prompt, prompt_marker

        import orbit.terminal.repl_input as repl_input

        original = repl_input.sys.stdout
        repl_input.sys.stdout = mock.Mock(**{"isatty.return_value": False})
        try:
            for label, autonomous in (("chat", False), ("analysis", True)):
                with self.subTest(label=label):
                    self.assertEqual(
                        prompt_marker(label, autonomous=autonomous),
                        input_prompt(label, autonomous=autonomous),
                    )
        finally:
            repl_input.sys.stdout = original

    def _rows_erased(self, fn, prompt: str, label: str, autonomous: bool) -> int:
        """The row count `fn` actually emits, read off its escape sequence."""
        import re

        import orbit.terminal.repl_input as repl_input

        # A stream that both claims to be a terminal and records what is
        # written to it: the function checks `isatty` and then prints.
        class _RecordingTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        out = _RecordingTTY()
        original = repl_input.sys.stdout
        repl_input.sys.stdout = out
        try:
            with mock.patch(
                "orbit.terminal.repl_input.get_terminal_size",
                return_value=os.terminal_size((80, 20)),
            ), contextlib.redirect_stdout(out):
                fn(prompt, label, autonomous=autonomous)
        finally:
            repl_input.sys.stdout = original
        match = re.search(r"\x1b\[(\d+)F", out.getvalue())
        self.assertIsNotNone(match, "the function must emit a cursor-up count")
        return int(match.group(1))

    def test_clear_input_echo_erases_every_row_it_wrote(self) -> None:
        """The real function, at the width where a narrower marker under-counts.

        70 typed characters fit on one row after `chat> ` but wrap after
        `chat [auto:off]> `, so a marker taken from the bare mode leaves a
        stale row behind on screen.
        """
        from orbit.terminal.repl_input import clear_input_echo

        rows = self._rows_erased(clear_input_echo, "x" * 70, "chat", False)
        self.assertEqual(rows, 2)

    def test_row_count_reflects_the_wider_marker(self) -> None:
        from orbit.terminal.repl_input import prompt_marker, visual_row_count

        marker = prompt_marker("chat", autonomous=False)
        typed = "x" * 70
        self.assertEqual(
            visual_row_count(f"{marker}{typed}", columns=80),
            2,
            "marker plus 70 characters wraps at 80 columns",
        )
        # The bare mode would have under-counted this exact case.
        self.assertEqual(visual_row_count(f"chat> {typed}", columns=80), 1)


class StatusPanelTests(UXTestBase):
    """`/status` is what the banner points at; the two must agree."""

    def _panel(self, autonomous: bool) -> str:
        from orbit.terminal.commands import runtime_status

        repl = self._repl()
        repl.autonomous_analysis = autonomous
        return runtime_status(
            repl.runtime, repl.config, repl.backend,
            tools_mode=repl.tools_mode,
            autonomous=bool(repl.autonomous_analysis),
        )

    def test_status_reports_autonomy_on(self) -> None:
        panel = self._panel(True)
        self.assertIn("Autonomous", panel)
        self.assertRegex(panel, r"Autonomous\s+on")

    def test_status_reports_autonomy_off(self) -> None:
        self.assertRegex(self._panel(False), r"Autonomous\s+off")


class AnalysisHelpTests(unittest.TestCase):
    """Four commands, four obviously different contracts."""

    def _analysis_help(self) -> dict[str, str]:
        from orbit.terminal.command_registry import COMMANDS

        return {
            command.name: command.description
            for command in COMMANDS
            if command.category == "Analysis"
        }

    def test_analysis_starts_and_collects(self) -> None:
        text = self._analysis_help()["/analysis"].lower()
        self.assertIn("start analysis", text)
        self.assertIn("evidence", text)

    def test_autonomous_controls_progress_and_starts_nothing(self) -> None:
        text = self._analysis_help()["/autonomous"].lower()
        self.assertIn("control how analysis advances", text)
        self.assertIn("one evidence step at a time", text)
        self.assertIn("does not enter analysis mode by itself", text)
        self.assertNotIn("start analysis", text)

    def test_report_runs_no_new_action(self) -> None:
        text = self._analysis_help()["/report"].lower()
        self.assertIn("evidence already collected", text)
        self.assertIn("runs no new analysis action", text)

    def test_chat_leaves_analysis(self) -> None:
        text = self._analysis_help()["/chat"].lower()
        self.assertIn("leave analysis", text)

    def test_the_four_contracts_are_distinct(self) -> None:
        descriptions = self._analysis_help()
        self.assertEqual(len(descriptions), 4)
        self.assertEqual(len(set(descriptions.values())), 4)


if __name__ == "__main__":
    unittest.main()
