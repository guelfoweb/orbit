"""Interactive control of autonomous analysis, for one session only.

Autonomy was reachable only by exporting an environment variable before
starting Orbit, so changing your mind meant restarting and losing the session.
`/autonomous` moves that decision inside the process without moving the policy:
the runtime still owns what autonomy does, when it stops and what it costs.
The terminal owns only whether the next analysis turn asks for it.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.analysis_runtime import ANALYSIS_AUTONOMY_ENV
from orbit.terminal.command_registry import COMMANDS, commands_matching, resolve_command
from orbit.terminal.commands import help_text

from tests.test_workflow_mode import ModeTestBase


class CommandDiscoveryTests(unittest.TestCase):
    def test_the_command_resolves_with_and_without_an_argument(self) -> None:
        for value, expected in (
            ("/autonomous", ""),
            ("/autonomous on", "on"),
            ("/autonomous off", "off"),
            ("/autonomous   on  ", "on"),
        ):
            with self.subTest(value=value):
                invocation = resolve_command(value)
                self.assertIsNotNone(invocation)
                self.assertEqual(invocation.spec.handler, "autonomous")
                self.assertEqual(invocation.arguments, expected)

    def test_the_command_is_discoverable_by_prefix(self) -> None:
        self.assertIn("/autonomous", [c.name for c in commands_matching("/auto")])

    def test_it_is_grouped_with_the_other_analysis_commands(self) -> None:
        spec = next(c for c in COMMANDS if c.name == "/autonomous")
        self.assertEqual(spec.category, "Analysis")


class HelpTests(unittest.TestCase):
    """Help must make the mode controls findable without a redesign."""

    def _headings(self) -> list[str]:
        return [line for line in help_text().splitlines() if line and not line.startswith("/")]

    def test_no_duplicate_headings(self) -> None:
        from collections import Counter

        repeated = [name for name, count in Counter(self._headings()).items() if count > 1]
        self.assertEqual(repeated, [])

    def test_every_command_sits_under_exactly_one_heading(self) -> None:
        """Coherence, asserted structurally rather than by counting headings.

        A budget on the number of groups fails for reasons unrelated to this
        feature the moment someone adds a category. What matters is that each
        command is reachable under one heading and no heading is emitted twice.
        """
        from orbit.terminal.command_registry import COMMANDS

        text = help_text()
        headings = self._headings()
        for command in COMMANDS:
            with self.subTest(command=command.name):
                self.assertIn(command.usage, text)
                self.assertIn(command.category, headings)
        self.assertEqual(len(set(headings)), len(headings))

    def test_the_mode_commands_are_discoverable_together(self) -> None:
        text = help_text()
        analysis_block = text[text.index("Analysis") :]
        for usage in ("/analysis <path>", "/report", "/autonomous [off|on]", "/chat"):
            self.assertIn(usage, analysis_block)

    def test_help_describes_the_autonomy_control(self) -> None:
        self.assertIn("autonomous analysis for this session", help_text().lower())


class SessionStateTests(ModeTestBase):
    """State is initialised from the runtime gate and overridden interactively."""

    def test_a_startup_default_is_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ANALYSIS_AUTONOMY_ENV, None)
            self.assertFalse(self.repl().autonomous_analysis)

    def test_the_startup_environment_enables_it(self) -> None:
        with mock.patch.dict(os.environ, {ANALYSIS_AUTONOMY_ENV: "1"}):
            self.assertTrue(self.repl().autonomous_analysis)

    def test_only_the_exact_value_enables_it(self) -> None:
        for value in ("true", "TRUE", "yes", "on", "01", "2", "", " "):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {ANALYSIS_AUTONOMY_ENV: value}):
                    self.assertFalse(self.repl().autonomous_analysis)

    def test_status_reports_the_current_state(self) -> None:
        repl = self.repl()
        repl.autonomous_analysis = False
        self.assertIn("off", repl._handle_autonomous_command(""))
        repl.autonomous_analysis = True
        self.assertIn("on", repl._handle_autonomous_command(""))

    def test_asking_for_the_state_does_not_change_it(self) -> None:
        """A query is not a command.

        Asserting only the reported string cannot catch a status path that also
        assigns: each assertion would pass while `/autonomous` silently turned
        autonomy off for anyone who merely asked what it was set to.
        """
        for before in (True, False):
            with self.subTest(before=before):
                repl = self.repl()
                repl.autonomous_analysis = before
                for _ in range(3):
                    repl._handle_autonomous_command("")
                    self.assertEqual(repl.autonomous_analysis, before)
                # Whitespace is a query too, not an invalid argument.
                repl._handle_autonomous_command("   ")
                self.assertEqual(repl.autonomous_analysis, before)

    def test_enabling_and_disabling_apply_immediately(self) -> None:
        repl = self.repl()
        repl._handle_autonomous_command("on")
        self.assertTrue(repl.autonomous_analysis)
        repl._handle_autonomous_command("off")
        self.assertFalse(repl.autonomous_analysis)

    def test_an_interactive_override_beats_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {ANALYSIS_AUTONOMY_ENV: "1"}):
            repl = self.repl()
            self.assertTrue(repl.autonomous_analysis)
            repl._handle_autonomous_command("off")
            self.assertFalse(repl.autonomous_analysis)
            # And the environment is not rewritten to make it so.
            self.assertEqual(os.environ[ANALYSIS_AUTONOMY_ENV], "1")

    def test_the_environment_is_never_mutated(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ANALYSIS_AUTONOMY_ENV, None)
            repl = self.repl()
            repl._handle_autonomous_command("on")
            self.assertNotIn(ANALYSIS_AUTONOMY_ENV, os.environ)

    def test_an_invalid_argument_leaves_the_state_alone(self) -> None:
        """Checked from BOTH starting states.

        Testing only from `on` would miss a mutant that sets `on` on rejection:
        the assertion would still pass while the command silently enabled
        autonomy for anyone who mistyped it.
        """
        for value in ("maybe", "1", "true", "ON!", "enable"):
            for before in (True, False):
                with self.subTest(value=value, before=before):
                    repl = self.repl()
                    repl.autonomous_analysis = before
                    message = repl._handle_autonomous_command(value)
                    self.assertTrue(message.startswith("error:"), message)
                    self.assertIn("/autonomous [off|on]", message)
                    self.assertEqual(
                        repl.autonomous_analysis, before, "state must not change"
                    )

    def test_case_is_accepted_but_the_value_must_be_exact(self) -> None:
        repl = self.repl()
        repl.autonomous_analysis = False
        repl._handle_autonomous_command("ON")
        self.assertTrue(repl.autonomous_analysis)


class ModeTransitionTests(ModeTestBase):
    """Autonomy is a property of the session, not of the current mode."""

    def test_the_selection_survives_chat_and_analysis_switching(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        repl._handle_autonomous_command("on")

        self.run_command(repl, "/chat")
        self.assertTrue(repl.autonomous_analysis, "cleared by leaving analysis")

        self.run_command(repl, f"/analysis {self.artifact}")
        self.assertTrue(repl.autonomous_analysis, "cleared by re-entering analysis")

        self.run_command(repl, "/chat")
        repl._handle_autonomous_command("off")
        self.run_command(repl, f"/analysis {self.artifact}")
        self.assertFalse(repl.autonomous_analysis)

    def test_existing_chat_and_analysis_commands_still_work(self) -> None:
        """`/chat` switches mode and deliberately keeps the analysis session.

        Closing it would destroy artifacts the analyst may still want, so the
        session is released by the ordinary lifecycle instead. Asserted here so
        that adding a mode control cannot quietly change it.
        """
        from orbit.runtime.workflow_mode import WorkflowMode

        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.assertIsNotNone(repl.analysis)
        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)

        self.run_command(repl, "/chat")
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNotNone(repl.analysis, "the session is kept, not closed")

    def test_the_runtime_gate_is_read_once_not_per_turn(self) -> None:
        """The session decides, so a later env change cannot override a choice."""
        import inspect

        from orbit.terminal.repl import Repl

        source = inspect.getsource(Repl._ask_analysis)
        self.assertIn("self.autonomous_analysis", source)
        self.assertNotIn("analysis_autonomy_enabled()", source)


class PolicyOwnershipTests(unittest.TestCase):
    """The terminal controls the switch; the runtime keeps the policy."""

    def test_the_terminal_does_not_reimplement_autonomy_policy(self) -> None:
        import inspect

        from orbit.terminal.repl import Repl

        source = inspect.getsource(Repl._handle_autonomous_command)
        for policy in ("MAX_AUTONOMOUS", "SOFT_MAX", "replan", "NEW_CONTENT", "stop_reason"):
            self.assertNotIn(policy, source, f"{policy} belongs to the runtime")

    def test_the_runtime_gate_is_unchanged(self) -> None:
        """`/autonomous` must not have altered the environment contract."""
        from orbit.runtime.analysis_runtime import analysis_autonomy_enabled

        self.assertEqual(ANALYSIS_AUTONOMY_ENV, "ORBIT_ANALYSIS_AUTONOMOUS")
        self.assertFalse(analysis_autonomy_enabled({}))
        self.assertTrue(analysis_autonomy_enabled({ANALYSIS_AUTONOMY_ENV: "1"}))
        self.assertFalse(analysis_autonomy_enabled({ANALYSIS_AUTONOMY_ENV: "true"}))


if __name__ == "__main__":
    unittest.main()
