from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.sessions import SessionStore
from orbit.terminal.session_selection import display_datetime, preview_prompt, select_interactive_session


class CliTests(unittest.TestCase):
    def test_one_shot_status_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/status")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("base_url: http://127.0.0.1:18080", completed.stdout)
        self.assertIn("server:", completed.stdout)
        self.assertIn("messages: 1", completed.stdout)
        self.assertNotIn("model: fake", completed.stdout)

    def test_one_shot_status_context_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/status ctx")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Context\n-------", completed.stdout)
        self.assertIn("Token estimate\n--------------", completed.stdout)
        self.assertIn("Message count\n-------------", completed.stdout)
        self.assertIn("system:", completed.stdout)
        self.assertNotIn("model: fake", completed.stdout)

    def test_one_shot_tools_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/tools")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("tools: off", completed.stdout)
        self.assertIn("/tools files = read/inspect local files", completed.stdout)
        self.assertIn("/tools web   = search/fetch URLs", completed.stdout)
        self.assertNotIn("/tools time", completed.stdout)
        self.assertNotIn("Groups:", completed.stdout)
        self.assertNotIn("Single tools:", completed.stdout)
        self.assertNotIn("llama-server:", completed.stdout)
        self.assertNotIn("orbit-only:", completed.stdout)

    def test_one_shot_max_tokens_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/max-tokens 2048")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("max_tokens: 2048", completed.stdout)

    def test_one_shot_compact_command_is_interactive_only(self) -> None:
        completed = _run_cli("", "/compact")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("error: /compact is available only in interactive mode", completed.stdout)

    def test_repl_status_command_does_not_call_model(self) -> None:
        completed = _run_cli("/status\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("orbit interactive mode", completed.stdout)
        self.assertIn("base_url: http://127.0.0.1:18080", completed.stdout)
        self.assertIn("server:", completed.stdout)
        self.assertIn("messages: 1", completed.stdout)
        self.assertIn("workdir:", completed.stdout)

    def test_repl_status_context_command_does_not_call_model(self) -> None:
        completed = _run_cli("/status ctx\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("orbit interactive mode", completed.stdout)
        self.assertIn("Context\n-------", completed.stdout)
        self.assertIn("tool_result:", completed.stdout)

    def test_status_context_alias_still_works(self) -> None:
        completed = _run_cli("", "/status context")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Context\n-------", completed.stdout)

    def test_repl_unknown_command_is_not_sent_to_model(self) -> None:
        completed = _run_cli("/unknown\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("unknown command: /unknown", completed.stderr)

    def test_repl_max_tokens_command_updates_status(self) -> None:
        completed = _run_cli("/max-tokens\n/max-tokens 2048\n/status\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("max_tokens: 512", completed.stdout)
        self.assertIn("max_tokens: 2048", completed.stdout)

    def test_select_interactive_session_uses_new_session_when_stdin_is_not_tty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            root = Path(tmp) / "sessions"
            existing = SessionStore.for_workdir(workdir, root=root)
            existing.save(messages=[{"role": "user", "content": "old"}], workdir=workdir, model="m", base_url="u")

            selected = select_interactive_session(workdir, root=root)

            self.assertNotEqual(selected.path, existing.path)
            self.assertIsNone(selected.load())

    def test_select_interactive_session_can_choose_existing_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            root = Path(tmp) / "sessions"
            existing = SessionStore.for_workdir(workdir, root=root)
            existing.save(messages=[{"role": "user", "content": "old"}], workdir=workdir, model="m", base_url="u")
            fake_stdin = mock.Mock()
            fake_stdin.isatty.return_value = True

            with (
                mock.patch("sys.stdin", fake_stdin),
                mock.patch("builtins.input", return_value="1"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                selected = select_interactive_session(workdir, root=root)

            self.assertEqual(selected.path, existing.path)

    def test_select_interactive_session_blank_starts_clean_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            root = Path(tmp) / "sessions"
            existing = SessionStore.for_workdir(workdir, root=root)
            existing.save(messages=[{"role": "user", "content": "old"}], workdir=workdir, model="m", base_url="u")
            fake_stdin = mock.Mock()
            fake_stdin.isatty.return_value = True

            with (
                mock.patch("sys.stdin", fake_stdin),
                mock.patch("builtins.input", return_value=""),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                selected = select_interactive_session(workdir, root=root)

            self.assertNotEqual(selected.path, existing.path)
            self.assertIsNone(selected.load())

    def test_preview_prompt_truncates_long_text(self) -> None:
        preview = preview_prompt("a" * 100, limit=10)

        self.assertEqual(preview, "aaaaaaaaaa...")

    def test_display_datetime_formats_iso_timestamp(self) -> None:
        value = display_datetime("2026-06-11T10:00:00+00:00")

        self.assertRegex(value, r"2026-06-11 \d{2}:00:00")


def _run_cli(stdin: str, *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as home:
        return subprocess.run(
            [sys.executable, "-m", "orbit.terminal.cli", *args],
            cwd=ROOT,
            input=stdin,
            text=True,
            capture_output=True,
            env={"PYTHONPATH": str(ROOT / "src"), "HOME": home},
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
