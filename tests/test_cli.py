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
from orbit.terminal import cli
from orbit.backend.base import ChatResult


class CliTests(unittest.TestCase):
    def test_one_shot_think_on_does_not_crash(self) -> None:
        completed = _run_cli("", "--think", "on", "/think")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("think: on", completed.stdout)

    def test_one_shot_status_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/status")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("base_url: http://127.0.0.1:12120", completed.stdout)
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

    def test_health_flag_does_not_enter_chat(self) -> None:
        completed = _run_cli("", "--health")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Health\n------", completed.stdout)
        self.assertIn("base_url: http://127.0.0.1:12120", completed.stdout)
        self.assertIn("server:", completed.stdout)
        self.assertNotIn("orbit interactive mode", completed.stdout)

    def test_one_shot_tools_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/tools")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("tools: off", completed.stdout)
        self.assertIn("/tools off = chat only", completed.stdout)
        self.assertIn("/tools on  = unrestricted local shell", completed.stdout)
        self.assertNotIn("/tools files", completed.stdout)
        self.assertNotIn("/tools web", completed.stdout)
        self.assertNotIn("/tools time", completed.stdout)
        self.assertNotIn("Groups:", completed.stdout)
        self.assertNotIn("Single tools:", completed.stdout)
        self.assertNotIn("llama-server:", completed.stdout)
        self.assertNotIn("orbit-only:", completed.stdout)

    def test_one_shot_think_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/think")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("think: off", completed.stdout)
        self.assertIn("/think off = suppress reasoning", completed.stdout)
        self.assertIn("/think on  = show reasoning before the final answer", completed.stdout)

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
        self.assertIn("base_url: http://127.0.0.1:12120", completed.stdout)
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

    def test_repl_think_command_updates_status(self) -> None:
        completed = _run_cli("/think on\n/status\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("think: on", completed.stdout)
        self.assertIn("thinking_mode: on", completed.stdout)

    def test_one_shot_length_footer_suggests_larger_budget(self) -> None:
        class FakeRuntime:
            messages = []
            context_tokens = None

            def ask_chat(self, *args, **kwargs):
                on_final_delta = kwargs["on_final_delta"]
                on_final_delta("partial")
                return ChatResult(
                    content="partial",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=32,
                    cached_tokens=0,
                    prompt_tokens_per_second=100.0,
                    generation_tokens_per_second=10.0,
                )

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = cli._run_one_shot(
                FakeRuntime(),
                "hello",
                image_paths=[],
                audio_paths=[],
                temperature=0.0,
                max_tokens=32,
                workdir=ROOT,
                tools="off",
                thinking=False,
            )

        output = stream.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("output stopped because max_tokens was reached", output)
        self.assertIn("rerun with --max-tokens N for a larger one-shot budget", output)

    def test_one_shot_length_footer_mentions_thinking_when_enabled(self) -> None:
        class FakeRuntime:
            messages = []
            context_tokens = None

            def ask_chat(self, *args, **kwargs):
                on_final_delta = kwargs["on_final_delta"]
                on_final_delta("partial")
                return ChatResult(
                    content="partial",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=32,
                    cached_tokens=0,
                    prompt_tokens_per_second=100.0,
                    generation_tokens_per_second=10.0,
                )

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = cli._run_one_shot(
                FakeRuntime(),
                "hello",
                image_paths=[],
                audio_paths=[],
                temperature=0.0,
                max_tokens=32,
                workdir=ROOT,
                tools="off",
                thinking=True,
            )

        output = stream.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("thinking or final output stopped because max_tokens was reached", output)
        self.assertIn("rerun with --max-tokens N for a larger one-shot budget", output)

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
