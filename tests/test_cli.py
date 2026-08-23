from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.sessions import SessionStore
from orbit.runtime.turn_trace import ModelStepMetrics
from orbit.terminal.session_selection import display_datetime, preview_prompt, select_interactive_session
from orbit.terminal import cli
from orbit.backend.base import ChatResult


class CliTests(unittest.TestCase):
    def test_cli_startup_runs_bounded_artifact_recovery_for_active_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            config = SimpleNamespace(
                workdir=workdir,
                base_url="http://127.0.0.1:12120",
                timeout=30.0,
                think=False,
            )
            stream = io.StringIO()
            with mock.patch(
                "orbit.terminal.cli.load_app_config", return_value=config
            ), mock.patch(
                "orbit.terminal.cli.cleanup_stale_artifact_entries"
            ) as cleanup, mock.patch(
                "orbit.terminal.cli.LlamaServerBackend"
            ), mock.patch(
                "orbit.terminal.cli.health_text", return_value="health"
            ), contextlib.redirect_stdout(stream):
                code = cli.main(["--health"])

            self.assertEqual(code, 0)
            cleanup.assert_called_once_with(workdir)
            self.assertEqual(stream.getvalue().strip(), "health")

    def test_one_shot_footer_reports_complete_turn_token_usage(self) -> None:
        class FakeRuntime:
            messages = []
            context_tokens = None

            def ask_chat(self, *args, **kwargs):
                on_model_step = kwargs["on_model_step"]
                on_model_step(ModelStepMetrics(1, "route", "stop", 100, 5, 20, 10.0, 2.0, 0))
                on_model_step(ModelStepMetrics(1, "chat_final", "stop", 300, 15, 0, 10.0, 2.0, 0))
                return ChatResult(
                    content="complete",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=300,
                    completion_tokens=15,
                    cached_tokens=0,
                    prompt_tokens_per_second=10.0,
                    generation_tokens_per_second=2.0,
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

        self.assertEqual(code, 0)
        self.assertIn("2 calls · stop", stream.getvalue())
        self.assertIn("tokens: 400 in · 380 eval · 20 cache · 20 out", stream.getvalue())
        self.assertIn("last call: 10.0 tok/s prefill · 2.0 tok/s decode", stream.getvalue())

    def test_one_shot_think_on_does_not_crash(self) -> None:
        completed = _run_cli("", "--think", "on", "/think")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("think: on", completed.stdout)

    def test_one_shot_status_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/status")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("┌─ Orbit Runtime", completed.stdout)
        self.assertIn("Backend", completed.stdout)
        self.assertIn("Messages     1", completed.stdout)
        self.assertNotIn("model: fake", completed.stdout)
        self.assertNotIn("Type /help for commands", completed.stdout)

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
        self.assertIn("tools: on", completed.stdout)
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
        self.assertIn("/help for commands · /status for details", completed.stdout)
        self.assertIn("┌─ Orbit Runtime", completed.stdout)
        self.assertIn("Messages     1", completed.stdout)

    def test_repl_status_context_command_does_not_call_model(self) -> None:
        completed = _run_cli("/status ctx\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("/help for commands · /status for details", completed.stdout)
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
        self.assertIn("Max tokens   2048", completed.stdout)

    def test_repl_think_command_updates_status(self) -> None:
        completed = _run_cli("/think on\n/status\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("think: on", completed.stdout)
        self.assertIn("Think        on", completed.stdout)

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
            env={
                "PYTHONPATH": str(ROOT / "src"),
                "HOME": home,
                # These drive the REPL itself, not readiness; there is no
                # server behind the subprocess.
                "ORBIT_SKIP_SERVER_READINESS": "1",
            },
            check=False,
        )



class ServerReadinessTests(unittest.TestCase):
    """An interactive session must not open against a server that is not there.

    Drawing `chat>` and then failing on whatever the analyst types reads as
    their mistake rather than a missing server.
    """

    def _args(self, tmp: Path):
        return [
            "--workdir", str(tmp),
            "--base-url", "http://127.0.0.1:12120",
        ]

    def _run(self, *, healthy: bool):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            backend = mock.MagicMock()
            backend.health.return_value = healthy
            backend.model_info.return_value = None
            err = io.StringIO()
            out = io.StringIO()
            with (
                mock.patch.object(cli, "LlamaServerBackend", return_value=backend),
                mock.patch.object(cli, "Repl") as repl_cls,
                contextlib.redirect_stderr(err),
                contextlib.redirect_stdout(out),
            ):
                repl_cls.return_value.run.return_value = 0
                code = cli.main(self._args(tmp))
            return code, err.getvalue(), out.getvalue(), backend, repl_cls

    def test_an_unavailable_server_exits_before_the_prompt(self) -> None:
        code, err, out, backend, repl_cls = self._run(healthy=False)

        self.assertEqual(code, 1)
        repl_cls.assert_not_called()
        self.assertNotIn("chat>", out)
        self.assertIn("Orbit server is not ready", err)

    def test_the_error_names_the_endpoint_and_the_remedy(self) -> None:
        _, err, _, _, _ = self._run(healthy=False)

        self.assertIn("127.0.0.1:12120", err)
        self.assertIn("Start the server and try again", err)
        self.assertEqual(len(err.strip().splitlines()), 1, "one concise line")

    def test_it_does_not_retry(self) -> None:
        _, _, _, backend, _ = self._run(healthy=False)

        self.assertEqual(backend.health.call_count, 1, "no hidden retry loop")

    def test_a_ready_server_enters_the_repl(self) -> None:
        code, err, _, backend, repl_cls = self._run(healthy=True)

        self.assertEqual(code, 0)
        repl_cls.assert_called_once()
        self.assertNotIn("not ready", err)

    def test_it_reuses_the_existing_health_mechanism(self) -> None:
        """No second protocol: the check is `backend.health()`."""
        _, _, _, backend, _ = self._run(healthy=False)

        backend.health.assert_called()

    def test_the_escape_hatch_is_explicit_and_narrow(self) -> None:
        """Only the exact opt-in string skips the check.

        The hatch exists so subprocess REPL tests can run without a server. A
        loose truthiness test would turn any stray value into a silent bypass.
        """
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            for value, expect_repl in (("1", True), ("0", False), ("true", False), ("", False)):
                backend = mock.MagicMock()
                backend.health.return_value = False
                backend.model_info.return_value = None
                with (
                    mock.patch.dict("os.environ", {"ORBIT_SKIP_SERVER_READINESS": value}),
                    mock.patch.object(cli, "LlamaServerBackend", return_value=backend),
                    mock.patch.object(cli, "Repl") as repl_cls,
                    contextlib.redirect_stderr(io.StringIO()),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    repl_cls.return_value.run.return_value = 0
                    cli.main(self._args(tmp))
                    with self.subTest(value=value):
                        self.assertEqual(repl_cls.called, expect_repl)

    def test_endpoint_label_falls_back_to_the_configured_url(self) -> None:
        self.assertEqual(cli._endpoint_label("http://127.0.0.1:12120"), "127.0.0.1:12120")
        self.assertEqual(cli._endpoint_label("http://example.test"), "example.test")
        self.assertEqual(cli._endpoint_label("not-a-url"), "not-a-url")


if __name__ == "__main__":
    unittest.main()
