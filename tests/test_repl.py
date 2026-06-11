from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.sessions import SessionStore
from orbit.terminal.config import AppConfig
from orbit.terminal.history import PromptHistory
from orbit.terminal.prompt_preview import compact_prompt_preview
from orbit.terminal.repl import Repl, _read_available_paste_tail, colorize_paste_preview
from orbit.terminal.tool_events import format_tool_result_event


class InterruptingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        return ChatResult(
            content='{"_route":"FILESYSTEM"}',
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=1,
            completion_tokens=1,
            cached_tokens=0,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None) -> ChatResult:
        assert on_delta is not None
        on_delta("partial")
        raise KeyboardInterrupt


class CountingRuntime(ChatRuntime):
    def __init__(self) -> None:
        super().__init__(backend=InterruptingBackend(), system_prompt=None)
        self.ask_calls = 0

    def ask_auto(self, *args, **kwargs) -> ChatResult:
        self.ask_calls += 1
        return ChatResult(
            content="ok",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=1,
            completion_tokens=1,
            cached_tokens=0,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def continue_last_response(self, *args, **kwargs) -> ChatResult:
        self.ask_calls += 1
        on_final_delta = kwargs.get("on_final_delta")
        if on_final_delta:
            on_final_delta("continued")
        return ChatResult(
            content="continued",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=1,
            completion_tokens=1,
            cached_tokens=0,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )


class ReplTests(unittest.TestCase):
    def test_stream_interrupt_restores_messages_and_returns_to_prompt(self) -> None:
        backend = InterruptingBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="system")
        repl = Repl(
            runtime=runtime,
            backend=backend,
            config=AppConfig(workdir=Path(".")),
        )
        before = list(runtime.messages)
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            repl._ask("hello")

        self.assertEqual(runtime.messages, before)
        self.assertIn("partial", stdout.getvalue())
        self.assertIn("interrupted", stdout.getvalue())

    def test_tool_result_event_marks_large_context(self) -> None:
        self.assertEqual(format_tool_result_event("read_file", 9999), " └ read_file 9999 chars")
        self.assertEqual(format_tool_result_event("read_file", 10000), " └ read_file 10000 chars | large context")
        self.assertEqual(format_tool_result_event("read_file", 5, "llama-server"), " └ read_file 5 chars | src: llama-server")

    def test_unresolved_history_preview_is_not_sent_to_model(self) -> None:
        long_prompt = "Copied history " + ("x" * 900)
        preview = compact_prompt_preview(long_prompt)
        with tempfile.TemporaryDirectory() as tmp:
            readline_path = Path(tmp) / "history"
            readline_path.write_text(preview + "\n", encoding="utf-8")
            history = PromptHistory(path=readline_path, readline_module=None)
            runtime = CountingRuntime()
            repl = Repl(
                runtime=runtime,
                backend=runtime.backend,
                config=AppConfig(workdir=Path(".")),
                history=history,
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                resolution = history.resolve_prompt(preview)
                if resolution.missing_full_text:
                    print("error: full pasted text is unavailable for this history entry", file=sys.stderr)

        self.assertTrue(resolution.missing_full_text)
        self.assertEqual(runtime.ask_calls, 0)
        self.assertIn("full pasted text is unavailable", stderr.getvalue())

    def test_multiline_paste_tail_is_aggregated_into_one_prompt(self) -> None:
        read_fd, write_fd = os.pipe()
        original_stdin = sys.stdin
        try:
            os.write(write_fd, b"second line\nthird line\n")
            os.close(write_fd)
            with os.fdopen(read_fd, "r", encoding="utf-8") as fake_stdin:
                sys.stdin = fake_stdin
                prompt = _read_available_paste_tail("first line", timeout=0.0, require_tty=False)
        finally:
            sys.stdin = original_stdin

        self.assertEqual(prompt, "first line\nsecond line\nthird line")

    def test_prompt_without_available_tail_stays_unchanged(self) -> None:
        read_fd, write_fd = os.pipe()
        original_stdin = sys.stdin
        try:
            os.close(write_fd)
            with os.fdopen(read_fd, "r", encoding="utf-8") as fake_stdin:
                sys.stdin = fake_stdin
                prompt = _read_available_paste_tail("single line", timeout=0.0, require_tty=False)
        finally:
            sys.stdin = original_stdin

        self.assertEqual(prompt, "single line")

    def test_colorize_paste_preview_highlights_only_badge(self) -> None:
        preview = "Lorem ipsum...\n[text 5108 chars #a1b2c3d4]"

        colored = colorize_paste_preview(preview)

        self.assertIn("Lorem ipsum...\n", colored)
        self.assertIn("\033[2m\033[33m[text 5108 chars #a1b2c3d4]\033[0m", colored)
        self.assertNotIn("\033[2m\033[33mLorem", colored)

    def test_length_footer_suggests_continue_and_max_tokens(self) -> None:
        runtime = CountingRuntime()
        repl = Repl(runtime=runtime, backend=runtime.backend, config=AppConfig(workdir=Path(".")))
        result = ChatResult(
            content="partial",
            model="fake",
            finish_reason="length",
            tool_calls=[],
            prompt_tokens=10,
            completion_tokens=32,
            cached_tokens=0,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            repl._print_turn_footer(result, elapsed_seconds=1)

        self.assertTrue(repl.can_continue)
        self.assertIn("output stopped because max_tokens was reached", stdout.getvalue())
        self.assertIn("/continue", stdout.getvalue())
        self.assertIn("/max-tokens N", stdout.getvalue())

    def test_continue_command_requires_truncated_answer(self) -> None:
        runtime = CountingRuntime()
        repl = Repl(runtime=runtime, backend=runtime.backend, config=AppConfig(workdir=Path(".")))
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            handled = repl._handle_command("/continue")

        self.assertTrue(handled)
        self.assertEqual(runtime.ask_calls, 0)
        self.assertIn("no truncated answer", stderr.getvalue())

    def test_continue_command_calls_local_continuation_when_available(self) -> None:
        runtime = CountingRuntime()
        repl = Repl(runtime=runtime, backend=runtime.backend, config=AppConfig(workdir=Path(".")))
        repl.can_continue = True
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            handled = repl._handle_command("/continue")

        self.assertTrue(handled)
        self.assertEqual(runtime.ask_calls, 1)
        self.assertIn("continued", stdout.getvalue())
        self.assertFalse(repl.can_continue)

    def test_sessions_clear_can_be_cancelled(self) -> None:
        runtime = CountingRuntime()
        repl = Repl(runtime=runtime, backend=runtime.backend, config=AppConfig(workdir=Path(".")))
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = True

        with mock.patch("sys.stdin", fake_stdin), mock.patch("builtins.input", return_value="n"):
            message = repl._clear_workdir_sessions()

        self.assertEqual(message, "sessions clear cancelled")

    def test_sessions_clear_resets_runtime_and_uses_new_session(self) -> None:
        runtime = CountingRuntime()
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            session = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            session.save(messages=[{"role": "user", "content": "hello"}], workdir=workdir, model="m", base_url="u")
            repl = Repl(runtime=runtime, backend=runtime.backend, config=AppConfig(workdir=workdir), session=session)
            fake_stdin = mock.Mock()
            fake_stdin.isatty.return_value = False

            with mock.patch("sys.stdin", fake_stdin), mock.patch.object(SessionStore, "clear_for_workdir", return_value=1):
                message = repl._clear_workdir_sessions()

        self.assertEqual(message, "sessions cleared: 1")
        self.assertEqual(runtime.messages, [])
        self.assertIsNotNone(repl.session)
        self.assertNotEqual(repl.session.path, session.path)


if __name__ == "__main__":
    unittest.main()
