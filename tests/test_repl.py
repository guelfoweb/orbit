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

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.terminal.config import AppConfig
from orbit.terminal.repl import Repl
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


if __name__ == "__main__":
    unittest.main()
