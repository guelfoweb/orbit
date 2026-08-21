"""An intentionally aborted route stream is an attempt, not a failure.

The route phase stops its own stream as soon as the output cannot become a
tool decision. That early stop travels out of the backend as an exception, so
it used to be counted as a failed backend call and every ordinary
conversational turn reported a failure that never happened. These tests pin
the distinction end to end, through the real backend and runtime, with only
the HTTP layer scripted.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.backend.llama_server as llama_server
from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime.chat import ChatRuntime
from orbit.terminal.status import (
    TokenUsageAccumulator,
    format_session_token_usage,
    format_turn_status,
)

PROSE_CHUNKS = ["I'm", " doing", " well", ", thanks!"]
ROUTE_CHUNKS = ['{"command"', ': "pwd"}']


class _ScriptedStream:
    """The native SSE wire format, replayed from a list of text chunks."""

    def __init__(self, chunks: list[str], *, prompt_tokens: int, completion_tokens: int) -> None:
        self._lines: list[bytes] = []
        for chunk in chunks:
            self._lines.append(b"event: delta\n")
            self._lines.append(b"data: " + json.dumps({"text": chunk}).encode() + b"\n")
            self._lines.append(b"\n")
        self._lines.append(b"event: metrics\n")
        self._lines.append(
            b"data: "
            + json.dumps(
                {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}
            ).encode()
            + b"\n"
        )
        self._lines.append(b"\n")
        self._lines.append(b"event: done\n")
        self._lines.append(b"data: " + json.dumps({"finish_reason": "stop", "model": "m"}).encode() + b"\n")
        self._lines.append(b"\n")

    def __enter__(self) -> "_ScriptedStream":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return b""


class RouteAbortTelemetryTest(unittest.TestCase):
    def _backend(self) -> tuple[LlamaServerBackend, TokenUsageAccumulator]:
        backend = LlamaServerBackend(base_url="http://127.0.0.1:1", timeout=5)
        usage = TokenUsageAccumulator()
        backend.set_result_observer(usage.add_result)
        backend.set_failure_observer(usage.add_failed_call)
        backend.set_aborted_observer(usage.add_aborted_call)
        return backend, usage

    def _run(self, backend: LlamaServerBackend, phases: list[tuple[str, int, str | None]]):
        runtime = ChatRuntime(backend=backend, system_prompt="sys")
        return runtime.ask_auto(
            "hi, how are you?",
            temperature=0,
            max_tokens=64,
            workdir=Path("."),
            on_final_delta=lambda _text: None,
            on_progress=lambda _item: None,
            on_phase_start=lambda ph: phases.append((ph.phase, ph.attempt, ph.reason)),
            allowed_tool_names=("exec_shell_full_command",),
        )

    def _patched(self, responses):
        """Serve each backend call from `responses`, which may raise instead."""
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            index = min(calls["n"], len(responses) - 1)
            calls["n"] += 1
            item = responses[index]
            if isinstance(item, Exception):
                raise item
            return item()

        return mock.patch.object(llama_server, "urlopen", fake_urlopen), calls

    def _stub_profile(self):
        return mock.patch.multiple(
            LlamaServerBackend,
            _is_orbit_native_backend=lambda self: True,
            request_model_name=lambda self: "test-model",
            _props_or_empty=lambda self: {},
            _serialize_for_profile=lambda self, messages: list(messages),
        )

    def test_prose_route_abort_counts_an_attempt_not_a_failure(self) -> None:
        backend, usage = self._backend()
        phases: list[tuple[str, int, str | None]] = []
        patch, calls = self._patched(
            [lambda: _ScriptedStream(PROSE_CHUNKS, prompt_tokens=40, completion_tokens=8)]
        )
        with self._stub_profile(), patch:
            result = self._run(backend, phases)

        # The early stop still happens, and the runtime still recovers through
        # the existing handler: route -> chat_final_retry(route_not_command).
        self.assertEqual(
            [(phase, reason) for phase, _attempt, reason in phases],
            [("route", "tool_decision"), ("chat_final_retry", "route_not_command")],
        )
        self.assertEqual(result.content, "I'm doing well, thanks!")

        snapshot = usage.snapshot()
        self.assertEqual(snapshot.model_calls, 2, "route attempt and final call both count")
        self.assertEqual(snapshot.failed_calls, 0, "an intentional abort is not a failure")
        # The aborted stream never delivered its metrics event, so the totals
        # are real but short -- saying otherwise would invent usage.
        self.assertTrue(snapshot.usage_incomplete)

        # What the user actually reads: no failure warning, but still an honest
        # "partial" on the totals the abort left short.
        turn_status = format_turn_status(
            result, elapsed_seconds=1.0, turn_token_usage=snapshot
        )
        self.assertNotIn("failed attempt", turn_status)
        self.assertIn("(partial)", turn_status)
        session_line = format_session_token_usage(snapshot)
        self.assertNotIn("failed attempts", session_line)
        self.assertIn("partial", session_line)

    def test_genuine_route_decision_is_unaffected(self) -> None:
        backend, usage = self._backend()
        phases: list[tuple[str, int, str | None]] = []
        patch, _calls = self._patched(
            [lambda: _ScriptedStream(ROUTE_CHUNKS, prompt_tokens=40, completion_tokens=8)]
        )
        with self._stub_profile(), patch:
            self._run(backend, phases)

        snapshot = usage.snapshot()
        self.assertEqual(snapshot.failed_calls, 0)
        self.assertFalse(snapshot.usage_incomplete, "a completed route reports full usage")
        self.assertEqual([phase for phase, _a, _r in phases][0], "route")

    def test_genuine_backend_failure_still_counts_as_failed(self) -> None:
        backend, usage = self._backend()
        with self._stub_profile(), mock.patch.object(
            llama_server, "urlopen", side_effect=URLError("connection reset")
        ):
            with self.assertRaises(LlamaServerError):
                backend.chat_stream(
                    [{"role": "user", "content": "hi"}],
                    temperature=0,
                    max_tokens=8,
                    on_delta=lambda _text: None,
                )

        snapshot = usage.snapshot()
        self.assertEqual(snapshot.model_calls, 1)
        self.assertEqual(snapshot.failed_calls, 1, "a real transport error is still a failure")
        self.assertTrue(snapshot.usage_incomplete)


if __name__ == "__main__":
    unittest.main()
