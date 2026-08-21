"""The REPL is the observer the backend actually calls on an aborted stream.

Wiring prompt metrics into the accumulator is only half the job: the value has
to survive the seam between the backend and the REPL's own observer. A test
that hands `add_aborted_call` straight to the backend proves the accumulator
works and nothing about the seam, which is exactly where the metrics were
being dropped.

These drive the real `Repl._record_backend_abort` and assert both the turn and
session accumulators receive the same snapshot exactly once.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.backend.llama_server as llama_server
from orbit.backend.base import StreamConsumerAbort, StreamPromptMetrics
from orbit.backend.llama_server import LlamaServerBackend
from orbit.terminal.repl import Repl
from orbit.terminal.status import TokenUsageAccumulator, format_session_token_usage

PROMPT_TOKENS = 1105
REUSED_TOKENS = 1085
EVALUATED_TOKENS = 20


def _event(name: str, payload: dict) -> list[bytes]:
    return [
        f"event: {name}\n".encode(),
        b"data: " + json.dumps(payload).encode() + b"\n",
        b"\n",
    ]


class _Stream:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return b""


def aborted_route_stream() -> _Stream:
    lines = _event(
        "progress.prefill",
        {
            "current": PROMPT_TOKENS,
            "total": PROMPT_TOKENS,
            "percent": 100,
            "evaluated_current": EVALUATED_TOKENS,
            "evaluated_total": EVALUATED_TOKENS,
            "cached_tokens": REUSED_TOKENS,
        },
    )
    lines += _event("delta", {"text": "I'm doing well, thanks!"})
    return _Stream(lines)


class _RouteAbort(StreamConsumerAbort):
    """Stands in for the runtime's route abort: same marker, same propagation."""


class _ReplSeam:
    """Only the observer surface of Repl, bound to real accumulators.

    `Repl` is a dataclass with a large constructor, so the real method is bound
    to a minimal holder. The method under test is production code.
    """

    def __init__(self) -> None:
        self.turn_backend_token_usage = TokenUsageAccumulator()
        self.session_token_usage = TokenUsageAccumulator()
        self.calls = 0

    def record(self, prompt_metrics):
        self.calls += 1
        return Repl._record_backend_abort(self, prompt_metrics)


def backend_with_seam():
    backend = LlamaServerBackend(base_url="http://127.0.0.1:1", timeout=5)
    seam = _ReplSeam()
    backend.set_aborted_observer(seam.record)
    patches = mock.patch.multiple(
        LlamaServerBackend,
        _is_orbit_native_backend=lambda self: True,
        request_model_name=lambda self: "m",
        _props_or_empty=lambda self: {},
        _serialize_for_profile=lambda self, messages: list(messages),
    )
    return backend, seam, patches


class ReplSeamReceivesPromptMetricsTest(unittest.TestCase):
    def test_turn_and_session_both_receive_the_snapshot_once(self) -> None:
        backend, seam, patches = backend_with_seam()

        def on_delta(_text: str) -> None:
            raise _RouteAbort("route stream produced non-command prose")

        with patches, mock.patch.object(
            llama_server, "urlopen", lambda *a, **k: aborted_route_stream()
        ):
            with self.assertRaises(_RouteAbort):
                backend.chat_stream(
                    [{"role": "user", "content": "hi"}],
                    temperature=0,
                    max_tokens=8,
                    on_delta=on_delta,
                )

        self.assertEqual(seam.calls, 1, "the observer runs exactly once")

        for label, usage in (
            ("turn", seam.turn_backend_token_usage),
            ("session", seam.session_token_usage),
        ):
            with self.subTest(accumulator=label):
                snapshot = usage.snapshot()
                self.assertEqual(snapshot.model_calls, 1)
                self.assertEqual(snapshot.failed_calls, 0, "a deliberate stop is not a failure")
                self.assertEqual(
                    snapshot.cached_tokens,
                    REUSED_TOKENS,
                    "metrics must survive the backend -> REPL seam",
                )
                self.assertEqual(snapshot.prompt_tokens, PROMPT_TOKENS)
                self.assertEqual(snapshot.evaluated_tokens, EVALUATED_TOKENS)
                self.assertTrue(snapshot.usage_incomplete, "decode metrics never arrived")

        line = format_session_token_usage(seam.session_token_usage.snapshot())
        self.assertNotIn("cache: 0 (0%)", line)
        self.assertNotIn("failed attempts", line)
        self.assertIn("partial", line)

    def test_abort_without_metrics_reaches_the_seam_with_none(self) -> None:
        backend, seam, patches = backend_with_seam()
        empty = _Stream(_event("delta", {"text": "prose"}))

        def on_delta(_text: str) -> None:
            raise _RouteAbort("stop")

        with patches, mock.patch.object(llama_server, "urlopen", lambda *a, **k: empty):
            with self.assertRaises(_RouteAbort):
                backend.chat_stream(
                    [{"role": "user", "content": "hi"}],
                    temperature=0,
                    max_tokens=8,
                    on_delta=on_delta,
                )

        self.assertEqual(seam.calls, 1)
        for usage in (seam.turn_backend_token_usage, seam.session_token_usage):
            self.assertEqual(usage.model_calls, 1)
            self.assertEqual(usage.failed_calls, 0)
            self.assertEqual(usage.prompt_tokens, 0, "nothing measured, nothing recorded")
            self.assertEqual(usage.cached_tokens, 0)


class ObserverExceptionIsNotRetriedTest(unittest.TestCase):
    """A bug inside the observer must surface, not masquerade as bad arity."""

    def test_internal_type_error_propagates_after_one_call(self) -> None:
        backend = LlamaServerBackend(base_url="http://127.0.0.1:1", timeout=5)
        calls: list[object] = []

        def broken_observer(prompt_metrics: StreamPromptMetrics | None) -> None:
            calls.append(prompt_metrics)
            raise TypeError("observer bug")

        backend.set_aborted_observer(broken_observer)

        def on_delta(_text: str) -> None:
            raise _RouteAbort("stop")

        patches = mock.patch.multiple(
            LlamaServerBackend,
            _is_orbit_native_backend=lambda self: True,
            request_model_name=lambda self: "m",
            _props_or_empty=lambda self: {},
            _serialize_for_profile=lambda self, messages: list(messages),
        )
        with patches, mock.patch.object(
            llama_server, "urlopen", lambda *a, **k: aborted_route_stream()
        ):
            with self.assertRaises(TypeError) as raised:
                backend.chat_stream(
                    [{"role": "user", "content": "hi"}],
                    temperature=0,
                    max_tokens=8,
                    on_delta=on_delta,
                )

        self.assertEqual(str(raised.exception), "observer bug")
        self.assertEqual(len(calls), 1, "a failing observer must not be retried")


if __name__ == "__main__":
    unittest.main()
