"""Prompt metrics survive a stream the consumer stops on purpose.

A route stream is cut short as soon as the output is clearly prose, which
happens on every ordinary conversational turn. The backend has already
finished prefill by then and knows exactly how many prompt tokens were reused
and evaluated -- but that knowledge only reaches the client in the terminal
`metrics` event, which an aborted stream never sees. The turn then reports
`cache: 0` for a call that reused most of its prompt.

These tests pin the honest outcome: what prefill measured is reported, what
decode never produced stays unknown, and the deliberate stop is still not a
failure.
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
from orbit.backend.base import StreamConsumerAbort
from orbit.backend.llama_server import LlamaServerBackend
from orbit.terminal.status import (
    TokenUsageAccumulator,
    format_session_token_usage,
)

PROMPT_TOKENS = 1105
REUSED_TOKENS = 1085
EVALUATED_TOKENS = 20


class _Stream:
    """The native SSE wire format, assembled from explicit events."""

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


def _event(name: str, payload: dict) -> list[bytes]:
    return [
        f"event: {name}\n".encode(),
        b"data: " + json.dumps(payload).encode() + b"\n",
        b"\n",
    ]


def prefill_progress_event() -> list[bytes]:
    """What the server sends once prefill is complete, before any token."""
    return _event(
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


def aborted_route_stream() -> _Stream:
    """Prefill metrics, then prose: the consumer stops before `metrics`."""
    lines = prefill_progress_event()
    lines += _event("delta", {"text": "I'm doing well, thanks!"})
    # No `metrics`, no `done`: the consumer raised and stopped reading.
    return _Stream(lines)


def complete_stream() -> _Stream:
    """A normal call: prefill metrics, deltas, then the terminal metrics."""
    lines = prefill_progress_event()
    lines += _event("delta", {"text": '{"command"'})
    lines += _event("delta", {"text": ': "pwd"}'})
    lines += _event(
        "metrics",
        {
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": REUSED_TOKENS},
            },
            "timings": {},
        },
    )
    lines += _event("done", {"finish_reason": "stop", "model": "m"})
    return _Stream(lines)


class _RouteAbort(StreamConsumerAbort):
    """Stands in for the runtime's route abort: same marker, same propagation."""


def backend_with(stream: _Stream):
    backend = LlamaServerBackend(base_url="http://127.0.0.1:1", timeout=5)
    usage = TokenUsageAccumulator()
    backend.set_result_observer(usage.add_result)
    backend.set_failure_observer(usage.add_failed_call)
    aborted = getattr(backend, "set_aborted_observer", None)
    if callable(aborted):
        aborted(usage.add_aborted_call)
    patches = mock.patch.multiple(
        LlamaServerBackend,
        _is_orbit_native_backend=lambda self: True,
        request_model_name=lambda self: "m",
        _props_or_empty=lambda self: {},
        _serialize_for_profile=lambda self, messages: list(messages),
    )
    return backend, usage, patches


class AbortedStreamKeepsPrefillMetricsTest(unittest.TestCase):
    def test_known_prompt_and_cache_survive_the_abort(self) -> None:
        backend, usage, patches = backend_with(aborted_route_stream())

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

        snapshot = usage.snapshot()
        self.assertEqual(snapshot.model_calls, 1, "the attempt counts exactly once")
        self.assertEqual(snapshot.failed_calls, 0, "a deliberate stop is not a failure")
        # The point of the fix: prefill already measured these.
        self.assertEqual(
            snapshot.cached_tokens,
            REUSED_TOKENS,
            "reuse measured during prefill must not be discarded by the abort",
        )
        self.assertEqual(snapshot.prompt_tokens, PROMPT_TOKENS)
        self.assertEqual(snapshot.evaluated_tokens, EVALUATED_TOKENS)
        # Decode never reported, so the totals stay honestly incomplete.
        self.assertTrue(snapshot.usage_incomplete)

        line = format_session_token_usage(snapshot)
        self.assertNotIn("cache: 0 (0%)", line, "a reusing turn must not read as zero cache")
        self.assertNotIn("failed attempts", line)
        self.assertIn("partial", line)

    def test_abort_before_any_prompt_metrics_invents_nothing(self) -> None:
        empty = _Stream(_event("delta", {"text": "prose"}))

        def on_delta(_text: str) -> None:
            raise _RouteAbort("stop")

        backend, usage, patches = backend_with(empty)
        with patches, mock.patch.object(llama_server, "urlopen", lambda *a, **k: empty):
            with self.assertRaises(_RouteAbort):
                backend.chat_stream(
                    [{"role": "user", "content": "hi"}],
                    temperature=0,
                    max_tokens=8,
                    on_delta=on_delta,
                )

        snapshot = usage.snapshot()
        self.assertEqual(snapshot.model_calls, 1)
        self.assertEqual(snapshot.failed_calls, 0)
        self.assertTrue(snapshot.usage_incomplete)
        # Nothing was measured, so nothing may be written: an accumulator that
        # stamps zeros looks identical to one that measured zero reuse.
        self.assertEqual(usage.prompt_tokens, 0, "no prompt tokens may be recorded")
        self.assertEqual(usage.cached_tokens, 0)
        self.assertEqual(usage.evaluated_tokens, 0)

        # Prove it by contrast: a prior real call's totals must survive intact.
        with_history = TokenUsageAccumulator()
        with_history.prompt_tokens = 500
        with_history.cached_tokens = 100
        with_history.evaluated_tokens = 400
        with_history.add_aborted_call(None)
        self.assertEqual(with_history.prompt_tokens, 500, "an empty abort must add nothing")
        self.assertEqual(with_history.cached_tokens, 100)
        self.assertEqual(with_history.evaluated_tokens, 400)


class CompleteStreamCountsOnceTest(unittest.TestCase):
    def test_prefill_and_final_metrics_are_not_double_counted(self) -> None:
        backend, usage, patches = backend_with(complete_stream())
        with patches, mock.patch.object(
            llama_server, "urlopen", lambda *a, **k: complete_stream()
        ):
            backend.chat_stream(
                [{"role": "user", "content": "hi"}],
                temperature=0,
                max_tokens=8,
                on_delta=lambda _t: None,
            )

        snapshot = usage.snapshot()
        self.assertEqual(snapshot.model_calls, 1, "one attempt, one count")
        self.assertEqual(snapshot.failed_calls, 0)
        self.assertEqual(
            snapshot.prompt_tokens,
            PROMPT_TOKENS,
            "prefill progress must not be added on top of the final metrics",
        )
        self.assertEqual(snapshot.cached_tokens, REUSED_TOKENS)
        self.assertEqual(snapshot.completion_tokens, 8)
        self.assertFalse(snapshot.usage_incomplete, "a complete stream is not partial")


if __name__ == "__main__":
    unittest.main()
