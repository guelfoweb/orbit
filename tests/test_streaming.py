from __future__ import annotations

import io
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.streaming import StreamRenderer, format_elapsed


class StreamingRendererTests(unittest.TestCase):
    def test_write_prints_delta(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer()
            renderer.write("hello")
        finally:
            sys.stdout = original

        self.assertIn("hello", stream.getvalue())

    def test_event_prints_dim_message(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer()
            renderer.event('list_files {"path":"."}', restart_timer=False)
            renderer.event(" └ list_files 90 chars", trailing_blank_line=True)
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn('list_files {"path":"."}', output)
        self.assertIn(" └ list_files 90 chars", output)
        self.assertIn("chars\033[0m\n\n", output)

    def test_elapsed_format_switches_to_minutes(self) -> None:
        self.assertEqual(format_elapsed(0), "0s")
        self.assertEqual(format_elapsed(59.9), "59s")
        self.assertEqual(format_elapsed(79), "1m 19s")

    def test_wait_timer_prints_working_message(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(interval=0.01)
            renderer.start()
            time.sleep(0.02)
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Working (0s - Ctrl+C to interrupt)", output)

    def test_wait_timer_prints_prefill_progress_when_estimated(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(interval=0.01, prefill_estimate_seconds=10)
            renderer.start()
            time.sleep(0.02)
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Working (0s, pf ~1% - Ctrl+C to interrupt)", output)

    def test_wait_timer_prints_prefill_token_progress_when_estimated(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)

        self.assertIn("pf ~500/1000 tk", renderer._working_status(5))
        self.assertIn("processing prompt", renderer._working_status(10))

    def test_wait_timer_prints_prefill_finalizing_after_estimate(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10)

        self.assertIn("processing prompt", renderer._working_status(10))

    def test_wait_timer_stops_before_first_delta(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(interval=0.01)
            renderer.start()
            time.sleep(0.02)
            renderer.write("hello")
            time.sleep(0.03)
            renderer.finish()
        finally:
            sys.stdout = original

        after_delta = stream.getvalue().split("hello", 1)[1]
        self.assertNotIn("Working", after_delta)

    def test_restarted_timer_stops_before_later_delta(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(interval=0.01)
            renderer.start()
            renderer.write('{"_route":"WEB"}')
            renderer.event('search_web {"query":"x"}')
            time.sleep(0.02)
            renderer.write("final answer")
            time.sleep(0.03)
            renderer.finish()
        finally:
            sys.stdout = original

        after_final = stream.getvalue().split("final answer", 1)[1]
        self.assertNotIn("Working", after_final)


if __name__ == "__main__":
    unittest.main()
