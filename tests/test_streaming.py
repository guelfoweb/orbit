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

from orbit.terminal.streaming import StreamRenderer, _pad_to_terminal_width, format_elapsed
from orbit.backend.base import StreamProgress


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

    def test_thinking_mode_dims_reasoning_and_keeps_final_answer_normal(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write("### Reasoning\nstep 1\n\n**Final Answer:** done")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Thinking...", output)
        self.assertIn("\033[2m### Reasoning\nstep 1\n\n\033[0m", output)
        self.assertIn("\n\n**Final Answer:** done", output)

    def test_thinking_mode_handles_split_final_answer_marker(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write("### Reasoning\nstep 1\n\n**Final")
            renderer.write(" Answer:** done")
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("**Final Answer:** done", output)
        self.assertNotIn("\033[2m**Final Answer:**", output)

    def test_thinking_mode_uses_real_thought_channel_boundary_when_present(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write("<|channel>thought\nprivate chain<channel|>final answer")
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Thinking...", output)
        self.assertIn("\033[2mprivate chain\033[0m", output)
        self.assertIn("\n\nfinal answer", output)

    def test_thinking_mode_separates_tool_phase_thought_from_following_final_answer(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write("<|channel>thought\nfrom tool result")
            renderer.write("<channel|>final answer")
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Thinking...", output)
        self.assertIn("\033[2mfrom tool result\033[0m", output)
        self.assertIn("\n\nfinal answer", output)
        self.assertNotIn("<|channel>thought", output)
        self.assertNotIn("<channel|>", output)

    def test_thinking_mode_hides_split_thought_marker_chunks(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write("<|chan")
            renderer.write("nel>thought\nprivate chain")
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertNotIn("<|channel>thought", output)
        self.assertIn("Thinking...", output)
        self.assertIn("\033[2mprivate chain\033[0m", output)

    def test_thinking_mode_emits_fallback_final_answer_when_reasoning_leaks_without_marker(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write(
                '"What is the main difference between essay and wise?"\n'
                "The user likely meant essay and wise.\n"
                "* Possibility A: typo.\n"
                "* Possibility B: meaning.\n"
                '* The main difference is that an essay is a written composition, while "wise" means having good judgment.'
            )
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Thinking...", output)
        self.assertIn("The user likely meant essay and wise.", output)
        self.assertIn(
            '\n\nThe main difference is that an essay is a written composition, while "wise" means having good judgment.',
            output,
        )
        self.assertNotIn('\n\nPossibility A: typo.', output)

    def test_thinking_mode_does_not_invent_final_answer_when_only_reasoning_is_present(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write(
                "The user likely meant essay and wise.\n"
                "* Possibility A: typo.\n"
                "* Possibility B: meaning.\n"
            )
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Thinking...", output)
        self.assertNotIn("\n\nPossibility", output)
        self.assertNotIn("\n\nThe main difference", output)

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

    def test_wait_timer_includes_phase_label_when_present(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(interval=0.01, prefill_estimate_seconds=10)
            renderer.set_phase_label("final answer")
            renderer.start()
            time.sleep(0.02)
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Working [final answer] (0s, pf ~1% - Ctrl+C to interrupt)", output)

    def test_wait_timer_prints_prefill_token_progress_when_estimated(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)

        self.assertIn("pf ~500/1000 tk", renderer._working_status(5))
        self.assertIn("waiting for model...", renderer._working_status(10))

    def test_wait_timer_includes_phase_label_in_real_progress_status(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)
        renderer.set_phase_label("forced final")
        renderer.progress(StreamProgress(phase="prefill", current=243, total=935, percent=25))

        self.assertEqual(renderer._working_phase_prefix(), " [forced final]")
        self.assertIn("pf 243/935 tk (25%)", renderer._working_status(5))

    def test_wait_timer_prints_prefill_finalizing_after_estimate(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10)

        self.assertIn("waiting for model...", renderer._working_status(10))

    def test_wait_timer_prefers_real_prefill_progress_when_available(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)
        renderer.progress(StreamProgress(phase="prefill", current=243, total=935, percent=25))

        self.assertIn("pf 243/935 tk (25%)", renderer._working_status(5))

    def test_wait_timer_shows_real_generation_progress_when_available(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)
        renderer.progress(StreamProgress(phase="generation", current=7, total=32, percent=21))

        self.assertIn("gen 7/32 tk (21%)", renderer._working_status(5))

    def test_wait_timer_accumulates_generation_across_continuation_passes(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=32, total=32, percent=100))
        renderer.progress(StreamProgress(phase="generation", current=1, total=128, percent=0))

        self.assertIn("gen 33/160 tk (20%)", renderer._working_status(5))

    def test_first_progress_renders_immediately_before_timer_tick(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(interval=10.0)
            renderer.start()
            renderer.progress(StreamProgress(phase="prefill", current=12, total=48, percent=25))
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("pf 12/48 tk (25%)", output)

    def test_wait_line_is_padded_to_clear_previous_content(self) -> None:
        padded = _pad_to_terminal_width("\033[2mshort\033[0m")

        self.assertGreater(len(padded), len("\033[2mshort\033[0m"))

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
            renderer.write('{"command":"curl https://example.com"}')
            renderer.event('search_web {"query":"x"}')
            time.sleep(0.02)
            renderer.write("final answer")
            time.sleep(0.03)
            renderer.finish()
        finally:
            sys.stdout = original

        after_final = stream.getvalue().split("final answer", 1)[1]
        self.assertNotIn("Working", after_final)

    def test_restart_timer_clears_previous_progress_state(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=6, total=32, percent=18))

        self.assertIn("gen 6/32 tk (18%)", renderer._working_status(5))

        renderer._restart_timer()

        self.assertNotIn("gen 6/32 tk (18%)", renderer._working_status(0))

    def test_restart_timer_clears_generation_accumulator(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=32, total=32, percent=100))
        renderer.progress(StreamProgress(phase="generation", current=1, total=128, percent=0))

        renderer._restart_timer()
        renderer.progress(StreamProgress(phase="generation", current=1, total=128, percent=0))

        self.assertIn("gen 1/128 tk (0%)", renderer._working_status(0))


if __name__ == "__main__":
    unittest.main()
