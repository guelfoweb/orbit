from __future__ import annotations

import io
import sys
import time
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.streaming import StreamRenderer, WorkProgress, _pad_to_terminal_width, _visible_len, format_elapsed
from orbit.backend.base import StreamProgress


class StreamingRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tty_patches = [
            mock.patch("orbit.terminal.streaming.is_tty", return_value=True),
            mock.patch("orbit.terminal.streaming.supports_ansi", return_value=True),
            mock.patch("orbit.terminal.theme.supports_ansi", return_value=True),
        ]
        for patcher in self._tty_patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self._tty_patches):
            patcher.stop()

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

    def test_plain_markdown_rendering_keeps_literal_text(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="plain")
            renderer.write("Hello **world**\n\n")
            renderer.finish()
        finally:
            sys.stdout = original

        self.assertEqual(stream.getvalue(), "Hello **world**\n\n")

    def test_live_markdown_rendering_does_not_apply_to_thinking_fragments(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True, render_markdown_mode="live")
            renderer.write("### Reasoning\nstep 1\n")
            renderer.set_final_output_mode(True)
            renderer.write("**Final** answer.\n")
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Thinking...", output)
        self.assertIn("### Reasoning\nstep 1\n", output)
        self.assertIn("\033[1mFinal\033[22m answer.\n", output)

    def test_live_markdown_rendering_does_not_apply_to_tool_events(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.event("Read: **cat** README.md", restart_timer=False)
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Read: **cat** README.md", output)
        self.assertNotIn("\033[1mcat", output)

    def test_live_markdown_rendering_styles_heading_immediately(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("# Title")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Title", output)
        self.assertNotIn("# Title", output)
        self.assertIn("\033[1m\033[36m", output)

    def test_live_markdown_rendering_handles_split_heading_marker(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("#")
            renderer.write(" Title")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Title", output)
        self.assertNotIn("# Title", output)
        self.assertIn("\033[1m\033[36m", output)

    def test_live_markdown_rendering_emits_partial_paragraph_before_blank_line(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("This paragraph is still growing")
        finally:
            sys.stdout = original

        self.assertEqual(stream.getvalue(), "This paragraph is still growing")

    def test_live_markdown_rendering_styles_list_immediately(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("- item")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("- item", output)
        self.assertIn("\033[36m", output)

    def test_live_markdown_rendering_emits_list_item_before_list_end(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("- first item")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("- first item", output)
        self.assertNotEqual(output, "")

    def test_live_markdown_rendering_handles_split_list_marker(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("1")
            renderer.write(". item")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("1. item", output)
        self.assertIn("\033[36m", output)

    def test_live_markdown_rendering_styles_complete_inline_bold(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("The **first** item")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("The ", output)
        self.assertIn("\033[1mfirst\033[22m", output)
        self.assertIn(" item", output)
        self.assertNotIn("**first**", output)

    def test_live_markdown_rendering_styles_line_start_inline_bold(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("**First** item")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("\033[1mFirst\033[22m", output)
        self.assertIn(" item", output)
        self.assertNotIn("**First**", output)

    def test_live_markdown_rendering_handles_split_inline_bold(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("The **bo")
            renderer.write("ld** word")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("The ", output)
        self.assertIn("\033[1mbold\033[22m", output)
        self.assertIn(" word", output)
        self.assertNotIn("**bold**", output)

    def test_live_markdown_rendering_preserves_incomplete_inline_bold(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("The **first item")
            renderer.finish()
        finally:
            sys.stdout = original

        self.assertEqual(stream.getvalue(), "The **first item")

    def test_live_markdown_rendering_styles_complete_inline_italic(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("The *first* and _second_ item")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("\033[3mfirst\033[23m", output)
        self.assertIn("\033[3msecond\033[23m", output)
        self.assertNotIn("*first*", output)
        self.assertNotIn("_second_", output)

    def test_live_markdown_rendering_handles_split_inline_italic(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("The *ita")
            renderer.write("lic* word")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("\033[3mitalic\033[23m", output)
        self.assertIn(" word", output)

    def test_live_markdown_rendering_does_not_style_snake_case_as_italic(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("use snake_case_name here")
        finally:
            sys.stdout = original

        self.assertEqual(stream.getvalue(), "use snake_case_name here")

    def test_live_markdown_rendering_does_not_hold_plain_asterisk_math(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("2 * 3 = 6")
        finally:
            sys.stdout = original

        self.assertEqual(stream.getvalue(), "2 * 3 = 6")

    def test_live_markdown_rendering_restores_heading_style_after_inline_bold(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("# A **bold** title")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("\033[1mbold\033[0m\033[1m\033[36m", output)
        self.assertIn(" title", output)

    def test_live_markdown_rendering_shows_code_fence_before_close(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("```python\nprint('x')")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("print('x')", output)
        self.assertNotIn("```python", output)
        self.assertNotIn("python\n", output)
        self.assertIn("\033[2m", output)

    def test_live_markdown_rendering_handles_split_code_fence(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("`")
            renderer.write("``python\nprint('x')")
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("print('x')", output)
        self.assertNotIn("```python", output)
        self.assertNotIn("python\n", output)
        self.assertIn("\033[2m", output)

    def test_live_markdown_tables_fall_back_to_plain_text(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            renderer.write("| a | b |\n| - | - |\n")
        finally:
            sys.stdout = original

        self.assertEqual(stream.getvalue(), "| a | b |\n| - | - |\n")

    def test_live_markdown_falls_back_to_plain_text_on_renderer_error(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(render_markdown_mode="live")
            with mock.patch.object(renderer._markdown_live, "write", side_effect=RuntimeError("boom")):
                renderer.write("hello **world**")
        finally:
            sys.stdout = original

        self.assertEqual(stream.getvalue(), "hello **world**")

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

    def test_thinking_mode_does_not_extract_final_answer_from_reasoning_leakage(self) -> None:
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
            'The main difference is that an essay is a written composition, while "wise" means having good judgment.',
            output,
        )
        self.assertNotIn(
            '\n\nThe main difference is that an essay is a written composition, while "wise" means having good judgment.',
            output,
        )

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
        self.assertNotIn("\n\nThe main difference", output)

    def test_final_output_mode_switches_following_text_out_of_thinking_color(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(thinking=True)
            renderer.write("brief reasoning")
            renderer.set_final_output_mode(True)
            renderer.write("final answer")
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("Thinking...", output)
        self.assertIn("\033[2mbrief reasoning\033[0m", output)
        self.assertIn("\n\nfinal answer", output)
        self.assertNotIn("\033[2mfinal answer", output)

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

    def test_wait_timer_prints_compact_activity(self) -> None:
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
        self.assertIn("model · working · 0s", output)

    def test_wait_timer_does_not_print_speculative_prefill_estimate(self) -> None:
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
        self.assertIn("model · working · 0s", output)
        self.assertNotIn("estimate", output)

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
        self.assertIn("model · final answer · 0s", output)

    def test_wait_timer_prints_prefill_token_progress_when_estimated(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)

        self.assertIn("prefill estimate ~500/1000 tk", renderer._working_status(5))
        self.assertIn("waiting for model...", renderer._working_status(10))

    def test_wait_timer_includes_phase_label_in_real_progress_status(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)
        renderer.set_phase_label("forced final")
        renderer.progress(
            StreamProgress(
                phase="prefill",
                current=1011,
                total=1703,
                percent=59,
                evaluated_current=243,
                evaluated_total=935,
                cached_tokens=768,
                tokens_per_second=31.4,
            )
        )

        self.assertEqual(renderer._working_phase_prefix(), " [forced final prefill]")
        self.assertEqual(renderer._working_status(5), "5s, prefill · 26% · 243/935 eval · 768 cached · 31.4 tok/s")

    def test_wait_timer_omits_repeated_generation_label_from_status(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=51, total=256, percent=19))

        self.assertEqual(renderer._working_phase_prefix(), " [generation]")
        self.assertEqual(renderer._working_status(30), "30s, generating · 51 tok")

    def test_wait_timer_includes_generation_detail_in_phase_label(self) -> None:
        renderer = StreamRenderer()
        renderer.set_phase_label("final answer")
        renderer.progress(StreamProgress(phase="generation", current=7, total=512, percent=1))

        self.assertEqual(renderer._working_phase_prefix(), " [final answer generation]")

    def test_wait_timer_names_generation_progress_explicitly(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(
            StreamProgress(
                phase="generation",
                current=7,
                total=512,
                percent=1,
                elapsed_seconds=1.25,
                tokens_per_second=5.6,
            )
        )

        status = renderer._working_status(1)
        self.assertIn("generating · 7 tok · 5.6 tok/s", status)
        self.assertNotIn("%", status)
        self.assertNotIn("/512", status)

    def test_wait_timer_prints_prefill_finalizing_after_estimate(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10)

        self.assertIn("waiting for model...", renderer._working_status(10))

    def test_wait_timer_prefers_real_prefill_progress_when_available(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)
        renderer.progress(
            StreamProgress(
                phase="prefill",
                current=1011,
                total=1703,
                percent=59,
                evaluated_current=243,
                evaluated_total=935,
                cached_tokens=768,
            )
        )

        self.assertIn("prefill · 26% · 243/935 eval · 768 cached", renderer._working_status(5))

    def test_prefill_progress_uses_authoritative_evaluated_work_from_zero_to_complete(self) -> None:
        renderer = StreamRenderer()
        for current, expected in ((0, "0%"), (512, "63%"), (813, "100%")):
            renderer.progress(
                StreamProgress(
                    phase="prefill",
                    current=768 + current,
                    total=1581,
                    percent=0,
                    evaluated_current=current,
                    evaluated_total=813,
                    cached_tokens=768,
                )
            )
            status = renderer._working_status(0)
            self.assertIn(expected, status)
            self.assertIn(f"{current}/813 eval", status)
            self.assertIn("768 cached", status)

    def test_prefill_progress_does_not_render_contradictory_work_as_percentage(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(
            StreamProgress(
                phase="prefill",
                current=20,
                total=10,
                percent=200,
                evaluated_current=20,
                evaluated_total=10,
                cached_tokens=0,
            )
        )

        self.assertEqual(renderer._working_status(0), "0s, prefill")

    def test_wait_timer_shows_real_generation_progress_when_available(self) -> None:
        renderer = StreamRenderer(prefill_estimate_seconds=10, prefill_estimate_tokens=1000)
        renderer.progress(StreamProgress(phase="generation", current=7, total=32, percent=21))

        status = renderer._working_status(5)
        self.assertIn("generating · 7 tok", status)
        self.assertNotIn("21%", status)

    def test_wait_timer_reports_current_generation_pass_without_budget_percentage(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=32, total=32, percent=100))
        renderer.progress(StreamProgress(phase="generation", current=1, total=128, percent=0))

        self.assertEqual(renderer._working_status(5), "5s, generating · 1 tok")

    def test_known_tool_progress_formats_bytes_and_lines_without_estimation(self) -> None:
        renderer = StreamRenderer()
        renderer.set_activity("tool", "read")
        renderer.progress(WorkProgress(phase="reading", current=7_130_317, total=10_485_760, unit="bytes"))
        self.assertEqual(renderer._working_status(0), "0s, reading · 68% · 6.8/10.0 MB")

        renderer.progress(WorkProgress(phase="scanning", current=10_770, total=14_753, unit="lines"))
        self.assertEqual(renderer._working_status(0), "0s, scanning · 73% · 10770/14753 lines")

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
        self.assertIn("model · prefill · 0s", output)

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
        self.assertNotIn("model ·", after_delta)

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
        self.assertNotIn("model ·", after_final)

    def test_tool_activity_replaces_model_activity_until_result(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            renderer = StreamRenderer(interval=0.01)
            renderer.start()
            renderer.event("› Exec  pwd", next_activity=("tool", "exec_shell_full_command"))
            time.sleep(0.02)
            renderer.event("└ /tmp", next_activity=("model", "final answer"))
            time.sleep(0.02)
            renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertIn("tool · exec_shell_full_command · 0s", output)
        self.assertIn("model · final answer · 0s", output)

    def test_activity_line_fits_40_and_60_columns(self) -> None:
        for columns in (40, 60):
            stream = io.StringIO()
            original = sys.stdout
            try:
                sys.stdout = stream
                renderer = StreamRenderer(interactive=True)
                renderer.set_activity("tool", "exec_shell_full_command")
                renderer._start_time = time.monotonic()
                with mock.patch("orbit.terminal.streaming._terminal_columns", return_value=columns):
                    renderer._render_wait_line()
            finally:
                sys.stdout = original

            line = stream.getvalue().rsplit("\r", 1)[-1]
            self.assertLessEqual(_visible_len(line), columns)
            self.assertIn("exec_shell_full_command", line)

    def test_progress_line_fits_40_and_60_columns(self) -> None:
        for columns in (40, 60):
            stream = io.StringIO()
            original = sys.stdout
            try:
                sys.stdout = stream
                renderer = StreamRenderer(interactive=True)
                renderer.progress(
                    StreamProgress(
                        phase="prefill",
                        current=1280,
                        total=1581,
                        percent=80,
                        evaluated_current=512,
                        evaluated_total=813,
                        cached_tokens=768,
                        tokens_per_second=31.4,
                    )
                )
                renderer._start_time = time.monotonic()
                with mock.patch("orbit.terminal.streaming._terminal_columns", return_value=columns):
                    renderer._render_wait_line()
            finally:
                sys.stdout = original

            line = stream.getvalue().rsplit("\r", 1)[-1]
            self.assertLessEqual(_visible_len(line), columns)
            self.assertIn("prefill", line)

    def test_non_tty_is_plain_linear_and_keeps_markdown_literal(self) -> None:
        stream = io.StringIO()
        original = sys.stdout
        try:
            sys.stdout = stream
            with (
                mock.patch("orbit.terminal.streaming.is_tty", return_value=False),
                mock.patch("orbit.terminal.streaming.supports_ansi", return_value=False),
                mock.patch("orbit.terminal.theme.supports_ansi", return_value=False),
            ):
                renderer = StreamRenderer(interval=0.01, render_markdown_mode="live")
                renderer.start()
                renderer.progress(StreamProgress(phase="prefill", current=1, total=2, percent=50))
                renderer.event("› Exec  pwd", next_activity=("tool", "exec_shell_full_command"))
                renderer.write("**answer**\n")
                renderer.finish()
        finally:
            sys.stdout = original

        output = stream.getvalue()
        self.assertEqual(output, "› Exec  pwd\n**answer**\n")
        self.assertNotIn("\033", output)
        self.assertNotIn("\r", output)

    def test_non_tty_progress_is_byte_identical_across_repeated_runs(self) -> None:
        def render_once() -> str:
            stream = io.StringIO()
            original = sys.stdout
            try:
                sys.stdout = stream
                renderer = StreamRenderer(interactive=False)
                renderer.start()
                renderer.progress(
                    StreamProgress(
                        phase="prefill",
                        current=1280,
                        total=1581,
                        percent=80,
                        evaluated_current=512,
                        evaluated_total=813,
                        cached_tokens=768,
                        elapsed_seconds=16.3,
                        tokens_per_second=31.4,
                    )
                )
                renderer.progress(
                    StreamProgress(
                        phase="generation",
                        current=12,
                        total=128,
                        percent=9,
                        elapsed_seconds=1.0,
                        tokens_per_second=12.0,
                    )
                )
                renderer.write("answer\n")
                renderer.finish()
            finally:
                sys.stdout = original
            return stream.getvalue()

        first = render_once()
        second = render_once()
        self.assertEqual(first, second)
        self.assertEqual(first, "answer\n")
        self.assertNotIn("\033", first)
        self.assertNotIn("\r", first)

    def test_restart_timer_clears_previous_progress_state(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=6, total=32, percent=18))

        self.assertIn("generating · 6 tok", renderer._working_status(5))

        renderer._restart_timer()

        self.assertNotIn("generating · 6 tok", renderer._working_status(0))

    def test_new_model_phase_clears_previous_call_progress(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=6, total=32, percent=18))

        renderer.set_phase_label("final answer")

        self.assertNotIn("generating", renderer._working_status(0))
        self.assertEqual(renderer._working_phase_detail(), None)

    def test_restart_timer_clears_generation_accumulator(self) -> None:
        renderer = StreamRenderer()
        renderer.progress(StreamProgress(phase="generation", current=32, total=32, percent=100))
        renderer.progress(StreamProgress(phase="generation", current=1, total=128, percent=0))

        renderer._restart_timer()
        renderer.progress(StreamProgress(phase="generation", current=1, total=128, percent=0))

        self.assertIn("generating · 1 tok", renderer._working_status(0))


if __name__ == "__main__":
    unittest.main()
