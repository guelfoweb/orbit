from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import (
    _has_closed_thought_with_final,
    _looks_like_degenerate_thought_continuation,
    NativeClientConfig,
    NativeLlamaClient,
    _has_open_thought_channel,
    _merge_completions,
    _strip_reasoning_preamble,
)
from orbit.native_llama.events import NativeCompletion, NativeTimings
from orbit.native_llama.paths import NativeLlamaPaths


class NativeThinkingTests(unittest.TestCase):
    def _paths(self) -> NativeLlamaPaths:
        return NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/target.gguf"),
            mmproj_model=None,
            draft_mtp_model=None,
            multimodal_available=False,
            multimodal_fallback_reason="mmproj-missing",
            mtp_available=False,
            fallback_reason="draft-mtp-missing",
            model_id="gemma4-12b-it-q4km",
        )

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_text_passes_plain_reasoning_when_thinking_is_on(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(thinking=True))
        emitted: list[str] = []
        timings = NativeTimings(10, 4, 0, 10, 10.0, 20.0, False)

        def fake_complete_chat(_messages, **kwargs):
            kwargs["on_token"]("### Reasoning\nplain text\n\nFinal answer")
            return timings

        with mock.patch.object(client, "complete_chat", side_effect=fake_complete_chat):
            result = client.complete_chat_text(
                [{"role": "user", "content": "hello"}],
                max_tokens=32,
                thinking=True,
                on_token=emitted.append,
            )

        self.assertEqual(result.content, "### Reasoning\nplain text\n\nFinal answer")
        self.assertEqual("".join(emitted), result.content)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_text_can_continue_thought_multiple_times(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(thinking=True))
        emitted: list[str] = []
        first = NativeCompletion(
            content="<|channel>thought\npart 1",
            timings=NativeTimings(10, 32, 0, 10, 1.0, 2.0, False),
            stopped_by_stop=False,
        )
        second = NativeCompletion(
            content=" part 2",
            timings=NativeTimings(0, 48, 0, 0, 0.0, 2.0, False),
            stopped_by_stop=False,
        )
        third = NativeCompletion(
            content="<channel|>Final",
            timings=NativeTimings(0, 8, 0, 0, 0.0, 1.0, False),
            stopped_by_stop=False,
        )

        with (
            mock.patch.object(client, "_complete_chat_text_once", return_value=first),
            mock.patch.object(client, "_continue_chat_text_from_current_context", side_effect=[second, third]) as cont,
        ):
            client._last_completion_used_mtp = True
            client._last_completion_generation_cap = 32
            result = client.complete_chat_text(
                [{"role": "user", "content": "hello"}],
                max_tokens=48,
                thinking=True,
                on_token=emitted.append,
            )

        self.assertEqual(cont.call_count, 2)
        self.assertEqual(result.content, "<|channel>thought\npart 1 part 2<channel|>Final")
        self.assertTrue(result.completed_after_thought)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_text_can_continue_thought_multiple_times_with_tools_enabled(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(thinking=True))
        emitted: list[str] = []
        first = NativeCompletion(
            content="<|channel>thought\npart 1",
            timings=NativeTimings(10, 32, 0, 10, 1.0, 2.0, False),
            stopped_by_stop=False,
        )
        second = NativeCompletion(
            content=" part 2",
            timings=NativeTimings(0, 48, 0, 0, 0.0, 2.0, False),
            stopped_by_stop=False,
        )
        third = NativeCompletion(
            content="<channel|>Final",
            timings=NativeTimings(0, 8, 0, 0, 0.0, 1.0, False),
            stopped_by_stop=False,
        )

        with (
            mock.patch.object(client, "_complete_chat_text_once", return_value=first),
            mock.patch.object(client, "_continue_chat_text_from_current_context", side_effect=[second, third]) as cont,
        ):
            client._last_completion_used_mtp = True
            client._last_completion_generation_cap = 32
            result = client.complete_chat_text(
                [{"role": "user", "content": "hello"}],
                max_tokens=48,
                tools=[{"type": "function", "function": {"name": "exec_shell_full_command"}}],
                thinking=True,
                on_token=emitted.append,
            )

        self.assertEqual(cont.call_count, 2)
        self.assertEqual(result.content, "<|channel>thought\npart 1 part 2<channel|>Final")
        self.assertTrue(result.completed_after_thought)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_text_drops_degenerate_thought_continuation_with_tools_enabled(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(thinking=True))
        emitted: list[str] = []
        first = NativeCompletion(
            content="<|channel>thought\npart 1",
            timings=NativeTimings(10, 32, 0, 10, 1.0, 2.0, False),
            stopped_by_stop=False,
        )
        degenerate = NativeCompletion(
            content=".\n.\n.\n.\n",
            timings=NativeTimings(0, 128, 0, 0, 0.0, 2.0, False),
            stopped_by_stop=False,
        )

        def first_pass(_messages, **kwargs):
            kwargs["on_token"]("<|channel>thought\npart 1")
            return first

        with (
            mock.patch.object(client, "_complete_chat_text_once", side_effect=first_pass),
            mock.patch.object(client, "_continue_chat_text_from_current_context", return_value=degenerate) as cont,
        ):
            client._last_completion_used_mtp = True
            client._last_completion_generation_cap = 32
            result = client.complete_chat_text(
                [{"role": "user", "content": "hello"}],
                max_tokens=48,
                tools=[{"type": "function", "function": {"name": "exec_shell_full_command"}}],
                thinking=True,
                on_token=emitted.append,
            )

        self.assertEqual(cont.call_count, 1)
        self.assertEqual(result.content, "<|channel>thought\npart 1")
        self.assertEqual("".join(emitted), "<|channel>thought\npart 1")
        self.assertFalse(result.completed_after_thought)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_text_still_strips_control_channel_when_thinking_is_off(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(thinking=False))
        emitted: list[str] = []
        timings = NativeTimings(10, 4, 0, 10, 10.0, 20.0, False)

        def fake_complete_chat(_messages, **kwargs):
            kwargs["on_token"]("<|channel>thought\nhidden<channel|>Final answer")
            return timings

        with mock.patch.object(client, "complete_chat", side_effect=fake_complete_chat):
            result = client.complete_chat_text(
                [{"role": "user", "content": "hello"}],
                max_tokens=32,
                thinking=False,
                on_token=emitted.append,
            )

        self.assertEqual(result.content, "Final answer")
        self.assertEqual("".join(emitted), "Final answer")

    def test_strip_reasoning_preamble_keeps_final_answer_section(self) -> None:
        content = (
            "### Reasoning\n"
            "1. Analyze\n"
            "2. Decide\n\n"
            "**Final Answer:**\n"
            "Use mv \"old report 2026.txt\" \"final report 2026.txt\""
        )
        self.assertEqual(
            _strip_reasoning_preamble(content),
            'Use mv "old report 2026.txt" "final report 2026.txt"',
        )

    def test_strip_reasoning_preamble_keeps_plain_answer_when_no_boundary_exists(self) -> None:
        content = "This is a direct explanation without a separate final-answer section."
        self.assertEqual(_strip_reasoning_preamble(content), content)

    def test_strip_reasoning_preamble_handles_the_final_answer_is_variant(self) -> None:
        content = (
            "Plan:\n"
            "1. Inspect\n"
            "2. Solve\n\n"
            "The final answer is:\n"
            "42"
        )
        self.assertEqual(_strip_reasoning_preamble(content), "42")

    def test_detects_open_thought_channel(self) -> None:
        self.assertTrue(_has_open_thought_channel("<|channel>thought\npartial"))
        self.assertFalse(_has_open_thought_channel("<|channel>thought\nx<channel|>final"))

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_thought_continuation_uses_requested_budget_for_standard_path(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(thinking=True))
        result = NativeCompletion(
            content="<|channel>thought\npartial",
            timings=NativeTimings(10, 32, 0, 10, 1.0, 2.0, False),
            stopped_by_stop=False,
        )
        client._last_completion_used_mtp = False
        client._last_completion_generation_cap = 0
        self.assertTrue(client._should_continue_thought_after_completion(result, max_tokens=32, thinking=True))
        self.assertFalse(client._should_continue_thought_after_completion(result, max_tokens=64, thinking=True))
        self.assertFalse(client._should_continue_thought_after_completion(result, max_tokens=32, thinking=False))

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_thought_continuation_uses_internal_mtp_cap(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(thinking=True))
        client._last_completion_used_mtp = True
        client._last_completion_generation_cap = 32
        result = NativeCompletion(
            content="<|channel>thought\npartial",
            timings=NativeTimings(10, 32, 0, 10, 1.0, 2.0, False),
            stopped_by_stop=False,
        )
        self.assertTrue(client._should_continue_thought_after_completion(result, max_tokens=48, thinking=True))

    def test_merge_completions_sums_generation_and_output_tokens(self) -> None:
        first = NativeCompletion("<|channel>thought\nx", NativeTimings(10, 32, 3, 7, 11.0, 22.0, False), False)
        second = NativeCompletion("<channel|>b", NativeTimings(0, 5, 0, 0, 0.0, 3.0, False), False)
        merged = _merge_completions(first, second)
        self.assertEqual(merged.content, "<|channel>thought\nx<channel|>b")
        self.assertEqual(merged.timings.output_tokens, 37)
        self.assertEqual(merged.timings.generation_ms, 25.0)
        self.assertTrue(merged.completed_after_thought)

    def test_has_closed_thought_with_final_requires_tail_after_channel_end(self) -> None:
        self.assertTrue(_has_closed_thought_with_final("<|channel>thought\nx<channel|>4"))
        self.assertFalse(_has_closed_thought_with_final("<|channel>thought\nx<channel|>"))

    def test_detects_degenerate_thought_continuation(self) -> None:
        self.assertTrue(_looks_like_degenerate_thought_continuation(".\n.\n.\n.\n"))
        self.assertFalse(_looks_like_degenerate_thought_continuation("<channel|>done"))
        self.assertFalse(_looks_like_degenerate_thought_continuation(" part 2"))


if __name__ == "__main__":
    unittest.main()
