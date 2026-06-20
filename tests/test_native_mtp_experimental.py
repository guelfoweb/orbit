from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.mtp_completion import MtpCompletionResult
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_server.app import build_parser


class NativeMtpExperimentalTests(unittest.TestCase):
    def _paths(self, *, mtp_available: bool = True, fallback_reason: str | None = None) -> NativeLlamaPaths:
        return NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/target.gguf"),
            draft_mtp_model=Path("/models/draft.gguf") if mtp_available else None,
            mtp_available=mtp_available,
            fallback_reason=fallback_reason,
            model_id="gemma4-12b-it-q4km",
        )

    def test_parser_accepts_mtp_flag(self) -> None:
        args = build_parser().parse_args(["--mtp"])
        self.assertTrue(args.enable_mtp_experimental)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_prompt_skips_mtp_on_chat_followup_and_resets_session(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._session.prompt_cache_mode = "chat:thinking=off"
        client._session.cached_prompt_tokens = [1, 2, 3]
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True

        with (
            mock.patch.object(client, "reset_session_state") as reset_state,
            mock.patch.object(client, "_try_complete_with_mtp_experimental") as mtp_try,
            mock.patch.object(client, "_complete_prompt_standard", return_value=NativeTimings(3, 1, 0, 3, 1.0, 2.0, False)) as standard,
        ):
            result = client.complete_prompt("hello", max_tokens=8, allow_mtp_experimental=True, thinking=False)

        self.assertEqual(result.output_tokens, 1)
        mtp_try.assert_not_called()
        reset_state.assert_called_once()
        standard.assert_called_once()
        self.assertEqual(client.mtp_fallback_reason, "chat-followup-fallback")
        self.assertEqual(client.last_mtp_completion.error, "chat-followup-fallback")

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_falls_back_when_draft_missing(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(mtp_available=False, fallback_reason="draft-mtp-missing"), NativeClientConfig(use_mtp_experimental=True))

        result = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNone(result)
        self.assertEqual(client.mtp_fallback_reason, "draft-mtp-missing")

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_skips_when_thinking_is_on(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True, thinking=True))

        result = client._try_complete_with_mtp_experimental("hello", max_tokens=8, thinking=True)

        self.assertIsNone(result)
        self.assertEqual(client.mtp_fallback_reason, "thinking-mode")
        self.assertEqual(client.last_mtp_completion.error, "thinking-mode")

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_returns_timings_on_success(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [9]])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="ok",
            output_tokens=1,
            draft_tokens_total=3,
            accepted_tokens_total=2,
            rejected_tokens_total=1,
            acceptance_ratio=2 / 3,
            target_decode_calls=2,
            draft_decode_calls=1,
            elapsed_ms=12.5,
            tokens_per_second=80.0,
            full_accept_steps=1,
            replay_steps=0,
            partial_accept_steps=0,
            partial_no_replay_steps=0,
            replay_fallback_steps=0,
            seq_rm_supported=False,
            rollback_tokens_total=0,
        )
        emitted: list[str] = []

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8, on_token=emitted.append)

        self.assertIsNotNone(timings)
        assert timings is not None
        self.assertEqual(timings.output_tokens, 1)
        self.assertEqual(timings.reused_prompt_tokens, 0)
        self.assertEqual("".join(emitted), "ok")
        self.assertIsNone(client.mtp_fallback_reason)
        self.assertEqual(client._session.cached_prompt_tokens, [1, 2, 3, 9])
        self.assertEqual(client._session.raw_emitted_token_ids, [])
        self.assertEqual(client._session.committed_frontier_tokens, [])
        self.assertEqual(client.last_mtp_completion.full_accept_steps, 1)
        self.assertEqual(client.last_mtp_completion.replay_steps, 0)
        self.assertEqual(client.last_mtp_completion.partial_accept_steps, 0)
        self.assertEqual(client.last_mtp_completion.partial_no_replay_steps, 0)
        self.assertEqual(client.last_mtp_completion.replay_fallback_steps, 0)
        self.assertFalse(client.last_mtp_completion.seq_rm_supported)
        self.assertEqual(client.last_mtp_completion.rollback_tokens_total, 0)

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_strips_control_channel_tokens(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [8, 9]])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="<|channel>thought\n<channel|>ok.",
            output_tokens=2,
            draft_tokens_total=3,
            accepted_tokens_total=2,
            rejected_tokens_total=1,
            acceptance_ratio=2 / 3,
            target_decode_calls=2,
            draft_decode_calls=1,
            elapsed_ms=12.5,
            tokens_per_second=80.0,
        )
        emitted: list[str] = []

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8, on_token=emitted.append)

        self.assertIsNotNone(timings)
        self.assertEqual("".join(emitted), "ok.")
        self.assertEqual(client.last_mtp_completion.content, "ok.")
        self.assertEqual(client.last_mtp_completion.acceptance_ratio, 2 / 3)
        self.assertEqual(client._session.cached_prompt_tokens, [1, 2, 3, 8, 9])

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_strips_plain_reasoning_preview_tokens(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [8, 9, 10]])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="thought preview\nDante Alighieri was an Italian poet.",
            output_tokens=6,
            elapsed_ms=12.5,
        )
        emitted: list[str] = []

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8, on_token=emitted.append)

        self.assertIsNotNone(timings)
        self.assertEqual("".join(emitted), "Dante Alighieri was an Italian poet.")
        self.assertEqual(client.last_mtp_completion.content, "Dante Alighieri was an Italian poet.")

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_falls_back_when_visible_content_is_empty(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(return_value=[1, 2, 3])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="<|channel>thought\nhidden<channel|>",
            output_tokens=2,
            elapsed_ms=12.5,
        )
        emitted: list[str] = []

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8, on_token=emitted.append)

        self.assertIsNone(timings)
        self.assertEqual(emitted, [])
        self.assertEqual(client.mtp_fallback_reason, "empty-visible-content")
        self.assertEqual(client.last_mtp_completion.error, "empty-visible-content")

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_avoids_double_emit_when_streaming_callback_is_used(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [9]])

        def fake_run(**kwargs):
            kwargs["on_token"]("o")
            kwargs["on_token"]("k")
            return MtpCompletionResult(
                enabled=True,
                success=True,
                error=None,
                content="ok",
                output_tokens=1,
                elapsed_ms=12.5,
            )

        mocked_run.side_effect = fake_run
        emitted: list[str] = []

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8, on_token=emitted.append)

        self.assertIsNotNone(timings)
        self.assertEqual("".join(emitted), "ok")

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_maps_prefill_and_generation_progress(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [9]])
        progress = []

        def fake_run(**kwargs):
            kwargs["on_progress"](0, 12, 48)
            kwargs["on_progress"](1, 2, 32)
            return MtpCompletionResult(
                enabled=True,
                success=True,
                error=None,
                content="ok",
                output_tokens=2,
                elapsed_ms=12.5,
            )

        mocked_run.side_effect = fake_run

        client._try_complete_with_mtp_experimental(
            "hello",
            max_tokens=8,
            on_progress=lambda item: progress.append((item.phase, item.current, item.total)),
        )

        self.assertEqual(progress, [("prefill", 12, 48), ("generation", 2, 32)])

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_wraps_raw_prompt_in_gemma_chat_template(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [9]])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="ok.",
            output_tokens=1,
        )

        client._try_complete_with_mtp_experimental("Say only: ok.", max_tokens=8)

        called_prompt = mocked_run.call_args.kwargs["prompt"]
        self.assertIn("<bos>", called_prompt)
        self.assertIn("<|turn>user", called_prompt)
        self.assertIn("Say only: ok.", called_prompt)

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_reuses_cached_prompt_tokens_from_previous_turn(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [9]])
        client._session.cached_prompt_tokens = [1, 2, 3]
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="ok",
            output_tokens=1,
            draft_tokens_total=3,
            accepted_tokens_total=2,
            rejected_tokens_total=1,
            acceptance_ratio=2 / 3,
            target_decode_calls=2,
            draft_decode_calls=1,
            elapsed_ms=12.5,
            tokens_per_second=80.0,
        )

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNotNone(timings)
        assert timings is not None
        self.assertEqual(timings.reused_prompt_tokens, 2)
        self.assertEqual(client._session.cached_prompt_tokens, [1, 2, 3, 9])

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_stores_raw_frontier_tokens_separately(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [9]])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="ok",
            output_tokens=1,
            elapsed_ms=12.5,
            raw_emitted_token_ids=[41, 42],
            end_turn_frontier_token_ids=[1, 2, 3, 41, 42],
        )

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNotNone(timings)
        self.assertEqual(client._session.cached_prompt_tokens, [1, 2, 3, 9])
        self.assertEqual(client._session.raw_emitted_token_ids, [41, 42])
        self.assertEqual(client._session.committed_frontier_tokens, [1, 2, 3, 41, 42])

    @mock.patch.dict("os.environ", {"ORBIT_MTP_CHAT_REUSE_RAW": "1"}, clear=False)
    @mock.patch("orbit.native_llama.client.set_persistent_mtp_followup_suffix_tokens")
    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_uses_raw_followup_suffix_when_debug_flag_is_on(
        self,
        _mocked_lib,
        mocked_run,
        mocked_set_suffix,
    ) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client._session.prompt_cache_mode = "chat:thinking=off"
        client._session.cached_prompt_tokens = [1, 2, 3]
        client._session.chat_visible_frontier_tokens = [1, 2, 3]
        client._session.committed_frontier_tokens = [11, 12, 13, 14]
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3, 4, 5], [9]])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="ok",
            output_tokens=1,
            elapsed_ms=12.5,
            raw_emitted_token_ids=[41, 42],
            end_turn_frontier_token_ids=[11, 12, 13, 14, 4, 5, 41, 42],
        )

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNotNone(timings)
        mocked_set_suffix.assert_called_once()
        self.assertEqual(mocked_set_suffix.call_args.kwargs["suffix_tokens"], [4, 5])

    @mock.patch.dict("os.environ", {"ORBIT_MTP_CHAT_REUSE_RAW": "1"}, clear=False)
    @mock.patch("orbit.native_llama.client.set_persistent_mtp_followup_suffix_tokens")
    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_falls_back_when_visible_prefix_mismatches_raw_reuse(
        self,
        _mocked_lib,
        mocked_run,
        mocked_set_suffix,
    ) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client._session.prompt_cache_mode = "chat:thinking=off"
        client._session.cached_prompt_tokens = [1, 2, 3]
        client._session.chat_visible_frontier_tokens = [1, 2, 3]
        client._session.committed_frontier_tokens = [11, 12, 13]
        client.tokenize = mock.Mock(return_value=[1, 99, 3, 4])

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNone(timings)
        mocked_set_suffix.assert_not_called()
        mocked_run.assert_not_called()
        self.assertEqual(client.mtp_fallback_reason, "chat-followup-visible-prefix-mismatch")

    @mock.patch.dict("os.environ", {"ORBIT_MTP_CHAT_REUSE_RAW": "1"}, clear=False)
    @mock.patch("orbit.native_llama.client.set_persistent_mtp_followup_suffix_tokens")
    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_falls_back_when_raw_frontier_is_missing(
        self,
        _mocked_lib,
        mocked_run,
        mocked_set_suffix,
    ) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client._session.prompt_cache_mode = "chat:thinking=off"
        client._session.cached_prompt_tokens = [1, 2, 3]
        client._session.chat_visible_frontier_tokens = [1, 2, 3]
        client._session.committed_frontier_tokens = []
        client.tokenize = mock.Mock(return_value=[1, 2, 3, 4])

        timings = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNone(timings)
        mocked_set_suffix.assert_not_called()
        mocked_run.assert_not_called()
        self.assertEqual(client.mtp_fallback_reason, "chat-followup-raw-frontier-missing")

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_falls_back_on_helper_error(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        mocked_run.return_value = MtpCompletionResult(enabled=True, success=False, error="mtp helper failed")

        result = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNone(result)
        self.assertEqual(client.mtp_fallback_reason, "mtp helper failed")

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_falls_back_when_persistent_session_is_uninitialized(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._session.mtp_failure_reason = "persistent-mtp-uninitialized"

        result = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNone(result)
        self.assertEqual(client.mtp_fallback_reason, "persistent-mtp-uninitialized")

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_prompt_clears_stale_cancel_before_mtp_attempt(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [9]])
        client.cancel()
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="ok",
            output_tokens=1,
            elapsed_ms=12.5,
        )

        timings = client.complete_prompt("hello", max_tokens=8)

        self.assertEqual(timings.output_tokens, 1)
        self.assertFalse(client.cancel_event.is_set())
        self.assertTrue(client.last_mtp_completion.success)
        self.assertIsNone(client.mtp_fallback_reason)

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_caches_generated_frontier_for_next_turn(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = mock.Mock(side_effect=[[1, 2, 3], [41, 42]])
        mocked_run.return_value = MtpCompletionResult(
            enabled=True,
            success=True,
            error=None,
            content="dante",
            output_tokens=2,
            elapsed_ms=12.5,
        )

        client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertEqual(client._session.cached_prompt_tokens, [1, 2, 3, 41, 42])

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_keeps_mtp_enabled_for_final_from_tool_history(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client.apply_chat_template = mock.Mock(return_value="prompt")
        expected = NativeTimings(
            prompt_tokens=10,
            output_tokens=2,
            reused_prompt_tokens=3,
            evaluated_prompt_tokens=7,
            prefill_ms=12.0,
            generation_ms=34.0,
        )
        client.complete_prompt = mock.Mock(return_value=expected)

        result = client.complete_chat(
            [{"role": "user", "content": "read note.txt"}, {"role": "tool", "content": "hello"}],
            max_tokens=16,
        )

        self.assertEqual(result, expected)
        kwargs = client.complete_prompt.call_args.kwargs
        self.assertTrue(kwargs["allow_mtp_experimental"])

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_disables_mtp_during_tool_call_rounds(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client.apply_chat_template = mock.Mock(return_value="prompt")
        expected = NativeTimings(
            prompt_tokens=10,
            output_tokens=2,
            reused_prompt_tokens=3,
            evaluated_prompt_tokens=7,
            prefill_ms=12.0,
            generation_ms=34.0,
        )
        client.complete_prompt = mock.Mock(return_value=expected)

        result = client.complete_chat(
            [{"role": "user", "content": "read note.txt"}],
            max_tokens=16,
            tools=[{"type": "function", "function": {"name": "exec_shell_full_command"}}],
        )

        self.assertEqual(result, expected)
        kwargs = client.complete_prompt.call_args.kwargs
        self.assertFalse(kwargs["allow_mtp_experimental"])

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_prompt_can_use_mtp_even_when_should_cancel_callback_is_present(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        expected = NativeTimings(
            prompt_tokens=5,
            output_tokens=1,
            reused_prompt_tokens=0,
            evaluated_prompt_tokens=5,
            prefill_ms=0.0,
            generation_ms=10.0,
        )
        client._try_complete_with_mtp_experimental = mock.Mock(return_value=expected)
        client._complete_prompt_standard = mock.Mock(side_effect=AssertionError("standard path should not be used"))

        result = client.complete_prompt(
            "hello",
            allow_mtp_experimental=True,
            should_cancel=lambda: False,
        )

        self.assertEqual(result, expected)
        client._try_complete_with_mtp_experimental.assert_called_once()

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_prompt_skips_mtp_when_should_cancel_is_already_true(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client.cancel = mock.Mock()
        expected = NativeTimings(
            prompt_tokens=5,
            output_tokens=0,
            reused_prompt_tokens=0,
            evaluated_prompt_tokens=5,
            prefill_ms=1.0,
            generation_ms=2.0,
        )
        client._try_complete_with_mtp_experimental = mock.Mock(side_effect=AssertionError("mtp path should be skipped"))
        client._complete_prompt_standard = mock.Mock(return_value=expected)

        result = client.complete_prompt(
            "hello",
            allow_mtp_experimental=True,
            should_cancel=lambda: True,
        )

        self.assertEqual(result, expected)
        client.cancel.assert_called_once()
        client._complete_prompt_standard.assert_called_once()

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_prompt_skips_mtp_when_thinking_is_on(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True, thinking=True))
        expected = NativeTimings(
            prompt_tokens=5,
            output_tokens=1,
            reused_prompt_tokens=0,
            evaluated_prompt_tokens=5,
            prefill_ms=1.0,
            generation_ms=2.0,
        )
        client._try_complete_with_mtp_experimental = mock.Mock(side_effect=AssertionError("mtp path should be skipped"))
        client._complete_prompt_standard = mock.Mock(return_value=expected)

        result = client.complete_prompt(
            "hello",
            allow_mtp_experimental=True,
            thinking=True,
        )

        self.assertEqual(result, expected)
        client._complete_prompt_standard.assert_called_once()
        self.assertEqual(client.mtp_fallback_reason, "thinking-mode")
        self.assertEqual(client.last_mtp_completion.error, "thinking-mode")


if __name__ == "__main__":
    unittest.main()
