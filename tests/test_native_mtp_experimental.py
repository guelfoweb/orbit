from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
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

    def test_parser_accepts_enable_mtp_experimental_flag(self) -> None:
        args = build_parser().parse_args(["--enable-mtp-experimental"])
        self.assertTrue(args.enable_mtp_experimental)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_falls_back_when_draft_missing(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(mtp_available=False, fallback_reason="draft-mtp-missing"), NativeClientConfig(use_mtp_experimental=True))

        result = client._try_complete_with_mtp_experimental("hello", max_tokens=8)

        self.assertIsNone(result)
        self.assertEqual(client.mtp_fallback_reason, "draft-mtp-missing")

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_returns_timings_on_success(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = lambda prompt: [1, 2, 3]
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
        self.assertEqual(client._session.cached_prompt_tokens, [1, 2, 3])
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
        client.tokenize = lambda prompt: [1, 2, 3]
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

    @mock.patch("orbit.native_llama.client.run_persistent_mtp_completion")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_try_complete_with_mtp_experimental_wraps_raw_prompt_in_gemma_chat_template(self, _mocked_lib, mocked_run) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
        client._vocab = object()
        client._session.ctx_tgt = object()
        client._session.mtp_enabled = True
        client._persistent_mtp_runtime = object()
        client.tokenize = lambda prompt: [1, 2, 3]
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
        client.tokenize = lambda prompt: [1, 2, 3]
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


if __name__ == "__main__":
    unittest.main()
