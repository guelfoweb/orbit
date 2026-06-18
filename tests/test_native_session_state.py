from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.session_state import DEFAULT_NATIVE_SESSION_ID
from orbit.native_server.app import OrbitNativeServer


class NativeSessionStateTests(unittest.TestCase):
    def _paths(self) -> NativeLlamaPaths:
        return NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/target.gguf"),
            draft_mtp_model=None,
            mtp_available=False,
            fallback_reason="draft-mtp-missing",
            model_id="gemma4-12b-it-q4km",
        )

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_session_snapshot_defaults_to_no_mtp_idle_state(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())

        snapshot = client.session_snapshot()

        self.assertEqual(snapshot.session_id, DEFAULT_NATIVE_SESSION_ID)
        self.assertEqual(snapshot.cached_tokens, 0)
        self.assertFalse(snapshot.in_flight)
        self.assertFalse(snapshot.cancel_requested)
        self.assertEqual(snapshot.backend_mode, "no-mtp")
        self.assertIsNone(snapshot.last_metrics)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_session_snapshot_reflects_cached_tokens_cancel_and_last_metrics(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._session.cached_prompt_tokens = [1, 2, 3, 4]
        client._session.in_flight = True
        client.cancel()
        client._session.last_metrics = NativeTimings(
            prompt_tokens=10,
            output_tokens=2,
            reused_prompt_tokens=7,
            evaluated_prompt_tokens=3,
            prefill_ms=100.0,
            generation_ms=50.0,
        )

        snapshot = client.session_snapshot()

        self.assertEqual(snapshot.cached_tokens, 4)
        self.assertTrue(snapshot.in_flight)
        self.assertTrue(snapshot.cancel_requested)
        self.assertEqual(snapshot.last_metrics.reused_prompt_tokens, 7)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_server_session_info_uses_client_snapshot(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._session.cached_prompt_tokens = [1, 2]
        client._session.in_flight = True
        server = OrbitNativeServer(client=client, model_alias="m")

        session = server.session_info()

        self.assertEqual(session["id"], DEFAULT_NATIVE_SESSION_ID)
        self.assertEqual(session["backend_mode"], "no-mtp")
        self.assertEqual(session["cached_tokens"], 2)
        self.assertTrue(session["in_flight"])

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_prompt_updates_last_metrics_and_clears_in_flight(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        expected = NativeTimings(
            prompt_tokens=5,
            output_tokens=1,
            reused_prompt_tokens=2,
            evaluated_prompt_tokens=3,
            prefill_ms=10.0,
            generation_ms=20.0,
        )
        client._complete_prompt_standard = mock.Mock(return_value=expected)

        result = client.complete_prompt("hello", allow_mtp_experimental=False)

        self.assertEqual(result, expected)
        self.assertFalse(client.session_snapshot().in_flight)
        self.assertEqual(client.session_snapshot().last_metrics, expected)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_prompt_emits_initial_prefill_progress_from_reused_tokens(self, mocked_lib_class) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        mocked_lib = mocked_lib_class.return_value.lib
        mocked_lib.llama_time_us.side_effect = [0, 1000, 1000, 2000]
        mocked_lib.llama_batch_get_one.return_value = object()
        mocked_lib.llama_decode.return_value = 0

        client._session.ctx_tgt = object()
        client._vocab = object()
        client._session.sampler = object()
        client.tokenize = mock.Mock(return_value=[1, 2, 3, 4])
        client._prepare_memory_for_prompt = mock.Mock(return_value=3)
        progress: list[tuple[str, int, int, int]] = []

        timings = client._complete_prompt_standard(
            "hello",
            max_tokens=0,
            on_progress=lambda item: progress.append((item.phase, item.current, item.total, item.percent)),
        )

        self.assertEqual(progress[0], ("prefill", 3, 4, 75))
        self.assertEqual(progress[1], ("prefill", 4, 4, 100))
        self.assertEqual(timings.reused_prompt_tokens, 3)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_prompt_cache_mode_change_resets_session_state(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._session.ctx_tgt = object()
        client._session.prompt_cache_mode = "chat:thinking=off"
        client.reset_session_state = mock.Mock()

        client._ensure_prompt_cache_mode("chat:thinking=on")

        client.reset_session_state.assert_called_once()
        self.assertEqual(client._session.prompt_cache_mode, "chat:thinking=on")

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_prompt_cache_mode_same_value_keeps_session_state(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._session.prompt_cache_mode = "chat:thinking=off"
        client.reset_session_state = mock.Mock()

        client._ensure_prompt_cache_mode("chat:thinking=off")

        client.reset_session_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
