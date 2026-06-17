from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.persistent_mtp import PersistentMtpSessionRuntime


class NativePersistentMtpTests(unittest.TestCase):
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

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_persistent_mtp_stays_disabled_when_draft_is_missing(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(mtp_available=False, fallback_reason="draft-mtp-missing"), NativeClientConfig())
        client._session.ctx_tgt = object()

        client._initialize_persistent_mtp_session()

        snapshot = client.session_snapshot()
        self.assertFalse(snapshot.mtp_enabled)
        self.assertFalse(snapshot.mtp_initialized)
        self.assertEqual(snapshot.mtp_failure_reason, "draft-mtp-missing")

    @mock.patch("orbit.native_llama.client.create_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_persistent_mtp_initializes_and_exposes_runtime_handles(self, _mocked_lib, mocked_create) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._session.ctx_tgt = object()
        mocked_create.return_value = PersistentMtpSessionRuntime(
            handle=object(),
            ctx_dft=object(),
            spec=object(),
            rss_before_kb=100,
            rss_after_init_kb=200,
            rss_peak_kb=300,
        )

        client._initialize_persistent_mtp_session()

        snapshot = client.session_snapshot()
        self.assertTrue(snapshot.mtp_enabled)
        self.assertTrue(snapshot.mtp_initialized)
        self.assertIsNone(snapshot.mtp_failure_reason)
        self.assertIsNotNone(client._session.ctx_dft)
        self.assertIsNotNone(client._session.spec)

    @mock.patch("orbit.native_llama.client.create_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_persistent_mtp_records_init_failure(self, _mocked_lib, mocked_create) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._session.ctx_tgt = object()
        mocked_create.side_effect = RuntimeError("init failed")

        client._initialize_persistent_mtp_session()

        snapshot = client.session_snapshot()
        self.assertFalse(snapshot.mtp_enabled)
        self.assertFalse(snapshot.mtp_initialized)
        self.assertEqual(snapshot.mtp_failure_reason, "init failed")

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_reset_session_state_clears_target_cache_and_reinitializes_persistent_mtp(self, mocked_lib_cls, mocked_reset) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        fake_lib = mock.Mock()
        fake_lib.llama_get_memory.return_value = object()
        mocked_lib_cls.return_value.lib = fake_lib
        client._session.ctx_tgt = object()
        client._session.cached_prompt_tokens = [1, 2, 3]
        client._session.last_metrics = NativeTimings(5, 1, 2, 3, 1.0, 2.0)
        client._persistent_mtp_runtime = PersistentMtpSessionRuntime(handle=object(), ctx_dft=object(), spec=object())
        mocked_reset.return_value = PersistentMtpSessionRuntime(handle=object(), ctx_dft=object(), spec=object())

        client.reset_session_state()

        self.assertEqual(client._session.cached_prompt_tokens, [])
        self.assertIsNone(client._session.last_metrics)
        self.assertTrue(client._session.mtp_enabled)
        fake_lib.llama_memory_clear.assert_called()
        mocked_reset.assert_called_once()

    @mock.patch("orbit.native_llama.client.free_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_close_frees_persistent_mtp_before_releasing_target_context(self, mocked_lib_cls, mocked_free) -> None:
        fake_lib = mock.Mock()
        mocked_lib_cls.return_value.lib = fake_lib
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._persistent_mtp_runtime = PersistentMtpSessionRuntime(handle=object(), ctx_dft=object(), spec=object())
        client._session.ctx_tgt = object()
        client._session.sampler = object()
        client._model = object()

        client.close()

        mocked_free.assert_called_once()
        fake_lib.llama_sampler_free.assert_called_once()
        fake_lib.llama_free.assert_called_once()
        fake_lib.llama_model_free.assert_called_once()


if __name__ == "__main__":
    unittest.main()
