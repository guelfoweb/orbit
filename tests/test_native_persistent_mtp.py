from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.persistent_mtp import PersistentMtpSessionRuntime, build_persistent_mtp_shim, run_persistent_mtp_completion


class NativePersistentMtpTests(unittest.TestCase):
    def test_build_persistent_shim_prefers_packaged_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packaged = Path(tmp) / "liborbit-persistent-mtp.so"
            packaged.write_text("", encoding="utf-8")
            with mock.patch("orbit.native_llama.persistent_mtp.packaged_shim_path", return_value=packaged), mock.patch(
                "orbit.native_llama.persistent_mtp._shim_exports_required_symbols", return_value=True
            ):
                shim = build_persistent_mtp_shim(llama_root=None)

        self.assertEqual(shim, packaged)

    def test_build_persistent_shim_rebuilds_when_packaged_artifact_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            packaged = tmp_path / "liborbit-persistent-mtp.so"
            packaged.write_text("stale", encoding="utf-8")
            llama_root = tmp_path / "llama"
            (llama_root / "build/bin").mkdir(parents=True)
            for name in ("libllama-common.so", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"):
                (llama_root / "build/bin" / name).write_text("", encoding="utf-8")
            runner = mock.Mock(return_value=mock.Mock(returncode=0, stderr="", stdout=""))
            with mock.patch("orbit.native_llama.persistent_mtp.packaged_shim_path", return_value=packaged), mock.patch(
                "orbit.native_llama.persistent_mtp._shim_exports_required_symbols", side_effect=[False, False, True]
            ):
                shim = build_persistent_mtp_shim(llama_root=llama_root, build_dir=tmp_path, runner=runner)

        self.assertEqual(shim, packaged)
        runner.assert_called_once()

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
    def test_persistent_mtp_stays_disabled_by_default_when_experimental_flag_is_off(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client._session.ctx_tgt = object()

        with mock.patch("orbit.native_llama.client.create_persistent_mtp_session") as mocked_create:
            client._initialize_persistent_mtp_session()

        snapshot = client.session_snapshot()
        self.assertFalse(snapshot.mtp_enabled)
        self.assertFalse(snapshot.mtp_initialized)
        self.assertIsNone(snapshot.mtp_failure_reason)
        mocked_create.assert_not_called()

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_persistent_mtp_stays_disabled_when_draft_is_missing(self, _mocked_lib) -> None:
        client = NativeLlamaClient(
            self._paths(mtp_available=False, fallback_reason="draft-mtp-missing"),
            NativeClientConfig(use_mtp_experimental=True),
        )
        client._session.ctx_tgt = object()

        client._initialize_persistent_mtp_session()

        snapshot = client.session_snapshot()
        self.assertFalse(snapshot.mtp_enabled)
        self.assertFalse(snapshot.mtp_initialized)
        self.assertEqual(snapshot.mtp_failure_reason, "draft-mtp-missing")

    @mock.patch("orbit.native_llama.client.create_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_persistent_mtp_initializes_and_exposes_runtime_handles(self, _mocked_lib, mocked_create) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
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
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
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
        client = NativeLlamaClient(self._paths(), NativeClientConfig(use_mtp_experimental=True))
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

    def test_build_persistent_shim_requires_legacy_root_when_no_packaged_artifact_exists(self) -> None:
        with mock.patch("orbit.native_llama.persistent_mtp.packaged_shim_path", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "missing native build inputs for liborbit-persistent-mtp.so"):
                build_persistent_mtp_shim(llama_root=None)

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

    def test_run_persistent_mtp_completion_uses_noop_callbacks_when_callbacks_are_missing(self) -> None:
        class FakeLib:
            def orbit_mtp_session_complete(self, handle, ctx_tgt, prompt, max_tokens, token_cb, progress_cb, user_data):
                self.args = (handle, ctx_tgt, prompt, max_tokens, token_cb, progress_cb, user_data)
                return True

            def orbit_mtp_session_last_content(self, _handle):
                return b"ok"

            def orbit_mtp_session_last_output_tokens(self, _handle):
                return 1

            def orbit_mtp_session_last_draft_tokens_total(self, _handle):
                return 0

            def orbit_mtp_session_last_accepted_tokens_total(self, _handle):
                return 0

            def orbit_mtp_session_last_rejected_tokens_total(self, _handle):
                return 0

            def orbit_mtp_session_last_reused_draft_tokens_total(self, _handle):
                return 0

            def orbit_mtp_session_last_reused_accepted_tokens_total(self, _handle):
                return 0

            def orbit_mtp_session_last_reused_rejected_tokens_total(self, _handle):
                return 0

            def orbit_mtp_session_last_acceptance_ratio(self, _handle):
                return 0.0

            def orbit_mtp_session_last_fresh_acceptance_ratio(self, _handle):
                return 0.0

            def orbit_mtp_session_last_consumed_acceptance_ratio(self, _handle):
                return 0.0

            def orbit_mtp_session_last_target_decode_calls(self, _handle):
                return 0

            def orbit_mtp_session_last_draft_decode_calls(self, _handle):
                return 0

            def orbit_mtp_session_last_elapsed_ms(self, _handle):
                return 1.0

            def orbit_mtp_session_last_tokens_per_second(self, _handle):
                return 1.0

            def orbit_mtp_session_last_full_accept_steps(self, _handle):
                return 0

            def orbit_mtp_session_last_replay_steps(self, _handle):
                return 0

            def orbit_mtp_session_last_partial_accept_steps(self, _handle):
                return 0

            def orbit_mtp_session_last_partial_no_replay_steps(self, _handle):
                return 0

            def orbit_mtp_session_last_replay_fallback_steps(self, _handle):
                return 0

            def orbit_mtp_session_last_seq_rm_supported(self, _handle):
                return False

            def orbit_mtp_session_last_rollback_tokens_total(self, _handle):
                return 0

            def orbit_mtp_session_last_checkpoint_count(self, _handle):
                return 0

            def orbit_mtp_session_last_restore_count(self, _handle):
                return 0

        class FakeLibrary:
            def __init__(self, _build_bin, _shim_path) -> None:
                self.lib = FakeLib()

        runtime = PersistentMtpSessionRuntime(handle=object(), ctx_dft=object(), spec=object())
        with mock.patch("orbit.native_llama.persistent_mtp.build_persistent_mtp_shim", return_value=Path("/tmp/fake.so")):
            result = run_persistent_mtp_completion(
                llama_root=Path("/llama"),
                paths=self._paths(),
                runtime=runtime,
                ctx_tgt=object(),
                prompt="hello",
                max_tokens=8,
                library_factory=FakeLibrary,
            )

        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
