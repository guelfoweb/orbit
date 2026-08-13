from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.model_profiles import QWEN3_CODER_PROFILE_ID
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.prefix_anchor import PrefixAnchorState
from orbit.native_server.app import _model_load_props, build_parser


def _profile(profile_id: str, *, verified: bool = True):
    return SimpleNamespace(profile_id=profile_id, verified=verified)


class Qwen3CoderLowMemoryTests(unittest.TestCase):
    def _client(self, *, low_memory: bool) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/qwen3-coder.gguf"),
            model_id="qwen3-coder-30b-a3b-instruct-q4-k-m",
        )
        binding = mock.Mock()
        binding.lib.llama_model_default_params.return_value = SimpleNamespace(
            use_extra_bufts=True,
            progress_callback=None,
            progress_callback_user_data=None,
            n_gpu_layers=0,
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary", return_value=binding):
            return NativeLlamaClient(paths, NativeClientConfig(low_memory=low_memory))

    def test_parser_defaults_off_and_documents_opt_in(self) -> None:
        parser = build_parser()

        self.assertFalse(parser.parse_args([]).low_memory)
        self.assertTrue(parser.parse_args(["--low-memory"]).low_memory)
        help_text = parser.format_help()
        self.assertIn("--low-memory", help_text)
        self.assertIn(
            "Use a qualified low-memory profile when supported by the selected model.",
            " ".join(help_text.split()),
        )

    def test_default_preserves_backend_cpu_repack_setting(self) -> None:
        client = self._client(low_memory=False)
        with mock.patch("orbit.native_llama.client.inspect_native_model_profile") as inspect:
            params = client._model_load_params(None)

        self.assertTrue(params.use_extra_bufts)
        self.assertEqual(client.model_load_status(), {"low_memory": False, "cpu_repack": True})
        inspect.assert_not_called()

    def test_low_memory_disables_repack_only_for_verified_qwen3_coder(self) -> None:
        client = self._client(low_memory=True)
        with mock.patch(
            "orbit.native_llama.client.inspect_native_model_profile",
            return_value=_profile(QWEN3_CODER_PROFILE_ID),
        ) as inspect:
            params = client._model_load_params(None)

        self.assertFalse(params.use_extra_bufts)
        self.assertEqual(client.model_load_status(), {"low_memory": True, "cpu_repack": False})
        inspect.assert_called_once_with(client.lib, client.paths.model)

    def test_unsupported_or_unverified_profiles_fail_before_model_load(self) -> None:
        cases = (
            _profile("orbit-qwen36-native-v1"),
            _profile("orbit-gemma4-native-v1"),
            _profile(QWEN3_CODER_PROFILE_ID, verified=False),
        )
        for profile in cases:
            with self.subTest(profile=profile.profile_id, verified=profile.verified):
                client = self._client(low_memory=True)
                with mock.patch(
                    "orbit.native_llama.client.inspect_native_model_profile",
                    return_value=profile,
                ):
                    with self.assertRaisesRegex(RuntimeError, "requires verified native profile"):
                        client._model_load_params(None)
                client.lib.lib.llama_model_default_params.assert_not_called()

    def test_malformed_preflight_failure_is_clear_and_fail_closed(self) -> None:
        client = self._client(low_memory=True)
        with mock.patch(
            "orbit.native_llama.client.inspect_native_model_profile",
            side_effect=RuntimeError("vocab-only GGUF inspection failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "--low-memory model verification failed"):
                client._model_load_params(None)
        client.lib.lib.llama_model_default_params.assert_not_called()

    def test_loaded_identity_is_checked_again_after_preflight(self) -> None:
        client = self._client(low_memory=True)
        client.model_profile = _profile("orbit-qwen36-native-v1")

        with self.assertRaisesRegex(RuntimeError, "identity changed during load"):
            client._validate_loaded_low_memory_profile()

    def test_startup_and_reload_pass_no_repack_to_native_model_load(self) -> None:
        for reloading in (False, True):
            with self.subTest(reloading=reloading):
                client = self._client(low_memory=True)
                if reloading:
                    client._model = 1  # type: ignore[assignment]
                    client._qwen3_coder_route_prefix_anchor_state = PrefixAnchorState(
                        valid=True,
                        checkpoint_data=b"route",
                        checkpoint_size=5,
                    )
                client.lib.lib.llama_model_load_from_file.return_value = None
                with mock.patch(
                    "orbit.native_llama.client.inspect_native_model_profile",
                    return_value=_profile(QWEN3_CODER_PROFILE_ID),
                ):
                    with self.assertRaisesRegex(RuntimeError, "failed to load model"):
                        client.load()

                params = client.lib.lib.llama_model_load_from_file.call_args.args[1]
                self.assertFalse(params.use_extra_bufts)
                if reloading:
                    self.assertFalse(client._qwen3_coder_route_prefix_anchor_state.valid)

    def test_low_memory_mode_survives_lifecycle_cleanup_without_preserving_state(self) -> None:
        for operation in ("cancel", "reset", "close"):
            with self.subTest(operation=operation):
                client = self._client(low_memory=True)
                client._cpu_repack_enabled = False
                client.model_profile = _profile(QWEN3_CODER_PROFILE_ID)
                client._qwen3_coder_route_prefix_anchor_state = PrefixAnchorState(
                    valid=True,
                    checkpoint_data=b"route",
                    checkpoint_size=5,
                )
                if operation == "reset":
                    client._session.ctx_tgt = 1  # type: ignore[assignment]
                    client.lib.lib.llama_get_memory.return_value = 2
                    client.reset_session_state()
                elif operation == "close":
                    client.close()
                else:
                    client.cancel()

                self.assertFalse(client._qwen3_coder_route_prefix_anchor_state.valid)
                self.assertEqual(
                    client.model_load_status(),
                    {"low_memory": True, "cpu_repack": False},
                )

    def test_diagnostics_expose_only_mode_and_repack_state(self) -> None:
        client = self._client(low_memory=True)
        client._cpu_repack_enabled = False

        self.assertEqual(
            _model_load_props(client),
            {"low_memory": True, "cpu_repack": False},
        )


if __name__ == "__main__":
    unittest.main()
