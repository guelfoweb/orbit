from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from orbit.native_llama.client import NativeRoutePrefixPrefillResult
from orbit.native_llama.native_names import runtime_library_filename
from orbit.native_server.app import (
    PREFIX_PREWARM_OFF,
    PREFIX_PREWARM_STARTUP,
    build_parser,
    prewarm_startup_route_prefix,
    resolve_bootstrap_paths,
    route_prefix_prewarm_mode,
    run_server,
    tools_startup_enabled,
)


class _FakeNativeClient:
    instances: list["_FakeNativeClient"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.config = _args[1] if len(_args) > 1 else None
        self.loaded = False
        self.closed = False
        self.capture_calls = 0
        self.raise_on_capture = False
        _FakeNativeClient.instances.append(self)

    def set_quiet_logging(self) -> None:
        return None

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    def capture_route_prefix_prefill_only(self, segments, *, tools_mode: str = "on", should_cancel=None):
        del should_cancel
        self.capture_calls += 1
        if self.raise_on_capture:
            raise RuntimeError("synthetic capture failure")
        if tools_mode != "on":
            return NativeRoutePrefixPrefillResult(
                attempted=False,
                succeeded=False,
                skipped=True,
                skip_reason="tools_mode_ineligible",
            )
        if not getattr(segments, "boundary_available", False):
            return NativeRoutePrefixPrefillResult(
                attempted=True,
                succeeded=False,
                skipped=False,
                failed_reason="route_boundary_unavailable",
            )
        return NativeRoutePrefixPrefillResult(
            attempted=True,
            succeeded=True,
            skipped=False,
            prefix_hash="prefix-hash-alpha",
            prefix_token_count=693,
            checkpoint_size_bytes=238454176,
            prefill_ms=12.0,
            decode_calls=3,
            restore_ready=True,
        )


class _FakeHTTPServer:
    instances: list["_FakeHTTPServer"] = []

    def __init__(self, address, handler) -> None:
        self.address = address
        self.handler = handler
        self.orbit_state = None
        self.closed = False
        _FakeHTTPServer.instances.append(self)

    def serve_forever(self) -> None:
        return None

    def server_close(self) -> None:
        self.closed = True


class NativeServerBootstrapTests(unittest.TestCase):
    def test_bootstrap_can_use_packaged_vendor_lib_without_llama_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_lib = root / "vendor/lib"
            models_dir = root / "models"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            vendor_lib.mkdir(parents=True)
            (vendor_lib / runtime_library_filename("llama")).write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            with mock.patch("orbit.native_llama.paths.DEFAULT_VENDOR_LIB_DIR", vendor_lib), mock.patch(
                "orbit.native_llama.paths.DEFAULT_VENDOR_BUILD_BIN", root / "missing-vendor-build-bin"
            ):
                args = build_parser().parse_args(["--models-dir", str(models_dir), "--hf-cache", str(root / "hf")])
                paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.build_bin, vendor_lib)
        self.assertIsNotNone(paths.llama_root)
        self.assertEqual(paths.model, target)

    def test_bootstrap_can_use_orbit_llama_lib_dir_without_llama_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_lib = root / "custom-lib"
            models_dir = root / "models"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            env_lib.mkdir(parents=True)
            (env_lib / runtime_library_filename("llama")).write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            with (
                mock.patch("orbit.native_llama.paths.DEFAULT_VENDOR_LIB_DIR", root / "missing-vendor-lib"),
                mock.patch("orbit.native_llama.paths.DEFAULT_VENDOR_BUILD_BIN", root / "missing-vendor-build-bin"),
                mock.patch("orbit.native_llama.paths.DEFAULT_LLAMA_LIB_DIR", env_lib),
            ):
                args = build_parser().parse_args(["--models-dir", str(models_dir), "--hf-cache", str(root / "hf")])
                paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.build_bin, env_lib)
        self.assertIsNotNone(paths.llama_root)
        self.assertEqual(paths.model, target)

    def test_bootstrap_defaults_to_model_id_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            models_dir = root / "models"
            build_bin = llama_root / "build/bin"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / runtime_library_filename("llama")).write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--models-dir", str(models_dir), "--hf-cache", str(root / "hf")])
            paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.model, target)
        self.assertEqual(paths.mmproj_model, mmproj)
        self.assertEqual(paths.model_id, "gemma4-12b-it-q4km")

    def test_parser_accepts_think_flag(self) -> None:
        args = build_parser().parse_args(["--think", "on"])

        self.assertEqual(args.think, "on")

    def test_parser_defaults_to_new_user_port(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.port, 12120)

    def test_route_prefix_prewarm_defaults_to_startup(self) -> None:
        self.assertEqual(route_prefix_prewarm_mode({}), PREFIX_PREWARM_STARTUP)

    def test_route_prefix_prewarm_accepts_startup(self) -> None:
        self.assertEqual(route_prefix_prewarm_mode({"ORBIT_KV_PREFIX_PREWARM": "startup"}), PREFIX_PREWARM_STARTUP)

    def test_route_prefix_prewarm_accepts_off(self) -> None:
        self.assertEqual(route_prefix_prewarm_mode({"ORBIT_KV_PREFIX_PREWARM": "off"}), PREFIX_PREWARM_OFF)

    def test_route_prefix_prewarm_invalid_value_falls_back_to_off(self) -> None:
        self.assertEqual(route_prefix_prewarm_mode({"ORBIT_KV_PREFIX_PREWARM": "soon"}), PREFIX_PREWARM_OFF)

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_tools_startup_enabled_defaults_to_true(self) -> None:
        self.assertTrue(tools_startup_enabled())

    @mock.patch.dict("os.environ", {"ORBIT_TOOLS": "off"}, clear=True)
    def test_tools_startup_enabled_accepts_off(self) -> None:
        self.assertFalse(tools_startup_enabled())

    @mock.patch.dict("os.environ", {"ORBIT_TOOLS": "browser"}, clear=True)
    def test_tools_startup_enabled_invalid_value_falls_back_to_false(self) -> None:
        self.assertFalse(tools_startup_enabled())

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_startup_prewarm_default_invokes_native_hook(self) -> None:
        client = _FakeNativeClient()

        result = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.succeeded)
        self.assertTrue(result.restore_ready)
        self.assertEqual(client.capture_calls, 1)

    @mock.patch.dict("os.environ", {"ORBIT_KV_PREFIX_PREWARM": "off"}, clear=True)
    def test_startup_prewarm_explicit_off_skips_without_capture(self) -> None:
        client = _FakeNativeClient()

        result = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "disabled")
        self.assertEqual(client.capture_calls, 0)

    @mock.patch.dict("os.environ", {"ORBIT_TOOLS": "off"}, clear=True)
    def test_startup_prewarm_tools_off_skips_without_capture(self) -> None:
        client = _FakeNativeClient()

        result = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "tools_disabled")
        self.assertEqual(client.capture_calls, 0)

    @mock.patch.dict("os.environ", {"ORBIT_KV_PREFIX_PREWARM": "startup"}, clear=True)
    def test_startup_prewarm_invokes_native_hook(self) -> None:
        client = _FakeNativeClient()

        result = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.succeeded)
        self.assertTrue(result.restore_ready)
        self.assertEqual(result.sampled_tokens, 0)
        self.assertEqual(result.generated_tokens, 0)
        self.assertEqual(client.capture_calls, 1)

    @mock.patch.dict(
        "os.environ",
        {"ORBIT_KV_PREFIX_PREWARM": "startup", "ORBIT_KV_PREFIX_ANCHOR": "off", "ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT": "1"},
        clear=True,
    )
    def test_startup_prewarm_anchor_off_wins_without_capture(self) -> None:
        client = _FakeNativeClient()

        result = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "anchor_disabled")
        self.assertEqual(client.capture_calls, 0)

    @mock.patch.dict("os.environ", {"ORBIT_KV_PREFIX_PREWARM": "startup"}, clear=True)
    def test_startup_prewarm_failure_is_metadata_only_and_safe(self) -> None:
        client = _FakeNativeClient()
        client.raise_on_capture = True

        result = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]
        rendered = str(result.to_metadata())

        self.assertTrue(result.attempted)
        self.assertFalse(result.succeeded)
        self.assertFalse(result.restore_ready)
        self.assertEqual(result.failed_reason, "startup_prewarm_failed:RuntimeError")
        self.assertNotIn("synthetic capture failure", rendered)
        self.assertNotIn("route policy", rendered)

    def test_bootstrap_supports_legacy_direct_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            build_bin = llama_root / "build/bin"
            target = root / "manual.gguf"
            mmproj = root / "manual-mmproj.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / runtime_library_filename("llama")).write_text("", encoding="utf-8")
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--model", str(target), "--mmproj", str(mmproj)])
            paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.model, target.resolve())
        self.assertEqual(paths.mmproj_model, mmproj.resolve())
        self.assertEqual(paths.model_id, "legacy-path")
        self.assertEqual(paths.fallback_reason, "legacy-model-path")

    def test_bootstrap_with_model_id_and_draft_present_exposes_mtp_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            models_dir = root / "models"
            build_bin = llama_root / "build/bin"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            draft = models_dir / "unsloth--gemma-4-12b-it-GGUF" / "MTP/gemma-4-12b-it-Q8_0-MTP.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / runtime_library_filename("llama")).write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            draft.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")
            draft.write_text("draft", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--model-id", "gemma4-12b-it-q4km", "--models-dir", str(models_dir), "--hf-cache", str(root / "hf")])
            paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.model, target)
        self.assertEqual(paths.mmproj_model, mmproj)
        self.assertEqual(paths.draft_mtp_model, draft)
        self.assertTrue(paths.multimodal_available)
        self.assertTrue(paths.mtp_available)

    def test_bootstrap_errors_when_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            build_bin = llama_root / "build/bin"
            build_bin.mkdir(parents=True)
            (build_bin / runtime_library_filename("llama")).write_text("", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--model-id", "gemma4-12b-it-q4km", "--models-dir", str(root / "models"), "--hf-cache", str(root / "hf")])
            with self.assertRaises(FileNotFoundError):
                resolve_bootstrap_paths(args)

    def test_run_server_reports_clear_error_when_native_runtime_is_missing(self) -> None:
        stderr = io.StringIO()
        with mock.patch(
            "orbit.native_server.app.resolve_bootstrap_paths",
            side_effect=FileNotFoundError("libllama.so not found. Searched: /missing/libllama.so."),
        ):
            with redirect_stderr(stderr):
                code = run_server([])

        self.assertEqual(code, 1)
        output = stderr.getvalue()
        self.assertIn("error: native backend libraries are missing.", output)
        self.assertIn("--llama-root", output)
        self.assertIn("ORBIT_LLAMA_ROOT", output)

    def test_run_server_reports_clear_error_when_mtp_shim_inputs_are_missing(self) -> None:
        stderr = io.StringIO()
        with mock.patch("orbit.native_server.app.resolve_bootstrap_paths") as mocked_paths, mock.patch(
            "orbit.native_server.app.NativeLlamaClient"
        ) as mocked_client:
            mocked_paths.return_value = mock.Mock()
            mocked_client.return_value.load.side_effect = RuntimeError(
                "missing native build inputs for liborbit-persistent-mtp.so"
            )
            with redirect_stderr(stderr):
                code = run_server(["--mtp"])

        self.assertEqual(code, 1)
        output = stderr.getvalue()
        self.assertIn("error: native MTP shim inputs are missing.", output)
        self.assertIn("--llama-root", output)

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_run_server_default_startup_prewarm_invokes_hook_before_serving(self) -> None:
        _FakeNativeClient.instances.clear()
        _FakeHTTPServer.instances.clear()
        with (
            mock.patch("orbit.native_server.app.resolve_bootstrap_paths", return_value=SimpleNamespace()),
            mock.patch("orbit.native_server.app.NativeLlamaClient", _FakeNativeClient),
            mock.patch("orbit.native_server.app.ThreadingHTTPServer", _FakeHTTPServer),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            code = run_server([])

        self.assertEqual(code, 0)
        self.assertEqual(len(_FakeNativeClient.instances), 1)
        self.assertEqual(_FakeNativeClient.instances[0].capture_calls, 1)
        self.assertTrue(_FakeNativeClient.instances[0].closed)
        self.assertTrue(_FakeHTTPServer.instances[0].closed)

    @mock.patch.dict("os.environ", {"ORBIT_KV_PREFIX_PREWARM": "off"}, clear=True)
    def test_run_server_explicit_prewarm_off_skips_hook(self) -> None:
        _FakeNativeClient.instances.clear()
        _FakeHTTPServer.instances.clear()
        with (
            mock.patch("orbit.native_server.app.resolve_bootstrap_paths", return_value=SimpleNamespace()),
            mock.patch("orbit.native_server.app.NativeLlamaClient", _FakeNativeClient),
            mock.patch("orbit.native_server.app.ThreadingHTTPServer", _FakeHTTPServer),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            code = run_server([])

        self.assertEqual(code, 0)
        self.assertEqual(len(_FakeNativeClient.instances), 1)
        self.assertEqual(_FakeNativeClient.instances[0].capture_calls, 0)
        self.assertIsNotNone(_FakeHTTPServer.instances[0].orbit_state)

    @mock.patch.dict(
        "os.environ",
        {"ORBIT_FINAL_PREFIX_REUSE": "0", "ORBIT_FINAL_PREFIX_EXPERIMENT": "1"},
        clear=True,
    )
    def test_run_server_uses_stable_final_prefix_precedence(self) -> None:
        _FakeNativeClient.instances.clear()
        _FakeHTTPServer.instances.clear()
        with (
            mock.patch("orbit.native_server.app.resolve_bootstrap_paths", return_value=SimpleNamespace()),
            mock.patch("orbit.native_server.app.NativeLlamaClient", _FakeNativeClient),
            mock.patch("orbit.native_server.app.ThreadingHTTPServer", _FakeHTTPServer),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            code = run_server([])

        self.assertEqual(code, 0)
        config = _FakeNativeClient.instances[0].config
        self.assertFalse(config.final_prefix_experiment_enabled)
        self.assertEqual(config.final_prefix_reuse_source, "stable")
        self.assertTrue(config.final_prefix_reuse_legacy_detected)
        self.assertIsNone(config.final_prefix_reuse_config_error)


if __name__ == "__main__":
    unittest.main()
