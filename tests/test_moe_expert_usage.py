from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.bindings import LlamaLibrary, llama_token
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.expert_usage import MAX_EXPERTS, MAX_LAYERS, summarize_expert_usage
from orbit.native_server.app import OrbitNativeHandler, OrbitNativeServer, build_parser


class ExpertUsageTests(unittest.TestCase):
    def test_summary_separates_phases_and_reports_top_n(self) -> None:
        counts = [0] * (2 * MAX_LAYERS * MAX_EXPERTS)
        tokens = [0] * (2 * MAX_LAYERS)
        counts[0] = 8
        counts[1] = 2
        counts[MAX_LAYERS * MAX_EXPERTS + 2] = 4
        tokens[0] = 5
        tokens[MAX_LAYERS] = 2

        result = summarize_expert_usage(counts, tokens, layers=1, experts=16, active=2)

        self.assertEqual(result["phases"]["prefill"]["selections"], 10)
        self.assertEqual(result["phases"]["decode"]["selections"], 4)
        self.assertEqual(result["aggregate"]["experts_observed"], 3)
        self.assertEqual(result["aggregate"]["top_n_coverage"]["16"], 1.0)
        self.assertEqual(result["aggregate"]["routed_tokens"], 7)
        self.assertTrue(result["aggregate"]["selection_accounting_valid"])

    def test_invalid_snapshot_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "snapshot size"):
            summarize_expert_usage([], [], layers=1, experts=16, active=2)
        counts = [0] * (2 * MAX_LAYERS * MAX_EXPERTS)
        tokens = [0] * (2 * MAX_LAYERS)
        with self.assertRaisesRegex(ValueError, "active expert"):
            summarize_expert_usage(counts, tokens, layers=1, experts=16, active=17)

    def test_disabled_native_build_only_fails_when_enabled(self) -> None:
        library = LlamaLibrary.__new__(LlamaLibrary)
        library.expert_usage_available = False
        library.configure_expert_usage(False)
        with self.assertRaisesRegex(RuntimeError, "build-native"):
            library.configure_expert_usage(True)

    def test_off_does_not_materialize_counter_pages(self) -> None:
        library = LlamaLibrary.__new__(LlamaLibrary)
        library.expert_usage_available = True
        library.cpu_lib = mock.Mock()
        library.configure_expert_usage(False)
        library.cpu_lib.ggml_backend_cpu_expert_usage_reset.assert_not_called()

    def test_reset_does_not_reset_session(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.config = NativeClientConfig(moe_expert_usage_enabled=True)
        client.lib = mock.Mock()
        client.moe_expert_usage_status = mock.Mock(return_value={"enabled": True})
        client.reset_session_state = mock.Mock()
        self.assertEqual(client.reset_moe_expert_usage(), {"enabled": True})
        client.reset_session_state.assert_not_called()

    def test_server_serializes_snapshot_and_reset_with_inference_lock(self) -> None:
        client = mock.Mock()
        client.moe_expert_usage_status.return_value = {"enabled": True}
        client.reset_moe_expert_usage.return_value = {"enabled": True}
        server = OrbitNativeServer.__new__(OrbitNativeServer)
        server.client = client
        server.lock = mock.MagicMock()
        server.moe_expert_usage_status()
        server.reset_moe_expert_usage()
        self.assertEqual(server.lock.__enter__.call_count, 2)

    def test_diagnostic_endpoints_are_thin_read_only_adapters(self) -> None:
        state = mock.Mock()
        state.moe_expert_usage_status.return_value = {"enabled": True}
        state.reset_moe_expert_usage.return_value = {"enabled": True, "selections": 0}
        handler = OrbitNativeHandler.__new__(OrbitNativeHandler)
        handler._state = mock.Mock(return_value=state)
        handler._json = mock.Mock()

        handler.path = "/diagnostics/expert-usage"
        handler.do_GET()
        handler._json.assert_called_once_with({"enabled": True})

        handler._json.reset_mock()
        handler.path = "/diagnostics/expert-usage/reset"
        handler.do_POST()
        handler._json.assert_called_once_with({"enabled": True, "selections": 0})

    def test_disabled_reset_endpoint_fails_closed(self) -> None:
        state = mock.Mock()
        state.reset_moe_expert_usage.side_effect = RuntimeError("telemetry is disabled")
        handler = OrbitNativeHandler.__new__(OrbitNativeHandler)
        handler._state = mock.Mock(return_value=state)
        handler._json = mock.Mock()
        handler.path = "/diagnostics/expert-usage/reset"

        handler.do_POST()

        handler._json.assert_called_once_with({"error": "telemetry is disabled"}, status=409)

    def test_phase_is_set_once_per_prefill_and_decode_operation(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.config = NativeClientConfig(moe_expert_usage_enabled=True)
        client.lib = mock.Mock()
        client.lib.lib.llama_batch_get_one.return_value = object()
        client.lib.lib.llama_decode.return_value = 0
        client.lib.lib.llama_time_us.return_value = 0
        client.cancel_event = mock.Mock()
        client.cancel_event.is_set.return_value = False
        client._session = SimpleNamespace(ctx_tgt=object(), sampler=object())
        client._vocab = object()
        client._decode_prompt_range((llama_token * 1)(1), processed=0, end=1, step=1, total=1)
        client._generate_from_current_context(max_tokens=0)
        self.assertEqual(client.lib.set_expert_usage_phase.call_args_list, [mock.call(1), mock.call(2)])

    def test_flag_and_native_hook_are_narrow_and_lock_free(self) -> None:
        self.assertFalse(build_parser().parse_args([]).moe_expert_usage)
        self.assertTrue(build_parser().parse_args(["--moe-expert-usage"]).moe_expert_usage)
        source = (
            Path(__file__).parents[1]
            / "src/orbit/native_llama/vendor/source/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c"
        ).read_text()
        hook = source[source.index("#define GGML_CPU_EXPERT_USAGE_PHASES"):source.index("#define GGML_THREADPOOL_N_THREADS_MASK")]
        self.assertIn('strcmp(end, ".ffn_down_exps.weight")', hook)
        self.assertNotIn("atomic_", hook)
        self.assertNotIn("mutex", hook)
        repack = (
            Path(__file__).parents[1]
            / "src/orbit/native_llama/vendor/source/llama.cpp/ggml/src/ggml-cpu/repack.cpp"
        ).read_text()
        self.assertIn("ggml_backend_cpu_expert_usage_record(src0->name, ids);", repack)


if __name__ == "__main__":
    unittest.main()
