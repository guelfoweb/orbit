"""R1 (RUNTIME-DECOMPOSITION-CAMPAIGN-1): the read-only status/metrics accessors
moved to native_llama/client_status.py, with the client keeping thin delegates.

These prove the delegate and the module function return identical values (so the
delegation is correct and complete), that the accessors mutate no client state,
and that client_status never imports client (no runtime cycle).
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import client_status  # noqa: E402
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient  # noqa: E402
from orbit.native_llama.model_profiles import NativeModelProfile, ORNITH15_PROFILE_ID  # noqa: E402
from orbit.native_llama.paths import NativeLlamaPaths  # noqa: E402


ACCESSORS = (
    "final_prefix_experiment_status",
    "qwen_route_prefix_reuse_status",
    "qwen3_coder_route_prefix_reuse_status",
    "ornith_route_prefix_reuse_status",
    "qwen36_shell_tool_prefix_reuse_status",
    "compatibility_diagnostics",
    "model_load_status",
    "moe_expert_usage_status",
)


def _profile():
    return NativeModelProfile(
        profile_id=ORNITH15_PROFILE_ID, family="ornith1.5", model_name="Ornith-1.5-35B",
        architecture="qwen35moe", renderer="llama.cpp-jinja", reasoning_protocol="qwen-think",
        tool_call_protocol="qwen3.6-xml", history_serialization="qwen-leading-system-only",
        verified=True, failure_reason=None, template_source="gguf-embedded-official",
        template_sha256="f" * 64, thinking_supported=True, mtp_supported=False,
        gemma_prefix_reuse_supported=False, route_prefix_reuse_supported=True,
        verified_quantization="Q4_K_M",
    )


class ClientStatusDelegationTests(unittest.TestCase):
    def _client(self) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=pathlib.Path("/llama"), build_bin=pathlib.Path("/llama/build/bin"),
            library=pathlib.Path("/llama/build/bin/libllama.so"),
            model=pathlib.Path("/models/ornith.gguf"), model_id="legacy-path",
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(paths, NativeClientConfig())
        client.model_profile = _profile()
        client._model_metadata_identity = {"general.file_type": "15"}
        return client

    def test_each_delegate_returns_the_module_functions_value(self) -> None:
        client = self._client()
        for name in ACCESSORS:
            with self.subTest(accessor=name):
                delegate = getattr(client, name)()
                direct = getattr(client_status, name)(client)
                self.assertEqual(delegate, direct)

    def test_accessors_do_not_mutate_client_state(self) -> None:
        client = self._client()
        before = {k: v for k, v in vars(client).items()}
        for name in ACCESSORS:
            getattr(client, name)()
        after = vars(client)
        self.assertEqual(set(before), set(after))
        for k in before:
            self.assertIs(before[k], after[k], f"{k} was rebound by a read-only accessor")

    def test_client_status_does_not_import_client(self) -> None:
        src = (ROOT / "src/orbit/native_llama/client_status.py").read_text()
        # a runtime import of client would be a cycle; only a TYPE_CHECKING import is allowed
        self.assertNotIn("\nfrom .client import", src.replace("if TYPE_CHECKING", "#guard"))
        self.assertNotIn("\nimport orbit.native_llama.client\n", src)

    def test_route_prefix_accessor_reports_eligible_state(self) -> None:
        # a real value assertion so a dropped/renamed field is caught here too
        client = self._client()
        status = client.ornith_route_prefix_reuse_status()
        self.assertIn("failure_reason", status)
        self.assertEqual(status["profile_identity"], ORNITH15_PROFILE_ID)


if __name__ == "__main__":
    unittest.main()
