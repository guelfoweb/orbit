from __future__ import annotations

import hashlib
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.capabilities import LlamaCppBuildInfo, safe_native_capability_manifest
from orbit.native_llama.model_profiles import (
    GEMMA4_PROFILE_ID,
    QWEN36_PROFILE_ID,
    NativeModelProfile,
    detect_native_model_profile,
)
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT


QWEN_METADATA = {
    "general.architecture": "qwen35moe",
    "general.name": "Qwen3.6-35B-A3B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
}


class NativeModelProfileTests(unittest.TestCase):
    def test_detects_verified_qwen_only_from_complete_identity(self) -> None:
        template = "official-qwen-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.QWEN36_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(QWEN_METADATA, template)

        self.assertEqual(profile.profile_id, QWEN36_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")
        self.assertEqual(profile.history_serialization, "qwen-leading-system-only")
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.gemma_prefix_reuse_supported)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")

    def test_qwen_template_drift_fails_closed(self) -> None:
        profile = detect_native_model_profile(QWEN_METADATA, "unreviewed-template")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen36_template_identity_mismatch")

    def test_qwen_model_size_drift_fails_closed(self) -> None:
        metadata = {**QWEN_METADATA, "general.name": "Qwen3.6-8B"}

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen36_model_identity_mismatch")

    def test_qwen_quantization_drift_fails_closed(self) -> None:
        template = "official-qwen-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        metadata = {**QWEN_METADATA, "general.file_type": "2"}

        with mock.patch("orbit.native_llama.model_profiles.QWEN36_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(metadata, template)

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen36_quantization_identity_mismatch")

    def test_existing_gemma_family_keeps_orbit_profile(self) -> None:
        profile = detect_native_model_profile(
            {
                "general.architecture": "gemma4",
                "general.name": "Gemma 4 26B-A4B",
                "tokenizer.ggml.model": "gemma4",
            },
            "embedded-template-not-used-by-orbit-renderer",
        )

        self.assertEqual(profile.profile_id, GEMMA4_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.renderer, "orbit-gemma4")
        self.assertTrue(profile.mtp_supported)
        self.assertTrue(profile.gemma_prefix_reuse_supported)

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_qwen_capability_manifest_is_profile_aware(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(9551, "6f79e02", "x86_64", "GNU", "a" * 64, "runtime_symbols")
        profile = NativeModelProfile(
            profile_id=QWEN36_PROFILE_ID,
            family="qwen3.6",
            model_name="Qwen3.6-35B-A3B",
            architecture="qwen35moe",
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256="b" * 64,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
        )
        client = SimpleNamespace(model_profile=profile, paths=SimpleNamespace(build_bin="/native"))

        manifest = safe_native_capability_manifest(client, final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)

        self.assertEqual(manifest["profile_id"], QWEN36_PROFILE_ID)
        self.assertEqual(manifest["status"], "verified")
        self.assertTrue(manifest["behavior_enforced"])
        self.assertEqual(manifest["renderer"]["template_hash"], "b" * 64)
        self.assertFalse(manifest["requirements"]["mtp_supported"])


if __name__ == "__main__":
    unittest.main()
