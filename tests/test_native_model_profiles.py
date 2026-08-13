from __future__ import annotations

import hashlib
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.capabilities import LlamaCppBuildInfo, safe_native_capability_manifest
from orbit.native_llama.model_profiles import (
    GEMMA4_PROFILE_ID,
    QWEN36_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
    NativeModelProfile,
    detect_native_model_profile,
    supports_low_memory_mode,
)
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT


QWEN_METADATA = {
    "general.architecture": "qwen35moe",
    "general.name": "Qwen3.6-35B-A3B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
}

QWEN3_CODER_METADATA = {
    "general.architecture": "qwen3moe",
    "general.name": "Qwen3-Coder-30B-A3B-Instruct",
    "general.file_type": "15",
    "general.quantization_version": "2",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen2",
    "tokenizer.ggml.add_bos_token": "false",
    "tokenizer.ggml.eos_token_id": "151645",
    "tokenizer.ggml.padding_token_id": "151654",
    "qwen3moe.context_length": "262144",
    "qwen3moe.expert_count": "128",
    "qwen3moe.expert_used_count": "8",
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
        self.assertFalse(supports_low_memory_mode(profile))
        self.assertTrue(profile.diagnostics(thinking_enabled=False)["capabilities"]["full_document_analysis"])

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
        self.assertFalse(supports_low_memory_mode(profile))
        self.assertTrue(profile.diagnostics(thinking_enabled=False)["capabilities"]["full_document_analysis"])

    def test_detects_verified_qwen3_coder_only_from_complete_identity(self) -> None:
        template = "official-qwen3-coder-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(QWEN3_CODER_METADATA, template)

        self.assertEqual(profile.profile_id, QWEN3_CODER_PROFILE_ID)
        self.assertEqual(profile.family, "qwen3-coder")
        self.assertTrue(profile.verified)
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        self.assertEqual(profile.tool_call_protocol, "qwen3-coder-xml")
        self.assertEqual(profile.history_serialization, "qwen3-coder-chatml")
        self.assertEqual(profile.artifact_content_protocol, "qwen3-coder-json-string-v1")
        self.assertFalse(profile.thinking_supported)
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.gemma_prefix_reuse_supported)
        self.assertTrue(profile.route_prefix_reuse_supported)
        self.assertFalse(profile.multimodal_supported)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertTrue(supports_low_memory_mode(profile))
        self.assertTrue(profile.diagnostics(thinking_enabled=False)["capabilities"]["full_document_analysis"])

    def test_qwen3_coder_template_drift_fails_closed(self) -> None:
        profile = detect_native_model_profile(QWEN3_CODER_METADATA, "unreviewed-template")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen3_coder_template_identity_mismatch")
        self.assertFalse(supports_low_memory_mode(profile))

    def test_qwen3_coder_architecture_drift_fails_closed(self) -> None:
        metadata = {**QWEN3_CODER_METADATA, "general.architecture": "qwen3"}

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "unsupported_model_profile")

    def test_unknown_qwen3_variant_fails_closed(self) -> None:
        metadata = {**QWEN3_CODER_METADATA, "general.name": "Qwen3-Coder-Next"}

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen3_coder_model_identity_mismatch")

    def test_qwen3_coder_filename_like_metadata_cannot_authorize_profile(self) -> None:
        metadata = {
            "general.filename": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            "general.architecture": "unknown",
        }

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.profile_id, "unsupported")

    def test_qwen3_coder_metadata_drift_fails_closed(self) -> None:
        template = "official-qwen3-coder-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        metadata = {**QWEN3_CODER_METADATA, "qwen3moe.expert_used_count": "4"}

        with mock.patch("orbit.native_llama.model_profiles.QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(metadata, template)

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen3_coder_metadata_identity_mismatch")

    def test_each_qwen3_coder_authorizing_metadata_field_fails_closed_on_drift(self) -> None:
        template = "official-qwen3-coder-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        drifted_values = {
            "general.file_type": "14",
            "general.quantization_version": "3",
            "tokenizer.ggml.model": "qwen",
            "tokenizer.ggml.pre": "qwen3",
            "tokenizer.ggml.add_bos_token": "true",
            "tokenizer.ggml.eos_token_id": "151643",
            "tokenizer.ggml.padding_token_id": "151645",
            "qwen3moe.context_length": "131072",
            "qwen3moe.expert_count": "64",
            "qwen3moe.expert_used_count": "4",
        }

        with mock.patch("orbit.native_llama.model_profiles.QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256", digest):
            for key, value in drifted_values.items():
                with self.subTest(key=key):
                    profile = detect_native_model_profile({**QWEN3_CODER_METADATA, key: value}, template)
                    self.assertFalse(profile.verified)
                    self.assertNotEqual(profile.profile_id, QWEN3_CODER_PROFILE_ID)

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

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_qwen3_coder_capability_manifest_declares_profile_limits(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(9551, "6f79e02", "x86_64", "GNU", "a" * 64, "runtime_symbols")
        profile = NativeModelProfile(
            profile_id=QWEN3_CODER_PROFILE_ID,
            family="qwen3-coder",
            model_name="Qwen3-Coder-30B-A3B-Instruct",
            architecture="qwen3moe",
            renderer="llama.cpp-jinja",
            reasoning_protocol="none",
            tool_call_protocol="qwen3-coder-xml",
            history_serialization="qwen3-coder-chatml",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256="c" * 64,
            thinking_supported=False,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization="Q4_K_M",
            artifact_content_protocol="qwen3-coder-json-string-v1",
        )
        client = SimpleNamespace(model_profile=profile, paths=SimpleNamespace(build_bin="/native"))

        manifest = safe_native_capability_manifest(client, final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)

        self.assertEqual(manifest["profile_id"], QWEN3_CODER_PROFILE_ID)
        self.assertEqual(manifest["renderer"]["artifact_content_protocol"], "qwen3-coder-json-string-v1")
        self.assertFalse(manifest["requirements"]["thinking_supported"])
        self.assertFalse(manifest["requirements"]["route_prefix_reuse_supported"])
        self.assertFalse(manifest["requirements"]["multimodal_supported"])


if __name__ == "__main__":
    unittest.main()
