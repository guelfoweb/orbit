from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping


GEMMA4_PROFILE_ID = "orbit-gemma4-native-v1"
QWEN36_PROFILE_ID = "orbit-qwen36-native-v1"
QWEN36_OFFICIAL_TEMPLATE_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
QWEN36_VERIFIED_FILE_TYPE = "15"
QWEN36_VERIFIED_QUANTIZATION = "Q4_K_M"
QWEN3_CODER_PROFILE_ID = "orbit-qwen3-coder-native-v1"
QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256 = "87710339d25b4e789c1d723f93c91ee861a86d305bb3d20a845536f251d6ea8a"
QWEN3_CODER_VERIFIED_FILE_TYPE = "15"
QWEN3_CODER_VERIFIED_QUANTIZATION = "Q4_K_M"


@dataclass(frozen=True)
class NativeModelProfile:
    profile_id: str
    family: str
    model_name: str
    architecture: str
    renderer: str
    reasoning_protocol: str
    tool_call_protocol: str
    history_serialization: str
    verified: bool
    failure_reason: str | None
    template_source: str
    template_sha256: str
    thinking_supported: bool
    mtp_supported: bool
    gemma_prefix_reuse_supported: bool
    verified_quantization: str | None = None
    artifact_content_protocol: str = "literal-content-v1"
    route_prefix_reuse_supported: bool = False
    multimodal_supported: bool = False

    @property
    def uses_native_chat_bridge(self) -> bool:
        return self.renderer == "llama.cpp-jinja"

    def diagnostics(self, *, thinking_enabled: bool) -> dict[str, object]:
        return {
            "model_family": self.family,
            "model_name": self.model_name,
            "compatibility_profile": self.profile_id,
            "architecture": self.architecture,
            "template_source": self.template_source,
            "template_hash": self.template_sha256,
            "renderer": self.renderer,
            "thinking_supported": self.thinking_supported,
            "thinking_enabled": thinking_enabled,
            "reasoning_protocol": self.reasoning_protocol,
            "tool_call_protocol": self.tool_call_protocol,
            "history_serialization": self.history_serialization,
            "verified": self.verified,
            "failure_reason": self.failure_reason,
            "mtp_supported": self.mtp_supported,
            "verified_quantization": self.verified_quantization,
            "artifact_content_protocol": self.artifact_content_protocol,
            "capabilities": {
                "chat": self.verified,
                "tools": self.verified,
                "tool_history_results": self.verified,
                "write_artifact": self.verified,
                "verify_artifact": self.verified,
                "mtp": self.mtp_supported,
                "route_prefix_reuse": self.route_prefix_reuse_supported,
                "multimodal": self.multimodal_supported,
                "arbitrary_exact_copy_artifact": False,
                "empty_artifact": False,
            },
        }


def detect_native_model_profile(metadata: Mapping[str, str], template: str) -> NativeModelProfile:
    architecture = metadata.get("general.architecture", "").strip().lower()
    tokenizer_model = metadata.get("tokenizer.ggml.model", "").strip().lower()
    tokenizer_pre = metadata.get("tokenizer.ggml.pre", "").strip().lower()
    model_name = metadata.get("general.name", "").strip()
    file_type = metadata.get("general.file_type", "").strip()
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()

    if architecture == "gemma4" and tokenizer_model == "gemma4":
        return NativeModelProfile(
            profile_id=GEMMA4_PROFILE_ID,
            family="gemma4",
            model_name=model_name,
            architecture=architecture,
            renderer="orbit-gemma4",
            reasoning_protocol="gemma4-control-channel",
            tool_call_protocol="gemma4-native",
            history_serialization="orbit-native-roles",
            verified=True,
            failure_reason=None,
            template_source="orbit-reviewed",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=True,
            gemma_prefix_reuse_supported=True,
            route_prefix_reuse_supported=True,
            multimodal_supported=True,
        )

    qwen_identity = (
        architecture == "qwen35moe"
        and model_name == "Qwen3.6-35B-A3B"
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN36_VERIFIED_FILE_TYPE
        and template_hash == QWEN36_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen_identity:
        return NativeModelProfile(
            profile_id=QWEN36_PROFILE_ID,
            family="qwen3.6",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN36_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
        )

    qwen3_coder_identity = (
        architecture == "qwen3moe"
        and model_name == "Qwen3-Coder-30B-A3B-Instruct"
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen2"
        and file_type == QWEN3_CODER_VERIFIED_FILE_TYPE
        and metadata.get("general.quantization_version", "").strip() == "2"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() == "false"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "151645"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "151654"
        and metadata.get("qwen3moe.context_length", "").strip() == "262144"
        and metadata.get("qwen3moe.expert_count", "").strip() == "128"
        and metadata.get("qwen3moe.expert_used_count", "").strip() == "8"
        and template_hash == QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen3_coder_identity:
        return NativeModelProfile(
            profile_id=QWEN3_CODER_PROFILE_ID,
            family="qwen3-coder",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="none",
            tool_call_protocol="qwen3-coder-xml",
            history_serialization="qwen3-coder-chatml",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=False,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN3_CODER_VERIFIED_QUANTIZATION,
            artifact_content_protocol="qwen3-coder-json-string-v1",
        )

    reason = _unverified_reason(
        architecture=architecture,
        model_name=model_name,
        tokenizer_model=tokenizer_model,
        tokenizer_pre=tokenizer_pre,
        file_type=file_type,
        template_hash=template_hash,
    )
    return NativeModelProfile(
        profile_id="unsupported",
        family=architecture or "unknown",
        model_name=model_name,
        architecture=architecture or "unknown",
        renderer="unsupported",
        reasoning_protocol="unsupported",
        tool_call_protocol="unsupported",
        history_serialization="unsupported",
        verified=False,
        failure_reason=reason,
        template_source="gguf-embedded" if template else "missing",
        template_sha256=template_hash,
        thinking_supported=False,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
    )


def _unverified_reason(
    *,
    architecture: str,
    model_name: str,
    tokenizer_model: str,
    tokenizer_pre: str,
    file_type: str,
    template_hash: str,
) -> str:
    if architecture == "qwen35moe":
        if model_name != "Qwen3.6-35B-A3B":
            return "qwen36_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
            return "qwen36_tokenizer_identity_mismatch"
        if file_type != QWEN36_VERIFIED_FILE_TYPE:
            return "qwen36_quantization_identity_mismatch"
        if template_hash != QWEN36_OFFICIAL_TEMPLATE_SHA256:
            return "qwen36_template_identity_mismatch"
    if architecture == "qwen3moe":
        if model_name != "Qwen3-Coder-30B-A3B-Instruct":
            return "qwen3_coder_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen2":
            return "qwen3_coder_tokenizer_identity_mismatch"
        if file_type != QWEN3_CODER_VERIFIED_FILE_TYPE:
            return "qwen3_coder_quantization_identity_mismatch"
        if template_hash != QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256:
            return "qwen3_coder_template_identity_mismatch"
        return "qwen3_coder_metadata_identity_mismatch"
    return "unsupported_model_profile"
