from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping


GEMMA4_PROFILE_ID = "orbit-gemma4-native-v1"
GEMMA4_VERIFIED_MODEL_NAME = "gemma-4-26B-A4B-it"
QWEN36_PROFILE_ID = "orbit-qwen36-native-v1"
QWEN36_VERIFIED_MODEL_NAME = "Qwen3.6-35B-A3B"
QWEN36_OFFICIAL_TEMPLATE_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
QWEN36_VERIFIED_FILE_TYPE = "15"
QWEN36_VERIFIED_QUANTIZATION = "Q4_K_M"
QWEN38_PROFILE_ID = "orbit-qwen38-native-v1"
QWEN38_VERIFIED_MODEL_NAME = "Qwen3.8-27B"
QWEN38_OFFICIAL_TEMPLATE_SHA256 = "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"
QWEN38_VERIFIED_FILE_TYPE = "15"
QWEN38_VERIFIED_QUANTIZATION = "Q4_K_M"
# Metadata keys read when identifying a model profile. Every key any pinned
# identity below compares must appear here: a key absent from this set is read
# as the empty string, which silently fails the comparison and reports a
# verified model as unsupported. Callers that extract GGUF metadata filter
# through this one set, so adding a key to an identity and forgetting the
# extraction cannot happen twice in two places.
PROFILE_METADATA_KEYS = frozenset(
    {
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "qwen3moe.context_length",
        "qwen3moe.block_count",
        "qwen3moe.expert_count",
        "qwen3moe.expert_used_count",
        "qwen35.context_length",
        "qwen35.block_count",
        "qwen35moe.context_length",
        "qwen35moe.block_count",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
    }
)

ORNITH15_PROFILE_ID = "orbit-ornith15-native-v1"
ORNITH15_VERIFIED_MODEL_NAME = "Ornith-1.5-35B"
ORNITH15_OFFICIAL_TEMPLATE_SHA256 = "f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b"
ORNITH15_VERIFIED_FILE_TYPE = "15"
ORNITH15_VERIFIED_QUANTIZATION = "Q4_K_M"
QWEN3_CODER_PROFILE_ID = "orbit-qwen3-coder-native-v1"
QWEN3_CODER_VERIFIED_MODEL_NAME = "Qwen3-Coder-30B-A3B-Instruct"
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
                "full_document_analysis": self.verified,
                "mtp": self.mtp_supported,
                "route_prefix_reuse": self.route_prefix_reuse_supported,
                "multimodal": self.multimodal_supported,
                "arbitrary_exact_copy_artifact": False,
                "empty_artifact": False,
            },
        }


def supports_low_memory_mode(profile: NativeModelProfile | None) -> bool:
    return bool(
        profile is not None
        and profile.verified
        and profile.profile_id == QWEN3_CODER_PROFILE_ID
    )


_VERIFIED_TEMPLATE_HISTORY_SERIALIZATION = {
    QWEN36_OFFICIAL_TEMPLATE_SHA256: "qwen-leading-system-only",
    QWEN38_OFFICIAL_TEMPLATE_SHA256: "qwen-leading-system-only",
    ORNITH15_OFFICIAL_TEMPLATE_SHA256: "qwen-leading-system-only",
}


@dataclass(frozen=True)
class VerifiedNativeModelIdentity:
    profile_id: str
    model_name: str
    architecture: str


VERIFIED_NATIVE_MODEL_IDENTITIES = (
    VerifiedNativeModelIdentity(GEMMA4_PROFILE_ID, GEMMA4_VERIFIED_MODEL_NAME, "gemma4"),
    VerifiedNativeModelIdentity(QWEN36_PROFILE_ID, QWEN36_VERIFIED_MODEL_NAME, "qwen35moe"),
    VerifiedNativeModelIdentity(ORNITH15_PROFILE_ID, ORNITH15_VERIFIED_MODEL_NAME, "qwen35moe"),
    VerifiedNativeModelIdentity(QWEN38_PROFILE_ID, QWEN38_VERIFIED_MODEL_NAME, "qwen35"),
    VerifiedNativeModelIdentity(
        QWEN3_CODER_PROFILE_ID,
        QWEN3_CODER_VERIFIED_MODEL_NAME,
        "qwen3moe",
    ),
)


def verified_native_model_identity(profile_id: str) -> VerifiedNativeModelIdentity | None:
    return next(
        (identity for identity in VERIFIED_NATIVE_MODEL_IDENTITIES if identity.profile_id == profile_id),
        None,
    )


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
        and model_name == QWEN36_VERIFIED_MODEL_NAME
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

    ornith15_identity = (
        architecture == "qwen35moe"
        and model_name == ORNITH15_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == ORNITH15_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("qwen35moe.context_length", "").strip() == "262144"
        and metadata.get("qwen35moe.block_count", "").strip() == "41"
        and metadata.get("qwen35moe.expert_count", "").strip() == "256"
        and metadata.get("qwen35moe.expert_used_count", "").strip() == "8"
        and template_hash == ORNITH15_OFFICIAL_TEMPLATE_SHA256
    )
    if ornith15_identity:
        return NativeModelProfile(
            profile_id=ORNITH15_PROFILE_ID,
            family="ornith1.5",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            # Tool envelope verified byte-identical to the reviewed Qwen3.6
            # template: same <tool_call>/<function=>/<parameter=> form and
            # <tool_response> results.
            tool_call_protocol="qwen3.6-xml",
            # This template silently DROPS a system message past the leading
            # run instead of raising, so Orbit evidence and citation cards
            # would vanish without the leading-system-only contract.
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=ORNITH15_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
        )

    qwen38_identity = (
        architecture == "qwen35"
        and model_name == QWEN38_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN38_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("qwen35.context_length", "").strip() == "262144"
        and metadata.get("qwen35.block_count", "").strip() == "65"
        and template_hash == QWEN38_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen38_identity:
        return NativeModelProfile(
            profile_id=QWEN38_PROFILE_ID,
            family="qwen3.8",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            # Wire protocol verified byte-identical to Qwen3.6: same
            # <tool_call>/<function=>/<parameter=> envelope, same
            # <tool_response> result form. Qwen3.8 only adds template-side
            # argument validation, which does not change the emitted format.
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN38_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
        )

    qwen3_coder_identity = (
        architecture == "qwen3moe"
        and model_name == QWEN3_CODER_VERIFIED_MODEL_NAME
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
            route_prefix_reuse_supported=True,
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
        if model_name == ORNITH15_VERIFIED_MODEL_NAME:
            if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
                return "ornith15_tokenizer_identity_mismatch"
            if file_type != ORNITH15_VERIFIED_FILE_TYPE:
                return "ornith15_quantization_identity_mismatch"
            if template_hash != ORNITH15_OFFICIAL_TEMPLATE_SHA256:
                return "ornith15_template_identity_mismatch"
            return "ornith15_metadata_identity_mismatch"
        if model_name != QWEN36_VERIFIED_MODEL_NAME:
            return "qwen36_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
            return "qwen36_tokenizer_identity_mismatch"
        if file_type != QWEN36_VERIFIED_FILE_TYPE:
            return "qwen36_quantization_identity_mismatch"
        if template_hash != QWEN36_OFFICIAL_TEMPLATE_SHA256:
            return "qwen36_template_identity_mismatch"
    if architecture == "qwen35":
        if model_name != QWEN38_VERIFIED_MODEL_NAME:
            return "qwen38_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
            return "qwen38_tokenizer_identity_mismatch"
        if file_type != QWEN38_VERIFIED_FILE_TYPE:
            return "qwen38_quantization_identity_mismatch"
        if template_hash != QWEN38_OFFICIAL_TEMPLATE_SHA256:
            return "qwen38_template_identity_mismatch"
        return "qwen38_metadata_identity_mismatch"
    if architecture == "qwen3moe":
        if model_name != QWEN3_CODER_VERIFIED_MODEL_NAME:
            return "qwen3_coder_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen2":
            return "qwen3_coder_tokenizer_identity_mismatch"
        if file_type != QWEN3_CODER_VERIFIED_FILE_TYPE:
            return "qwen3_coder_quantization_identity_mismatch"
        if template_hash != QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256:
            return "qwen3_coder_template_identity_mismatch"
        return "qwen3_coder_metadata_identity_mismatch"
    return "unsupported_model_profile"


def history_serialization_for_template(template: str) -> str | None:
    """History-serialization contract for a template that matches a verified pin.

    Matches the exact template text against the digests of verified profile
    templates, so a backend without Orbit-native metadata can still honour the
    profile's message-shape contract. Returns None when the template matches no
    verified profile, which keeps unverified models free of model-specific
    handling.
    """

    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return _VERIFIED_TEMPLATE_HISTORY_SERIALIZATION.get(digest)
