from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.model_profiles import (
    _VERIFIED_TEMPLATE_HISTORY_SERIALIZATION,
    ORNITH15_OFFICIAL_TEMPLATE_SHA256,
    ORNITH15_PROFILE_ID,
    ORNITH15_VERIFIED_MODEL_NAME,
    QWEN36_PROFILE_ID,
    detect_native_model_profile,
    history_serialization_for_template,
)
from orbit.runtime.history_serialization import serialize_profile_messages


# The verified Ornith-1.5-35B metadata, transcribed from the audited GGUF. Any
# drift in these values must fail closed rather than resolve a near-match.
ORNITH15_METADATA = {
    "general.architecture": "qwen35moe",
    "general.name": ORNITH15_VERIFIED_MODEL_NAME,
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "qwen35moe.context_length": "262144",
    "qwen35moe.block_count": "41",
    "qwen35moe.expert_count": "256",
    "qwen35moe.expert_used_count": "8",
}


class Ornith15DetectionFixture(unittest.TestCase):
    """Shared fixture only: carries no assertions of its own.

    Subclasses inherit setUp/_detect. Keeping the identity assertions in a
    separate leaf class stops them re-running once per subclass.
    """

    def setUp(self) -> None:
        self.template = "ornith-official-template-body"
        self.digest = hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def _detect(self, metadata=None, template=None):
        from unittest import mock

        with mock.patch(
            "orbit.native_llama.model_profiles.ORNITH15_OFFICIAL_TEMPLATE_SHA256",
            self.digest,
        ):
            return detect_native_model_profile(
                ORNITH15_METADATA if metadata is None else metadata,
                self.template if template is None else template,
            )


class Ornith15IdentityTests(Ornith15DetectionFixture):
    def test_exact_identity_is_accepted_with_isolated_profile(self) -> None:
        profile = self._detect()

        self.assertTrue(profile.verified)
        self.assertEqual(profile.profile_id, ORNITH15_PROFILE_ID)
        self.assertEqual(profile.family, "ornith1.5")
        self.assertEqual(profile.architecture, "qwen35moe")
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertIsNone(profile.failure_reason)

    def test_does_not_resolve_to_the_qwen36_identity(self) -> None:
        profile = self._detect()

        # Ornith shares Qwen3.6's architecture, tokenizer, block count, and
        # expert layout. Only the template digest and model name separate them,
        # so this must never collapse into the Qwen3.6 profile.
        self.assertNotEqual(profile.profile_id, QWEN36_PROFILE_ID)
        self.assertNotEqual(profile.family, "qwen3.6")

    def test_mtp_and_prefix_reuse_acceleration_stay_off(self) -> None:
        profile = self._detect()

        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.gemma_prefix_reuse_supported)

    def test_reasoning_and_tool_protocols_match_the_verified_envelope(self) -> None:
        profile = self._detect()

        self.assertEqual(profile.reasoning_protocol, "qwen-think")
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")
        self.assertTrue(profile.thinking_supported)
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        self.assertEqual(profile.template_source, "gguf-embedded-official")
        self.assertEqual(profile.template_sha256, self.digest)


class Ornith15FailClosedTests(Ornith15DetectionFixture):
    def test_wrong_template_is_rejected(self) -> None:
        profile = self._detect(template="not-the-official-template")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "ornith15_template_identity_mismatch")

    def test_wrong_tokenizer_is_rejected(self) -> None:
        for key, value in (
            ("tokenizer.ggml.model", "llama"),
            ("tokenizer.ggml.pre", "qwen2"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**ORNITH15_METADATA, key: value})

                self.assertFalse(profile.verified)
                self.assertEqual(
                    profile.failure_reason, "ornith15_tokenizer_identity_mismatch"
                )

    def test_wrong_quantization_is_rejected(self) -> None:
        profile = self._detect({**ORNITH15_METADATA, "general.file_type": "7"})

        self.assertFalse(profile.verified)
        self.assertEqual(
            profile.failure_reason, "ornith15_quantization_identity_mismatch"
        )

    def test_critical_metadata_drift_is_rejected(self) -> None:
        for key, value in (
            ("tokenizer.ggml.bos_token_id", "1"),
            ("tokenizer.ggml.eos_token_id", "2"),
            ("qwen35moe.context_length", "32768"),
            ("qwen35moe.block_count", "40"),
            ("qwen35moe.expert_count", "128"),
            ("qwen35moe.expert_used_count", "4"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**ORNITH15_METADATA, key: value})

                self.assertFalse(profile.verified)
                self.assertEqual(
                    profile.failure_reason, "ornith15_metadata_identity_mismatch"
                )

    def test_near_match_model_name_is_rejected(self) -> None:
        for name in ("Ornith-1.5-35B-A3B", "ornith-1.5-35b", "Ornith-1.5-30B"):
            with self.subTest(name=name):
                profile = self._detect({**ORNITH15_METADATA, "general.name": name})

                self.assertFalse(profile.verified)
                # A name that is not the exact pin must not fall through to the
                # Ornith branch, and must not be adopted by Qwen3.6 either.
                self.assertNotEqual(profile.profile_id, ORNITH15_PROFILE_ID)
                self.assertNotEqual(profile.profile_id, QWEN36_PROFILE_ID)

    def test_ornith_template_does_not_verify_a_qwen36_named_model(self) -> None:
        profile = self._detect(
            {**ORNITH15_METADATA, "general.name": "Qwen3.6-35B-A3B"}
        )

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen36_template_identity_mismatch")


class Ornith15SystemMessageContractTests(Ornith15DetectionFixture):
    def test_profile_declares_the_leading_system_only_contract(self) -> None:
        profile = self._detect()

        # The Ornith template merges only the leading system run and silently
        # drops any later system message instead of raising. Orbit appends
        # trailing system messages for evidence, citation policy, and
        # full-document analysis, so the contract is mandatory here.
        self.assertEqual(profile.history_serialization, "qwen-leading-system-only")

    def test_trailing_system_evidence_survives_serialization(self) -> None:
        profile = self._detect()
        messages = [
            {"role": "system", "content": "orbit policy"},
            {"role": "user", "content": "analyze"},
            {"role": "system", "content": "evidence:e1 payload"},
        ]

        serialized = serialize_profile_messages(
            messages, history_serialization=profile.history_serialization
        )

        self.assertEqual(serialized[0]["role"], "system")
        # Demoted to user rather than dropped: the evidence still reaches the
        # model instead of vanishing inside the template loop.
        self.assertEqual(serialized[2]["role"], "user")
        self.assertEqual(serialized[2]["content"], "evidence:e1 payload")

    def test_utf8_and_json_argument_payloads_are_preserved_verbatim(self) -> None:
        profile = self._detect()
        payload = '{"path": "/tmp/fattura — é\\u00e8.js", "flags": ["--dry-run"]}'
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "试探 — прове́рка"},
            {"role": "system", "content": payload},
        ]

        serialized = serialize_profile_messages(
            messages, history_serialization=profile.history_serialization
        )

        self.assertEqual(serialized[1]["content"], "试探 — прове́рка")
        self.assertEqual(serialized[2]["content"], payload)

    def test_trailing_developer_message_is_also_demoted(self) -> None:
        # Orbit's context manager and session memory treat `developer` as a
        # peer of `system`. The Ornith template drops both silently past the
        # leading run, so the contract must cover the developer role too.
        profile = self._detect()
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "analyze"},
            {"role": "developer", "content": "evidence:e2 payload"},
        ]

        serialized = serialize_profile_messages(
            messages, history_serialization=profile.history_serialization
        )

        self.assertEqual(serialized[2]["role"], "user")
        self.assertEqual(serialized[2]["content"], "evidence:e2 payload")

    def test_leading_developer_message_is_preserved(self) -> None:
        profile = self._detect()
        messages = [
            {"role": "developer", "content": "policy"},
            {"role": "user", "content": "analyze"},
        ]

        serialized = serialize_profile_messages(
            messages, history_serialization=profile.history_serialization
        )

        self.assertEqual(serialized[0]["role"], "developer")

    def test_serialization_does_not_mutate_the_caller_messages(self) -> None:
        profile = self._detect()
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "system", "content": "trailing"},
        ]

        serialize_profile_messages(
            messages, history_serialization=profile.history_serialization
        )

        self.assertEqual(messages[1]["role"], "system")


class Ornith15RuntimeMetadataAllowlistTests(Ornith15DetectionFixture):
    """The pinned keys must survive the native client's metadata allowlist.

    detect_native_model_profile only ever sees keys that _read_model_metadata
    kept. A pin on a key missing from that allowlist verifies in tests and is
    then rejected at load time, which is how the Qwen3.8 profile first failed.
    """

    def _allowlisted_keys(self) -> set[str]:
        import ast

        source = (SRC / "orbit" / "native_llama" / "client.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "_read_model_metadata":
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Set):
                    return {
                        element.value
                        for element in inner.elts
                        if isinstance(element, ast.Constant) and isinstance(element.value, str)
                    }
        self.fail("could not locate the metadata allowlist in _read_model_metadata")

    def test_every_pinned_metadata_key_is_allowlisted(self) -> None:
        allowlisted = self._allowlisted_keys()

        missing = sorted(set(ORNITH15_METADATA) - allowlisted)

        self.assertEqual(missing, [], f"pinned keys stripped before detection: {missing}")

    def test_profile_still_verifies_after_allowlist_filtering(self) -> None:
        allowlisted = self._allowlisted_keys()
        filtered = {k: v for k, v in ORNITH15_METADATA.items() if k in allowlisted}

        profile = self._detect(filtered)

        self.assertTrue(profile.verified, profile.failure_reason)
        self.assertEqual(profile.profile_id, ORNITH15_PROFILE_ID)


class Ornith15RealTemplateTests(unittest.TestCase):
    """Validates the pins against the real audited template, unmocked.

    Every other class patches ORNITH15_OFFICIAL_TEMPLATE_SHA256, so a typo in
    the 64-character constant would otherwise ship green and then reject the
    real GGUF at load time.
    """

    def setUp(self) -> None:
        self.template = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "ornith15_chat_template.jinja"
        ).read_text(encoding="utf-8")

    def test_fixture_is_the_audited_template(self) -> None:
        self.assertEqual(len(self.template), 7828)

    def test_pinned_digest_matches_the_real_template(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.template.encode("utf-8")).hexdigest(),
            ORNITH15_OFFICIAL_TEMPLATE_SHA256,
        )

    def test_real_template_and_metadata_verify_without_any_patching(self) -> None:
        profile = detect_native_model_profile(ORNITH15_METADATA, self.template)

        self.assertTrue(profile.verified, profile.failure_reason)
        self.assertEqual(profile.profile_id, ORNITH15_PROFILE_ID)
        self.assertEqual(profile.history_serialization, "qwen-leading-system-only")
        self.assertFalse(profile.mtp_supported)

    def test_real_template_lookup_returns_the_contract(self) -> None:
        self.assertEqual(
            history_serialization_for_template(self.template),
            "qwen-leading-system-only",
        )

    def test_pinned_model_name_matches_the_audited_gguf_string(self) -> None:
        self.assertEqual(ORNITH15_VERIFIED_MODEL_NAME, "Ornith-1.5-35B")

    def test_real_template_has_no_leading_system_guard(self) -> None:
        # Qwen3.6 raises on a non-leading system message; Ornith does not. The
        # absence of this guard is precisely why the contract is mandatory.
        self.assertNotIn("System message must be at the beginning", self.template)

    def test_real_template_drops_trailing_system_and_developer_roles(self) -> None:
        self.assertIn(
            'message.role != "system" and message.role != "developer"',
            self.template,
        )


class Ornith15PinnedDigestTests(unittest.TestCase):
    """Exercises the real pinned digest, with no constant patched."""

    def test_pinned_template_digest_maps_to_the_leading_system_only_contract(self) -> None:
        # history_serialization_for_template takes template TEXT and hashes it.
        # The real 7828-character Ornith template is not vendored here, so the
        # digest map is asserted directly against the pinned constant: an
        # external backend seeing that template must get the same contract the
        # native profile declares.
        self.assertEqual(
            _VERIFIED_TEMPLATE_HISTORY_SERIALIZATION.get(ORNITH15_OFFICIAL_TEMPLATE_SHA256),
            "qwen-leading-system-only",
        )

    def test_lookup_hashes_template_text_and_honors_the_ornith_pin(self) -> None:
        from unittest import mock

        template = "ornith-official-template-body"
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()

        with mock.patch.dict(
            _VERIFIED_TEMPLATE_HISTORY_SERIALIZATION,
            {digest: "qwen-leading-system-only"},
        ):
            self.assertEqual(
                history_serialization_for_template(template),
                "qwen-leading-system-only",
            )
        self.assertIsNone(history_serialization_for_template("unreviewed-template"))

    def test_pinned_digest_is_distinct_from_every_other_verified_template(self) -> None:
        from orbit.native_llama.model_profiles import (
            QWEN36_OFFICIAL_TEMPLATE_SHA256,
            QWEN38_OFFICIAL_TEMPLATE_SHA256,
        )

        self.assertNotEqual(ORNITH15_OFFICIAL_TEMPLATE_SHA256, QWEN36_OFFICIAL_TEMPLATE_SHA256)
        self.assertNotEqual(ORNITH15_OFFICIAL_TEMPLATE_SHA256, QWEN38_OFFICIAL_TEMPLATE_SHA256)
        self.assertEqual(len(ORNITH15_OFFICIAL_TEMPLATE_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
