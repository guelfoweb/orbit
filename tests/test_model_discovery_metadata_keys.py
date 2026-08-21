"""Discovery must read every metadata key the pinned identities compare.

A key missing from the extraction allowlist is read as the empty string, so the
comparison fails and a correctly installed verified model is reported as
MISSING while the same file appears again as AVAILABLE / UNSUPPORTED. That is
what happened to Ornith 1.5: the allowlist existed in two copies and only one
gained the `qwen35moe.*` keys.

These exercise the extraction step rather than `detect_native_model_profile`
with a ready-made dict, because a preconstructed dict is exactly what bypasses
the defect.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from orbit.native_llama import model_discovery
from orbit.native_llama.model_profiles import (
    ORNITH15_PROFILE_ID,
    PROFILE_METADATA_KEYS,
    detect_native_model_profile,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ornith15_chat_template.jinja"

# The metadata a real Ornith GGUF reports, verified field-by-field against the
# installed file. Extra keys are deliberate: the extraction step must drop them
# and keep the pinned ones.
ORNITH_METADATA = {
    "general.architecture": "qwen35moe",
    "general.name": "Ornith-1.5-35B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "qwen35moe.context_length": "262144",
    "qwen35moe.block_count": "41",
    "qwen35moe.expert_count": "256",
    "qwen35moe.expert_used_count": "8",
    "general.license": "apache-2.0",          # not pinned, must be dropped
    "qwen35moe.attention.head_count": "16",   # not pinned, must be dropped
}


def _extract(metadata: dict[str, str]) -> dict[str, str]:
    """Run the production filtering step over a metadata mapping.

    Mirrors `_read_profile_metadata`'s enumerate-and-filter without a GGUF
    handle: the property under test is which keys survive the allowlist.
    """
    return {k: v for k, v in metadata.items() if k in model_discovery.PROFILE_METADATA_KEYS}


class OrnithDiscoveryMetadataTests(unittest.TestCase):
    def test_filtering_retains_every_pinned_qwen35moe_key(self) -> None:
        filtered = _extract(ORNITH_METADATA)
        for key in (
            "qwen35moe.context_length",
            "qwen35moe.block_count",
            "qwen35moe.expert_count",
            "qwen35moe.expert_used_count",
        ):
            self.assertIn(key, filtered, f"{key} was filtered out of discovery metadata")
            self.assertTrue(filtered[key], f"{key} survived but is empty")

    def test_filtering_drops_unpinned_keys(self) -> None:
        filtered = _extract(ORNITH_METADATA)
        self.assertNotIn("general.license", filtered)
        self.assertNotIn("qwen35moe.attention.head_count", filtered)

    def test_filtered_metadata_still_verifies_as_ornith(self) -> None:
        """The end-to-end property: filter, then identify, and stay verified."""
        template = FIXTURE.read_text(encoding="utf-8")
        profile = detect_native_model_profile(_extract(ORNITH_METADATA), template)
        self.assertEqual(profile.profile_id, ORNITH15_PROFILE_ID)
        self.assertTrue(profile.verified, profile.failure_reason)
        self.assertIsNone(profile.failure_reason)

    def test_unfiltered_and_filtered_agree(self) -> None:
        """Filtering must not change the verdict, only the key set."""
        template = FIXTURE.read_text(encoding="utf-8")
        direct = detect_native_model_profile(ORNITH_METADATA, template)
        through_filter = detect_native_model_profile(_extract(ORNITH_METADATA), template)
        self.assertEqual(direct.profile_id, through_filter.profile_id)
        self.assertEqual(direct.verified, through_filter.verified)


class AllowlistContractTests(unittest.TestCase):
    """Every key any pinned identity compares must be extractable."""

    def _keys_compared_by_identities(self) -> set[str]:
        source = Path(model_discovery.__file__).with_name("model_profiles.py").read_text()
        tree = ast.parse(source)
        keys: set[str] = set()
        for node in ast.walk(tree):
            # metadata.get("some.key", ...)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "metadata"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
        return keys

    def test_every_compared_key_is_extractable(self) -> None:
        compared = self._keys_compared_by_identities()
        self.assertTrue(compared, "no metadata.get() keys found; parser is broken")
        missing = sorted(compared - set(PROFILE_METADATA_KEYS))
        self.assertEqual(
            missing, [], f"identities compare keys discovery cannot read: {missing}"
        )

    def test_discovery_and_profiles_share_one_source(self) -> None:
        """Two copies is how the keys drifted apart in the first place."""
        self.assertIs(model_discovery.PROFILE_METADATA_KEYS, PROFILE_METADATA_KEYS)

    def test_client_shares_the_same_source(self) -> None:
        from orbit.native_llama import client

        self.assertIs(client.PROFILE_METADATA_KEYS, PROFILE_METADATA_KEYS)


if __name__ == "__main__":
    unittest.main()
