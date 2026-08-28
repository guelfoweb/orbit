"""Exact-artifact capability identity.

The property under test is that capability follows CONTENT, never the path.
Two Ornith builds share their profile, model name, architecture and chat
template; only their bytes differ. Anything keyed on a filename would qualify
the wrong one.

Small fixtures with injected digests do the work. One integration test hashes a
real file end to end so the wiring is not merely mocked.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import artifact_capabilities as ac  # noqa: E402
from orbit.native_llama.model_profiles import (  # noqa: E402
    ORNITH15_CURRENT_ARTIFACT_SHA256,
    artifact_capability_map,
    ORNITH15_PROFILE_ID,
    SELF_MTP_CAPABILITY,
    VERIFIED_NATIVE_MODEL_IDENTITIES,
    VerifiedNativeModelIdentity,
    verified_native_model_identity,
)

LEGACY_SHA = "ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f"


def _profile(profile_id: str = ORNITH15_PROFILE_ID, verified: bool = True):
    """The fields `verified_artifact_capabilities` actually reads."""
    return mock.Mock(profile_id=profile_id, verified=verified)


class SchemaTests(unittest.TestCase):
    def test_only_ornith_declares_an_artifact_capability(self) -> None:
        declaring = [
            i.profile_id for i in VERIFIED_NATIVE_MODEL_IDENTITIES if i.artifact_capabilities
        ]
        self.assertEqual(declaring, [ORNITH15_PROFILE_ID])

    def test_current_sha_maps_to_self_mtp(self) -> None:
        identity = verified_native_model_identity(ORNITH15_PROFILE_ID)
        self.assertEqual(
            artifact_capability_map(identity)[ORNITH15_CURRENT_ARTIFACT_SHA256],
            frozenset({SELF_MTP_CAPABILITY}),
        )

    def test_legacy_sha_is_absent_from_the_table(self) -> None:
        identity = verified_native_model_identity(ORNITH15_PROFILE_ID)
        self.assertNotIn(LEGACY_SHA, artifact_capability_map(identity))

    def test_exactly_one_digest_is_qualified(self) -> None:
        """The table is an allowlist of one, not "the current one plus others".

        Asserting only that a couple of known-bad digests are absent would let
        any additional entry slip in. What matters is the size of the set.
        """
        identity = verified_native_model_identity(ORNITH15_PROFILE_ID)
        self.assertEqual(
            list(artifact_capability_map(identity)), [ORNITH15_CURRENT_ARTIFACT_SHA256]
        )

    def test_no_profile_qualifies_more_than_the_digests_it_declares(self) -> None:
        for identity in VERIFIED_NATIVE_MODEL_IDENTITIES:
            with self.subTest(identity.profile_id):
                for digest in artifact_capability_map(identity):
                    self.assertEqual(len(digest), 64, "keys must be sha256 hex")
                    self.assertEqual(digest, digest.lower())

    def test_capability_sets_are_immutable(self) -> None:
        identity = verified_native_model_identity(ORNITH15_PROFILE_ID)
        with self.assertRaises(AttributeError):
            identity.artifact_capabilities.add(("x", frozenset({"y"})))
        with self.assertRaises(TypeError):
            artifact_capability_map(identity)["x"] = frozenset({"y"})

    def test_identities_without_capabilities_default_to_empty(self) -> None:
        blank = VerifiedNativeModelIdentity("p", "m", "a")
        self.assertEqual(dict(artifact_capability_map(blank)), {})


class EligibilityTests(unittest.TestCase):
    """Digest-driven verdicts, with the hash injected so no 20 GiB is read."""

    def _supports(self, digest: str, profile=None) -> bool:
        with mock.patch.object(ac, "artifact_sha256", return_value=digest):
            return ac.verified_artifact_supports(
                Path("/nonexistent.gguf"),
                SELF_MTP_CAPABILITY,
                profile if profile is not None else _profile(),
            )

    def test_current_artifact_is_eligible(self) -> None:
        self.assertTrue(self._supports(ORNITH15_CURRENT_ARTIFACT_SHA256))

    def test_legacy_artifact_is_not_eligible(self) -> None:
        self.assertFalse(self._supports(LEGACY_SHA))

    def test_unknown_artifact_is_not_eligible(self) -> None:
        self.assertFalse(self._supports("00" * 32))

    def test_unverified_profile_is_not_eligible(self) -> None:
        self.assertFalse(
            self._supports(ORNITH15_CURRENT_ARTIFACT_SHA256, _profile(verified=False))
        )

    def test_absent_profile_is_not_eligible(self) -> None:
        self.assertFalse(
            ac.verified_artifact_supports(
                Path("/nonexistent.gguf"), SELF_MTP_CAPABILITY, None
            )
        )

    def test_other_profile_is_not_eligible_even_with_the_current_digest(self) -> None:
        """The digest alone must not qualify an artifact under another profile."""
        self.assertFalse(
            self._supports(
                ORNITH15_CURRENT_ARTIFACT_SHA256, _profile("orbit-qwen38-native-v1")
            )
        )

    def test_digest_lookup_is_case_sensitive(self) -> None:
        """hexdigest() emits lowercase; nothing may normalise around that.

        Latent rather than live, but a lookup that lowercases (or uppercases)
        its key widens the allowlist by exactly the set of digests that differ
        only in case -- a silent relaxation of an exact-match guarantee.
        """
        self.assertFalse(self._supports(ORNITH15_CURRENT_ARTIFACT_SHA256.upper()))

    def test_non_oserror_failures_are_not_swallowed(self) -> None:
        """The narrow `except OSError` is deliberate: a bug must surface.

        Widening it to `except Exception` would turn a programming error into
        a silent "not qualified", which reads as a policy decision rather than
        the defect it is.
        """
        with mock.patch.object(ac, "artifact_sha256", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                ac.verified_artifact_supports(
                    Path("/x.gguf"), SELF_MTP_CAPABILITY, _profile()
                )

    def test_unknown_capability_name_fails_closed(self) -> None:
        with mock.patch.object(
            ac, "artifact_sha256", return_value=ORNITH15_CURRENT_ARTIFACT_SHA256
        ):
            self.assertFalse(
                ac.verified_artifact_supports(
                    Path("/x.gguf"), "teleportation", _profile()
                )
            )


class ContentNotPathTests(unittest.TestCase):
    """Section 4: capability follows the bytes, under any name or link."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="artifact-caps-"))
        self.addCleanup(self._cleanup)
        self.payload = b"pretend GGUF bytes\n"
        self.real = self.tmp / "Ornith-1.5-35B-Q4_K_M.gguf"
        self.real.write_bytes(self.payload)
        self.digest = ac.artifact_sha256(self.real)
        # Qualify these exact bytes under the Ornith profile for the test.
        identity = verified_native_model_identity(ORNITH15_PROFILE_ID)
        self.patched = VerifiedNativeModelIdentity(
            identity.profile_id,
            identity.model_name,
            identity.architecture,
            frozenset({(self.digest, frozenset({SELF_MTP_CAPABILITY}))}),
        )

    def _cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _supports(self, path: Path) -> bool:
        with mock.patch.object(
            ac, "verified_native_model_identity", return_value=self.patched
        ):
            return ac.verified_artifact_supports(path, SELF_MTP_CAPABILITY, _profile())

    def test_original_path_qualifies(self) -> None:
        self.assertTrue(self._supports(self.real))

    def test_renamed_file_still_qualifies(self) -> None:
        renamed = self.tmp / "totally-different-name.bin"
        renamed.write_bytes(self.payload)
        self.assertTrue(self._supports(renamed))

    def test_symlink_to_the_artifact_qualifies(self) -> None:
        link = self.tmp / "link.gguf"
        link.symlink_to(self.real)
        self.assertTrue(self._supports(link))

    def test_same_name_wrong_bytes_does_not_qualify(self) -> None:
        """The impostor case: correct filename, different content."""
        other = self.tmp / "impostor"
        other.mkdir()
        impostor = other / "Ornith-1.5-35B-Q4_K_M.gguf"
        impostor.write_bytes(self.payload + b"tampered")
        self.assertEqual(impostor.name, self.real.name)
        self.assertFalse(self._supports(impostor))

    def test_one_flipped_byte_does_not_qualify(self) -> None:
        near = self.tmp / "near.gguf"
        near.write_bytes(self.payload[:-1] + b"X")
        self.assertFalse(self._supports(near))

    def test_missing_file_fails_closed(self) -> None:
        self.assertFalse(self._supports(self.tmp / "absent.gguf"))

    def test_directory_fails_closed(self) -> None:
        self.assertFalse(self._supports(self.tmp))

    def test_unreadable_file_fails_closed(self) -> None:
        locked = self.tmp / "locked.gguf"
        locked.write_bytes(self.payload)
        locked.chmod(0o000)
        self.addCleanup(locked.chmod, 0o644)
        if os.geteuid() == 0:
            self.skipTest("root ignores file permissions")
        self.assertFalse(self._supports(locked))


class WholeFileIsHashedTests(unittest.TestCase):
    """The digest must cover every byte, not just the first chunk.

    Every other fixture here is a few dozen bytes -- under `_CHUNK` -- so a
    truncated read would hash them entirely and no test would notice. That is
    exactly the bug this module exists to prevent: the CURRENT and LEGACY
    Ornith artifacts share ~20 GiB of identical prefix and differ only in
    trailing blk.40 weights, so a first-chunk-only digest would hand LEGACY
    the CURRENT verdict.
    """

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="whole-file-hash-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # Shared prefix strictly larger than one read chunk, then a divergent
        # tail -- the CURRENT/LEGACY shape in miniature.
        prefix = b"\xa5" * (ac._CHUNK + 4096)
        self.a = self.tmp / "a.gguf"
        self.b = self.tmp / "b.gguf"
        self.a.write_bytes(prefix + b"TAIL-A")
        self.b.write_bytes(prefix + b"TAIL-B")

    def test_files_sharing_a_chunk_sized_prefix_hash_differently(self) -> None:
        self.assertGreater(self.a.stat().st_size, ac._CHUNK)
        self.assertEqual(
            self.a.read_bytes()[: ac._CHUNK], self.b.read_bytes()[: ac._CHUNK]
        )
        self.assertNotEqual(ac.artifact_sha256(self.a), ac.artifact_sha256(self.b))

    def test_a_tail_only_difference_changes_the_verdict(self) -> None:
        """End to end: the impostor must not inherit the qualified verdict."""
        identity = VerifiedNativeModelIdentity(
            ORNITH15_PROFILE_ID,
            "Ornith-1.5-35B",
            "qwen35moe",
            frozenset({(ac.artifact_sha256(self.a), frozenset({SELF_MTP_CAPABILITY}))}),
        )
        with mock.patch.object(
            ac, "verified_native_model_identity", return_value=identity
        ):
            self.assertTrue(
                ac.verified_artifact_supports(self.a, SELF_MTP_CAPABILITY, _profile())
            )
            self.assertFalse(
                ac.verified_artifact_supports(self.b, SELF_MTP_CAPABILITY, _profile())
            )

    def test_digest_matches_a_whole_file_reference_hash(self) -> None:
        import hashlib

        self.assertEqual(
            ac.artifact_sha256(self.a),
            hashlib.sha256(self.a.read_bytes()).hexdigest(),
        )


class IdentityRecordTests(unittest.TestCase):
    """The identity record must stay hashable and frozen."""

    def test_identities_remain_hashable(self) -> None:
        for identity in VERIFIED_NATIVE_MODEL_IDENTITIES:
            with self.subTest(identity.profile_id):
                self.assertIsInstance(hash(identity), int)
        self.assertEqual(len(set(VERIFIED_NATIVE_MODEL_IDENTITIES)),
                         len(VERIFIED_NATIVE_MODEL_IDENTITIES))

    def test_identities_are_usable_as_dict_keys(self) -> None:
        mapping = {i: i.profile_id for i in VERIFIED_NATIVE_MODEL_IDENTITIES}
        self.assertEqual(len(mapping), len(VERIFIED_NATIVE_MODEL_IDENTITIES))

    def test_identities_differing_only_in_capabilities_are_distinct(self) -> None:
        """Capabilities must participate in equality, not just live alongside it.

        Excluding them (compare=False) restores hashability but collapses
        equality, and a legacy-qualified record then silently overwrites a
        current-qualified one used as a dict key -- the substitution this
        module exists to prevent, moved from the digest lookup into record
        identity.
        """
        current = VerifiedNativeModelIdentity(
            "p", "m", "a", frozenset({("aa" * 32, frozenset({SELF_MTP_CAPABILITY}))})
        )
        legacy = VerifiedNativeModelIdentity(
            "p", "m", "a", frozenset({("bb" * 32, frozenset({SELF_MTP_CAPABILITY}))})
        )
        none = VerifiedNativeModelIdentity("p", "m", "a")
        self.assertNotEqual(current, legacy)
        self.assertNotEqual(current, none)
        self.assertEqual(len({current, legacy, none}), 3)
        keyed = {current: "current", legacy: "legacy"}
        self.assertEqual(keyed[current], "current", "legacy overwrote current")

    def test_value_equality_holds_for_identical_records(self) -> None:
        """`eq=False` would silently switch this to identity comparison."""
        caps = frozenset({("aa" * 32, frozenset({SELF_MTP_CAPABILITY}))})
        self.assertEqual(
            VerifiedNativeModelIdentity("p", "m", "a", caps),
            VerifiedNativeModelIdentity("p", "m", "a", caps),
        )
        self.assertEqual(
            VerifiedNativeModelIdentity("p", "m", "a"),
            VerifiedNativeModelIdentity("p", "m", "a"),
        )

    def test_identity_is_frozen(self) -> None:
        identity = verified_native_model_identity(ORNITH15_PROFILE_ID)
        with self.assertRaises(Exception):
            identity.profile_id = "mutated"


class DiscoveryDoesNotHashTests(unittest.TestCase):
    """Normal discovery must not acquire a 20 GiB hashing cost."""

    def test_model_discovery_module_never_hashes_artifacts(self) -> None:
        from orbit.native_llama import model_discovery

        source = Path(model_discovery.__file__).read_text()
        self.assertNotIn("artifact_sha256", source)
        self.assertNotIn("artifact_capabilities", source)

    def test_model_profiles_module_does_not_hash_artifacts(self) -> None:
        from orbit.native_llama import model_profiles

        source = Path(model_profiles.__file__).read_text()
        self.assertNotIn("artifact_sha256(", source)

    def test_capability_query_short_circuits_before_hashing(self) -> None:
        """A profile with no declared artifacts never reaches the hash."""
        with mock.patch.object(ac, "artifact_sha256") as hashed:
            ac.verified_artifact_supports(
                Path("/x.gguf"), SELF_MTP_CAPABILITY, _profile("orbit-qwen38-native-v1")
            )
            hashed.assert_not_called()

    def test_capability_query_does_hash_when_it_must(self) -> None:
        with mock.patch.object(ac, "artifact_sha256", return_value="00" * 32) as hashed:
            ac.verified_artifact_supports(Path("/x.gguf"), SELF_MTP_CAPABILITY, _profile())
            hashed.assert_called_once()


class ExternalDraftUntouchedTests(unittest.TestCase):
    """Existing external-draft MTP semantics must be unchanged."""

    def test_ornith_mtp_supported_is_false_without_any_fixture(self) -> None:
        """Machine-independent: no checkpoint, no captured JSON, no skip.

        The JSON-backed test below is stronger but skips where the checkpoint
        is absent -- and with it skipped, flipping this flag left the whole
        suite green. The external-draft guarantee needs an anchor that cannot
        skip.
        """
        import inspect

        from orbit.native_llama import model_profiles

        source = inspect.getsource(model_profiles)
        start = source.index("if ornith15_identity:")
        end = source.index("return NativeModelProfile", start)
        end = source.index(")", source.index("template_source", end))
        block = source[start:end]
        self.assertIn("mtp_supported=False", block)
        self.assertNotIn("mtp_supported=True", block)

    def test_ornith_profile_mtp_supported_stays_false(self) -> None:
        """`mtp_supported` gates external-draft MTP and must not absorb self_mtp.

        Asserted against the shipped profile table rather than a hand-built
        metadata dict: Ornith detection requires twelve exact fields plus the
        official template hash, and a fixture that drifts from those would
        silently stop testing Ornith at all.
        """
        import json

        from orbit.native_llama.model_profiles import detect_native_model_profile

        captured = Path(
            "/home/guelfoweb/LAB/orbit-checkpoints/"
            "ornith-new-mtp-qualification-20260828/gguf_metadata_old_vs_new.json"
        )
        if not captured.is_file():
            self.skipTest("captured Ornith GGUF metadata not present")
        meta = json.loads(captured.read_text())["new"]["meta"]
        profile = detect_native_model_profile(meta, meta["tokenizer.chat_template"])
        self.assertEqual(profile.profile_id, ORNITH15_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertFalse(profile.mtp_supported)

    def test_registry_mtp_block_semantics_unchanged(self) -> None:
        from orbit.native_llama.model_registry import load_registry

        ornith = [m for m in load_registry() if "ornith" in m.id][0]
        self.assertIsNone(ornith.mtp)


class RealFileIntegrationTests(unittest.TestCase):
    """One end-to-end hash, so the wiring is not only mocked."""

    def test_real_file_digest_drives_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.gguf"
            path.write_bytes(b"orbit self-mtp capability integration fixture")
            digest = ac.artifact_sha256(path)
            identity = VerifiedNativeModelIdentity(
                ORNITH15_PROFILE_ID,
                "Ornith-1.5-35B",
                "qwen35moe",
                frozenset({(digest, frozenset({SELF_MTP_CAPABILITY}))}),
            )
            with mock.patch.object(
                ac, "verified_native_model_identity", return_value=identity
            ):
                self.assertTrue(
                    ac.verified_artifact_supports(path, SELF_MTP_CAPABILITY, _profile())
                )
                path.write_bytes(b"different bytes entirely")
                self.assertFalse(
                    ac.verified_artifact_supports(path, SELF_MTP_CAPABILITY, _profile())
                )


if __name__ == "__main__":
    unittest.main()
