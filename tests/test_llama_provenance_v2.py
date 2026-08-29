"""Provenance V2: a locally reproducible attestation of the Orbit patchset.

The legacy `patchset_sha256` came from a packaging pipeline that no longer exists
here; 121 candidate algorithms failed to reproduce either of its two historical
values. It is preserved as an imported attestation and must never be recomputed
or reinterpreted. V2 is additive and locally reproducible.

These tests exercise the real generator against real trees and real mutations,
not source text.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import llama_provenance_v2 as pv2  # noqa: E402

UPSTREAM = Path.home() / "LAB/llama.cpp"
VENDORED = ROOT / "src/orbit/native_llama/vendor/source/llama.cpp"
MANIFEST = ROOT / "src/orbit/native_llama/vendor/LLAMA_PROVENANCE.json"
COMMIT = json.loads(MANIFEST.read_text())["upstream_commit"]


def _available() -> bool:
    return (UPSTREAM / ".git").exists() and VENDORED.is_dir()


@unittest.skipUnless(_available(), "upstream llama.cpp checkout not present")
class PatchsetV2Tests(unittest.TestCase):
    def _entries(self):
        return pv2.build_patchset(UPSTREAM, VENDORED, COMMIT)

    def test_generation_is_deterministic(self) -> None:
        a = pv2.patchset_v2_sha256(self._entries())
        b = pv2.patchset_v2_sha256(self._entries())
        self.assertEqual(a, b)

    def test_input_set_equals_declared_patched_paths(self) -> None:
        """The derived set must be exactly what the manifest declares.

        Proven in the recovery mission at both historical commits (60/60, 62/62).
        """
        declared = sorted(json.loads(MANIFEST.read_text())["patched_paths"])
        computed = sorted(e["path"] for e in self._entries())
        self.assertEqual(computed, declared)

    def test_entries_are_sorted_by_path(self) -> None:
        paths = [e["path"] for e in self._entries()]
        self.assertEqual(paths, sorted(paths))

    def test_canonical_encoding_is_compact_and_unterminated(self) -> None:
        blob = pv2.canonical_bytes(self._entries())
        self.assertNotIn(b", ", blob)
        self.assertNotIn(b'": ', blob)
        self.assertFalse(blob.endswith(b"\n"))
        self.assertEqual(blob[:1], b"[")

    def test_empty_patchset_matches_the_existing_sentinel(self) -> None:
        from orbit.native_llama.llama_provenance import EMPTY_PATCHSET_SHA256

        self.assertEqual(pv2.patchset_v2_sha256([]), EMPTY_PATCHSET_SHA256)

    def test_every_entry_binds_both_sides(self) -> None:
        for e in self._entries():
            self.assertIsInstance(e["vendored_sha256"], str)
            self.assertEqual(len(e["vendored_sha256"]), 64)
            self.assertTrue(e["upstream_sha256"] is None or len(e["upstream_sha256"]) == 64)

    def test_hash_changes_when_a_patched_file_changes_and_reverts(self) -> None:
        """Content sensitivity, then exact restoration.

        Restoration is registered as a cleanup rather than a `finally`, so the
        vendored tree is repaired even if the process is interrupted mid-test.
        A leftover probe here poisons every later run, including the baseline of
        a mutation campaign.
        """
        entries = self._entries()
        target = VENDORED / entries[0]["path"]
        original = target.read_bytes()
        before = pv2.patchset_v2_sha256(entries)
        self.addCleanup(target.write_bytes, original)
        target.write_bytes(original + b"\n// provenance v2 probe\n")
        after = pv2.patchset_v2_sha256(self._entries())
        self.assertNotEqual(after, before)
        target.write_bytes(original)
        self.assertEqual(pv2.patchset_v2_sha256(self._entries()), before)

    def test_a_file_restored_to_upstream_leaves_the_patchset(self) -> None:
        """Input set is derived from divergence, not from the declared list."""
        entries = self._entries()
        rel = entries[0]["path"]
        target = VENDORED / rel
        original = target.read_bytes()
        upstream = pv2._upstream_blob(UPSTREAM, COMMIT, rel)
        self.assertIsNotNone(upstream)
        self.addCleanup(target.write_bytes, original)
        target.write_bytes(upstream)
        paths = [e["path"] for e in self._entries()]
        self.assertNotIn(rel, paths)
        target.write_bytes(original)
        self.assertIn(rel, [e["path"] for e in self._entries()])

    def test_orbit_only_file_is_represented_with_null_upstream(self) -> None:
        added = VENDORED / "orbit_provenance_v2_probe.txt"
        self.assertFalse(added.exists())
        self.addCleanup(added.unlink, True)
        added.write_bytes(b"orbit-only probe\n")
        entry = next(e for e in self._entries() if e["path"] == added.name)
        self.assertIsNone(entry["upstream_sha256"])
        self.assertEqual(len(entry["vendored_sha256"]), 64)
        added.unlink()

    def test_binary_content_is_hashed_byte_for_byte(self) -> None:
        """No newline normalization: CRLF and LF must differ."""
        a = [{"path": "x", "upstream_sha256": None, "vendored_sha256": pv2._sha256_bytes(b"a\r\nb")}]
        b = [{"path": "x", "upstream_sha256": None, "vendored_sha256": pv2._sha256_bytes(b"a\nb")}]
        self.assertNotEqual(pv2.patchset_v2_sha256(a), pv2.patchset_v2_sha256(b))

    def test_path_rename_changes_the_hash(self) -> None:
        entries = self._entries()
        renamed = [dict(e) for e in entries]
        renamed[0]["path"] = renamed[0]["path"] + ".renamed"
        renamed.sort(key=lambda e: e["path"])
        self.assertNotEqual(pv2.patchset_v2_sha256(renamed), pv2.patchset_v2_sha256(entries))

    def test_upstream_side_participates_in_the_hash(self) -> None:
        entries = self._entries()
        tweaked = [dict(e) for e in entries]
        tweaked[0]["upstream_sha256"] = "0" * 64
        self.assertNotEqual(pv2.patchset_v2_sha256(tweaked), pv2.patchset_v2_sha256(entries))

    def test_declared_order_does_not_matter(self) -> None:
        entries = self._entries()
        shuffled = list(reversed(entries))
        self.assertEqual(
            pv2.patchset_v2_sha256(sorted(shuffled, key=lambda e: e["path"])),
            pv2.patchset_v2_sha256(entries),
        )

    def test_entries_are_sorted_independently_of_walk_order(self) -> None:
        """The explicit sort must not rely on `_vendored_files` already sorting.

        Mutation P1 (removing `entries.sort`) survived because the walker returns
        sorted paths, making the sort a no-op TODAY. It is still load-bearing:
        it pins ordering independently of the walker. This test removes the
        coupling by feeding an unsorted walk.
        """
        real = pv2._vendored_files

        def shuffled(root):
            return list(reversed(real(root)))

        with unittest.mock.patch.object(pv2, "_vendored_files", shuffled):
            entries = pv2.build_patchset(UPSTREAM, VENDORED, COMMIT)
        paths = [e["path"] for e in entries]
        self.assertEqual(paths, sorted(paths), "build_patchset must sort its own output")
        self.assertEqual(
            pv2.patchset_v2_sha256(entries),
            pv2.patchset_v2_sha256(pv2.build_patchset(UPSTREAM, VENDORED, COMMIT)),
            "walk order must not change the attestation",
        )

    def test_crlf_only_change_is_detected(self) -> None:
        """Line-ending normalization must never be introduced.

        Mutation P9 (normalizing CRLF->LF on read) survived only because no
        ATTESTABLE vendored file currently contains CRLF -- the 11 that do are
        all __pycache__/.pyc, which the filter excludes. A future vendored file
        with CRLF would then let a line-ending-only edit produce an identical
        hash, so an undeclared modification would pass the gate.

        This drives the real reader over a genuine CRLF file.
        """
        probe = VENDORED / "orbit_v2_crlf_probe.txt"
        self.assertFalse(probe.exists())
        self.addCleanup(probe.unlink, True)

        probe.write_bytes(b"line one\r\nline two\r\n")
        crlf = next(e for e in pv2.build_patchset(UPSTREAM, VENDORED, COMMIT)
                    if e["path"] == probe.name)["vendored_sha256"]

        probe.write_bytes(b"line one\nline two\n")
        lf = next(e for e in pv2.build_patchset(UPSTREAM, VENDORED, COMMIT)
                  if e["path"] == probe.name)["vendored_sha256"]

        probe.unlink()
        self.assertNotEqual(crlf, lf, "CRLF and LF content must hash differently")

    def test_excluded_components_match_source_tree_filter(self) -> None:
        self.assertEqual(pv2.EXCLUDED_PARTS, {".git", "build", "__pycache__"})
        self.assertEqual(pv2.EXCLUDED_SUFFIXES, {".pyc", ".pyo"})

    def test_missing_vendored_root_fails_closed(self) -> None:
        rc = pv2.main(["check", "--vendored-root", "/nonexistent/vendored"])
        self.assertEqual(rc, 2)

    def test_missing_upstream_checkout_fails_closed(self) -> None:
        rc = pv2.main(["check", "--upstream-root", "/nonexistent/upstream"])
        self.assertEqual(rc, 2)

    def test_check_fails_on_a_manifest_without_v2_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "m.json"
            payload = json.loads(MANIFEST.read_text())
            payload.pop("patchset_algorithm", None)
            payload.pop("patchset_v2_sha256", None)
            copy.write_text(json.dumps(payload))
            rc = pv2.main(["check", "--manifest", str(copy)])
            self.assertEqual(rc, 1)

    def test_generate_then_check_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "m.json"
            shutil.copy2(MANIFEST, copy)
            self.assertEqual(pv2.main(["generate", "--manifest", str(copy)]), 0)
            self.assertEqual(pv2.main(["check", "--manifest", str(copy)]), 0)
            written = json.loads(copy.read_text())
            self.assertEqual(written["patchset_algorithm"], "orbit-patchset-v2")
            self.assertEqual(len(written["patchset_v2_sha256"]), 64)

    def test_legacy_fields_are_never_recomputed_or_removed(self) -> None:
        """V2 must not touch the imported attestation."""
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "m.json"
            shutil.copy2(MANIFEST, copy)
            before = json.loads(copy.read_text())
            pv2.main(["generate", "--manifest", str(copy)])
            after = json.loads(copy.read_text())
            for key in ("patchset_sha256", "source_tree_sha256",
                        "omitted_upstream_paths_sha256", "omitted_upstream_path_count",
                        "upstream_commit", "upstream_tag", "format"):
                self.assertEqual(after[key], before[key], f"V2 modified legacy field {key}")

    def test_v2_digest_differs_from_the_legacy_digest(self) -> None:
        """Guards against silently aliasing the two attestations."""
        legacy = json.loads(MANIFEST.read_text())["patchset_sha256"]
        self.assertNotEqual(pv2.patchset_v2_sha256(self._entries()), legacy)

    def test_dirty_upstream_worktree_cannot_affect_the_attestation(self) -> None:
        """BEHAVIOURAL: dirty an attested upstream file, digest must not move.

        This previously asserted on `inspect.getsource` text, which would keep
        passing through a refactor that kept the substring but read the working
        tree anyway. It now proves the property instead of describing it.
        """
        entries = pv2.build_patchset(UPSTREAM, VENDORED, COMMIT)
        before = pv2.patchset_v2_sha256(entries)

        target = UPSTREAM / entries[0]["path"]
        if not target.is_file():
            self.skipTest("attested path not present in the upstream worktree")
        original = target.read_bytes()
        self.addCleanup(target.write_bytes, original)

        target.write_bytes(original + b"\n// upstream worktree drift probe\n")
        after = pv2.patchset_v2_sha256(pv2.build_patchset(UPSTREAM, VENDORED, COMMIT))
        target.write_bytes(original)

        self.assertEqual(after, before,
                         "upstream working-tree edits must not affect the attestation")

    def test_duplicate_declared_paths_fail_the_gate(self) -> None:
        """A malformed declared list must not be blessed by `check`."""
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "m.json"
            payload = json.loads(MANIFEST.read_text())
            payload["patched_paths"] = payload["patched_paths"] + [payload["patched_paths"][0]]
            copy.write_text(json.dumps(payload))
            self.assertEqual(pv2.main(["check", "--manifest", str(copy)]), 1)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_available(), "upstream llama.cpp checkout not present")
class PatchsetV2EnforcementTests(unittest.TestCase):
    """V2 must be a GATE, not a note in a file.

    The legacy `source_tree_sha256` is already enforced: `load_llama_provenance`
    raises on drift and `test_vendored_provenance_matches_current_source_tree`
    fails. V2 needs the same automatic property, so vendored drift cannot reach a
    green suite merely because someone forgot to run a command by hand.

    Every check here drives the REAL verifier (`pv2.main(["check", ...])`), never
    a reimplementation of its algorithm.
    """

    def _check(self, manifest: Path | None = None) -> int:
        argv = ["check"]
        if manifest is not None:
            argv += ["--manifest", str(manifest)]
        return pv2.main(argv)

    def test_current_tree_passes_the_gate(self) -> None:
        self.assertEqual(self._check(), 0)

    def test_vendored_drift_fails_the_gate_and_restores(self) -> None:
        """The decisive property: edit a patched file, do NOT regenerate."""
        entries = pv2.build_patchset(UPSTREAM, VENDORED, COMMIT)
        target = VENDORED / entries[0]["path"]
        original = target.read_bytes()
        self.addCleanup(target.write_bytes, original)

        target.write_bytes(original + b"\n// undeclared drift\n")
        self.assertEqual(self._check(), 1, "vendored drift must fail the gate")

        target.write_bytes(original)
        self.assertEqual(self._check(), 0, "restoring the file must clear the gate")

    def test_a_new_undeclared_patched_file_fails_the_gate(self) -> None:
        added = VENDORED / "orbit_v2_enforcement_probe.txt"
        self.assertFalse(added.exists())
        self.addCleanup(added.unlink, True)
        added.write_bytes(b"undeclared orbit-only file\n")
        self.assertEqual(self._check(), 1)
        added.unlink()
        self.assertEqual(self._check(), 0)

    def test_tampered_v2_hash_fails_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "m.json"
            payload = json.loads(MANIFEST.read_text())
            payload["patchset_v2_sha256"] = "0" * 64
            copy.write_text(json.dumps(payload))
            self.assertEqual(self._check(copy), 1)

    def test_tampered_patched_path_membership_fails_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "m.json"
            payload = json.loads(MANIFEST.read_text())
            payload["patched_paths"] = payload["patched_paths"][:-1]
            copy.write_text(json.dumps(payload))
            self.assertEqual(self._check(copy), 1)

    def test_wrong_algorithm_name_fails_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "m.json"
            payload = json.loads(MANIFEST.read_text())
            payload["patchset_algorithm"] = "orbit-patchset-v1"
            copy.write_text(json.dumps(payload))
            self.assertEqual(self._check(copy), 1)

    def test_check_never_mutates_the_manifest(self) -> None:
        before = MANIFEST.read_bytes()
        self._check()
        self.assertEqual(MANIFEST.read_bytes(), before)

    def test_legacy_attestation_is_untouched_by_the_gate(self) -> None:
        payload = json.loads(MANIFEST.read_text())
        self.assertEqual(len(payload["patchset_sha256"]), 64)
        self.assertNotEqual(payload["patchset_sha256"], payload.get("patchset_v2_sha256"))
