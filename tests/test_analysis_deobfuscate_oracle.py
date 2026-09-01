"""The deterministic pass against the real artifact, with no model involved.

Qualification only. The sample path and the expected digests live here, in the
test, because that is where knowledge of a particular artifact belongs -- the
production module must be able to decode this file without ever having heard
of it, and `GenericityTests` enforces that separately.

Digests rather than plaintext for the recovered stages: this file states what
the transformation must produce without reproducing a live indicator in the
repository. The one exception is the scheme-and-shape assertion on the final
stage, which checks a URI was recovered without writing the address out.

Skipped when the sample is absent, so a checkout without it still qualifies.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from orbit.runtime.analysis_deobfuscate import (
    JSCRIPT_XOR,
    POWERSHELL_XOR,
    deobfuscate,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "workdir" / "samples" / "Fattura981033956.js"
ORIGIN = ROOT / "workdir" / "samples" / "Fattura981033956_origin.js"

SAMPLE_SHA = "b7cfd5fdeb16d7b5ecea1063419bdad6ad280ed9b73c636707874c3f4001dc0c"
ORIGIN_SHA = "924fb89c69486151861e6eb82788718804aeee26f0989bb7c17356b15da622f4"

# Output digests of the five stages the artifact determines: three WMI
# monikers/classes, the command stage, and the address the command builds.
EXPECTED = {
    "S1": "6a892b624a2a237ca8b6fdd64d2dc02a30aa1afc08db9b9a21f269d9fb67eac2",
    "S2": "cafa14b8acab599aeb5425fabb3bdb519935cd39947e653dd89619db19b1b3fe",
    "S3": "8716f5d4559ed0f5dbfdb7bfb1717abcc9cfa1171caeea548aaf9c38fccd04db",
    "S4": "ec8ccda0cbdce79a76748c0e32c1fb788276c762abc5fd8c6f77609a0c8f58f1",
    "S5": "6a4277aa4ae872f43b368c35fcee79ea2ee40824822ae215cf52483f8faa48a3",
}
S4_CHARS = 1008


def _load(path: Path, expected_sha: str) -> str:
    if not path.exists():
        raise unittest.SkipTest(f"{path.name} not present")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise unittest.SkipTest(f"{path.name} is not the pinned artifact")
    return data.decode("utf-8", "replace")


class RealSampleOracleTests(unittest.TestCase):
    def test_every_stage_is_recovered_from_the_pinned_sample(self) -> None:
        stages = deobfuscate(_load(SAMPLE, SAMPLE_SHA))
        digests = {s.output_sha256 for s in stages}
        for name, expected in EXPECTED.items():
            with self.subTest(stage=name):
                self.assertIn(expected, digests)

    def test_the_stage_shape_is_what_the_artifact_determines(self) -> None:
        stages = deobfuscate(_load(SAMPLE, SAMPLE_SHA))
        by_digest = {s.output_sha256: s for s in stages}

        for name in ("S1", "S2", "S3", "S4"):
            self.assertEqual(by_digest[EXPECTED[name]].kind, JSCRIPT_XOR)
            self.assertEqual(by_digest[EXPECTED[name]].depth, 0)

        nested = by_digest[EXPECTED["S5"]]
        self.assertEqual(nested.kind, POWERSHELL_XOR)
        self.assertEqual(nested.depth, 1, "the address is reached through the command")

        command = by_digest[EXPECTED["S4"]]
        self.assertEqual(len(command.output), S4_CHARS)

    def test_input_digests_are_recorded_for_every_stage(self) -> None:
        for stage in deobfuscate(_load(SAMPLE, SAMPLE_SHA)):
            with self.subTest(line=stage.line):
                self.assertEqual(
                    stage.input_sha256,
                    hashlib.sha256(stage.encoded.encode("utf-8")).hexdigest(),
                )
                self.assertEqual(
                    stage.output_sha256,
                    hashlib.sha256(stage.output.encode("utf-8")).hexdigest(),
                )

    def test_an_absolute_uri_is_recovered_by_the_final_stage(self) -> None:
        """Asserted by shape, so the address itself is not written here."""
        from orbit.runtime.analysis_runtime import _uris_in

        stages = deobfuscate(_load(SAMPLE, SAMPLE_SHA))
        final = next(s for s in stages if s.output_sha256 == EXPECTED["S5"])
        uris = _uris_in(final.output)
        self.assertEqual(len(uris), 1)
        self.assertTrue(uris[0].startswith("http://"))
        self.assertEqual(uris[0], final.output)

    def test_the_original_and_cleaned_artifacts_decode_identically(self) -> None:
        """Comments and formatting differ; the executable chain does not."""
        cleaned = {s.output_sha256 for s in deobfuscate(_load(SAMPLE, SAMPLE_SHA))}
        original = {s.output_sha256 for s in deobfuscate(_load(ORIGIN, ORIGIN_SHA))}
        self.assertEqual(cleaned, original)
        self.assertEqual(cleaned, set(EXPECTED.values()))


if __name__ == "__main__":
    unittest.main()
