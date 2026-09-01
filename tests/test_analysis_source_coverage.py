"""COVER supplies the complete source in one call, or supplies none.

ANALYSIS-SOURCE-COVERAGE-1. The failure being addressed is not bad reasoning:
it is actions spent *acquiring* source Orbit already holds -- the artifact read
raw, then numbered, then as a repr, then in slices -- because nothing told the
model the bytes were already available. Prompt guidance did not move it.

Two things are pinned here, and the second matters as much as the first.

SOURCE COVERAGE currently supports complete textual artifacts that fit the safe
single-shot context budget. Large-artifact chunked COVER is NOT supported: the
history is append-only, so every turn stays resident and a multi-part coverage
would leave the final call carrying the whole artifact anyway, plus one header
per part. Chunking cannot get an artifact under a budget its bytes exceed.

And coverage is never completion. It says the model was supplied the source; it
says nothing about whether the analysis may stop.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_coverage import (  # noqa: E402
    COVERAGE_COMPLETE,
    COVERAGE_NOT_ELIGIBLE,
    COVERAGE_TOO_LARGE,
    COVERAGE_UNADMISSIBLE,
    SourceCoverage,
    attest_coverage,
    decode_artifact,
    plan_coverage,
)

SAMPLE = ROOT / "workdir" / "samples" / "Fattura981033956.js"
SAMPLE_SHA = "b7cfd5fdeb16d7b5ecea1063419bdad6ad280ed9b73c636707874c3f4001dc0c"


def _cover(raw: bytes, limit: int = 10**9) -> SourceCoverage:
    """Plan with an oracle admitting at most `limit` characters."""
    return plan_coverage(
        raw,
        fits=lambda text: len(text) <= limit,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


class CompleteCoverageTests(unittest.TestCase):
    """A. An eligible textual artifact is supplied whole, in one call."""

    def test_the_whole_artifact_is_offered(self) -> None:
        raw = b"var a = 1;\nprint(a);\n"
        coverage = _cover(raw)
        self.assertTrue(coverage.covered)
        self.assertEqual(coverage.text, raw.decode())
        self.assertEqual(coverage.covered_bytes, len(raw))

    def test_coverage_reconstructs_the_artifact_byte_for_byte(self) -> None:
        raw = b"".join(b"line %04d\n" % n for n in range(500))
        coverage = _cover(raw)
        self.assertEqual(coverage.text.encode("utf-8"), raw)
        self.assertTrue(coverage.attest().complete)

    def test_attestation_names_the_artifact(self) -> None:
        raw = b"body\n"
        attestation = _cover(raw).attest()
        self.assertEqual(attestation.sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(attestation.size_bytes, len(raw))
        self.assertEqual(attestation.covered_bytes, len(raw))


class BoundaryByteTests(unittest.TestCase):
    """C. Awkward bytes survive exactly."""

    def test_multibyte_codepoints_survive(self) -> None:
        raw = ("é中\U0001f600" * 400).encode("utf-8")
        self.assertEqual(_cover(raw).text.encode("utf-8"), raw)

    def test_crlf_and_blank_lines_survive(self) -> None:
        raw = b"a\r\nb\r\n\r\nc\n\n"
        self.assertEqual(_cover(raw).text.encode("utf-8"), raw)

    def test_no_trailing_newline_survives(self) -> None:
        raw = b"first\nsecond\nend"
        coverage = _cover(raw)
        self.assertEqual(coverage.text.encode("utf-8"), raw)
        self.assertFalse(coverage.text.endswith("\n"))

    def test_single_byte_artifact(self) -> None:
        self.assertTrue(_cover(b"x").attest().complete)


class TooLargeTests(unittest.TestCase):
    """E. Beyond the safe single-shot budget: no coverage, never partial."""

    def test_an_oversized_artifact_is_refused(self) -> None:
        coverage = _cover(b"z" * 5000, limit=1000)
        self.assertFalse(coverage.covered)
        self.assertEqual(coverage.status, COVERAGE_TOO_LARGE)
        self.assertEqual(coverage.text, "")
        self.assertEqual(coverage.covered_bytes, 0)

    def test_exactly_at_the_budget_is_covered(self) -> None:
        raw = b"y" * 1000
        self.assertTrue(_cover(raw, limit=1000).covered)

    def test_one_character_over_the_budget_is_refused(self) -> None:
        self.assertFalse(_cover(b"y" * 1001, limit=1000).covered)

    def test_a_refused_coverage_is_never_complete(self) -> None:
        self.assertFalse(_cover(b"z" * 5000, limit=10).attest().complete)


class ChunkingIsNotSupportedTests(unittest.TestCase):
    """The architectural limit, stated as a test rather than a comment.

    ANALYSIS history is append-only: every turn stays resident, so a multi-part
    coverage would leave the final call carrying the whole artifact plus one
    header per part -- strictly worse than one call. There is therefore no
    chunking machinery to exercise, and its absence is deliberate.
    """

    def test_the_module_offers_no_chunked_coverage(self) -> None:
        import orbit.runtime.analysis_coverage as module

        names = " ".join(dir(module)).lower()
        for absent in ("chunk", "part", "split", "range", "max_chunks"):
            self.assertNotIn(absent, names, absent)

    def test_coverage_is_all_or_nothing(self) -> None:
        """There is no state between covered and refused."""
        for raw, limit in ((b"a" * 100, 10), (b"a" * 100, 1000)):
            coverage = _cover(raw, limit=limit)
            if coverage.covered:
                self.assertEqual(coverage.covered_bytes, len(raw))
            else:
                self.assertEqual(coverage.covered_bytes, 0)

    def test_the_docstring_states_the_limit(self) -> None:
        import orbit.runtime.analysis_coverage as module

        text = (module.__doc__ or "").lower()
        self.assertIn("not supported", text)
        self.assertIn("append-only", text)


class NonTextArtifactTests(unittest.TestCase):
    """H. Binary artifacts are not covered; existing behaviour stands."""

    def test_invalid_utf8_is_not_eligible(self) -> None:
        coverage = _cover(b"\xff\xfe\x00\x01\x02binary")
        self.assertEqual(coverage.status, COVERAGE_NOT_ELIGIBLE)
        self.assertEqual(coverage.text, "")

    def test_embedded_nul_is_not_eligible(self) -> None:
        """UTF-16 text decodes as UTF-8 when it is all ASCII; NUL betrays it."""
        self.assertEqual(
            _cover("var a = 1;".encode("utf-16-le")).status, COVERAGE_NOT_ELIGIBLE
        )

    def test_empty_artifact_is_not_eligible(self) -> None:
        self.assertEqual(_cover(b"").status, COVERAGE_NOT_ELIGIBLE)

    def test_decode_never_invents_a_character(self) -> None:
        self.assertIsNone(decode_artifact(b"\xc3\x28"))
        self.assertIsNone(decode_artifact(b"ok\x00"))
        self.assertEqual(decode_artifact(b"ok"), "ok")


class AttestationTests(unittest.TestCase):
    """3. Proved from the artifact and the bytes held, never from the model."""

    def test_a_short_coverage_is_incomplete(self) -> None:
        truncated = SourceCoverage("abc", COVERAGE_COMPLETE, "0" * 64, 10)
        attestation = attest_coverage(truncated)
        self.assertNotEqual(attestation.covered_bytes, attestation.size_bytes)
        self.assertFalse(attestation.complete)

    def test_an_overlong_coverage_is_incomplete(self) -> None:
        overlong = SourceCoverage("abcdefghijkl", COVERAGE_COMPLETE, "0" * 64, 4)
        self.assertFalse(attest_coverage(overlong).complete)

    def test_status_is_part_of_the_proof(self) -> None:
        """Correct bytes do not rescue a coverage that was refused."""
        mislabelled = SourceCoverage("abcd", COVERAGE_TOO_LARGE, "0" * 64, 4)
        self.assertFalse(attest_coverage(mislabelled).complete)

    def test_an_empty_artifact_is_never_complete(self) -> None:
        self.assertFalse(
            attest_coverage(SourceCoverage("", COVERAGE_COMPLETE, "0" * 64, 0)).complete
        )

    def test_byte_length_is_measured_in_bytes_not_characters(self) -> None:
        """A multi-byte artifact must not read as complete by character count."""
        raw = "é" * 10  # 10 characters, 20 bytes
        coverage = SourceCoverage(raw, COVERAGE_COMPLETE, "0" * 64, 10)
        self.assertFalse(attest_coverage(coverage).complete)


class CoverageIsNotCompletenessTests(unittest.TestCase):
    """8. SOURCE_COVERED must never be read as ANALYSIS_COMPLETE."""

    def test_the_module_exposes_no_completion_signal(self) -> None:
        import orbit.runtime.analysis_coverage as module

        for name in dir(module):
            self.assertNotIn("analysis_complete", name.lower())
            self.assertNotIn("decisive", name.lower())

    def test_covering_the_source_leaves_behaviour_unaccounted_for(self) -> None:
        """The concrete case: a behaviour that is not in the source bytes."""
        if not SAMPLE.exists():
            self.skipTest("pinned sample not present")
        raw = SAMPLE.read_bytes()
        coverage = _cover(raw)
        self.assertTrue(coverage.covered)
        # Present only inside a decoded stage, nowhere in the artifact itself.
        self.assertNotIn(b"Win32_Process", raw)
        self.assertNotIn("Win32_Process", coverage.text)


class GenericityTests(unittest.TestCase):
    """4. No language-, format- or malware-specific rules in the code."""

    def test_no_banned_token_appears_in_executable_code(self) -> None:
        """Docstrings are stripped: prose about the guarantee is not a rule."""
        import ast

        path = ROOT / "src" / "orbit" / "runtime" / "analysis_coverage.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree).lower()
        for banned in (
            "javascript", "jscript", "powershell", "vbscript", ".js", ".ps1",
            "xor", "base64", "malware", "http://", "https://", "ioc",
            "smartmaket", "fattura", "wscript", "win32_process",
        ):
            self.assertNotIn(banned, code, banned)

    def test_eligibility_depends_only_on_bytes_and_admission(self) -> None:
        """Same length, different kinds of content, same verdict."""
        payloads = (b"a" * 600, b"function f(){return 1}" * 27 + b"aaaaaa")
        verdicts = {_cover(payload, limit=1000).status for payload in payloads}
        self.assertEqual(len(verdicts), 1)

    def test_the_module_never_evaluates_the_artifact(self) -> None:
        source = (
            ROOT / "src" / "orbit" / "runtime" / "analysis_coverage.py"
        ).read_text()
        for banned in ("eval(", "exec(", "subprocess", "os.system", "__import__"):
            self.assertNotIn(banned, source, banned)


class PinnedArtifactTests(unittest.TestCase):
    """9. The prototype gate, held as a regression."""

    def setUp(self) -> None:
        if not SAMPLE.exists():
            self.skipTest("pinned sample not present")
        self.raw = SAMPLE.read_bytes()
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), SAMPLE_SHA)

    def test_the_artifact_is_covered_when_the_budget_allows(self) -> None:
        coverage = _cover(self.raw)
        self.assertTrue(coverage.covered)
        self.assertEqual(coverage.text.encode("utf-8"), self.raw)
        self.assertEqual(coverage.sha256, SAMPLE_SHA)

    def test_in_source_behaviour_is_present_in_the_coverage(self) -> None:
        """F. Deterministic transforms do not displace host behaviour."""
        coverage = _cover(self.raw)
        for needle in ("WScript.Sleep", "ShowWindow", ".Create("):
            self.assertIn(needle, coverage.text, needle)

    def test_transform_evidence_coexists_with_coverage(self) -> None:
        from orbit.runtime.analysis_deobfuscate import deobfuscate_with_status

        result = deobfuscate_with_status(self.raw.decode("utf-8"))
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.stages), 5)
        self.assertTrue(
            any("smartmaket.com/1.php" in stage.output for stage in result.stages)
        )
        self.assertTrue(_cover(self.raw).covered)

    def test_no_action_is_needed_to_acquire_the_source(self) -> None:
        self.assertEqual(_cover(self.raw).text.encode("utf-8"), self.raw)


class ArtifactWithoutTransformsTests(unittest.TestCase):
    """G. Coverage does not depend on there being anything to decode."""

    def test_plain_text_is_covered(self) -> None:
        raw = b"# notes\nnothing encoded here at all\n" * 40
        from orbit.runtime.analysis_deobfuscate import deobfuscate_with_status

        self.assertEqual(deobfuscate_with_status(raw.decode()).stages, [])
        self.assertTrue(_cover(raw).covered)


if __name__ == "__main__":
    unittest.main()
