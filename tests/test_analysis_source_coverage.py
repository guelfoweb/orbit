"""COVER must supply every artifact byte, exactly once, or supply none.

ANALYSIS-SOURCE-COVERAGE-1. The failure being addressed is not bad reasoning:
it is actions spent *acquiring* source Orbit already holds -- the artifact read
raw, then numbered, then as a repr, then in slices -- because nothing told the
model the bytes were already available. Prompt guidance did not move it.

What is pinned here is the guarantee and its limits. Coverage means every
eligible source byte was presented, in order, once. It does not mean the
analysis is finished: the pinned sample's `Win32_Process` is not in its source
bytes at all, so a fully covered artifact still has behaviour left to resolve.
Tests below hold both halves.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_coverage import (  # noqa: E402
    COVERAGE_BUDGET_EXCEEDED,
    SourceChunk,
    COVERAGE_COMPLETE,
    COVERAGE_NOT_ELIGIBLE,
    COVERAGE_UNADMISSIBLE,
    CoveragePlan,
    attest_coverage,
    decode_artifact,
    plan_coverage,
    reconstruct,
)

SAMPLE = ROOT / "workdir" / "samples" / "Fattura981033956.js"
SAMPLE_SHA = "b7cfd5fdeb16d7b5ecea1063419bdad6ad280ed9b73c636707874c3f4001dc0c"


def _chars(limit: int, *, cumulative: bool = False):
    """An oracle admitting a chunk of at most `limit` characters.

    `cumulative` models the append-only history: the parts already planned
    stay resident, so each later part has less room than the one before it.
    """

    def fits(text: str, index: int, preceding: tuple[str, ...]) -> bool:
        used = sum(len(part) for part in preceding) if cumulative else 0
        return used + len(text) <= limit

    return fits


def _plan(raw: bytes, limit: int, *, max_chunks: int = 64) -> CoveragePlan:
    return plan_coverage(
        raw,
        fits=_chars(limit),
        sha256=hashlib.sha256(raw).hexdigest(),
        max_chunks=max_chunks,
    )


def _assert_covers(case: unittest.TestCase, plan: CoveragePlan, raw: bytes) -> None:
    """The whole invariant, asserted the same way everywhere it matters."""
    case.assertTrue(plan.covered)
    attestation = plan.attest()
    case.assertTrue(attestation.complete)
    case.assertEqual(attestation.gaps, ())
    case.assertEqual(attestation.overlaps, ())
    case.assertEqual(attestation.covered_bytes, len(raw))
    case.assertEqual(attestation.sha256, hashlib.sha256(raw).hexdigest())
    # Ordered, contiguous, non-empty.
    case.assertEqual(plan.chunks[0].start, 0)
    case.assertEqual(plan.chunks[-1].end, len(raw))
    for earlier, later in zip(plan.chunks, plan.chunks[1:]):
        case.assertEqual(earlier.end, later.start)
        case.assertLess(earlier.start, earlier.end)
    # Byte-exact reconstruction, and each chunk's text is its own range.
    case.assertEqual(reconstruct(plan, raw), raw)
    for chunk in plan.chunks:
        case.assertEqual(chunk.text.encode("utf-8"), raw[chunk.start : chunk.end])
    # Exactly one chunk is final, and it is the last.
    case.assertEqual([c.index for c in plan.chunks if c.is_final], [len(plan.chunks)])


class SingleCallCoverageTests(unittest.TestCase):
    """A. A small textual artifact is supplied in one COVER call."""

    def test_small_artifact_is_one_chunk(self) -> None:
        raw = b"var a = 1;\nprint(a);\n"
        plan = _plan(raw, 10_000)
        self.assertEqual(len(plan.chunks), 1)
        self.assertEqual(plan.chunks[0].text, raw.decode())
        _assert_covers(self, plan, raw)

    def test_the_single_chunk_is_marked_final(self) -> None:
        plan = _plan(b"x = 1\n", 10_000)
        self.assertTrue(plan.chunks[0].is_final)


class MultiChunkCoverageTests(unittest.TestCase):
    """B. A larger artifact is covered in exact ordered chunks."""

    def test_multi_chunk_plan_has_no_gaps_or_overlaps(self) -> None:
        raw = ("line %04d\n" % 0).encode() * 0 + b"".join(
            b"line %04d\n" % n for n in range(500)
        )
        plan = _plan(raw, 900)
        self.assertGreater(len(plan.chunks), 1)
        _assert_covers(self, plan, raw)

    def test_chunks_are_numbered_in_order(self) -> None:
        raw = b"".join(b"row %03d\n" % n for n in range(300))
        plan = _plan(raw, 500)
        self.assertEqual(
            [c.index for c in plan.chunks], list(range(1, len(plan.chunks) + 1))
        )
        self.assertTrue(all(c.total == len(plan.chunks) for c in plan.chunks))

    def test_every_byte_belongs_to_exactly_one_chunk(self) -> None:
        raw = b"".join(b"%d;" % n for n in range(4000))
        plan = _plan(raw, 700)
        owners = [0] * len(raw)
        for chunk in plan.chunks:
            for offset in range(chunk.start, chunk.end):
                owners[offset] += 1
        self.assertEqual(set(owners), {1})


class BoundaryByteTests(unittest.TestCase):
    """C. Boundaries reconstruct byte-for-byte, including awkward ones."""

    def test_multibyte_codepoints_are_never_split(self) -> None:
        raw = ("é中\U0001f600" * 400).encode("utf-8")
        plan = _plan(raw, 90)
        _assert_covers(self, plan, raw)
        for chunk in plan.chunks:
            # Each chunk decodes independently -- the point of splitting on
            # code-point boundaries rather than byte counts.
            raw[chunk.start : chunk.end].decode("utf-8")

    def test_crlf_and_trailing_newline_survive(self) -> None:
        raw = b"a\r\nb\r\n\r\nc\n\n"
        plan = _plan(raw, 4)
        _assert_covers(self, plan, raw)

    def test_artifact_without_trailing_newline(self) -> None:
        raw = b"".join(b"%d," % n for n in range(900)) + b"end"
        plan = _plan(raw, 300)
        _assert_covers(self, plan, raw)
        self.assertTrue(plan.chunks[-1].text.endswith("end"))

    def test_single_byte_artifact(self) -> None:
        _assert_covers(self, _plan(b"x", 10), b"x")

    def test_chunk_limit_equal_to_artifact_length(self) -> None:
        raw = b"abcdefghij"
        plan = _plan(raw, len(raw))
        self.assertEqual(len(plan.chunks), 1)
        _assert_covers(self, plan, raw)

    def test_chunk_limit_one_byte_short_splits(self) -> None:
        raw = b"abcdefghij"
        plan = _plan(raw, len(raw) - 1)
        self.assertEqual(len(plan.chunks), 2)
        _assert_covers(self, plan, raw)


class TokenAdmissionTests(unittest.TestCase):
    """D. Sizing is the tokenizer's answer, never a character count."""

    def test_token_heavy_text_yields_more_chunks_than_character_sizing(self) -> None:
        """The density case: same characters, honest counter, smaller chunks.

        A character cap cannot bound tokens -- this repo measured 1.123
        chars/token on obfuscated content -- so the oracle here charges four
        tokens per character and the plan must shrink accordingly.
        """
        raw = b"".join(b"\\x%02x" % (n % 256) for n in range(1200))
        by_chars = plan_coverage(
            raw, fits=lambda t, i, p: len(t) <= 2000,
            sha256=hashlib.sha256(raw).hexdigest(), max_chunks=64,
        )
        by_tokens = plan_coverage(
            raw, fits=lambda t, i, p: len(t) * 4 <= 2000,
            sha256=hashlib.sha256(raw).hexdigest(), max_chunks=64,
        )
        self.assertGreater(len(by_tokens.chunks), len(by_chars.chunks))
        _assert_covers(self, by_tokens, raw)
        for chunk in by_tokens.chunks:
            self.assertLessEqual(len(chunk.text) * 4, 2000)

    def test_no_chunk_ever_exceeds_the_oracle(self) -> None:
        raw = b"".join(b"%d " % n for n in range(2000))
        plan = _plan(raw, 331)
        self.assertTrue(all(len(c.text) <= 331 for c in plan.chunks))

    def test_chunks_are_as_large_as_the_oracle_permits(self) -> None:
        """Coverage must not waste model calls it does not need."""
        raw = b"a" * 1000
        plan = _plan(raw, 400)
        self.assertEqual([c.end - c.start for c in plan.chunks], [400, 400, 200])


class BudgetRefusalTests(unittest.TestCase):
    """E. Too large for the permitted calls means no coverage, never partial."""

    def test_artifact_needing_more_chunks_than_permitted_is_refused(self) -> None:
        raw = b"a" * 5000
        plan = _plan(raw, 400, max_chunks=4)
        self.assertFalse(plan.covered)
        self.assertEqual(plan.status, COVERAGE_BUDGET_EXCEEDED)
        self.assertEqual(plan.chunks, ())

    def test_a_refused_plan_carries_no_partial_coverage(self) -> None:
        raw = b"b" * 9000
        plan = _plan(raw, 300, max_chunks=2)
        self.assertEqual(plan.covered_bytes, 0)
        self.assertFalse(plan.attest().complete)

    def test_exactly_the_permitted_number_of_chunks_is_covered(self) -> None:
        raw = b"c" * 1200
        plan = _plan(raw, 400, max_chunks=3)
        self.assertTrue(plan.covered)
        self.assertEqual(len(plan.chunks), 3)
        _assert_covers(self, plan, raw)

    def test_one_chunk_over_the_ceiling_is_refused(self) -> None:
        raw = b"c" * 1201
        plan = _plan(raw, 400, max_chunks=3)
        self.assertFalse(plan.covered)

    def test_zero_permitted_chunks_refuses(self) -> None:
        plan = _plan(b"anything", 400, max_chunks=0)
        self.assertFalse(plan.covered)
        self.assertEqual(plan.status, COVERAGE_BUDGET_EXCEEDED)

    def test_a_chunk_that_can_never_fit_is_refused(self) -> None:
        """Not even one code point admissible: no plan exists at all."""
        plan = plan_coverage(
            b"hello", fits=lambda text, index, preceding: False,
            sha256="0" * 64, max_chunks=8,
        )
        self.assertEqual(plan.status, COVERAGE_UNADMISSIBLE)
        self.assertEqual(plan.chunks, ())


class AppendOnlyHistoryTests(unittest.TestCase):
    """Chunking splits messages, not the resident context.

    ANALYSIS history is append-only, so every COVER turn stays resident and the
    final call carries the whole artifact however many parts it arrived in. An
    artifact whose bytes do not fit the input budget therefore cannot be
    covered at all -- and the planner must discover that and refuse, rather
    than send parts it cannot follow through.
    """

    def test_artifact_larger_than_the_budget_is_refused_not_split(self) -> None:
        raw = b"z" * 5000
        plan = _plan(raw, 1000, max_chunks=64)  # cumulative oracle below
        self.assertTrue(plan.covered)  # non-cumulative oracle still splits
        cumulative = plan_coverage(
            raw, fits=_chars(1000, cumulative=True),
            sha256=hashlib.sha256(raw).hexdigest(), max_chunks=64,
        )
        self.assertFalse(cumulative.covered)
        self.assertEqual(cumulative.chunks, ())
        # "Outgrew the budget", not "could not start".
        self.assertEqual(cumulative.status, COVERAGE_BUDGET_EXCEEDED)

    def test_artifact_within_the_budget_is_covered_under_cumulation(self) -> None:
        raw = b"z" * 900
        plan = plan_coverage(
            raw, fits=_chars(1000, cumulative=True),
            sha256=hashlib.sha256(raw).hexdigest(), max_chunks=64,
        )
        _assert_covers(self, plan, raw)

    def test_each_part_is_measured_against_the_parts_before_it(self) -> None:
        """The oracle must actually receive the planned prefix."""
        seen: list[tuple[int, int]] = []

        def fits(text: str, index: int, preceding: tuple[str, ...]) -> bool:
            seen.append((index, len(preceding)))
            return len(text) <= 100

        plan_coverage(b"q" * 350, fits=fits, sha256="0" * 64, max_chunks=64)
        # Part n is always offered exactly n-1 predecessors.
        self.assertTrue(all(index - 1 == count for index, count in seen))
        self.assertGreater(max(index for index, _ in seen), 1)

    def test_a_partially_admissible_artifact_yields_no_partial_plan(self) -> None:
        """The first parts fitting is not a reason to send them."""
        raw = b"w" * 4000
        plan = plan_coverage(
            raw, fits=_chars(1500, cumulative=True),
            sha256=hashlib.sha256(raw).hexdigest(), max_chunks=64,
        )
        self.assertFalse(plan.covered)
        self.assertEqual(plan.covered_bytes, 0)


class NonTextArtifactTests(unittest.TestCase):
    """H. Binary artifacts are not covered; existing behaviour stands."""

    def test_invalid_utf8_is_not_eligible(self) -> None:
        plan = _plan(b"\xff\xfe\x00\x01\x02binary", 4000)
        self.assertEqual(plan.status, COVERAGE_NOT_ELIGIBLE)
        self.assertEqual(plan.chunks, ())

    def test_embedded_nul_is_not_eligible(self) -> None:
        """UTF-16 text decodes as UTF-8 when it is all ASCII; NUL betrays it."""
        plan = _plan("var a = 1;".encode("utf-16-le"), 4000)
        self.assertEqual(plan.status, COVERAGE_NOT_ELIGIBLE)

    def test_empty_artifact_is_not_eligible(self) -> None:
        self.assertEqual(_plan(b"", 4000).status, COVERAGE_NOT_ELIGIBLE)

    def test_decode_artifact_never_invents_a_character(self) -> None:
        self.assertIsNone(decode_artifact(b"\xc3\x28"))
        self.assertIsNone(decode_artifact(b"ok\x00"))
        self.assertEqual(decode_artifact(b"ok"), "ok")


class AttestationTests(unittest.TestCase):
    """3. The runtime proves coverage from the plan, never from model output."""

    def test_attestation_reports_sha_bytes_and_ranges(self) -> None:
        raw = b"".join(b"%d\n" % n for n in range(400))
        plan = _plan(raw, 300)
        attestation = plan.attest()
        self.assertEqual(attestation.sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(attestation.size_bytes, len(raw))
        self.assertEqual(
            attestation.ranges, tuple((c.start, c.end) for c in plan.chunks)
        )

    def test_a_gap_is_detected_and_defeats_completeness(self) -> None:
        raw = b"0123456789"
        plan = _plan(raw, 4)
        holed = CoveragePlan(
            chunks=(plan.chunks[0], plan.chunks[2]),
            status=COVERAGE_COMPLETE, sha256=plan.sha256, size_bytes=plan.size_bytes,
        )
        attestation = attest_coverage(holed)
        self.assertTrue(attestation.gaps)
        self.assertFalse(attestation.complete)

    def test_an_overlap_is_detected_and_defeats_completeness(self) -> None:
        raw = b"0123456789"
        plan = _plan(raw, 4)
        doubled = CoveragePlan(
            chunks=(plan.chunks[0], plan.chunks[0], plan.chunks[1], plan.chunks[2]),
            status=COVERAGE_COMPLETE, sha256=plan.sha256, size_bytes=plan.size_bytes,
        )
        attestation = attest_coverage(doubled)
        self.assertTrue(attestation.overlaps)
        self.assertFalse(attestation.complete)

    def test_a_truncated_tail_is_a_gap(self) -> None:
        raw = b"0123456789"
        plan = _plan(raw, 4)
        short = CoveragePlan(
            chunks=plan.chunks[:-1], status=COVERAGE_COMPLETE,
            sha256=plan.sha256, size_bytes=plan.size_bytes,
        )
        self.assertFalse(attest_coverage(short).complete)

    def test_a_gap_offset_by_an_overlap_is_still_incomplete(self) -> None:
        """The byte total can be right while the coverage is wrong.

        A duplicated part and a missing part of the same size leave
        `covered_bytes == size_bytes`, so the count alone cannot detect it.
        Only the gap and overlap checks can, which is why completeness may
        never be reduced to arithmetic on the total.
        """
        raw = b"0123456789"
        plan = _plan(raw, 2)
        self.assertGreaterEqual(len(plan.chunks), 4)
        # Cover chunk 0 twice and drop chunk 1: same byte count, real hole.
        broken = CoveragePlan(
            chunks=(plan.chunks[0], plan.chunks[0], *plan.chunks[2:]),
            status=COVERAGE_COMPLETE, sha256=plan.sha256, size_bytes=plan.size_bytes,
        )
        attestation = attest_coverage(broken)
        self.assertEqual(attestation.covered_bytes, attestation.size_bytes)
        self.assertTrue(attestation.gaps)
        self.assertTrue(attestation.overlaps)
        self.assertFalse(attestation.complete)

    def test_a_gap_alone_defeats_completeness(self) -> None:
        raw = b"0123456789"
        plan = _plan(raw, 2)
        holed = CoveragePlan(
            chunks=(plan.chunks[0], *plan.chunks[2:]),
            status=COVERAGE_COMPLETE, sha256=plan.sha256, size_bytes=plan.size_bytes,
        )
        attestation = attest_coverage(holed)
        self.assertTrue(attestation.gaps)
        self.assertFalse(attestation.complete)

    def test_an_overlap_alone_defeats_completeness(self) -> None:
        raw = b"0123456789"
        plan = _plan(raw, 2)
        doubled = CoveragePlan(
            chunks=(plan.chunks[0], *plan.chunks),
            status=COVERAGE_COMPLETE, sha256=plan.sha256, size_bytes=plan.size_bytes,
        )
        attestation = attest_coverage(doubled)
        self.assertTrue(attestation.overlaps)
        self.assertFalse(attestation.complete)

    def test_gap_and_overlap_checks_are_jointly_load_bearing(self) -> None:
        """Both structural checks together, without the byte total.

        Individually neither is observable: on a fixed artifact size, a gap
        forces a compensating overlap and vice versa, so the byte total plus
        either check already decides every case (verified exhaustively over
        every range-set of up to three chunks on a six-byte artifact: 0
        divergences for each check alone, 1224 when both are removed). What is
        load-bearing is the pair, and this pins that.
        """
        chunks = (
            SourceChunk(index=1, total=2, start=0, end=1, text="0"),
            SourceChunk(index=2, total=2, start=0, end=5, text="01234"),
        )
        plan = CoveragePlan(
            chunks=chunks, status=COVERAGE_COMPLETE, sha256="0" * 64, size_bytes=6,
        )
        attestation = attest_coverage(plan)
        # The byte total alone is satisfied -- and the coverage is still wrong.
        self.assertEqual(attestation.covered_bytes, attestation.size_bytes)
        self.assertTrue(attestation.gaps)
        self.assertTrue(attestation.overlaps)
        self.assertFalse(attestation.complete)

    def test_a_refused_status_is_never_complete(self) -> None:
        """Status is part of the proof: correct ranges do not rescue it."""
        raw = b"0123456789"
        plan = _plan(raw, 4)
        mislabelled = CoveragePlan(
            chunks=plan.chunks, status=COVERAGE_BUDGET_EXCEEDED,
            sha256=plan.sha256, size_bytes=plan.size_bytes,
        )
        self.assertFalse(attest_coverage(mislabelled).complete)


class CoverageIsNotAnalysisCompletenessTests(unittest.TestCase):
    """8. SOURCE_COVERED must never be read as ANALYSIS_COMPLETE."""

    def test_the_module_exposes_no_analysis_completion_signal(self) -> None:
        import orbit.runtime.analysis_coverage as module

        exported = dir(module)
        for name in exported:
            self.assertNotIn("analysis_complete", name.lower())
            self.assertNotIn("decisive", name.lower())
        self.assertNotIn("ANALYSIS_COMPLETE", exported)

    def test_covered_reports_bytes_not_conclusions(self) -> None:
        raw = SAMPLE.read_bytes()
        plan = _plan(raw, 5200)
        self.assertTrue(plan.covered)
        # The behaviour that proves the distinction: this string is nowhere in
        # the artifact's own bytes -- it exists only inside a decoded stage --
        # so complete source coverage does not account for it.
        self.assertNotIn(b"Win32_Process", raw)
        self.assertTrue(all("Win32_Process" not in c.text for c in plan.chunks))


class PinnedArtifactCoverageTests(unittest.TestCase):
    """9. The prototype gate, held as a regression."""

    def setUp(self) -> None:
        if not SAMPLE.exists():
            self.skipTest("pinned sample not present")
        self.raw = SAMPLE.read_bytes()
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), SAMPLE_SHA)

    def test_pinned_artifact_is_fully_covered(self) -> None:
        plan = _plan(self.raw, 5200)
        _assert_covers(self, plan, self.raw)
        self.assertEqual(plan.sha256, SAMPLE_SHA)

    def test_host_behaviour_regions_are_covered(self) -> None:
        """F. Deterministic transforms do not displace in-source behaviour."""
        plan = _plan(self.raw, 5200)
        for needle in (b"WScript.Sleep", b"ShowWindow", b".Create("):
            offset = self.raw.find(needle)
            self.assertGreaterEqual(offset, 0, needle)
            owners = [c for c in plan.chunks if c.start <= offset < c.end]
            self.assertEqual(len(owners), 1, needle)
            self.assertIn(needle.decode(), owners[0].text)

    def test_transform_evidence_coexists_with_coverage(self) -> None:
        from orbit.runtime.analysis_deobfuscate import deobfuscate_with_status

        result = deobfuscate_with_status(self.raw.decode("utf-8"))
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.stages), 5)
        self.assertTrue(
            any("smartmaket.com/1.php" in stage.output for stage in result.stages)
        )
        self.assertTrue(_plan(self.raw, 5200).covered)

    def test_coverage_needs_no_actions(self) -> None:
        """Zero execute_analysis actions are required to acquire the source."""
        plan = _plan(self.raw, 5200)
        self.assertEqual(reconstruct(plan, self.raw), self.raw)


class ArtifactWithoutTransformsTests(unittest.TestCase):
    """G. Coverage does not depend on there being anything to decode."""

    def test_plain_text_with_no_transforms_is_covered(self) -> None:
        raw = b"# notes\nnothing encoded here at all\n" * 40
        from orbit.runtime.analysis_deobfuscate import deobfuscate_with_status

        self.assertEqual(deobfuscate_with_status(raw.decode()).stages, [])
        _assert_covers(self, _plan(raw, 400), raw)


class GenericityTests(unittest.TestCase):
    """4. No language-, format- or malware-specific rules anywhere."""

    def test_coverage_source_names_no_language_or_indicator(self) -> None:
        """No banned token may appear in executable code.

        Comments and docstrings are stripped first, deliberately. The module
        docstring names `Win32_Process` because explaining why coverage is not
        analysis completeness requires the concrete example -- prose about the
        limits of the guarantee is not a rule keyed on an artifact. What must
        stay clean is anything that runs.
        """
        import ast

        path = ROOT / "src" / "orbit" / "runtime" / "analysis_coverage.py"
        tree = ast.parse(path.read_text())
        # Drop every docstring, then unparse: what is left is only code.
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

    def test_coverage_is_driven_only_by_bytes_and_admission(self) -> None:
        """Identical-length inputs of different kinds plan identically."""
        plans = [
            _plan(payload, 100)
            for payload in (
                b"a" * 600,
                b"function f(){return 1}" * 27 + b"aaaaaa",
            )
        ]
        self.assertEqual(len(plans[0].chunks), len(plans[1].chunks))
        for plan, payload in zip(plans, (b"a" * 600, None)):
            self.assertTrue(plan.covered)


if __name__ == "__main__":
    unittest.main()
