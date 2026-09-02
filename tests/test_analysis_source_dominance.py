"""Only the covered source plus recomputable facts may count as adding nothing.

ANALYSIS-SOURCE-DOMINANCE-1. Exact equivalence catches an output that is only
the source; the live run showed the model almost always prints the source and
then one deterministic fact about it -- `LEN: 1495`, `TOTAL_LINES: 46`, a SHA,
a hex prefix -- so equivalence almost never fired and nine actions were spent
on source-derived observations.

The boundary here is recomputation, not correctness. A property qualifies only
if Orbit can derive it from the artifact's own bytes without interpreting the
program: length, line count, digest, the hex of an explicitly named range. A
count of functions does not qualify however true it is, because deriving it
means parsing the language.

The proof is total: every component explained, every value recomputed and
compared exactly, no unexplained bytes. Most of what follows tests what must
NOT be suppressed, because a false positive discards real evidence while a
missed one costs only a model call.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_source_dominance import (  # noqa: E402
    BYTE_LENGTH,
    HEX_PREFIX,
    LINE_COUNT,
    MAX_PROPERTIES,
    SHA256,
    TEXT_LENGTH,
    classify_dominated,
)
from orbit.runtime.analysis_source_identity import NUMBERED, RAW, REPR  # noqa: E402

SOURCE = (
    "import os\n"
    "\n"
    "\n"
    "def handler(name):\n"
    "    return os.environ.get(name)\n"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _numbered(text: str) -> str:
    return "\n".join(f"{i:3}: {line}" for i, line in enumerate(text.splitlines()))


class LengthTests(unittest.TestCase):
    """A/B. Byte and text length, exact or not at all."""

    def test_source_plus_correct_length_is_dominated(self) -> None:
        found = classify_dominated(f"{SOURCE}\nLEN: {len(SOURCE)}\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertEqual(found.representation, RAW)
        self.assertIn(TEXT_LENGTH, found.properties)

    def test_length_off_by_one_is_useful(self) -> None:
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                self.assertIsNone(
                    classify_dominated(
                        f"{SOURCE}\nLEN: {len(SOURCE) + delta}\n", SOURCE
                    )
                )

    def test_byte_length_and_text_length_are_distinguished(self) -> None:
        """They differ on non-ASCII, and each label accepts only its own."""
        source = "héllo\n"  # 6 characters, 7 bytes
        self.assertNotEqual(len(source), len(source.encode("utf-8")))
        self.assertIsNotNone(
            classify_dominated(f"{source}\nlen_str: {len(source)}\n", source)
        )
        self.assertIsNotNone(
            classify_dominated(
                f"{source}\nlen_bytes: {len(source.encode())}\n", source
            )
        )
        # The wrong quantity under a specific label is refused.
        self.assertIsNone(
            classify_dominated(f"{source}\nlen_str: {len(source.encode())}\n", source)
        )
        self.assertIsNone(
            classify_dominated(f"{source}\nlen_bytes: {len(source)}\n", source)
        )

    def test_a_non_numeric_length_is_useful(self) -> None:
        self.assertIsNone(classify_dominated(f"{SOURCE}\nLEN: many\n", SOURCE))


class LineCountTests(unittest.TestCase):
    """C/D. Line count, under either counting rule the label may mean."""

    def test_source_plus_correct_total_lines_is_dominated(self) -> None:
        count = len(SOURCE.split("\n"))
        found = classify_dominated(f"{SOURCE}\nTOTAL_LINES: {count}\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertIn(LINE_COUNT, found.properties)

    def test_splitlines_counting_is_also_accepted(self) -> None:
        count = len(SOURCE.splitlines())
        self.assertNotEqual(count, len(SOURCE.split("\n")))
        self.assertIsNotNone(
            classify_dominated(f"{SOURCE}\nTOTAL_LINES: {count}\n", SOURCE)
        )

    def test_only_the_two_recomputable_counts_are_accepted(self) -> None:
        """Two rules for one label, and nothing between them.

        On text containing U+2028 the rules differ by two. Accepting both is
        not a loosening -- Orbit recomputes each from the same bytes, so
        neither tells the session anything -- but every other value, including
        the ones in between, is refused.
        """
        text = "a\u2028b\u2028c"
        by_split = len(text.split("\n"))
        by_lines = len(text.splitlines())
        self.assertEqual((by_split, by_lines), (1, 3))
        for count in (by_split, by_lines):
            self.assertIsNotNone(
                classify_dominated(f"{text}\nTOTAL_LINES: {count}\n", text)
            )
        for count in (0, 2, 4, 5):
            with self.subTest(count=count):
                self.assertIsNone(
                    classify_dominated(f"{text}\nTOTAL_LINES: {count}\n", text)
                )

    def test_an_incorrect_line_count_is_useful(self) -> None:
        for count in (0, 1, 999, len(SOURCE.split("\n")) + 2):
            with self.subTest(count=count):
                self.assertIsNone(
                    classify_dominated(f"{SOURCE}\nTOTAL_LINES: {count}\n", SOURCE)
                )


class DigestTests(unittest.TestCase):
    """E/F. SHA-256 of the complete source."""

    def test_source_plus_exact_digest_is_dominated(self) -> None:
        found = classify_dominated(f"{SOURCE}\nsha256: {_sha(SOURCE)}\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertIn(SHA256, found.properties)

    def test_a_wrong_digest_is_useful(self) -> None:
        self.assertIsNone(classify_dominated(f"{SOURCE}\nsha256: {'0' * 64}\n", SOURCE))

    def test_a_digest_of_something_else_is_useful(self) -> None:
        """The digest of a substring is a real finding about that substring."""
        self.assertIsNone(
            classify_dominated(f"{SOURCE}\nsha256: {_sha(SOURCE[:20])}\n", SOURCE)
        )

    def test_md5_is_not_an_allowed_property(self) -> None:
        """Not on the allow-list, however deterministic it is.

        The list is exhaustive by design: adding a label is a deliberate act
        justified by an observed output form, not a convenience.
        """
        digest = hashlib.md5(SOURCE.encode()).hexdigest()
        self.assertIsNone(classify_dominated(f"{SOURCE}\nmd5: {digest}\n", SOURCE))


class HexRangeTests(unittest.TestCase):
    """G/H. A hex range, only when the label names it exactly."""

    def test_an_exact_hex_prefix_is_dominated(self) -> None:
        hexed = SOURCE[:20].encode("utf-8").hex()
        found = classify_dominated(f"{SOURCE}\nHEX20: {hexed}\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertIn(HEX_PREFIX, found.properties)

    def test_the_count_in_the_label_selects_the_range(self) -> None:
        """A correct hex under the wrong count is refused."""
        hexed = SOURCE[:20].encode("utf-8").hex()
        self.assertIsNone(classify_dominated(f"{SOURCE}\nHEX21: {hexed}\n", SOURCE))

    def test_wrong_hex_is_useful(self) -> None:
        self.assertIsNone(classify_dominated(f"{SOURCE}\nHEX20: {'ab' * 20}\n", SOURCE))

    def test_an_unlabelled_hex_range_is_useful(self) -> None:
        """Nothing is guessed: without a count there is no proven range."""
        hexed = SOURCE[:20].encode("utf-8").hex()
        for label in ("HEX", "HEXDUMP", "hex"):
            with self.subTest(label=label):
                self.assertIsNone(
                    classify_dominated(f"{SOURCE}\n{label}: {hexed}\n", SOURCE)
                )

    def test_a_range_longer_than_the_source_is_useful(self) -> None:
        """Refused rather than clamped: nobody proved that range."""
        hexed = SOURCE.encode("utf-8").hex()
        self.assertIsNone(
            classify_dominated(f"{SOURCE}\nHEX99999: {hexed}\n", SOURCE)
        )

    def test_a_zero_length_range_is_useful(self) -> None:
        self.assertIsNone(classify_dominated(f"{SOURCE}\nHEX0: \n", SOURCE))

    def test_character_slicing_is_what_the_label_means(self) -> None:
        """`data[:n].encode().hex()` -- not the first n BYTES.

        The two differ on non-ASCII, and only the form the observed code
        produced is accepted.
        """
        source = "héllo world\n"
        by_chars = source[:5].encode("utf-8").hex()
        by_bytes = source.encode("utf-8")[:5].hex()
        self.assertNotEqual(by_chars, by_bytes)
        self.assertIsNotNone(
            classify_dominated(f"{source}\nHEX5: {by_chars}\n", source)
        )
        self.assertIsNone(classify_dominated(f"{source}\nHEX5: {by_bytes}\n", source))


class RepresentationTests(unittest.TestCase):
    """Every representation equivalence proves, plus properties."""

    def test_repr_plus_length_is_dominated(self) -> None:
        found = classify_dominated(f"{SOURCE!r}\nLEN: {len(SOURCE)}\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertEqual(found.representation, REPR)

    def test_a_numbered_listing_plus_line_count_is_dominated(self) -> None:
        listing = _numbered(SOURCE)
        count = len(SOURCE.splitlines())
        found = classify_dominated(f"{listing}\nTOTAL_LINES: {count}\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertEqual(found.representation, NUMBERED)

    def test_several_exact_properties_together(self) -> None:
        """I. Length and digest, both exact."""
        observed = f"{SOURCE}\nLEN: {len(SOURCE)}\nsha256: {_sha(SOURCE)}\n"
        found = classify_dominated(observed, SOURCE)
        self.assertIsNotNone(found)
        self.assertEqual(set(found.properties), {TEXT_LENGTH, SHA256})

    def test_the_source_alone_is_not_dominance(self) -> None:
        """That is exact equivalence, which the other module answers."""
        self.assertIsNone(classify_dominated(SOURCE, SOURCE))
        self.assertIsNone(classify_dominated(SOURCE + "\n", SOURCE))


class FailClosedTests(unittest.TestCase):
    """J-N. Everything that must remain useful evidence."""

    def test_a_semantic_line_defeats_the_proof(self) -> None:
        """J/K. True or not, a function count is not byte-derived."""
        for extra in ("FUNCTIONS: 1", "IMPORTS: 1", "CALLS: 2", "VULNS: 0"):
            with self.subTest(extra=extra):
                self.assertIsNone(
                    classify_dominated(
                        f"{SOURCE}\nLEN: {len(SOURCE)}\n{extra}\n", SOURCE
                    )
                )

    def test_partial_source_with_its_own_correct_length_is_useful(self) -> None:
        """L. The length is right for what was printed, and that is the point.

        Half the file plus the length of that half is a real observation about
        which half was taken.
        """
        half = SOURCE[: len(SOURCE) // 2]
        self.assertIsNone(classify_dominated(f"{half}\nLEN: {len(half)}\n", SOURCE))

    def test_a_modified_source_with_matching_properties_is_useful(self) -> None:
        """M. Properties correct for the modification, not for the artifact."""
        modified = SOURCE.replace("os", "0s", 1)
        observed = f"{modified}\nLEN: {len(modified)}\nsha256: {_sha(modified)}\n"
        self.assertIsNone(classify_dominated(observed, SOURCE))

    def test_an_unexplained_marker_is_useful(self) -> None:
        """The live `===END===` case: a component the grammar does not name."""
        self.assertIsNone(
            classify_dominated(
                f"{SOURCE}\n===END===\nLEN: {len(SOURCE)}\n", SOURCE
            )
        )

    def test_a_heading_before_the_source_is_useful(self) -> None:
        self.assertIsNone(
            classify_dominated(f"=== source ===\n{SOURCE}\nLEN: {len(SOURCE)}\n", SOURCE)
        )

    def test_a_property_must_begin_on_its_own_line(self) -> None:
        """The separator is part of the grammar, not incidental formatting.

        `print(data)` then `print("LEN:", n)` puts a newline between them.
        Without requiring it, output where the property is merely appended --
        after a space, or with no separator that the source itself did not
        supply -- would be accepted, and the boundary between the source and
        what follows it would be whatever the text happened to allow.
        """
        for observed in (
            f"{SOURCE} LEN: {len(SOURCE)}\n",
            f"{SOURCE}\tLEN: {len(SOURCE)}\n",
            f"{SOURCE}  LEN: {len(SOURCE)}\n",
        ):
            with self.subTest(observed=observed[-24:]):
                self.assertIsNone(classify_dominated(observed, SOURCE))
        self.assertIsNotNone(
            classify_dominated(f"{SOURCE}\nLEN: {len(SOURCE)}\n", SOURCE)
        )

    def test_a_property_without_the_source_is_useful(self) -> None:
        self.assertIsNone(classify_dominated(f"LEN: {len(SOURCE)}\n", SOURCE))

    def test_a_bare_value_with_no_label_is_useful(self) -> None:
        self.assertIsNone(classify_dominated(f"{SOURCE}\n{len(SOURCE)}\n", SOURCE))

    def test_a_different_separator_is_useful(self) -> None:
        """`print("X:", v)` emits exactly one space; nothing else is named."""
        for line in (f"LEN:{len(SOURCE)}", f"LEN =  {len(SOURCE)}",
                     f"LEN:  {len(SOURCE)}", f"LEN\t{len(SOURCE)}"):
            with self.subTest(line=line):
                self.assertIsNone(classify_dominated(f"{SOURCE}\n{line}\n", SOURCE))

    def test_too_many_properties_is_useful(self) -> None:
        lines = "\n".join(f"LEN: {len(SOURCE)}" for _ in range(MAX_PROPERTIES + 1))
        self.assertIsNone(classify_dominated(f"{SOURCE}\n{lines}\n", SOURCE))

    def test_an_empty_source_proves_nothing(self) -> None:
        self.assertIsNone(classify_dominated("anything", ""))

    def test_absurdly_large_output_is_not_scanned(self) -> None:
        self.assertIsNone(classify_dominated("x" * 10_000_000, SOURCE))

    def test_a_targeted_calculation_stays_useful(self) -> None:
        """P. Real work that mentions no source at all."""
        for output in ("FINDING: pickle.loads on cookie",
                       "AST: Module(body=[Import, FunctionDef])",
                       "grep 'def': line 4"):
            with self.subTest(output=output):
                self.assertIsNone(classify_dominated(output, SOURCE))


class TrailingByteFidelityTests(unittest.TestCase):
    """A listing must show every line, including ones made of whitespace.

    `rstrip()` would treat form feeds, carriage returns, vertical tabs, spaces
    and tabs as absent -- they are ordinary bytes of a source file, and a
    listing that never showed the lines they belong to is incomplete. Only the
    single trailing newline a line-wise print drops may be tolerated.
    """

    def test_a_listing_omitting_whitespace_only_tail_lines_is_useful(self) -> None:
        for tail in ("\x0c\r\x0b", " \t \n \t \n", "\x0c\n", "   ", "\t\t"):
            with self.subTest(tail=tail):
                source = "a = 1\nb = 2\n" + tail
                listing = "1: a = 1\n2: b = 2"
                self.assertIsNone(
                    classify_dominated(
                        f"{listing}\nLEN: {len(source)}\n", source
                    )
                )

    def test_extra_trailing_newlines_are_not_stripped_away(self) -> None:
        source = "a\nb\n\n\n"
        listing = "1: a\n2: b"
        self.assertIsNone(
            classify_dominated(f"{listing}\nLEN: {len(source)}\n", source)
        )

    def test_one_trailing_newline_is_still_tolerated(self) -> None:
        """The control: the newline `splitlines()` drops is not a missing line."""
        source = "a\nb\n"
        listing = "1: a\n2: b"
        self.assertIsNotNone(
            classify_dominated(f"{listing}\nLEN: {len(source)}\n", source)
        )

    def test_blank_lines_between_properties_are_unexplained(self) -> None:
        """Tolerant parsing in a module that claims none.

        A blank line is a byte the program printed. Discarding it would mean
        the observation contained something the grammar never accounted for,
        which is the one thing a total proof cannot do -- and it is the seam
        through which a line could later be smuggled.
        """
        for observed in (
            f"{SOURCE}\n\nLEN: {len(SOURCE)}\n",
            f"{SOURCE}\nLEN: {len(SOURCE)}\n\n\n",
            f"{SOURCE}\nLEN: {len(SOURCE)}\n\nsha256: {_sha(SOURCE)}\n",
            f"{SOURCE}\nLEN: {len(SOURCE)}\n\n",
        ):
            with self.subTest(observed=observed[-30:]):
                self.assertIsNone(classify_dominated(observed, SOURCE))
        # The control: exactly one separator and one trailing newline is the
        # shape `print` produces, and it is accepted.
        self.assertIsNotNone(
            classify_dominated(f"{SOURCE}\nLEN: {len(SOURCE)}\n", SOURCE)
        )


class ComplexityTests(unittest.TestCase):
    """Bounded host CPU: the sandbox timeout does not cover this code.

    `_split_numbered` once retried every shorter prefix, re-joining and
    re-stripping each -- quadratic, and reachable from ordinary output: a model
    printing a numbered listing with one trailing line paid it, in the host
    process, after the sandbox had already returned and with no timeout around
    it.
    """

    def test_numbered_shaped_output_is_classified_in_linear_time(self) -> None:
        """The costly shape: every prefix parses, none of them matches.

        A trailing non-numbered line is cheap -- the strip fails at once on
        every prefix. What is expensive is output that is numbered
        consecutively all the way down and simply is not this source: each
        prefix parses successfully and only the final comparison rejects it,
        so a search over prefixes does the full parse once per line. The
        candidate is kept inside `_too_large` so the search is genuinely
        reached rather than short-circuited.
        """
        import time

        timings = []
        for count in (1000, 2000, 4000):
            source = "\n".join(f"other {i}" for i in range(count)) + "\n"
            candidate = "\n".join(f"{i}: line {i}" for i in range(count))
            started = time.monotonic()
            self.assertIsNone(classify_dominated(candidate, source))
            timings.append(time.monotonic() - started)
        # Quadratic here measured 0.8s / 2.2s / 9.9s; linear is milliseconds.
        self.assertLess(max(timings), 1.0)

    def test_a_real_listing_is_still_recognised(self) -> None:
        source = "x = 1\n" * 50
        listing = "\n".join(
            f"{i:3}: {line}" for i, line in enumerate(source.splitlines())
        )
        self.assertIsNotNone(
            classify_dominated(
                f"{listing}\nTOTAL_LINES: {len(source.splitlines())}\n", source
            )
        )


class SecurityTests(unittest.TestCase):
    """R. Bounded, deterministic, non-executing."""

    def test_the_module_never_evaluates(self) -> None:
        import ast

        path = ROOT / "src" / "orbit" / "runtime" / "analysis_source_dominance.py"
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
        code = ast.unparse(tree)
        for banned in ("eval(", "exec(", "literal_eval", "__import__",
                       "subprocess", "os.system", "pickle"):
            self.assertNotIn(banned, code, banned)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(
                    node.func.id, {"eval", "exec", "compile", "__import__", "open"}
                )

    def test_hostile_property_lines_are_safe(self) -> None:
        import time

        hostile = [
            f"{SOURCE}\n" + "LEN: " + "9" * 100_000,
            f"{SOURCE}\nHEX" + "9" * 100_000 + ": ab",
            f"{SOURCE}\n" + "A" * 100_000 + ": 1",
            f"{SOURCE}\n\x00\x00: 1",
            f"{SOURCE}\nLEN: __import__('os').system('id')",
            f"{SOURCE}\n" + "\n".join(":" for _ in range(10_000)),
            f"{SOURCE}\nHEX5: " + "z" * 10_000,
        ]
        started = time.monotonic()
        for candidate in hostile:
            with self.subTest(candidate=candidate[:30]):
                self.assertIsNone(classify_dominated(candidate, SOURCE))
        self.assertLess(time.monotonic() - started, 10.0)

    def test_a_numbered_listing_search_terminates(self) -> None:
        import time

        started = time.monotonic()
        classify_dominated("\n".join(f"{i}: x" for i in range(5000)), SOURCE)
        self.assertLess(time.monotonic() - started, 10.0)


class LiveObservedPatternTests(unittest.TestCase):
    """The exact forms the last live run produced, judged individually."""

    SAMPLE = ROOT / "workdir" / "samples" / "vulnerable_service.py"

    def setUp(self) -> None:
        if not self.SAMPLE.exists():
            self.skipTest("sample missing")
        self.src = self.SAMPLE.read_text()

    def test_action_1_repr_plus_len(self) -> None:
        self.assertIsNotNone(
            classify_dominated(f"{self.src!r}\nLEN: {len(self.src)}\n", self.src)
        )

    def test_action_2_plain_plus_len(self) -> None:
        self.assertIsNotNone(
            classify_dominated(f"{self.src}\nLEN: {len(self.src)}\n", self.src)
        )

    def test_action_3_marker_between_source_and_len_stays_useful(self) -> None:
        """`===END===` is a component the grammar does not name."""
        self.assertIsNone(
            classify_dominated(
                f"{self.src}\n===END===\nLEN: {len(self.src)}\n", self.src
            )
        )

    def test_action_5_hex_prefix_plus_a_bytes_repr_stays_useful(self) -> None:
        """`BYTES: b'...'` is a range representation the grammar does not name."""
        hexed = self.src[:200].encode("utf-8").hex()
        observed = f"HEX200: {hexed}\nBYTES: {self.src.encode()[:50]!r}\n"
        self.assertIsNone(classify_dominated(observed, self.src))

    def test_action_6_md5_and_counts_stay_useful(self) -> None:
        observed = (
            f"{self.src}\n"
            f"sha256: {_sha(self.src)}\n"
            f"md5: {hashlib.md5(self.src.encode()).hexdigest()}\n"
            "nonascii_count: 0\n"
        )
        self.assertIsNone(classify_dominated(observed, self.src))

    def test_action_7_ast_output_stays_useful(self) -> None:
        self.assertIsNone(
            classify_dominated("FUNC find_user @ line 14: doc='...'", self.src)
        )


if __name__ == "__main__":
    unittest.main()


class RuntimeSeamTests(unittest.TestCase):
    """The gate, end to end: dominance reuses the equivalence seam exactly.

    No parallel progress system -- a dominated observation takes the same path
    an equivalent one does, so the ledger, the accounting and the provenance
    are the ones already qualified.
    """

    def setUp(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_reacq", ROOT / "tests" / "test_analysis_reacquisition_runtime.py"
        )
        self.helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helpers)

    def _runtime(self):
        case = self.helpers._Case()
        case.setUp() if hasattr(case, "setUp") else None
        runtime = case._runtime()
        self.addCleanup(runtime.close)
        self._case = case
        return runtime

    def _cover(self, runtime) -> None:
        coverage = runtime.plan_source_coverage()
        self.assertTrue(coverage.covered)
        runtime.cover_source(coverage)
        self._case.backend.chat_calls.clear()

    def _step(self, runtime, result):
        from unittest import mock

        from orbit.runtime import analysis_runtime as module

        with mock.patch.object(module, "execute_analysis", lambda **kw: result):
            return runtime.step("go")

    def test_a_dominated_observation_consumes_no_action(self) -> None:
        source = self.helpers.SOURCE
        runtime = self._runtime()
        self._cover(runtime)
        before = runtime.actions_executed
        step = self._step(
            runtime,
            self.helpers._result(stdout=f"{source}\nLEN: {len(source)}"),
        )
        self.assertIsNotNone(step.suppressed_duplicate_of)
        self.assertEqual(runtime.actions_executed, before)
        self.assertEqual(runtime.suppressed_duplicates, 1)
        self.assertEqual(runtime.model_calls, 2)  # cover + this step

    def test_the_ledger_sees_no_progress(self) -> None:
        from orbit.runtime.analysis_progress import NO_PROGRESS, ProgressLedger

        source = self.helpers.SOURCE
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(
            runtime,
            self.helpers._result(stdout=f"{source}\nLEN: {len(source)}"),
        )
        self.assertEqual(
            ProgressLedger().classify(1, step).classification, NO_PROGRESS
        )

    def test_provenance_names_the_verified_properties(self) -> None:
        from orbit.runtime.analysis_source_dominance import SOURCE_DOMINATED

        source = self.helpers.SOURCE
        runtime = self._runtime()
        self._cover(runtime)
        step = self._step(
            runtime,
            self.helpers._result(stdout=f"{source}\nLEN: {len(source)}"),
        )
        record = runtime.evidence_store.records[step.suppressed_duplicate_of]
        self.assertEqual(record.metadata["suppressed_as"], SOURCE_DOMINATED)
        self.assertEqual(record.metadata["verified_properties"], [TEXT_LENGTH])
        # The raw output stays retained and re-attestable.
        raw_id = record.metadata["raw_output_evidence_id"]
        self.assertIsNotNone(runtime.evidence_store.reattest_exact(raw_id))

    def test_an_artifact_defeats_dominance(self) -> None:
        """A written file is new state whatever the stdout says."""
        from orbit.runtime.analysis_sandbox import DerivedArtifact

        source = self.helpers.SOURCE
        runtime = self._runtime()
        self._cover(runtime)
        written = DerivedArtifact(name="notes.txt", size_bytes=5, sha256="f" * 64)
        step = self._step(
            runtime,
            self.helpers._result(
                stdout=f"{source}\nLEN: {len(source)}", artifacts=[written]
            ),
        )
        self.assertIsNone(step.suppressed_duplicate_of)

    def test_nothing_is_dominated_without_coverage(self) -> None:
        """O. Without COVER the model was never given the source."""
        source = self.helpers.SOURCE
        runtime = self._runtime()
        before = runtime.actions_executed
        step = self._step(
            runtime,
            self.helpers._result(stdout=f"{source}\nLEN: {len(source)}"),
        )
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertEqual(runtime.actions_executed, before + 1)

    def test_a_failed_or_altered_action_is_never_dominated(self) -> None:
        """N. Every guard equivalence already applies, applies here too."""
        source = self.helpers.SOURCE
        stdout = f"{source}\nLEN: {len(source)}"
        for label, result in (
            ("error", self.helpers._result(stdout=stdout, status="error")),
            ("stderr", self.helpers._result(stdout=stdout, stderr="warn")),
            ("truncated", self.helpers._truncated_result(stdout)),
            ("replaced", self.helpers._replaced_result(stdout)),
        ):
            with self.subTest(label=label):
                runtime = self._runtime()
                self._cover(runtime)
                step = self._step(runtime, result)
                self.assertIsNone(step.suppressed_duplicate_of)

    def test_the_repeated_four_way_sequence_never_progresses(self) -> None:
        """Q. The live shapes in a row, none of them useful."""
        from orbit.runtime.analysis_progress import NO_PROGRESS, ProgressLedger

        source = self.helpers.SOURCE
        runtime = self._runtime()
        self._cover(runtime)
        listing = _numbered(source)
        outputs = [
            f"{source}\nLEN: {len(source)}",
            f"{source!r}\nsha256: {_sha(source)}",
            f"{listing}\nTOTAL_LINES: {len(source.splitlines())}",
            f"{source}\nHEX20: {source[:20].encode('utf-8').hex()}",
        ]
        ledger = ProgressLedger()
        for index, stdout in enumerate(outputs, 1):
            self._case.backend.code = f"# variant {index}\nprint(1)"
            step = self._step(runtime, self.helpers._result(stdout=stdout))
            self.assertIsNotNone(step.suppressed_duplicate_of, stdout[:40])
            self.assertEqual(
                ledger.classify(index, step).classification, NO_PROGRESS
            )
        self.assertEqual(runtime.actions_executed, 0)
        self.assertEqual(runtime.suppressed_duplicates, 4)

    def test_a_legitimate_action_still_runs_after_dominance(self) -> None:
        """P. Suppression is per-observation, never a mode the run enters."""
        source = self.helpers.SOURCE
        runtime = self._runtime()
        self._cover(runtime)
        self._case.backend.code = "# one\nprint(1)"
        self._step(runtime, self.helpers._result(stdout=f"{source}\nLEN: {len(source)}"))
        self._case.backend.code = "# two\nprint(2)"
        step = self._step(
            runtime, self.helpers._result(stdout="FINDING: os.environ read")
        )
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertEqual(runtime.actions_executed, 1)

    def test_global_ceilings_are_unchanged(self) -> None:
        from orbit.runtime import analysis_runtime as module

        self.assertEqual(module.MAX_AUTONOMOUS_ACTIONS, 12)
        self.assertEqual(module.SOFT_MAX_AUTONOMOUS_ACTIONS, 8)
        self.assertEqual(module.MAX_AUTONOMOUS_MODEL_CALLS, 15)
