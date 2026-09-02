"""Only a provable full-source reacquisition may be treated as adding nothing.

ANALYSIS-SOURCE-EQUIVALENCE-1. After COVER supplies the complete artifact, the
observed behaviour is that the model reads the same bytes back anyway -- as a
repr, as plain text, as a numbered listing. Those observations carry nothing
the session does not already hold.

The rule is exact proof or nothing. Every recognizer reconstructs candidate
bytes and compares them to the artifact's own; one differing byte, one extra
printed line, one reordered line, and the answer is no. The failure that must
never happen is suppressing something new, so the tests below spend far more
effort on what must NOT be recognised than on what must.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_source_identity import (  # noqa: E402
    ARTIFACT,
    NUMBERED,
    RAW,
    REPR,
    classify_artifacts,
    classify_output,
    decode_repr_literal,
    strip_line_numbers,
)

SOURCE = (
    "from __future__ import annotations\n"
    "\n"
    "import os\n"
    "\n"
    "\n"
    "def handler(name: str) -> str:\n"
    '    """Do a thing."""\n'
    "    return os.environ.get(name, 'fallback')\n"
)


class _Artifact:
    def __init__(self, name: str, sha256: str, size_bytes: int) -> None:
        self.name = name
        self.sha256 = sha256
        self.size_bytes = size_bytes


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RawRecognizerTests(unittest.TestCase):
    """A. The output is the source, byte for byte."""

    def test_exact_source_is_recognised(self) -> None:
        found = classify_output(SOURCE, SOURCE)
        self.assertIsNotNone(found)
        self.assertEqual(found.recognizer, RAW)

    def test_print_adds_one_newline_and_that_is_tolerated(self) -> None:
        """`print(data)` emits the text plus exactly one newline."""
        self.assertIsNotNone(classify_output(SOURCE + "\n", SOURCE))

    def test_two_trailing_newlines_are_not_the_source(self) -> None:
        """Only the one newline `print` itself adds; more is different text."""
        self.assertIsNone(classify_output(SOURCE + "\n\n", SOURCE))


class ReprRecognizerTests(unittest.TestCase):
    """B. The output is `repr(source)` and decodes back exactly."""

    def test_repr_of_the_source_is_recognised(self) -> None:
        found = classify_output(repr(SOURCE) + "\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertEqual(found.recognizer, REPR)

    def test_repr_round_trips_for_every_escape_repr_emits(self) -> None:
        awkward = "a\\b\tc\nd\re'f\"g\x00h\x1biéj\U0001f600k"
        self.assertEqual(decode_repr_literal(repr(awkward)), awkward)

    def test_a_repr_with_more_than_one_trailing_newline_is_refused(self) -> None:
        """`$` under `re.S` also matches before a final newline.

        That made `repr(source) + "\\n\\n"` match with the extra newline
        silently tolerated -- accepting a candidate that is the repr plus
        something else. The pattern is anchored with `\\A`/`\\Z` instead.
        """
        self.assertIsNone(classify_output(repr(SOURCE) + "\n\n", SOURCE))
        self.assertIsNone(classify_output(repr(SOURCE) + "\n \n", SOURCE))
        self.assertIsNone(classify_output(repr(SOURCE) + " ", SOURCE))
        # The single newline `print` adds is still accepted.
        self.assertIsNotNone(classify_output(repr(SOURCE) + "\n", SOURCE))

    def test_a_repr_of_different_text_is_not_recognised(self) -> None:
        self.assertIsNone(classify_output(repr(SOURCE + "x") + "\n", SOURCE))

    def test_forms_repr_never_emits_are_refused(self) -> None:
        """The decoder covers `repr(str)`, not the Python literal grammar."""
        for text in (
            "b'bytes'",           # a prefix
            "r'raw'",
            "f'formatted'",
            "'a' 'b'",            # implicit concatenation
            "'unterminated",
            "'trailing backslash\\",
            "'\\N{BULLET}'",      # valid Python, never from repr
            "'\\777'",            # octal
            "('tuple',)",
            "['list']",
        ):
            with self.subTest(text=text):
                self.assertIsNone(decode_repr_literal(text))

    def test_a_lone_unescaped_quote_cannot_end_the_literal_early(self) -> None:
        self.assertIsNone(decode_repr_literal("'a'b'"))

    def test_malformed_escapes_are_refused(self) -> None:
        for text in (r"'\xZZ'", r"'\x1'", r"'\u12'", r"'\U0011FFFF'", r"'\ud800'"):
            with self.subTest(text=text):
                self.assertIsNone(decode_repr_literal(text))


class NumberedRecognizerTests(unittest.TestCase):
    """C. A complete consecutive numbered listing, reversibly stripped."""

    def _numbered(self, text: str, start: int = 0, sep: str = ": ") -> str:
        return "\n".join(
            f"{i + start:3}{sep}{line}" for i, line in enumerate(text.splitlines())
        )

    def test_zero_based_listing_is_recognised(self) -> None:
        found = classify_output(self._numbered(SOURCE) + "\n", SOURCE)
        self.assertIsNotNone(found)
        self.assertEqual(found.recognizer, NUMBERED)

    def test_one_based_listing_is_recognised(self) -> None:
        self.assertIsNotNone(
            classify_output(self._numbered(SOURCE, start=1) + "\n", SOURCE)
        )

    def test_a_partial_listing_is_refused_not_merely_unequal(self) -> None:
        """The comparison must be equality, never containment.

        A listing of the first two lines of a three-line file reconstructs to a
        PREFIX of the source. An implementation testing `stripped in source`
        would suppress it -- discarding the fact that the third line, which may
        be the whole finding, was never shown.
        """
        source = "alpha\nbeta\nSECRET=evil.example.com\n"
        partial = "1: alpha\n2: beta"
        self.assertIsNone(classify_output(partial, source))
        # And the same for a suffix, which containment would also accept.
        self.assertIsNone(classify_output("1: beta\n2: SECRET=evil.example.com",
                                          source))

    def test_a_listing_starting_at_two_is_refused(self) -> None:
        """Starting at 2 says line 1 was not shown: a filtered view.

        Pinned separately from "any start" because the off-by-one is the
        plausible mistake -- a recognizer accepting 0, 1 or 2 looks reasonable
        and silently admits a listing that is missing its first line.
        """
        listing = "\n".join(
            f"{i + 2:3}: {line}" for i, line in enumerate(SOURCE.splitlines())
        )
        self.assertIsNone(classify_output(listing, SOURCE))
        self.assertIsNone(strip_line_numbers(listing))

    def test_a_listing_starting_elsewhere_is_refused(self) -> None:
        """Starting at 7 means this is a slice, not the whole file."""
        self.assertIsNone(classify_output(self._numbered(SOURCE, start=7), SOURCE))

    def test_a_gap_in_the_numbers_is_refused(self) -> None:
        lines = self._numbered(SOURCE).split("\n")
        del lines[3]
        self.assertIsNone(classify_output("\n".join(lines), SOURCE))

    def test_mixed_separators_are_refused(self) -> None:
        lines = self._numbered(SOURCE).split("\n")
        lines[2] = lines[2].replace(": ", " | ", 1)
        self.assertIsNone(classify_output("\n".join(lines), SOURCE))

    def test_an_unnumbered_line_is_refused(self) -> None:
        lines = self._numbered(SOURCE).split("\n")
        lines.append("summary: 8 lines")
        self.assertIsNone(classify_output("\n".join(lines), SOURCE))

    def test_wrong_numbers_are_refused_even_when_the_bodies_reconstruct(self) -> None:
        """Consecutiveness is load-bearing on its own.

        A deleted line changes the bodies too, so the equality check catches it
        independently -- which means only numbers that are wrong while the text
        is right can show that the consecutiveness check does any work. A
        listing whose numbers do not count is not a listing of a whole file:
        it could be a filtered view whose selection is itself the finding.
        """
        for numbers in ((0, 0, 0, 0, 0, 0, 0, 0), (0, 5, 9, 2, 1, 3, 4, 6),
                        (1, 1, 1, 1, 1, 1, 1, 1)):
            with self.subTest(numbers=numbers):
                candidate = "\n".join(
                    f"{n}: {line}"
                    for n, line in zip(numbers, SOURCE.splitlines())
                )
                self.assertIsNone(classify_output(candidate, SOURCE))

    def test_descending_numbers_are_refused(self) -> None:
        candidate = "\n".join(
            f"{n}: {line}"
            for n, line in zip(range(7, -1, -1), SOURCE.splitlines())
        )
        self.assertIsNone(classify_output(candidate, SOURCE))

    def test_only_number_padding_is_absorbed_before_a_listing(self) -> None:
        """Alignment whitespace belongs to the number; content does not.

        `f"{i:3}"` right-aligns, so leading spaces are part of the number's own
        rendering and absorbing them is reversible. Anything else in front --
        a heading, a marker, a digit that is not the line number -- stops the
        line matching and leaves the observation as evidence.
        """
        listing = "\n".join(
            f"{i:3}: {line}" for i, line in enumerate(SOURCE.splitlines())
        )
        for padding in (" ", "\t", "  "):
            with self.subTest(padding=padding):
                self.assertIsNotNone(classify_output(padding + listing, SOURCE))
        for content in ("x", " x", "0", " 0: ", "#", "== source ==\n"):
            with self.subTest(content=content):
                self.assertIsNone(classify_output(content + listing, SOURCE))

    def test_every_separator_form_is_reachable(self) -> None:
        """Each documented separator actually matches something.

        Checked by running the pattern rather than by parsing it: the
        alternation contains an escaped `|`, so splitting the source text on
        `|` mis-reads the grammar it is meant to audit.
        """
        from orbit.runtime.analysis_source_identity import _NUMBERED_LINE

        separators = {": ", " | ", "\t", ". ", "  ", " "}
        matched = set()
        for separator in separators:
            match = _NUMBERED_LINE.match(f"1{separator}body")
            self.assertIsNotNone(match, separator)
            matched.add(match.group("sep"))
        # Every form maps to a distinct captured separator: none is shadowed
        # into unreachability by an earlier alternative.
        self.assertEqual(matched, separators)

    def test_a_single_numbered_line_is_never_a_listing(self) -> None:
        """One line cannot be distinguished from a grep hit or a result index.

        `1: evil.example.com` against a one-line source is exactly the shape of
        a search result whose leading number is a match count -- real
        information about WHERE something was found. A listing needs at least
        two lines before its numbering means anything, so a single line is
        refused outright.
        """
        for candidate in (
            "1: evil.example.com",
            "1 evil.example.com",
            "  0: evil.example.com",
            "1\tevil.example.com",
        ):
            with self.subTest(candidate=candidate):
                self.assertIsNone(classify_output(candidate, "evil.example.com"))
                self.assertIsNone(strip_line_numbers(candidate))
        # Two lines is where a listing becomes meaningful.
        self.assertIsNotNone(classify_output("0: a\n1: b", "a\nb"))

    def test_stripping_returns_none_rather_than_guessing(self) -> None:
        self.assertIsNone(strip_line_numbers("no numbers here\nat all"))
        self.assertIsNone(strip_line_numbers("single line"))


class ArtifactRecognizerTests(unittest.TestCase):
    """D. A produced file whose bytes are the source."""

    def test_a_copy_is_recognised_by_digest(self) -> None:
        found = classify_artifacts(
            [_Artifact("copy.py", _sha(SOURCE), len(SOURCE.encode()))],
            _sha(SOURCE), len(SOURCE.encode()),
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.recognizer, ARTIFACT)

    def test_the_name_does_not_matter(self) -> None:
        self.assertIsNotNone(
            classify_artifacts(
                [_Artifact("anything.txt", _sha(SOURCE), len(SOURCE.encode()))],
                _sha(SOURCE), len(SOURCE.encode()),
            )
        )

    def test_a_different_digest_is_refused(self) -> None:
        self.assertIsNone(
            classify_artifacts(
                [_Artifact("copy.py", _sha(SOURCE + "x"), len(SOURCE.encode()))],
                _sha(SOURCE), len(SOURCE.encode()),
            )
        )

    def test_a_size_disagreeing_with_the_digest_is_refused(self) -> None:
        self.assertIsNone(
            classify_artifacts(
                [_Artifact("copy.py", _sha(SOURCE), 1)],
                _sha(SOURCE), len(SOURCE.encode()),
            )
        )

    def test_more_than_one_artifact_is_refused(self) -> None:
        """A second file is new state whatever the first one holds."""
        digest, size = _sha(SOURCE), len(SOURCE.encode())
        self.assertIsNone(
            classify_artifacts(
                [_Artifact("copy.py", digest, size),
                 _Artifact("notes.txt", _sha("notes"), 5)],
                digest, size,
            )
        )

    def test_no_artifacts_is_refused(self) -> None:
        self.assertIsNone(
            classify_artifacts([], _sha(SOURCE), len(SOURCE.encode()))
        )


class FailClosedTests(unittest.TestCase):
    """E-I. Everything that must stay ordinary evidence."""

    def test_one_byte_difference(self) -> None:
        self.assertIsNone(classify_output(SOURCE.replace("os", "0s", 1), SOURCE))

    def test_one_trailing_space_added(self) -> None:
        self.assertIsNone(classify_output(SOURCE + " ", SOURCE))

    def test_full_source_plus_one_computed_line(self) -> None:
        """F. The live case: the source AND something the model worked out."""
        self.assertIsNone(classify_output(f"{SOURCE}\nLEN: {len(SOURCE)}", SOURCE))

    def test_full_source_preceded_by_a_heading(self) -> None:
        self.assertIsNone(classify_output(f"=== source ===\n{SOURCE}", SOURCE))

    def test_repr_plus_a_computed_line(self) -> None:
        self.assertIsNone(
            classify_output(f"{repr(SOURCE)}\nLEN: {len(SOURCE)}", SOURCE)
        )

    def test_numbered_listing_plus_a_summary(self) -> None:
        numbered = "\n".join(
            f"{i:3}: {line}" for i, line in enumerate(SOURCE.splitlines())
        )
        self.assertIsNone(classify_output(f"{numbered}\nIMPORTS: ['os']", SOURCE))

    def test_partial_source(self) -> None:
        """G. Half the file is a real observation about which half."""
        self.assertIsNone(classify_output(SOURCE[: len(SOURCE) // 2], SOURCE))

    def test_source_with_the_last_line_dropped(self) -> None:
        self.assertIsNone(
            classify_output("\n".join(SOURCE.splitlines()[:-1]), SOURCE)
        )

    def test_reordered_lines(self) -> None:
        """H. Same bytes, different file."""
        lines = SOURCE.splitlines()
        lines[0], lines[-1] = lines[-1], lines[0]
        self.assertIsNone(classify_output("\n".join(lines), SOURCE))

    def test_normalised_whitespace(self) -> None:
        """I. Byte identity is the standard; a reflow is different text."""
        self.assertIsNone(classify_output(SOURCE.replace("\n\n", "\n"), SOURCE))
        self.assertIsNone(classify_output(SOURCE.replace("\n", "\r\n"), SOURCE))
        self.assertIsNone(classify_output(SOURCE.replace("    ", "\t"), SOURCE))

    def test_stripped_output(self) -> None:
        self.assertIsNone(classify_output(SOURCE.strip(), SOURCE))

    def test_legitimate_targeted_calculation(self) -> None:
        """J. Real work that happens to mention the source."""
        for output in (
            "SHA256: " + _sha(SOURCE),
            "IMPORTS: ['os']",
            "LINE COUNT: 8",
            "grep 'def': line 6",
            "AST: Module(body=[ImportFrom, Import, FunctionDef])",
        ):
            with self.subTest(output=output):
                self.assertIsNone(classify_output(output, SOURCE))

    def test_empty_output(self) -> None:
        self.assertIsNone(classify_output("", SOURCE))

    def test_an_empty_source_proves_nothing(self) -> None:
        self.assertIsNone(classify_output("anything", ""))

    def test_absurdly_large_output_is_not_scanned(self) -> None:
        self.assertIsNone(classify_output("x" * 10_000_000, SOURCE))


class SecurityTests(unittest.TestCase):
    """L. The decoder is a parser, never an interpreter."""

    def test_the_module_never_evaluates(self) -> None:
        """No banned construct in executable code.

        Docstrings are stripped first: the module docstring names
        `literal_eval` precisely to explain why it is NOT used, and prose about
        a rejected approach is not the approach.
        """
        import ast

        path = ROOT / "src" / "orbit" / "runtime" / "analysis_source_identity.py"
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
        for banned in (
            "eval(", "exec(", "literal_eval", "__import__",
            "subprocess", "os.system", "pickle",
        ):
            self.assertNotIn(banned, code, banned)
        # `compile` only as the builtin, never `re.compile`, which is the one
        # legitimate use and the reason a bare substring check misfires here.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(
                    node.func.id,
                    {"eval", "exec", "compile", "__import__", "open"},
                    f"forbidden builtin call: {node.func.id}",
                )

    def test_the_module_imports_nothing_that_can_execute(self) -> None:
        import ast

        path = ROOT / "src" / "orbit" / "runtime" / "analysis_source_identity.py"
        imported = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(
            imported <= {"re", "dataclasses", "__future__"},
            f"unexpected imports: {imported}",
        )

    def test_hostile_repr_shaped_input_is_refused_not_run(self) -> None:
        for hostile in (
            "__import__('os').system('id')",
            "'a' + __import__('os').popen('id').read()",
            "'\\x27); import os; os.system('id'); ('",
            "eval('1+1')",
            "'{}'.format(__import__('os'))",
        ):
            with self.subTest(hostile=hostile):
                self.assertIsNone(classify_output(hostile, SOURCE))

    def test_an_artifact_cannot_steer_the_numbered_parser(self) -> None:
        """Source lines that look like numbering do not make a listing."""
        numeric = "1: alpha\n2: beta\n3: gamma\n"
        # Read back verbatim, this is the source and matches as RAW.
        self.assertEqual(classify_output(numeric, numeric).recognizer, RAW)
        # Numbered on top of that, it still reconstructs exactly.
        listed = "\n".join(
            f"{i}: {line}" for i, line in enumerate(numeric.splitlines())
        )
        self.assertEqual(classify_output(listed, numeric).recognizer, NUMBERED)
        # But a DIFFERENT file whose lines merely look numbered is refused.
        self.assertIsNone(classify_output(listed, "0: other\n1: text\n"))

    def test_an_absurd_line_number_cannot_crash_the_parser(self) -> None:
        """Artifact-controlled input must never raise out of a recognizer.

        `int()` on a very long digit string raises rather than returning, so a
        printed line with a hundred thousand leading digits crashed the parser
        before the digit run was bounded. A line number is at most as large as
        the file has lines, so a long one cannot be a real listing anyway.
        """
        hostile = "9" * 100_000 + ": x\n" + "9" * 100_000 + ": y"
        self.assertIsNone(strip_line_numbers(hostile))
        self.assertIsNone(classify_output(hostile, SOURCE))

    def test_hostile_input_never_raises_from_any_recognizer(self) -> None:
        for hostile in (
            "9" * 50_000 + ": x\n" + "9" * 50_000 + ": y",
            "\\" * 100_000,
            "'" + "\\x41" * 50_000 + "'",
            "\x00" * 1000,
            "\r\n" * 10_000,
            "0" * 5000 + ": line",
        ):
            with self.subTest(hostile=hostile[:20]):
                self.assertIsNone(classify_output(hostile, SOURCE))
                self.assertIsNone(strip_line_numbers(hostile))
                decode_repr_literal(hostile)  # must not raise

    def test_decoding_is_bounded_and_terminates(self) -> None:
        import time

        started = time.monotonic()
        decode_repr_literal("'" + ("\\x41" * 200_000) + "'")
        self.assertLess(time.monotonic() - started, 10.0)


class LiveObservedPatternTests(unittest.TestCase):
    """The four patterns the live run actually produced, judged honestly."""

    SAMPLE = ROOT / "workdir" / "samples" / "vulnerable_service.py"

    def setUp(self) -> None:
        if not self.SAMPLE.exists():
            self.skipTest("sample missing")
        self.src = self.SAMPLE.read_text()

    def test_pattern_1_repr_with_a_length_line_stays_evidence(self) -> None:
        """The run printed `repr(data)` AND `LEN:`, so it is not bare."""
        observed = f"{self.src!r}\nLEN: {len(self.src)}"
        self.assertIsNone(classify_output(observed, self.src))

    def test_pattern_2_plain_with_a_length_line_stays_evidence(self) -> None:
        observed = f"{self.src}\nLEN: {len(self.src)}"
        self.assertIsNone(classify_output(observed, self.src))

    def test_pattern_3_numbered_listing_is_recognised(self) -> None:
        observed = "\n".join(
            f"{i:3}: {line}" for i, line in enumerate(self.src.splitlines())
        )
        found = classify_output(observed + "\n", self.src)
        self.assertIsNotNone(found)
        self.assertEqual(found.recognizer, NUMBERED)

    def test_pattern_4_copy_plus_computed_facts_stays_evidence(self) -> None:
        """The copy is recognisable, but the action also computed real facts."""
        observed = "WROTE 1495 bytes\nLINE COUNT: 46\nHAS 'def': 5"
        self.assertIsNone(classify_output(observed, self.src))

    def test_a_bare_repr_of_the_sample_is_recognised(self) -> None:
        self.assertIsNotNone(classify_output(repr(self.src) + "\n", self.src))

    def test_a_bare_read_of_the_sample_is_recognised(self) -> None:
        self.assertIsNotNone(classify_output(self.src + "\n", self.src))


if __name__ == "__main__":
    unittest.main()
