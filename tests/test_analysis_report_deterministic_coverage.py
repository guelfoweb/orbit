"""The report must not silently drop a fact the runtime decoded exactly.

A decoded stage IS the finding: a dropper's fetch/write/run chain lives in the
body of its decoded script, not its first line. The deterministic transform
appendix -- which grounds the REPORT prompt and is also appended to the answer --
renders a decoded stage in full up to a bounded limit, so that chain reaches the
narrative rather than being cut to a 120-char prefix.

These tests exercise the rendering contract deterministically (no model): full
inline within the bound, digest-only past it, URI survival, the per-appendix
total budget, single rendering, and provenance. They use synthetic stages with
no IBAN-specific identifiers.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orbit.runtime.analysis_runtime import (
    TRANSFORM_INLINE_CHARS,
    TRANSFORM_INLINE_TOTAL_BUDGET,
    AnalysisRuntime,
    acquire_analysis_source,
)
from orbit.runtime.evidence import EvidenceStore


def _fcc_source(plaintext: str, *, const: int = 1000, var: str = "QQ",
                target: str = "PP", evald: bool = True) -> str:
    """A synthetic String.fromCharCode(N-const) artifact that decodes to
    `plaintext`. Generic: no IBAN names/constants."""
    args = ",".join(f"{const + ord(c)}-{var}" for c in plaintext)
    tail = f"eval({target});" if evald else f"use({target});"
    return f"var {var}={const};\nvar {target} = String.fromCharCode({args});\n{tail}\n"


class _Base(unittest.TestCase):
    def _runtime(self, source_text: str) -> AnalysisRuntime:
        tmp = tempfile.mkdtemp(prefix="orbit-rdc-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        p = Path(tmp) / "a.js"
        p.write_text(source_text, encoding="utf-8")
        source = acquire_analysis_source(p, Path(tmp) / "owned")
        store = EvidenceStore(root=Path(tmp) / "evidence")
        rt = AnalysisRuntime(backend=None, source=source, evidence_store=store)
        self.addCleanup(rt.close)
        return rt


class InlineCoverageTests(_Base):
    def test_full_decoded_body_is_inlined_within_bound(self):
        # A decoded script longer than the old 400-char prefix but within the
        # new bound is rendered in full, so its whole behaviour is in the report.
        body = (
            'var o = new ActiveXObject("Some.COMObject");\n'
            'o.DoThing("arg1", "arg2");\n'
            'var f = o.Path("C:/tmp/dropped.bin");\n'
            "o.Execute(f);\n"
        ) * 5  # > old 400-char prefix, < 2048 inline bound
        rt = self._runtime(_fcc_source(body))
        appendix = rt.transform_appendix()
        self.assertGreater(len(body), 400)
        self.assertLess(len(body), TRANSFORM_INLINE_CHARS)
        # every line of the decoded body is present (not just the first 120 chars)
        self.assertIn("Some.COMObject", appendix)
        self.assertIn("C:/tmp/dropped.bin", appendix)
        self.assertIn("o.Execute(f)", appendix)

    def test_stage_past_bound_is_digest_only_but_uri_survives(self):
        uri = "http://synthetic.invalid/x?t=1"
        # A space delimits the filler from the URI so the URI extractor gets the
        # URI alone (a real decoded payload delimits its literals the same way).
        body = ("A " * (TRANSFORM_INLINE_CHARS // 2 + 200)) + uri
        rt = self._runtime(_fcc_source(body))
        appendix = rt.transform_appendix()
        self.assertGreater(len(body), TRANSFORM_INLINE_CHARS)
        self.assertNotIn(body, appendix)  # not inlined whole
        self.assertIn(f"{len(body)} chars", appendix)  # named by length
        self.assertIn(f"decoded URI: {uri}", appendix)  # URI survives

    def test_provenance_preserved(self):
        rt = self._runtime(_fcc_source('new ActiveXObject("X.Y");'))
        appendix = rt.transform_appendix()
        stage, record = rt.transform_stages[0]
        self.assertIn(record.evidence_id, appendix)  # evidence id
        self.assertIn(stage.output_sha256, appendix)  # output sha
        self.assertIn(f"offset {stage.offset}", appendix)  # source location

    def test_single_rendering(self):
        rt = self._runtime(_fcc_source('new ActiveXObject("X.Y");'))
        appendix = rt.transform_appendix()
        self.assertEqual(appendix.count("## Deterministic transformations"), 1)

    def test_deterministic_sections_render_each_part_once(self):
        # A fromCharCode stage whose decode carries a URL, so both sections are
        # present. Each must appear exactly once (no duplicated indicators or
        # transformations block).
        rt = self._runtime(_fcc_source('x="http://synthetic.invalid/p";'))
        sections = rt.deterministic_sections()
        self.assertEqual(sections.count("## Verified indicators"), 1)
        self.assertEqual(sections.count("## Deterministic transformations"), 1)


class TotalBudgetTests(_Base):
    def test_many_stages_are_bounded_by_total_budget(self):
        # Many distinct decoded stages, each within the per-stage bound but
        # together exceeding the total budget: earlier ones inline, later ones
        # fall back to digest, so the appendix cannot explode.
        big = "Z" * 1500  # < per-stage bound
        calls = []
        for i in range(12):
            # distinct output per call so each is its own stage
            pt = f"{big}{i:04d}"
            args = ",".join(f"{1000 + ord(c)}-QQ" for c in pt)
            calls.append(f"String.fromCharCode({args})")
        src = "var QQ=1000;\n" + ";\n".join(f"var v{i}={c}" for i, c in enumerate(calls)) + ";\n"
        rt = self._runtime(src)
        appendix = rt.transform_appendix()
        # Count verbatim-inlined stages (an `output: ` line NOT of the digest
        # `chars, begins` form). Total inlined bytes must respect the budget, so
        # the number of fully-inlined ~1504-char stages is bounded.
        import re

        inlined = re.findall(r"^  output: (?!\d+ chars, begins)(.*)$", appendix, re.M)
        total_inlined = sum(len(x) for x in inlined)
        self.assertLessEqual(total_inlined, TRANSFORM_INLINE_TOTAL_BUDGET)
        # and at least one stage must have fallen back to the digest form.
        self.assertIn("chars, begins", appendix)


class GenericityTests(_Base):
    def test_no_sample_specific_tokens_in_rendering_code(self):
        # The appendix mechanism must carry no IBAN-specific identifier.
        import orbit.runtime.analysis_runtime as m

        src = Path(m.__file__).read_text(encoding="utf-8")
        # isolate the transform_appendix method text
        start = src.index("def transform_appendix")
        end = src.index("def ", start + 10)
        block = src[start:end]
        for token in ("J7f", "MMGCLZ", "productoslili", "TKFSIK", "IBAN",
                      "5d51e765", "86e23fa"):
            self.assertNotIn(token, block)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
