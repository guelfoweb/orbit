"""A large extracted source is windowed to fit, never admitted over ctx.

ANALYSIS-OVERSIZED-SOURCE-WINDOW-CLOSURE-1. The frozen Office VBA module is
~55 KB -- larger than the 8192 context. Rehydrating it whole builds a prompt
that cannot be admitted, and the run then ended on a ContextAdmissionError. The
runtime now delivers the largest line-aligned exact HEAD window of an extracted
source that fits the remaining budget, sized by the real backend tokenizer, with
provenance -- so no ctx-overflowing call is constructed. A decoded stage (a
computed result) is never windowed; only extracted source (read material) is.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import TokenCount  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    ANALYSIS_OFFICE_PHASE,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
    ContextAdmissionError,
)
from orbit.runtime.context_manager import (  # noqa: E402
    DEFAULT_NEXT_ACTION_RESERVE,
    DEFAULT_SAFETY_MARGIN,
)
from orbit.runtime.evidence import EvidenceStore, rehydrated_evidence_block  # noqa: E402

CTX = 8192
DOC = ROOT / "workdir" / "samples" / (
    "99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc"
)


class _Tokenizer:
    """A deterministic exact-admission backend: ~`per_tok` chars per token."""

    thinking = False

    def __init__(self, per_tok: float = 2.5) -> None:
        self.per_tok = per_tok
        self.calls = 0

    def supports_exact_context_admission(self) -> bool:
        return True

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        self.calls += 1
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(
            tokens=int(40 + chars / self.per_tok), context_tokens=CTX,
            rendered_hash="a" * 64, token_hash="b" * 64,
        )


def _runtime(backend, data: bytes = b"x = 1\n") -> AnalysisRuntime:
    ws = AnalysisWorkspace.create()
    p = ws.source_root / "a.bin"
    p.write_bytes(data)
    rt = AnalysisRuntime(
        backend=backend,
        source=AnalysisSource(
            snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data), original_path=str(p),
        ),
        evidence_store=EvidenceStore(root=ws.root / "evidence"),
        workspace=ws,
    )
    return rt


def _add_source(rt, chars: int):
    """A large EXTRACTED-SOURCE record (Office phase), line-structured."""
    body = "".join(f"Line {i:05d} of the extracted macro source\n" for i in range(chars // 40 + 1))
    return rt.evidence_store.add(
        "execute_analysis", body,
        metadata={"produced_by_phase": ANALYSIS_OFFICE_PHASE,
                  "office_module_name": "ThisDocument",
                  "tool_call_id": "office_vba_1", "user_turn_id": "turn_0"},
    ), body


def _input_limit(output_reserve=512, next_action=DEFAULT_NEXT_ACTION_RESERVE):
    return CTX - output_reserve - next_action - DEFAULT_SAFETY_MARGIN


class WindowingTests(unittest.TestCase):
    def test_oversized_source_is_windowed_and_admits(self) -> None:
        backend = _Tokenizer()
        rt = _runtime(backend)
        self.addCleanup(rt.close)
        rec, _ = _add_source(rt, 60_000)  # far larger than ctx
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": f"show source evidence:{rec.evidence_id}"}]
        # T1: _admit does not raise; the assembled prompt fits input_limit.
        admitted = rt._admit(msgs, max_tokens=512, tools=None)
        joined = admitted[-1]["content"]
        self.assertIn("windowed_to_fit_context", joined)
        tokens = backend.count_chat_tokens(admitted).tokens
        self.assertLessEqual(tokens, _input_limit())

    def test_window_is_the_largest_line_aligned_range_that_fits(self) -> None:
        # T4: one more line would exceed the budget.
        backend = _Tokenizer()
        rt = _runtime(backend)
        self.addCleanup(rt.close)
        rec, body = _add_source(rt, 60_000)
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": f"evidence:{rec.evidence_id}"}]
        out, _ = rt._with_evidence_rehydration(
            msgs, tools=None, next_action_reserve=DEFAULT_NEXT_ACTION_RESERVE,
            output_reserve=512)
        kept = rt.last_rehydration_diag["windowed"][0][1]
        total = rt.last_rehydration_diag["windowed"][0][2]
        self.assertGreater(kept, 0)
        self.assertLess(kept, total)
        # The window is an exact line-aligned prefix of the real source.
        prefix = "".join(body.splitlines(keepends=True)[:kept])
        self.assertIn(prefix.rstrip("\n").splitlines()[-1], out[-1]["content"])

    def test_window_size_tracks_the_tokenizer_not_chars(self) -> None:
        # T3: a denser tokenizer (more chars/token) admits MORE lines.
        rec_lines = {}
        for per_tok in (1.5, 4.0):
            backend = _Tokenizer(per_tok)
            rt = _runtime(backend)
            rec, _ = _add_source(rt, 60_000)
            rt._with_evidence_rehydration(
                [{"role": "system", "content": "S"},
                 {"role": "user", "content": f"evidence:{rec.evidence_id}"}],
                tools=None, next_action_reserve=DEFAULT_NEXT_ACTION_RESERVE,
                output_reserve=512)
            rec_lines[per_tok] = rt.last_rehydration_diag["windowed"][0][1]
            rt.close()
        self.assertGreater(rec_lines[4.0], rec_lines[1.5])

    def test_repeated_identical_request_is_deterministic(self) -> None:
        # T7: the same impossible request yields the same bounded window, so a
        # repeat is a duplicate, not an ever-growing loop.
        backend = _Tokenizer()
        rt = _runtime(backend)
        self.addCleanup(rt.close)
        rec, _ = _add_source(rt, 60_000)
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": f"evidence:{rec.evidence_id}"}]
        a, _ = rt._with_evidence_rehydration(
            msgs, tools=None, next_action_reserve=256, output_reserve=512)
        b, _ = rt._with_evidence_rehydration(
            msgs, tools=None, next_action_reserve=256, output_reserve=512)
        self.assertEqual(a[-1]["content"], b[-1]["content"])

    def test_oversized_source_before_other_ids_still_fits(self) -> None:
        # MAJOR-1 regression: the oversized source is named FIRST, then several
        # other ids. The trailing withheld-refs must be counted in the window
        # budget so the assembled prompt still fits and _admit does not re-raise.
        backend = _Tokenizer()
        rt = _runtime(backend)
        self.addCleanup(rt.close)
        big, _ = _add_source(rt, 60_000)
        others = [
            rt.evidence_store.add(
                "execute_analysis", f"small evidence {i}\n" * 3,
                metadata={"produced_by_phase": "analysis_transform",
                          "tool_call_id": f"t{i}", "user_turn_id": "turn_0"})
            for i in range(7)
        ]
        named = " ".join([f"evidence:{big.evidence_id}"]
                         + [f"evidence:{o.evidence_id}" for o in others])
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": named}]
        admitted = rt._admit(msgs, max_tokens=512, tools=None)  # must not raise
        self.assertLessEqual(backend.count_chat_tokens(admitted).tokens, _input_limit())
        block = admitted[-1]["content"]
        self.assertIn("windowed_to_fit_context", block)
        # the trailing ids are referenced (withheld), not inlined
        for o in others:
            self.assertIn(o.evidence_id, block)

    def test_decoded_stage_is_never_windowed(self) -> None:
        # A computed decoded stage is delivered whole (or the opening withdraws);
        # it is never partially delivered.
        backend = _Tokenizer()
        rt = _runtime(backend)
        self.addCleanup(rt.close)
        body = "decoded " * 20000  # large, but a decoded stage, not source
        rec = rt.evidence_store.add(
            "execute_analysis", body,
            metadata={"produced_by_phase": "analysis_transform",
                      "tool_call_id": "t1", "user_turn_id": "turn_0"})
        out, _ = rt._with_evidence_rehydration(
            [{"role": "system", "content": "S"},
             {"role": "user", "content": f"evidence:{rec.evidence_id}"}],
            tools=None, next_action_reserve=256, output_reserve=512)
        self.assertNotIn("windowed_to_fit_context", out[-1]["content"])

    def test_unsatisfiable_history_still_refuses_not_overflows(self) -> None:
        # T5: when even a minimal window plus the history cannot fit, admission
        # refuses (ContextAdmissionError) before any backend generation call.
        backend = _Tokenizer()
        rt = _runtime(backend)
        self.addCleanup(rt.close)
        rec, _ = _add_source(rt, 60_000)
        huge_history = "H" * (CTX * 3)  # history alone blows the budget
        msgs = [{"role": "system", "content": huge_history},
                {"role": "user", "content": f"evidence:{rec.evidence_id}"}]
        with self.assertRaises(ContextAdmissionError):
            rt._admit(msgs, max_tokens=512, tools=None)


class ByteEquivalenceTests(unittest.TestCase):
    def test_small_fitting_source_is_byte_identical_to_baseline(self) -> None:
        # T13: a record that fits is rehydrated exactly as before.
        backend = _Tokenizer()
        rt = _runtime(backend)
        self.addCleanup(rt.close)
        rec = rt.evidence_store.add(
            "execute_analysis", "small\nsource\n",
            metadata={"produced_by_phase": ANALYSIS_OFFICE_PHASE,
                      "office_module_name": "M",
                      "tool_call_id": "office_vba_1", "user_turn_id": "turn_0"})
        out, _ = rt._with_evidence_rehydration(
            [{"role": "system", "content": "S"},
             {"role": "user", "content": f"evidence:{rec.evidence_id}"}],
            tools=None, next_action_reserve=256, output_reserve=512)
        baseline = rehydrated_evidence_block(rt.evidence_store, (rec.evidence_id,))
        self.assertEqual(out[-1]["content"], baseline)

    def test_non_orbit_backend_is_unchanged(self) -> None:
        class NonOrbit:
            thinking = False
            def supports_exact_context_admission(self):
                return False
        rt = _runtime(NonOrbit())
        self.addCleanup(rt.close)
        rec, _ = _add_source(rt, 60_000)
        out, _ = rt._with_evidence_rehydration(
            [{"role": "system", "content": "S"},
             {"role": "user", "content": f"evidence:{rec.evidence_id}"}],
            tools=None)
        baseline = rehydrated_evidence_block(rt.evidence_store, (rec.evidence_id,))
        self.assertEqual(out[-1]["content"], baseline)


@unittest.skipUnless(DOC.exists(), "Office sample required")
class RealOfficeTests(unittest.TestCase):
    def test_module_windows_and_authority_and_chain_intact(self) -> None:
        backend = _Tokenizer()
        data = DOC.read_bytes()
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "s.doc"
        p.write_bytes(data)
        rt = AnalysisRuntime(
            backend=backend,
            source=AnalysisSource(
                snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(p)),
            evidence_store=EvidenceStore(root=ws.root / "evidence"), workspace=ws)
        self.addCleanup(rt.close)
        rt._run_transform_preflight()
        mod, rec = rt.office_modules[0]
        # T9: source authority intact (exact module sha).
        self.assertEqual(
            hashlib.sha256(mod.source.encode("utf-8", "surrogatepass")).hexdigest(),
            "d034bd8381f4663af80a7f516cf104ae9fd2a49581379d6a7683bfdf1c8713f0")
        # T10/T11: chain + stage + C2 intact in deterministic grounding.
        grounding = rt.deterministic_sections()
        self.assertIn("Document_Open", grounding)
        self.assertIn("reaches a Shell execution call", grounding)
        self.assertIn("http://185.189.58.222/x.exe", grounding)
        self.assertIn("PHfW.exe", grounding)
        stage_shas = {s.output_sha256 for s, _ in rt.transform_stages}
        self.assertIn(
            "f1fa67e3f558443e9604b0f0b26715dc8065c53c92db7acbab2e6d32874e5c21",
            stage_shas)
        # T1 on the real module: rehydrating it admits (windowed), never over ctx.
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": f"evidence:{rec.evidence_id}"}]
        admitted = rt._admit(msgs, max_tokens=512, tools=None)
        self.assertLessEqual(backend.count_chat_tokens(admitted).tokens, _input_limit())
        self.assertIn("windowed_to_fit_context", admitted[-1]["content"])
        # T8: the window is explicitly partial (never a full-source claim).
        self.assertIn(f"of {rec.raw_lines}", admitted[-1]["content"])


if __name__ == "__main__":
    unittest.main()
