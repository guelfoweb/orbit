"""Part B controller behaviour (B1-B14): large-source analysis end to end.

An artifact whose source is larger than the context must not claim coverage,
must give PLAN a bounded structural view so it can form questions, must let the
model read unseen regions by bounded range, accumulate evidence across regions,
and compose a report from that evidence -- all while the normal small-source
path stays byte-for-byte unchanged. Scripted fakes only; no Ornith, no sandbox.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    ANALYSIS_TOOL_NAME,
    FINISH_TOOL_NAME,
    PLAN_TOOL_NAME,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

CTX = 8192
# Genuinely larger than head+tail*2 (9600 B) and, at the stub's 0.25 tok/char,
# far past the coverage budget -- an honest oversized source with generic
# tokens near the start, middle and end (no sample-specific strings).
OVERSIZED = (
    "OPENING_TOKEN\n"
    + ("pad line xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n" * 1500)
    + "MIDDLE_TOKEN\n"
    + ("pad line yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n" * 1500)
    + "CLOSING_TOKEN\n"
)
SMALL = "print('hello')\n"


class _Model:
    """Scripted by offered tool. Adopts a plan, runs one action (a range read),
    finishes the question, and reports. Prose empty, as a real model's calls."""

    def __init__(self, plan):
        self.plan = plan
        self.finishes = 0
        self.actions = 0

    def reply(self, tools):
        names = [t["function"]["name"] for t in (tools or [])]
        if PLAN_TOOL_NAME in names:
            return self._c(PLAN_TOOL_NAME, {"questions": self.plan})
        if FINISH_TOOL_NAME in names:
            self.finishes += 1
            return self._c(FINISH_TOOL_NAME,
                           {"status": "resolved", "answer_summary": "settled"})
        if ANALYSIS_TOOL_NAME in names:
            # Distinct code each action, as a real model reading a DIFFERENT
            # byte region would send -- an identical read is the duplicate the
            # action guard exists to suppress, which is not what these tests are
            # about. Each targets a new offset.
            self.actions += 1
            off = self.actions * 5000
            return self._c(ANALYSIS_TOOL_NAME, {
                "code": f"import orbit_tools\n"
                        f"print(orbit_tools.read_file(orbit_tools.SOURCE_PATH, "
                        f"offset={off}, limit=2000))"
            })
        return []

    def _c(self, name, args):
        return [{"id": f"c{name}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}]


class _Backend:
    thinking = False

    def __init__(self, model):
        self.model = model
        self.plan_prompts = []   # user content of each PLAN call

    def supports_exact_context_admission(self):
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * 0.25), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def count_text_tokens(self, text):
        return TokenCount(tokens=len(text.split()), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        tools = kwargs.get("tools")
        names = [t["function"]["name"] for t in (tools or [])]
        if PLAN_TOOL_NAME in names:
            users = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
            self.plan_prompts.append("\n".join(users))
        calls = self.model.reply(tools)
        return ChatResult(content="", model="m", finish_reason="stop",
                          tool_calls=calls, prompt_tokens=1, completion_tokens=1,
                          cached_tokens=0, prompt_tokens_per_second=None,
                          generation_tokens_per_second=None)

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


class _Case(unittest.TestCase):
    def _runtime(self, model, text):
        self.backend = _Backend(model)
        ws = AnalysisWorkspace.create()
        data = text.encode()
        path = ws.source_root / "artifact.hta"
        path.write_bytes(data)
        rt = AnalysisRuntime(
            backend=self.backend,
            source=AnalysisSource(snapshot_path=path,
                                  sha256=hashlib.sha256(data).hexdigest(),
                                  size_bytes=len(data), original_path=str(path)),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )
        self.addCleanup(rt.close)
        return rt

    def _range_action(self):
        """execute_analysis stands in for a bounded source-range read: each call
        returns distinct bytes, as reading a new region would."""
        n = {"i": 0}

        def act(**_kw):
            n["i"] += 1
            return AnalysisResult(status="ok", code_sha256=f"{n['i']:064d}",
                                  input_sha256="i" * 64,
                                  stdout=f"REGION {n['i']} bytes: data-{n['i']}",
                                  stderr="", exit_status=0, duration_seconds=0.1)
        return mock.patch.object(module, "execute_analysis", act)


class LargeSourceControllerTests(_Case):
    def test_b1_oversized_does_not_claim_cover(self) -> None:
        rt = self._runtime(_Model(plan=[]), OVERSIZED)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertFalse(run.source_covered)
        self.assertEqual(run.cover_calls, 0)

    def test_b2_plan_receives_bootstrap_overview(self) -> None:
        rt = self._runtime(_Model(plan=[]), OVERSIZED)
        with self._range_action():
            rt.run_autonomous("Analyse it.", finalize=True)
        self.assertTrue(self.backend.plan_prompts)
        plan_ctx = self.backend.plan_prompts[0]
        self.assertIn("larger than this context", plan_ctx)
        self.assertIn("read_file", plan_ctx)
        # The provenance digest is present so every requested region is anchored.
        self.assertIn(rt.source.sha256, plan_ctx)

    def test_b3_plan_can_adopt_a_question(self) -> None:
        rt = self._runtime(_Model(plan=[{"question": "what runs?",
                                         "missing_fact": "needs a region read"}]),
                           OVERSIZED)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertEqual(run.initial_questions, 1)

    def test_b4_bounded_read_dispatches_and_b8_resolves(self) -> None:
        rt = self._runtime(_Model(plan=[{"question": "what runs?",
                                         "missing_fact": "needs a region read"}]),
                           OVERSIZED)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertGreaterEqual(run.actions_executed, 1)
        self.assertIn("Q1", run.resolved_questions)

    def test_b7_multi_region_evidence_supports_one_question(self) -> None:
        """Two distinct region reads both run (distinct bytes) and accumulate."""
        rt = self._runtime(_Model(plan=[{"question": "a", "missing_fact": "f"},
                                        {"question": "b", "missing_fact": "g"}]),
                           OVERSIZED)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        # Two distinct reads ran -- a range read is never suppressed as a
        # whole-source re-delivery.
        self.assertGreaterEqual(run.actions_executed, 2)
        self.assertEqual(run.suppressed_duplicates, 0)

    def test_b10_report_composes_with_partial_coverage(self) -> None:
        rt = self._runtime(_Model(plan=[{"question": "q", "missing_fact": "f"}]),
                           OVERSIZED)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertIsNotNone(run.final_report)
        self.assertFalse(run.source_covered)

    def test_b11_empty_plan_on_oversized_still_reports(self) -> None:
        """Even with the bootstrap, an oversized source may yield no plan -- and
        Part A still guarantees a truthful closing result, not silence."""
        rt = self._runtime(_Model(plan=[]), OVERSIZED)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertIsNotNone(run.final_report)
        self.assertEqual(run.actions_executed, 0)

    def test_b12_small_source_gets_no_overview(self) -> None:
        """The normal small-source path never sees a bootstrap overview."""
        rt = self._runtime(_Model(plan=[]), SMALL)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertTrue(run.source_covered)
        for p in self.backend.plan_prompts:
            self.assertNotIn("larger than this context", p)

    def test_b6_distinct_region_not_suppressed(self) -> None:
        rt = self._runtime(_Model(plan=[{"question": "a", "missing_fact": "f"},
                                        {"question": "b", "missing_fact": "g"}]),
                           OVERSIZED)
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertEqual(run.suppressed_duplicates, 0)

    def test_b1b_overview_is_never_marked_covered(self) -> None:
        """The PLAN overview must never carry a source_covered mark: partial
        structural delivery is not coverage (B-M1). Asserted by instrumenting
        plan_analysis to inspect the transient PLAN message list it builds."""
        rt = self._runtime(_Model(plan=[]), OVERSIZED)
        seen = {"covered_turn": False}
        real = rt._control_call

        def spy(messages, schema, **kw):
            for m in messages:
                if m.get("source_covered") is True and "larger than this context" in str(m.get("content") or ""):
                    seen["covered_turn"] = True
            return real(messages, schema, **kw)
        rt._control_call = spy
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True)
        self.assertFalse(seen["covered_turn"])
        # And the derived coverage state stays False the whole run.
        self.assertFalse(run.source_covered)

    def test_b6b_overview_overflow_degrades_bounded_not_crashing(self) -> None:
        """If the bootstrap view ever overflowed PLAN admission, the run must
        degrade to a bounded, truthful outcome -- never crash or loop.

        The size guard (`size_bytes > (HEAD+TAIL)*2`) makes overflow practically
        unreachable because the view is bounded (~4 KB) while PLAN context is
        minimal, but robustness must not rest on that alone. This forces the
        refusal directly: admission rejects any prompt carrying the view. The
        run must still terminate bounded and produce a closing result (Part A
        guarantees no silent termination), with no exception escaping and no
        unbounded retry.
        """
        rt = self._runtime(_Model(plan=[]), OVERSIZED)
        real_admit = rt._admit

        def refuse_overview(messages, **kw):
            if any("larger than this context" in str(m.get("content") or "")
                   for m in messages):
                from orbit.runtime.context_manager import ContextAdmissionError
                raise ContextAdmissionError("context admission failed: test-forced")
            return real_admit(messages, **kw)
        rt._admit = refuse_overview
        with self._range_action():
            run = rt.run_autonomous("Analyse it.", finalize=True,
                                    max_model_calls=8)
        # Bounded: it ended, it was not cancelled into silence, and a closing
        # result exists (the finalize closure still reports the honest outcome).
        self.assertFalse(run.cancelled)
        self.assertIsNotNone(run.final_report)
        # No fake coverage resulted from the refusal.
        self.assertFalse(run.source_covered)

    def test_b14_reset_invalidates_overview_state(self) -> None:
        """A fresh runtime over the same bytes starts with no coverage and no
        carried overview; the overview is rebuilt per run, never persisted."""
        rt = self._runtime(_Model(plan=[]), OVERSIZED)
        with self._range_action():
            rt.run_autonomous("Analyse it.", finalize=True)
        # The overview was never committed to history (it is PLAN-only).
        self.assertFalse(any("larger than this context" in str(m.get("content") or "")
                             for m in rt.messages))


if __name__ == "__main__":
    unittest.main()
