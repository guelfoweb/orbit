"""A decoded stage's explicit entry-routine invocation is an established fact.

MINE-HTA-IOC-GROUNDING-CLOSURE-1, defect B. The recovered PowerShell stage
defines `function ROmYsTcn` and ends with the bare call `ROmYsTcn;`. That the
stage's own entry routine is invoked WITHIN the stage is established by the
bytes; how the outer HTA/VBScript reaches the stage is a separate question. The
runtime surfaces the inner invocation as a deterministic fact and flags a report
that denies it, without upgrading it into a proven outer-container chain.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_deobfuscate import TransformStage, deobfuscate  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

MINE = ROOT / "workdir" / "samples" / "mine.hta"


class DetectorTests(unittest.TestCase):
    def test_powershell_bare_call_is_an_entry_invocation(self) -> None:
        text = "function ROmYsTcn(){ do_thing };ROmYsTcn;"
        self.assertEqual(module._stage_entry_invocations(text), [("ROmYsTcn", "ROmYsTcn;")])

    def test_argument_passing_call_is_not_an_entry_invocation(self) -> None:
        """A helper called WITH arguments is not the entry-point signal."""
        text = "function pYUxYe($a,$b){ x };function main(){ pYUxYe $f $g };"
        # `pYUxYe $f $g` passes args -> not matched; `main` is never called.
        self.assertEqual(module._stage_entry_invocations(text), [])

    def test_defined_but_never_called_is_not_flagged(self) -> None:
        self.assertEqual(module._stage_entry_invocations("function lonely(){ x }"), [])

    def test_vbscript_sub_self_call(self) -> None:
        text = "Sub Go()\n do\nEnd Sub\nGo"
        self.assertEqual(module._stage_entry_invocations(text), [("Go", "Go")])

    def test_call_without_definition_is_not_flagged(self) -> None:
        """A bare name that is never defined here is not a self-invocation."""
        self.assertEqual(module._stage_entry_invocations("SomethingElse;"), [])

    def test_call_inside_a_string_is_not_an_invocation(self) -> None:
        """A name in a string literal is text, not an executed call."""
        self.assertEqual(
            module._stage_entry_invocations('function ROmYsTcn(){ x };$s="ROmYsTcn;"'),
            [],
        )

    def test_call_inside_a_comment_is_not_an_invocation(self) -> None:
        self.assertEqual(
            module._stage_entry_invocations("function ROmYsTcn(){ x }\n# ROmYsTcn;"),
            [],
        )
        self.assertEqual(
            module._stage_entry_invocations("Sub A()\nEnd Sub\n' A"), []
        )

    def test_consecutive_vbscript_subs_are_both_seen(self) -> None:
        """`End Sub` must not swallow the next sub's keyword (false negative)."""
        text = "Sub A()\nEnd Sub\nSub B()\nEnd Sub\nB"
        self.assertEqual(module._stage_entry_invocations(text), [("B", "B")])


class ContradictionTests(unittest.TestCase):
    INV = [("ROmYsTcn", "ROmYsTcn;", "ev_x")]

    def test_uncertainty_about_the_named_routine_is_flagged(self) -> None:
        text = "It is unclear whether the ROmYsTcn routine is actually invoked."
        self.assertEqual(module._invocation_contradictions(text, self.INV), self.INV)

    def test_explicit_negation_is_flagged(self) -> None:
        text = "The ROmYsTcn function is defined but not invoked anywhere."
        self.assertEqual(module._invocation_contradictions(text, self.INV), self.INV)

    def test_correct_affirmation_is_not_flagged(self) -> None:
        text = "The stage invokes ROmYsTcn at the end (ROmYsTcn;)."
        self.assertEqual(module._invocation_contradictions(text, self.INV), [])

    def test_report_not_naming_the_routine_is_untouched(self) -> None:
        self.assertEqual(
            module._invocation_contradictions("Nothing about that here.", self.INV), []
        )

    def test_name_without_invoke_word_is_not_flagged(self) -> None:
        self.assertEqual(
            module._invocation_contradictions("ROmYsTcn is a function name.", self.INV), []
        )

    def test_outer_container_hedge_is_not_a_contradiction(self) -> None:
        """The correct nuanced statement the fact block asks for -- inner call
        established, outer reach unresolved -- must not be flagged."""
        for correct in (
            "The stage defines ROmYsTcn and calls it as a bare statement; "
            "whether the outer HTA container executes this stage is unclear.",
            "ROmYsTcn is invoked within the stage, but it is not confirmed that "
            "the outer container ever reaches the stage.",
        ):
            self.assertEqual(module._invocation_contradictions(correct, self.INV), [])

    def test_negation_about_a_different_function_is_not_flagged(self) -> None:
        text = "ROmYsTcn is defined here. Separately, the helper Qux is never invoked."
        self.assertEqual(module._invocation_contradictions(text, self.INV), [])

    def test_bare_call_in_sentence_affirms_and_is_not_flagged(self) -> None:
        text = "Although obfuscated, ROmYsTcn; is not something we could confirm at runtime."
        # The bare call form is present in the sentence -> the call is shown.
        self.assertEqual(module._invocation_contradictions(text, self.INV), [])


class RuntimeSurfacingTests(unittest.TestCase):
    def _runtime_with_stage(self, output: str) -> AnalysisRuntime:
        data = b"x\n"
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "a.hta"
        p.write_bytes(data)
        rt = AnalysisRuntime(
            backend=None,
            source=AnalysisSource(
                snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(p),
            ),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )
        self.addCleanup(rt.close)
        digest = hashlib.sha256(output.encode()).hexdigest()
        stage = TransformStage(
            kind="vbscript_chr_offset", key=0, delimiter="", line=1, offset=0,
            depth=0, encoded="<e>", output=output, input_sha256="e" * 64,
            output_sha256=digest,
        )
        rec = rt.evidence_store.add(
            "execute_analysis", output,
            metadata={"produced_by_phase": "analysis_transform"},
        )
        rt.transform_stages.append((stage, rec))
        self.eid = rec.evidence_id
        return rt

    def test_fact_surfaced_with_inner_outer_distinction(self) -> None:
        rt = self._runtime_with_stage("function ROmYsTcn(){ x };ROmYsTcn;")
        fact = rt.stage_invocations()
        self.assertIn("defines function ROmYsTcn and invokes it", fact)
        self.assertIn(self.eid, fact)
        # T5: outer-container linkage explicitly left separate/unresolved.
        self.assertIn("outer container", fact.lower())
        self.assertIn("separate question", fact.lower())
        self.assertIn(fact, rt.deterministic_sections())

    def test_empty_when_no_self_invocation(self) -> None:
        rt = self._runtime_with_stage("function lonely(){ x }")
        self.assertEqual(rt.stage_invocations(), "")

    def test_report_grounding_flags_denied_invocation(self) -> None:
        rt = self._runtime_with_stage("function ROmYsTcn(){ x };ROmYsTcn;")
        report = "The routine ROmYsTcn is defined; it is not clear it is invoked."
        grounded = rt._flag_invocation_contradictions(report)
        self.assertIn(module.STAGE_INVOCATION_CONTRADICTION_NOTICE, grounded)
        self.assertIn("the routine IS invoked within its stage", grounded)
        self.assertIn(report, grounded)  # prose preserved, not rewritten

    def test_report_grounding_leaves_correct_report_untouched(self) -> None:
        rt = self._runtime_with_stage("function ROmYsTcn(){ x };ROmYsTcn;")
        report = "The stage invokes ROmYsTcn (ROmYsTcn;); outer execution is unproven."
        self.assertEqual(rt._flag_invocation_contradictions(report), report)


@unittest.skipUnless(MINE.exists(), "mine.hta sample required")
class RealSampleTests(unittest.TestCase):
    def test_entry_invocation_from_real_decode(self) -> None:
        src = MINE.read_bytes().decode("utf-8", "surrogatepass")
        stage0 = deobfuscate(src)[0].output
        self.assertEqual(
            module._stage_entry_invocations(stage0), [("ROmYsTcn", "ROmYsTcn;")]
        )


if __name__ == "__main__":
    unittest.main()
