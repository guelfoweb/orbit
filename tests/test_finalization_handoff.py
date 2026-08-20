from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.finalization import (
    FINAL_ONLY_INSTRUCTION,
    FINALIZATION_FINAL_MAX_TOKENS,
    FINALIZATION_SAFETY_TOKENS,
    BundleEntry,
    admit_finalization,
    deduplicate_evidence,
    render_bundle,
    resolve_output_budget,
)

CTX = 16384
TASK = "Analyze the artifact and report what it does."


def entry(evidence_id: str, content: str, kind: str = "stdout") -> BundleEntry:
    return BundleEntry(
        evidence_id=evidence_id,
        kind=kind,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        chars=len(content),
        content=content,
    )


class DeduplicationTests(unittest.TestCase):
    def test_byte_identical_evidence_collapses(self) -> None:
        entries = [entry("E1", "same bytes"), entry("E2", "same bytes")]
        self.assertEqual(len(deduplicate_evidence(entries)), 1)

    def test_first_occurrence_identity_is_kept(self) -> None:
        entries = [entry("E1", "dup"), entry("E9", "dup")]
        self.assertEqual(deduplicate_evidence(entries)[0].evidence_id, "E1")

    def test_byte_different_evidence_is_not_merged(self) -> None:
        # Similar but not identical must remain distinct: no fuzzy equivalence.
        entries = [entry("E1", "value: 86"), entry("E2", "value: 73")]
        self.assertEqual(len(deduplicate_evidence(entries)), 2)

    def test_whitespace_difference_is_not_merged(self) -> None:
        entries = [entry("E1", "a b"), entry("E2", "a  b")]
        self.assertEqual(len(deduplicate_evidence(entries)), 2)

    def test_provenance_and_content_survive(self) -> None:
        payload = "line1\nline2 — ünïcode\t\\escaped"
        kept = deduplicate_evidence([entry("E4", payload, kind="file:x")])[0]
        self.assertEqual(kept.content, payload)
        self.assertEqual(kept.kind, "file:x")
        self.assertEqual(kept.sha256, hashlib.sha256(payload.encode()).hexdigest())

    def test_empty_evidence_yields_empty_bundle(self) -> None:
        self.assertEqual(deduplicate_evidence([]), [])


class RenderTests(unittest.TestCase):
    def test_bundle_contains_task_instruction_and_ids(self) -> None:
        text = render_bundle(TASK, [entry("E1", "PAYLOAD")])
        self.assertIn(TASK, text)
        self.assertIn(FINAL_ONLY_INSTRUCTION, text)
        self.assertIn("[E1]", text)
        self.assertIn("PAYLOAD", text)

    def test_evidence_bytes_are_verbatim(self) -> None:
        payload = "exact\tbytes\\here\n"
        self.assertIn(payload, render_bundle(TASK, [entry("E1", payload)]))

    def test_instruction_requires_unresolved_and_forbids_tools(self) -> None:
        low = FINAL_ONLY_INSTRUCTION.lower()
        self.assertIn("unresolved", low)
        self.assertIn("do not request tools", low)

    def test_instruction_carries_no_domain_vocabulary(self) -> None:
        low = FINAL_ONLY_INSTRUCTION.lower()
        for banned in ("malware", "xor", "powershell", "wmi", "url", "ioc",
                       "decode", "cleanup", "payload"):
            self.assertNotIn(banned, low)

    def test_bundle_carries_only_supplied_evidence(self) -> None:
        # Reasoning, failed repairs and generated programs are not evidence
        # records, so they cannot reach the bundle.
        text = render_bundle(TASK, [entry("E1", "EVIDENCE")])
        for banned in ("Traceback", "SyntaxError", "import orbit_tools"):
            self.assertNotIn(banned, text)


class OutputBudgetTests(unittest.TestCase):
    def test_budget_is_not_inherited_from_investigation(self) -> None:
        # The investigation's per-call cap is 2048; finalization must not be
        # silently limited to it when the context has more room.
        budget = resolve_output_budget(7008, CTX)
        self.assertGreater(budget, 2048)
        self.assertEqual(budget, FINALIZATION_FINAL_MAX_TOKENS)

    def test_budget_is_capped(self) -> None:
        self.assertEqual(resolve_output_budget(100, CTX), FINALIZATION_FINAL_MAX_TOKENS)

    def test_budget_shrinks_to_available_context(self) -> None:
        prompt = CTX - FINALIZATION_SAFETY_TOKENS - 1000
        self.assertEqual(resolve_output_budget(prompt, CTX), 1000)

    def test_budget_is_zero_when_nothing_remains(self) -> None:
        self.assertEqual(resolve_output_budget(CTX, CTX), 0)

    def test_safety_reserve_is_always_deducted(self) -> None:
        prompt = CTX - FINALIZATION_SAFETY_TOKENS
        self.assertEqual(resolve_output_budget(prompt, CTX), 0)


class AdmissionTests(unittest.TestCase):
    def test_preserved_bundle_is_admitted(self) -> None:
        admission = admit_finalization(7008, CTX)
        self.assertTrue(admission.admitted)
        self.assertIsNone(admission.reason)
        self.assertEqual(admission.output_budget, FINALIZATION_FINAL_MAX_TOKENS)
        self.assertGreaterEqual(admission.headroom, 0)

    def test_oversized_bundle_is_refused_before_inference(self) -> None:
        admission = admit_finalization(CTX - 100, CTX)
        self.assertFalse(admission.admitted)
        self.assertEqual(admission.reason, "finalization_bundle_exceeds_context")

    def test_exact_boundary_is_admitted(self) -> None:
        prompt = CTX - FINALIZATION_SAFETY_TOKENS - 256
        admission = admit_finalization(prompt, CTX, minimum_output=256)
        self.assertTrue(admission.admitted)
        self.assertEqual(admission.output_budget, 256)

    def test_one_token_over_is_refused(self) -> None:
        prompt = CTX - FINALIZATION_SAFETY_TOKENS - 255
        self.assertFalse(admit_finalization(prompt, CTX, minimum_output=256).admitted)

    def test_admission_never_exceeds_context(self) -> None:
        for prompt in (0, 1000, 7008, 12000, 15000, 16000):
            admission = admit_finalization(prompt, CTX)
            if admission.admitted:
                total = (
                    admission.prompt_tokens
                    + admission.output_budget
                    + admission.safety_tokens
                )
                self.assertLessEqual(total, CTX, f"prompt={prompt}")

    def test_investigation_history_size_does_not_affect_admission(self) -> None:
        # The whole point: a saturated investigation must not block an answer.
        admission = admit_finalization(7008, CTX)
        self.assertTrue(admission.admitted)


if __name__ == "__main__":
    unittest.main()
