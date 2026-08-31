"""ANALYSIS can compact its own history and read the evidence back exactly.

Admission gave ANALYSIS a safe ceiling: an over-budget step stops rather than
driving the KV sequence into the context wall. It could not do better than stop,
because `plan_context` externalises a completed tool turn only when the tool
message carries real evidence identity AND its content is already a canonical
reference -- and ANALYSIS persisted raw observations with neither.

Adding identity alone achieves nothing; that is measured here, not assumed.
Replacing observations with references alone would be worse than the ceiling:
evidence above `COMPAT_INLINE_CHARS` stops being inlined, and without a way to
read it back the model would lose its own results. So the three parts stand
together -- identity, canonical reference, exact rehydration -- and these tests
pin all three, including the failure mode where evidence cannot be re-attested.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisRuntime,
    acquire_analysis_source,
)
from orbit.runtime.context_manager import (  # noqa: E402
    ContextAdmissionError,
    ContextBudget,
    _eligible_tool_turn,
    _parse_turns,
    plan_context,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

SMALL = "S" * 800    # under COMPAT_INLINE_CHARS: stays readable inline
LARGE = "B" * 3000   # over it: becomes a compact reference


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="orbit-evcompact-")
        self.addCleanup(self._dir.cleanup)
        tmp = pathlib.Path(self._dir.name)
        artifact = tmp / "artifact.js"
        artifact.write_text("var x = 1;", encoding="utf-8")
        self.source = acquire_analysis_source(artifact, tmp / "owned")
        self.store = EvidenceStore(root=tmp / "evidence")
        self.runtime = AnalysisRuntime(
            backend=object(), source=self.source, evidence_store=self.store
        )
        self.addCleanup(self.runtime.close)

    def _turn(self, messages: list[dict], tag: str, body: str):
        """One completed tool turn, built through the shipped append path."""
        call = {"id": f"call_{tag}",
                "function": {"name": "execute_analysis", "arguments": "{}"}}
        record = self.store.add(
            "execute_analysis", body,
            metadata={"tool_call_id": call["id"], "user_turn_id": "turn_1",
                      "produced_by_phase": "analysis_action"},
        )
        messages.append({"role": "user", "content": f"step {tag}"})
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        self.runtime.messages = messages
        self.runtime._append_tool_result(call, body, record=record)
        messages = self.runtime.messages
        messages.append({"role": "assistant", "content": f"finding from {tag}"})
        self.runtime.messages = messages
        return record

    def _history(self):
        messages = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "artifact"}]
        small = self._turn(messages, "small", SMALL)
        large = self._turn(messages, "large", LARGE)
        return self.runtime.messages, small, large


class PersistedReferenceTests(_Base):
    def test_the_tool_message_carries_real_evidence_identity(self) -> None:
        messages, _, large = self._history()
        tool = [m for m in messages if m.get("role") == "tool"][-1]
        self.assertEqual(tool["evidence_id"], large.evidence_id)
        self.assertEqual(tool["user_turn_id"], large.user_turn_id)
        self.assertEqual(tool["tool_call_id"], "call_large")
        self.assertTrue(tool["content"].startswith("tool_evidence_ref: true"))

    def test_small_evidence_keeps_its_inline_compatibility_content(self) -> None:
        """Existing EvidenceStore policy decides this, not ANALYSIS."""
        messages, _, _ = self._history()
        tool = [m for m in messages if m.get("role") == "tool"][0]
        self.assertIn("S" * 20, tool["content"])

    def test_large_evidence_becomes_a_compact_reference(self) -> None:
        messages, _, _ = self._history()
        tool = [m for m in messages if m.get("role") == "tool"][-1]
        self.assertNotIn("B" * 20, tool["content"])
        self.assertLess(len(tool["content"]), len(LARGE))

    def test_a_result_without_evidence_carries_no_identity(self) -> None:
        """A refused action must not claim evidence that does not exist."""
        self.runtime.messages = []
        self.runtime._append_tool_result({"id": "c1"}, "action not executed: nope")
        message = self.runtime.messages[-1]
        self.assertNotIn("evidence_id", message)
        self.assertNotIn("user_turn_id", message)
        self.assertEqual(message["content"], "action not executed: nope")


class CompactabilityTests(_Base):
    def test_completed_evidence_turns_are_eligible(self) -> None:
        messages, _, _ = self._history()
        available, covered = self.runtime._compactable_evidence_sets(messages, ())
        eligible = [
            turn for turn in _parse_turns(messages)
            if _eligible_tool_turn(turn, available=frozenset(available),
                                   covered=frozenset(covered))
        ]
        self.assertEqual(len(eligible), 2)

    def test_an_over_budget_history_is_compacted_and_admitted(self) -> None:
        messages, _, _ = self._history()
        available, covered = self.runtime._compactable_evidence_sets(messages, ())
        budget = ContextBudget(context_tokens=8192, output_reserve=2048,
                               next_action_reserve=256, safety_margin=256)

        def count(candidate: list[dict]) -> int:
            inlined = any(
                ("B" * 20) in str(m.get("content")) or ("S" * 20) in str(m.get("content"))
                for m in candidate
            )
            return 7000 if inlined else 3000

        plan = plan_context(messages, budget=budget,
                            available_evidence_ids=available,
                            covered_evidence_ids=covered, count_tokens=count)

        self.assertEqual(plan.status, "compacted")
        self.assertTrue(plan.admitted)
        self.assertLessEqual(plan.tokens_after, budget.input_limit)
        self.assertGreaterEqual(plan.compacted_turns, 1)
        self.assertTrue(plan.externalized_evidence_ids)

    def test_identity_without_a_reference_is_not_compactable(self) -> None:
        """Measured, not assumed: identity alone achieves nothing."""
        messages, _, _ = self._history()
        raw = [dict(m) for m in messages]
        for message in raw:
            if message.get("role") == "tool":
                message["content"] = "RAW OBSERVATION TEXT"
        self.assertTrue(all(t.evidence_ids == () for t in _parse_turns(raw)))


class RehydrationTests(_Base):
    def test_exact_large_evidence_is_restored_verbatim(self) -> None:
        messages, _, large = self._history()
        asked = [*messages,
                 {"role": "user", "content": f"need evidence:{large.evidence_id}"}]
        rehydrated, ids = self.runtime._with_evidence_rehydration(asked)
        self.assertEqual(ids, (large.evidence_id,))
        block = rehydrated[-1]["content"]
        self.assertIn(LARGE, block, "the exact archived bytes must come back")
        self.assertIn(large.raw_sha256, block, "with their digest, for attestation")

    def test_a_reference_does_not_rehydrate_itself(self) -> None:
        """Scanning tool messages would undo every compaction immediately.

        A canonical reference names its own id in `exact_content_ref`, so a scan
        that included tool messages would re-inline each compacted turn on the
        very next step.
        """
        messages, _, _ = self._history()
        out, ids = self.runtime._with_evidence_rehydration(messages)
        self.assertEqual(ids, ())
        self.assertEqual(out, messages)

    def test_rehydrated_evidence_is_withheld_from_compaction(self) -> None:
        messages, _, large = self._history()
        available, covered = self.runtime._compactable_evidence_sets(
            messages, (large.evidence_id,)
        )
        self.assertIn(large.evidence_id, available)
        self.assertNotIn(large.evidence_id, covered)

    def test_missing_evidence_fails_closed(self) -> None:
        with self.assertRaises(ContextAdmissionError):
            self.runtime._with_evidence_rehydration(
                [{"role": "user",
                  "content": "need evidence:ev_000000000000_0000000000000000"}]
            )

    def test_a_tampered_reference_is_not_treated_as_available(self) -> None:
        messages, _, _ = self._history()
        tampered = [dict(m) for m in messages]
        for message in tampered:
            if message.get("role") == "tool":
                message["content"] = message["content"] + "\ntampered: true"
        available, _ = self.runtime._compactable_evidence_sets(tampered, ())
        self.assertEqual(available, set())

    def test_a_pairing_whose_identity_disagrees_with_the_record_is_rejected(
        self,
    ) -> None:
        """A tool message must agree with the record it claims.

        ANALYSIS builds both sides from one record, so today this cannot arise.
        The check exists so a reloaded or externally assembled history cannot
        launder one turn's evidence into another turn's slot, and so this stays
        at parity with CHAT's `_context_evidence_sets`.
        """
        for field, value in (
            ("tool_call_id", "orbit_analysis_call_999"),
            ("name", "some_other_tool"),
            ("user_turn_id", "utid_mismatched"),
        ):
            with self.subTest(field=field):
                messages, _, _ = self._history()
                mismatched = [dict(m) for m in messages]
                for message in mismatched:
                    if message.get("role") == "tool":
                        message[field] = value
                available, covered = self.runtime._compactable_evidence_sets(
                    mismatched, ()
                )
                self.assertEqual(available, set())
                self.assertEqual(covered, set())

    def test_the_evidence_store_survives_compaction(self) -> None:
        messages, small, large = self._history()
        available, covered = self.runtime._compactable_evidence_sets(messages, ())
        plan_context(
            messages,
            budget=ContextBudget(context_tokens=8192, output_reserve=2048,
                                 next_action_reserve=256, safety_margin=256),
            available_evidence_ids=available, covered_evidence_ids=covered,
            count_tokens=lambda m: 7000,
        )
        for record in (small, large):
            self.assertIsNotNone(
                self.store.reattest_exact(record.evidence_id),
                "compaction externalises a reference; it never deletes evidence",
            )


class ShippedPathWiringTests(_Base):
    """The wiring itself: `step()` must persist a record, `_admit` must offer it."""

    def test_a_real_step_persists_an_evidence_backed_reference(self) -> None:
        class Backend:
            thinking = False

            def supports_exact_context_admission(self) -> bool:
                return True

            def model_info(self):
                class _Info:
                    context_length = 8192
                return _Info()

            def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                return TokenCount(tokens=900, context_tokens=8192,
                                  rendered_hash="a" * 64, token_hash="b" * 64)

            def chat_stream(self, messages, **kwargs):
                return ChatResult(
                    content="", model="m", finish_reason="tool_calls",
                    tool_calls=[{"function": {"name": "execute_analysis",
                                              "arguments": '{"code": "print(1)"}'}}],
                    prompt_tokens=900, completion_tokens=5, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        runtime = AnalysisRuntime(
            backend=Backend(), source=self.source, evidence_store=self.store
        )
        self.addCleanup(runtime.close)
        runtime.step("analyze")

        tool = [m for m in runtime.messages if m.get("role") == "tool"][-1]
        self.assertIn("evidence_id", tool)
        self.assertTrue(tool["content"].startswith("tool_evidence_ref: true"))
        self.assertIsNotNone(
            self.store.reattest_exact(tool["evidence_id"],
                                      expected_reference=tool["content"]),
        )

    def test_admit_rehydrates_evidence_the_analyst_asked_for(self) -> None:
        """`_admit` must actually call rehydration, not just be able to.

        Without this the request path can drop the call entirely and every unit
        test still passes -- the model would name an evidence id mid-analysis
        and silently receive nothing, which is the exact silent-loss failure
        this capability exists to prevent.
        """
        class Backend:
            thinking = False

            def __init__(self) -> None:
                self.seen: list[list[dict]] = []

            def supports_exact_context_admission(self) -> bool:
                return True

            def model_info(self):
                class _Info:
                    context_length = 8192
                return _Info()

            def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                self.seen.append([dict(m) for m in messages])
                return TokenCount(tokens=900, context_tokens=8192,
                                  rendered_hash="a" * 64, token_hash="b" * 64)

        messages, _, large = self._history()
        backend = Backend()
        object.__setattr__(self.runtime, "backend", backend)

        admitted = self.runtime._admit(
            [*messages, {"role": "user", "content": f"need evidence:{large.evidence_id}"}],
            max_tokens=2048, tools=[],
        )

        self.assertTrue(
            any(LARGE in str(m.get("content")) for m in admitted),
            "the exact evidence the analyst named must reach the request",
        )

    def test_admit_offers_the_evidence_sets_to_the_planner(self) -> None:
        import ast

        source = (ROOT / "src/orbit/runtime/analysis_runtime.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef) or node.name != "_admit":
                continue
            body = ast.unparse(node)
            self.assertIn("available_evidence_ids=available", body)
            self.assertIn("covered_evidence_ids=covered", body)
            self.assertIn("_compactable_evidence_sets(", body)
            return
        self.fail("_admit not found")


class PromptContractTests(unittest.TestCase):
    def test_the_prompt_teaches_reference_semantics_and_retrieval(self) -> None:
        from orbit.runtime.analysis_runtime import ANALYSIS_SYSTEM_PROMPT

        self.assertIn("tool_evidence_ref", ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("evidence:<evidence_id>", ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("Never infer content from a reference alone",
                      ANALYSIS_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
