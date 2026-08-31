"""Every ANALYSIS tool call carries a non-empty, unique, matching id.

Backends may return tool calls without an `id`. ANALYSIS tolerated that by
writing `call.get("id") or ""` into the matching tool result -- self-consistent,
but structurally invalid: Orbit's shared context planner requires a non-empty id
on every assistant tool call (`context_manager._tool_call_id`) and refuses the
whole history without one, with `invalid-message-structure:missing-tool-call-id`.

That is what made exact-context admission impossible for ANALYSIS, and what made
a first attempt at admission stop a real run at step 2. These tests pin the
prerequisite: a canonical id at the single point where a step accepts backend
calls, before anything is persisted.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_runtime import AnalysisRuntime  # noqa: E402
from orbit.runtime.context_manager import ContextBudget, plan_context  # noqa: E402


def _runtime() -> AnalysisRuntime:
    runtime = object.__new__(AnalysisRuntime)
    object.__setattr__(runtime, "_synthetic_call_seq", 0)
    object.__setattr__(runtime, "messages", [])
    return runtime


def _call(**extra) -> dict:
    call = {"function": {"name": "execute_analysis", "arguments": "{}"}}
    call.update(extra)
    return call


class IdPreservationTests(unittest.TestCase):
    """A real backend id is authoritative and must survive untouched."""

    def test_a_backend_supplied_id_is_preserved_exactly(self) -> None:
        out = _runtime()._with_canonical_call_ids([_call(id="call_abc123")])
        self.assertEqual(out[0]["id"], "call_abc123")

    def test_preserving_an_id_does_not_consume_the_counter(self) -> None:
        """A run of real ids must not leave gaps that look like lost calls."""
        runtime = _runtime()
        runtime._with_canonical_call_ids([_call(id="a"), _call(id="b")])
        generated = runtime._with_canonical_call_ids([_call()])
        self.assertEqual(generated[0]["id"], "orbit_analysis_call_1")

    def test_the_rest_of_the_call_is_carried_through_unchanged(self) -> None:
        out = _runtime()._with_canonical_call_ids(
            [_call(id="keep", function={"name": "execute_analysis", "arguments": '{"code":"x"}'})]
        )
        self.assertEqual(out[0]["function"]["arguments"], '{"code":"x"}')


class GeneratedIdTests(unittest.TestCase):
    """A missing or empty id is filled, never left invalid."""

    def test_a_missing_id_is_generated(self) -> None:
        out = _runtime()._with_canonical_call_ids([_call()])
        self.assertTrue(out[0]["id"], "a generated id must be non-empty")

    def test_an_empty_id_is_treated_as_missing(self) -> None:
        """`""` is the exact shape the old code produced, and is invalid."""
        out = _runtime()._with_canonical_call_ids([_call(id="")])
        self.assertTrue(out[0]["id"])

    def test_a_non_string_id_is_replaced(self) -> None:
        out = _runtime()._with_canonical_call_ids([_call(id=7)])
        self.assertIsInstance(out[0]["id"], str)
        self.assertTrue(out[0]["id"])

    def test_two_id_less_calls_get_different_ids(self) -> None:
        out = _runtime()._with_canonical_call_ids([_call(), _call()])
        self.assertNotEqual(out[0]["id"], out[1]["id"])

    def test_ids_stay_unique_across_steps_of_one_runtime(self) -> None:
        runtime = _runtime()
        first = runtime._with_canonical_call_ids([_call()])
        second = runtime._with_canonical_call_ids([_call()])
        self.assertNotEqual(first[0]["id"], second[0]["id"])

    def test_the_generated_id_is_stable_on_the_returned_object(self) -> None:
        """Downstream steps reuse the returned dict, so its id cannot drift."""
        call = _runtime()._with_canonical_call_ids([_call()])[0]
        self.assertEqual(call["id"], call["id"])
        self.assertTrue(call["id"])

    def test_a_non_dict_call_is_passed_through_untouched(self) -> None:
        """Not ours to repair: the structural gate rejects it with its own message."""
        out = _runtime()._with_canonical_call_ids(["not-a-call"])
        self.assertEqual(out, ["not-a-call"])


class NoCollisionWithBackendIdsTests(unittest.TestCase):
    """A generated id must never duplicate one already in use."""

    def _runtime_with_history(self, history):
        runtime = object.__new__(AnalysisRuntime)
        object.__setattr__(runtime, "_synthetic_call_seq", 0)
        object.__setattr__(runtime, "messages", history)
        return runtime

    def test_a_persisted_backend_id_in_the_generated_form_is_skipped(self) -> None:
        """A backend may return an id shaped exactly like a generated one.

        Preserving it is correct, but the next synthetic id must then skip it:
        a duplicate silently breaks the assistant/result pairing the shared
        planner depends on.
        """
        runtime = self._runtime_with_history([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "orbit_analysis_call_1"}]},
        ])
        out = runtime._with_canonical_call_ids([_call()])
        self.assertNotEqual(out[0]["id"], "orbit_analysis_call_1")

    def test_an_id_preserved_earlier_in_the_same_batch_is_skipped(self) -> None:
        runtime = self._runtime_with_history([])
        out = runtime._with_canonical_call_ids(
            [_call(id="orbit_analysis_call_1"), _call()]
        )
        ids = [c["id"] for c in out]
        self.assertEqual(ids[0], "orbit_analysis_call_1", "the real id is kept")
        self.assertEqual(len(set(ids)), len(ids), "and the generated one differs")


class AssistantAndResultAgreeTests(unittest.TestCase):
    """The id on the assistant call and on its tool result must match."""

    def test_the_tool_result_uses_the_same_id_as_the_assistant_call(self) -> None:
        """Reproduces the real message shapes the step builds from `calls`."""
        call = _runtime()._with_canonical_call_ids([_call()])[0]
        assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
        # `_append_tool_result` writes `call.get("id") or ""` -- with a canonical
        # id present that is now the same non-empty value.
        result = {"role": "tool", "tool_call_id": call.get("id") or "",
                  "name": "execute_analysis", "content": "observation"}

        self.assertEqual(assistant["tool_calls"][0]["id"], result["tool_call_id"])
        self.assertTrue(result["tool_call_id"])


class PlannerAcceptsAnalysisHistoryTests(unittest.TestCase):
    """The point of the change: real ANALYSIS history now parses."""

    def _history(self, call: dict) -> list[dict]:
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "artifact"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call.get("id") or "",
             "name": "execute_analysis", "content": "result"},
            {"role": "assistant", "content": "finding"},
        ]

    def _plan(self, messages: list[dict]):
        budget = ContextBudget(context_tokens=8192, output_reserve=2048,
                               next_action_reserve=256, safety_margin=256)
        return plan_context(messages, budget=budget, count_tokens=lambda m: 1000)

    def test_normalized_history_is_accepted_by_the_shared_planner(self) -> None:
        call = _runtime()._with_canonical_call_ids([_call()])[0]
        plan = self._plan(self._history(call))
        self.assertTrue(plan.admitted)
        self.assertEqual(plan.status, "unchanged")

    def test_unnormalized_history_is_still_rejected(self) -> None:
        """The regression this prerequisite exists to remove, pinned.

        Without normalization the planner refuses the entire history, which is
        what stopped a real analysis at step 2.
        """
        plan = self._plan(self._history(_call()))
        self.assertFalse(plan.admitted)
        self.assertEqual(plan.reason, "invalid-message-structure:missing-tool-call-id")


class StepUsesNormalizedCallsTests(unittest.TestCase):
    """The wiring: a step must normalize before persisting anything."""

    def test_the_step_normalizes_the_backend_calls_it_accepts(self) -> None:
        import ast

        source = (ROOT / "src/orbit/runtime/analysis_runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "step":
                continue
            body = ast.unparse(node)
            self.assertIn("_with_canonical_call_ids(", body)
            self.assertNotIn("list(response.tool_calls or [])", body,
                             "raw backend calls must not be persisted unnormalized")
            return
        self.fail("step() not found")


if __name__ == "__main__":
    unittest.main()
