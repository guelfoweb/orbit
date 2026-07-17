from __future__ import annotations

import unittest

from orbit.inference_audit_config import resolve_inference_audit_shadow
from orbit.runtime.inference_audit import audit_model_calls, summarize_inference_audits


def metric(
    phase: str,
    *,
    loop: int,
    prompt: int,
    cached: int,
    output: int,
    tool_calls: int = 0,
    finish: str = "stop",
    retry_reason: str | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "loop": loop,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "evaluated_tokens": prompt - cached,
        "completion_tokens": output,
        "tool_calls": tool_calls,
        "finish_reason": finish,
        "retry_reason": retry_reason,
    }


class InferenceAuditConfigTests(unittest.TestCase):
    def test_default_is_off_and_invalid_value_disables(self) -> None:
        self.assertEqual(resolve_inference_audit_shadow({}).enabled, False)
        self.assertEqual(resolve_inference_audit_shadow({}).source, "default")
        self.assertEqual(resolve_inference_audit_shadow({"ORBIT_INFERENCE_AUDIT_SHADOW": "1"}).enabled, True)
        self.assertEqual(resolve_inference_audit_shadow({"ORBIT_INFERENCE_AUDIT_SHADOW": "0"}).enabled, False)
        invalid = resolve_inference_audit_shadow({"ORBIT_INFERENCE_AUDIT_SHADOW": "yes"})
        self.assertFalse(invalid.enabled)
        self.assertEqual(invalid.validation_error, "invalid_boolean")


class InferenceAuditTests(unittest.TestCase):
    def test_bounded_single_tool_graph_marks_only_post_tool_and_final_as_candidates(self) -> None:
        audit = audit_model_calls(
            model_steps=[
                metric("tool_call", loop=1, prompt=900, cached=4, output=15, tool_calls=1),
                metric("final", loop=2, prompt=400, cached=300, output=3),
                metric("final_from_tool", loop=3, prompt=120, cached=64, output=3),
            ],
            phase_starts=[
                {"phase": "tool_call", "reason": "tool_selection"},
                {"phase": "final_from_tool", "reason": "tool_evidence"},
            ],
            phase_timings=[
                {"phase": "tool_call", "wall_ms": 1000.0},
                {"phase": "tool_call", "wall_ms": 200.0},
                {"phase": "final_from_tool", "wall_ms": 100.0},
            ],
            tool_names=["exec_shell_full_command"],
            expectation="bounded_confirmation",
            correctness="correct",
        )
        calls = audit["calls"]
        self.assertEqual([call["category"] for call in calls], ["tool_call", "post_tool_route", "confirmation_only"])
        self.assertEqual(
            [call["disposition"] for call in calls],
            ["necessary", "deterministic_completion_candidate", "deterministic_completion_candidate"],
        )
        self.assertEqual(audit["summary"]["theoretical_model_calls"], 2)
        self.assertEqual(audit["summary"]["theoretical_evaluated_tokens"], 156)
        self.assertEqual(audit["summary"]["theoretical_wall_ms"], 300.0)

    def test_synthesis_keeps_post_tool_route_and_final_necessary(self) -> None:
        audit = audit_model_calls(
            model_steps=[
                metric("tool_call", loop=1, prompt=900, cached=4, output=15, tool_calls=1),
                metric("final", loop=2, prompt=400, cached=300, output=12),
                metric("final_from_tool", loop=3, prompt=500, cached=64, output=40),
            ],
            tool_names=["system_info"],
            expectation="synthesis_required",
            correctness="correct",
        )
        self.assertTrue(all(call["disposition"] == "necessary" for call in audit["calls"]))
        self.assertEqual(audit["summary"]["theoretical_model_calls"], 0)

    def test_route_and_chat_final_preserve_route_decision(self) -> None:
        audit = audit_model_calls(
            model_steps=[
                metric("route", loop=1, prompt=700, cached=600, output=5),
                metric("chat_final", loop=2, prompt=300, cached=4, output=20),
            ],
            route_outputs=[{"route_call": "initial", "parsed_route": "CHAT"}],
            expectation="synthesis_required",
            correctness="correct",
        )
        self.assertEqual([call["category"] for call in audit["calls"]], ["initial_route", "chat_final"])
        self.assertEqual(audit["calls"][0]["decision"], "CHAT")
        self.assertEqual(audit["summary"]["theoretical_model_calls"], 0)

    def test_retry_and_completion_repair_are_not_counted_as_safe_savings(self) -> None:
        audit = audit_model_calls(
            model_steps=[
                metric("route_retry", loop=2, prompt=700, cached=600, output=8, retry_reason="length"),
                metric("final_from_tool_completion_repair", loop=4, prompt=200, cached=64, output=20),
            ],
            expectation="bounded_confirmation",
            correctness="correct",
        )
        self.assertEqual([call["category"] for call in audit["calls"]], ["error_recovery", "formatting_only"])
        self.assertEqual([call["disposition"] for call in audit["calls"]], ["retry_caused", "formatting_only"])
        self.assertEqual(audit["summary"]["theoretical_model_calls"], 0)

    def test_unreported_backend_retry_remains_visible_and_uncorrelated(self) -> None:
        audit = audit_model_calls(
            model_steps=[metric("tool_call", loop=1, prompt=100, cached=0, output=4, tool_calls=1)],
            phase_timings=[
                {"phase": "tool_call", "wall_ms": 20.0},
                {"phase": "tool_call_json_retry", "wall_ms": 30.0},
            ],
            tool_names=["system_info"],
        )
        self.assertEqual(audit["summary"]["model_calls"], 2)
        self.assertEqual(audit["summary"]["uncorrelated_backend_calls"], 1)
        self.assertIn("retry", {call["category"] for call in audit["calls"]})

    def test_records_are_content_free_and_deterministic(self) -> None:
        kwargs = {
            "model_steps": [metric("final_from_tool", loop=2, prompt=120, cached=64, output=3)],
            "tool_names": ["system_info"],
            "expectation": "bounded_confirmation",
            "correctness": "correct",
        }
        first = audit_model_calls(**kwargs)
        second = audit_model_calls(**kwargs)
        self.assertEqual(first, second)
        rendered = repr(first)
        for forbidden in ("sensitive request", "raw arguments", "secret evidence", "https://secret", "/tmp/private"):
            self.assertNotIn(forbidden, rendered)

    def test_summary_aggregates_only_observational_data(self) -> None:
        audit = audit_model_calls(
            model_steps=[metric("chat_final", loop=1, prompt=50, cached=4, output=5)],
            expectation="synthesis_required",
            correctness="correct",
        )
        summary = summarize_inference_audits([audit, audit])
        self.assertEqual(summary["model_calls"], 2)
        self.assertEqual(summary["categories"], {"chat_final": 2})
        self.assertFalse(summary["active_behavior_changed"])


if __name__ == "__main__":
    unittest.main()
