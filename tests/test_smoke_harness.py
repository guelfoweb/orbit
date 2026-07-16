from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from orbit.runtime.turn_trace import ModelStepMetrics


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "orbit_smoke_harness.py"
SPEC = importlib.util.spec_from_file_location("orbit_smoke_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_harness = importlib.util.module_from_spec(SPEC)
sys.modules["orbit_smoke_harness"] = smoke_harness
SPEC.loader.exec_module(smoke_harness)


def route_step_report(**overrides: object):
    values = {
        "case": "route",
        "step": 1,
        "prompt": "fixture",
        "prompt_kind": "auto",
        "completion_kind": "route",
        "route_tokens": 10,
        "final_tokens": 10,
        "prompt_tokens": 10,
        "cached_tokens": 0,
        "evaluated_tokens": 10,
        "finish_reason": "stop",
        "tool_calls": 0,
        "tool_names": [],
        "wall_ms": 1.0,
        "correctness_category": "correct",
        "raw_leak": False,
        "fake_output": False,
        "loop": False,
        "notes": "",
        "answer_excerpt": "answer",
        "model_steps": [],
    }
    values.update(overrides)
    return smoke_harness.StepReport(**values)


class SmokeHarnessTests(unittest.TestCase):
    def test_tool_call_generation_only_uses_one_model_call_without_execution_or_finalization(self) -> None:
        class Backend:
            base_url = "http://127.0.0.1:9"

            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                return smoke_harness.ChatResult(
                    content="",
                    model="fake",
                    finish_reason="tool_calls",
                    tool_calls=[{"id": "c", "type": "function", "function": {"name": "system_info", "arguments": "{}"}}],
                    prompt_tokens=20,
                    completion_tokens=3,
                    cached_tokens=4,
                    prompt_tokens_per_second=100.0,
                    generation_tokens_per_second=10.0,
                )

        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "wait_for_backend_idle", return_value={"in_flight": False}), \
            mock.patch("orbit.runtime.tool_calls.execute_tool_call") as execute, \
            mock.patch.object(smoke_harness.ChatRuntime, "_answer_from_tool_results") as finalize:
            row = smoke_harness.run_tool_call_generation_sample(
                backend,
                base_url=backend.base_url,
                timeout=1.0,
                workdir=Path(tmp),
                scenario="system_info_1",
                prompt="synthetic prompt",
                expected_tool="system_info",
                repetition=1,
                temperature=0,
                max_tokens=96,
            )

        self.assertEqual(backend.calls, 1)
        self.assertEqual(row["model_calls"], 1)
        self.assertTrue(row["strict_valid"])
        self.assertEqual(row["candidate_source"], "backend")
        self.assertTrue(row["tool_name_exact_match"])
        self.assertFalse(row["tool_executed"])
        self.assertFalse(row["finalization_started"])
        execute.assert_not_called()
        finalize.assert_not_called()

    def test_tool_call_generation_timeout_cancels_and_reports_cleanup(self) -> None:
        release = threading.Event()

        class Backend:
            base_url = "http://127.0.0.1:9"

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                release.wait(1.0)
                return smoke_harness.ChatResult(
                    content="", model="fake", finish_reason="cancelled", tool_calls=[],
                    prompt_tokens=1, completion_tokens=0, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        def cancel(*_args, **_kwargs):
            release.set()
            return True

        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "request_backend_cancel", side_effect=cancel), \
            mock.patch.object(smoke_harness, "wait_for_backend_idle", return_value={"in_flight": False}):
            row = smoke_harness.run_tool_call_generation_sample(
                Backend(), base_url="http://127.0.0.1:9", timeout=0.01,
                workdir=Path(tmp), scenario="timeout", prompt="fixture",
                expected_tool="system_info", repetition=1, temperature=0, max_tokens=96,
            )

        self.assertTrue(row["timeout"])
        self.assertTrue(row["cancel_requested"])
        self.assertTrue(row["cleanup_healthy"])
        self.assertFalse(row["evaluable"])
        self.assertEqual(row["model_calls"], 1)

    def test_tool_call_generation_summary_and_jsonl_are_content_free(self) -> None:
        rows = [
            {
                "type": "tool_call_generation", "scenario": "safe-id", "repetition": 1,
                "evaluable": True, "timeout": False, "cancelled": False, "cleanup_healthy": True,
                "strict_valid": True, "repairs": [], "attempt_detected": True,
                "tool_name_exact_match": True, "multiple_candidates": False, "truncation": False,
                "budget_truncation": False, "structural_truncation": False,
                "semantic_outcome": "expected_tool",
                "formal_repairable": False,
                "markup_leakage": False, "categories": ["valid_first_pass"],
                "generation_wall_ms": 10.0, "healing_us": 80.0, "model_calls": 1,
                "tool_executed": False, "finalization_started": False,
            }
        ]
        summary = smoke_harness.summarize_tool_call_generation(rows, mtp="off")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generation.jsonl"
            smoke_harness.write_tool_call_generation_jsonl(path, {"type": "environment", "model": "fake"}, rows, summary)
            text = path.read_text(encoding="utf-8")

        self.assertEqual(summary["completion_rate"], 1.0)
        self.assertEqual(summary["valid_first_pass_rate"], 1.0)
        self.assertEqual(summary["valid_first_pass_output_rate"], 1.0)
        self.assertEqual(summary["formal_error_rate"], 0.0)
        self.assertEqual(summary["deterministic_repair_candidate_rate"], 0.0)
        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["tools_executed"], 0)
        self.assertEqual(summary["finalizations_started"], 0)
        self.assertNotIn("synthetic prompt", text)
        self.assertNotIn("arguments", text)

    def test_validation_divergence_replay_measures_legacy_without_healing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = smoke_harness.run_tool_validation_divergence_replay(Path(tmp))

        cases = [row for row in rows if row["type"] == "tool_validation_divergence_case"]
        summary = rows[-1]
        self.assertEqual(summary["cases"], 16)
        self.assertEqual(summary["matched_expectations"], 16)
        self.assertEqual(summary["active_default_used"], 4)
        self.assertEqual(summary["active_clamp_used"], 3)
        self.assertEqual(summary["active_ignored_extra"], 4)
        self.assertEqual(summary["policy_denied"], 1)
        self.assertEqual(summary["missing_required"], 1)
        self.assertFalse(summary["repair_executed"])
        self.assertEqual(summary["new_executions_enabled"], 0)
        self.assertEqual(summary["valid_off_on_terminal_matches"], 3)
        self.assertTrue(all(row["formal_repairable"] is False for row in cases))
        self.assertTrue(all("arguments" not in row for row in cases))

    def test_tool_call_generation_distinguishes_no_attempt_and_truncation_kinds(self) -> None:
        no_attempt = smoke_harness.tool_healing_categories(
            {"attempt_detected": False, "strict_outcome": "no_attempt", "finish_reason": "stop"}
        )
        budget = smoke_harness.tool_healing_categories(
            {
                "attempt_detected": True,
                "strict_outcome": "rejected_parse",
                "parse_error": "unterminated_json_string",
                "finish_reason": "length",
                "repairs": ["close_json_structure"],
            }
        )
        structural = smoke_harness.tool_healing_categories(
            {
                "attempt_detected": True,
                "strict_outcome": "rejected_parse",
                "parse_error": "unterminated_json_string",
                "finish_reason": "stop",
            }
        )

        self.assertEqual(no_attempt, ["no_attempt"])
        self.assertNotIn("ambiguous_attempt", no_attempt)
        self.assertIn("budget_truncation", budget)
        self.assertIn("truncated_attempt", budget)
        self.assertNotIn("budget_truncation", structural)
        self.assertIn("truncated_attempt", structural)

        summary = smoke_harness.summarize_tool_call_generation(
            [
                {
                    "evaluable": True, "expected_tool": "fetch_url", "attempt_detected": True,
                    "categories": budget, "semantic_outcome": "missing_tool_call",
                    "formal_repairable": False,
                    "generation_wall_ms": 10.0, "healing_us": 50.0,
                    "cleanup_healthy": True, "timeout": False, "cancelled": False,
                    "multiple_candidates": False, "truncation": True,
                    "budget_truncation": True, "structural_truncation": True,
                    "markup_leakage": True, "model_calls": 1,
                }
            ],
            mtp="off",
        )
        self.assertEqual(summary["deterministic_repair_candidate_rate"], 0.0)
        self.assertEqual(summary["budget_truncation_rate"], 1.0)
        self.assertEqual(summary["structural_truncation_rate"], 1.0)

    def test_tool_call_generation_summary_excludes_non_evaluable_and_semantic_errors_from_repair(self) -> None:
        rows = [
            {
                "evaluable": True, "expected_tool": None, "attempt_detected": False,
                "categories": ["no_attempt"], "semantic_outcome": "no_attempt",
                "formal_repairable": False,
                "generation_wall_ms": 10.0, "healing_us": 50.0,
                "cleanup_healthy": True, "timeout": False, "cancelled": False,
                "multiple_candidates": False, "truncation": False,
                "budget_truncation": False, "structural_truncation": False,
                "markup_leakage": False, "model_calls": 1,
            },
            {
                "evaluable": True, "expected_tool": "read_file", "attempt_detected": True,
                "categories": ["valid_first_pass"], "semantic_outcome": "wrong_tool",
                "formal_repairable": False,
                "tool_name_exact_match": False, "generation_wall_ms": 20.0, "healing_us": 60.0,
                "cleanup_healthy": True, "timeout": False, "cancelled": False,
                "multiple_candidates": False, "truncation": False,
                "budget_truncation": False, "structural_truncation": False,
                "markup_leakage": False, "model_calls": 1,
            },
            {
                "evaluable": False, "expected_tool": "system_info", "attempt_detected": False,
                "categories": ["uncorrelated"], "semantic_outcome": "not_evaluable",
                "formal_repairable": False,
                "generation_wall_ms": 80.0, "cleanup_healthy": True,
                "timeout": True, "cancelled": True, "model_calls": 1,
            },
        ]

        summary = smoke_harness.summarize_tool_call_generation(rows, mtp="off")

        self.assertEqual(summary["evaluable"], 2)
        self.assertEqual(summary["no_attempt_rate"], 0.5)
        self.assertEqual(summary["valid_first_pass_rate"], 1.0)
        self.assertEqual(summary["valid_first_pass_output_rate"], 0.5)
        self.assertEqual(summary["semantic_wrong_tool_rate"], 1.0)
        self.assertEqual(summary["deterministic_repair_candidate_rate"], 0.0)
        self.assertEqual(summary["formal_error_rate"], 0.0)
        self.assertEqual(summary["generation_wall_ms_median"], 15.0)

    def test_parser_accepts_output_and_selection_options(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            [
                "--base-url",
                "http://127.0.0.1:12120",
                "--scenario",
                "pwd_followup",
                "--no-web",
                "--jsonl",
                "/tmp/out.jsonl",
                "--markdown",
                "/tmp/out.md",
                "--canonical-gate",
                "on",
                "--tool-healing-mode",
                "off",
            ]
        )

        self.assertEqual(args.scenario, ["pwd_followup"])
        self.assertTrue(args.no_web)
        self.assertEqual(args.jsonl, "/tmp/out.jsonl")
        self.assertEqual(args.markdown, "/tmp/out.md")
        self.assertEqual(args.canonical_gate, "on")
        self.assertEqual(args.tool_healing_mode, "off")

    def test_canonical_gate_environment_and_metadata_are_bounded(self) -> None:
        key = "ORBIT_TOOL_CALL_CANONICAL_GATE"
        with mock.patch.dict(os.environ, {key: "0"}, clear=False):
            with smoke_harness.canonical_gate_environment("on"):
                self.assertEqual(os.environ[key], "1")
                self.assertEqual(
                    smoke_harness.canonical_gate_metadata("inherit"),
                    {"requested_mode": "inherit", "enabled": True, "source": "stable", "validation_error": None},
                )
            self.assertEqual(os.environ[key], "0")

    def test_tool_healing_environment_and_metadata_are_bounded(self) -> None:
        key = "ORBIT_TOOL_CALL_HEALING"
        with mock.patch.dict(os.environ, {key: "0"}, clear=False):
            with smoke_harness.tool_healing_environment("on"):
                self.assertEqual(os.environ[key], "1")
                metadata = smoke_harness.tool_healing_metadata("inherit")
                self.assertEqual(metadata["requested_mode"], "inherit")
                self.assertTrue(metadata["enabled"])
                self.assertEqual(metadata["source"], "stable")
                self.assertIsNone(metadata["validation_error"])
                self.assertIsNone(metadata["blocked_reason"])
                self.assertIsInstance(metadata["repair_count"], int)
                self.assertIsInstance(metadata["rejection_count"], int)
                self.assertIsInstance(metadata["last_rules"], list)
            self.assertEqual(os.environ[key], "0")

    def test_parser_accepts_managed_final_prefix_options(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            ["--manage-server", "--final-prefix-mode", "on", "--repetitions", "5"]
        )

        self.assertTrue(args.manage_server)
        self.assertEqual(args.final_prefix_mode, "on")
        self.assertEqual(args.repetitions, 5)

    def test_parser_accepts_first_class_lifecycle_checks(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            [
                "--manage-server",
                "--final-prefix-mode",
                "on",
                "--lifecycle-check",
                "restart",
                "--lifecycle-check",
                "ctx-change",
                "--ctx-change-to",
                "4096",
            ]
        )

        self.assertEqual(args.lifecycle_check, ["restart", "ctx-change"])
        self.assertEqual(args.ctx_change_to, 4096)

    def test_parser_accepts_opt_in_route_output_diagnostics(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            ["--route-output-diagnostics", "--route-diagnostic-store", "existing-snapshot"]
        )

        self.assertTrue(args.route_output_diagnostics)
        self.assertEqual(args.route_diagnostic_store, "existing-snapshot")

        defaults = smoke_harness.build_parser().parse_args([])
        self.assertFalse(defaults.route_output_diagnostics)
        self.assertEqual(defaults.route_diagnostic_store, "clean")

    def test_select_scenarios_defaults_to_all_without_optional_or_web_when_disabled(self) -> None:
        selected = smoke_harness.select_scenarios(None, no_web=True, include_optional=False)
        names = [scenario.name for scenario in selected]

        self.assertIn("simple_chat", names)
        self.assertIn("dual_shell", names)
        self.assertNotIn("web_shell", names)
        self.assertNotIn("grep_read", names)

    def test_select_scenarios_allows_single_scenario_without_implicit_all(self) -> None:
        selected = smoke_harness.select_scenarios(["pwd_followup"], no_web=False, include_optional=False)

        self.assertEqual([scenario.name for scenario in selected], ["pwd_followup"])

    def test_final_prefix_mixed_has_eleven_isolated_eligible_cases(self) -> None:
        scenario = smoke_harness.scenarios()["final_prefix_mixed"]

        self.assertEqual(len(scenario.steps), 11)
        self.assertTrue(scenario.isolated_steps)
        self.assertIn("system_info", scenario.allowed_tool_names)

    def test_final_prefix_groups_share_canonical_case_definitions(self) -> None:
        registry = smoke_harness.scenarios()

        self.assertEqual(registry["final_prefix_local"].steps, smoke_harness.FINAL_PREFIX_LOCAL_STEPS)
        self.assertEqual(registry["final_prefix_web"].steps, smoke_harness.FINAL_PREFIX_WEB_STEPS)
        self.assertEqual(
            registry["final_prefix_mixed"].steps,
            smoke_harness.FINAL_PREFIX_LOCAL_STEPS + smoke_harness.FINAL_PREFIX_WEB_STEPS,
        )
        self.assertEqual(registry["final_prefix_paired"].steps, smoke_harness.FINAL_PREFIX_PAIRED_STEPS)

    def test_route_classification_matrix_is_declarative_and_optional(self) -> None:
        registry = smoke_harness.scenarios()
        names = (
            "route_classification_recap",
            "route_classification_chat",
            "route_classification_local",
            "route_classification_web",
            "route_classification_evidence",
            "route_classification_ambiguous",
            "route_classification_refresh",
            "route_classification_verify",
            "route_classification_error_success",
            "route_classification_web_error",
        )

        self.assertTrue(all(registry[name].optional for name in names))
        self.assertEqual(registry["route_classification_recap"].steps, smoke_harness.ROUTE_CLASS_RECAP_STEPS)
        self.assertEqual(registry["route_classification_web"].family, "web")
        self.assertEqual(registry["route_classification_ambiguous"].steps, smoke_harness.ROUTE_CLASS_AMBIGUOUS_STEPS)
        self.assertEqual(registry["route_classification_refresh"].steps, smoke_harness.ROUTE_CLASS_REFRESH_STEPS)
        self.assertEqual(registry["route_classification_verify"].steps, smoke_harness.ROUTE_CLASS_VERIFY_STEPS)
        self.assertEqual(registry["route_classification_error_success"].steps, smoke_harness.ROUTE_CLASS_ERROR_SUCCESS_STEPS)
        self.assertEqual(registry["route_classification_web_error"].steps, smoke_harness.ROUTE_CLASS_WEB_ERROR_STEPS)
        self.assertTrue(
            all(step.expected_route is not None for name in names for step in registry[name].steps)
        )

    def test_route_diagnostic_lines_are_sanitized_and_missing_fields_are_safe(self) -> None:
        lines = [
            *(
                json.dumps(
                    {
                        "event": "kv_diag_route_outcome",
                        "phase": "route",
                        "route_output_class": route_class,
                        "route_output_reason": f"fixture_{route_class}",
                        "route_parser_accepted": route_class in {"canonical", "legacy_tolerated"},
                    }
                )
                for route_class in smoke_harness.ROUTE_OUTPUT_CLASSES
            ),
            json.dumps(
                {
                    "event": "kv_diag_route_outcome",
                    "phase": "route",
                    "route_output_class": "malformed",
                    "route_output_reason": "unaccepted_output",
                    "route_parser_accepted": False,
                    "route_finish_reason": "length",
                    "route_output_tokens": 64,
                    "decision_type": None,
                    "outcome": "route_no_decision_length_retry",
                    "retry_reason": "length_without_decision",
                    "raw_route_text": "route-secret",
                    "user_request": "request-secret",
                    "evidence": "evidence-secret",
                }
            ),
            json.dumps({"event": "kv_diag_route_outcome", "phase": "route_retry"}),
            "not-json",
        ]

        events = smoke_harness.parse_route_diagnostic_lines(lines)
        serialized = json.dumps(events)

        self.assertEqual(len(events), 7)
        self.assertEqual(
            [event["route_output_class"] for event in events[:5]],
            list(smoke_harness.ROUTE_OUTPUT_CLASSES),
        )
        self.assertEqual(events[0]["route_call"], "initial")
        self.assertEqual(events[-1]["route_call"], "retry")
        self.assertIsNone(events[-1]["route_output_class"])
        self.assertNotIn("route-secret", serialized)
        self.assertNotIn("request-secret", serialized)
        self.assertNotIn("evidence-secret", serialized)

    def test_route_diagnostic_snapshot_does_not_modify_source_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = smoke_harness.EvidenceStore(root / "source")
            source.add("system_info", "source evidence")
            source_hash_before = hashlib.sha256(
                b"".join(
                    path.relative_to(source.root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
                    for path in sorted(source.root.rglob("*"))
                    if path.is_file()
                )
            ).hexdigest()
            collector = smoke_harness.RouteDiagnosticCollector(root / "collector", store_mode="existing-snapshot")
            with mock.patch.object(smoke_harness.EvidenceStore, "for_workdir", return_value=source):
                snapshot = collector.new_evidence_store(Path("workdir"))
            snapshot.add("exec_shell_full_command", "snapshot evidence")
            source_hash_after = hashlib.sha256(
                b"".join(
                    path.relative_to(source.root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
                    for path in sorted(source.root.rglob("*"))
                    if path.is_file()
                )
            ).hexdigest()

            reloaded_source = smoke_harness.EvidenceStore(source.root)
            reloaded_source.load_index()

        self.assertEqual(len(reloaded_source.records), 1)
        self.assertEqual(len(snapshot.records), 2)
        self.assertEqual(source_hash_after, source_hash_before)

    def test_tools_off_removes_runtime_tool_names(self) -> None:
        scenario = smoke_harness.scenarios()["final_prefix_local"]

        self.assertEqual(smoke_harness.effective_allowed_tool_names(scenario, "off"), ())
        self.assertEqual(smoke_harness.effective_allowed_tool_names(scenario, "on"), scenario.allowed_tool_names)

    def test_run_scenario_passes_no_tools_when_tools_mode_is_off(self) -> None:
        report = mock.Mock(finish_reason="stop")
        with mock.patch.object(smoke_harness, "run_step", return_value=report) as run_step, \
            mock.patch.object(smoke_harness, "ChatRuntime"):
            smoke_harness.run_scenario(
                smoke_harness.SmokeScenario("chat", (smoke_harness.SmokeStep("hi"),)),
                backend=mock.Mock(base_url="http://127.0.0.1:12120"),
                workdir=Path("workdir"),
                max_tokens=32,
                temperature=0.0,
                timeout=30.0,
                tools_mode="off",
            )

        self.assertEqual(run_step.call_args.kwargs["allowed_tool_names"], ())

    def test_correctness_categorizer_detects_shell_error_and_mixed_output(self) -> None:
        self.assertEqual(smoke_harness.check_shell_error("exit_code=127 command not found", []), "correct")
        self.assertEqual(smoke_harness.check_shell_error_focus("failed with 127", []), "correct")
        self.assertEqual(smoke_harness.check_shell_error_focus("failed with 127 and printed line-19", []), "mixed_wrong")

    def test_correctness_categorizer_detects_fake_shell20_without_tool_call(self) -> None:
        self.assertEqual(smoke_harness.check_shell20("line-19", []), "fake_tool_output")
        self.assertEqual(smoke_harness.check_shell20("line-19", ["exec_shell_full_command"]), "correct")

    def test_web_error_checker_rejects_answer_from_memory(self) -> None:
        self.assertEqual(smoke_harness.check_web_error("The web search failed.", []), "correct")
        self.assertEqual(
            smoke_harness.check_web_error("The search failed, but Avola is in Sicily.", []),
            "wrong",
        )

    def test_deterministic_web_fixture_covers_success_none_and_error(self) -> None:
        from orbit.runtime import shell_guardrails

        original = shell_guardrails.search_web
        with smoke_harness.deterministic_web(True):
            self.assertIn("Orbit deterministic fixture", shell_guardrails.search_web("orbit fixture success"))
            self.assertIn("results: none", shell_guardrails.search_web("orbit fixture none"))
            self.assertIn("error:", shell_guardrails.search_web("where is Avola located?"))
        self.assertIs(shell_guardrails.search_web, original)

    def test_mtp_state_marks_failed_props_as_not_usable(self) -> None:
        state = smoke_harness.mtp_state_from_props(
            {
                "mtp_experimental_enabled": True,
                "mtp_initialized": False,
                "mtp_last_completion_success": False,
                "mtp_failure_reason": "failed to decode target prefill suffix",
            }
        )

        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["usable"])
        self.assertEqual(state["failure_reason"], "failed to decode target prefill suffix")

    def test_mtp_state_marks_session_ready_without_attempt_as_ready(self) -> None:
        state = smoke_harness.mtp_state_from_props(
            {
                "mtp_experimental_enabled": True,
                "mtp_initialized": True,
                "mtp_last_completion_success": False,
                "mtp_failure_reason": None,
                "mtp_fallback_reason": None,
            }
        )

        self.assertEqual(state["status"], "ready")
        self.assertFalse(state["usable"])

    def test_environment_summary_uses_mtp_state_fields(self) -> None:
        class Backend:
            def model_info(self):
                return type("Info", (), {"id": "m"})()

        args = smoke_harness.build_parser().parse_args([])
        env = smoke_harness.environment_summary(
            args=args,
            backend=Backend(),
            props={
                "backend": "orbit-native",
                "model_id": "m",
                "multimodal_available": True,
                "mtp_experimental_enabled": True,
                "mtp_initialized": False,
                "mtp_last_completion_success": False,
                "mtp_failure_reason": "probe failed",
            },
        )

        self.assertEqual(env["mtp"], "failed")
        self.assertFalse(env["mtp_usable"])
        self.assertEqual(env["mtp_failure_reason"], "probe failed")

    def test_environment_summary_redacts_locations_for_healing_diagnostics(self) -> None:
        class Backend:
            def model_info(self):
                return type("Info", (), {"id": "m"})()

        args = smoke_harness.build_parser().parse_args(
            [
                "--base-url", "http://127.0.0.1:9999",
                "--workdir", "/tmp/private-fixture",
                "--tool-healing-diagnostics",
            ]
        )
        env = smoke_harness.environment_summary(
            args=args,
            backend=Backend(),
            props={"backend": "orbit-native", "model_id": "m"},
        )

        self.assertEqual(env["base_url"], "<redacted>")
        self.assertEqual(env["workdir"], "<redacted>")
        self.assertNotIn("http://", json.dumps(env))
        self.assertNotIn("private-fixture", json.dumps(env))

    def test_environment_summary_records_final_prefix_and_runtime_metadata(self) -> None:
        class Backend:
            def model_info(self):
                return type("Info", (), {"id": "m"})()

        args = smoke_harness.build_parser().parse_args(
            [
                "--scenario", "final_prefix_local", "--final-prefix-mode", "on", "--timeout", "30",
                "--block-id", "abba-1-on-a", "--run-order", "ON", "--cooling-seconds", "15",
            ]
        )
        env = smoke_harness.environment_summary(
            args=args,
            backend=Backend(),
            props={
                "backend": "orbit-native",
                "ctx_size": 8192,
                "threads": 6,
                "threads_batch": 6,
                "batch_size": 256,
                "ubatch_size": 128,
                "final_prefix_experiment_enabled": True,
                "final_prefix_experiment_restore_count": 2,
                "final_prefix_reuse_enabled": True,
                "final_prefix_reuse_source": "stable",
                "final_prefix_reuse_config_error": None,
                "final_prefix_reuse_legacy_detected": False,
            },
        )

        self.assertEqual(env["final_prefix_mode"], "on")
        self.assertEqual(env["scenario"], ["final_prefix_local"])
        self.assertEqual(env["ctx"], 8192)
        self.assertEqual(env["final_prefix"]["restore_count"], 2)
        self.assertEqual(env["final_prefix_config"]["requested"], "on")
        self.assertTrue(env["final_prefix_config"]["server_client_parity"])
        self.assertNotIn("raw_value", env["final_prefix_config"]["client"])
        self.assertEqual(env["timeout"], 30.0)
        self.assertEqual(env["block_id"], "abba-1-on-a")
        self.assertEqual(env["run_order"], "ON")
        self.assertEqual(env["cooling_seconds"], 15.0)
        self.assertIsInstance(env["cpu_affinity"], list)

    def test_final_prefix_environment_controls_client_flag_and_restores_it(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"ORBIT_FINAL_PREFIX_REUSE": "old-stable", "ORBIT_FINAL_PREFIX_EXPERIMENT": "old-legacy"},
            clear=False,
        ):
            with smoke_harness.final_prefix_environment("on"):
                self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_REUSE"], "1")
                self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT", smoke_harness.os.environ)
            self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_REUSE"], "old-stable")
            self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_EXPERIMENT"], "old-legacy")

            with smoke_harness.final_prefix_environment("off"):
                self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_REUSE"], "0")
                self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT", smoke_harness.os.environ)
            self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_REUSE"], "old-stable")
            self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_EXPERIMENT"], "old-legacy")

    def test_final_prefix_harness_modes_cover_stable_legacy_conflict_and_invalid(self) -> None:
        expected = {
            "off": (False, "stable", None, False),
            "on": (True, "stable", None, False),
            "legacy-off": (False, "legacy", None, True),
            "legacy-on": (True, "legacy", None, True),
            "stable-off-legacy-on": (False, "stable", None, True),
            "stable-on-legacy-off": (True, "stable", None, True),
            "stable-invalid": (False, "stable", "invalid_stable_value", True),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode), mock.patch.dict(smoke_harness.os.environ, {}, clear=True):
                with smoke_harness.final_prefix_environment(mode):
                    resolved = smoke_harness.resolve_final_prefix_reuse()
                    self.assertEqual(
                        (resolved.enabled, resolved.source, resolved.validation_error, resolved.legacy_detected),
                        values,
                    )

    def test_managed_server_passes_final_prefix_flag_only_when_on(self) -> None:
        args = smoke_harness.build_parser().parse_args(["--manage-server", "--final-prefix-mode", "on"])
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(smoke_harness.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(smoke_harness, "wait_for_server"), \
            mock.patch.object(
                smoke_harness,
                "fresh_backend_props",
                side_effect=[
                    {},
                    {
                        "final_prefix_experiment_enabled": True,
                        "final_prefix_reuse_enabled": True,
                        "final_prefix_reuse_source": "stable",
                        "final_prefix_reuse_config_error": None,
                        "final_prefix_reuse_legacy_detected": False,
                    },
                ],
            ):
            with smoke_harness.managed_server(args):
                pass
        self.assertEqual(popen.call_args.kwargs["env"]["ORBIT_FINAL_PREFIX_REUSE"], "1")
        self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT", popen.call_args.kwargs["env"])

        args = smoke_harness.build_parser().parse_args(["--manage-server", "--final-prefix-mode", "off"])
        with mock.patch.dict(smoke_harness.os.environ, {"ORBIT_FINAL_PREFIX_EXPERIMENT": "1"}), \
            mock.patch.object(smoke_harness.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(smoke_harness, "wait_for_server"), \
            mock.patch.object(
                smoke_harness,
                "fresh_backend_props",
                side_effect=[
                    {},
                    {
                        "final_prefix_experiment_enabled": False,
                        "final_prefix_reuse_enabled": False,
                        "final_prefix_reuse_source": "stable",
                        "final_prefix_reuse_config_error": None,
                        "final_prefix_reuse_legacy_detected": False,
                    },
                ],
            ):
            with smoke_harness.managed_server(args):
                pass
        self.assertEqual(popen.call_args.kwargs["env"]["ORBIT_FINAL_PREFIX_REUSE"], "0")
        self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT", popen.call_args.kwargs["env"])

    def test_managed_server_rejects_an_already_used_base_url(self) -> None:
        args = smoke_harness.build_parser().parse_args(["--manage-server", "--final-prefix-mode", "off"])
        with mock.patch.object(smoke_harness, "fresh_backend_props", return_value={"backend": "orbit-native"}), \
            self.assertRaisesRegex(RuntimeError, "unused base URL"):
            with smoke_harness.managed_server(args):
                pass

    def test_server_command_can_enable_mtp_without_changing_final_prefix_mode(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            ["--manage-server", "--final-prefix-mode", "on", "--server-mtp"]
        )

        command = smoke_harness.server_command(args)

        self.assertIn("--mtp", command)
        self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT=1", command)

    def test_server_command_records_thinking_and_managed_tools_mode(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            ["--manage-server", "--server-thinking", "on", "--tools", "off"]
        )
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(smoke_harness.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(smoke_harness, "wait_for_server"), \
            mock.patch.object(
                smoke_harness,
                "fresh_backend_props",
                side_effect=[
                    {},
                    {
                        "final_prefix_experiment_enabled": True,
                        "final_prefix_reuse_enabled": True,
                        "final_prefix_reuse_source": "default",
                        "final_prefix_reuse_config_error": None,
                        "final_prefix_reuse_legacy_detected": False,
                    },
                ],
            ):
            with smoke_harness.managed_server(args):
                pass

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--think") + 1], "on")
        self.assertEqual(popen.call_args.kwargs["env"]["ORBIT_TOOLS"], "off")

    def test_process_rss_reads_linux_status(self) -> None:
        process = mock.Mock(pid=123)
        with mock.patch.object(smoke_harness.Path, "read_text", return_value="Name:\torbit\nVmRSS:\t2048 kB\n"):
            self.assertEqual(smoke_harness.process_rss_kib(process), 2048)

    def test_final_prefix_step_state_reports_counter_deltas(self) -> None:
        before = {
            "final_prefix_experiment_capture_count": 1,
            "final_prefix_experiment_restore_count": 3,
            "final_prefix_experiment_fallback_count": 0,
        }
        after = {
            "final_prefix_experiment_enabled": True,
            "final_prefix_experiment_initialized": True,
            "final_prefix_experiment_prefix_tokens": 64,
            "final_prefix_experiment_capture_count": 1,
            "final_prefix_experiment_restore_count": 4,
            "final_prefix_experiment_fallback_count": 0,
        }

        state = smoke_harness.final_prefix_step_state(before, after)

        self.assertEqual(state["capture_count_delta"], 0)
        self.assertEqual(state["restore_count_delta"], 1)
        self.assertEqual(state["fallback_count_delta"], 0)
        self.assertEqual(state["prefix_tokens"], 64)

    def test_final_prefix_validation_encodes_off_capture_and_on_restore_contract(self) -> None:
        final = smoke_harness.StepReport(
            case="pwd_followup",
            step=1,
            prompt="run pwd",
            prompt_kind="auto",
            completion_kind="route,final_from_tool",
            route_tokens=800,
            final_tokens=100,
            prompt_tokens=100,
            cached_tokens=64,
            evaluated_tokens=57,
            finish_reason="stop",
            tool_calls=1,
            tool_names=["exec_shell_full_command"],
            wall_ms=100.0,
            correctness_category="correct",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="/tmp",
            model_steps=[],
        )
        off_props = {
            "final_prefix_experiment_enabled": False,
            "final_prefix_experiment_capture_count": 0,
        }
        on_props = {
            "final_prefix_experiment_enabled": True,
            "final_prefix_experiment_prefix_tokens": 64,
            "final_prefix_experiment_capture_count": 1,
            "final_prefix_experiment_restore_count": 1,
            "final_prefix_experiment_fallback_count": 0,
        }

        self.assertIsNone(smoke_harness.final_prefix_validation_failure("off", [final], off_props))
        self.assertIsNone(smoke_harness.final_prefix_validation_failure("on", [final, final], on_props))
        self.assertEqual(
            smoke_harness.final_prefix_validation_failure(
                "on",
                [final, final],
                {**on_props, "final_prefix_experiment_restore_count": 0},
            ),
            "restore_missing",
        )
        self.assertIsNone(
            smoke_harness.final_prefix_validation_failure(
                "on",
                [final],
                {
                    "mtp_experimental_enabled": True,
                    "final_prefix_experiment_enabled": True,
                    "final_prefix_experiment_capture_count": 0,
                    "final_prefix_experiment_restore_count": 0,
                },
            )
        )
        self.assertIsNone(
            smoke_harness.final_prefix_validation_failure(
                "on",
                [],
                {
                    "final_prefix_experiment_enabled": True,
                    "final_prefix_experiment_capture_count": 0,
                    "final_prefix_experiment_restore_count": 0,
                },
                tools_mode="off",
            )
        )
        self.assertEqual(
            smoke_harness.final_prefix_validation_failure(
                "on",
                [],
                {
                    "final_prefix_experiment_enabled": True,
                    "final_prefix_experiment_capture_count": 1,
                    "final_prefix_experiment_restore_count": 0,
                },
                tools_mode="off",
            ),
            "tools_off_guard_failed",
        )

    def test_lifecycle_transition_reports_restart_and_thinking_state(self) -> None:
        eligible_props = {
            "final_prefix_experiment_initialized": True,
            "final_prefix_experiment_capture_count": 1,
            "final_prefix_experiment_restore_count": 1,
            "final_prefix_experiment_fallback_count": 0,
        }
        blocks = [
            smoke_harness.LifecycleBlock(
                block_id="before",
                server_pid=101,
                ctx=8192,
                thinking="off",
                initial_props={"final_prefix_experiment_initialized": False},
                final_props=eligible_props,
                reports=[],
                rss_samples=[],
            ),
            smoke_harness.LifecycleBlock(
                block_id="after",
                server_pid=202,
                ctx=4096,
                thinking="off",
                initial_props={"final_prefix_experiment_initialized": False},
                final_props=eligible_props,
                reports=[],
                rss_samples=[],
            ),
        ]

        restart = smoke_harness.lifecycle_transition_row("restart", blocks)
        thinking = smoke_harness.lifecycle_transition_row(
            "thinking",
            [
                smoke_harness.LifecycleBlock(
                    block_id="thinking-on",
                    server_pid=303,
                    ctx=8192,
                    thinking="on",
                    initial_props={"final_prefix_experiment_initialized": False},
                    final_props={
                        "final_prefix_experiment_capture_count": 0,
                        "final_prefix_experiment_restore_count": 0,
                    },
                    reports=[],
                    rss_samples=[],
                )
            ],
        )

        self.assertTrue(restart["passed"])
        self.assertEqual(restart["process_ids"], [101, 202])
        self.assertFalse(restart["transitions"][1]["initial_initialized"])
        self.assertTrue(thinking["passed"])
        self.assertEqual(thinking["transitions"][0]["eligibility"], "ineligible_thinking")

    def test_rss_samples_preserve_pid_order_and_compute_neutral_deltas(self) -> None:
        process = mock.Mock(pid=4242)
        values = iter((1000, 1200, 1210, 1220, 1230, 1240, 1250))
        labels = (
            "startup",
            "after_capture",
            "after_restore_10",
            "after_restore_25",
            "after_restore_50",
            "after_invalidation",
            "after_recapture",
        )
        with mock.patch.object(smoke_harness, "process_rss_kib", side_effect=lambda _process: next(values)):
            samples = []
            for index, label in enumerate(labels):
                props = {}
                if label == "after_invalidation":
                    props = {
                        "final_prefix_experiment_initialized": False,
                        "final_prefix_experiment_prefix_tokens": 0,
                    }
                elif label == "after_recapture":
                    props = {
                        "final_prefix_experiment_initialized": True,
                        "final_prefix_experiment_capture_count": 2,
                    }
                samples.append(
                    smoke_harness.rss_sample(
                        process,
                        label=label,
                        block_id="rss",
                        sequence=index,
                        props=props,
                    )
                )

        summary = smoke_harness.summarize_rss_samples(samples, block_id="rss")

        self.assertEqual(summary["sample_labels"], list(labels))
        self.assertEqual(summary["server_pid"], 4242)
        self.assertEqual(summary["startup_to_capture_delta_kib"], 200)
        self.assertEqual(summary["capture_to_restore50_delta_kib"], 30)
        self.assertFalse(summary["linear_growth_suspected"])
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["passed"])

    def test_rss_summary_handles_missing_samples_without_inference(self) -> None:
        summary = smoke_harness.summarize_rss_samples(
            [{"label": "startup", "rss_kib": None, "server_pid": 7}],
            block_id="rss",
        )

        self.assertIsNone(summary["startup_to_capture_delta_kib"])
        self.assertIsNone(summary["linear_growth_suspected"])
        self.assertFalse(summary["complete"])

    def test_lifecycle_runner_exercises_restart_ctx_and_thinking_transitions(self) -> None:
        recorded: list[tuple[str, int, str]] = []

        def fake_block(args, *, block_id, calls: int, rss_series=False):
            del calls, rss_series
            test_args = args
            eligible = test_args.server_thinking == "off"
            calls_state = 1 if eligible else 0
            recorded.append((block_id, test_args.ctx, test_args.server_thinking))
            return smoke_harness.LifecycleBlock(
                block_id=block_id,
                server_pid=100 + len(recorded),
                ctx=test_args.ctx,
                thinking=test_args.server_thinking,
                initial_props={"final_prefix_experiment_initialized": False},
                final_props={
                    "backend": "orbit-native",
                    "final_prefix_experiment_capture_count": calls_state,
                    "final_prefix_experiment_restore_count": calls_state,
                    "final_prefix_experiment_fallback_count": 0,
                },
                reports=[],
                rss_samples=[],
            )

        args = smoke_harness.build_parser().parse_args(
            [
                "--manage-server",
                "--final-prefix-mode",
                "on",
                "--lifecycle-check",
                "restart",
                "--lifecycle-check",
                "ctx-change",
                "--lifecycle-check",
                "thinking",
                "--ctx",
                "8192",
                "--ctx-change-to",
                "4096",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "run_lifecycle_block", side_effect=fake_block), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.run_lifecycle_checks(
                args,
                jsonl_path=Path(tmp) / "out.jsonl",
                markdown_path=Path(tmp) / "out.md",
            )

        self.assertEqual(rc, 0)
        self.assertEqual(
            recorded,
            [
                ("restart-before", 8192, "off"),
                ("restart-after", 8192, "off"),
                ("ctx-8192", 8192, "off"),
                ("ctx-4096", 4096, "off"),
                ("thinking-off", 8192, "off"),
                ("thinking-on", 8192, "on"),
            ],
        )

    def test_main_mtp_required_fails_when_post_run_props_not_usable(self) -> None:
        initial = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        final = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": False,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": "failed to decode target prefill suffix",
            "mtp_fallback_reason": "failed to decode target prefill suffix",
        }
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=initial), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value=final), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 2)

    def test_main_mtp_required_passes_when_post_run_props_usable(self) -> None:
        props = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": True,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=props), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value=props), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 0)

    def test_jsonl_output_contains_environment_and_step_rows(self) -> None:
        report = smoke_harness.StepReport(
            case="simple_chat",
            step=1,
            prompt="hi",
            prompt_kind="chat",
            completion_kind="final",
            route_tokens=None,
            final_tokens=12,
            prompt_tokens=12,
            cached_tokens=4,
            evaluated_tokens=8,
            finish_reason="stop",
            tool_calls=0,
            tool_names=[],
            wall_ms=10.0,
            correctness_category="correct",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="hello",
            model_steps=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.jsonl"
            smoke_harness.write_jsonl(
                path,
                {"version": "0.0.1", "git_head": "abc"},
                [report],
                extra_rows=[{"type": "lifecycle_summary", "operation": "restart", "passed": True}],
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["type"], "environment")
        self.assertEqual(rows[1]["type"], "step")
        self.assertEqual(rows[1]["case"], "simple_chat")
        self.assertIn("output_tokens", rows[1])
        self.assertIn("phase_wall_ms", rows[1])
        self.assertEqual(rows[2]["type"], "summary")
        self.assertEqual(rows[3]["type"], "lifecycle_summary")

    def test_route_diagnostics_keep_initial_retry_and_tool_correlation_separate(self) -> None:
        report = route_step_report(
            prompt="fixture request",
            completion_kind="route,route_retry,final_from_tool",
            route_tokens=100,
            final_tokens=50,
            prompt_tokens=50,
            cached_tokens=4,
            evaluated_tokens=46,
            tool_calls=1,
            tool_names=["exec_shell_full_command"],
            wall_ms=10.0,
            answer_excerpt="fixture answer",
            model_steps=[{"phase": "route_retry"}],
        )
        events = [
            {
                "route_call": "initial",
                "route_output_class": "malformed",
                "route_output_reason": "unaccepted_output",
                "parser_accepted": False,
                "finish_reason": "stop",
                "output_tokens": 10,
                "parsed_route": None,
                "outcome": "route_other_retry",
                "retry_reason": "explicit_web_search",
                "control_loop_surrogate": False,
            },
            {
                "route_call": "retry",
                "route_output_class": "canonical",
                "route_output_reason": "canonical_command",
                "parser_accepted": True,
                "finish_reason": "stop",
                "output_tokens": 5,
                "parsed_route": "FILESYSTEM",
                "outcome": "route_parsed_tool",
                "retry_reason": None,
                "control_loop_surrogate": False,
            },
        ]

        enriched = smoke_harness.enrich_step_route_diagnostics(
            report,
            events,
            step=smoke_harness.SmokeStep(
                "fixture request",
                expected_route="FILESYSTEM",
                expected_tool_names=("exec_shell_full_command",),
            ),
            enabled=True,
            scenario_family="web",
            process_id=4321,
            block_id="block-a",
            run_order="2",
            repetition=1,
        )

        self.assertEqual([event["route_call"] for event in enriched.route_outputs], ["initial", "retry"])
        self.assertEqual(enriched.final_parsed_route, "FILESYSTEM")
        self.assertTrue(enriched.route_correct)
        self.assertTrue(enriched.tool_correct)
        self.assertTrue(enriched.retry_required)
        self.assertFalse(enriched.route_fallback_used)
        self.assertEqual(enriched.process_id, 4321)
        self.assertEqual(enriched.block_id, "block-a")

    def test_route_classification_summary_counts_transitions_and_correlations(self) -> None:
        def report_with(events, *, family, final_route, fallback, retry, downstream=True):
            return route_step_report(
                case=family,
                correctness_category="correct" if downstream else "length_failure",
                scenario_family=family,
                route_diagnostics_enabled=True,
                route_outputs=events,
                final_parsed_route=final_route,
                route_correct=True,
                tool_correct=True,
                downstream_final_correct=downstream,
                retry_required=retry,
                route_fallback_used=fallback,
            )

        malformed = {
            "route_call": "initial", "route_output_class": "malformed", "route_correct": True,
            "tool_correct": True, "downstream_final_correct": True, "retry_required": True,
            "control_loop_surrogate": False,
        }
        canonical_retry = {
            "route_call": "retry", "route_output_class": "canonical", "route_correct": True,
            "tool_correct": True, "downstream_final_correct": True, "retry_required": True,
            "control_loop_surrogate": False,
        }
        control = {
            "route_call": "initial", "route_output_class": "control_loop", "route_correct": False,
            "tool_correct": False, "downstream_final_correct": True, "retry_required": True,
            "control_loop_surrogate": True,
        }
        legacy = {
            "route_call": "initial", "route_output_class": "legacy_tolerated", "route_correct": True,
            "tool_correct": True, "downstream_final_correct": True, "retry_required": False,
            "control_loop_surrogate": False,
        }
        direct = {
            "route_call": "initial", "route_output_class": "direct_prose", "route_correct": True,
            "tool_correct": True, "downstream_final_correct": True, "retry_required": False,
            "control_loop_surrogate": False,
        }
        malformed_failure = {
            "route_call": "initial", "route_output_class": "malformed", "route_correct": False,
            "tool_correct": True, "downstream_final_correct": False, "retry_required": True,
            "control_loop_surrogate": False,
        }
        malformed_retry = {**malformed_failure, "route_call": "retry"}
        reports = [
            report_with([malformed, canonical_retry], family="web", final_route="FILESYSTEM", fallback=False, retry=True),
            report_with([control], family="chat", final_route=None, fallback=True, retry=True),
            report_with([legacy], family="local", final_route="FILESYSTEM", fallback=False, retry=False),
            report_with([direct], family="chat", final_route=None, fallback=False, retry=False),
            report_with(
                [malformed_failure, malformed_retry],
                family="evidence",
                final_route=None,
                fallback=True,
                retry=True,
                downstream=False,
            ),
        ]

        summary = smoke_harness.summarize_route_classifications(reports)

        self.assertEqual(summary["class_counts"]["canonical"], 1)
        self.assertEqual(summary["class_counts"]["legacy_tolerated"], 1)
        self.assertEqual(summary["class_counts"]["direct_prose"], 1)
        self.assertEqual(summary["initial_class_counts"]["malformed"], 2)
        self.assertEqual(summary["retry_class_counts"]["canonical"], 1)
        self.assertEqual(summary["retry_class_counts"]["malformed"], 1)
        self.assertEqual(summary["malformed_to_retry_transitions"], 2)
        self.assertEqual(summary["control_loop_to_retry_transitions"], 1)
        self.assertEqual(summary["final_successful_decision_count"], 2)
        self.assertEqual(summary["fallback_count"], 2)
        self.assertEqual(summary["class_distribution_by_scenario_family"]["web"]["canonical"], 1)
        self.assertEqual(summary["class_distribution_by_scenario_family"]["evidence"]["malformed"], 2)
        self.assertEqual(summary["correctness_by_class"]["malformed"]["downstream_wrong"], 2)
        self.assertEqual(summary["empty_visible_control_output_surrogate_count"], 1)
        self.assertIn("not exact token-cycle proof", summary["control_loop_surrogate_note"])

    def test_route_classification_summary_aggregates_five_repetitions(self) -> None:
        reports = []
        for repetition in range(1, 6):
            reports.append(
                route_step_report(
                    case="fragile",
                    completion_kind="route,chat_final",
                    scenario_family="fragile_ambiguous",
                    repetition=repetition,
                    route_diagnostics_enabled=True,
                    route_outputs=[
                        {
                            "route_call": "initial",
                            "route_output_class": "canonical",
                            "route_correct": True,
                            "tool_correct": True,
                            "downstream_final_correct": True,
                            "retry_required": False,
                            "control_loop_surrogate": False,
                        }
                    ],
                    final_parsed_route="CHAT",
                    route_correct=True,
                    tool_correct=True,
                    downstream_final_correct=True,
                )
            )

        summary = smoke_harness.summarize_route_classifications(reports)

        self.assertEqual(summary["diagnostic_steps"], 5)
        self.assertEqual(summary["class_counts"]["canonical"], 5)
        self.assertEqual(summary["final_successful_decision_count"], 5)
        self.assertEqual(summary["class_distribution_by_scenario_family"]["fragile_ambiguous"]["canonical"], 5)
        self.assertEqual(summary["correctness_by_class"]["canonical"]["downstream_correct"], 5)

    def test_route_jsonl_rows_are_additive_and_do_not_copy_raw_diagnostic_content(self) -> None:
        report = route_step_report(
            prompt="existing step prompt",
            answer_excerpt="existing answer excerpt",
            route_diagnostics_enabled=True,
            route_outputs=[
                {
                    "route_call": "initial",
                    "route_output_class": "canonical",
                    "route_output_reason": "canonical_chat",
                    "parser_accepted": True,
                    "finish_reason": "stop",
                    "output_tokens": 5,
                    "parsed_route": "CHAT",
                }
            ],
            final_parsed_route="CHAT",
            route_correct=True,
            tool_correct=True,
            downstream_final_correct=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.jsonl"
            smoke_harness.write_jsonl(path, {"version": "test"}, [report])
            text = path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in text.splitlines()]

        self.assertEqual([row["type"] for row in rows], ["environment", "step", "summary", "route_classification_summary"])
        self.assertEqual(rows[1]["prompt"], "existing step prompt")
        route_metadata = json.dumps(rows[1]["route_outputs"])
        self.assertNotIn("existing step prompt", route_metadata)
        self.assertNotIn("existing answer excerpt", route_metadata)
        self.assertNotIn("raw_route_text", text)
        self.assertNotIn("route-secret", text)

    def test_missing_route_diagnostics_do_not_add_summary_or_change_old_rows(self) -> None:
        report = route_step_report(
            case="simple_chat",
            prompt="hi",
            prompt_kind="chat",
            completion_kind="final",
            route_tokens=None,
            final_tokens=2,
            prompt_tokens=2,
            evaluated_tokens=2,
            answer_excerpt="hello",
        )

        self.assertIsNone(smoke_harness.summarize_route_classifications([report]))

        values = report.__dict__.copy()
        values["route_diagnostics_enabled"] = True
        partial = smoke_harness.StepReport(**values)
        summary = smoke_harness.summarize_route_classifications([partial])

        self.assertEqual(summary["diagnostic_steps"], 1)
        self.assertEqual(summary["steps_with_route_events"], 0)
        self.assertEqual(summary["missing_route_diagnostic_steps"], 1)
        self.assertEqual(summary["class_counts"], smoke_harness.empty_route_class_counts())

    def test_tool_healing_events_join_by_attempt_id_and_classify_active_divergence(self) -> None:
        attempt_id = "a" * 32
        lines = [
            json.dumps(
                {
                    "event": "kv_diag_tool_healing_shadow",
                    "attempt_id": attempt_id,
                    "attempt_detected": True,
                    "signals": ["backend_tool_calls"],
                    "candidate_count": 1,
                    "candidate_source": "backend",
                    "repairs": ["unwrap_function_object", "decode_arguments_string"],
                    "shadow_outcome": "rejected_validation",
                    "validation_error": "additional_property",
                    "finish_reason": "tool_calls",
                    "output_tokens": 3,
                    "healing_us": 82.5,
                    "candidate_hash": "1" * 16,
                    "normalized_tool_name_hash": "2" * 16,
                    "normalized_arguments_hash": "3" * 16,
                }
            ),
            json.dumps(
                {
                    "event": "kv_diag_tool_healing_terminal",
                    "attempt_id": attempt_id,
                    "active_candidate_count": 1,
                    "active_tool_name_hash": "2" * 16,
                    "active_arguments_hash": "3" * 16,
                    "active_outcome": "executed",
                    "agreement": "active_only",
                }
            ),
        ]

        attempts = smoke_harness.parse_tool_healing_diagnostic_lines(lines)

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["attempt_id"], attempt_id)
        self.assertEqual(attempts[0]["active_outcome"], "executed")
        self.assertIn("extra_argument", attempts[0]["categories"])
        self.assertIn("active_ignored_extra", attempts[0]["categories"])
        self.assertNotIn("candidate_excerpt", attempts[0])

    def test_tool_healing_summary_reports_rates_latency_and_execution_assessment(self) -> None:
        step = smoke_harness.SmokeStep(
            "fixture",
            mode="tool",
            expected_tool_names=("system_info",),
            expected_active_outcome="executed",
        )
        report = route_step_report(
            case="healing",
            prompt="secret request",
            answer_excerpt="secret answer",
            tool_names=["system_info"],
            tool_calls=1,
            model_steps=[{"phase": "tool_call", "prompt_tokens": 12}],
        )
        attempts = [
            {
                "attempt_id": "b" * 32,
                "attempt_detected": True,
                "categories": ["valid_first_pass"],
                "formal_repairable": False,
                "strict_outcome": "valid_shadow_candidate",
                "active_outcome": "executed",
                "agreement": "exact_match",
                "healing_us": 80.0,
            },
            {
                "attempt_id": "c" * 32,
                "attempt_detected": True,
                "categories": ["recoverable_trailing_comma", "superseded"],
                "formal_repairable": True,
                "strict_outcome": "valid_shadow_candidate",
                "active_outcome": "superseded",
                "agreement": "shadow_only",
                "healing_us": 120.0,
            },
        ]
        enriched = smoke_harness.enrich_step_tool_healing(report, attempts, step=step, enabled=True)

        summary = smoke_harness.summarize_tool_healing([enriched])

        self.assertEqual(summary["attempts"], 2)
        self.assertEqual(summary["category_counts"]["valid_first_pass"], 1)
        self.assertEqual(summary["category_counts"]["recoverable_trailing_comma"], 1)
        self.assertEqual(summary["first_pass_valid_rate"], 0.5)
        self.assertEqual(summary["deterministic_repair_candidate_rate"], 0.5)
        self.assertEqual(summary["strict_active_agreement_rate"], 0.5)
        self.assertEqual(summary["healing_us_median"], 100.0)
        self.assertEqual(summary["healing_us_p95"], 120.0)
        self.assertEqual(enriched.tool_healing_attempts[0]["execution_assessment"], "correct_execution")

    def test_tool_healing_jsonl_is_content_free_and_additive(self) -> None:
        secret = "SECRET-REQUEST-TOKEN"
        report = route_step_report(
            prompt=secret,
            answer_excerpt="SECRET-ANSWER",
            tool_healing_diagnostics_enabled=True,
            tool_healing_attempts=[
                {
                    "attempt_id": "d" * 32,
                    "scenario": "healing",
                    "categories": ["valid_first_pass"],
                    "active_outcome": "executed",
                    "healing_us": 75.0,
                }
            ],
        )
        env = {
            "version": "test",
            "model": "gemma4:12b-it-native",
            "ctx": 8192,
            "threads": 6,
            "mtp": "on",
            "mmproj": "loaded",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "healing.jsonl"
            smoke_harness.write_jsonl(path, env, [report])
            text = path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in text.splitlines()]

        self.assertNotIn(secret, text)
        self.assertNotIn("SECRET-ANSWER", text)
        self.assertEqual(rows[1]["prompt"], "<redacted>")
        self.assertEqual(rows[1]["answer_excerpt"], "<redacted>")
        self.assertIn("tool_healing_attempt", [row["type"] for row in rows])
        self.assertEqual(rows[-1]["type"], "tool_healing_summary")

    def test_tool_healing_replay_is_labelled_and_contains_no_raw_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = smoke_harness.run_tool_healing_replay(Path(tmp))
        summary = rows[-1]
        serialized = json.dumps(rows)

        self.assertEqual(summary["type"], "tool_healing_replay_summary")
        self.assertEqual(summary["cases"], 17)
        self.assertEqual(summary["false_positives"], 0)
        self.assertEqual(summary["false_negatives"], 0)
        self.assertNotIn("file:///etc/passwd", serialized)
        self.assertNotIn("rm -f", serialized)

    def test_main_tools_off_metadata_matches_runtime_mode(self) -> None:
        captured_env: list[dict[str, object]] = []
        captured_modes: list[str] = []

        def fake_run_scenario(*_args, **kwargs):
            captured_modes.append(kwargs["tools_mode"])
            return []

        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value={}), \
            mock.patch.object(smoke_harness, "settled_backend_props", return_value={}), \
            mock.patch.object(smoke_harness, "run_scenario", side_effect=fake_run_scenario), \
            mock.patch.object(smoke_harness, "write_jsonl", side_effect=lambda _p, env, _r: captured_env.append(env)), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(
                ["--scenario", "simple_chat", "--no-web", "--tools", "off", "--output-dir", tmp]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(captured_modes, ["off"])
        self.assertEqual(captured_env[0]["tools"], "off")

    def test_markdown_summary_contains_expected_columns(self) -> None:
        report = smoke_harness.StepReport(
            case="pwd_followup",
            step=2,
            prompt="what directory was that?",
            prompt_kind="auto",
            completion_kind="route,final",
            route_tokens=700,
            final_tokens=300,
            prompt_tokens=300,
            cached_tokens=4,
            evaluated_tokens=296,
            finish_reason="stop",
            tool_calls=0,
            tool_names=[],
            wall_ms=123.0,
            correctness_category="partial_baseline",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="/tmp",
            model_steps=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            smoke_harness.write_markdown(path, {"version": "0.0.1", "git_head": "abc", "backend": "orbit-native"}, [report])
            text = path.read_text(encoding="utf-8")

        self.assertIn("| Case | Step | Kind | Route | Final |", text)
        self.assertIn("pwd_followup", text)
        self.assertIn("partial_baseline", text)
        self.assertIn("## Repetition Summary", text)

    def test_summary_reports_median_and_final_prefix_counter_deltas(self) -> None:
        rows = []
        for cached, evaluated, wall, restore in (
            (4, 96, 20.0, 0),
            (64, 36, 14.0, 1),
            (64, 36, 15.0, 1),
            (64, 36, 13.0, 1),
            (64, 36, 16.0, 1),
        ):
            rows.append(
                smoke_harness.StepReport(
                    case="pwd",
                    step=1,
                    prompt="run pwd",
                    prompt_kind="auto",
                    completion_kind="route,final_from_tool",
                    route_tokens=800,
                    final_tokens=100,
                    prompt_tokens=100,
                    cached_tokens=cached,
                    evaluated_tokens=evaluated,
                    finish_reason="stop",
                    tool_calls=1,
                    tool_names=["exec_shell_full_command"],
                    wall_ms=wall,
                    correctness_category="correct",
                    raw_leak=False,
                    fake_output=False,
                    loop=False,
                    notes="",
                    answer_excerpt="/tmp",
                    model_steps=[],
                    final_prefix={"restore_count_delta": restore},
                )
            )

        summary = smoke_harness.summarize_reports(rows)[0]

        self.assertEqual(summary["cached_tokens"]["median"], 64.0)
        self.assertEqual(summary["evaluated_tokens"]["median"], 36.0)
        self.assertEqual(summary["runs"], 5)
        self.assertEqual(summary["restore_delta"], 4)
        self.assertEqual(summary["restored"]["runs"], 4)
        self.assertEqual(summary["restored"]["cached_tokens"]["min"], 64.0)

    def test_summary_aggregates_fifty_stability_calls(self) -> None:
        row = smoke_harness.StepReport(
            case="final_prefix_mixed",
            step=1,
            prompt="run pwd",
            prompt_kind="auto",
            completion_kind="route,final_from_tool",
            route_tokens=780,
            final_tokens=100,
            prompt_tokens=100,
            cached_tokens=64,
            evaluated_tokens=36,
            finish_reason="stop",
            tool_calls=1,
            tool_names=["exec_shell_full_command"],
            wall_ms=12.0,
            correctness_category="correct",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="/tmp",
            model_steps=[],
            final_prefix={"restore_count_delta": 1, "fallback_count_delta": 0},
        )

        summary = smoke_harness.summarize_reports([row] * 50)[0]

        self.assertEqual(summary["runs"], 50)
        self.assertEqual(summary["correct"], 50)
        self.assertEqual(summary["restore_delta"], 50)
        self.assertEqual(summary["fallback_delta"], 0)

    def test_phase_timing_summary_separates_model_and_non_model_wall(self) -> None:
        timing = smoke_harness.phase_timing_summary(
            [("route", 10.0), ("final_from_tool", 20.0), ("final_from_tool", 5.0)],
            40.0,
        )

        self.assertEqual(timing["route"], 10.0)
        self.assertEqual(timing["final_from_tool"], 25.0)
        self.assertEqual(timing["non_model_wall_ms"], 5.0)
        self.assertEqual(smoke_harness.estimated_generation_ms(10, 5.0), 2000.0)

    def test_model_step_to_json_includes_evaluated_tokens(self) -> None:
        row = smoke_harness.model_step_to_json(
            ModelStepMetrics(
                loop=1,
                phase="final",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=5,
                cached_tokens=40,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
                tool_calls=0,
            )
        )

        self.assertEqual(row["evaluated_tokens"], 60)

    def test_run_step_timeout_requests_cancel_and_returns_timeout_report(self) -> None:
        runtime = mock.Mock()

        def slow_ask_chat(*args, **kwargs):
            time.sleep(0.2)
            raise AssertionError("worker should have been abandoned after timeout")

        runtime.ask_chat.side_effect = slow_ask_chat
        with mock.patch.object(smoke_harness, "request_backend_cancel", return_value=True) as mocked_cancel, \
            mock.patch.object(smoke_harness, "wait_for_backend_idle", return_value={"in_flight": False}):
            report = smoke_harness.run_step(
                runtime,
                scenario="simple_chat",
                step_index=1,
                step=smoke_harness.SmokeStep("hi", mode="chat", checker_name="nonempty"),
                workdir=Path("workdir"),
                max_tokens=16,
                temperature=0.0,
                base_url="http://127.0.0.1:12120",
                timeout=0.01,
            )

        mocked_cancel.assert_called_once()
        self.assertEqual(report.finish_reason, "timeout")
        self.assertEqual(report.completion_kind, "timeout")
        self.assertIn("cancel_requested", report.notes)
        self.assertIn("cleanup_ok", report.notes)
        self.assertTrue(report.lifecycle["timeout_observed"])
        self.assertFalse(report.lifecycle["automatic_cancel"])
        self.assertTrue(report.lifecycle["explicit_cancel_used"])
        self.assertTrue(report.lifecycle["cleanup_healthy"])

    def test_main_returns_timeout_failure_when_report_times_out(self) -> None:
        timeout_report = smoke_harness.StepReport(
            case="shell20",
            step=1,
            prompt="run python3 -c 'for i in range(20): print(f\"line-{i}\")'",
            prompt_kind="auto",
            completion_kind="timeout",
            route_tokens=None,
            final_tokens=None,
            prompt_tokens=None,
            cached_tokens=None,
            evaluated_tokens=None,
            finish_reason="timeout",
            tool_calls=0,
            tool_names=[],
            wall_ms=60001.0,
            correctness_category="not_evaluated",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="timeout,cancel_requested,cleanup_ok",
            answer_excerpt="timeout",
            model_steps=[],
        )
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value={}), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value={}), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[timeout_report]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "shell20", "--no-web", "--output-dir", tmp, "--timeout", "60"])

        self.assertEqual(rc, 1)

    def test_wait_for_backend_idle_returns_when_inflight_clears(self) -> None:
        with mock.patch.object(
            smoke_harness,
            "fresh_backend_props",
            side_effect=[{"in_flight": True}, {"in_flight": False, "mtp_enabled": True}],
        ):
            props = smoke_harness.wait_for_backend_idle("http://127.0.0.1:12120", 1.0, poll_interval=0.0)

        self.assertEqual(props.get("in_flight"), False)

    def test_settled_backend_props_waits_past_cancelled_snapshot(self) -> None:
        with mock.patch.object(
            smoke_harness,
            "fresh_backend_props",
            side_effect=[
                {"in_flight": False, "mtp_fallback_reason": "cancelled", "mtp_failure_reason": None},
                {"in_flight": False, "mtp_fallback_reason": None, "mtp_failure_reason": None, "mtp_initialized": True},
            ],
        ):
            props = smoke_harness.settled_backend_props("http://127.0.0.1:12120", 1.0, settle_seconds=0.5, poll_interval=0.0)

        self.assertIsNone(props.get("mtp_fallback_reason"))
        self.assertTrue(props.get("mtp_initialized"))

    def test_run_scenario_stops_after_timeout_report(self) -> None:
        scenario = smoke_harness.SmokeScenario(
            "shell20",
            (
                smoke_harness.SmokeStep("first"),
                smoke_harness.SmokeStep("second"),
            ),
        )
        timeout_report = smoke_harness.StepReport(
            case="shell20",
            step=1,
            prompt="first",
            prompt_kind="auto",
            completion_kind="timeout",
            route_tokens=None,
            final_tokens=None,
            prompt_tokens=None,
            cached_tokens=None,
            evaluated_tokens=None,
            finish_reason="timeout",
            tool_calls=0,
            tool_names=[],
            wall_ms=60000.0,
            correctness_category="not_evaluated",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="timeout",
            answer_excerpt="timeout",
            model_steps=[],
        )
        with mock.patch.object(smoke_harness, "run_step", return_value=timeout_report) as mocked_run_step, \
            mock.patch.object(smoke_harness, "ChatRuntime"):
            reports = smoke_harness.run_scenario(
                scenario,
                backend=mock.Mock(base_url="http://127.0.0.1:12120"),
                workdir=Path("workdir"),
                max_tokens=128,
                temperature=0.0,
                timeout=60.0,
            )

        self.assertEqual(len(reports), 1)
        mocked_run_step.assert_called_once()

    def test_main_uses_settled_props_for_environment_row(self) -> None:
        initial = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        settled = {
            "backend": "orbit-native",
            "model_id": "m",
            "multimodal_available": True,
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": True,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        captured_env: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=initial), \
            mock.patch.object(smoke_harness, "settled_backend_props", return_value=settled), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[]), \
            mock.patch.object(smoke_harness, "write_jsonl", side_effect=lambda _p, env, _r: captured_env.append(env)), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 0)
        self.assertEqual(captured_env[0]["mtp"], "on")
        self.assertTrue(captured_env[0]["mtp_usable"])


if __name__ == "__main__":
    unittest.main()
