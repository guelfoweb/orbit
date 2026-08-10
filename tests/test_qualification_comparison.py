from __future__ import annotations

from dataclasses import replace
import json
import unittest
from pathlib import Path

from orbit.qualification.fixtures import FixtureError, load_fixture_set, load_fixture_text
from orbit.qualification.reporting import comparison_json, format_comparison_summary
from orbit.qualification.runner import build_optimization_comparison, compare_runs
from orbit.qualification.schema import (
    AggregateMetrics,
    CallMetric,
    CommonGate,
    ComparisonExecution,
    FixtureObservation,
    LifecycleOutcome,
    QualificationRun,
    Reason,
    RunProvenance,
    Status,
    ToolCallRecord,
    ToolOutcomeRecord,
)
from orbit.qualification.validators import validate_observation
from scripts.orbit_qualify_compare import _parse_overrides


ROOT = Path(__file__).parents[1]


def fixtures():
    return load_fixture_text(json.dumps({
        "schema_version": 1,
        "fixtures": [{
            "name": "pwd_route",
            "capability": "tools",
            "profiles": ["profile-a"],
            "request": {"prompt": "pwd", "tools": True},
            "expect": {
                "route": "FILESYSTEM",
                "tool_calls": [{
                    "name": "exec_shell_full_command",
                    "arguments": {"command": "pwd"},
                }],
                "finish_reason": "stop",
                "max_model_calls": 1,
            },
            "parity": {"mode": "structural"},
        }],
    }))


def metric(*, evaluated=800, cached=0, wall=20.0, input_tokens=800):
    return CallMetric("route", input_tokens, evaluated, cached, 5, 30.0, 8.0, wall, "stop")


def observation(*, call=None, value=None, wall=20.0):
    tool = call or ToolCallRecord("exec_shell_full_command", {"command": "pwd"})
    item = value or metric(wall=wall)
    return FixtureObservation(
        "FILESYSTEM", (tool,), (), "FILESYSTEM", "stop", 1, 0, (item,), None,
        LifecycleOutcome(True, "clean"), 1024, wall_seconds=wall,
    )


def provenance(fixture_hash, *, profile="profile-a", model="model-a"):
    return RunProvenance(
        1, fixture_hash, "deadbeef", profile, model, "embedded", "template-hash",
        "orbit-native", "backend-revision", {"ctx_size": 8192}, {"cpu": "fake"},
        {"peak_rss_bytes": "fake"},
    )


def run(value, *, profile="profile-a", model="model-a"):
    fixture_set = fixtures()
    result = validate_observation(fixture_set.fixtures[0], value)
    status = Status.PASS if result.status is Status.PASS else Status.FAIL
    return QualificationRun(
        provenance(fixture_set.content_hash, profile=profile, model=model),
        (CommonGate("identity", Status.PASS, Reason("verified_profile", "verified")),),
        (result,), result.aggregate_metrics, status, status.value,
    )


def side(label, pid, value, *, startup=10.0, environment=None, profile="profile-a", model="model-a"):
    return ComparisonExecution(
        label, pid, startup, {"environment": environment or {}},
        run(value, profile=profile, model=model),
    )


class OptimizationComparisonTests(unittest.TestCase):
    def test_optimization_fixture_file_has_only_controlled_cases(self) -> None:
        value = load_fixture_set(ROOT / "qualification/fixtures/optimizations-v1.json")
        self.assertEqual(
            [item.name for item in value.fixtures],
            ["chat_route_first", "pwd_route", "pwd_final", "chat_route_second"],
        )
        self.assertTrue(value.fixtures[0].request.full_request)
        invalid = (ROOT / "qualification/fixtures/optimizations-v1.json").read_text().replace(
            '"full_request": true', '"full_request": 1', 1
        )
        with self.assertRaisesRegex(FixtureError, "invalid_type"):
            load_fixture_text(invalid)

    def test_cache_redistribution_preserves_parity_and_reports_wall_delta(self) -> None:
        baseline = side("baseline", 101, observation())
        candidate = side(
            "candidate", 202,
            observation(value=metric(evaluated=32, cached=768, wall=2.0), wall=2.0),
            environment={"ORBIT_QWEN_ROUTE_PREFIX_REUSE": "1"},
        )
        comparison = build_optimization_comparison(fixtures(), baseline, candidate)
        self.assertTrue(comparison.parity[0].equivalent)
        self.assertTrue(comparison.performance_comparison_valid)
        change = comparison.performance["fixtures"]["pwd_route"]["wall"]
        self.assertEqual(change["absolute_change_seconds"], -18.0)
        self.assertEqual(change["percent_change"], -90.0)

    def test_extra_tool_argument_invalidates_parity_and_suppresses_gain(self) -> None:
        baseline = side("baseline", 101, observation())
        candidate = side("candidate", 202, observation(call=ToolCallRecord(
            "exec_shell_full_command", {"command": "pwd", "timeout": 10}
        ), wall=1.0))
        comparison = build_optimization_comparison(fixtures(), baseline, candidate)
        self.assertFalse(comparison.parity[0].equivalent)
        self.assertIn("tool_calls", comparison.parity[0].mismatches)
        self.assertFalse(comparison.performance_comparison_valid)
        self.assertIsNone(comparison.performance)
        payload = comparison_json(comparison)
        self.assertNotIn("percent_change", payload)
        self.assertNotIn("change +", format_comparison_summary(comparison))

    def test_deterministic_tool_output_mismatch_invalidates_parity(self) -> None:
        baseline_value = replace(
            observation(),
            tool_outcomes=(ToolOutcomeRecord("exec_shell_full_command", "success", 0, "a"),),
        )
        candidate_value = replace(
            observation(),
            tool_outcomes=(ToolOutcomeRecord("exec_shell_full_command", "success", 0, "b"),),
        )
        comparison = build_optimization_comparison(
            fixtures(), side("baseline", 101, baseline_value), side("candidate", 202, candidate_value)
        )
        self.assertFalse(comparison.performance_comparison_valid)
        self.assertIn("tool_outcomes", comparison.parity[0].mismatches)

    def test_startup_and_request_wall_are_separate(self) -> None:
        baseline = side("baseline", 101, observation(wall=20.0), startup=10.0)
        candidate = side("candidate", 202, observation(wall=2.0), startup=30.0)
        comparison = build_optimization_comparison(fixtures(), baseline, candidate)
        self.assertEqual(comparison.performance["startup"]["absolute_change_seconds"], 20.0)
        self.assertEqual(comparison.performance["aggregate"]["wall"]["absolute_change_seconds"], -18.0)
        self.assertIn("pwd_route wall: 20.00s -> 2.00s", format_comparison_summary(comparison))

    def test_zero_and_nonfinite_startup_walls_are_safe(self) -> None:
        zero = build_optimization_comparison(
            fixtures(), side("baseline", 101, observation(), startup=0.0),
            side("candidate", 202, observation(), startup=1.0),
        )
        self.assertIsNone(zero.performance["startup"]["percent_change"])
        nonfinite = build_optimization_comparison(
            fixtures(), side("baseline", 101, observation(), startup=float("nan")),
            side("candidate", 202, observation(), startup=float("inf")),
        )
        payload = json.loads(comparison_json(nonfinite))
        self.assertIsNone(payload["baseline"]["startup_wall_seconds"])
        self.assertIsNone(payload["candidate"]["startup_wall_seconds"])
        self.assertIsNone(payload["performance"]["startup"]["absolute_change_seconds"])

    def test_nonfinite_fixture_and_call_metrics_serialize_as_null(self) -> None:
        baseline_value = observation(
            value=replace(metric(), wall_seconds=float("inf")),
            wall=float("nan"),
        )
        comparison = build_optimization_comparison(
            fixtures(), side("baseline", 101, baseline_value),
            side("candidate", 202, observation(wall=1.0)),
        )
        payload = json.loads(comparison_json(comparison))
        fixture = payload["baseline"]["result"]["fixtures"][0]
        self.assertIsNone(fixture["calls"][0]["wall_seconds"])
        self.assertIsNone(fixture["aggregate_metrics"]["wall_seconds"])
        self.assertIsNone(payload["performance"]["fixtures"]["pwd_route"]["wall"]["percent_change"])

    def test_missing_metrics_remain_null_without_changing_parity(self) -> None:
        missing = metric(input_tokens=None, evaluated=None, cached=None)
        comparison = build_optimization_comparison(
            fixtures(), side("baseline", 101, observation()),
            side("candidate", 202, observation(value=missing)),
        )
        self.assertTrue(comparison.performance_comparison_valid)
        self.assertIsNone(comparison.candidate.result.aggregate_metrics.input_tokens)
        self.assertIsNone(comparison.performance["aggregate"]["candidate"].input_tokens)

    def test_token_count_differences_are_metrics_not_operational_parity(self) -> None:
        candidate_metric = metric(input_tokens=804, evaluated=36, cached=768, wall=2.0)
        comparison = build_optimization_comparison(
            fixtures(), side("baseline", 101, observation()),
            side("candidate", 202, observation(value=candidate_metric, wall=2.0)),
        )
        self.assertTrue(comparison.parity[0].equivalent)
        self.assertTrue(comparison.performance_comparison_valid)
        self.assertEqual(comparison.performance["aggregate"]["baseline"].input_tokens, 800)
        self.assertEqual(comparison.performance["aggregate"]["candidate"].input_tokens, 804)

    def test_process_and_configuration_provenance_are_retained(self) -> None:
        baseline = side("baseline", 101, observation(), environment={"MODE": "off"})
        candidate = side("candidate", 202, observation(), environment={"MODE": "on"})
        candidate = replace(
            candidate,
            result=replace(
                candidate.result,
                provenance=replace(
                    candidate.result.provenance,
                    measurement_scope={"peak_rss_bytes": "server PID 202"},
                ),
            ),
        )
        comparison = build_optimization_comparison(fixtures(), baseline, candidate)
        payload = json.loads(comparison_json(comparison))
        self.assertEqual(payload["baseline"]["server_pid"], 101)
        self.assertEqual(payload["candidate"]["configuration"]["environment"], {"MODE": "on"})
        same_process = build_optimization_comparison(
            fixtures(), baseline, replace(candidate, server_pid=101)
        )
        self.assertFalse(same_process.performance_comparison_valid)
        self.assertIn("process_not_isolated", same_process.mismatches)
        invalid_process = build_optimization_comparison(
            fixtures(), baseline, replace(candidate, server_pid=0)
        )
        self.assertFalse(invalid_process.performance_comparison_valid)
        self.assertIn("invalid_server_pid", invalid_process.mismatches)
        different_config = build_optimization_comparison(
            fixtures(), baseline,
            replace(candidate, configuration={"ctx": 4096, "environment": {"MODE": "on"}}),
        )
        self.assertFalse(different_config.performance_comparison_valid)
        self.assertIn("execution_configuration_mismatch", different_config.mismatches)

    def test_volatile_available_ram_is_descriptive_and_non_gating(self) -> None:
        hardware = {
            "machine": "host-a", "cpu": "cpu-a", "physical_cores": "6",
            "logical_cores": "12", "ram_total": "64 GB", "ram_available": "31 GB",
            "os_name": "Linux",
        }
        baseline = side("baseline", 101, observation())
        baseline = replace(
            baseline,
            result=replace(
                baseline.result,
                provenance=replace(baseline.result.provenance, hardware=hardware),
            ),
        )
        candidate = side("candidate", 202, observation())
        candidate = replace(
            candidate,
            result=replace(
                candidate.result,
                provenance=replace(
                    candidate.result.provenance,
                    hardware={**hardware, "ram_available": "30 GB"},
                ),
            ),
        )
        comparison = build_optimization_comparison(fixtures(), baseline, candidate)
        payload = json.loads(comparison_json(comparison))
        self.assertTrue(comparison.performance_comparison_valid)
        self.assertEqual(payload["baseline"]["result"]["provenance"]["hardware"]["ram_available"], "31 GB")
        self.assertEqual(payload["candidate"]["result"]["provenance"]["hardware"]["ram_available"], "30 GB")

    def test_stable_hardware_mismatch_is_rejected(self) -> None:
        hardware = {
            "machine": "host-a", "cpu": "cpu-a", "physical_cores": "6",
            "logical_cores": "12", "ram_total": "64 GB", "ram_available": "31 GB",
            "os_name": "Linux",
        }
        baseline = run(observation())
        baseline = replace(
            baseline,
            provenance=replace(baseline.provenance, hardware=hardware),
        )
        for field, value in (
            ("machine", "host-b"), ("cpu", "cpu-b"), ("physical_cores", "4"),
            ("logical_cores", "8"), ("ram_total", "32 GB"),
            ("os_name", "OtherOS x86_64"),
        ):
            with self.subTest(field=field):
                candidate = replace(
                    baseline,
                    provenance=replace(
                        baseline.provenance,
                        hardware={**hardware, field: value},
                    ),
                )
                with self.assertRaisesRegex(ValueError, "same model and configuration"):
                    compare_runs(fixtures(), baseline, candidate)

    def test_comparison_environment_is_explicit_and_bounded(self) -> None:
        self.assertEqual(
            _parse_overrides(["ORBIT_QWEN_ROUTE_PREFIX_REUSE=0"]),
            {"ORBIT_QWEN_ROUTE_PREFIX_REUSE": "0"},
        )
        with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
            _parse_overrides(["PATH=/tmp"])
        with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
            _parse_overrides([
                "ORBIT_KV_PREFIX_PREWARM=off",
                "ORBIT_KV_PREFIX_PREWARM=startup",
            ])

    def test_optimization_mode_rejects_cross_model_or_configuration_comparison(self) -> None:
        baseline = run(observation())
        different_model = run(observation(), profile="profile-b", model="model-b")
        with self.assertRaisesRegex(ValueError, "same model and configuration"):
            compare_runs(fixtures(), baseline, different_model)
        different_config = replace(
            baseline,
            provenance=replace(baseline.provenance, runtime_configuration={"ctx_size": 4096}),
        )
        with self.assertRaisesRegex(ValueError, "same model and configuration"):
            compare_runs(fixtures(), baseline, different_config)
        different_revision = replace(
            baseline,
            provenance=replace(baseline.provenance, git_revision="different"),
        )
        with self.assertRaisesRegex(ValueError, "same model and configuration"):
            compare_runs(fixtures(), baseline, different_revision)


if __name__ == "__main__":
    unittest.main()
