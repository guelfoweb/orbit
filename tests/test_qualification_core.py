from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from orbit.backend.base import ChatResult
from orbit.qualification.fixtures import load_fixture_text
from orbit.qualification.reporting import format_summary, result_json
from orbit.qualification.runner import QualificationRunner, RuntimeFixtureExecutor, _artifact_evidence, compare_runs
from orbit.qualification.schema import (
    AggregateMetrics, ArtifactEvidence, CallMetric, ComparisonMode, FixtureObservation, LifecycleOutcome,
    RunProvenance, Status, ToolCallRecord,
)
from orbit.qualification.validators import compare_fixture_results, validate_observation


def fixture(*, artifact: bool = False):
    expect: dict[str, object] = {
        "route": "FILESYSTEM", "tool_calls": [{"name": "exec_shell_full_command", "arguments": {"command": "pwd"}}],
        "finish_reason": "stop", "max_model_calls": 3,
    }
    capability = "tools"
    if artifact:
        capability = "artifacts"
        expect = {
            "tool_calls": [{"name": "write_artifact"}, {"name": "verify_artifact"}],
            "finish_reason": "stop", "max_model_calls": 5,
            "artifact": {"path": "qualification.json", "json_equals": {"orbit": "qualified", "version": 1},
                         "publication": True, "verification": True},
        }
    payload = {"schema_version": 1, "fixtures": [{
        "name": "fixture", "capability": capability, "profiles": ["profile-a"],
        "request": {"prompt": "test", "tools": True}, "expect": expect,
        "parity": {"mode": "structural"},
    }]}
    return load_fixture_text(json.dumps(payload)).fixtures[0]


def call(
    phase: str = "route", *, input_tokens: int | None = 800, evaluated: int = 32,
    cached: int | None = 768, output: int | None = 5, prefill: float | None = 30.0,
) -> CallMetric:
    return CallMetric(phase, input_tokens, evaluated, cached, output, prefill, 8.0, 1.5, "stop")


def observation(**changes: object) -> FixtureObservation:
    values: dict[str, object] = {
        "route": "FILESYSTEM", "tool_calls": (ToolCallRecord("exec_shell_full_command", {"command": "pwd"}),),
        "executed_tools": (),
        "final_output": "/tmp/workdir", "finish_reason": "stop", "model_call_count": 2,
        "retry_count": 0, "calls": (call(), call("final_from_tool")), "artifact": None,
        "lifecycle": LifecycleOutcome(True, "clean"), "peak_rss_bytes": 2048,
    }
    values.update(changes)
    return FixtureObservation(**values)  # type: ignore[arg-type]


def provenance(fixture_hash: str) -> RunProvenance:
    return RunProvenance(1, fixture_hash, "deadbeef", "profile-a", "model-a", "embedded", "abc",
                         "orbit-native", "rev-a", {"ctx": 8192}, {"cpu": "fake"},
                         {"aggregate_wall_seconds": "fixtures", "peak_rss_bytes": "fake"})


class FakeExecutor:
    def __init__(self, error: bool = False) -> None:
        self.error = error
        self.names: list[str] = []

    def execute(self, item, workdir: Path) -> FixtureObservation:
        self.names.append(item.name)
        if self.error:
            raise RuntimeError("sensitive backend detail")
        return FixtureObservation("CHAT", (), (), "OK", "stop", 1, 0, (call("chat"),), None,
                                  LifecycleOutcome(True, "clean"), 2048)


class FakeChatBackend:
    def __init__(self, content: str = "OK") -> None:
        self.content = content

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        return ChatResult(self.content, "fake", "stop", [], 10, 1, 0, 20.0, 5.0)


class QualificationValidationTests(unittest.TestCase):
    def test_exact_route_and_arguments_pass(self) -> None:
        result = validate_observation(fixture(), observation())
        self.assertEqual((result.status, result.reason.code), (Status.PASS, "validated"))

    def test_route_argument_finish_call_and_lifecycle_failures(self) -> None:
        cases = (
            (observation(route="CHAT"), "route_mismatch"),
            (observation(tool_calls=(ToolCallRecord("exec_shell_full_command", {"command": "pwd", "timeout": 10}),)), "tool_arguments_mismatch"),
            (observation(finish_reason="length"), "finish_reason_mismatch"),
            (observation(model_call_count=4), "model_call_limit"),
            (observation(lifecycle=LifecycleOutcome(False, "residue")), "lifecycle_not_clean"),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(validate_observation(fixture(), value).reason.code, reason)

    def test_artifact_publication_verification_and_json_are_structural(self) -> None:
        raw = b'{"version":1,"orbit":"qualified"}\n'
        canonical = hashlib.sha256(b'{"orbit":"qualified","version":1}').hexdigest()
        evidence = ArtifactEvidence("qualification.json", True, True, True, len(raw),
                                    hashlib.sha256(raw).hexdigest(), "created", "text_integrity", canonical)
        value = observation(route=None, tool_calls=(ToolCallRecord("write_artifact", {}), ToolCallRecord("verify_artifact", {})),
                            executed_tools=("write_artifact", "verify_artifact"), artifact=evidence)
        self.assertEqual(validate_observation(fixture(artifact=True), value).status, Status.PASS)
        missing_execution = observation(route=None, tool_calls=value.tool_calls, executed_tools=(), artifact=evidence)
        self.assertEqual(validate_observation(fixture(artifact=True), missing_execution).reason.code,
                         "tool_execution_mismatch")
        bad = observation(route=None, tool_calls=value.tool_calls, executed_tools=value.executed_tools, artifact=ArtifactEvidence(
            "qualification.json", True, False, True, 2, hashlib.sha256(b"{}").hexdigest()))
        self.assertEqual(validate_observation(fixture(artifact=True), bad).reason.code, "artifact_verification_missing")

    def test_artifact_evidence_is_bound_to_actual_path_size_and_hash(self) -> None:
        item = fixture(artifact=True)
        raw = b'{"orbit":"qualified","version":1}'
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qualification.json").write_bytes(raw)
            valid = [("write_artifact", f"artifact_publication: complete\npath: qualification.json\nbytes: {len(raw)}\nsha256: {digest}\npublication_action: created"),
                     ("verify_artifact", f"artifact_verification: complete\npath: qualification.json\nbytes: {len(raw)}\nsha256: {digest}\npublication_action: created\nstatus: pass")]
            self.assertTrue(_artifact_evidence(item, root, valid).verified)  # type: ignore[union-attr]
            wrong = [(name, content.replace("path: qualification.json", "path: other.json")) for name, content in valid]
            self.assertFalse(_artifact_evidence(item, root, wrong).published)  # type: ignore[union-attr]

    def test_optimization_parity_treats_token_counts_as_metrics(self) -> None:
        item = fixture()
        baseline = validate_observation(item, observation())
        same = compare_fixture_results(item, baseline, validate_observation(item, observation()))
        self.assertTrue(same.performance_comparison_valid)
        different = validate_observation(item, observation(calls=(call(input_tokens=801, evaluated=33), call("final_from_tool"))))
        parity = compare_fixture_results(item, baseline, different)
        self.assertTrue(parity.equivalent)
        self.assertTrue(parity.performance_comparison_valid)

    def test_optimization_parity_requires_equivalent_call_behavior(self) -> None:
        item = fixture()
        baseline = validate_observation(item, observation())
        different = validate_observation(item, observation(calls=(call(), call("retry"))))
        parity = compare_fixture_results(item, baseline, different)
        self.assertFalse(parity.equivalent)
        self.assertFalse(parity.performance_comparison_valid)
        self.assertIn("call_behavior", parity.mismatches)

    def test_optimization_parity_allows_evaluated_cached_redistribution(self) -> None:
        item = fixture()
        baseline = validate_observation(item, observation())
        redistributed = observation(calls=(call(evaluated=100, cached=700), call("final_from_tool")))
        parity = compare_fixture_results(item, baseline, validate_observation(item, redistributed))
        self.assertTrue(parity.equivalent)
        self.assertTrue(parity.performance_comparison_valid)

    def test_aggregate_rates_are_weighted_and_missing_metrics_propagate(self) -> None:
        calls = (call(input_tokens=100, evaluated=100, cached=0, prefill=10.0),
                 call(input_tokens=300, evaluated=300, cached=0, prefill=30.0))
        aggregate = AggregateMetrics.from_calls(calls, 2.0, None)
        self.assertEqual(aggregate.input_tokens, 400)
        self.assertAlmostEqual(aggregate.prefill_tokens_per_second or 0, 20.0)
        incomplete = AggregateMetrics.from_calls(
            (calls[0], call(input_tokens=300, evaluated=300, cached=None, prefill=None)), 2.0, None,
        )
        self.assertIsNone(incomplete.cached_tokens)
        self.assertIsNone(incomplete.prefill_tokens_per_second)

    def test_missing_token_metrics_remain_descriptive(self) -> None:
        item = fixture()
        baseline = validate_observation(item, observation())
        missing = observation(calls=(call(input_tokens=None), call("final_from_tool")))
        parity = compare_fixture_results(item, baseline, validate_observation(item, missing))
        self.assertTrue(parity.equivalent)
        self.assertTrue(parity.performance_comparison_valid)

    def test_argument_mismatch_invalidates_optimization_comparison(self) -> None:
        item = fixture()
        left = validate_observation(item, observation())
        right = validate_observation(item, observation(tool_calls=(
            ToolCallRecord("exec_shell_full_command", {"command": "pwd", "timeout": 10}),)))
        parity = compare_fixture_results(item, left, right)
        self.assertFalse(parity.equivalent)
        self.assertFalse(parity.performance_comparison_valid)

    def test_cross_model_parity_ignores_token_and_call_count_diversity(self) -> None:
        item = fixture()
        left = validate_observation(item, observation())
        right = validate_observation(item, observation(
            final_output="model-specific wording", model_call_count=3,
            calls=(call(input_tokens=50, evaluated=50, cached=0, output=9),) * 3,
        ))
        parity = compare_fixture_results(item, left, right, comparison_mode=ComparisonMode.CROSS_MODEL)
        self.assertTrue(parity.equivalent)
        self.assertFalse(parity.performance_comparison_valid)
        self.assertIsNone(parity.performance)

    def test_visible_control_markup_fails_protocol_gate(self) -> None:
        result = validate_observation(fixture(), observation(protocol_issue="visible_control_markup"))
        self.assertEqual(result.reason.code, "protocol_leak")


class QualificationRunnerReportingTests(unittest.TestCase):
    def fixture_set(self):
        payload = {"schema_version": 1, "fixtures": [
            {"name": "simple_chat", "capability": "chat", "profiles": ["profile-a"],
             "request": {"prompt": "Reply OK", "tools": False},
             "expect": {"finish_reason": "stop", "max_model_calls": 1, "exact_output": "OK"},
             "parity": {"mode": "exact"}},
            {"name": "other", "capability": "tools", "profiles": ["profile-b"],
             "request": {"prompt": "pwd", "tools": True},
             "expect": {"finish_reason": "stop", "max_model_calls": 1},
             "parity": {"mode": "structural"}},
            {"name": "unsupported", "capability": "artifacts", "profiles": ["profile-a"],
             "request": {"prompt": "write", "tools": True},
             "expect": {"finish_reason": "stop", "max_model_calls": 3,
                        "tool_calls": [{"name": "write_artifact"}, {"name": "verify_artifact"}],
                        "artifact": {"path": "x.json", "json_equals": 1,
                                     "publication": True, "verification": True}},
             "parity": {"mode": "structural"}},
        ]}
        return load_fixture_text(json.dumps(payload))

    def run_harness(self, profile: dict[str, object], executor: FakeExecutor, names=None):
        fixtures = self.fixture_set()
        with tempfile.TemporaryDirectory() as directory:
            return QualificationRunner(fixtures, profile, provenance(fixtures.content_hash), executor,
                                       Path(directory)).run(names)

    def test_pass_and_not_applicable_are_distinct(self) -> None:
        executor = FakeExecutor()
        run = self.run_harness({"compatibility_profile": "profile-a", "verified": True,
                        "capabilities": {"chat": True, "tools": True, "write_artifact": False}}, executor)
        self.assertEqual([item.status for item in run.fixtures],
                         [Status.PASS, Status.NOT_APPLICABLE, Status.NOT_APPLICABLE])
        self.assertEqual(executor.names, ["simple_chat"])

    def test_execution_error_is_bounded_technical_stop(self) -> None:
        run = self.run_harness({"compatibility_profile": "profile-a", "verified": True,
                        "capabilities": {"chat": True}}, FakeExecutor(True), ("simple_chat",))
        self.assertEqual(run.fixtures[0].status, Status.TECHNICAL_STOP)
        self.assertNotIn("sensitive", run.fixtures[0].reason.detail)

    def test_unverified_profile_never_executes(self) -> None:
        executor = FakeExecutor()
        run = self.run_harness({"compatibility_profile": "profile-a", "verified": False,
                        "capabilities": {"chat": True}}, executor)
        self.assertEqual(run.common[0].status, Status.FAIL)
        self.assertEqual(executor.names, [])

    def test_all_not_applicable_is_not_a_qualified_pass(self) -> None:
        run = self.run_harness({"compatibility_profile": "profile-a", "verified": True,
                        "capabilities": {"chat": False}}, FakeExecutor(), ("simple_chat",))
        self.assertEqual(run.fixtures[0].status, Status.NOT_APPLICABLE)
        self.assertEqual(run.overall_status, Status.NOT_APPLICABLE)

    def test_profile_identity_mismatch_fails_without_execution(self) -> None:
        executor = FakeExecutor()
        run = self.run_harness({"compatibility_profile": "unknown", "verified": True,
                        "capabilities": {"chat": True}}, executor, ("simple_chat",))
        self.assertEqual(run.common[0].status, Status.FAIL)
        self.assertEqual(executor.names, [])

    def test_json_and_terminal_reporting_are_stable(self) -> None:
        run = self.run_harness({"compatibility_profile": "profile-a", "verified": True,
                        "capabilities": {"chat": True}}, FakeExecutor(), ("simple_chat",))
        payload = json.loads(result_json(run))
        self.assertEqual(result_json(run), result_json(run))
        self.assertEqual(payload["fixtures"][0]["status"], "PASS")
        self.assertIsNone(payload["aggregate_metrics"]["ttft_seconds"])
        self.assertEqual(payload["provenance"]["measurement_scope"]["peak_rss_bytes"], "fake")
        summary = format_summary(run)
        self.assertIn("QUALIFIED FOR TESTED CAPABILITIES", summary)
        self.assertNotIn("score", summary.lower())

    def test_run_level_cross_model_comparison_is_descriptive(self) -> None:
        profile = {"compatibility_profile": "profile-a", "verified": True,
                   "capabilities": {"chat": True}}
        left = self.run_harness(profile, FakeExecutor(), ("simple_chat",))
        right = self.run_harness(profile, FakeExecutor(), ("simple_chat",))
        comparison = compare_runs(self.fixture_set(), left, right, comparison_mode=ComparisonMode.CROSS_MODEL)
        self.assertTrue(comparison[0].equivalent)
        self.assertFalse(comparison[0].performance_comparison_valid)

    def test_exact_chat_output_mismatch_fails(self) -> None:
        item = self.fixture_set().fixtures[0]
        value = FixtureObservation("CHAT", (), (), "Almost OK", "stop", 1, 0, (call("chat"),), None,
                                   LifecycleOutcome(True, "clean"), 1024)
        self.assertEqual(validate_observation(item, value).reason.code, "exact_output_mismatch")

    def test_runtime_adapter_observes_production_chat_runtime(self) -> None:
        item = self.fixture_set().fixtures[0]
        with tempfile.TemporaryDirectory() as directory:
            value = RuntimeFixtureExecutor(FakeChatBackend()).execute(item, Path(directory))
        self.assertEqual(value.final_output, "OK")
        self.assertEqual(value.calls[0].input_tokens, 10)
        self.assertEqual(value.calls[0].evaluated_tokens, 10)
        self.assertEqual(value.model_call_count, 1)

    def test_runtime_adapter_observes_full_tools_on_chat_request(self) -> None:
        payload = {"schema_version": 1, "fixtures": [{
            "name": "full_chat", "capability": "tools", "profiles": ["profile-a"],
            "request": {"prompt": "hello", "tools": True, "full_request": True},
            "expect": {"route": "CHAT", "finish_reason": "stop", "max_model_calls": 2},
            "parity": {"mode": "structural"},
        }]}

        class Backend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                content = '{"route":"CHAT"}' if self.calls == 1 else "Hello"
                return ChatResult(content, "fake", "stop", [], 10, 1, 0, 20.0, 5.0)

        item = load_fixture_text(json.dumps(payload)).fixtures[0]
        with tempfile.TemporaryDirectory() as directory:
            value = RuntimeFixtureExecutor(Backend()).execute(item, Path(directory))
        self.assertEqual((value.route, value.model_call_count), ("CHAT", 2))
        self.assertEqual([call.phase for call in value.calls], ["route", "chat_final"])

    def test_runtime_adapter_detects_visible_reasoning_markup(self) -> None:
        item = self.fixture_set().fixtures[0]
        with tempfile.TemporaryDirectory() as directory:
            value = RuntimeFixtureExecutor(FakeChatBackend("<think>private</think>OK")).execute(item, Path(directory))
        self.assertEqual(value.protocol_issue, "visible_control_markup")

    def test_runtime_adapter_allows_authoritative_inert_control_text(self) -> None:
        payload = {"schema_version": 1, "fixtures": [{
            "name": "quoted_marker", "capability": "chat", "profiles": ["profile-a"],
            "request": {"prompt": "Quote the literal marker.", "tools": False},
            "expect": {
                "finish_reason": "stop", "max_model_calls": 1,
                "exact_output": 'The literal string "<tool_call>" is inert.',
            },
            "parity": {"mode": "exact"},
        }]}
        item = load_fixture_text(json.dumps(payload)).fixtures[0]
        with tempfile.TemporaryDirectory() as directory:
            value = RuntimeFixtureExecutor(
                FakeChatBackend('The literal string "<tool_call>" is inert.')
            ).execute(item, Path(directory))
        self.assertIsNone(value.protocol_issue)

    def test_runtime_adapter_rejects_inconsistent_cache_accounting(self) -> None:
        item = self.fixture_set().fixtures[0]
        backend = FakeChatBackend()
        original_chat = backend.chat

        def inconsistent_chat(messages, *, temperature, max_tokens, tools=None):
            result = original_chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
            return ChatResult(
                result.content, result.model, result.finish_reason, result.tool_calls,
                10, result.completion_tokens, 20, result.prompt_tokens_per_second,
                result.generation_tokens_per_second,
            )

        backend.chat = inconsistent_chat  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as directory:
            value = RuntimeFixtureExecutor(backend).execute(item, Path(directory))
        self.assertIsNone(value.calls[0].evaluated_tokens)
        self.assertIsNone(value.calls[0].cached_tokens)


if __name__ == "__main__":
    unittest.main()
