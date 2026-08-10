from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orbit.backend.base import ChatResult
from orbit.qualification.fixtures import FixtureError, load_fixture_set, load_fixture_text
from orbit.qualification.runner import (
    QualificationRunner,
    RuntimeFixtureExecutor,
    _file_state,
    _unexpected_paths,
)
from orbit.qualification.schema import (
    CallMetric,
    ComparisonMode,
    FileStateEvidence,
    FixtureObservation,
    LifecycleOutcome,
    RunProvenance,
    Status,
    TestEvidence,
    ToolCallRecord,
    ToolOutcomeRecord,
    WorkflowEvidence,
)
from orbit.qualification.validators import compare_fixture_results, validate_observation
from orbit.runtime.shell_guardrails import classify_explicit_no_mutation_constraint


ROOT = Path(__file__).parents[1]
WORKFLOW_FIXTURES = ROOT / "qualification/fixtures/workflows-v1.json"


def workflow_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fixtures": [{
            "name": "modify", "capability": "tools", "profiles": ["profile-a", "profile-b"],
            "workspace": {"files": [
                {"path": "value.txt", "content": "old\n"},
                {"path": "keep.txt", "content": "keep\n"},
            ]},
            "request": {"prompt": "Change value.txt.", "tools": True},
            "expect": {
                "finish_reason": "stop", "max_model_calls": 5,
                "workflow": {
                    "files": [{"path": "value.txt", "content": "new\n"}],
                    "unchanged_files": ["keep.txt"], "max_tool_calls": 4, "timeout_seconds": 30,
                },
            },
            "parity": {"mode": "structural"},
        }],
    }


def metric() -> CallMetric:
    return CallMetric("tool_call", 20, 20, 0, 5, 10.0, 5.0, 1.0, "stop")


def evidence(
    *,
    content: bytes = b"new\n",
    outcomes: tuple[ToolOutcomeRecord, ...] = (ToolOutcomeRecord("exec_shell_full_command", "success", None),),
    test: TestEvidence | None = None,
    recovery: bool = False,
    repeated: bool = False,
    unexpected: tuple[str, ...] = (),
) -> WorkflowEvidence:
    return WorkflowEvidence(
        files=(
            FileStateEvidence.from_bytes("value.txt", content),
            FileStateEvidence.from_bytes("keep.txt", b"keep\n"),
        ),
        unexpected_paths=unexpected,
        failed_tool_calls=sum(item.status == "failure" for item in outcomes),
        recovery_observed=recovery,
        repeated_failed_command=repeated,
        test=test,
    )


def observation(
    *,
    calls: tuple[ToolCallRecord, ...] = (
        ToolCallRecord("exec_shell_full_command", {"command": "printf 'new\\n' > value.txt"}),
    ),
    workflow: WorkflowEvidence | None = None,
    outcomes: tuple[ToolOutcomeRecord, ...] | None = None,
) -> FixtureObservation:
    recorded = outcomes or tuple(ToolOutcomeRecord(item.name, "success", None) for item in calls)
    return FixtureObservation(
        route=None, tool_calls=calls, executed_tools=tuple(item.name for item in calls),
        final_output="updated", finish_reason="stop", model_call_count=2, retry_count=0,
        calls=(metric(), metric()), artifact=None, lifecycle=LifecycleOutcome(True, "clean"),
        peak_rss_bytes=None, tool_outcomes=recorded,
        workflow=workflow or evidence(outcomes=recorded),
    )


def provenance(content_hash: str) -> RunProvenance:
    return RunProvenance(
        1, content_hash, "deadbeef", "profile-a", "model", "embedded", "hash",
        "native", "revision", {}, {}, {"peak_rss_bytes": "unavailable"},
    )


class WorkflowFixtureSchemaTests(unittest.TestCase):
    def test_workflow_fixture_file_has_exact_three_contracts(self) -> None:
        fixtures = load_fixture_set(WORKFLOW_FIXTURES)
        self.assertEqual(
            [item.name for item in fixtures.fixtures],
            ["bug_fix_and_test", "existing_file_modification", "failed_command_recovery"],
        )
        self.assertTrue(all(item.workspace and item.expect.workflow for item in fixtures.fixtures))
        self.assertTrue(all(
            classify_explicit_no_mutation_constraint(item.request.prompt) == "none"
            for item in fixtures.fixtures
        ))

    def test_workspace_and_workflow_schema_fail_closed(self) -> None:
        cases = []
        unknown = workflow_document()
        unknown["fixtures"][0]["workspace"]["command"] = "rm -rf /"  # type: ignore[index]
        cases.append((unknown, "unknown_key"))
        boolean = workflow_document()
        boolean["fixtures"][0]["expect"]["workflow"]["max_tool_calls"] = True  # type: ignore[index]
        cases.append((boolean, "invalid_type"))
        unsafe = workflow_document()
        unsafe["fixtures"][0]["workspace"]["files"][0]["path"] = "../escape"  # type: ignore[index]
        cases.append((unsafe, "invalid_value"))
        duplicate = workflow_document()
        duplicate["fixtures"][0]["workspace"]["files"].append(  # type: ignore[index]
            {"path": "value.txt", "content": "other"}
        )
        cases.append((duplicate, "duplicate_workspace_path"))
        for payload, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(FixtureError, reason):
                load_fixture_text(json.dumps(payload))


class WorkflowValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = load_fixture_text(json.dumps(workflow_document())).fixtures[0]

    def test_success_and_cross_model_strategy_diversity(self) -> None:
        left = validate_observation(self.fixture, observation())
        right = validate_observation(self.fixture, observation(
            calls=(
                ToolCallRecord("exec_shell_full_command", {"command": "cat value.txt"}),
                ToolCallRecord("exec_shell_full_command", {"command": "sed -i s/old/new/ value.txt"}),
            ),
            outcomes=(
                ToolOutcomeRecord("exec_shell_full_command", "success", None),
                ToolOutcomeRecord("exec_shell_full_command", "success", None),
            ),
        ))
        self.assertEqual((left.status, right.status), (Status.PASS, Status.PASS))
        parity = compare_fixture_results(self.fixture, left, right, comparison_mode=ComparisonMode.CROSS_MODEL)
        self.assertTrue(parity.equivalent)
        self.assertFalse(parity.performance_comparison_valid)

    def test_tool_bound_and_filesystem_mismatches_fail(self) -> None:
        too_many = tuple(
            ToolCallRecord("exec_shell_full_command", {"command": f"echo {index}"}) for index in range(5)
        )
        cases = (
            (observation(calls=too_many, outcomes=tuple(
                ToolOutcomeRecord("exec_shell_full_command", "success", None) for _ in too_many
            )), "tool_call_limit"),
            (observation(workflow=evidence(content=b"wrong\n")), "filesystem_state_mismatch"),
            (observation(workflow=WorkflowEvidence(
                files=(
                    FileStateEvidence.from_bytes("value.txt", b"new\n"),
                    FileStateEvidence.from_bytes("keep.txt", b"changed\n"),
                ),
                unexpected_paths=(),
                failed_tool_calls=0, recovery_observed=False, repeated_failed_command=False, test=None,
            )), "unrelated_file_changed"),
            (observation(workflow=evidence(unexpected=("unexpected.txt",))), "unexpected_filesystem_state"),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(validate_observation(self.fixture, value).reason.code, reason)

    def test_model_and_tool_limits_are_independent_and_inclusive(self) -> None:
        at_model_bound = replace(observation(), model_call_count=5)
        over_model_bound = replace(observation(), model_call_count=6)
        at_tool_bound = tuple(
            ToolCallRecord("exec_shell_full_command", {"command": f"echo {index}"})
            for index in range(4)
        )
        at_tool_outcomes = tuple(
            ToolOutcomeRecord("exec_shell_full_command", "success", 0)
            for _item in at_tool_bound
        )
        self.assertEqual(validate_observation(self.fixture, at_model_bound).status, Status.PASS)
        self.assertEqual(
            validate_observation(self.fixture, over_model_bound).reason.code,
            "model_call_limit",
        )
        self.assertEqual(
            validate_observation(
                self.fixture,
                observation(calls=at_tool_bound, outcomes=at_tool_outcomes),
            ).status,
            Status.PASS,
        )

    def test_recovery_accepts_different_successful_strategy(self) -> None:
        contract = replace(self.fixture.expect.workflow, require_recovery=True)
        fixture = replace(self.fixture, expect=replace(self.fixture.expect, workflow=contract))
        calls = (
            ToolCallRecord("exec_shell_full_command", {"command": "cat absent.txt"}),
            ToolCallRecord("exec_shell_full_command", {"command": "printf 'new\\n' > value.txt"}),
        )
        outcomes = (
            ToolOutcomeRecord("exec_shell_full_command", "failure", 1),
            ToolOutcomeRecord("exec_shell_full_command", "success", 0),
        )
        value = observation(
            calls=calls,
            outcomes=outcomes,
            workflow=evidence(outcomes=outcomes, recovery=True),
        )
        self.assertEqual(validate_observation(fixture, value).status, Status.PASS)

    def test_test_failure_and_recovery_loop_fail(self) -> None:
        payload = workflow_document()
        contract = payload["fixtures"][0]["expect"]["workflow"]  # type: ignore[index]
        contract["test_runner"] = "python_unittest"
        contract["require_recovery"] = True
        item = load_fixture_text(json.dumps(payload)).fixtures[0]
        failed_test = TestEvidence("python_unittest", "failure", 1, False)
        failed = ToolOutcomeRecord("exec_shell_full_command", "failure", 1)
        cases = (
            (observation(workflow=evidence(outcomes=(failed,), test=failed_test), outcomes=(failed,)), "test_failure"),
            (observation(
                calls=(
                    ToolCallRecord("exec_shell_full_command", {"command": "false"}),
                    ToolCallRecord("exec_shell_full_command", {"command": "false"}),
                ),
                workflow=evidence(
                    outcomes=(failed, failed), recovery=False, repeated=True,
                    test=TestEvidence("python_unittest", "pass", 0, True),
                ),
                outcomes=(failed, failed),
            ), "repeated_failed_command"),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(validate_observation(item, value).reason.code, reason)

    def test_declared_test_must_be_run_by_the_model(self) -> None:
        payload = workflow_document()
        payload["fixtures"][0]["expect"]["workflow"]["test_runner"] = "python_unittest"  # type: ignore[index]
        item = load_fixture_text(json.dumps(payload)).fixtures[0]
        value = observation(workflow=evidence(test=TestEvidence("python_unittest", "pass", 0, False)))
        self.assertEqual(validate_observation(item, value).reason.code, "model_test_run_missing")


class WorkflowRunnerTests(unittest.TestCase):
    def test_filesystem_inventory_rejects_extra_paths_but_allows_declared_python_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calculator.py").write_text("pass\n", encoding="utf-8")
            (root / "test_calculator.py").write_text("pass\n", encoding="utf-8")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "calculator.cpython-313.pyc").write_bytes(b"cache")
            (cache / "test_calculator.cpython-313.pyc").write_bytes(b"cache")
            allowed = {"calculator.py", "test_calculator.py"}
            self.assertEqual(_unexpected_paths(root, allowed), ())
            (cache / "undeclared.cpython-313.pyc").write_bytes(b"cache")
            self.assertEqual(
                _unexpected_paths(root, allowed),
                ("__pycache__/undeclared.cpython-313.pyc",),
            )
            (cache / "undeclared.cpython-313.pyc").unlink()
            (root / "unexpected.txt").write_text("mutation\n", encoding="utf-8")
            self.assertEqual(_unexpected_paths(root, allowed), ("unexpected.txt",))

    def test_file_state_rejects_symlink_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            external = Path(outside)
            (external / "value.txt").write_text("new\n", encoding="utf-8")
            (root / "nested").symlink_to(external, target_is_directory=True)
            state = _file_state(root, "nested/value.txt")
        self.assertTrue(state.exists)
        self.assertFalse(state.regular_file)
        self.assertIsNone(state.sha256)

    def test_runtime_model_loop_bound_is_independent_from_tool_call_bound(self) -> None:
        fixture = next(
            item for item in load_fixture_set(WORKFLOW_FIXTURES).fixtures
            if item.name == "existing_file_modification"
        )
        captured: dict[str, object] = {}

        class RuntimeSpy:
            def __init__(self, backend) -> None:
                pass

            def ask_with_tools(self, prompt, **kwargs):
                captured.update(kwargs)
                return ChatResult("complete", "fake", "stop", [], 0, 0, 0, None, None)

        with tempfile.TemporaryDirectory() as directory, patch(
            "orbit.qualification.runner.ChatRuntime", RuntimeSpy
        ):
            RuntimeFixtureExecutor(object()).execute(fixture, Path(directory))  # type: ignore[arg-type]
        self.assertEqual(captured["max_loops"], fixture.expect.max_model_calls)
        self.assertNotEqual(
            fixture.expect.max_model_calls,
            fixture.expect.workflow.max_tool_calls,  # type: ignore[union-attr]
        )

    def test_runner_prepares_workspace_and_observes_final_state(self) -> None:
        fixtures = load_fixture_text(json.dumps(workflow_document()))

        class Executor:
            def execute(self, fixture, workdir: Path) -> FixtureObservation:
                self.assert_initial = (workdir / "value.txt").read_text(encoding="utf-8")
                if self.assert_initial != "old\n":
                    raise AssertionError("workspace was not prepared")
                (workdir / "value.txt").write_text("new\n", encoding="utf-8")
                return observation()

        profile = {"compatibility_profile": "profile-a", "verified": True, "capabilities": {"tools": True}}
        with tempfile.TemporaryDirectory() as directory:
            run = QualificationRunner(
                fixtures, profile, provenance(fixtures.content_hash), Executor(), Path(directory)
            ).run()
        self.assertEqual(run.fixtures[0].status, Status.PASS)
        self.assertEqual(run.fixtures[0].workflow.files[0].path, "value.txt")  # type: ignore[union-attr]

    def test_runner_executes_declared_test_and_records_pass(self) -> None:
        fixtures = load_fixture_set(WORKFLOW_FIXTURES)
        bug_fix = replace(
            next(item for item in fixtures.fixtures if item.name == "bug_fix_and_test"),
            profiles=("profile-a",),
        )

        class Executor:
            def execute(self, fixture, workdir: Path) -> FixtureObservation:
                (workdir / "calculator.py").write_text(
                    "def add(left, right):\n    return left + right\n", encoding="utf-8"
                )
                return observation(
                    calls=(ToolCallRecord(
                        "exec_shell_full_command",
                        {"command": "sed -i s/-/+/ calculator.py && python3 -m unittest -q"},
                    ),),
                    outcomes=(ToolOutcomeRecord("exec_shell_full_command", "success", 0),),
                )

        one = type(fixtures)(fixtures.schema_version, (bug_fix,), fixtures.content_hash)
        profile = {"compatibility_profile": "profile-a", "verified": True, "capabilities": {"tools": True}}
        with tempfile.TemporaryDirectory() as directory:
            run = QualificationRunner(one, profile, provenance(one.content_hash), Executor(), Path(directory)).run()
        self.assertEqual(run.fixtures[0].status, Status.PASS)
        self.assertEqual(run.fixtures[0].workflow.test.status, "pass")  # type: ignore[union-attr]

    def test_runner_rejects_test_failure_masked_before_mutation(self) -> None:
        fixtures = load_fixture_set(WORKFLOW_FIXTURES)
        bug_fix = replace(
            next(item for item in fixtures.fixtures if item.name == "bug_fix_and_test"),
            profiles=("profile-a",),
        )

        class Executor:
            def execute(self, fixture, workdir: Path) -> FixtureObservation:
                (workdir / "calculator.py").write_text(
                    "def add(left, right):\n    return left + right\n", encoding="utf-8"
                )
                command = "python3 -m unittest -q || true; sed -i s/-/+/ calculator.py"
                return observation(
                    calls=(ToolCallRecord("exec_shell_full_command", {"command": command}),),
                    outcomes=(ToolOutcomeRecord("exec_shell_full_command", "success", 0),),
                )

        one = type(fixtures)(fixtures.schema_version, (bug_fix,), fixtures.content_hash)
        profile = {"compatibility_profile": "profile-a", "verified": True, "capabilities": {"tools": True}}
        with tempfile.TemporaryDirectory() as directory:
            run = QualificationRunner(
                one, profile, provenance(one.content_hash), Executor(), Path(directory)
            ).run()
        self.assertEqual(run.fixtures[0].status, Status.FAIL)
        self.assertEqual(run.fixtures[0].reason.code, "model_test_run_missing")


if __name__ == "__main__":
    unittest.main()
