from __future__ import annotations

import hashlib
from pathlib import Path

from .fixtures import FileSpec, FixtureSpec
from .schema import (
    AggregateMetrics, ArtifactEvidence, ComparisonMode, FixtureObservation, FixtureResult,
    LifecycleOutcome, ParityMode, ParityResult, Reason, Status, WorkflowEvidence,
)


def validate_observation(
    fixture: FixtureSpec,
    observation: FixtureObservation,
    *,
    workdir: Path | None = None,
) -> FixtureResult:
    failure = _first_failure(fixture, observation, workdir)
    status = Status.FAIL if failure else Status.PASS
    reason = failure or Reason("validated", "all deterministic expectations passed")
    return _result(fixture, observation, status, reason, applicable=True)


def _result(
    fixture: FixtureSpec,
    observation: FixtureObservation,
    status: Status,
    reason: Reason,
    *,
    applicable: bool,
) -> FixtureResult:
    wall = observation.wall_seconds
    if wall is None:
        wall = sum(item.wall_seconds or 0.0 for item in observation.calls)
    return FixtureResult(
        name=fixture.name,
        capability=fixture.capability,
        fixture_hash=fixture.fixture_hash,
        status=status,
        reason=reason,
        applicable=applicable,
        route=observation.route,
        tool_calls=observation.tool_calls,
        executed_tools=observation.executed_tools,
        final_output_sha256=(
            hashlib.sha256(observation.final_output.encode("utf-8")).hexdigest()
            if observation.model_call_count or observation.final_output
            else None
        ),
        finish_reason=observation.finish_reason,
        model_call_count=observation.model_call_count,
        retry_count=observation.retry_count,
        calls=observation.calls,
        aggregate_metrics=AggregateMetrics.from_calls(
            observation.calls,
            wall,
            observation.peak_rss_bytes,
        ),
        artifact=observation.artifact,
        lifecycle=observation.lifecycle,
        tool_outcomes=observation.tool_outcomes,
        workflow=observation.workflow,
    )


def unavailable_result(
    fixture: FixtureSpec,
    *,
    code: str,
    detail: str,
    status: Status = Status.NOT_APPLICABLE,
) -> FixtureResult:
    observation = FixtureObservation(
        route=None,
        tool_calls=(),
        executed_tools=(),
        final_output="",
        finish_reason=None,
        model_call_count=0,
        retry_count=0,
        calls=(),
        artifact=None,
        lifecycle=LifecycleOutcome(clean=True, detail="not executed"),
        peak_rss_bytes=None,
        wall_seconds=0.0,
    )
    return _result(
        fixture,
        observation,
        status,
        Reason(code, detail),
        applicable=status is not Status.NOT_APPLICABLE,
    )


def compare_fixture_results(
    fixture: FixtureSpec,
    baseline: FixtureResult,
    candidate: FixtureResult,
    *,
    comparison_mode: ComparisonMode = ComparisonMode.OPTIMIZATION,
) -> ParityResult:
    mismatches: list[str] = []
    if baseline.status is not Status.PASS or candidate.status is not Status.PASS:
        mismatches.append("status")
    _different(mismatches, "route", baseline.route, candidate.route)
    if comparison_mode is ComparisonMode.OPTIMIZATION or fixture.expect.workflow is None:
        baseline_tools = _parity_tools(fixture, baseline, comparison_mode)
        candidate_tools = _parity_tools(fixture, candidate, comparison_mode)
        _different(mismatches, "tool_calls", baseline_tools, candidate_tools)
        _different(mismatches, "executed_tools", baseline.executed_tools, candidate.executed_tools)
        _different(mismatches, "tool_outcomes", _tool_outcome_state(baseline), _tool_outcome_state(candidate))
    _different(mismatches, "finish_reason", baseline.finish_reason, candidate.finish_reason)
    _different(mismatches, "artifact_state", _artifact_state(baseline.artifact), _artifact_state(candidate.artifact))
    _different(mismatches, "workflow_state", _workflow_state(baseline.workflow), _workflow_state(candidate.workflow))
    if comparison_mode is ComparisonMode.OPTIMIZATION and fixture.parity_mode is ParityMode.EXACT:
        _different(
            mismatches,
            "output_hash",
            baseline.final_output_sha256,
            candidate.final_output_sha256,
        )
        _different(mismatches, "model_call_count", baseline.model_call_count, candidate.model_call_count)
        _different(mismatches, "retry_count", baseline.retry_count, candidate.retry_count)
    unique = tuple(dict.fromkeys(mismatches))
    equivalent = not unique
    performance = None
    performance_valid = False
    performance_mismatches: list[str] = []
    if comparison_mode is ComparisonMode.OPTIMIZATION and equivalent:
        baseline_work = _model_work(baseline)
        candidate_work = _model_work(candidate)
        if baseline_work is None or candidate_work is None:
            performance_mismatches.append("model_work_unavailable")
        else:
            _different(performance_mismatches, "model_work", baseline_work, candidate_work)
        performance_valid = not performance_mismatches
    all_mismatches = tuple(dict.fromkeys([*unique, *performance_mismatches]))
    if performance_valid:
        performance = {
            "baseline": baseline.aggregate_metrics,
            "candidate": candidate.aggregate_metrics,
        }
    return ParityResult(
        comparison_mode=comparison_mode,
        mode=fixture.parity_mode,
        equivalent=equivalent,
        performance_comparison_valid=performance_valid,
        mismatches=all_mismatches,
        performance=performance,
    )


def _first_failure(
    fixture: FixtureSpec,
    observation: FixtureObservation,
    workdir: Path | None,
) -> Reason | None:
    expect = fixture.expect
    if not observation.lifecycle.clean:
        return Reason("lifecycle_not_clean", observation.lifecycle.detail)
    if observation.protocol_issue is not None:
        return Reason("protocol_leak", observation.protocol_issue)
    if observation.finish_reason != expect.finish_reason:
        return Reason(
            "finish_reason_mismatch",
            f"expected {expect.finish_reason!r}, got {observation.finish_reason!r}",
        )
    if observation.model_call_count > expect.max_model_calls:
        return Reason(
            "model_call_limit",
            f"expected at most {expect.max_model_calls}, got {observation.model_call_count}",
        )
    if expect.workflow is not None:
        return _workflow_failure(fixture, observation)
    if expect.route is not None and observation.route != expect.route:
        return Reason("route_mismatch", f"expected {expect.route!r}, got {observation.route!r}")
    if len(observation.tool_calls) != len(expect.tool_calls):
        return Reason(
            "tool_call_count_mismatch",
            f"expected {len(expect.tool_calls)}, got {len(observation.tool_calls)}",
        )
    for index, (expected, actual) in enumerate(zip(expect.tool_calls, observation.tool_calls)):
        if expected.name != actual.name:
            return Reason(
                "tool_name_mismatch",
                f"call {index}: expected {expected.name!r}, got {actual.name!r}",
            )
        if expected.arguments is not None and expected.arguments != actual.arguments:
            return Reason(
                "tool_arguments_mismatch",
                f"call {index}: typed arguments differ",
            )
    if expect.route is None and observation.executed_tools != tuple(item.name for item in expect.tool_calls):
        return Reason("tool_execution_mismatch", "ordered executed tools differ from selected tools")
    if expect.exact_output is not None and observation.final_output != expect.exact_output:
        return Reason("exact_output_mismatch", "visible output differs from authoritative fixture text")
    if expect.final_reports_workdir and (
        workdir is None or str(workdir.resolve()) not in observation.final_output
    ):
        return Reason("final_workdir_mismatch", "final output does not report the fixture workdir")
    if expect.artifact is not None:
        return _artifact_failure(expect.artifact, observation.artifact)
    return None


def _workflow_failure(fixture: FixtureSpec, observation: FixtureObservation) -> Reason | None:
    expected = fixture.expect.workflow
    actual = observation.workflow
    assert expected is not None
    if len(observation.tool_calls) > expected.max_tool_calls:
        return Reason("tool_call_limit", f"expected at most {expected.max_tool_calls}, got {len(observation.tool_calls)}")
    if len(observation.tool_calls) != len(observation.tool_outcomes):
        return Reason("tool_execution_incomplete", "selected and completed tool-call counts differ")
    if any(item.name != "exec_shell_full_command" for item in observation.tool_calls):
        return Reason("workflow_tool_mismatch", "workflow selected a tool outside its shell capability")
    if observation.wall_seconds is not None and observation.wall_seconds > expected.timeout_seconds:
        return Reason("workflow_timeout", "fixture execution exceeded its declared timeout")
    states = {item.path: item for item in actual.files} if actual is not None else {}
    for item in expected.files:
        mismatch = _file_mismatch(item, states.get(item.path))
        if mismatch:
            return Reason("filesystem_state_mismatch", f"{item.path}: {mismatch}")
    workspace = {item.path: item for item in fixture.workspace.files} if fixture.workspace else {}
    for path in expected.unchanged_files:
        mismatch = _file_mismatch(workspace[path], states.get(path))
        if mismatch:
            return Reason("unrelated_file_changed", f"{path}: {mismatch}")
    if actual is not None and actual.unexpected_paths:
        return Reason("unexpected_filesystem_state", actual.unexpected_paths[0])
    if expected.test_runner is not None:
        if actual is None or actual.test is None:
            return Reason("test_evidence_missing", "post-workflow test evidence is unavailable")
        if actual.test.status != "pass":
            return Reason("test_failure", f"{expected.test_runner} did not pass")
        if not actual.test.model_invocation_observed:
            return Reason("model_test_run_missing", "model did not complete the declared test command")
    if expected.require_recovery:
        if actual is None or actual.failed_tool_calls < 1:
            return Reason("failed_command_missing", "workflow did not observe the required command failure")
        if actual.repeated_failed_command:
            return Reason("repeated_failed_command", "the same failing tool call was repeated")
        if not actual.recovery_observed:
            return Reason("recovery_missing", "no successful tool call followed the failure")
    return None


def _file_mismatch(expected: FileSpec, actual) -> str | None:
    if actual is None or not actual.exists:
        return "file is missing"
    if not actual.regular_file:
        return "path is not a regular file"
    raw = expected.content.encode("utf-8")
    if actual.byte_count != len(raw) or actual.sha256 != expected.sha256:
        return "byte count or SHA-256 differs"
    return None


def _artifact_failure(expected, actual: ArtifactEvidence | None) -> Reason | None:
    if actual is None:
        return Reason("artifact_evidence_missing", "no artifact evidence was captured")
    if actual.path != expected.path:
        return Reason("artifact_path_mismatch", f"expected {expected.path!r}, got {actual.path!r}")
    if expected.publication and not (actual.published and actual.exists):
        return Reason("artifact_publication_missing", "artifact was not published and present")
    if expected.verification and not actual.verified:
        return Reason("artifact_verification_missing", "model-selected verification did not complete")
    if actual.byte_count is None or actual.sha256 is None:
        return Reason("artifact_hash_missing", "byte count or SHA-256 is unavailable")
    if actual.canonical_json_sha256 != expected.canonical_json_sha256:
        return Reason("artifact_json_mismatch", "parsed JSON value differs from the fixture")
    return None


def _artifact_state(value: ArtifactEvidence | None):
    if value is None:
        return None
    return (
        value.path,
        value.published,
        value.verified,
        value.exists,
        value.canonical_json_sha256,
    )


def _tool_outcome_state(result: FixtureResult):
    return tuple((item.name, item.status, item.exit_code) for item in result.tool_outcomes)


def _workflow_state(value: WorkflowEvidence | None):
    if value is None:
        return None
    return (
        tuple((item.path, item.exists, item.regular_file, item.byte_count, item.sha256) for item in value.files),
        value.unexpected_paths,
        value.test.status if value.test is not None else None,
        value.recovery_observed,
        value.repeated_failed_command,
    )


def _model_work(result: FixtureResult):
    if any(item.input_tokens is None or item.output_tokens is None for item in result.calls):
        return None
    return tuple(
        (
            item.phase,
            item.input_tokens,
            item.output_tokens,
            item.retry_reason,
        )
        for item in result.calls
    )


def _parity_tools(fixture: FixtureSpec, result: FixtureResult, mode: ComparisonMode):
    if mode is ComparisonMode.OPTIMIZATION:
        return result.tool_calls
    return tuple(
        (actual.name, actual.arguments if expected.arguments is not None else None)
        for expected, actual in zip(fixture.expect.tool_calls, result.tool_calls)
    )


def _different(items: list[str], name: str, baseline, candidate) -> None:
    if baseline != candidate:
        items.append(name)
