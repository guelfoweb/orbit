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
    status = (
        Status.TECHNICAL_STOP
        if failure and failure.code in {
            "lifecycle_evidence_missing", "lifecycle_evidence_incomplete", "lifecycle_evidence_invalid",
        }
        else Status.FAIL if failure else Status.PASS
    )
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
        state_reuse=observation.state_reuse,
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
    _different(mismatches, "state_reuse", _state_reuse_parity(baseline.state_reuse), _state_reuse_parity(candidate.state_reuse))
    if comparison_mode is ComparisonMode.OPTIMIZATION and fixture.expect.lifecycle is None:
        _different(mismatches, "model_call_count", baseline.model_call_count, candidate.model_call_count)
        _different(mismatches, "retry_count", baseline.retry_count, candidate.retry_count)
        _different(mismatches, "call_behavior", _call_behavior(baseline), _call_behavior(candidate))
    if comparison_mode is ComparisonMode.OPTIMIZATION and fixture.parity_mode is ParityMode.EXACT:
        _different(
            mismatches,
            "output_hash",
            baseline.final_output_sha256,
            candidate.final_output_sha256,
        )
    unique = tuple(dict.fromkeys(mismatches))
    equivalent = not unique
    performance = None
    performance_valid = comparison_mode is ComparisonMode.OPTIMIZATION and equivalent
    if performance_valid:
        performance = {
            "baseline": baseline.aggregate_metrics,
            "candidate": candidate.aggregate_metrics,
        }
    return ParityResult(
        fixture_name=fixture.name,
        comparison_mode=comparison_mode,
        mode=fixture.parity_mode,
        equivalent=equivalent,
        performance_comparison_valid=performance_valid,
        mismatches=unique,
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
    if expect.lifecycle is not None:
        return _lifecycle_failure(expect.lifecycle.operation, expect.lifecycle.min_restores, observation)
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


def _lifecycle_failure(operation: str, min_restores: int, observation: FixtureObservation) -> Reason | None:
    actual = observation.state_reuse
    if actual is None:
        return Reason("lifecycle_evidence_missing", "state-reuse evidence is unavailable")
    if actual.operation != operation:
        return Reason("lifecycle_operation_mismatch", "evidence belongs to a different lifecycle operation")
    booleans, integers = {
        "reset_invalidation": (
            (actual.initialized_before, actual.initialized_after, actual.invalidated, actual.recapture_observed),
            (actual.capture_count, actual.restore_count, actual.fallback_count,
             actual.invalidation_count, actual.cached_tokens_after),
        ),
        "cancellation": (
            (actual.initialized_before, actual.initialized_after, actual.cancellation_observed,
             actual.invalidated, actual.partial_state_accepted),
            (actual.capture_count, actual.checkpoint_size_after),
        ),
        "restore_failure_fallback": (
            (actual.initialized_before, actual.initialized_after, actual.restore_rejected,
             actual.fallback_succeeded, actual.partial_state_accepted),
            (actual.capture_count, actual.restore_count, actual.fallback_attempts,
             actual.fallback_count, actual.invalidation_count, actual.checkpoint_size_after),
        ),
        "repeated_restore_rss": (
            (actual.initialized_before, actual.initialized_after),
            (actual.restore_count, actual.cached_tokens_after, actual.checkpoint_size_after, actual.rss_start_bytes,
             actual.rss_end_bytes, actual.rss_peak_bytes, actual.rss_tolerance_bytes),
        ),
        "teardown_cleanup": ((actual.initialized_before, actual.port_released), (actual.process_pid,)),
    }[operation]
    if any(item is None for item in booleans + integers):
        return Reason("lifecycle_evidence_incomplete", "required lifecycle evidence is unavailable")
    if any(type(item) is not bool for item in booleans) or any(type(item) is not int or item < 0 for item in integers):
        return Reason("lifecycle_evidence_invalid", "lifecycle evidence has an invalid type or range")
    if type(actual.residual_state) is not tuple or any(type(item) is not str or not item for item in actual.residual_state):
        return Reason("lifecycle_evidence_invalid", "residual-state evidence is malformed")
    if operation == "teardown_cleanup" and (
        actual.process_pid <= 0
        or (actual.process_exit_code is not None and type(actual.process_exit_code) is not int)
    ):
        return Reason("lifecycle_evidence_invalid", "process teardown evidence is malformed")
    if actual.residual_state:
        return Reason("lifecycle_residue", actual.residual_state[0])
    if actual.partial_state_accepted is True:
        code = "partial_restore_accepted" if operation == "restore_failure_fallback" else "partial_state_accepted"
        return Reason(code, "an incomplete reusable state was accepted")
    if operation == "reset_invalidation":
        if not (
            actual.initialized_before and actual.initialized_after and actual.invalidated
            and actual.recapture_observed and actual.capture_count >= 3
            and actual.restore_count >= 1 and actual.fallback_count == 0
            and actual.invalidation_count >= 2 and actual.cached_tokens_after == 0
        ):
            return Reason("stale_state_after_reset", "reset did not force a safe cold recapture")
    elif operation == "cancellation":
        if actual.capture_count < 1 or not actual.initialized_before or actual.initialized_after or actual.checkpoint_size_after != 0 or not actual.cancellation_observed or not actual.invalidated:
            return Reason("cancellation_cleanup_failed", "cancellation did not invalidate reusable state")
    elif operation == "restore_failure_fallback":
        if actual.capture_count < 1 or actual.restore_count != 0 or actual.invalidation_count < 1 or not actual.initialized_before or actual.initialized_after or actual.checkpoint_size_after != 0 or not actual.restore_rejected:
            return Reason("restore_failure_not_rejected", "invalid reusable state was not rejected")
        if actual.fallback_attempts != 1 or actual.fallback_count != 1:
            return Reason("fallback_loop", "restore failure did not use exactly one cold fallback")
        if not actual.fallback_succeeded:
            return Reason("cold_fallback_failed", "cold fallback did not complete safely")
    elif operation == "repeated_restore_rss":
        samples = actual.rss_samples
        if (
            type(samples) is not tuple or len(samples) < 2
            or any(type(item) is not int or item < 0 for item in samples)
            or samples[0] != actual.rss_start_bytes or samples[-1] != actual.rss_end_bytes
            or max(samples) != actual.rss_peak_bytes
        ):
            return Reason("lifecycle_evidence_invalid", "RSS samples are missing or inconsistent")
        if not actual.initialized_before or not actual.initialized_after or actual.cached_tokens_after <= 0 or actual.checkpoint_size_after <= 0 or actual.restore_count < min_restores:
            return Reason("restore_count_shortfall", "bounded restore sequence did not complete")
        growth = actual.rss_end_bytes - actual.rss_start_bytes
        sustained_floor = max(1024**2, actual.rss_tolerance_bytes // min_restores)
        sustained = len(samples) >= 3 and all(right > left for left, right in zip(samples, samples[1:]))
        if growth > actual.rss_tolerance_bytes or (sustained and growth > sustained_floor):
            return Reason("rss_growth_unbounded", "resident growth exceeded the declared allocator tolerance")
    elif not actual.initialized_before:
        return Reason("teardown_state_missing", "reusable state was not established before teardown")
    elif actual.process_exit_code is None:
        return Reason("process_not_exited", "qualification-owned server is still running")
    elif not actual.port_released:
        return Reason("port_not_released", "qualification-owned server port remains bound")
    return None


def _state_reuse_parity(value):
    if value is None:
        return None
    return (
        value.operation, value.initialized_before, value.initialized_after, value.invalidated,
        value.recapture_observed,
        None if value.checkpoint_size_after is None else value.checkpoint_size_after == 0,
        value.partial_state_accepted, value.cancellation_observed, value.restore_rejected,
        value.fallback_succeeded, value.fallback_attempts, value.port_released, value.residual_state,
    )


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
    return tuple(
        (item.name, item.status, item.exit_code, item.content_sha256)
        for item in result.tool_outcomes
    )


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


def _call_behavior(result: FixtureResult):
    return tuple(
        (
            item.phase,
            item.finish_reason,
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
