from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from orbit.backend.base import ChatBackend
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.command_request import (
    command_tool_call_from_content,
    command_tool_call_from_tool_calls,
    parse_command_decision,
    parse_command_decision_from_tool_calls,
)
from orbit.runtime.kv_diag import model_call_context
from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT
from orbit.runtime.shell_guardrails import shell_failure_from_output
from orbit.runtime.turn_trace import ModelPhaseStart, ModelStepMetrics

from .fixtures import CAPABILITY_REQUIREMENTS, FixtureSet, FixtureSpec
from .schema import (
    AggregateMetrics,
    ArtifactEvidence,
    CallMetric,
    CommonGate,
    ComparisonMode,
    FileStateEvidence,
    FixtureObservation,
    LifecycleOutcome,
    ParityResult,
    QualificationRun,
    Reason,
    RunProvenance,
    Status,
    TestEvidence,
    ToolCallRecord,
    ToolOutcomeRecord,
    WorkflowEvidence,
)
from .validators import compare_fixture_results, unavailable_result, validate_observation


class FixtureExecutor(Protocol):
    def execute(self, fixture: FixtureSpec, workdir: Path) -> FixtureObservation: ...


@dataclass
class QualificationRunner:
    fixture_set: FixtureSet
    profile: dict[str, Any]
    provenance: RunProvenance
    executor: FixtureExecutor
    workdir: Path

    def run(self, names: tuple[str, ...] | None = None) -> QualificationRun:
        selected = self._selected(names)
        profile_id = self.profile.get("compatibility_profile")
        verified = self.profile.get("verified") is True and profile_id == self.provenance.profile_identity
        identity = CommonGate(
            "identity",
            Status.PASS if verified else Status.FAIL,
            Reason(
                "verified_profile" if verified else "unverified_profile",
                "profile identity is verified" if verified else "profile identity is not verified",
            ),
        )
        results = []
        if verified:
            for fixture in selected:
                applicability = self._applicability(fixture, str(profile_id))
                if applicability is not None:
                    results.append(applicability)
                    continue
                fixture_workdir = self.workdir / fixture.name
                fixture_workdir.mkdir(parents=True, exist_ok=False)
                try:
                    _prepare_workspace(fixture, fixture_workdir)
                    observation = self.executor.execute(fixture, fixture_workdir)
                    if fixture.expect.workflow is not None:
                        observation = replace(
                            observation,
                            workflow=_workflow_evidence(fixture, fixture_workdir, observation),
                        )
                    results.append(validate_observation(fixture, observation, workdir=fixture_workdir))
                except Exception as error:
                    results.append(
                        unavailable_result(
                            fixture,
                            code="execution_error",
                            detail=f"fixture execution failed ({type(error).__name__})",
                            status=Status.TECHNICAL_STOP,
                        )
                    )

        lifecycle_ok = all(item.lifecycle.clean for item in results)
        protocol_ok = all(item.reason.code != "protocol_leak" for item in results)
        common = (
            identity,
            CommonGate(
                "lifecycle",
                Status.PASS if lifecycle_ok else Status.FAIL,
                Reason(
                    "clean" if lifecycle_ok else "residue",
                    "fixture lifecycle is clean" if lifecycle_ok else "fixture lifecycle residue detected",
                ),
            ),
            CommonGate(
                "protocol",
                Status.PASS if protocol_ok else Status.FAIL,
                Reason(
                    "valid" if protocol_ok else "control_leak",
                    "protocol output is structurally valid" if protocol_ok else "visible control markup detected",
                ),
            ),
        )
        calls = tuple(call for result in results for call in result.calls)
        aggregate = AggregateMetrics.from_calls(
            calls,
            sum(item.aggregate_metrics.wall_seconds for item in results),
            max((item.aggregate_metrics.peak_rss_bytes or 0 for item in results), default=0) or None,
        )
        overall = _overall(common, results)
        return QualificationRun(
            provenance=self.provenance,
            common=common,
            fixtures=tuple(results),
            aggregate_metrics=aggregate,
            overall_status=overall,
            overall_detail=(
                "QUALIFIED FOR TESTED CAPABILITIES" if overall is Status.PASS else overall.value
            ),
        )

    def _selected(self, names: tuple[str, ...] | None) -> tuple[FixtureSpec, ...]:
        if names is None:
            return self.fixture_set.fixtures
        wanted = set(names)
        selected = tuple(item for item in self.fixture_set.fixtures if item.name in wanted)
        missing = sorted(wanted - {item.name for item in selected})
        if missing:
            raise ValueError(f"unknown qualification fixture: {missing[0]}")
        return selected

    def _applicability(self, fixture: FixtureSpec, profile_id: str):
        if profile_id not in fixture.profiles:
            return unavailable_result(
                fixture,
                code="profile_not_applicable",
                detail="fixture does not declare this profile",
            )
        capabilities = self.profile.get("capabilities")
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        required = CAPABILITY_REQUIREMENTS[fixture.capability]
        if not all(capabilities.get(name) is True for name in required):
            return unavailable_result(
                fixture,
                code="capability_not_supported",
                detail=f"profile does not declare {fixture.capability}",
            )
        return None


class RuntimeFixtureExecutor:
    def __init__(self, backend: ChatBackend, *, process_pid: int | None = None) -> None:
        self.backend = backend
        self.process_pid = process_pid

    def execute(self, fixture: FixtureSpec, workdir: Path) -> FixtureObservation:
        started = time.perf_counter()
        steps: list[ModelStepMetrics] = []
        step_walls: list[float | None] = []
        phase_starts: list[float] = []
        tool_calls: list[ToolCallRecord] = []
        tool_results: list[tuple[str, str]] = []
        tool_outcomes: list[ToolOutcomeRecord] = []
        protocol_issue = None

        def phase_start(_phase: ModelPhaseStart) -> None:
            phase_starts.append(time.perf_counter())

        def model_step(step: ModelStepMetrics) -> None:
            steps.append(step)
            step_walls.append(time.perf_counter() - phase_starts.pop(0) if phase_starts else None)

        def tool_call(name: str, raw: str) -> None:
            nonlocal protocol_issue
            try:
                arguments = json.loads(raw)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments are not an object")
            except (json.JSONDecodeError, TypeError, ValueError):
                arguments = {}
                protocol_issue = "malformed_tool_arguments"
            tool_calls.append(ToolCallRecord(name, arguments))

        def tool_result(name: str, _chars: int, _kind: str, content: str) -> None:
            tool_results.append((name, content))
            tool_outcomes.append(_tool_outcome(name, content))

        route = None
        runtime = ChatRuntime(backend=self.backend)
        if fixture.expect.route is not None:
            result, route, route_call = self._route(fixture)
            tool_calls.extend(route_call)
            steps.append(ModelStepMetrics.from_result(loop=1, result=result, phase="route"))
            step_walls.append(time.perf_counter() - started)
        elif not fixture.request.tools:
            route = "CHAT"
            result = runtime.ask_chat(
                fixture.request.prompt,
                temperature=0,
                max_tokens=64,
                on_model_step=model_step,
                on_phase_start=phase_start,
            )
        elif fixture.expect.artifact is not None:
            result = runtime.ask_auto(
                fixture.request.prompt,
                temperature=0,
                max_tokens=512,
                workdir=workdir,
                max_loops=6,
                allowed_tool_names=("write_artifact",),
                on_tool_call=tool_call,
                on_tool_result=tool_result,
                on_model_step=model_step,
                on_phase_start=phase_start,
            )
        else:
            names = (
                ("exec_shell_full_command",)
                if fixture.expect.workflow is not None
                else tuple(item.name for item in fixture.expect.tool_calls)
            )
            result = runtime.ask_with_tools(
                fixture.request.prompt,
                temperature=0,
                max_tokens=160,
                workdir=workdir,
                max_loops=(
                    fixture.expect.max_model_calls
                    if fixture.expect.workflow is not None
                    else 6
                ),
                tool_names=names or None,
                on_tool_call=tool_call,
                on_tool_result=tool_result,
                on_model_step=model_step,
                on_phase_start=phase_start,
            )

        if (
            _has_control_markup(result.content)
            and result.content != fixture.expect.exact_output
        ):
            protocol_issue = "visible_control_markup"
        calls = tuple(_metric(item, step_walls[index]) for index, item in enumerate(steps))
        artifact = _artifact_evidence(fixture, workdir, tool_results)
        residue = _artifact_residue(workdir)
        lifecycle = LifecycleOutcome(
            not residue,
            "clean" if not residue else "private artifact state remains",
        )
        return FixtureObservation(
            route=route,
            tool_calls=tuple(tool_calls),
            executed_tools=tuple(name for name, _content in tool_results),
            final_output=result.content,
            finish_reason=result.finish_reason,
            model_call_count=len(calls),
            retry_count=sum(bool(item.retry_reason or "retry" in item.phase) for item in calls),
            calls=calls,
            artifact=artifact,
            lifecycle=lifecycle,
            peak_rss_bytes=_peak_rss_bytes(self.process_pid),
            wall_seconds=time.perf_counter() - started,
            protocol_issue=protocol_issue,
            tool_outcomes=tuple(tool_outcomes),
        )

    def _route(self, fixture: FixtureSpec):
        allowed = tuple(item.name for item in fixture.expect.tool_calls) or ("exec_shell_full_command",)
        messages = [
            {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
            {"role": "user", "content": fixture.request.prompt},
        ]
        with model_call_context(phase="route", tools_mode="on"):
            result = self.backend.chat(messages, temperature=0, max_tokens=64)
        decision = parse_command_decision_from_tool_calls(result.tool_calls) or parse_command_decision(result.content)
        raw_call = command_tool_call_from_tool_calls(result.tool_calls, allowed)
        raw_call = raw_call or command_tool_call_from_content(result.content, allowed)
        calls = (_tool_record(raw_call),) if raw_call is not None else ()
        return result, decision.route.value if decision else None, calls


def compare_runs(
    fixture_set: FixtureSet,
    baseline: QualificationRun,
    candidate: QualificationRun,
    *,
    comparison_mode: ComparisonMode = ComparisonMode.OPTIMIZATION,
) -> tuple[ParityResult, ...]:
    if baseline.provenance.fixture_set_hash != candidate.provenance.fixture_set_hash:
        raise ValueError("qualification fixture sets differ")
    left = {item.name: item for item in baseline.fixtures}
    right = {item.name: item for item in candidate.fixtures}
    fixtures = {item.name: item for item in fixture_set.fixtures}
    if set(left) != set(right) or not set(left) <= set(fixtures):
        raise ValueError("qualification result fixtures differ")
    return tuple(
        compare_fixture_results(
            fixtures[name], left[name], right[name], comparison_mode=comparison_mode
        )
        for name in sorted(left)
    )


def _metric(value: ModelStepMetrics, wall: float | None) -> CallMetric:
    input_tokens = _token_count(value.prompt_tokens)
    cached_tokens = _token_count(value.cached_tokens)
    valid_cache = (
        input_tokens is not None
        and cached_tokens is not None
        and cached_tokens <= input_tokens
    )
    evaluated = input_tokens - cached_tokens if valid_cache else None
    return CallMetric(
        phase=value.phase,
        input_tokens=input_tokens,
        evaluated_tokens=evaluated,
        cached_tokens=cached_tokens if valid_cache else None,
        output_tokens=_token_count(value.completion_tokens),
        prefill_tokens_per_second=_positive_rate(value.prompt_tokens_per_second),
        generation_tokens_per_second=_positive_rate(value.generation_tokens_per_second),
        wall_seconds=wall,
        finish_reason=value.finish_reason,
        retry_reason=value.retry_reason,
    )


def _token_count(value: int | None) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _positive_rate(value: float | None) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    ):
        return float(value)
    return None


def _tool_record(value: dict[str, Any]) -> ToolCallRecord:
    function = value.get("function") if isinstance(value, dict) else None
    function = function if isinstance(function, dict) else {}
    raw = function.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        arguments = {}
    return ToolCallRecord(
        str(function.get("name") or ""),
        arguments if isinstance(arguments, dict) else {},
    )


def _tool_outcome(name: str, content: str) -> ToolOutcomeRecord:
    failure = shell_failure_from_output(content) if name == "exec_shell_full_command" else None
    failed = failure is not None or content.startswith("error:") or "\nerror:" in content
    return ToolOutcomeRecord(
        name=name,
        status="failure" if failed else "success",
        exit_code=failure.exit_code if failure is not None else None,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _prepare_workspace(fixture: FixtureSpec, workdir: Path) -> None:
    if fixture.workspace is None:
        return
    for item in fixture.workspace.files:
        target = workdir / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8", newline="")


def _workflow_evidence(
    fixture: FixtureSpec,
    workdir: Path,
    observation: FixtureObservation,
) -> WorkflowEvidence:
    expected = fixture.expect.workflow
    assert expected is not None
    paths = tuple(dict.fromkeys([*(item.path for item in expected.files), *expected.unchanged_files]))
    files = tuple(_file_state(workdir, path) for path in paths)
    allowed_paths = {
        *(item.path for item in fixture.workspace.files),
        *(item.path for item in expected.files),
    }
    failed = tuple(
        (index, call)
        for index, (call, outcome) in enumerate(zip(observation.tool_calls, observation.tool_outcomes))
        if outcome.status == "failure"
    )
    failed_signatures = tuple(
        (call.name, json.dumps(call.arguments, sort_keys=True, separators=(",", ":")))
        for _index, call in failed
    )
    success_indices = tuple(
        index for index, outcome in enumerate(observation.tool_outcomes) if outcome.status == "success"
    )
    return WorkflowEvidence(
        files=files,
        unexpected_paths=_unexpected_paths(workdir, allowed_paths),
        failed_tool_calls=len(failed),
        recovery_observed=bool(failed and success_indices and max(success_indices) > failed[0][0]),
        repeated_failed_command=len(failed_signatures) != len(set(failed_signatures)),
        test=_run_workflow_test(expected.test_runner, expected.timeout_seconds, workdir, observation),
    )


def _unexpected_paths(workdir: Path, allowed_paths: set[str]) -> tuple[str, ...]:
    allowed_parents = {
        parent.as_posix()
        for value in allowed_paths
        for parent in Path(value).parents
        if parent.as_posix() != "."
    }
    python_sources = {
        (Path(value).parent / "__pycache__", Path(value).stem)
        for value in allowed_paths
        if value.endswith(".py")
    }
    unexpected = []
    for path in workdir.rglob("*"):
        relative = path.relative_to(workdir).as_posix()
        if relative in allowed_paths or relative in allowed_parents:
            continue
        if not path.is_symlink() and _is_declared_python_cache(path.relative_to(workdir), python_sources):
            continue
        unexpected.append(relative)
    return tuple(sorted(unexpected))


def _is_declared_python_cache(
    relative: Path,
    sources: set[tuple[Path, str]],
) -> bool:
    for cache_directory, stem in sources:
        if relative == cache_directory:
            return True
        if relative.parent == cache_directory:
            prefix = f"{stem}.cpython-"
            if relative.name.startswith(prefix) and relative.name.endswith(".pyc"):
                return True
    return False


def _file_state(workdir: Path, relative: str) -> FileStateEvidence:
    parts = Path(relative).parts
    if any(
        workdir.joinpath(*parts[:index]).is_symlink()
        for index in range(1, len(parts))
    ):
        return FileStateEvidence(relative, True, False, None, None)
    path = workdir / relative
    try:
        info = path.lstat()
        regular = stat.S_ISREG(info.st_mode) and not path.is_symlink()
        content = path.read_bytes() if regular else None
    except OSError:
        return FileStateEvidence(relative, False, False, None, None)
    return FileStateEvidence(
        relative,
        True,
        regular,
        len(content) if content is not None else None,
        hashlib.sha256(content).hexdigest() if content is not None else None,
    )


def _run_workflow_test(
    runner: str | None,
    timeout_seconds: int,
    workdir: Path,
    observation: FixtureObservation,
) -> TestEvidence | None:
    if runner is None:
        return None
    assert runner == "python_unittest"
    command = "python3 -m unittest -q"
    observed = any(
        call.name == "exec_shell_full_command"
        and _has_successful_command_tail(call.arguments.get("command"), command)
        and index < len(observation.tool_outcomes)
        and observation.tool_outcomes[index].status == "success"
        for index, call in enumerate(observation.tool_calls)
    )
    env = dict(os.environ)
    env.update({"HOME": str(workdir), "PWD": str(workdir)})
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "-q"],
            cwd=workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        status = "pass" if completed.returncode == 0 else "failure"
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        status, exit_code = "timeout", None
    except OSError:
        status, exit_code = "error", None
    return TestEvidence(runner, status, exit_code, observed, time.perf_counter() - started)


def _has_successful_command_tail(raw: Any, expected: str) -> bool:
    if not isinstance(raw, str):
        return False
    if "||" in raw or ";" in raw or "\n" in raw:
        return False
    segments = tuple(segment.strip() for segment in raw.split("&&") if segment.strip())
    return bool(segments) and segments[-1] == expected


def _artifact_evidence(
    fixture: FixtureSpec,
    workdir: Path,
    results: list[tuple[str, str]],
) -> ArtifactEvidence | None:
    expected = fixture.expect.artifact
    if expected is None:
        return None
    path = workdir / expected.path
    publication = _evidence_fields(results, "write_artifact")
    verification = _evidence_fields(results, "verify_artifact")
    exists = False
    content = None
    try:
        info = path.lstat()
        exists = stat.S_ISREG(info.st_mode) and not path.is_symlink()
        content = path.read_bytes() if exists else None
    except OSError:
        pass
    actual_hash = hashlib.sha256(content).hexdigest() if content is not None else None
    expected_bytes = str(len(content)) if content is not None else None
    publication_action = publication.get("publication_action")
    published = (
        publication.get("artifact_publication") == "complete"
        and publication.get("path") == expected.path
        and publication.get("bytes") == expected_bytes
        and publication.get("sha256") == actual_hash
        and publication_action in frozenset({"created", "replaced"})
    )
    verified = (
        verification.get("artifact_verification") == "complete"
        and verification.get("path") == expected.path
        and verification.get("bytes") == expected_bytes
        and verification.get("sha256") == actual_hash
        and verification.get("publication_action") == publication_action
        and verification.get("status") == "pass"
    )
    canonical = None
    if content is not None:
        try:
            value = json.loads(content.decode("utf-8"))
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            canonical = hashlib.sha256(encoded).hexdigest()
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return ArtifactEvidence(
        path=expected.path,
        published=published,
        verified=verified,
        exists=exists,
        byte_count=len(content) if content is not None else None,
        sha256=actual_hash,
        publication_action=publication_action,
        verification_check=verification.get("check"),
        canonical_json_sha256=canonical,
    )


def _evidence_fields(results: list[tuple[str, str]], tool: str) -> dict[str, str]:
    fields = {}
    for name, content in results:
        if name != tool:
            continue
        for line in content.splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            fields[key] = value.strip()
    return fields


def _artifact_residue(workdir: Path) -> bool:
    for path in workdir.rglob(".orbit-artifact-*"):
        if path.name == ".orbit-artifact-state" and path.is_dir() and not any(path.iterdir()):
            continue
        return True
    return False


def _peak_rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _has_control_markup(text: str) -> bool:
    markers = (
        "<think>", "</think>", "<|channel>", "<channel|>",
        "<|im_start|>", "<|im_end|>", "<|tool_call|>",
        "<tool_call>", "</tool_call>", "<start_of_turn>", "<end_of_turn>",
    )
    return any(marker in text for marker in markers)


def _overall(common, results) -> Status:
    common_statuses = [item.status for item in common]
    applicable = [item.status for item in results if item.applicable]
    if Status.FAIL in common_statuses or Status.FAIL in applicable:
        return Status.FAIL
    if Status.TECHNICAL_STOP in common_statuses or Status.TECHNICAL_STOP in applicable:
        return Status.TECHNICAL_STOP
    if Status.PASS in applicable:
        return Status.PASS
    return Status.NOT_APPLICABLE
