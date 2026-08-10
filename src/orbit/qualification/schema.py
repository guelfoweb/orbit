from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import math
from typing import Any


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    TECHNICAL_STOP = "TECHNICAL_STOP"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ParityMode(str, Enum):
    EXACT = "exact"
    STRUCTURAL = "structural"


class ComparisonMode(str, Enum):
    OPTIMIZATION = "optimization"
    CROSS_MODEL = "cross_model"


@dataclass(frozen=True)
class Reason:
    code: str
    detail: str


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolOutcomeRecord:
    name: str
    status: str
    exit_code: int | None
    content_sha256: str | None = None


@dataclass(frozen=True)
class CallMetric:
    phase: str
    input_tokens: int | None
    evaluated_tokens: int | None
    cached_tokens: int | None
    output_tokens: int | None
    prefill_tokens_per_second: float | None
    generation_tokens_per_second: float | None
    wall_seconds: float | None
    finish_reason: str | None
    retry_reason: str | None = None


@dataclass(frozen=True)
class AggregateMetrics:
    input_tokens: int | None
    evaluated_tokens: int | None
    cached_tokens: int | None
    output_tokens: int | None
    calls: int
    wall_seconds: float
    prefill_tokens_per_second: float | None
    generation_tokens_per_second: float | None
    peak_rss_bytes: int | None
    ttft_seconds: None = None

    @classmethod
    def from_calls(
        cls,
        calls: tuple[CallMetric, ...],
        wall_seconds: float,
        peak_rss_bytes: int | None,
    ) -> AggregateMetrics:
        input_tokens = _complete_sum(item.input_tokens for item in calls)
        evaluated = _complete_sum(item.evaluated_tokens for item in calls)
        cached = _complete_sum(item.cached_tokens for item in calls)
        output = _complete_sum(item.output_tokens for item in calls)
        return cls(
            input_tokens=input_tokens, evaluated_tokens=evaluated, cached_tokens=cached, output_tokens=output,
            calls=len(calls), wall_seconds=wall_seconds,
            prefill_tokens_per_second=_weighted_rate(calls, "evaluated_tokens", "prefill_tokens_per_second"),
            generation_tokens_per_second=_weighted_rate(calls, "output_tokens", "generation_tokens_per_second"),
            peak_rss_bytes=peak_rss_bytes,
        )


@dataclass(frozen=True)
class ArtifactEvidence:
    path: str
    published: bool
    verified: bool
    exists: bool
    byte_count: int | None
    sha256: str | None
    publication_action: str | None = None
    verification_check: str | None = None
    canonical_json_sha256: str | None = None


@dataclass(frozen=True)
class LifecycleOutcome:
    clean: bool
    detail: str


@dataclass(frozen=True)
class DocumentEvidence:
    coverage: str | None
    analysis_executed: bool
    required_context: int | None
    available_context: int | None
    snapshot_clean: bool


@dataclass(frozen=True)
class FileStateEvidence:
    path: str
    exists: bool
    regular_file: bool
    byte_count: int | None
    sha256: str | None

    @classmethod
    def from_bytes(cls, path: str, content: bytes) -> FileStateEvidence:
        return cls(path, True, True, len(content), hashlib.sha256(content).hexdigest())


@dataclass(frozen=True)
class TestEvidence:
    runner: str
    status: str
    exit_code: int | None
    model_invocation_observed: bool
    wall_seconds: float | None = None


@dataclass(frozen=True)
class WorkflowEvidence:
    files: tuple[FileStateEvidence, ...]
    unexpected_paths: tuple[str, ...]
    failed_tool_calls: int
    recovery_observed: bool
    repeated_failed_command: bool
    test: TestEvidence | None


@dataclass(frozen=True)
class StateReuseEvidence:
    operation: str
    initialized_before: bool | None
    initialized_after: bool | None
    invalidated: bool | None
    recapture_observed: bool | None
    capture_count: int | None
    restore_count: int | None
    fallback_count: int | None
    invalidation_count: int | None
    cached_tokens_after: int | None
    checkpoint_size_after: int | None
    partial_state_accepted: bool | None
    cancellation_observed: bool | None
    restore_rejected: bool | None
    fallback_succeeded: bool | None
    fallback_attempts: int | None
    rss_start_bytes: int | None
    rss_end_bytes: int | None
    rss_peak_bytes: int | None
    rss_tolerance_bytes: int | None
    rss_samples: tuple[int, ...]
    process_pid: int | None
    process_exit_code: int | None
    port_released: bool | None
    residual_state: tuple[str, ...]


@dataclass(frozen=True)
class FixtureObservation:
    route: str | None
    tool_calls: tuple[ToolCallRecord, ...]
    executed_tools: tuple[str, ...]
    final_output: str
    finish_reason: str | None
    model_call_count: int
    retry_count: int
    calls: tuple[CallMetric, ...]
    artifact: ArtifactEvidence | None
    lifecycle: LifecycleOutcome
    peak_rss_bytes: int | None
    wall_seconds: float | None = None
    protocol_issue: str | None = None
    tool_outcomes: tuple[ToolOutcomeRecord, ...] = ()
    workflow: WorkflowEvidence | None = None
    state_reuse: StateReuseEvidence | None = None
    document: DocumentEvidence | None = None


@dataclass(frozen=True)
class ParityResult:
    fixture_name: str
    comparison_mode: ComparisonMode
    mode: ParityMode
    equivalent: bool
    performance_comparison_valid: bool
    mismatches: tuple[str, ...]
    performance: dict[str, Any] | None


@dataclass(frozen=True)
class FixtureResult:
    name: str
    capability: str
    fixture_hash: str
    status: Status
    reason: Reason
    applicable: bool
    route: str | None
    tool_calls: tuple[ToolCallRecord, ...]
    executed_tools: tuple[str, ...]
    final_output_sha256: str | None
    finish_reason: str | None
    model_call_count: int
    retry_count: int
    calls: tuple[CallMetric, ...]
    aggregate_metrics: AggregateMetrics
    artifact: ArtifactEvidence | None
    lifecycle: LifecycleOutcome
    tool_outcomes: tuple[ToolOutcomeRecord, ...]
    workflow: WorkflowEvidence | None
    state_reuse: StateReuseEvidence | None = None
    document: DocumentEvidence | None = None


@dataclass(frozen=True)
class CommonGate:
    name: str
    status: Status
    reason: Reason


@dataclass(frozen=True)
class RunProvenance:
    qualification_schema_version: int
    fixture_set_hash: str
    git_revision: str | None
    profile_identity: str
    model_identity: str | None
    template_identity: str | None
    template_hash: str | None
    backend_identity: str | None
    backend_revision: str | None
    runtime_configuration: dict[str, Any]
    hardware: dict[str, Any]
    measurement_scope: dict[str, str]


@dataclass(frozen=True)
class QualificationRun:
    provenance: RunProvenance
    common: tuple[CommonGate, ...]
    fixtures: tuple[FixtureResult, ...]
    aggregate_metrics: AggregateMetrics
    overall_status: Status
    overall_detail: str


@dataclass(frozen=True)
class ComparisonExecution:
    label: str
    server_pid: int
    startup_wall_seconds: float | None
    configuration: dict[str, Any]
    result: QualificationRun


@dataclass(frozen=True)
class OptimizationComparison:
    baseline: ComparisonExecution
    candidate: ComparisonExecution
    parity: tuple[ParityResult, ...]
    performance_comparison_valid: bool
    mismatches: tuple[str, ...]
    performance: dict[str, Any] | None


def as_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value):
        return {item.name: as_primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): as_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_primitive(item) for item in value]
    return value


def _complete_sum(values) -> int | None:
    items = tuple(values)
    return sum(items) if all(type(item) is int and item >= 0 for item in items) else None


def _weighted_rate(calls: tuple[CallMetric, ...], token_field: str, rate_field: str) -> float | None:
    pairs = tuple((getattr(item, token_field), getattr(item, rate_field)) for item in calls)
    if any(type(tokens) is not int or tokens < 0 for tokens, _rate in pairs):
        return None
    contributing = tuple((tokens, rate) for tokens, rate in pairs if tokens)
    if not contributing or any(
        not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0
        for _tokens, rate in contributing
    ):
        return None
    total = sum(tokens for tokens, _rate in contributing)
    seconds = sum(tokens / rate for tokens, rate in contributing)
    return total / seconds if seconds else None
