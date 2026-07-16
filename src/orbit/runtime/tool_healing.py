from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable
import uuid

from orbit.runtime.kv_diag import emit_tool_healing_shadow
from orbit.runtime.tool_contract import CanonicalToolDecision, validate_canonical_tool_call
from orbit.tool_contract_config import resolve_tool_call_canonical_gate
from orbit.tool_healing_config import resolve_tool_call_healing, resolve_tool_call_healing_shadow


_FIELD_RE = re.compile(r'"(?:name|tool|function|arguments)"\s*:')
_TOOL_TAG_RE = re.compile(r"<tool_call(?:\s[^>]*)?>", re.IGNORECASE)
_ALT_TOOL_TAG_RE = re.compile(r"<\|tool_call>", re.IGNORECASE)
_TOOL_CALLS_RE = re.compile(r"\[TOOL_CALLS]", re.IGNORECASE)
_ARGS_RE = re.compile(r"\[ARGS]", re.IGNORECASE)
_MAX_EXCERPT_CHARS = 96
_FORMAL_REPAIR_WHITELIST = frozenset(
    {
        "remove_trailing_comma",
        "decode_arguments_string",
        "unwrap_function_object",
        "unwrap_named_wrapper",
    }
)
_KNOWN_ENVELOPE_SOURCES = frozenset(
    {"tool_call_tag", "gemma_tool_call_wrapper", "tool_calls_args_wrapper"}
)
_HEALING_DIAGNOSTIC_LOCK = threading.Lock()
_HEALING_REPAIR_COUNT = 0
_HEALING_REJECTION_COUNT = 0
_HEALING_LAST_RULES: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolAttemptDetection:
    detected: bool
    signals: tuple[str, ...]


@dataclass(frozen=True)
class ExtractedCandidate:
    payload: Any | None
    source: str | None
    repairs: tuple[str, ...]
    raw_fragment: str | None
    error: str | None


@dataclass(frozen=True)
class CanonicalCandidate:
    name: str
    arguments: dict[str, Any]
    repairs: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    error: str | None
    error_path: str | None = None


@dataclass(frozen=True)
class ToolHealingShadowReport:
    attempt_id: str
    attempt_detected: bool
    signals: tuple[str, ...]
    candidate_count: int
    candidate_extracted: bool
    candidate_source: str | None
    candidate_hash: str | None
    candidate_excerpt: str | None
    repairs: tuple[str, ...]
    tool_name: str | None
    original_tool_name_hash: str | None
    normalized_tool_name_hash: str | None
    normalized_arguments_hash: str | None
    parse_error: str | None
    validation_error: str | None
    validation_path: str | None
    outcome: str
    formal_repairable: bool = False
    formal_repair_reason: str | None = None
    formal_argument_count: int | None = None


@dataclass(frozen=True)
class ToolHealingActiveCorrelation:
    active_candidate_count: int
    active_tool_name_hash: str | None
    active_arguments_hash: str | None
    active_outcome: str
    terminal_reason: str | None
    agreement: str
    canonical_outcome: str | None = None
    canonical_error: str | None = None


@dataclass(frozen=True)
class FormalRepairProof:
    authorized: bool
    reason: str
    tool_name_hash: str | None = None
    arguments_hash: str | None = None
    argument_count: int | None = None


@dataclass(frozen=True)
class ToolRepairCertificate:
    repair_categories: tuple[str, ...]
    candidate_count_one: bool
    tool_name_unchanged: bool
    argument_count_unchanged: bool
    argument_keys_unchanged: bool
    argument_values_unchanged: bool
    idempotent: bool
    schema_valid: bool
    policy_valid: bool
    permission_valid: bool
    operational_limits_valid: bool
    finish_complete: bool
    tool_name_hash: str
    arguments_hash: str
    argument_count: int


@dataclass(frozen=True)
class ToolCallRepairResult:
    tool_call: dict[str, Any] | None
    certificate: ToolRepairCertificate | None
    reason: str
    canonical_decision: CanonicalToolDecision | None = None


class _DuplicateKeyError(ValueError):
    pass


def tool_call_healing_status() -> dict[str, Any]:
    config = resolve_tool_call_healing()
    canonical = resolve_tool_call_canonical_gate()
    enabled = config.enabled and canonical.enabled
    with _HEALING_DIAGNOSTIC_LOCK:
        repair_count = _HEALING_REPAIR_COUNT
        rejection_count = _HEALING_REJECTION_COUNT
        last_rules = list(_HEALING_LAST_RULES)
    return {
        "tool_call_healing_enabled": enabled,
        "tool_call_healing_source": config.source,
        "tool_call_healing_config_error": config.validation_error,
        "tool_call_healing_blocked_reason": None if enabled or not config.enabled else "canonical_gate_disabled",
        "tool_call_healing_repair_count": repair_count,
        "tool_call_healing_rejection_count": rejection_count,
        "tool_call_healing_last_rules": last_rules,
    }


def reset_tool_call_healing_diagnostics() -> None:
    global _HEALING_REPAIR_COUNT, _HEALING_REJECTION_COUNT, _HEALING_LAST_RULES
    with _HEALING_DIAGNOSTIC_LOCK:
        _HEALING_REPAIR_COUNT = 0
        _HEALING_REJECTION_COUNT = 0
        _HEALING_LAST_RULES = ()


def _record_healing_repair(rules: tuple[str, ...]) -> None:
    global _HEALING_REPAIR_COUNT, _HEALING_LAST_RULES
    with _HEALING_DIAGNOSTIC_LOCK:
        _HEALING_REPAIR_COUNT += 1
        _HEALING_LAST_RULES = rules


def _rejected_repair(reason: str) -> ToolCallRepairResult:
    global _HEALING_REJECTION_COUNT
    with _HEALING_DIAGNOSTIC_LOCK:
        _HEALING_REJECTION_COUNT += 1
    return ToolCallRepairResult(None, None, reason)


class ToolAttemptDetector:
    def detect(
        self,
        *,
        text: str,
        tool_calls: list[dict[str, Any]],
        registered_tool_names: Iterable[str],
    ) -> ToolAttemptDetection:
        signals: set[str] = set()
        if tool_calls:
            signals.add("backend_tool_calls")
        if _TOOL_TAG_RE.search(text):
            signals.add("tool_call_tag")
        if _ALT_TOOL_TAG_RE.search(text):
            signals.add("gemma_tool_call_wrapper")
        if _TOOL_CALLS_RE.search(text) and _ARGS_RE.search(text):
            signals.add("tool_calls_args_wrapper")

        stripped = _strip_outer_fence(text.strip())
        structured_object_count = _structured_json_payload_count(stripped)
        whole_json_shape = structured_object_count > 0
        if whole_json_shape and _FIELD_RE.search(stripped):
            signals.add("whole_structured_object")
        if structured_object_count > 1:
            signals.add("multiple_structured_objects")
        if any(_quoted_name_present(stripped, name) for name in registered_tool_names):
            if signals or whole_json_shape:
                signals.add("exact_registered_tool_name")
        if whole_json_shape and '"arguments"' in stripped:
            signals.add("arguments_field")
        if whole_json_shape and ('"name"' in stripped or '"tool"' in stripped or '"function"' in stripped):
            signals.add("tool_identity_field")
        return ToolAttemptDetection(detected=bool(signals), signals=tuple(sorted(signals)))


class CandidateExtractor:
    def extract(
        self,
        *,
        text: str,
        tool_calls: list[dict[str, Any]],
        detection: ToolAttemptDetection,
    ) -> ExtractedCandidate:
        if not detection.detected:
            return ExtractedCandidate(None, None, (), None, "not_detected")
        if tool_calls:
            if len(tool_calls) != 1 or _candidate_regions(text):
                return ExtractedCandidate(None, "backend", (), None, "multiple_candidates")
            return ExtractedCandidate(tool_calls[0], "backend", (), _json_for_diagnostic(tool_calls[0]), None)

        regions = _candidate_regions(text)
        if len(regions) != 1:
            error = "multiple_candidates" if len(regions) > 1 else "candidate_not_found"
            return ExtractedCandidate(None, None, (), None, error)
        source, region, wrapper_name, wrapper_repair = regions[0]
        fragments = _json_fragments(region)
        if len(fragments) != 1:
            error = "multiple_candidates" if len(fragments) > 1 else "candidate_not_found"
            return ExtractedCandidate(None, source, (), None, error)
        fragment, complete, closers, scan_error = fragments[0]
        if scan_error:
            return ExtractedCandidate(None, source, (), fragment, scan_error)

        repairs: list[str] = []
        repaired = fragment
        if not complete:
            if not closers:
                return ExtractedCandidate(None, source, (), fragment, "incomplete_json")
            repaired += closers
            repairs.append("close_json_structure")
        without_trailing = _remove_trailing_commas(repaired)
        if without_trailing != repaired:
            repaired = without_trailing
            repairs.append("remove_trailing_comma")
        if wrapper_repair:
            repairs.append(wrapper_repair)
        try:
            payload = json.loads(repaired, object_pairs_hook=_unique_object)
        except _DuplicateKeyError:
            return ExtractedCandidate(None, source, tuple(repairs), fragment, "duplicate_key")
        except json.JSONDecodeError:
            return ExtractedCandidate(None, source, tuple(repairs), fragment, "invalid_json")
        if wrapper_name is not None:
            payload = {"name": wrapper_name, "arguments": payload}
            repairs.append("unwrap_named_wrapper")
        return ExtractedCandidate(payload, source, tuple(repairs), fragment, None)


class CanonicalToolCallValidator:
    def __init__(
        self,
        *,
        tool_definitions: list[dict[str, Any]],
        allowed_tool_names: Iterable[str],
        workdir: Path,
        user_prompt: str | None,
    ) -> None:
        self.tool_definitions = tool_definitions
        self.definitions = _definitions_by_name(tool_definitions)
        self.allowed = frozenset(allowed_tool_names)
        self.workdir = workdir
        self.user_prompt = user_prompt

    def canonicalize(self, candidate: ExtractedCandidate) -> tuple[CanonicalCandidate | None, ValidationResult]:
        if candidate.error:
            return None, ValidationResult(False, candidate.error)
        if not isinstance(candidate.payload, dict):
            return None, ValidationResult(False, "candidate_not_object")
        payload = candidate.payload
        repairs = list(candidate.repairs)

        if "function" in payload:
            if any(key not in {"id", "type", "function"} for key in payload):
                return None, ValidationResult(False, "ambiguous_wrapper")
            function = payload.get("function")
            if not isinstance(function, dict):
                return None, ValidationResult(False, "function_not_object", "function")
            payload = function
            repairs.append("unwrap_function_object")

        identity_keys = [key for key in ("name", "tool") if key in payload]
        if len(identity_keys) != 1:
            return None, ValidationResult(False, "ambiguous_tool_identity")
        identity_key = identity_keys[0]
        name = payload.get(identity_key)
        if not isinstance(name, str) or not name:
            return None, ValidationResult(False, "invalid_tool_name", identity_key)
        if identity_key == "tool":
            repairs.append("normalize_tool_field")

        if "arguments" in payload:
            if any(key not in {"name", "tool", "arguments", "id", "type"} for key in payload):
                return None, ValidationResult(False, "ambiguous_wrapper")
            arguments: Any = payload.get("arguments")
        else:
            top_level_arguments = {
                key: value
                for key, value in payload.items()
                if key not in {"name", "tool", "id", "type"}
            }
            schema = self.definitions.get(name)
            if schema is None:
                return None, ValidationResult(False, "unknown_tool", "name")
            properties = schema.get("properties")
            properties = properties if isinstance(properties, dict) else {}
            if any(key not in properties for key in top_level_arguments):
                return None, ValidationResult(False, "additional_property", "arguments.<unknown>")
            arguments = top_level_arguments
            repairs.append("wrap_top_level_arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments, object_pairs_hook=_unique_object)
            except _DuplicateKeyError:
                return None, ValidationResult(False, "duplicate_key", "arguments")
            except json.JSONDecodeError:
                return None, ValidationResult(False, "arguments_invalid_json", "arguments")
            repairs.append("decode_arguments_string")
        if not isinstance(arguments, dict):
            return None, ValidationResult(False, "arguments_not_object", "arguments")
        return CanonicalCandidate(name=name, arguments=arguments, repairs=tuple(dict.fromkeys(repairs))), ValidationResult(True, None)

    def decide(self, candidate: CanonicalCandidate) -> CanonicalToolDecision:
        return validate_canonical_tool_call(
            candidate.name,
            candidate.arguments,
            tool_definitions=self.tool_definitions,
            allowed_tool_names=self.allowed,
            workdir=self.workdir,
            user_prompt=self.user_prompt,
        )

    def validate(
        self,
        candidate: CanonicalCandidate,
        *,
        decision: CanonicalToolDecision | None = None,
    ) -> ValidationResult:
        decision = decision or self.decide(candidate)
        if decision.accepted:
            return ValidationResult(True, None)
        path = next(
            (
                outcome.path
                for outcome in (
                    decision.schema_outcome,
                    decision.permission_outcome,
                    decision.policy_outcome,
                    decision.operational_limit_outcome,
                )
                if not outcome.accepted
            ),
            None,
        )
        return ValidationResult(False, decision.rejection_code, path)


def prove_formal_repair(
    *,
    extracted: ExtractedCandidate,
    canonical: CanonicalCandidate,
    validation: ValidationResult,
    validator: CanonicalToolCallValidator,
    candidate_count: int,
    finish_reason: str | None,
) -> FormalRepairProof:
    if finish_reason in {"length", "cancelled", "timeout"}:
        return FormalRepairProof(False, "incomplete_generation")
    if candidate_count != 1:
        return FormalRepairProof(False, "candidate_count_not_one")
    if not validation.valid:
        return FormalRepairProof(False, "canonical_validation_failed")
    if extracted.source == "backend":
        return FormalRepairProof(False, "active_backend_shape")
    repairs = frozenset(extracted.repairs) | frozenset(canonical.repairs)
    if repairs - _FORMAL_REPAIR_WHITELIST:
        return FormalRepairProof(False, "repair_not_whitelisted")
    envelope_removed = extracted.source in _KNOWN_ENVELOPE_SOURCES
    if not repairs and not envelope_removed:
        return FormalRepairProof(False, "no_repair_required")
    original_name, original_arguments, identity_error = _identity_from_extracted_payload(extracted.payload)
    if identity_error is not None:
        return FormalRepairProof(False, identity_error)
    if original_name != canonical.name:
        return FormalRepairProof(False, "tool_name_changed")
    if not _json_values_identical(original_arguments, canonical.arguments):
        return FormalRepairProof(False, "argument_values_changed")

    stable = ExtractedCandidate(
        payload={"name": canonical.name, "arguments": canonical.arguments},
        source="canonical",
        repairs=(),
        raw_fragment=None,
        error=None,
    )
    repeated, repeated_result = validator.canonicalize(stable)
    if (
        repeated is None
        or repeated.repairs
        or repeated.name != canonical.name
        or not _json_values_identical(repeated.arguments, canonical.arguments)
    ):
        return FormalRepairProof(False, "repair_not_idempotent")
    return FormalRepairProof(
        True,
        "formal_equivalence_proven",
        tool_name_hash=_hash_identifier(canonical.name),
        arguments_hash=_hash_value(canonical.arguments),
        argument_count=len(canonical.arguments),
    )


def build_tool_call_repair(
    *,
    text: str,
    tool_calls: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
    finish_reason: str | None,
    attempt_id: str | None = None,
) -> ToolCallRepairResult:
    if tool_calls:
        return ToolCallRepairResult(None, None, "active_tool_call_present")
    allowed = tuple(allowed_tool_names)
    detection = ToolAttemptDetector().detect(
        text=text,
        tool_calls=tool_calls,
        registered_tool_names=allowed,
    )
    if not detection.detected:
        return ToolCallRepairResult(None, None, "no_attempt")
    if finish_reason in {"length", "cancelled", "timeout"}:
        return _rejected_repair("incomplete_generation")
    extracted = CandidateExtractor().extract(
        text=text,
        tool_calls=tool_calls,
        detection=detection,
    )
    candidate_count = _candidate_count(text, tool_calls, detection)
    if candidate_count != 1:
        return _rejected_repair("candidate_count_not_one")
    if not _has_strong_template_evidence(text, extracted):
        return _rejected_repair("insufficient_template_evidence")
    if extracted.error:
        return _rejected_repair(extracted.error)

    validator = CanonicalToolCallValidator(
        tool_definitions=tool_definitions,
        allowed_tool_names=allowed,
        workdir=workdir,
        user_prompt=user_prompt,
    )
    canonical, canonical_result = validator.canonicalize(extracted)
    if canonical is None:
        return _rejected_repair(canonical_result.error or "canonicalization_failed")
    decision = validator.decide(canonical)
    validation = validator.validate(canonical, decision=decision)
    proof = prove_formal_repair(
        extracted=extracted,
        canonical=canonical,
        validation=validation,
        validator=validator,
        candidate_count=candidate_count,
        finish_reason=finish_reason,
    )
    if not proof.authorized:
        return _rejected_repair(proof.reason)

    original_name, original_arguments, identity_error = _identity_from_extracted_payload(extracted.payload)
    if identity_error is not None or original_arguments is None or original_name is None:
        return _rejected_repair(identity_error or "identity_unavailable")
    same_keys = set(original_arguments) == set(canonical.arguments)
    same_values = _json_values_identical(original_arguments, canonical.arguments)
    certificate = ToolRepairCertificate(
        repair_categories=_repair_categories(extracted, canonical),
        candidate_count_one=True,
        tool_name_unchanged=original_name == canonical.name,
        argument_count_unchanged=len(original_arguments) == len(canonical.arguments),
        argument_keys_unchanged=same_keys,
        argument_values_unchanged=same_values,
        idempotent=True,
        schema_valid=decision.schema_outcome.accepted,
        policy_valid=decision.policy_outcome.accepted,
        permission_valid=decision.permission_outcome.accepted,
        operational_limits_valid=decision.operational_limit_outcome.accepted,
        finish_complete=True,
        tool_name_hash=_hash_identifier(canonical.name) or "",
        arguments_hash=_hash_value(canonical.arguments),
        argument_count=len(canonical.arguments),
    )
    if not all(
        (
            certificate.candidate_count_one,
            certificate.tool_name_unchanged,
            certificate.argument_count_unchanged,
            certificate.argument_keys_unchanged,
            certificate.argument_values_unchanged,
            certificate.idempotent,
            certificate.schema_valid,
            certificate.policy_valid,
            certificate.permission_valid,
            certificate.operational_limits_valid,
            certificate.finish_complete,
        )
    ):
        return _rejected_repair("certificate_incomplete")
    call_id = attempt_id or f"healed-{certificate.arguments_hash[:12]}"
    _record_healing_repair(certificate.repair_categories)
    return ToolCallRepairResult(
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": canonical.name,
                "arguments": json.dumps(canonical.arguments, ensure_ascii=False, separators=(",", ":")),
            },
        },
        certificate,
        "formal_repair_authorized",
        decision,
    )


def _identity_from_extracted_payload(payload: Any) -> tuple[str | None, dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, None, "candidate_not_object"
    if "function" in payload:
        if any(key not in {"id", "type", "function"} for key in payload):
            return None, None, "ambiguous_wrapper"
        payload = payload.get("function")
        if not isinstance(payload, dict):
            return None, None, "function_not_object"
    if set(payload) - {"name", "arguments", "id", "type"}:
        return None, None, "identity_contains_noncanonical_fields"
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return None, None, "invalid_tool_name"
    if "arguments" not in payload:
        return None, None, "arguments_missing"
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, object_pairs_hook=_unique_object)
        except (_DuplicateKeyError, json.JSONDecodeError):
            return None, None, "arguments_not_unique_json"
    if not isinstance(arguments, dict):
        return None, None, "arguments_not_object"
    return name, arguments, None


def analyze_tool_attempt(
    *,
    text: str,
    tool_calls: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
    attempt_id: str | None = None,
    finish_reason: str | None = None,
) -> ToolHealingShadowReport:
    resolved_attempt_id = attempt_id or uuid.uuid4().hex
    allowed = tuple(allowed_tool_names)
    detector = ToolAttemptDetector()
    detection = detector.detect(text=text, tool_calls=tool_calls, registered_tool_names=allowed)
    extracted = CandidateExtractor().extract(text=text, tool_calls=tool_calls, detection=detection)
    candidate_count = _candidate_count(text, tool_calls, detection)
    candidate_hash = _hash_fragment(extracted.raw_fragment)
    excerpt = _bounded_excerpt(extracted.raw_fragment)
    original_tool_name_hash = _hash_identifier(_original_tool_name(extracted.payload))
    if not detection.detected:
        return ToolHealingShadowReport(
            resolved_attempt_id,
            False,
            detection.signals,
            candidate_count,
            False,
            None,
            None,
            None,
            (),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "no_attempt",
        )
    if extracted.error:
        fragment_extracted = extracted.raw_fragment is not None
        return ToolHealingShadowReport(
            resolved_attempt_id,
            True,
            detection.signals,
            candidate_count,
            fragment_extracted,
            extracted.source,
            candidate_hash,
            excerpt,
            extracted.repairs,
            None,
            original_tool_name_hash,
            None,
            None,
            extracted.error,
            None,
            None,
            "rejected_parse",
        )
    validator = CanonicalToolCallValidator(
        tool_definitions=tool_definitions,
        allowed_tool_names=allowed,
        workdir=workdir,
        user_prompt=user_prompt,
    )
    canonical, canonical_result = validator.canonicalize(extracted)
    repairs = canonical.repairs if canonical is not None else extracted.repairs
    if canonical is None:
        return ToolHealingShadowReport(
            resolved_attempt_id,
            True,
            detection.signals,
            candidate_count,
            True,
            extracted.source,
            candidate_hash,
            excerpt,
            repairs,
            None,
            original_tool_name_hash,
            None,
            None,
            None,
            canonical_result.error,
            canonical_result.error_path,
            "rejected_validation",
        )
    result = validator.validate(canonical)
    proof = prove_formal_repair(
        extracted=extracted,
        canonical=canonical,
        validation=result,
        validator=validator,
        candidate_count=candidate_count,
        finish_reason=finish_reason,
    )
    return ToolHealingShadowReport(
        resolved_attempt_id,
        True,
        detection.signals,
        candidate_count,
        True,
        extracted.source,
        candidate_hash,
        excerpt,
        canonical.repairs,
        canonical.name,
        original_tool_name_hash,
        _hash_identifier(canonical.name),
        _hash_value(canonical.arguments),
        None,
        result.error,
        result.error_path,
        "valid_shadow_candidate" if result.valid else "rejected_validation",
        proof.authorized,
        proof.reason,
        proof.argument_count,
    )


def observe_tool_attempt_shadow(
    *,
    text: str,
    tool_calls: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
    finish_reason: str | None,
    output_tokens: int | None,
    phase: str = "tool_call",
    attempt_id: str | None = None,
) -> ToolHealingShadowReport | None:
    config = resolve_tool_call_healing_shadow()
    if not config.enabled:
        return None
    started_ns = time.perf_counter_ns()
    try:
        report = analyze_tool_attempt(
            text=text,
            tool_calls=tool_calls,
            tool_definitions=tool_definitions,
            allowed_tool_names=allowed_tool_names,
            workdir=workdir,
            user_prompt=user_prompt,
            attempt_id=attempt_id,
            finish_reason=finish_reason,
        )
    except Exception:
        report = ToolHealingShadowReport(
            attempt_id or uuid.uuid4().hex,
            attempt_detected=False,
            signals=(),
            candidate_count=0,
            candidate_extracted=False,
            candidate_source=None,
            candidate_hash=None,
            candidate_excerpt=None,
            repairs=(),
            tool_name=None,
            original_tool_name_hash=None,
            normalized_tool_name_hash=None,
            normalized_arguments_hash=None,
            parse_error="shadow_internal_error",
            validation_error=None,
            validation_path=None,
            outcome="shadow_error",
        )
    try:
        emit_tool_healing_shadow(
            report=report,
            finish_reason=finish_reason,
            output_tokens=output_tokens,
            output_hash=_hash_fragment(text) or hashlib.sha256(b"").hexdigest()[:16],
            output_chars=len(text),
            phase=phase,
            config_source=config.source,
            config_error=config.validation_error,
            healing_us=round((time.perf_counter_ns() - started_ns) / 1000, 1),
        )
    except Exception:
        # A shadow diagnostic sink must never affect the active tool path.
        pass
    return report


def correlate_tool_attempt_shadow(
    *,
    report: ToolHealingShadowReport | None,
    active_tool_calls: list[dict[str, Any]],
    phase: str,
) -> ToolHealingActiveCorrelation | None:
    del phase
    if report is None:
        return None
    active_count = len(active_tool_calls)
    active_name, active_arguments = _single_active_identity(active_tool_calls)
    active_name_hash = _hash_identifier(active_name)
    active_arguments_hash = _hash_value(active_arguments)
    active_outcome = (
        "no_tool_call"
        if active_count == 0
        else "single_tool_call"
        if active_count == 1
        else "multiple_tool_calls"
    )
    agreement = _active_agreement(
        report,
        active_count=active_count,
        active_name_hash=active_name_hash,
        active_arguments_hash=active_arguments_hash,
    )
    correlation = ToolHealingActiveCorrelation(
        active_candidate_count=active_count,
        active_tool_name_hash=active_name_hash,
        active_arguments_hash=active_arguments_hash,
        active_outcome=active_outcome,
        terminal_reason=None,
        agreement=agreement,
    )
    return correlation


def begin_tool_healing_attempt() -> str | None:
    return uuid.uuid4().hex if resolve_tool_call_healing_shadow().enabled else None


def record_tool_healing_terminal(
    *,
    attempt_id: str | None,
    report: ToolHealingShadowReport | None,
    active_tool_calls: list[dict[str, Any]],
    active_outcome: str,
    terminal_reason: str | None,
    phase: str,
    tool_definitions: list[dict[str, Any]] | None = None,
    allowed_tool_names: Iterable[str] = (),
    workdir: Path | None = None,
    user_prompt: str | None = None,
    canonical_decision: CanonicalToolDecision | None = None,
    canonical_evaluated: bool = False,
) -> ToolHealingActiveCorrelation | None:
    if attempt_id is None:
        return None
    active_count = len(active_tool_calls)
    active_name, active_arguments = _single_active_identity(active_tool_calls)
    active_name_hash = _hash_identifier(active_name)
    active_arguments_hash = _hash_value(active_arguments)
    agreement = (
        _active_agreement(
            report,
            active_count=active_count,
            active_name_hash=active_name_hash,
            active_arguments_hash=active_arguments_hash,
        )
        if report is not None
        else "uncorrelated"
    )
    canonical_outcome = None
    canonical_error = None
    if canonical_evaluated:
        canonical_outcome = "valid" if canonical_decision is not None and canonical_decision.accepted else "rejected_validation"
        canonical_error = canonical_decision.rejection_code if canonical_decision is not None else "canonical_decision_missing"
    elif tool_definitions is not None and workdir is not None:
        try:
            canonical_outcome, canonical_error = validate_active_tool_calls(
                active_tool_calls,
                tool_definitions=tool_definitions,
                allowed_tool_names=allowed_tool_names,
                workdir=workdir,
                user_prompt=user_prompt,
            )
        except Exception:
            canonical_outcome, canonical_error = "shadow_error", "active_validation_error"
    correlation = ToolHealingActiveCorrelation(
        active_candidate_count=active_count,
        active_tool_name_hash=active_name_hash,
        active_arguments_hash=active_arguments_hash,
        active_outcome=active_outcome,
        terminal_reason=terminal_reason,
        agreement=agreement,
        canonical_outcome=canonical_outcome,
        canonical_error=canonical_error,
    )
    try:
        from orbit.runtime.kv_diag import emit_tool_healing_terminal

        emit_tool_healing_terminal(
            report=report,
            correlation=correlation,
            phase=phase,
            attempt_id=attempt_id,
        )
    except Exception:
        # Terminal observation cannot affect the active tool lifecycle.
        pass
    return correlation


def validate_active_tool_calls(
    active_tool_calls: list[dict[str, Any]],
    *,
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
) -> tuple[str, str | None]:
    if not active_tool_calls:
        return "no_tool_call", None
    if len(active_tool_calls) != 1:
        return "rejected_validation", "multiple_candidates"
    validator = CanonicalToolCallValidator(
        tool_definitions=tool_definitions,
        allowed_tool_names=allowed_tool_names,
        workdir=workdir,
        user_prompt=user_prompt,
    )
    extracted = ExtractedCandidate(active_tool_calls[0], "backend", (), None, None)
    canonical, canonical_result = validator.canonicalize(extracted)
    if canonical is None:
        return "rejected_validation", canonical_result.error
    result = validator.validate(canonical)
    return ("valid" if result.valid else "rejected_validation"), result.error


def _has_strong_template_evidence(text: str, extracted: ExtractedCandidate) -> bool:
    if extracted.source not in _KNOWN_ENVELOPE_SOURCES:
        return False
    stripped = text.strip()
    if "```" in stripped:
        return False
    if extracted.source == "tool_call_tag":
        if re.fullmatch(r"<tool_call(?:\s[^>]*)?>.*</tool_call\s*>", stripped, re.IGNORECASE | re.DOTALL) is None:
            return False
    elif extracted.source == "gemma_tool_call_wrapper":
        if re.fullmatch(r"<\|tool_call>.*<tool_call\|>", stripped, re.IGNORECASE | re.DOTALL) is None:
            return False
    elif extracted.source == "tool_calls_args_wrapper":
        if re.fullmatch(r"\[TOOL_CALLS].*\[ARGS].*", stripped, re.IGNORECASE | re.DOTALL) is None:
            return False
    regions = _candidate_regions(stripped)
    if len(regions) != 1:
        return False
    source, region, _wrapper_name, wrapper_repair = regions[0]
    if source != extracted.source or wrapper_repair is not None:
        return False
    fragments = _json_fragments(region)
    if len(fragments) != 1:
        return False
    fragment, complete, _closers, scan_error = fragments[0]
    return scan_error is None and complete and region.strip() == fragment.strip()


def _repair_categories(
    extracted: ExtractedCandidate,
    canonical: CanonicalCandidate,
) -> tuple[str, ...]:
    categories = ["remove_known_envelope"]
    repairs = set(extracted.repairs) | set(canonical.repairs)
    if "remove_trailing_comma" in repairs:
        categories.append("remove_trailing_comma")
    if "decode_arguments_string" in repairs:
        categories.append("decode_arguments_string")
    if repairs & {"unwrap_function_object", "unwrap_named_wrapper"}:
        categories.append("unwrap_registered_wrapper")
    return tuple(categories)


def _json_values_identical(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_json_values_identical(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_identical(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _candidate_regions(text: str) -> list[tuple[str, str, str | None, str | None]]:
    regions: list[tuple[str, str, str | None, str | None]] = []
    tag_matches = list(_TOOL_TAG_RE.finditer(text))
    for match in tag_matches:
        close = re.search(r"</tool_call\s*>", text[match.end() :], re.IGNORECASE)
        end = match.end() + close.start() if close else len(text)
        regions.append(("tool_call_tag", text[match.end() : end], None, None if close else "unclosed_tool_call_tag"))
    alt_matches = list(_ALT_TOOL_TAG_RE.finditer(text))
    for match in alt_matches:
        close_index = text.find("<tool_call|>", match.end())
        end = close_index if close_index >= 0 else len(text)
        body = text[match.end() : end].strip()
        named = re.match(r"call:([A-Za-z_][A-Za-z0-9_]*)\s*", body)
        regions.append(
            (
                "gemma_tool_call_wrapper",
                body[named.end() :] if named else body,
                named.group(1) if named else None,
                None if close_index >= 0 else "unclosed_tool_call_wrapper",
            )
        )
    wrapper_matches = list(_TOOL_CALLS_RE.finditer(text))
    for match in wrapper_matches:
        args = _ARGS_RE.search(text, match.end())
        if args is None:
            regions.append(("tool_calls_args_wrapper", "", None, None))
            continue
        name = text[match.end() : args.start()].strip()
        regions.append(("tool_calls_args_wrapper", text[args.end() :], name or None, None))
    if regions:
        return regions
    stripped = _strip_outer_fence(text.strip())
    if stripped.startswith("{"):
        return [("whole_structured_object", stripped, None, None)]
    return []


def _json_fragments(text: str) -> list[tuple[str, bool, str, str | None]]:
    fragments: list[tuple[str, bool, str, str | None]] = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        end, complete, closers, error = _scan_json_object(text, index)
        fragments.append((text[index:end], complete, closers, error))
        index = max(index + 1, end)
    return fragments


def _scan_json_object(text: str, start: int) -> tuple[int, bool, str, str | None]:
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or (char == "}" and stack[-1] != "{") or (char == "]" and stack[-1] != "["):
                return index + 1, False, "", "mismatched_json_delimiter"
            stack.pop()
            if not stack:
                return index + 1, True, "", None
    if in_string or escaped:
        return len(text), False, "", "unterminated_json_string"
    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return len(text), False, closers, None


def _remove_trailing_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _definitions_by_name(tool_definitions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for definition in tool_definitions:
        function = definition.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, dict):
            definitions[name] = parameters
    return definitions


def _candidate_count(
    text: str,
    tool_calls: list[dict[str, Any]],
    detection: ToolAttemptDetection,
) -> int:
    if not detection.detected:
        return 0
    text_regions = _candidate_regions(text)
    text_count = 0
    for _source, region, _name, _repair in text_regions:
        text_count += max(1, len(_json_fragments(region)))
    return len(tool_calls) + text_count


def _original_tool_name(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    function = payload.get("function")
    if isinstance(function, dict):
        payload = function
    names = [payload.get(key) for key in ("name", "tool") if key in payload]
    if len(names) != 1 or not isinstance(names[0], str):
        return None
    return names[0]


def _single_active_identity(
    active_tool_calls: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    if len(active_tool_calls) != 1:
        return None, None
    function = active_tool_calls[0].get("function")
    if not isinstance(function, dict):
        return None, None
    name = function.get("name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, _DuplicateKeyError):
            arguments = None
    return (name if isinstance(name, str) else None, arguments if isinstance(arguments, dict) else None)


def _active_agreement(
    report: ToolHealingShadowReport,
    *,
    active_count: int,
    active_name_hash: str | None,
    active_arguments_hash: str | None,
) -> str:
    shadow_valid = report.outcome == "valid_shadow_candidate"
    if active_count > 1 or report.candidate_count > 1:
        return "multiple_candidates"
    if shadow_valid and active_count == 0:
        return "shadow_only"
    if not shadow_valid and active_count == 1:
        return "active_only"
    if not shadow_valid and active_count == 0:
        return "both_no_active_call"
    if report.normalized_tool_name_hash != active_name_hash:
        return "tool_name_divergence"
    if report.normalized_arguments_hash != active_arguments_hash:
        return "arguments_divergence"
    return "exact_match"


def _hash_identifier(value: str | None) -> str | None:
    return _hash_fragment(value)


def _hash_value(value: Any | None) -> str | None:
    if value is None:
        return None
    return _hash_fragment(_json_for_diagnostic(value))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _structured_json_payload_count(text: str) -> int:
    fragments = _json_fragments(text)
    if not fragments:
        return 0
    cursor = 0
    for fragment, _complete, _closers, _error in fragments:
        start = text.find(fragment, cursor)
        if start < 0 or text[cursor:start].strip():
            return 0
        cursor = start + len(fragment)
    return len(fragments) if not text[cursor:].strip() else 0


def _strip_outer_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _quoted_name_present(text: str, name: str) -> bool:
    return re.search(rf'"{re.escape(name)}"', text) is not None


def _json_for_diagnostic(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _hash_fragment(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _bounded_excerpt(value: str | None) -> str | None:
    if value is None:
        return None
    compact = " ".join(_redacted_structural_excerpt(value).split())
    if len(compact) > _MAX_EXCERPT_CHARS:
        return compact[: _MAX_EXCERPT_CHARS - 3] + "..."
    return compact


def _redacted_structural_excerpt(value: str) -> str:
    allowed_keys = {"name", "tool", "function", "arguments", "id", "type"}
    output: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == '"':
            end = index + 1
            escaped = False
            while end < len(value):
                current = value[end]
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
                end += 1
            closed = end < len(value) and value[end] == '"'
            raw = value[index : end + 1] if closed else value[index:]
            lookahead = end + 1 if closed else len(value)
            while lookahead < len(value) and value[lookahead].isspace():
                lookahead += 1
            try:
                decoded = json.loads(raw) if closed else None
            except json.JSONDecodeError:
                decoded = None
            is_key = lookahead < len(value) and value[lookahead] == ":"
            output.append(raw if is_key and decoded in allowed_keys else '"<redacted>"')
            index = end + 1 if closed else len(value)
            continue
        if char.isdigit() or (char == "-" and index + 1 < len(value) and value[index + 1].isdigit()):
            end = index + 1
            while end < len(value) and (value[end].isdigit() or value[end] in ".eE+-"):
                end += 1
            output.append("#")
            index = end
            continue
        if char.isalpha() or char == "_":
            end = index + 1
            while end < len(value) and (value[end].isalnum() or value[end] == "_"):
                end += 1
            token = value[index:end]
            output.append(token if token in {"true", "false", "null"} else "<token>")
            index = end
            continue
        output.append(char)
        index += 1
    return "".join(output)
