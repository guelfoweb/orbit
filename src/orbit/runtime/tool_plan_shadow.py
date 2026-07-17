from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from orbit.runtime.tool_contract import CanonicalToolDecision, validate_canonical_tool_call


TOOL_PLAN_SCHEMA_VERSION = 2
TOOL_PLAN_STEP_COUNT = 2
TOOL_PLAN_INITIAL_TOOLS = frozenset({"list_directory", "system_info"})
_PLAN_MARKER = re.compile(r'"type"\s*:\s*"(?:tool_plan|unsupported_plan)"')
_PLAN_REFERENCE = re.compile(r"(?:\$\{?|\{\{)[^}]{0,80}(?:step|output|result)", re.IGNORECASE)


class DuplicatePlanKey(ValueError):
    pass


@dataclass(frozen=True)
class ToolPlanStep:
    id: str
    name: str
    arguments: dict[str, Any]
    canonical_decision: CanonicalToolDecision


@dataclass(frozen=True)
class ToolPlan:
    steps: tuple[ToolPlanStep, ToolPlanStep]


@dataclass(frozen=True)
class ToolPlanShadowReport:
    detected: bool
    candidate_count: int
    json_compliant: bool
    valid: bool
    response_kind: str | None
    rejection_code: str | None
    plan: ToolPlan | None
    payload_hash: str | None

    def diagnostic(self) -> dict[str, object]:
        steps = self.plan.steps if self.plan is not None else ()
        return {
            "schema_version": TOOL_PLAN_SCHEMA_VERSION,
            "detected": self.detected,
            "candidate_count": self.candidate_count,
            "json_compliant": self.json_compliant,
            "valid": self.valid,
            "response_kind": self.response_kind,
            "rejection_code": self.rejection_code,
            "step_count": len(steps),
            "step_ids": [step.id for step in steps],
            "tool_hashes": [_hash_text(step.name) for step in steps],
            "argument_hashes": [_hash_json(step.arguments) for step in steps],
            "payload_hash": self.payload_hash,
            "prose_leakage": self.rejection_code in {"external_text", "no_plan"},
            "invalid_json": self.rejection_code in {"invalid_json", "duplicate_key"},
            "raw_content_included": False,
            "tool_executed": False,
            "finalization_started": False,
        }


def analyze_tool_plan_shadow(
    text: str,
    *,
    finish_reason: str | None,
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
) -> ToolPlanShadowReport:
    markers = len(_PLAN_MARKER.findall(text))
    if markers == 0:
        return _report(False, 0, False, False, None, "no_plan", None, text)
    if markers != 1:
        return _report(True, markers, False, False, None, "multiple_candidates", None, text)
    if finish_reason in {"length", "cancelled", "timeout"}:
        return _report(True, 1, False, False, None, f"nonrecoverable_{finish_reason}", None, text)
    if finish_reason not in {None, "stop"}:
        return _report(True, 1, False, False, None, "nonrecoverable_finish_reason", None, text)
    stripped = text.strip()
    if not stripped.startswith("{"):
        return _report(True, 1, False, False, None, "external_text", None, text)
    try:
        payload = json.loads(stripped, object_pairs_hook=_unique_object)
    except DuplicatePlanKey:
        return _report(True, 1, False, False, None, "duplicate_key", None, stripped)
    except json.JSONDecodeError:
        if _has_complete_json_prefix(stripped):
            return _report(True, 1, False, False, None, "external_text", None, stripped)
        return _report(True, 1, False, False, None, "invalid_json", None, stripped)
    plan, kind, rejection = validate_tool_plan_payload(
        payload,
        tool_definitions=tool_definitions,
        allowed_tool_names=allowed_tool_names,
        workdir=workdir,
        user_prompt=user_prompt,
    )
    return _report(True, 1, True, rejection is None, kind, rejection, plan, stripped)


def validate_tool_plan_payload(
    payload: object,
    *,
    tool_definitions: list[dict[str, Any]],
    allowed_tool_names: Iterable[str],
    workdir: Path,
    user_prompt: str | None,
) -> tuple[ToolPlan | None, str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None, "invalid_response_shape"
    response_type = payload.get("type")
    if response_type == "unsupported_plan":
        if set(payload) != {"type"}:
            return None, "unsupported", "invalid_unsupported_shape"
        return None, "unsupported", None
    if response_type != "tool_plan" or set(payload) != {"type", "steps"}:
        return None, None, "invalid_plan_shape"
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != TOOL_PLAN_STEP_COUNT:
        return None, "plan", "invalid_step_count"
    allowed = tuple(allowed_tool_names)
    steps: list[ToolPlanStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict) or set(raw_step) != {"name", "arguments"}:
            return None, "plan", "invalid_step_shape"
        name = raw_step.get("name")
        arguments = raw_step.get("arguments")
        if not isinstance(name, str) or not name:
            return None, "plan", "invalid_tool_name"
        if name not in TOOL_PLAN_INITIAL_TOOLS:
            return None, "plan", "tool_not_plan_eligible"
        if not isinstance(arguments, dict):
            return None, "plan", "arguments_not_object"
        if _contains_dynamic_reference(arguments):
            return None, "plan", "dynamic_dependency"
        decision = validate_canonical_tool_call(
            name,
            arguments,
            tool_definitions=tool_definitions,
            allowed_tool_names=allowed,
            workdir=workdir,
            user_prompt=user_prompt,
        )
        if not decision.accepted:
            return None, "plan", f"canonical_{decision.rejection_code or decision.terminal_decision}"
        steps.append(ToolPlanStep(f"step_{index}", name, arguments, decision))
    assert len(steps) == TOOL_PLAN_STEP_COUNT
    return ToolPlan((steps[0], steps[1])), "plan", None


def _report(
    detected: bool,
    candidate_count: int,
    json_compliant: bool,
    valid: bool,
    response_kind: str | None,
    rejection_code: str | None,
    plan: ToolPlan | None,
    hashed_text: str,
) -> ToolPlanShadowReport:
    return ToolPlanShadowReport(
        detected,
        candidate_count,
        json_compliant,
        valid,
        response_kind,
        rejection_code,
        plan,
        _hash_text(hashed_text),
    )


def _contains_dynamic_reference(value: object) -> bool:
    if isinstance(value, str):
        return bool(_PLAN_REFERENCE.search(value))
    if isinstance(value, dict):
        return any(_contains_dynamic_reference(key) or _contains_dynamic_reference(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_dynamic_reference(item) for item in value)
    return False


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicatePlanKey(key)
        result[key] = value
    return result


def _has_complete_json_prefix(text: str) -> bool:
    try:
        _value, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return False
    return bool(text[end:].strip())


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return _hash_text(encoded)
