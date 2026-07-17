from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence


CALL_CATEGORIES = {
    "initial_route",
    "tool_call",
    "post_tool_route",
    "retry",
    "final_from_tool",
    "chat_final",
    "formatting_only",
    "confirmation_only",
    "error_recovery",
}

DISPOSITIONS = {
    "necessary",
    "avoidable",
    "duplicate",
    "formatting_only",
    "deterministic_completion_candidate",
    "retry_caused",
}

AUDIT_EXPECTATIONS = {
    "unknown",
    "bounded_confirmation",
    "structured_error",
    "synthesis_required",
    "next_tool_required",
}

_RETRY_PHASES = {
    "route_retry",
    "tool_call_retry",
    "tool_call_json_retry",
    "chat_final_retry",
    "final_from_tool_retry",
    "chat_continue_native",
}
_FORMATTING_PHASES = {
    "chat_final_completion_repair",
    "final_from_tool_completion_repair",
    "final_from_tool_compact_retry",
}
_ERROR_REASONS = {
    "empty_or_invalid",
    "empty_response",
    "length",
    "length_retry",
    "tool_contract_retry",
}


def audit_model_calls(
    *,
    model_steps: Sequence[Mapping[str, object]],
    phase_starts: Sequence[Mapping[str, object]] = (),
    phase_timings: Sequence[Mapping[str, object]] = (),
    route_outputs: Sequence[Mapping[str, object]] = (),
    tool_names: Sequence[str] = (),
    expectation: str = "unknown",
    correctness: str = "not_evaluated",
) -> dict[str, object]:
    """Classify completed calls without changing or interpreting runtime behavior."""
    if expectation not in AUDIT_EXPECTATIONS:
        expectation = "unknown"
    occurrences = _correlate_occurrences(model_steps, phase_timings)
    starts = _phase_start_queues(phase_starts)
    route_queues = _route_output_queues(route_outputs)
    calls: list[dict[str, object]] = []
    allocated_tools = 0

    for index, occurrence in enumerate(occurrences):
        metric = occurrence.get("metric")
        metric = metric if isinstance(metric, Mapping) else {}
        timing_phase = _identifier(occurrence.get("phase")) or "unknown"
        metric_phase = _identifier(metric.get("phase"))
        phase = metric_phase or timing_phase
        start = _take_phase_start(starts, timing_phase, phase)
        reason = _identifier(start.get("reason")) if start else None
        next_occurrence = occurrences[index + 1] if index + 1 < len(occurrences) else None
        next_phase = _occurrence_phase(next_occurrence)
        tool_call_count = _integer(metric.get("tool_calls")) or 0
        category = _call_category(
            phase=phase,
            timing_phase=timing_phase,
            loop=_integer(metric.get("loop")),
            tool_call_count=tool_call_count,
            tools_executed=bool(tool_names),
            expectation=expectation,
            reason=reason or _identifier(metric.get("retry_reason")),
        )
        route_event = _take_route_output(route_queues, timing_phase, phase)
        decision = _decision_kind(category, tool_call_count, route_event)
        executed_tool = None
        if tool_call_count > 0 and allocated_tools < len(tool_names):
            executed_tool = tool_names[allocated_tools]
            allocated_tools += 1
        elif category == "initial_route" and route_event:
            selected = route_event.get("selected_tool")
            executed_tool = selected if isinstance(selected, str) else None
        disposition = _disposition(
            category=category,
            expectation=expectation,
            correctness=correctness,
            finish_reason=_identifier(metric.get("finish_reason")),
            tool_count=len(tool_names),
            tool_call_count=tool_call_count,
            next_phase=next_phase,
        )
        evaluated_tokens = _integer(metric.get("evaluated_tokens"))
        if evaluated_tokens is None:
            prompt_tokens = _integer(metric.get("prompt_tokens"))
            cached_tokens = _integer(metric.get("cached_tokens"))
            if prompt_tokens is not None and cached_tokens is not None:
                evaluated_tokens = prompt_tokens - cached_tokens
        counted = disposition in {"avoidable", "duplicate", "deterministic_completion_candidate"}
        calls.append(
            {
                "call_id": f"call_{index + 1:03d}",
                "category": category,
                "phase": phase,
                "timing_phase": timing_phase,
                "entry_reason": reason or _default_entry_reason(category),
                "prompt_view": _prompt_view(phase, timing_phase),
                "prompt_tokens": _integer(metric.get("prompt_tokens")),
                "cached_tokens": _integer(metric.get("cached_tokens")),
                "evaluated_tokens": evaluated_tokens,
                "output_tokens": _integer(metric.get("completion_tokens")),
                "wall_ms": _number(occurrence.get("wall_ms")),
                "finish_reason": _identifier(metric.get("finish_reason")),
                "decision": decision,
                "tool_executed": executed_tool,
                "output_changes_next_step": _changes_next_step(category, decision),
                "repeats_determined_decision": _repeats_determined_decision(category, expectation),
                "disposition": disposition,
                "counted_as_theoretical_saving": counted,
                "potential_evaluated_tokens": evaluated_tokens if counted else 0,
                "potential_wall_ms": _number(occurrence.get("wall_ms")) if counted else 0.0,
                "information_loss_if_removed": _information_loss(category, disposition),
                "required_invariant": _required_invariant(category, disposition),
                "metrics_correlated": bool(metric),
            }
        )

    category_counts = Counter(str(call["category"]) for call in calls)
    disposition_counts = Counter(str(call["disposition"]) for call in calls)
    return {
        "calls": calls,
        "summary": {
            "model_calls": len(calls),
            "reported_model_steps": len(model_steps),
            "timed_backend_calls": len(phase_timings),
            "uncorrelated_backend_calls": sum(not call["metrics_correlated"] for call in calls),
            "categories": dict(sorted(category_counts.items())),
            "dispositions": dict(sorted(disposition_counts.items())),
            "theoretical_model_calls": sum(bool(call["counted_as_theoretical_saving"]) for call in calls),
            "theoretical_evaluated_tokens": sum(int(call["potential_evaluated_tokens"] or 0) for call in calls),
            "theoretical_wall_ms": round(sum(float(call["potential_wall_ms"] or 0.0) for call in calls), 1),
            "expectation": expectation,
            "correctness": correctness,
        },
    }


def summarize_inference_audits(audits: Iterable[Mapping[str, object]]) -> dict[str, object]:
    category_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    calls = 0
    theoretical_calls = 0
    theoretical_tokens = 0
    theoretical_wall_ms = 0.0
    scenarios = 0
    for audit in audits:
        summary = audit.get("summary")
        if not isinstance(summary, Mapping):
            continue
        scenarios += 1
        calls += _integer(summary.get("model_calls")) or 0
        theoretical_calls += _integer(summary.get("theoretical_model_calls")) or 0
        theoretical_tokens += _integer(summary.get("theoretical_evaluated_tokens")) or 0
        theoretical_wall_ms += _number(summary.get("theoretical_wall_ms")) or 0.0
        _update_counter(category_counts, summary.get("categories"))
        _update_counter(disposition_counts, summary.get("dispositions"))
    return {
        "type": "inference_audit_summary",
        "scenarios": scenarios,
        "model_calls": calls,
        "categories": dict(sorted(category_counts.items())),
        "dispositions": dict(sorted(disposition_counts.items())),
        "theoretical_model_calls": theoretical_calls,
        "theoretical_evaluated_tokens": theoretical_tokens,
        "theoretical_wall_ms": round(theoretical_wall_ms, 1),
        "active_behavior_changed": False,
    }


def _correlate_occurrences(
    model_steps: Sequence[Mapping[str, object]],
    phase_timings: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not phase_timings:
        return [
            {"phase": _identifier(metric.get("phase")) or "unknown", "wall_ms": None, "metric": metric}
            for metric in model_steps
        ]
    if len(phase_timings) <= len(model_steps):
        occurrences = [
            {
                "phase": _identifier(metric.get("phase")) or "unknown",
                "wall_ms": None,
                "metric": metric,
            }
            for metric in model_steps
        ]
        available = set(range(len(occurrences)))
        extra_timings: list[dict[str, object]] = []
        for timing in phase_timings:
            timing_phase = _identifier(timing.get("phase")) or "unknown"
            match = next(
                (
                    index
                    for index in sorted(available)
                    if _phases_compatible(timing_phase, str(occurrences[index]["phase"]))
                ),
                None,
            )
            if match is None:
                extra_timings.append(
                    {"phase": timing_phase, "wall_ms": _number(timing.get("wall_ms")), "metric": None}
                )
                continue
            occurrences[match]["phase"] = timing_phase
            occurrences[match]["wall_ms"] = _number(timing.get("wall_ms"))
            available.remove(match)
        occurrences.extend(extra_timings)
        return occurrences
    occurrences = [
        {
            "phase": _identifier(item.get("phase")) or "unknown",
            "wall_ms": _number(item.get("wall_ms")),
            "metric": None,
        }
        for item in phase_timings
    ]
    available = set(range(len(occurrences)))
    for metric in reversed(model_steps):
        metric_phase = _identifier(metric.get("phase")) or "unknown"
        match = next(
            (
                index
                for index in sorted(available, reverse=True)
                if _phases_compatible(str(occurrences[index]["phase"]), metric_phase)
            ),
            None,
        )
        if match is None:
            continue
        occurrences[match]["metric"] = metric
        available.remove(match)
    unmatched_metrics = [metric for metric in model_steps if all(item.get("metric") is not metric for item in occurrences)]
    occurrences.extend(
        {"phase": _identifier(metric.get("phase")) or "unknown", "wall_ms": None, "metric": metric}
        for metric in unmatched_metrics
    )
    return occurrences


def _phases_compatible(timing_phase: str, metric_phase: str) -> bool:
    if timing_phase == metric_phase:
        return True
    if metric_phase == "post_tool_route" and timing_phase == "tool_call":
        return True
    return metric_phase == "final" and timing_phase in {"tool_call", "tool_call_retry", "tool_call_json_retry"}


def _call_category(
    *,
    phase: str,
    timing_phase: str,
    loop: int | None,
    tool_call_count: int,
    tools_executed: bool,
    expectation: str,
    reason: str | None,
) -> str:
    if phase in _FORMATTING_PHASES:
        return "formatting_only"
    if phase in _RETRY_PHASES or timing_phase in _RETRY_PHASES:
        return "error_recovery" if reason in _ERROR_REASONS else "retry"
    if phase == "route":
        return "initial_route"
    if phase == "post_tool_route":
        return "post_tool_route"
    if phase == "final" and tools_executed:
        return "post_tool_route"
    if phase == "tool_call" or timing_phase == "tool_call":
        if tools_executed and (loop or 0) > 1 and tool_call_count == 0:
            return "post_tool_route"
        return "tool_call"
    if phase == "final_from_tool":
        if expectation in {"bounded_confirmation", "structured_error"}:
            return "confirmation_only"
        return "final_from_tool"
    if phase == "chat_final" or phase.startswith("chat_continue"):
        return "chat_final"
    return "chat_final"


def _disposition(
    *,
    category: str,
    expectation: str,
    correctness: str,
    finish_reason: str | None,
    tool_count: int,
    tool_call_count: int,
    next_phase: str | None,
) -> str:
    if category in {"retry", "error_recovery"}:
        return "retry_caused"
    if category == "formatting_only":
        return "formatting_only"
    bounded = expectation in {"bounded_confirmation", "structured_error"}
    complete = correctness == "correct" and finish_reason == "stop"
    if category == "post_tool_route" and bounded and tool_count == 1 and tool_call_count == 0:
        if next_phase == "final_from_tool":
            return "deterministic_completion_candidate"
    if category == "confirmation_only" and bounded and complete and tool_count == 1:
        return "deterministic_completion_candidate"
    return "necessary"


def _decision_kind(category: str, tool_call_count: int, route_event: Mapping[str, object] | None) -> str:
    if route_event is not None and isinstance(route_event.get("parsed_route"), str):
        return str(route_event["parsed_route"])
    if tool_call_count:
        return "tool_call"
    if category == "post_tool_route":
        return "stop_tools"
    if category in {"final_from_tool", "confirmation_only", "chat_final", "formatting_only"}:
        return "final_answer"
    if category in {"retry", "error_recovery"}:
        return "replacement_output"
    return "unknown"


def _changes_next_step(category: str, decision: str) -> bool | None:
    if category in {"initial_route", "tool_call", "post_tool_route", "retry", "error_recovery"}:
        return True
    if decision == "final_answer":
        return False
    return None


def _repeats_determined_decision(category: str, expectation: str) -> bool | None:
    if category == "confirmation_only" and expectation in {"bounded_confirmation", "structured_error"}:
        return True
    return None


def _prompt_view(phase: str, timing_phase: str) -> str:
    effective = timing_phase if timing_phase != "unknown" else phase
    if effective == "route":
        return "route"
    if effective == "route_retry":
        return "route_retry"
    if effective.startswith("tool_call"):
        return effective
    if effective.startswith("final_from_tool"):
        return "final_from_tool_family"
    if effective.startswith("chat_final"):
        return "chat_final_family"
    return effective


def _default_entry_reason(category: str) -> str:
    return {
        "initial_route": "tool_decision",
        "tool_call": "tool_selection",
        "post_tool_route": "post_tool_decision",
        "retry": "retry",
        "final_from_tool": "tool_evidence_finalization",
        "chat_final": "chat_response",
        "formatting_only": "completion_format_repair",
        "confirmation_only": "bounded_tool_result_finalization",
        "error_recovery": "error_recovery",
    }[category]


def _information_loss(category: str, disposition: str) -> str:
    if disposition == "deterministic_completion_candidate":
        if category == "post_tool_route":
            return "model_decision_to_stop_or_select_another_tool"
        return "model_wording_and_evidence_interpretation"
    if disposition == "formatting_only":
        return "clean_final_answer_when_first_output_is_incomplete"
    if disposition == "retry_caused":
        return "recovery_from_an_invalid_or_incomplete_prior_output"
    return "semantic_decision_or_user_facing_synthesis"


def _required_invariant(category: str, disposition: str) -> str:
    if disposition == "deterministic_completion_candidate":
        if category == "post_tool_route":
            return "single_terminal_tool_result_and_no_possible_follow_up_tool_needed"
        return "bounded_structured_result_fully_satisfies_an_explicit_response_contract"
    if disposition == "formatting_only":
        return "first_pass_completion_is_already_valid_and_complete"
    if disposition == "retry_caused":
        return "first_pass_format_and_budget_reliability"
    return "model_output_remains_required"


def _phase_start_queues(phase_starts: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    result: dict[str, list[Mapping[str, object]]] = {}
    for item in phase_starts:
        phase = _identifier(item.get("phase"))
        if phase:
            result.setdefault(phase, []).append(item)
    return result


def _take_phase_start(
    queues: dict[str, list[Mapping[str, object]]],
    timing_phase: str,
    metric_phase: str,
) -> Mapping[str, object] | None:
    for phase in (timing_phase, metric_phase):
        values = queues.get(phase)
        if values:
            return values.pop(0)
    return None


def _route_output_queues(
    route_outputs: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    result = {"route": [], "route_retry": []}
    for item in route_outputs:
        key = "route_retry" if item.get("route_call") == "retry" else "route"
        result[key].append(item)
    return result


def _take_route_output(
    queues: dict[str, list[Mapping[str, object]]],
    timing_phase: str,
    metric_phase: str,
) -> Mapping[str, object] | None:
    for phase in (timing_phase, metric_phase):
        values = queues.get(phase)
        if values:
            return values.pop(0)
    return None


def _occurrence_phase(occurrence: Mapping[str, object] | None) -> str | None:
    if occurrence is None:
        return None
    metric = occurrence.get("metric")
    if isinstance(metric, Mapping):
        phase = _identifier(metric.get("phase"))
        if phase:
            return phase
    return _identifier(occurrence.get("phase"))


def _update_counter(counter: Counter[str], value: object) -> None:
    if not isinstance(value, Mapping):
        return
    for key, count in value.items():
        if isinstance(key, str) and isinstance(count, int):
            counter[key] += count


def _identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 80:
        return None
    return value


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
