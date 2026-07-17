#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys


METRICS = (
    "completion_rate",
    "valid_first_pass_rate",
    "exact_tool_match_rate",
    "model_unwanted_attempt_rate",
    "detector_false_positive_rate",
    "detector_false_negative_rate",
    "multiple_candidate_rate",
    "budget_truncation_rate",
    "structural_truncation_rate",
    "markup_leakage_rate",
    "generation_wall_ms_median",
    "generation_wall_ms_p95",
    "healing_us_median",
    "healing_us_p95",
)

NONCOMPARABLE_BLOCKERS = {
    "baseline_capability_unavailable",
    "baseline_renderer_or_tokenizer_mismatch",
    "capability_contract_mismatch",
    "candidate_capability_unavailable",
    "candidate_renderer_or_tokenizer_mismatch",
    "configuration_mismatch",
    "corpus_mismatch",
    "protocol_mismatch",
    "sample_set_mismatch",
}


@dataclass(frozen=True)
class GenerationRun:
    environment: dict[str, object]
    samples: dict[tuple[str, int], dict[str, object]]
    summary: dict[str, object]


class ComparisonInputError(ValueError):
    pass


def load_generation_run(path: Path) -> GenerationRun:
    environments: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    samples: dict[tuple[str, int], dict[str, object]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ComparisonInputError("row_not_object")
                row_type = value.get("type")
                if row_type == "environment":
                    environments.append(value)
                elif row_type == "tool_call_generation_summary":
                    summaries.append(value)
                elif row_type == "tool_call_generation":
                    key = _sample_key(value)
                    if key in samples:
                        raise ComparisonInputError("duplicate_sample")
                    samples[key] = value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonInputError(f"unreadable_jsonl:{type(exc).__name__}") from exc
    if len(environments) != 1 or environments[0].get("benchmark_mode") != "tool_call_generation_only":
        raise ComparisonInputError("missing_generation_environment")
    if len(summaries) != 1:
        raise ComparisonInputError("missing_generation_summary")
    if not samples:
        raise ComparisonInputError("missing_generation_samples")
    return GenerationRun(environments[0], samples, summaries[0])


def compare_generation_runs(baseline: GenerationRun, candidate: GenerationRun) -> dict[str, object]:
    blockers: list[str] = []
    baseline_corpus = _corpus_identity(baseline.environment)
    candidate_corpus = _corpus_identity(candidate.environment)
    if baseline_corpus != candidate_corpus or None in baseline_corpus:
        blockers.append("corpus_mismatch")

    baseline_protocol = _versioned_hash_identity(baseline.environment, "protocol")
    candidate_protocol = _versioned_hash_identity(candidate.environment, "protocol")
    if baseline_protocol != candidate_protocol or None in baseline_protocol:
        blockers.append("protocol_mismatch")

    baseline_configuration = _versioned_hash_identity(baseline.environment, "configuration")
    candidate_configuration = _versioned_hash_identity(candidate.environment, "configuration")
    if baseline_configuration != candidate_configuration or None in baseline_configuration:
        blockers.append("configuration_mismatch")

    blockers.extend(_capability_blockers("baseline", baseline.environment.get("native_backend_capabilities")))
    blockers.extend(_capability_blockers("candidate", candidate.environment.get("native_backend_capabilities")))
    if _capability_contract_identity(baseline.environment) != _capability_contract_identity(candidate.environment):
        blockers.append("capability_contract_mismatch")

    baseline_keys = set(baseline.samples)
    candidate_keys = set(candidate.samples)
    if baseline_keys != candidate_keys:
        blockers.append("sample_set_mismatch")

    if not NONCOMPARABLE_BLOCKERS.intersection(blockers):
        blockers.extend(_sample_regressions(baseline.samples, candidate.samples))

    if _safe_count(candidate.summary.get("tools_executed")) != 0:
        blockers.append("candidate_executed_tool")
    if _safe_count(candidate.summary.get("finalizations_started")) != 0:
        blockers.append("candidate_started_finalization")
    if _safe_count(candidate.summary.get("model_calls")) != len(candidate.samples):
        blockers.append("candidate_model_call_count")

    deltas = {metric: _numeric_delta(baseline.summary.get(metric), candidate.summary.get(metric)) for metric in METRICS}
    comparable = not NONCOMPARABLE_BLOCKERS.intersection(blockers)
    return {
        "type": "tool_call_generation_comparison",
        "decision": "pass" if not blockers else "fail" if comparable else "incomparable",
        "comparable": comparable,
        "corpus": {"version": baseline_corpus[0], "hash": baseline_corpus[1]},
        "protocol": {"version": baseline_protocol[0], "hash": baseline_protocol[1]},
        "configuration": {"version": baseline_configuration[0], "hash": baseline_configuration[1]},
        "baseline": _run_identity(baseline),
        "candidate": _run_identity(candidate),
        "identity_changes": _identity_changes(baseline.environment, candidate.environment),
        "metric_deltas": deltas,
        "blockers": sorted(set(blockers)),
        "raw_content_included": False,
    }


def _sample_regressions(
    baseline: dict[tuple[str, int], dict[str, object]],
    candidate: dict[tuple[str, int], dict[str, object]],
) -> list[str]:
    blockers: list[str] = []
    for key in sorted(baseline):
        before = baseline[key]
        after = candidate[key]
        if before.get("expected_tool") != after.get("expected_tool"):
            blockers.append("expected_tool_mismatch")
        if before.get("template") != after.get("template"):
            blockers.append("template_mismatch")
        if before.get("evaluable") is True and after.get("evaluable") is not True:
            blockers.append("completion_regression")
        if before.get("semantic_outcome") == "expected_tool" and after.get("semantic_outcome") != "expected_tool":
            blockers.append("tool_selection_regression")
        if before.get("semantic_outcome") == "no_attempt" and after.get("semantic_outcome") != "no_attempt":
            blockers.append("negative_false_positive_regression")
        for field, blocker in (
            ("multiple_candidates", "multiple_candidate_regression"),
            ("markup_leakage", "markup_leakage_regression"),
            ("tool_executed", "sample_tool_execution"),
            ("finalization_started", "sample_finalization"),
        ):
            if before.get(field) is not True and after.get(field) is True:
                blockers.append(blocker)
        if _safe_count(after.get("model_calls")) != 1:
            blockers.append("sample_model_call_count")
    return blockers


def _run_identity(run: GenerationRun) -> dict[str, object]:
    capability = _capability_identity(run.environment.get("native_backend_capabilities"))
    return {
        "samples": len(run.samples),
        "model": _safe_identifier(run.environment.get("model")),
        "mtp": _safe_identifier(run.environment.get("mtp")),
        "protocol": _versioned_hash_mapping(run.environment, "protocol"),
        "configuration": _versioned_hash_mapping(run.environment, "configuration"),
        "capability": capability,
    }


def _identity_changes(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, bool]:
    before = _capability_identity(baseline.get("native_backend_capabilities"))
    after = _capability_identity(candidate.get("native_backend_capabilities"))
    return {
        "profile": before.get("profile_id") != after.get("profile_id"),
        "backend_commit": before.get("backend_commit") != after.get("backend_commit"),
        "backend_library": before.get("backend_library_hash") != after.get("backend_library_hash"),
        "tool_protocol": before.get("tool_protocol_text_hash") != after.get("tool_protocol_text_hash"),
        "final_prefix": before.get("final_prefix_text_hash") != after.get("final_prefix_text_hash"),
        "renderer_fixture_suite": before.get("renderer_fixture_suite_hash") != after.get("renderer_fixture_suite_hash"),
        "tokenizer": before.get("tokenizer_prefix_hash") != after.get("tokenizer_prefix_hash"),
    }


def _capability_identity(value: object) -> dict[str, object]:
    manifest = value if isinstance(value, dict) else {}
    build_number = manifest.get("backend_build_number")
    return {
        "profile_id": _safe_identifier(manifest.get("profile_id")),
        "status": _safe_identifier(manifest.get("status")),
        "backend_build_number": build_number if isinstance(build_number, int) and not isinstance(build_number, bool) else None,
        "backend_commit": _safe_identifier(manifest.get("backend_commit")),
        "backend_library_hash": _safe_hash(manifest.get("backend_library_hash")),
        "tool_protocol_text_hash": _safe_hash(manifest.get("tool_protocol_text_hash")),
        "final_prefix_text_hash": _safe_hash(manifest.get("final_prefix_text_hash")),
        "renderer_fixture_suite_hash": _safe_hash(manifest.get("renderer_fixture_suite_hash")),
        "tokenizer_prefix_hash": _safe_hash(manifest.get("tokenizer_prefix_hash")),
    }


def _corpus_identity(environment: dict[str, object]) -> tuple[int | None, str | None]:
    version = environment.get("corpus_version")
    return (
        version if isinstance(version, int) and not isinstance(version, bool) else None,
        _safe_hash(environment.get("corpus_hash")),
    )


def _versioned_hash_identity(environment: dict[str, object], name: str) -> tuple[int | None, str | None]:
    version = environment.get(f"{name}_version")
    return (
        version if isinstance(version, int) and not isinstance(version, bool) else None,
        _safe_hash(environment.get(f"{name}_hash")),
    )


def _versioned_hash_mapping(environment: dict[str, object], name: str) -> dict[str, object]:
    version, value_hash = _versioned_hash_identity(environment, name)
    return {"version": version, "hash": value_hash}


def _capability_blockers(label: str, value: object) -> list[str]:
    capability = _capability_identity(value)
    status = capability.get("status")
    if capability.get("profile_id") is None or status is None:
        return [f"{label}_capability_unavailable"]
    if any(
        capability.get(field) is None
        for field in (
            "tool_protocol_text_hash",
            "final_prefix_text_hash",
            "renderer_fixture_suite_hash",
            "tokenizer_prefix_hash",
        )
    ):
        return [f"{label}_capability_unavailable"]
    if status not in {"verified", "backend_unverified"}:
        return [f"{label}_renderer_or_tokenizer_mismatch"]
    return []


def _capability_contract_identity(environment: dict[str, object]) -> tuple[object, ...]:
    capability = _capability_identity(environment.get("native_backend_capabilities"))
    return (
        capability.get("profile_id"),
        capability.get("tool_protocol_text_hash"),
        capability.get("final_prefix_text_hash"),
        capability.get("renderer_fixture_suite_hash"),
        capability.get("tokenizer_prefix_hash"),
    )


def _sample_key(row: dict[str, object]) -> tuple[str, int]:
    scenario = _safe_identifier(row.get("scenario"))
    repetition = row.get("repetition")
    if scenario is None or not isinstance(repetition, int) or isinstance(repetition, bool) or repetition < 1:
        raise ComparisonInputError("invalid_sample_identity")
    return scenario, repetition


def _safe_identifier(value: object, *, limit: int = 96) -> str | None:
    if not isinstance(value, str) or not value or len(value) > limit:
        return None
    if not all(character.isalnum() or character in {"_", "-", ".", ":"} for character in value):
        return None
    return value


def _safe_hash(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    return value if all(character in "0123456789abcdef" for character in value) else None


def _numeric_delta(before: object, after: object) -> float | None:
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if not isinstance(before, int | float) or not isinstance(after, int | float):
        return None
    return round(float(after) - float(before), 6)


def _safe_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else -1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two redacted Orbit tool-call generation benchmark JSONL files.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_generation_runs(load_generation_run(args.baseline), load_generation_run(args.candidate))
    except ComparisonInputError as exc:
        print(json.dumps({"type": "tool_call_generation_comparison", "decision": "invalid", "error": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"] == "pass" else 2 if result["decision"] == "incomparable" else 1


if __name__ == "__main__":
    sys.exit(main())
