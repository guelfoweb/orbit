#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


class ComparisonInputError(ValueError):
    pass


def load_run(path: Path) -> tuple[dict[str, object], dict[tuple[str, int, int], dict[str, object]]]:
    environments: list[dict[str, object]] = []
    steps: dict[tuple[str, int, int], dict[str, object]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ComparisonInputError("row_not_object")
                if row.get("type") == "environment":
                    environments.append(row)
                elif row.get("type") == "step":
                    key = _step_key(row)
                    if key in steps:
                        raise ComparisonInputError("duplicate_step")
                    steps[key] = row
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonInputError(f"unreadable_jsonl:{type(exc).__name__}") from exc
    if len(environments) != 1 or not steps:
        raise ComparisonInputError("missing_environment_or_steps")
    return environments[0], steps


def compare_runs(
    baseline: tuple[dict[str, object], dict[tuple[str, int, int], dict[str, object]]],
    candidate: tuple[dict[str, object], dict[tuple[str, int, int], dict[str, object]]],
) -> dict[str, object]:
    baseline_env, baseline_steps = baseline
    candidate_env, candidate_steps = candidate
    blockers: list[str] = []
    fingerprint = baseline_env.get("post_tool_final_reuse_comparison_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != candidate_env.get(
        "post_tool_final_reuse_comparison_fingerprint"
    ):
        blockers.append("fingerprint_mismatch")
    if baseline_env.get("server_pid") == candidate_env.get("server_pid"):
        blockers.append("process_not_isolated")
    if set(baseline_steps) != set(candidate_steps):
        blockers.append("step_set_mismatch")
    if _reuse_enabled(baseline_env) is not False or _reuse_enabled(candidate_env) is not True:
        blockers.append("mode_mismatch")

    rows: list[dict[str, object]] = []
    if not blockers:
        for key in sorted(baseline_steps):
            before = baseline_steps[key]
            after = candidate_steps[key]
            row_blockers: list[str] = []
            if before.get("tool_names") != after.get("tool_names"):
                row_blockers.append("tool_mismatch")
            if before.get("correctness_category") != after.get("correctness_category"):
                row_blockers.append("correctness_mismatch")
            if before.get("finish_reason") != after.get("finish_reason"):
                row_blockers.append("finish_reason_mismatch")
            reused = _reuse_delta(after) > 0
            before_calls = len(_model_steps(before))
            after_calls = len(_model_steps(after))
            if reused and before_calls - after_calls != 1:
                row_blockers.append("unexpected_model_call_delta")
            blockers.extend(row_blockers)
            rows.append(
                {
                    "case": key[0],
                    "step": key[1],
                    "repetition": key[2],
                    "reused": reused,
                    "model_calls_saved": before_calls - after_calls,
                    "evaluated_tokens_saved": _evaluated_tokens(before) - _evaluated_tokens(after),
                    "wall_ms_saved": _number(before.get("wall_ms")) - _number(after.get("wall_ms")),
                    "correctness_unchanged": before.get("correctness_category") == after.get("correctness_category"),
                    "finish_reason_unchanged": before.get("finish_reason") == after.get("finish_reason"),
                    "tool_unchanged": before.get("tool_names") == after.get("tool_names"),
                    "blockers": row_blockers,
                }
            )
    reused_rows = [row for row in rows if row["reused"]]
    return {
        "type": "post_tool_final_reuse_comparison",
        "decision": "pass" if not blockers and reused_rows else "fail",
        "fingerprint": fingerprint if isinstance(fingerprint, str) else None,
        "process_isolated": baseline_env.get("server_pid") != candidate_env.get("server_pid"),
        "steps": len(rows),
        "reused_steps": len(reused_rows),
        "model_calls_saved": sum(int(row["model_calls_saved"]) for row in reused_rows),
        "evaluated_tokens_saved": sum(int(row["evaluated_tokens_saved"]) for row in reused_rows),
        "wall_ms_saved": round(sum(float(row["wall_ms_saved"]) for row in reused_rows), 1),
        "rows": rows,
        "blockers": sorted(set(blockers or (["no_reused_step"] if not reused_rows else []))),
        "raw_content_included": False,
    }


def _step_key(row: dict[str, object]) -> tuple[str, int, int]:
    case = row.get("case")
    step = row.get("step")
    repetition = row.get("repetition")
    if not isinstance(case, str) or not isinstance(step, int):
        raise ComparisonInputError("invalid_step_identity")
    return case, step, repetition if isinstance(repetition, int) else 1


def _reuse_enabled(environment: dict[str, object]) -> bool | None:
    value = environment.get("post_tool_final_reuse")
    return value.get("enabled") if isinstance(value, dict) and isinstance(value.get("enabled"), bool) else None


def _reuse_delta(row: dict[str, object]) -> int:
    value = row.get("post_tool_final_reuse")
    delta = value.get("reused_count_delta") if isinstance(value, dict) else None
    return delta if isinstance(delta, int) else 0


def _model_steps(row: dict[str, object]) -> list[object]:
    value = row.get("model_steps")
    return value if isinstance(value, list) else []


def _evaluated_tokens(row: dict[str, object]) -> int:
    return sum(
        int(step.get("evaluated_tokens") or 0)
        for step in _model_steps(row)
        if isinstance(step, dict)
    )


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare process-isolated post-tool final reuse smoke runs.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_runs(load_run(args.baseline), load_run(args.candidate))
    except ComparisonInputError as exc:
        print(json.dumps({"type": "post_tool_final_reuse_comparison", "decision": "invalid", "error": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
