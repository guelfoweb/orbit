from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "compare_post_tool_final_reuse.py"
SPEC = importlib.util.spec_from_file_location("compare_post_tool_final_reuse", PATH)
assert SPEC is not None and SPEC.loader is not None
comparator = importlib.util.module_from_spec(SPEC)
sys.modules["compare_post_tool_final_reuse"] = comparator
SPEC.loader.exec_module(comparator)


def _run(*, enabled: bool, pid: int, calls: int, reused: int):
    environment = {
        "post_tool_final_reuse_comparison_fingerprint": "a" * 64,
        "post_tool_final_reuse": {"enabled": enabled},
        "server_pid": pid,
    }
    step = {
        "case": "fixture",
        "step": 1,
        "repetition": 1,
        "tool_names": ["exec_shell_full_command"],
        "correctness_category": "correct",
        "finish_reason": "stop",
        "wall_ms": float(calls * 100),
        "model_steps": [{"evaluated_tokens": 10} for _ in range(calls)],
        "post_tool_final_reuse": {"reused_count_delta": reused},
    }
    return environment, {("fixture", 1, 1): step}


class PostToolFinalReuseComparatorTests(unittest.TestCase):
    def test_process_isolated_pair_reports_bounded_savings(self) -> None:
        result = comparator.compare_runs(
            _run(enabled=False, pid=10, calls=3, reused=0),
            _run(enabled=True, pid=11, calls=2, reused=1),
        )
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["model_calls_saved"], 1)
        self.assertEqual(result["evaluated_tokens_saved"], 10)
        self.assertFalse(result["raw_content_included"])

    def test_rejects_same_process_or_fingerprint_mismatch(self) -> None:
        baseline = _run(enabled=False, pid=10, calls=3, reused=0)
        candidate = _run(enabled=True, pid=10, calls=2, reused=1)
        self.assertEqual(comparator.compare_runs(baseline, candidate)["decision"], "fail")
        candidate[0]["server_pid"] = 11
        candidate[0]["post_tool_final_reuse_comparison_fingerprint"] = "b" * 64
        self.assertEqual(comparator.compare_runs(baseline, candidate)["decision"], "fail")


if __name__ == "__main__":
    unittest.main()
