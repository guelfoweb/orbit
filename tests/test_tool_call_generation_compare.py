from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_tool_call_generation.py"
SPEC = importlib.util.spec_from_file_location("compare_tool_call_generation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
sys.modules["compare_tool_call_generation"] = comparison
SPEC.loader.exec_module(comparison)


def _environment(*, corpus_hash: str = "a" * 64, commit: str = "base") -> dict[str, object]:
    return {
        "type": "environment",
        "benchmark_mode": "tool_call_generation_only",
        "corpus_version": 1,
        "corpus_hash": corpus_hash,
        "protocol_version": 1,
        "protocol_hash": "1" * 64,
        "configuration_version": 1,
        "configuration_hash": "2" * 64,
        "model": "gemma4-test",
        "mtp": "off",
        "native_backend_capabilities": {
            "profile_id": "orbit-gemma4-native-v1",
            "status": "verified",
            "backend_build_number": 278,
            "backend_commit": commit,
            "backend_library_hash": "b" * 64,
            "tool_protocol_text_hash": "c" * 64,
            "final_prefix_text_hash": "e" * 64,
            "renderer_fixture_suite_hash": "f" * 64,
            "tokenizer_prefix_hash": "d" * 64,
        },
    }


def _sample(*, scenario: str = "system_info_1", outcome: str = "expected_tool") -> dict[str, object]:
    return {
        "type": "tool_call_generation",
        "scenario": scenario,
        "repetition": 1,
        "expected_tool": "system_info" if outcome != "no_attempt" else None,
        "template": "production_tool_call_system_prompt",
        "evaluable": True,
        "semantic_outcome": outcome,
        "multiple_candidates": False,
        "markup_leakage": False,
        "tool_executed": False,
        "finalization_started": False,
        "model_calls": 1,
    }


def _summary() -> dict[str, object]:
    return {
        "type": "tool_call_generation_summary",
        "samples": 1,
        "completion_rate": 1.0,
        "valid_first_pass_rate": 1.0,
        "exact_tool_match_rate": 1.0,
        "model_unwanted_attempt_rate": 0.0,
        "detector_false_positive_rate": 0.0,
        "detector_false_negative_rate": 0.0,
        "multiple_candidate_rate": 0.0,
        "budget_truncation_rate": 0.0,
        "structural_truncation_rate": 0.0,
        "markup_leakage_rate": 0.0,
        "generation_wall_ms_median": 10.0,
        "generation_wall_ms_p95": 12.0,
        "healing_us_median": 50.0,
        "healing_us_p95": 60.0,
        "model_calls": 1,
        "tools_executed": 0,
        "finalizations_started": 0,
    }


def _write(path: Path, environment: dict[str, object], sample: dict[str, object], summary: dict[str, object]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in (environment, sample, summary)) + "\n",
        encoding="utf-8",
    )


class ToolCallGenerationComparisonTests(unittest.TestCase):
    def _run_cli(self, baseline: Path, candidate: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(baseline), str(candidate)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_equivalent_runs_pass_and_report_backend_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(commit="old"), _sample(), _summary())
            _write(candidate_path, _environment(commit="new"), _sample(), _summary())

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertEqual(result["decision"], "pass")
        self.assertTrue(result["identity_changes"]["backend_commit"])
        self.assertEqual(result["blockers"], [])
        self.assertFalse(result["raw_content_included"])

    def test_corpus_mismatch_is_incomparable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, _environment(corpus_hash="e" * 64), _sample(), _summary())

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertEqual(result["decision"], "incomparable")
        self.assertIn("corpus_mismatch", result["blockers"])

    def test_protocol_or_configuration_mismatch_is_incomparable(self) -> None:
        candidate_environment = _environment()
        candidate_environment["configuration_hash"] = "3" * 64
        candidate_environment["protocol_hash"] = "4" * 64
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, candidate_environment, _sample(), _summary())

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertEqual(result["decision"], "incomparable")
        self.assertIn("configuration_mismatch", result["blockers"])
        self.assertIn("protocol_mismatch", result["blockers"])

    def test_backend_unverified_is_comparable_but_tokenizer_mismatch_is_not(self) -> None:
        unverified_environment = _environment(commit="new")
        unverified_environment["native_backend_capabilities"]["status"] = "backend_unverified"
        mismatch_environment = _environment(commit="new")
        mismatch_environment["native_backend_capabilities"]["status"] = "tokenizer_mismatch"
        unknown_environment = _environment(commit="new")
        unknown_environment["native_backend_capabilities"]["status"] = "future_status"
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            unverified_path = Path(tmp) / "unverified.jsonl"
            mismatch_path = Path(tmp) / "mismatch.jsonl"
            unknown_path = Path(tmp) / "unknown.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(unverified_path, unverified_environment, _sample(), _summary())
            _write(mismatch_path, mismatch_environment, _sample(), _summary())
            _write(unknown_path, unknown_environment, _sample(), _summary())
            baseline = comparison.load_generation_run(baseline_path)

            unverified = comparison.compare_generation_runs(baseline, comparison.load_generation_run(unverified_path))
            mismatch = comparison.compare_generation_runs(baseline, comparison.load_generation_run(mismatch_path))
            unknown = comparison.compare_generation_runs(baseline, comparison.load_generation_run(unknown_path))

        self.assertEqual(unverified["decision"], "pass")
        self.assertEqual(mismatch["decision"], "incomparable")
        self.assertIn("candidate_renderer_or_tokenizer_mismatch", mismatch["blockers"])
        self.assertEqual(unknown["decision"], "incomparable")

    def test_declared_verified_capability_contract_drift_is_incomparable(self) -> None:
        candidate_environment = _environment()
        candidate_environment["native_backend_capabilities"]["tokenizer_prefix_hash"] = "f" * 64
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, candidate_environment, _sample(), _summary())

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertEqual(result["decision"], "incomparable")
        self.assertIn("capability_contract_mismatch", result["blockers"])

    def test_semantic_and_markup_regressions_fail(self) -> None:
        candidate_sample = _sample(outcome="wrong_tool")
        candidate_sample["markup_leakage"] = True
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, _environment(), candidate_sample, _summary())

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertEqual(result["decision"], "fail")
        self.assertIn("tool_selection_regression", result["blockers"])
        self.assertIn("markup_leakage_regression", result["blockers"])

    def test_tool_execution_or_extra_model_call_fail(self) -> None:
        candidate_sample = _sample()
        candidate_sample["tool_executed"] = True
        candidate_sample["model_calls"] = 2
        candidate_summary = _summary()
        candidate_summary["tools_executed"] = 1
        candidate_summary["model_calls"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, _environment(), candidate_sample, candidate_summary)

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertEqual(result["decision"], "fail")
        self.assertIn("candidate_executed_tool", result["blockers"])
        self.assertIn("candidate_model_call_count", result["blockers"])
        self.assertIn("sample_model_call_count", result["blockers"])

    def test_loader_rejects_duplicate_samples_without_echoing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in (_environment(), _sample(), _sample(), _summary())) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(comparison.ComparisonInputError, "duplicate_sample"):
                comparison.load_generation_run(path)

    def test_malformed_candidate_counts_fail_closed(self) -> None:
        candidate_summary = _summary()
        candidate_summary["model_calls"] = True
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, _environment(), _sample(), candidate_summary)

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertEqual(result["decision"], "fail")
        self.assertIn("candidate_model_call_count", result["blockers"])

    def test_boolean_metrics_are_not_reported_as_numeric_deltas(self) -> None:
        candidate_summary = _summary()
        candidate_summary["completion_rate"] = True
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, _environment(), _sample(), candidate_summary)

            result = comparison.compare_generation_runs(
                comparison.load_generation_run(baseline_path),
                comparison.load_generation_run(candidate_path),
            )

        self.assertIsNone(result["metric_deltas"]["completion_rate"])

    def test_cli_exit_codes_distinguish_pass_fail_and_incomparable(self) -> None:
        wrong_tool = _sample(outcome="wrong_tool")
        mismatched_environment = _environment()
        mismatched_environment["configuration_hash"] = "9" * 64
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            passing_path = Path(tmp) / "passing.jsonl"
            failing_path = Path(tmp) / "failing.jsonl"
            incomparable_path = Path(tmp) / "incomparable.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(passing_path, _environment(commit="new"), _sample(), _summary())
            _write(failing_path, _environment(), wrong_tool, _summary())
            _write(incomparable_path, mismatched_environment, _sample(), _summary())

            passing = self._run_cli(baseline_path, passing_path)
            failing = self._run_cli(baseline_path, failing_path)
            incomparable = self._run_cli(baseline_path, incomparable_path)

        self.assertEqual(passing.returncode, 0)
        self.assertEqual(passing.stderr, "")
        self.assertEqual(json.loads(passing.stdout)["decision"], "pass")
        self.assertEqual(failing.returncode, 1)
        self.assertEqual(failing.stderr, "")
        self.assertEqual(json.loads(failing.stdout)["decision"], "fail")
        self.assertEqual(incomparable.returncode, 2)
        self.assertEqual(incomparable.stderr, "")
        self.assertEqual(json.loads(incomparable.stdout)["decision"], "incomparable")

    def test_cli_invalid_input_is_bounded_and_content_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            invalid_path = Path(tmp) / "invalid.jsonl"
            valid_path = Path(tmp) / "valid.jsonl"
            invalid_path.write_text('{"secret":"must-not-leak"', encoding="utf-8")
            _write(valid_path, _environment(), _sample(), _summary())

            result = self._run_cli(invalid_path, valid_path)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["decision"], "invalid")
        self.assertNotIn("must-not-leak", result.stdout)

    def test_cli_failure_does_not_echo_unrecognized_sensitive_fields(self) -> None:
        candidate_sample = _sample(outcome="wrong_tool")
        candidate_sample.update(
            {
                "raw_output": "secret-output",
                "arguments": {"path": "/private/example", "token": "secret-token"},
                "evidence": "secret-evidence",
                "url": "https://secret.invalid/query",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.jsonl"
            candidate_path = Path(tmp) / "candidate.jsonl"
            _write(baseline_path, _environment(), _sample(), _summary())
            _write(candidate_path, _environment(), candidate_sample, _summary())

            result = self._run_cli(baseline_path, candidate_path)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "")
        for sensitive in ("secret-output", "/private/example", "secret-token", "secret-evidence", "secret.invalid"):
            self.assertNotIn(sensitive, result.stdout)


if __name__ == "__main__":
    unittest.main()
