from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from orbit.runtime.turn_trace import ModelStepMetrics


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "orbit_smoke_harness.py"
SPEC = importlib.util.spec_from_file_location("orbit_smoke_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_harness = importlib.util.module_from_spec(SPEC)
sys.modules["orbit_smoke_harness"] = smoke_harness
SPEC.loader.exec_module(smoke_harness)


class SmokeHarnessTests(unittest.TestCase):
    def test_parser_accepts_output_and_selection_options(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            [
                "--base-url",
                "http://127.0.0.1:12120",
                "--scenario",
                "pwd_followup",
                "--no-web",
                "--jsonl",
                "/tmp/out.jsonl",
                "--markdown",
                "/tmp/out.md",
            ]
        )

        self.assertEqual(args.scenario, ["pwd_followup"])
        self.assertTrue(args.no_web)
        self.assertEqual(args.jsonl, "/tmp/out.jsonl")
        self.assertEqual(args.markdown, "/tmp/out.md")

    def test_select_scenarios_defaults_to_all_without_optional_or_web_when_disabled(self) -> None:
        selected = smoke_harness.select_scenarios(None, no_web=True, include_optional=False)
        names = [scenario.name for scenario in selected]

        self.assertIn("simple_chat", names)
        self.assertIn("dual_shell", names)
        self.assertNotIn("web_shell", names)
        self.assertNotIn("grep_read", names)

    def test_select_scenarios_allows_single_scenario_without_implicit_all(self) -> None:
        selected = smoke_harness.select_scenarios(["pwd_followup"], no_web=False, include_optional=False)

        self.assertEqual([scenario.name for scenario in selected], ["pwd_followup"])

    def test_correctness_categorizer_detects_shell_error_and_mixed_output(self) -> None:
        self.assertEqual(smoke_harness.check_shell_error("exit_code=127 command not found", []), "correct")
        self.assertEqual(smoke_harness.check_shell_error_focus("failed with 127", []), "correct")
        self.assertEqual(smoke_harness.check_shell_error_focus("failed with 127 and printed line-19", []), "mixed_wrong")

    def test_correctness_categorizer_detects_fake_shell20_without_tool_call(self) -> None:
        self.assertEqual(smoke_harness.check_shell20("line-19", []), "fake_tool_output")
        self.assertEqual(smoke_harness.check_shell20("line-19", ["exec_shell_full_command"]), "correct")

    def test_jsonl_output_contains_environment_and_step_rows(self) -> None:
        report = smoke_harness.StepReport(
            case="simple_chat",
            step=1,
            prompt="hi",
            prompt_kind="chat",
            completion_kind="final",
            route_tokens=None,
            final_tokens=12,
            prompt_tokens=12,
            cached_tokens=4,
            evaluated_tokens=8,
            finish_reason="stop",
            tool_calls=0,
            tool_names=[],
            wall_ms=10.0,
            correctness_category="correct",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="hello",
            model_steps=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.jsonl"
            smoke_harness.write_jsonl(path, {"version": "0.0.1", "git_head": "abc"}, [report])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["type"], "environment")
        self.assertEqual(rows[1]["type"], "step")
        self.assertEqual(rows[1]["case"], "simple_chat")

    def test_markdown_summary_contains_expected_columns(self) -> None:
        report = smoke_harness.StepReport(
            case="pwd_followup",
            step=2,
            prompt="what directory was that?",
            prompt_kind="auto",
            completion_kind="route,final",
            route_tokens=700,
            final_tokens=300,
            prompt_tokens=300,
            cached_tokens=4,
            evaluated_tokens=296,
            finish_reason="stop",
            tool_calls=0,
            tool_names=[],
            wall_ms=123.0,
            correctness_category="partial_baseline",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="/tmp",
            model_steps=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            smoke_harness.write_markdown(path, {"version": "0.0.1", "git_head": "abc", "backend": "orbit-native"}, [report])
            text = path.read_text(encoding="utf-8")

        self.assertIn("| Case | Step | Kind | Route | Final |", text)
        self.assertIn("pwd_followup", text)
        self.assertIn("partial_baseline", text)

    def test_model_step_to_json_includes_evaluated_tokens(self) -> None:
        row = smoke_harness.model_step_to_json(
            ModelStepMetrics(
                loop=1,
                phase="final",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=5,
                cached_tokens=40,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
                tool_calls=0,
            )
        )

        self.assertEqual(row["evaluated_tokens"], 60)


if __name__ == "__main__":
    unittest.main()
