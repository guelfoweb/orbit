from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.runtime.turn_trace import ModelStepMetrics


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "orbit_smoke_harness.py"
SPEC = importlib.util.spec_from_file_location("orbit_smoke_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_harness = importlib.util.module_from_spec(SPEC)
sys.modules["orbit_smoke_harness"] = smoke_harness
SPEC.loader.exec_module(smoke_harness)


class SmokeHarnessTests(unittest.TestCase):
    def _step_report(self, *, notes: str = "", answer_excerpt: str = "", finish_reason: str = "stop") -> smoke_harness.StepReport:
        return smoke_harness.StepReport(
            case="simple_chat",
            step=1,
            prompt="hi",
            prompt_kind="chat",
            completion_kind="chat_final",
            route_tokens=None,
            final_tokens=12,
            prompt_tokens=12,
            cached_tokens=4,
            evaluated_tokens=8,
            finish_reason=finish_reason,
            tool_calls=0,
            tool_names=[],
            wall_ms=10.0,
            correctness_category="correct",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes=notes,
            answer_excerpt=answer_excerpt,
            model_steps=[],
        )

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

    def test_mtp_state_marks_failed_props_as_not_usable(self) -> None:
        state = smoke_harness.mtp_state_from_props(
            {
                "mtp_experimental_enabled": True,
                "mtp_initialized": False,
                "mtp_last_completion_success": False,
                "mtp_failure_reason": "failed to decode target prefill suffix",
            }
        )

        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["usable"])
        self.assertEqual(state["failure_reason"], "failed to decode target prefill suffix")

    def test_mtp_state_marks_session_ready_without_attempt_as_ready(self) -> None:
        state = smoke_harness.mtp_state_from_props(
            {
                "mtp_experimental_enabled": True,
                "mtp_initialized": True,
                "mtp_last_completion_success": False,
                "mtp_failure_reason": None,
                "mtp_fallback_reason": None,
            }
        )

        self.assertEqual(state["status"], "ready")
        self.assertFalse(state["usable"])

    def test_environment_summary_uses_mtp_state_fields(self) -> None:
        class Backend:
            def model_info(self):
                return type("Info", (), {"id": "m"})()

        args = smoke_harness.build_parser().parse_args([])
        env = smoke_harness.environment_summary(
            args=args,
            backend=Backend(),
            props={
                "backend": "orbit-native",
                "model_id": "m",
                "multimodal_available": True,
                "mtp_experimental_enabled": True,
                "mtp_initialized": False,
                "mtp_last_completion_success": False,
                "mtp_failure_reason": "probe failed",
            },
        )

        self.assertEqual(env["mtp"], "failed")
        self.assertFalse(env["mtp_usable"])
        self.assertEqual(env["mtp_failure_reason"], "probe failed")

    def test_main_mtp_required_fails_when_post_run_props_not_usable(self) -> None:
        initial = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        final = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": False,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": "failed to decode target prefill suffix",
            "mtp_fallback_reason": "failed to decode target prefill suffix",
        }
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=initial), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value=final), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 2)

    def test_main_mtp_required_passes_when_post_run_props_usable(self) -> None:
        props = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": True,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=props), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value=props), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 0)

    def test_main_mtp_required_uses_settled_final_props_over_stale_cancelled_state(self) -> None:
        initial = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        stale = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": "cancelled",
            "mtp_fallback_reason": "cancelled",
        }
        healthy = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": True,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        captured_env: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=initial), \
            mock.patch.object(smoke_harness, "fresh_backend_props", side_effect=[stale, healthy, healthy]), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[]), \
            mock.patch.object(smoke_harness, "write_jsonl", side_effect=lambda _p, env, _r: captured_env.append(env)), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 0)
        self.assertTrue(captured_env)
        self.assertEqual(captured_env[0]["mtp"], "on")
        self.assertTrue(captured_env[0]["mtp_usable"])

    def test_main_timeout_with_healthy_final_props_fails_as_timeout_not_mtp(self) -> None:
        initial = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        healthy = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": True,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        report = self._step_report(
            notes="exception",
            answer_excerpt="LlamaServerError: backend server request timed out after 60s",
            finish_reason="error",
        )
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=initial), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value=healthy), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[report]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "shell20", "--no-web", "--output-dir", tmp])

        self.assertEqual(rc, 1)

    def test_main_correct_scenario_with_failed_final_props_still_fails_mtp_required(self) -> None:
        initial = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        failed = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": False,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": "failed to decode target prefill suffix",
            "mtp_fallback_reason": "failed to decode target prefill suffix",
        }
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=initial), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value=failed), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[self._step_report()]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 2)

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
