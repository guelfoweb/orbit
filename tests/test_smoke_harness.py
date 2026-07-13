from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
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

    def test_parser_accepts_managed_final_prefix_options(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            ["--manage-server", "--final-prefix-mode", "on", "--repetitions", "5"]
        )

        self.assertTrue(args.manage_server)
        self.assertEqual(args.final_prefix_mode, "on")
        self.assertEqual(args.repetitions, 5)

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

    def test_final_prefix_mixed_has_eleven_isolated_eligible_cases(self) -> None:
        scenario = smoke_harness.scenarios()["final_prefix_mixed"]

        self.assertEqual(len(scenario.steps), 11)
        self.assertTrue(scenario.isolated_steps)
        self.assertIn("system_info", scenario.allowed_tool_names)

    def test_correctness_categorizer_detects_shell_error_and_mixed_output(self) -> None:
        self.assertEqual(smoke_harness.check_shell_error("exit_code=127 command not found", []), "correct")
        self.assertEqual(smoke_harness.check_shell_error_focus("failed with 127", []), "correct")
        self.assertEqual(smoke_harness.check_shell_error_focus("failed with 127 and printed line-19", []), "mixed_wrong")

    def test_correctness_categorizer_detects_fake_shell20_without_tool_call(self) -> None:
        self.assertEqual(smoke_harness.check_shell20("line-19", []), "fake_tool_output")
        self.assertEqual(smoke_harness.check_shell20("line-19", ["exec_shell_full_command"]), "correct")

    def test_web_error_checker_rejects_answer_from_memory(self) -> None:
        self.assertEqual(smoke_harness.check_web_error("The web search failed.", []), "correct")
        self.assertEqual(
            smoke_harness.check_web_error("The search failed, but Avola is in Sicily.", []),
            "wrong",
        )

    def test_deterministic_web_fixture_covers_success_none_and_error(self) -> None:
        from orbit.runtime import shell_guardrails

        original = shell_guardrails.search_web
        with smoke_harness.deterministic_web(True):
            self.assertIn("Orbit deterministic fixture", shell_guardrails.search_web("orbit fixture success"))
            self.assertIn("results: none", shell_guardrails.search_web("orbit fixture none"))
            self.assertIn("error:", shell_guardrails.search_web("where is Avola located?"))
        self.assertIs(shell_guardrails.search_web, original)

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

    def test_environment_summary_records_final_prefix_and_runtime_metadata(self) -> None:
        class Backend:
            def model_info(self):
                return type("Info", (), {"id": "m"})()

        args = smoke_harness.build_parser().parse_args(
            [
                "--scenario", "final_prefix_local", "--final-prefix-mode", "on", "--timeout", "30",
                "--block-id", "abba-1-on-a", "--run-order", "ON", "--cooling-seconds", "15",
            ]
        )
        env = smoke_harness.environment_summary(
            args=args,
            backend=Backend(),
            props={
                "backend": "orbit-native",
                "ctx_size": 8192,
                "threads": 6,
                "threads_batch": 6,
                "batch_size": 256,
                "ubatch_size": 128,
                "final_prefix_experiment_enabled": True,
                "final_prefix_experiment_restore_count": 2,
            },
        )

        self.assertEqual(env["final_prefix_mode"], "on")
        self.assertEqual(env["scenario"], ["final_prefix_local"])
        self.assertEqual(env["ctx"], 8192)
        self.assertEqual(env["final_prefix"]["restore_count"], 2)
        self.assertEqual(env["timeout"], 30.0)
        self.assertEqual(env["block_id"], "abba-1-on-a")
        self.assertEqual(env["run_order"], "ON")
        self.assertEqual(env["cooling_seconds"], 15.0)
        self.assertIsInstance(env["cpu_affinity"], list)

    def test_final_prefix_environment_controls_client_flag_and_restores_it(self) -> None:
        with mock.patch.dict("os.environ", {"ORBIT_FINAL_PREFIX_EXPERIMENT": "old"}, clear=False):
            with smoke_harness.final_prefix_environment("on"):
                self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_EXPERIMENT"], "1")
            self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_EXPERIMENT"], "old")

            with smoke_harness.final_prefix_environment("off"):
                self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT", smoke_harness.os.environ)
            self.assertEqual(smoke_harness.os.environ["ORBIT_FINAL_PREFIX_EXPERIMENT"], "old")

    def test_managed_server_passes_final_prefix_flag_only_when_on(self) -> None:
        args = smoke_harness.build_parser().parse_args(["--manage-server", "--final-prefix-mode", "on"])
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(smoke_harness.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(smoke_harness, "wait_for_server"), \
            mock.patch.object(
                smoke_harness,
                "fresh_backend_props",
                side_effect=[{}, {"final_prefix_experiment_enabled": True}],
            ):
            with smoke_harness.managed_server(args):
                pass
        self.assertEqual(popen.call_args.kwargs["env"]["ORBIT_FINAL_PREFIX_EXPERIMENT"], "1")

        args = smoke_harness.build_parser().parse_args(["--manage-server", "--final-prefix-mode", "off"])
        with mock.patch.dict(smoke_harness.os.environ, {"ORBIT_FINAL_PREFIX_EXPERIMENT": "1"}), \
            mock.patch.object(smoke_harness.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(smoke_harness, "wait_for_server"), \
            mock.patch.object(
                smoke_harness,
                "fresh_backend_props",
                side_effect=[{}, {"final_prefix_experiment_enabled": False}],
            ):
            with smoke_harness.managed_server(args):
                pass
        self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT", popen.call_args.kwargs["env"])

    def test_managed_server_rejects_an_already_used_base_url(self) -> None:
        args = smoke_harness.build_parser().parse_args(["--manage-server", "--final-prefix-mode", "off"])
        with mock.patch.object(smoke_harness, "fresh_backend_props", return_value={"backend": "orbit-native"}), \
            self.assertRaisesRegex(RuntimeError, "unused base URL"):
            with smoke_harness.managed_server(args):
                pass

    def test_server_command_can_enable_mtp_without_changing_final_prefix_mode(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            ["--manage-server", "--final-prefix-mode", "on", "--server-mtp"]
        )

        command = smoke_harness.server_command(args)

        self.assertIn("--mtp", command)
        self.assertNotIn("ORBIT_FINAL_PREFIX_EXPERIMENT=1", command)

    def test_server_command_records_thinking_and_managed_tools_mode(self) -> None:
        args = smoke_harness.build_parser().parse_args(
            ["--manage-server", "--server-thinking", "on", "--tools", "off"]
        )
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(smoke_harness.subprocess, "Popen", return_value=process) as popen, \
            mock.patch.object(smoke_harness, "wait_for_server"), \
            mock.patch.object(
                smoke_harness,
                "fresh_backend_props",
                side_effect=[{}, {"final_prefix_experiment_enabled": False}],
            ):
            with smoke_harness.managed_server(args):
                pass

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--think") + 1], "on")
        self.assertEqual(popen.call_args.kwargs["env"]["ORBIT_TOOLS"], "off")

    def test_process_rss_reads_linux_status(self) -> None:
        process = mock.Mock(pid=123)
        with mock.patch.object(smoke_harness.Path, "read_text", return_value="Name:\torbit\nVmRSS:\t2048 kB\n"):
            self.assertEqual(smoke_harness.process_rss_kib(process), 2048)

    def test_final_prefix_step_state_reports_counter_deltas(self) -> None:
        before = {
            "final_prefix_experiment_capture_count": 1,
            "final_prefix_experiment_restore_count": 3,
            "final_prefix_experiment_fallback_count": 0,
        }
        after = {
            "final_prefix_experiment_enabled": True,
            "final_prefix_experiment_initialized": True,
            "final_prefix_experiment_prefix_tokens": 43,
            "final_prefix_experiment_capture_count": 1,
            "final_prefix_experiment_restore_count": 4,
            "final_prefix_experiment_fallback_count": 0,
        }

        state = smoke_harness.final_prefix_step_state(before, after)

        self.assertEqual(state["capture_count_delta"], 0)
        self.assertEqual(state["restore_count_delta"], 1)
        self.assertEqual(state["fallback_count_delta"], 0)
        self.assertEqual(state["prefix_tokens"], 43)

    def test_final_prefix_validation_encodes_off_capture_and_on_restore_contract(self) -> None:
        final = smoke_harness.StepReport(
            case="pwd_followup",
            step=1,
            prompt="run pwd",
            prompt_kind="auto",
            completion_kind="route,final_from_tool",
            route_tokens=800,
            final_tokens=100,
            prompt_tokens=100,
            cached_tokens=43,
            evaluated_tokens=57,
            finish_reason="stop",
            tool_calls=1,
            tool_names=["exec_shell_full_command"],
            wall_ms=100.0,
            correctness_category="correct",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="/tmp",
            model_steps=[],
        )
        off_props = {
            "final_prefix_experiment_enabled": False,
            "final_prefix_experiment_capture_count": 0,
        }
        on_props = {
            "final_prefix_experiment_enabled": True,
            "final_prefix_experiment_prefix_tokens": 43,
            "final_prefix_experiment_capture_count": 1,
            "final_prefix_experiment_restore_count": 1,
            "final_prefix_experiment_fallback_count": 0,
        }

        self.assertIsNone(smoke_harness.final_prefix_validation_failure("off", [final], off_props))
        self.assertIsNone(smoke_harness.final_prefix_validation_failure("on", [final, final], on_props))
        self.assertEqual(
            smoke_harness.final_prefix_validation_failure(
                "on",
                [final, final],
                {**on_props, "final_prefix_experiment_restore_count": 0},
            ),
            "restore_missing",
        )
        self.assertIsNone(
            smoke_harness.final_prefix_validation_failure(
                "on",
                [final],
                {
                    "mtp_experimental_enabled": True,
                    "final_prefix_experiment_enabled": True,
                    "final_prefix_experiment_capture_count": 0,
                    "final_prefix_experiment_restore_count": 0,
                },
            )
        )

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
        self.assertIn("output_tokens", rows[1])
        self.assertIn("phase_wall_ms", rows[1])
        self.assertEqual(rows[2]["type"], "summary")

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
        self.assertIn("## Repetition Summary", text)

    def test_summary_reports_median_and_final_prefix_counter_deltas(self) -> None:
        rows = []
        for cached, evaluated, wall, restore in (
            (4, 96, 20.0, 0),
            (43, 57, 14.0, 1),
            (43, 57, 15.0, 1),
            (43, 57, 13.0, 1),
            (43, 57, 16.0, 1),
        ):
            rows.append(
                smoke_harness.StepReport(
                    case="pwd",
                    step=1,
                    prompt="run pwd",
                    prompt_kind="auto",
                    completion_kind="route,final_from_tool",
                    route_tokens=800,
                    final_tokens=100,
                    prompt_tokens=100,
                    cached_tokens=cached,
                    evaluated_tokens=evaluated,
                    finish_reason="stop",
                    tool_calls=1,
                    tool_names=["exec_shell_full_command"],
                    wall_ms=wall,
                    correctness_category="correct",
                    raw_leak=False,
                    fake_output=False,
                    loop=False,
                    notes="",
                    answer_excerpt="/tmp",
                    model_steps=[],
                    final_prefix={"restore_count_delta": restore},
                )
            )

        summary = smoke_harness.summarize_reports(rows)[0]

        self.assertEqual(summary["cached_tokens"]["median"], 43.0)
        self.assertEqual(summary["evaluated_tokens"]["median"], 57.0)
        self.assertEqual(summary["runs"], 5)
        self.assertEqual(summary["restore_delta"], 4)
        self.assertEqual(summary["restored"]["runs"], 4)
        self.assertEqual(summary["restored"]["cached_tokens"]["min"], 43.0)

    def test_summary_aggregates_fifty_stability_calls(self) -> None:
        row = smoke_harness.StepReport(
            case="final_prefix_mixed",
            step=1,
            prompt="run pwd",
            prompt_kind="auto",
            completion_kind="route,final_from_tool",
            route_tokens=780,
            final_tokens=100,
            prompt_tokens=100,
            cached_tokens=43,
            evaluated_tokens=57,
            finish_reason="stop",
            tool_calls=1,
            tool_names=["exec_shell_full_command"],
            wall_ms=12.0,
            correctness_category="correct",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="",
            answer_excerpt="/tmp",
            model_steps=[],
            final_prefix={"restore_count_delta": 1, "fallback_count_delta": 0},
        )

        summary = smoke_harness.summarize_reports([row] * 50)[0]

        self.assertEqual(summary["runs"], 50)
        self.assertEqual(summary["correct"], 50)
        self.assertEqual(summary["restore_delta"], 50)
        self.assertEqual(summary["fallback_delta"], 0)

    def test_phase_timing_summary_separates_model_and_non_model_wall(self) -> None:
        timing = smoke_harness.phase_timing_summary(
            [("route", 10.0), ("final_from_tool", 20.0), ("final_from_tool", 5.0)],
            40.0,
        )

        self.assertEqual(timing["route"], 10.0)
        self.assertEqual(timing["final_from_tool"], 25.0)
        self.assertEqual(timing["non_model_wall_ms"], 5.0)
        self.assertEqual(smoke_harness.estimated_generation_ms(10, 5.0), 2000.0)

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

    def test_run_step_timeout_requests_cancel_and_returns_timeout_report(self) -> None:
        runtime = mock.Mock()

        def slow_ask_chat(*args, **kwargs):
            time.sleep(0.2)
            raise AssertionError("worker should have been abandoned after timeout")

        runtime.ask_chat.side_effect = slow_ask_chat
        with mock.patch.object(smoke_harness, "request_backend_cancel", return_value=True) as mocked_cancel, \
            mock.patch.object(smoke_harness, "wait_for_backend_idle", return_value={"in_flight": False}):
            report = smoke_harness.run_step(
                runtime,
                scenario="simple_chat",
                step_index=1,
                step=smoke_harness.SmokeStep("hi", mode="chat", checker_name="nonempty"),
                workdir=Path("workdir"),
                max_tokens=16,
                temperature=0.0,
                base_url="http://127.0.0.1:12120",
                timeout=0.01,
            )

        mocked_cancel.assert_called_once()
        self.assertEqual(report.finish_reason, "timeout")
        self.assertEqual(report.completion_kind, "timeout")
        self.assertIn("cancel_requested", report.notes)
        self.assertIn("cleanup_ok", report.notes)
        self.assertTrue(report.lifecycle["timeout_observed"])
        self.assertFalse(report.lifecycle["automatic_cancel"])
        self.assertTrue(report.lifecycle["explicit_cancel_used"])
        self.assertTrue(report.lifecycle["cleanup_healthy"])

    def test_main_returns_timeout_failure_when_report_times_out(self) -> None:
        timeout_report = smoke_harness.StepReport(
            case="shell20",
            step=1,
            prompt="run python3 -c 'for i in range(20): print(f\"line-{i}\")'",
            prompt_kind="auto",
            completion_kind="timeout",
            route_tokens=None,
            final_tokens=None,
            prompt_tokens=None,
            cached_tokens=None,
            evaluated_tokens=None,
            finish_reason="timeout",
            tool_calls=0,
            tool_names=[],
            wall_ms=60001.0,
            correctness_category="not_evaluated",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="timeout,cancel_requested,cleanup_ok",
            answer_excerpt="timeout",
            model_steps=[],
        )
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value={}), \
            mock.patch.object(smoke_harness, "fresh_backend_props", return_value={}), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[timeout_report]), \
            mock.patch.object(smoke_harness, "write_jsonl"), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "shell20", "--no-web", "--output-dir", tmp, "--timeout", "60"])

        self.assertEqual(rc, 1)

    def test_wait_for_backend_idle_returns_when_inflight_clears(self) -> None:
        with mock.patch.object(
            smoke_harness,
            "fresh_backend_props",
            side_effect=[{"in_flight": True}, {"in_flight": False, "mtp_enabled": True}],
        ):
            props = smoke_harness.wait_for_backend_idle("http://127.0.0.1:12120", 1.0, poll_interval=0.0)

        self.assertEqual(props.get("in_flight"), False)

    def test_settled_backend_props_waits_past_cancelled_snapshot(self) -> None:
        with mock.patch.object(
            smoke_harness,
            "fresh_backend_props",
            side_effect=[
                {"in_flight": False, "mtp_fallback_reason": "cancelled", "mtp_failure_reason": None},
                {"in_flight": False, "mtp_fallback_reason": None, "mtp_failure_reason": None, "mtp_initialized": True},
            ],
        ):
            props = smoke_harness.settled_backend_props("http://127.0.0.1:12120", 1.0, settle_seconds=0.5, poll_interval=0.0)

        self.assertIsNone(props.get("mtp_fallback_reason"))
        self.assertTrue(props.get("mtp_initialized"))

    def test_run_scenario_stops_after_timeout_report(self) -> None:
        scenario = smoke_harness.SmokeScenario(
            "shell20",
            (
                smoke_harness.SmokeStep("first"),
                smoke_harness.SmokeStep("second"),
            ),
        )
        timeout_report = smoke_harness.StepReport(
            case="shell20",
            step=1,
            prompt="first",
            prompt_kind="auto",
            completion_kind="timeout",
            route_tokens=None,
            final_tokens=None,
            prompt_tokens=None,
            cached_tokens=None,
            evaluated_tokens=None,
            finish_reason="timeout",
            tool_calls=0,
            tool_names=[],
            wall_ms=60000.0,
            correctness_category="not_evaluated",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes="timeout",
            answer_excerpt="timeout",
            model_steps=[],
        )
        with mock.patch.object(smoke_harness, "run_step", return_value=timeout_report) as mocked_run_step, \
            mock.patch.object(smoke_harness, "ChatRuntime"):
            reports = smoke_harness.run_scenario(
                scenario,
                backend=mock.Mock(base_url="http://127.0.0.1:12120"),
                workdir=Path("workdir"),
                max_tokens=128,
                temperature=0.0,
                timeout=60.0,
            )

        self.assertEqual(len(reports), 1)
        mocked_run_step.assert_called_once()

    def test_main_uses_settled_props_for_environment_row(self) -> None:
        initial = {
            "backend": "orbit-native",
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": False,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        settled = {
            "backend": "orbit-native",
            "model_id": "m",
            "multimodal_available": True,
            "mtp_experimental_enabled": True,
            "mtp_initialized": True,
            "mtp_last_completion_success": True,
            "mtp_failure_reason": None,
            "mtp_fallback_reason": None,
        }
        captured_env: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(smoke_harness, "safe_backend_props", return_value=initial), \
            mock.patch.object(smoke_harness, "settled_backend_props", return_value=settled), \
            mock.patch.object(smoke_harness, "run_scenario", return_value=[]), \
            mock.patch.object(smoke_harness, "write_jsonl", side_effect=lambda _p, env, _r: captured_env.append(env)), \
            mock.patch.object(smoke_harness, "write_markdown"):
            rc = smoke_harness.main(["--scenario", "simple_chat", "--no-web", "--output-dir", tmp, "--mtp-required"])

        self.assertEqual(rc, 0)
        self.assertEqual(captured_env[0]["mtp"], "on")
        self.assertTrue(captured_env[0]["mtp_usable"])


if __name__ == "__main__":
    unittest.main()
