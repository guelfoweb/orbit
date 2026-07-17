from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "orbit_smoke_harness.py"
SPEC = importlib.util.spec_from_file_location("orbit_tool_plan_smoke_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
sys.modules["orbit_tool_plan_smoke_harness"] = harness
SPEC.loader.exec_module(harness)


EXPECTED_STEPS = (
    ("system_info", {}),
    ("list_directory", {"path": ".", "max_entries": 20}),
)


def _valid_plan() -> str:
    return json.dumps(
        {
            "type": "tool_plan",
            "steps": [
                {"name": name, "arguments": arguments}
                for name, arguments in EXPECTED_STEPS
            ],
        }
    )


class ToolPlanHarnessTests(unittest.TestCase):
    def test_cli_shadow_requires_centralized_opt_in(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), \
            mock.patch.object(harness, "run_tool_plan_shadow_benchmark") as run:
            self.assertEqual(harness.main(["--tool-plan-shadow"]), 2)
        run.assert_not_called()

        with mock.patch.dict("os.environ", {"ORBIT_TOOL_PLAN_SHADOW": "invalid"}, clear=True), \
            mock.patch.object(harness, "run_tool_plan_shadow_benchmark") as run:
            self.assertEqual(harness.main(["--tool-plan-shadow"]), 2)
        run.assert_not_called()

        with mock.patch.dict("os.environ", {"ORBIT_TOOL_PLAN_SHADOW": "1"}, clear=True), \
            mock.patch.object(harness, "run_tool_plan_shadow_benchmark", return_value=0) as run:
            self.assertEqual(harness.main(["--tool-plan-shadow"]), 0)
        run.assert_called_once()

    def test_exact_plan_uses_one_model_call_without_tool_or_finalization(self) -> None:
        class Backend:
            base_url = "http://127.0.0.1:9"

            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                self.tools = tools
                return harness.ChatResult(
                    content=_valid_plan(),
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=100,
                    completion_tokens=40,
                    cached_tokens=4,
                    prompt_tokens_per_second=100.0,
                    generation_tokens_per_second=10.0,
                )

        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(harness, "wait_for_backend_idle", return_value={"in_flight": False}), \
            mock.patch("orbit.runtime.tool_calls.execute_tool_call") as execute, \
            mock.patch.object(harness.ChatRuntime, "_answer_from_tool_results") as finalize:
            row = harness.run_tool_plan_shadow_sample(
                backend,
                base_url=backend.base_url,
                timeout=1.0,
                workdir=Path(tmp),
                scenario="fixture",
                group="positive_mixed",
                prompt="sensitive synthetic request",
                prompt_view="contract",
                system_prompt=harness.TOOL_PLAN_SHADOW_PROMPT_VIEWS["contract"],
                expected_kind="plan",
                expected_steps=EXPECTED_STEPS,
                repetition=1,
                temperature=0,
                max_tokens=128,
            )

        self.assertEqual(backend.calls, 1)
        self.assertIsNone(backend.tools)
        self.assertEqual(row["model_calls"], 1)
        self.assertTrue(row["exact_plan"])
        self.assertTrue(row["exact_tool_sequence"])
        self.assertTrue(row["exact_arguments"])
        self.assertEqual(row["scenario_outcome"], "exact_plan")
        self.assertEqual(row["potential_model_call_reduction"], 2)
        self.assertEqual(row["realized_model_call_reduction"], 0)
        self.assertFalse(row["tool_executed"])
        self.assertFalse(row["finalization_started"])
        execute.assert_not_called()
        finalize.assert_not_called()

    def test_unsupported_response_is_scored_separately(self) -> None:
        class Backend:
            base_url = "http://127.0.0.1:9"

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                return harness.ChatResult(
                    content='{"type":"unsupported_plan"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=80,
                    completion_tokens=7,
                    cached_tokens=4,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(harness, "wait_for_backend_idle", return_value={"in_flight": False}):
            row = harness.run_tool_plan_shadow_sample(
                Backend(),
                base_url="http://127.0.0.1:9",
                timeout=1.0,
                workdir=Path(tmp),
                scenario="unsupported",
                group="negative",
                prompt="fixture",
                prompt_view="contract",
                system_prompt=harness.TOOL_PLAN_SHADOW_PROMPT_VIEWS["contract"],
                expected_kind="unsupported",
                expected_steps=None,
                repetition=1,
                temperature=0,
                max_tokens=128,
            )

        self.assertTrue(row["correct_unsupported"])
        self.assertEqual(row["scenario_outcome"], "correct_unsupported")
        self.assertFalse(row["wrong_plan"])

    def test_timeout_cancels_and_never_executes(self) -> None:
        release = threading.Event()

        class Backend:
            base_url = "http://127.0.0.1:9"

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                release.wait(1.0)
                return harness.ChatResult(
                    content="",
                    model="fake",
                    finish_reason="cancelled",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=0,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        def cancel(*_args, **_kwargs):
            release.set()
            return True

        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(harness, "request_backend_cancel", side_effect=cancel), \
            mock.patch.object(harness, "wait_for_backend_idle", return_value={"in_flight": False}), \
            mock.patch("orbit.runtime.tool_calls.execute_tool_call") as execute:
            row = harness.run_tool_plan_shadow_sample(
                Backend(),
                base_url="http://127.0.0.1:9",
                timeout=0.01,
                workdir=Path(tmp),
                scenario="timeout",
                group="negative",
                prompt="fixture",
                prompt_view="contract",
                system_prompt=harness.TOOL_PLAN_SHADOW_PROMPT_VIEWS["contract"],
                expected_kind="unsupported",
                expected_steps=None,
                repetition=1,
                temperature=0,
                max_tokens=128,
            )

        self.assertTrue(row["timeout"])
        self.assertTrue(row["cancel_requested"])
        self.assertTrue(row["cleanup_healthy"])
        self.assertFalse(row["evaluable"])
        execute.assert_not_called()

    def test_native_backend_tool_call_is_rejected(self) -> None:
        class Backend:
            base_url = "http://127.0.0.1:9"

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                return harness.ChatResult(
                    content=_valid_plan(),
                    model="fake",
                    finish_reason="tool_call",
                    tool_calls=[{"function": {"name": "system_info", "arguments": {}}}],
                    prompt_tokens=100,
                    completion_tokens=40,
                    cached_tokens=4,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(harness, "wait_for_backend_idle", return_value={"in_flight": False}), \
            mock.patch("orbit.runtime.tool_calls.execute_tool_call") as execute:
            row = harness.run_tool_plan_shadow_sample(
                Backend(),
                base_url="http://127.0.0.1:9",
                timeout=1.0,
                workdir=Path(tmp),
                scenario="native-call",
                group="positive_mixed",
                prompt="fixture",
                prompt_view="contract",
                system_prompt=harness.TOOL_PLAN_SHADOW_PROMPT_VIEWS["contract"],
                expected_kind="plan",
                expected_steps=EXPECTED_STEPS,
                repetition=1,
                temperature=0,
                max_tokens=128,
            )

        self.assertFalse(row["valid"])
        self.assertEqual(row["rejection_code"], "unexpected_backend_tool_call")
        self.assertTrue(row["unexpected_backend_tool_calls"])
        execute.assert_not_called()

    def test_view_metrics_and_wrong_plan_credit(self) -> None:
        exact = {
            "prompt_view": "contract", "evaluable": True, "expected_kind": "plan",
            "json_compliant": True, "exact_plan": True, "correct_unsupported": False,
            "wrong_plan": False, "prose_leakage": False, "invalid_json": False,
            "model_calls": 1, "potential_model_call_reduction": 2,
        }
        unsupported = {
            "prompt_view": "contract", "evaluable": True, "expected_kind": "unsupported",
            "json_compliant": True, "exact_plan": False, "correct_unsupported": True,
            "wrong_plan": False, "prose_leakage": False, "invalid_json": False,
            "model_calls": 1, "potential_model_call_reduction": 0,
        }
        wrong = {
            "prompt_view": "json_only", "evaluable": True, "expected_kind": "unsupported",
            "json_compliant": True, "exact_plan": False, "correct_unsupported": False,
            "wrong_plan": True, "prose_leakage": False, "invalid_json": False,
            "model_calls": 1, "potential_model_call_reduction": 0,
        }
        summary = harness.summarize_tool_plan_shadow([exact, unsupported, wrong])

        self.assertEqual(summary["exact_plans"], 1)
        self.assertEqual(summary["correct_unsupported"], 1)
        self.assertEqual(summary["wrong_plans"], 1)
        self.assertEqual(summary["potential_model_call_reduction"], 2)
        self.assertEqual(summary["realized_model_call_reduction"], 0)
        self.assertTrue(summary["views"]["contract"]["smoke_gate_pass"])
        self.assertFalse(summary["views"]["json_only"]["smoke_gate_pass"])

    def test_jsonl_is_bounded_and_content_free(self) -> None:
        row = {
            "type": "tool_plan_shadow", "scenario": "safe-id", "prompt_view": "contract",
            "evaluable": True, "expected_kind": "unsupported", "json_compliant": True,
            "exact_plan": False, "correct_unsupported": True, "wrong_plan": False,
            "prose_leakage": False, "invalid_json": False, "model_calls": 1,
            "tool_executed": False, "finalization_started": False,
        }
        summary = harness.summarize_tool_plan_shadow([row])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.jsonl"
            harness.write_tool_plan_shadow_jsonl(
                path, {"type": "environment", "workdir": "<redacted>"}, [row], summary
            )
            text = path.read_text(encoding="utf-8")

        self.assertNotIn('"prompt":', text)
        self.assertNotIn('"arguments":', text)
        self.assertNotIn('"raw_content":', text)

    def test_corpus_has_three_groups_and_three_incremental_views(self) -> None:
        groups = {item[1] for item in harness.TOOL_PLAN_SHADOW_CORPUS}
        self.assertEqual(groups, {"positive_mixed", "positive_two_lists", "negative"})
        self.assertEqual(tuple(harness.TOOL_PLAN_SHADOW_PROMPT_VIEWS), ("contract", "json_only", "exactness"))
        self.assertGreaterEqual(len(harness.TOOL_PLAN_SHADOW_CORPUS), 9)
        self.assertEqual(len(harness.tool_plan_shadow_corpus(smoke=True)), 4)


if __name__ == "__main__":
    unittest.main()
