from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from orbit.runtime.tool_plan_shadow import analyze_tool_plan_shadow
from orbit.runtime.tools import TOOL_NAMES, tool_definitions
from orbit.tool_plan_config import resolve_tool_plan_shadow


def _step(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"name": name, "arguments": arguments}


def _plan(*steps: dict[str, object]) -> dict[str, object]:
    return {"type": "tool_plan", "steps": list(steps)}


class ToolPlanShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)
        self.definitions = tool_definitions(TOOL_NAMES)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def analyze(self, payload: object, *, finish_reason: str = "stop"):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return analyze_tool_plan_shadow(
            text,
            finish_reason=finish_reason,
            tool_definitions=self.definitions,
            allowed_tool_names=TOOL_NAMES,
            workdir=self.workdir,
            user_prompt="inspect synthetic local state",
        )

    def test_config_is_off_by_default_and_invalid_values_disable(self) -> None:
        self.assertFalse(resolve_tool_plan_shadow({}).enabled)
        self.assertTrue(resolve_tool_plan_shadow({"ORBIT_TOOL_PLAN_SHADOW": "1"}).enabled)
        self.assertFalse(resolve_tool_plan_shadow({"ORBIT_TOOL_PLAN_SHADOW": "0"}).enabled)
        invalid = resolve_tool_plan_shadow({"ORBIT_TOOL_PLAN_SHADOW": "yes"})
        self.assertFalse(invalid.enabled)
        self.assertEqual(invalid.validation_error, "invalid_boolean")

    def test_exact_two_step_plan_passes_and_runtime_ids_are_internal(self) -> None:
        report = self.analyze(
            _plan(
                _step("system_info", {}),
                _step("list_directory", {"path": ".", "max_entries": 20}),
            )
        )

        self.assertTrue(report.valid)
        self.assertEqual(report.response_kind, "plan")
        self.assertEqual([step.id for step in report.plan.steps], ["step_1", "step_2"])  # type: ignore[union-attr]
        diagnostic = report.diagnostic()
        self.assertEqual(diagnostic["step_count"], 2)
        self.assertFalse(diagnostic["raw_content_included"])
        self.assertNotIn("arguments", diagnostic)

    def test_exact_unsupported_response_passes(self) -> None:
        report = self.analyze({"type": "unsupported_plan"})

        self.assertTrue(report.valid)
        self.assertEqual(report.response_kind, "unsupported")
        self.assertIsNone(report.plan)

    def test_only_exact_two_step_shape_is_accepted(self) -> None:
        one = _plan(_step("system_info", {}))
        three = _plan(_step("system_info", {}), _step("system_info", {}), _step("system_info", {}))
        old_shape = {
            "type": "tool_plan",
            "steps": [
                {"id": "step_1", "tool": "system_info", "arguments": {}, "expect": None},
                {"id": "step_2", "tool": "system_info", "arguments": {}, "expect": None},
            ],
            "completion": "none",
        }

        self.assertEqual(self.analyze(one).rejection_code, "invalid_step_count")
        self.assertEqual(self.analyze(three).rejection_code, "invalid_step_count")
        self.assertEqual(self.analyze(old_shape).rejection_code, "invalid_plan_shape")

    def test_unsupported_shape_is_exact(self) -> None:
        report = self.analyze({"type": "unsupported_plan", "reason": "shell"})
        self.assertEqual(report.rejection_code, "invalid_unsupported_shape")

    def test_shell_network_and_unknown_tools_are_not_eligible(self) -> None:
        for name, arguments in (
            ("exec_shell_full_command", {"command": "pwd"}),
            ("fetch_url", {"url": "https://synthetic.invalid"}),
            ("missing_tool", {}),
        ):
            with self.subTest(name=name):
                report = self.analyze(_plan(_step(name, arguments), _step("system_info", {})))
                self.assertEqual(report.rejection_code, "tool_not_plan_eligible")

    def test_dynamic_arguments_fail_closed(self) -> None:
        report = self.analyze(
            _plan(
                _step("list_directory", {"path": "."}),
                _step("list_directory", {"path": "${step_1.output}"}),
            )
        )
        self.assertEqual(report.rejection_code, "dynamic_dependency")

    def test_canonical_schema_permission_and_limits_are_authoritative(self) -> None:
        cases = (
            ({"path": ".", "extra": True}, "canonical_additional_property"),
            ({"path": 7}, "canonical_type_mismatch"),
            ({"path": ".", "max_entries": 100000}, "canonical_limit_out_of_range"),
        )
        for arguments, reason in cases:
            with self.subTest(reason=reason):
                report = self.analyze(
                    _plan(_step("list_directory", arguments), _step("system_info", {}))
                )
                self.assertEqual(report.rejection_code, reason)
        disabled = analyze_tool_plan_shadow(
            json.dumps(_plan(_step("system_info", {}), _step("system_info", {}))),
            finish_reason="stop",
            tool_definitions=self.definitions,
            allowed_tool_names=("list_directory",),
            workdir=self.workdir,
            user_prompt="inspect",
        )
        self.assertEqual(disabled.rejection_code, "canonical_tool_not_enabled")

    def test_duplicate_multiple_prose_invalid_and_finish_fail_closed(self) -> None:
        duplicate = (
            '{"type":"tool_plan","steps":[{"name":"system_info","arguments":{},"arguments":{}},'
            '{"name":"system_info","arguments":{}}]}'
        )
        multiple = json.dumps({"type": "unsupported_plan"}) + json.dumps({"type": "unsupported_plan"})
        prose = "Result: " + json.dumps({"type": "unsupported_plan"})
        invalid = '{"type":"unsupported_plan"'
        plan = _plan(_step("system_info", {}), _step("system_info", {}))

        self.assertEqual(self.analyze(duplicate).rejection_code, "duplicate_key")
        self.assertEqual(self.analyze(multiple).rejection_code, "multiple_candidates")
        self.assertEqual(self.analyze(prose).rejection_code, "external_text")
        self.assertEqual(self.analyze(invalid).rejection_code, "invalid_json")
        self.assertEqual(self.analyze(plan, finish_reason="length").rejection_code, "nonrecoverable_length")
        self.assertEqual(self.analyze(plan, finish_reason="cancelled").rejection_code, "nonrecoverable_cancelled")

    def test_normal_text_is_not_a_plan(self) -> None:
        report = self.analyze("Explain system_info and list_directory.")
        self.assertFalse(report.detected)
        self.assertEqual(report.candidate_count, 0)
        self.assertEqual(report.rejection_code, "no_plan")
        self.assertTrue(report.diagnostic()["prose_leakage"])

    def test_analysis_cannot_execute_a_tool(self) -> None:
        with mock.patch("orbit.runtime.tool_backends.HybridToolExecutor.execute") as execute:
            report = self.analyze(
                _plan(_step("system_info", {}), _step("list_directory", {"path": "."}))
            )
        self.assertTrue(report.valid)
        execute.assert_not_called()

    def test_validation_is_deterministic_for_many_literal_plans(self) -> None:
        for index in range(100):
            payload = _plan(
                _step("list_directory", {"path": f"alpha-{index}", "max_entries": 5}),
                _step("list_directory", {"path": f"beta-{index}", "max_entries": 7}),
            )
            first = self.analyze(payload)
            second = self.analyze(payload)
            self.assertTrue(first.valid)
            self.assertEqual(first.diagnostic(), second.diagnostic())


if __name__ == "__main__":
    unittest.main()
