from __future__ import annotations

import unittest

from orbit.final_prefix_config import resolve_final_prefix_reuse


class FinalPrefixReuseConfigTests(unittest.TestCase):
    def test_precedence_matrix(self) -> None:
        cases = (
            ({}, True, "default", False, None),
            ({"ORBIT_FINAL_PREFIX_EXPERIMENT": "0"}, False, "legacy", True, None),
            ({"ORBIT_FINAL_PREFIX_EXPERIMENT": "1"}, True, "legacy", True, None),
            ({"ORBIT_FINAL_PREFIX_REUSE": "0"}, False, "stable", False, None),
            ({"ORBIT_FINAL_PREFIX_REUSE": "1"}, True, "stable", False, None),
            (
                {"ORBIT_FINAL_PREFIX_REUSE": "0", "ORBIT_FINAL_PREFIX_EXPERIMENT": "1"},
                False,
                "stable",
                True,
                None,
            ),
            (
                {"ORBIT_FINAL_PREFIX_REUSE": "1", "ORBIT_FINAL_PREFIX_EXPERIMENT": "0"},
                True,
                "stable",
                True,
                None,
            ),
            (
                {"ORBIT_FINAL_PREFIX_REUSE": "invalid", "ORBIT_FINAL_PREFIX_EXPERIMENT": "1"},
                False,
                "stable",
                True,
                "invalid_stable_value",
            ),
        )
        for env, enabled, source, legacy_detected, error in cases:
            with self.subTest(env=env):
                result = resolve_final_prefix_reuse(env)
                self.assertIs(result.enabled, enabled)
                self.assertEqual(result.source, source)
                self.assertIs(result.legacy_detected, legacy_detected)
                self.assertEqual(result.validation_error, error)

    def test_invalid_legacy_value_preserves_disabled_behavior_with_diagnostic(self) -> None:
        result = resolve_final_prefix_reuse({"ORBIT_FINAL_PREFIX_EXPERIMENT": "old"})

        self.assertFalse(result.enabled)
        self.assertEqual(result.source, "legacy")
        self.assertEqual(result.raw_value, "old")
        self.assertEqual(result.validation_error, "invalid_legacy_value")

    def test_unbounded_or_control_raw_values_are_not_returned(self) -> None:
        for value in ("x" * 17, "bad\nvalue", "inv\N{SNOWMAN}lid"):
            with self.subTest(value=value):
                result = resolve_final_prefix_reuse({"ORBIT_FINAL_PREFIX_REUSE": value})
                self.assertFalse(result.enabled)
                self.assertIsNone(result.raw_value)
                self.assertEqual(result.validation_error, "invalid_stable_value")


if __name__ == "__main__":
    unittest.main()
