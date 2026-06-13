from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.cli import build_parser
from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT, TOOL_CALL_SYSTEM_PROMPT
from orbit.terminal.config import DEFAULT_SYSTEM_PROMPT, load_app_config
from orbit.terminal.tool_mode import allowed_tool_names_for_spec


class ConfigTests(unittest.TestCase):
    def test_default_system_prompt_allows_attached_media_answers(self) -> None:
        self.assertIn("Answer normally unless shell is needed", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Environment: OS=", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Use curl for URLs", DEFAULT_SYSTEM_PROMPT)

    def test_command_system_prompt_sends_local_hardware_queries_to_shell(self) -> None:
        self.assertIn('{"command":"..."}', ROUTE_SYSTEM_PROMPT)
        self.assertIn("Return valid one-line JSON only", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Use native commands", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Environment: OS=", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Use curl for URLs", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Quote spaced paths", ROUTE_SYSTEM_PROMPT)
        self.assertIn("not metadata", ROUTE_SYSTEM_PROMPT)

    def test_tool_call_prompt_mentions_quoted_shell_paths(self) -> None:
        self.assertIn("Call exec_shell_full_command exactly once", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("one-line shell command", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Use curl for URLs when content is needed", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Quote paths containing spaces", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("collect direct evidence", TOOL_CALL_SYSTEM_PROMPT)

    def test_command_prompt_mentions_shell_full_path_quoting(self) -> None:
        self.assertIn("Quote spaced paths", ROUTE_SYSTEM_PROMPT)
        self.assertIn("files/edit/create/append/delete", ROUTE_SYSTEM_PROMPT)
        self.assertIn("source, binaries, strings, logs, archives", ROUTE_SYSTEM_PROMPT)

    def test_missing_config_uses_defaults(self) -> None:
        args = _parse("--config", "/tmp/orbit-missing-config.json")

        config = load_app_config(args)

        self.assertEqual(config.base_url, "http://127.0.0.1:18080")
        self.assertEqual(config.workdir, Path(".").resolve())
        self.assertEqual(config.max_tokens, 512)
        self.assertIsNone(config.context_tokens)
        self.assertEqual(config.tools, "off")

    def test_config_file_sets_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "base_url": "http://127.0.0.1:19090",
                        "workdir": str(Path(tmp)),
                        "timeout": 120,
                        "temperature": 0,
                        "max_tokens": 512,
                        "context_tokens": 4096,
                        "no_system": True,
                        "tools": "on",
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(_parse("--config", str(path)))

        self.assertEqual(config.base_url, "http://127.0.0.1:19090")
        self.assertEqual(config.workdir, Path(tmp).resolve())
        self.assertEqual(config.timeout, 120.0)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.context_tokens, 4096)
        self.assertTrue(config.no_system)
        self.assertEqual(config.tools, "on")

    def test_cli_flags_override_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"workdir": "/tmp", "max_tokens": 128}), encoding="utf-8")

            config = load_app_config(
                _parse(
                    "--config",
                    str(path),
                    "--workdir",
                    str(ROOT),
                    "--max-tokens",
                    "64",
                    "--context-tokens",
                    "2048",
                    "--tools",
                    "on",
                )
            )

        self.assertEqual(config.workdir, ROOT.resolve())
        self.assertEqual(config.max_tokens, 64)
        self.assertEqual(config.context_tokens, 2048)
        self.assertEqual(config.tools, "on")

    def test_config_rejects_unknown_tool_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "tools"):
            load_app_config(_parse("--tools", "browser"))

    def test_tools_on_uses_shell_full_only(self) -> None:
        names = allowed_tool_names_for_spec("on")

        self.assertEqual(names, ("exec_shell_full_command",))

    def test_legacy_tools_object_config_key_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"tools": {"max_loops": 10}, "workdir": str(ROOT)}), encoding="utf-8")

            config = load_app_config(_parse("--config", str(path)))

        self.assertEqual(config.tools, "off")

    def test_legacy_model_config_key_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"model": "legacy", "workdir": str(ROOT)}), encoding="utf-8")

            config = load_app_config(_parse("--config", str(path)))

        self.assertEqual(config.workdir, ROOT.resolve())

    def test_config_rejects_operational_values_outside_safe_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"timeout": 0, "max_tokens": 4, "context_tokens": 128}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "timeout"):
                load_app_config(_parse("--config", str(path)))

    def test_cli_flags_reject_operational_values_outside_safe_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            load_app_config(_parse("--max-tokens", "4"))


def _parse(*args: str) -> argparse.Namespace:
    return build_parser().parse_args(list(args))


if __name__ == "__main__":
    unittest.main()
