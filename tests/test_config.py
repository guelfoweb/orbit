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
        self.assertIn("Attached image/audio => answer normally", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Never emit raw tool-call syntax", DEFAULT_SYSTEM_PROMPT)
        self.assertIn('{"_route":"FILESYSTEM","tool":"<tool>"}', DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Never answer file contents from memory", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Common args: path, pattern, command, url, query, content.", DEFAULT_SYSTEM_PROMPT)

    def test_route_system_prompt_routes_local_hardware_queries_to_shell(self) -> None:
        self.assertIn("follow-up questions about previous answers or tool results", ROUTE_SYSTEM_PROMPT)
        self.assertIn("specs/specifications/configuration", ROUTE_SYSTEM_PROMPT)
        self.assertIn("this/local PC hardware or resources", ROUTE_SYSTEM_PROMPT)
        self.assertIn("do not ask for photos, brand, or model", ROUTE_SYSTEM_PROMPT)
        self.assertIn("FILESYSTEM/exec_shell_command", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Current date/time requests use FILESYSTEM/get_datetime", ROUTE_SYSTEM_PROMPT)
        self.assertIn("choose enough allowed read-only commands", ROUTE_SYSTEM_PROMPT)
        self.assertIn("short && chain", ROUTE_SYSTEM_PROMPT)
        self.assertIn("For line counts use wc -l file", ROUTE_SYSTEM_PROMPT)
        self.assertIn("For file_glob_search, use one simple glob only", ROUTE_SYSTEM_PROMPT)
        self.assertIn("For content-based edits without explicit line numbers", ROUTE_SYSTEM_PROMPT)
        self.assertIn("For shell-full analysis requests", ROUTE_SYSTEM_PROMPT)
        self.assertIn("file_glob_search: one simple glob only; no brace expansion", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Content-based edits without explicit line numbers need read_file and edit_file", DEFAULT_SYSTEM_PROMPT)

    def test_tool_call_prompt_mentions_quoted_shell_paths(self) -> None:
        self.assertIn("strings -a samples/suspicious_dropper_demo.js", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Available tools have already been enabled by the user", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Tool arguments must be valid compact JSON", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("no literal newlines inside string values", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("use one single-line command string only", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("for line counts use wc -l file", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("For external tools, verify availability with command -v", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("For edit_file append at end", TOOL_CALL_SYSTEM_PROMPT)

    def test_route_prompt_mentions_shell_full_path_quoting(self) -> None:
        self.assertIn("any path containing whitespace MUST be one double-quoted shell argument", ROUTE_SYSTEM_PROMPT)
        self.assertIn("strings -a samples/suspicious_dropper_demo.js", ROUTE_SYSTEM_PROMPT)
        self.assertIn("use one single-line command string only", ROUTE_SYSTEM_PROMPT)
        self.assertIn("For external tools, verify availability with command -v", ROUTE_SYSTEM_PROMPT)

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
                        "tools": "files,web",
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
        self.assertEqual(config.tools, "files,web")

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
                    "web",
                )
            )

        self.assertEqual(config.workdir, ROOT.resolve())
        self.assertEqual(config.max_tokens, 64)
        self.assertEqual(config.context_tokens, 2048)
        self.assertEqual(config.tools, "web")

    def test_config_rejects_unknown_tool_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "tools"):
            load_app_config(_parse("--tools", "browser"))

    def test_shell_tool_group_includes_datetime(self) -> None:
        self.assertEqual(allowed_tool_names_for_spec("shell"), ("exec_shell_command", "get_datetime"))

    def test_tools_on_does_not_include_shell_full(self) -> None:
        names = allowed_tool_names_for_spec("on")

        self.assertIsNotNone(names)
        self.assertIn("exec_shell_command", names)
        self.assertNotIn("exec_shell_full_command", names)

    def test_shell_full_is_explicit_tool_group(self) -> None:
        self.assertEqual(allowed_tool_names_for_spec("shell-full"), ("exec_shell_full_command",))

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
