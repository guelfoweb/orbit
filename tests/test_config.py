from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.cli import build_parser
from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT, TOOL_CALL_SYSTEM_PROMPT
from orbit.runtime.session_memory import estimate_text_tokens
from orbit.terminal.config import DEFAULT_SYSTEM_PROMPT, load_app_config
from orbit.terminal.tool_mode import allowed_tool_names_for_spec


class ConfigTests(unittest.TestCase):
    def test_default_system_prompt_allows_attached_media_answers(self) -> None:
        self.assertIn("Decide compactly whether the user request needs local tools", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("files/read/edit/create/append/delete", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("do not answer directly or return CHAT", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Web/search/latest/current/online and URL fetch/read/open/explain/summarize/analyze requests are tool tasks", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("return a compact tool decision, not a direct answer", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("File read/explain/summarize/analyze requests require file content evidence", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Use directory listing only when the user asks to list files", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("one-sentence direct-answer exception", DEFAULT_SYSTEM_PROMPT)
        self.assertIn('{"command":"cat README.md"}', DEFAULT_SYSTEM_PROMPT)
        self.assertIn('{"command":"orbit-web-search \\"query\\""}', DEFAULT_SYSTEM_PROMPT)
        self.assertIn('{"url":"https://example.com"}', DEFAULT_SYSTEM_PROMPT)
        self.assertIn("no external evidence is needed", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("fits in one short sentence", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("needing explanation, a list, a paragraph", DEFAULT_SYSTEM_PROMPT)
        self.assertIn('{"route":"CHAT"}', DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Do not write long prose in the route pass", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Environment: OS=", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("orbit-web-search", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("prefer the fetch_url tool", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("system_info", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("curl are still allowed", DEFAULT_SYSTEM_PROMPT)

    def test_command_system_prompt_sends_local_hardware_queries_to_shell(self) -> None:
        self.assertIn('{"command":"..."}', ROUTE_SYSTEM_PROMPT)
        self.assertIn("Return valid one-line JSON only for route/tool decisions", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Tool tasks: files/read/edit/create/append/delete", ROUTE_SYSTEM_PROMPT)
        self.assertIn("do not answer directly or return CHAT", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Web/search/latest/current/online and URL fetch/read/open/explain/summarize/analyze requests are tool tasks", ROUTE_SYSTEM_PROMPT)
        self.assertIn("return a compact tool decision, not a direct answer", ROUTE_SYSTEM_PROMPT)
        self.assertIn("File read/explain/summarize/analyze requests require file content evidence", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Use directory listing only when the user asks to list files", ROUTE_SYSTEM_PROMPT)
        self.assertIn("one-sentence direct-answer exception", ROUTE_SYSTEM_PROMPT)
        self.assertIn('{"command":"cat README.md"}', ROUTE_SYSTEM_PROMPT)
        self.assertIn('{"command":"orbit-web-search \\"query\\""}', ROUTE_SYSTEM_PROMPT)
        self.assertIn('{"url":"https://example.com"}', ROUTE_SYSTEM_PROMPT)
        self.assertIn("no external evidence is needed", ROUTE_SYSTEM_PROMPT)
        self.assertIn("fits in one short sentence", ROUTE_SYSTEM_PROMPT)
        self.assertIn("needing explanation, a list, a paragraph", ROUTE_SYSTEM_PROMPT)
        self.assertIn('{"route":"CHAT"}', ROUTE_SYSTEM_PROMPT)
        self.assertIn("Do not write long prose in the route pass", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Use native commands", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Environment: OS=", ROUTE_SYSTEM_PROMPT)
        self.assertIn("orbit-web-search", ROUTE_SYSTEM_PROMPT)
        self.assertIn("prefer the fetch_url tool", ROUTE_SYSTEM_PROMPT)
        self.assertIn("system_info", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Quote spaced paths", ROUTE_SYSTEM_PROMPT)
        self.assertIn("not metadata", ROUTE_SYSTEM_PROMPT)

    def test_chat_system_prompt_is_short_and_non_operational(self) -> None:
        self.assertLess(estimate_text_tokens(CHAT_SYSTEM_PROMPT), 30)
        self.assertIn("Answer normally", CHAT_SYSTEM_PROMPT)
        self.assertNotIn('{"command"', CHAT_SYSTEM_PROMPT)
        self.assertNotIn("orbit-web-search", CHAT_SYSTEM_PROMPT)

    def test_tool_call_prompt_mentions_quoted_shell_paths(self) -> None:
        self.assertIn("Call exactly one available tool", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Prefer fetch_url", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Prefer system_info", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("orbit-web-search", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("exec_shell_full_command", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Quote paths containing spaces", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("collect direct evidence", TOOL_CALL_SYSTEM_PROMPT)

    def test_command_prompt_mentions_shell_full_path_quoting(self) -> None:
        self.assertIn("Quote spaced paths", ROUTE_SYSTEM_PROMPT)
        self.assertIn("files/read/edit/create/append/delete", ROUTE_SYSTEM_PROMPT)
        self.assertIn("source, binaries, strings, logs, archives", ROUTE_SYSTEM_PROMPT)

    def test_missing_config_uses_defaults(self) -> None:
        args = _parse("--config", "/tmp/orbit-missing-config.json")

        config = load_app_config(args)

        self.assertEqual(config.base_url, "http://127.0.0.1:12120")
        self.assertEqual(config.workdir, Path(".").resolve())
        self.assertEqual(config.max_tokens, 512)
        self.assertIsNone(config.context_tokens)
        self.assertEqual(config.tools, "off")
        self.assertEqual(config.render_markdown, "live")

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
                        "think": True,
                        "tools": "on",
                        "render_markdown": "live",
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
        self.assertTrue(config.think)
        self.assertEqual(config.tools, "on")
        self.assertEqual(config.render_markdown, "live")

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
                    "--think",
                    "on",
                    "--tools",
                    "on",
                    "--render-markdown-live",
                )
            )

        self.assertEqual(config.workdir, ROOT.resolve())
        self.assertEqual(config.max_tokens, 64)
        self.assertEqual(config.context_tokens, 2048)
        self.assertTrue(config.think)
        self.assertEqual(config.tools, "on")
        self.assertEqual(config.render_markdown, "live")

    def test_env_can_enable_live_markdown_rendering(self) -> None:
        with mock.patch.dict(os.environ, {"ORBIT_RENDER_MARKDOWN": "live"}):
            config = load_app_config(_parse("--config", "/tmp/orbit-missing-config.json"))

        self.assertEqual(config.render_markdown, "live")

    def test_no_render_markdown_disables_default_live_rendering(self) -> None:
        config = load_app_config(_parse("--config", "/tmp/orbit-missing-config.json", "--no-render-markdown"))

        self.assertEqual(config.render_markdown, "plain")

    def test_cli_has_precedence_over_markdown_render_env(self) -> None:
        with mock.patch.dict(os.environ, {"ORBIT_RENDER_MARKDOWN": "off"}):
            enabled = load_app_config(_parse("--config", "/tmp/orbit-missing-config.json", "--render-markdown-live"))
        self.assertEqual(enabled.render_markdown, "live")

        with mock.patch.dict(os.environ, {"ORBIT_RENDER_MARKDOWN": "live"}):
            disabled = load_app_config(_parse("--config", "/tmp/orbit-missing-config.json", "--no-render-markdown"))
        self.assertEqual(disabled.render_markdown, "plain")

    def test_env_can_disable_markdown_rendering(self) -> None:
        for value in ("0", "false", "off", "plain"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"ORBIT_RENDER_MARKDOWN": value}):
                    config = load_app_config(_parse("--config", "/tmp/orbit-missing-config.json"))

                self.assertEqual(config.render_markdown, "plain")

    def test_cli_can_enable_live_markdown_rendering(self) -> None:
        config = load_app_config(_parse("--config", "/tmp/orbit-missing-config.json", "--render-markdown-live"))

        self.assertEqual(config.render_markdown, "live")

    def test_no_render_markdown_forces_plain(self) -> None:
        with mock.patch.dict(os.environ, {"ORBIT_RENDER_MARKDOWN": "live"}):
            config = load_app_config(_parse("--config", "/tmp/orbit-missing-config.json", "--no-render-markdown"))

        self.assertEqual(config.render_markdown, "plain")

    def test_config_rejects_unknown_tool_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "tools"):
            load_app_config(_parse("--tools", "browser"))

    def test_tools_on_uses_shell_fetch_url_list_directory_and_system_info(self) -> None:
        names = allowed_tool_names_for_spec("on")

        self.assertEqual(names, ("exec_shell_full_command", "fetch_url", "list_directory", "system_info"))

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
