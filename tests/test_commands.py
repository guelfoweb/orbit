from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ModelInfo
from orbit.runtime import ChatRuntime
from orbit.terminal.commands import help_text, runtime_status, set_max_tokens, tools_text
from orbit.terminal.config import AppConfig


class CommandTests(unittest.TestCase):
    def test_help_mentions_max_tokens(self) -> None:
        self.assertIn("/max-tokens <n>", help_text())

    def test_set_max_tokens_without_value_reports_current_value(self) -> None:
        config = AppConfig(max_tokens=512)

        updated, message = set_max_tokens(config, "")

        self.assertEqual(updated.max_tokens, 512)
        self.assertEqual(message, "max_tokens: 512")

    def test_set_max_tokens_updates_runtime_config(self) -> None:
        config = AppConfig(max_tokens=512)

        updated, message = set_max_tokens(config, "2048")

        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(updated.max_tokens, 2048)
        self.assertEqual(message, "max_tokens: 2048")

    def test_set_max_tokens_rejects_invalid_values(self) -> None:
        config = AppConfig(max_tokens=512)

        updated, message = set_max_tokens(config, "99999")

        self.assertEqual(updated.max_tokens, 512)
        self.assertIn("between 32 and 4096", message)

    def test_runtime_status_shows_tool_backends(self) -> None:
        backend = FakeStatusBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        status = runtime_status(runtime, AppConfig(model="gemma4:12b"), backend)

        self.assertIn("tools_llama_server: grep_search, read_file", status)
        self.assertIn("tools_orbit_only:", status)
        self.assertNotIn("tools_orbit:", status)

    def test_tools_text_hides_orbit_duplicates_when_server_tools_exist(self) -> None:
        output = tools_text(FakeFullToolBackend())

        self.assertIn("llama-server: apply_diff, edit_file, exec_shell_command, read_file, write_file", output)
        self.assertIn("orbit-only:", output)
        self.assertIn("fetch_url", output)
        self.assertIn("search_web", output)
        self.assertIn("list_files", output)
        self.assertNotIn("stat_path", output)
        self.assertNotIn("append_file", output)
        self.assertNotIn("replace_in_file", output)


class FakeStatusBackend:
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="gemma4:12b",
            capabilities=("completion",),
            context_length=8192,
            parameter_count=None,
            size_bytes=None,
        )

    def display_model_name(self) -> str:
        return "gemma4:12b"

    def health(self) -> bool:
        return True

    def server_tools(self):
        return [{"tool": "grep_search"}, {"tool": "read_file"}]


class FakeFullToolBackend(FakeStatusBackend):
    def server_tools(self):
        return [
            {"tool": "read_file"},
            {"tool": "write_file"},
            {"tool": "exec_shell_command"},
            {"tool": "edit_file"},
            {"tool": "apply_diff"},
        ]


if __name__ == "__main__":
    unittest.main()
