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
from orbit.runtime.session_memory import MemoryRefresh
from orbit.terminal.commands import help_text, runtime_status, set_max_tokens, tools_text
from orbit.terminal.config import AppConfig


class CommandTests(unittest.TestCase):
    def test_help_mentions_max_tokens(self) -> None:
        self.assertIn("/compact [tools]", help_text())
        self.assertIn("/max-tokens [n]", help_text())
        self.assertIn("/continue", help_text())
        self.assertIn("/sessions clear", help_text())
        self.assertIn("/status [ctx]", help_text())
        self.assertIn("/tools [spec]", help_text())

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

        status = runtime_status(runtime, AppConfig(), backend)

        self.assertIn("Backend\n-------", status)
        self.assertIn("Runtime\n-------", status)
        self.assertIn("Tools\n-------", status)
        self.assertIn("Memory\n-------", status)
        self.assertIn("Model\n-------", status)
        self.assertIn("tools_mode: n/a", status)
        self.assertIn("tools_llama_server: grep_search, read_file", status)
        self.assertIn("tools_orbit_only:", status)
        self.assertIn("memory_refresh_threshold: 6963/8192", status)
        self.assertIn("memory_refreshes: 0", status)
        self.assertIn("last_refresh_outcome: none", status)
        self.assertNotIn("tools_orbit:", status)

    def test_runtime_status_shows_memory_refresh_observability(self) -> None:
        backend = FakeStatusBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, context_tokens=1000)
        runtime.messages = [{"role": "user", "content": "hello"}]
        runtime.last_memory_refresh = MemoryRefresh(
            changed=True,
            reason="memory-refreshed",
            estimated_tokens_before=900,
            estimated_tokens_after=300,
            context_tokens=1000,
            threshold_tokens=850,
        )
        runtime.last_memory_refresh_attempt = runtime.last_memory_refresh
        runtime.last_memory_refresh_message_count = len(runtime.messages)
        runtime.memory_refreshes = 2
        runtime.total_memory_tokens_saved = 1200

        status = runtime_status(runtime, AppConfig(), backend)

        self.assertIn("memory_refresh_threshold: 850/1000", status)
        self.assertIn("memory_refreshes: 2", status)
        self.assertIn("last_refresh_before: 900", status)
        self.assertIn("last_refresh_after: 300", status)
        self.assertIn("last_refresh_saved: 600", status)
        self.assertIn("total_tokens_saved: 1200", status)
        self.assertIn("memory_cooldown: active (4 message(s) remaining)", status)
        self.assertIn("last_refresh_outcome: success (memory-refreshed)", status)

    def test_runtime_status_shows_discarded_memory_refresh_attempt(self) -> None:
        backend = FakeStatusBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, context_tokens=1000)
        runtime.last_memory_refresh_attempt = MemoryRefresh(
            changed=False,
            reason="memory-not-smaller",
            estimated_tokens_before=900,
            estimated_tokens_after=900,
            context_tokens=1000,
            threshold_tokens=850,
        )

        status = runtime_status(runtime, AppConfig(), backend)

        self.assertIn("last_refresh_outcome: discarded (memory-not-smaller)", status)
        self.assertIn("last_refresh_before: none", status)

    def test_runtime_status_can_show_interactive_tools_mode(self) -> None:
        backend = FakeStatusBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        status = runtime_status(runtime, AppConfig(), backend, tools_mode="files,web")

        self.assertIn("tools_mode: files,web", status)

    def test_tools_text_shows_user_selectable_specs_only(self) -> None:
        output = tools_text("off")

        self.assertIn("tools: off", output)
        self.assertIn("/tools files = read/inspect local files", output)
        self.assertIn("/tools edit  = create/modify/delete files or directories", output)
        self.assertIn("/tools web   = search/fetch URLs", output)
        self.assertIn("/tools shell = read-only local/system commands", output)
        self.assertIn("/tools shell-full = DANGEROUS unrestricted local shell", output)
        self.assertNotIn("/tools time", output)
        self.assertNotIn("Single tools:", output)
        self.assertNotIn("/tools read_file,grep_search", output)
        self.assertNotIn("llama-server:", output)
        self.assertNotIn("orbit-only:", output)


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
