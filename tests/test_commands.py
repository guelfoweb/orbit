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
        self.assertNotIn("Compact conversation memory or old tool results.", help_text())
        self.assertIn("/max-tokens [n]", help_text())
        self.assertIn("/continue", help_text())
        self.assertIn("/sessions clear", help_text())
        self.assertIn("/think [off|on]", help_text())
        self.assertIn("/status [ctx]", help_text())
        self.assertIn("/tools [off|on]", help_text())
        self.assertNotIn("Show or set tools: off or on.", help_text())

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
        self.assertIn("thinking_mode: off", status)
        self.assertIn("model_tools: exec_shell_full_command", status)
        self.assertIn("memory_refresh_threshold: 6963/8192", status)
        self.assertIn("memory_refreshes: 0", status)
        self.assertIn("last_refresh_outcome: none", status)
        self.assertNotIn("tools_llama_server:", status)
        self.assertNotIn("tools_orbit_only:", status)

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

        status = runtime_status(runtime, AppConfig(), backend, tools_mode="on")

        self.assertIn("tools_mode: on", status)

    def test_runtime_status_shows_thinking_mode_from_config(self) -> None:
        backend = FakeStatusBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        status = runtime_status(runtime, AppConfig(think=True), backend)

        self.assertIn("thinking_mode: on", status)

    def test_runtime_status_shows_native_backend_runtime_details(self) -> None:
        backend = FakeNativeStatusBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        status = runtime_status(runtime, AppConfig(think=True), backend)

        self.assertIn("Backend runtime\n---------------", status)
        self.assertIn("backend: orbit-native", status)
        self.assertIn("backend_mode: no-mtp", status)
        self.assertIn("session_id: default", status)
        self.assertIn("threads: 6", status)
        self.assertIn("ctx_size: 8192", status)
        self.assertIn("mtp_available: yes", status)
        self.assertIn("multimodal_available: yes", status)

    def test_tools_text_shows_user_selectable_specs_only(self) -> None:
        output = tools_text("off")

        self.assertIn("tools: off", output)
        self.assertIn("/tools off = chat only", output)
        self.assertIn("/tools on  = unrestricted local shell", output)
        self.assertNotIn("/tools files", output)
        self.assertNotIn("/tools web", output)
        self.assertNotIn("/tools shell-full", output)
        self.assertNotIn("Single tools:", output)
        self.assertNotIn("/tools read_file,grep_search", output)
        self.assertNotIn("llama-server:", output)
        self.assertNotIn("orbit-only:", output)


class FakeStatusBackend:
    def backend_props(self) -> dict[str, object]:
        return {}

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


class FakeNativeStatusBackend(FakeStatusBackend):
    def backend_props(self) -> dict[str, object]:
        return {
            "backend": "orbit-native",
            "backend_mode": "no-mtp",
            "session_id": "default",
            "cached_tokens": 128,
            "in_flight": False,
            "threads": 6,
            "threads_batch": 6,
            "ctx_size": 8192,
            "batch_size": 256,
            "ubatch_size": 128,
            "parallel_slots": 1,
            "mtp_available": True,
            "mtp_enabled": False,
            "multimodal_available": True,
        }


if __name__ == "__main__":
    unittest.main()
