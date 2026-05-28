from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.core.events import SessionAutoCompactEvent
import orbit.core.runtime as runtime_module
from orbit.core.runtime import OrbitRuntime
from orbit.core.client import ModelMetadata


class FakeAgent:
    def __init__(self, *, should_compact: bool) -> None:
        self.should_compact = should_compact
        self.compact_called = 0
        self.compact_overflow_tokens = []
        self.run_turn_called = 0
        self.messages = [{"role": "system", "content": "base"}]
        self.skill = None

    def context_pressure(self, pending_user_input=None):
        return type(
            "Pressure",
            (),
            {
                "should_compact": self.should_compact,
                "reason": "soft pressure: score=1.67, msg=80, est_tokens=20000",
                "level": "soft",
                "score": 1.67,
                "session_messages": 80,
                "estimated_prompt_tokens": 20000,
                "overflow_tokens": 8000,
            },
        )()

    def compact(self, *, overflow_tokens: int = 0) -> bool:
        self.compact_called += 1
        self.compact_overflow_tokens.append(overflow_tokens)
        return self.should_compact

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": "base"}]

    def set_skill(self, skill) -> None:
        self.skill = skill

    def run_turn(self, user_input: str, on_event=None):
        self.run_turn_called += 1
        return {"content": user_input}


class RuntimeTests(unittest.TestCase):
    def test_run_turn_auto_compacts_under_pressure(self) -> None:
        agent = FakeAgent(should_compact=True)
        runtime = OrbitRuntime(config=None, client=None, registry=None, agent=agent, session_name="demo")
        saved = []
        events = []
        runtime.save_session = lambda: saved.append("saved")

        def on_event(event):
            events.append(event)

        result = runtime.run_turn("hello", on_event=on_event)
        self.assertEqual(result, {"content": "hello"})
        self.assertEqual(agent.compact_called, 1)
        self.assertEqual(agent.compact_overflow_tokens, [8000])
        self.assertEqual(agent.run_turn_called, 1)
        self.assertGreaterEqual(len(saved), 2)
        self.assertIsInstance(events[0], SessionAutoCompactEvent)
        self.assertEqual(events[0].level, "soft")

    def test_run_turn_skips_auto_compact_when_not_needed(self) -> None:
        agent = FakeAgent(should_compact=False)
        runtime = OrbitRuntime(config=None, client=None, registry=None, agent=agent, session_name="demo")
        saved = []
        runtime.save_session = lambda: saved.append("saved")
        result = runtime.run_turn("hello", on_event=None)
        self.assertEqual(result, {"content": "hello"})
        self.assertEqual(agent.compact_called, 0)
        self.assertEqual(agent.run_turn_called, 1)
        self.assertEqual(saved, ["saved"])

    def test_clear_skill_restores_default_skill(self) -> None:
        agent = FakeAgent(should_compact=False)
        runtime = OrbitRuntime(config=type("Config", (), {"workdir": Path("/tmp"), "base_url": "", "timeout": 1, "model": None, "max_loops": 1, "temperature": 0.0, "think_explicit": False, "show_thinking_explicit": False})(), client=None, registry=None, agent=agent, session_name="demo")
        runtime.save_session = lambda: None
        runtime.clear_skill()
        self.assertIsNotNone(agent.skill)
        self.assertEqual(runtime.active_skill_ref, "orbit-default")

    def test_clear_sessions_for_workdir_resets_agent_and_rotates_session_name(self) -> None:
        agent = FakeAgent(should_compact=False)
        config = type("Config", (), {"workdir": Path("/tmp/project")})()
        runtime = OrbitRuntime(config=config, client=None, registry=None, agent=agent, session_name="demo")
        with (
            patch.object(runtime_module, "delete_sessions_for_workdir", return_value=3),
            patch.object(runtime_module, "create_session_name", return_value="project-12345678"),
        ):
            deleted = runtime.clear_sessions_for_workdir()
        self.assertEqual(deleted, 3)
        self.assertEqual(runtime.session_name, "project-12345678")
        self.assertEqual(agent.messages, [{"role": "system", "content": "base"}])

    def test_startup_notice_for_model_without_tool_support(self) -> None:
        agent = FakeAgent(should_compact=False)
        runtime = OrbitRuntime(
            config=None,
            client=type("Client", (), {"model": "demo"})(),
            registry=None,
            agent=agent,
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="demo",
                context_window=8192,
                capabilities=("completion",),
                tools_supported=False,
            ),
        )
        agent.tools_enabled = False
        self.assertFalse(runtime.tools_enabled)
        self.assertIn("chat-only mode", runtime.startup_notice)

    def test_startup_summary_omits_profile_and_session(self) -> None:
        runtime = OrbitRuntime(
            config=type("Config", (), {"workdir": Path("/tmp")})(),
            client=type("Client", (), {"model": "demo"})(),
            registry=None,
            agent=type("Agent", (), {"think_mode": "off", "show_thinking": False})(),
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="demo",
                context_window=8192,
                capabilities=("completion", "tools"),
                tools_supported=True,
            ),
        )
        first, _ = runtime.startup_summary
        self.assertEqual(first, "orbit v0.1.0 | workdir=/tmp")

    def test_startup_summary_reports_think_off_for_supported_gemma_model(self) -> None:
        runtime = OrbitRuntime(
            config=type("Config", (), {"workdir": Path("/tmp")})(),
            client=type("Client", (), {"model": "gemma4:e4b"})(),
            registry=None,
            agent=type("Agent", (), {"think_mode": "off", "show_thinking": False})(),
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="gemma4:e4b",
                context_window=131072,
                capabilities=("completion", "tools", "thinking"),
                tools_supported=True,
            ),
        )
        _, second = runtime.startup_summary
        self.assertIn("think=off", second)
        self.assertIn("show-thinking=off", second)

    def test_thinking_status_text_separates_think_and_show_thinking(self) -> None:
        runtime = OrbitRuntime(
            config=type("Config", (), {"workdir": Path("/tmp")})(),
            client=type("Client", (), {"model": "demo"})(),
            registry=None,
            agent=type("Agent", (), {"think_mode": "on", "show_thinking": False})(),
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="demo",
                context_window=8192,
                capabilities=("completion", "thinking"),
                tools_supported=True,
            ),
        )
        self.assertEqual(runtime.effective_think_state(), "on")
        self.assertEqual(runtime.effective_show_thinking_state(), "off")
        self.assertEqual(runtime.thinking_status_text(), "think: on | show-thinking: off")

    def test_initial_auto_thinking_enables_both_flags_when_supported(self) -> None:
        config = type("Config", (), {"think_mode": "auto", "show_thinking": False, "think_explicit": False, "show_thinking_explicit": False})()
        runtime = OrbitRuntime(
            config=config,
            client=type("Client", (), {"model": "demo"})(),
            registry=None,
            agent=type("Agent", (), {"think_mode": "auto", "show_thinking": False})(),
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="demo",
                context_window=8192,
                capabilities=("completion", "thinking"),
                tools_supported=True,
            ),
        )
        runtime._apply_initial_thinking_mode()
        self.assertEqual(runtime.agent.think_mode, "on")
        self.assertTrue(runtime.agent.show_thinking)

    def test_initial_auto_thinking_disables_flags_for_gemma_model_first_runtime(self) -> None:
        config = type("Config", (), {"think_mode": "auto", "show_thinking": False, "think_explicit": False, "show_thinking_explicit": False})()
        runtime = OrbitRuntime(
            config=config,
            client=type("Client", (), {"model": "gemma4:e4b"})(),
            registry=None,
            agent=type("Agent", (), {"think_mode": "auto", "show_thinking": False})(),
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="gemma4:e4b",
                context_window=131072,
                capabilities=("completion", "tools", "thinking"),
                tools_supported=True,
            ),
        )
        runtime._apply_initial_thinking_mode()
        self.assertEqual(runtime.agent.think_mode, "off")
        self.assertFalse(runtime.agent.show_thinking)

    def test_initial_auto_thinking_disables_both_flags_when_unsupported(self) -> None:
        config = type("Config", (), {"think_mode": "auto", "show_thinking": True, "think_explicit": False, "show_thinking_explicit": False})()
        runtime = OrbitRuntime(
            config=config,
            client=type("Client", (), {"model": "demo"})(),
            registry=None,
            agent=type("Agent", (), {"think_mode": "auto", "show_thinking": True})(),
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="demo",
                context_window=8192,
                capabilities=("completion",),
                tools_supported=True,
            ),
        )
        runtime._apply_initial_thinking_mode()
        self.assertEqual(runtime.agent.think_mode, "off")
        self.assertFalse(runtime.agent.show_thinking)

    def test_explicit_think_off_disables_both_flags(self) -> None:
        agent = type("Agent", (), {"think_mode": "auto", "show_thinking": True})()
        runtime = OrbitRuntime(
            config=type("Config", (), {"think_mode": "off", "show_thinking": True, "think_explicit": True, "show_thinking_explicit": False})(),
            client=type("Client", (), {"model": "demo"})(),
            registry=None,
            agent=agent,
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="demo",
                context_window=8192,
                capabilities=("completion", "thinking"),
                tools_supported=True,
            ),
        )
        runtime._apply_initial_thinking_mode()
        self.assertEqual(runtime.agent.think_mode, "off")
        self.assertFalse(runtime.agent.show_thinking)

    def test_set_think_mode_off_disables_show_thinking(self) -> None:
        agent = type("Agent", (), {"think_mode": "on", "show_thinking": True})()
        runtime = OrbitRuntime(config=None, client=None, registry=None, agent=agent, session_name="demo")
        runtime.set_think_mode("off")
        self.assertEqual(runtime.agent.think_mode, "off")
        self.assertFalse(runtime.agent.show_thinking)

    def test_set_show_thinking_off_disables_think_mode(self) -> None:
        agent = type("Agent", (), {"think_mode": "on", "show_thinking": True})()
        runtime = OrbitRuntime(config=None, client=None, registry=None, agent=agent, session_name="demo")
        runtime.set_show_thinking(False)
        self.assertEqual(runtime.agent.think_mode, "off")
        self.assertFalse(runtime.agent.show_thinking)

    def test_set_show_thinking_on_without_support_keeps_both_off(self) -> None:
        agent = type("Agent", (), {"think_mode": "auto", "show_thinking": False})()
        runtime = OrbitRuntime(
            config=None,
            client=type("Client", (), {"model": "demo"})(),
            registry=None,
            agent=agent,
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="demo",
                context_window=8192,
                capabilities=("completion",),
                tools_supported=True,
            ),
        )
        runtime.set_show_thinking(True)
        self.assertEqual(runtime.agent.think_mode, "off")
        self.assertFalse(runtime.agent.show_thinking)

    def test_initial_auto_thinking_prefers_off_for_model_first_runtime(self) -> None:
        config = type("Config", (), {"think_mode": "auto", "show_thinking": False, "think_explicit": False, "show_thinking_explicit": False})()
        runtime = OrbitRuntime(
            config=config,
            client=type("Client", (), {"model": "gemma4:e2b"})(),
            registry=None,
            agent=type("Agent", (), {"think_mode": "auto", "show_thinking": False})(),
            session_name="demo",
            model_metadata=ModelMetadata(
                active_model="gemma4:e2b",
                context_window=8192,
                capabilities=("completion", "thinking", "tools"),
                tools_supported=True,
            ),
        )
        runtime._apply_initial_thinking_mode()
        self.assertEqual(runtime.agent.think_mode, "off")
        self.assertFalse(runtime.agent.show_thinking)
