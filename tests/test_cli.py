from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.cli import (
    InterruptTracker,
    LastTurnDebug,
    TurnTimer,
    _input_preview,
    _rendered_input_line_count,
    _rewrite_long_input_line,
)
from orbit.session import SessionSummary
from orbit.terminal.config import AppConfig
from orbit.terminal.cli import _choose_startup_session, _read_stdin_prompt, main
from orbit.terminal import history as history_module
from orbit.terminal.ui import format_input_prompt, format_status, format_user_prompt, print_live_event
from orbit.terminal.config import parse_config
from orbit.core.agent import TurnStatus
from orbit.core.events import DebugTimingEvent, ThinkingChunkEvent, ThinkingEndEvent, ThinkingStartEvent, ToolCallEvent, ToolResultEvent, ToolRouteEvent


class InterruptTrackerTests(unittest.TestCase):
    def test_last_turn_debug_formats_compact_route_tool_and_timing(self) -> None:
        debug = LastTurnDebug()
        self.assertEqual(debug.format(), "No turn debug data yet.")

        debug.record(ToolRouteEvent(loop=1, intent="codebase_inspection", categories=("filesystem",), reason="source prompt"))
        debug.record(ToolCallEvent(loop=1, name="list_files", arguments={"path": "."}))
        debug.record(ToolResultEvent(loop=1, name="list_files", ok=True, elapsed_ms=1.2))
        debug.record(DebugTimingEvent(phase="model", elapsed_ms=42.0, detail="loop=1"))
        debug.status_text = "demo | ctx: 8192"

        rendered = debug.format()
        self.assertIn("route: codebase_inspection -> filesystem", rendered)
        self.assertIn("tools: list_files ok 1.2ms", rendered)
        self.assertIn("timing: model loop=1 42.0ms", rendered)
        self.assertIn("status: demo | ctx: 8192", rendered)

    def test_first_interrupt_does_not_exit(self) -> None:
        tracker = InterruptTracker(window_sec=1.5)
        self.assertFalse(tracker.register(now=10.0))

    def test_second_interrupt_within_window_exits(self) -> None:
        tracker = InterruptTracker(window_sec=1.5)
        self.assertFalse(tracker.register(now=10.0))
        self.assertTrue(tracker.register(now=11.0))

    def test_second_interrupt_after_window_does_not_exit(self) -> None:
        tracker = InterruptTracker(window_sec=1.5)
        self.assertFalse(tracker.register(now=10.0))
        self.assertFalse(tracker.register(now=12.0))

    def test_reset_clears_pending_interrupt(self) -> None:
        tracker = InterruptTracker(window_sec=1.5)
        self.assertFalse(tracker.register(now=10.0))
        tracker.reset()
        self.assertFalse(tracker.register(now=10.5))

    def test_choose_startup_session_uses_existing_selection(self) -> None:
        config = AppConfig(
            base_url="http://127.0.0.1:11434",
            model="demo",
            timeout=300,
            workdir=Path("/tmp/project"),
            max_loops=10,
            max_loops_explicit=False,
            temperature=0.0,
            think_mode="auto",
            show_thinking=False,
            think_explicit=False,
            show_thinking_explicit=False,
            skill_ref=None,
            session_name=None,
            prompt=None,
        )
        sessions = [
            SessionSummary(name="demo-2", path=Path("/tmp/demo-2.json"), workdir=config.workdir, first_prompt="second"),
            SessionSummary(name="demo-1", path=Path("/tmp/demo-1.json"), workdir=config.workdir, first_prompt="first"),
        ]
        with (
            patch("orbit.terminal.cli.list_sessions_for_workdir", return_value=sessions),
            patch("builtins.input", return_value="2"),
        ):
            selected = _choose_startup_session(config)
        self.assertEqual(selected.session_name, "demo-1")

    def test_choose_startup_session_can_start_new(self) -> None:
        config = AppConfig(
            base_url="http://127.0.0.1:11434",
            model="demo",
            timeout=300,
            workdir=Path("/tmp/project"),
            max_loops=10,
            max_loops_explicit=False,
            temperature=0.0,
            think_mode="auto",
            show_thinking=False,
            think_explicit=False,
            show_thinking_explicit=False,
            skill_ref=None,
            session_name=None,
            prompt=None,
        )
        with (
            patch("orbit.terminal.cli.list_sessions_for_workdir", return_value=[]),
            patch("orbit.terminal.cli.create_session_name", return_value="project-12345678"),
        ):
            selected = _choose_startup_session(config)
        self.assertEqual(selected.session_name, "project-12345678")

    def test_format_user_prompt_plain_output(self) -> None:
        self.assertEqual(format_user_prompt("analizza il progetto"), "> analizza il progetto")

    def test_format_input_prompt_plain_output(self) -> None:
        self.assertEqual(format_input_prompt(), "> ")

    def test_input_preview_collapses_long_text(self) -> None:
        self.assertEqual(_input_preview("x" * 600), f"{'x' * 50} [text 600 chars]")
        self.assertIsNone(_input_preview("short text"))

    def test_input_preview_collapses_long_multiline_text_with_prefix(self) -> None:
        text = "first line\nsecond line\n" + ("x" * 260)
        expected_prefix = " ".join(text[:50].split())
        self.assertEqual(_input_preview(text), f"{expected_prefix} [text 283 chars]")

    def test_rendered_input_line_count_accounts_for_wrapping_and_newlines(self) -> None:
        self.assertEqual(_rendered_input_line_count("abc", width=80), 1)
        self.assertEqual(_rendered_input_line_count("x" * 90, width=40), 3)
        self.assertEqual(_rendered_input_line_count("abc\ndef", width=80), 2)

    def test_rewrite_long_input_line_prints_placeholder(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.buffer: list[str] = []

            def write(self, text: str) -> None:
                self.buffer.append(text)

            def flush(self) -> None:
                return None

            def isatty(self) -> bool:
                return True

        stream = FakeStream()
        with patch("orbit.terminal.cli.shutil.get_terminal_size", return_value=type("TS", (), {"columns": 80})()):
            _rewrite_long_input_line("x" * 600, stream=stream)
        output = "".join(stream.buffer)
        self.assertIn(f"{'x' * 50} [text 600 chars]", output)
        self.assertIn("F", output)
        self.assertIn("\x1b[2K", output)

    def test_rewrite_long_single_line_does_not_move_above_input_line(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.buffer: list[str] = []

            def write(self, text: str) -> None:
                self.buffer.append(text)

            def flush(self) -> None:
                return None

            def isatty(self) -> bool:
                return True

        stream = FakeStream()
        with patch("orbit.terminal.cli.shutil.get_terminal_size", return_value=type("TS", (), {"columns": 1200})()):
            _rewrite_long_input_line("x" * 600, stream=stream)
        output = "".join(stream.buffer)
        self.assertIn(f"{'x' * 50} [text 600 chars]", output)
        self.assertIn("\x1b[1F", output)
        self.assertNotIn("\x1b[2F", output)

    def test_parse_config_supports_thinking_flags(self) -> None:
        config = parse_config(["--think", "on", "--show-thinking"])
        self.assertEqual(config.max_loops, 10)
        self.assertEqual(config.think_mode, "on")
        self.assertTrue(config.show_thinking)
        self.assertTrue(config.think_explicit)
        self.assertTrue(config.show_thinking_explicit)
        self.assertFalse(config.max_loops_explicit)

    def test_parse_config_marks_max_loops_as_explicit(self) -> None:
        config = parse_config(["--max-loops", "5"])
        self.assertEqual(config.max_loops, 5)
        self.assertTrue(config.max_loops_explicit)

    def test_parse_config_supports_debug_timing(self) -> None:
        config = parse_config(["--debug-timing"])
        self.assertTrue(config.debug_timing)

    def test_parse_config_joins_multiword_prompt(self) -> None:
        config = parse_config(["--model", "demo", "perchè", "il", "cielo", "è", "blu?"])
        self.assertEqual(config.prompt, "perchè il cielo è blu?")

    def test_read_stdin_prompt_returns_none_for_tty(self) -> None:
        with patch("sys.stdin.isatty", return_value=True):
            self.assertIsNone(_read_stdin_prompt())

    def test_read_stdin_prompt_reads_piped_input(self) -> None:
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdin.read", return_value="hello from pipe\n"),
        ):
            self.assertEqual(_read_stdin_prompt(), "hello from pipe")

    def test_setup_history_enables_bracketed_paste(self) -> None:
        calls: list[str] = []

        class FakeReadline:
            def parse_and_bind(self, command: str) -> None:
                calls.append(command)

            def read_history_file(self, path) -> None:
                calls.append(f"read:{path}")

            def set_history_length(self, length: int) -> None:
                calls.append(f"len:{length}")

            def write_history_file(self, path) -> None:
                calls.append(f"write:{path}")

            def get_current_history_length(self) -> int:
                return 0

            def add_history(self, text: str) -> None:
                calls.append(f"add:{text}")

            def get_history_item(self, index: int) -> str | None:
                return None

        with (
            patch.object(history_module, "readline", FakeReadline()),
            patch.object(history_module, "ensure_orbit_home"),
            patch.object(history_module.atexit, "register"),
        ):
            history_module.setup_history()
        self.assertIn("set enable-bracketed-paste on", calls[0])

    def test_main_uses_stdin_prompt_in_one_shot_mode(self) -> None:
        fake_runtime = type(
            "Runtime",
            (),
            {
                "startup_notice": None,
                "run_turn": lambda self, prompt, on_event=None: type(
                    "Result",
                    (),
                    {
                        "content": f"echo: {prompt}",
                        "status": type(
                            "Status",
                            (),
                            {
                                "active_model": "demo",
                                "context_window": 8192,
                                "session_messages": 1,
                                "session_turns": 1,
                                "prompt_tokens": 10,
                                "estimated_prompt_tokens": 10,
                                "output_tokens": 2,
                                "prefill_tps": None,
                                "decode_tps": None,
                                "model_elapsed_sec": None,
                                "wall_elapsed_sec": None,
                                "tool_elapsed_sec": None,
                                "usage_ratio": None,
                                "warning": None,
                                "think_state": "no",
                            },
                        )(),
                    },
                )(),
            },
        )()
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdin.read", return_value="hello from stdin\n"),
            patch("orbit.terminal.cli.OrbitRuntime.from_config", return_value=fake_runtime),
            patch("builtins.print") as mock_print,
        ):
            exit_code = main(["--model", "demo"])
        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("echo: hello from stdin", printed)

    def test_main_one_shot_uses_turn_timer_wrapper(self) -> None:
        fake_runtime = type(
            "Runtime",
            (),
            {
                "startup_notice": None,
                "run_turn": lambda self, prompt, on_event=None: type(
                    "Result",
                    (),
                    {
                        "content": f"echo: {prompt}",
                        "status": type(
                            "Status",
                            (),
                            {
                                "active_model": "demo",
                                "context_window": 8192,
                                "session_messages": 1,
                                "session_turns": 1,
                                "prompt_tokens": 10,
                                "estimated_prompt_tokens": 10,
                                "output_tokens": 2,
                                "prefill_tps": None,
                                "decode_tps": None,
                                "model_elapsed_sec": None,
                                "wall_elapsed_sec": None,
                                "tool_elapsed_sec": None,
                                "usage_ratio": None,
                                "warning": None,
                                "think_state": "no",
                                "show_thinking_state": "off",
                            },
                        )(),
                    },
                )(),
            },
        )()
        timer_calls = []

        class FakeTimer:
            def start(self_nonlocal):
                timer_calls.append("start")

            def stop(self_nonlocal):
                timer_calls.append("stop")

        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdin.read", return_value="hello from pipe\n"),
            patch("orbit.terminal.cli.OrbitRuntime.from_config", return_value=fake_runtime),
            patch("orbit.terminal.cli.TurnTimer", return_value=FakeTimer()),
            patch("builtins.print") as mock_print,
        ):
            exit_code = main(["--model", "demo"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(timer_calls, ["start", "stop"])
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("echo: hello from pipe", printed)

    def test_turn_timer_keeps_final_neutral_line(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.buffer: list[str] = []

            def write(self, text: str) -> None:
                self.buffer.append(text)

            def flush(self) -> None:
                return None

            def isatty(self) -> bool:
                return True

        stream = FakeStream()
        timer = TurnTimer(stream=stream, update_sec=10.0)
        with patch("orbit.terminal.cli.time.monotonic", side_effect=[100.0, 100.0, 101.3]):
            timer.start()
            timer.stop()
        output = "".join(stream.buffer)
        self.assertIn("0.0s ", output)
        self.assertIn("1.3s ", output)
        self.assertNotIn("⏳", output)
        self.assertNotIn("time:", output)
        self.assertTrue(output.endswith("\x1b[0m\n"))

    def test_print_live_event_renders_thinking_on_separate_lines(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.buffer: list[str] = []

            def write(self, text: str) -> None:
                self.buffer.append(text)

            def flush(self) -> None:
                return None

            def isatty(self) -> bool:
                return False

        fake_stderr = FakeStream()
        with patch("orbit.terminal.ui.sys.stderr", fake_stderr):
            print_live_event(ThinkingStartEvent(loop=0))
            print_live_event(ThinkingChunkEvent(loop=0, text="first chunk"))
            print_live_event(ThinkingChunkEvent(loop=0, text="second line\nthird line"))
            print_live_event(ThinkingEndEvent(loop=0))
        output = "".join(fake_stderr.buffer)
        self.assertIn("└ thinking\n", output)
        self.assertIn("  first chunk\n", output)
        self.assertIn("  second line\n", output)
        self.assertIn("  third line\n", output)

    def test_format_status_omits_final_time_and_tool_fields(self) -> None:
        status = TurnStatus(
            active_model="demo",
            context_window=8192,
            session_messages=3,
            session_turns=2,
            prompt_tokens=120,
            estimated_prompt_tokens=120,
            output_tokens=18,
            prefill_tps=45.0,
            decode_tps=12.0,
            model_elapsed_sec=4.0,
            wall_elapsed_sec=31.0,
            tool_elapsed_sec=2.5,
            usage_ratio=0.25,
            warning=None,
            think_state="off",
            show_thinking_state="off",
        )
        rendered = format_status(status)
        self.assertIn("tk_pf: 45.0/s", rendered)
        self.assertIn("tk_gen: 12.0/s", rendered)
        self.assertIn("msg: 2", rendered)
        self.assertNotIn("think: off", rendered)
        self.assertNotIn("show-thinking: off", rendered)

    def test_main_accepts_session_clear_alias_without_sending_to_model(self) -> None:
        runtime = type(
            "Runtime",
            (),
            {
                "startup_notice": None,
                "startup_summary": ("first", "second"),
                "config": type("Config", (), {"workdir": Path("/tmp/project")})(),
                "session_name": "project-12345678",
                "agent": type("Agent", (), {"current_status": lambda self: type(
                    "Status",
                    (),
                    {
                        "active_model": "demo",
                        "context_window": 8192,
                        "session_messages": 1,
                        "session_turns": 1,
                        "prompt_tokens": 10,
                        "estimated_prompt_tokens": 10,
                        "output_tokens": 2,
                        "prefill_tps": None,
                        "decode_tps": None,
                        "model_elapsed_sec": None,
                        "wall_elapsed_sec": None,
                        "tool_elapsed_sec": None,
                        "usage_ratio": None,
                        "warning": None,
                        "think_state": "no",
                    },
                )()})(),
                "clear_sessions_for_workdir": lambda self: 2,
                "run_turn": lambda self, prompt, on_event=None: (_ for _ in ()).throw(AssertionError("run_turn should not be called")),
            },
        )()
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("orbit.terminal.cli.list_sessions_for_workdir", return_value=[]),
            patch("orbit.terminal.cli.OrbitRuntime.from_config", return_value=runtime),
            patch("builtins.input", side_effect=["/session clear", "/exit"]),
            patch("builtins.print") as mock_print,
        ):
            exit_code = main(["--model", "demo"])
        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("cleared 2 session(s) for /tmp/project", printed)

    def test_main_reports_effective_thinking_state_after_think_command(self) -> None:
        runtime = type(
            "Runtime",
            (),
            {
                "startup_notice": None,
                "startup_summary": ("first", "second"),
                "config": type("Config", (), {"workdir": Path("/tmp/project")})(),
                "session_name": "project-12345678",
                "agent": type("Agent", (), {"current_status": lambda self: type(
                    "Status",
                    (),
                    {
                        "active_model": "demo",
                        "context_window": 8192,
                        "session_messages": 1,
                        "session_turns": 1,
                        "prompt_tokens": 10,
                        "estimated_prompt_tokens": 10,
                        "output_tokens": 2,
                        "prefill_tps": None,
                        "decode_tps": None,
                        "model_elapsed_sec": None,
                        "wall_elapsed_sec": None,
                        "tool_elapsed_sec": None,
                        "usage_ratio": None,
                        "warning": None,
                        "think_state": "off",
                        "show_thinking_state": "off",
                    },
                )()})(),
                "set_think_mode": lambda self, mode: None,
                "thinking_status_text": lambda self: "think: on | show-thinking: on",
                "run_turn": lambda self, prompt, on_event=None: (_ for _ in ()).throw(AssertionError("run_turn should not be called")),
            },
        )()
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("orbit.terminal.cli.list_sessions_for_workdir", return_value=[]),
            patch("orbit.terminal.cli.OrbitRuntime.from_config", return_value=runtime),
            patch("builtins.input", side_effect=["/think on", "/exit"]),
            patch("builtins.print") as mock_print,
        ):
            exit_code = main(["--model", "demo"])
        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("think: on | show-thinking: on", printed)

    def test_main_reports_effective_thinking_state_after_thinking_command(self) -> None:
        runtime = type(
            "Runtime",
            (),
            {
                "startup_notice": None,
                "startup_summary": ("first", "second"),
                "config": type("Config", (), {"workdir": Path("/tmp/project")})(),
                "session_name": "project-12345678",
                "agent": type("Agent", (), {"current_status": lambda self: type(
                    "Status",
                    (),
                    {
                        "active_model": "demo",
                        "context_window": 8192,
                        "session_messages": 1,
                        "session_turns": 1,
                        "prompt_tokens": 10,
                        "estimated_prompt_tokens": 10,
                        "output_tokens": 2,
                        "prefill_tps": None,
                        "decode_tps": None,
                        "model_elapsed_sec": None,
                        "wall_elapsed_sec": None,
                        "tool_elapsed_sec": None,
                        "usage_ratio": None,
                        "warning": None,
                        "think_state": "on",
                        "show_thinking_state": "off",
                    },
                )()})(),
                "set_show_thinking": lambda self, enabled: None,
                "thinking_status_text": lambda self: "think: on | show-thinking: on",
                "run_turn": lambda self, prompt, on_event=None: (_ for _ in ()).throw(AssertionError("run_turn should not be called")),
            },
        )()
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("orbit.terminal.cli.list_sessions_for_workdir", return_value=[]),
            patch("orbit.terminal.cli.OrbitRuntime.from_config", return_value=runtime),
            patch("builtins.input", side_effect=["/thinking on", "/exit"]),
            patch("builtins.print") as mock_print,
        ):
            exit_code = main(["--model", "demo"])
        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("think: on | show-thinking: on", printed)

    def test_main_rejects_unknown_slash_command_locally(self) -> None:
        runtime = type(
            "Runtime",
            (),
            {
                "startup_notice": None,
                "startup_summary": ("first", "second"),
                "config": type("Config", (), {"workdir": Path("/tmp/project")})(),
                "session_name": "project-12345678",
                "agent": type("Agent", (), {"current_status": lambda self: type(
                    "Status",
                    (),
                    {
                        "active_model": "demo",
                        "context_window": 8192,
                        "session_messages": 1,
                        "session_turns": 1,
                        "prompt_tokens": 10,
                        "estimated_prompt_tokens": 10,
                        "output_tokens": 2,
                        "prefill_tps": None,
                        "decode_tps": None,
                        "model_elapsed_sec": None,
                        "wall_elapsed_sec": None,
                        "tool_elapsed_sec": None,
                        "usage_ratio": None,
                        "warning": None,
                        "think_state": "no",
                    },
                )()})(),
                "run_turn": lambda self, prompt, on_event=None: (_ for _ in ()).throw(AssertionError("run_turn should not be called")),
            },
        )()
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("orbit.terminal.cli.list_sessions_for_workdir", return_value=[]),
            patch("orbit.terminal.cli.OrbitRuntime.from_config", return_value=runtime),
            patch("builtins.input", side_effect=["/session unknown", "/exit"]),
            patch("builtins.print") as mock_print,
        ):
            exit_code = main(["--model", "demo"])
        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("error: unknown command: /session unknown", printed)
