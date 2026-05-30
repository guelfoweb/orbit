from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
import shutil
import sys
import threading
import time
from typing import Any

from ..core.client import OllamaError
from ..core.events import DebugTimingEvent, ToolCallEvent, ToolResultEvent, ToolRouteEvent
from ..core.runtime import OrbitRuntime
from ..session import create_session_name, list_sessions_for_workdir
from ..skills import list_skills
from .config import parse_config
from .history import remember_input, setup_history
from .ui import (
    LIVE_OUTPUT_LOCK,
    format_input_prompt,
    format_user_prompt,
    format_skill,
    format_runtime_status,
    format_startup_line,
    format_status,
    make_live_event_printer,
    print_help,
    print_tools,
)


DOUBLE_CTRL_C_WINDOW_SEC = 1.5
TURN_TIMER_UPDATE_SEC = 1.0
TIMER_COLOR = "\x1b[90m"
TIMER_RESET = "\x1b[0m"
LONG_INPUT_PREVIEW_CHARS = 512
LONG_INPUT_MULTILINE_CHARS = 256
LONG_INPUT_PREVIEW_PREFIX_CHARS = 50
IMAGE_PATH_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:png|jpe?g|webp|bmp|gif)\b", re.IGNORECASE)
AUDIO_PATH_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:wav|mp3|m4a|flac|ogg)\b", re.IGNORECASE)


@dataclass
class InterruptTracker:
    window_sec: float = DOUBLE_CTRL_C_WINDOW_SEC
    _last_interrupt_at: float | None = None

    def register(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        should_exit = self._last_interrupt_at is not None and (current - self._last_interrupt_at) <= self.window_sec
        self._last_interrupt_at = current
        return should_exit

    def reset(self) -> None:
        self._last_interrupt_at = None


@dataclass
class TurnTimer:
    stream: object = sys.stderr
    update_sec: float = TURN_TIMER_UPDATE_SEC
    _started_at: float | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._is_enabled():
            return
        self._started_at = time.monotonic()
        self._stop_event.clear()
        self._render()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._is_enabled() or self._started_at is None:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.2)
        self._write(f"\r{self._formatted_elapsed()}\n")
        self._thread = None
        self._started_at = None

    def _run(self) -> None:
        while not self._stop_event.wait(self.update_sec):
            self._render()

    def _render(self) -> None:
        self._write(f"\r{self._formatted_elapsed()}")

    def _write(self, text: str) -> None:
        with LIVE_OUTPUT_LOCK:
            if hasattr(self.stream, "write"):
                self.stream.write(text)
            if hasattr(self.stream, "flush"):
                self.stream.flush()

    def _is_enabled(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def _formatted_elapsed(self) -> str:
        elapsed = time.monotonic() - self._started_at if self._started_at is not None else 0.0
        text = f"{elapsed:.1f}s "
        return f"{TIMER_COLOR}{text}{TIMER_RESET}"


@dataclass
class LastTurnDebug:
    route: str | None = None
    route_categories: tuple[str, ...] = ()
    route_reason: str | None = None
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    tool_results: list[tuple[str, bool, str | None, float | None]] = field(default_factory=list)
    timings: list[tuple[str, float, str | None]] = field(default_factory=list)
    status_text: str | None = None

    def reset(self) -> None:
        self.route = None
        self.route_categories = ()
        self.route_reason = None
        self.tool_calls.clear()
        self.tool_results.clear()
        self.timings.clear()
        self.status_text = None

    def record(self, event) -> None:
        if isinstance(event, ToolRouteEvent):
            self.route = event.intent
            self.route_categories = event.categories
            self.route_reason = event.reason
        elif isinstance(event, ToolCallEvent):
            self.tool_calls.append((event.name, event.arguments))
        elif isinstance(event, ToolResultEvent):
            self.tool_results.append((event.name, event.ok, event.error, event.elapsed_ms))
        elif isinstance(event, DebugTimingEvent):
            self.timings.append((event.phase, event.elapsed_ms, event.detail))

    def has_data(self) -> bool:
        return bool(self.route or self.tool_calls or self.tool_results or self.timings or self.status_text)

    def format(self) -> str:
        if not self.has_data():
            return "No turn debug data yet."
        lines = ["Last turn:"]
        if self.route:
            categories = f" -> {', '.join(self.route_categories)}" if self.route_categories else ""
            reason = f" ({self.route_reason})" if self.route_reason else ""
            lines.append(f"- route: {self.route}{categories}{reason}")
        if self.tool_calls or self.tool_results:
            result_by_index = list(self.tool_results)
            rendered_tools: list[str] = []
            for index, (name, _arguments) in enumerate(self.tool_calls):
                if index < len(result_by_index):
                    result_name, ok, error, elapsed_ms = result_by_index[index]
                    label = result_name or name
                    state = "ok" if ok else f"error: {error or 'failed'}"
                    elapsed = f" {elapsed_ms:.1f}ms" if elapsed_ms is not None else ""
                    rendered_tools.append(f"{label} {state}{elapsed}")
                else:
                    rendered_tools.append(f"{name} pending")
            lines.append(f"- tools: {'; '.join(rendered_tools)}")
        if self.timings:
            rendered_timings = []
            for phase, elapsed_ms, detail in self.timings[-8:]:
                suffix = f" {detail}" if detail else ""
                rendered_timings.append(f"{phase}{suffix} {elapsed_ms:.1f}ms")
            lines.append(f"- timing: {'; '.join(rendered_timings)}")
        if self.status_text:
            lines.append(f"- status: {self.status_text}")
        return "\n".join(lines)

    def response_source(self, status: object) -> str:
        has_tools = bool(self.tool_results or self.tool_calls)
        output_tokens = getattr(status, "output_tokens", None)
        if isinstance(output_tokens, int) and output_tokens > 0:
            return "tool+model" if has_tools else "model"
        if has_tools:
            return "local"
        return "local"


def _run_turn_with_timer(runtime, prompt: str, on_event=None):
    timer = TurnTimer()
    timer.start()
    try:
        return runtime.run_turn(prompt, on_event=on_event)
    finally:
        timer.stop()


def _format_turn_status(status: object, source: str | None = None, prep: str | None = None) -> str:
    text = format_status(status)
    suffix_parts = [f"msg: {getattr(status, 'session_turns', '-')}"]
    if source:
        suffix_parts.append(f"src: {source}")
    if prep:
        suffix_parts.append(f"prep: {prep}")
    if suffix_parts:
        suffix = " | ".join(suffix_parts)
        reset = "\x1b[0m"
        if text.endswith(reset):
            return f"{text[: -len(reset)]} | {suffix}{reset}"
        return f"{text} | {suffix}"
    return text


def _print_turn_output(content: str, status: object, source: str | None = None, prep: str | None = None) -> None:
    print(content)
    if _should_print_turn_separator(content):
        print(_turn_separator())
    print(_format_turn_status(status, source=source, prep=prep))


def _should_print_turn_separator(content: str) -> bool:
    if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
        return False
    if not isinstance(content, str):
        return False
    return len(content) > 1200 or content.count("\n") >= 8


def _turn_separator() -> str:
    width = max(20, min(80, shutil.get_terminal_size(fallback=(80, 24)).columns))
    return "-" * width


def _turn_preprocessing_label(prompt: str) -> str | None:
    if AUDIO_PATH_RE.search(prompt):
        return "audio"
    if IMAGE_PATH_RE.search(prompt):
        return "vision"
    return None


def _input_preview(text: str) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    is_multiline = "\n" in text
    if len(text) <= LONG_INPUT_PREVIEW_CHARS and not (is_multiline and len(text) > LONG_INPUT_MULTILINE_CHARS):
        return None
    prefix = " ".join(text[:LONG_INPUT_PREVIEW_PREFIX_CHARS].split())
    if not prefix:
        return f"[text {len(text)} chars]"
    return f"{prefix} [text {len(text)} chars]"


def _rewrite_long_input_line(text: str, *, stream=None) -> None:
    preview = _input_preview(text)
    if preview is None:
        return
    target = sys.stdout if stream is None else stream
    if not bool(getattr(target, "isatty", lambda: False)()):
        return
    width = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns)
    rendered_lines = _rendered_input_line_count(text, width=width, prompt_width=2)
    with LIVE_OUTPUT_LOCK:
        if hasattr(target, "write"):
            target.write(f"\x1b[{rendered_lines}F")
            for index in range(rendered_lines):
                target.write("\x1b[2K")
                if index < rendered_lines - 1:
                    target.write("\x1b[1E")
            if rendered_lines > 1:
                target.write(f"\x1b[{rendered_lines - 1}F")
            target.write(format_user_prompt(preview))
            target.write("\n")
        if hasattr(target, "flush"):
            target.flush()


def _rendered_input_line_count(text: str, *, width: int, prompt_width: int = 2) -> int:
    total = 0
    lines = text.splitlines() or [""]
    if text.endswith("\n"):
        lines.append("")
    for index, line in enumerate(lines):
        visual_len = len(line) + (prompt_width if index == 0 else 0)
        total += max(1, math.ceil(max(visual_len, 1) / width))
    return max(total, 1)


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_config(argv)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    stdin_prompt = _read_stdin_prompt()
    if config.prompt is None and stdin_prompt is not None:
        config = replace(config, prompt=stdin_prompt)

    if config.prompt is None and config.session_name is None:
        config = _choose_startup_session(config)

    try:
        runtime = OrbitRuntime.from_config(config)
    except OllamaError as exc:
        print(f"error: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    if config.prompt:
        try:
            if runtime.startup_notice:
                print(runtime.startup_notice)
            debug_recorder = LastTurnDebug()
            event_printer = make_live_event_printer(debug_timing=config.debug_timing)

            def record_and_print_event(event) -> None:
                debug_recorder.record(event)
                event_printer(event)

            result = _run_turn_with_timer(runtime, config.prompt, on_event=record_and_print_event)
            _print_turn_output(
                result.content,
                result.status,
                source=debug_recorder.response_source(result.status),
                prep=_turn_preprocessing_label(config.prompt),
            )
            return 0
        except OllamaError as exc:
            print(f"error: {exc}")
            return 1

    setup_history()
    interrupts = InterruptTracker()
    startup_first, startup_second = runtime.startup_summary
    print(format_startup_line(startup_first))
    print(format_startup_line(startup_second))
    if runtime.startup_notice:
        print(runtime.startup_notice)
    print(format_status(runtime.agent.current_status()))
    print("Type /help for commands.")
    event_printer = make_live_event_printer(debug_timing=config.debug_timing)
    debug_recorder = LastTurnDebug()

    def record_and_print_event(event) -> None:
        debug_recorder.record(event)
        event_printer(event)

    while True:
        try:
            user_input = input(format_input_prompt()).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            if interrupts.register():
                return 130
            print("interrupted")
            continue
        interrupts.reset()
        if not user_input:
            continue
        _rewrite_long_input_line(user_input)
        remember_input(user_input)
        if user_input == "/exit":
            return 0
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/tools":
            print_tools(runtime.registry)
            continue
        if user_input == "/skill list":
            skills = list_skills()
            if not skills:
                print("no skills found")
                continue
            print("Skills:")
            for skill in skills:
                print(f"  {skill.name}")
            continue
        if user_input == "/status":
            try:
                print(format_runtime_status(runtime))
            except OllamaError as exc:
                print(f"error: {exc}")
            continue
        if user_input in {"/debug", "/debug last"}:
            print(debug_recorder.format())
            continue
        if user_input == "/skill show":
            print(format_skill(runtime.agent))
            continue
        if user_input in {"/think on", "/think off", "/think auto"}:
            mode = user_input.split()[-1]
            runtime.set_think_mode(mode)
            print(runtime.thinking_status_text())
            continue
        if user_input in {"/thinking on", "/thinking off"}:
            enabled = user_input.endswith("on")
            runtime.set_show_thinking(enabled)
            print(runtime.thinking_status_text())
            continue
        if user_input == "/skill clear":
            runtime.clear_skill()
            print("default skill restored")
            continue
        if user_input.startswith("/skill use "):
            ref = user_input[len("/skill use ") :].strip()
            if not ref:
                print("error: usage: /skill use <name-or-path>")
                continue
            try:
                runtime.set_skill(ref)
            except FileNotFoundError as exc:
                print(f"error: {exc}")
                continue
            print(format_skill(runtime.agent))
            print(format_status(runtime.agent.current_status()))
            continue
        if user_input == "/compact":
            changed = runtime.compact_session()
            print("session compacted" if changed else "nothing to compact")
            print(format_status(runtime.agent.current_status()))
            continue
        if user_input == "/reset":
            runtime.reset_session()
            print("session reset")
            continue
        if user_input == "/sessions clear":
            deleted = runtime.clear_sessions_for_workdir()
            print(f"cleared {deleted} session(s) for {runtime.config.workdir}")
            print(f"new session: {runtime.session_name}")
            continue
        if user_input == "/session clear":
            deleted = runtime.clear_sessions_for_workdir()
            print(f"cleared {deleted} session(s) for {runtime.config.workdir}")
            print(f"new session: {runtime.session_name}")
            continue
        if user_input.startswith("/"):
            print(f"error: unknown command: {user_input}")
            continue
        try:
            debug_recorder.reset()
            result = _run_turn_with_timer(runtime, user_input, on_event=record_and_print_event)
            source = debug_recorder.response_source(result.status)
            prep = _turn_preprocessing_label(user_input)
            debug_recorder.status_text = _format_turn_status(result.status, source=source, prep=prep)
            interrupts.reset()
            _print_turn_output(result.content, result.status, source=source, prep=prep)
        except KeyboardInterrupt:
            print()
            if interrupts.register():
                return 130
            print("request interrupted")
        except OllamaError as exc:
            interrupts.reset()
            print(f"error: {exc}")
    return 0


def _choose_startup_session(config):
    sessions = list_sessions_for_workdir(config.workdir)
    if not sessions:
        return replace(config, session_name=create_session_name(config.workdir))
    print(f"Found {len(sessions)} session(s) for workdir {config.workdir}:")
    for index, session in enumerate(sessions, start=1):
        print(f"  {index}. {session.name}")
        print(f"     {session.first_prompt}")
    print("  n. new session")
    while True:
        default_choice = "1"
        choice = input(f"Choose session [{default_choice}/n]: ").strip().lower()
        if not choice:
            selected = sessions[0]
            return replace(config, session_name=selected.name)
        if choice == "n":
            return replace(config, session_name=create_session_name(config.workdir))
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(sessions):
                return replace(config, session_name=sessions[index - 1].name)
        print("error: choose a session number or 'n'")


def _read_stdin_prompt() -> str | None:
    if sys.stdin.isatty():
        return None
    content = sys.stdin.read()
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    return stripped or None
