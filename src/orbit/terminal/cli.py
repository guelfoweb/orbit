from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import shutil
import sys
import threading
import time

from ..core.client import OllamaError
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


def _run_turn_with_timer(runtime, prompt: str, on_event=None):
    timer = TurnTimer()
    timer.start()
    try:
        return runtime.run_turn(prompt, on_event=on_event)
    finally:
        timer.stop()


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
            result = _run_turn_with_timer(runtime, config.prompt, on_event=make_live_event_printer(debug_timing=config.debug_timing))
            print(result.content)
            print(format_status(result.status))
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
                print(format_status(runtime.agent.current_status()))
            except OllamaError as exc:
                print(f"error: {exc}")
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
            result = _run_turn_with_timer(runtime, user_input, on_event=event_printer)
            interrupts.reset()
            print(result.content)
            print(format_status(result.status))
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
