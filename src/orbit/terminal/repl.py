from __future__ import annotations

import sys
import time
import select
import re
from shutil import get_terminal_size
from dataclasses import dataclass

from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.sessions import SessionStore
from orbit.terminal.commands import help_text, reset_session, runtime_status, set_max_tokens, tools_text
from orbit.terminal.config import AppConfig
from orbit.terminal.history import PromptHistory
from orbit.terminal.prefill import estimate_prefill_seconds, estimate_prefill_tokens
from orbit.terminal.prompt_preview import compact_prompt_preview, is_long_text_prompt
from orbit.terminal.status import estimate_context_status_tokens, format_memory_refresh, format_turn_status
from orbit.terminal.streaming import StreamRenderer
from orbit.terminal.tool_events import format_tool_result_event
from orbit.terminal.tool_mode import USAGE, ToolSpec, allowed_tool_names_for_spec, normalize_tool_spec, tools_are_enabled
from orbit.terminal.theme import dim, yellow_dim


PASTE_BADGE_PATTERN = re.compile(r"(\[text \d+ chars #[0-9a-f]{8}\])$")


@dataclass
class Repl:
    runtime: ChatRuntime
    backend: LlamaServerBackend
    config: AppConfig
    session: SessionStore | None = None
    history: PromptHistory | None = None
    prompt_tokens_per_second: float | None = None
    can_continue: bool = False
    tools_mode: ToolSpec | None = None

    def __post_init__(self) -> None:
        if self.tools_mode is None:
            self.tools_mode = self.config.tools

    def run(self) -> int:
        if self.history:
            self.history.load()
        print("orbit interactive mode. Type /help for commands.")
        print(f"tools: {self.tools_mode}")
        while True:
            try:
                prompt = _read_prompt_input().strip()
            except EOFError:
                self._save_history()
                print()
                return 0
            except KeyboardInterrupt:
                self._save_history()
                print()
                return 130
            if not prompt:
                continue
            if prompt.startswith("/"):
                if self._handle_command(prompt):
                    continue
                self._save_history()
                return 0
            if self.history:
                resolution = self.history.resolve_prompt(prompt)
                if resolution.missing_full_text:
                    print("error: full pasted text is unavailable for this history entry", file=sys.stderr)
                    continue
                if resolution.prompt != prompt:
                    prompt = resolution.prompt
                else:
                    _replace_long_input_echo(prompt)
            else:
                _replace_long_input_echo(prompt)
            if self.history:
                self.history.add(prompt)
                self.history.save()
            self._ask(prompt)

    def _ask(self, prompt: str) -> None:
        renderer = StreamRenderer(
            prefill_estimate_seconds=estimate_prefill_seconds(
                self.runtime.messages,
                prompt,
                prompt_tokens_per_second=self.prompt_tokens_per_second,
            ),
            prefill_estimate_tokens=estimate_prefill_tokens(self.runtime.messages, prompt),
        )
        checkpoint = len(self.runtime.messages)
        print()
        started = time.monotonic()
        renderer.start()
        try:
            if tools_are_enabled(self.tools_mode or "off"):
                result = self.runtime.ask_auto(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    workdir=self.config.workdir,
                    allowed_tool_names=allowed_tool_names_for_spec(self.tools_mode or "off"),
                    on_final_delta=renderer.write,
                    on_tool_call=lambda name, args: renderer.event(f"{name} {args}", restart_timer=False),
                    on_tool_result=lambda name, chars, source: renderer.event(
                        format_tool_result_event(name, chars, source),
                        trailing_blank_line=True,
                    ),
                    on_model_step=self._record_model_step,
                )
            else:
                result = self.runtime.ask_chat(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    on_final_delta=renderer.write,
                    on_model_step=self._record_model_step,
                )
        except KeyboardInterrupt:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(dim("interrupted"), flush=True)
            return
        except LlamaServerError as exc:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(f"error: {exc}", file=sys.stderr)
            return
        renderer.finish()
        self._save_session()
        elapsed = time.monotonic() - started
        print("\n\n", end="", flush=True)
        self._print_turn_footer(result, elapsed_seconds=elapsed)

    def _print_turn_footer(self, result, *, elapsed_seconds: float) -> None:
        self.can_continue = result.finish_reason == "length"
        if self.runtime.last_memory_refresh:
            refresh = self.runtime.last_memory_refresh
            print(dim(format_memory_refresh(refresh)), flush=True)
        print(
            dim(
                format_turn_status(
                    result,
                    elapsed_seconds=elapsed_seconds,
                    estimated_context_tokens=estimate_context_status_tokens(self.runtime.messages),
                    context_tokens=self.runtime.context_tokens,
                )
            ),
            flush=True,
        )
        if result.finish_reason == "length":
            print(dim("output stopped because max_tokens was reached"), flush=True)
            print(dim("/continue       continue the answer"), flush=True)
            print(dim("/max-tokens N   increase output budget"), flush=True)

    def _record_model_step(self, metrics) -> None:
        if metrics.prompt_tokens_per_second is not None and metrics.prompt_tokens_per_second > 0:
            self.prompt_tokens_per_second = metrics.prompt_tokens_per_second

    def _handle_command(self, command: str) -> bool:
        if command == "/exit":
            return False
        if command == "/continue":
            self._continue_last_answer()
            return True
        if command == "/help":
            print(help_text())
            return True
        if command == "/reset":
            print(reset_session(self.runtime, self.session))
            self.can_continue = False
            return True
        if command == "/sessions clear":
            print(self._clear_workdir_sessions())
            return True
        if command == "/health":
            print("llama-server: ok" if self.backend.health() else "llama-server: unavailable")
            return True
        if command == "/max-tokens" or command.startswith("/max-tokens "):
            value = command.removeprefix("/max-tokens").strip()
            self.config, message = set_max_tokens(self.config, value)
            print(message)
            return True
        if command == "/status":
            print(runtime_status(self.runtime, self.config, self.backend, tools_mode=self.tools_mode))
            return True
        if command == "/tools" or command.startswith("/tools "):
            print(self._handle_tools_command(command))
            return True
        print(f"unknown command: {command}", file=sys.stderr)
        return True

    def _clear_workdir_sessions(self) -> str:
        if not _confirm_clear_sessions():
            return "sessions clear cancelled"
        removed = SessionStore.clear_for_workdir(self.config.workdir)
        self.runtime.reset()
        self.can_continue = False
        self.session = SessionStore.new_for_workdir(self.config.workdir)
        return f"sessions cleared: {removed}"

    def _handle_tools_command(self, command: str) -> str:
        value = command.removeprefix("/tools").strip().lower()
        if not value:
            return tools_text(self.tools_mode)
        try:
            self.tools_mode = normalize_tool_spec(value)
        except ValueError:
            return f"error: usage: /tools [{USAGE}]"
        if self.tools_mode:
            return f"tools: {self.tools_mode}"
        return f"error: usage: /tools [{USAGE}]"

    def _continue_last_answer(self) -> None:
        if not self.can_continue:
            print("error: no truncated answer to continue", file=sys.stderr)
            return
        self._ask_continue()

    def _save_session(self) -> None:
        if not self.session:
            return
        self.session.save(
            messages=self.runtime.messages,
            workdir=self.config.workdir,
            model=self.backend.display_model_name() or "unknown",
            base_url=self.config.base_url,
        )

    def _save_history(self) -> None:
        if self.history:
            self.history.save()

    def _ask_continue(self) -> None:
        renderer = StreamRenderer(
            prefill_estimate_seconds=estimate_prefill_seconds(
                self.runtime.messages,
                "",
                prompt_tokens_per_second=self.prompt_tokens_per_second,
            ),
            prefill_estimate_tokens=estimate_prefill_tokens(self.runtime.messages, ""),
        )
        checkpoint = len(self.runtime.messages)
        print()
        started = time.monotonic()
        renderer.start()
        try:
            result = self.runtime.continue_last_response(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                on_final_delta=renderer.write,
                on_model_step=self._record_model_step,
            )
        except KeyboardInterrupt:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(dim("interrupted"), flush=True)
            return
        except LlamaServerError as exc:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(f"error: {exc}", file=sys.stderr)
            return
        renderer.finish()
        self._save_session()
        elapsed = time.monotonic() - started
        print("\n\n", end="", flush=True)
        self._print_turn_footer(result, elapsed_seconds=elapsed)


def _replace_long_input_echo(prompt: str) -> None:
    if not is_long_text_prompt(prompt) or not sys.stdout.isatty():
        return
    preview = colorize_paste_preview(compact_prompt_preview(prompt, multiline=True))
    columns = max(20, get_terminal_size((80, 20)).columns)
    visual_rows = max(1, (len("> ") + len(prompt)) // columns + 1)
    print(f"\x1b[{visual_rows}F\x1b[J> {preview}", flush=True)


def colorize_paste_preview(preview: str) -> str:
    return PASTE_BADGE_PATTERN.sub(lambda match: yellow_dim(match.group(1)), preview)


def _read_prompt_input() -> str:
    first_line = input("> ")
    return _read_available_paste_tail(first_line)


def _read_available_paste_tail(first_line: str, *, timeout: float = 0.01, require_tty: bool = True) -> str:
    if require_tty and not sys.stdin.isatty():
        return first_line
    try:
        fileno = sys.stdin.fileno()
    except (AttributeError, OSError):
        return first_line
    lines = [first_line]
    while True:
        try:
            ready, _, _ = select.select([fileno], [], [], timeout)
        except (OSError, ValueError):
            break
        if not ready:
            break
        line = sys.stdin.readline()
        if line == "":
            break
        lines.append(line.rstrip("\n"))
        timeout = 0.0
    return "\n".join(lines)


def _confirm_clear_sessions() -> bool:
    if not sys.stdin.isatty():
        return True
    try:
        answer = input("Delete all saved sessions for this workdir? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}
