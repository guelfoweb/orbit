from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.sessions import SessionStore
from orbit.terminal.compact_reports import format_memory_compaction_report, format_tool_compaction_report
from orbit.terminal.commands import health_text, help_text, reset_session, runtime_status, set_max_tokens, tools_text
from orbit.terminal.config import AppConfig
from orbit.terminal.context_status import context_status_text
from orbit.terminal.history import PromptHistory
from orbit.terminal.prefill import MIN_PREFILL_ESTIMATE_SECONDS, estimate_prefill_tokens
from orbit.terminal.prefill_estimator import CHAT_PREFILL_PROFILE, FINAL_FROM_TOOL_PREFILL_PROFILE, TOOL_PREFILL_PROFILE, PrefillEstimator, prefill_profile_for_phase
from orbit.terminal.repl_input import clear_input_echo, read_prompt_input, replace_input_echo
from orbit.terminal.session_preview import format_recent_session_messages, has_existing_session_context
from orbit.terminal.status import estimate_context_status_tokens, format_memory_refresh, format_turn_status
from orbit.terminal.streaming import StreamRenderer
from orbit.terminal.tool_events import format_tool_call_event, format_tool_result_event
from orbit.terminal.tool_mode import USAGE, ToolSpec, allowed_tool_names_for_spec, normalize_tool_spec, tools_are_enabled
from orbit.terminal.theme import danger, dim


@dataclass
class Repl:
    runtime: ChatRuntime
    backend: LlamaServerBackend
    config: AppConfig
    session: SessionStore | None = None
    history: PromptHistory | None = None
    prefill_estimator: PrefillEstimator = field(default_factory=PrefillEstimator)
    can_continue: bool = False
    tools_mode: ToolSpec | None = None
    queued_prompts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tools_mode is None:
            self.tools_mode = self.config.tools

    def run(self) -> int:
        if self.history:
            self.history.load()
        print(dim("orbit interactive mode. Type /help for commands."))
        print(dim(f"tools: {self.tools_mode}"))
        if has_existing_session_context(self.runtime.messages):
            print(dim("recent session context:"))
            for line in format_recent_session_messages(self.runtime.messages):
                print(dim(line))
        while True:
            try:
                prompt = self._read_next_prompt().strip()
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
                clear_input_echo(prompt)
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
                    replace_input_echo(prompt)
            else:
                replace_input_echo(prompt)
            if self.history:
                self.history.add(prompt)
                self.history.save()
            self._ask(prompt)

    def _read_next_prompt(self) -> str:
        if self.queued_prompts:
            return self.queued_prompts.pop(0)
        return read_prompt_input()

    def _ask(self, prompt: str) -> None:
        prefill_tokens = estimate_prefill_tokens(self.runtime.messages, prompt)
        prefill_profile = _prefill_profile_for_turn(self.runtime.messages, tools_enabled=tools_are_enabled(self.tools_mode or "off"))
        prefill_seconds = self.prefill_estimator.estimate_seconds(prefill_tokens, profile=prefill_profile)
        renderer = StreamRenderer(
            prefill_estimate_seconds=_visible_prefill_seconds(prefill_seconds),
            prefill_estimate_tokens=prefill_tokens,
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
                    on_tool_call=lambda name, args: renderer.event(format_tool_call_event(name, args), restart_timer=False),
                    on_tool_result=lambda name, chars, source, content: renderer.event(
                        format_tool_result_event(name, chars, source, content),
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
        self.prefill_estimator.update(
            prompt_tokens=metrics.prompt_tokens,
            prompt_tokens_per_second=metrics.prompt_tokens_per_second,
            profile=prefill_profile_for_phase(metrics.phase),
        )

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
        if command == "/compact":
            print(format_memory_compaction_report(self.runtime.compact_memory_now(temperature=self.config.temperature)))
            self._save_session()
            return True
        if command == "/compact tools":
            print(format_tool_compaction_report(self.runtime.compact_old_tool_results(temperature=self.config.temperature)))
            self._save_session()
            return True
        if command == "/sessions clear":
            print(self._clear_workdir_sessions())
            return True
        if command == "/health":
            print(health_text(self.backend, self.config))
            return True
        if command == "/max-tokens" or command.startswith("/max-tokens "):
            value = command.removeprefix("/max-tokens").strip()
            self.config, message = set_max_tokens(self.config, value)
            print(message)
            return True
        if command == "/status":
            print(runtime_status(self.runtime, self.config, self.backend, tools_mode=self.tools_mode))
            return True
        if command in {"/status ctx", "/status context"}:
            print(context_status_text(self.runtime.messages, context_tokens=self.runtime.context_tokens))
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
            if self.tools_mode == "on":
                return (
                    f"tools: {self.tools_mode}\n"
                    + danger(
                        "warning: tools on gives the model unrestricted local shell access. Commands may read, modify, delete files, execute programs, or access network. "
                        "Use only in an isolated lab."
                    )
                )
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
            messages=self.runtime.persistent_messages(),
            workdir=self.config.workdir,
            model=self.backend.display_model_name() or "unknown",
            base_url=self.config.base_url,
        )

    def _save_history(self) -> None:
        if self.history:
            self.history.save()

    def _ask_continue(self) -> None:
        prefill_tokens = estimate_prefill_tokens(self.runtime.messages, "")
        prefill_seconds = self.prefill_estimator.estimate_seconds(prefill_tokens)
        renderer = StreamRenderer(
            prefill_estimate_seconds=_visible_prefill_seconds(prefill_seconds),
            prefill_estimate_tokens=prefill_tokens,
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


def _confirm_clear_sessions() -> bool:
    if not sys.stdin.isatty():
        return True
    try:
        answer = input("Delete all saved sessions for this workdir? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def _visible_prefill_seconds(seconds: float | None) -> float | None:
    if seconds is None or seconds < MIN_PREFILL_ESTIMATE_SECONDS:
        return None
    return seconds


def _prefill_profile_for_turn(messages: list[dict[str, object]], *, tools_enabled: bool) -> str:
    if not tools_enabled:
        return CHAT_PREFILL_PROFILE
    if any(message.get("role") == "tool" for message in messages[-4:]):
        return FINAL_FROM_TOOL_PREFILL_PROFILE
    return TOOL_PREFILL_PROFILE
