from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace

from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT
from orbit.runtime.sessions import SessionStore
from orbit.terminal.compact_reports import format_memory_compaction_report, format_tool_compaction_report
from orbit.terminal.commands import health_text, help_text, reset_session, runtime_status, set_max_tokens, think_mode_text, tools_text
from orbit.terminal.config import AppConfig
from orbit.terminal.context_status import context_status_text
from orbit.terminal.history import PromptHistory
from orbit.terminal.prefill import MIN_PREFILL_ESTIMATE_SECONDS, estimate_prefill_tokens, estimate_prefill_tokens_after_tool_result
from orbit.terminal.prefill_estimator import CHAT_PREFILL_PROFILE, FINAL_FROM_TOOL_PREFILL_PROFILE, TOOL_PREFILL_PROFILE, PrefillEstimator, prefill_profile_for_phase
from orbit.terminal.repl_input import clear_input_echo, read_prompt_input, replace_input_echo
from orbit.terminal.session_preview import format_recent_session_messages, has_existing_session_context
from orbit.terminal.status import estimate_context_status_tokens, format_memory_refresh, format_turn_status
from orbit.terminal.streaming import StreamRenderer
from orbit.terminal.tool_events import format_tool_call_event, format_tool_result_event
from orbit.terminal.tool_mode import USAGE, ToolSpec, allowed_tool_names_for_spec, normalize_tool_spec, tools_are_enabled
from orbit.terminal.theme import danger, dim
from orbit.runtime.thinking_mode import ThinkingMode
from orbit.runtime.turn_trace import ModelPhaseStart


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
    _last_phase_label: str | None = None

    def __post_init__(self) -> None:
        if self.tools_mode is None:
            self.tools_mode = self.config.tools
        self.backend.thinking = self.config.think

    def run(self) -> int:
        if self.history:
            self.history.load()
        print(dim("orbit interactive mode. Type /help for commands."))
        print(dim(f"tools: {self.tools_mode}"))
        print(dim(f"think: {'on' if self.config.think else 'off'}"))
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
        self._last_phase_label = None
        tools_enabled = tools_are_enabled(self.tools_mode or "off")
        system_prompt = ROUTE_SYSTEM_PROMPT if tools_enabled else CHAT_SYSTEM_PROMPT
        prefill_tokens = estimate_prefill_tokens(self.runtime.messages, prompt, system_prompt=system_prompt)
        prefill_profile = _prefill_profile_for_turn(self.runtime.messages, tools_enabled=tools_enabled)
        prefill_seconds = self.prefill_estimator.estimate_seconds(prefill_tokens, profile=prefill_profile)
        renderer = StreamRenderer(
            prefill_estimate_seconds=_visible_prefill_seconds(prefill_seconds),
            prefill_estimate_tokens=prefill_tokens,
            thinking=self.config.think,
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
                    on_progress=renderer.progress,
                    on_tool_call=lambda name, args: renderer.event(format_tool_call_event(name, args), restart_timer=False),
                    on_tool_result=lambda name, chars, source, content: self._show_tool_result(renderer, name, chars, source, content),
                    on_model_step=self._record_model_step,
                    on_phase_start=lambda phase: self._record_phase_start(renderer, phase),
                )
            else:
                result = self.runtime.ask_chat(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    on_final_delta=renderer.write,
                    on_progress=renderer.progress,
                    on_model_step=self._record_model_step,
                    on_phase_start=lambda phase: self._record_phase_start(renderer, phase),
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
        self.can_continue = self.runtime.can_continue_last_response() or (
            ThinkingMode(enabled=self.config.think).continuation_kind_for(
                content=result.content,
                finish_reason=result.finish_reason,
            )
            is not None
        )
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
        if self.can_continue:
            if result.finish_reason == "length":
                message = _length_footer_message(self.config.think)
            else:
                message = "reasoning finished without a complete final answer"
            print(dim(message), flush=True)
            print(dim("/continue       continue the answer"), flush=True)
            print(dim("/max-tokens N   increase output budget"), flush=True)

    def _record_model_step(self, metrics) -> None:
        self.prefill_estimator.update(
            prompt_tokens=metrics.prompt_tokens,
            prompt_tokens_per_second=metrics.prompt_tokens_per_second,
            profile=prefill_profile_for_phase(metrics.phase),
        )

    def _record_phase_start(self, renderer: StreamRenderer, phase: ModelPhaseStart) -> None:
        label = _phase_label(phase)
        renderer.set_phase_label(_phase_progress_label(phase))
        if not label or label == self._last_phase_label:
            return
        self._last_phase_label = label
        renderer.event(label, restart_timer=True)

    def _show_tool_result(self, renderer: StreamRenderer, name: str, chars: int, source: str | None, content: str | None) -> None:
        if content is not None:
            tokens = estimate_prefill_tokens_after_tool_result(self.runtime.messages, content)
            seconds = self.prefill_estimator.estimate_seconds(tokens, profile=FINAL_FROM_TOOL_PREFILL_PROFILE)
            renderer.set_prefill_estimate(_visible_prefill_seconds(seconds), tokens)
        renderer.event(format_tool_result_event(name, chars, source, content), trailing_blank_line=True)

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
        if command == "/think" or command.startswith("/think "):
            print(self._handle_think_command(command))
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

    def _handle_think_command(self, command: str) -> str:
        value = command.removeprefix("/think").strip().lower()
        if not value:
            return think_mode_text(self.config.think)
        if value not in {"on", "off"}:
            return "error: usage: /think [off|on]"
        self.config = replace(self.config, think=value == "on")
        self.backend.thinking = self.config.think
        self.runtime.thinking_mode = self.config.think
        return f"think: {value}"

    def _continue_last_answer(self) -> None:
        self.can_continue = self.can_continue or self.runtime.can_continue_last_response()
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
        self._last_phase_label = None
        prefill_tokens = estimate_prefill_tokens(self.runtime.messages, "")
        prefill_seconds = self.prefill_estimator.estimate_seconds(prefill_tokens)
        renderer = StreamRenderer(
            prefill_estimate_seconds=_visible_prefill_seconds(prefill_seconds),
            prefill_estimate_tokens=prefill_tokens,
            thinking=self.config.think,
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
                on_progress=renderer.progress,
                on_model_step=self._record_model_step,
                on_phase_start=lambda phase: self._record_phase_start(renderer, phase),
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


def _length_footer_message(thinking: bool) -> str:
    if thinking:
        return "thinking or final output stopped because max_tokens was reached"
    return "output stopped because max_tokens was reached"


def _phase_label(phase: ModelPhaseStart) -> str | None:
    suffix = ""
    if not phase.streamed:
        suffix = " (non-streaming)"

    if phase.phase == "tool_plan":
        return "phase: thinking"
    if phase.phase == "route":
        return f"phase: deciding tool use{suffix}"
    if phase.phase == "chat_final":
        pass_label = f" pass {phase.attempt}" if phase.attempt else ""
        return f"phase: final answer{pass_label}{suffix}"
    if phase.phase == "chat_final_retry":
        reason = " after length" if phase.reason == "length" else ""
        return f"phase: retrying final answer{reason}{suffix}"
    if phase.phase == "chat_final_completion_repair":
        if phase.reason == "reasoning_like":
            return f"phase: forced final answer only{suffix}"
        if phase.reason:
            return f"phase: repairing final answer ({phase.reason}){suffix}"
        return f"phase: repairing final answer{suffix}"
    if phase.phase == "tool_call":
        pass_label = f" pass {phase.attempt}" if phase.attempt else ""
        return f"phase: tool call{pass_label}{suffix}"
    if phase.phase == "tool_call_retry":
        if phase.reason == "tool_contract_retry":
            return "phase: retrying tool call after contract mismatch (non-streaming)"
        return f"phase: retrying tool call{suffix}"
    if phase.phase == "final_from_tool":
        pass_label = f" pass {phase.attempt}" if phase.attempt else ""
        return f"phase: final answer from tool result{pass_label}{suffix}"
    if phase.phase == "final_from_tool_retry":
        if phase.reason == "length":
            return f"phase: continuing final answer from tool result after length{suffix}"
        if phase.reason:
            return f"phase: final answer from tool result retry ({phase.reason}){suffix}"
        return f"phase: final answer from tool result retry{suffix}"
    if phase.phase == "final_from_tool_completion_repair":
        if phase.reason == "reasoning_like":
            return f"phase: forced final answer from tool result only{suffix}"
        if phase.reason:
            return f"phase: repairing final answer from tool result ({phase.reason}){suffix}"
        return f"phase: repairing final answer from tool result{suffix}"
    if phase.phase == "final_from_tool_compact_retry":
        if phase.reason:
            return f"phase: compact final answer retry ({phase.reason}){suffix}"
        return f"phase: compact final answer retry{suffix}"
    if phase.phase == "chat_continue_native":
        return f"phase: continuing after length{suffix}"
    return None


def _phase_progress_label(phase: ModelPhaseStart) -> str | None:
    if phase.phase == "tool_plan":
        return "thinking"
    if phase.phase == "route":
        return "tool decision"
    if phase.phase == "chat_final":
        if phase.attempt and phase.attempt > 1:
            return f"final answer #{phase.attempt}"
        return "final answer"
    if phase.phase == "chat_final_retry":
        if phase.reason == "length":
            return "final retry"
        return "final retry"
    if phase.phase == "chat_final_completion_repair":
        if phase.reason == "reasoning_like":
            return "forced final"
        return "repair final"
    if phase.phase == "tool_call":
        if phase.attempt and phase.attempt > 1:
            return f"tool call #{phase.attempt}"
        return "tool call"
    if phase.phase == "tool_call_retry":
        return "tool retry"
    if phase.phase == "final_from_tool":
        if phase.attempt and phase.attempt > 1:
            return f"tool final #{phase.attempt}"
        return "tool final"
    if phase.phase == "final_from_tool_retry":
        if phase.reason == "length":
            return "tool final continue"
        return "tool final retry"
    if phase.phase == "final_from_tool_completion_repair":
        if phase.reason == "reasoning_like":
            return "forced tool final"
        return "tool final repair"
    if phase.phase == "final_from_tool_compact_retry":
        return "compact retry"
    if phase.phase == "chat_continue_native":
        return "continue"
    return None


def _prefill_profile_for_turn(messages: list[dict[str, object]], *, tools_enabled: bool) -> str:
    if not tools_enabled:
        return CHAT_PREFILL_PROFILE
    if any(message.get("role") == "tool" for message in messages[-4:]):
        return FINAL_FROM_TOOL_PREFILL_PROFILE
    return TOOL_PREFILL_PROFILE
