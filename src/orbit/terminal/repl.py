from __future__ import annotations

import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Callable

from orbit.backend.base import ChatResult, StreamPromptMetrics
from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.analysis_runtime import (
    AnalysisRuntime,
    AutonomousRunResult,
    analysis_autonomy_enabled,
)
from orbit.runtime.context_manager import ContextAdmissionError
from orbit.runtime.kv_diag import instrument_backend
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT
from orbit.runtime.sessions import SessionStore
from orbit.runtime.workflow_mode import DEFAULT_WORKFLOW_MODE, WorkflowMode
from orbit.terminal.analysis_mode import (
    AnalysisModeError,
    format_analysis_step,
    format_step_diagnostics,
    open_analysis_session,
    open_confined_analysis_session,
)
from orbit.terminal.compact_reports import format_memory_compaction_report, format_tool_compaction_report
from orbit.terminal.command_actions import CommandAction, build_list_action, build_models_action, build_read_action, build_search_action
from orbit.terminal.command_registry import resolve_command
from orbit.terminal.commands import health_text, help_text, props_text, reset_session, runtime_status, set_max_tokens, think_mode_text, tools_text
from orbit.terminal.config import AppConfig
from orbit.terminal.context_status import context_status_text
from orbit.terminal.history import PromptHistory
from orbit.terminal.prefill import MIN_PREFILL_ESTIMATE_SECONDS, estimate_prefill_tokens, estimate_prefill_tokens_after_tool_result
from orbit.terminal.prefill_estimator import CHAT_PREFILL_PROFILE, FINAL_FROM_TOOL_PREFILL_PROFILE, TOOL_PREFILL_PROFILE, PrefillEstimator, prefill_profile_for_phase
from orbit.terminal.repl_input import clear_input_echo, read_prompt_input, replace_input_echo
from orbit.terminal.runtime_status import collect_runtime_status, format_startup_banner
from orbit.terminal.session_preview import format_recent_session_messages, has_existing_session_context
from orbit.terminal.status import (
    TokenUsageAccumulator,
    TurnTokenUsage,
    estimate_context_status_tokens,
    format_memory_refresh,
    format_session_token_usage,
    format_turn_status,
    summarize_turn_token_usage,
)
from orbit.terminal.streaming import StreamRenderer
from orbit.terminal.tool_events import format_tool_activity_label, format_tool_call_event, format_tool_result_event
from orbit.terminal.tool_mode import USAGE, ToolSpec, allowed_tool_names_for_spec, normalize_tool_spec, tools_are_enabled
from orbit.terminal.theme import dim, runtime_error_text, sanitize_terminal_text
from orbit.runtime.thinking_mode import ThinkingMode
from orbit.runtime.turn_trace import ModelPhaseStart, ModelStepMetrics


def _abbreviate_home(path: Path) -> str:
    """`~/...` for paths under HOME, because the absolute form is noise."""
    try:
        relative = Path(path).resolve().relative_to(Path.home())
    except (ValueError, RuntimeError, OSError):
        return str(path)
    return "~" if str(relative) == "." else f"~/{relative}"


def _print_orbit_summary(summary: str) -> str:
    """Print one Orbit-authored telemetry line, set apart from the answer above.

    The line Orbit writes after an analysis -- `analysis | mode: ANALYSIS |
    ...` -- sits directly under the model's closing report, and both are plain
    left-aligned prose. `dim()` used to be the only thing telling them apart,
    and it is a no-op wherever ANSI is unavailable: a pipe, a redirected log,
    `NO_COLOR`, `TERM=dumb`. In those places the analyst reads Orbit's own
    counters as the model's last sentence, or the model's last sentence as
    Orbit's counters -- and a report is free to contain a line that looks
    exactly like the summary.

    A blank line and the `›` prefix already mean "Orbit is speaking" elsewhere
    in this terminal, so the separation reuses them instead of introducing new
    vocabulary. The blank line survives the loss of colour; the prefix survives
    both that and the line being copied out of context.

    One blank line, unconditionally. A report cannot arrive already ending in
    one: `AnalysisReport.text` is stripped at construction, and the two other
    values it can hold -- the no-evidence constant and the no-usable-text
    fallback -- are newline-free literals. Sanitizing does not add a trailing
    newline, but it does preserve one, so the strip is what makes this safe.
    An earlier version took a flag for the ends-in-a-newline case, but the
    flag could never be true, which left the branch untestable and the caller
    carrying state for a situation that does not occur.

    Returns what was printed so a test can assert on it without capturing
    stdout.
    """
    line = f"› {summary}"
    print()
    print(dim(line), flush=True)
    return line


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
    workflow_mode: WorkflowMode = DEFAULT_WORKFLOW_MODE
    # Runs between acquiring the artifact's bytes and storing them. Only a
    # test sets this; it exists so the swap window can be exercised
    # deterministically instead of raced for.
    _analysis_acquired_hook: Callable[[], None] | None = field(default=None, repr=False)
    _announced_workdir: str | None = field(default=None, repr=False)
    analysis: AnalysisRuntime | None = field(default=None, repr=False)
    turn_model_steps: list[ModelStepMetrics] = field(default_factory=list, repr=False)
    turn_backend_token_usage: TokenUsageAccumulator = field(default_factory=TokenUsageAccumulator, repr=False)
    session_token_usage: TokenUsageAccumulator = field(default_factory=TokenUsageAccumulator, repr=False)
    backend_usage_observer_installed: bool = field(default=False, init=False, repr=False)
    prompt_gap_pending: bool = field(default=True, init=False, repr=False)
    prompt_redisplay_pending: bool = field(default=False, init=False, repr=False)
    # Whether an analysis may continue itself, for this process only.
    #
    # Read once at startup from the runtime's own gate rather than consulted
    # per turn, so `/autonomous` can override the environment without writing
    # to it: the setting belongs to this session, and a later analysis in the
    # same process should honour what the analyst last chose rather than what
    # they exported before starting. `None` means "not yet resolved" and is
    # replaced in __post_init__; it is not a third state.
    autonomous_analysis: bool | None = field(default=None)

    def __post_init__(self) -> None:
        if self.autonomous_analysis is None:
            self.autonomous_analysis = analysis_autonomy_enabled()
        if self.tools_mode is None:
            self.tools_mode = self.config.tools
        self.backend.thinking = self.config.think
        set_result_observer = getattr(self.backend, "set_result_observer", None)
        if callable(set_result_observer):
            set_result_observer(self._record_backend_result)
            self.backend_usage_observer_installed = True
        set_failure_observer = getattr(self.backend, "set_failure_observer", None)
        if callable(set_failure_observer):
            set_failure_observer(self._record_backend_failure)
        set_aborted_observer = getattr(self.backend, "set_aborted_observer", None)
        if callable(set_aborted_observer):
            set_aborted_observer(self._record_backend_abort)

    def run(self) -> int:
        if self.history:
            self.history.load()
        status = collect_runtime_status(self.runtime, self.config, self.backend, tools_mode=self.tools_mode)
        for line in format_startup_banner(status).splitlines():
            print(dim(line))
        self._announce_workdir()
        if has_existing_session_context(self.runtime.messages):
            print(dim("recent session context:"))
            for line in format_recent_session_messages(self.runtime.messages):
                print(dim(line))
        while True:
            try:
                prompt = self._read_next_prompt().strip()
            except EOFError:
                return self._finish_interactive_session(0, leading_newline=True)
            except KeyboardInterrupt:
                return self._finish_interactive_session(130, leading_newline=True)
            if not prompt:
                continue
            # The marker this line was actually displayed with. A command may
            # change the mode, so re-deriving it after the fact would erase
            # the wrong number of rows.
            echo_label = self._prompt_label()
            if prompt.startswith("/"):
                clear_input_echo(prompt, echo_label)
                self.prompt_redisplay_pending = True
                if self._handle_command(prompt):
                    self.prompt_gap_pending = True
                    continue
                return self._finish_interactive_session(0)
            if self.history:
                resolution = self.history.resolve_prompt(prompt)
                if resolution.missing_full_text:
                    print("error: full pasted text is unavailable for this history entry", file=sys.stderr)
                    self.prompt_gap_pending = True
                    continue
                if resolution.prompt != prompt:
                    prompt = resolution.prompt
                else:
                    replace_input_echo(prompt, echo_label)
            else:
                replace_input_echo(prompt, echo_label)
            if self.history:
                self.history.add(prompt)
                self.history.save()
            self._ask(prompt)
            self.prompt_gap_pending = True

    def _announce_workdir(self) -> None:
        """Say where this session is working, once.

        Called at startup only. `config.workdir` is resolved when the config
        loads and nothing mutates it afterwards -- there is no `/cd` -- so a
        per-turn change check would be a string comparison that can never
        fire. Guarded anyway, so calling it twice stays harmless.

        Display only: the model's view of the workdir is whatever the runtime
        already sends, and this neither adds to nor alters it.
        """
        current = str(self.config.workdir)
        if current == self._announced_workdir:
            return
        self._announced_workdir = current
        print(dim(f"workdir: {_abbreviate_home(self.config.workdir)}"), flush=True)

    def _prompt_label(self) -> str:
        """The marker for the runtime that owns the next line.

        Read straight off `workflow_mode`, the state the Repl already keeps,
        so the marker cannot drift from the mode it names. Display only: it
        is never appended to history and never reaches the backend.
        """
        return "analysis" if self.workflow_mode is WorkflowMode.ANALYSIS else "chat"

    def _read_next_prompt(self) -> str:
        if self.prompt_gap_pending:
            print(flush=True)
            self.prompt_gap_pending = False
        label = self._prompt_label()
        if self.prompt_redisplay_pending:
            self.prompt_redisplay_pending = False
            return read_prompt_input(redisplay=True, label=label)
        return read_prompt_input(label=label)

    def _ask(self, prompt: str, *, command_action: CommandAction | None = None) -> None:
        # The mode decides which runtime owns this line before anything else
        # happens. ANALYSIS returns here, so no CHAT machinery -- route,
        # tools, finalization -- is reached while a session is open.
        if self.workflow_mode is WorkflowMode.ANALYSIS:
            if self.analysis is None:
                # The two are set together everywhere; if they ever diverge,
                # say so rather than quietly answering as CHAT.
                print("error: analysis session unavailable: returning to CHAT", file=sys.stderr)
                self.workflow_mode = DEFAULT_WORKFLOW_MODE
                return
            self._ask_analysis(prompt)
            return
        self.turn_model_steps.clear()
        self.turn_backend_token_usage = TokenUsageAccumulator()
        tools_enabled = tools_are_enabled(self.tools_mode or "off")
        system_prompt = ROUTE_SYSTEM_PROMPT if tools_enabled else CHAT_SYSTEM_PROMPT
        prefill_tokens = estimate_prefill_tokens(self.runtime.messages, prompt, system_prompt=system_prompt)
        prefill_profile = _prefill_profile_for_turn(self.runtime.messages, tools_enabled=tools_enabled)
        prefill_seconds = self.prefill_estimator.estimate_seconds(prefill_tokens, profile=prefill_profile)
        renderer = StreamRenderer(
            prefill_estimate_seconds=_visible_prefill_seconds(prefill_seconds),
            prefill_estimate_tokens=prefill_tokens,
            thinking=self.config.think,
            render_markdown_mode=self.config.render_markdown,
        )
        checkpoint = len(self.runtime.messages)
        print()
        started = time.monotonic()
        renderer.start()
        try:
            if command_action is not None and command_action.full_document_path is not None:
                result = self.runtime.answer_full_document_command(
                    prompt,
                    path=command_action.full_document_path,
                    workdir=self.config.workdir,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    on_final_delta=renderer.write,
                    on_progress=renderer.progress,
                    on_model_step=self._record_model_step,
                    on_phase_start=lambda phase: self._record_phase_start(renderer, phase),
                )
            elif command_action is not None and command_action.evidence is not None:
                result = self.runtime.answer_from_acquired_evidence(
                    prompt,
                    evidence=command_action.evidence,
                    workdir=self.config.workdir,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    on_final_delta=renderer.write,
                    on_progress=renderer.progress,
                    on_model_step=self._record_model_step,
                    on_phase_start=lambda phase: self._record_phase_start(renderer, phase),
                )
            elif tools_are_enabled(self.tools_mode or "off"):
                result = self.runtime.ask_auto(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    workdir=self.config.workdir,
                    allowed_tool_names=allowed_tool_names_for_spec(self.tools_mode or "off"),
                    on_final_delta=renderer.write,
                    on_progress=renderer.progress,
                    on_tool_call=lambda name, args: renderer.event(
                        format_tool_call_event(name, args),
                        next_activity=("tool", format_tool_activity_label(name, args)),
                    ),
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
            renderer.finish(interrupted=True)
            self.runtime.restore_message_count(checkpoint)
            print(dim("cancelled"), flush=True)
            return
        except (ContextAdmissionError, LlamaServerError, TimeoutError) as exc:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(runtime_error_text(exc), file=sys.stderr)
            return
        except Exception:
            renderer.finish()
            raise
        renderer.finish()
        if self.workflow_mode is WorkflowMode.CHAT and self.runtime.last_analysis_request:
            # The route asked for artifact analysis. Nothing was answered and
            # no tool ran, so the analyst's original request is still owed a
            # reply -- it becomes the first analyst step of the new session.
            if self._enter_analysis_from_route(self.runtime.last_analysis_request):
                self._ask_analysis(prompt)
                return
            # Refused: the route turn answered nothing, so rewinding leaves no
            # unanswered user turn behind for every later prompt to carry.
            self.runtime.restore_message_count(checkpoint)
            self.runtime.last_analysis_request = None
            return
        self._save_session()
        elapsed = time.monotonic() - started
        print("\n\n", end="", flush=True)
        self._print_turn_footer(result, elapsed_seconds=elapsed)

    def _ask_analysis(self, analyst_message: str) -> None:
        """One analyst line -> exactly one `AnalysisRuntime.step()`.

        There is no loop and no retry: whatever the step returns is printed
        and control is back with the analyst. Steering, `continue`, and any
        other text are all just the next analyst message, passed verbatim.
        """
        assert self.analysis is not None
        print()
        started = time.monotonic()
        # The same renderer CHAT uses. An analysis step is one long model call
        # with no intermediate output, so without this the terminal shows
        # nothing at all until the step returns -- a wait that is
        # indistinguishable from a hang. `thinking=False` because ANALYSIS
        # never displays reasoning.
        renderer = StreamRenderer(
            thinking=False,
            render_markdown_mode="plain",
        )
        renderer.set_activity("analysis")
        # `step()` appends the analyst line before it calls the model, so a
        # cancelled or failed step would otherwise leave an unanswered user
        # turn in the history. CHAT rewinds to a checkpoint for the same
        # reason; ANALYSIS has more at stake, because that history is the
        # append-only record a later exact-prefix strategy depends on.
        checkpoint = self._analysis_checkpoint()
        renderer.start()
        run: AutonomousRunResult | None = None
        try:
            if self.autonomous_analysis:
                # Same step primitive, issued repeatedly while each one adds
                # verifiable state. `step()` still owns every guarantee; the
                # loop only decides whether to ask again.
                #
                # Each completed step is rendered as it lands. Showing only the
                # last one would hide every action an autonomous run took --
                # and its terminating step is usually prose, so the analyst
                # would be told actions ran and given no evidence id to cite or
                # re-attest.
                def show(step, record) -> None:
                    # The wait line is still being redrawn in place. End it
                    # first, or this block lands on the same row and the
                    # elapsed duration is overwritten instead of kept.
                    renderer.settle_progress_line()
                    block = format_analysis_step(
                        step, prose_already_shown=renderer.rendered_visible_text
                    )
                    if renderer.rendered_visible_text:
                        print(flush=True)
                    if block:
                        print(block, flush=True)
                    renderer.reset_visible_text()

                run = self.analysis.run_autonomous(
                    analyst_message,
                    on_progress=renderer.progress,
                    on_delta=renderer.write,
                    on_step=show,
                )
                result = run.last_step
                if result is None:
                    # No step completed -- cancelled, a backend failure, or a
                    # zero call budget -- so there is nothing to render and the
                    # pre-run turn is safe to undo. Why it ended still has to be
                    # told truthfully: a backend that died must not be reported
                    # as an analyst interrupt.
                    renderer.finish(interrupted=True)
                    self._restore_analysis_checkpoint(checkpoint)
                    if run.cancelled:
                        print(dim("cancelled"), flush=True)
                    else:
                        print(dim(run.stop_reason), flush=True)
                    return
            else:
                result = self.analysis.step(
                    analyst_message,
                    on_progress=renderer.progress,
                    on_delta=renderer.write,
                )
        except KeyboardInterrupt:
            renderer.finish(interrupted=True)
            self._restore_analysis_checkpoint(checkpoint)
            print(dim("cancelled"), flush=True)
            return
        except (ContextAdmissionError, LlamaServerError, TimeoutError) as exc:
            # Recoverable: the analyst keeps the session and can steer again.
            renderer.finish(interrupted=True)
            self._restore_analysis_checkpoint(checkpoint)
            print(runtime_error_text(exc), file=sys.stderr)
            return
        except BaseException:
            # Anything else ends the process, as it does in CHAT. CHAT leaves
            # nothing behind when it does; an analysis session would leave a
            # temporary workspace, so release it before the exception goes up.
            renderer.finish(interrupted=True)
            self._close_analysis()
            raise
        # A guided step prints its block after the renderer has stopped, so
        # it asks for the finished progress line to be kept rather than
        # settling over a live one.
        if run is None:
            renderer.keep_progress_line()
        renderer.finish()
        # The renderer already showed the prose if the backend streamed it;
        # the final block then carries only what streaming could not: action
        # status, evidence preview, artifacts.
        prose_already_shown = renderer.rendered_visible_text
        # An autonomous run already rendered every step through `on_step`.
        step_block = (
            ""
            if run is not None
            else format_analysis_step(result, prose_already_shown=prose_already_shown)
        )
        if prose_already_shown and run is None:
            # Streamed deltas leave the cursor mid-line; close that line so the
            # block below starts on its own. Previously the reprinted copy of
            # the prose supplied this break.
            print(flush=True)
        if step_block:
            print(step_block, flush=True)
        elapsed = time.monotonic() - started
        if run is not None:
            # The closing report streamed as it was generated, like any other
            # prose. Print it here only if it did not: a backend that does not
            # stream would otherwise produce a run whose grounded conclusion
            # never reached the analyst at all.
            #
            # Sanitized because this print is the one that did not pass the
            # renderer. The streamed copy crosses that boundary on its way out
            # and the empty-report branch sanitizes too, so without this the
            # report is safe exactly when a backend happens to stream it. The
            # text itself is untouched: `run.final_report.text` still holds
            # what the model wrote.
            if run.final_report is not None and not renderer.rendered_visible_text:
                print(
                    sanitize_terminal_text(
                        run.final_report.text, allow_newlines=True
                    ),
                    flush=True,
                )
            replans = f" | replans: {run.replans}" if run.replans else ""
            summary = (
                f"analysis | mode: ANALYSIS | model calls: {run.model_calls} | "
                f"actions: {run.actions_executed} | steps: {len(run.steps)}{replans} | "
                f"stopped: {run.stop_reason} | {elapsed:.1f}s"
            )
        else:
            summary = (
                f"analysis | mode: ANALYSIS | model calls: {result.model_calls} | "
                f"actions: {1 if result.action_executed else 0} | {elapsed:.1f}s"
            )
        detail = format_step_diagnostics(result.diagnostics)
        if detail:
            summary += f" | {detail}"
        _print_orbit_summary(summary)

    def _analysis_checkpoint(self) -> tuple[int, int]:
        """History length and analyst-turn count before a step is attempted."""
        assert self.analysis is not None
        return len(self.analysis.messages), self.analysis.analyst_turns

    def _restore_analysis_checkpoint(self, checkpoint: tuple[int, int]) -> None:
        """Undo a step that produced nothing, so the record stays truthful.

        A step that never reached the model must not leave a turn behind: the
        `turn_N` in evidence provenance counts analyst turns, and a counter
        that advances on failed attempts would name turns that produced no
        evidence at all.
        """
        if self.analysis is None:
            return
        message_count, analyst_turns = checkpoint
        del self.analysis.messages[message_count:]
        self.analysis.analyst_turns = analyst_turns

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
                    turn_token_usage=self._turn_token_usage(),
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

    def _record_model_step(self, metrics: ModelStepMetrics) -> None:
        self.turn_model_steps.append(metrics)
        if not self.backend_usage_observer_installed:
            self.session_token_usage.add(metrics)
        self.prefill_estimator.update(
            prompt_tokens=metrics.prompt_tokens,
            prompt_tokens_per_second=metrics.prompt_tokens_per_second,
            profile=prefill_profile_for_phase(metrics.phase),
        )

    def _record_backend_result(self, result: ChatResult) -> None:
        self.turn_backend_token_usage.add_result(result)
        self.session_token_usage.add_result(result)

    def _record_backend_failure(self) -> None:
        self.turn_backend_token_usage.add_failed_call()
        self.session_token_usage.add_failed_call()

    def _record_backend_abort(self, prompt_metrics: StreamPromptMetrics | None) -> None:
        # The same snapshot goes to both ledgers: whatever prefill measured
        # before the stream was stopped, or nothing at all if it got no further.
        self.turn_backend_token_usage.add_aborted_call(prompt_metrics)
        self.session_token_usage.add_aborted_call(prompt_metrics)

    def _turn_token_usage(self) -> TurnTokenUsage | None:
        if self.backend_usage_observer_installed:
            return self.turn_backend_token_usage.snapshot()
        return summarize_turn_token_usage(self.turn_model_steps)

    def _record_phase_start(self, renderer: StreamRenderer, phase: ModelPhaseStart) -> None:
        renderer.set_phase_label(_phase_progress_label(phase))
        renderer.set_final_output_mode(_phase_starts_final_output(phase))

    def _show_tool_result(self, renderer: StreamRenderer, name: str, chars: int, source: str | None, content: str | None) -> None:
        if content is not None:
            tokens = estimate_prefill_tokens_after_tool_result(self.runtime.messages, content)
            seconds = self.prefill_estimator.estimate_seconds(tokens, profile=FINAL_FROM_TOOL_PREFILL_PROFILE)
            renderer.set_prefill_estimate(_visible_prefill_seconds(seconds), tokens)
        renderer.event(
            format_tool_result_event(name, chars, source, content),
            trailing_blank_line=True,
            next_activity=("model", "working"),
        )

    def _handle_command(self, command: str) -> bool:
        invocation = resolve_command(command)
        if invocation is None:
            print(f"error: unknown command: {command}", file=sys.stderr)
            return True
        handler = invocation.spec.handler
        arguments = invocation.arguments
        if handler == "exit":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
                return True
            return False
        if handler == "read":
            return self._handle_data_action(build_read_action(arguments, workdir=self.config.workdir))
        if handler == "search":
            return self._handle_data_action(build_search_action(arguments, workdir=self.config.workdir))
        if handler == "list":
            return self._handle_data_action(build_list_action(arguments, workdir=self.config.workdir))
        if handler == "models":
            return self._handle_data_action(build_models_action(arguments))
        if handler == "continue":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
                return True
            self._continue_last_answer()
            return True
        if handler == "help":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
                return True
            print(help_text())
            return True
        if handler == "clear":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
            elif sys.stdout.isatty():
                print("\033[2J\033[H", end="", flush=True)
            else:
                print("terminal display clear unavailable in non-TTY mode")
            return True
        if handler == "reset":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
                return True
            print(reset_session(self.runtime, self.session))
            self._close_analysis()
            self.workflow_mode = DEFAULT_WORKFLOW_MODE
            self._reset_token_usage()
            self.can_continue = False
            return True
        if handler == "compact" and not arguments:
            print(format_memory_compaction_report(self.runtime.compact_memory_now(temperature=self.config.temperature)))
            self._save_session()
            return True
        if handler == "compact" and arguments == "tools":
            print(format_tool_compaction_report(self.runtime.compact_old_tool_results(temperature=self.config.temperature)))
            self._save_session()
            return True
        if handler == "compact":
            print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
            return True
        if handler == "sessions" and arguments == "clear":
            print(self._clear_workdir_sessions())
            return True
        if handler == "sessions":
            print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
            return True
        if handler == "health":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
                return True
            print(health_text(self.backend, self.config))
            return True
        if handler == "max_tokens":
            self.config, message = set_max_tokens(self.config, arguments)
            print(message)
            return True
        if handler == "autonomous":
            print(self._handle_autonomous_command(arguments))
            return True
        if handler == "think":
            print(self._handle_think_command(f"/think {arguments}".rstrip()))
            return True
        if handler == "status" and not arguments:
            print(runtime_status(self.runtime, self.config, self.backend, tools_mode=self.tools_mode))
            print(format_session_token_usage(self.session_token_usage.snapshot()))
            return True
        if handler == "status" and arguments in {"ctx", "context"}:
            print(context_status_text(self.runtime.messages, context_tokens=self.runtime.context_tokens))
            return True
        if handler == "status":
            print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
            return True
        if handler == "props":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
            else:
                print(props_text(self.backend))
            return True
        if handler == "tools":
            print(self._handle_tools_command(f"/tools {arguments}".rstrip()))
            return True
        if handler == "analysis":
            print(self._handle_analysis_command(arguments))
            return True
        if handler == "report":
            self._handle_report_command(arguments)
            return True
        if handler == "chat":
            if arguments:
                print(self._command_usage_error(invocation.spec.usage), file=sys.stderr)
                return True
            print(self._handle_chat_command())
            return True
        print(f"error: command handler unavailable: {invocation.spec.name}", file=sys.stderr)
        return True

    def _handle_data_action(self, action: CommandAction) -> bool:
        if action.needs_model:
            assert action.prompt is not None
            self._ask(action.prompt, command_action=action)
        else:
            # A command's output carries content Orbit did not author: file
            # bytes, and filenames from whatever directory is being examined.
            # A crafted name is enough to move the cursor or erase the screen,
            # so it is neutralized here, where it is displayed. `action.output`
            # itself is untouched, and the model never sees it -- a command
            # that needs one sends `prompt` and `evidence` instead.
            print(sanitize_terminal_text(action.output, allow_newlines=True))
        return True

    @staticmethod
    def _command_usage_error(usage: str) -> str:
        return f"error: usage: {usage}"

    def _clear_workdir_sessions(self) -> str:
        if not _confirm_clear_sessions():
            return "sessions clear cancelled"
        removed = SessionStore.clear_for_workdir(self.config.workdir)
        self.runtime.reset()
        self._close_analysis()
        self.workflow_mode = DEFAULT_WORKFLOW_MODE
        self._reset_token_usage()
        self.can_continue = False
        self.session = SessionStore.new_for_workdir(self.config.workdir)
        return f"sessions cleared: {removed}"

    def _reset_token_usage(self) -> None:
        self.turn_model_steps.clear()
        self.turn_backend_token_usage = TokenUsageAccumulator()
        self.session_token_usage = TokenUsageAccumulator()

    def _handle_tools_command(self, command: str) -> str:
        value = command.removeprefix("/tools").strip().lower()
        if not value:
            return tools_text(self.tools_mode)
        if value == "status":
            return self.runtime.local_capabilities.format_tools_status()
        if value == "refresh":
            capabilities = self.runtime.refresh_local_capabilities()
            return "tools capabilities refreshed\n" + capabilities.format_tools_status()
        try:
            self.tools_mode = normalize_tool_spec(value)
        except ValueError:
            return f"error: usage: /tools [{USAGE}]"
        if self.tools_mode:
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

    def _handle_autonomous_command(self, arguments: str) -> str:
        """Show or set autonomous analysis for this session.

        The terminal owns the switch, not the policy: what autonomy means, when
        it stops, and what it costs all stay in the runtime. This decides only
        whether the next analysis turn asks for it.

        Nothing is written to the environment and nothing is persisted. An
        override lives for this process, so a later `orbit` starts from the
        environment again -- an analyst who turned autonomy on to finish one
        artifact does not silently keep it on tomorrow.
        """
        value = arguments.strip().lower()
        if not value:
            return f"autonomous analysis: {'on' if self.autonomous_analysis else 'off'}"
        if value not in {"on", "off"}:
            # The state is left exactly as it was: a mistyped argument must not
            # decide whether an analysis runs itself.
            return "error: usage: /autonomous [off|on]"
        self.autonomous_analysis = value == "on"
        if self.autonomous_analysis:
            return (
                "autonomous analysis: on -- an analysis continues by itself "
                "while each step adds new evidence"
            )
        return "autonomous analysis: off -- each analysis step returns to you"

    def _enter_analysis_from_route(self, artifact: str) -> bool:
        """Open an analysis session the route asked for. True if it opened.

        The path came from the model, not the analyst, so it is acquired
        rather than merely checked: opened once under the workdir, following
        no symlink on any component, with the bytes read from that same
        descriptor. An earlier version validated the name and let the session
        reopen it, which left a window in which the name could be pointed at
        an outside file between the two -- reproducibly, on a few racing turns
        in a hundred. Nothing here hands a pathname on to be opened again.

        Explicit `/analysis` keeps its own, wider policy: the analyst typed
        that path and may legitimately name anything on the machine.
        """
        try:
            runtime = open_confined_analysis_session(
                artifact,
                backend=self._analysis_backend(),
                workdir=self.config.workdir,
                evidence_store_factory=self._analysis_evidence_store,
                on_acquired=self._analysis_acquired_hook,
            )
        except AnalysisModeError as exc:
            print(f"error: refusing analysis: {exc}", file=sys.stderr)
            return False
        self._close_analysis()
        self.analysis = runtime
        self.workflow_mode = WorkflowMode.ANALYSIS
        self.can_continue = False
        source = runtime.source
        print(
            dim(
                f"mode: ANALYSIS | {source.original_path} "
                f"({source.size_bytes} bytes, sha256 {source.sha256})"
            ),
            flush=True,
        )
        return True

    def _analysis_backend(self):
        """The backend an ANALYSIS session should call.

        `ChatRuntime` wraps its backend with the KV diagnostic recorder, but the
        REPL holds the raw `LlamaServerBackend` it was constructed with, and the
        analysis session was being handed that one -- so ANALYSIS model calls
        produced no `kv_diag` records at all while CHAT calls did.

        `instrument_backend` returns the backend unchanged when diagnostics are
        disabled and is idempotent when they are on, so this adds no wrapper and
        no behaviour to an ordinary run; it only restores correlation (phase and
        model_call_id) for the calls an analysis actually makes.
        """
        return instrument_backend(self.backend)

    def _handle_analysis_command(self, arguments: str) -> str:
        """Enter ANALYSIS on one artifact. No model call happens here."""
        try:
            runtime = open_analysis_session(
                arguments,
                backend=self._analysis_backend(),
                workdir=self.config.workdir,
                evidence_store_factory=self._analysis_evidence_store,
            )
        except AnalysisModeError as exc:
            # A refused session leaves the current mode untouched: a typo must
            # not silently drop the analyst out of the session they were in.
            return f"error: {exc}"
        self._close_analysis()
        self.analysis = runtime
        self.workflow_mode = WorkflowMode.ANALYSIS
        self.can_continue = False
        source = runtime.source
        return (
            f"mode: ANALYSIS | {source.original_path} "
            f"({source.size_bytes} bytes, sha256 {source.sha256})"
        )

    def _handle_report_command(self, question: str) -> None:
        """`/report` -- answer from the evidence already collected.

        Valid only inside a session, because there is nothing to report on
        otherwise. It runs no action, so the analysis history and the evidence
        store are both left exactly as they were; what it returns is displayed
        and not recorded.
        """
        if self.workflow_mode is not WorkflowMode.ANALYSIS or self.analysis is None:
            print(
                "error: /report needs an analysis session; start one with /analysis <path>",
                file=sys.stderr,
            )
            return
        print()
        started = time.monotonic()
        renderer = StreamRenderer(thinking=False, render_markdown_mode="plain")
        renderer.set_activity("analysis")
        renderer.start()
        try:
            report = self.analysis.report(
                question,
                on_progress=renderer.progress,
                on_delta=renderer.write,
            )
        except KeyboardInterrupt:
            renderer.finish(interrupted=True)
            print(dim("cancelled"), flush=True)
            return
        except (ContextAdmissionError, LlamaServerError, TimeoutError) as exc:
            # Recoverable: a failed report leaves the session exactly as it was.
            renderer.finish(interrupted=True)
            print(runtime_error_text(exc), file=sys.stderr)
            return
        renderer.finish()
        if report.model_calls == 0:
            # Today this branch can only carry NO_EVIDENCE_REPORT, a constant.
            # It is sanitized anyway because it is a print that never passed
            # the renderer's boundary: if the branch ever comes to carry model
            # prose, it is already safe rather than newly vulnerable.
            print(sanitize_terminal_text(report.text, allow_newlines=True), flush=True)
        elapsed = time.monotonic() - started
        summary = (
            f"report | mode: ANALYSIS | model calls: {report.model_calls} | "
            f"actions: 0 | {elapsed:.1f}s"
        )
        detail = format_step_diagnostics(report.diagnostics)
        if detail:
            summary += f" | {detail}"
        _print_orbit_summary(summary)

    def _handle_chat_command(self) -> str:
        """Return to CHAT. No model call, and the analysis is kept.

        Closing the workspace here would destroy artifacts the analyst may
        still want to come back to within this process, and nothing about
        answering a chat question requires that. The session is released by
        the ordinary lifecycle instead: `/reset`, `/analysis` on another
        artifact, or leaving the REPL.
        """
        if self.workflow_mode is WorkflowMode.CHAT:
            return "mode: CHAT"
        self.workflow_mode = WorkflowMode.CHAT
        self.can_continue = False
        if self.analysis is not None:
            return "mode: CHAT | analysis session kept (/analysis <path> starts a new one)"
        return "mode: CHAT"

    def _analysis_evidence_store(self, workspace_root: Path) -> EvidenceStore:
        """A store of its own, per analysis session.

        Sharing CHAT's store looks economical and is not: CHAT reads that
        store to decide which route window to build and which evidence ids a
        final answer may cite, both keyed on the most recent record and on a
        `turn_N` string that each runtime mints from its own counter. An
        analysis record landing there truncates the next CHAT route window,
        advertises `execute_analysis` inside it, and can be authorised for
        citation by a CHAT turn that never produced it. The two evidence
        spaces are therefore kept apart, which is also what lets `/reset`
        clear CHAT without reaching into a live analysis session.
        """
        return EvidenceStore(root=workspace_root / "evidence")

    def _close_analysis(self) -> None:
        """Release the analysis workspace, if one is open. Idempotent."""
        if self.analysis is None:
            return
        self.analysis.close()
        self.analysis = None

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
            workflow_mode=str(self.workflow_mode),
        )

    def _save_history(self) -> None:
        if self.history:
            self.history.save()

    def _finish_interactive_session(self, code: int, *, leading_newline: bool = False) -> int:
        self._close_analysis()
        self._save_history()
        if leading_newline:
            print()
        print(dim(format_session_token_usage(self.session_token_usage.snapshot())), flush=True)
        return code

    def _ask_continue(self) -> None:
        self.turn_model_steps.clear()
        self.turn_backend_token_usage = TokenUsageAccumulator()
        prefill_tokens = estimate_prefill_tokens(self.runtime.messages, "")
        prefill_seconds = self.prefill_estimator.estimate_seconds(prefill_tokens)
        renderer = StreamRenderer(
            prefill_estimate_seconds=_visible_prefill_seconds(prefill_seconds),
            prefill_estimate_tokens=prefill_tokens,
            thinking=self.config.think,
            render_markdown_mode=self.config.render_markdown,
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
            renderer.finish(interrupted=True)
            self.runtime.restore_message_count(checkpoint)
            print(dim("cancelled"), flush=True)
            return
        except (ContextAdmissionError, LlamaServerError, TimeoutError) as exc:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(runtime_error_text(exc), file=sys.stderr)
            return
        except Exception:
            renderer.finish()
            raise
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


def _phase_starts_final_output(phase: ModelPhaseStart) -> bool:
    return phase.phase in {
        "chat_final",
        "chat_final_retry",
        "chat_final_completion_repair",
        "final_from_tool",
        "final_from_tool_retry",
        "final_from_tool_completion_repair",
        "final_from_tool_compact_retry",
        "chat_continue_native",
    }
