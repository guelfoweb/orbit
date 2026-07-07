from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable
import json
import re

from orbit.backend import ChatBackend, ChatResult
from orbit.backend.base import Message, StreamProgress
from orbit.runtime.capabilities import LocalCapabilities, discover_local_capabilities
from orbit.runtime.client_state import ClientState
from orbit.runtime.completion_budget import resolve_max_tokens
from orbit.runtime.environments import (
    ContinueEnvironment,
    FileInputEnvironment,
    FinalFromToolEnvironment,
    PureChatEnvironment,
    ToolLoopEnvironment,
    TransportEnvironment,
    is_empty_final_response as _is_empty_final_response,
    is_tool_argument_json_error as _is_tool_argument_json_error,
    merge_chat_results as _merge_chat_results,
    needs_final_completion_repair as _needs_final_completion_repair,
    pdf_extraction_repair_prompt as _pdf_extraction_repair_prompt,
    contradicts_successful_pdf_extraction as _contradicts_successful_pdf_extraction,
    unsupported_tool_mode_result as _unsupported_tool_mode_result,
)
from orbit.runtime.evidence import (
    EvidenceStore,
    build_compact_final_evidence_context,
    build_final_evidence_context,
    build_post_tool_route_evidence_context,
    build_route_evidence_context,
    build_web_final_evidence_context,
)
from orbit.runtime.final_policy import (
    has_list_like_tool_result as _has_list_like_tool_result,
)
from orbit.runtime.file_input_resolver import FileInputResolver
from orbit.runtime.kv_diag import emit_evidence_lineage, emit_route_outcome, instrument_backend, model_call_context, user_request
from orbit.runtime.media import AudioInput, ImageInput
from orbit.runtime.messages import (
    TOOL_CALL_JSON_RETRY_PROMPT,
    message_content,
    with_chat_system_prompt,
    with_final_tool_system_prompt,
    with_media_system_prompt,
    with_command_system_prompt,
    with_tool_call_system_prompt,
)

from orbit.runtime.command_request import (
    ToolRoute,
    command_stream_state,
    decision_tool_names,
    parse_command_decision,
    parse_command_decision_from_tool_calls,
    command_tool_call_from_content,
    command_tool_call_from_tool_calls,
)
from orbit.runtime.results import error_result
from orbit.runtime.session_memory import MemoryRefresh, maybe_refresh_memory
from orbit.runtime.shell_guardrails import is_metadata_only_shell_command
from orbit.runtime.thinking_mode import (
    ThinkingMode,
    contains_control_channel_markup,
    last_assistant_has_open_reasoning,
)
from orbit.runtime.tool_result_compaction import (
    ToolResultCompactionReport,
    compact_tool_results,
    persistent_messages as persistent_tool_result_messages,
)
from orbit.runtime.turn_trace import ModelPhaseStart, ModelStepMetrics


FINAL_FROM_TOOL_COMPACT_MIN_RAW_CHARS = 512
FINAL_SMALL_EVIDENCE_MAX_RAW_CHARS = 500
FINAL_SMALL_EVIDENCE_KINDS = {"shell", "unknown", "grep_search"}
_ROUTE_TOOL_RETRY_PROMPT_RE = re.compile(
    r"\b(?:search|online|web|url|http|https|read|show|print|inspect|analy[sz]e|review|list|find|grep|curl|fetch|directory|folder|file|path|pdf|html|source|current\s+directory|computer|system|specs?)\b|[/\\]|[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}\b",
    re.IGNORECASE,
)
_EXPLICIT_WEB_SEARCH_RE = re.compile(
    r"\b(?:search\s+(?:online|the\s+web|web)|web\s+search|look\s+up\s+online|find\s+online|cerca\s+(?:online|sul\s+web|nel\s+web))\b",
    re.IGNORECASE,
)
_FILE_CONTENT_ACTION_RE = re.compile(
    r"\b(?:read|show|print|explain|summari[sz]e|analy[sz]e|review|inspect|open|cat|leggi|spiega|riassumi|analizza|mostra)\b",
    re.IGNORECASE,
)
_FILE_LIKE_REFERENCE_RE = re.compile(
    r"(?:[\"'`])[^\"'`\n]+?\.[A-Za-z0-9]{1,8}(?:[\"'`])|(?:^|[\s(])[A-Za-z0-9_./-]+?\.[A-Za-z0-9]{1,8}(?=$|[\s),:;])",
    re.IGNORECASE,
)


@dataclass
class ChatRuntime:
    backend: ChatBackend
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    thinking_mode: bool = False
    client_state: ClientState = field(default_factory=ClientState)
    context_tokens: int | None = None
    last_memory_refresh: MemoryRefresh | None = None
    last_memory_refresh_message_count: int | None = None
    memory_refresh_cooldown_messages: int = 4
    memory_refreshes: int = 0
    total_memory_tokens_saved: int = 0
    last_memory_refresh_attempt: MemoryRefresh | None = None
    mutation_verifications: int = 0
    mutation_verification_repairs: int = 0
    mutation_verification_failures: int = 0
    completion_guard_nudges: int = 0
    completion_guard_commands: int = 0
    completion_guard_successes: int = 0
    completion_guard_failures: int = 0
    minimal_patch_guard_nudges: int = 0
    minimal_patch_guard_commands: int = 0
    minimal_patch_guard_successes: int = 0
    minimal_patch_guard_failures: int = 0
    mutation_semantic_repairs: int = 0
    mutation_semantic_repair_commands: int = 0
    mutation_semantic_repair_failures: int = 0
    content_evidence_guard_nudges: int = 0
    content_evidence_guard_commands: int = 0
    content_evidence_guard_successes: int = 0
    content_evidence_guard_failures: int = 0
    local_capabilities: LocalCapabilities = field(default_factory=discover_local_capabilities)
    diagnostic_session_id: str = "default"
    evidence_store: EvidenceStore | None = None

    def __post_init__(self) -> None:
        self.backend = instrument_backend(self.backend)
        if hasattr(self.backend, "thinking"):
            self.thinking_mode = bool(getattr(self.backend, "thinking"))
        if not self.messages and self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def refresh_local_capabilities(self) -> LocalCapabilities:
        self.local_capabilities = discover_local_capabilities()
        return self.local_capabilities

    @user_request
    def ask(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        images: list[ImageInput] | None = None,
        audios: list[AudioInput] | None = None,
        on_final_delta: Callable[[str], None] | None = None,
        on_progress: Callable[[StreamProgress], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
    ) -> ChatResult:
        user_content = message_content(prompt, images or [], audios or [])
        result = self._pure_chat_environment().ask_user_content(
            user_content,
            temperature=temperature,
            max_tokens=max_tokens,
            call_messages=self.messages,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=1,
        )
        return self._remember_visible_result(result)

    @user_request
    def ask_chat(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None = None,
        on_progress: Callable[[StreamProgress], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
    ) -> ChatResult:
        self.last_memory_refresh = None
        self.refresh_memory_if_needed(temperature=temperature)
        call_messages = with_chat_system_prompt([*self.messages, {"role": "user", "content": prompt}])
        result = self._pure_chat_environment().ask_user_content(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            call_messages=call_messages,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=1,
        )
        return self._remember_visible_result(result)

    @user_request
    def continue_last_response(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None = None,
        on_progress: Callable[[StreamProgress], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
    ) -> ChatResult:
        result = self._continue_environment().continue_last_response(
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
        )
        return self._remember_visible_result(_tool_loop_result_value(result))

    @user_request
    def ask_with_tools(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        workdir,
        max_loops: int = 10,
        on_final_delta: Callable[[str], None] | None = None,
        on_progress: Callable[[StreamProgress], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_tool_result: Callable[[str, int, str, str], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
        tool_names: tuple[str, ...] | None = None,
    ) -> ChatResult:
        self.last_memory_refresh = None
        self.refresh_memory_if_needed(temperature=temperature)
        self.messages.append({"role": "user", "content": prompt})
        self._stream_tool_thinking_plan(
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
        )
        result = self._run_tool_loop(
            temperature=temperature,
            max_tokens=max_tokens,
            workdir=workdir,
            max_loops=max_loops,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            tool_names=tool_names,
        )
        return self._remember_visible_result(result.result)

    @user_request
    def ask_auto(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        workdir,
        max_loops: int = 10,
        on_final_delta: Callable[[str], None] | None = None,
        on_progress: Callable[[StreamProgress], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_tool_result: Callable[[str, int, str, str], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> ChatResult:
        self.last_memory_refresh = None
        self.refresh_memory_if_needed(temperature=temperature)
        resolution = self._file_input_environment(workdir).resolve(
            prompt,
            allowed_tool_names=allowed_tool_names,
        )
        media_result = self._ask_media_if_referenced(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            resolution=resolution,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
        )
        if media_result is not None:
            return self._remember_visible_result(media_result)
        self.messages.append({"role": "user", "content": prompt})
        self._stream_tool_thinking_plan(
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
        )
        if resolution.bypass_tool_route:
            bundle = self._run_tool_loop(
                temperature=temperature,
                max_tokens=max_tokens,
                workdir=workdir,
                max_loops=max_loops,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_model_step=on_model_step,
                on_phase_start=on_phase_start,
                tool_names=("exec_shell_full_command",),
            )
            return self._remember_visible_result(_tool_loop_result_value(bundle))
        command_max_tokens = resolve_max_tokens("route", max_tokens)
        streamed_final_retry = False
        retried_empty_final = False
        route_abort_reason: str | None = None
        command_messages = self._route_messages()
        with self._temporary_backend_thinking(False):
            if on_phase_start:
                on_phase_start(ModelPhaseStart("route", streamed=False, attempt=1, reason="tool_decision"))
            with model_call_context(phase="route", tools_mode="on"):
                if on_progress is None:
                    first = self.backend.chat(command_messages, temperature=temperature, max_tokens=command_max_tokens)
                else:
                    route_stream_filter = _RouteDecisionStreamAbort()
                    try:
                        first = self.backend.chat_stream(
                            command_messages,
                            temperature=temperature,
                            max_tokens=command_max_tokens,
                            on_delta=route_stream_filter.write,
                            on_progress=on_progress,
                        )
                    except _RouteNotCommandAbort as exc:
                        route_abort_reason = "route_not_command"
                        first = ChatResult(
                            content=exc.content,
                            model=None,
                            finish_reason="length",
                            tool_calls=[],
                            prompt_tokens=None,
                            completion_tokens=exc.chunks,
                            cached_tokens=None,
                            prompt_tokens_per_second=None,
                            generation_tokens_per_second=None,
                        )
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=1, result=first, phase="route"))
        command_content = first.content
        decision = parse_command_decision_from_tool_calls(first.tool_calls) or parse_command_decision(command_content)
        route_outcome_emitted = False
        if decision is not None:
            _emit_route_outcome_for_result(first, decision=decision, outcome=_route_parsed_outcome(decision))
            route_outcome_emitted = True
        if decision is None:
            explicit_web_search = _requires_explicit_web_search_tool(prompt, allowed_tool_names)
            file_content_evidence = _requires_file_content_evidence_retry(prompt, allowed_tool_names, first.finish_reason)
            if _should_force_tool_route_retry(
                prompt,
                allowed_tool_names,
                first.finish_reason,
            ) and not _should_skip_route_repair(first):
                retry_reason = (
                    "explicit_web_search"
                    if explicit_web_search
                    else "file_content_evidence"
                    if file_content_evidence
                    else "length_without_decision_tool_related"
                )
                _emit_route_outcome_for_result(
                    first,
                    decision=None,
                    outcome="route_other_retry",
                    retry_reason=retry_reason,
                )
                route_outcome_emitted = True
                with self._temporary_backend_thinking(False):
                    route_retry_messages = _route_retry_messages(
                        self.messages,
                        explicit_web_search=explicit_web_search,
                        file_content_evidence=file_content_evidence,
                    )
                    route_retry_messages = self._with_route_evidence_context(route_retry_messages)
                    if on_phase_start:
                        on_phase_start(
                            ModelPhaseStart(
                                "route_retry",
                                streamed=False,
                                attempt=2,
                                reason=retry_reason,
                            )
                        )
                    if on_progress is None:
                        with model_call_context(phase="route_retry", tools_mode="on"):
                            route_retry = self.backend.chat(
                                route_retry_messages,
                                temperature=temperature,
                                max_tokens=command_max_tokens,
                            )
                    else:
                        with model_call_context(phase="route_retry", tools_mode="on"):
                            route_retry = self.backend.chat_stream(
                                route_retry_messages,
                                temperature=temperature,
                                max_tokens=command_max_tokens,
                                on_delta=lambda _text: None,
                                on_progress=on_progress,
                            )
                if on_model_step:
                    on_model_step(ModelStepMetrics.from_result(loop=2, result=route_retry, phase="route_retry"))
                retry_decision = parse_command_decision_from_tool_calls(route_retry.tool_calls) or parse_command_decision(route_retry.content)
                if retry_decision is not None:
                    _emit_route_outcome_for_result(route_retry, decision=retry_decision, outcome=_route_parsed_outcome(retry_decision))
                    route_outcome_emitted = True
                if retry_decision is not None:
                    first = route_retry
                    command_content = route_retry.content
                    decision = retry_decision
            if decision is None:
                if explicit_web_search or file_content_evidence:
                    bundle = self._run_tool_loop(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        workdir=workdir,
                        max_loops=max_loops,
                        on_final_delta=on_final_delta,
                        on_progress=on_progress,
                        on_tool_call=on_tool_call,
                        on_tool_result=on_tool_result,
                        on_model_step=on_model_step,
                        on_phase_start=on_phase_start,
                        tool_names=("exec_shell_full_command",),
                    )
                    return self._remember_visible_result(_tool_loop_result_value(bundle))
                initial_shell_tool_call = command_tool_call_from_tool_calls(first.tool_calls, ("exec_shell_full_command",)) or command_tool_call_from_content(command_content, ("exec_shell_full_command",))
                route_requires_chat_final = contains_control_channel_markup(first.content)
                if route_requires_chat_final and first.finish_reason != "length":
                    if not route_outcome_emitted:
                        _emit_route_outcome_for_result(
                            first,
                            decision=None,
                            outcome="route_invalid_output",
                            retry_reason="control_channel_markup",
                        )
                        route_outcome_emitted = True
                    first = self._transport_environment().chat_final(
                        self._with_final_evidence_context(with_chat_system_prompt(self.messages)),
                        temperature=temperature,
                        max_tokens=resolve_max_tokens("chat", max_tokens),
                        on_final_delta=on_final_delta,
                        on_progress=on_progress,
                        on_model_step=on_model_step,
                        on_phase_start=on_phase_start,
                        loop=2,
                    )
                    retried_empty_final = True
                if first.finish_reason == "length":
                    retry_reason = route_abort_reason or "length_without_decision"
                    if not route_outcome_emitted:
                        _emit_route_outcome_for_result(
                            first,
                            decision=None,
                            outcome="route_no_decision_length_retry",
                            retry_reason=retry_reason,
                        )
                        route_outcome_emitted = True
                    if on_phase_start:
                        on_phase_start(
                            ModelPhaseStart(
                                "chat_final_retry",
                                streamed=on_final_delta is not None,
                                attempt=2,
                                reason=retry_reason,
                            )
                        )
                    retry_messages = self._chat_final_retry_messages()
                    retry_max_tokens = self._chat_final_retry_max_tokens(
                        max_tokens,
                        previous_finish_reason=first.finish_reason,
                    )
                    if on_final_delta is None:
                        with model_call_context(phase="chat_final_retry", tools_mode="on"):
                            first = self.backend.chat(
                                retry_messages,
                                temperature=temperature,
                                max_tokens=retry_max_tokens,
                            )
                    else:
                        streamed_final_retry = True
                        with model_call_context(phase="chat_final_retry", tools_mode="on"):
                            first = self.backend.chat_stream(
                                retry_messages,
                                temperature=temperature,
                                max_tokens=retry_max_tokens,
                                on_delta=on_final_delta,
                                on_progress=on_progress,
                            )
                    if on_model_step:
                        on_model_step(ModelStepMetrics.from_result(loop=2, result=first, phase="chat_final_retry"))
                if _is_empty_final_response(first):
                    if not route_outcome_emitted:
                        _emit_route_outcome_for_result(
                            first,
                            decision=None,
                            outcome="route_invalid_output",
                            retry_reason="empty_response",
                        )
                        route_outcome_emitted = True
                    retried_empty_final = True
                    first = self._transport_environment().chat_final(
                        self._with_final_evidence_context(self.messages),
                        temperature=temperature,
                        max_tokens=resolve_max_tokens("chat", max_tokens),
                        on_final_delta=on_final_delta,
                        on_progress=on_progress,
                        on_model_step=on_model_step,
                        on_phase_start=on_phase_start,
                        loop=2,
                    )
                if not route_outcome_emitted:
                    _emit_route_outcome_for_result(
                        first,
                        decision=None,
                        outcome="route_direct_final_stop",
                        retry_reason=None,
                    )
                self.messages.append({"role": "assistant", "content": first.content})
                if on_final_delta and not streamed_final_retry and not retried_empty_final:
                    on_final_delta(first.content)
                return self._remember_visible_result(first)
        if decision.route == ToolRoute.CHAT:
            chat_messages = self._chat_final_messages()
            with model_call_context(phase="chat_final", tools_mode="on"):
                result = self._chat_final(
                    chat_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                    on_model_step=on_model_step,
                    on_phase_start=on_phase_start,
                    loop=2,
                )
            self.messages.append({"role": "assistant", "content": result.content})
            return self._remember_visible_result(result)
        if decision.route == ToolRoute.MEDIA:
            if allowed_tool_names is not None:
                result = _unsupported_tool_mode_result(first)
                self.messages.append({"role": "assistant", "content": result.content})
                if on_final_delta:
                    on_final_delta(result.content)
                return self._remember_visible_result(result)
            return self._ask_referenced_media(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                resolution=resolution,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
                on_model_step=on_model_step,
                on_phase_start=on_phase_start,
                command_result=first,
            )
        tools = decision_tool_names(decision, prompt)
        if allowed_tool_names is not None:
            allowed = set(allowed_tool_names)
            tools = tuple(tool for tool in tools if tool in allowed)
            if (
                not tools
                and decision.route in {ToolRoute.FILESYSTEM, ToolRoute.FILE_EDIT, ToolRoute.WEB}
                and "exec_shell_full_command" in allowed
            ):
                tools = ("exec_shell_full_command",)
        if not tools:
            result = _unsupported_tool_mode_result(first)
            self.messages.append({"role": "assistant", "content": result.content})
            if on_final_delta:
                on_final_delta(result.content)
            return self._remember_visible_result(result)
        bundle = self._run_tool_loop(
            temperature=temperature,
            max_tokens=max_tokens,
            workdir=workdir,
            max_loops=max_loops,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            tool_names=tools,
            initial_tool_calls=(
                command_tool_call_from_tool_calls(first.tool_calls, tools)
                or command_tool_call_from_content(command_content, tools)
            ),
        )
        return self._remember_visible_result(_tool_loop_result_value(bundle))

    def _remember_visible_result(self, result: ChatResult) -> ChatResult:
        self.client_state.update_from_result(result, thinking=self._thinking())
        return result

    def _ask_media_if_referenced(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        resolution,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
    ) -> ChatResult | None:
        if resolution.error:
            return None
        if not resolution.has_media:
            return None
        user_content = message_content(prompt, resolution.images, resolution.audios)
        call_messages = with_media_system_prompt([*self.messages, {"role": "user", "content": user_content}])
        result = self._pure_chat_environment().ask_user_content(
            user_content,
            temperature=temperature,
            max_tokens=max_tokens,
            call_messages=call_messages,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=1,
        )
        return result

    def _ask_referenced_media(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        resolution,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
        command_result: ChatResult,
    ) -> ChatResult:
        if resolution.error:
            result = error_result(resolution.error, command_result)
            self.messages.append({"role": "assistant", "content": result.content})
            if on_final_delta:
                on_final_delta(result.content)
            return result
        if not resolution.has_media:
            result = error_result("error: MEDIA route requested but no local image/audio path was found in the prompt", command_result)
            self.messages.append({"role": "assistant", "content": result.content})
            if on_final_delta:
                on_final_delta(result.content)
            return result

        self.messages[-1] = {"role": "user", "content": message_content(prompt, resolution.images, resolution.audios)}
        call_messages = with_media_system_prompt(self.messages)
        result = self._transport_environment().chat_final(
            call_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=2,
        )
        self.messages.append({"role": "assistant", "content": result.content})
        return result

    def _run_tool_loop(
        self,
        *,
        temperature: float,
        max_tokens: int,
        workdir,
        max_loops: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_tool_call: Callable[[str, str], None] | None,
        on_tool_result: Callable[[str, int, str, str], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
        tool_names: tuple[str, ...] | None,
        initial_tool_calls: list[dict[str, object]] | dict[str, object] | None = None,
        ):
        return self._tool_loop_environment().run(
            temperature=temperature,
            max_tokens=max_tokens,
            workdir=workdir,
            max_loops=max_loops,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            tool_names=tool_names,
            initial_tool_calls=initial_tool_calls,
            local_capabilities=self.local_capabilities,
        )
    def _stream_tool_thinking_plan(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
    ) -> None:
        if not self._thinking().should_stream_tool_plan(
            has_delta_sink=on_final_delta is not None,
            backend_supports_streaming=hasattr(self.backend, "chat_stream"),
        ):
            return
        planning_messages = [
            *with_chat_system_prompt(self.messages),
            {
                "role": "user",
                "content": (
                    "Think briefly about the approach only. "
                    "Do not answer the user yet. "
                    "Do not call tools in this step."
                ),
            },
        ]
        thought_filter = _ThoughtOnlyDeltaFilter(on_final_delta)
        with self._transport_environment().backend_thinking(True):
            if on_phase_start:
                on_phase_start(ModelPhaseStart("tool_plan", streamed=True, attempt=1, reason="pre_tool_thinking"))
            with model_call_context(phase="tool_plan", tools_mode="on"):
                result = self.backend.chat_stream(
                    planning_messages,
                    temperature=temperature,
                    max_tokens=resolve_max_tokens("route", max_tokens),
                    on_delta=thought_filter.write,
                    on_progress=on_progress,
                )
        thought_filter.finish()
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=1, result=result, phase="tool_plan"))

    def _answer_from_tool_results(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
        loop: int,
        use_tool_prompt: bool,
        compact_window: bool = False,
    ) -> ChatResult:
        return self._final_from_tool_environment().answer(
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=loop,
            use_tool_prompt=use_tool_prompt,
            compact_window=compact_window,
        ).result

    def _chat_final(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None = None,
        loop: int,
    ) -> ChatResult:
        result = self._transport_environment().chat_final(
            messages,
            temperature=temperature,
            max_tokens=resolve_max_tokens("chat", max_tokens),
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=loop,
        )
        return result

    def _chat_once(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
    ) -> ChatResult:
        return self._transport_environment().chat_once(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
        )

    def _chat_tool_call_once(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, object]],
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
    ) -> ChatResult:
        return self._transport_environment().chat_tool_call_once(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
        )

    def _temporary_backend_thinking(self, value: bool):
        return self._transport_environment().backend_thinking(value)

    def reset(self) -> None:
        self.messages.clear()
        if self.evidence_store is not None:
            self.evidence_store.clear_memory()
        self.client_state.reset()
        self.last_memory_refresh = None
        self.last_memory_refresh_message_count = None
        self.last_memory_refresh_attempt = None
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def compact_old_tool_results(self, *, temperature: float) -> ToolResultCompactionReport:
        return compact_tool_results(self.messages, backend=self.backend, temperature=temperature)

    def compact_memory_now(self, *, temperature: float) -> MemoryRefresh:
        before_count = len(self.messages)
        result = maybe_refresh_memory(
            self.messages,
            backend=self.backend,
            context_tokens=self.context_tokens,
            temperature=temperature,
            force=True,
        )
        if result.changed:
            self.last_memory_refresh = result
            self.last_memory_refresh_message_count = len(self.messages)
            self.memory_refreshes += 1
            self.total_memory_tokens_saved += max(0, result.estimated_tokens_before - result.estimated_tokens_after)
        elif len(self.messages) != before_count:
            self.messages[:] = self.messages[:before_count]
        self.last_memory_refresh_attempt = result
        return result

    def persistent_messages(self) -> list[Message]:
        return persistent_tool_result_messages(self.messages)

    def restore_message_count(self, count: int) -> None:
        if count < 0:
            count = 0
        del self.messages[count:]
        self.client_state.reset()
        self.last_memory_refresh = None
        if self.last_memory_refresh_message_count is not None and self.last_memory_refresh_message_count > len(self.messages):
            self.last_memory_refresh_message_count = None

    def can_continue_last_response(self) -> bool:
        return self.client_state.can_continue or last_assistant_has_open_reasoning(self.messages)

    @property
    def last_visible_finish_reason(self) -> str | None:
        return self.client_state.last_finish_reason

    @last_visible_finish_reason.setter
    def last_visible_finish_reason(self, value: str | None) -> None:
        self.client_state.last_finish_reason = value
        if value == "length" and not self.client_state.continuation_kind:
            self.client_state.continuation_kind = "final_answer"

    def _thinking(self) -> ThinkingMode:
        return ThinkingMode(enabled=self.thinking_mode)

    def _continue_environment(self) -> ContinueEnvironment:
        return ContinueEnvironment(runtime=self, transport=self._transport_environment())

    def _transport_environment(self) -> TransportEnvironment:
        return TransportEnvironment(runtime=self)

    def _pure_chat_environment(self) -> PureChatEnvironment:
        return PureChatEnvironment(runtime=self, transport=self._transport_environment())

    def _tool_loop_environment(self) -> ToolLoopEnvironment:
        return ToolLoopEnvironment(runtime=self)

    def _final_from_tool_environment(self) -> FinalFromToolEnvironment:
        return FinalFromToolEnvironment(runtime=self, transport=self._transport_environment())

    def _file_input_environment(self, workdir) -> FileInputEnvironment:
        return FileInputEnvironment(FileInputResolver(workdir=workdir))

    def _with_final_tool_prompt(self) -> list[Message]:
        return with_final_tool_system_prompt(self.messages)

    def _compact_final_from_tool_messages(self) -> list[Message]:
        messages: list[Message] = []
        latest_user = _latest_user_message(self.messages)
        if latest_user is not None:
            messages.append(latest_user)
        context = build_compact_final_evidence_context(self.evidence_store)
        self._emit_evidence_lineage_context(context_kind="compact_final", consumer_phase="final_from_tool", limit=1)
        final_messages = with_final_tool_system_prompt(messages)
        if not context:
            return final_messages
        return [*final_messages, {"role": "system", "content": context}]

    def _web_final_from_tool_messages(self) -> list[Message]:
        messages: list[Message] = []
        latest_user = _latest_user_message(self.messages)
        if latest_user is not None:
            messages.append(latest_user)
        final_messages = with_final_tool_system_prompt(messages)
        context = build_web_final_evidence_context(self.evidence_store)
        if not context:
            return final_messages
        return [*final_messages, {"role": "system", "content": context}]

    def _should_use_web_final_view(self, *, use_tool_prompt: bool) -> bool:
        if self.evidence_store is None:
            return False
        record = next(iter(self.evidence_store.recent_records(1)), None)
        if record is None or record.kind != "web_search":
            return False
        if record.status == "none":
            return True
        if use_tool_prompt:
            return False
        snippets = record.metadata.get("top_snippets")
        return isinstance(snippets, list) and bool(snippets)

    def _should_use_final_small_evidence_view(self, *, use_tool_prompt: bool) -> bool:
        if use_tool_prompt or self.evidence_store is None:
            return False
        record = next(iter(self.evidence_store.recent_records(1)), None)
        if record is None:
            return False
        if record.kind not in FINAL_SMALL_EVIDENCE_KINDS:
            return False
        if record.raw_chars > FINAL_SMALL_EVIDENCE_MAX_RAW_CHARS:
            return False
        return not _latest_tool_message_has_legacy_direct_content(self.messages)

    def _should_compact_final_from_tool_window(self) -> bool:
        if self.evidence_store is None:
            return False
        record = next(iter(self.evidence_store.recent_records(1)), None)
        if record is None:
            return False
        if record.kind in {"web_search", "fetch"}:
            return False
        return record.raw_chars >= FINAL_FROM_TOOL_COMPACT_MIN_RAW_CHARS

    def _with_route_evidence_context(self, messages: list[Message]) -> list[Message]:
        context = build_route_evidence_context(self.evidence_store)
        if not context:
            return messages
        return [*messages, {"role": "system", "content": context}]

    def _route_messages(self) -> list[Message]:
        if self._should_use_post_tool_route_window():
            return self._post_tool_route_messages()
        return self._with_route_evidence_context(with_command_system_prompt(self.messages))

    def _should_use_post_tool_route_window(self) -> bool:
        if self.evidence_store is None or not self.evidence_store.recent_records(1):
            return False
        return _latest_user_message(self.messages) is not None

    def _post_tool_route_messages(self) -> list[Message]:
        messages: list[Message] = []
        latest_user = _latest_user_message(self.messages)
        if latest_user is not None:
            messages.append(latest_user)
        route_messages = with_command_system_prompt(messages)
        context = build_post_tool_route_evidence_context(self.evidence_store)
        if not context:
            return route_messages
        return [*route_messages, {"role": "system", "content": context}]

    def _with_final_evidence_context(self, messages: list[Message]) -> list[Message]:
        context = build_final_evidence_context(self.evidence_store)
        self._emit_evidence_lineage_context(context_kind="final", consumer_phase="final", limit=2)
        if not context:
            return messages
        return [*messages, {"role": "system", "content": context}]

    def _chat_final_messages(self) -> list[Message]:
        if self.evidence_store is None or not self.evidence_store.recent_records(1):
            return self._with_final_evidence_context(with_chat_system_prompt(self.messages))
        messages: list[Message] = []
        assistant = _latest_operational_assistant_message(self.messages)
        if assistant is not None:
            messages.append(assistant)
        latest_user = _latest_user_message(self.messages)
        if latest_user is not None:
            messages.append(latest_user)
        return self._with_chat_final_retry_evidence_context(with_chat_system_prompt(messages), consumer_phase="chat_final")

    def _with_chat_final_retry_evidence_context(self, messages: list[Message], *, consumer_phase: str) -> list[Message]:
        if self._should_compact_final_from_tool_window():
            context = build_compact_final_evidence_context(self.evidence_store)
            self._emit_evidence_lineage_context(context_kind="compact_final", consumer_phase=consumer_phase, limit=1)
        else:
            context = build_final_evidence_context(self.evidence_store)
            self._emit_evidence_lineage_context(context_kind="final", consumer_phase=consumer_phase, limit=2)
        if not context:
            return messages
        return [*messages, {"role": "system", "content": context}]

    def _emit_evidence_lineage_context(self, *, context_kind: str, consumer_phase: str, limit: int) -> None:
        if self.evidence_store is None:
            return
        records = self.evidence_store.recent_records(limit)
        for index, record in enumerate(records, start=1):
            emit_evidence_lineage(
                {
                    "context_kind": context_kind,
                    "phase": consumer_phase,
                    "card_index": index,
                    "evidence_id": record.evidence_id,
                    "evidence_sequence": record.evidence_sequence,
                    "tool_call_id": record.tool_call_id,
                    "user_turn_id": record.user_turn_id,
                    "producer_model_call_id": record.producer_model_call_id,
                    "produced_by_phase": record.produced_by_phase,
                    "kind": record.kind,
                    "status": record.status,
                }
            )

    def _chat_final_retry_messages(self) -> list[Message]:
        messages: list[Message] = []
        latest_user = _latest_user_message(self.messages)
        if self.evidence_store is not None and self.evidence_store.recent_records(1):
            assistant = _latest_operational_assistant_message(self.messages)
        else:
            assistant = _latest_short_assistant_message(self.messages)
        if assistant is not None:
            messages.append(assistant)
        if latest_user is not None:
            messages.append(latest_user)
        return self._with_chat_final_retry_evidence_context(with_chat_system_prompt(messages), consumer_phase="chat_final_retry")

    def chat_final_completion_repair_messages(self, repair_instruction: str) -> list[Message] | None:
        if self.evidence_store is None or not self.evidence_store.recent_records(1):
            return None
        latest_user = _latest_user_message(self.messages)
        if latest_user is None:
            return None
        window: list[Message] = []
        assistant = _latest_assistant_excerpt_message(self.messages)
        if assistant is not None:
            window.append(assistant)
        window.append(latest_user)
        messages = with_chat_system_prompt(window)
        context = build_compact_final_evidence_context(self.evidence_store)
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": repair_instruction})
        return messages

    def _chat_final_retry_max_tokens(self, max_tokens: int, *, previous_finish_reason: str | None = None) -> int:
        if self.evidence_store is not None and self.evidence_store.recent_records(1):
            return resolve_max_tokens(
                "chat_final_retry",
                max_tokens,
                previous_finish_reason=previous_finish_reason,
            )
        return max_tokens

    def refresh_memory_if_needed(self, *, temperature: float, force: bool = False) -> bool:
        if not force and self._memory_refresh_in_cooldown():
            return False
        result = maybe_refresh_memory(
            self.messages,
            backend=self.backend,
            context_tokens=self.context_tokens,
            temperature=temperature,
            force=force,
        )
        if result.changed:
            self.last_memory_refresh = result
            self.last_memory_refresh_message_count = len(self.messages)
            self.memory_refreshes += 1
            self.total_memory_tokens_saved += max(0, result.estimated_tokens_before - result.estimated_tokens_after)
        self.last_memory_refresh_attempt = result
        return result.changed

    def _memory_refresh_in_cooldown(self) -> bool:
        if self.last_memory_refresh_message_count is None:
            return False
        return len(self.messages) - self.last_memory_refresh_message_count < self.memory_refresh_cooldown_messages


class _ThoughtOnlyDeltaFilter:
    _THOUGHT_START = "<|channel>thought\n"
    _THOUGHT_END = "<channel|>"

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._buffer = ""
        self._in_thought = False

    def write(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        self._drain(final=False)

    def finish(self) -> None:
        self._drain(final=True)

    def _drain(self, *, final: bool) -> None:
        while self._buffer:
            if self._in_thought:
                end = self._buffer.find(self._THOUGHT_END)
                if end < 0:
                    if final:
                        visible = self._buffer
                        self._buffer = ""
                        if visible:
                            self._emit(visible)
                    break
                visible = self._buffer[:end]
                if visible:
                    self._emit(visible)
                self._buffer = self._buffer[end + len(self._THOUGHT_END) :]
                self._in_thought = False
                continue
            start = self._buffer.find(self._THOUGHT_START)
            if start < 0:
                if final:
                    self._buffer = ""
                break
            self._buffer = self._buffer[start + len(self._THOUGHT_START) :]
            self._in_thought = True


class _RouteNotCommandAbort(Exception):
    def __init__(self, content: str, *, chunks: int) -> None:
        super().__init__("route stream produced non-command prose")
        self.content = content
        self.chunks = chunks


class _RouteDecisionStreamAbort:
    """Stop an internal route stream once it cannot still become a route decision."""

    _MAX_PENDING_WHITESPACE_CHUNKS = 8

    def __init__(self) -> None:
        self._buffer = ""
        self._chunks = 0

    def write(self, text: str) -> None:
        if not text:
            return
        self._chunks += 1
        self._buffer += text
        if not self._buffer.strip() and self._chunks >= self._MAX_PENDING_WHITESPACE_CHUNKS:
            raise _RouteNotCommandAbort(self._buffer, chunks=self._chunks)
        if command_stream_state(self._buffer) == "not_command":
            raise _RouteNotCommandAbort(self._buffer, chunks=self._chunks)


class _BufferedDeltaSink:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> None:
        if text:
            self.chunks.append(text)


def _tool_call_command(tool_call: dict[str, object] | None) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    command = arguments.get("command")
    return command if isinstance(command, str) else None


def _should_retry_route_as_tool_call(prompt: str, allowed_tool_names: tuple[str, ...] | None) -> bool:
    if "exec_shell_full_command" not in set(allowed_tool_names or ()):
        return False
    return bool(_ROUTE_TOOL_RETRY_PROMPT_RE.search(prompt))


def _requires_explicit_web_search_tool(prompt: str, allowed_tool_names: tuple[str, ...] | None) -> bool:
    if "exec_shell_full_command" not in set(allowed_tool_names or ()):
        return False
    return bool(_EXPLICIT_WEB_SEARCH_RE.search(prompt))


def _should_force_tool_route_retry(prompt: str, allowed_tool_names: tuple[str, ...] | None, finish_reason: str | None) -> bool:
    if _requires_explicit_web_search_tool(prompt, allowed_tool_names):
        return True
    if _requires_file_content_evidence_retry(prompt, allowed_tool_names, finish_reason):
        return True
    return finish_reason == "length" and _should_retry_route_as_tool_call(prompt, allowed_tool_names)


def _should_attempt_route_repair(result: ChatResult) -> bool:
    if result.tool_calls:
        return True
    text = result.content.strip()
    if not text:
        return False
    if command_stream_state(text) == "pending":
        return True
    if text.startswith("<|tool_call>"):
        return True
    if text.startswith("{") and any(key in text for key in ('"command"', '"url"', '"path"', '"route"', '"include_')):
        return True
    return False


def _should_skip_route_repair(result: ChatResult) -> bool:
    if result.finish_reason != "length" or _should_attempt_route_repair(result):
        return False
    text = result.content.strip()
    if not text:
        return True
    if result.completion_tokens is not None:
        return result.completion_tokens <= 2 and len(text) <= 16
    return len(text) <= 16


def _requires_file_content_evidence_retry(prompt: str, allowed_tool_names: tuple[str, ...] | None, finish_reason: str | None) -> bool:
    if finish_reason != "stop":
        return False
    if "exec_shell_full_command" not in set(allowed_tool_names or ()):
        return False
    return bool(_FILE_CONTENT_ACTION_RE.search(prompt) and _FILE_LIKE_REFERENCE_RE.search(prompt))


def _route_parsed_outcome(decision: RouteDecision) -> str:
    if decision.route == ToolRoute.CHAT:
        return "route_parsed_chat"
    return "route_parsed_tool"


def _emit_route_outcome_for_result(
    result: ChatResult,
    *,
    decision: RouteDecision | None,
    outcome: str,
    retry_reason: str | None = None,
) -> None:
    decision_type = decision.route.value if decision is not None else None
    emit_route_outcome(
        outcome=outcome,
        finish_reason=result.finish_reason,
        decision_type=decision_type,
        output_chars=len(result.content),
        output_tokens=result.completion_tokens,
        retry_reason=retry_reason,
    )


def _route_retry_messages(
    messages: list[Message],
    *,
    explicit_web_search: bool,
    file_content_evidence: bool = False,
) -> list[Message]:
    if explicit_web_search:
        reminder = (
            "The user explicitly asked for an online/web search. "
            "Do not answer from memory. "
            "Use the latest user request only, and ignore older tool results or file/page content unless that latest request refers to them. "
            "Return exactly one shell tool call now that performs the search. Output no prose."
        )
    elif file_content_evidence:
        reminder = (
            "The latest user request asks for a specific file's contents. "
            "Do not answer from directory listings, older tool results, or assumptions. "
            'Return a compact JSON command decision such as {"command":"cat requested-file"} that reads the requested file content. '
            "Output no prose."
        )
    else:
        reminder = (
            "The previous routing pass did not return a valid decision. "
            "Return exactly one shell tool call now if shell/web/file access is needed. Output no prose."
        )
    retry_messages = with_command_system_prompt(messages) if file_content_evidence else with_tool_call_system_prompt(messages)
    return [
        *retry_messages,
        {"role": "user", "content": reminder},
    ]


def _latest_user_message(messages: list[Message]) -> Message | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return dict(message)
    return None


def _latest_short_assistant_message(messages: list[Message], *, max_chars: int = 600) -> Message | None:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip() and len(content) <= max_chars:
            return {"role": "assistant", "content": content}
    return None


def _latest_operational_assistant_message(messages: list[Message], *, max_chars: int = 160) -> Message | None:
    return _latest_assistant_excerpt_message(messages, max_chars=max_chars)


def _latest_assistant_excerpt_message(messages: list[Message], *, max_chars: int = 140) -> Message | None:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."
            return {"role": "assistant", "content": text}
    return None


def _latest_tool_message_has_legacy_direct_content(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = str(message.get("content", ""))
        return any(
            marker in content
            for marker in (
                "shell-full output",
                "large_file_excerpt: true",
                "pdf_text_result: true",
                "fetch_url_result: true",
            )
        )
    return False


def _tool_loop_result_value(result):
    return result.result if hasattr(result, "result") else result


def _merge_chat_results(first: ChatResult, second: ChatResult) -> ChatResult:
    def add_int(a: int | None, b: int | None) -> int | None:
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)

    return ChatResult(
        content=f"{first.content}{second.content}",
        model=second.model or first.model,
        finish_reason=second.finish_reason or first.finish_reason,
        tool_calls=second.tool_calls or first.tool_calls,
        prompt_tokens=add_int(first.prompt_tokens, second.prompt_tokens),
        completion_tokens=add_int(first.completion_tokens, second.completion_tokens),
        cached_tokens=add_int(first.cached_tokens, second.cached_tokens),
        prompt_tokens_per_second=second.prompt_tokens_per_second or first.prompt_tokens_per_second,
        generation_tokens_per_second=second.generation_tokens_per_second or first.generation_tokens_per_second,
    )


def _is_empty_final_response(result: ChatResult) -> bool:
    return not result.tool_calls and result.finish_reason == "stop" and not result.content.strip()


def _unsupported_tool_mode_result(result: ChatResult) -> ChatResult:
    return ChatResult(
        content="error: no suitable tool is available for this request",
        model=result.model,
        finish_reason="unsupported_command",
        tool_calls=[],
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cached_tokens=result.cached_tokens,
        prompt_tokens_per_second=result.prompt_tokens_per_second,
        generation_tokens_per_second=result.generation_tokens_per_second,
    )


def _is_tool_argument_json_error(exc: RuntimeError) -> bool:
    text = str(exc)
    return "Failed to parse tool call arguments as JSON" in text


def _needs_final_completion_repair(content: str) -> bool:
    if "<channel|>" in content:
        tail = content.split("<channel|>", 1)[1].strip()
        if tail:
            return False
    lowered = content.strip().lower()
    if "final answer:" in lowered or "**final answer:**" in lowered:
        return False
    stripped = content.rstrip()
    if re.search(r"(?:^|\n)\s*#{1,6}\s*$", stripped):
        return True
    if re.search(r"(?:^|\n)\s*(?:[-*]|\d+\.)\s+\*\*[^*\n]+:\*\*\s*$", stripped):
        return True
    if re.search(r":\s*(?:\n\s*)?(?:#{1,6}|\*|-)?\s*$", stripped):
        return True
    if looks_like_incomplete_final(content):
        return True
    return content.count("`") % 2 == 1


_FILE_MISSING_RE = re.compile(
    r"(?:"
    r"file\b.*(?:not\s+found|missing|inaccessible)"
    r"|could\s+not\s+be\s+found"
    r"|cannot\s+confirm.*file"
    r"|non\s+[\w'\s]*trovat\w*"
    r"|potrebbe\s+non\s+essere\s+stato\s+trovat\w*"
    r"|impossibile.*trovare"
    r")",
    re.IGNORECASE,
)


def _contradicts_successful_pdf_extraction(messages: list[Message], content: str) -> bool:
    if not content.strip():
        return False
    has_pdf_success = any(
        message.get("role") == "tool"
        and isinstance(message.get("content"), str)
        and "shell_output_pdf_text: true" in str(message.get("content"))
        for message in messages
    )
    return has_pdf_success and _FILE_MISSING_RE.search(content) is not None


def _pdf_extraction_repair_prompt(messages: list[Message]) -> str:
    path = None
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or "shell_output_pdf_text: true" not in content:
            continue
        for line in content.splitlines():
            if line.startswith("path: "):
                path = line.removeprefix("path: ").strip()
                break
        break
    path_text = f' for "{path}"' if path else ""
    return (
        "A successful PDF text extraction already exists"
        f"{path_text}. "
        "The previous answer incorrectly claimed the file was missing or inaccessible. "
        "Restate the answer using only the extracted PDF text already available. "
        "Do not say the file is missing. Do not call tools."
    )
