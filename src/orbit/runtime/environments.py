from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable
import json
import re

from orbit.backend import ChatResult
from orbit.backend.base import Message, StreamProgress
from orbit.runtime.completion_budget import CompletionBudget
from orbit.runtime.continue_controller import ContinueController
from orbit.runtime.file_input_resolver import FileInputResolver
from orbit.runtime.final_policy import (
    build_final_tool_policy,
    classify_final_answer_completeness,
    final_from_tool_compact_retry_reason,
    final_from_tool_retry_reason,
    final_tool_compact_retry_instruction,
    final_tool_compact_retry_max_tokens,
    final_tool_retry_instruction,
    final_tool_retry_max_tokens,
    is_repetitive_final_answer,
    last_user_text,
    looks_like_incomplete_final,
)
from orbit.runtime.media import AudioInput, ImageInput
from orbit.runtime.messages import TOOL_CALL_JSON_RETRY_PROMPT
from orbit.runtime.thinking_mode import ThinkingMode, last_assistant_has_open_reasoning
from orbit.runtime.tool_loop import run_tool_loop
from orbit.runtime.turn_trace import ModelPhaseStart, ModelStepMetrics

if TYPE_CHECKING:
    from pathlib import Path

    from orbit.runtime.chat import ChatRuntime

CHAT_THINKING_MAX_TOKENS = 64


@dataclass(frozen=True)
class ClientTurnState:
    prompt: str
    temperature: float
    max_tokens: int
    workdir: Path | None = None
    max_loops: int = 10
    allowed_tool_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ToolResultBundle:
    result: ChatResult
    tool_names: tuple[str, ...] | None


@dataclass(frozen=True)
class FinalAnswerResult:
    result: ChatResult
    used_retry_or_repair_pass: bool
    repair_attempts: int = 0
    repair_failed: bool = False
    compact_retry_attempted: bool = False
    compact_retry_failed: bool = False


@dataclass(frozen=True)
class ContinueResult:
    result: ChatResult
    used_native_continue: bool
    used_prompt_fallback: bool


@dataclass(frozen=True)
class FileResolutionResult:
    images: list[ImageInput]
    audios: list[AudioInput]
    bypass_tool_route: bool
    error: str | None = None

    @property
    def has_media(self) -> bool:
        return bool(self.images or self.audios)


@dataclass(frozen=True)
class TransportEnvironment:
    runtime: ChatRuntime

    def chat_final(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None,
        loop: int,
        repair_incomplete_final: bool = True,
    ) -> ChatResult:
        if on_phase_start:
            on_phase_start(ModelPhaseStart("chat_final", streamed=on_final_delta is not None, attempt=1))
        result = self.chat_once(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
        )
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop, result=result, phase="chat_final"))
        completeness = classify_final_answer_completeness(result.content, messages=messages)
        if not repair_incomplete_final and not is_empty_final_response(result):
            return result
        if not is_empty_final_response(result) and completeness.is_complete:
            return result

        retry_messages = messages
        retry_phase = "chat_final_retry"
        retry_reason = "empty_or_invalid"
        with_final_only_retry = False
        if not completeness.is_complete:
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Stop reasoning and provide the final answer only. "
                        "No thinking, no plan, no headings, no bullet list unless explicitly requested. "
                        "Answer the user directly now."
                    ),
                },
            ]
            retry_phase = "chat_final_completion_repair"
            retry_reason = completeness.status
            with_final_only_retry = True
        if on_phase_start:
            on_phase_start(
                ModelPhaseStart(
                    retry_phase,
                    streamed=on_final_delta is not None,
                    attempt=2,
                    reason=retry_reason,
                )
            )
        retry_context = self.backend_thinking(False) if with_final_only_retry else nullcontext()
        with retry_context:
            retry = self.chat_once(
                retry_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
            )
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop + 1, result=retry, phase=retry_phase))
        retry_completeness = classify_final_answer_completeness(retry.content, messages=retry_messages)
        if not is_empty_final_response(retry) and retry_completeness.is_complete:
            return retry

        if is_empty_final_response(retry):
            error = ChatResult(
                content="error: model returned an empty response twice",
                model=retry.model,
                finish_reason="empty_response",
                tool_calls=retry.tool_calls,
                prompt_tokens=retry.prompt_tokens,
                completion_tokens=retry.completion_tokens,
                cached_tokens=retry.cached_tokens,
                prompt_tokens_per_second=retry.prompt_tokens_per_second,
                generation_tokens_per_second=retry.generation_tokens_per_second,
            )
            if on_final_delta:
                on_final_delta(error.content)
            return error

        error = ChatResult(
            content="error: model did not produce a clean final answer",
            model=retry.model,
            finish_reason="stop",
            tool_calls=retry.tool_calls,
            prompt_tokens=retry.prompt_tokens,
            completion_tokens=retry.completion_tokens,
            cached_tokens=retry.cached_tokens,
            prompt_tokens_per_second=retry.prompt_tokens_per_second,
            generation_tokens_per_second=retry.generation_tokens_per_second,
        )
        if on_final_delta:
            on_final_delta(error.content)
        return error

    def chat_once(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
    ) -> ChatResult:
        if on_final_delta is None:
            return self.runtime.backend.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self.runtime.backend.chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_delta=on_final_delta,
            on_progress=on_progress,
        )

    def chat_tool_call_once(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, object]],
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
    ) -> ChatResult:
        try:
            with self.backend_thinking(False):
                if on_final_delta is None:
                    return self.runtime.backend.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
                return self.runtime.backend.chat_stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    on_delta=on_final_delta,
                    on_progress=on_progress,
                )
        except RuntimeError as exc:
            if not is_tool_argument_json_error(exc):
                raise
        retry_messages = [*messages, {"role": "system", "content": TOOL_CALL_JSON_RETRY_PROMPT}]
        with self.backend_thinking(False):
            if on_final_delta is None:
                return self.runtime.backend.chat(retry_messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
            return self.runtime.backend.chat_stream(
                retry_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                on_delta=on_final_delta,
                on_progress=on_progress,
            )

    @contextmanager
    def backend_thinking(self, value: bool):
        if not hasattr(self.runtime.backend, "thinking"):
            yield
            return
        previous = getattr(self.runtime.backend, "thinking")
        setattr(self.runtime.backend, "thinking", value)
        try:
            yield
        finally:
            setattr(self.runtime.backend, "thinking", previous)


@dataclass(frozen=True)
class PureChatEnvironment:
    runtime: ChatRuntime
    transport: TransportEnvironment

    def ask_user_content(
        self,
        user_content: object,
        *,
        temperature: float,
        max_tokens: int,
        call_messages: list[Message],
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None,
        loop: int,
    ) -> ChatResult:
        self.runtime.messages.append({"role": "user", "content": user_content})
        if self.runtime.thinking_mode:
            thinking_max_tokens = CompletionBudget(max_tokens).internal(CHAT_THINKING_MAX_TOKENS)
            thinking_messages = [
                *call_messages,
                {
                    "role": "user",
                    "content": (
                        "Think briefly about how to answer. "
                        "Do not answer the user yet. "
                        "Reason only in this pass. "
                        "Keep it to at most three very short bullets or two short sentences, then stop."
                    ),
                },
            ]
            if on_phase_start:
                on_phase_start(ModelPhaseStart("tool_plan", streamed=on_final_delta is not None, attempt=1, reason="chat_thinking"))
            with self.transport.backend_thinking(True):
                thinking_result = self.transport.chat_once(
                    thinking_messages,
                    temperature=temperature,
                    max_tokens=thinking_max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                )
            if on_model_step:
                on_model_step(ModelStepMetrics.from_result(loop=loop, result=thinking_result, phase="chat_thinking"))
            final_messages = [
                *call_messages,
                {"role": "assistant", "content": thinking_result.content},
                {
                    "role": "user",
                    "content": (
                        "Now provide the final answer only. "
                        "Start the answer with 'Final answer:'. "
                        "Do not include reasoning, plan, or prompt analysis. "
                        "Answer the user directly now."
                    ),
                },
            ]
            with self.transport.backend_thinking(False):
                result = self.transport.chat_final(
                    final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                    on_model_step=on_model_step,
                    on_phase_start=on_phase_start,
                    loop=loop + 1,
                )
            self.runtime.messages.append({"role": "assistant", "content": result.content})
            return result
        result = self.transport.chat_final(
            call_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=loop,
        )
        self.runtime.messages.append({"role": "assistant", "content": result.content})
        return result


@dataclass(frozen=True)
class ToolLoopEnvironment:
    runtime: ChatRuntime

    def run(
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
        on_phase_start: Callable[[ModelPhaseStart], None] | None,
        tool_names: tuple[str, ...] | None,
        initial_tool_calls: list[dict[str, object]] | dict[str, object] | None = None,
    ) -> ToolResultBundle:
        result = run_tool_loop(
            self.runtime,
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
        )
        return ToolResultBundle(
            result=result,
            tool_names=tool_names,
        )


@dataclass(frozen=True)
class FinalFromToolEnvironment:
    runtime: ChatRuntime
    transport: TransportEnvironment

    def answer(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None,
        loop: int,
        use_tool_prompt: bool,
    ) -> FinalAnswerResult:
        call_messages = self.runtime._with_final_tool_prompt() if use_tool_prompt else self.runtime.messages
        policy = build_final_tool_policy(call_messages, max_tokens=max_tokens, streamed=on_final_delta is not None)
        stream_buffer = _BufferedDeltaSink() if on_final_delta is not None and policy.incomplete_retry_allowed else None
        final_delta = stream_buffer.write if stream_buffer is not None else on_final_delta
        used_retry_or_repair_pass = False
        repair_attempt = 0
        repair_failed = False
        compact_retry_attempted = False
        compact_retry_failed = False
        if on_phase_start:
            on_phase_start(
                ModelPhaseStart(
                    "final_from_tool",
                    streamed=stream_buffer is None and on_final_delta is not None,
                    attempt=1,
                )
            )
        with self.transport.backend_thinking(False):
            if on_final_delta is None:
                result = self.runtime.backend.chat(policy.messages, temperature=temperature, max_tokens=policy.max_tokens)
            else:
                result = self.runtime.backend.chat_stream(
                    policy.messages,
                    temperature=temperature,
                    max_tokens=policy.max_tokens,
                    on_delta=final_delta,
                    on_progress=on_progress,
                )
        best_non_empty_result = result if result.content.strip() else None
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop, result=result, phase="final_from_tool"))
        retry_reason = final_from_tool_retry_reason(
            result,
            length_retry_allowed=policy.length_retry_allowed,
            incomplete_retry_allowed=policy.incomplete_retry_allowed,
            messages=policy.messages,
        )
        if retry_reason is not None:
            used_retry_or_repair_pass = True
            previous_result = result
            retry_messages = [*policy.messages, final_tool_retry_instruction()]
            retry_max_tokens = final_tool_retry_max_tokens(max_tokens, web_fetch_result=policy.web_fetch_result)
            if on_phase_start:
                on_phase_start(
                    ModelPhaseStart(
                        "final_from_tool_retry",
                        streamed=stream_buffer is None and on_final_delta is not None,
                        attempt=2,
                        reason=retry_reason,
                    )
                )
            with self.transport.backend_thinking(False):
                if on_final_delta is None:
                    retry_candidate = self.runtime.backend.chat(retry_messages, temperature=temperature, max_tokens=retry_max_tokens)
                else:
                    retry_candidate = self.runtime.backend.chat_stream(
                        retry_messages,
                        temperature=temperature,
                        max_tokens=retry_max_tokens,
                        on_delta=final_delta,
                        on_progress=on_progress,
                    )
            if retry_candidate.content.strip():
                best_non_empty_result = retry_candidate
            result = _prefer_non_empty_retry_result(previous_result, retry_candidate)
            if on_model_step:
                on_model_step(
                    ModelStepMetrics.from_result(
                        loop=loop + 1,
                        result=retry_candidate,
                        phase="final_from_tool_retry",
                        retry_reason=retry_reason,
                    )
                )
        completeness = classify_final_answer_completeness(result.content, messages=policy.messages)
        if (
            policy.incomplete_retry_allowed
            and result.finish_reason == "stop"
            and (
                not completeness.is_complete
                or contradicts_successful_pdf_extraction(policy.messages, result.content)
            )
            and repair_attempt < 1
        ):
            used_retry_or_repair_pass = True
            repair_attempt += 1
            contradiction = contradicts_successful_pdf_extraction(policy.messages, result.content)
            repair_instruction = (
                pdf_extraction_repair_prompt(policy.messages)
                if contradiction
                else (
                    "The previous answer was incomplete, malformed, or not yet a final answer. "
                    "Write one short final answer now using only the existing tool results. "
                    "Use plain prose only. No headings. No bullet list unless absolutely necessary. "
                    "No code fences. No thinking or plan. No partial identifiers. "
                    "Limit yourself to three to five sentences. "
                    "Do not repeat fragments from the previous answer. Do not call tools."
                )
            )
            repair_messages = [*policy.messages, {"role": "user", "content": repair_instruction}]
            repair_max_tokens = final_tool_retry_max_tokens(max_tokens, web_fetch_result=policy.web_fetch_result)
            if on_phase_start:
                on_phase_start(
                    ModelPhaseStart(
                        "final_from_tool_completion_repair",
                        streamed=stream_buffer is None and on_final_delta is not None,
                        attempt=3,
                        reason="pdf_contradiction" if contradiction else completeness.status,
                    )
                )
            with self.transport.backend_thinking(False):
                if on_final_delta is None:
                    result = self.runtime.backend.chat(repair_messages, temperature=temperature, max_tokens=repair_max_tokens)
                else:
                    result = self.runtime.backend.chat_stream(
                        repair_messages,
                        temperature=temperature,
                        max_tokens=repair_max_tokens,
                        on_delta=final_delta,
                        on_progress=on_progress,
                    )
            if result.content.strip():
                best_non_empty_result = result
            if on_model_step:
                on_model_step(
                    ModelStepMetrics.from_result(
                        loop=loop + 2 + repair_attempt,
                        result=result,
                        phase="final_from_tool_completion_repair",
                        retry_reason="incomplete_final",
                    )
                )
            completeness = classify_final_answer_completeness(result.content, messages=policy.messages)
            if (
                result.finish_reason == "stop"
                and (
                    not completeness.is_complete
                    or contradicts_successful_pdf_extraction(policy.messages, result.content)
                )
            ):
                repair_failed = True
        compact_retry_reason = final_from_tool_compact_retry_reason(result, messages=policy.messages)
        if compact_retry_reason is not None:
            used_retry_or_repair_pass = True
            compact_retry_attempted = True
            original_result = best_non_empty_result or result
            compact_messages = [*policy.messages, final_tool_compact_retry_instruction()]
            compact_max_tokens = final_tool_compact_retry_max_tokens(max_tokens, messages=policy.messages)
            if on_phase_start:
                on_phase_start(
                    ModelPhaseStart(
                        "final_from_tool_compact_retry",
                        streamed=stream_buffer is None and on_final_delta is not None,
                        attempt=4,
                        reason=compact_retry_reason,
                    )
                )
            with self.transport.backend_thinking(False):
                if on_final_delta is None:
                    candidate = self.runtime.backend.chat(compact_messages, temperature=temperature, max_tokens=compact_max_tokens)
                else:
                    candidate = self.runtime.backend.chat_stream(
                        compact_messages,
                        temperature=temperature,
                        max_tokens=compact_max_tokens,
                        on_delta=final_delta,
                        on_progress=on_progress,
                    )
            if candidate.content.strip():
                best_non_empty_result = candidate
            if on_model_step:
                on_model_step(
                    ModelStepMetrics.from_result(
                        loop=loop + 3,
                        result=candidate,
                        phase="final_from_tool_compact_retry",
                        retry_reason=compact_retry_reason,
                    )
                )
            if _prefer_compact_retry_result(candidate, original_result, messages=policy.messages):
                result = candidate
            else:
                result = original_result
                compact_retry_failed = True
        if not result.content.strip() and best_non_empty_result is not None:
            result = best_non_empty_result
        if stream_buffer is not None and on_final_delta is not None:
            if used_retry_or_repair_pass and result.content:
                on_final_delta(result.content)
            else:
                for chunk in stream_buffer.chunks:
                    on_final_delta(chunk)
        self.runtime.messages.append({"role": "assistant", "content": result.content})
        return FinalAnswerResult(
            result=result,
            used_retry_or_repair_pass=used_retry_or_repair_pass,
            repair_attempts=repair_attempt,
            repair_failed=repair_failed,
            compact_retry_attempted=compact_retry_attempted,
            compact_retry_failed=compact_retry_failed,
        )


@dataclass(frozen=True)
class ContinueEnvironment:
    runtime: ChatRuntime
    transport: TransportEnvironment

    def continue_last_response(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None,
    ) -> ContinueResult:
        initial_kind = self._continuation_kind_before_continue()
        if initial_kind == "thinking":
            result = self._continue_with_prompt_fallback(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
                on_model_step=on_model_step,
                on_phase_start=on_phase_start,
                loop=1,
                force_disable_backend_thinking=True,
            )
            return ContinueResult(result=result, used_native_continue=False, used_prompt_fallback=True)
        if self.runtime.can_continue_last_response() and hasattr(self.runtime.backend, "continue_current"):
            result = ContinueController(
                backend=self.runtime.backend,
                thinking=self.runtime._thinking(),
                merge_results=merge_chat_results,
            ).continue_until_settled(
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
                max_passes=3,
            )
            if on_phase_start:
                on_phase_start(
                    ModelPhaseStart(
                        "chat_continue_native",
                        streamed=on_final_delta is not None,
                        attempt=1,
                        reason="length",
                    )
                )
            if on_model_step:
                on_model_step(ModelStepMetrics.from_result(loop=1, result=result, phase="chat_continue_native"))
            if self.runtime._thinking().continuation_kind_for(content=result.content, finish_reason=result.finish_reason) is not None:
                self.runtime.messages.append({"role": "assistant", "content": result.content})
                fallback = self._continue_with_prompt_fallback(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                    on_model_step=on_model_step,
                    on_phase_start=on_phase_start,
                    loop=2,
                    force_disable_backend_thinking=True,
                )
                return ContinueResult(
                    result=merge_chat_results(result, fallback),
                    used_native_continue=True,
                    used_prompt_fallback=True,
                )
            self.runtime.messages.append({"role": "assistant", "content": result.content})
            return ContinueResult(result=result, used_native_continue=True, used_prompt_fallback=False)
        result = self._continue_with_prompt_fallback(
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_progress=on_progress,
            on_model_step=on_model_step,
            on_phase_start=on_phase_start,
            loop=1,
        )
        return ContinueResult(result=result, used_native_continue=False, used_prompt_fallback=True)

    def _continuation_kind_before_continue(self) -> str | None:
        if self.runtime.client_state.last_content:
            current_kind = self.runtime._thinking().continuation_kind_for(
                content=self.runtime.client_state.last_content,
                finish_reason=self.runtime.client_state.last_finish_reason,
            )
            if current_kind is not None:
                return current_kind
        if self.runtime.client_state.continuation_kind is not None:
            return self.runtime.client_state.continuation_kind
        if last_assistant_has_open_reasoning(self.runtime.messages):
            return "thinking"
        if self.runtime.last_visible_finish_reason == "length":
            return "final_answer"
        return None

    def _continue_with_prompt_fallback(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_progress: Callable[[StreamProgress], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        on_phase_start: Callable[[ModelPhaseStart], None] | None,
        loop: int,
        force_disable_backend_thinking: bool = False,
    ) -> ChatResult:
        if force_disable_backend_thinking:
            original_request = last_user_text(self.runtime.messages)
            continuation_prompt = (
                "Stop reasoning now and write only the missing final answer. "
                "Start the answer with 'Final answer:'. "
                "Write exactly one short final-answer sentence. "
                "Do not continue or repeat any thought text. "
                "Do not explain the plan again. "
                "No hidden reasoning, no plan, no thinking markers. "
                + (
                    f"Answer concisely and directly to the original user request: {original_request!r}. "
                    if original_request
                    else "Answer concisely and directly from the existing context. "
                )
            )
        else:
            continuation_prompt = (
                "Continue exactly from where the previous answer stopped. "
                "If the previous answer stopped inside reasoning, do not restart it from the beginning. "
                "Finish the remaining reasoning briefly, then continue with the missing final answer only. "
                "Do not repeat already written text."
                if last_assistant_has_open_reasoning(self.runtime.messages)
                else "Continue exactly from where the previous answer stopped. Do not repeat already written text."
            )
        self.runtime.messages.append({"role": "user", "content": continuation_prompt})
        if force_disable_backend_thinking:
            with self.transport.backend_thinking(False):
                result = self.transport.chat_final(
                    self.runtime.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                    on_model_step=on_model_step,
                    on_phase_start=on_phase_start,
                    loop=loop,
                    repair_incomplete_final=False,
                )
        else:
            result = self.transport.chat_final(
                self.runtime.messages,
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
                on_model_step=on_model_step,
                on_phase_start=on_phase_start,
                loop=loop,
                repair_incomplete_final=False,
            )
        if force_disable_backend_thinking and (
            self.runtime._thinking().continuation_kind_for(
                content=result.content,
                finish_reason=result.finish_reason,
            ) is not None
            or looks_like_incomplete_final(result.content)
        ):
            result = replace(
                result,
                content="error: model did not produce a final answer after continuation",
                finish_reason="stop",
            )
        self.runtime.messages.append({"role": "assistant", "content": result.content})
        return result


@dataclass(frozen=True)
class FileInputEnvironment:
    resolver: FileInputResolver

    def resolve(self, prompt: str, *, allowed_tool_names: tuple[str, ...] | None) -> FileResolutionResult:
        try:
            images, audios = self.resolver.resolve_media(prompt)
        except ValueError as exc:
            return FileResolutionResult(images=[], audios=[], bypass_tool_route=False, error=str(exc))
        return FileResolutionResult(
            images=images,
            audios=audios,
            bypass_tool_route=self.resolver.should_bypass_tool_route(prompt, allowed_tool_names),
        )


def merge_chat_results(first: ChatResult, second: ChatResult) -> ChatResult:
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


def is_empty_final_response(result: ChatResult) -> bool:
    return not result.tool_calls and result.finish_reason == "stop" and not result.content.strip()


def unsupported_tool_mode_result(result: ChatResult) -> ChatResult:
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


def is_tool_argument_json_error(exc: RuntimeError) -> bool:
    return "Failed to parse tool call arguments as JSON" in str(exc)


def needs_final_completion_repair(content: str) -> bool:
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


def contradicts_successful_pdf_extraction(messages: list[Message], content: str) -> bool:
    if not content.strip():
        return False
    has_pdf_success = any(
        message.get("role") == "tool"
        and isinstance(message.get("content"), str)
        and "shell_output_pdf_text: true" in str(message.get("content"))
        for message in messages
    )
    return has_pdf_success and _FILE_MISSING_RE.search(content) is not None


def pdf_extraction_repair_prompt(messages: list[Message]) -> str:
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


class _BufferedDeltaSink:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> None:
        if text:
            self.chunks.append(text)


def _prefer_compact_retry_result(candidate: ChatResult, original: ChatResult, *, messages: list[Message]) -> bool:
    candidate_text = candidate.content.strip()
    if not candidate_text:
        return False
    candidate_complete = classify_final_answer_completeness(candidate.content, messages=messages).is_complete
    original_complete = classify_final_answer_completeness(original.content, messages=messages).is_complete
    candidate_repetitive = is_repetitive_final_answer(candidate.content)
    original_repetitive = is_repetitive_final_answer(original.content)
    if candidate.finish_reason == "stop" and original.finish_reason != "stop":
        return candidate_complete and not candidate_repetitive
    if candidate_complete and not candidate_repetitive and (not original_complete or original_repetitive):
        return True
    if candidate_complete and original.finish_reason == "length" and len(candidate_text) <= len(original.content.strip()):
        return True
    return False


def _prefer_non_empty_retry_result(previous: ChatResult, candidate: ChatResult) -> ChatResult:
    if candidate.content.strip():
        return candidate
    if previous.content.strip():
        return previous
    return candidate
