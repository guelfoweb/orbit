from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from orbit.backend import ChatBackend, ChatResult
from orbit.backend.base import Message
from orbit.runtime.final_policy import (
    build_final_tool_policy,
    final_from_tool_retry_reason,
    final_tool_retry_instruction,
    final_tool_retry_max_tokens,
    has_list_like_tool_result as _has_list_like_tool_result,
)
from orbit.runtime.media import AudioInput, ImageInput, load_referenced_media
from orbit.runtime.messages import (
    TOOL_CALL_JSON_RETRY_PROMPT,
    message_content,
    with_chat_system_prompt,
    with_final_tool_system_prompt,
    with_media_system_prompt,
    with_command_system_prompt,
)

from orbit.runtime.command_request import (
    ToolRoute,
    decision_tool_names,
    parse_command_decision,
    parse_command_decision_from_tool_calls,
    command_tool_call_from_content,
    command_tool_call_from_tool_calls,
)
from orbit.runtime.results import error_result
from orbit.runtime.session_memory import MemoryRefresh, maybe_refresh_memory
from orbit.runtime.tool_loop import run_tool_loop
from orbit.runtime.tool_result_compaction import (
    ToolResultCompactionReport,
    compact_tool_results,
    persistent_messages as persistent_tool_result_messages,
)
from orbit.runtime.turn_trace import ModelStepMetrics


ROUTE_MAX_TOKENS = 128


@dataclass
class ChatRuntime:
    backend: ChatBackend
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
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

    def __post_init__(self) -> None:
        if not self.messages and self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def ask(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        images: list[ImageInput] | None = None,
        audios: list[AudioInput] | None = None,
        on_final_delta: Callable[[str], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
    ) -> ChatResult:
        self.messages.append({"role": "user", "content": message_content(prompt, images or [], audios or [])})
        result = self._chat_final(
            self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_model_step=on_model_step,
            loop=1,
        )
        self.messages.append({"role": "assistant", "content": result.content})
        return result

    def ask_chat(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
    ) -> ChatResult:
        self.last_memory_refresh = None
        self.refresh_memory_if_needed(temperature=temperature)
        self.messages.append({"role": "user", "content": prompt})
        result = self._chat_final(
            with_chat_system_prompt(self.messages),
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_model_step=on_model_step,
            loop=1,
        )
        self.messages.append({"role": "assistant", "content": result.content})
        return result

    def continue_last_response(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
    ) -> ChatResult:
        self.messages.append(
            {
                "role": "user",
                "content": "Continue exactly from where the previous answer stopped. Do not repeat already written text.",
            }
        )
        result = self._chat_final(
            self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_model_step=on_model_step,
            loop=1,
        )
        self.messages.append({"role": "assistant", "content": result.content})
        return result

    def ask_with_tools(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        workdir,
        max_loops: int = 10,
        on_final_delta: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_tool_result: Callable[[str, int, str, str], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
        tool_names: tuple[str, ...] | None = None,
    ) -> ChatResult:
        self.last_memory_refresh = None
        self.refresh_memory_if_needed(temperature=temperature)
        self.messages.append({"role": "user", "content": prompt})
        return self._run_tool_loop(
            temperature=temperature,
            max_tokens=max_tokens,
            workdir=workdir,
            max_loops=max_loops,
            on_final_delta=on_final_delta,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_model_step=on_model_step,
            tool_names=tool_names,
        )

    def ask_auto(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        workdir,
        max_loops: int = 10,
        on_final_delta: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_tool_result: Callable[[str, int, str, str], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> ChatResult:
        self.last_memory_refresh = None
        self.refresh_memory_if_needed(temperature=temperature)
        media_result = self._ask_media_if_referenced(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            workdir=workdir,
            on_final_delta=on_final_delta,
            on_model_step=on_model_step,
        )
        if media_result is not None:
            return media_result
        self.messages.append({"role": "user", "content": prompt})
        command_max_tokens = _bounded_internal_max_tokens(max_tokens, ROUTE_MAX_TOKENS)
        streamed_final_retry = False
        retried_empty_final = False
        command_messages = with_command_system_prompt(self.messages)
        first = self.backend.chat(command_messages, temperature=temperature, max_tokens=command_max_tokens)
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=1, result=first, phase="route"))
        command_content = first.content
        decision = parse_command_decision_from_tool_calls(first.tool_calls) or parse_command_decision(command_content)
        if decision is None:
            if first.finish_reason == "length":
                if on_final_delta is None:
                    first = self.backend.chat(self.messages, temperature=temperature, max_tokens=max_tokens)
                else:
                    streamed_final_retry = True
                    first = self.backend.chat_stream(
                        self.messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        on_delta=on_final_delta,
                    )
                if on_model_step:
                    on_model_step(ModelStepMetrics.from_result(loop=2, result=first, phase="chat_final_retry"))
            if _is_empty_final_response(first):
                retried_empty_final = True
                first = self._chat_final(
                    self.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=2,
                )
            self.messages.append({"role": "assistant", "content": first.content})
            if on_final_delta and not streamed_final_retry and not retried_empty_final:
                on_final_delta(first.content)
            return first
        if decision.route == ToolRoute.CHAT:
            chat_messages = with_chat_system_prompt(self.messages)
            result = self._chat_final(
                chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_model_step=on_model_step,
                loop=2,
            )
            self.messages.append({"role": "assistant", "content": result.content})
            return result
        if decision.route == ToolRoute.MEDIA:
            if allowed_tool_names is not None:
                result = _unsupported_tool_mode_result(first)
                self.messages.append({"role": "assistant", "content": result.content})
                if on_final_delta:
                    on_final_delta(result.content)
                return result
            return self._ask_referenced_media(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                workdir=workdir,
                on_final_delta=on_final_delta,
                on_model_step=on_model_step,
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
            return result
        return self._run_tool_loop(
            temperature=temperature,
            max_tokens=max_tokens,
            workdir=workdir,
            max_loops=max_loops,
            on_final_delta=on_final_delta,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_model_step=on_model_step,
            tool_names=tools,
            initial_tool_calls=(
                command_tool_call_from_tool_calls(first.tool_calls, tools)
                or command_tool_call_from_content(command_content, tools)
            ),
        )

    def _ask_media_if_referenced(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        workdir,
        on_final_delta: Callable[[str], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
    ) -> ChatResult | None:
        try:
            images, audios = load_referenced_media(prompt, workdir=workdir)
        except ValueError:
            return None
        if not images and not audios:
            return None
        self.messages.append({"role": "user", "content": message_content(prompt, images, audios)})
        call_messages = with_media_system_prompt(self.messages)
        result = self._chat_final(
            call_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_model_step=on_model_step,
            loop=1,
        )
        self.messages.append({"role": "assistant", "content": result.content})
        return result

    def _ask_referenced_media(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        workdir,
        on_final_delta: Callable[[str], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        command_result: ChatResult,
    ) -> ChatResult:
        try:
            images, audios = load_referenced_media(prompt, workdir=workdir)
        except ValueError as exc:
            result = error_result(str(exc), command_result)
            self.messages.append({"role": "assistant", "content": result.content})
            if on_final_delta:
                on_final_delta(result.content)
            return result
        if not images and not audios:
            result = error_result("error: MEDIA route requested but no local image/audio path was found in the prompt", command_result)
            self.messages.append({"role": "assistant", "content": result.content})
            if on_final_delta:
                on_final_delta(result.content)
            return result

        self.messages[-1] = {"role": "user", "content": message_content(prompt, images, audios)}
        call_messages = with_media_system_prompt(self.messages)
        result = self._chat_final(
            call_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
            on_model_step=on_model_step,
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
        on_tool_call: Callable[[str, str], None] | None,
        on_tool_result: Callable[[str, int, str, str], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        tool_names: tuple[str, ...] | None,
        initial_tool_calls: list[dict[str, object]] | dict[str, object] | None = None,
    ) -> ChatResult:
        return run_tool_loop(
            self,
            temperature=temperature,
            max_tokens=max_tokens,
            workdir=workdir,
            max_loops=max_loops,
            on_final_delta=on_final_delta,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_model_step=on_model_step,
            tool_names=tool_names,
            initial_tool_calls=initial_tool_calls,
        )

    def _answer_from_tool_results(
        self,
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        loop: int,
        use_tool_prompt: bool,
    ) -> ChatResult:
        call_messages = with_final_tool_system_prompt(self.messages) if use_tool_prompt else self.messages
        policy = build_final_tool_policy(call_messages, max_tokens=max_tokens, streamed=on_final_delta is not None)
        if on_final_delta is None:
            result = self.backend.chat(policy.messages, temperature=temperature, max_tokens=policy.max_tokens)
        else:
            result = self.backend.chat_stream(
                policy.messages,
                temperature=temperature,
                max_tokens=policy.max_tokens,
                on_delta=on_final_delta,
            )
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop, result=result, phase="final_from_tool"))
        retry_reason = final_from_tool_retry_reason(result, length_retry_allowed=policy.length_retry_allowed)
        if retry_reason is not None:
            retry_messages = [*policy.messages, final_tool_retry_instruction()]
            retry_max_tokens = final_tool_retry_max_tokens(max_tokens, web_fetch_result=policy.web_fetch_result)
            if on_final_delta is None:
                result = self.backend.chat(retry_messages, temperature=temperature, max_tokens=retry_max_tokens)
            else:
                result = self.backend.chat_stream(
                    retry_messages,
                    temperature=temperature,
                    max_tokens=retry_max_tokens,
                    on_delta=on_final_delta,
                )
            if on_model_step:
                on_model_step(
                    ModelStepMetrics.from_result(
                        loop=loop + 1,
                        result=result,
                        phase="final_from_tool_retry",
                        retry_reason=retry_reason,
                    )
                )
        self.messages.append({"role": "assistant", "content": result.content})
        return result

    def _chat_final(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        loop: int,
    ) -> ChatResult:
        result = self._chat_once(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
        )
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop, result=result, phase="chat_final"))
        if not _is_empty_final_response(result):
            return result

        retry = self._chat_once(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_final_delta=on_final_delta,
        )
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop + 1, result=retry, phase="chat_final_retry"))
        if not _is_empty_final_response(retry):
            return retry

        error = replace(retry, content="error: model returned an empty response twice", finish_reason="empty_response")
        if on_final_delta:
            on_final_delta(error.content)
        return error

    def _chat_once(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_final_delta: Callable[[str], None] | None,
    ) -> ChatResult:
        if on_final_delta is None:
            return self.backend.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self.backend.chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            on_delta=on_final_delta,
        )

    def _chat_tool_call_once(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, object]],
        on_final_delta: Callable[[str], None] | None,
    ) -> ChatResult:
        try:
            if on_final_delta is None:
                return self.backend.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
            return self.backend.chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                on_delta=on_final_delta,
            )
        except RuntimeError as exc:
            if not _is_tool_argument_json_error(exc):
                raise
        retry_messages = [*messages, {"role": "system", "content": TOOL_CALL_JSON_RETRY_PROMPT}]
        if on_final_delta is None:
            return self.backend.chat(retry_messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        return self.backend.chat_stream(
            retry_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            on_delta=on_final_delta,
        )

    def reset(self) -> None:
        self.messages.clear()
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
        self.last_memory_refresh = None
        if self.last_memory_refresh_message_count is not None and self.last_memory_refresh_message_count > len(self.messages):
            self.last_memory_refresh_message_count = None

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


def _bounded_internal_max_tokens(max_tokens: int, internal_max: int) -> int:
    return max(1, min(max_tokens, internal_max))


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
