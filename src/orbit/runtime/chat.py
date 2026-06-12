from __future__ import annotations

from dataclasses import dataclass, field
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
    message_content,
    with_chat_system_prompt,
    with_final_tool_system_prompt,
    with_media_system_prompt,
    with_route_system_prompt,
    with_tool_call_system_prompt,
)
from dataclasses import replace

from orbit.runtime.route_request import (
    ToolRoute,
    decision_tool_names,
    parse_route_decision,
    parse_route_decision_from_tool_calls,
    route_like_tool_call,
    route_tool_call_from_content,
    route_tool_call_from_tool_calls,
)
from orbit.runtime.results import error_result
from orbit.runtime.session_memory import MemoryRefresh, maybe_refresh_memory, should_refresh_for_append
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_calls import execute_tool_call
from orbit.runtime.tool_loop_state import ToolLoopState
from orbit.runtime.tool_message import assistant_tool_call_message, tool_result_message
from orbit.runtime.tool_result_compaction import (
    ToolResultCompactionReport,
    compact_tool_results,
    persistent_messages as persistent_tool_result_messages,
)
from orbit.runtime.tools import tool_names as all_tool_names
from orbit.runtime.turn_trace import ModelStepMetrics


ROUTE_MAX_TOKENS = 64
TOOL_CALL_MAX_TOKENS = 96


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
        on_tool_result: Callable[[str, int, str], None] | None = None,
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
        on_tool_result: Callable[[str, int, str], None] | None = None,
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
        route_max_tokens = _bounded_internal_max_tokens(max_tokens, ROUTE_MAX_TOKENS)
        streamed_final_retry = False
        retried_empty_final = False
        route_messages = with_route_system_prompt(self.messages)
        first = self.backend.chat(route_messages, temperature=temperature, max_tokens=route_max_tokens)
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=1, result=first, phase="route"))
        route_content = first.content
        decision = parse_route_decision_from_tool_calls(first.tool_calls) or parse_route_decision(route_content)
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
                route_result=first,
            )
        tools = decision_tool_names(decision, prompt)
        if allowed_tool_names is not None:
            allowed = set(allowed_tool_names)
            tools = tuple(tool for tool in tools if tool in allowed)
            if decision.route == ToolRoute.FILE_EDIT and not _has_edit_capability(tools):
                tools = ()
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
                route_tool_call_from_tool_calls(first.tool_calls, tools)
                or route_tool_call_from_content(route_content, tools)
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
        route_result: ChatResult,
    ) -> ChatResult:
        try:
            images, audios = load_referenced_media(prompt, workdir=workdir)
        except ValueError as exc:
            result = error_result(str(exc), route_result)
            self.messages.append({"role": "assistant", "content": result.content})
            if on_final_delta:
                on_final_delta(result.content)
            return result
        if not images and not audios:
            result = error_result("error: MEDIA route requested but no local image/audio path was found in the prompt", route_result)
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
        on_tool_result: Callable[[str, int, str], None] | None,
        on_model_step: Callable[[ModelStepMetrics], None] | None,
        tool_names: tuple[str, ...] | None,
        initial_tool_calls: list[dict[str, object]] | dict[str, object] | None = None,
    ) -> ChatResult:
        allowed_tool_names = tool_names or all_tool_names()
        executor = HybridToolExecutor(
            backend=self.backend if hasattr(self.backend, "server_tools") else None,
            workdir=workdir,
            allowed_tool_names=allowed_tool_names,
        )
        tools = executor.tool_definitions()
        last_result: ChatResult | None = None
        state = ToolLoopState(allowed_tool_names)
        if initial_tool_calls:
            calls = [initial_tool_calls] if isinstance(initial_tool_calls, dict) else list(initial_tool_calls)
            state.increment_round()
            self.messages.append(assistant_tool_call_message("", calls))
            for tool_call in calls:
                signature = state.mark_tool_call(tool_call)
                if on_tool_call:
                    on_tool_call(*signature)
                execution = execute_tool_call(tool_call, chunk_budget=state.chunk_budget, executor=executor)
                tool_result = execution.result
                if on_tool_result:
                    on_tool_result(tool_result.name, len(tool_result.content), execution.source)
                self.messages.append(tool_result_message(tool_call, tool_result))
            if state.round_limit_reached():
                return self._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=state.tool_rounds + 1,
                    use_tool_prompt=state.used_tool_call_prompt,
                )
        for loop_index in range(1, max_loops + 1):
            call_messages = with_tool_call_system_prompt(self.messages)
            state.used_tool_call_prompt = True
            tool_max_tokens = _bounded_internal_max_tokens(max_tokens, TOOL_CALL_MAX_TOKENS)
            if on_final_delta is None:
                result = self.backend.chat(call_messages, temperature=temperature, max_tokens=tool_max_tokens, tools=tools)
            else:
                result = self.backend.chat_stream(
                    call_messages,
                    temperature=temperature,
                    max_tokens=tool_max_tokens,
                    tools=tools,
                    on_delta=on_final_delta,
                )
            last_result = result
            if on_model_step:
                on_model_step(ModelStepMetrics.from_result(loop=loop_index, result=result, phase="tool_call" if result.tool_calls else None))
            if result.finish_reason == "length" and not result.tool_calls:
                result = self.backend.chat(call_messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
                if on_model_step:
                    on_model_step(ModelStepMetrics.from_result(loop=loop_index + 1, result=result, phase="tool_call_retry" if result.tool_calls else None))
            if not result.tool_calls and _is_empty_final_response(result):
                result = self.backend.chat(call_messages, temperature=temperature, max_tokens=tool_max_tokens, tools=tools)
                if on_model_step:
                    on_model_step(ModelStepMetrics.from_result(loop=loop_index + 1, result=result, phase="tool_call_retry" if result.tool_calls else None))
                if not result.tool_calls and _is_empty_final_response(result):
                    return self._chat_final(
                        with_chat_system_prompt(self.messages),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        on_final_delta=on_final_delta,
                        on_model_step=on_model_step,
                        loop=loop_index + 2,
                    )
            if result.tool_calls and not _all_tool_calls_allowed(result.tool_calls, allowed_tool_names):
                route_tool_call = route_tool_call_from_tool_calls(result.tool_calls, allowed_tool_names)
                if route_tool_call is not None:
                    result = replace(result, content="", finish_reason="tool_calls", tool_calls=[route_tool_call])
            if not result.tool_calls:
                route_tool_call = route_like_tool_call(result.content, allowed_tool_names)
                if route_tool_call is not None:
                    result = replace(result, content="", finish_reason="tool_calls", tool_calls=[route_tool_call])
            self.messages.append(assistant_tool_call_message(result.content, result.tool_calls))
            if not result.tool_calls:
                return result
            state.increment_round()
            for tool_call in result.tool_calls:
                if state.has_seen_tool_call(tool_call):
                    return self._answer_from_tool_results(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        on_final_delta=on_final_delta,
                        on_model_step=on_model_step,
                        loop=loop_index + 1,
                        use_tool_prompt=state.used_tool_call_prompt,
                    )
                signature = state.mark_tool_call(tool_call)
                if on_tool_call:
                    on_tool_call(*signature)
                execution = execute_tool_call(
                    tool_call,
                    chunk_budget=state.chunk_budget,
                    executor=executor,
                )
                tool_result = execution.result
                if on_tool_result:
                    on_tool_result(tool_result.name, len(tool_result.content), execution.source)
                if should_refresh_for_append(self.messages, tool_result.content, context_tokens=self.context_tokens):
                    self.refresh_memory_if_needed(temperature=temperature, force=True)
                self.messages.append(tool_result_message(tool_call, tool_result))
            if state.round_limit_reached():
                return self._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=loop_index + 1,
                    use_tool_prompt=state.used_tool_call_prompt,
                )
        return last_result or ChatResult(
            content="error: tool loop did not produce a response",
            model=None,
            finish_reason=None,
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
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
        finish_reason="unsupported_route",
        tool_calls=[],
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cached_tokens=result.cached_tokens,
        prompt_tokens_per_second=result.prompt_tokens_per_second,
        generation_tokens_per_second=result.generation_tokens_per_second,
    )


def _has_edit_capability(tool_names: tuple[str, ...]) -> bool:
    return bool({"write_file", "edit_file", "apply_diff", "make_directory", "delete_path"}.intersection(tool_names))


def _all_tool_calls_allowed(tool_calls: list[dict[str, object]], allowed_tool_names: tuple[str, ...]) -> bool:
    allowed = set(allowed_tool_names)
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return False
        name = function.get("name")
        if not isinstance(name, str) or name not in allowed:
            return False
    return True
