from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Callable

from orbit.backend import ChatBackend, ChatResult
from orbit.backend.base import Message
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
from orbit.runtime.tool_calls import execute_tool_call, tool_call_id, tool_call_signature
from orbit.runtime.tools import tool_names as all_tool_names
from orbit.runtime.turn_trace import ModelStepMetrics


ROUTE_MAX_TOKENS = 64
TOOL_CALL_MAX_TOKENS = 96
FINAL_FROM_TOOL_MIN_TOKENS = 256
LARGE_FILE_FINAL_MAX_TOKENS = 128
WEB_FETCH_FINAL_MAX_TOKENS = 72
LIST_FINAL_MAX_TOKENS = 96


@dataclass
class ChatRuntime:
    backend: ChatBackend
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    context_tokens: int | None = None
    last_memory_refresh: MemoryRefresh | None = None
    last_memory_refresh_message_count: int | None = None
    memory_refresh_cooldown_messages: int = 4

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
        if not tools:
            result = ChatResult(
                content="error: no suitable tool is available for this request",
                model=first.model,
                finish_reason="unsupported_route",
                tool_calls=[],
                prompt_tokens=first.prompt_tokens,
                completion_tokens=first.completion_tokens,
                cached_tokens=first.cached_tokens,
                prompt_tokens_per_second=first.prompt_tokens_per_second,
                generation_tokens_per_second=first.generation_tokens_per_second,
            )
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
        chunk_budget = {"read_file_chunks": 0, "fetch_url_chunks": 0}
        seen_tool_calls: set[tuple[str, str]] = set()
        tool_rounds = 0
        tool_round_limit = _tool_round_limit(allowed_tool_names)
        used_tool_call_prompt = False
        if initial_tool_calls:
            calls = [initial_tool_calls] if isinstance(initial_tool_calls, dict) else list(initial_tool_calls)
            tool_rounds += 1
            self.messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            for tool_call in calls:
                signature = tool_call_signature(tool_call)
                seen_tool_calls.add(signature)
                if on_tool_call:
                    on_tool_call(*signature)
                execution = execute_tool_call(tool_call, chunk_budget=chunk_budget, executor=executor)
                tool_result = execution.result
                if on_tool_result:
                    on_tool_result(tool_result.name, len(tool_result.content), execution.source)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id(tool_call),
                        "name": tool_result.name,
                        "content": tool_result.content,
                    }
                )
            if tool_rounds >= tool_round_limit:
                return self._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=tool_rounds + 1,
                    use_tool_prompt=used_tool_call_prompt,
                )
        for loop_index in range(1, max_loops + 1):
            call_messages = with_tool_call_system_prompt(self.messages)
            used_tool_call_prompt = True
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
            assistant_message: Message = {"role": "assistant", "content": result.content}
            if result.tool_calls:
                assistant_message["tool_calls"] = result.tool_calls
            self.messages.append(assistant_message)
            if not result.tool_calls:
                return result
            tool_rounds += 1
            for tool_call in result.tool_calls:
                signature = tool_call_signature(tool_call)
                if signature in seen_tool_calls:
                    return self._answer_from_tool_results(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        on_final_delta=on_final_delta,
                        on_model_step=on_model_step,
                        loop=loop_index + 1,
                        use_tool_prompt=used_tool_call_prompt,
                    )
                seen_tool_calls.add(signature)
                if on_tool_call:
                    on_tool_call(*signature)
                execution = execute_tool_call(
                    tool_call,
                    chunk_budget=chunk_budget,
                    executor=executor,
                )
                tool_result = execution.result
                if on_tool_result:
                    on_tool_result(tool_result.name, len(tool_result.content), execution.source)
                if should_refresh_for_append(self.messages, tool_result.content, context_tokens=self.context_tokens):
                    self.refresh_memory_if_needed(temperature=temperature, force=True)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id(tool_call),
                        "name": tool_result.name,
                        "content": tool_result.content,
                    }
                )
            if tool_rounds >= tool_round_limit:
                return self._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=loop_index + 1,
                    use_tool_prompt=used_tool_call_prompt,
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
        large_file_excerpt = _has_large_file_excerpt(self.messages)
        web_fetch_result = _has_web_fetch_tool_result(self.messages)
        web_search_result = _has_tool_result(self.messages, "search_web")
        list_like_result = _has_list_like_tool_result(self.messages)
        shell_result = _has_tool_result(self.messages, "exec_shell_command")
        read_file_result = _has_tool_result(self.messages, "read_file")
        if large_file_excerpt:
            call_messages = [
                *call_messages,
                {
                    "role": "user",
                    "content": (
                        "Use the available large-file excerpt only. "
                        "Answer in at most five short bullets, each under twelve words. Do not quote long passages. "
                        "Do not request more chunks unless the user explicitly asked for exhaustive analysis."
                    ),
                },
            ]
        elif web_search_result:
            call_messages = [
                *call_messages,
                {
                    "role": "user",
                    "content": (
                        "Use only the search results already available. "
                        "Answer in at most four short bullets. "
                        "Keep the main facts and cite source names only when useful. "
                        "Do not add background beyond the results. "
                        "Expand only if the user asks for more detail."
                    ),
                },
            ]
        elif web_fetch_result:
            call_messages = [
                *call_messages,
                {
                    "role": "user",
                    "content": (
                        "Use only the fetched page text already available. "
                        "Write exactly two concise bullets. "
                        "Use the requested language; if unspecified, use the fetched page language. "
                        "Focus on the central thesis and key messages. "
                        "Each bullet must be under eighteen words. No introduction. Stop after the second bullet. "
                        "Do not request more chunks unless the user explicitly asked for exhaustive analysis."
                    ),
                },
            ]
        elif list_like_result:
            call_messages = [
                *call_messages,
                {"role": "user", "content": "Return only the listed names, compactly. No categories or explanations."},
            ]
        elif shell_result:
            call_messages = [
                *call_messages,
                {
                    "role": "user",
                    "content": (
                        "Use only the command output. "
                        "Return at most six compact findings. "
                        "Preserve important numbers and names. "
                        "Do not explain generic concepts. Expand only if asked."
                    ),
                },
            ]
        elif read_file_result:
            call_messages = [
                *call_messages,
                {
                    "role": "user",
                    "content": (
                        "Use only the file content. "
                        "Respect any requested length. "
                        "If no length is requested, answer concisely. "
                        "Expand only if asked."
                    ),
                },
            ]
        if large_file_excerpt:
            final_max_tokens = min(max_tokens, LARGE_FILE_FINAL_MAX_TOKENS)
        elif web_fetch_result:
            final_max_tokens = min(max_tokens, WEB_FETCH_FINAL_MAX_TOKENS)
        elif list_like_result:
            final_max_tokens = min(max_tokens, LIST_FINAL_MAX_TOKENS)
        else:
            final_max_tokens = max(max_tokens, FINAL_FROM_TOOL_MIN_TOKENS)
        if on_final_delta is None:
            result = self.backend.chat(call_messages, temperature=temperature, max_tokens=final_max_tokens)
        else:
            result = self.backend.chat_stream(
                call_messages,
                temperature=temperature,
                max_tokens=final_max_tokens,
                on_delta=on_final_delta,
            )
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop, result=result, phase="final_from_tool"))
        length_retry_allowed = on_final_delta is None and (large_file_excerpt or web_fetch_result)
        retry_reason = _final_from_tool_retry_reason(result, length_retry_allowed=length_retry_allowed)
        if retry_reason is not None:
            retry_messages = [
                *call_messages,
                {
                    "role": "user",
                    "content": (
                        "Do not call tools. Provide a shorter final answer from the available tool result now."
                    ),
                },
            ]
            retry_max_tokens = min(max(max_tokens, FINAL_FROM_TOOL_MIN_TOKENS), WEB_FETCH_FINAL_MAX_TOKENS if web_fetch_result else max(max_tokens, FINAL_FROM_TOOL_MIN_TOKENS))
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
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

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
        return result.changed

    def _memory_refresh_in_cooldown(self) -> bool:
        if self.last_memory_refresh_message_count is None:
            return False
        return len(self.messages) - self.last_memory_refresh_message_count < self.memory_refresh_cooldown_messages


def _bounded_internal_max_tokens(max_tokens: int, internal_max: int) -> int:
    return max(1, min(max_tokens, internal_max))


def _tool_round_limit(tool_names: tuple[str, ...]) -> int:
    edit_tools = {"write_file", "edit_file", "apply_diff", "make_directory", "delete_path"}
    return 2 if edit_tools.intersection(tool_names) else 1


def _has_large_file_excerpt(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = message.get("content")
            return isinstance(content, str) and "large_file_excerpt: true" in content
    return False


def _has_web_fetch_tool_result(messages: list[Message]) -> bool:
    return _has_tool_result(messages, "fetch_url")


def _has_tool_result(messages: list[Message], name: str) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        return message.get("name") == name
    return False


def _has_list_like_tool_result(messages: list[Message]) -> bool:
    last_shell_command = _last_exec_shell_command(messages)
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        if name in {"list_files", "file_glob_search"}:
            return True
        if name == "exec_shell_command":
            return _is_list_shell_command(last_shell_command)
        return False
    return False


def _last_exec_shell_command(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for tool_call in reversed(calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or function.get("name") != "exec_shell_command":
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, dict):
                continue
            command = arguments.get("command")
            if isinstance(command, str):
                return command
    return None


def _is_list_shell_command(command: str | None) -> bool:
    if not command:
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    return tokens[0] in {"ls", "find"}


def _is_empty_final_response(result: ChatResult) -> bool:
    return not result.tool_calls and result.finish_reason == "stop" and not result.content.strip()


def _final_from_tool_retry_reason(result: ChatResult, *, length_retry_allowed: bool) -> str | None:
    if result.tool_calls:
        return "tool_call_in_final"
    if _contains_raw_tool_call(result.content):
        return "raw_tool_call"
    if not result.content and result.finish_reason == "stop":
        return "empty_final"
    if length_retry_allowed and result.finish_reason == "length":
        return "length"
    return None


def _contains_raw_tool_call(content: str) -> bool:
    return "<|tool_call>" in content or "<tool_call|>" in content


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
