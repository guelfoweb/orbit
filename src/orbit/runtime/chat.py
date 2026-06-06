from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from orbit.backend import ChatBackend, ChatResult
from orbit.backend.base import Message
from orbit.runtime.media import AudioInput, ImageInput
from orbit.runtime.session_memory import MemoryRefresh, maybe_refresh_memory, should_refresh_for_append
from orbit.runtime.tools import execute_tool, tool_definitions
from orbit.runtime.turn_trace import ModelStepMetrics


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
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
    ) -> ChatResult:
        self.messages.append({"role": "user", "content": _message_content(prompt, images or [], audios or [])})
        result = self.backend.chat(self.messages, temperature=temperature, max_tokens=max_tokens)
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=1, result=result))
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
        on_tool_result: Callable[[str, int], None] | None = None,
        on_model_step: Callable[[ModelStepMetrics], None] | None = None,
    ) -> ChatResult:
        self.last_memory_refresh = None
        self.refresh_memory_if_needed(temperature=temperature)
        self.messages.append({"role": "user", "content": prompt})
        tools = tool_definitions()
        last_result: ChatResult | None = None
        chunk_budget = {"read_file_chunks": 0, "fetch_url_chunks": 0}
        seen_tool_calls: set[tuple[str, str]] = set()
        for loop_index in range(1, max_loops + 1):
            if on_final_delta is None:
                result = self.backend.chat(self.messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
            else:
                result = self.backend.chat_stream(
                    self.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    on_delta=on_final_delta,
                )
            last_result = result
            if on_model_step:
                on_model_step(ModelStepMetrics.from_result(loop=loop_index, result=result))
            assistant_message: Message = {"role": "assistant", "content": result.content}
            if result.tool_calls:
                assistant_message["tool_calls"] = result.tool_calls
            self.messages.append(assistant_message)
            if not result.tool_calls:
                return result
            for tool_call in result.tool_calls:
                signature = _tool_call_signature(tool_call)
                if signature in seen_tool_calls:
                    return ChatResult(
                        content=f"error: repeated tool call stopped: {signature[0]} {signature[1]}",
                        model=result.model,
                        finish_reason="repeated_tool_call",
                        tool_calls=[],
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        cached_tokens=result.cached_tokens,
                        prompt_tokens_per_second=result.prompt_tokens_per_second,
                        generation_tokens_per_second=result.generation_tokens_per_second,
                    )
                seen_tool_calls.add(signature)
                if on_tool_call:
                    on_tool_call(*signature)
                tool_result = _execute_tool_call(tool_call, workdir=workdir, chunk_budget=chunk_budget)
                if on_tool_result:
                    on_tool_result(tool_result.name, len(tool_result.content))
                if should_refresh_for_append(self.messages, tool_result.content, context_tokens=self.context_tokens):
                    self.refresh_memory_if_needed(temperature=temperature, force=True)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(tool_call),
                        "name": tool_result.name,
                        "content": tool_result.content,
                    }
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


def _message_content(
    prompt: str,
    images: list[ImageInput],
    audios: list[AudioInput],
) -> str | list[dict[str, object]]:
    if not images and not audios:
        return prompt
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image.data_url}})
    for audio in audios:
        content.append({"type": "input_audio", "input_audio": {"data": audio.data, "format": audio.format}})
    return content


def _execute_tool_call(tool_call: dict[str, object], *, workdir, chunk_budget: dict[str, int] | None = None) -> object:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return execute_tool("unknown", {}, workdir=workdir, chunk_budget=chunk_budget)
    name = function.get("name")
    arguments = function.get("arguments", {})
    return execute_tool(
        name if isinstance(name, str) else "unknown",
        arguments,
        workdir=workdir,
        chunk_budget=chunk_budget,
    )


def _tool_call_id(tool_call: dict[str, object]) -> str:
    value = tool_call.get("id")
    return value if isinstance(value, str) and value else "tool-call"


def _tool_call_signature(tool_call: dict[str, object]) -> tuple[str, str]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ("unknown", "{}")
    name = function.get("name")
    arguments = function.get("arguments", "")
    return (name if isinstance(name, str) else "unknown", arguments if isinstance(arguments, str) else str(arguments))
