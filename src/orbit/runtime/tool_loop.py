from __future__ import annotations

from dataclasses import replace
from typing import Callable

from orbit.backend import ChatResult
from orbit.backend.base import Message
from orbit.runtime.command_request import command_like_tool_call, command_tool_call_from_tool_calls
from orbit.runtime.messages import with_chat_system_prompt, with_tool_call_system_prompt
from orbit.runtime.session_memory import should_refresh_for_append
from orbit.runtime.shell_guardrails import (
    SHELL_FULL_CONTRACT_RETRY_PROMPT,
    SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT,
    is_repairable_shell_error,
    is_shell_full_contract_error,
    is_shell_full_execution_error,
    shell_repair_prompt,
)
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_calls import execute_tool_call
from orbit.runtime.tool_loop_state import ToolLoopState
from orbit.runtime.tool_message import assistant_tool_call_message, tool_result_message
from orbit.runtime.tools import default_tool_names
from orbit.runtime.turn_trace import ModelStepMetrics


TOOL_CALL_MAX_TOKENS = 96
MAX_SHELL_REPAIR_RETRIES = 2


def run_tool_loop(
    runtime,
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
    allowed_tool_names = tool_names or default_tool_names()
    executor = HybridToolExecutor(
        backend=runtime.backend if hasattr(runtime.backend, "server_tools") else None,
        workdir=workdir,
        allowed_tool_names=allowed_tool_names,
        user_prompt=_last_user_text(runtime.messages),
    )
    tools = executor.tool_definitions()
    last_result: ChatResult | None = None
    state = ToolLoopState(allowed_tool_names)
    contract_retry_pending = False
    shell_empty_result_check_pending = False
    shell_empty_result_check_used = False
    shell_error_final_pending = False
    shell_repair_prompt_pending: str | None = None
    shell_repair_retries = 0
    shell_full_enabled = "exec_shell_full_command" in allowed_tool_names
    suppress_tool_delta = (lambda _delta: None) if on_final_delta is not None and shell_full_enabled else None
    if initial_tool_calls:
        calls = [initial_tool_calls] if isinstance(initial_tool_calls, dict) else list(initial_tool_calls)
        state.increment_round()
        runtime.messages.append(assistant_tool_call_message("", calls))
        for tool_call in calls:
            signature = state.mark_tool_call(tool_call)
            if on_tool_call:
                on_tool_call(*signature)
            execution = execute_tool_call(tool_call, chunk_budget=state.chunk_budget, executor=executor)
            tool_result = execution.result
            if tool_result.name == "exec_shell_full_command" and is_shell_full_contract_error(tool_result.content):
                contract_retry_pending = True
            elif (
                tool_result.name == "exec_shell_full_command"
                and is_shell_full_execution_error(tool_result.content)
                and is_repairable_shell_error(tool_result.content)
                and shell_repair_retries < MAX_SHELL_REPAIR_RETRIES
            ):
                shell_repair_retries += 1
                shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
            elif tool_result.name == "exec_shell_full_command" and is_shell_full_execution_error(tool_result.content):
                shell_error_final_pending = True
            elif tool_result.name == "exec_shell_full_command" and not shell_empty_result_check_used and not tool_result.content.strip():
                shell_empty_result_check_pending = True
                shell_empty_result_check_used = True
            if on_tool_result:
                on_tool_result(tool_result.name, len(tool_result.content), execution.source, tool_result.content)
            runtime.messages.append(tool_result_message(tool_call, tool_result))
        if shell_error_final_pending or (
            not contract_retry_pending and shell_repair_prompt_pending is None and not shell_empty_result_check_pending
        ):
            return runtime._answer_from_tool_results(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_model_step=on_model_step,
                loop=state.tool_rounds + 1,
                use_tool_prompt=state.used_tool_call_prompt,
            )
        if state.round_limit_reached() and not contract_retry_pending:
            return runtime._answer_from_tool_results(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_model_step=on_model_step,
                loop=state.tool_rounds + 1,
                use_tool_prompt=state.used_tool_call_prompt,
            )
    for loop_index in range(1, max_loops + 1):
        call_messages = with_tool_call_system_prompt(runtime.messages)
        if shell_repair_prompt_pending is not None:
            call_messages = [*call_messages, {"role": "user", "content": shell_repair_prompt_pending}]
        elif shell_empty_result_check_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT}]
        state.used_tool_call_prompt = True
        tool_max_tokens = _bounded_internal_max_tokens(max_tokens, TOOL_CALL_MAX_TOKENS)
        tool_delta_callback = suppress_tool_delta if shell_full_enabled and (state.tool_rounds > 0 or contract_retry_pending) else on_final_delta
        result = runtime._chat_tool_call_once(
            call_messages,
            temperature=temperature,
            max_tokens=tool_max_tokens,
            tools=tools,
            on_final_delta=tool_delta_callback,
        )
        last_result = result
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop_index, result=result, phase="tool_call" if result.tool_calls else None))
        if contract_retry_pending and not result.tool_calls:
            retry_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_CONTRACT_RETRY_PROMPT}]
            result = runtime._chat_tool_call_once(
                retry_messages,
                temperature=temperature,
                max_tokens=tool_max_tokens,
                tools=tools,
                on_final_delta=suppress_tool_delta,
            )
            last_result = result
            contract_retry_pending = False
            if on_model_step:
                on_model_step(
                    ModelStepMetrics.from_result(
                        loop=loop_index + 1,
                        result=result,
                        phase="tool_call_retry" if result.tool_calls else None,
                    )
                )
            elif result.tool_calls:
                contract_retry_pending = False
        if shell_repair_prompt_pending is not None:
            shell_repair_prompt_pending = None
        if shell_empty_result_check_pending:
            shell_empty_result_check_pending = False
        if result.finish_reason == "length" and not result.tool_calls and state.tool_rounds > 0:
            return runtime._answer_from_tool_results(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_model_step=on_model_step,
                loop=loop_index + 1,
                use_tool_prompt=state.used_tool_call_prompt,
            )
        if result.finish_reason == "length" and not result.tool_calls:
            result = runtime.backend.chat(call_messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
            if on_model_step:
                on_model_step(ModelStepMetrics.from_result(loop=loop_index + 1, result=result, phase="tool_call_retry" if result.tool_calls else None))
        if not result.tool_calls and _is_empty_final_response(result):
            result = runtime.backend.chat(call_messages, temperature=temperature, max_tokens=tool_max_tokens, tools=tools)
            if on_model_step:
                on_model_step(ModelStepMetrics.from_result(loop=loop_index + 1, result=result, phase="tool_call_retry" if result.tool_calls else None))
            if not result.tool_calls and _is_empty_final_response(result):
                return runtime._chat_final(
                    with_chat_system_prompt(runtime.messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=loop_index + 2,
                )
        if result.tool_calls and not _all_tool_calls_allowed(result.tool_calls, allowed_tool_names):
            route_tool_call = command_tool_call_from_tool_calls(result.tool_calls, allowed_tool_names)
            if route_tool_call is not None:
                result = replace(result, content="", finish_reason="tool_calls", tool_calls=[route_tool_call])
        if not result.tool_calls:
            route_tool_call = command_like_tool_call(result.content, allowed_tool_names)
            if route_tool_call is not None:
                result = replace(result, content="", finish_reason="tool_calls", tool_calls=[route_tool_call])
        runtime.messages.append(assistant_tool_call_message(result.content, result.tool_calls))
        if not result.tool_calls:
            if state.tool_rounds > 0 and shell_full_enabled:
                return runtime._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=loop_index + 1,
                    use_tool_prompt=state.used_tool_call_prompt,
                )
            return result
        state.increment_round()
        for tool_call in result.tool_calls:
            if state.has_seen_tool_call(tool_call):
                return runtime._answer_from_tool_results(
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
            if tool_result.name == "exec_shell_full_command" and is_shell_full_contract_error(tool_result.content):
                contract_retry_pending = True
            elif (
                tool_result.name == "exec_shell_full_command"
                and is_shell_full_execution_error(tool_result.content)
                and is_repairable_shell_error(tool_result.content)
                and shell_repair_retries < MAX_SHELL_REPAIR_RETRIES
            ):
                shell_repair_retries += 1
                shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
            elif tool_result.name == "exec_shell_full_command" and is_shell_full_execution_error(tool_result.content):
                shell_error_final_pending = True
            elif tool_result.name == "exec_shell_full_command" and not shell_empty_result_check_used and not tool_result.content.strip():
                shell_empty_result_check_pending = True
                shell_empty_result_check_used = True
            if on_tool_result:
                on_tool_result(tool_result.name, len(tool_result.content), execution.source, tool_result.content)
            if should_refresh_for_append(runtime.messages, tool_result.content, context_tokens=runtime.context_tokens):
                runtime.refresh_memory_if_needed(temperature=temperature, force=True)
            runtime.messages.append(tool_result_message(tool_call, tool_result))
        if shell_error_final_pending:
            return runtime._answer_from_tool_results(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_model_step=on_model_step,
                loop=loop_index + 1,
                use_tool_prompt=state.used_tool_call_prompt,
            )
        if state.round_limit_reached() and not contract_retry_pending:
            return runtime._answer_from_tool_results(
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


def _bounded_internal_max_tokens(max_tokens: int, internal_max: int) -> int:
    return max(1, min(max_tokens, internal_max))


def _is_empty_final_response(result: ChatResult) -> bool:
    return not result.tool_calls and result.finish_reason == "stop" and not result.content.strip()


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


def _last_user_text(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        return content if isinstance(content, str) else None
    return None
