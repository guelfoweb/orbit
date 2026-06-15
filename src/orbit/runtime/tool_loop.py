from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable

from orbit.backend import ChatResult
from orbit.backend.base import Message
from orbit.runtime.command_request import command_like_tool_call, command_tool_call_from_tool_calls
from orbit.runtime.messages import with_chat_system_prompt, with_tool_call_system_prompt
from orbit.runtime.session_memory import should_refresh_for_append
from orbit.runtime.shell_guardrails import (
    SHELL_FULL_COMPLETION_GUARD_PROMPT,
    SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT,
    SHELL_FULL_CONTRACT_RETRY_PROMPT,
    SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT,
    SHELL_FULL_MINIMAL_PATCH_GUARD_PROMPT,
    SHELL_FULL_SEMANTIC_REPAIR_PROMPT,
    is_incomplete_shell_json_or_command_error,
    is_content_evidence_shell_command,
    is_metadata_only_shell_command,
    is_mutating_shell_command,
    is_mutative_user_request,
    is_repairable_shell_error,
    is_shell_full_contract_error,
    is_shell_full_execution_error,
    looks_like_broad_file_rewrite,
    shell_repair_prompt,
    should_verify_shell_mutation,
)
from orbit.runtime.tool_arguments import parse_tool_arguments_or_empty
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_calls import execute_tool_call
from orbit.runtime.tool_loop_state import ToolLoopState
from orbit.runtime.tool_message import assistant_tool_call_message, tool_result_message
from orbit.runtime.tools import default_tool_names
from orbit.runtime.turn_trace import ModelStepMetrics


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


TOOL_CALL_MAX_TOKENS = 96
MUTATIVE_TOOL_CALL_MAX_TOKENS = _env_int("ORBIT_MUTATIVE_TOOL_CALL_MAX_TOKENS", 160)
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
    mutation_verification_pending = False
    mutation_verification_repair_pending = False
    mutation_verification_repair_used = False
    mutation_semantic_repair_pending = False
    mutation_semantic_repair_used = False
    content_evidence_guard_pending = False
    content_evidence_guard_used = False
    completion_guard_pending = False
    completion_guard_used = False
    minimal_patch_guard_pending = False
    minimal_patch_guard_used = False
    shell_commands_seen = 0
    content_evidence_seen = False
    metadata_only_rejections = 0
    shell_mutation_attempted = False
    shell_mutation_succeeded = False
    shell_full_enabled = "exec_shell_full_command" in allowed_tool_names
    user_prompt = _last_user_text(runtime.messages)
    suppress_tool_delta = (lambda _delta: None) if on_final_delta is not None and shell_full_enabled else None

    def should_nudge_completion() -> bool:
        return (
            shell_full_enabled
            and not completion_guard_used
            and shell_commands_seen > 0
            and not shell_mutation_attempted
            and is_mutative_user_request(user_prompt)
        )

    def request_completion_guard() -> None:
        nonlocal completion_guard_pending
        nonlocal completion_guard_used
        completion_guard_pending = True
        completion_guard_used = True
        runtime.completion_guard_nudges += 1

    def should_nudge_minimal_patch(*, broad_rewrite_seen: bool, length_truncated: bool = False) -> bool:
        return (
            shell_full_enabled
            and not minimal_patch_guard_used
            and is_mutative_user_request(user_prompt)
            and shell_commands_seen > 0
            and not shell_mutation_succeeded
            and (broad_rewrite_seen or length_truncated)
        )

    def request_minimal_patch_guard() -> None:
        nonlocal minimal_patch_guard_pending
        nonlocal minimal_patch_guard_used
        minimal_patch_guard_pending = True
        minimal_patch_guard_used = True
        runtime.minimal_patch_guard_nudges += 1

    def request_mutation_semantic_repair() -> None:
        nonlocal mutation_semantic_repair_pending
        nonlocal mutation_semantic_repair_used
        mutation_semantic_repair_pending = True
        mutation_semantic_repair_used = True
        runtime.mutation_semantic_repairs += 1

    def should_nudge_content_evidence() -> bool:
        return (
            shell_full_enabled
            and not content_evidence_guard_used
            and is_mutative_user_request(user_prompt)
            and metadata_only_rejections > 0
            and not content_evidence_seen
            and not shell_mutation_attempted
        )

    def request_content_evidence_guard() -> None:
        nonlocal content_evidence_guard_pending
        nonlocal content_evidence_guard_used
        content_evidence_guard_pending = True
        content_evidence_guard_used = True
        runtime.content_evidence_guard_nudges += 1

    def has_pending_internal_request() -> bool:
        return (
            contract_retry_pending
            or shell_repair_prompt_pending is not None
            or shell_empty_result_check_pending
            or mutation_verification_pending
            or mutation_verification_repair_pending
            or mutation_semantic_repair_pending
            or content_evidence_guard_pending
            or completion_guard_pending
            or minimal_patch_guard_pending
        )

    def update_state_after_tool_result(
        tool_call: dict[str, object],
        tool_result,
        *,
        is_mutation_verification: bool,
        is_mutation_verification_repair: bool,
        is_mutation_semantic_repair: bool,
        is_content_evidence_guard: bool,
        is_completion_guard: bool,
        is_minimal_patch_guard: bool,
    ) -> None:
        nonlocal contract_retry_pending
        nonlocal mutation_verification_pending
        nonlocal mutation_verification_repair_pending
        nonlocal mutation_verification_repair_used
        nonlocal mutation_semantic_repair_pending
        nonlocal content_evidence_seen
        nonlocal metadata_only_rejections
        nonlocal shell_empty_result_check_pending
        nonlocal shell_empty_result_check_used
        nonlocal shell_error_final_pending
        nonlocal shell_repair_prompt_pending
        nonlocal shell_repair_retries
        nonlocal shell_commands_seen
        nonlocal shell_mutation_attempted
        nonlocal shell_mutation_succeeded

        if tool_result.name != "exec_shell_full_command":
            return
        command = _shell_command_from_tool_call(tool_call)
        raw_arguments = _shell_raw_arguments_from_tool_call(tool_call)
        command_is_mutating = bool(command and is_mutating_shell_command(command))
        command_is_content_evidence = bool(command and is_content_evidence_shell_command(command))
        if command:
            shell_commands_seen += 1
        if command_is_mutating:
            shell_mutation_attempted = True
        if is_content_evidence_guard:
            runtime.content_evidence_guard_commands += 1
        if is_mutation_semantic_repair:
            runtime.mutation_semantic_repair_commands += 1
        if is_minimal_patch_guard:
            runtime.minimal_patch_guard_commands += 1
            if command_is_mutating:
                runtime.minimal_patch_guard_successes += 1
            else:
                runtime.minimal_patch_guard_failures += 1
        if (
            is_incomplete_shell_json_or_command_error(tool_result.content)
            and should_nudge_minimal_patch(
                broad_rewrite_seen=looks_like_broad_file_rewrite(command) or looks_like_broad_file_rewrite(raw_arguments),
                length_truncated=True,
            )
        ):
            request_minimal_patch_guard()
            return
        if is_completion_guard:
            runtime.completion_guard_commands += 1
            if command_is_mutating:
                runtime.completion_guard_successes += 1
            else:
                runtime.completion_guard_failures += 1
        if is_shell_full_contract_error(tool_result.content):
            if command and is_metadata_only_shell_command(command):
                metadata_only_rejections += 1
            if should_nudge_content_evidence():
                request_content_evidence_guard()
            else:
                contract_retry_pending = True
            if is_content_evidence_guard:
                runtime.content_evidence_guard_failures += 1
            return
        if is_shell_full_execution_error(tool_result.content):
            if is_mutation_verification:
                if (
                    is_repairable_shell_error(tool_result.content)
                    and not mutation_verification_repair_used
                ):
                    mutation_verification_repair_used = True
                    mutation_verification_repair_pending = True
                    runtime.mutation_verification_repairs += 1
                    shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
                    return
                runtime.mutation_verification_failures += 1
                shell_error_final_pending = True
                return
            if is_mutation_verification_repair:
                runtime.mutation_verification_failures += 1
                shell_error_final_pending = True
                return
            if is_mutation_semantic_repair:
                if is_repairable_shell_error(tool_result.content) and shell_repair_retries < MAX_SHELL_REPAIR_RETRIES:
                    shell_repair_retries += 1
                    shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
                    return
                runtime.mutation_semantic_repair_failures += 1
                shell_error_final_pending = True
                return
            if is_repairable_shell_error(tool_result.content) and shell_repair_retries < MAX_SHELL_REPAIR_RETRIES:
                shell_repair_retries += 1
                shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
                return
            shell_error_final_pending = True
            return
        if command_is_mutating:
            shell_mutation_succeeded = True
        if tool_result.content.strip():
            if command_is_content_evidence and not command_is_mutating:
                content_evidence_seen = True
                if is_content_evidence_guard:
                    runtime.content_evidence_guard_successes += 1
            if is_mutation_verification and not mutation_semantic_repair_used:
                request_mutation_semantic_repair()
            return
        if is_content_evidence_guard:
            runtime.content_evidence_guard_failures += 1
        if is_mutation_verification_repair and command_is_mutating and not mutation_semantic_repair_used:
            request_mutation_semantic_repair()
            return
        if is_mutation_verification or is_mutation_verification_repair:
            runtime.mutation_verification_failures += 1
            shell_error_final_pending = True
            return
        if is_mutation_semantic_repair:
            runtime.mutation_semantic_repair_failures += 1
            shell_error_final_pending = True
            return
        if (
            command
            and not shell_empty_result_check_used
            and should_verify_shell_mutation(command, user_prompt=_last_user_text(runtime.messages))
        ):
            shell_empty_result_check_pending = True
            shell_empty_result_check_used = True
            mutation_verification_pending = True
            runtime.mutation_verifications += 1
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
            update_state_after_tool_result(
                tool_call,
                tool_result,
                is_mutation_verification=False,
                is_mutation_verification_repair=False,
                is_mutation_semantic_repair=False,
                is_content_evidence_guard=False,
                is_completion_guard=False,
                is_minimal_patch_guard=False,
            )
            if on_tool_result:
                on_tool_result(tool_result.name, len(tool_result.content), execution.source, tool_result.content)
            runtime.messages.append(tool_result_message(tool_call, tool_result))
        if (
            not shell_error_final_pending
            and not contract_retry_pending
            and shell_repair_prompt_pending is None
            and not has_pending_internal_request()
            and should_nudge_completion()
        ):
            request_completion_guard()
        if shell_error_final_pending or (
            not has_pending_internal_request()
        ):
            if not completion_guard_pending:
                return runtime._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_model_step=on_model_step,
                    loop=state.tool_rounds + 1,
                    use_tool_prompt=state.used_tool_call_prompt,
                )
        if state.round_limit_reached() and not has_pending_internal_request():
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
        executing_mutation_verification = mutation_verification_pending
        executing_mutation_verification_repair = mutation_verification_repair_pending
        executing_mutation_semantic_repair = mutation_semantic_repair_pending
        executing_content_evidence_guard = content_evidence_guard_pending
        executing_completion_guard = completion_guard_pending
        executing_minimal_patch_guard = minimal_patch_guard_pending
        if shell_repair_prompt_pending is not None:
            call_messages = [*call_messages, {"role": "user", "content": shell_repair_prompt_pending}]
        elif shell_empty_result_check_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT}]
        elif mutation_semantic_repair_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_SEMANTIC_REPAIR_PROMPT}]
        elif content_evidence_guard_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT}]
        elif minimal_patch_guard_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_MINIMAL_PATCH_GUARD_PROMPT}]
        elif completion_guard_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_COMPLETION_GUARD_PROMPT}]
        state.used_tool_call_prompt = True
        tool_max_tokens = _tool_call_max_tokens(
            max_tokens,
            mutative=(
                shell_repair_prompt_pending is not None
                or shell_empty_result_check_pending
                or mutation_semantic_repair_pending
                or content_evidence_guard_pending
                or minimal_patch_guard_pending
                or completion_guard_pending
                or mutation_verification_pending
                or mutation_verification_repair_pending
                or (shell_full_enabled and is_mutative_user_request(user_prompt))
            ),
        )
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
        if mutation_verification_pending:
            mutation_verification_pending = False
        if mutation_verification_repair_pending:
            mutation_verification_repair_pending = False
        if mutation_semantic_repair_pending:
            mutation_semantic_repair_pending = False
        if content_evidence_guard_pending:
            content_evidence_guard_pending = False
        if completion_guard_pending:
            completion_guard_pending = False
        if minimal_patch_guard_pending:
            minimal_patch_guard_pending = False
        if (
            result.finish_reason == "length"
            and result.tool_calls
            and should_nudge_minimal_patch(
                broad_rewrite_seen=any(
                    looks_like_broad_file_rewrite(_shell_raw_arguments_from_tool_call(tool_call))
                    for tool_call in result.tool_calls
                ),
                length_truncated=True,
            )
        ):
            request_minimal_patch_guard()
            continue
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
        if result.tool_calls:
            route_tool_call = command_tool_call_from_tool_calls(result.tool_calls, allowed_tool_names)
            if route_tool_call is not None:
                result = replace(result, content="", finish_reason="tool_calls", tool_calls=[route_tool_call])
        if not result.tool_calls:
            route_tool_call = command_like_tool_call(result.content, allowed_tool_names)
            if route_tool_call is not None:
                result = replace(result, content="", finish_reason="tool_calls", tool_calls=[route_tool_call])
        runtime.messages.append(assistant_tool_call_message(result.content, result.tool_calls))
        if not result.tool_calls:
            if executing_mutation_semantic_repair and result.content.strip().upper() != "OK":
                runtime.mutation_semantic_repair_failures += 1
            if executing_content_evidence_guard:
                runtime.content_evidence_guard_failures += 1
            if executing_completion_guard:
                runtime.completion_guard_failures += 1
            if executing_minimal_patch_guard:
                runtime.minimal_patch_guard_failures += 1
            if state.tool_rounds > 0 and shell_full_enabled:
                if should_nudge_completion():
                    request_completion_guard()
                    continue
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
            update_state_after_tool_result(
                tool_call,
                tool_result,
                is_mutation_verification=executing_mutation_verification,
                is_mutation_verification_repair=executing_mutation_verification_repair,
                is_mutation_semantic_repair=executing_mutation_semantic_repair,
                is_content_evidence_guard=executing_content_evidence_guard,
                is_completion_guard=executing_completion_guard,
                is_minimal_patch_guard=executing_minimal_patch_guard,
            )
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
        if state.round_limit_reached() and not has_pending_internal_request():
            if should_nudge_completion():
                request_completion_guard()
                continue
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


def _tool_call_max_tokens(max_tokens: int, *, mutative: bool) -> int:
    internal_max = MUTATIVE_TOOL_CALL_MAX_TOKENS if mutative else TOOL_CALL_MAX_TOKENS
    return _bounded_internal_max_tokens(max_tokens, internal_max)


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


def _shell_command_from_tool_call(tool_call: dict[str, object]) -> str | None:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    if function.get("name") != "exec_shell_full_command":
        return None
    args = parse_tool_arguments_or_empty(function.get("arguments"))
    command = args.get("command") if isinstance(args, dict) else None
    return command if isinstance(command, str) and command.strip() else None


def _shell_raw_arguments_from_tool_call(tool_call: dict[str, object]) -> str | None:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    if function.get("name") != "exec_shell_full_command":
        return None
    arguments = function.get("arguments")
    return arguments if isinstance(arguments, str) and arguments.strip() else None
