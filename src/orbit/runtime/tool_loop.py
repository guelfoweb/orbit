from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable

from orbit.backend import ChatResult
from orbit.backend.base import Message, StreamProgress
from orbit.runtime.command_request import command_like_tool_call, command_tool_call_from_tool_calls
from orbit.runtime.messages import with_chat_system_prompt, with_tool_call_system_prompt
from orbit.runtime.session_memory import should_refresh_for_append
from orbit.runtime.shell_guardrails import (
    SHELL_FULL_COMPLETION_GUARD_PROMPT,
    SHELL_FULL_ANALYSIS_COMPLETION_GUARD_PROMPT,
    SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT,
    SHELL_FULL_CONTRACT_RETRY_PROMPT,
    SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT,
    build_shell_full_file_recovery_guard_prompt,
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
from orbit.runtime.tool_loop_state import (
    EVIDENCE_CANDIDATE_PATHS_FOUND,
    EVIDENCE_DIRECT_READ_FAILED,
    RECONSIDER_ANALYSIS_COMPLETION,
    RECONSIDER_COMPLETION,
    RECONSIDER_CONTENT_EVIDENCE,
    RECONSIDER_FILE_RECOVERY,
    RECONSIDER_MINIMAL_PATCH,
    ToolLoopState,
    ToolTurnState,
)
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
FILE_RECOVERY_TOOL_CALL_MAX_TOKENS = 64
MAX_SHELL_REPAIR_RETRIES = 2


def run_tool_loop(
    runtime,
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
    user_prompt = _last_user_text(runtime.messages)
    turn = ToolTurnState(requested_user_path=_extract_requested_user_path(user_prompt))
    repair = turn.repair_state
    shell_full_enabled = "exec_shell_full_command" in allowed_tool_names
    suppress_tool_delta = (lambda _delta: None) if on_final_delta is not None and shell_full_enabled else None

    # These local helpers still couple policy and runtime counters in one place.
    # The state objects only store turn-local facts; they do not decide tasks.

    def should_nudge_completion() -> bool:
        return (
            shell_full_enabled
            and turn.can_reconsider(RECONSIDER_COMPLETION)
            and turn.shell_commands_seen > 0
            and not turn.shell_mutation_attempted
            and is_mutative_user_request(user_prompt)
        )

    def request_completion_guard() -> None:
        turn.pending_completion_guard = True
        turn.mark_reconsider(RECONSIDER_COMPLETION)
        runtime.completion_guard_nudges += 1

    def should_nudge_minimal_patch(
        *,
        broad_rewrite_seen: bool,
        length_truncated: bool = False,
        existing_file_rewrite: bool = False,
    ) -> bool:
        return (
            shell_full_enabled
            and turn.can_reconsider(RECONSIDER_MINIMAL_PATCH)
            and is_mutative_user_request(user_prompt)
            and (turn.shell_commands_seen > 0 or existing_file_rewrite)
            and not turn.shell_mutation_succeeded
            and (broad_rewrite_seen or length_truncated)
        )

    def request_minimal_patch_guard() -> None:
        turn.pending_minimal_patch_guard = True
        turn.mark_reconsider(RECONSIDER_MINIMAL_PATCH)
        runtime.minimal_patch_guard_nudges += 1

    def request_mutation_semantic_repair() -> None:
        repair.mutation_semantic_repair_pending = True
        repair.mutation_semantic_repair_used = True
        runtime.mutation_semantic_repairs += 1

    def should_nudge_content_evidence() -> bool:
        return (
            shell_full_enabled
            and turn.can_reconsider(RECONSIDER_CONTENT_EVIDENCE)
            and is_mutative_user_request(user_prompt)
            and turn.metadata_only_rejections > 0
            and not turn.content_evidence_satisfied
            and not turn.shell_mutation_attempted
        )

    def request_content_evidence_guard() -> None:
        turn.pending_content_evidence_guard = True
        turn.mark_reconsider(RECONSIDER_CONTENT_EVIDENCE)
        runtime.content_evidence_guard_nudges += 1

    def should_nudge_analysis_completion(result_tool_calls: list[dict[str, object]] | None) -> bool:
        if (
            not shell_full_enabled
            or not turn.can_reconsider(RECONSIDER_ANALYSIS_COMPLETION)
            or not turn.can_reconsider(RECONSIDER_FILE_RECOVERY)
            or is_mutative_user_request(user_prompt)
            or not turn.content_evidence_satisfied
            or state.tool_rounds <= 0
            or not result_tool_calls
        ):
            return False
        for tool_call in result_tool_calls:
            command = _shell_command_from_tool_call(tool_call)
            if not command:
                continue
            if is_mutating_shell_command(command):
                return False
            if not is_content_evidence_shell_command(command):
                return True
        return False

    def request_analysis_completion_guard() -> None:
        turn.pending_analysis_completion_guard = True
        turn.mark_reconsider(RECONSIDER_ANALYSIS_COMPLETION)

    def should_nudge_file_recovery(result_tool_calls: list[dict[str, object]] | None) -> bool:
        if (
            not shell_full_enabled
            or not turn.can_reconsider(RECONSIDER_FILE_RECOVERY)
            or not turn.requested_user_path
            or not result_tool_calls
            or turn.content_evidence_satisfied
        ):
            return False
        if turn.evidence_state not in {EVIDENCE_DIRECT_READ_FAILED, EVIDENCE_CANDIDATE_PATHS_FOUND}:
            return False
        for tool_call in result_tool_calls:
            command = _shell_command_from_tool_call(tool_call)
            if not command:
                continue
            if _is_direct_content_read_for_known_path(
                command,
                requested_path=turn.requested_user_path,
                candidate_paths=turn.candidate_paths,
            ):
                return False
            if turn.evidence_state == EVIDENCE_DIRECT_READ_FAILED and _is_targeted_file_discovery(command, requested_path=turn.requested_user_path):
                return False
            if _looks_like_discovery_command(command):
                return True
        return False

    def request_file_recovery_guard(
        *,
        last_command: str | None = None,
        last_failure_content: str | None = None,
        requested_path_exists: bool = False,
    ) -> None:
        if not turn.requested_user_path:
            return
        turn.pending_file_recovery_guard = True
        turn.mark_reconsider(RECONSIDER_FILE_RECOVERY)
        turn.pending_file_recovery_guard_prompt = build_shell_full_file_recovery_guard_prompt(
            requested_path=turn.requested_user_path,
            last_error=turn.last_error,
            candidate_paths=turn.candidate_paths,
            requested_path_exists=requested_path_exists,
            last_command=last_command,
            last_failure_content=last_failure_content,
        )

    def has_pending_internal_request() -> bool:
        return turn.has_pending_internal_request()

    def should_handoff_to_final_from_tool() -> bool:
        return (
            shell_full_enabled
            and turn.content_evidence_satisfied
            and turn.finalizable
            and not has_pending_internal_request()
            and not repair.shell_error_final_pending
            and not is_mutative_user_request(user_prompt)
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
        # This remains the main transition reducer for tool evidence and bounded
        # repair state. It updates turn-local state, while runtime counters stay
        # here to avoid leaking telemetry into the state objects themselves.
        if tool_result.name != "exec_shell_full_command":
            return
        command = _shell_command_from_tool_call(tool_call)
        raw_arguments = _shell_raw_arguments_from_tool_call(tool_call)
        command_is_mutating = bool(command and is_mutating_shell_command(command))
        command_is_content_evidence = bool(command and is_content_evidence_shell_command(command))
        tool_output_kind = _classify_shell_output_kind(command, tool_result.content, requested_path=turn.requested_user_path)
        turn.set_tool_result_kind(tool_output_kind)
        if command:
            turn.shell_commands_seen += 1
        if command_is_mutating:
            turn.shell_mutation_attempted = True
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
                turn.metadata_only_rejections += 1
            if should_nudge_content_evidence():
                request_content_evidence_guard()
            else:
                repair.contract_retry_pending = True
            if is_content_evidence_guard:
                runtime.content_evidence_guard_failures += 1
            return
        if is_shell_full_execution_error(tool_result.content):
            direct_requested_read_failed = bool(
                turn.requested_user_path
                and command
                and _looks_like_direct_requested_read(command, turn.requested_user_path)
            )
            if tool_output_kind == "file_not_found" or direct_requested_read_failed:
                turn.mark_direct_read_failed(_summarize_shell_error(tool_result.content))
            if is_mutation_verification:
                if (
                    is_repairable_shell_error(tool_result.content)
                    and not repair.mutation_verification_repair_used
                ):
                    repair.mutation_verification_repair_used = True
                    repair.mutation_verification_repair_pending = True
                    runtime.mutation_verification_repairs += 1
                    repair.shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
                    return
                runtime.mutation_verification_failures += 1
                repair.shell_error_final_pending = True
                turn.mark_exhausted()
                return
            if is_mutation_verification_repair:
                runtime.mutation_verification_failures += 1
                repair.shell_error_final_pending = True
                turn.mark_exhausted()
                return
            if is_mutation_semantic_repair:
                if is_repairable_shell_error(tool_result.content) and repair.shell_repair_retries < MAX_SHELL_REPAIR_RETRIES:
                    repair.shell_repair_retries += 1
                    repair.shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
                    return
                runtime.mutation_semantic_repair_failures += 1
                repair.shell_error_final_pending = True
                turn.mark_exhausted()
                return
            requested_path_exists = _requested_user_path_exists(turn.requested_user_path, workdir=workdir)
            requested_pdf_path = bool(turn.requested_user_path and turn.requested_user_path.lower().endswith(".pdf"))
            if (
                is_repairable_shell_error(tool_result.content)
                and direct_requested_read_failed
                and requested_pdf_path
                and requested_path_exists
                and turn.can_reconsider(RECONSIDER_FILE_RECOVERY)
            ):
                request_file_recovery_guard(
                    last_command=command,
                    last_failure_content=tool_result.content,
                    requested_path_exists=True,
                )
                return
            if is_repairable_shell_error(tool_result.content) and repair.shell_repair_retries < MAX_SHELL_REPAIR_RETRIES:
                repair.shell_repair_retries += 1
                repair.shell_repair_prompt_pending = shell_repair_prompt(tool_result.content)
                return
            repair.shell_error_final_pending = True
            turn.mark_exhausted()
            return
        if command_is_mutating:
            turn.shell_mutation_succeeded = True
        if tool_result.content.strip():
            if command_is_content_evidence and not command_is_mutating:
                turn.mark_direct_content_read()
                if is_content_evidence_guard:
                    runtime.content_evidence_guard_successes += 1
            if tool_output_kind == "discovery_result":
                discovered = _extract_candidate_paths_from_output(tool_result.content)
                if discovered:
                    turn.mark_candidate_paths_found(_merge_candidate_paths(turn.candidate_paths, discovered))
            if is_mutation_verification and not repair.mutation_semantic_repair_used:
                request_mutation_semantic_repair()
            if turn.content_evidence_satisfied or repair.shell_error_final_pending:
                turn.mark_finalizable()
            return
        if is_content_evidence_guard:
            runtime.content_evidence_guard_failures += 1
        if is_mutation_verification_repair and command_is_mutating and not repair.mutation_semantic_repair_used:
            request_mutation_semantic_repair()
            return
        if is_mutation_verification or is_mutation_verification_repair:
            runtime.mutation_verification_failures += 1
            repair.shell_error_final_pending = True
            turn.mark_exhausted()
            return
        if is_mutation_semantic_repair:
            runtime.mutation_semantic_repair_failures += 1
            repair.shell_error_final_pending = True
            turn.mark_exhausted()
            return
        if (
            command
            and not repair.shell_empty_result_check_used
            and should_verify_shell_mutation(command, user_prompt=_last_user_text(runtime.messages))
        ):
            repair.shell_empty_result_check_pending = True
            repair.shell_empty_result_check_used = True
            repair.mutation_verification_pending = True
            runtime.mutation_verifications += 1
    if initial_tool_calls:
        calls = [initial_tool_calls] if isinstance(initial_tool_calls, dict) else list(initial_tool_calls)
        if any(
            _should_guard_existing_file_rewrite(
                tool_call,
                workdir=workdir,
                should_nudge_minimal_patch=should_nudge_minimal_patch,
            )
            for tool_call in calls
        ):
            request_minimal_patch_guard()
        else:
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
                if should_handoff_to_final_from_tool():
                    return runtime._answer_from_tool_results(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        on_final_delta=on_final_delta,
                        on_progress=on_progress,
                        on_model_step=on_model_step,
                        loop=state.tool_rounds + 1,
                        use_tool_prompt=state.used_tool_call_prompt,
                    )
        if (
            not repair.shell_error_final_pending
            and not repair.contract_retry_pending
            and repair.shell_repair_prompt_pending is None
            and not has_pending_internal_request()
            and should_nudge_completion()
        ):
            request_completion_guard()
        if repair.shell_error_final_pending or (
            not has_pending_internal_request()
        ):
            if not turn.pending_completion_guard:
                return runtime._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                    on_model_step=on_model_step,
                    loop=state.tool_rounds + 1,
                    use_tool_prompt=state.used_tool_call_prompt,
                )
        if state.round_limit_reached() and not has_pending_internal_request():
            return runtime._answer_from_tool_results(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
                on_model_step=on_model_step,
                loop=state.tool_rounds + 1,
                use_tool_prompt=state.used_tool_call_prompt,
            )
    for loop_index in range(1, max_loops + 1):
        call_messages = with_tool_call_system_prompt(runtime.messages)
        executing_mutation_verification = repair.mutation_verification_pending
        executing_mutation_verification_repair = repair.mutation_verification_repair_pending
        executing_mutation_semantic_repair = repair.mutation_semantic_repair_pending
        executing_content_evidence_guard = turn.pending_content_evidence_guard
        executing_completion_guard = turn.pending_completion_guard
        executing_minimal_patch_guard = turn.pending_minimal_patch_guard
        if repair.shell_repair_prompt_pending is not None:
            call_messages = [*call_messages, {"role": "user", "content": repair.shell_repair_prompt_pending}]
        elif repair.shell_empty_result_check_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT}]
        elif repair.mutation_semantic_repair_pending:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_SEMANTIC_REPAIR_PROMPT}]
        elif turn.pending_content_evidence_guard:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT}]
        elif turn.pending_analysis_completion_guard:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_ANALYSIS_COMPLETION_GUARD_PROMPT}]
        elif turn.pending_file_recovery_guard:
            call_messages = [*call_messages, {"role": "user", "content": turn.pending_file_recovery_guard_prompt or ""}]
        elif turn.pending_minimal_patch_guard:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_MINIMAL_PATCH_GUARD_PROMPT}]
        elif turn.pending_completion_guard:
            call_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_COMPLETION_GUARD_PROMPT}]
        state.used_tool_call_prompt = True
        tool_max_tokens = _tool_call_max_tokens(
            max_tokens,
            mutative=(
                repair.shell_repair_prompt_pending is not None
                or repair.shell_empty_result_check_pending
                or repair.mutation_semantic_repair_pending
                or turn.pending_content_evidence_guard
                or turn.pending_analysis_completion_guard
                or turn.pending_minimal_patch_guard
                or turn.pending_completion_guard
                or repair.mutation_verification_pending
                or repair.mutation_verification_repair_pending
                or (shell_full_enabled and is_mutative_user_request(user_prompt))
            ),
            file_recovery=turn.pending_file_recovery_guard,
        )
        tool_delta_callback = suppress_tool_delta if shell_full_enabled and (state.tool_rounds > 0 or repair.contract_retry_pending) else on_final_delta
        result = runtime._chat_tool_call_once(
            call_messages,
            temperature=temperature,
            max_tokens=tool_max_tokens,
            tools=tools,
            on_final_delta=tool_delta_callback,
            on_progress=on_progress,
        )
        last_result = result
        if on_model_step:
            on_model_step(ModelStepMetrics.from_result(loop=loop_index, result=result, phase="tool_call" if result.tool_calls else None))
        if repair.contract_retry_pending and not result.tool_calls:
            retry_messages = [*call_messages, {"role": "user", "content": SHELL_FULL_CONTRACT_RETRY_PROMPT}]
            result = runtime._chat_tool_call_once(
                retry_messages,
                temperature=temperature,
                max_tokens=tool_max_tokens,
                tools=tools,
                on_final_delta=suppress_tool_delta,
                on_progress=on_progress,
            )
            last_result = result
            repair.contract_retry_pending = False
            if on_model_step:
                on_model_step(
                    ModelStepMetrics.from_result(
                        loop=loop_index + 1,
                        result=result,
                        phase="tool_call_retry" if result.tool_calls else None,
                    )
                )
            elif result.tool_calls:
                repair.contract_retry_pending = False
        turn.clear_pending_after_model_call()
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
                on_progress=on_progress,
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
                    on_progress=on_progress,
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
        if result.tool_calls and any(
            _should_guard_existing_file_rewrite(
                tool_call,
                workdir=workdir,
                should_nudge_minimal_patch=should_nudge_minimal_patch,
            )
            for tool_call in result.tool_calls
        ):
            request_minimal_patch_guard()
            continue
        if result.tool_calls and should_nudge_analysis_completion(result.tool_calls):
            request_analysis_completion_guard()
            continue
        if result.tool_calls and should_nudge_file_recovery(result.tool_calls):
            request_file_recovery_guard()
            continue
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
                turn.mark_finalizable()
                return runtime._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                    on_model_step=on_model_step,
                    loop=loop_index + 1,
                    use_tool_prompt=state.used_tool_call_prompt,
                )
            return result
        state.increment_round()
        turn.increment_round()
        for tool_call in result.tool_calls:
            if state.has_seen_tool_call(tool_call):
                turn.mark_finalizable()
                return runtime._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
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
            if should_handoff_to_final_from_tool():
                return runtime._answer_from_tool_results(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_final_delta=on_final_delta,
                    on_progress=on_progress,
                    on_model_step=on_model_step,
                    loop=loop_index + 1,
                    use_tool_prompt=state.used_tool_call_prompt,
                )
        if repair.shell_error_final_pending:
            turn.mark_finalizable()
            return runtime._answer_from_tool_results(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
                on_model_step=on_model_step,
                loop=loop_index + 1,
                use_tool_prompt=state.used_tool_call_prompt,
            )
        if state.round_limit_reached() and not has_pending_internal_request():
            if should_nudge_completion():
                request_completion_guard()
                continue
            turn.mark_exhausted()
            return runtime._answer_from_tool_results(
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=on_final_delta,
                on_progress=on_progress,
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


def _tool_call_max_tokens(max_tokens: int, *, mutative: bool, file_recovery: bool = False) -> int:
    if file_recovery:
        return _bounded_internal_max_tokens(max_tokens, FILE_RECOVERY_TOOL_CALL_MAX_TOKENS)
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


def _should_guard_existing_file_rewrite(
    tool_call: dict[str, object],
    *,
    workdir,
    should_nudge_minimal_patch,
) -> bool:
    command = _shell_command_from_tool_call(tool_call)
    raw_arguments = _shell_raw_arguments_from_tool_call(tool_call)
    broad_rewrite_seen = _looks_like_preexecution_broad_rewrite(command) or _looks_like_preexecution_broad_rewrite(raw_arguments)
    target_source = command or raw_arguments
    if not broad_rewrite_seen or not target_source:
        return False
    return should_nudge_minimal_patch(
        broad_rewrite_seen=True,
        existing_file_rewrite=_targets_existing_file(target_source, workdir=workdir),
    )


def _looks_like_preexecution_broad_rewrite(text: str | None) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"\bcat\s+<<\s*['\"]?\w+['\"]?\s*>\s*[^\s]+", text)
        or re.search(r"\bcat\s*>\s*[^\s]+\s*<<\s*['\"]?\w+['\"]?", text)
        or re.search(r"\btee\b.*\s>\s*", text)
        or re.search(r"\btee\s+(?:-a\s+)?['\"]?[^'\"\s;|&]+", text)
        or re.search(r"\bdd\b.*\bof=", text)
    )


def _targets_existing_file(command: str, *, workdir) -> bool:
    for candidate in _rewrite_target_candidates(command):
        try:
            path = Path(workdir, candidate).expanduser().resolve()
            root = Path(workdir).expanduser().resolve()
            path.relative_to(root)
        except (OSError, ValueError):
            continue
        if path.is_file():
            return True
    return False


def _rewrite_target_candidates(command: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"(?:^|[\s;|&])>{1,2}\s*(['\"]?)([^'\"\s;|&]+)\1", command):
        candidates.append(match.group(2))
    for match in re.finditer(r"\btee\s+(?:-a\s+)?(['\"]?)([^'\"\s;|&]+)\1", command):
        candidates.append(match.group(2))
    for match in re.finditer(r"\bof=(['\"]?)([^'\"\s;|&]+)\1", command):
        candidates.append(match.group(2))
    for match in re.finditer(r"\bPath\(\s*['\"]([^'\"]+)['\"]\s*\).*?\bwrite_(?:text|bytes)\s*\(", command):
        candidates.append(match.group(1))
    return candidates


_USER_PATH_RE = re.compile(
    r"(?:[\"'`])([^\"'`\n]+?\.[A-Za-z0-9]{1,8})(?:[\"'`])|(?:^|[\s(])([A-Za-z0-9_./-]+?\.[A-Za-z0-9]{1,8})(?=$|[\s),:;])"
)
_FILENAME_ONLY_RE = re.compile(r"^[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}$")
_DISCOVERY_COMMAND_RE = re.compile(r"^\s*(?:find|ls|tree|fd|locate|rg\s+--files|rg\b(?!.*(?:cat|sed|head|tail)))", re.IGNORECASE)
_FILE_NOT_FOUND_RE = re.compile(
    r"(?:No such file or directory|cannot open|can't open|not found|missing file|impossibile.*trovare|non trovat\w*)",
    re.IGNORECASE,
)


def _extract_requested_user_path(prompt: str | None) -> str | None:
    if not prompt:
        return None
    for match in _USER_PATH_RE.finditer(prompt):
        candidate = match.group(1) or match.group(2)
        if candidate:
            return candidate.strip()
    return None


def _requested_user_path_exists(requested_path: str | None, *, workdir) -> bool:
    if not requested_path:
        return False
    root = Path(workdir).expanduser().resolve()
    target = (root / requested_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return target.is_file()


def _looks_like_discovery_command(command: str) -> bool:
    return bool(_DISCOVERY_COMMAND_RE.search(command))


def _is_targeted_file_discovery(command: str, *, requested_path: str) -> bool:
    if not _looks_like_discovery_command(command):
        return False
    basename = Path(requested_path).name
    return requested_path in command or basename in command


def _classify_shell_output_kind(command: str | None, content: str, *, requested_path: str | None) -> str:
    stripped = content.strip()
    if not stripped:
        return "empty"
    if is_shell_full_contract_error(stripped):
        return "metadata_listing"
    if stripped.startswith("shell_command_failed: true"):
        if requested_path and command and _looks_like_direct_requested_read(command, requested_path) and _FILE_NOT_FOUND_RE.search(stripped):
            return "file_not_found"
        return "error"
    if command and _looks_like_discovery_command(command):
        return "discovery_result"
    if requested_path and command and _looks_like_direct_requested_read(command, requested_path):
        return "direct_content"
    if command and is_content_evidence_shell_command(command):
        return "direct_content"
    return "other"


def _looks_like_direct_requested_read(command: str, requested_path: str) -> bool:
    basename = Path(requested_path).name
    return requested_path in command or basename in command


def _extract_candidate_paths_from_output(content: str) -> list[str]:
    candidates: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("shell_"):
            continue
        if "/" not in stripped and not _FILENAME_ONLY_RE.match(stripped):
            continue
        if "." not in stripped:
            continue
        candidates.append(stripped)
    return candidates


def _merge_candidate_paths(existing: list[str], discovered: list[str]) -> list[str]:
    merged = list(existing)
    for path in discovered:
        if path not in merged:
            merged.append(path)
    return merged


def _is_direct_content_read_for_known_path(command: str, *, requested_path: str, candidate_paths: list[str]) -> bool:
    if not is_content_evidence_shell_command(command):
        return False
    if _looks_like_direct_requested_read(command, requested_path):
        return True
    return any(path in command or Path(path).name in command for path in candidate_paths)


def _summarize_shell_error(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "unknown shell error"
    for line in lines:
        if line.startswith("stderr:"):
            return line.removeprefix("stderr:").strip() or lines[-1]
    return lines[-1]
