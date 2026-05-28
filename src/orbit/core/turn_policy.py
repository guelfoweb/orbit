from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re
from .intent_router import INTENT_CODEBASE_INSPECTION, INTENT_FILE_EDIT
from .loop_guard import (
    ToolCallRecord,
    repeated_tool_count,
    repeated_read_path_record,
    repeated_write_path_record,
    repeated_tool_records,
    sampled_read_paths,
    register_tool_calls as register_loop_tool_calls,
)
from .turn_policy_helpers import completed_edit_paths, edited_paths, file_edit_completion_message, read_target_paths, target_edit_paths
from .turn_policy_prompts import repeated_tool_retry_prompt


EMPTY_REPLY_RETRY_SYSTEM_PROMPT = (
    "Your previous reply was empty. "
    "Reply now with either a short plain-text answer or one valid tool call. "
    "Do not return an empty response."
)

_CODEBASE_SAMPLE_COUNT_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")
_CODEBASE_SAMPLE_COUNT_WORDS = {
    "one": 1,
    "uno": 1,
    "una": 1,
    "two": 2,
    "due": 2,
    "three": 3,
    "tre": 3,
    "four": 4,
    "quattro": 4,
    "five": 5,
    "cinque": 5,
    "six": 6,
    "sei": 6,
    "seven": 7,
    "sette": 7,
    "eight": 8,
    "otto": 8,
    "nine": 9,
    "nove": 9,
    "ten": 10,
    "dieci": 10,
}
_DEFAULT_CODEBASE_SAMPLE_COUNT = 3


@dataclass
class TurnPolicyState:
    loop_count: int = 0
    tool_steps: int = 0
    empty_reply_retries: int = 0
    repeated_tool_retries: int = 0
    synthesis_retries: int = 0
    tool_history: list[ToolCallRecord] = field(default_factory=list)


@dataclass(frozen=True)
class TurnPolicyDecision:
    action: str
    content: str | None = None


def classify_model_reply(
    *,
    content: str,
    tool_calls: list[dict[str, Any]],
    state: TurnPolicyState,
    intent: str | None = None,
    user_input: str | None = None,
) -> TurnPolicyDecision:
    if tool_calls:
        synthesis = detect_synthesis_cutoff(tool_calls=tool_calls, state=state, intent=intent, user_input=user_input)
        if synthesis is not None:
            return synthesis
        repeated = detect_repeated_tool_loop(tool_calls=tool_calls, state=state)
        if repeated is not None:
            return repeated
        return TurnPolicyDecision(action="tool_phase")

    if content.strip():
        return TurnPolicyDecision(action="final_text", content=content.strip())

    if state.empty_reply_retries < 1:
        return TurnPolicyDecision(action="retry_empty_reply", content=EMPTY_REPLY_RETRY_SYSTEM_PROMPT)

    return TurnPolicyDecision(
        action="abort_empty_reply",
        content=(
            "Model returned an empty reply twice in the same turn. "
            "Try a smaller prompt, reduce context, or reset the session."
        ),
    )


def detect_synthesis_cutoff(
    *,
    tool_calls: list[dict[str, Any]],
    state: TurnPolicyState,
    intent: str | None,
    user_input: str | None = None,
) -> TurnPolicyDecision | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        if intent != INTENT_FILE_EDIT:
            return None
        return _detect_file_edit_synthesis_cutoff(tool_calls=tool_calls, state=state)
    if not tool_calls:
        return None
    if any((call.get("function", {}) or {}).get("name") != "read_file" for call in tool_calls):
        return None
    sampled_paths = sampled_read_paths(state.tool_history)
    min_unique_paths = _requested_codebase_sample_count(user_input) or _DEFAULT_CODEBASE_SAMPLE_COUNT
    if len(sampled_paths) < min_unique_paths:
        return None
    if state.synthesis_retries < 1:
        return TurnPolicyDecision(
            action="retry_repeated_tool",
            content=(
                "You already sampled enough implementation files for a first architectural answer. "
                f"Sampled files: {', '.join(sampled_paths[:8])}. "
                f"Stop reading more files and answer now with the {min_unique_paths} most important files to inspect first and the main architectural weaknesses you can already infer."
            ),
        )
    return TurnPolicyDecision(
        action="abort_repeated_tool_loop",
        content=(
            "Stopped because the model kept reading more files after a sufficient architectural sample. "
            "Try /reset or ask a narrower follow-up on one specific file."
        ),
    )


def _detect_file_edit_synthesis_cutoff(
    *,
    tool_calls: list[dict[str, Any]],
    state: TurnPolicyState,
) -> TurnPolicyDecision | None:
    if not tool_calls:
        return None
    tool_names = {(call.get("function", {}) or {}).get("name") for call in tool_calls}
    if not tool_names.issubset({"write_file", "append_file", "replace_in_file", "make_directory", "delete_path", "read_file"}):
        return None
    completed_paths = _completed_edit_paths(state.tool_history)
    read_paths = read_target_paths(tool_calls)
    if read_paths:
        touched_paths = edited_paths(state.tool_history)
        if touched_paths and read_paths.issubset(touched_paths):
            return TurnPolicyDecision(
                action="final_text",
                content=_file_edit_completion_message(touched_paths),
            )
    if not completed_paths:
        return None
    if read_paths and read_paths.issubset(completed_paths):
        return TurnPolicyDecision(
            action="final_text",
            content=_file_edit_completion_message(completed_paths),
        )
    target_paths = _target_edit_paths(tool_calls)
    if not target_paths or not target_paths.issubset(completed_paths):
        return None
    if state.synthesis_retries < 1:
        return TurnPolicyDecision(
            action="retry_repeated_tool",
            content=(
                "You already created or updated the target file. "
                f"Edited files: {', '.join(sorted(completed_paths))}. "
                "Stop editing now and answer with a short confirmation of what was written and any next steps."
            ),
        )
    return TurnPolicyDecision(
        action="final_text",
        content=_file_edit_completion_message(completed_paths),
    )


def detect_repeated_tool_loop(
    *,
    tool_calls: list[dict[str, Any]],
    state: TurnPolicyState,
) -> TurnPolicyDecision | None:
    repeated_records = repeated_tool_records(tool_calls=tool_calls, history=state.tool_history)
    if not repeated_records:
        repeated_read_path = repeated_read_path_record(tool_calls=tool_calls, history=state.tool_history)
        if repeated_read_path is not None:
            repeated_records = [repeated_read_path]
    if not repeated_records:
        repeated_write_path = repeated_write_path_record(tool_calls=tool_calls, history=state.tool_history)
        if repeated_write_path is not None:
            repeated_records = [repeated_write_path]
    if not repeated_records:
        return None
    names = ", ".join(sorted({record.name for record in repeated_records}))
    detail = repeated_records[0].detail
    repeated_count = repeated_tool_count(repeated_records[0], state.tool_history)
    if state.repeated_tool_retries < 1:
        return TurnPolicyDecision(
            action="retry_repeated_tool",
            content=_repeated_tool_retry_prompt(repeated_records[0], state),
        )
    detail_suffix = f" | repeated: {detail}" if detail else ""
    return TurnPolicyDecision(
        action="abort_repeated_tool_loop",
        content=(
            f"Stopped because the tool loop started repeating the same call pattern ({names}). "
            f"Detected {repeated_count} occurrences of the same call{detail_suffix}. "
            "Try /compact, /reset, or a smaller bounded prompt."
        ),
    )


def register_tool_calls(state: TurnPolicyState, tool_calls: list[dict[str, Any]]) -> None:
    register_loop_tool_calls(state.tool_history, tool_calls)


def format_max_loops_message(max_loops: int, state: TurnPolicyState) -> str:
    return (
        f"Stopped after reaching max_loops={max_loops} "
        f"(loops={state.loop_count}, tool_steps={state.tool_steps})."
    )


def _repeated_tool_retry_prompt(record: ToolCallRecord, state: TurnPolicyState) -> str:
    return repeated_tool_retry_prompt(record, state)

def _file_edit_completion_message(paths: set[str]) -> str:
    return file_edit_completion_message(paths)


def _completed_edit_paths(history: list[ToolCallRecord]) -> set[str]:
    return completed_edit_paths(history)


def _target_edit_paths(tool_calls: list[dict[str, Any]]) -> set[str]:
    return target_edit_paths(tool_calls)


def _requested_codebase_sample_count(user_input: str | None) -> int | None:
    if not user_input:
        return None
    tokens = _CODEBASE_SAMPLE_COUNT_RE.findall(user_input.lower())
    if not tokens:
        return None
    for index, token in enumerate(tokens):
        if token not in {"file", "files", "path", "paths"}:
            continue
        for prior in range(max(0, index - 4), index):
            if tokens[prior].isdigit():
                return max(1, min(10, int(tokens[prior])))
            count = _CODEBASE_SAMPLE_COUNT_WORDS.get(tokens[prior])
            if count is not None:
                return count
    return None
