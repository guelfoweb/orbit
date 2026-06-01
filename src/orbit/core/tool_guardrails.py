from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any

from .guardrail_documents import (
    SHOW_CONTENT_HINTS,
    SUMMARY_HINTS,
    TEXT_PATH_RE,
    extract_explicit_pdf_path as _extract_explicit_pdf_path,
    extract_explicit_text_path as _extract_explicit_text_path,
    has_pdf_text_extract_in_current_turn as _has_pdf_text_extract_in_current_turn,
    is_strings_extract_result as _is_strings_extract_result,
    latest_pdf_strings_result as _latest_pdf_strings_result,
    latest_pdf_text_extract_result as _latest_pdf_text_extract_result,
    local_explicit_pdf_result as _local_explicit_pdf_result,
    local_explicit_text_result as _local_explicit_text_result,
    condense_explicit_text_summary_messages as _condense_explicit_text_summary_messages,
    should_defer_explicit_text_summary_to_model as _should_defer_explicit_text_summary_to_model,
    result_has_useful_text as _result_has_useful_text,
    seed_document_summary_reads as _seed_document_summary_reads,
    seed_explicit_pdf_read_impl,
    seed_explicit_text_read_impl,
    summarize_text_content as _summarize_text_content,
)
from .guardrail_patterns import (
    local_markdown_checkbox_extraction_result as _local_markdown_checkbox_extraction_result,
    markdown_checkbox_redundant_read_prompt as _markdown_checkbox_redundant_read_prompt,
    seed_markdown_checkbox_extraction as _seed_markdown_checkbox_extraction,
)
from .binary_guardrails import (
    ARCHIVE_CONTAINER_EXTENSIONS,
    binary_analysis_guard_prompt as _binary_analysis_guard_prompt,
    binary_listing_guidance as _binary_listing_guidance_impl,
    binary_listing_retry_prompt as _binary_listing_retry_prompt,
    binary_seeded_summary_for_candidate as _binary_seeded_summary_for_candidate,
    binary_text_reply_handling as _binary_text_reply_handling,
    binary_tool_strategy_prompt as _binary_tool_strategy_prompt,
)
from .guardrail_chat import (
    assistant_identity_system_prompt as _assistant_identity_system_prompt,
    local_assistant_identity_result as _local_assistant_identity_result,
    local_directory_listing_result as _local_directory_listing_result,
    local_pure_chitchat_result as _local_pure_chitchat_result,
    needs_directory_discovery as _needs_directory_discovery,
)
from .guardrail_file_edit import (
    apply_deterministic_file_edit as _apply_deterministic_file_edit,
    file_edit_placeholder_handling as _file_edit_placeholder_handling,
    file_edit_post_write_reply_handling as _file_edit_post_write_reply_handling,
    infer_file_edit_section_append as _infer_file_edit_section_append,
    placeholder_write_replacement_text as _placeholder_write_replacement_text,
)
from .guardrail_factual import (
    PROJECT_METADATA_CANDIDATES,
    VERSION_QUERY_HINTS,
    apply_deterministic_bounded_command as _apply_deterministic_bounded_command,
    local_codebase_metadata_result as _local_codebase_metadata_result,
    local_current_factual_result as _local_current_factual_result,
    local_tooling_concept_result as _local_tooling_concept_result,
    seed_current_factual_tool as _seed_current_factual_tool,
)
from .guardrail_review import (
    codebase_redundant_listing_prompt as _codebase_redundant_listing_prompt,
    codebase_review_reply_handling as _codebase_review_reply_handling,
    local_codebase_architecture_result as _local_codebase_architecture_result,
    local_codebase_hotspot_result as _local_codebase_hotspot_result,
    local_codebase_priority_files_result as _local_codebase_priority_files_result,
    local_codebase_review_result as _local_codebase_review_result,
    seed_codebase_listing_impl,
    seed_codebase_review_reads_impl,
)
from .static_analysis_guardrails import (
    local_static_reverse_inspection_result,
    local_static_sample_evidence_result,
    seed_binary_discovery_impl,
)
from .events import ToolCallEvent, ToolResultEvent, ToolRouteEvent
from .intent_router import INTENT_CODEBASE_INSPECTION, INTENT_TEXT_DOCUMENT_ANALYSIS, is_binary_or_pdf_analysis_intent
from .message_ops import (
    has_recent_tool_result,
    last_fetch_url_result,
    last_read_file_result,
    latest_search_web_result_in_current_turn,
    latest_fetch_url_result_in_current_turn,
    latest_successful_read_result_in_current_turn,
    listed_entry_type,
    likely_binary_candidates_from_recent_listing,
    recent_archive_container_for_member,
    recent_listed_paths,
    recent_listed_directory_paths,
    recent_listed_file_paths,
    normalize_relative_path,
    successful_bash_results_in_current_turn,
    successful_read_results_in_current_turn,
    successful_write_results_in_current_turn,
    was_listed_by_list_files,
)
from .turn_policy import TurnPolicyState
from .turn_policy_helpers import file_edit_completion_message
from .text_utils import prefers_english_output


LOCAL_ACCESS_REFUSAL_HINTS = (
    "do not have access",
    "don't have access",
    "cannot access",
    "can't access",
    "no access to file",
    "no access to directory",
    "provide the path",
    "fornisci il percorso",
    "non ho accesso",
    "non dispongo di accesso",
)
FAKE_TOOL_RESPONSE_HINTS = (
    "<tool_response>",
    "</tool_response>",
    '"ok": true',
    '"ok": false',
)
CODE_REVIEW_READ_MAX_FILES = 3


def _run_guardrail_tool(
    *,
    name: str,
    arguments: dict[str, Any],
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
    emit_route: bool = False,
) -> dict[str, Any]:
    if emit_route and on_event is not None:
        on_event(ToolRouteEvent(loop=0, intent=route.intent, categories=route.categories, reason=route.reason))
    if on_event is not None:
        on_event(ToolCallEvent(loop=0, name=name, arguments=arguments))
    started_at = time.monotonic_ns()
    result = registry.call(name, arguments)
    elapsed_ns = time.monotonic_ns() - started_at
    if elapsed_ns > 0:
        metrics.tool_duration_ns += elapsed_ns
    policy_state.tool_steps += 1
    if on_event is not None:
        on_event(
            ToolResultEvent(
                loop=0,
                name=name,
                ok=bool(result.get("ok")),
                error=result.get("error"),
                returncode=result.get("returncode"),
                stderr=result.get("stderr"),
                stdout=result.get("stdout"),
                elapsed_ms=elapsed_ns / 1_000_000,
            )
        )
    messages.append(
        {
            "role": "tool",
            "tool_name": name,
            "content": registry.encode_tool_result(result),
        }
    )
    return result


def _seed_guardrail_tool(
    *,
    name: str,
    arguments: dict[str, Any],
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
    emit_route: bool = False,
) -> dict[str, Any]:
    return _run_guardrail_tool(
        name=name,
        arguments=arguments,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=emit_route,
    )


def local_text_document_result(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    if intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return None
    if len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    if function.get("name") != "read_file":
        return None
    arguments = parse_arguments(function.get("arguments"))
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    result = last_read_file_result(messages, path.strip())
    if result is None:
        return None
    content = result.get("content")
    if not isinstance(content, str) or not content:
        return None
    if result.get("truncated") or result.get("has_more"):
        return content + "\n\n[truncated: ask for a specific range or continue reading this file if needed]"
    return content


def seed_binary_discovery(
    *,
    user_input: str = "",
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    seed_binary_discovery_impl(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_seed_guardrail_tool,
        binary_listing_guidance=_binary_listing_guidance,
        has_pdf_text_extract_in_current_turn=_has_pdf_text_extract_in_current_turn,
    )


def seed_directory_discovery(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    if route.intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return
    if not _needs_directory_discovery(user_input):
        return
    if has_recent_tool_result(messages, "list_files"):
        return
    _seed_guardrail_tool(
        name="list_files",
        arguments={"path": ".", "recursive": False, "max_entries": 12},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )
    messages.append(
        {
            "role": "system",
            "content": (
                "A directory listing for the current workdir has already been collected. "
                "Use that listing to answer what the directory contains or to choose one concrete path for read_file. "
                "Do not claim that you lack local file access."
            ),
        }
    )


def seed_filesystem_metadata(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    if route.intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return
    lowered = user_input.lower()
    if not _asks_for_filesystem_metadata(lowered):
        return
    if has_recent_tool_result(messages, "stat_path"):
        return
    path = _extract_explicit_text_path(user_input)
    recursive = False
    if path is None:
        workspace_hints = ("workspace", "workdir", "directory", "folder", "cartella", "progetto", "project")
        if not any(hint in lowered for hint in workspace_hints):
            return
        path = "."
        recursive = any(hint in lowered for hint in ("newest", "latest", "oldest", "recente", "recenti", "nuovo"))
    _seed_guardrail_tool(
        name="stat_path",
        arguments={"path": path, "recursive": recursive},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )


def seed_workspace_file_classification(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    if route.intent not in {INTENT_TEXT_DOCUMENT_ANALYSIS, INTENT_CODEBASE_INSPECTION}:
        return
    if not _asks_for_workspace_file_classification(user_input.lower()):
        return
    if has_recent_tool_result(messages, "list_files"):
        return
    _seed_guardrail_tool(
        name="list_files",
        arguments={"path": ".", "recursive": False, "max_entries": 80},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )


def seed_workspace_file_presence_check(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    if route.intent not in {INTENT_TEXT_DOCUMENT_ANALYSIS, INTENT_CODEBASE_INSPECTION}:
        return
    if not _asks_for_workspace_file_presence(user_input.lower()):
        return
    if has_recent_tool_result(messages, "list_files"):
        return
    _seed_guardrail_tool(
        name="list_files",
        arguments={"path": ".", "recursive": False, "max_entries": 80},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )


def local_workspace_file_presence_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent not in {INTENT_TEXT_DOCUMENT_ANALYSIS, INTENT_CODEBASE_INSPECTION}:
        return None
    lowered = user_input.lower()
    if not _asks_for_workspace_file_presence(lowered):
        return None
    paths = recent_listed_file_paths(messages, limit=80)
    if not paths:
        return None
    if "json" in lowered and "config" in lowered:
        matches = [
            path
            for path in paths
            if path.lower().endswith(".json") and ("config" in path.lower().rsplit("/", 1)[-1])
        ]
        if matches:
            return f"JSON configuration file found: {', '.join(matches)}."
        return "No JSON configuration file exists in the current workspace."
    return None


def seed_workspace_security_scan(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    if route.intent != INTENT_CODEBASE_INSPECTION:
        return
    if not _asks_for_workspace_security_scan(user_input.lower()):
        return
    if not has_recent_tool_result(messages, "list_files"):
        _seed_guardrail_tool(
            name="list_files",
            arguments={"path": ".", "recursive": False, "max_entries": 80},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )
    seed_codebase_review_reads_impl(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_run_guardrail_tool,
        max_chunks_per_file=1,
    )


def local_workspace_security_scan_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    if not _asks_for_workspace_security_scan(user_input.lower()):
        return None
    review = _local_codebase_review_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        prefers_english_output=_prefers_english_output,
    )
    if review is not None:
        if "should be inspected before claiming" in review.lower() or "va ispezionato prima" in review.lower():
            inspected = successful_read_results_in_current_turn(messages)
            paths = [str(item.get("path")) for item in inspected if isinstance(item.get("path"), str)]
            path_list = ", ".join(paths[:3]) if paths else "the selected files"
            return f"I cannot prove a concrete security issue from the inspected workspace files. Inspected code evidence: {path_list}."
        if not _review_text_has_security_evidence(review):
            inspected = successful_read_results_in_current_turn(messages)
            paths = [str(item.get("path")) for item in inspected if isinstance(item.get("path"), str)]
            path_list = ", ".join(paths[:3]) if paths else "the selected files"
            return f"I cannot prove a concrete security issue from the inspected workspace files. Inspected code evidence: {path_list}."
        return review + "\nEvidence category: code."
    inspected = successful_read_results_in_current_turn(messages)
    if inspected:
        paths = [str(item.get("path")) for item in inspected if isinstance(item.get("path"), str)]
        path_list = ", ".join(paths[:3]) if paths else "the selected files"
        return f"I cannot prove a concrete security issue from the inspected workspace files. Inspected code evidence: {path_list}."
    return None


def local_workspace_file_classification_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent not in {INTENT_TEXT_DOCUMENT_ANALYSIS, INTENT_CODEBASE_INSPECTION}:
        return None
    if not _asks_for_workspace_file_classification(user_input.lower()):
        return None
    paths = recent_listed_file_paths(messages, limit=80)
    if not paths:
        return None
    source: list[str] = []
    config: list[str] = []
    documentation: list[str] = []
    for path in paths:
        lowered = path.lower()
        suffix = "." + lowered.rsplit(".", 1)[-1] if "." in lowered.rsplit("/", 1)[-1] else ""
        base = lowered.rsplit("/", 1)[-1]
        if suffix in _SOURCE_FILE_EXTENSIONS:
            source.append(path)
        elif suffix in _CONFIG_FILE_EXTENSIONS or base in _CONFIG_FILE_NAMES:
            config.append(path)
        elif suffix in _DOCUMENTATION_FILE_EXTENSIONS or base in _DOCUMENTATION_FILE_NAMES:
            documentation.append(path)
    lines: list[str] = []
    if source:
        lines.append(f"Source code: {', '.join(source)}")
    if config:
        lines.append(f"Configuration: {', '.join(config)}")
    if documentation:
        lines.append(f"Documentation/text: {', '.join(documentation)}")
    if not lines:
        return "No source code, configuration, or documentation files are evident from the current workspace listing."
    return "\n".join(lines)


def local_filesystem_metadata_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return None
    lowered = user_input.lower()
    if not _asks_for_filesystem_metadata(lowered):
        return None
    result = _latest_tool_json(messages, "stat_path")
    if result is None or result.get("ok") is not True:
        return None
    if result.get("type") == "file":
        return _format_file_metadata_result(result, lowered)
    if result.get("type") == "dir":
        return _format_directory_metadata_result(result, lowered)
    return None


def seed_codebase_listing(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    seed_codebase_listing_impl(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_run_guardrail_tool,
    )


def seed_markdown_checkbox_extraction(
    *,
    skill: Any,
    user_input: str,
    route: ToolRoute,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: TurnMetrics,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    _seed_markdown_checkbox_extraction(
        skill=skill,
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_run_guardrail_tool,
    )


def local_markdown_checkbox_extraction_result(
    *,
    skill: Any,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_markdown_checkbox_extraction_result(skill=skill, user_input=user_input, messages=messages)


def markdown_checkbox_redundant_read_prompt(
    *,
    skill: Any,
    user_input: str,
    name: str,
    arguments: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str | None:
    return _markdown_checkbox_redundant_read_prompt(
        skill=skill,
        user_input=user_input,
        name=name,
        arguments=arguments,
        messages=messages,
    )


def codebase_redundant_listing_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    return _codebase_redundant_listing_prompt(
        intent=intent,
        tool_calls=tool_calls,
        messages=messages,
        parse_arguments=parse_arguments,
    )


def seed_codebase_review_reads(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    seed_codebase_review_reads_impl(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_run_guardrail_tool,
    )


def local_directory_listing_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_directory_listing_result(
        intent=intent,
        text_document_intent=INTENT_TEXT_DOCUMENT_ANALYSIS,
        user_input=user_input,
        messages=messages,
        recent_listed_paths=recent_listed_paths,
        recent_listed_directory_paths=recent_listed_directory_paths,
        prefers_english_output=_prefers_english_output,
    )


def local_codebase_priority_files_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION or not _has_recursive_listing_in_current_turn(messages):
        return None
    return _local_codebase_priority_files_result(intent=intent, user_input=user_input, messages=messages)


def local_codebase_architecture_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_codebase_architecture_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        prefers_english_output=_prefers_english_output,
    )


def local_codebase_hotspot_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_codebase_hotspot_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        prefers_english_output=_prefers_english_output,
    )


def local_codebase_review_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_codebase_review_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        prefers_english_output=_prefers_english_output,
    )


def local_codebase_review_after_missing_read_path(
    *,
    intent: str | None,
    user_input: str,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    if not successful_read_results_in_current_turn(messages):
        return None
    has_missing_read_path = False
    for tool_call in tool_calls:
        function = tool_call.get("function", {}) or {}
        if function.get("name") != "read_file":
            continue
        arguments = parse_arguments(function.get("arguments"))
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            has_missing_read_path = True
            break
    if not has_missing_read_path:
        return None
    review = _local_codebase_review_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        prefers_english_output=_prefers_english_output,
    )
    if review is not None:
        return review
    paths = [str(item.get("path")) for item in successful_read_results_in_current_turn(messages) if isinstance(item.get("path"), str)]
    if not paths:
        return None
    return (
        "No concrete bug found in the inspected portions. "
        f"The model attempted another read_file call without a path after inspecting: {', '.join(paths[:3])}."
    )


def codebase_review_reply_handling(
    *,
    intent: str | None,
    user_input: str,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    return _codebase_review_reply_handling(
        intent=intent,
        user_input=user_input,
        content=content,
        messages=messages,
        policy_state=policy_state,
        prefers_english_output=_prefers_english_output,
    )


def seed_project_metadata_read(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    if route.intent != INTENT_CODEBASE_INSPECTION:
        return
    lowered = user_input.lower()
    if not any(hint in lowered for hint in VERSION_QUERY_HINTS):
        return
    if successful_read_results_in_current_turn(messages):
        return
    for path in PROJECT_METADATA_CANDIDATES:
        result = _seed_guardrail_tool(
            name="read_file",
            arguments={"path": path, "start_line": 1, "max_lines": 80, "max_chars": 4000},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )
        if result.get("ok"):
            return


def seed_explicit_text_read(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    seed_explicit_text_read_impl(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_run_guardrail_tool,
    )


def seed_explicit_pdf_read(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    seed_explicit_pdf_read_impl(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_run_guardrail_tool,
    )


def seed_current_factual_tool(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> None:
    _seed_current_factual_tool(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        has_recent_tool_result=has_recent_tool_result,
        run_guardrail_tool=_run_guardrail_tool,
    )


def apply_deterministic_file_edit(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> str | None:
    return _apply_deterministic_file_edit(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        summary_hints=SUMMARY_HINTS,
        run_guardrail_tool=_run_guardrail_tool,
        last_read_file_result=last_read_file_result,
        summarize_text_content=_summarize_text_content,
        successful_read_results_in_current_turn=successful_read_results_in_current_turn,
        file_edit_completion_message=file_edit_completion_message,
        extract_explicit_text_path=_extract_explicit_text_path,
        extract_explicit_pdf_path=_extract_explicit_pdf_path,
        text_path_re=TEXT_PATH_RE,
        normalize_relative_path=normalize_relative_path,
    )


def apply_deterministic_bounded_command(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: TurnPolicyState,
    on_event: Any,
) -> str | None:
    return _apply_deterministic_bounded_command(
        user_input=user_input,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        run_guardrail_tool=_run_guardrail_tool,
    )


def local_codebase_metadata_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_codebase_metadata_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        codebase_inspection_intent=INTENT_CODEBASE_INSPECTION,
        successful_read_results_in_current_turn=successful_read_results_in_current_turn,
        normalize_relative_path=normalize_relative_path,
    )


def local_tooling_concept_result(user_input: str) -> str | None:
    return _local_tooling_concept_result(user_input)


def local_explicit_text_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_explicit_text_result(intent=intent, user_input=user_input, messages=messages)


def condense_explicit_text_summary_messages(
    *,
    user_input: str,
    messages: list[dict[str, Any]],
    summary_text: str | None = None,
) -> None:
    _condense_explicit_text_summary_messages(user_input=user_input, messages=messages, summary_text=summary_text)


def should_defer_explicit_text_summary_to_model(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> bool:
    return _should_defer_explicit_text_summary_to_model(intent=intent, user_input=user_input, messages=messages)


def local_explicit_pdf_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_explicit_pdf_result(intent=intent, user_input=user_input, messages=messages)


def local_current_factual_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    return _local_current_factual_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        latest_search_web_result_in_current_turn=latest_search_web_result_in_current_turn,
        latest_fetch_url_result_in_current_turn=latest_fetch_url_result_in_current_turn,
    )


def local_mixed_local_web_evidence_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent != "current_factual_lookup":
        return None
    lowered = user_input.lower()
    if not (
        "local evidence" in lowered
        and "web evidence" in lowered
        and any(term in lowered for term in ("versus", "vs", "separate", "difference", "what is"))
    ):
        return None
    read_results = successful_read_results_in_current_turn(messages)
    search = latest_search_web_result_in_current_turn(messages)
    if not read_results or search is None:
        return None
    latest_read = read_results[-1]
    path = latest_read.get("path")
    content = latest_read.get("content")
    if not isinstance(path, str) or not isinstance(content, str) or not content.strip():
        return None
    local_summary = _summarize_text_content(content, single_line=True) or content.strip().splitlines()[0].strip()
    results = search.get("results")
    web_titles: list[str] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                web_titles.append(title.strip())
            if len(web_titles) >= 2:
                break
    web_summary = "; ".join(web_titles) if web_titles else "the bounded web search results for the requested current topic"
    return (
        f"- Local evidence: `{path}` says {local_summary}\n"
        f"- Web evidence: the search results include {web_summary}.\n"
        "- Difference: local evidence comes from files in the current workspace; web evidence comes from external current sources."
    )


def local_assistant_identity_result(user_input: str) -> str | None:
    return _local_assistant_identity_result(user_input, prefers_english_output=_prefers_english_output)


def local_pure_chitchat_result(user_input: str) -> str | None:
    return _local_pure_chitchat_result(user_input)


def assistant_identity_system_prompt(user_input: str) -> str | None:
    return _assistant_identity_system_prompt(user_input, prefers_english_output=_prefers_english_output)


_extract_explicit_text_path = _extract_explicit_text_path
_extract_explicit_pdf_path = _extract_explicit_pdf_path
_seed_document_summary_reads = _seed_document_summary_reads


_summarize_text_content = _summarize_text_content
_latest_pdf_text_extract_result = _latest_pdf_text_extract_result
_latest_pdf_strings_result = _latest_pdf_strings_result
_has_pdf_text_extract_in_current_turn = _has_pdf_text_extract_in_current_turn
_result_has_useful_text = _result_has_useful_text
_is_strings_extract_result = _is_strings_extract_result


def _has_recursive_listing_in_current_turn(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if message.get("role") == "user":
            return False
        if message.get("role") != "tool" or message.get("tool_name") != "list_files":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
        return bool(payload.get("recursive"))
    return False


def _prefers_english_output(text: str) -> bool:
    return prefers_english_output(text)


_SOURCE_FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".sh",
    ".ps1",
}
_CONFIG_FILE_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env"}
_CONFIG_FILE_NAMES = {"dockerfile", "makefile", "modelfile", "requirements.txt", "package.json", "pyproject.toml"}
_DOCUMENTATION_FILE_EXTENSIONS = {".md", ".rst", ".txt"}
_DOCUMENTATION_FILE_NAMES = {"readme", "license", "changelog"}


def _asks_for_workspace_file_classification(lowered: str) -> bool:
    if _asks_for_workspace_file_presence(lowered):
        return False
    classification_hints = (
        "source code",
        "configuration",
        "documentation",
        "documentazione",
        "configurazione",
        "codice sorgente",
    )
    workspace_hints = ("workspace", "workdir", "directory", "folder", "cartella", "progetto", "project")
    action_hints = ("inspect", "identify", "which files", "quali file", "classify", "classifica")
    return (
        any(hint in lowered for hint in classification_hints)
        and any(hint in lowered for hint in workspace_hints)
        and any(hint in lowered for hint in action_hints)
    )


def _asks_for_workspace_file_presence(lowered: str) -> bool:
    metadata_hints = (
        "newest",
        "latest",
        "oldest",
        "modified",
        "modification",
        "mtime",
        "size",
        "how many",
        "count",
        "recente",
        "modificato",
        "dimensione",
        "quanti",
    )
    if any(hint in lowered for hint in metadata_hints):
        return False
    workspace_hints = ("workspace", "workdir", "directory", "folder", "cartella", "project", "progetto")
    presence_hints = ("whether", "if", "any", "exists", "exist", "present", "there is", "there are", "whether there is")
    file_hints = ("file", "files", "configuration", "config", "json", "document", "source")
    return (
        any(hint in lowered for hint in workspace_hints)
        and any(hint in lowered for hint in presence_hints)
        and any(hint in lowered for hint in file_hints)
    )


def _asks_for_workspace_security_scan(lowered: str) -> bool:
    workspace_hints = ("workspace", "workdir", "directory", "folder", "cartella", "repo", "repository", "project", "progetto")
    search_hints = ("search", "find", "look for", "cerca", "trova")
    security_hints = (
        "security issue",
        "security",
        "vulnerability",
        "vulnerabilities",
        "vuln",
        "secret",
        "password",
        "credential",
        "token",
        "insecure",
        "sicurezza",
        "vulnerabilita",
        "vulnerabilità",
        "segreto",
        "credenziali",
    )
    return (
        any(hint in lowered for hint in workspace_hints)
        and any(hint in lowered for hint in search_hints)
        and any(hint in lowered for hint in security_hints)
    )


def _review_text_has_security_evidence(text: str) -> bool:
    lowered = text.lower()
    security_terms = (
        "security",
        "vulnerability",
        "vulnerab",
        "secret",
        "credential",
        "password",
        "token",
        "injection",
        "traversal",
        "arbitrary",
        "unsafe",
        "insecure",
        "shell",
        "command",
        "destructive",
        "sicurezza",
        "credenzial",
        "segreto",
    )
    return any(term in lowered for term in security_terms)



def binary_analysis_guard_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    return _binary_analysis_guard_prompt(
        intent=intent,
        tool_calls=tool_calls,
        messages=messages,
        parse_arguments=parse_arguments,
        normalize_relative_path=normalize_relative_path,
        recent_archive_container_for_member=recent_archive_container_for_member,
        was_listed_by_list_files=was_listed_by_list_files,
    )


def filesystem_read_path_guard_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    if intent not in {"text_document_analysis", "codebase_inspection", "file_edit"} or len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    if function.get("name") != "read_file":
        return None
    if not has_recent_tool_result(messages, "list_files"):
        return None
    arguments = parse_arguments(function.get("arguments"))
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    normalized = normalize_relative_path(path)
    entry_type = listed_entry_type(messages, normalized)
    if entry_type == "file":
        return None
    if entry_type == "dir":
        return (
            f"`{normalized}` is a directory, not a text file. "
            "Use one exact file path returned by list_files. Do not call read_file on directories."
        )
    return (
        f"`{normalized}` was not returned exactly by the recent list_files result. "
        "Reuse one exact relative path from that listing instead of guessing or shortening the path."
    )


def binary_tool_strategy_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    return _binary_tool_strategy_prompt(
        intent=intent,
        tool_calls=tool_calls,
        parse_arguments=parse_arguments,
    )


def storage_command_strategy_prompt(
    *,
    intent_class: str | None,
    user_input: str,
    tool_calls: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    if len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    if function.get("name") != "bash":
        return None
    arguments = parse_arguments(function.get("arguments"))
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens or tokens[0] != "du":
        return None
    target = next((token for token in tokens[1:] if not token.startswith("-")), "")
    lowered = user_input.lower()
    wants_filesystem_space = any(
        hint in lowered
        for hint in (
            "available storage",
            "free space",
            "filesystem",
            "mounted on",
            "spazio disponibile",
            "spazio libero",
            "filesystem",
            "workspace",
        )
    )
    if target == "/" or wants_filesystem_space:
        target_hint = target or "."
        return (
            "This request is about filesystem capacity, not recursive directory size. "
            f"Do not use `du` here. Use `df -h {target_hint}` instead, then answer briefly from that filesystem result."
        )
    return None


def binary_listing_retry_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    return _binary_listing_retry_prompt(
        intent=intent,
        tool_calls=tool_calls,
        messages=messages,
        parse_arguments=parse_arguments,
        normalize_relative_path=normalize_relative_path,
        has_recent_tool_result=has_recent_tool_result,
        likely_binary_candidates_from_recent_listing=likely_binary_candidates_from_recent_listing,
    )


def binary_text_reply_handling(
    *,
    intent: str | None,
    content: str,
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    return _binary_text_reply_handling(
        intent=intent,
        content=content,
        policy_state=policy_state,
    )


def filesystem_text_reply_handling(
    *,
    intent: str | None,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return None
    lowered = content.lower()
    if not any(hint in lowered for hint in LOCAL_ACCESS_REFUSAL_HINTS):
        return None
    if not has_recent_tool_result(messages, "list_files"):
        return None
    if policy_state.synthesis_retries >= 1:
        return (
            "final",
            "A directory listing is already available via the local tools, but the model still answered with a generic local-access refusal. "
            "Retry the request or ask for one specific file or subtree.",
        )
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "You do have access to the current workdir through local tools, and a directory listing is already available in the tool results. "
        "Use that existing list_files result to answer what this directory contains, or call read_file on one exact returned path if needed. "
        "Do not say that you lack local file or directory access.",
    )


def filesystem_metadata_reply_handling(
    *,
    intent: str | None,
    user_input: str,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return None
    lowered_input = user_input.lower()
    if not _asks_for_filesystem_metadata(lowered_input):
        return None
    if has_recent_tool_result(messages, "stat_path"):
        return None
    lowered_content = content.lower()
    missing_metadata_hints = (
        "do not have the modification times",
        "don't have the modification times",
        "cannot determine the newest",
        "can't determine the newest",
        "modification times",
        "modified times",
        "mtime",
        "no metadata",
        "newest file is not explicitly provided",
        "newest file is not provided",
        "not explicitly provided by the `list_files` tool",
        "not explicitly provided by list_files",
        "list_files tool does not provide",
        "non ho i tempi di modifica",
        "non posso determinare il file più recente",
        "non posso determinare il file piu recente",
    )
    if not any(hint in lowered_content for hint in missing_metadata_hints):
        return None
    if policy_state.synthesis_retries >= 1:
        return None
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "The request asks for local filesystem metadata. Call stat_path now on the relevant file or directory. "
        "Use recursive=true for workspace-level newest/oldest questions, then answer every requested fact from the stat_path result.",
    )


def _latest_tool_json(messages: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("tool_name") != tool_name:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _format_file_metadata_result(result: dict[str, Any], lowered_input: str) -> str:
    path = str(result.get("path") or ".")
    parts: list[str] = [f"`{path}`"]
    if any(hint in lowered_input for hint in ("size", "bytes", "dimensione", "dimensioni", "byte")):
        size = result.get("size_bytes")
        if isinstance(size, int):
            parts.append(f"size: {size} bytes")
    if any(hint in lowered_input for hint in ("modified", "modification", "mtime", "modificato", "modifica", "recente")):
        modified = result.get("modified_at")
        if isinstance(modified, str) and modified:
            parts.append(f"modified: {modified}")
    if any(hint in lowered_input for hint in ("permissions", "permission", "mode", "permessi", "permesso")):
        mode = result.get("mode")
        if isinstance(mode, str) and mode:
            parts.append(f"mode: {mode}")
    if any(hint in lowered_input for hint in ("exists", "exist", "esiste")):
        parts.append("exists: yes")
    if len(parts) == 1:
        size = result.get("size_bytes")
        modified = result.get("modified_at")
        if isinstance(size, int):
            parts.append(f"size: {size} bytes")
        if isinstance(modified, str) and modified:
            parts.append(f"modified: {modified}")
    output = "; ".join(parts)
    if _asks_for_metadata_method(lowered_input):
        output += ". Determined with `stat_path` on the requested path."
    return output


def _format_directory_metadata_result(result: dict[str, Any], lowered_input: str) -> str | None:
    entries = result.get("entries")
    file_count = result.get("file_count")
    dir_count = result.get("dir_count")
    count = result.get("total_entries", result.get("count"))
    wants_newest = any(hint in lowered_input for hint in ("newest", "latest", "recente", "recenti", "nuovo"))
    wants_oldest = any(hint in lowered_input for hint in ("oldest", "meno recente"))
    wants_files = any(hint in lowered_input for hint in ("file", "files"))
    if wants_newest or wants_oldest:
        if not isinstance(entries, list) or not entries:
            return "No entries found."
        ordered = [entry for entry in entries if isinstance(entry, dict)]
        if wants_files:
            ordered = [entry for entry in ordered if entry.get("type") == "file"]
        if not ordered:
            return "No entries found."
        chosen = ordered[-1] if wants_oldest else ordered[0]
        path = chosen.get("path")
        modified = chosen.get("modified_at")
        label = "oldest" if wants_oldest else "newest"
        noun = "file" if wants_files else "entry"
        prefix = []
        if isinstance(file_count, int):
            prefix.append(f"There are {file_count} files")
        elif isinstance(count, int):
            prefix.append(f"There are {count} entries")
        if isinstance(path, str) and path:
            suffix = f"The {label} {noun} is `{path}`"
            if isinstance(modified, str) and modified:
                suffix += f" modified at {modified}"
            prefix.append(suffix)
        output = ". ".join(prefix) + "."
        if _asks_for_metadata_method(lowered_input):
            output += " Determined with `stat_path` on the workspace directory, using modification times from bounded filesystem metadata."
        return output
    parts: list[str] = []
    if isinstance(file_count, int):
        parts.append(f"files: {file_count}")
    if isinstance(dir_count, int):
        parts.append(f"directories: {dir_count}")
    if isinstance(count, int):
        parts.append(f"entries: {count}")
    path = result.get("path")
    if isinstance(path, str):
        parts.insert(0, f"`{path}`")
    return "; ".join(parts) if parts else None


def _asks_for_metadata_method(lowered_input: str) -> bool:
    return any(
        hint in lowered_input
        for hint in (
            "how did you determine",
            "how did you know",
            "how was this determined",
            "method",
            "determined it",
            "determine it",
            "come lo hai determinato",
            "come l'hai determinato",
            "come lo sai",
            "metodo",
        )
    )


def generic_tool_access_reply_handling(
    *,
    content: str,
    available_tool_names: set[str],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if not available_tool_names:
        return None
    lowered = content.lower()
    if not any(hint in lowered for hint in LOCAL_ACCESS_REFUSAL_HINTS):
        return None
    if policy_state.synthesis_retries >= 1:
        return None
    preferred: list[str] = []
    for name in ("bash", "read_file", "list_files", "search_web", "fetch_url"):
        if name in available_tool_names:
            preferred.append(name)
    tool_hint = ", ".join(preferred) if preferred else ", ".join(sorted(available_tool_names))
    action_hint = ""
    if "bash" in available_tool_names:
        action_hint = (
            " For local machine or environment questions, call bash now with one small inspection command such as "
            "`nproc`, `uname -srm`, `lscpu`, or `cat /etc/os-release`."
        )
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "You do have access to local runtime tools in this environment. "
        f"If the request needs external evidence or machine inspection, choose the most relevant tool from: {tool_hint}. "
        f"{action_hint}"
        "If the answer is already in the conversation, answer directly from the existing context. "
        "Do not answer with a generic access refusal.",
    )


def _asks_for_filesystem_metadata(lowered_input: str) -> bool:
    metadata_hints = (
        "metadata",
        "stat",
        "size",
        "modified",
        "modification",
        "mtime",
        "newest",
        "latest",
        "oldest",
        "permissions",
        "mode",
        "exists",
        "metadati",
        "dimensione",
        "modificato",
        "modifica",
        "recente",
        "nuovo",
        "permessi",
        "esiste",
    )
    return any(hint in lowered_input for hint in metadata_hints)


def read_only_tool_hesitation_reply_handling(
    *,
    content: str,
    available_tool_names: set[str],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if policy_state.synthesis_retries >= 1:
        return None
    read_only_tools = available_tool_names & {"bash", "read_file", "list_files", "search_web", "fetch_url"}
    if not read_only_tools:
        return None
    lowered = content.lower()
    token_set = set(re.findall(r"[a-z0-9àèéìòù]+", lowered))
    action_tokens = {"use", "using", "run", "running", "list", "read", "search", "fetch", "open", "inspect", "eseguire", "usare", "leggere", "elencare", "cercare", "aprire"}
    permission_tokens = {"want", "like", "should", "can", "may", "proceed", "permission", "vuoi", "vorresti", "posso", "proceda", "procedere", "permesso"}
    future_tokens = {"need", "would", "first", "have", "bisogno", "dovrei", "devo", "prima"}
    if not (token_set & action_tokens and token_set & permission_tokens):
        return None
    if not (token_set & future_tokens):
        return None
    preferred = ", ".join(name for name in ("list_files", "read_file", "search_web", "fetch_url", "bash") if name in read_only_tools)
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "A safe read-only tool is already available for this request. "
        f"Choose the most relevant tool now from: {preferred}. "
        "Do not ask the user for permission before using a safe read-only tool. "
        "After the tool result arrives, answer directly.",
    )


def guarded_shell_reply_handling(
    *,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if policy_state.synthesis_retries >= 1:
        return None
    for message in reversed(messages):
        if message.get("role") == "user":
            return None
        if message.get("role") != "tool" or message.get("tool_name") != "bash":
            continue
        raw_content = message.get("content")
        if not isinstance(raw_content, str):
            return None
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            return None
        if payload.get("ok") is not False:
            return None
        error = str(payload.get("error") or "").lower()
        if "shell operators are not allowed" not in error and "shell redirection is not allowed" not in error:
            return None
        lowered = content.lower()
        if "not riuscito" not in lowered and "could not" not in lowered and "non sono riuscito" not in lowered:
            return None
        policy_state.synthesis_retries += 1
        return (
            "retry",
            "The previous bash command used blocked shell operators or redirection. "
            "Do not explain the failure and stop there. "
            "Retry now using separate safe bash calls, one command at a time, such as `lscpu`, `free -h`, or `cat /proc/cpuinfo`, then summarize the combined results.",
        )
    return None


def machine_resource_tool_misdirection_handling(
    *,
    intent: str | None,
    user_input: str,
    tool_calls: list[dict[str, Any]],
    content: str,
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if policy_state.synthesis_retries >= 2:
        return None
    lowered_input = user_input.lower()
    machine_tokens = {"pc", "computer", "machine", "macchina", "laptop", "notebook", "portatile", "system", "sistema", "risorse", "resources", "hardware"}
    if not any(token in lowered_input for token in machine_tokens):
        return None
    if successful_bash_results_in_current_turn(messages):
        return None
    if tool_calls:
        names = {
            (call.get("function", {}) or {}).get("name")
            for call in tool_calls
        }
        if names & {"list_files", "read_file"}:
            policy_state.synthesis_retries += 1
            return (
                "retry",
                "This request is about local machine resources, not workspace files. "
                "Do not use list_files or read_file for this turn. "
                "Use bash with safe machine-inspection commands such as `uname -srm`, `lscpu`, `free -h`, `nproc`, or `cat /etc/os-release`, then summarize the results briefly in a few bullets.",
            )
        return None
    lowered = content.lower()
    hesitation_hints = (
        "what specific resources",
        "what resources would you like",
        "i have listed the files",
        "i listed the files",
        "source code",
        "configuration files",
        "documentation",
        "quali risorse",
        "che risorse",
        "ho elencato i file",
        "ho già elencato i file",
        "codice sorgente",
        "file di configurazione",
        "documentazione",
    )
    if not any(hint in lowered for hint in hesitation_hints):
        return None
    if not has_recent_tool_result(messages, "list_files") and "listed the files" not in lowered and "elencato i file" not in lowered:
        return None
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "This request is about local machine resources, not project files. "
        "Do not ask which project files to inspect. "
        "Use bash now with safe machine-inspection commands such as `uname -srm`, `lscpu`, `free -h`, `nproc`, or `cat /etc/os-release`, then answer briefly from those results.",
    )


def fake_tool_response_handling(
    *,
    intent: str | None,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    lowered = content.lower()
    if not any(hint in lowered for hint in FAKE_TOOL_RESPONSE_HINTS):
        return None
    if intent == INTENT_CODEBASE_INSPECTION:
        listed = recent_listed_file_paths(messages)
        if listed:
            read_results = successful_read_results_in_current_turn(messages)
            preferred = []
            seen: set[str] = set()
            for item in read_results:
                path = item.get("path")
                if isinstance(path, str) and path not in seen:
                    preferred.append(path)
                    seen.add(path)
            for path in listed:
                if path not in seen:
                    preferred.append(path)
                    seen.add(path)
            top = preferred[:5]
            if top:
                lines = "\n".join(f"- `{path}`" for path in top)
                return ("final", f"I file più importanti da leggere per primi sono:\n{lines}")
    if is_binary_or_pdf_analysis_intent(intent):
        candidates = likely_binary_candidates_from_recent_listing(messages, limit=1)
        if candidates:
            write_results = successful_write_results_in_current_turn(messages)
            written_paths = {
                str(item.get("path")).strip()
                for item in write_results
                if isinstance(item.get("path"), str) and str(item.get("path")).strip()
            }
            docs_note = ""
            if {"AGENTS.md", "REPORT.md"}.issubset(written_paths):
                docs_note = " AGENTS.md and REPORT.md were initialized."
            return (
                "final",
                _binary_seeded_summary_for_candidate(
                    candidates[0],
                    normalize_relative_path=normalize_relative_path,
                )
                + docs_note,
            )
    if intent == "current_factual_lookup":
        fetched = latest_fetch_url_result_in_current_turn(messages)
        if fetched is not None:
            if fetched.get("has_more"):
                return None
            title = fetched.get("title") if isinstance(fetched.get("title"), str) else ""
            highlights = fetched.get("highlights") if isinstance(fetched.get("highlights"), list) else []
            highlights = [item for item in highlights if isinstance(item, str) and item.strip()]
            excerpt = fetched.get("text") if isinstance(fetched.get("text"), str) else ""
            parts = []
            if title:
                parts.append(f"Title: {title}")
            if highlights:
                parts.append("Highlights: " + " | ".join(highlights[:3]))
            elif excerpt.strip():
                parts.append(f"Excerpt: {excerpt.strip()[:600]}")
            return (
                "final",
                "A concrete page was already fetched for this lookup. Use this evidence instead of fabricating tool output.\n\n"
                + "\n".join(parts),
            )
    if intent == "file_edit":
        write_results = successful_write_results_in_current_turn(messages)
        if write_results:
            paths = {
                str(item.get("path")).strip()
                for item in write_results
                if isinstance(item.get("path"), str) and str(item.get("path")).strip()
            }
            if paths:
                return ("final", file_edit_completion_message(paths))
        read_results = successful_read_results_in_current_turn(messages)
        if not write_results and len(read_results) >= 2:
            if policy_state.synthesis_retries >= 2:
                return (
                    "final",
                    "The model kept fabricating tool results instead of applying the requested file edit, even after the source and target files were read. Retry with a smaller edit request or reset the session.",
                )
            policy_state.synthesis_retries += 1
            return (
                "retry",
                "Do not fabricate tool outputs or <tool_response> blocks. The source text and the current target file have already been read. "
                "Now emit one real edit tool call only: use append_file to add a new section, or replace_in_file to update existing text. "
                "Prefer append_file when the request is to add a new section. Do not use write_file for a full rewrite unless the user explicitly asked for replacement.",
            )
    if policy_state.synthesis_retries >= 1:
        return (
            "final",
            "The model started fabricating tool results in plain text instead of using real tool calls or answering from existing evidence. Retry with a narrower request or reset the session.",
        )
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "Do not fabricate tool outputs or <tool_response> blocks in assistant text. Use a real tool call, or answer only from the tool results that already exist in the conversation.",
    )


def binary_seeded_summary(messages: list[dict[str, Any]]) -> str | None:
    candidates = likely_binary_candidates_from_recent_listing(messages, limit=1)
    if not candidates:
        return None
    return _binary_seeded_summary_for_candidate(candidates[0], normalize_relative_path=normalize_relative_path)


def text_document_followup_handling(
    *,
    intent: str | None,
    user_input: str,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return None
    read_results = successful_read_results_in_current_turn(messages)
    if not read_results:
        return None
    lowered = user_input.lower()
    latest = read_results[-1]
    content = latest.get("content")
    path = latest.get("path")
    if not isinstance(content, str) or not isinstance(path, str):
        return None
    if any(hint in lowered for hint in SHOW_CONTENT_HINTS):
        if tool_calls:
            return ("final", content)
    if any(hint in lowered for hint in SUMMARY_HINTS):
        if tool_calls:
            if policy_state.synthesis_retries >= 1:
                return (
                    "final",
                    f"I read `{path}` but the model did not produce the requested summary. Try a more specific summary request or a smaller file chunk.",
                )
            policy_state.synthesis_retries += 1
            return (
                "retry",
                f"You already read `{path}`. Answer now using only that file content. "
                "Do not inspect other files or metadata. Produce the requested summary directly from the file you already read.",
            )
    return None


def current_factual_lookup_fetch_followup(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
    parse_arguments: Any,
) -> tuple[str, str] | None:
    if intent != "current_factual_lookup":
        return None
    if len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    name = function.get("name")
    if name != "fetch_url":
        return None
    fetched = latest_fetch_url_result_in_current_turn(messages)
    if fetched is None:
        return None
    arguments = parse_arguments(function.get("arguments"))
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    fetched = last_fetch_url_result(messages, url.strip())
    if fetched is None:
        return None
    if fetched.get("has_more"):
        next_start_char = fetched.get("next_start_char")
        if isinstance(next_start_char, int) and next_start_char >= 0 and policy_state.synthesis_retries < 1:
            current_start_char = arguments.get("start_char")
            if current_start_char == next_start_char:
                return None
            policy_state.synthesis_retries += 1
            title = fetched.get("title") if isinstance(fetched.get("title"), str) else ""
            details = [f"Next chunk start_char: {next_start_char}"]
            if title:
                details.insert(0, f"Page title: {title}")
            return (
                "retry",
                "You already fetched one chunk of this page. If you need more evidence, fetch the same URL again with "
                f"start_char={next_start_char}. Answer from the chunks you have and continue only if needed. "
                + " ".join(details),
            )
    title = fetched.get("title") if isinstance(fetched.get("title"), str) else ""
    text = fetched.get("text") if isinstance(fetched.get("text"), str) else ""
    highlights = fetched.get("highlights") if isinstance(fetched.get("highlights"), list) else []
    highlights = [item for item in highlights if isinstance(item, str) and item.strip()]
    excerpt = text.strip()[:600]
    if policy_state.synthesis_retries >= 1:
        return None
    policy_state.synthesis_retries += 1
    details = []
    if title:
        details.append(f"Page title: {title}")
    if highlights:
        details.append("Highlights: " + " | ".join(highlights[:3]))
    if excerpt:
        details.append(f"Fetched page excerpt: {excerpt}")
    details_text = " ".join(details)
    return (
        "retry",
        "You already opened this exact page with fetch_url. Do not fetch the same URL again. "
        "Answer now from the fetched page content already in memory. "
        f"{details_text}".strip(),
    )


def url_inspection_fetch_followup(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
    parse_arguments: Any,
) -> tuple[str, str] | None:
    if intent != "url_inspection":
        return None
    if len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    if function.get("name") != "fetch_url":
        return None
    fetched = latest_fetch_url_result_in_current_turn(messages)
    if fetched is None:
        return None
    if not fetched.get("has_more"):
        return None
    arguments = parse_arguments(function.get("arguments"))
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    next_start_char = fetched.get("next_start_char")
    if not isinstance(next_start_char, int) or next_start_char < 0:
        return None
    current_start_char = arguments.get("start_char")
    if current_start_char == next_start_char:
        return None
    if policy_state.synthesis_retries >= 1:
        return None
    policy_state.synthesis_retries += 1
    title = fetched.get("title") if isinstance(fetched.get("title"), str) else ""
    chunk_index = fetched.get("chunk_index")
    chunk_count = fetched.get("chunk_count")
    chunk_note = ""
    if isinstance(chunk_index, int) and isinstance(chunk_count, int):
        chunk_note = f" Chunk {chunk_index}/{chunk_count}."
    details = []
    if title:
        details.append(f"Page title: {title}")
    details.append(f"Next chunk start_char: {next_start_char}")
    return (
        "retry",
        "You already fetched one chunk of this page. If you need more evidence, fetch the same URL again with start_char="
        f"{next_start_char}. Answer from the chunks you have and continue only if needed."
        f"{chunk_note} {' '.join(details)}".strip(),
    )


def file_edit_placeholder_handling(
    *,
    intent: str | None,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    return _file_edit_placeholder_handling(
        intent=intent,
        content=content,
        messages=messages,
        policy_state=policy_state,
        successful_write_results_in_current_turn=successful_write_results_in_current_turn,
        successful_read_results_in_current_turn=successful_read_results_in_current_turn,
        file_edit_completion_message=file_edit_completion_message,
    )


def file_edit_post_write_reply_handling(
    *,
    intent: str | None,
    content: str,
    messages: list[dict[str, Any]],
) -> tuple[str, str] | None:
    return _file_edit_post_write_reply_handling(
        intent=intent,
        content=content,
        messages=messages,
        successful_write_results_in_current_turn=successful_write_results_in_current_turn,
        file_edit_completion_message=file_edit_completion_message,
    )


def placeholder_write_replacement_text(messages: list[dict[str, Any]], content: str) -> str | None:
    return _placeholder_write_replacement_text(
        messages,
        content,
        successful_read_results_in_current_turn=successful_read_results_in_current_turn,
        latest_successful_read_result_in_current_turn=latest_successful_read_result_in_current_turn,
        normalize_relative_path=normalize_relative_path,
    )


def infer_file_edit_section_append(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return _infer_file_edit_section_append(
        intent=intent,
        user_input=user_input,
        messages=messages,
        successful_write_results_in_current_turn=successful_write_results_in_current_turn,
        successful_read_results_in_current_turn=successful_read_results_in_current_turn,
        normalize_relative_path=normalize_relative_path,
    )


def tool_names_from_definitions(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in tools:
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def unsupported_tool_prompt(
    *,
    tool_calls: list[dict[str, Any]],
    allowed_tool_names: set[str],
    route: Any,
) -> str | None:
    if not tool_calls:
        return None
    if not allowed_tool_names:
        blocked = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) or {}
            name = function.get("name")
            if isinstance(name, str) and name:
                blocked.append(name)
        if not blocked:
            return None
        return (
            f"The previous tool call used tools that are not needed for this request: {', '.join(sorted(set(blocked)))}. "
            "Do not use any tools for this turn. Answer directly from general knowledge."
        )
    unsupported = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {}) or {}
        name = function.get("name")
        if isinstance(name, str) and name and name not in allowed_tool_names:
            unsupported.append(name)
    if not unsupported:
        return None
    allowed = ", ".join(sorted(allowed_tool_names))
    blocked = ", ".join(sorted(set(unsupported)))
    binary_hint = ""
    if is_binary_or_pdf_analysis_intent(route.intent):
        binary_hint = (
            " For binary or PDF analysis, first discover a real candidate path with list_files if the file name is not explicit, "
            "then use a shell-oriented tool such as bash for strings, pdftotext, file, or another bounded binary-aware command."
        )
    return (
        f"The previous tool call used unsupported tools for this request: {blocked}. "
        f"Use only these tools for this turn: {allowed}.{binary_hint}"
    )


def workspace_listing_scope_prompt(
    *,
    intent_class: str | None,
    tool_calls: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    if intent_class != "workspace_discovery":
        return None
    for tool_call in tool_calls:
        function = tool_call.get("function", {}) or {}
        if function.get("name") != "list_files":
            continue
        arguments = parse_arguments(function.get("arguments"))
        if arguments.get("recursive") is True:
            return (
                "For workspace discovery, start with a top-level directory listing only. "
                "Retry with list_files on the current workdir using recursive=false, then answer from that listing. "
                "Do not expand recursively unless the user explicitly asks for a subtree or deeper exploration."
            )
    return None


def _binary_listing_guidance(messages: list[dict[str, Any]]) -> str:
    return _binary_listing_guidance_impl(
        messages,
        likely_binary_candidates_from_recent_listing=likely_binary_candidates_from_recent_listing,
    )
