from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

from .compaction import (
    CompactionPlan,
    SUMMARY_MARKER,
    apply_compaction,
    build_hybrid_refinement_messages,
    normalize_model_summary,
    plan_compaction,
)
from .ollama_client import OllamaClient
from .ollama_client import is_thinking_unsupported_error
from .policy.context import BudgetPressure, evaluate_budget_pressure
from .guardrails.factual import allows_fetch_url_query
from .guardrails.audio import (
    AudioChunk,
    ExplicitAudioRequest,
    prepare_audio_chunks,
    resolve_explicit_audio_requests,
)
from .intent.gate import intent_gate_decision, intent_gate_messages, parse_intent_gate_reply
from .intent.router import INTENT_CODEBASE_INSPECTION, is_static_file_analysis_intent
from .model.guidance import MODEL_FIRST_INTENT_GUIDANCE, model_first_post_tool_prompt
from .model.payloads import (
    ModelPayloadCompactor,
    compact_message_for_model as _compact_message_for_model,
    compact_tool_payload as _compact_tool_payload,
)
from .events import (
    DebugTimingEvent,
    EmptyReplyRetryEvent,
    EventSink,
    ModelRequestEvent,
    RepeatedToolRetryEvent,
    ToolRouteEvent,
    ToolResultCompactEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingUnavailableEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .messages import (
    assistant_message_for_history,
    estimate_prompt_tokens,
    recent_listed_paths,
    recent_listed_file_paths,
    successful_read_results_in_current_turn,
    successful_write_results_in_current_turn,
)
from .tools.call_parser import fallback_tool_calls, parse_arguments, repair_placeholder_write_payload
from .tools.guardrails import (
    binary_analysis_guard_prompt,
    binary_listing_retry_prompt,
    binary_seeded_summary,
    binary_tool_strategy_prompt,
    binary_text_reply_handling,
    codebase_redundant_listing_prompt,
    current_factual_lookup_fetch_followup,
    codebase_review_reply_handling,
    directory_listing_target_from_recent_listing,
    has_structured_explicit_code_review_path,
    local_missing_explicit_code_review_file_result,
    apply_deterministic_file_edit,
    apply_deterministic_bounded_command,
    file_edit_post_write_reply_handling,
    file_edit_placeholder_handling,
    fake_tool_response_handling,
    filesystem_metadata_reply_handling,
    filesystem_read_path_guard_prompt,
    filesystem_text_reply_handling,
    generic_tool_access_reply_handling,
    guarded_shell_reply_handling,
    machine_resource_tool_misdirection_handling,
    read_only_tool_hesitation_reply_handling,
    infer_file_edit_section_append,
    local_text_document_result,
    local_codebase_metadata_result,
    local_codebase_architecture_result,
    local_codebase_hotspot_result,
    local_codebase_priority_files_result,
    local_codebase_review_result,
    local_codebase_review_after_missing_read_path,
    local_current_factual_result,
    local_tooling_concept_result,
    local_mixed_local_web_evidence_result,
    local_directory_listing_result,
    local_filesystem_metadata_result,
    local_markdown_checkbox_extraction_result,
    local_workspace_file_classification_result,
    local_workspace_file_presence_result,
    local_workspace_security_scan_result,
    markdown_checkbox_redundant_read_prompt,
    local_static_sample_evidence_result,
    local_static_reverse_inspection_result,
    local_explicit_text_result,
    condense_explicit_text_summary_messages,
    should_defer_explicit_text_summary_to_model,
    local_explicit_pdf_result,
    local_assistant_identity_result,
    local_pure_chitchat_result,
    assistant_identity_system_prompt,
    placeholder_write_replacement_text,
    seed_current_factual_tool,
    seed_codebase_listing,
    seed_codebase_review_reads,
    seed_project_metadata_read,
    seed_explicit_text_read,
    seed_explicit_pdf_read,
    seed_directory_discovery,
    seed_filesystem_metadata,
    seed_markdown_checkbox_extraction,
    seed_workspace_file_classification,
    seed_workspace_file_presence_check,
    seed_workspace_security_scan,
    seed_binary_discovery,
    storage_command_strategy_prompt,
    text_document_followup_handling,
    url_inspection_fetch_followup,
    tool_names_from_definitions,
    unsupported_tool_prompt,
    workspace_listing_scope_prompt,
)
from .tools.router import ToolRoute, route_tool_categories
from .tools.execution_policy import ToolExecutionPolicy
from .skill_hints import should_bootstrap_workspace_docs, startup_prompt_for_skill, workspace_doc_bootstrap_actions
from ..skills import DEFAULT_SKILL_REF, Skill
from ..tooling.registry import ToolRegistry
from .policy.turn import (
    TurnPolicyState,
    classify_model_reply,
    format_max_loops_message,
    register_tool_calls,
)
from .guardrails.vision import ExplicitImageRequest, encode_image_base64, resolve_explicit_image_requests
from ..tooling.common import ToolError


BASE_SYSTEM_PROMPT = """You are the local assistant running inside the Orbit CLI in the user's environment.
Do not invent a personal name, vendor, or creator.
Use tools only when needed and follow tool outputs exactly.
Answer directly when no tool is needed.
"""

MODEL_FIRST_SYSTEM_PROMPT = """You are the local assistant running inside the Orbit CLI in the user's environment.
Do not invent a personal name, vendor, or creator.
Reuse the conversation and prior tool results before calling a tool.
If a tool is needed, choose the smallest valid tool and prefer one tool at a time.
Use bash for machine and environment inspection.
Use bash with rg or grep for extracting explicit textual markers or line patterns from files, such as Markdown checkboxes, TODO/FIXME, tags, dates, URLs, errors, warnings, CVEs, IPs, or hashes.
Use list_files for workspace structure.
Use read_file for a known file path.
Use write_file, append_file, or replace_in_file for file edits.
If the user asks for code generation without an explicit file path or save request, answer inline and do not write a file.
Use search_web for current information.
Use fetch_url only for an explicit URL already provided by the user.
For machine or resource questions, prefer bash over workspace file tools. Use df for filesystem free space at the requested path or mount point, including /. Use du only for directory size.
For safe read-only inspection, do not ask permission: use the tool.
Never guess file contents.
Use filesystem tools when filesystem evidence is needed.
Prefer read_file before editing.
Use bash only for bounded inspection and safe commands.
When using bash, do not combine commands with &&, ;, |, >, or <. Use separate calls.
Do not claim you lack tool access when a relevant tool is available.
Do not volunteer identity, vendor, or model family unless asked.
Answer directly when no tool is needed.
"""

MINIMAL_CHAT_SYSTEM_PROMPT = """You are the concise local assistant running inside the Orbit CLI in the user's environment.
Do not invent a personal name, vendor, or creator.
Answer directly.
"""

MINIMAL_CHAT_HISTORY_MESSAGES = 4
TOOL_APPEND_MIN_RESPONSE_TOKENS = 1200
TOOL_APPEND_RESPONSE_CTX_RATIO = 0.12

WARN_USAGE_RATIO = 0.80
CRIT_USAGE_RATIO = 0.92

VISION_SYSTEM_PROMPT = """You are the local assistant running inside the Orbit CLI.
Analyze the attached local image from the user's workspace and answer directly from visible evidence only.
If the user asks to read text from the image, transcribe only visible text.
If the image does not contain enough evidence, say so briefly.
When multiple images are attached, treat them as distinct images in the same order as their labels.
When comparing images, mention major content differences, including visible text if present.
Keep the answer concise.
"""

AUDIO_SYSTEM_PROMPT = """You are the local assistant running inside the Orbit CLI.
Analyze attached local audio chunks from the user's workspace.
Transcribe speech faithfully when possible.
If a chunk is unclear, say unclear for that chunk.
Do not claim that no audio was provided when an audio attachment is present.
Keep the answer concise.
"""

TRANSIENT_SYSTEM_FLAG = "_orbit_transient_system"


@dataclass(frozen=True)
class TurnStatus:
    active_model: str
    context_window: int | None
    session_messages: int
    session_turns: int
    prompt_tokens: int | None
    estimated_prompt_tokens: int
    output_tokens: int | None
    prefill_tps: float | None
    decode_tps: float | None
    model_elapsed_sec: float | None
    wall_elapsed_sec: float | None
    tool_elapsed_sec: float | None
    usage_ratio: float | None
    warning: str | None
    think_state: str = "no"
    show_thinking_state: str = "off"


@dataclass(frozen=True)
class TurnResult:
    content: str
    status: TurnStatus


@dataclass
class TurnMetrics:
    prompt_tokens: int | None = None
    prompt_eval_duration_ns: int | None = None
    output_tokens: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None
    tool_duration_ns: int = 0
    wall_duration_ns: int | None = None


class AgentLoop:
    def __init__(
        self,
        *,
        client: OllamaClient,
        registry: ToolRegistry,
        max_loops: int = 12,
        temperature: float = 0.0,
        skill: Skill | None = None,
        tools_enabled: bool = True,
        think_mode: str = "auto",
        show_thinking: bool = False,
        debug_timing: bool = False,
    ) -> None:
        self.client = client
        self.registry = registry
        self.max_loops = max_loops
        self.temperature = temperature
        self.skill = skill
        self.tools_enabled = tools_enabled
        self.think_mode = think_mode
        self.show_thinking = show_thinking
        self.debug_timing = debug_timing
        self._model_metadata = None
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        workdir = getattr(self.registry, "workdir", Path("."))
        self._tool_policy = ToolExecutionPolicy(Path(workdir))
        self._model_payloads = ModelPayloadCompactor()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self._tool_policy.reset()
        self._model_payloads.clear_message_cache()

    def set_skill(self, skill: Skill | None) -> None:
        self.skill = skill
        self.reset()

    def restore_messages(self, messages: list[dict[str, Any]]) -> None:
        if messages:
            restored = [
                dict(item)
                for item in messages
                if not (item.get("role") == "system" and item.get(TRANSIENT_SYSTEM_FLAG))
            ]
            if restored and restored[0].get("role") == "system":
                restored[0]["content"] = self._system_prompt()
            else:
                restored.insert(0, {"role": "system", "content": self._system_prompt()})
            self.messages = restored
            self._tool_policy.rehydrate_from_messages(self.messages)
            self._model_payloads.clear_message_cache()
        else:
            self.reset()

    def compact(self, *, overflow_tokens: int = 0) -> bool:
        plan = plan_compaction(self.messages, overflow_tokens=overflow_tokens)
        if plan is None:
            return False
        summary = self._refine_compaction_summary(plan)
        self.messages = apply_compaction(plan, summary)
        return True

    def run_turn(
        self,
        user_input: str,
        on_event: EventSink | None = None,
        image_attachments: list[ExplicitImageRequest] | None = None,
    ) -> TurnResult:
        turn_started_at = time.monotonic_ns()
        self.messages.append({"role": "user", "content": user_input})
        metrics = TurnMetrics()
        policy_state = TurnPolicyState()
        model_first_runtime = self._prefers_model_first_runtime()
        route_started_at = time.monotonic_ns()
        route = route_tool_categories(user_input, skill=self.skill) if self.tools_enabled else None
        self._emit_timing(on_event, "route", route_started_at, route.intent if route is not None else "chat-only")
        pre_model_started_at = time.monotonic_ns()
        explicit_audio_result = self._run_explicit_audio_request(
            user_input=user_input,
            metrics=metrics,
            on_event=on_event,
        )
        if explicit_audio_result is not None:
            metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
            return self._result(explicit_audio_result, metrics)
        explicit_image_result = self._run_explicit_image_request(
            user_input=user_input,
            metrics=metrics,
            on_event=on_event,
            image_attachments=image_attachments,
        )
        if explicit_image_result is not None:
            metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
            return self._result(explicit_image_result, metrics)
        if not model_first_runtime:
            chitchat_result = local_pure_chitchat_result(user_input)
            if chitchat_result is not None:
                metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                return self._result(chitchat_result, metrics)
            identity_result = local_assistant_identity_result(user_input)
            if identity_result is not None:
                metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                return self._result(identity_result, metrics)
        if not model_first_runtime:
            identity_prompt = assistant_identity_system_prompt(user_input)
            if identity_prompt is not None:
                self._append_system_message(identity_prompt)
        if (
            self.tools_enabled
            and model_first_runtime
            and recent_listed_paths(self.messages, 1)
            and directory_listing_target_from_recent_listing(user_input, self.messages) is None
        ):
            local_listing_result = local_directory_listing_result(
                intent=route.intent,
                user_input=user_input,
                messages=self.messages,
            )
            if local_listing_result is not None:
                metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                return self._result(local_listing_result, metrics)
        gate_decision = intent_gate_decision(user_input, route)
        if self.tools_enabled and model_first_runtime and gate_decision.confirm:
            if not self._model_confirms_tool_route(user_input=user_input, route=route, metrics=metrics, on_event=on_event):
                route = ToolRoute(
                    intent="chitchat",
                    intent_class="chat_general",
                    categories=(),
                    reason=f"{route.reason}; {gate_decision.reason}; model declined tool route",
                )
        if self.tools_enabled and route is not None:
            if model_first_runtime and route.categories:
                intent_guidance = MODEL_FIRST_INTENT_GUIDANCE.get(route.intent_class)
                if intent_guidance is not None:
                    self._append_system_message(intent_guidance)
            skill_startup_prompt = startup_prompt_for_skill(self.skill, route.intent, self.messages)
            if skill_startup_prompt is not None:
                self._append_system_message(skill_startup_prompt)
            self._bootstrap_skill_workspace_docs(route=route, metrics=metrics, policy_state=policy_state, on_event=on_event)
            if "base64" in user_input.lower():
                deterministic_bounded_result = apply_deterministic_bounded_command(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                if deterministic_bounded_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(deterministic_bounded_result, metrics)
            if model_first_runtime:
                defer_text_summary_to_model = False
                seed_directory_discovery(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_workspace_file_presence_check(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_workspace_file_classification(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_listing_result = local_directory_listing_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_listing_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_listing_result, metrics)
                local_file_presence = local_workspace_file_presence_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_file_presence is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_file_presence, metrics)
                local_file_classification = local_workspace_file_classification_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_file_classification is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_file_classification, metrics)
                seed_filesystem_metadata(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_filesystem_metadata = local_filesystem_metadata_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_filesystem_metadata is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_filesystem_metadata, metrics)
                seed_workspace_security_scan(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_security_scan = local_workspace_security_scan_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_security_scan is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_security_scan, metrics)
                local_checkbox_result = self._try_markdown_checkbox_extraction(
                    user_input=user_input,
                    route=route,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                if local_checkbox_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_checkbox_result, metrics)
                if has_structured_explicit_code_review_path(user_input):
                    seed_codebase_listing(
                        user_input=user_input,
                        route=route,
                        registry=self.registry,
                        messages=self.messages,
                        metrics=metrics,
                        policy_state=policy_state,
                        on_event=on_event,
                    )
                    seed_codebase_review_reads(
                        user_input=user_input,
                        route=route,
                        registry=self.registry,
                        messages=self.messages,
                        metrics=metrics,
                        policy_state=policy_state,
                        on_event=on_event,
                    )
                    missing_code_file = local_missing_explicit_code_review_file_result(
                        intent=route.intent,
                        user_input=user_input,
                        messages=self.messages,
                    )
                    if missing_code_file is not None:
                        metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                        return self._result(missing_code_file, metrics)
                seed_explicit_pdf_read(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_pdf_result = local_explicit_pdf_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_pdf_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_pdf_result, metrics)
                seed_binary_discovery(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_static_evidence = local_static_sample_evidence_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_static_evidence is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_static_evidence, metrics)
                local_static_reverse = local_static_reverse_inspection_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_static_reverse is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_static_reverse, metrics)
                seed_explicit_text_read(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                defer_text_summary_to_model = should_defer_explicit_text_summary_to_model(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if defer_text_summary_to_model:
                    condense_explicit_text_summary_messages(
                        user_input=user_input,
                        messages=self.messages,
                    )
                    model_text_result = self._summarize_explicit_text_evidence(
                        user_input=user_input,
                        metrics=metrics,
                    )
                    if model_text_result is not None:
                        self.messages.append({"role": "assistant", "content": model_text_result})
                        metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                        return self._result(model_text_result, metrics)
                local_text_result = local_explicit_text_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_text_result is not None and not defer_text_summary_to_model:
                    condense_explicit_text_summary_messages(
                        user_input=user_input,
                        messages=self.messages,
                        summary_text=local_text_result,
                    )
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_text_result, metrics)
                seed_current_factual_tool(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_factual_result = local_current_factual_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_factual_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_factual_result, metrics)
            if not model_first_runtime:
                seed_directory_discovery(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_project_metadata_read(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_filesystem_metadata(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_workspace_file_presence_check(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_workspace_file_classification(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_workspace_security_scan(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_checkbox_result = self._try_markdown_checkbox_extraction(
                    user_input=user_input,
                    route=route,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                if local_checkbox_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_checkbox_result, metrics)
                seed_codebase_listing(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_codebase_review_reads(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_explicit_text_read(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_explicit_pdf_read(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_current_factual_tool(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                seed_binary_discovery(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                local_metadata_result = local_codebase_metadata_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_metadata_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_metadata_result, metrics)
                local_codebase_files_result = local_codebase_priority_files_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_codebase_files_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_codebase_files_result, metrics)
                local_codebase_arch_result = local_codebase_architecture_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_codebase_arch_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_codebase_arch_result, metrics)
                local_codebase_hotspot_result_text = local_codebase_hotspot_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_codebase_hotspot_result_text is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_codebase_hotspot_result_text, metrics)
                local_codebase_review_result_text = local_codebase_review_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_codebase_review_result_text is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_codebase_review_result_text, metrics)
                local_file_presence = local_workspace_file_presence_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_file_presence is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_file_presence, metrics)
                local_security_scan = local_workspace_security_scan_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_security_scan is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_security_scan, metrics)
                local_file_classification = local_workspace_file_classification_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_file_classification is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_file_classification, metrics)
                local_listing_result = local_directory_listing_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_listing_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_listing_result, metrics)
                local_filesystem_metadata = local_filesystem_metadata_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_filesystem_metadata is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_filesystem_metadata, metrics)
                local_text_result = local_explicit_text_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_text_result is not None:
                    condense_explicit_text_summary_messages(
                        user_input=user_input,
                        messages=self.messages,
                        summary_text=local_text_result,
                    )
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_text_result, metrics)
                local_pdf_result = local_explicit_pdf_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_pdf_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_pdf_result, metrics)
            if not model_first_runtime:
                local_factual_result = local_current_factual_result(
                    intent=route.intent,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_factual_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_factual_result, metrics)
            if not model_first_runtime:
                if route.intent != INTENT_CODEBASE_INSPECTION:
                    local_tooling_result = local_tooling_concept_result(user_input)
                    if local_tooling_result is not None:
                        metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                        return self._result(local_tooling_result, metrics)
                deterministic_bounded_result = apply_deterministic_bounded_command(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                if deterministic_bounded_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(deterministic_bounded_result, metrics)
                deterministic_edit_result = apply_deterministic_file_edit(
                    user_input=user_input,
                    route=route,
                    registry=self.registry,
                    messages=self.messages,
                    metrics=metrics,
                    policy_state=policy_state,
                    on_event=on_event,
                )
                if deterministic_edit_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(deterministic_edit_result, metrics)
        self._emit_timing(on_event, "pre-model", pre_model_started_at)
        for _ in range(self.max_loops):
            policy_state.loop_count += 1
            if on_event is not None:
                on_event(ModelRequestEvent(loop=policy_state.loop_count))
            tools = []
            allowed_tool_names: set[str] = set()
            if self.tools_enabled:
                tools = self.registry.definitions_for_categories(route.categories)
                tools = self._tool_policy.filter_tools(tools)
                allowed_tool_names = tool_names_from_definitions(tools)
                if on_event is not None:
                    on_event(
                        ToolRouteEvent(
                            loop=policy_state.loop_count,
                            intent=route.intent,
                            categories=route.categories,
                            reason=route.reason,
                        )
                    )
            response = self._chat_response(loop=policy_state.loop_count, tools=tools, route=route, on_event=on_event)
            self._update_metrics(metrics, response)
            message = response.get("message", {})
            self.messages.append(assistant_message_for_history(message))
            tool_calls = []
            repaired_placeholder_tool_call = False
            if self.tools_enabled:
                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    tool_calls = fallback_tool_calls(message.get("content"))
                raw_content = message.get("content")
                if route.intent == "file_edit" and (
                    not tool_calls
                    or self._should_infer_file_edit_after_repeated_reads(tool_calls)
                    or self._should_force_file_edit_inference(
                        content=str(raw_content or ""),
                        tool_calls=tool_calls,
                    )
                ):
                    inferred_edit = infer_file_edit_section_append(
                        intent=route.intent,
                        user_input=user_input,
                        messages=self.messages,
                    )
                    if inferred_edit is not None:
                        tool_calls = [{"function": {"name": inferred_edit["name"], "arguments": inferred_edit["arguments"]}}]
                    replacement_text = None
                    if isinstance(raw_content, str):
                        replacement_text = placeholder_write_replacement_text(self.messages, raw_content)
                    if isinstance(raw_content, str) and replacement_text is not None:
                        repaired = repair_placeholder_write_payload(raw_content, replacement_text)
                        if repaired is not None:
                            repaired_placeholder_tool_call = True
                            tool_calls = [{"function": {"name": repaired["name"], "arguments": repaired.get("arguments", {})}}]
                workspace_scope_prompt = workspace_listing_scope_prompt(
                    intent_class=route.intent_class,
                    tool_calls=tool_calls,
                    parse_arguments=parse_arguments,
                )
                if workspace_scope_prompt is not None:
                    self._append_system_message(workspace_scope_prompt)
                    continue
                redundant_codebase_listing = codebase_redundant_listing_prompt(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    parse_arguments=parse_arguments,
                )
                if redundant_codebase_listing is not None:
                    self._append_system_message(redundant_codebase_listing)
                    continue
                binary_guard_prompt = binary_analysis_guard_prompt(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    parse_arguments=parse_arguments,
                )
                if binary_guard_prompt is not None:
                    self._append_system_message(binary_guard_prompt)
                    continue
                binary_tool_prompt = binary_tool_strategy_prompt(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    parse_arguments=parse_arguments,
                )
                if binary_tool_prompt is not None:
                    self._append_system_message(binary_tool_prompt)
                    continue
                storage_tool_prompt = storage_command_strategy_prompt(
                    intent_class=route.intent_class,
                    user_input=user_input,
                    tool_calls=tool_calls,
                    parse_arguments=parse_arguments,
                )
                if storage_tool_prompt is not None:
                    self._append_system_message(storage_tool_prompt)
                    continue
                machine_resource_misdirection = machine_resource_tool_misdirection_handling(
                    intent=route.intent,
                    user_input=user_input,
                    tool_calls=tool_calls,
                    content=str(message.get("content") or ""),
                    messages=self.messages,
                    policy_state=policy_state,
                )
                if machine_resource_misdirection is not None:
                    action, payload = machine_resource_misdirection
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                unsupported_prompt = unsupported_tool_prompt(
                    tool_calls=tool_calls,
                    allowed_tool_names=allowed_tool_names,
                    route=route,
                )
                if unsupported_prompt is not None:
                    self._append_system_message(unsupported_prompt)
                    continue
                binary_listing_prompt = binary_listing_retry_prompt(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    parse_arguments=parse_arguments,
                )
                if binary_listing_prompt is not None:
                    self._append_system_message(binary_listing_prompt)
                    continue
                codebase_missing_read_path = local_codebase_review_after_missing_read_path(
                    intent=route.intent,
                    user_input=user_input,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    parse_arguments=parse_arguments,
                )
                if codebase_missing_read_path is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(codebase_missing_read_path, metrics)
                filesystem_path_prompt = filesystem_read_path_guard_prompt(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    parse_arguments=parse_arguments,
                )
                if filesystem_path_prompt is not None:
                    self._append_system_message(filesystem_path_prompt)
                    continue
                local_document_result = local_text_document_result(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    parse_arguments=parse_arguments,
                )
                if local_document_result is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_document_result, metrics)
                document_followup = text_document_followup_handling(
                    intent=route.intent,
                    user_input=user_input,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    policy_state=policy_state,
                )
                if document_followup is not None:
                    action, payload = document_followup
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                factual_fetch_followup = current_factual_lookup_fetch_followup(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    policy_state=policy_state,
                    parse_arguments=parse_arguments,
                )
                if factual_fetch_followup is not None:
                    action, payload = factual_fetch_followup
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                url_fetch_followup = url_inspection_fetch_followup(
                    intent=route.intent,
                    tool_calls=tool_calls,
                    messages=self.messages,
                    policy_state=policy_state,
                    parse_arguments=parse_arguments,
                )
                if url_fetch_followup is not None:
                    action, payload = url_fetch_followup
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                filesystem_text_handling = filesystem_text_reply_handling(
                    intent=route.intent,
                    content=str(message.get("content") or ""),
                    messages=self.messages,
                    policy_state=policy_state,
                )
                if filesystem_text_handling is not None:
                    action, payload = filesystem_text_handling
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                filesystem_metadata_reply = filesystem_metadata_reply_handling(
                    intent=route.intent,
                    user_input=user_input,
                    content=str(message.get("content") or ""),
                    messages=self.messages,
                    policy_state=policy_state,
                )
                if filesystem_metadata_reply is not None:
                    action, payload = filesystem_metadata_reply
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                generic_access_retry = generic_tool_access_reply_handling(
                    content=str(message.get("content") or ""),
                    available_tool_names=allowed_tool_names,
                    policy_state=policy_state,
                )
                if generic_access_retry is not None:
                    action, payload = generic_access_retry
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                guarded_shell_retry = guarded_shell_reply_handling(
                    content=str(message.get("content") or ""),
                    messages=self.messages,
                    policy_state=policy_state,
                )
                if guarded_shell_retry is not None:
                    action, payload = guarded_shell_retry
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                read_only_hesitation = read_only_tool_hesitation_reply_handling(
                    content=str(message.get("content") or ""),
                    available_tool_names=allowed_tool_names,
                    policy_state=policy_state,
                )
                if read_only_hesitation is not None:
                    action, payload = read_only_hesitation
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                fake_tool_response = fake_tool_response_handling(
                    intent=route.intent,
                    content=str(message.get("content") or ""),
                    messages=self.messages,
                    policy_state=policy_state,
                )
                if fake_tool_response is not None:
                    action, payload = fake_tool_response
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                file_edit_post_write = file_edit_post_write_reply_handling(
                    intent=route.intent,
                    content=str(message.get("content") or ""),
                    messages=self.messages,
                )
                if file_edit_post_write is not None:
                    action, payload = file_edit_post_write
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                if not tool_calls:
                    codebase_review_reply = codebase_review_reply_handling(
                        intent=route.intent,
                        user_input=user_input,
                        content=str(message.get("content") or ""),
                        messages=self.messages,
                        policy_state=policy_state,
                    )
                    if codebase_review_reply is not None:
                        action, payload = codebase_review_reply
                        if action == "retry":
                            self._append_system_message(payload)
                            continue
                        metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                        return self._result(payload, metrics)
                binary_text_handling = binary_text_reply_handling(
                    intent=route.intent,
                    content=str(message.get("content") or ""),
                    policy_state=policy_state,
                )
                if binary_text_handling is not None:
                    action, payload = binary_text_handling
                    if action == "retry":
                        self._append_system_message(payload)
                        continue
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(payload, metrics)
                if not repaired_placeholder_tool_call:
                    file_edit_placeholder = file_edit_placeholder_handling(
                        intent=route.intent,
                        content=str(message.get("content") or ""),
                        messages=self.messages,
                        policy_state=policy_state,
                    )
                    if file_edit_placeholder is not None:
                        action, payload = file_edit_placeholder
                        if action == "retry":
                            self._append_system_message(payload)
                            continue
                        metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                        return self._result(payload, metrics)
            decision = classify_model_reply(
                content=str(message.get("content") or ""),
                tool_calls=tool_calls,
                state=policy_state,
                intent=route.intent if self.tools_enabled else None,
                user_input=user_input,
            )
            if decision.action == "final_text":
                metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                return self._result(decision.content or "", metrics)
            if decision.action == "retry_empty_reply":
                policy_state.empty_reply_retries += 1
                if on_event is not None:
                    on_event(EmptyReplyRetryEvent(loop=policy_state.loop_count))
                self._append_system_message(decision.content or "")
                continue
            if decision.action == "retry_repeated_tool":
                policy_state.repeated_tool_retries += 1
                policy_state.synthesis_retries += 1
                if on_event is not None:
                    on_event(RepeatedToolRetryEvent(loop=policy_state.loop_count, detail=decision.content))
                self._append_system_message(decision.content or "")
                continue
            if decision.action in {"abort_empty_reply", "abort_repeated_tool_loop"}:
                metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                return self._result(decision.content or "Turn aborted by policy.", metrics)
            self._execute_tool_calls(
                tool_calls,
                metrics=metrics,
                policy_state=policy_state,
                on_event=on_event,
                intent=route.intent if self.tools_enabled else None,
                user_input=user_input,
            )
            local_factual_result = local_current_factual_result(
                intent=route.intent if self.tools_enabled else None,
                user_input=user_input,
                messages=self.messages,
            )
            if local_factual_result is not None:
                metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                return self._result(local_factual_result, metrics)
            if self._should_stop_after_codebase_assessment_evidence(user_input):
                local_codebase_assessment_text = self._local_codebase_assessment_result(user_input)
                if local_codebase_assessment_text is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_codebase_assessment_text, metrics)
                local_codebase_review_result_text = local_codebase_review_result(
                    intent=route.intent if self.tools_enabled else None,
                    user_input=user_input,
                    messages=self.messages,
                )
                if local_codebase_review_result_text is not None:
                    metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                    return self._result(local_codebase_review_result_text, metrics)
            local_mixed_evidence = local_mixed_local_web_evidence_result(
                intent=route.intent if self.tools_enabled else None,
                user_input=user_input,
                messages=self.messages,
            )
            if local_mixed_evidence is not None:
                metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
                return self._result(local_mixed_evidence, metrics)
            if model_first_runtime and route is not None:
                self._append_model_first_post_tool_prompt(route, tool_calls)
        final_content = format_max_loops_message(self.max_loops, policy_state)
        if self.tools_enabled and route is not None and is_static_file_analysis_intent(route.intent):
            seeded = binary_seeded_summary(self.messages)
            if seeded is not None:
                final_content = seeded
        metrics.wall_duration_ns = time.monotonic_ns() - turn_started_at
        return self._result(final_content, metrics)

    @staticmethod
    def _should_infer_file_edit_after_repeated_reads(tool_calls: list[dict[str, Any]]) -> bool:
        if not tool_calls:
            return False
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) or {}
            if function.get("name") != "read_file":
                return False
        return True

    def _should_force_file_edit_inference(self, *, content: str, tool_calls: list[dict[str, Any]]) -> bool:
        if tool_calls:
            return False
        if successful_write_results_in_current_turn(self.messages):
            return False
        lowered = content.lower()
        return any(
            hint in lowered
            for hint in (
                "<tool_response",
                "tool_response.content",
                "<readme_content>",
                "_content>",
                "open_file",
            )
        )

    @staticmethod
    def _should_stop_after_codebase_assessment_evidence(user_input: str) -> bool:
        lowered = user_input.lower()
        assessment_hints = ("technical assessment", "assessment", "valutazione tecnica")
        risk_hints = ("concrete risk", "one risk", "risk and", "rischio concreto", "un rischio")
        improvement_hints = ("improvement suggestion", "one improvement", "suggestion", "miglioramento")
        return (
            any(hint in lowered for hint in assessment_hints)
            and any(hint in lowered for hint in risk_hints)
            and any(hint in lowered for hint in improvement_hints)
        )

    def _try_markdown_checkbox_extraction(
        self,
        *,
        user_input: str,
        route: ToolRoute,
        metrics: TurnMetrics,
        policy_state: TurnPolicyState,
        on_event: EventSink | None,
    ) -> str | None:
        seed_markdown_checkbox_extraction(
            skill=self.skill,
            user_input=user_input,
            route=route,
            registry=self.registry,
            messages=self.messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
        )
        return local_markdown_checkbox_extraction_result(
            skill=self.skill,
            user_input=user_input,
            messages=self.messages,
        )

    @staticmethod
    def _sanitize_fetch_url_arguments(*, user_input: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if "query" not in arguments:
            return arguments
        if allows_fetch_url_query(user_input):
            return arguments
        sanitized = dict(arguments)
        for key in ("query", "query_mode", "max_matches", "context_chars"):
            sanitized.pop(key, None)
        return sanitized

    def _local_codebase_assessment_result(self, user_input: str) -> str | None:
        read_results = successful_read_results_in_current_turn(self.messages)
        if not read_results:
            return None
        listed = recent_listed_file_paths(self.messages, limit=40)
        ranked = self._rank_codebase_paths(listed)
        files = ranked[:3] or [str(item.get("path")) for item in read_results if isinstance(item.get("path"), str)][:3]
        latest = read_results[-1]
        path = str(latest.get("path") or files[0])
        total_lines = latest.get("total_lines")
        line_count = total_lines if isinstance(total_lines, int) and total_lines > 0 else len(str(latest.get("content") or "").splitlines())
        english = not any(token in user_input.lower() for token in ("valutazione", "rischio", "miglioramento"))
        if english:
            return (
                f"Most relevant files visible from the workspace: {', '.join(f'`{item}`' for item in files)}.\n"
                f"Assessment: `{path}` appears to be the main execution surface; it is large enough ({line_count} lines) that loop, routing, or guardrail changes can have broad effects.\n"
                f"Concrete risk: regressions in `{path}` can alter tool execution, stop conditions, or session behavior beyond the touched feature.\n"
                "Improvement: keep focused regression tests around agent-loop decisions before changing routing, guardrails, or post-tool synthesis."
            )
        return (
            f"File piu` rilevanti visibili nel workspace: {', '.join(f'`{item}`' for item in files)}.\n"
            f"Valutazione: `{path}` sembra la superficie esecutiva principale; con circa {line_count} righe, modifiche a loop, routing o guardrail possono avere effetti ampi.\n"
            f"Rischio concreto: regressioni in `{path}` possono alterare esecuzione tool, condizioni di stop o comportamento delle sessioni oltre alla feature modificata.\n"
            "Miglioramento: mantenere test regressivi mirati sulle decisioni del loop agente prima di cambiare routing, guardrail o sintesi post-tool."
        )

    @staticmethod
    def _rank_codebase_paths(paths: list[str]) -> list[str]:
        priority_names = ("agent.py", "AGENTS.md", "README.md", "REPORT.md", "pyproject.toml", "PROMPTS.md")
        priority_exts = (".py", ".md", ".toml", ".yaml", ".yml", ".json")

        def score(path: str) -> tuple[int, int, str]:
            name_score = next((index for index, name in enumerate(priority_names) if path == name or path.endswith(f"/{name}")), len(priority_names))
            ext_score = 0 if path.endswith(priority_exts) else 1
            return (name_score, ext_score, path)

        seen: set[str] = set()
        unique = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return sorted(unique, key=score)

    def current_status(self) -> TurnStatus:
        return self._build_status(metrics=TurnMetrics())

    def context_pressure(self, pending_user_input: str | None = None) -> BudgetPressure:
        if pending_user_input is not None:
            projected_messages = [*self.messages, {"role": "user", "content": pending_user_input}]
            session_messages = max(0, len(projected_messages) - 1)
            estimated_prompt_tokens = estimate_prompt_tokens(projected_messages)
        else:
            session_messages = max(0, len(self.messages) - 1)
            estimated_prompt_tokens = estimate_prompt_tokens(self.messages)
        model_name = self._active_model_name()
        context_window = None
        if self._model_metadata is not None:
            context_window = self._model_metadata.context_window
        return evaluate_budget_pressure(
            model_name=model_name,
            session_messages=session_messages,
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_window=context_window,
        )

    def _active_model_name(self) -> str | None:
        if self._model_metadata is not None:
            return self._model_metadata.active_model
        if self.client.model:
            return self.client.model
        return None

    def _build_status(
        self,
        *,
        metrics: TurnMetrics,
    ) -> TurnStatus:
        if self._model_metadata is None:
            self._model_metadata = self.client.inspect_model()
        estimated_prompt_tokens = estimate_prompt_tokens(self.messages)
        effective_prompt_tokens = metrics.prompt_tokens if metrics.prompt_tokens is not None else estimated_prompt_tokens
        prefill_tps = self._tokens_per_second(metrics.prompt_tokens, metrics.prompt_eval_duration_ns)
        decode_tps = self._tokens_per_second(metrics.output_tokens, metrics.eval_duration_ns)
        model_elapsed_sec = self._duration_seconds(metrics.total_duration_ns)
        wall_elapsed_sec = self._duration_seconds(metrics.wall_duration_ns)
        tool_elapsed_sec = self._duration_seconds(metrics.tool_duration_ns)
        context_window = self._model_metadata.context_window
        usage_ratio = None
        warning = None
        if context_window and context_window > 0:
            usage_ratio = effective_prompt_tokens / context_window
            if usage_ratio >= CRIT_USAGE_RATIO:
                warning = "critical: context window nearly exhausted"
            elif usage_ratio >= WARN_USAGE_RATIO:
                warning = "warning: context window getting tight"
        if not self.tools_enabled:
            warning = "chat-only: model does not advertise tool support"
        think_state = "no"
        show_thinking_state = "off"
        if self._model_metadata is not None and "thinking" in self._model_metadata.capabilities:
            think_state = "on" if self.think_mode != "off" else "off"
            if think_state == "on" and self.show_thinking:
                show_thinking_state = "on"
        return TurnStatus(
            active_model=self._model_metadata.active_model,
            context_window=context_window,
            session_messages=max(0, len(self.messages) - 1),
            session_turns=sum(1 for message in self.messages if message.get("role") == "user"),
            prompt_tokens=metrics.prompt_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            output_tokens=metrics.output_tokens,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            model_elapsed_sec=model_elapsed_sec,
            wall_elapsed_sec=wall_elapsed_sec,
            tool_elapsed_sec=tool_elapsed_sec,
            usage_ratio=usage_ratio,
            warning=warning,
            think_state=think_state,
            show_thinking_state=show_thinking_state,
        )

    @staticmethod
    def _tokens_per_second(tokens: int | None, duration_ns: int | None) -> float | None:
        if tokens is None or duration_ns is None or tokens <= 0 or duration_ns <= 0:
            return None
        return tokens / (duration_ns / 1_000_000_000)

    @staticmethod
    def _duration_seconds(duration_ns: int | None) -> float | None:
        if duration_ns is None or duration_ns <= 0:
            return None
        return duration_ns / 1_000_000_000

    def _model_confirms_tool_route(
        self,
        *,
        user_input: str,
        route: ToolRoute,
        metrics: TurnMetrics,
        on_event: EventSink | None,
    ) -> bool:
        messages = intent_gate_messages(user_input=user_input, route=route)
        try:
            started_at = time.monotonic_ns()
            response = self.client.chat(
                messages=messages,
                tools=[],
                options={"temperature": 0.0},
                think=False,
            )
            self._update_metrics(metrics, response)
        except Exception:
            self._emit_timing(on_event, "intent-check", started_at, f"{route.intent} -> fail-open")
            return True
        parsed = parse_intent_gate_reply((response.get("message") or {}).get("content"))
        outcome = "unclear, fail-open" if parsed is None else ("YES" if parsed else "NO")
        self._emit_timing(on_event, "intent-check", started_at, f"{route.intent} -> {outcome}")
        return True if parsed is None else parsed

    def _system_prompt(self) -> str:
        if self._prefers_model_first_runtime():
            base_prompt = MODEL_FIRST_SYSTEM_PROMPT
        else:
            base_prompt = BASE_SYSTEM_PROMPT
        model_note = ""
        active_model = self._active_model_name()
        if self._prefers_model_first_runtime() and active_model:
            model_note = (
                f"\nThe current underlying model name is `{active_model}`. "
                "If asked who you are, answer as the assistant naturally without claiming that your personal name is Orbit. "
                "If asked which model you are, distinguish the Orbit CLI from the underlying model name."
            )
        mode_note = ""
        if not self.tools_enabled:
            mode_note = "\nTool calling is disabled for this model. Answer in chat-only mode without using tools."
        if self.skill is None:
            return f"{base_prompt}{model_note}{mode_note}"
        skill_content = self._skill_prompt_content()
        if self.skill.name == DEFAULT_SKILL_REF:
            return f"{base_prompt}{model_note}{mode_note}\n\n{skill_content}"
        return (
            f"{base_prompt}{model_note}{mode_note}\n\n"
            f"Active skill: {self.skill.name}\n\n{skill_content}"
        )

    def _minimal_chat_system_prompt(self) -> str:
        model_note = ""
        active_model = self._active_model_name()
        if self._prefers_model_first_runtime() and active_model:
            model_note = (
                f"\nThe current underlying model name is `{active_model}`. "
                "If asked who you are, answer as the assistant naturally without claiming that your personal name is Orbit. "
                "If asked which model you are, distinguish the Orbit CLI from the underlying model name."
            )
        mode_note = ""
        if not self.tools_enabled:
            mode_note = "\nTool calling is disabled for this model."
        return f"{MINIMAL_CHAT_SYSTEM_PROMPT}{model_note}{mode_note}"

    def _chat_request_messages(self, route: Any | None, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if route is None:
            if self._prefers_model_first_runtime():
                return [self._compact_message_for_model(item) for item in self.messages]
            return self.messages
        if tools:
            if self._prefers_model_first_runtime():
                return [self._compact_message_for_model(item) for item in self.messages]
            return self.messages
        if route.intent not in {"chitchat", "general_knowledge"}:
            if self._prefers_model_first_runtime():
                return [self._compact_message_for_model(item) for item in self.messages]
            return self.messages
        recent_transient_systems = [
            message
            for message in self.messages[1:]
            if message.get("role") == "system" and SUMMARY_MARKER not in str(message.get("content", ""))
        ]
        if recent_transient_systems:
            return self.messages
        history_limit = MINIMAL_CHAT_HISTORY_MESSAGES
        recent_dialogue: list[dict[str, Any]] = []
        for message in reversed(self.messages[1:]):
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            recent_dialogue.append({"role": role, "content": message.get("content", "")})
            if len(recent_dialogue) >= history_limit:
                break
        recent_dialogue.reverse()
        return [{"role": "system", "content": self._minimal_chat_system_prompt()}, *recent_dialogue]

    @staticmethod
    def _update_metrics(metrics: TurnMetrics, response: dict[str, Any]) -> None:
        prompt_eval_count = response.get("prompt_eval_count")
        if isinstance(prompt_eval_count, int) and prompt_eval_count > 0:
            metrics.prompt_tokens = prompt_eval_count
        prompt_duration = response.get("prompt_eval_duration")
        if isinstance(prompt_duration, int) and prompt_duration > 0:
            metrics.prompt_eval_duration_ns = prompt_duration
        eval_count = response.get("eval_count")
        if isinstance(eval_count, int) and eval_count >= 0:
            metrics.output_tokens = eval_count
        eval_duration = response.get("eval_duration")
        if isinstance(eval_duration, int) and eval_duration > 0:
            metrics.eval_duration_ns = eval_duration
        total_duration = response.get("total_duration")
        if isinstance(total_duration, int) and total_duration > 0:
            metrics.total_duration_ns = total_duration

    def _result(self, content: str, metrics: TurnMetrics) -> TurnResult:
        self._prune_transient_system_messages()
        return TurnResult(content=content, status=self._build_status(metrics=metrics))

    def _summarize_explicit_text_evidence(self, *, user_input: str, metrics: TurnMetrics) -> str | None:
        evidence = self._latest_explicit_summary_read_payload()
        if evidence is None:
            return None
        projected_evidence = _compact_tool_payload("read_file", evidence)
        requested_lines = _requested_summary_line_count(user_input)
        line_instruction = (
            f" Return exactly {requested_lines} short lines."
            if requested_lines is not None
            else ""
        )
        system_prompt = (
            "You are the local assistant running inside the Orbit CLI. "
            "Summarize the whole file from bounded sampled chunk notes and document-map evidence only. "
            "Write a real synthesis, not the notes themselves. "
            "Prefer recurring entities, themes, conflicts, progression, and ending signals over isolated fragments. "
            "If evidence is sampled or partial, say so briefly without weakening the summary. "
            "Do not repeat raw excerpts unless the user explicitly asked for quotes."
            f"{line_instruction}"
        )
        requests = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"{user_input}\n\nBounded file evidence:\n{json.dumps(projected_evidence, ensure_ascii=False)}",
            },
        ]
        for attempt in range(2):
            try:
                response = self.client.chat(
                    messages=requests,
                    tools=[],
                    options={"temperature": 0.0},
                    think=False,
                )
            except Exception:
                return None
            self._update_metrics(metrics, response)
            message = response.get("message", {})
            content = str(message.get("content", "")).strip()
            if content and not _looks_like_summary_evidence_echo(content):
                if requested_lines is not None and _summary_line_count(content) != requested_lines:
                    if attempt == 0:
                        requests.append(
                            {
                                "role": "system",
                                "content": (
                                    f"Your previous answer did not have exactly {requested_lines} lines. "
                                    f"Rewrite it now as exactly {requested_lines} short lines, one line per sentence, with no bullets and no extra commentary."
                                    f"{line_instruction}"
                                ),
                            }
                        )
                        continue
                return content
            if attempt == 0:
                requests.append(
                    {
                        "role": "system",
                        "content": (
                            "Do not restate the evidence format. "
                            "Summarize characters, events, goals, conflicts, and major outcomes when apparent."
                            f"{line_instruction}"
                        ),
                    }
                )
        return None

    def _latest_explicit_summary_read_payload(self) -> dict[str, Any] | None:
        for message in reversed(self.messages):
            if message.get("role") != "tool" or message.get("tool_name") != "read_file":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("summary_read"):
                return payload
        return None

    def _refine_compaction_summary(self, plan: CompactionPlan) -> str | None:
        try:
            response = self.client.chat(
                messages=build_hybrid_refinement_messages(plan),
                tools=[],
                options={"temperature": 0.0},
                think=False,
            )
        except Exception:
            return None
        message = response.get("message", {})
        if message.get("tool_calls"):
            return None
        content = message.get("content")
        if not isinstance(content, str):
            return None
        normalized = normalize_model_summary(content)
        if normalized is None:
            return None
        if not _is_acceptable_refined_summary(normalized, plan.fallback_summary):
            return None
        return normalized

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        metrics: TurnMetrics,
        policy_state: TurnPolicyState,
        on_event: EventSink | None = None,
        intent: str | None = None,
        user_input: str = "",
    ) -> None:
        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            name = fn.get("name")
            arguments = parse_arguments(fn.get("arguments"))
            if not name:
                continue
            if name == "fetch_url":
                arguments = self._sanitize_fetch_url_arguments(user_input=user_input, arguments=arguments)
            redundant_markdown_read = markdown_checkbox_redundant_read_prompt(
                skill=self.skill,
                user_input=user_input,
                name=name,
                arguments=arguments,
                messages=self.messages,
            )
            if redundant_markdown_read is not None:
                self._append_system_message(redundant_markdown_read)
                continue
            redundant_listing = self._redundant_codebase_listing_reuse(
                intent=intent,
                name=name,
                arguments=arguments,
                user_input=user_input,
            )
            if redundant_listing is not None:
                message_text, read_path = redundant_listing
                self._append_system_message(message_text)
                if read_path is not None:
                    read_arguments = {"path": read_path}
                    if on_event is not None:
                        on_event(ToolCallEvent(loop=policy_state.loop_count, name="read_file", arguments=read_arguments))
                    read_result, read_elapsed_ns = self._tool_policy.call_tool(
                        registry=self.registry,
                        name="read_file",
                        arguments=read_arguments,
                    )
                    if read_elapsed_ns > 0:
                        metrics.tool_duration_ns += read_elapsed_ns
                    policy_state.tool_steps += 1
                    if on_event is not None:
                        on_event(
                            ToolResultEvent(
                                loop=policy_state.loop_count,
                                name="read_file",
                                ok=bool(read_result.get("ok")),
                                error=read_result.get("error"),
                                returncode=read_result.get("returncode"),
                                stderr=read_result.get("stderr"),
                                stdout=read_result.get("stdout"),
                                elapsed_ms=read_elapsed_ns / 1_000_000,
                            )
                        )
                    self._append_tool_message_with_compaction(
                        {
                            "role": "tool",
                            "tool_name": "read_file",
                            "content": self.registry.encode_tool_result(read_result),
                        },
                        tool_name="read_file",
                        on_event=on_event,
                    )
                continue
            if on_event is not None:
                on_event(ToolCallEvent(loop=policy_state.loop_count, name=name, arguments=arguments))
            result, elapsed_ns = self._tool_policy.call_tool(
                registry=self.registry,
                name=name,
                arguments=arguments,
            )
            if elapsed_ns > 0:
                metrics.tool_duration_ns += elapsed_ns
            policy_state.tool_steps += 1
            if on_event is not None:
                on_event(
                    ToolResultEvent(
                        loop=policy_state.loop_count,
                        name=name,
                        ok=bool(result.get("ok")),
                        error=result.get("error"),
                        returncode=result.get("returncode"),
                        stderr=result.get("stderr"),
                        stdout=result.get("stdout"),
                        elapsed_ms=elapsed_ns / 1_000_000,
                    )
                )
            tool_message = {
                "role": "tool",
                "tool_name": name,
                "content": self.registry.encode_tool_result(result),
            }
            self._append_tool_message_with_compaction(tool_message, tool_name=name, on_event=on_event)
            shell_blocked = (
                name == "bash"
                and result.get("ok") is False
                and isinstance(result.get("error"), str)
                and (
                    "shell operators are not allowed" in str(result.get("error")).lower()
                    or "shell redirection is not allowed" in str(result.get("error")).lower()
                )
            )
            if not result.get("_guarded") and not shell_blocked:
                register_tool_calls(policy_state, [tool_call])
            if shell_blocked:
                self.messages.append(
                    {
                        "role": "system",
                        "content": (
                            "The previous bash command used blocked shell operators or redirection. "
                            "Retry now with separate safe bash calls, one command at a time. "
                            "For machine inspection, prefer small commands such as `uname -srm`, `free -h`, `lscpu`, `nproc`, or `cat /etc/os-release`."
                        ),
                    }
                )
            if result.get("_guarded") and name in {"write_file", "append_file", "replace_in_file"}:
                guarded_path = result.get("path") or arguments.get("path")
                if isinstance(guarded_path, str) and guarded_path.strip():
                    self.messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"The previous write target `{guarded_path}` already exists and has not been read in this session. "
                                f"Read `{guarded_path}` first, then apply a focused update. "
                                "Prefer replace_in_file or append_file over rewriting the whole file when possible."
                            ),
                        }
                    )
                    read_arguments = {"path": guarded_path}
                    if on_event is not None:
                        on_event(ToolCallEvent(loop=policy_state.loop_count, name="read_file", arguments=read_arguments))
                    read_result, read_elapsed_ns = self._tool_policy.call_tool(
                        registry=self.registry,
                        name="read_file",
                        arguments=read_arguments,
                    )
                    if read_elapsed_ns > 0:
                        metrics.tool_duration_ns += read_elapsed_ns
                    policy_state.tool_steps += 1
                    if on_event is not None:
                        on_event(
                            ToolResultEvent(
                                loop=policy_state.loop_count,
                                name="read_file",
                                ok=bool(read_result.get("ok")),
                                error=read_result.get("error"),
                                returncode=read_result.get("returncode"),
                                stderr=read_result.get("stderr"),
                                stdout=read_result.get("stdout"),
                                elapsed_ms=read_elapsed_ns / 1_000_000,
                            )
                        )
                    self._append_tool_message_with_compaction(
                        {
                            "role": "tool",
                            "tool_name": "read_file",
                            "content": self.registry.encode_tool_result(read_result),
                        },
                        tool_name="read_file",
                        on_event=on_event,
                    )
                    if read_result.get("ok"):
                        register_tool_calls(
                            policy_state,
                            [{"function": {"name": "read_file", "arguments": read_arguments}}],
                    )

    def _append_model_first_post_tool_prompt(self, route: ToolRoute, tool_calls: list[dict[str, Any]]) -> None:
        if not tool_calls:
            return
        prompt = self._model_first_post_tool_prompt(route)
        if prompt is None:
            return
        if self.messages and self.messages[-1].get("role") == "system" and self.messages[-1].get("content") == prompt:
            return
        self._append_system_message(prompt)

    def _append_tool_message_with_compaction(
        self,
        tool_message: dict[str, Any],
        *,
        tool_name: str | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        pressure = self._projected_pressure_after_tool_message(tool_message)
        if pressure.level == "hard" and pressure.should_compact:
            changed = self.compact(overflow_tokens=pressure.overflow_tokens)
            if changed and on_event is not None:
                on_event(
                    ToolResultCompactEvent(
                        level=pressure.level,
                        reason=pressure.reason,
                        session_messages=pressure.session_messages,
                        estimated_prompt_tokens=pressure.estimated_prompt_tokens,
                        tool_name=tool_name,
                    )
                )
        self.messages.append(tool_message)

    def _redundant_codebase_listing_reuse(
        self,
        *,
        intent: str | None,
        name: str,
        arguments: dict[str, Any],
        user_input: str,
    ) -> tuple[str, str | None] | None:
        if intent != "codebase_inspection" or name != "list_files":
            return None
        path = str(arguments.get("path") or ".").strip().rstrip("/") or "."
        if path not in {".", "./"}:
            return None
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return None
            if message.get("role") != "tool" or message.get("tool_name") != "list_files":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                continue
            listed_path = str(payload.get("path") or ".").strip().rstrip("/") or "."
            if listed_path == ".":
                paths = self._listed_paths_from_payload(payload, limit=24)
                read_path = None if self._asks_only_for_codebase_file_list(user_input) else self._best_codebase_read_path(paths)
                suffix = f" Existing paths include: {', '.join(paths)}." if paths else ""
                if read_path is not None:
                    suffix += f" Reading `{read_path}` now."
                return (
                    "Skipped a redundant list_files call because a workspace listing is already available in this turn. "
                    "Do not ask the user for the listing output and do not answer yet. "
                    "Reuse these returned paths and call read_file on one concrete source or documentation file before the final assessment."
                    f"{suffix}"
                ), read_path
        return None

    @staticmethod
    def _asks_only_for_codebase_file_list(user_input: str) -> bool:
        lowered = user_input.lower()
        only_hints = ("only", "solo", "soltanto")
        file_hints = ("most relevant files", "most important files", "file più importanti", "file piu importanti")
        return any(hint in lowered for hint in only_hints) and any(hint in lowered for hint in file_hints)

    @staticmethod
    def _listed_paths_from_payload(payload: dict[str, Any], *, limit: int) -> list[str]:
        entries = payload.get("entries")
        if not isinstance(entries, list):
            summary = payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                return [item.strip() for item in summary.split(",") if item.strip()][:limit]
            return []
        paths: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if isinstance(path, str) and path.strip():
                paths.append(path.strip())
                if len(paths) >= limit:
                    break
        return paths

    @staticmethod
    def _best_codebase_read_path(paths: list[str]) -> str | None:
        source_exts = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".ps1", ".sh")
        preferred_names = ("agent.py", "main.py", "app.py", "runtime.py", "cli.py", "README.md", "AGENTS.md")
        for name in preferred_names:
            for path in paths:
                if path == name or path.endswith(f"/{name}"):
                    return path
        for path in paths:
            lowered = path.lower()
            if lowered.endswith(source_exts):
                return path
        for path in paths:
            if path.lower().endswith((".md", ".txt", ".toml", ".json", ".yaml", ".yml")):
                return path
        return None

    def _bootstrap_skill_workspace_docs(
        self,
        *,
        route: Any,
        metrics: TurnMetrics,
        policy_state: TurnPolicyState,
        on_event: EventSink | None = None,
    ) -> None:
        if not should_bootstrap_workspace_docs(self.skill, route.intent):
            return
        workdir = Path(getattr(self.registry, "workdir", Path(".")))
        actions = workspace_doc_bootstrap_actions(self.skill, workdir)
        for name, arguments in actions:
            if on_event is not None:
                on_event(ToolCallEvent(loop=0, name=name, arguments=arguments))
            started_at = time.monotonic_ns()
            result = self.registry.call(name, arguments)
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
            self.messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": self.registry.encode_tool_result(result),
                }
            )
        if actions:
            self._append_system_message(
                "Skill workspace bootstrap completed. Continue with bounded triage and keep AGENTS.md and REPORT.md updated during the analysis."
            )

    def _chat_response(
        self,
        *,
        loop: int,
        tools: list[dict[str, Any]],
        route: Any | None,
        on_event: EventSink | None,
    ) -> dict[str, Any]:
        request_messages = self._chat_request_messages(route, tools)
        request_tools = self._request_tool_definitions(tools)
        return self._perform_chat_request(
            loop=loop,
            request_messages=request_messages,
            request_tools=request_tools,
            on_event=on_event,
        )

    def _perform_chat_request(
        self,
        *,
        loop: int,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        on_event: EventSink | None,
    ) -> dict[str, Any]:
        think = self._resolved_think_value()
        try:
            started_at = time.monotonic_ns()
            if self._should_stream_chat_response():
                response = self._stream_chat_response(
                    loop=loop,
                    think=think,
                    request_messages=request_messages,
                    tools=request_tools,
                    on_event=on_event,
                )
            else:
                response = self.client.chat(
                    messages=request_messages,
                    tools=request_tools,
                    options={"temperature": self.temperature},
                    think=think,
                )
            self._emit_timing(on_event, "model", started_at, f"loop={loop}")
            return response
        except Exception as exc:
            if think not in {None, False} and is_thinking_unsupported_error(exc):
                if on_event is not None:
                    on_event(ThinkingUnavailableEvent(loop=loop, detail=str(exc)))
                started_at = time.monotonic_ns()
                response = self.client.chat(
                    messages=request_messages,
                    tools=request_tools,
                    options={"temperature": self.temperature},
                    think=False,
                )
                self._emit_timing(on_event, "model", started_at, f"loop={loop} think=off")
                return response
            raise

    def _run_explicit_audio_request(
        self,
        *,
        user_input: str,
        metrics: TurnMetrics,
        on_event: EventSink | None,
    ) -> str | None:
        try:
            requests = resolve_explicit_audio_requests(user_input=user_input, workdir=self.registry.workdir)
        except ToolError as exc:
            return str(exc)
        if not requests:
            return None
        if not self._supports_audio():
            return "The current model does not advertise audio support for local audio inspection."
        if len(requests) > 1:
            return "Audio inspection currently supports one local audio file per request."
        request = requests[0]
        try:
            chunks = prepare_audio_chunks(request.full_path)
        except ToolError as exc:
            return str(exc)
        return self._run_audio_chunks_request(
            user_input=user_input,
            label=request.path,
            chunks=chunks,
            metrics=metrics,
            on_event=on_event,
        )

    def _run_audio_chunks_request(
        self,
        *,
        user_input: str,
        label: str,
        chunks: list[AudioChunk],
        metrics: TurnMetrics,
        on_event: EventSink | None,
    ) -> str:
        transcripts: list[str] = []
        for chunk in chunks:
            request_messages = [
                {"role": "system", "content": AUDIO_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Transcribe audio chunk {chunk.index} from {label}. "
                        f"Time range: {chunk.start_seconds:.1f}s-{chunk.start_seconds + chunk.duration_seconds:.1f}s. "
                        "Return only the spoken words if clear; otherwise return unclear."
                    ),
                    "images": [chunk.payload_base64],
                },
            ]
            try:
                response = self._perform_vision_chat_with_retry(
                    loop=chunk.index,
                    request_messages=request_messages,
                    on_event=on_event,
                )
            except Exception as exc:
                return str(exc)
            self._update_metrics(metrics, response)
            message = response.get("message", {})
            content = str(message.get("content", "")).strip() or "unclear"
            transcripts.append(f"[{chunk.start_seconds:.1f}-{chunk.start_seconds + chunk.duration_seconds:.1f}s] {content}")
        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's audio request using only these chunk transcripts. "
                    "If the user asked for transcription, provide a cleaned transcript. "
                    "If the user asked for a summary, summarize the transcript. "
                    "Do not claim that audio was unavailable."
                ),
            },
            {
                "role": "user",
                "content": f"{user_input}\n\nAudio chunk transcripts for {label}:\n" + "\n".join(transcripts),
            },
        ]
        try:
            response = self._perform_vision_chat_with_retry(
                loop=len(chunks) + 1,
                request_messages=synthesis_messages,
                on_event=on_event,
            )
        except Exception as exc:
            return str(exc)
        self._update_metrics(metrics, response)
        message = response.get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            content = "\n".join(transcripts)
        self.messages.append({"role": "assistant", "content": content})
        return content

    def _run_explicit_image_request(
        self,
        *,
        user_input: str,
        metrics: TurnMetrics,
        on_event: EventSink | None,
        image_attachments: list[ExplicitImageRequest] | None = None,
    ) -> str | None:
        try:
            requests = self._resolve_image_requests(user_input=user_input, image_attachments=image_attachments)
        except ToolError as exc:
            return str(exc)
        if not requests:
            return None
        if not self._supports_vision():
            return "The current model does not advertise vision support for local image inspection."
        encoded_images: list[str] = []
        labels: list[str] = []
        try:
            for request in requests:
                encoded_images.append(encode_image_base64(request.full_path))
                labels.append(request.path)
        except ToolError as exc:
            return str(exc)
        if len(encoded_images) > 1:
            return self._run_multi_image_request(
                user_input=user_input,
                labels=labels,
                encoded_images=encoded_images,
                metrics=metrics,
                on_event=on_event,
            )
        request_messages = self._build_vision_messages(
            user_input=user_input,
            labels=labels,
            encoded_images=encoded_images,
        )
        for attempt in range(2):
            if on_event is not None:
                on_event(ModelRequestEvent(loop=attempt + 1))
            try:
                response = self._perform_chat_request(
                    loop=attempt + 1,
                    request_messages=request_messages,
                    request_tools=[],
                    on_event=on_event,
                )
            except Exception as exc:
                return str(exc)
            self._update_metrics(metrics, response)
            message = response.get("message", {})
            content = str(message.get("content", "")).strip()
            if content:
                self.messages.append({"role": "assistant", "content": content})
                return content
            if attempt == 0:
                request_messages.append(
                    {
                        "role": "system",
                        "content": "Reply briefly from the attached image evidence only.",
                    }
                )
        return "The model returned an empty reply for the requested image."

    def _run_multi_image_request(
        self,
        *,
        user_input: str,
        labels: list[str],
        encoded_images: list[str],
        metrics: TurnMetrics,
        on_event: EventSink | None,
    ) -> str:
        descriptions: list[str] = []
        for index, (label, encoded) in enumerate(zip(labels, encoded_images, strict=False), start=1):
            request_messages = [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Inspect Image {index}: {label}. "
                        "Return exactly two short lines: "
                        "Visible text: transcribe any readable text, or say none/unclear. "
                        "Subject: describe the main visible subject."
                    ),
                    "images": [encoded],
                },
            ]
            try:
                response = self._perform_vision_chat_with_retry(
                    loop=index,
                    request_messages=request_messages,
                    on_event=on_event,
                )
            except Exception as exc:
                return str(exc)
            self._update_metrics(metrics, response)
            message = response.get("message", {})
            content = str(message.get("content", "")).strip()
            if not content:
                content = "No description returned."
            descriptions.append(f"Image {index} ({label}): {content}")
        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's multi-image request using only these per-image descriptions. "
                    "Do not claim that images were unavailable. Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": f"{user_input}\n\nPer-image evidence:\n" + "\n".join(descriptions),
            },
        ]
        try:
            response = self._perform_vision_chat_with_retry(
                loop=len(descriptions) + 1,
                request_messages=synthesis_messages,
                on_event=on_event,
            )
        except Exception as exc:
            return str(exc)
        self._update_metrics(metrics, response)
        message = response.get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            content = "\n".join(descriptions)
        self.messages.append({"role": "assistant", "content": content})
        return content

    def _perform_vision_chat_with_retry(
        self,
        *,
        loop: int,
        request_messages: list[dict[str, Any]],
        on_event: EventSink | None,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return self._perform_chat_request(
                    loop=loop,
                    request_messages=request_messages,
                    request_tools=[],
                    on_event=on_event,
                )
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(1.0)
                    continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("vision request failed")

    @staticmethod
    def _build_vision_messages(
        *,
        user_input: str,
        labels: list[str],
        encoded_images: list[str],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": VISION_SYSTEM_PROMPT}]
        if len(encoded_images) <= 1:
            label_text = labels[0] if labels else "image"
            messages.append(
                {
                    "role": "user",
                    "content": f"{user_input}\n\nAttached image: Image 1 = {label_text}",
                    "images": encoded_images,
                }
            )
            return messages
        for index, (label, encoded) in enumerate(zip(labels, encoded_images, strict=False), start=1):
            messages.append(
                {
                    "role": "user",
                    "content": f"Image {index}: {label}",
                    "images": [encoded],
                }
            )
        mapping = ", ".join(f"Image {index} = {label}" for index, label in enumerate(labels, start=1))
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{user_input}\n\n"
                    f"Attached images are distinct and ordered as: {mapping}. "
                    "Compare or describe them by these image numbers and file names."
                ),
            }
        )
        return messages

    def _resolve_image_requests(
        self,
        *,
        user_input: str,
        image_attachments: list[ExplicitImageRequest] | None,
    ) -> list[ExplicitImageRequest]:
        workdir = Path(getattr(self.registry, "workdir", Path(".")))
        requests: list[ExplicitImageRequest] = []
        if image_attachments:
            return list(image_attachments)
        requests = resolve_explicit_image_requests(
            user_input=user_input,
            workdir=workdir,
        )
        if requests:
            return requests
        return requests

    def _stream_chat_response(
        self,
        *,
        loop: int,
        tools: list[dict[str, Any]],
        think: bool | str,
        request_messages: list[dict[str, Any]],
        on_event: EventSink | None,
    ) -> dict[str, Any]:
        stream = self.client.chat(
            messages=request_messages,
            tools=tools,
            options={"temperature": self.temperature},
            think=think,
            stream=True,
        )
        thinking_started = False
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls = None
        last_chunk: dict[str, Any] = {}
        for chunk in stream:
            last_chunk = chunk
            message = chunk.get("message") or {}
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking:
                thinking_parts.append(thinking)
                if on_event is not None:
                    if not thinking_started:
                        thinking_started = True
                        on_event(ThinkingStartEvent(loop=loop))
                    on_event(ThinkingChunkEvent(loop=loop, text=thinking))
            content = message.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
            chunk_tool_calls = message.get("tool_calls")
            if chunk_tool_calls:
                tool_calls = chunk_tool_calls
        if thinking_started and on_event is not None:
            on_event(ThinkingEndEvent(loop=loop))
        merged = dict(last_chunk)
        last_message = dict(merged.get("message") or {})
        last_message["content"] = "".join(content_parts)
        if thinking_parts:
            last_message["thinking"] = "".join(thinking_parts)
        if tool_calls:
            last_message["tool_calls"] = tool_calls
        merged["message"] = last_message
        return merged

    def _resolved_think_value(self) -> bool | str | None:
        if self.think_mode == "on":
            return True
        if self.think_mode == "off":
            return False
        return None

    def _should_stream_chat_response(self) -> bool:
        if self.show_thinking and self.think_mode != "off":
            return True
        return False

    def _prefers_model_first_runtime(self) -> bool:
        active_model = (self._active_model_name() or "").lower()
        return active_model.startswith("gemma4:")

    def _supports_vision(self) -> bool:
        if self._model_metadata is None:
            self._model_metadata = self.client.inspect_model()
        return "vision" in self._model_metadata.capabilities

    def _supports_audio(self) -> bool:
        if self._model_metadata is None:
            self._model_metadata = self.client.inspect_model()
        return "audio" in self._model_metadata.capabilities

    def _skill_prompt_content(self) -> str:
        if self.skill is None:
            return ""
        return self.skill.content

    def _request_tool_definitions(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._prefers_model_first_runtime():
            return [self._compact_tool_definition(item) for item in tools]
        return tools

    def _compact_tool_definition(self, tool: dict[str, Any]) -> dict[str, Any]:
        return self._model_payloads.compact_tool_definition(tool)

    def _compact_message_for_model(self, message: dict[str, Any]) -> dict[str, Any]:
        return self._model_payloads.compact_message_for_model(message)

    def _emit_timing(self, on_event: EventSink | None, phase: str, started_at_ns: int, detail: str | None = None) -> None:
        if not self.debug_timing or on_event is None:
            return
        elapsed_ms = 0.0
        if started_at_ns > 0:
            elapsed_ms = (time.monotonic_ns() - started_at_ns) / 1_000_000
        on_event(DebugTimingEvent(phase=phase, elapsed_ms=elapsed_ms, detail=detail))

    def _model_first_post_tool_prompt(self, route: ToolRoute) -> str | None:
        return model_first_post_tool_prompt(route.intent_class, self._latest_user_text())

    def _latest_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def _append_system_message(self, content: str, *, transient: bool = True) -> None:
        message = {"role": "system", "content": content}
        if transient:
            message[TRANSIENT_SYSTEM_FLAG] = True
        self.messages.append(message)

    def _prune_transient_system_messages(self) -> None:
        self.messages = [
            message
            for message in self.messages
            if not (message.get("role") == "system" and message.get(TRANSIENT_SYSTEM_FLAG))
        ]

    def _projected_pressure_after_tool_message(self, tool_message: dict[str, Any]) -> BudgetPressure:
        projected_messages = [*self.messages, tool_message]
        estimated_prompt_tokens = estimate_prompt_tokens(projected_messages) + self._tool_append_response_reserve_tokens()
        return evaluate_budget_pressure(
            model_name=self._active_model_name(),
            session_messages=max(0, len(projected_messages) - 1),
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_window=self._model_metadata.context_window if self._model_metadata is not None else None,
        )

    def _tool_append_response_reserve_tokens(self) -> int:
        context_window = self._model_metadata.context_window if self._model_metadata is not None else None
        if isinstance(context_window, int) and context_window > 0:
            return max(TOOL_APPEND_MIN_RESPONSE_TOKENS, int(context_window * TOOL_APPEND_RESPONSE_CTX_RATIO))
        return TOOL_APPEND_MIN_RESPONSE_TOKENS


def _is_acceptable_refined_summary(candidate: str, fallback: str) -> bool:
    lowered = candidate.lower()
    if "working memory:" not in lowered or "durable memory:" not in lowered:
        return False
    if len(fallback) >= 400 and len(candidate) < int(len(fallback) * 0.35):
        return False
    fallback_paths = _summary_path_markers(fallback)
    if not fallback_paths:
        return True
    candidate_paths = _summary_path_markers(candidate)
    retained = sum(1 for path in fallback_paths if path in candidate_paths)
    minimum_retained = min(2, len(fallback_paths))
    return retained >= minimum_retained


def _summary_path_markers(value: str) -> set[str]:
    markers = set()
    for match in re.findall(r"(?:path=)?([A-Za-z0-9_./-]+\.[A-Za-z0-9_+-]+)", value):
        markers.add(match.lower())
    return markers

def _looks_like_summary_evidence_echo(content: str) -> bool:
    lowered = content.lower()
    if "sampled file evidence" in lowered:
        return True
    if "chunk_notes" in lowered:
        return True
    line_matches = re.findall(r"lines?\s+\d+\s*-\s*\d+", lowered)
    return len(line_matches) >= 2 and "focus:" in lowered


def _requested_summary_line_count(user_input: str) -> int | None:
    match = re.search(r"\b(?P<count>\d{1,2})\s+(?:short\s+)?(?:lines?|righe?)\b", user_input, flags=re.IGNORECASE)
    if match is None:
        return None
    return max(1, min(12, int(match.group("count"))))


def _summary_line_count(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip())
