from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Mapping

from orbit.backend.payloads import ARTIFACT_CONTENT_PROTOCOL_ID, ARTIFACT_CONTENT_PROTOCOL_VERSION
from orbit.final_prefix_config import resolve_final_prefix_reuse
from orbit.native_llama.capabilities import safe_native_capability_manifest
from orbit.native_llama.chat_template import render_gemma4_route_prompt_segments
from orbit.native_llama.client import (
    NativeClientConfig,
    NativeLlamaClient,
    NativeRoutePrefixPrefillResult,
    _has_open_thought_channel,
)
from orbit.native_llama.kv_diag import emit_route_prefix_prewarm_event, request_context as native_kv_request_context
from orbit.native_server.server_calibration import (
    WARMUP_REJECTION,
    calibrate_threads,
    restore_threads,
)
from orbit.native_server.server_profile import (
    thread_candidates,
    QualifiedStartupProfile,
    Resolution,
    detect_topology,
    qualified_startup_profile,
    render_profile_lines,
    resolve_cpu_repack,
    resolve_profile,
)
from orbit.native_llama.download_cli import _DownloadProgress as DownloadProgress
from orbit.native_llama.model_discovery import (
    ModelDiscoveryRow,
    NativeProfileInspector,
    discover_models,
    format_model_discovery,
    paint_model_status,
)
from orbit.terminal.theme import supports_ansi
from orbit.native_llama.model_download import download_model, huggingface_resolve_url, parse_huggingface_spec
from orbit.native_llama.model_store import download_advisory
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID
from orbit.native_llama.model_registry import default_hf_cache, effective_models_dir, get_manifest, local_model_path
from orbit.native_llama.paths import (
    DEFAULT_LLAMA_ROOT,
    DEFAULT_MODEL_ID,
    NativeLlamaPaths,
    _resolve_native_runtime,
    resolve_legacy_paths,
    resolve_paths,
)
from orbit.native_llama.prefix_anchor import prefix_anchor_enabled
from orbit.native_llama.qwen_route_prefix import resolve_qwen_route_prefix_reuse
from orbit.native_llama.qwen36_shell_tool_prefix import (
    exact_qwen36_shell_tool_schema,
    resolve_qwen36_shell_tool_prefix_reuse,
)
from orbit.native_llama.ornith_analysis_prefix import (
    resolve_ornith_analysis_prefix_prewarm,
    resolve_ornith_analysis_prefix_reuse,
)
from orbit.native_llama.ornith_route_prefix import resolve_ornith_route_prefix_reuse
from orbit.native_llama.qwen3_coder_route_prefix import resolve_qwen3_coder_route_prefix_reuse
from orbit.native_server.protocol import (
    ContinueRequest,
    DEFAULT_SESSION_ID,
    ChatRequest,
    parse_continue_request,
    native_chat_response,
    openai_chat_response,
    parse_chat_request,
    sse_data,
    sse_event,
    trim_at_stop,
    validate_session_id,
)
from orbit.runtime.analysis_runtime import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_TOOL_SCHEMA
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT
from orbit.runtime.tool_healing import tool_call_healing_status


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 12120
PREFIX_PREWARM_ENV = "ORBIT_KV_PREFIX_PREWARM"
PREFIX_PREWARM_OFF = "off"
PREFIX_PREWARM_STARTUP = "startup"
TOOLS_ENV = "ORBIT_TOOLS"


def _progress_payload(progress: Any, *, session_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "current": progress.current,
        "total": progress.total,
        "percent": progress.percent,
        "session_id": session_id,
    }
    for field in (
        "evaluated_current",
        "evaluated_total",
        "cached_tokens",
        "elapsed_seconds",
        "tokens_per_second",
    ):
        value = getattr(progress, field, None)
        if value is not None:
            payload[field] = value
    return payload


class OrbitNativeServer:
    def __init__(self, *, client: NativeLlamaClient, model_alias: str) -> None:
        self.client = client
        self.model_alias = model_alias
        self.lock = threading.Lock()
        # Requests waiting for (or holding) the model. The post-final route
        # shadow runs between requests and yields to the first one that
        # arrives: it never starts while one waits, and it stops at the next
        # prompt batch when one does.
        self._waiters = 0
        self._waiters_lock = threading.Lock()
        self._route_shadow_thread: threading.Thread | None = None
        self._route_shadow_serial = 0
        self._route_shadow_stopping = False
        self.last_route_shadow: dict[str, Any] | None = None
        self.native_backend_capabilities = safe_native_capability_manifest(
            client,
            final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT,
        )

    @contextlib.contextmanager
    def _model_lock(self):
        """Hold the model for a request, announcing the wait first.

        The waiter count is what preempts a running route shadow, so it is
        raised BEFORE blocking on the lock and lowered only once the lock is
        released: a shadow that starts while a request is on its way would
        otherwise cost that request up to one whole shadow.
        """
        guard = self._waiters_guard()
        with guard:
            self._waiters = self.__dict__.get("_waiters", 0) + 1
        try:
            with self.lock:
                yield
        finally:
            with guard:
                self._waiters -= 1

    def _waiters_guard(self) -> threading.Lock:
        # Created on demand: a server built without `__init__` (several tests
        # do) must still take the model lock exactly as before.
        guard = self.__dict__.get("_waiters_lock")
        if guard is None:
            guard = threading.Lock()
            self._waiters_lock = guard
        return guard

    def _requests_waiting(self) -> bool:
        return self.__dict__.get("_waiters", 0) > 0

    def _schedule_route_shadow(self, request: ChatRequest, *, content: str, finish_reason: str, stopped: bool, tool_calls) -> bool:
        """Advance the route checkpoint past this reply, after it is delivered.

        Only for a call the runtime declared as extending the route history,
        and only when the reply is exactly what the runtime will commit: it
        stopped on its own (not cancelled, not cut by the budget, not by a
        stop sequence, not empty) and carries no tool call. The work runs on
        its own thread and takes the model lock only after this request has
        released it, so it never delays the completion; the handler writes
        the response while the shadow may already be decoding. The next
        request preempts it at a prompt-batch boundary. A later shadow
        supersedes an earlier one that has not run yet: a final followed by
        its retry commits the retry's reply, never the first.
        """
        if not request.route_history_continuation:
            return False
        if finish_reason != "stop" or stopped or tool_calls or not content.strip():
            return False
        advance = getattr(self.client, "advance_route_checkpoint_after_final", None)
        if not callable(advance):
            return False
        with self._waiters_guard():
            self._route_shadow_serial += 1
            serial = self._route_shadow_serial
        thread = threading.Thread(
            target=self._run_route_shadow,
            args=(advance, content, serial),
            name="orbit-route-shadow",
            daemon=True,
        )
        self._route_shadow_thread = thread
        thread.start()
        return True

    def _run_route_shadow(self, advance, content: str, serial: int) -> None:
        with self.lock:
            if self.__dict__.get("_route_shadow_stopping"):
                self.last_route_shadow = {"status": "skipped", "reason": "stopping"}
                return
            if serial != self._route_shadow_serial:
                self.last_route_shadow = {"status": "skipped", "reason": "superseded"}
                return
            if self._requests_waiting():
                self.last_route_shadow = {"status": "skipped", "reason": "request_waiting"}
                return
            try:
                with native_kv_request_context(
                    endpoint="/route-shadow",
                    payload={"_orbit_kv_phase": "route_shadow", "_orbit_kv_tools_mode": "on"},
                ):
                    self.last_route_shadow = advance(content, should_cancel=self._requests_waiting)
            except Exception as exc:  # never let a shadow take the server down
                self.last_route_shadow = {"status": "skipped", "reason": "error", "error": str(exc)}

    def stop_route_shadow(self, timeout: float | None = 30.0) -> None:
        """Cancel a running shadow and wait for it: the context is about to go.

        A shadow still waiting for the lock behind an in-flight request must
        not start once that request drains, so the stop is also a flag it
        checks before doing anything.
        """
        self._route_shadow_stopping = True
        thread = self._route_shadow_thread
        if thread is None or not thread.is_alive():
            return
        cancel = getattr(self.client, "cancel", None)
        if callable(cancel):
            cancel()
        thread.join(timeout)

    def wait_for_route_shadow(self, timeout: float | None = None) -> None:
        """Join the last scheduled shadow, for tests and diagnostics."""
        thread = self._route_shadow_thread
        if thread is not None:
            thread.join(timeout)

    def chat(self, payload: dict[str, Any], *, on_token=None, on_progress=None, should_cancel=None) -> dict[str, Any]:
        request = parse_chat_request(payload)
        validate_session_id(request.session_id)
        return self.complete(request, on_token=on_token, on_progress=on_progress, should_cancel=should_cancel)

    def moe_expert_usage_status(self) -> dict[str, object]:
        with self._model_lock():
            return self.client.moe_expert_usage_status()

    def reset_moe_expert_usage(self) -> dict[str, object]:
        with self._model_lock():
            return self.client.reset_moe_expert_usage()

    def reset_session(self) -> dict[str, object]:
        with self._model_lock():
            profile = getattr(self.client, "model_profile", None)
            preserve_coder_checkpoint = bool(
                getattr(profile, "verified", False)
                and getattr(profile, "profile_id", None) == QWEN3_CODER_PROFILE_ID
            )
            self.client.reset_session_state(
                preserve_qwen3_coder_route_checkpoint=preserve_coder_checkpoint
            )
            snapshot = self.client.session_snapshot(DEFAULT_SESSION_ID)
            return {
                "status": "reset",
                "session_id": snapshot.session_id,
                "cached_tokens": snapshot.cached_tokens,
                "in_flight": snapshot.in_flight,
                "qwen3_coder_checkpoint_preserved": preserve_coder_checkpoint,
            }

    def validate_request(self, request: ChatRequest) -> None:
        thinking = self.client.config.thinking if request.thinking is None else request.thinking
        if request.artifact_content and (request.tools or thinking or request.stop):
            raise ValueError(
                "artifact content generation requires tools, thinking, and stop sequences to be disabled"
            )

    def complete(self, request: ChatRequest, *, on_token=None, on_progress=None, should_cancel=None) -> dict[str, Any]:
        self.validate_request(request)
        parts: list[str] = []

        def collect(text: str) -> None:
            parts.append(text)
            if on_token:
                on_token(text)

        with self._model_lock():
            thinking = self.client.config.thinking if request.thinking is None else request.thinking
            final_prefix_experiment = request.final_prefix_experiment and _is_final_from_tool_prompt(request.messages)
            qwen_route_prefix_anchor = (
                request.qwen_route_prefix_anchor
                and not thinking
                and not request.tools
                and _is_qwen_route_prompt(request.messages)
            )
            qwen36_shell_tool_prefix_anchor = (
                request.qwen36_shell_tool_prefix_anchor
                and not thinking
                and exact_qwen36_shell_tool_schema(request.tools)
            )
            if request.artifact_content:
                completion = self.client.complete_artifact_text(
                    request.messages,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                    on_progress=on_progress,
                    on_token=collect,
                    should_cancel=should_cancel,
                )
            else:
                completion = self.client.complete_chat_text(
                    request.messages,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                    tools=request.tools,
                    thinking=thinking,
                    route_prefix_anchor=request.route_prefix_anchor,
                    analysis_rolling_anchor=request.analysis_rolling_anchor,
                    analysis_step_anchor=request.analysis_step_anchor,
                    qwen_route_prefix_anchor=qwen_route_prefix_anchor,
                    qwen36_shell_tool_prefix_anchor=qwen36_shell_tool_prefix_anchor,
                    allow_mtp_experimental=request.allow_mtp_experimental,
                    final_prefix_experiment=final_prefix_experiment,
                    on_progress=on_progress,
                    on_token=collect,
                    should_cancel=should_cancel,
                )
        timings = completion.timings
        content, stopped = trim_at_stop(completion.content, request.stop)
        stopped = stopped or completion.stopped_by_stop
        open_thought = thinking and _has_open_thought_channel(completion.content)
        finish_reason = "cancelled" if timings.cancelled else "stop"
        if not timings.cancelled and not content.strip():
            finish_reason = "empty_response"
        if stopped:
            finish_reason = "stop"
        if completion.completed_after_thought and not timings.cancelled:
            finish_reason = "stop"
        elif open_thought and not timings.cancelled:
            finish_reason = "length"
        elif timings.output_tokens >= request.max_tokens and not timings.cancelled and not stopped:
            finish_reason = "length"
        tool_calls = getattr(completion, "tool_calls", ())
        # A structurally parsed call must not turn a budget-truncated generation
        # into an executable completion. The canonical gate remains authoritative
        # after this transport-level finish reason is preserved.
        if tool_calls and finish_reason not in {"cancelled", "length"}:
            finish_reason = "tool_calls"
        self._schedule_route_shadow(
            request, content=content, finish_reason=finish_reason, stopped=stopped, tool_calls=tool_calls
        )
        return native_chat_response(
            content=content,
            model=self.model_alias,
            finish_reason=finish_reason,
            session_id=request.session_id,
            prompt_tokens=timings.prompt_tokens,
            completion_tokens=timings.output_tokens,
            reused_prompt_tokens=timings.reused_prompt_tokens,
            evaluated_prompt_tokens=timings.evaluated_prompt_tokens,
            prefill_ms=timings.prefill_ms,
            generation_ms=timings.generation_ms,
            cancelled=timings.cancelled and not stopped,
            backend_ttft_ms=getattr(timings, "backend_ttft_ms", None),
            reasoning_content=getattr(completion, "reasoning_content", ""),
            reasoning_tokens=getattr(completion, "reasoning_tokens", 0),
            tool_calls=tool_calls,
        )

    def continue_current(self, request: ContinueRequest, *, on_token=None, on_progress=None, should_cancel=None) -> dict[str, Any]:
        with self._model_lock():
            thinking = self.client.config.thinking if request.thinking is None else request.thinking
            completion = self.client.continue_chat_text_current_context(
                max_tokens=request.max_tokens,
                stop=request.stop,
                thinking=thinking,
                on_progress=on_progress,
                on_token=on_token,
                should_cancel=should_cancel,
            )
        timings = completion.timings
        content, stopped = trim_at_stop(completion.content, request.stop)
        stopped = stopped or completion.stopped_by_stop
        open_thought = thinking and _has_open_thought_channel(completion.content)
        finish_reason = "cancelled" if timings.cancelled else "stop"
        if not timings.cancelled and not content.strip():
            finish_reason = "empty_response"
        if stopped:
            finish_reason = "stop"
        elif open_thought and not timings.cancelled:
            finish_reason = "length"
        elif timings.output_tokens >= request.max_tokens and not timings.cancelled and not stopped:
            finish_reason = "length"
        return native_chat_response(
            content=content,
            model=self.model_alias,
            finish_reason=finish_reason,
            session_id=DEFAULT_SESSION_ID,
            prompt_tokens=timings.prompt_tokens,
            completion_tokens=timings.output_tokens,
            reused_prompt_tokens=timings.reused_prompt_tokens,
            evaluated_prompt_tokens=timings.evaluated_prompt_tokens,
            prefill_ms=timings.prefill_ms,
            generation_ms=timings.generation_ms,
            cancelled=timings.cancelled and not stopped,
            backend_ttft_ms=getattr(timings, "backend_ttft_ms", None),
        )

    def cancel(self, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
        validate_session_id(session_id)
        self.client.cancel()
        return {"status": "cancel_requested", "session_id": session_id}

    def session_info(self, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
        validate_session_id(session_id)
        snapshot = self.client.session_snapshot(session_id)
        return {
            "id": snapshot.session_id,
            "active": True,
            "backend_mode": snapshot.backend_mode,
            "thinking_mode": "on" if self.client.config.thinking else "off",
            "cached_tokens": snapshot.cached_tokens,
            "in_flight": snapshot.in_flight,
            "cancel_requested": snapshot.cancel_requested,
            "mtp_enabled": snapshot.mtp_enabled,
            "mtp_initialized": snapshot.mtp_initialized,
            "mtp_failure_reason": snapshot.mtp_failure_reason,
        }

    def runtime_info(self) -> dict[str, Any]:
        # `threads` is the resolved profile; `native_threads` is the live
        # context read back through llama.cpp. They are published side by
        # side so a mismatch is visible from `/props` instead of from a
        # benchmark that came out slow. Read without `self.lock` on purpose:
        # `/props` is polled while a completion holds that lock for the whole
        # decode, and the read is a plain field on the context. The only race
        # is against `close()` at shutdown, which is narrowed rather than
        # eliminated: `close()` publishes a None handle before it frees the
        # context, so a reader that already passed the handle check has a
        # window of one call. Accepted for a diagnostic on a dying process.
        native = _native_thread_counts(self.client)
        return {
            "threads": self.client.config.threads,
            "threads_batch": self.client.config.threads_batch,
            "native_threads": native[0] if native else None,
            "native_threads_batch": native[1] if native else None,
            "ctx_size": self.client.config.context_tokens,
            "batch_size": self.client.config.batch_size,
            "ubatch_size": self.client.config.ubatch_size,
            "parallel_slots": 1,
            "thinking_mode": "on" if self.client.config.thinking else "off",
        }

    def count_text_tokens(self, text: str) -> dict[str, int]:
        return {
            "tokens": self.client.count_text_tokens(text),
            "context_tokens": self.client.config.context_tokens,
        }

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        thinking: bool,
    ) -> dict[str, int | str]:
        # The Qwen bridge keeps the parser associated with its latest render.
        # Serialize inspection with completion so concurrent token accounting
        # cannot replace parser state while generated text is being decoded.
        with self._model_lock():
            tokens, rendered_hash, token_hash = self.client.inspect_chat_tokens(
                messages,
                tools=tools,
                thinking=thinking,
            )
        return {
            "tokens": tokens,
            "context_tokens": self.client.config.context_tokens,
            "rendered_hash": rendered_hash,
            "token_hash": token_hash,
        }

    def count_artifact_content_tokens(self, messages: list[dict[str, Any]]) -> dict[str, int | str]:
        with self._model_lock():
            tokens, rendered_hash, token_hash = self.client.inspect_artifact_content_tokens(messages)
        return {
            "tokens": tokens,
            "context_tokens": self.client.config.context_tokens,
            "rendered_hash": rendered_hash,
            "token_hash": token_hash,
        }

    def error_result(self, message: str, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = DEFAULT_SESSION_ID
        try:
            request = parse_chat_request(payload)
            session_id = request.session_id
        except ValueError:
            raw_session_id = payload.get("session_id")
            if isinstance(raw_session_id, str) and raw_session_id.strip():
                session_id = raw_session_id.strip()
        return native_chat_response(
            content=f"error: {message}",
            model=self.model_alias,
            finish_reason="error",
            session_id=session_id,
            prompt_tokens=0,
            completion_tokens=0,
            reused_prompt_tokens=0,
            evaluated_prompt_tokens=0,
            prefill_ms=0.0,
            generation_ms=0.0,
            cancelled=False,
        )


class OrbitNativeHandler(BaseHTTPRequestHandler):
    server_version = "orbit-server"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok"})
            return
        if self.path == "/v1/models":
            state = self._state()
            capabilities = ["completion"]
            if state.client.supports_vision or state.client.supports_audio:
                capabilities.append("multimodal")
            self._json({"object": "list", "data": [{"id": state.model_alias, "object": "model", "capabilities": capabilities}]})
            return
        if self.path == "/diagnostics/expert-usage":
            self._json(self._state().moe_expert_usage_status())
            return
        if self.path == "/props":
            state = self._state()
            session = state.session_info()
            runtime = state.runtime_info()
            mtp_last_completion = _mtp_last_completion_payload(state.client)
            mtp_last_timing = _mtp_last_timing_payload(state.client)
            mtp_last_validate_efficiency = _mtp_last_validate_efficiency_payload(state.client)
            mtp_last_validate_equivalence = _mtp_last_validate_equivalence_payload(state.client)
            mtp_config = _mtp_config_payload(state.client)
            final_prefix = state.client.final_prefix_experiment_status()
            qwen_route_prefix = state.client.qwen_route_prefix_reuse_status()
            qwen36_shell_tool_prefix = state.client.qwen36_shell_tool_prefix_reuse_status()
            qwen3_coder_route_prefix = state.client.qwen3_coder_route_prefix_reuse_status()
            ornith_route_prefix = state.client.ornith_route_prefix_reuse_status()
            final_prefix_config = _final_prefix_reuse_props(state.client)
            route_shadow = state.last_route_shadow
            self._json(
                {
                    "model_path": str(state.client.paths.model),
                    "mmproj_path": str(state.client.paths.mmproj_model) if state.client.paths.mmproj_model else None,
                    "draft_model_path": str(state.client.paths.draft_mtp_model) if state.client.paths.draft_mtp_model else None,
                    "multimodal_available": state.client.paths.multimodal_available,
                    "multimodal_fallback_reason": state.client.paths.multimodal_fallback_reason,
                    "supports_vision": state.client.supports_vision,
                    "supports_audio": state.client.supports_audio,
                    # "mtp_available" means an EXTERNAL DRAFT MODEL is present
                    # (`draft_path is not None`). Several internal callers gate
                    # on exactly that, so its meaning is preserved rather than
                    # widened. It is therefore false for a qualified single-GGUF
                    # self-MTP artifact, which reads as "MTP never ran" beside a
                    # successful self-MTP completion -- hence the field below.
                    "mtp_available": state.client.paths.mtp_available,
                    # Whether THIS session actually constructed a single-GGUF
                    # self-MTP runtime. Observed state, not a capability probe:
                    # deciding capability means hashing a 20 GiB artifact, which
                    # must never happen per /props request. False here therefore
                    # means "not running self-MTP now", which for an un-requested
                    # or failed session is the honest answer.
                    "self_mtp_active": bool(
                        getattr(state.client._persistent_mtp_runtime, "self_mtp", False)
                    ),
                    "fallback_reason": state.client.paths.fallback_reason,
                    "mtp_probe_enabled": state.client.mtp_probe.enabled,
                    "mtp_probe_initialized": state.client.mtp_probe.initialized,
                    "mtp_probe_error": state.client.mtp_probe.error,
                    "mtp_dry_run_enabled": state.client.mtp_dry_run.enabled,
                    "mtp_dry_run_success": state.client.mtp_dry_run.success,
                    "mtp_draft_tokens": state.client.mtp_dry_run.draft_tokens,
                    "mtp_dry_run_error": state.client.mtp_dry_run.error,
                    "mtp_accept_probe_enabled": state.client.mtp_accept_probe.enabled,
                    "mtp_accept_probe_success": state.client.mtp_accept_probe.success,
                    "mtp_accept_probe_draft_tokens": state.client.mtp_accept_probe.draft_tokens,
                    "mtp_accept_probe_accepted_tokens": state.client.mtp_accept_probe.accepted_tokens,
                    "mtp_accept_probe_error": state.client.mtp_accept_probe.error,
                    "mtp_decode_probe_enabled": state.client.mtp_decode_probe.enabled,
                    "mtp_decode_probe_success": state.client.mtp_decode_probe.success,
                    "mtp_decode_probe_error": state.client.mtp_decode_probe.error,
                    "mtp_experimental_enabled": state.client.config.use_mtp_experimental,
                    "mtp_last_completion_success": state.client.last_mtp_completion.success,
                    "mtp_last_completion": mtp_last_completion,
                    "mtp_last_timing": mtp_last_timing,
                    "mtp_last_validate_efficiency": mtp_last_validate_efficiency,
                    "mtp_last_validate_equivalence": mtp_last_validate_equivalence,
                    "mtp_config": mtp_config,
                    "mtp_fallback_reason": state.client.mtp_fallback_reason,
                    "mtp_enabled": session["mtp_enabled"],
                    "mtp_initialized": session["mtp_initialized"],
                    "mtp_failure_reason": session["mtp_failure_reason"],
                    "model_id": state.client.paths.model_id,
                    "backend": "orbit-native",
                    "artifact_content_protocol": {
                        "id": ARTIFACT_CONTENT_PROTOCOL_ID,
                        "version": ARTIFACT_CONTENT_PROTOCOL_VERSION,
                        "literal_stream": True,
                    },
                    "native_backend_capabilities": state.native_backend_capabilities,
                    "model_compatibility": state.client.compatibility_diagnostics(),
                    "backend_mode": session["backend_mode"],
                    "thinking_mode": runtime["thinking_mode"],
                    "session_id": session["id"],
                    "cached_tokens": session["cached_tokens"],
                    "in_flight": session["in_flight"],
                    "threads": runtime["threads"],
                    "threads_batch": runtime["threads_batch"],
                    "native_threads": runtime["native_threads"],
                    "native_threads_batch": runtime["native_threads_batch"],
                    "ctx_size": runtime["ctx_size"],
                    "batch_size": runtime["batch_size"],
                    "ubatch_size": runtime["ubatch_size"],
                    "parallel_slots": runtime["parallel_slots"],
                    "final_prefix_experiment_enabled": final_prefix["enabled"],
                    "final_prefix_experiment_initialized": final_prefix["initialized"],
                    "final_prefix_experiment_prefix_tokens": final_prefix["prefix_tokens"],
                    "final_prefix_experiment_capture_count": final_prefix["capture_count"],
                    "final_prefix_experiment_restore_count": final_prefix["restore_count"],
                    "final_prefix_experiment_fallback_count": final_prefix["fallback_count"],
                    "final_prefix_experiment_failure_reason": final_prefix["failure_reason"],
                    "final_prefix_experiment_last_used": final_prefix["last_used"],
                    "final_prefix_experiment_checkpoint_size_bytes": final_prefix["checkpoint_size_bytes"],
                    "qwen_route_prefix_reuse": qwen_route_prefix,
                    "qwen36_shell_tool_prefix_reuse": qwen36_shell_tool_prefix,
                    "qwen3_coder_route_prefix_reuse": qwen3_coder_route_prefix,
                    "ornith_route_prefix_reuse": ornith_route_prefix,
                    "route_shadow": route_shadow,
                    **_model_load_props(state.client),
                    **final_prefix_config,
                    **_tool_call_healing_props(),
                }
            )
            return
        if self.path == "/tools":
            self._json([])
            return
        if self.path == "/sessions":
            self._json({"sessions": [self._state().session_info()]})
            return
        self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            if self.path == "/diagnostics/expert-usage/reset":
                try:
                    self._json(self._state().reset_moe_expert_usage())
                except RuntimeError as exc:
                    self._json({"error": str(exc)}, status=409)
                return
            if self.path == "/session/reset":
                self._json(self._state().reset_session())
                return
            payload = self._read_json()
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return

        try:
            if self.path == "/tokens/count":
                if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > 8 * 1024 * 1024:
                    raise ValueError("token count payload exceeds 8388608 bytes")
                mode = payload.get("mode")
                if mode == "text":
                    text = payload.get("text")
                    if not isinstance(text, str):
                        raise ValueError("text token count requires a string")
                    self._json(self._state().count_text_tokens(text))
                    return
                if mode == "chat":
                    request = parse_chat_request(payload)
                    self._json(
                        self._state().count_chat_tokens(
                            request.messages,
                            tools=request.tools,
                            thinking=bool(request.thinking),
                        )
                    )
                    return
                if mode == "artifact_content":
                    request = parse_chat_request(payload)
                    if request.tools or request.thinking:
                        raise ValueError("artifact content token count does not accept tools or thinking")
                    self._json(self._state().count_artifact_content_tokens(request.messages))
                    return
                raise ValueError("token count mode must be text, chat, or artifact_content")
            if self.path == "/chat":
                try:
                    with native_kv_request_context(endpoint="/chat", payload=payload):
                        self._json(self._state().chat(payload))
                except RuntimeError as exc:
                    self._json(self._state().error_result(str(exc), payload), status=500)
                return
            if self.path == "/chat/continue":
                try:
                    request = parse_continue_request(payload)
                    self._json(self._state().continue_current(request))
                except RuntimeError as exc:
                    self._json(self._state().error_result(str(exc), payload), status=500)
                return
            if self.path == "/chat/stream":
                self._native_stream(payload)
                return
            if self.path == "/chat/continue/stream":
                self._native_continue_stream(payload)
                return
            if self.path == "/cancel":
                self._json(self._state().cancel(_session_id_from_payload(payload)))
                return
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return
        if self.path == "/v1/chat/completions":
            if payload.get("stream") is True:
                self._openai_stream(payload)
            else:
                try:
                    with native_kv_request_context(endpoint="/v1/chat/completions", payload=payload):
                        self._json(openai_chat_response(self._state().chat(payload)))
                except RuntimeError as exc:
                    self._json(openai_chat_response(self._state().error_result(str(exc), payload)), status=500)
                except ValueError as exc:
                    self._json({"error": str(exc)}, status=400)
            return
        self._json({"error": "not found"}, status=404)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _state(self) -> OrbitNativeServer:
        return self.server.orbit_state  # type: ignore[attr-defined]

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _json(self, data: dict[str, Any] | list[Any], *, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except CLIENT_DISCONNECT_ERRORS:
            return

    def _openai_stream(self, payload: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        disconnect = self._start_disconnect_watcher()

        def emit(data: dict[str, Any]) -> None:
            try:
                if disconnect.is_set():
                    self._state().client.cancel()
                    raise BrokenPipeError("client disconnected")
                if self._client_disconnected():
                    self._state().client.cancel()
                    raise BrokenPipeError("client disconnected")
                self.wfile.write(sse_data(data))
                self.wfile.flush()
            except CLIENT_DISCONNECT_ERRORS:
                self._state().client.cancel()
                raise

        def on_token(text: str) -> None:
            emit({"model": self._state().model_alias, "choices": [{"delta": {"content": text}}]})

        try:
            with native_kv_request_context(endpoint="/v1/chat/completions", payload=payload):
                result = self._state().chat(
                    payload,
                    on_token=on_token,
                    should_cancel=lambda: disconnect.is_set() or self._client_disconnected(),
                )
            disconnect.disarm()
            emit(openai_chat_response(result, content=""))
            self.wfile.write(sse_data("[DONE]"))
            self.wfile.flush()
        except ValueError as exc:
            emit({"error": str(exc)})
        except RuntimeError as exc:
            emit(openai_chat_response(self._state().error_result(str(exc), payload), content=""))
            self.wfile.write(sse_data("[DONE]"))
            self.wfile.flush()
        except CLIENT_DISCONNECT_ERRORS:
            self._state().client.cancel()
        finally:
            disconnect.stop()

    def _native_stream(self, payload: dict[str, Any]) -> None:
        try:
            request = parse_chat_request(payload)
            validate_session_id(request.session_id)
            self._state().validate_request(request)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        disconnect = self._start_disconnect_watcher()

        def emit(event: str, data: dict[str, Any]) -> None:
            try:
                if disconnect.is_set():
                    self._state().client.cancel()
                    raise BrokenPipeError("client disconnected")
                if self._client_disconnected():
                    self._state().client.cancel()
                    raise BrokenPipeError("client disconnected")
                self.wfile.write(sse_event(event, data))
                self.wfile.flush()
            except CLIENT_DISCONNECT_ERRORS:
                self._state().client.cancel()
                raise

        def on_token(text: str) -> None:
            emit("delta", {"text": text, "session_id": request.session_id})

        def on_progress(progress) -> None:
            emit(
                f"progress.{progress.phase}",
                _progress_payload(progress, session_id=request.session_id),
            )

        try:
            with native_kv_request_context(endpoint="/chat/stream", payload=payload):
                result = self._state().complete(
                    request,
                    on_progress=on_progress,
                    on_token=on_token,
                    should_cancel=lambda: disconnect.is_set() or self._client_disconnected(),
                )
            disconnect.disarm()
            reasoning = result.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                emit("reasoning", {"text": reasoning, "session_id": request.session_id})
            if result.get("tool_calls"):
                emit("tool_calls", {"tool_calls": result["tool_calls"], "session_id": request.session_id})
            emit("metrics", {"usage": result["usage"], "timings": result["timings"], "native": result["native"]})
            emit("done", {"finish_reason": result["finish_reason"], "session_id": request.session_id})
        except RuntimeError as exc:
            emit("error", {"message": str(exc), "session_id": request.session_id})
            emit("done", {"finish_reason": "error", "session_id": request.session_id})
        except CLIENT_DISCONNECT_ERRORS:
            self._state().client.cancel()
        finally:
            disconnect.stop()

    def _native_continue_stream(self, payload: dict[str, Any]) -> None:
        try:
            request = parse_continue_request(payload)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        disconnect = self._start_disconnect_watcher()

        def emit(event: str, data: dict[str, Any]) -> None:
            try:
                if disconnect.is_set():
                    self._state().client.cancel()
                    raise BrokenPipeError("client disconnected")
                if self._client_disconnected():
                    self._state().client.cancel()
                    raise BrokenPipeError("client disconnected")
                self.wfile.write(sse_event(event, data))
                self.wfile.flush()
            except CLIENT_DISCONNECT_ERRORS:
                self._state().client.cancel()
                raise

        def on_token(text: str) -> None:
            emit("delta", {"text": text, "session_id": DEFAULT_SESSION_ID})

        def on_progress(progress) -> None:
            emit(
                f"progress.{progress.phase}",
                _progress_payload(progress, session_id=DEFAULT_SESSION_ID),
            )

        try:
            result = self._state().continue_current(
                request,
                on_progress=on_progress,
                on_token=on_token,
                should_cancel=lambda: disconnect.is_set() or self._client_disconnected(),
            )
            disconnect.disarm()
            emit("metrics", {"usage": result["usage"], "timings": result["timings"], "native": result["native"]})
            emit("done", {"finish_reason": result["finish_reason"], "session_id": DEFAULT_SESSION_ID})
        except RuntimeError as exc:
            emit("error", {"message": str(exc), "session_id": DEFAULT_SESSION_ID})
            emit("done", {"finish_reason": "error", "session_id": DEFAULT_SESSION_ID})
        except CLIENT_DISCONNECT_ERRORS:
            self._state().client.cancel()
        finally:
            disconnect.stop()

    def _start_disconnect_watcher(self) -> "_DisconnectWatcher":
        watcher = _DisconnectWatcher(self.connection, self._state().client.cancel)
        watcher.start()
        return watcher

    def _client_disconnected(self) -> bool:
        try:
            poll = select.poll()
            events = select.POLLHUP | select.POLLERR
            if hasattr(select, "POLLRDHUP"):
                events |= select.POLLRDHUP
            poll.register(self.connection, events)
            if poll.poll(0):
                return True
        except (AttributeError, OSError, ValueError):
            pass
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except BlockingIOError:
            return False
        except CLIENT_DISCONNECT_ERRORS:
            return True


def _backend_identity() -> str:
    """The vendored llama.cpp revision a profile was measured against.

    A backend rebuild can move inference rates, so a stored measurement should
    not outlive the build it describes. Read from the vendored provenance
    manifest; unreadable means an empty identity, which only costs a
    recalibration.
    """
    try:
        manifest = (
            Path(__file__).resolve().parents[1]
            / "native_llama" / "vendor" / "LLAMA_PROVENANCE.json"
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        tag = str(payload.get("upstream_tag") or "")
        if tag and tag != "untagged":
            return tag
        # A pin without an upstream release tag (41abbfd) is identified by its
        # commit, so two untagged pins never share one calibration entry.
        return str(payload.get("upstream_commit") or "")[:12]
    except (OSError, ValueError, KeyError):
        return ""


def _native_thread_counts(client) -> "tuple[int, int] | None":
    """The live context's thread counts, or None when unobservable.

    Tolerates a client without the read-back (a test double, an older
    backend) because publishing None is correct there and raising would turn
    a diagnostic into an outage.
    """
    reader = getattr(client, "native_thread_counts", None)
    if reader is None:
        return None
    try:
        counts = reader()
    except Exception:
        return None
    if not counts:
        return None
    try:
        return int(counts[0]), int(counts[1])
    except (TypeError, ValueError, IndexError):
        return None


def _log_native_threads(client, stage: str) -> None:
    """One startup line per lifecycle stage: what the context runs on NOW.

    The resolved profile is printed once; this is printed at every point
    where the count could have been moved -- after load, after calibration,
    after the prewarm, at bind -- so a stage that leaves a different count
    active is visible in the log rather than only in a slow benchmark.
    """
    counts = _native_thread_counts(client)
    if counts is None:
        print(f"orbit-server native threads: unobservable ({stage})", file=sys.stderr)
        return
    print(
        f"orbit-server native threads: {counts[0]}/{counts[1]} ({stage})",
        file=sys.stderr,
    )


def _model_identity_for_profile(args) -> "tuple[str, int]":
    """The model bytes a profile was measured against, best-effort.

    A SHA of 20 GiB per start is not affordable, so identity is the resolved
    path plus its size and mtime -- enough that a different model, or the same
    path re-downloaded, produces a different fingerprint, which is the property
    the cache needs. Anything unreadable yields an empty identity, which simply
    means the profile is not cached across runs.
    """
    # Nothing in here may raise: it runs before the model loads, and a profile
    # that cannot be identified is a cache miss, never a failed start. The
    # broad excepts are deliberate -- `args` is whatever the caller built, and
    # a test double or an odd type must cost a recalibration, not a traceback.
    candidate = getattr(args, "model", None)
    if not candidate:
        # The ordinary path: the model comes from `--model-id` (or its default)
        # through the same resolver the server uses, so a profile is fingerprinted
        # against the model that will actually load rather than only against an
        # explicit `--model`. Resolution can legitimately fail here -- a missing
        # model is reported later, by the real bootstrap, with a better message.
        try:
            candidate = resolve_bootstrap_paths(args).model
        except Exception:
            candidate = None
    try:
        path = Path(candidate) if candidate else None
        if path is None or not path.is_file():
            return "", 0
        stat = path.stat()
        return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}", int(stat.st_size)
    except Exception:
        return "", 0


# The context size a start uses when neither the operator (`--ctx`) nor a
# qualified (machine, model) profile names one. The reference number behind
# every qualified corpus run.
DEFAULT_CTX_TOKENS = 8192

def _qualified_startup_profile(args) -> "QualifiedStartupProfile | None":
    """The qualified startup profile for this invocation, or None.

    Matched by DMI product name and the registry model id (`--model-id` or
    the interactive selection), then CONFIRMED against the GGUF's own metadata
    through the same vocab-only inspection discovery uses: the tier applies
    only when the file at the registry path detects as the verified qualified
    artifact (architecture, model name, quantization, template). A legacy
    `--model <path>` start carries no registry id and never matches, and
    nothing here may raise -- an unidentifiable model simply gets no
    qualified tier.
    """
    model_id = getattr(args, "model_id", None)
    if not model_id:
        return None
    try:
        machine = detect_topology().machine_model
    except Exception:
        return None
    candidate = qualified_startup_profile(machine_model=machine, model_id=str(model_id))
    if candidate is None:
        return None
    try:
        paths = resolve_bootstrap_paths(args)
        inspector = NativeProfileInspector(paths.build_bin)
        try:
            profile = inspector(paths.model)
        finally:
            inspector.close()
    except Exception:
        return None
    if not (getattr(profile, "verified", False) and profile.profile_id == candidate.profile_id):
        return None
    return candidate


def _resolve_ctx(args, qualified: "QualifiedStartupProfile | None") -> "tuple[int, str]":
    """`--ctx` beats the qualified profile beats the reference default."""
    explicit = getattr(args, "ctx", None)
    if explicit is not None:
        return int(explicit), "cli"
    if qualified is not None:
        return qualified.ctx, "qualified"
    return DEFAULT_CTX_TOKENS, "default"


def _resolve_startup_profile(
    args, *, calibrator=None, model_identity=None, qualified=None
) -> Resolution:
    """Walk the precedence chain for this invocation.

    Separated from `run_server` so `--show-profile` and the real start resolve
    through exactly the same code: a preview that could disagree with the thing
    it previews would be worse than no preview.

    `model_identity` lets the preview pass the fingerprint it already resolved
    for the selected model (so it does not silently fall back to the default
    model's identity); real startup passes nothing and fingerprints from args
    exactly as before. `qualified` is the `QualifiedStartupProfile` the caller
    looked up once (`_qualified_startup_profile`) and hands to every resolution
    of the same start, so preview, start and the post-load re-resolution never
    disagree and the GGUF is inspected once; None means no tier applies.
    """
    if model_identity is None:
        model_identity, model_bytes = _model_identity_for_profile(args)
    else:
        model_identity, model_bytes = model_identity
    ctx_tokens, _ctx_source = _resolve_ctx(args, qualified)
    return resolve_profile(
        cli={
            "threads": args.threads,
            "threads_batch": args.threads_batch,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "cache_ram_mib": None,
        },
        topology=detect_topology(),
        qualified_profile=qualified.tuning_fields() if qualified is not None else None,
        model_bytes=model_bytes,
        model_sha256=model_identity,
        backend_id=_backend_identity(),
        ctx_tokens=ctx_tokens,
        low_memory=bool(getattr(args, "low_memory", False)),
        mtp_enabled=bool(getattr(args, "enable_mtp_experimental", False)),
        expert_usage_enabled=bool(getattr(args, "moe_expert_usage", False)),
        calibrator=calibrator,
        recalibrate=bool(getattr(args, "recalibrate", False)),
        allow_calibration=calibrator is not None,
    )


def _resolve_preview_target(args) -> "tuple[str, str, tuple[str, int], bool] | int":
    """Resolve, read-only, which model `--show-profile` should describe.

    Mirrors the model resolution a real start uses -- an explicit path/id, the
    shared interactive selection, or the default model -- and returns the
    display name, a path description, the fingerprint identity, and whether the
    model is absent locally. It never downloads or loads a model. Returns an int
    exit code only when an interactive selection is cancelled or invalid.
    """
    explicit_model = getattr(args, "model", None)
    if explicit_model is not None:
        path = Path(explicit_model)
        present = path.is_file()
        disp = str(path) if present else f"{path} (not present locally)"
        return path.name, disp, _model_identity_for_profile(args), (not present)
    if getattr(args, "model_id", None) is not None:
        try:
            path = resolve_bootstrap_paths(args).model
        except Exception:
            path = None
        present = bool(path and Path(path).is_file())
        name = Path(path).name if path else str(args.model_id)
        disp = str(path) if present else (f"{path} (not present locally)" if path else "(unresolved)")
        return name, disp, _model_identity_for_profile(args), (not present)
    if _interactive_model_selection_requested(args):
        chosen = _choose_verified_model(args)
        if isinstance(chosen, int):
            return chosen
        row, _build_bin = chosen
        if row.local == "AVAILABLE":
            # Set the resolved path AND the registry id so the fingerprint and
            # the qualified (machine, model) tier match what a real start would
            # use (`_select_startup_model` sets both); no download, no load --
            # this is still a preview.
            args.model = Path(row.path_or_action)
            args.model_id = row.model_id
            # Memory mode is part of the same interactive selection and is a
            # fingerprint axis, so a low-memory-capable model must ask here too;
            # otherwise the preview would report the standard-mode cache while a
            # real start under low memory keys a different fingerprint.
            memory_exit = _select_memory_mode(args, row)
            if memory_exit is not None:
                return memory_exit
            return row.model, row.path_or_action, _model_identity_for_profile(args), False
        # MISSING: an honest heuristic preview, never a download.
        return row.model, "(not present locally; downloaded and calibrated on normal startup)", ("", 0), True
    # Non-interactive with no explicit model: the default a real non-interactive
    # start would use, named so the operator is never shown an unlabelled generic
    # profile.
    try:
        path = resolve_bootstrap_paths(args).model
    except Exception:
        path = None
    present = bool(path and Path(path).is_file())
    name = Path(path).name if path else str(DEFAULT_MODEL_ID)
    disp = str(path) if present else (f"{path} (not present locally)" if path else f"{DEFAULT_MODEL_ID} (default)")
    return name, disp, _model_identity_for_profile(args), (not present)


def _show_profile(args) -> int:
    """`--show-profile`: report the resolved profile for the model a real start
    would use, read-only. No calibrator, no model load, no cache mutation."""
    target = _resolve_preview_target(args)
    if isinstance(target, int):
        return target
    name, path_display, model_identity, missing = target
    qualified = _qualified_startup_profile(args)
    preview = _resolve_startup_profile(args, model_identity=model_identity, qualified=qualified)
    ctx_tokens, ctx_source = _resolve_ctx(args, qualified)
    print(f"model: {name}")
    print(f"path: {path_display}")
    if qualified is not None:
        print(f"qualified profile: {qualified.source}")
    print(f"ctx: {ctx_tokens} ({ctx_source})")
    for line in render_profile_lines(preview):
        print(line)
    if missing:
        print("note: this model is not present locally; the values above are a "
              "heuristic preview -- normal startup downloads it and calibrates once")
    elif getattr(args, "recalibrate", False):
        # The cache was deliberately not consulted, so nothing here can say
        # whether one exists; claiming there is none would guess about a file
        # this command chose not to read.
        print("note: --recalibrate shown from the heuristic; any stored "
              "measurement is left untouched until the server starts")
    elif preview.cache_path is None:
        print("note: no cached measurement for this model on this machine yet; "
              "starting the server normally will calibrate once")
    return 0


def _startup_note(message: str) -> None:
    """One line of startup progress, on the same stream and prefix as the
    existing server lines. Presentation only: line-based, no cursor control,
    no ANSI, so it is identical on a TTY and a redirect and honors NO_COLOR and
    dumb terminals by adding nothing to color."""
    print(f"orbit-server {message}", file=sys.stderr, flush=True)


def _report_startup_prewarm(label: str, result: "NativeRoutePrefixPrefillResult") -> None:
    """Truthful one-line outcome for a prewarm that was announced.

    Never prints success unless the prewarm actually succeeded; a skip or a
    failure says so, using only metrics the result already carries. Metrics that
    are absent are simply omitted rather than estimated."""
    if result.succeeded:
        parts: list[str] = []
        if result.prefix_token_count is not None:
            parts.append(f"{result.prefix_token_count} tokens")
        if result.prefill_ms is not None:
            parts.append(f"{result.prefill_ms / 1000:.1f}s")
        if result.checkpoint_size_bytes is not None:
            parts.append(f"{result.checkpoint_size_bytes // 1024} KiB")
        detail = f": {', '.join(parts)}" if parts else ""
        _startup_note(f"prewarm complete ({label}){detail}")
    elif result.skipped:
        _startup_note(f"prewarm skipped ({label}): {result.skip_reason or 'ineligible'}")
    elif result.attempted:
        _startup_note(f"prewarm failed ({label}): {result.failed_reason or 'unknown'}")


def run_server(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "show_profile", False):
        # Preview only, and model-aware: it resolves the model the same way a
        # real start would (explicit path/id, the shared interactive selection,
        # or the default), but is strictly read-only -- it never downloads,
        # loads a context, calibrates, prewarms, binds a socket or touches the
        # profile cache. Handled before the real selection so the download path
        # in `_select_startup_model` can never run for a preview.
        return _show_profile(args)
    selected_interactively = False
    if _interactive_model_selection_requested(args):
        try:
            selection_exit = _select_startup_model(args)
            if selection_exit is not None:
                return selection_exit
            selected_interactively = True
        except KeyboardInterrupt:
            print("\nmodel selection cancelled", file=sys.stderr)
            return 130
    final_prefix_config = resolve_final_prefix_reuse()
    qwen_route_prefix_config = resolve_qwen_route_prefix_reuse()
    qwen36_shell_tool_prefix_config = resolve_qwen36_shell_tool_prefix_reuse()
    qwen3_coder_route_prefix_config = resolve_qwen3_coder_route_prefix_reuse()
    ornith_route_prefix_config = resolve_ornith_route_prefix_reuse()
    ornith_analysis_prefix_config = resolve_ornith_analysis_prefix_reuse()

    # Resolved BEFORE the model loads, because batch and ubatch are context
    # creation parameters and cannot be retuned afterwards. Threads can be, so
    # they may still be replaced by a measurement below; everything else this
    # returns is final.
    _startup_note("resolving server profile...")
    qualified = _qualified_startup_profile(args)
    args.ctx, ctx_source = _resolve_ctx(args, qualified)
    if qualified is not None:
        _startup_note(
            f"qualified profile: {qualified.source} "
            f"(ctx {qualified.ctx}, threads {qualified.threads}/{qualified.threads_batch}, "
            f"batch {qualified.batch}/{qualified.ubatch}, lazy_mode {qualified.lazy_mode}, "
            f"load_mtp {'on' if qualified.load_mtp else 'off'})"
        )
    _startup_note(f"ctx: {args.ctx} ({ctx_source})")
    resolution = _resolve_startup_profile(args, qualified=qualified)
    profile = resolution.profile
    # A cached calibrated profile means the long calibration sweep below will not
    # run; say so early, before the model load, so the operator knows the wait
    # ahead is the load and not a calibration.
    if str(getattr(profile, "source", "")).startswith("cached"):
        _startup_note("server profile: cached auto-calibrated (no calibration needed)")

    paths: NativeLlamaPaths | None = None
    try:
        paths = resolve_bootstrap_paths(args)
        # CPU weight-repack resolution lives in the backend-invocation layer:
        # explicit --repack beats ORBIT_CPU_REPACK beats the qualified
        # per-(machine, model) default beats the backend default. Only the
        # qualified Dell pairs (Ornith, Qwen3.8 Flash Next) default to repack off.
        cpu_repack_cli = {"on": True, "off": False, "auto": None}[
            getattr(args, "repack", "auto")
        ]
        cpu_repack, cpu_repack_source = resolve_cpu_repack(
            machine_model=detect_topology().machine_model,
            model_id=getattr(paths, "model_id", "") or "",
            cli=cpu_repack_cli,
            environ=os.environ,
        )
        if cpu_repack is not None:
            _startup_note(
                f"cpu repack: {'on' if cpu_repack else 'off'} ({cpu_repack_source})"
            )
        client = NativeLlamaClient(
            paths,
            NativeClientConfig(
                context_tokens=args.ctx,
                threads=profile.threads,
                threads_batch=profile.threads_batch,
                batch_size=profile.batch,
                ubatch_size=profile.ubatch,
                thinking=args.think == "on",
                mtp_probe_enabled=args.enable_mtp_probe,
                mtp_dry_run_enabled=args.enable_mtp_dry_run,
                mtp_accept_probe_enabled=args.enable_mtp_accept_probe,
                mtp_decode_probe_enabled=args.enable_mtp_decode_probe,
                use_mtp_experimental=args.enable_mtp_experimental,
                final_prefix_experiment_enabled=final_prefix_config.enabled,
                final_prefix_reuse_source=final_prefix_config.source,
                final_prefix_reuse_config_error=final_prefix_config.validation_error,
                final_prefix_reuse_legacy_detected=final_prefix_config.legacy_detected,
                qwen_route_prefix_reuse_enabled=qwen_route_prefix_config.enabled,
                qwen_route_prefix_reuse_source=qwen_route_prefix_config.source,
                qwen_route_prefix_reuse_config_error=qwen_route_prefix_config.validation_error,
                qwen36_shell_tool_prefix_reuse_enabled=qwen36_shell_tool_prefix_config.enabled,
                qwen36_shell_tool_prefix_reuse_source=qwen36_shell_tool_prefix_config.source,
                qwen36_shell_tool_prefix_reuse_config_error=qwen36_shell_tool_prefix_config.validation_error,
                qwen3_coder_route_prefix_reuse_enabled=qwen3_coder_route_prefix_config.enabled,
                qwen3_coder_route_prefix_reuse_source=qwen3_coder_route_prefix_config.source,
                qwen3_coder_route_prefix_reuse_config_error=qwen3_coder_route_prefix_config.validation_error,
                ornith_route_prefix_reuse_enabled=ornith_route_prefix_config.enabled,
                ornith_route_prefix_reuse_source=ornith_route_prefix_config.source,
                ornith_route_prefix_reuse_config_error=ornith_route_prefix_config.validation_error,
                ornith_analysis_prefix_reuse_enabled=ornith_analysis_prefix_config.enabled,
                ornith_analysis_prefix_reuse_source=ornith_analysis_prefix_config.source,
                ornith_analysis_prefix_reuse_config_error=ornith_analysis_prefix_config.validation_error,
                moe_expert_usage_enabled=args.moe_expert_usage,
                low_memory=args.low_memory,
                use_extra_bufts=cpu_repack,
                load_mode=qualified.load_mode if qualified is not None else None,
                lazy_mode=qualified.lazy_mode if qualified is not None else None,
                load_mtp=qualified.load_mtp if qualified is not None else None,
            ),
        )
        if not args.verbose_llama_log:
            client.set_quiet_logging()
        _startup_note(f"loading model: {resolve_model_alias(args.alias, paths)}...")
        client.load()
        _log_native_threads(client, "after model load")

        # Threads are the one measurable field, and this is the only point at
        # which they can be measured: after the weights are resident, before
        # the prefix prewarm and before the socket binds. Running it after the
        # prewarm would time a checkpoint restore on one candidate and a real
        # prefill on the next -- the exact warm/cold mismatch the CHAT cache
        # work turned up. `calibrate_threads` never raises; it returns None and
        # the pre-load resolution stands.
        if not resolution.calibrated and (args.threads is None or args.threads_batch is None):
            # Progress is emitted from INSIDE the calibrator, which the resolver
            # invokes only when a real sweep is needed. A cached run reaches this
            # branch too (its thread fields are filled, so `measurable` is empty
            # and the calibrator is never called), and must NOT announce a sweep
            # -- so `calibrator_ran` gates the completion line as well.
            calibrator_ran = {"invoked": False}

            def _startup_calibrator(*, topology, fields):
                calibrator_ran["invoked"] = True
                total = len(thread_candidates(topology))
                _startup_note("auto-calibrating server profile...")
                seen = {"n": 0}

                def _on_candidate(measurement) -> None:
                    marker = getattr(measurement, "rejected", None)
                    if isinstance(marker, str) and marker.startswith(WARMUP_REJECTION):
                        return  # the warm-up walk is not a scored candidate
                    seen["n"] += 1
                    _startup_note(
                        f"  candidate {seen['n']}/{total}: "
                        f"threads={measurement.threads}"
                    )

                return calibrate_threads(
                    client, topology=topology, fields=fields, on_event=_on_candidate
                )

            measured = _resolve_startup_profile(
                args, calibrator=_startup_calibrator, qualified=qualified
            )
            if measured.calibrated:
                resolution = measured
                profile = measured.profile
                _startup_note(
                    f"calibration complete: threads={profile.threads}, "
                    f"threads_batch={profile.threads_batch}"
                )
            elif calibrator_ran["invoked"]:
                _startup_note(
                    "calibration did not settle; keeping the pre-load profile"
                )
            # Whether or not a winner emerged, the context is left tuned to the
            # last candidate that ran. Put it back on the resolved counts, or
            # the server would serve on one thread count while reporting
            # another.
            restore_threads(client, profile.threads, profile.threads_batch)
            # The context now runs on the resolved counts, but `client.config`
            # still holds the pre-calibration ones -- and it is not decoration:
            # `/props` publishes it, the smoke harness records it, and both the
            # native-version and final-prefix identities hash it. Left stale,
            # one process would report threads it is not using and two
            # identical runtimes would compute different checkpoint identities.
            # The config is frozen, so this is the only way to correct it.
            for field, value in (
                ("threads", profile.threads),
                ("threads_batch", profile.threads_batch),
            ):
                # Narrow on purpose. A frozen dataclass accepts this; a config
                # that grew `__slots__` or became a NamedTuple would raise
                # AttributeError/TypeError, and swallowing everything would
                # leave `/props` and two identity hashes quietly stale with no
                # signal. Anything else is a real bug and should surface.
                try:
                    object.__setattr__(client.config, field, value)
                except (AttributeError, TypeError):
                    pass

        for line in render_profile_lines(resolution):
            print(f"orbit-server {line}", file=sys.stderr)
        _log_native_threads(client, "after profile resolution")

        # Announce prewarm only when it is actually enabled for this server, so
        # a build with prewarm switched off stays quiet. The prewarm calls
        # themselves are unchanged -- each still runs exactly once -- and the
        # start line is emitted immediately before the expensive call, the
        # outcome only after it returns.
        announce_prewarm = route_prefix_prewarm_mode() == PREFIX_PREWARM_STARTUP
        if getattr(getattr(client, "model_profile", None), "profile_id", None) not in (
            QWEN3_CODER_PROFILE_ID,
            ORNITH15_PROFILE_ID,
        ):
            if announce_prewarm:
                _startup_note("prewarming route-prefix cache...")
            result = prewarm_startup_route_prefix(client)
            if announce_prewarm:
                _report_startup_prewarm("route-prefix", result)
        else:
            prewarm_interrupted = False
            previous_sigint = signal.getsignal(signal.SIGINT)

            def cancel_startup_prewarm(_signum, _frame) -> None:
                nonlocal prewarm_interrupted
                prewarm_interrupted = True
                client.cancel()

            signal.signal(signal.SIGINT, cancel_startup_prewarm)
            try:
                if announce_prewarm:
                    _startup_note("prewarming route-prefix cache...")
                route_result = prewarm_startup_route_prefix(client)
                if announce_prewarm:
                    _report_startup_prewarm("route-prefix", route_result)
                # Beside the CHAT capture, never instead of it: the CHAT prewarm
                # has already run and recorded its result, and this one owns a
                # separate slot. It is inside the same cancellable window because
                # it is more startup prefill the operator may want to interrupt.
                if not prewarm_interrupted:
                    if announce_prewarm:
                        _startup_note("prewarming analysis-prefix cache...")
                    analysis_result = prewarm_startup_analysis_prefix(client)
                    if announce_prewarm:
                        _report_startup_prewarm("analysis-prefix", analysis_result)
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
            if prewarm_interrupted:
                client.close()
                return 130
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        print(_format_native_bootstrap_error(exc), file=sys.stderr)
        if _is_model_bootstrap_failure(exc) and not selected_interactively:
            _print_model_discovery(args, paths)
        return 1

    assert paths is not None
    _log_native_threads(client, "after prewarm, before bind")
    model_alias = resolve_model_alias(args.alias, paths)
    httpd = ThreadingHTTPServer((args.host, args.port), OrbitNativeHandler)
    httpd.orbit_state = OrbitNativeServer(client=client, model_alias=model_alias)  # type: ignore[attr-defined]
    print(f"orbit-server model: {model_alias}", flush=True)
    print(f"orbit-server listening on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\norbit-server stopped", flush=True)
    finally:
        # A post-final route shadow may be inside llama_decode on its own
        # thread; the context must not be freed underneath it.
        httpd.orbit_state.stop_route_shadow()  # type: ignore[attr-defined]
        client.close()
        httpd.server_close()
    return 0


def route_prefix_prewarm_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    value = env.get(PREFIX_PREWARM_ENV, PREFIX_PREWARM_STARTUP).strip().lower()
    if value == PREFIX_PREWARM_OFF:
        return PREFIX_PREWARM_OFF
    if value in {"", PREFIX_PREWARM_STARTUP}:
        return PREFIX_PREWARM_STARTUP
    return PREFIX_PREWARM_OFF


def tools_startup_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    value = env.get(TOOLS_ENV, "on").strip().lower()
    if value == "on":
        return True
    if value == "off":
        return False
    return False


def _final_prefix_reuse_props(client: NativeLlamaClient) -> dict[str, object]:
    return {
        "final_prefix_reuse_enabled": client.config.final_prefix_experiment_enabled,
        "final_prefix_reuse_source": client.config.final_prefix_reuse_source,
        "final_prefix_reuse_config_error": client.config.final_prefix_reuse_config_error,
        "final_prefix_reuse_legacy_detected": client.config.final_prefix_reuse_legacy_detected,
    }


def _model_load_props(client: NativeLlamaClient) -> dict[str, object]:
    return client.model_load_status()


def _tool_call_healing_props() -> dict[str, object]:
    return tool_call_healing_status()


def _is_final_from_tool_prompt(messages: list[dict[str, Any]]) -> bool:
    return (
        len(messages) == 3
        and [message.get("role") for message in messages] == ["system", "user", "system"]
        and messages[0].get("content") == FINAL_FROM_TOOL_SYSTEM_PROMPT
    )


def _is_qwen_route_prompt(messages: list[dict[str, Any]]) -> bool:
    return bool(
        len(messages) >= 2
        and messages[0].get("role") == "system"
        and messages[0].get("content") == ROUTE_SYSTEM_PROMPT
    )


def prewarm_startup_route_prefix(client: NativeLlamaClient) -> NativeRoutePrefixPrefillResult:
    mode = route_prefix_prewarm_mode()
    tools_enabled = tools_startup_enabled()
    if mode != PREFIX_PREWARM_STARTUP:
        result = _startup_prewarm_skipped("disabled")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result)
        return result
    if not tools_enabled:
        result = _startup_prewarm_skipped("tools_disabled")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result)
        return result
    if not prefix_anchor_enabled():
        result = _startup_prewarm_skipped("anchor_disabled")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result)
        return result
    profile = getattr(client, "model_profile", None)
    if getattr(profile, "profile_id", None) in (QWEN3_CODER_PROFILE_ID, ORNITH15_PROFILE_ID):
        try:
            result = client.capture_qwen3_coder_route_prefix_prefill_only(
                system_prompt=ROUTE_SYSTEM_PROMPT,
                tools_mode="on",
            )
        except Exception as exc:
            result = NativeRoutePrefixPrefillResult(
                attempted=True,
                succeeded=False,
                skipped=False,
                failed_reason=f"startup_prewarm_failed:{type(exc).__name__}",
                restore_ready=False,
            )
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result)
        return result
    if profile is not None and not profile.gemma_prefix_reuse_supported:
        result = _startup_prewarm_skipped("model_profile_ineligible")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result)
        return result
    try:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": ROUTE_SYSTEM_PROMPT}],
            thinking=False,
        )
        result = client.capture_route_prefix_prefill_only(segments, tools_mode="on")
    except Exception as exc:
        result = NativeRoutePrefixPrefillResult(
            attempted=True,
            succeeded=False,
            skipped=False,
            failed_reason=f"startup_prewarm_failed:{type(exc).__name__}",
            restore_ready=False,
        )
    _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result)
    return result


def prewarm_startup_analysis_prefix(client: NativeLlamaClient) -> NativeRoutePrefixPrefillResult:
    """Capture the ANALYSIS prefix at startup, beside the CHAT one.

    Same mechanism, same gates, one more slot. The prefix is derived from the
    ANALYSIS system contract and the real `execute_analysis` schema -- both
    module constants -- so nothing about a session, an artifact or an analyst
    enters it, and the client's own exact-prefix derivation re-proves that
    against the prompt it is actually serving before anything is reused.

    Without this the first ANALYSIS step of a server's life is always cold:
    the checkpoint it would have restored is the one that step captures.

    A failure here is contained. It returns a result rather than raising, and
    the CHAT prewarm has already completed and been recorded by the time this
    runs, so a broken ANALYSIS capture costs the first analysis step nothing it
    was not already paying.
    """
    mode = route_prefix_prewarm_mode()
    tools_enabled = tools_startup_enabled()
    if mode != PREFIX_PREWARM_STARTUP:
        result = _startup_prewarm_skipped("disabled")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result, lineage="analysis")
        return result
    if not tools_enabled:
        result = _startup_prewarm_skipped("tools_disabled")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result, lineage="analysis")
        return result
    if not prefix_anchor_enabled():
        result = _startup_prewarm_skipped("anchor_disabled")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result, lineage="analysis")
        return result
    if not resolve_ornith_analysis_prefix_prewarm().enabled:
        # Off unless the operator asked for it. An eager ANALYSIS capture costs
        # real startup time and resident memory, and a server that only chats
        # would pay both for nothing. Skipping here changes nothing else: the
        # first analysis step still captures the checkpoint on its way past,
        # exactly as it did before this prewarm existed.
        result = _startup_prewarm_skipped("analysis_prewarm_not_requested")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result, lineage="analysis")
        return result
    profile = getattr(client, "model_profile", None)
    if getattr(profile, "profile_id", None) != ORNITH15_PROFILE_ID:
        # Only Ornith has a qualified ANALYSIS prefix. Every other profile keeps
        # the lazy capture it has now.
        result = _startup_prewarm_skipped("model_profile_ineligible")
        _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result, lineage="analysis")
        return result
    try:
        result = client.capture_qwen3_coder_route_prefix_prefill_only(
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            tools_mode="on",
            tools=[ANALYSIS_TOOL_SCHEMA],
            analysis_lineage=True,
        )
    except Exception as exc:
        result = NativeRoutePrefixPrefillResult(
            attempted=True,
            succeeded=False,
            skipped=False,
            failed_reason=f"startup_prewarm_failed:{type(exc).__name__}",
            restore_ready=False,
        )
    _emit_startup_prewarm_diag(mode=mode, tools_enabled=tools_enabled, result=result, lineage="analysis")
    return result


def _startup_prewarm_skipped(reason: str) -> NativeRoutePrefixPrefillResult:
    return NativeRoutePrefixPrefillResult(
        attempted=False,
        succeeded=False,
        skipped=True,
        skip_reason=reason,
        sampled_tokens=0,
        generated_tokens=0,
        sampler_touched=False,
        session_history_touched=False,
        restore_ready=False,
    )


def _emit_startup_prewarm_diag(
    *,
    mode: str,
    tools_enabled: bool,
    result: NativeRoutePrefixPrefillResult,
    lineage: str = "chat",
) -> None:
    emit_route_prefix_prewarm_event(
        {
            # Two prewarms now run at startup and both emit here, so the event
            # says which lineage it describes. The backend still learns nothing
            # about CHAT or ANALYSIS from a request: this is diagnostics only.
            "prewarm_lineage": lineage,
            "tools_default_enabled": tools_startup_enabled({}),
            "tools_startup_enabled": tools_enabled,
            "prewarm_enabled": mode == PREFIX_PREWARM_STARTUP and tools_enabled,
            "prewarm_mode": mode,
            "prewarm_attempted": result.attempted,
            "prewarm_succeeded": result.succeeded,
            "prewarm_skipped_reason": result.skip_reason,
            "prewarm_failed_reason": result.failed_reason,
            "prewarm_prefix_token_count": result.prefix_token_count,
            "prewarm_checkpoint_size_bytes": result.checkpoint_size_bytes,
            "prewarm_ms": int(result.prefill_ms) if result.prefill_ms is not None else None,
            "decode_calls": result.decode_calls,
            "sampled_tokens": result.sampled_tokens,
            "generated_tokens": result.generated_tokens,
            "sampler_touched": result.sampler_touched,
            "session_history_touched": result.session_history_touched,
            "restore_ready": result.restore_ready,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit-server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--llama-root",
        type=Path,
        default=DEFAULT_LLAMA_ROOT,
        help=(
            "Optional legacy llama.cpp root. If omitted, Orbit first looks for native libraries under "
            "orbit/native_llama/vendor/lib, then ORBIT_LLAMA_LIB_DIR, then ORBIT_LLAMA_ROOT."
        ),
    )
    parser.add_argument("--model-id", default=None, help=f"Orbit model id. Defaults to {DEFAULT_MODEL_ID} when --model is not used.")
    parser.add_argument("--model", type=Path, help="Legacy direct target model path override.")
    parser.add_argument("--mmproj", type=Path, help="Optional multimodal projector override for native image/audio support.")
    parser.add_argument("--models-dir", type=Path, help="Orbit local models directory.")
    parser.add_argument("--hf-cache", type=Path, help="Hugging Face cache root fallback.")
    parser.add_argument("--alias", help="Model name exposed by the server. Defaults to the exact GGUF filename.")
    # Default None so a qualified (machine, model) profile can supply its own
    # ctx; an unsupplied flag otherwise resolves to DEFAULT_CTX_TOKENS.
    parser.add_argument("--ctx", type=int, default=None)
    # Default None, not the reference numbers. argparse cannot otherwise tell
    # `--threads 6` from an unsupplied flag, and the whole precedence contract
    # rests on that distinction: a value the operator named must never be
    # measured, cached over, or reported as chosen by Orbit.
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--threads-batch", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--ubatch", type=int, default=None)
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Resolve and print the startup profile, then exit without loading a model.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Discard any cached auto-calibrated profile and measure this machine again.",
    )
    parser.add_argument("--think", choices=("off", "on"), default="off", help="Default thinking visibility for native server requests.")
    parser.add_argument("--repack", choices=("on", "off", "auto"), default="auto", help="CPU weight repacking (llama.cpp use_extra_bufts). 'on'/'off' force it; 'auto' (default) uses the qualified per-machine+model default, else the backend default.")
    parser.add_argument("--enable-mtp-probe", action="store_true", help="Backend-only MTP load/init probe. No generation.")
    parser.add_argument("--enable-mtp-dry-run", action="store_true", help="Backend-only MTP draft generation dry run. No accept loop or user output.")
    parser.add_argument("--enable-mtp-accept-probe", action="store_true", help="Backend-only MTP single accept-loop probe. No user output or runtime integration.")
    parser.add_argument("--enable-mtp-decode-probe", action="store_true", help="Backend-only experimental MTP decode-loop probe. No user output or runtime integration.")
    parser.add_argument(
        "--mtp",
        "--enable-mtp-experimental",
        dest="enable_mtp_experimental",
        action="store_true",
        help="Enable native MTP completion path with automatic no-MTP fallback.",
    )
    parser.add_argument("--verbose-llama-log", action="store_true")
    parser.add_argument("--moe-expert-usage", action="store_true", help="Enable opt-in CPU MoE expert-selection counters.")
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Use a qualified low-memory profile when supported by the selected model.",
    )
    return parser


def resolve_bootstrap_paths(args: argparse.Namespace) -> NativeLlamaPaths:
    if args.model_id:
        return resolve_paths(
            llama_root=args.llama_root,
            model_id=args.model_id,
            model=args.model,
            mmproj=args.mmproj,
            models_dir=args.models_dir,
            hf_cache=args.hf_cache,
        )
    if args.model is not None:
        return resolve_legacy_paths(llama_root=args.llama_root, model=args.model, mmproj=args.mmproj)
    return resolve_paths(
        llama_root=args.llama_root,
        model_id=DEFAULT_MODEL_ID,
        mmproj=args.mmproj,
        models_dir=args.models_dir,
        hf_cache=args.hf_cache,
    )


def resolve_model_alias(alias: str | None, paths: NativeLlamaPaths) -> str:
    return alias or paths.model.name


def _format_native_bootstrap_error(exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    if "libllama.so not found" in detail:
        return (
            "error: native backend libraries are missing.\n"
            f"detail: {detail}\n"
            "hint: provide --llama-root /path/to/llama.cpp, or set ORBIT_LLAMA_ROOT, "
            "or package native libraries under src/orbit/native_llama/vendor/lib."
        )
    if "missing native build inputs for" in detail:
        return (
            "error: native MTP shim inputs are missing.\n"
            f"detail: {detail}\n"
            "hint: use --llama-root /path/to/llama.cpp (or ORBIT_LLAMA_ROOT) so Orbit can rebuild "
            "the required shim, or package the shim under src/orbit/native_llama/vendor/shim."
        )
    return f"error: failed to start native backend: {detail}"


def _is_model_bootstrap_failure(exc: Exception) -> bool:
    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in (
            "model not found:",
            "target model not found:",
            "failed to load model:",
            "unsupported or unverified native model compatibility",
            "unknown native model manifest",
        )
    )


def _print_model_discovery(args: argparse.Namespace, paths: NativeLlamaPaths | None) -> None:
    build_bin = paths.build_bin if paths is not None else None
    if build_bin is None:
        try:
            build_bin = _resolve_native_runtime(args.llama_root)[1]
        except FileNotFoundError:
            build_bin = None
    try:
        result = discover_models(
            models_dir=effective_models_dir(args.models_dir),
            hf_cache=args.hf_cache or default_hf_cache(),
            explicit_model=args.model,
            build_bin=build_bin,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"model discovery unavailable: {str(exc).strip() or exc.__class__.__name__}", file=sys.stderr)
        return
    print(format_model_discovery(result, color=supports_ansi(sys.stderr)), file=sys.stderr)


def _interactive_model_selection_requested(args: argparse.Namespace) -> bool:
    isatty = getattr(sys.stdin, "isatty", None)
    return args.model is None and args.model_id is None and callable(isatty) and bool(isatty())


def _choose_verified_model(args: argparse.Namespace) -> "tuple[ModelDiscoveryRow, Path] | int":
    """Discovery + the numbered prompt, shared by real startup and the preview.

    Read-only: it discovers, lists and reads one selection, and returns the
    chosen row plus the resolved build_bin. It never downloads, sets args, or
    loads a model -- those are the caller's job -- so `--show-profile` can reuse
    the exact same selection semantics without the side effects of a real start.
    Returns an int exit code on discovery failure, no choices, or a
    cancelled/invalid selection.
    """
    try:
        build_bin = _resolve_native_runtime(args.llama_root)[1]
        result = discover_models(
            models_dir=effective_models_dir(args.models_dir),
            hf_cache=args.hf_cache or default_hf_cache(),
            build_bin=build_bin,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(_format_native_bootstrap_error(exc), file=sys.stderr)
        return 1

    print(format_model_discovery(result, color=supports_ansi(sys.stderr)), file=sys.stderr)
    choices = tuple(
        row
        for row in result.rows
        if row.local in {"AVAILABLE", "MISSING"} and row.support == "VERIFIED"
    )
    if not choices:
        print("error: no verified model is available or downloadable", file=sys.stderr)
        return 1

    print("\nVerified models:", file=sys.stderr)
    color = supports_ansi(sys.stderr)
    for index, row in enumerate(choices, start=1):
        print(f"  {index}. {row.model} [{paint_model_status(row.local, color=color)}]", file=sys.stderr)
    print(f"Select model [1-{len(choices)}]: ", end="", file=sys.stderr, flush=True)
    response = sys.stdin.readline()
    if not response:
        print("model selection cancelled", file=sys.stderr)
        return 1
    try:
        selection = int(response.strip())
    except ValueError:
        print("error: invalid model selection", file=sys.stderr)
        return 1
    if not 1 <= selection <= len(choices):
        print("error: invalid model selection", file=sys.stderr)
        return 1
    return choices[selection - 1], build_bin


def _select_startup_model(args: argparse.Namespace) -> int | None:
    chosen = _choose_verified_model(args)
    if isinstance(chosen, int):
        return chosen
    selected, build_bin = chosen

    if selected.local == "MISSING":
        return _download_selected_model(args, selected, build_bin=build_bin)
    args.model = Path(selected.path_or_action)
    args.model_id = selected.model_id
    memory_exit = _select_memory_mode(args, selected)
    if memory_exit is not None:
        return memory_exit
    print(f"Starting {selected.model}...", file=sys.stderr)
    return None


def _select_memory_mode(args: argparse.Namespace, selected: ModelDiscoveryRow) -> int | None:
    if args.low_memory or not selected.low_memory_supported:
        return None
    print("\nMemory mode:", file=sys.stderr)
    print("  1. Standard (~31.3 GiB peak RSS)", file=sys.stderr)
    print("  2. Low memory (~18.3 GiB peak RSS)", file=sys.stderr)
    print("Low memory is recommended for hosts with >=24 GB RAM.", file=sys.stderr)
    print("Select memory mode [1-2] (default 1): ", end="", file=sys.stderr, flush=True)
    response = sys.stdin.readline()
    if not response:
        print("memory mode selection cancelled", file=sys.stderr)
        return 1
    answer = response.strip()
    if answer in {"", "1"}:
        return None
    if answer == "2":
        args.low_memory = True
        return None
    print("error: invalid memory mode selection", file=sys.stderr)
    return 1


def _download_selected_model(
    args: argparse.Namespace,
    selected: ModelDiscoveryRow,
    *,
    build_bin: Path,
) -> int | None:
    if selected.model_id is None:
        print("error: selected model has no canonical download identity", file=sys.stderr)
        return 1
    try:
        manifest = get_manifest(selected.model_id)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    target = f"{manifest.target.repo}/{manifest.target.file}"
    print(f"\nSelected: {manifest.display_name}", file=sys.stderr)
    print(f"Download: orbit download {target}", file=sys.stderr)
    print("Download now? [Y/n] ", end="", file=sys.stderr, flush=True)
    response = sys.stdin.readline()
    if not response:
        print("download cancelled", file=sys.stderr)
        return 0
    answer = response.strip().lower()
    if answer in {"n", "no"}:
        print("download cancelled", file=sys.stderr)
        return 0
    if answer not in {"", "y", "yes"}:
        print("error: invalid download confirmation", file=sys.stderr)
        return 1

    models_dir = effective_models_dir(args.models_dir)
    advisory = download_advisory(
        models_dir,
        url=huggingface_resolve_url(parse_huggingface_spec(target)),
        destination=local_model_path(manifest.target, models_dir=models_dir),
    )
    if advisory:
        print(advisory, file=sys.stderr)
    progress = DownloadProgress()

    def on_shard(index: int, count: int, name: str, action: str) -> None:
        progress.finish()
        label = {"present": "already present", "download": "downloading", "resume": "resuming"}[action]
        print(f"shard {index}/{count}: {name} ({label})", file=sys.stderr, flush=True)

    try:
        try:
            downloaded = download_model(target, models_dir=models_dir, progress=progress, on_shard=on_shard)
        finally:
            progress.finish()
    except Exception as exc:
        print(f"error: download failed: {str(exc).strip() or exc.__class__.__name__}", file=sys.stderr)
        return 1

    expected_path = local_model_path(manifest.target, models_dir=models_dir).expanduser().resolve()
    downloaded_path = downloaded.path.expanduser().resolve()
    if downloaded_path != expected_path:
        print("error: downloader returned an unexpected model destination", file=sys.stderr)
        return 1

    action = "downloaded" if downloaded.downloaded else "already present"
    print(f"{action}: {downloaded.path}", file=sys.stderr)
    try:
        verified = discover_models(
            models_dir=models_dir,
            hf_cache=args.hf_cache or default_hf_cache(),
            build_bin=build_bin,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: downloaded model verification failed: {str(exc).strip() or exc.__class__.__name__}", file=sys.stderr)
        return 1

    verified_row = next(
        (
            row
            for row in verified.rows
            if row.local == "AVAILABLE"
            and row.support == "VERIFIED"
            and row.model_id == selected.model_id
            and Path(row.path_or_action).expanduser().resolve() == downloaded_path
        ),
        None,
    )
    if verified_row is None:
        print("error: downloaded model is not the selected verified profile", file=sys.stderr)
        return 1

    args.model = downloaded_path
    args.model_id = selected.model_id
    print(f"Verified: {manifest.display_name}", file=sys.stderr)
    memory_exit = _select_memory_mode(args, verified_row)
    if memory_exit is not None:
        return memory_exit
    print(f"Starting {manifest.display_name}...", file=sys.stderr)
    return None


def _session_id_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("session_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_SESSION_ID


def _mtp_last_completion_payload(client: NativeLlamaClient) -> dict[str, object] | None:
    completion = client.last_mtp_completion
    if not completion.enabled:
        return None
    has_metrics = any(
        value is not None and value != 0
        for value in (
            completion.output_tokens,
            completion.draft_tokens_total,
            completion.accepted_tokens_total,
            completion.rejected_tokens_total,
            completion.acceptance_ratio,
            completion.fresh_acceptance_ratio,
            completion.consumed_acceptance_ratio,
            completion.reused_draft_tokens_total,
            completion.reused_accepted_tokens_total,
            completion.reused_rejected_tokens_total,
            completion.target_decode_calls,
            completion.draft_decode_calls,
            completion.elapsed_ms,
            completion.tokens_per_second,
            completion.full_accept_steps,
            completion.replay_steps,
            completion.partial_accept_steps,
            completion.partial_no_replay_steps,
            completion.replay_fallback_steps,
            completion.rollback_tokens_total,
            completion.checkpoint_count,
            completion.restore_count,
        )
    )
    if not completion.success and completion.error is None and not has_metrics:
        return None
    return {
        "success": completion.success,
        "error": completion.error,
        "output_tokens": completion.output_tokens,
        "draft_tokens_total": completion.draft_tokens_total,
        "accepted_tokens_total": completion.accepted_tokens_total,
        "rejected_tokens_total": completion.rejected_tokens_total,
        "acceptance_ratio": completion.acceptance_ratio,
        "fresh_acceptance_ratio": completion.fresh_acceptance_ratio,
        "consumed_acceptance_ratio": completion.consumed_acceptance_ratio,
        "reused_draft_tokens_total": completion.reused_draft_tokens_total,
        "reused_accepted_tokens_total": completion.reused_accepted_tokens_total,
        "reused_rejected_tokens_total": completion.reused_rejected_tokens_total,
        "target_decode_calls": completion.target_decode_calls,
        "draft_decode_calls": completion.draft_decode_calls,
        "elapsed_ms": completion.elapsed_ms,
        "tokens_per_second": completion.tokens_per_second,
        "full_accept_steps": completion.full_accept_steps,
        "replay_steps": completion.replay_steps,
        "partial_accept_steps": completion.partial_accept_steps,
        "partial_no_replay_steps": completion.partial_no_replay_steps,
        "replay_fallback_steps": completion.replay_fallback_steps,
        "rollback_tokens_total": completion.rollback_tokens_total,
        "checkpoint_count": completion.checkpoint_count,
        "restore_count": completion.restore_count,
        # The two persistent-pair verdicts. They answer different questions and
        # must never be conflated: `resident_reuse_active` is "did THIS
        # completion reuse resident state", `pair_canonical` is "is the pair
        # trustworthy for the NEXT one". A cold completion can end canonical,
        # and a resident one can end poisoned. Without these, reuse could only
        # be inferred from `cached_tokens`, which is non-zero for unrelated
        # reasons and would read as a false positive.
        "resident_reuse_active": completion.resident_reuse_active,
        "pair_canonical": completion.pair_canonical,
        "resident_token_count": len(completion.resident_tokens),
    }


def _mtp_last_timing_payload(client: NativeLlamaClient) -> dict[str, object] | None:
    completion = client.last_mtp_completion
    if not completion.enabled or not completion.timing_json:
        return None
    try:
        payload = json.loads(completion.timing_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return None

    def _number(name: str) -> float | int | None:
        value = summary.get(name)
        if isinstance(value, (int, float)):
            return value
        return None

    total_ms = _number("total_wall_ms")
    suffix_target_prefill_ms = _number("suffix_target_prefill_ms")
    speculative_loop_ms = _number("speculative_loop_ms")
    speculative_loop_including_suffix_ms = _number("speculative_loop_including_suffix_ms")
    target_validate_ms = _number("target_validate_ms")
    draft_generation_ms = _number("draft_generation_ms")
    checkpoint_restore_ms = _number("checkpoint_restore_ms")
    sampler_ms = _number("sampler_ms")
    seq_rm_ms = _number("seq_rm_ms")
    non_loop_overhead_ms = _number("non_loop_overhead_ms")

    other_ms = None
    if isinstance(total_ms, (int, float)):
        known_components = [
            suffix_target_prefill_ms,
            speculative_loop_ms,
            target_validate_ms,
            draft_generation_ms,
            checkpoint_restore_ms,
            sampler_ms,
            seq_rm_ms,
            non_loop_overhead_ms,
        ]
        known_total = sum(float(value) for value in known_components if isinstance(value, (int, float)))
        other_ms = max(0.0, float(total_ms) - known_total)

    return {
        "total_ms": total_ms,
        "suffix_target_prefill_ms": suffix_target_prefill_ms,
        "speculative_loop_ms": speculative_loop_ms,
        "speculative_loop_including_suffix_ms": speculative_loop_including_suffix_ms,
        "target_validate_ms": target_validate_ms,
        "draft_generation_ms": draft_generation_ms,
        "checkpoint_restore_ms": checkpoint_restore_ms,
        "sampler_ms": sampler_ms,
        "seq_rm_ms": seq_rm_ms,
        "non_loop_overhead_ms": non_loop_overhead_ms,
        "other_ms": other_ms,
    }


def _mtp_last_validate_efficiency_payload(client: NativeLlamaClient) -> dict[str, object] | None:
    completion = client.last_mtp_completion
    if not completion.enabled:
        return None
    has_metrics = completion.success or any(
        value is not None and value != 0
        for value in (
            completion.validate_steps,
            completion.rows_requested_total,
            completion.rows_consumed_estimated_total,
            completion.rows_wasted_estimated_total,
            completion.rows_wasted_estimated_ratio,
            completion.accepted_draft_hist_0,
            completion.accepted_draft_hist_1,
            completion.accepted_draft_hist_2,
            completion.accepted_draft_hist_3,
            completion.accepted_draft_hist_ge4,
        )
    )
    if not has_metrics:
        return None
    return {
        "validate_steps": completion.validate_steps,
        "rows_requested_total": completion.rows_requested_total,
        "rows_consumed_estimated_total": completion.rows_consumed_estimated_total,
        "rows_wasted_estimated_total": completion.rows_wasted_estimated_total,
        "rows_wasted_estimated_ratio": completion.rows_wasted_estimated_ratio,
        "accepted_draft_histogram": {
            "0": completion.accepted_draft_hist_0,
            "1": completion.accepted_draft_hist_1,
            "2": completion.accepted_draft_hist_2,
            "3": completion.accepted_draft_hist_3,
            "ge4": completion.accepted_draft_hist_ge4,
        },
        "full_accept_steps": completion.full_accept_steps,
        "partial_accept_steps": completion.partial_accept_steps,
    }


def _mtp_last_validate_equivalence_payload(client: NativeLlamaClient) -> dict[str, object] | None:
    completion = client.last_mtp_completion
    if not completion.enabled or not completion.validate_equivalence_json:
        return None
    try:
        payload = json.loads(completion.validate_equivalence_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping) or not payload:
        return None

    def _number(name: str) -> float | int | None:
        value = payload.get(name)
        if isinstance(value, (int, float)):
            return value
        return None

    histogram = payload.get("accepted_draft_histogram")
    if not isinstance(histogram, Mapping):
        histogram_payload: dict[str, object] | None = None
    else:
        histogram_payload = {
            "0": histogram.get("0") if isinstance(histogram.get("0"), int) else None,
            "1": histogram.get("1") if isinstance(histogram.get("1"), int) else None,
            "2": histogram.get("2") if isinstance(histogram.get("2"), int) else None,
            "3": histogram.get("3") if isinstance(histogram.get("3"), int) else None,
            "ge4": histogram.get("ge4") if isinstance(histogram.get("ge4"), int) else None,
        }

    step_sample = payload.get("step_sample")
    if not isinstance(step_sample, list):
        step_sample_payload: list[object] | None = None
    else:
        step_sample_payload = step_sample[:64]

    return {
        "steps": _number("steps"),
        "steps_recorded": _number("steps_recorded"),
        "rows_requested_total": _number("rows_requested_total"),
        "rows_consumed_estimated_total": _number("rows_consumed_estimated_total"),
        "rows_wasted_estimated_total": _number("rows_wasted_estimated_total"),
        "rows_wasted_estimated_ratio": _number("rows_wasted_estimated_ratio"),
        "accepted_draft_histogram": histogram_payload,
        "all_steps_have_frontier": payload.get("all_steps_have_frontier") if isinstance(payload.get("all_steps_have_frontier"), bool) else None,
        "all_steps_have_sampler_hash": payload.get("all_steps_have_sampler_hash") if isinstance(payload.get("all_steps_have_sampler_hash"), bool) else None,
        "step_sample": step_sample_payload,
    }


def _mtp_config_payload(client: NativeLlamaClient) -> dict[str, object] | None:
    if not client.config.use_mtp_experimental:
        return None
    return {
        # Keep this in sync with the hardcoded Orbit native MTP shim config.
        "n_max": 3,
        "n_min": None,
        "p_min": None,
        "backend_sampling": None,
        "ctx_tgt": client._session.ctx_tgt is not None,
        "ctx_dft": client._session.ctx_dft is not None,
    }


CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError)


class _DisconnectWatcher:
    def __init__(self, sock: socket.socket, on_disconnect) -> None:
        self._sock = sock
        self._on_disconnect = on_disconnect
        self._disconnected = threading.Event()
        self._stop = threading.Event()
        self._armed = threading.Event()
        self._armed.set()
        self._thread = threading.Thread(target=self._run, name="orbit-stream-disconnect", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_set(self) -> bool:
        return self._disconnected.is_set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def disarm(self) -> None:
        self._armed.clear()

    def _run(self) -> None:
        while not self._stop.is_set() and not self._disconnected.is_set():
            try:
                readable, _, exceptional = select.select([self._sock], [], [self._sock], 0.1)
                if exceptional:
                    self._mark_disconnected()
                    return
                if not readable:
                    continue
                data = self._sock.recv(1, socket.MSG_PEEK)
                if data == b"":
                    self._mark_disconnected()
                    return
            except BlockingIOError:
                continue
            except CLIENT_DISCONNECT_ERRORS:
                self._mark_disconnected()
                return
            except OSError:
                if not self._stop.is_set():
                    self._mark_disconnected()
                return
            time.sleep(0)

    def _mark_disconnected(self) -> None:
        self._disconnected.set()
        if self._armed.is_set():
            self._on_disconnect()
