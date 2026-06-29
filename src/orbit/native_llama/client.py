from __future__ import annotations

import codecs
from ctypes import POINTER, byref, c_char, cast, c_ubyte, create_string_buffer, c_void_p, sizeof
from dataclasses import dataclass, replace
from pathlib import Path
import threading

from .bindings import (
    GgmlAbortCallback,
    GgmlLogCallback,
    LlamaLibrary,
    LlamaChatMessage,
    LlamaProgressCallback,
    MtmdInputText,
    MtmdLibrary,
    llama_token,
    llama_pos,
)
from .chat_template import NativeMessage, RoutePromptSegments, render_gemma4_chat, render_gemma4_route_prompt_segments
from .events import NativeCompletion, NativeProgress, NativeTimings
from .kv_diag import emit_prompt_cache_event, emit_route_prefix_anchor_event
from .multimodal import flatten_message_content, prepare_multimodal_messages
from .mtp_completion import MtpCompletionResult
from .mtp_decode_probe import MtpDecodeProbeResult, run_mtp_decode_probe
from .mtp_accept_probe import MtpAcceptProbeResult, run_mtp_accept_probe
from .mtp_dry_run import MtpDryRunResult, run_mtp_dry_run
from .mtp_probe import MtpProbeResult, run_mtp_probe
from .native_names import runtime_library_filename
from .paths import NativeLlamaPaths
from .prefix_anchor import (
    PrefixAnchorState,
    capture_prefix_anchor,
    compute_prefix_anchor_key,
    prefix_anchor_enabled,
    restore_prefix_anchor,
)
from .persistent_mtp import (
    PersistentMtpSessionRuntime,
    create_persistent_mtp_session,
    free_persistent_mtp_session,
    reset_persistent_mtp_session,
    run_persistent_mtp_completion,
)
from .session_state import DEFAULT_NATIVE_SESSION_ID, NativeSessionSnapshot, NativeSessionState


DEFAULT_MEDIA_MARKER = "<__media__>"


@dataclass(frozen=True)
class NativeClientConfig:
    context_tokens: int = 8192
    threads: int = 6
    threads_batch: int = 6
    batch_size: int = 256
    ubatch_size: int = 128
    progress_step: int = 64
    gpu_layers: int = 0
    thinking: bool = False
    mtp_probe_enabled: bool = False
    mtp_dry_run_enabled: bool = False
    mtp_accept_probe_enabled: bool = False
    mtp_decode_probe_enabled: bool = False
    use_mtp_experimental: bool = False


@dataclass(frozen=True)
class _RouteAnchorRuntimePlan:
    segments: RoutePromptSegments
    prefix_tokens: list[int]
    prefix_hash: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class NativeRoutePrefixPrefillResult:
    attempted: bool
    succeeded: bool
    skipped: bool
    skip_reason: str | None = None
    failed_reason: str | None = None
    prefix_hash: str | None = None
    prefix_token_count: int | None = None
    checkpoint_size_bytes: int | None = None
    prefill_ms: float | None = None
    decode_calls: int | None = None
    sampled_tokens: int = 0
    generated_tokens: int = 0
    sampler_touched: bool = False
    session_history_touched: bool = False
    restore_ready: bool = False

    def to_metadata(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "failed_reason": self.failed_reason,
            "prefix_hash": self.prefix_hash,
            "prefix_token_count": self.prefix_token_count,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "prefill_ms": self.prefill_ms,
            "decode_calls": self.decode_calls,
            "sampled_tokens": self.sampled_tokens,
            "generated_tokens": self.generated_tokens,
            "sampler_touched": self.sampler_touched,
            "session_history_touched": self.session_history_touched,
            "restore_ready": self.restore_ready,
        }


class NativeLlamaClient:
    def __init__(self, paths: NativeLlamaPaths, config: NativeClientConfig | None = None) -> None:
        self.paths = paths
        self.config = config or NativeClientConfig()
        self.lib = LlamaLibrary(paths.build_bin)
        self.mtmd = MtmdLibrary(paths.build_bin) if (paths.build_bin / runtime_library_filename("mtmd")).exists() else None
        self.cancel_event = threading.Event()
        self._callbacks: list[object] = []
        self._model: c_void_p | None = None
        self._vocab: c_void_p | None = None
        self._mtmd_ctx: c_void_p | None = None
        self._media_marker = (
            self.mtmd.lib.mtmd_default_marker().decode("utf-8", errors="replace")
            if self.mtmd is not None else DEFAULT_MEDIA_MARKER
        )
        self.supports_vision = False
        self.supports_audio = False
        self._session = NativeSessionState(session_id=DEFAULT_NATIVE_SESSION_ID)
        self.mtp_probe = MtpProbeResult(enabled=self.config.mtp_probe_enabled, initialized=False, error=None)
        self.mtp_dry_run = MtpDryRunResult(enabled=self.config.mtp_dry_run_enabled, success=False, error=None)
        self.mtp_accept_probe = MtpAcceptProbeResult(enabled=self.config.mtp_accept_probe_enabled, success=False, error=None)
        self.mtp_decode_probe = MtpDecodeProbeResult(enabled=self.config.mtp_decode_probe_enabled, success=False, error=None)
        self.last_mtp_completion = MtpCompletionResult(enabled=self.config.use_mtp_experimental, success=False, error=None)
        self.mtp_fallback_reason: str | None = None
        self._persistent_mtp_runtime: PersistentMtpSessionRuntime | None = None
        self._last_completion_used_mtp = False
        self._last_completion_generation_cap = 0
        self._route_prefix_anchor_state = PrefixAnchorState()
        self._route_prefix_prefill_lock = threading.Lock()

    def session_snapshot(self, session_id: str = DEFAULT_NATIVE_SESSION_ID) -> NativeSessionSnapshot:
        if session_id != self._session.session_id:
            raise ValueError("only the default native session is supported in this experiment")
        return self._session.snapshot(backend_mode=self._current_backend_mode())

    def _current_backend_mode(self) -> str:
        if not self.config.use_mtp_experimental:
            return "no-mtp"
        if self._last_completion_used_mtp:
            return "mtp"
        if self._session.mtp_enabled:
            return "mtp-ready"
        return "no-mtp"

    def set_quiet_logging(self) -> None:
        def log_cb(_level: int, _text: bytes, _data) -> None:
            return None

        cb = GgmlLogCallback(log_cb)
        self._callbacks.append(cb)
        self.lib.lib.llama_log_set(cb, None)

    def close(self) -> None:
        lib = self.lib.lib
        self._free_persistent_mtp_session()
        if self._mtmd_ctx:
            assert self.mtmd is not None
            self.mtmd.lib.mtmd_free(self._mtmd_ctx)
            self._mtmd_ctx = None
        if self._session.sampler:
            lib.llama_sampler_free(self._session.sampler)
            self._session.sampler = None
        if self._session.ctx_tgt:
            lib.llama_free(self._session.ctx_tgt)
            self._session.ctx_tgt = None
        if self._model:
            lib.llama_model_free(self._model)
            self._model = None
        lib.llama_backend_free()

    def cancel(self) -> None:
        self._session.cancel_requested = True
        self._session.continuation_ready = False
        self.cancel_event.set()

    def reset_cancel(self) -> None:
        self._session.cancel_requested = False
        self.cancel_event.clear()

    def load(self, on_progress=None) -> None:
        lib = self.lib.lib
        lib.ggml_backend_load_all()

        def load_cb(progress: float, _data) -> bool:
            if on_progress:
                on_progress(NativeProgress("load", int(progress * 100), 100))
            return not self.cancel_event.is_set()

        progress_cb = LlamaProgressCallback(load_cb)
        abort_cb = GgmlAbortCallback(lambda _data: self.cancel_event.is_set())
        self._callbacks.extend([progress_cb, abort_cb])

        model_params = lib.llama_model_default_params()
        model_params.n_gpu_layers = self.config.gpu_layers
        model_params.progress_callback = progress_cb
        model_params.progress_callback_user_data = None

        self._model = lib.llama_model_load_from_file(str(self.paths.model).encode(), model_params)
        if not self._model:
            raise RuntimeError(f"failed to load model: {self.paths.model}")

        ctx_params = lib.llama_context_default_params()
        ctx_params.n_ctx = self.config.context_tokens
        ctx_params.n_batch = self.config.batch_size
        ctx_params.n_ubatch = self.config.ubatch_size
        ctx_params.n_threads = self.config.threads
        ctx_params.n_threads_batch = self.config.threads_batch
        ctx_params.n_outputs_max = 1 + 3
        ctx_params.abort_callback = abort_cb
        ctx_params.abort_callback_data = None
        ctx_params.no_perf = False

        self._session.ctx_tgt = lib.llama_init_from_model(self._model, ctx_params)
        if not self._session.ctx_tgt:
            raise RuntimeError("failed to create llama context")

        self._vocab = lib.llama_model_get_vocab(self._model)
        sampler_params = lib.llama_sampler_chain_default_params()
        sampler_params.no_perf = False
        self._session.sampler = lib.llama_sampler_chain_init(sampler_params)
        lib.llama_sampler_chain_add(self._session.sampler, lib.llama_sampler_init_greedy())
        self._initialize_multimodal_context()
        self._initialize_mtp_probe()
        self._initialize_mtp_dry_run()
        self._initialize_mtp_accept_probe()
        self._initialize_mtp_decode_probe()
        self._initialize_persistent_mtp_session()

    def _initialize_mtp_probe(self) -> None:
        if not self.config.mtp_probe_enabled:
            self.mtp_probe = MtpProbeResult(enabled=False, initialized=False, error=None)
            return
        self.mtp_probe = run_mtp_probe(llama_root=self.paths.llama_root, paths=self.paths)

    def _initialize_mtp_dry_run(self) -> None:
        if not self.config.mtp_dry_run_enabled:
            self.mtp_dry_run = MtpDryRunResult(enabled=False, success=False, error=None)
            return
        self.mtp_dry_run = run_mtp_dry_run(llama_root=self.paths.llama_root, paths=self.paths)

    def _initialize_mtp_accept_probe(self) -> None:
        if not self.config.mtp_accept_probe_enabled:
            self.mtp_accept_probe = MtpAcceptProbeResult(enabled=False, success=False, error=None)
            return
        self.mtp_accept_probe = run_mtp_accept_probe(llama_root=self.paths.llama_root, paths=self.paths)

    def _initialize_mtp_decode_probe(self) -> None:
        if not self.config.mtp_decode_probe_enabled:
            self.mtp_decode_probe = MtpDecodeProbeResult(enabled=False, success=False, error=None)
            return
        self.mtp_decode_probe = run_mtp_decode_probe(llama_root=self.paths.llama_root, paths=self.paths)

    def _initialize_persistent_mtp_session(self) -> None:
        self._free_persistent_mtp_session()
        self._session.ctx_dft = None
        self._session.spec = None
        self._session.mtp_enabled = False
        self._session.mtp_failed = False
        self._session.mtp_failure_reason = None
        if not self.config.use_mtp_experimental:
            return
        if not self.paths.mtp_available or self.paths.draft_mtp_model is None:
            self._session.mtp_failure_reason = self.paths.fallback_reason or "draft-mtp-unavailable"
            return
        if not self._session.ctx_tgt:
            self._session.mtp_failed = True
            self._session.mtp_failure_reason = "target-context-missing"
            return
        try:
            runtime = create_persistent_mtp_session(
                llama_root=self.paths.llama_root,
                paths=self.paths,
                ctx_tgt=self._session.ctx_tgt,
                context_tokens=self.config.context_tokens,
                batch_size=self.config.batch_size,
                ubatch_size=self.config.ubatch_size,
                threads=self.config.threads,
                threads_batch=self.config.threads_batch,
            )
        except Exception as exc:
            self._session.mtp_failed = True
            self._session.mtp_failure_reason = str(exc)
            return
        self._persistent_mtp_runtime = runtime
        self._session.ctx_dft = runtime.ctx_dft
        self._session.spec = runtime.spec
        self._session.mtp_enabled = True
        self._session.mtp_failed = False

    def reset_session_state(self) -> None:
        if not self._session.ctx_tgt:
            raise RuntimeError("native client not loaded")
        lib = self.lib.lib
        self.reset_cancel()
        mem = lib.llama_get_memory(self._session.ctx_tgt)
        if mem:
            lib.llama_memory_clear(mem, True)
        self._session.cached_prompt_tokens.clear()
        self._session.prompt_cache_mode = None
        self._session.continuation_ready = False
        self._session.last_metrics = None
        if self._persistent_mtp_runtime is None:
            return
        try:
            runtime = reset_persistent_mtp_session(
                llama_root=self.paths.llama_root,
                paths=self.paths,
                runtime=self._persistent_mtp_runtime,
                ctx_tgt=self._session.ctx_tgt,
            )
        except Exception as exc:
            self._session.mtp_enabled = False
            self._session.mtp_failed = True
            self._session.mtp_failure_reason = str(exc)
            self._session.ctx_dft = None
            self._session.spec = None
            self._persistent_mtp_runtime = None
            return
        self._persistent_mtp_runtime = runtime
        self._session.ctx_dft = runtime.ctx_dft
        self._session.spec = runtime.spec
        self._session.mtp_enabled = True
        self._session.mtp_failed = False
        self._session.mtp_failure_reason = None

    def _ensure_prompt_cache_mode(self, mode: str) -> None:
        current = self._session.prompt_cache_mode
        if current is None:
            self._session.prompt_cache_mode = mode
            return
        if current == mode:
            return
        self.reset_session_state()
        self._session.prompt_cache_mode = mode

    def _initialize_multimodal_context(self) -> None:
        self.supports_vision = False
        self.supports_audio = False
        if self._mtmd_ctx:
            assert self.mtmd is not None
            self.mtmd.lib.mtmd_free(self._mtmd_ctx)
            self._mtmd_ctx = None
        if self.mtmd is None or self.paths.mmproj_model is None or not self._model:
            return
        params = self.mtmd.lib.mtmd_context_params_default()
        params.use_gpu = False
        params.print_timings = False
        params.n_threads = self.config.threads
        params.media_marker = self._media_marker.encode()
        ctx = self.mtmd.lib.mtmd_init_from_file(str(self.paths.mmproj_model).encode(), self._model, params)
        if not ctx:
            raise RuntimeError(f"failed to load multimodal projector: {self.paths.mmproj_model}")
        self._mtmd_ctx = ctx
        self.supports_vision = bool(self.mtmd.lib.mtmd_support_vision(ctx))
        self.supports_audio = bool(self.mtmd.lib.mtmd_support_audio(ctx))

    def _free_persistent_mtp_session(self) -> None:
        if self._persistent_mtp_runtime is None:
            return
        try:
            free_persistent_mtp_session(
                llama_root=self.paths.llama_root,
                paths=self.paths,
                runtime=self._persistent_mtp_runtime,
            )
        finally:
            self._persistent_mtp_runtime = None
            self._session.ctx_dft = None
            self._session.spec = None
            self._session.mtp_enabled = False

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        return self.complete_prompt(
            prompt,
            max_tokens=max_tokens,
            on_progress=on_progress,
            on_token=on_token,
            should_cancel=should_cancel,
        )

    def capture_route_prefix_prefill_only(
        self,
        segments: RoutePromptSegments,
        *,
        tools_mode: str = "on",
        should_cancel=None,
    ) -> NativeRoutePrefixPrefillResult:
        if tools_mode != "on":
            return _route_prefix_prefill_skipped("tools_mode_ineligible")
        if not prefix_anchor_enabled():
            return _route_prefix_prefill_skipped("anchor_disabled")
        if not self._session.ctx_tgt or not self._vocab:
            return _route_prefix_prefill_failed("native_client_not_loaded")
        if self._session.in_flight:
            return _route_prefix_prefill_skipped("native_request_in_flight")
        if self._session.continuation_ready or self._session.cached_prompt_tokens:
            return _route_prefix_prefill_skipped("active_context_present")
        if not self._route_prefix_prefill_lock.acquire(blocking=False):
            return _route_prefix_prefill_skipped("prefill_in_flight")
        try:
            if self._session.in_flight:
                return _route_prefix_prefill_skipped("native_request_in_flight")
            if self._session.continuation_ready or self._session.cached_prompt_tokens:
                return _route_prefix_prefill_skipped("active_context_present")
            if not segments.boundary_available:
                return _route_prefix_prefill_failed("route_boundary_unavailable")
            prompt_tokens = self.tokenize(segments.full_prompt_text)
            if not prompt_tokens:
                return _route_prefix_prefill_failed("empty_full_prompt_tokens")
            plan = self._route_anchor_plan(segments.full_prompt_text, prompt_tokens, segments)
            if plan is None:
                return _route_prefix_prefill_failed("route_anchor_plan_unavailable")

            self.reset_cancel()
            self._clear_target_memory()
            token_array = (llama_token * len(plan.prefix_tokens))(*plan.prefix_tokens)
            step = max(1, min(self.config.progress_step, self.config.batch_size))
            processed = 0
            decode_calls = 0
            lib = self.lib.lib
            start_us = int(lib.llama_time_us()) if hasattr(lib, "llama_time_us") else 0
            try:
                while processed < len(plan.prefix_tokens) and not self.cancel_event.is_set():
                    if should_cancel and should_cancel():
                        self.cancel()
                        break
                    end = min(processed + step, len(plan.prefix_tokens))
                    processed = self._decode_prompt_range(
                        token_array,
                        processed=processed,
                        end=end,
                        step=step,
                        total=len(plan.prefix_tokens),
                        on_progress=None,
                        should_cancel=should_cancel,
                    )
                    decode_calls += 1
            except Exception as exc:
                self._clear_target_memory()
                self._route_prefix_anchor_state = PrefixAnchorState()
                return _route_prefix_prefill_failed(
                    f"prefix_decode_failed:{type(exc).__name__}",
                    prefix_hash=plan.prefix_hash,
                    prefix_token_count=len(plan.prefix_tokens),
                    decode_calls=max(1, decode_calls),
                )
            end_us = int(lib.llama_time_us()) if hasattr(lib, "llama_time_us") else start_us
            prefill_ms = max(0.0, (end_us - start_us) / 1000.0)
            if processed != len(plan.prefix_tokens) or self.cancel_event.is_set():
                self._clear_target_memory()
                self._route_prefix_anchor_state = PrefixAnchorState()
                return _route_prefix_prefill_failed(
                    "cancelled",
                    prefix_hash=plan.prefix_hash,
                    prefix_token_count=len(plan.prefix_tokens),
                    prefill_ms=prefill_ms,
                    decode_calls=decode_calls,
                )

            state, capture_meta = capture_prefix_anchor(
                lib=self.lib.lib,
                ctx=self._session.ctx_tgt,
                prefix_hash=plan.prefix_hash,
                token_count=len(plan.prefix_tokens),
                enabled=True,
                **self._route_anchor_state_kwargs(plan),
            )
            if not state.valid:
                reason = str(capture_meta.get("fallback_reason") or state.invalidation_reason or "capture_failed")
                self._clear_target_memory()
                self._route_prefix_anchor_state = PrefixAnchorState()
                return _route_prefix_prefill_failed(
                    reason,
                    prefix_hash=plan.prefix_hash,
                    prefix_token_count=len(plan.prefix_tokens),
                    prefill_ms=prefill_ms,
                    decode_calls=decode_calls,
                )

            self._route_prefix_anchor_state = state
            self._session.cached_prompt_tokens = list(plan.prefix_tokens)
            self._session.continuation_ready = False
            return NativeRoutePrefixPrefillResult(
                attempted=True,
                succeeded=True,
                skipped=False,
                prefix_hash=plan.prefix_hash,
                prefix_token_count=len(plan.prefix_tokens),
                checkpoint_size_bytes=state.checkpoint_size,
                prefill_ms=prefill_ms,
                decode_calls=decode_calls,
                restore_ready=True,
            )
        finally:
            self._route_prefix_prefill_lock.release()

    def complete_chat(
        self,
        messages: list[NativeMessage],
        *,
        max_tokens: int = 16,
        tools: list[dict] | None = None,
        thinking: bool | None = None,
        route_prefix_anchor: bool = False,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        thinking = self._thinking_enabled(thinking)
        prepared_multimodal = prepare_multimodal_messages(messages, media_marker=self._media_marker)
        mode = (
            f"multimodal:thinking={'on' if thinking else 'off'}"
            if prepared_multimodal is not None
            else f"{'tools' if tools else 'chat'}:thinking={'on' if thinking else 'off'}"
        )
        self._ensure_prompt_cache_mode(mode)
        if prepared_multimodal is not None:
            if prepared_multimodal.has_image and not self.supports_vision:
                raise RuntimeError("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj")
            if prepared_multimodal.has_audio and not self.supports_audio:
                raise RuntimeError("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj")
            prompt = self.apply_chat_template(prepared_multimodal.messages, tools=tools, thinking=thinking)
            self.reset_cancel()
            self._session.in_flight = True
            try:
                timings = self._complete_prompt_multimodal(
                    prompt,
                    media_payloads=prepared_multimodal.media_payloads,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    on_progress=on_progress,
                    on_token=on_token,
                    should_cancel=should_cancel,
                )
                self._session.last_metrics = timings
                self._session.continuation_ready = _can_continue_from_timings(timings)
                return timings
            finally:
                self._session.in_flight = False
        prompt = self.apply_chat_template(messages, tools=tools, thinking=thinking)
        route_anchor_segments = self._route_anchor_segments_for_prompt(
            messages,
            tools=tools,
            thinking=thinking,
            prompt=prompt,
        ) if route_prefix_anchor else None
        return self.complete_prompt(
            prompt,
            max_tokens=max_tokens,
            allow_mtp_experimental=not tools,
            thinking=thinking,
            route_anchor_segments=route_anchor_segments,
            on_progress=on_progress,
            on_token=on_token,
            should_cancel=should_cancel,
        )

    def complete_chat_text(
        self,
        messages: list[NativeMessage],
        *,
        max_tokens: int = 16,
        stop: tuple[str, ...] = (),
        tools: list[dict] | None = None,
        thinking: bool | None = None,
        route_prefix_anchor: bool = False,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeCompletion:
        thinking = self._thinking_enabled(thinking)
        result = self._complete_chat_text_once(
            messages,
            max_tokens=max_tokens,
            stop=stop,
            tools=tools,
            thinking=thinking,
            route_prefix_anchor=route_prefix_anchor,
            on_progress=on_progress,
            on_token=on_token,
            should_cancel=should_cancel,
        )
        latest = result
        extra_budget = max(1, min(max_tokens, 64))
        allow_auto_continuation = max_tokens >= 128
        continuation_attempts = 0
        while allow_auto_continuation and self._should_continue_thought_after_completion(
            latest,
            max_tokens=max_tokens if continuation_attempts == 0 else extra_budget,
            thinking=thinking,
            content_override=result.content if continuation_attempts > 0 else None,
        ):
            continuation_chunks: list[str] = []
            continuation = self._continue_chat_text_from_current_context(
                max_tokens=extra_budget,
                stop=stop,
                thinking=thinking,
                on_progress=on_progress,
                on_token=continuation_chunks.append,
                should_cancel=should_cancel,
            )
            if thinking and _looks_like_degenerate_thought_continuation(continuation.content):
                break
            if on_token:
                for chunk in continuation_chunks:
                    on_token(chunk)
            result = _merge_completions(result, continuation)
            latest = continuation
            continuation_attempts += 1
            if continuation_attempts >= 1 or not continuation.content:
                break
        return result

    def _complete_chat_text_once(
        self,
        messages: list[NativeMessage],
        *,
        max_tokens: int,
        stop: tuple[str, ...],
        tools: list[dict] | None,
        thinking: bool,
        route_prefix_anchor: bool = False,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeCompletion:
        parts: list[str] = []
        channel_filter = None if thinking else _ControlChannelStreamFilter()
        thought_label_filter = None if thinking else _LeadingThoughtLabelFilter()
        stop_filter = _StopSequenceStreamFilter(stop, emit=parts.append) if stop else None

        def collect(text: str) -> None:
            if channel_filter is None:
                visible_chunks = [text]
            else:
                visible_chunks = channel_filter.write(text)
            if thought_label_filter is not None:
                normalized_chunks: list[str] = []
                for visible_text in visible_chunks:
                    normalized_chunks.extend(thought_label_filter.write(visible_text))
                visible_chunks = normalized_chunks
            for visible_text in visible_chunks:
                if stop_filter:
                    for delta in stop_filter.write(visible_text):
                        if on_token:
                            on_token(delta)
                    if stop_filter.stopped:
                        self.cancel()
                    continue
                parts.append(visible_text)
                if on_token:
                    on_token(visible_text)
            if stop_filter and stop_filter.stopped:
                self.cancel()
                return

        def flush_filters() -> None:
            if channel_filter is None:
                visible_chunks = []
            else:
                visible_chunks = channel_filter.finish()
            if thought_label_filter is not None:
                normalized_chunks: list[str] = []
                for visible_text in visible_chunks:
                    normalized_chunks.extend(thought_label_filter.write(visible_text))
                normalized_chunks.extend(thought_label_filter.finish())
                visible_chunks = normalized_chunks
            for visible_text in visible_chunks:
                if stop_filter:
                    for delta in stop_filter.write(visible_text):
                        if on_token:
                            on_token(delta)
                    continue
                parts.append(visible_text)
                if on_token:
                    on_token(visible_text)
            if stop_filter:
                for delta in stop_filter.finish():
                    if on_token:
                        on_token(delta)

        timings = self.complete_chat(
            messages,
            max_tokens=max_tokens,
            tools=tools,
            thinking=thinking,
            route_prefix_anchor=route_prefix_anchor,
            on_progress=on_progress,
            on_token=collect,
            should_cancel=should_cancel,
        )
        flush_filters()
        content = _trim_at_stop("".join(parts), stop)
        if not thinking:
            content = _strip_reasoning_preamble(content)
        completion = NativeCompletion(content=content, timings=timings, stopped_by_stop=bool(stop_filter and stop_filter.stopped))
        self._session.continuation_ready = _can_continue_from_completion(completion, thinking=thinking)
        return completion

    def _continue_chat_text_from_current_context(
        self,
        *,
        max_tokens: int,
        stop: tuple[str, ...],
        thinking: bool,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeCompletion:
        parts: list[str] = []
        stop_filter = _StopSequenceStreamFilter(stop, emit=parts.append) if stop else None

        def collect(text: str) -> None:
            visible_text = text
            if stop_filter:
                for delta in stop_filter.write(visible_text):
                    if on_token:
                        on_token(delta)
                if stop_filter.stopped:
                    self.cancel()
                return
            parts.append(visible_text)
            if on_token:
                on_token(visible_text)

        timings = self._continue_generation_from_current_context(
            max_tokens=max_tokens,
            on_progress=on_progress,
            on_token=collect,
            should_cancel=should_cancel,
        )
        if stop_filter:
            for delta in stop_filter.finish():
                if on_token:
                    on_token(delta)
        content = _trim_at_stop("".join(parts), stop)
        if not thinking:
            content = _strip_reasoning_preamble(content)
        return NativeCompletion(content=content, timings=timings, stopped_by_stop=bool(stop_filter and stop_filter.stopped))

    def continue_chat_text_current_context(
        self,
        *,
        max_tokens: int = 16,
        stop: tuple[str, ...] = (),
        thinking: bool | None = None,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeCompletion:
        thinking = self._thinking_enabled(thinking)
        return self._continue_chat_text_from_current_context(
            max_tokens=max_tokens,
            stop=stop,
            thinking=thinking,
            on_progress=on_progress,
            on_token=on_token,
            should_cancel=should_cancel,
        )

    def _complete_prompt_multimodal(
        self,
        prompt: str,
        *,
        media_payloads: list[bytes],
        max_tokens: int = 16,
        thinking: bool | None = None,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        thinking = self._thinking_enabled(thinking)
        if not self._session.ctx_tgt or not self._session.sampler or not self._mtmd_ctx or self.mtmd is None:
            raise RuntimeError("native multimodal client not loaded")

        lib = self.lib.lib
        mtmd = self.mtmd.lib
        self._session.cached_prompt_tokens.clear()
        mem = lib.llama_get_memory(self._session.ctx_tgt)
        if mem:
            lib.llama_memory_clear(mem, True)

        bitmap_buffers: list[object] = []
        bitmaps: list[c_void_p] = []
        for payload in media_payloads:
            buf = (c_ubyte * len(payload)).from_buffer_copy(payload)
            bitmap_buffers.append(buf)
            bitmap = mtmd.mtmd_helper_bitmap_init_from_buf(self._mtmd_ctx, buf, len(payload), False)
            if not bitmap:
                raise RuntimeError("failed to decode multimodal input")
            bitmaps.append(bitmap)

        chunks = mtmd.mtmd_input_chunks_init()
        if not chunks:
            for bitmap in bitmaps:
                mtmd.mtmd_bitmap_free(bitmap)
            raise RuntimeError("failed to allocate multimodal chunks")

        try:
            text = MtmdInputText(prompt.encode(), True, True)
            bitmap_array = (c_void_p * len(bitmaps))(*bitmaps) if bitmaps else None
            rc = mtmd.mtmd_tokenize(self._mtmd_ctx, chunks, byref(text), bitmap_array, len(bitmaps))
            if rc != 0:
                raise RuntimeError("failed to tokenize multimodal prompt")

            total_tokens = int(mtmd.mtmd_helper_get_n_tokens(chunks))
            processed_tokens = 0
            n_chunks = int(mtmd.mtmd_input_chunks_size(chunks))
            n_past = llama_pos(0)
            if on_progress:
                on_progress(NativeProgress("prefill", 0, max(1, total_tokens)))
            pf_start = lib.llama_time_us()
            for idx in range(n_chunks):
                if should_cancel and should_cancel():
                    self.cancel()
                    break
                chunk = mtmd.mtmd_input_chunks_get(chunks, idx)
                new_n_past = llama_pos(0)
                rc = mtmd.mtmd_helper_eval_chunk_single(
                    self._mtmd_ctx,
                    self._session.ctx_tgt,
                    chunk,
                    n_past,
                    0,
                    self._multimodal_chunk_batch_size(),
                    idx == (n_chunks - 1),
                    byref(new_n_past),
                )
                if rc != 0:
                    raise RuntimeError(f"multimodal prefill failed: {rc}")
                n_past = new_n_past
                processed_tokens += int(mtmd.mtmd_input_chunk_get_n_tokens(chunk))
                if on_progress:
                    on_progress(NativeProgress("prefill", min(processed_tokens, total_tokens), max(1, total_tokens)))
            pf_ms = (lib.llama_time_us() - pf_start) / 1000.0
            if self.cancel_event.is_set():
                return NativeTimings(
                    prompt_tokens=total_tokens,
                    output_tokens=0,
                    reused_prompt_tokens=0,
                    evaluated_prompt_tokens=total_tokens,
                    prefill_ms=pf_ms,
                    generation_ms=0.0,
                    cancelled=True,
                )

            generated, gen_ms, cancelled = self._generate_from_current_context(
                max_tokens=max_tokens,
                on_progress=on_progress,
                on_token=on_token,
                should_cancel=should_cancel,
            )
            return NativeTimings(
                prompt_tokens=total_tokens,
                output_tokens=generated,
                reused_prompt_tokens=0,
                evaluated_prompt_tokens=total_tokens,
                prefill_ms=pf_ms,
                generation_ms=gen_ms,
                cancelled=cancelled,
            )
        finally:
            mtmd.mtmd_input_chunks_free(chunks)
            for bitmap in bitmaps:
                mtmd.mtmd_bitmap_free(bitmap)

    def _multimodal_chunk_batch_size(self) -> int:
        return max(1, min(self.config.batch_size, self.config.ubatch_size))

    def _thinking_enabled(self, thinking: bool | None) -> bool:
        if thinking is None:
            return self.config.thinking
        return thinking

    def _should_continue_thought_after_completion(
        self,
        result: NativeCompletion,
        *,
        max_tokens: int,
        thinking: bool,
        content_override: str | None = None,
    ) -> bool:
        if not thinking or result.stopped_by_stop or result.timings.cancelled:
            return False
        content_to_check = result.content if content_override is None else content_override
        if not _has_open_thought_channel(content_to_check):
            return False
        threshold = self._last_completion_generation_cap if self._last_completion_used_mtp else max_tokens
        if threshold <= 0:
            threshold = max_tokens
        return result.timings.output_tokens >= threshold

    def complete_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        allow_mtp_experimental: bool = True,
        thinking: bool | None = None,
        route_anchor_segments: RoutePromptSegments | None = None,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        thinking = self._thinking_enabled(thinking)
        self.reset_cancel()
        self._session.in_flight = True
        try:
            if allow_mtp_experimental and thinking:
                self.mtp_fallback_reason = "thinking-mode"
                self.last_mtp_completion = MtpCompletionResult(
                    enabled=self.config.use_mtp_experimental,
                    success=False,
                    error="thinking-mode",
                )
            if allow_mtp_experimental and not thinking:
                if should_cancel and should_cancel():
                    self.cancel()
                else:
                    mtp_result = self._try_complete_with_mtp_experimental(
                        prompt,
                        max_tokens=max_tokens,
                        thinking=thinking,
                        on_progress=on_progress,
                        on_token=on_token,
                    )
                    if mtp_result is not None:
                        self._last_completion_used_mtp = True
                        self._session.last_metrics = mtp_result
                        self._session.continuation_ready = False
                        return mtp_result
            self._last_completion_used_mtp = False
            self._last_completion_generation_cap = max_tokens
            timings = self._complete_prompt_standard(
                prompt,
                max_tokens=max_tokens,
                route_anchor_segments=route_anchor_segments,
                on_progress=on_progress,
                on_token=on_token,
                should_cancel=should_cancel,
            )
            self._session.last_metrics = timings
            self._session.continuation_ready = _can_continue_from_timings(timings)
            return timings
        finally:
            self._session.in_flight = False

    def _try_complete_with_mtp_experimental(
        self,
        prompt: str,
        *,
        max_tokens: int,
        thinking: bool | None = None,
        on_progress=None,
        on_token=None,
    ) -> NativeTimings | None:
        thinking = self._thinking_enabled(thinking)
        if thinking:
            self.mtp_fallback_reason = "thinking-mode"
            self.last_mtp_completion = MtpCompletionResult(enabled=self.config.use_mtp_experimental, success=False, error="thinking-mode")
            return None
        if not self.config.use_mtp_experimental:
            self.last_mtp_completion = MtpCompletionResult(enabled=False, success=False, error=None)
            return None
        if not self.paths.mtp_available:
            self.mtp_fallback_reason = self.paths.fallback_reason or "draft-mtp-unavailable"
            self.last_mtp_completion = MtpCompletionResult(enabled=True, success=False, error=self.mtp_fallback_reason)
            return None
        if self.cancel_event.is_set():
            self.mtp_fallback_reason = "cancelled"
            self.last_mtp_completion = MtpCompletionResult(enabled=True, success=False, error="cancelled")
            return None
        if self._persistent_mtp_runtime is None or not self._session.mtp_enabled or not self._session.ctx_tgt:
            self.mtp_fallback_reason = self._session.mtp_failure_reason or "persistent-mtp-uninitialized"
            self.last_mtp_completion = MtpCompletionResult(enabled=True, success=False, error=self.mtp_fallback_reason)
            return None

        mtp_prompt = _prepare_mtp_prompt(prompt, thinking=thinking)
        streamed_parts: list[str] = []
        generation_cap = max(1, min(max_tokens, 32))
        self._last_completion_generation_cap = generation_cap
        result = run_persistent_mtp_completion(
            llama_root=self.paths.llama_root,
            paths=self.paths,
            runtime=self._persistent_mtp_runtime,
            ctx_tgt=self._session.ctx_tgt,
            prompt=mtp_prompt,
            max_tokens=generation_cap,
            on_token=(lambda text: (streamed_parts.append(text), on_token(text))[1]) if on_token else None,
            on_progress=(
                lambda phase, current, total: on_progress(
                    NativeProgress("prefill" if phase == 0 else "generation", current, total)
                )
            ) if on_progress else None,
        )
        if result.success and result.content and not thinking:
            result = replace(result, content=_strip_control_channels(result.content))
        self.last_mtp_completion = result
        if not result.success:
            self.mtp_fallback_reason = result.error or "mtp-experimental-failed"
            self._session.mtp_failed = True
            self._session.mtp_failure_reason = self.mtp_fallback_reason
            self._session.mtp_enabled = False
            return None
        self.mtp_fallback_reason = None
        self._session.mtp_failed = False
        self._session.mtp_failure_reason = None
        self._session.mtp_enabled = True
        if result.content and on_token and not streamed_parts:
            on_token(result.content)
        prompt_token_list = self.tokenize(mtp_prompt) if self._vocab else []
        prompt_tokens = len(prompt_token_list)
        reused_prompt_tokens = 0
        max_common = min(len(prompt_token_list), len(self._session.cached_prompt_tokens))
        while reused_prompt_tokens < max_common and prompt_token_list[reused_prompt_tokens] == self._session.cached_prompt_tokens[reused_prompt_tokens]:
            reused_prompt_tokens += 1
        if prompt_token_list:
            reused_prompt_tokens = min(reused_prompt_tokens, len(prompt_token_list) - 1)
        self._session.cached_prompt_tokens = list(prompt_token_list)
        return NativeTimings(
            prompt_tokens=prompt_tokens,
            output_tokens=result.output_tokens,
            reused_prompt_tokens=reused_prompt_tokens,
            evaluated_prompt_tokens=max(0, prompt_tokens - reused_prompt_tokens),
            prefill_ms=0.0,
            generation_ms=result.elapsed_ms or 0.0,
            cancelled=False,
        )

    def _complete_prompt_standard(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        route_anchor_segments: RoutePromptSegments | None = None,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        if not self._session.ctx_tgt or not self._vocab or not self._session.sampler:
            raise RuntimeError("native client not loaded")

        lib = self.lib.lib
        prompt_tokens = self.tokenize(prompt)
        n_prompt = len(prompt_tokens)
        if not prompt_tokens:
            raise RuntimeError("failed to count prompt tokens")

        previous_prompt_tokens = list(self._session.cached_prompt_tokens)
        pf_start = lib.llama_time_us()
        anchor_plan = self._route_anchor_plan(prompt, prompt_tokens, route_anchor_segments)
        anchor_metadata: dict[str, object] | None = None
        processed_start = 0
        if anchor_plan is None:
            reused = self._prepare_memory_for_prompt(prompt_tokens)
        else:
            processed_start, reused, anchor_metadata = self._prepare_memory_with_route_anchor(anchor_plan)
            if processed_start < 0:
                reused = self._prepare_memory_for_prompt(prompt_tokens)
                processed_start = reused
        processed = 0
        step = max(1, min(self.config.progress_step, self.config.batch_size))
        processed = processed_start if anchor_plan is not None and anchor_metadata is not None else reused
        if on_progress:
            on_progress(NativeProgress("prefill", processed, n_prompt))
        token_array = (llama_token * n_prompt)(*prompt_tokens)
        while processed < n_prompt and not self.cancel_event.is_set():
            processed = self._decode_prompt_range(
                token_array,
                processed=processed,
                end=n_prompt,
                step=step,
                total=n_prompt,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
        pf_ms = (lib.llama_time_us() - pf_start) / 1000.0
        generated, gen_ms, cancelled = self._generate_from_current_context(
            max_tokens=max_tokens,
            on_progress=on_progress,
            on_token=on_token,
            should_cancel=should_cancel,
        )
        emit_prompt_cache_event(
            prompt_tokens=prompt_tokens,
            previous_prompt_tokens=previous_prompt_tokens,
            reused_prompt_tokens=reused,
            output_tokens=generated,
            cancelled=cancelled,
            slot_id=self._session.session_id,
        )
        if anchor_metadata is not None:
            anchor_metadata["cached_tokens"] = reused
            anchor_metadata["evaluated_tokens"] = n_prompt - reused
            anchor_metadata["lcp_tokens"] = anchor_plan.metadata.get("prefix_token_count") if anchor_plan is not None else None
            emit_route_prefix_anchor_event(anchor_metadata)
        return NativeTimings(
            prompt_tokens=n_prompt,
            output_tokens=generated,
            reused_prompt_tokens=reused,
            evaluated_prompt_tokens=n_prompt - reused,
            prefill_ms=pf_ms,
            generation_ms=gen_ms,
            cancelled=cancelled,
        )

    def _decode_prompt_range(
        self,
        token_array,
        *,
        processed: int,
        end: int,
        step: int,
        total: int,
        on_progress=None,
        should_cancel=None,
    ) -> int:
        if not self._session.ctx_tgt:
            raise RuntimeError("native client not loaded")
        lib = self.lib.lib
        while processed < end and not self.cancel_event.is_set():
            if should_cancel and should_cancel():
                self.cancel()
                break
            n = min(step, end - processed)
            token_ptr = cast(byref(token_array, processed * sizeof(llama_token)), POINTER(llama_token))
            batch = lib.llama_batch_get_one(token_ptr, n)
            decode_rc = lib.llama_decode(self._session.ctx_tgt, batch)
            processed += n
            if on_progress:
                on_progress(NativeProgress("prefill", processed, total))
            if decode_rc != 0:
                if decode_rc == 2 and self.cancel_event.is_set():
                    break
                raise RuntimeError(f"llama_decode failed during prefill: {decode_rc}")
        return processed

    def _route_anchor_plan(
        self,
        prompt: str,
        prompt_tokens: list[int],
        segments: RoutePromptSegments | None,
    ) -> _RouteAnchorRuntimePlan | None:
        if segments is None or not prefix_anchor_enabled():
            return None
        metadata = _route_anchor_metadata(
            enabled=True,
            attempted=True,
            prefix_hash=segments.stable_prefix_hash,
            fallback_reason=None,
        )
        if not segments.boundary_available:
            metadata["fallback_reason"] = "route_boundary_unavailable"
            emit_route_prefix_anchor_event(metadata)
            return None
        if segments.full_prompt_text != prompt:
            metadata["fallback_reason"] = "route_prompt_mismatch"
            emit_route_prefix_anchor_event(metadata)
            return None
        prefix_tokens = self.tokenize(segments.stable_prefix_text)
        if not prefix_tokens:
            metadata["fallback_reason"] = "empty_prefix_tokens"
            emit_route_prefix_anchor_event(metadata)
            return None
        if prompt_tokens[: len(prefix_tokens)] != prefix_tokens:
            metadata["fallback_reason"] = "token_boundary_mismatch"
            metadata["prefix_token_count"] = len(prefix_tokens)
            emit_route_prefix_anchor_event(metadata)
            return None
        prefix_hash = self._route_anchor_key(segments)
        metadata["prefix_hash"] = prefix_hash
        metadata["prefix_token_count"] = len(prefix_tokens)
        return _RouteAnchorRuntimePlan(
            segments=segments,
            prefix_tokens=prefix_tokens,
            prefix_hash=prefix_hash,
            metadata=metadata,
        )

    def _prepare_memory_with_route_anchor(self, plan: _RouteAnchorRuntimePlan) -> tuple[int, int, dict[str, object]]:
        if not self._session.ctx_tgt:
            raise RuntimeError("native client not loaded")
        metadata = dict(plan.metadata)
        metadata["restore_attempted"] = True
        state_kwargs = self._route_anchor_state_kwargs(plan)
        if self._route_prefix_anchor_state.valid:
            self._clear_target_memory()
            ok, restored_state, restore_meta = restore_prefix_anchor(
                self._route_prefix_anchor_state,
                lib=self.lib.lib,
                ctx=self._session.ctx_tgt,
                prefix_hash=plan.prefix_hash,
                token_count=len(plan.prefix_tokens),
                enabled=True,
                **state_kwargs,
            )
            self._route_prefix_anchor_state = restored_state
            metadata.update(_route_anchor_metadata_from_prefix(restore_meta, prefix_hash=plan.prefix_hash))
            if ok:
                self._session.cached_prompt_tokens = list(plan.prefix_tokens)
                return len(plan.prefix_tokens), len(plan.prefix_tokens), metadata
            return -1, 0, metadata

        self._clear_target_memory()
        token_array = (llama_token * len(plan.prefix_tokens))(*plan.prefix_tokens)
        step = max(1, min(self.config.progress_step, self.config.batch_size))
        self._decode_prompt_range(
            token_array,
            processed=0,
            end=len(plan.prefix_tokens),
            step=step,
            total=len(plan.prefix_tokens),
            on_progress=None,
            should_cancel=None,
        )
        self._session.cached_prompt_tokens = list(plan.prefix_tokens)
        state, capture_meta = capture_prefix_anchor(
            lib=self.lib.lib,
            ctx=self._session.ctx_tgt,
            prefix_hash=plan.prefix_hash,
            token_count=len(plan.prefix_tokens),
            enabled=True,
            **state_kwargs,
        )
        self._route_prefix_anchor_state = state
        metadata.update(_route_anchor_metadata_from_prefix(capture_meta, prefix_hash=plan.prefix_hash))
        metadata["route_anchor_miss"] = True
        if not state.valid:
            metadata["fallback_reason"] = metadata.get("fallback_reason") or state.invalidation_reason or "capture_failed"
            self._clear_target_memory()
            return -1, 0, metadata
        return len(plan.prefix_tokens), 0, metadata

    def _clear_target_memory(self) -> None:
        if not self._session.ctx_tgt:
            raise RuntimeError("native client not loaded")
        mem = self.lib.lib.llama_get_memory(self._session.ctx_tgt)
        if mem:
            self.lib.lib.llama_memory_clear(mem, True)
        self._session.cached_prompt_tokens.clear()

    def _route_anchor_state_kwargs(self, plan: _RouteAnchorRuntimePlan) -> dict[str, str | None]:
        return {
            "model_id": str(self.paths.model),
            "template_id": "gemma4-route-prefix-v1",
            "tool_schema_hash": plan.segments.stable_prefix_hash,
            "capability_summary_hash": plan.segments.stable_prefix_hash,
            "runtime_policy_hash": plan.segments.stable_prefix_hash,
            "route_contract_hash": plan.segments.stable_prefix_hash,
            "backend_version": "orbit-native",
            "native_version": runtime_library_filename("llama"),
            "tools_mode": "on",
        }

    def _route_anchor_key(self, segments: RoutePromptSegments) -> str:
        return compute_prefix_anchor_key(
            model_id=str(self.paths.model),
            template_id="gemma4-route-prefix-v1",
            tool_schema_hash=segments.stable_prefix_hash,
            capability_summary_hash=segments.stable_prefix_hash,
            runtime_policy_hash=segments.stable_prefix_hash,
            route_contract_hash=segments.stable_prefix_hash,
            backend_version="orbit-native",
            native_version=runtime_library_filename("llama"),
            tools_mode="on",
        )

    def _route_anchor_segments_for_prompt(
        self,
        messages: list[NativeMessage],
        *,
        tools: list[dict] | None,
        thinking: bool,
        prompt: str,
    ) -> RoutePromptSegments | None:
        if not prefix_anchor_enabled():
            emit_route_prefix_anchor_event(
                _route_anchor_metadata(
                    enabled=False,
                    attempted=True,
                    prefix_hash=None,
                    fallback_reason="anchor_disabled",
                )
            )
            return None
        if tools or thinking:
            emit_route_prefix_anchor_event(
                _route_anchor_metadata(
                    enabled=True,
                    attempted=True,
                    prefix_hash=None,
                    fallback_reason="route_anchor_ineligible_mode",
                )
            )
            return None
        try:
            segments = render_gemma4_route_prompt_segments([dict(message) for message in messages], thinking=False)
        except Exception:
            emit_route_prefix_anchor_event(
                _route_anchor_metadata(
                    enabled=True,
                    attempted=True,
                    prefix_hash=None,
                    fallback_reason="route_segment_render_failed",
                )
            )
            return None
        if segments.full_prompt_text != prompt:
            emit_route_prefix_anchor_event(
                _route_anchor_metadata(
                    enabled=True,
                    attempted=True,
                    prefix_hash=segments.stable_prefix_hash,
                    fallback_reason="route_prompt_mismatch",
                )
            )
            return None
        return segments

    def _continue_generation_from_current_context(
        self,
        *,
        max_tokens: int,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        if not self._session.continuation_ready:
            raise RuntimeError("no active continuation state")
        self.reset_cancel()
        self._last_completion_used_mtp = False
        self._last_completion_generation_cap = max_tokens
        generated, gen_ms, cancelled = self._generate_from_current_context(
            max_tokens=max_tokens,
            on_progress=on_progress,
            on_token=on_token,
            should_cancel=should_cancel,
        )
        timings = NativeTimings(
            prompt_tokens=0,
            output_tokens=generated,
            reused_prompt_tokens=0,
            evaluated_prompt_tokens=0,
            prefill_ms=0.0,
            generation_ms=gen_ms,
            cancelled=cancelled,
        )
        self._session.last_metrics = timings
        self._session.continuation_ready = _can_continue_from_timings(timings)
        return timings

    def _generate_from_current_context(
        self,
        *,
        max_tokens: int,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> tuple[int, float, bool]:
        if not self._session.ctx_tgt or not self._session.sampler or not self._vocab:
            raise RuntimeError("native client not loaded")
        lib = self.lib.lib
        lib.llama_sampler_reset(self._session.sampler)
        generated = 0
        gen_start = lib.llama_time_us()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while generated < max_tokens and not self.cancel_event.is_set():
            if should_cancel and should_cancel():
                self.cancel()
                break
            token = lib.llama_sampler_sample(self._session.sampler, self._session.ctx_tgt, -1)
            lib.llama_sampler_accept(self._session.sampler, token)
            if lib.llama_vocab_is_eog(self._vocab, token):
                break
            text = decoder.decode(self._token_to_bytes(token), final=False)
            if text and on_token:
                on_token(text)
            one_token = (llama_token * 1)(token)
            batch = lib.llama_batch_get_one(one_token, 1)
            decode_rc = lib.llama_decode(self._session.ctx_tgt, batch)
            generated += 1
            if on_progress:
                on_progress(NativeProgress("generation", generated, max_tokens))
            if decode_rc != 0:
                if decode_rc == 2 and self.cancel_event.is_set():
                    break
                raise RuntimeError(f"llama_decode failed during generation: {decode_rc}")
        text = decoder.decode(b"", final=True)
        if text and on_token:
            on_token(text)
        gen_ms = (lib.llama_time_us() - gen_start) / 1000.0
        return generated, gen_ms, self.cancel_event.is_set()

    def tokenize(self, prompt: str) -> list[int]:
        if not self._vocab:
            raise RuntimeError("native client not loaded")
        lib = self.lib.lib
        prompt_bytes = prompt.encode()
        add_special = not prompt.startswith("<bos>")
        n_prompt = -lib.llama_tokenize(self._vocab, prompt_bytes, len(prompt_bytes), None, 0, add_special, True)
        if n_prompt <= 0:
            return []
        token_array = (llama_token * n_prompt)()
        rc = lib.llama_tokenize(self._vocab, prompt_bytes, len(prompt_bytes), token_array, n_prompt, add_special, True)
        if rc < 0:
            raise RuntimeError("failed to tokenize prompt")
        return [int(token_array[i]) for i in range(n_prompt)]

    def _prepare_memory_for_prompt(self, prompt_tokens: list[int]) -> int:
        if not self._session.ctx_tgt:
            raise RuntimeError("native client not loaded")
        lib = self.lib.lib
        common = 0
        max_common = min(len(prompt_tokens), len(self._session.cached_prompt_tokens))
        while common < max_common and prompt_tokens[common] == self._session.cached_prompt_tokens[common]:
            common += 1
        if prompt_tokens:
            # The final prompt token must be evaluated to produce fresh logits
            # for the next sampled token.
            common = min(common, len(prompt_tokens) - 1)

        mem = lib.llama_get_memory(self._session.ctx_tgt)
        if mem:
            if common == 0:
                lib.llama_memory_clear(mem, True)
            else:
                lib.llama_memory_seq_rm(mem, 0, common, -1)
        self._session.cached_prompt_tokens = list(prompt_tokens)
        return common

    def apply_chat_template(
        self,
        messages: list[NativeMessage],
        *,
        tools: list[dict] | None = None,
        thinking: bool | None = None,
    ) -> str:
        if not self._model:
            raise RuntimeError("native client not loaded")
        thinking = self._thinking_enabled(thinking)
        rendered_messages = [dict(message) for message in messages]
        if thinking:
            return render_gemma4_chat(rendered_messages, tools=tools, thinking=True)
        if tools or any(message.get("role") == "tool" or message.get("tool_calls") for message in rendered_messages):
            return render_gemma4_chat(rendered_messages, tools=tools, thinking=thinking)
        encoded_messages = [
            (str(message.get("role", "user")).encode(), _message_content(message).encode())
            for message in rendered_messages
        ]
        chat_array = (LlamaChatMessage * len(encoded_messages))(
            *[
                LlamaChatMessage(role, content)
                for role, content in encoded_messages
            ]
        )
        tmpl = self.lib.lib.llama_model_chat_template(self._model, None)
        needed = self.lib.lib.llama_chat_apply_template(tmpl, chat_array, len(chat_array), True, None, 0)
        if needed < 0:
            return render_gemma4_chat(rendered_messages, thinking=thinking)
        buf = create_string_buffer(needed + 1)
        written = self.lib.lib.llama_chat_apply_template(tmpl, chat_array, len(chat_array), True, buf, len(buf))
        if written < 0:
            raise RuntimeError("failed to apply chat template")
        rendered = bytes(buf[:written]).decode(errors="replace")
        if not thinking:
            rendered = _strip_thinking_prompt(rendered)
        return rendered

    def _token_to_bytes(self, token: int) -> bytes:
        if not self._vocab:
            return b""
        buf = (c_char * 512)()
        n = self.lib.lib.llama_token_to_piece(self._vocab, token, buf, len(buf), 0, True)
        if n <= 0:
            return b""
        return bytes(buf[:n])


def _trim_at_stop(content: str, stops: tuple[str, ...]) -> str:
    first: int | None = None
    for stop in stops:
        idx = content.find(stop)
        if idx >= 0 and (first is None or idx < first):
            first = idx
    if first is None:
        return content
    return content[:first]


def _prepare_mtp_prompt(prompt: str, *, thinking: bool = False) -> str:
    stripped = prompt.lstrip()
    if stripped.startswith("<bos>") or "<|turn>" in prompt:
        if thinking:
            return prompt
        return _strip_thinking_prompt(prompt)
    return render_gemma4_chat([{"role": "user", "content": prompt}], thinking=thinking)


def _strip_thinking_prompt(prompt: str) -> str:
    suffix = "<|turn>model\n<|channel>thought\n<channel|>"
    if prompt.endswith(suffix):
        return prompt[: -len(suffix)]
    return prompt


def _route_anchor_metadata(
    *,
    enabled: bool,
    attempted: bool,
    prefix_hash: str | None,
    fallback_reason: str | None,
) -> dict[str, object]:
    return {
        "phase": "route",
        "route_anchor_enabled": enabled,
        "route_anchor_attempted": attempted,
        "route_anchor_hit": False,
        "route_anchor_miss": False,
        "capture_attempted": False,
        "restore_attempted": False,
        "restore_used": False,
        "fallback_reason": fallback_reason,
        "prefix_hash": prefix_hash,
        "prefix_token_count": None,
        "checkpoint_size": None,
        "checkpoint_size_bytes": None,
        "checkpoint_age_ms": None,
        "anchor_invalidated": False,
        "invalidation_reason": None,
        "cached_tokens": None,
        "evaluated_tokens": None,
        "lcp_tokens": None,
    }


def _route_anchor_metadata_from_prefix(metadata: dict[str, object], *, prefix_hash: str) -> dict[str, object]:
    return {
        "phase": "route",
        "route_anchor_enabled": bool(metadata.get("anchor_enabled")),
        "route_anchor_attempted": True,
        "route_anchor_hit": bool(metadata.get("anchor_hit")),
        "route_anchor_miss": bool(metadata.get("anchor_miss")),
        "capture_attempted": bool(metadata.get("capture_attempted")),
        "restore_attempted": bool(metadata.get("restore_attempted")),
        "restore_used": bool(metadata.get("restore_used")),
        "fallback_reason": metadata.get("fallback_reason"),
        "prefix_hash": prefix_hash,
        "prefix_token_count": metadata.get("token_count"),
        "checkpoint_size": metadata.get("checkpoint_size"),
        "checkpoint_size_bytes": metadata.get("checkpoint_size_bytes"),
        "checkpoint_age_ms": metadata.get("checkpoint_age_ms"),
        "anchor_invalidated": bool(metadata.get("anchor_invalidated")),
        "invalidation_reason": metadata.get("invalidation_reason"),
    }


def _route_prefix_prefill_skipped(reason: str) -> NativeRoutePrefixPrefillResult:
    return NativeRoutePrefixPrefillResult(
        attempted=False,
        succeeded=False,
        skipped=True,
        skip_reason=reason,
    )


def _route_prefix_prefill_failed(
    reason: str,
    *,
    prefix_hash: str | None = None,
    prefix_token_count: int | None = None,
    checkpoint_size_bytes: int | None = None,
    prefill_ms: float | None = None,
    decode_calls: int | None = None,
) -> NativeRoutePrefixPrefillResult:
    return NativeRoutePrefixPrefillResult(
        attempted=True,
        succeeded=False,
        skipped=False,
        failed_reason=reason,
        prefix_hash=prefix_hash,
        prefix_token_count=prefix_token_count,
        checkpoint_size_bytes=checkpoint_size_bytes,
        prefill_ms=prefill_ms,
        decode_calls=decode_calls,
    )


def _strip_control_channels(content: str) -> str:
    if not content:
        return ""
    channel_filter = _ControlChannelStreamFilter()
    parts = channel_filter.write(content)
    parts.extend(channel_filter.finish())
    return "".join(parts)


def _strip_reasoning_preamble(content: str) -> str:
    text = content.strip()
    if not text:
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().lower() == "thought":
        cleaned = "\n".join(lines[1:]).strip()
        if cleaned:
            return cleaned
    return text


def _has_open_thought_channel(content: str) -> bool:
    start = content.rfind("<|channel>thought")
    if start < 0:
        return False
    end = content.rfind("<channel|>")
    return end < start


def _merge_completions(first: NativeCompletion, second: NativeCompletion) -> NativeCompletion:
    merged_content = first.content + second.content
    return NativeCompletion(
        content=merged_content,
        timings=NativeTimings(
            prompt_tokens=first.timings.prompt_tokens,
            output_tokens=first.timings.output_tokens + second.timings.output_tokens,
            reused_prompt_tokens=first.timings.reused_prompt_tokens,
            evaluated_prompt_tokens=first.timings.evaluated_prompt_tokens,
            prefill_ms=first.timings.prefill_ms,
            generation_ms=first.timings.generation_ms + second.timings.generation_ms,
            cancelled=first.timings.cancelled or second.timings.cancelled,
        ),
        stopped_by_stop=first.stopped_by_stop or second.stopped_by_stop,
        completed_after_thought=_has_closed_thought_with_final(merged_content),
    )


def _has_closed_thought_with_final(content: str) -> bool:
    end = content.rfind("<channel|>")
    if end < 0:
        return False
    tail = content[end + len("<channel|>") :].strip()
    return bool(tail)


def _looks_like_degenerate_thought_continuation(content: str) -> bool:
    stripped = content.strip()
    if not stripped or _has_closed_thought_with_final(content):
        return False
    if any(char.isalnum() for char in stripped):
        return False
    punctuation = "".join(char for char in stripped if not char.isspace())
    return len(punctuation) >= 4 and len(set(punctuation)) <= 3


def _can_continue_from_timings(timings: NativeTimings) -> bool:
    return not timings.cancelled and timings.output_tokens > 0


def _can_continue_from_completion(completion: NativeCompletion, *, thinking: bool) -> bool:
    if completion.timings.cancelled or completion.timings.output_tokens <= 0:
        return False
    if completion.stopped_by_stop:
        return False
    if thinking:
        return _has_open_thought_channel(completion.content) or completion.completed_after_thought
    return True


def _message_content(message: NativeMessage) -> str:
    return flatten_message_content(message)


class _StopSequenceStreamFilter:
    def __init__(self, stops: tuple[str, ...], *, emit) -> None:
        self._stops = stops
        self._emit = emit
        self._buffer = ""
        self.stopped = False
        self._keep = max(0, max(len(stop) for stop in stops) - 1)

    def write(self, text: str) -> list[str]:
        if self.stopped or not text:
            return []
        self._buffer += text
        stop_index = self._first_stop_index()
        if stop_index is not None:
            return self._emit_and_stop(self._buffer[:stop_index])
        if self._keep <= 0 or len(self._buffer) <= self._keep:
            return []
        return self._emit_prefix(len(self._buffer) - self._keep)

    def finish(self) -> list[str]:
        if self.stopped or not self._buffer:
            return []
        return self._emit_prefix(len(self._buffer))

    def _first_stop_index(self) -> int | None:
        first: int | None = None
        for stop in self._stops:
            idx = self._buffer.find(stop)
            if idx >= 0 and (first is None or idx < first):
                first = idx
        return first

    def _emit_and_stop(self, text: str) -> list[str]:
        self.stopped = True
        self._buffer = ""
        if not text:
            return []
        self._emit(text)
        return [text]

    def _emit_prefix(self, length: int) -> list[str]:
        text = self._buffer[:length]
        self._buffer = self._buffer[length:]
        if not text:
            return []
        self._emit(text)
        return [text]


class _ControlChannelStreamFilter:
    _START = "<|channel>"
    _END = "<channel|>"
    _MARKERS = (_START, _END)

    def __init__(self) -> None:
        self._buffer = ""

    def write(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        return self._drain(final=False)

    def finish(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        emitted: list[str] = []
        while self._buffer:
            start = self._buffer.find(self._START)
            end = self._buffer.find(self._END)
            marker_positions = [idx for idx in (start, end) if idx >= 0]
            if not marker_positions:
                emit_len = len(self._buffer) if final else self._safe_emit_length()
                if emit_len <= 0:
                    break
                emitted.append(self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
                continue
            marker = min(marker_positions)
            if marker > 0:
                emitted.append(self._buffer[:marker])
                self._buffer = self._buffer[marker:]
                continue
            if self._buffer.startswith(self._END):
                self._buffer = self._buffer[len(self._END):]
                continue
            block_end = self._buffer.find(self._END, len(self._START))
            if block_end < 0:
                if final:
                    self._buffer = ""
                break
            self._buffer = self._buffer[block_end + len(self._END):]
        return [text for text in emitted if text]

    def _safe_emit_length(self) -> int:
        keep = 0
        for marker in self._MARKERS:
            max_prefix = min(len(marker) - 1, len(self._buffer))
            for size in range(max_prefix, 0, -1):
                if marker.startswith(self._buffer[-size:]):
                    keep = max(keep, size)
                    break
        return max(0, len(self._buffer) - keep)


class _LeadingThoughtLabelFilter:
    def __init__(self) -> None:
        self._buffer = ""
        self._resolved = False

    def write(self, text: str) -> list[str]:
        if self._resolved or not text:
            return [text] if text else []
        self._buffer += text
        newline_index = self._find_newline(self._buffer)
        if newline_index < 0:
            return []
        first_line = self._buffer[:newline_index]
        rest = self._buffer[newline_index + 1 :]
        self._resolved = True
        self._buffer = ""
        if first_line.strip().lower() == "thought":
            return [rest] if rest else []
        return [first_line + "\n" + rest] if rest else [first_line + "\n"]

    def finish(self) -> list[str]:
        if self._resolved or not self._buffer:
            return []
        self._resolved = True
        buffered = self._buffer
        self._buffer = ""
        return [buffered]

    @staticmethod
    def _find_newline(text: str) -> int:
        for marker in ("\r\n", "\n", "\r"):
            idx = text.find(marker)
            if idx >= 0:
                return idx if marker == "\n" else idx + (0 if marker == "\r" else 1)
        return -1
