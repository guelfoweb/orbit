from __future__ import annotations

import codecs
from ctypes import POINTER, byref, c_char, cast, c_float, create_string_buffer, c_void_p, sizeof
from dataclasses import dataclass, replace
from pathlib import Path
import threading

from .bindings import (
    GgmlAbortCallback,
    GgmlLogCallback,
    LlamaLibrary,
    LlamaChatMessage,
    LlamaProgressCallback,
    llama_token,
)
from .chat_template import NativeMessage, render_gemma4_chat
from .events import NativeCompletion, NativeProgress, NativeTimings
from .mtp_completion import MtpCompletionResult
from .mtp_decode_probe import MtpDecodeProbeResult, run_mtp_decode_probe
from .mtp_accept_probe import MtpAcceptProbeResult, run_mtp_accept_probe
from .mtp_dry_run import MtpDryRunResult, run_mtp_dry_run
from .mtp_probe import MtpProbeResult, run_mtp_probe
from .paths import NativeLlamaPaths
from .persistent_mtp import (
    PersistentMtpSessionRuntime,
    create_persistent_mtp_session,
    free_persistent_mtp_session,
    reset_persistent_mtp_session,
    run_persistent_mtp_completion,
)
from .session_state import DEFAULT_NATIVE_SESSION_ID, NativeSessionSnapshot, NativeSessionState


@dataclass(frozen=True)
class NativeClientConfig:
    context_tokens: int = 8192
    threads: int = 6
    threads_batch: int = 6
    batch_size: int = 256
    ubatch_size: int = 128
    progress_step: int = 64
    gpu_layers: int = 0
    mtp_probe_enabled: bool = False
    mtp_dry_run_enabled: bool = False
    mtp_accept_probe_enabled: bool = False
    mtp_decode_probe_enabled: bool = False
    use_mtp_experimental: bool = False


class NativeLlamaClient:
    def __init__(self, paths: NativeLlamaPaths, config: NativeClientConfig | None = None) -> None:
        self.paths = paths
        self.config = config or NativeClientConfig()
        self.lib = LlamaLibrary(paths.build_bin)
        self.cancel_event = threading.Event()
        self._callbacks: list[object] = []
        self._model: c_void_p | None = None
        self._vocab: c_void_p | None = None
        self._session = NativeSessionState(session_id=DEFAULT_NATIVE_SESSION_ID)
        self.mtp_probe = MtpProbeResult(enabled=self.config.mtp_probe_enabled, initialized=False, error=None)
        self.mtp_dry_run = MtpDryRunResult(enabled=self.config.mtp_dry_run_enabled, success=False, error=None)
        self.mtp_accept_probe = MtpAcceptProbeResult(enabled=self.config.mtp_accept_probe_enabled, success=False, error=None)
        self.mtp_decode_probe = MtpDecodeProbeResult(enabled=self.config.mtp_decode_probe_enabled, success=False, error=None)
        self.last_mtp_completion = MtpCompletionResult(enabled=self.config.use_mtp_experimental, success=False, error=None)
        self.mtp_fallback_reason: str | None = None
        self._persistent_mtp_runtime: PersistentMtpSessionRuntime | None = None

    def session_snapshot(self, session_id: str = DEFAULT_NATIVE_SESSION_ID) -> NativeSessionSnapshot:
        if session_id != self._session.session_id:
            raise ValueError("only the default native session is supported in this experiment")
        return self._session.snapshot(backend_mode="no-mtp")

    def set_quiet_logging(self) -> None:
        def log_cb(_level: int, _text: bytes, _data) -> None:
            return None

        cb = GgmlLogCallback(log_cb)
        self._callbacks.append(cb)
        self.lib.lib.llama_log_set(cb, None)

    def close(self) -> None:
        lib = self.lib.lib
        self._free_persistent_mtp_session()
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

    def complete_chat(
        self,
        messages: list[NativeMessage],
        *,
        max_tokens: int = 16,
        tools: list[dict] | None = None,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        prompt = self.apply_chat_template(messages, tools=tools)
        return self.complete_prompt(
            prompt,
            max_tokens=max_tokens,
            allow_mtp_experimental=not tools and not any(message.get("role") == "tool" or message.get("tool_calls") for message in messages),
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
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeCompletion:
        parts: list[str] = []
        channel_filter = _ControlChannelStreamFilter()
        stop_filter = _StopSequenceStreamFilter(stop, emit=parts.append) if stop else None

        def collect(text: str) -> None:
            for visible_text in channel_filter.write(text):
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
            for visible_text in channel_filter.finish():
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
            on_progress=on_progress,
            on_token=collect,
            should_cancel=should_cancel,
        )
        flush_filters()
        content = _trim_at_stop("".join(parts), stop)
        return NativeCompletion(content=content, timings=timings, stopped_by_stop=bool(stop_filter and stop_filter.stopped))

    def complete_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        allow_mtp_experimental: bool = True,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        self._session.in_flight = True
        try:
            if allow_mtp_experimental and should_cancel is None:
                mtp_result = self._try_complete_with_mtp_experimental(
                    prompt,
                    max_tokens=max_tokens,
                    on_token=on_token,
                )
                if mtp_result is not None:
                    self._session.last_metrics = mtp_result
                    return mtp_result
            timings = self._complete_prompt_standard(
                prompt,
                max_tokens=max_tokens,
                on_progress=on_progress,
                on_token=on_token,
                should_cancel=should_cancel,
            )
            self._session.last_metrics = timings
            return timings
        finally:
            self._session.in_flight = False

    def _try_complete_with_mtp_experimental(
        self,
        prompt: str,
        *,
        max_tokens: int,
        on_token=None,
    ) -> NativeTimings | None:
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

        mtp_prompt = _prepare_mtp_prompt(prompt)
        result = run_persistent_mtp_completion(
            llama_root=self.paths.llama_root,
            paths=self.paths,
            runtime=self._persistent_mtp_runtime,
            ctx_tgt=self._session.ctx_tgt,
            prompt=mtp_prompt,
            max_tokens=min(max_tokens, 32),
        )
        if result.success and result.content:
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
        if result.content and on_token:
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
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ) -> NativeTimings:
        if not self._session.ctx_tgt or not self._vocab or not self._session.sampler:
            raise RuntimeError("native client not loaded")
        self.reset_cancel()

        lib = self.lib.lib
        prompt_tokens = self.tokenize(prompt)
        n_prompt = len(prompt_tokens)
        if not prompt_tokens:
            raise RuntimeError("failed to count prompt tokens")

        reused = self._prepare_memory_for_prompt(prompt_tokens)
        processed = 0
        step = max(1, min(self.config.progress_step, self.config.batch_size))
        pf_start = lib.llama_time_us()
        processed = reused
        token_array = (llama_token * n_prompt)(*prompt_tokens)
        while processed < n_prompt and not self.cancel_event.is_set():
            if should_cancel and should_cancel():
                self.cancel()
                break
            n = min(step, n_prompt - processed)
            token_ptr = cast(byref(token_array, processed * sizeof(llama_token)), POINTER(llama_token))
            batch = lib.llama_batch_get_one(token_ptr, n)
            decode_rc = lib.llama_decode(self._session.ctx_tgt, batch)
            processed += n
            if on_progress:
                on_progress(NativeProgress("prefill", processed, n_prompt))
            if decode_rc != 0:
                if decode_rc == 2 and self.cancel_event.is_set():
                    break
                raise RuntimeError(f"llama_decode failed during prefill: {decode_rc}")
        pf_ms = (lib.llama_time_us() - pf_start) / 1000.0

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
            if should_cancel and should_cancel():
                self.cancel()
                break
        text = decoder.decode(b"", final=True)
        if text and on_token:
            on_token(text)

        gen_ms = (lib.llama_time_us() - gen_start) / 1000.0
        return NativeTimings(
            prompt_tokens=n_prompt,
            output_tokens=generated,
            reused_prompt_tokens=reused,
            evaluated_prompt_tokens=n_prompt - reused,
            prefill_ms=pf_ms,
            generation_ms=gen_ms,
            cancelled=self.cancel_event.is_set(),
        )

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

    def apply_chat_template(self, messages: list[NativeMessage], *, tools: list[dict] | None = None) -> str:
        if not self._model:
            raise RuntimeError("native client not loaded")
        if tools or any(message.get("role") == "tool" or message.get("tool_calls") for message in messages):
            return render_gemma4_chat(messages, tools=tools)
        encoded_messages = [
            (str(message.get("role", "user")).encode(), _message_content(message).encode())
            for message in messages
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
            return render_gemma4_chat(messages)
        buf = create_string_buffer(needed + 1)
        written = self.lib.lib.llama_chat_apply_template(tmpl, chat_array, len(chat_array), True, buf, len(buf))
        if written < 0:
            raise RuntimeError("failed to apply chat template")
        return bytes(buf[:written]).decode(errors="replace")

    def _token_to_bytes(self, token: int) -> bytes:
        if not self._vocab:
            return b""
        buf = (c_char * 512)()
        n = self.lib.lib.llama_token_to_piece(self._vocab, token, buf, len(buf), 0, True)
        if n <= 0:
            return b""
        return bytes(buf[:n])


def _contains_stop(content: str, stops: tuple[str, ...]) -> bool:
    return any(stop in content for stop in stops)


def _trim_at_stop(content: str, stops: tuple[str, ...]) -> str:
    first: int | None = None
    for stop in stops:
        idx = content.find(stop)
        if idx >= 0 and (first is None or idx < first):
            first = idx
    if first is None:
        return content
    return content[:first]


def _prepare_mtp_prompt(prompt: str) -> str:
    stripped = prompt.lstrip()
    if stripped.startswith("<bos>") or "<|turn>" in prompt:
        return prompt
    return render_gemma4_chat([{"role": "user", "content": prompt}])


def _strip_control_channels(content: str) -> str:
    if not content:
        return ""
    channel_filter = _ControlChannelStreamFilter()
    parts = channel_filter.write(content)
    parts.extend(channel_filter.finish())
    return "".join(parts)


def _message_content(message: NativeMessage) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return ""


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
