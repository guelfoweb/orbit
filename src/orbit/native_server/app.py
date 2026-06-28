from __future__ import annotations

import argparse
import json
import select
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient, _has_open_thought_channel
from orbit.native_llama.kv_diag import request_context as native_kv_request_context
from orbit.native_llama.paths import DEFAULT_LLAMA_ROOT, DEFAULT_MODEL_ID, NativeLlamaPaths, resolve_legacy_paths, resolve_paths
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


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 12120
DEFAULT_ALIAS = "gemma4:12b-it-native"


class OrbitNativeServer:
    def __init__(self, *, client: NativeLlamaClient, model_alias: str) -> None:
        self.client = client
        self.model_alias = model_alias
        self.lock = threading.Lock()

    def chat(self, payload: dict[str, Any], *, on_token=None, on_progress=None, should_cancel=None) -> dict[str, Any]:
        request = parse_chat_request(payload)
        validate_session_id(request.session_id)
        return self.complete(request, on_token=on_token, on_progress=on_progress, should_cancel=should_cancel)

    def complete(self, request: ChatRequest, *, on_token=None, on_progress=None, should_cancel=None) -> dict[str, Any]:
        parts: list[str] = []

        def collect(text: str) -> None:
            parts.append(text)
            if on_token:
                on_token(text)

        with self.lock:
            thinking = self.client.config.thinking if request.thinking is None else request.thinking
            completion = self.client.complete_chat_text(
                request.messages,
                max_tokens=request.max_tokens,
                stop=request.stop,
                tools=request.tools,
                thinking=thinking,
                route_prefix_anchor=request.route_prefix_anchor,
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
        )

    def continue_current(self, request: ContinueRequest, *, on_token=None, on_progress=None, should_cancel=None) -> dict[str, Any]:
        with self.lock:
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
        return {
            "threads": self.client.config.threads,
            "threads_batch": self.client.config.threads_batch,
            "ctx_size": self.client.config.context_tokens,
            "batch_size": self.client.config.batch_size,
            "ubatch_size": self.client.config.ubatch_size,
            "parallel_slots": 1,
            "thinking_mode": "on" if self.client.config.thinking else "off",
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
        if self.path == "/props":
            state = self._state()
            session = state.session_info()
            runtime = state.runtime_info()
            self._json(
                {
                    "model_path": str(state.client.paths.model),
                    "mmproj_path": str(state.client.paths.mmproj_model) if state.client.paths.mmproj_model else None,
                    "draft_model_path": str(state.client.paths.draft_mtp_model) if state.client.paths.draft_mtp_model else None,
                    "multimodal_available": state.client.paths.multimodal_available,
                    "multimodal_fallback_reason": state.client.paths.multimodal_fallback_reason,
                    "supports_vision": state.client.supports_vision,
                    "supports_audio": state.client.supports_audio,
                    "mtp_available": state.client.paths.mtp_available,
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
                    "mtp_fallback_reason": state.client.mtp_fallback_reason,
                    "mtp_enabled": session["mtp_enabled"],
                    "mtp_initialized": session["mtp_initialized"],
                    "mtp_failure_reason": session["mtp_failure_reason"],
                    "model_id": state.client.paths.model_id,
                    "backend": "orbit-native",
                    "backend_mode": session["backend_mode"],
                    "thinking_mode": runtime["thinking_mode"],
                    "session_id": session["id"],
                    "cached_tokens": session["cached_tokens"],
                    "in_flight": session["in_flight"],
                    "threads": runtime["threads"],
                    "threads_batch": runtime["threads_batch"],
                    "ctx_size": runtime["ctx_size"],
                    "batch_size": runtime["batch_size"],
                    "ubatch_size": runtime["ubatch_size"],
                    "parallel_slots": runtime["parallel_slots"],
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
            payload = self._read_json()
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return

        try:
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
                {
                    "current": progress.current,
                    "total": progress.total,
                    "percent": progress.percent,
                    "session_id": request.session_id,
                },
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
                {
                    "current": progress.current,
                    "total": progress.total,
                    "percent": progress.percent,
                    "session_id": DEFAULT_SESSION_ID,
                },
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


def run_server(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        paths = resolve_bootstrap_paths(args)
        client = NativeLlamaClient(
            paths,
            NativeClientConfig(
                context_tokens=args.ctx,
                threads=args.threads,
                threads_batch=args.threads_batch,
                batch_size=args.batch,
                ubatch_size=args.ubatch,
                thinking=args.think == "on",
                mtp_probe_enabled=args.enable_mtp_probe,
                mtp_dry_run_enabled=args.enable_mtp_dry_run,
                mtp_accept_probe_enabled=args.enable_mtp_accept_probe,
                mtp_decode_probe_enabled=args.enable_mtp_decode_probe,
                use_mtp_experimental=args.enable_mtp_experimental,
            ),
        )
        if not args.verbose_llama_log:
            client.set_quiet_logging()
        client.load()
    except (FileNotFoundError, RuntimeError) as exc:
        print(_format_native_bootstrap_error(exc), file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), OrbitNativeHandler)
    httpd.orbit_state = OrbitNativeServer(client=client, model_alias=args.alias)  # type: ignore[attr-defined]
    print(f"orbit-server listening on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\norbit-server stopped", flush=True)
    finally:
        client.close()
        httpd.server_close()
    return 0


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
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--threads-batch", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--think", choices=("off", "on"), default="off", help="Default thinking visibility for native server requests.")
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


def _session_id_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("session_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_SESSION_ID


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
