from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatResult, Message, ModelInfo, StreamProgress
from .model_names import resolve_model_display_name
from .payloads import ChatPayloadOptions, build_chat_payload
from orbit.final_prefix_config import resolve_final_prefix_reuse
from orbit.native_llama.prefix_anchor import prefix_anchor_enabled
from orbit.runtime.kv_diag import current_phase, current_tools_mode, enabled as kv_diag_enabled


class LlamaServerError(RuntimeError):
    pass


class LlamaServerBackend:
    def __init__(self, *, base_url: str, timeout: float, model: str | None = None, thinking: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.thinking = thinking
        self._model_info_cache: ModelInfo | None = None
        self._display_model_name: str | None = None
        self._server_tools_cache: list[dict[str, Any]] | None = None
        self._props_cache: dict[str, Any] | None = None

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        native_backend = self._is_orbit_native_backend()
        payload = build_chat_payload(
            ChatPayloadOptions(
                model=self.request_model_name(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=self.thinking,
                tools=tools,
                route_prefix_anchor=_route_prefix_anchor_requested(native_backend=native_backend),
                allow_mtp_experimental=_allow_mtp_experimental_requested(native_backend=native_backend),
                final_prefix_experiment=_final_prefix_experiment_requested(native_backend=native_backend),
            )
        )
        _attach_native_kv_diag_payload(payload, native_backend=native_backend)
        if native_backend:
            return self._with_display_model(
                self._post_native_stream(
                    "/chat/stream",
                    payload,
                    on_delta=lambda _text: None,
                    on_progress=None,
                )
            )
        data = self._post_json("/v1/chat/completions", payload)
        return self._with_display_model(_parse_chat_result(data))

    def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None],
        on_progress: Callable[[StreamProgress], None] | None = None,
    ) -> ChatResult:
        native_backend = self._is_orbit_native_backend()
        payload = build_chat_payload(
            ChatPayloadOptions(
                model=self.request_model_name(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=self.thinking,
                tools=tools,
                stream=True,
                route_prefix_anchor=_route_prefix_anchor_requested(native_backend=native_backend),
                allow_mtp_experimental=_allow_mtp_experimental_requested(native_backend=native_backend),
                final_prefix_experiment=_final_prefix_experiment_requested(native_backend=native_backend),
            )
        )
        _attach_native_kv_diag_payload(payload, native_backend=native_backend)
        if native_backend:
            return self._with_display_model(
                self._post_native_stream(
                    "/chat/stream",
                    payload,
                    on_delta=on_delta,
                    on_progress=on_progress,
                )
            )
        return self._with_display_model(self._post_stream("/v1/chat/completions", payload, on_delta=on_delta))

    def continue_current(
        self,
        *,
        max_tokens: int,
        on_delta: Callable[[str], None] | None = None,
        on_progress: Callable[[StreamProgress], None] | None = None,
    ) -> ChatResult:
        if not self._is_orbit_native_backend():
            raise LlamaServerError("native continuation is unavailable for this backend")
        payload = {"max_tokens": max_tokens, "thinking": self.thinking, "stream": True}
        if on_delta is None:
            return self._with_display_model(
                self._post_native_stream(
                    "/chat/continue/stream",
                    payload,
                    on_delta=lambda _text: None,
                    on_progress=None,
                )
            )
        return self._with_display_model(
            self._post_native_stream(
                "/chat/continue/stream",
                payload,
                on_delta=on_delta,
                on_progress=on_progress,
            )
        )

    def health(self) -> bool:
        try:
            data = self._get_json("/health")
        except LlamaServerError:
            return False
        return data.get("status") == "ok"

    def model_info(self) -> ModelInfo | None:
        if self._model_info_cache is not None:
            return self._model_info_cache
        props = self._props_or_empty()
        try:
            data = self._get_json("/v1/models")
        except LlamaServerError:
            self._model_info_cache = _enrich_model_info_with_props(None, props)
            return self._model_info_cache
        self._model_info_cache = _enrich_model_info_with_props(
            _parse_model_info(data, model_path=_str_or_none(props.get("model_path"))),
            props,
        )
        return self._model_info_cache

    def display_model_name(self) -> str | None:
        if self._display_model_name:
            return self._display_model_name
        info = self.model_info()
        if info and info.id:
            self._display_model_name = info.id
            return self._display_model_name
        return None

    def request_model_name(self) -> str:
        if self.model:
            return self.model
        return self.display_model_name() or "local-model"

    def server_tools(self) -> list[dict[str, Any]]:
        if self._server_tools_cache is not None:
            return self._server_tools_cache
        try:
            data = self._get_json("/tools")
        except LlamaServerError:
            self._server_tools_cache = []
            return []
        if not isinstance(data, list):
            self._server_tools_cache = []
            return []
        self._server_tools_cache = [item for item in data if isinstance(item, dict)]
        return self._server_tools_cache

    def backend_props(self) -> dict[str, Any]:
        return dict(self._props_or_empty())

    def execute_server_tool(self, name: str, arguments: dict[str, Any]) -> str:
        data = self._post_json("/tools", {"tool": name, "params": arguments})
        if isinstance(data.get("plain_text_response"), str):
            return data["plain_text_response"]
        return json.dumps(data, ensure_ascii=False)

    def _get_json(self, path: str) -> Any:
        request = Request(f"{self.base_url}{path}", method="GET")
        return self._send(request)

    def _props_or_empty(self) -> dict[str, Any]:
        if self._props_cache is not None:
            return self._props_cache
        try:
            data = self._get_json("/props")
        except LlamaServerError:
            self._props_cache = {}
            return {}
        self._props_cache = data if isinstance(data, dict) else {}
        return self._props_cache

    def _is_orbit_native_backend(self) -> bool:
        return _str_or_none(self._props_or_empty().get("backend")) == "orbit-native"

    def _with_display_model(self, result: ChatResult) -> ChatResult:
        display = self.display_model_name()
        if not display:
            return result
        return replace(result, model=display)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request)

    def _post_stream(self, path: str, payload: dict[str, Any], *, on_delta: Callable[[str], None]) -> ChatResult:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return _parse_chat_stream(response, on_delta=on_delta)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"backend server HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LlamaServerError(f"cannot connect to backend server at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LlamaServerError(f"backend server request timed out after {self.timeout:.0f}s") from exc

    def _post_native_stream(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        on_delta: Callable[[str], None],
        on_progress: Callable[[StreamProgress], None] | None,
    ) -> ChatResult:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return _parse_native_stream(response, on_delta=on_delta, on_progress=on_progress)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"backend server HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LlamaServerError(f"cannot connect to backend server at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LlamaServerError(f"backend server request timed out after {self.timeout:.0f}s") from exc

    def _send(self, request: Request) -> Any:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"backend server HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LlamaServerError(f"cannot connect to backend server at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LlamaServerError(f"backend server request timed out after {self.timeout:.0f}s") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LlamaServerError(f"backend server returned invalid JSON: {raw[:200]}") from exc
        return data

def _parse_chat_result(data: dict[str, Any]) -> ChatResult:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlamaServerError("backend server response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LlamaServerError("backend server choice is invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LlamaServerError("backend server choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    parsed_raw_tool_calls = _parse_raw_tool_call_content(content)
    if not tool_calls and parsed_raw_tool_calls:
        tool_calls = parsed_raw_tool_calls
        content = ""

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}

    return ChatResult(
        content=content,
        model=_str_or_none(data.get("model")),
        finish_reason=_str_or_none(first.get("finish_reason")),
        tool_calls=[tool_call for tool_call in tool_calls if isinstance(tool_call, dict)],
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
        cached_tokens=_int_or_none(details.get("cached_tokens")),
        prompt_tokens_per_second=_float_or_none(timings.get("prompt_per_second")),
        generation_tokens_per_second=_float_or_none(timings.get("predicted_per_second")),
    )


def _parse_chat_stream(response: Any, *, on_delta: Callable[[str], None]) -> ChatResult:
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    content_filter = _ContentStreamFilter(on_delta)
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data_text = line.removeprefix("data:").strip()
        if data_text == "[DONE]":
            break
        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        model = _str_or_none(data.get("model")) or model
        if isinstance(data.get("usage"), dict):
            usage = data["usage"]
        if isinstance(data.get("timings"), dict):
            timings = data["timings"]
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        finish_reason = _str_or_none(first.get("finish_reason")) or finish_reason
        delta = first.get("delta")
        if not isinstance(delta, dict):
            continue
        text = delta.get("content")
        if isinstance(text, str) and text:
            content_parts.append(text)
            content_filter.write(text)
        _merge_stream_tool_calls(tool_calls_by_index, delta.get("tool_calls"))

    content_filter.finish()
    content = "".join(content_parts)
    tool_calls = [tool_calls_by_index[index] for index in sorted(tool_calls_by_index)]
    parsed_raw_tool_calls = _parse_raw_tool_call_content(content)
    if not tool_calls and parsed_raw_tool_calls:
        tool_calls = parsed_raw_tool_calls
        content = ""

    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return ChatResult(
        content=content,
        model=model,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
        cached_tokens=_int_or_none(details.get("cached_tokens")),
        prompt_tokens_per_second=_float_or_none(timings.get("prompt_per_second")),
        generation_tokens_per_second=_float_or_none(timings.get("predicted_per_second")),
    )


def _parse_native_stream(
    response: Any,
    *,
    on_delta: Callable[[str], None],
    on_progress: Callable[[StreamProgress], None] | None,
) -> ChatResult:
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    content_filter = _ContentStreamFilter(on_delta)
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    current_event: str | None = None
    current_data_lines: list[str] = []
    stream_done = False

    def flush_event() -> None:
        nonlocal current_event, current_data_lines, model, finish_reason, usage, timings, stream_done
        if not current_event:
            current_data_lines = []
            return
        data_text = "\n".join(current_data_lines).strip()
        current_data_lines = []
        if not data_text:
            current_event = None
            return
        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            current_event = None
            return
        if not isinstance(data, dict):
            current_event = None
            return
        if current_event == "delta":
            text = data.get("text")
            if isinstance(text, str) and text:
                content_parts.append(text)
                content_filter.write(text)
        elif current_event.startswith("progress.") and on_progress:
            phase = current_event.split(".", maxsplit=1)[1]
            progress = StreamProgress(
                phase=phase,
                current=_int_or_none(data.get("current")) or 0,
                total=_int_or_none(data.get("total")) or 0,
                percent=_int_or_none(data.get("percent")) or 0,
            )
            on_progress(progress)
        elif current_event == "tool_calls":
            _merge_stream_tool_calls(tool_calls_by_index, data.get("tool_calls"))
        elif current_event == "metrics":
            if isinstance(data.get("usage"), dict):
                usage = data["usage"]
            if isinstance(data.get("timings"), dict):
                timings = data["timings"]
        elif current_event == "done":
            finish_reason = _str_or_none(data.get("finish_reason")) or finish_reason
            stream_done = True
        elif current_event == "error":
            message = _str_or_none(data.get("message")) or "native stream error"
            raise LlamaServerError(message)
        model = _str_or_none(data.get("model")) or model
        current_event = None

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            flush_event()
            if stream_done:
                break
            continue
        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
            continue
        if line.startswith("data:"):
            current_data_lines.append(line.removeprefix("data:").strip())

    if not stream_done:
        flush_event()
    content_filter.finish()
    content = "".join(content_parts)
    tool_calls = [tool_calls_by_index[index] for index in sorted(tool_calls_by_index)]
    parsed_raw_tool_calls = _parse_raw_tool_call_content(content)
    if not tool_calls and parsed_raw_tool_calls:
        tool_calls = parsed_raw_tool_calls
        content = ""
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return ChatResult(
        content=content,
        model=model,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
        cached_tokens=_int_or_none(details.get("cached_tokens")),
        prompt_tokens_per_second=_float_or_none(timings.get("prompt_per_second")),
        generation_tokens_per_second=_float_or_none(timings.get("predicted_per_second")),
    )


def _merge_stream_tool_calls(tool_calls_by_index: dict[int, dict[str, Any]], raw_tool_calls: object) -> None:
    if not isinstance(raw_tool_calls, list):
        return
    for raw in raw_tool_calls:
        if not isinstance(raw, dict):
            continue
        index = raw.get("index")
        if not isinstance(index, int):
            index = len(tool_calls_by_index)
        current = tool_calls_by_index.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if isinstance(raw.get("id"), str):
            current["id"] = raw["id"]
        if isinstance(raw.get("type"), str):
            current["type"] = raw["type"]
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        current_function = current.setdefault("function", {"name": "", "arguments": ""})
        if not isinstance(current_function, dict):
            current_function = {"name": "", "arguments": ""}
            current["function"] = current_function
        if isinstance(function.get("name"), str):
            current_function["name"] = str(current_function.get("name", "")) + function["name"]
        if isinstance(function.get("arguments"), str):
            current_function["arguments"] = str(current_function.get("arguments", "")) + function["arguments"]


class _ContentStreamFilter:
    def __init__(self, on_delta: Callable[[str], None]) -> None:
        self._on_delta = on_delta
        self._buffer = ""
        self._suppressing_raw_tool_call = False

    def write(self, text: str) -> None:
        self._buffer += text
        self._flush(complete=False)

    def finish(self) -> None:
        self._flush(complete=True)

    def _flush(self, *, complete: bool) -> None:
        while self._buffer:
            if self._suppressing_raw_tool_call:
                end = self._buffer.find("<tool_call|>")
                if end < 0:
                    if complete:
                        self._buffer = ""
                    return
                self._buffer = self._buffer[end + len("<tool_call|>") :]
                self._suppressing_raw_tool_call = False
                continue

            start = self._buffer.find("<|tool_call>")
            if start == 0:
                end = self._buffer.find("<tool_call|>")
                if end < 0:
                    if complete:
                        self._buffer = ""
                    else:
                        self._suppressing_raw_tool_call = True
                    return
                self._buffer = self._buffer[end + len("<tool_call|>") :]
                continue
            if start > 0:
                self._on_delta(self._buffer[:start])
                self._buffer = self._buffer[start:]
                continue

            keep = 0 if complete else _raw_tool_call_prefix_suffix_len(self._buffer)
            emit_len = len(self._buffer) - keep
            if emit_len > 0:
                self._on_delta(self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
            return


def _is_partial_prefix(text: str, prefix: str) -> bool:
    return len(text) < len(prefix) and prefix.startswith(text)


def _raw_tool_call_prefix_suffix_len(text: str) -> int:
    marker = "<|tool_call>"
    max_len = min(len(text), len(marker) - 1)
    for length in range(max_len, 0, -1):
        if marker.startswith(text[-length:]):
            return length
    return 0


def _parse_raw_tool_call_content(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if not text.startswith("<|tool_call>") or "<tool_call|>" not in text:
        return []
    inner = text.removeprefix("<|tool_call>").split("<tool_call|>", maxsplit=1)[0].strip()
    if inner.startswith("call:"):
        inner = inner.removeprefix("call:").strip()
    match = re.match(r"(?:[A-Za-z0-9_]+\.)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<args>\{.*\})\s*$", inner, re.DOTALL)
    if not match:
        return []
    name = match.group("name")
    args = _normalize_raw_tool_arguments(match.group("args"))
    if args is None:
        return []
    return [
        {
            "id": "raw-tool-call-1",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        }
    ]


def _normalize_raw_tool_arguments(arguments: str) -> str | None:
    raw_string_args = _normalize_raw_tool_string_argument(arguments)
    if raw_string_args is not None:
        return raw_string_args
    normalized = arguments.replace('<|"|>', '"')
    normalized = re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', normalized)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return json.dumps(parsed, ensure_ascii=False)


def _normalize_raw_tool_string_argument(arguments: str) -> str | None:
    match = re.match(
        r'^\{\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*<\|"\|>(?P<value>.*)<\|"\|>\s*\}\s*$',
        arguments,
        re.DOTALL,
    )
    if not match:
        return None
    return json.dumps({match.group("key"): match.group("value")}, ensure_ascii=False)


def _parse_model_info(data: dict[str, Any], *, model_path: str | None = None) -> ModelInfo | None:
    model_items = data.get("models")
    openai_items = data.get("data")
    model_item = model_items[0] if isinstance(model_items, list) and model_items else {}
    openai_item = openai_items[0] if isinstance(openai_items, list) and openai_items else {}
    if not isinstance(model_item, dict):
        model_item = {}
    if not isinstance(openai_item, dict):
        openai_item = {}

    meta = openai_item.get("meta") if isinstance(openai_item.get("meta"), dict) else {}
    capabilities = model_item.get("capabilities")
    if not isinstance(capabilities, list):
        capabilities = []

    raw_model_id = _str_or_none(openai_item.get("id")) or _str_or_none(model_item.get("model")) or _str_or_none(model_item.get("name"))
    model_id = resolve_model_display_name(raw_model_id, model_path=model_path)
    if not model_id and not capabilities and not meta:
        return None

    return ModelInfo(
        id=model_id,
        capabilities=tuple(value for value in capabilities if isinstance(value, str) and value),
        context_length=_int_or_none(meta.get("n_ctx")),
        parameter_count=_int_or_none(meta.get("n_params")),
        size_bytes=_int_or_none(meta.get("size")),
    )


def _enrich_model_info_with_props(info: ModelInfo | None, props: dict[str, Any]) -> ModelInfo | None:
    capabilities = list(info.capabilities if info else ())
    backend = _str_or_none(props.get("backend"))
    if backend == "orbit-native" and "completion" not in capabilities:
        capabilities.append("completion")
    if bool(props.get("supports_vision")) and "vision" not in capabilities:
        capabilities.append("vision")
    if bool(props.get("supports_audio")) and "audio" not in capabilities:
        capabilities.append("audio")
    if bool(props.get("multimodal_available")) and "multimodal" not in capabilities:
        capabilities.append("multimodal")
    context_length = _int_or_none(props.get("ctx_size"))
    if info is None:
        model_id = resolve_model_display_name(_str_or_none(props.get("model_id")), model_path=_str_or_none(props.get("model_path")))
        if not model_id and not capabilities and context_length is None:
            return None
        return ModelInfo(
            id=model_id,
            capabilities=tuple(capabilities),
            context_length=context_length,
            parameter_count=None,
            size_bytes=None,
        )
    return ModelInfo(
        id=info.id,
        capabilities=tuple(capabilities) or info.capabilities,
        context_length=info.context_length if info.context_length is not None else context_length,
        parameter_count=info.parameter_count,
        size_bytes=info.size_bytes,
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _route_prefix_anchor_requested(*, native_backend: bool) -> bool:
    if not native_backend:
        return False
    if not prefix_anchor_enabled():
        return False
    return current_phase() == "route" and current_tools_mode() == "on"


def _allow_mtp_experimental_requested(*, native_backend: bool) -> bool | None:
    if not native_backend:
        return None
    if current_tools_mode() != "on":
        return None
    phase = current_phase()
    if phase == "chat_final_retry" or phase.startswith("final_from_tool"):
        return False
    return None


def _final_prefix_experiment_requested(*, native_backend: bool) -> bool:
    if not native_backend or not resolve_final_prefix_reuse().enabled:
        return False
    phase = current_phase()
    return current_tools_mode() == "on" and phase is not None and phase.startswith("final_from_tool")


def _attach_native_kv_diag_payload(payload: dict[str, Any], *, native_backend: bool) -> None:
    if not native_backend or not kv_diag_enabled():
        return
    phase = current_phase()
    tools_mode = current_tools_mode()
    if phase is not None:
        payload["_orbit_kv_phase"] = phase
    if tools_mode is not None:
        payload["_orbit_kv_tools_mode"] = tools_mode
