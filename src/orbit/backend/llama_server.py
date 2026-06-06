from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatResult, Message, ModelInfo
from .model_names import resolve_model_display_name
from .payloads import ChatPayloadOptions, build_chat_payload


class LlamaServerError(RuntimeError):
    pass


class LlamaServerBackend:
    def __init__(self, *, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._display_model_name: str | None = None

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        payload = build_chat_payload(
            ChatPayloadOptions(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
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
    ) -> ChatResult:
        payload = build_chat_payload(
            ChatPayloadOptions(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
            )
        )
        return self._with_display_model(self._post_stream("/v1/chat/completions", payload, on_delta=on_delta))

    def health(self) -> bool:
        try:
            data = self._get_json("/health")
        except LlamaServerError:
            return False
        return data.get("status") == "ok"

    def model_info(self) -> ModelInfo | None:
        try:
            data = self._get_json("/v1/models")
        except LlamaServerError:
            return None
        props = self._props_or_empty()
        return _parse_model_info(data, model_path=_str_or_none(props.get("model_path")))

    def display_model_name(self) -> str | None:
        if self._display_model_name:
            return self._display_model_name
        info = self.model_info()
        if info and info.id:
            self._display_model_name = info.id
            return self._display_model_name
        return None

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", method="GET")
        return self._send(request)

    def _props_or_empty(self) -> dict[str, Any]:
        try:
            return self._get_json("/props")
        except LlamaServerError:
            return {}

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
            raise LlamaServerError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LlamaServerError(f"cannot connect to llama-server at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LlamaServerError(f"llama-server request timed out after {self.timeout:.0f}s") from exc

    def _send(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LlamaServerError(f"cannot connect to llama-server at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LlamaServerError(f"llama-server request timed out after {self.timeout:.0f}s") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LlamaServerError(f"llama-server returned invalid JSON: {raw[:200]}") from exc
        if not isinstance(data, dict):
            raise LlamaServerError("llama-server returned a non-object JSON response")
        return data

def _parse_chat_result(data: dict[str, Any]) -> ChatResult:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlamaServerError("llama-server response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LlamaServerError("llama-server choice is invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LlamaServerError("llama-server choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []

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
            on_delta(text)
        _merge_stream_tool_calls(tool_calls_by_index, delta.get("tool_calls"))

    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return ChatResult(
        content="".join(content_parts),
        model=model,
        finish_reason=finish_reason,
        tool_calls=[tool_calls_by_index[index] for index in sorted(tool_calls_by_index)],
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


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
