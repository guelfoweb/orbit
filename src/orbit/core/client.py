from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

try:
    from ollama import Client as OllamaSdkClient
    from ollama import ResponseError
except ImportError:  # pragma: no cover
    OllamaSdkClient = None
    ResponseError = Exception


class OllamaError(RuntimeError):
    pass


def is_thinking_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "does not support thinking" in text or "thinking" in text and "status code: 400" in text


def _format_connection_error(exc: Exception, timeout: int) -> str:
    text = str(exc)
    if "timed out" in text.lower():
        return (
            f"connection failed: timed out. "
            f"Try a higher timeout, for example: orbit --timeout {max(timeout * 2, timeout + 300)}"
        )
    return f"connection failed: {exc}"


@dataclass(frozen=True)
class ModelMetadata:
    active_model: str
    context_window: int | None
    capabilities: tuple[str, ...]
    tools_supported: bool | None
    parameter_size: str | None = None


def _normalize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL must use http or https")
    if not parsed.netloc:
        raise ValueError("base URL must include a host")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not include query parameters or fragments")
    return base_url.rstrip("/")


@dataclass
class OllamaClient:
    base_url: str
    model: str | None = None
    timeout: int = 300

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)
        if OllamaSdkClient is None:
            raise OllamaError("missing dependency: install the Python package 'ollama'")
        self._client = OllamaSdkClient(host=self.base_url, timeout=self.timeout)
        if self.model is None:
            self.model = self.resolve_running_model()

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        think: bool | str | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "tools": tools,
        }
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        return self._chat(payload)

    def inspect_model(self) -> ModelMetadata:
        if self.model is None:
            raise OllamaError("no model selected")
        body = self._show({"model": self.model})
        context_window = self._extract_context_window(body)
        capabilities = self._extract_capabilities(body)
        return ModelMetadata(
            active_model=self.model,
            context_window=context_window,
            capabilities=capabilities,
            tools_supported=self._extract_tools_supported(capabilities),
            parameter_size=self._extract_parameter_size(body),
        )

    def resolve_running_model(self) -> str:
        body = self._ps()
        model_name = self._extract_running_model(body)
        if model_name is None:
            raise OllamaError("no running Ollama model found; start one first or pass --model")
        return model_name

    def _chat(self, payload: dict[str, Any]):
        try:
            response = self._client.chat(**payload)
        except ResponseError as exc:
            raise OllamaError(f"ollama error: {exc}") from exc
        except Exception as exc:
            raise OllamaError(_format_connection_error(exc, self.timeout)) from exc
        if payload.get("stream"):
            return self._stream_to_dicts(response)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        raise OllamaError("invalid response returned by Ollama client")

    def _ps(self) -> dict[str, Any]:
        try:
            response = self._client.ps()
        except ResponseError as exc:
            raise OllamaError(f"ollama error: {exc}") from exc
        except Exception as exc:
            raise OllamaError(_format_connection_error(exc, self.timeout)) from exc
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        raise OllamaError("invalid process list returned by Ollama client")

    def _show(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.show(payload["model"])
        except ResponseError as exc:
            raise OllamaError(f"ollama error: {exc}") from exc
        except Exception as exc:
            raise OllamaError(_format_connection_error(exc, self.timeout)) from exc
        if hasattr(response, "model_dump"):
            body = response.model_dump()
            modelinfo = getattr(response, "modelinfo", None)
            if "model_info" not in body and isinstance(modelinfo, dict):
                body["model_info"] = dict(modelinfo)
            return body
        if isinstance(response, dict):
            return response
        raise OllamaError("invalid metadata returned by Ollama client")

    @staticmethod
    def _extract_context_window(body: dict[str, Any]) -> int | None:
        parameter_text = body.get("parameters")
        parsed_parameter_ctx = OllamaClient._extract_num_ctx_from_parameters(parameter_text)
        if parsed_parameter_ctx is not None:
            return parsed_parameter_ctx
        model_info = body.get("model_info") or body.get("modelinfo")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                    return value
            for key in ("context_length", "n_ctx_train", "num_ctx"):
                value = model_info.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        details = body.get("details")
        if isinstance(details, dict):
            for key in ("context_length", "num_ctx"):
                value = details.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        return None

    @staticmethod
    def _extract_num_ctx_from_parameters(parameters: Any) -> int | None:
        if not isinstance(parameters, str):
            return None
        for line in parameters.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                try:
                    value = int(parts[-1])
                except ValueError:
                    return None
                if value > 0:
                    return value
        return None

    @staticmethod
    def _extract_running_model(body: dict[str, Any]) -> str | None:
        models = body.get("models")
        if not isinstance(models, list) or not models:
            return None
        first = models[0]
        if not isinstance(first, dict):
            return None
        for key in ("model", "name"):
            value = first.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _extract_capabilities(body: dict[str, Any]) -> tuple[str, ...]:
        capabilities = body.get("capabilities")
        if not isinstance(capabilities, list):
            return ()
        normalized: list[str] = []
        for item in capabilities:
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip().lower())
        return tuple(normalized)

    @staticmethod
    def _extract_tools_supported(capabilities: tuple[str, ...]) -> bool | None:
        if not capabilities:
            return None
        return "tools" in capabilities

    @staticmethod
    def _extract_parameter_size(body: dict[str, Any]) -> str | None:
        details = body.get("details")
        if isinstance(details, dict):
            for key in ("parameter_size", "parameters"):
                value = details.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
        model_info = body.get("model_info") or body.get("modelinfo")
        if isinstance(model_info, dict):
            for key in ("parameter_size", "parameters"):
                value = model_info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
        return None

    @staticmethod
    def _stream_to_dicts(response):
        for chunk in response:
            if hasattr(chunk, "model_dump"):
                yield chunk.model_dump()
            elif isinstance(chunk, dict):
                yield chunk
            else:
                raise OllamaError("invalid streamed response returned by Ollama client")
