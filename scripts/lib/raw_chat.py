from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_chat(base_url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"cannot connect to llama-server: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SystemExit(f"request timed out after {timeout:.0f}s") from exc
    return json.loads(raw)


def assistant_message(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"role": "assistant", "content": ""}
    first = choices[0]
    if not isinstance(first, dict):
        return {"role": "assistant", "content": ""}
    message = first.get("message")
    if not isinstance(message, dict):
        return {"role": "assistant", "content": ""}
    return message


def assistant_content(data: dict[str, Any]) -> str:
    content = assistant_message(data).get("content")
    return content if isinstance(content, str) else ""


def finish_reason(data: dict[str, Any]) -> str | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    value = first.get("finish_reason")
    return value if isinstance(value, str) else None


def prompt_tokens(data: dict[str, Any]) -> int | None:
    return int_value(usage(data).get("prompt_tokens"))


def cached_tokens(data: dict[str, Any]) -> int | None:
    details = usage(data).get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    return int_value(details.get("cached_tokens"))


def prompt_per_second(data: dict[str, Any]) -> float | None:
    return float_value(timings(data).get("prompt_per_second"))


def predicted_per_second(data: dict[str, Any]) -> float | None:
    return float_value(timings(data).get("predicted_per_second"))


def cache_ratio(data: dict[str, Any]) -> float | None:
    cached = cached_tokens(data)
    prompt = prompt_tokens(data)
    if cached is None or not prompt:
        return None
    return cached / prompt


def usage(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("usage")
    return value if isinstance(value, dict) else {}


def timings(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("timings")
    return value if isinstance(value, dict) else {}


def int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def float_value(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None
