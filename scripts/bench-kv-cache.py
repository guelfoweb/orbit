#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:18080"
DEFAULT_MODEL = "gemma4:12b"


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure llama-server prompt/KV cache reuse across consecutive chat turns.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--prefix-repeats", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a concise local assistant. Keep answers short. "
                + " ".join(["stable orbit kv cache probe prefix"] * args.prefix_repeats)
            ),
        }
    ]
    prompts = [
        "Answer in one sentence: what is prompt caching?",
        "Answer in one sentence: why does prompt caching matter on CPU-only inference?",
        "Answer in one sentence: what stayed stable across the previous turns?",
    ]

    print(f"prefix_repeats: {args.prefix_repeats}", flush=True)
    print("turn | prompt | cached | cache% | pf/s | gen/s | wall | finish", flush=True)
    print("-----|--------|--------|--------|------|-------|------|-------", flush=True)
    for index, prompt in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": prompt})
        started = time.monotonic()
        data = post_chat(
            args.base_url,
            {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "cache_prompt": True,
            },
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - started
        assistant = assistant_content(data)
        messages.append({"role": "assistant", "content": assistant})
        print(format_row(index, data, elapsed), flush=True)
    return 0


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


def assistant_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def format_row(index: int, data: dict[str, Any], elapsed: float) -> str:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    finish = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
    prompt_tokens = int_value(usage.get("prompt_tokens"))
    cached_tokens = int_value(nested(usage, "prompt_tokens_details", "cached_tokens"))
    cache_ratio = ratio(cached_tokens, prompt_tokens)
    return (
        f"{index} | {value(prompt_tokens)} | {value(cached_tokens)} | {cache_ratio} | "
        f"{float_value(timings.get('prompt_per_second'))} | "
        f"{float_value(timings.get('predicted_per_second'))} | "
        f"{elapsed:.1f}s | {finish or '-'}"
    )


def nested(data: dict[str, Any], key: str, child: str) -> Any:
    value = data.get(key)
    if not isinstance(value, dict):
        return None
    return value.get(child)


def int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def value(item: int | None) -> str:
    return str(item) if item is not None else "-"


def ratio(part: int | None, total: int | None) -> str:
    if part is None or not total:
        return "-"
    return f"{part / total * 100:.0f}%"


def float_value(item: Any) -> str:
    if not isinstance(item, int | float):
        return "-"
    return f"{float(item):.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
