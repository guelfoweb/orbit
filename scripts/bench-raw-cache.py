#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from raw_chat import (
    assistant_content,
    assistant_message,
    cache_ratio,
    cached_tokens,
    finish_reason,
    post_chat,
    predicted_per_second,
    prompt_per_second,
    prompt_tokens,
)


DEFAULT_BASE_URL = "http://127.0.0.1:18080"
DEFAULT_MODEL = "gemma4:12b"
SYSTEM_PROMPT = "You are a concise local assistant. Keep answers short."


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw llama-server cache probe without Orbit runtime.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--mode", choices=("chat", "tools", "monolith", "all"), default="all")
    args = parser.parse_args()

    modes = ["chat", "tools", "monolith"] if args.mode == "all" else [args.mode]
    print("mode | turn | prompt | cached | cache% | pf/s | gen/s | wall | finish | note", flush=True)
    print("-----|------|--------|--------|--------|------|-------|------|--------|-----", flush=True)
    for mode in modes:
        if mode == "chat":
            run_chat(args)
        elif mode == "tools":
            run_tools(args)
        elif mode == "monolith":
            run_monolith(args)
    return 0


def run_chat(args: argparse.Namespace) -> None:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompts = [
        "Answer in one short sentence: what is local inference?",
        "Answer in one short sentence: why does prompt cache matter?",
        "Answer in one short sentence: what did we discuss in the previous answer?",
    ]
    for index, prompt in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": prompt})
        data, elapsed = request(args, messages)
        messages.append(clean_assistant_for_replay(assistant_message(data)))
        print(row("chat", index, data, elapsed, "-"), flush=True)


def run_tools(args: argparse.Namespace) -> None:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    tools = tool_definitions()

    # Turn 1: let the model produce a real tool call, then inject the result.
    messages.append({"role": "user", "content": "list files in this directory"})
    data, elapsed = request(args, messages, tools=tools)
    assistant = clean_assistant_for_replay(assistant_message(data))
    messages.append(assistant)
    print(row("tools", 1, data, elapsed, "tool-call"), flush=True)
    tool_call = first_tool_call(assistant)
    if tool_call:
        messages.append(tool_message(tool_call, "alpha.txt\nbeta.md\nnote.txt\nreport.pdf"))
        data, elapsed = request(args, messages, tools=tools)
        messages.append(clean_assistant_for_replay(assistant_message(data)))
        print(row("tools", 2, data, elapsed, "after-tool"), flush=True)

    # Turn 2: keep normal replay after a completed tool turn.
    messages.append({"role": "user", "content": "read note.txt and summarize it in one sentence"})
    data, elapsed = request(args, messages, tools=tools)
    assistant = clean_assistant_for_replay(assistant_message(data))
    messages.append(assistant)
    print(row("tools", 3, data, elapsed, "tool-call"), flush=True)
    tool_call = first_tool_call(assistant)
    if tool_call:
        messages.append(tool_message(tool_call, "Orbit reads UTF-8 text files correctly.\n"))
        data, elapsed = request(args, messages, tools=tools)
        messages.append(clean_assistant_for_replay(assistant_message(data)))
        print(row("tools", 4, data, elapsed, "after-tool"), flush=True)


def run_monolith(args: argparse.Namespace) -> None:
    transcript = f"SYSTEM: {SYSTEM_PROMPT}\n"
    prompts = [
        "USER: Answer in one short sentence: what is local inference?",
        "USER: Answer in one short sentence: why does prompt cache matter?",
        "USER: Answer in one short sentence: what did we discuss in the previous answer?",
    ]
    for index, prompt in enumerate(prompts, start=1):
        transcript += prompt + "\n"
        messages = [{"role": "user", "content": transcript + "ASSISTANT:"}]
        data, elapsed = request(args, messages)
        answer = assistant_content(data).strip()
        transcript += f"ASSISTANT: {answer}\n"
        print(row("monolith", index, data, elapsed, "-"), flush=True)


def request(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "cache_prompt": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
        payload["parse_tool_calls"] = True
    started = time.monotonic()
    data = post_chat(args.base_url, payload, timeout=args.timeout)
    return data, time.monotonic() - started


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List workdir files/directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative directory. Defaults to '.'."}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text/source file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file."},
                        "chunk_index": {"type": "integer", "description": "Zero-based chunk for large files."},
                        "chunk_chars": {"type": "integer", "description": "Chunk size. Default 12000, max 24000."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stat_path",
                "description": "Inspect bounded metadata for a workdir path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch an explicit http/https URL and return bounded readable text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Explicit http/https URL."},
                        "chunk_index": {"type": "integer", "description": "Zero-based chunk for long fetched pages."},
                        "chunk_chars": {"type": "integer", "description": "Chunk size. Default 6000, max 24000."},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web and return bounded structured results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "max_results": {"type": "integer", "description": "Number of results. Default 5, max 8."},
                        "site": {"type": "string", "description": "Optional bare domain filter, for example example.com. Do not pass full URLs."},
                        "timelimit": {"type": "string", "description": "Optional time filter: d, w, m, or y."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a new UTF-8 text/source file at an explicit workdir path. Never overwrites.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative target file path. Parent directory must already exist."},
                        "content": {"type": "string", "description": "UTF-8 text content, max 65536 characters."},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append UTF-8 text to an existing text/source file at an explicit workdir path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative existing file path."},
                        "content": {"type": "string", "description": "UTF-8 text to append, max 16384 characters."},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": "Replace one unique exact UTF-8 text fragment in an existing text/source file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative existing file path."},
                        "old": {"type": "string", "description": "Exact text to replace, max 16384 characters."},
                        "new": {"type": "string", "description": "Replacement UTF-8 text, max 16384 characters."},
                    },
                    "required": ["path", "old", "new"],
                },
            },
        },
    ]


def clean_assistant_for_replay(message: dict[str, Any]) -> dict[str, Any]:
    replay = copy.deepcopy(message)
    replay["role"] = "assistant"
    if "content" not in replay or replay["content"] is None:
        replay["content"] = ""
    return replay


def first_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first = tool_calls[0]
    return first if isinstance(first, dict) else None


def tool_message(tool_call: dict[str, Any], content: str) -> dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    return {
        "role": "tool",
        "tool_call_id": str(tool_call.get("id") or "tool-call"),
        "name": str(function.get("name") or "unknown"),
        "content": content,
    }


def row(mode: str, turn: int, data: dict[str, Any], elapsed: float, note: str) -> str:
    return (
        f"{mode} | {turn} | {value(prompt_tokens(data))} | {value(cached_tokens(data))} | "
        f"{percent(cache_ratio(data))} | {float_value(prompt_per_second(data))} | "
        f"{float_value(predicted_per_second(data))} | {elapsed:.1f}s | {finish_reason(data) or '-'} | {note}"
    )


def value(item: int | None) -> str:
    return str(item) if item is not None else "-"


def percent(item: float | None) -> str:
    return f"{item * 100:.0f}%" if item is not None else "-"


def float_value(item: float | None) -> str:
    return f"{item:.1f}" if item is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())
