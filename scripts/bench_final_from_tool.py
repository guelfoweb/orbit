#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.messages import DEFAULT_SYSTEM_PROMPT


WORKDIR = Path("workdir")


class ProbeBackend:
    def __init__(self, backend: LlamaServerBackend) -> None:
        self._backend = backend
        self.calls: list[dict[str, object]] = []

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def _classify(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None) -> str:
        if tools is not None:
            return "tool_call"
        if any(message.get("role") == "tool" for message in messages):
            return "final_from_tool"
        return "chat"

    def _message_chars(self, messages: list[dict[str, object]]) -> tuple[int, int]:
        total = 0
        tool_chars = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                serialized = content
            else:
                serialized = json.dumps(content, ensure_ascii=False) if content is not None else ""
            total += len(serialized)
            if message.get("role") == "tool":
                tool_chars += len(serialized)
        return total, tool_chars

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        phase = self._classify(messages, tools)
        total_chars, tool_chars = self._message_chars(messages)
        started = time.perf_counter()
        result = self._backend.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        self.calls.append(
            {
                "phase": phase,
                "max_tokens": max_tokens,
                "total_chars": total_chars,
                "tool_chars": tool_chars,
                "finish_reason": result.finish_reason,
                "tool_calls": len(result.tool_calls),
                "content_chars": len(result.content),
                "wall_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        return result

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)

    def continue_current(self, *, max_tokens, on_delta=None, on_progress=None):
        started = time.perf_counter()
        result = self._backend.continue_current(max_tokens=max_tokens, on_delta=on_delta, on_progress=on_progress)
        self.calls.append(
            {
                "phase": "continue_native",
                "max_tokens": max_tokens,
                "total_chars": 0,
                "tool_chars": 0,
                "finish_reason": result.finish_reason,
                "tool_calls": len(result.tool_calls),
                "content_chars": len(result.content),
                "wall_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        return result


def _run_case(
    name: str,
    fn: Callable[[ChatRuntime, list[dict[str, object]]], object],
    *,
    base_url: str,
    thinking: bool = False,
) -> dict[str, object]:
    backend = ProbeBackend(LlamaServerBackend(base_url=base_url, timeout=300, thinking=thinking))
    runtime = ChatRuntime(backend=backend, system_prompt=DEFAULT_SYSTEM_PROMPT)
    model_steps: list[dict[str, object]] = []
    started = time.perf_counter()
    result = fn(runtime, model_steps)
    final_calls = [call for call in backend.calls if call["phase"] == "final_from_tool"]
    return {
        "case": name,
        "wall_ms": round((time.perf_counter() - started) * 1000, 1),
        "tool_calls": sum(1 for call in backend.calls if call["phase"] == "tool_call" and call["tool_calls"]),
        "tool_wall_ms": round(sum(float(call["wall_ms"]) for call in backend.calls if call["phase"] == "tool_call"), 1),
        "final_from_tool_calls": len(final_calls),
        "final_from_tool_tool_chars": sum(int(call["tool_chars"]) for call in final_calls),
        "final_from_tool_prompt_chars": sum(int(call["total_chars"]) for call in final_calls),
        "finish_reason": result.finish_reason,
        "final_answer_chars": len(result.content),
        "answer": result.content.strip(),
        "model_steps": model_steps,
        "backend_calls": backend.calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark final_from_tool-heavy paths against a local Orbit server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12120")
    parser.add_argument("--workdir", default=str(WORKDIR))
    parser.add_argument("--json-out")
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()

    cases = [
        (
            "chat_simple",
            False,
            lambda rt, steps: rt.ask_chat(
                "Tell me what grep is used for in one sentence.",
                temperature=0.0,
                max_tokens=96,
                on_model_step=lambda step: steps.append(
                    {
                        "phase": step.phase,
                        "finish_reason": step.finish_reason,
                        "retry_reason": step.retry_reason,
                        "tool_calls": step.tool_calls,
                    }
                ),
            ),
        ),
        (
            "think_continue",
            True,
            lambda rt, steps: (
                rt.ask_chat(
                    "Explain the plan before the final answer for reviewing a local text file, but keep the final answer short.",
                    temperature=0.0,
                    max_tokens=48,
                    on_model_step=lambda step: steps.append(
                        {
                            "phase": step.phase,
                            "finish_reason": step.finish_reason,
                            "retry_reason": step.retry_reason,
                            "tool_calls": step.tool_calls,
                        }
                    ),
                ),
                rt.continue_last_response(
                    temperature=0.0,
                    max_tokens=96,
                    on_model_step=lambda step: steps.append(
                        {
                            "phase": step.phase,
                            "finish_reason": step.finish_reason,
                            "retry_reason": step.retry_reason,
                            "tool_calls": step.tool_calls,
                        }
                    ),
                ),
            )[1],
        ),
        (
            "shell_pwd",
            False,
            lambda rt, steps: rt.ask_auto(
                "Use the shell tool to print the current working directory.",
                temperature=0.0,
                max_tokens=96,
                workdir=workdir,
                max_loops=6,
                allowed_tool_names=("exec_shell_full_command",),
                on_model_step=lambda step: steps.append(
                    {
                        "phase": step.phase,
                        "finish_reason": step.finish_reason,
                        "retry_reason": step.retry_reason,
                        "tool_calls": step.tool_calls,
                    }
                ),
            ),
        ),
        (
            "text_summary",
            False,
            lambda rt, steps: rt.ask_auto(
                "Read text/summary.txt and summarize it in one concise sentence.",
                temperature=0.0,
                max_tokens=96,
                workdir=workdir,
                max_loops=6,
                allowed_tool_names=("exec_shell_full_command",),
                on_model_step=lambda step: steps.append(
                    {
                        "phase": step.phase,
                        "finish_reason": step.finish_reason,
                        "retry_reason": step.retry_reason,
                        "tool_calls": step.tool_calls,
                    }
                ),
            ),
        ),
        (
            "pdf_small",
            False,
            lambda rt, steps: rt.ask_auto(
                "Read pdf/piccolo.pdf and summarize the document topic in one concise sentence.",
                temperature=0.0,
                max_tokens=96,
                workdir=workdir,
                max_loops=6,
                allowed_tool_names=("exec_shell_full_command",),
                on_model_step=lambda step: steps.append(
                    {
                        "phase": step.phase,
                        "finish_reason": step.finish_reason,
                        "retry_reason": step.retry_reason,
                        "tool_calls": step.tool_calls,
                    }
                ),
            ),
        ),
        (
            "pdf_large",
            False,
            lambda rt, steps: rt.ask_auto(
                "Read pdf/grande.pdf and summarize the document topic in one concise sentence.",
                temperature=0.0,
                max_tokens=96,
                workdir=workdir,
                max_loops=6,
                allowed_tool_names=("exec_shell_full_command",),
                on_model_step=lambda step: steps.append(
                    {
                        "phase": step.phase,
                        "finish_reason": step.finish_reason,
                        "retry_reason": step.retry_reason,
                        "tool_calls": step.tool_calls,
                    }
                ),
            ),
        ),
        (
            "code_brief",
            False,
            lambda rt, steps: rt.ask_auto(
                "Inspect samples/vulnerable_service.py and tell me the main security issue briefly.",
                temperature=0.0,
                max_tokens=96,
                workdir=workdir,
                max_loops=6,
                allowed_tool_names=("exec_shell_full_command",),
                on_model_step=lambda step: steps.append(
                    {
                        "phase": step.phase,
                        "finish_reason": step.finish_reason,
                        "retry_reason": step.retry_reason,
                        "tool_calls": step.tool_calls,
                    }
                ),
            ),
        ),
        (
            "recursive_listing",
            False,
            lambda rt, steps: rt.ask_auto(
                "List all files and directories recursively.",
                temperature=0.0,
                max_tokens=128,
                workdir=workdir,
                max_loops=4,
                allowed_tool_names=("exec_shell_full_command",),
                on_model_step=lambda step: steps.append(
                    {
                        "phase": step.phase,
                        "finish_reason": step.finish_reason,
                        "retry_reason": step.retry_reason,
                        "tool_calls": step.tool_calls,
                    }
                ),
            ),
        ),
    ]

    results: list[dict[str, object]] = []
    for name, thinking, fn in cases:
        result = _run_case(name, fn, base_url=args.base_url, thinking=thinking)
        results.append(result)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
