#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime import ChatRuntime
from orbit.runtime.turn_trace import ModelStepMetrics
from orbit.terminal.config import DEFAULT_SYSTEM_PROMPT


DEFAULT_BASE_URL = "http://127.0.0.1:18080"
DEFAULT_MODEL = "gemma4:12b"


@dataclass(frozen=True)
class CapturedCall:
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    result: ChatResult


@dataclass(frozen=True)
class ProbeMetrics:
    prompt_tokens: int | None
    cached_tokens: int | None
    new_tokens: int | None
    prefill_ms: float | None


class CapturingBackend(LlamaServerBackend):
    def __init__(self, *, base_url: str, model: str, timeout: float) -> None:
        super().__init__(base_url=base_url, model=model, timeout=timeout)
        self.calls: list[CapturedCall] = []

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        result = super().chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        self.calls.append(CapturedCall(copy.deepcopy(messages), copy.deepcopy(tools), result))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Orbit prompt components with differential llama-server measurements.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--prompt", default="show a tree of this workdir files and directories")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        prepare_workdir(workdir)
        backend = CapturingBackend(base_url=args.base_url, model=args.model, timeout=args.timeout)
        runtime = ChatRuntime(backend=backend, system_prompt=DEFAULT_SYSTEM_PROMPT)
        steps: list[ModelStepMetrics] = []
        started = time.monotonic()
        result = runtime.ask_auto(
            args.prompt,
            temperature=0,
            max_tokens=args.max_tokens,
            workdir=workdir,
            on_model_step=steps.append,
        )
        elapsed = time.monotonic() - started

        print(f"workdir: {workdir}", flush=True)
        print(f"prompt: {args.prompt}", flush=True)
        print(f"answer: {one_line(result.content)}", flush=True)
        print(f"wall: {elapsed:.1f}s", flush=True)
        print(
            "note: component rows are differential measurements; cached/new/prefill deltas can be negative "
            "because llama-server prompt cache checkpoints are not additive per component.",
            flush=True,
        )
        print("", flush=True)

        for index, (call, step) in enumerate(zip(backend.calls, steps), start=1):
            print_phase(index, call, step, args)
    return 0


def prepare_workdir(workdir: Path) -> None:
    (workdir / "docs").mkdir()
    (workdir / "docs" / "nested").mkdir()
    (workdir / "summary.txt").write_text(
        "Orbit is a local CLI for llama-server. It focuses on small tool surfaces and stable local workflows.\n",
        encoding="utf-8",
    )
    (workdir / "medium.py").write_text("print('hello')\n", encoding="utf-8")
    (workdir / "long.txt").write_text("local inference\n" * 200, encoding="utf-8")


def print_phase(index: int, call: CapturedCall, step: ModelStepMetrics, args: argparse.Namespace) -> None:
    full = metrics_from_result(call.result)
    no_descriptions_tools = strip_tool_descriptions(call.tools)
    without_tools = None
    full_probe = probe(args, call.messages, call.tools)
    no_tools_probe = probe(args, call.messages, without_tools) if call.tools else zero_metrics()
    no_descriptions_probe = probe(args, call.messages, no_descriptions_tools) if call.tools else zero_metrics()
    components = [
        ("system prompt", probe(args, clear_system_prompt(call.messages), call.tools), full_probe),
        ("conversation history", probe(args, clear_conversation_history(call.messages), call.tools), full_probe),
        ("tool result", probe(args, clear_tool_results(call.messages), call.tools), full_probe),
        ("user message", probe(args, clear_last_user_message(call.messages), call.tools), full_probe),
    ]
    if call.tools:
        tool_schema = diff_metrics(no_descriptions_probe, no_tools_probe)
        tool_descriptions = diff_metrics(full_probe, no_descriptions_probe)
    else:
        tool_schema = zero_metrics()
        tool_descriptions = zero_metrics()

    print(
        f"## inference {index}: {step.phase} | finish={step.finish_reason or '-'} | "
        f"in={value(step.prompt_tokens)} out={value(step.completion_tokens)} "
        f"cached={value(step.cached_tokens)} new={value(new_tokens(step.prompt_tokens, step.cached_tokens))} "
        f"prefill_ms={ms_from_rate(step.prompt_tokens, step.prompt_tokens_per_second)}",
        flush=True,
    )
    print("component | tokens | pct | cached | new | prefill_ms", flush=True)
    print("----------|--------|-----|--------|-----|-----------", flush=True)
    print_component("system prompt", diff_metrics(full_probe, components[0][1]), full.prompt_tokens)
    print_component("tool descriptions", tool_descriptions, full.prompt_tokens)
    print_component("tool schema", tool_schema, full.prompt_tokens)
    print_component("conversation history", diff_metrics(full_probe, components[1][1]), full.prompt_tokens)
    print_component("tool result", diff_metrics(full_probe, components[2][1]), full.prompt_tokens)
    print_component("user message", diff_metrics(full_probe, components[3][1]), full.prompt_tokens)
    print("", flush=True)


def probe(args: argparse.Namespace, messages: list[Message], tools: list[dict[str, Any]] | None) -> ProbeMetrics:
    backend = LlamaServerBackend(base_url=args.base_url, model=args.model, timeout=args.timeout)
    result = backend.chat(messages, temperature=0, max_tokens=1, tools=tools)
    return metrics_from_result(result)


def metrics_from_result(result: ChatResult) -> ProbeMetrics:
    return ProbeMetrics(
        prompt_tokens=result.prompt_tokens,
        cached_tokens=result.cached_tokens,
        new_tokens=new_tokens(result.prompt_tokens, result.cached_tokens),
        prefill_ms=ms_value(result.prompt_tokens, result.prompt_tokens_per_second),
    )


def diff_metrics(full: ProbeMetrics, reduced: ProbeMetrics) -> ProbeMetrics:
    return ProbeMetrics(
        prompt_tokens=diff_optional(full.prompt_tokens, reduced.prompt_tokens),
        cached_tokens=diff_optional(full.cached_tokens, reduced.cached_tokens),
        new_tokens=diff_optional(full.new_tokens, reduced.new_tokens),
        prefill_ms=diff_optional_float(full.prefill_ms, reduced.prefill_ms),
    )


def strip_tool_descriptions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    stripped = copy.deepcopy(tools)
    remove_description_keys(stripped)
    return stripped


def remove_description_keys(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("description", None)
        for item in value.values():
            remove_description_keys(item)
    elif isinstance(value, list):
        for item in value:
            remove_description_keys(item)


def clear_system_prompt(messages: list[Message]) -> list[Message]:
    copied = copy.deepcopy(messages)
    for message in copied:
        if message.get("role") == "system":
            message["content"] = ""
    return copied


def clear_conversation_history(messages: list[Message]) -> list[Message]:
    copied = copy.deepcopy(messages)
    last_user_index = last_role_index(copied, "user")
    for index, message in enumerate(copied):
        if message.get("role") in {"system", "tool"}:
            continue
        if index == last_user_index:
            continue
        message["content"] = ""
        if "tool_calls" in message:
            message["tool_calls"] = []
    return copied


def clear_tool_results(messages: list[Message]) -> list[Message]:
    copied = copy.deepcopy(messages)
    for message in copied:
        if message.get("role") == "tool":
            message["content"] = ""
    return copied


def clear_last_user_message(messages: list[Message]) -> list[Message]:
    copied = copy.deepcopy(messages)
    index = last_role_index(copied, "user")
    if index is not None:
        copied[index]["content"] = ""
    return copied


def last_role_index(messages: list[Message], role: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == role:
            return index
    return None


def print_component(name: str, metrics: ProbeMetrics, total_tokens: int | None) -> None:
    print(
        f"{name} | {value(metrics.prompt_tokens)} | {percent_of(metrics.prompt_tokens, total_tokens)} | "
        f"{value(metrics.cached_tokens)} | {value(metrics.new_tokens)} | {float_value(metrics.prefill_ms)}",
        flush=True,
    )


def zero_metrics() -> ProbeMetrics:
    return ProbeMetrics(prompt_tokens=0, cached_tokens=0, new_tokens=0, prefill_ms=0.0)


def new_tokens(prompt_tokens: int | None, cached_tokens: int | None) -> int | None:
    if prompt_tokens is None or cached_tokens is None:
        return None
    return prompt_tokens - cached_tokens


def ms_value(tokens: int | None, tokens_per_second: float | None) -> float | None:
    if tokens is None or not tokens_per_second:
        return None
    return tokens / tokens_per_second * 1000


def ms_from_rate(tokens: int | None, tokens_per_second: float | None) -> str:
    value = ms_value(tokens, tokens_per_second)
    return float_value(value)


def diff_optional(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left - right


def diff_optional_float(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def value(item: int | None) -> str:
    return str(item) if item is not None else "-"


def float_value(item: float | None) -> str:
    return f"{item:.0f}" if item is not None else "-"


def percent_of(item: int | None, total: int | None) -> str:
    if item is None or not total:
        return "-"
    return f"{item / total * 100:.0f}%"


def one_line(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) > 160:
        return compact[:157] + "..."
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
