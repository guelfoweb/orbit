#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime import ChatRuntime
from orbit.runtime.turn_trace import ModelStepMetrics, cache_ratio
from orbit.terminal.config import DEFAULT_SYSTEM_PROMPT


DEFAULT_BASE_URL = "http://127.0.0.1:18080"
DEFAULT_MODEL = "gemma4:12b-it"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare prompt-cache reuse in no-tool and tool-call Orbit turns.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        prepare_workdir(workdir)
        print(f"workdir: {workdir}", flush=True)
        print("mode | turn | loop | phase | prompt | cached | cache% | pf/s | gen/s | wall | finish | tools", flush=True)
        print("-----|------|------|-------|--------|--------|--------|------|-------|------|--------|------", flush=True)
        run_chat_probe(args, workdir)
        run_tool_probe(args, workdir)
    return 0


def run_chat_probe(args: argparse.Namespace, workdir: Path) -> None:
    runtime = ChatRuntime(
        backend=LlamaServerBackend(base_url=args.base_url, model=args.model, timeout=args.timeout),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    prompts = [
        "Answer in one short sentence: what is local inference?",
        "Answer in one short sentence: why does prompt cache matter?",
        "Answer in one short sentence: what did we discuss in the previous answer?",
    ]
    for index, prompt in enumerate(prompts, start=1):
        steps: list[ModelStepMetrics] = []
        started = time.monotonic()
        runtime.ask(prompt, temperature=0, max_tokens=args.max_tokens, on_model_step=steps.append)
        elapsed = time.monotonic() - started
        for step in steps:
            print(format_row("chat", index, step, elapsed), flush=True)


def run_tool_probe(args: argparse.Namespace, workdir: Path) -> None:
    runtime = ChatRuntime(
        backend=LlamaServerBackend(base_url=args.base_url, model=args.model, timeout=args.timeout),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    prompts = [
        "list files in this directory",
        "read note.txt and summarize it in one sentence",
        "use available tools to read report.pdf and tell me what it contains",
    ]
    for index, prompt in enumerate(prompts, start=1):
        steps: list[ModelStepMetrics] = []
        started = time.monotonic()
        runtime.ask_with_tools(
            prompt,
            temperature=0,
            max_tokens=args.max_tokens,
            workdir=workdir,
            on_model_step=steps.append,
        )
        elapsed = time.monotonic() - started
        for step in steps:
            print(format_row("tools", index, step, elapsed if step is steps[-1] else None), flush=True)


def prepare_workdir(workdir: Path) -> None:
    (workdir / "docs").mkdir()
    (workdir / "alpha.txt").touch()
    (workdir / "beta.md").touch()
    (workdir / "note.txt").write_text("Orbit reads UTF-8 text files correctly.\n", encoding="utf-8")
    (workdir / "report.pdf").write_text("%PDF-1.7\n", encoding="utf-8")


def format_row(mode: str, turn_index: int, step: ModelStepMetrics, elapsed: float | None) -> str:
    ratio = cache_ratio(step)
    return (
        f"{mode} | {turn_index} | {step.loop} | {step.phase} | {value(step.prompt_tokens)} | "
        f"{value(step.cached_tokens)} | {percent(ratio)} | {float_value(step.prompt_tokens_per_second)} | "
        f"{float_value(step.generation_tokens_per_second)} | {elapsed_text(elapsed)} | "
        f"{step.finish_reason or '-'} | {step.tool_calls}"
    )


def value(item: int | None) -> str:
    return str(item) if item is not None else "-"


def percent(item: float | None) -> str:
    return f"{item * 100:.0f}%" if item is not None else "-"


def float_value(item: float | None) -> str:
    return f"{item:.1f}" if item is not None else "-"


def elapsed_text(item: float | None) -> str:
    return f"{item:.1f}s" if item is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())
