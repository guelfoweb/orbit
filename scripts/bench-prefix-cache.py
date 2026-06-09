#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BenchCase:
    name: str
    prompt: str
    max_tokens: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Orbit phase-level prompt-cache reuse.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--case", choices=("chat_short", "read_short", "read_medium", "read_long", "list_tree"))
    args = parser.parse_args()

    cases = [
        BenchCase("chat_short", "Answer in one short sentence: what is local inference?", 96),
        BenchCase("read_short", "read summary.txt and summarize it in one sentence", 128),
        BenchCase("read_medium", "read medium.py and summarize what the code does in three bullets", 192),
        BenchCase("read_long", "read long.txt and summarize it in five concise lines", 256),
        BenchCase("list_tree", "show a tree of this workdir files and directories", 128),
    ]
    if args.case:
        cases = [case for case in cases if case.name == args.case]

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        prepare_workdir(workdir)
        runtime = ChatRuntime(
            backend=LlamaServerBackend(base_url=args.base_url, model=args.model, timeout=args.timeout),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        print(f"workdir: {workdir}", flush=True)
        print(
            "case | inf | phase | in | out | cached | new | cache% | prefill_ms | generation_ms | wall | finish | tools",
            flush=True,
        )
        print(
            "-----|-----|-------|----|-----|--------|-----|--------|------------|---------------|------|--------|------",
            flush=True,
        )
        totals: list[tuple[str, float, int]] = []
        for case in cases:
            steps: list[ModelStepMetrics] = []
            started = time.monotonic()
            result = runtime.ask_auto(
                case.prompt,
                temperature=0,
                max_tokens=case.max_tokens,
                workdir=workdir,
                on_model_step=steps.append,
            )
            elapsed = time.monotonic() - started
            for index, step in enumerate(steps, start=1):
                print(format_row(case.name, index, step, elapsed if index == len(steps) else None), flush=True)
            totals.append((case.name, elapsed, len(steps)))
            print(f"answer[{case.name}]: {one_line(result.content)}", flush=True)
        print("total | wall | infer", flush=True)
        for name, elapsed, infer_count in sorted(totals, key=lambda item: item[1], reverse=True):
            print(f"{name} | {elapsed:.1f}s | {infer_count}", flush=True)
    return 0


def prepare_workdir(workdir: Path) -> None:
    (workdir / "docs").mkdir()
    (workdir / "docs" / "nested").mkdir()
    (workdir / "summary.txt").write_text(
        "Orbit is a local CLI for llama-server. It focuses on small tool surfaces and stable local workflows.\n",
        encoding="utf-8",
    )
    (workdir / "medium.py").write_text(_medium_source(), encoding="utf-8")
    long_block = (
        "Inferno canto one begins with a lost traveler in a dark wood. "
        "Virgil appears as a guide after the traveler is blocked by symbolic beasts. "
        "The scene introduces fear, moral confusion, and the need for guidance.\n"
    )
    (workdir / "long.txt").write_text("".join(long_block for _ in range(1500)), encoding="utf-8")
    for index in range(8):
        (workdir / "docs" / f"note-{index}.md").write_text(f"# note {index}\n", encoding="utf-8")


def _medium_source() -> str:
    header = """from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Item:
    name: str
    score: int


def normalize(items: list[Item]) -> list[Item]:
    if not items:
        return []
    highest = max(item.score for item in items) or 1
    return [Item(item.name, int(item.score * 100 / highest)) for item in items]


def report(items: list[Item]) -> str:
    normalized = normalize(items)
    return "\\n".join(f"{item.name}: {item.score}" for item in normalized)

"""
    return header + "\n".join(f"# filler line {index}" for index in range(900)) + "\n"


def format_row(case: str, index: int, step: ModelStepMetrics, elapsed: float | None) -> str:
    cached = step.cached_tokens
    prompt = step.prompt_tokens
    new_tokens = prompt - cached if prompt is not None and cached is not None else None
    return (
        f"{case} | {index} | {step.phase} | {value(prompt)} | {value(step.completion_tokens)} | "
        f"{value(cached)} | {value(new_tokens)} | {percent(cache_ratio(step))} | "
        f"{ms_from_rate(prompt, step.prompt_tokens_per_second)} | "
        f"{ms_from_rate(step.completion_tokens, step.generation_tokens_per_second)} | "
        f"{elapsed_text(elapsed)} | {step.finish_reason or '-'} | {step.tool_calls}"
    )


def value(item: int | None) -> str:
    return str(item) if item is not None else "-"


def percent(item: float | None) -> str:
    return f"{item * 100:.0f}%" if item is not None else "-"


def ms_from_rate(tokens: int | None, tokens_per_second: float | None) -> str:
    if tokens is None or not tokens_per_second:
        return "-"
    return f"{tokens / tokens_per_second * 1000:.0f}"


def elapsed_text(item: float | None) -> str:
    return f"{item:.1f}s" if item is not None else "-"


def one_line(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) > 140:
        return compact[:137] + "..."
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
