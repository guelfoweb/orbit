from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_url: str
    model: str | None
    timeout: int
    workdir: Path
    max_loops: int
    temperature: float
    think_mode: str
    show_thinking: bool
    think_explicit: bool
    show_thinking_explicit: bool
    skill_ref: str | None
    session_name: str | None
    prompt: str | None
    max_loops_explicit: bool = False
    debug_timing: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Ollama REPL with local tool calling")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--max-loops", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--skill")
    parser.add_argument("--session")
    parser.add_argument("prompt", nargs="*")
    return parser


def parse_config(argv: list[str] | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    argv = argv or []
    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        raise ValueError(f"workdir not found: {workdir}")
    if not workdir.is_dir():
        raise ValueError(f"workdir is not a directory: {workdir}")
    think_explicit = "--think" in argv
    show_thinking_explicit = "--show-thinking" in argv
    max_loops_explicit = "--max-loops" in argv
    return AppConfig(
        base_url=args.base_url,
        model=args.model,
        timeout=max(1, args.timeout),
        workdir=workdir,
        max_loops=max(1, args.max_loops),
        max_loops_explicit=max_loops_explicit,
        temperature=args.temperature,
        think_mode=args.think,
        show_thinking=bool(args.show_thinking),
        think_explicit=think_explicit,
        show_thinking_explicit=show_thinking_explicit,
        skill_ref=args.skill,
        session_name=args.session,
        prompt=" ".join(args.prompt).strip() or None,
        debug_timing=bool(args.debug_timing),
    )
