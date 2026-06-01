from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 300
DEFAULT_WORKDIR = "."
DEFAULT_MAX_LOOPS = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_THINK_MODE = "auto"
DEFAULT_RENDER_MARKDOWN = True
DEFAULT_COLLAPSE_LONG_INPUT = True
DEFAULT_LONG_INPUT_PREVIEW_CHARS = 50
CONFIG_PATH = Path.home() / ".orbit" / "config.json"
TOP_LEVEL_CONFIG_KEYS = {"model", "host", "base_url", "workdir", "timeout", "think", "debug_timing", "ui", "tools"}
UI_CONFIG_KEYS = {"markdown", "collapse_long_input", "long_input_preview_chars"}
TOOLS_CONFIG_KEYS = {"max_loops"}


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
    render_markdown: bool = DEFAULT_RENDER_MARKDOWN
    collapse_long_input: bool = DEFAULT_COLLAPSE_LONG_INPUT
    long_input_preview_chars: int = DEFAULT_LONG_INPUT_PREVIEW_CHARS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Ollama REPL with local tool calling")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--workdir")
    parser.add_argument("--max-loops", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--think", choices=("auto", "on", "off"))
    parser.add_argument("--show-thinking", action="store_true", default=None)
    parser.add_argument("--debug-timing", action="store_true", default=None)
    parser.add_argument("--skill")
    parser.add_argument("--session")
    parser.add_argument("prompt", nargs="*")
    return parser


def parse_config(argv: list[str] | None = None, *, config_path: Path | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    argv = argv or []
    file_config = load_user_config(CONFIG_PATH if config_path is None else config_path)
    base_url = args.base_url or _config_str(file_config, "base_url") or _config_str(file_config, "host") or DEFAULT_BASE_URL
    model = args.model or _config_str(file_config, "model")
    timeout = args.timeout if args.timeout is not None else _config_int(file_config, "timeout", DEFAULT_TIMEOUT)
    workdir_raw = args.workdir or _config_str(file_config, "workdir") or DEFAULT_WORKDIR
    max_loops = args.max_loops if args.max_loops is not None else _config_nested_int(file_config, "tools", "max_loops", DEFAULT_MAX_LOOPS)
    temperature = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE
    think_mode = args.think or _config_str(file_config, "think") or DEFAULT_THINK_MODE
    if think_mode not in {"auto", "on", "off"}:
        raise ValueError("config key 'think' must be one of: auto, on, off")
    show_thinking = bool(args.show_thinking)
    debug_timing = (
        bool(args.debug_timing)
        if args.debug_timing is not None
        else _config_bool(file_config, "debug_timing", False)
    )
    render_markdown = _config_nested_bool(file_config, "ui", "markdown", DEFAULT_RENDER_MARKDOWN)
    collapse_long_input = _config_nested_bool(
        file_config, "ui", "collapse_long_input", DEFAULT_COLLAPSE_LONG_INPUT
    )
    long_input_preview_chars = _config_nested_int(
        file_config, "ui", "long_input_preview_chars", DEFAULT_LONG_INPUT_PREVIEW_CHARS
    )
    workdir = Path(workdir_raw).expanduser().resolve()
    if not workdir.exists():
        raise ValueError(f"workdir not found: {workdir}")
    if not workdir.is_dir():
        raise ValueError(f"workdir is not a directory: {workdir}")
    think_explicit = "--think" in argv or "think" in file_config
    show_thinking_explicit = "--show-thinking" in argv
    max_loops_explicit = "--max-loops" in argv or _has_nested_config_key(file_config, "tools", "max_loops")
    return AppConfig(
        base_url=base_url,
        model=model,
        timeout=max(1, timeout),
        workdir=workdir,
        max_loops=max(1, max_loops),
        max_loops_explicit=max_loops_explicit,
        temperature=temperature,
        think_mode=think_mode,
        show_thinking=show_thinking,
        think_explicit=think_explicit,
        show_thinking_explicit=show_thinking_explicit,
        skill_ref=args.skill,
        session_name=args.session,
        prompt=" ".join(args.prompt).strip() or None,
        debug_timing=debug_timing,
        render_markdown=render_markdown,
        collapse_long_input=collapse_long_input,
        long_input_preview_chars=max(1, long_input_preview_chars),
    )


def load_user_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config JSON at {path}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    _validate_config_keys(raw, path=path)
    return raw


def _validate_config_keys(config: dict[str, Any], *, path: Path) -> None:
    unknown = sorted(set(config) - TOP_LEVEL_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown config key in {path}: {unknown[0]}")
    for section, allowed in (("ui", UI_CONFIG_KEYS), ("tools", TOOLS_CONFIG_KEYS)):
        value = config.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"config key '{section}' must be an object")
        section_unknown = sorted(set(value) - allowed)
        if section_unknown:
            raise ValueError(f"unknown config key in '{section}': {section_unknown[0]}")


def _config_str(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"config key '{key}' must be a string")
    return value


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"config key '{key}' must be a boolean")
    return value


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config key '{key}' must be an integer")
    return value


def _config_nested_bool(config: dict[str, Any], section: str, key: str, default: bool) -> bool:
    nested = config.get(section)
    if nested is None:
        return default
    value = nested.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"config key '{section}.{key}' must be a boolean")
    return value


def _config_nested_int(config: dict[str, Any], section: str, key: str, default: int) -> int:
    nested = config.get(section)
    if nested is None:
        return default
    value = nested.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config key '{section}.{key}' must be an integer")
    return value


def _has_nested_config_key(config: dict[str, Any], section: str, key: str) -> bool:
    nested = config.get(section)
    return isinstance(nested, dict) and key in nested
