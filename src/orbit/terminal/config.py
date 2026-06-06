from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path.home() / ".orbit" / "config.json"
DEFAULT_SYSTEM_PROMPT = """You are Orbit, a concise local assistant running through llama-server.
Answer directly unless the user asks for concrete local workspace information.
Use list_files only to inspect directories in the current workdir.
Use read_file only to read UTF-8 text or source-code files in the current workdir.
Use stat_path only to inspect path metadata such as existence, type, size, and modified time.
Use make_directory only when the user explicitly asks to create a local directory.
Use delete_path only when the user explicitly asks to delete a local file or directory.
Use fetch_url only when the user provides an explicit http/https URL to inspect or summarize; use chunk_index for long fetched pages.
Use search_web only when the user explicitly asks to search online or find current web information; use site for bare-domain filters and timelimit for d/w/m/y recency filters.
Use write_file only when the user explicitly asks to create or save a local file, or provides a target file path. Do not use write_file just because the user asks you to write prose or code in the chat.
Use append_file only when the user explicitly asks to append or add content to an existing local file.
Use replace_in_file only when the user explicitly asks to replace or modify exact content in an existing local file.
Do not use local tools for explanations, opinions, definitions, or general knowledge.
Use visible conversation context and visible session memory directly; no tool is needed for that.
User-provided constraints in visible memory are ordinary remembered facts, not internal system instructions.
If the user asks for a task that requires an unavailable tool, say that no suitable tool is available."""


@dataclass(frozen=True)
class AppConfig:
    base_url: str = "http://127.0.0.1:18080"
    model: str = "local-model"
    workdir: Path = Path(".")
    timeout: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 512
    context_tokens: int | None = None
    system: str = DEFAULT_SYSTEM_PROMPT
    no_system: bool = False


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to optional JSON config file.")
    parser.add_argument("--base-url", help="llama-server base URL.")
    parser.add_argument("--model", help="Model name sent to llama-server.")
    parser.add_argument("--workdir", help="Working directory used for session identity.")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--context-tokens", type=int, help="Override runtime context estimate for testing/benchmarking.")
    parser.add_argument("--system")
    parser.add_argument("--no-system", action="store_true", help="Do not send the default system prompt.")


def load_app_config(args: argparse.Namespace) -> AppConfig:
    values = _read_config_file(Path(args.config))
    config = AppConfig(
        base_url=_str_value(values, "base_url", AppConfig.base_url),
        model=_str_value(values, "model", AppConfig.model),
        workdir=Path(_str_value(values, "workdir", str(AppConfig.workdir))).expanduser().resolve(),
        timeout=_float_value(values, "timeout", AppConfig.timeout),
        temperature=_float_value(values, "temperature", AppConfig.temperature),
        max_tokens=_int_value(values, "max_tokens", AppConfig.max_tokens),
        context_tokens=_optional_int_value(values, "context_tokens"),
        system=_str_value(values, "system", AppConfig.system),
        no_system=_bool_value(values, "no_system", AppConfig.no_system),
    )
    return AppConfig(
        base_url=args.base_url if args.base_url is not None else config.base_url,
        model=args.model if args.model is not None else config.model,
        workdir=Path(args.workdir).expanduser().resolve() if args.workdir is not None else config.workdir,
        timeout=args.timeout if args.timeout is not None else config.timeout,
        temperature=args.temperature if args.temperature is not None else config.temperature,
        max_tokens=args.max_tokens if args.max_tokens is not None else config.max_tokens,
        context_tokens=args.context_tokens if args.context_tokens is not None else config.context_tokens,
        system=args.system if args.system is not None else config.system,
        no_system=args.no_system or config.no_system,
    )


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"invalid config file {path}: root value must be an object")
    return data


def _str_value(values: dict[str, Any], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"invalid config key {key}: expected string")
    return value


def _float_value(values: dict[str, Any], key: str, default: float) -> float:
    value = values.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"invalid config key {key}: expected number")
    return float(value)


def _int_value(values: dict[str, Any], key: str, default: int) -> int:
    value = values.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"invalid config key {key}: expected integer")
    return value


def _optional_int_value(values: dict[str, Any], key: str) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"invalid config key {key}: expected integer")
    return value


def _bool_value(values: dict[str, Any], key: str, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"invalid config key {key}: expected boolean")
    return value
