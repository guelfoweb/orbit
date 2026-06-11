from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit.runtime.messages import DEFAULT_SYSTEM_PROMPT
from orbit.terminal.tool_mode import ToolSpec, normalize_tool_spec


DEFAULT_CONFIG_PATH = Path.home() / ".orbit" / "config.json"
MIN_TIMEOUT_SECONDS = 1.0
MAX_TIMEOUT_SECONDS = 3600.0
MIN_MAX_TOKENS = 32
MAX_MAX_TOKENS = 4096
MIN_CONTEXT_TOKENS = 512
MAX_CONTEXT_TOKENS = 262_144
DEFAULT_TOOLS = "off"


@dataclass(frozen=True)
class AppConfig:
    base_url: str = "http://127.0.0.1:18080"
    workdir: Path = Path(".")
    timeout: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 512
    context_tokens: int | None = None
    system: str = DEFAULT_SYSTEM_PROMPT
    no_system: bool = False
    tools: ToolSpec = DEFAULT_TOOLS


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to optional JSON config file.")
    parser.add_argument("--base-url", help="llama-server base URL.")
    parser.add_argument("--workdir", help="Working directory used for session identity.")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--context-tokens", type=int, help="Override runtime context estimate for testing/benchmarking.")
    parser.add_argument("--system")
    parser.add_argument("--no-system", action="store_true", help="Do not send the default system prompt.")
    parser.add_argument("--tools", help="Initial tool mode: off, on, files, edit, web, shell, or comma-separated groups.")


def load_app_config(args: argparse.Namespace) -> AppConfig:
    values = _read_config_file(Path(args.config))
    config = AppConfig(
        base_url=_str_value(values, "base_url", AppConfig.base_url),
        workdir=Path(_str_value(values, "workdir", str(AppConfig.workdir))).expanduser().resolve(),
        timeout=_ranged_float_value(
            values,
            "timeout",
            AppConfig.timeout,
            minimum=MIN_TIMEOUT_SECONDS,
            maximum=MAX_TIMEOUT_SECONDS,
        ),
        temperature=_float_value(values, "temperature", AppConfig.temperature),
        max_tokens=_ranged_int_value(
            values,
            "max_tokens",
            AppConfig.max_tokens,
            minimum=MIN_MAX_TOKENS,
            maximum=MAX_MAX_TOKENS,
        ),
        context_tokens=_optional_ranged_int_value(
            values,
            "context_tokens",
            minimum=MIN_CONTEXT_TOKENS,
            maximum=MAX_CONTEXT_TOKENS,
        ),
        system=_str_value(values, "system", AppConfig.system),
        no_system=_bool_value(values, "no_system", AppConfig.no_system),
        tools=_tool_spec_value(values),
    )
    return AppConfig(
        base_url=args.base_url if args.base_url is not None else config.base_url,
        workdir=Path(args.workdir).expanduser().resolve() if args.workdir is not None else config.workdir,
        timeout=_validate_optional_float_range(
            args.timeout,
            "timeout",
            minimum=MIN_TIMEOUT_SECONDS,
            maximum=MAX_TIMEOUT_SECONDS,
        )
        if args.timeout is not None
        else config.timeout,
        temperature=args.temperature if args.temperature is not None else config.temperature,
        max_tokens=_validate_optional_int_range(
            args.max_tokens,
            "max_tokens",
            minimum=MIN_MAX_TOKENS,
            maximum=MAX_MAX_TOKENS,
        )
        if args.max_tokens is not None
        else config.max_tokens,
        context_tokens=_validate_optional_int_range(
            args.context_tokens,
            "context_tokens",
            minimum=MIN_CONTEXT_TOKENS,
            maximum=MAX_CONTEXT_TOKENS,
        )
        if args.context_tokens is not None
        else config.context_tokens,
        system=args.system if args.system is not None else config.system,
        no_system=args.no_system or config.no_system,
        tools=normalize_tool_spec(args.tools) if args.tools is not None else config.tools,
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


def _ranged_float_value(
    values: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    return _validate_optional_float_range(_float_value(values, key, default), key, minimum=minimum, maximum=maximum)


def _ranged_int_value(
    values: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = values.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"invalid config key {key}: expected integer")
    return _validate_optional_int_range(value, key, minimum=minimum, maximum=maximum)


def _optional_ranged_int_value(
    values: dict[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"invalid config key {key}: expected integer")
    return _validate_optional_int_range(value, key, minimum=minimum, maximum=maximum)


def _validate_optional_float_range(value: float, key: str, *, minimum: float, maximum: float) -> float:
    if value < minimum or value > maximum:
        raise ValueError(f"invalid config key {key}: expected value between {minimum:g} and {maximum:g}")
    return value


def _validate_optional_int_range(value: int, key: str, *, minimum: int, maximum: int) -> int:
    if value < minimum or value > maximum:
        raise ValueError(f"invalid config key {key}: expected value between {minimum} and {maximum}")
    return value


def _bool_value(values: dict[str, Any], key: str, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"invalid config key {key}: expected boolean")
    return value


def _tool_spec_value(values: dict[str, Any]) -> ToolSpec:
    value = values.get("tool_mode", values.get("tools", DEFAULT_TOOLS))
    if isinstance(value, dict):
        value = DEFAULT_TOOLS
    return normalize_tool_spec(value, key="tool_mode")
