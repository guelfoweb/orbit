from __future__ import annotations

import os
import sys
from typing import TextIO


DIM = "\033[2m"
CYAN = "\033[36m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def supports_ansi(stream: TextIO | None = None) -> bool:
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    return is_tty(stream)


def is_tty(stream: TextIO | None = None) -> bool:
    target = stream or sys.stdout
    try:
        return bool(target.isatty())
    except (AttributeError, OSError):
        return False


def accent(text: str) -> str:
    return _styled(text, CYAN)


def dim(text: str) -> str:
    return _styled(text, DIM)


def yellow_dim(text: str) -> str:
    return _styled(text, DIM + YELLOW)


def danger(text: str) -> str:
    return _styled(text, RED)


def warning_text(value: object) -> str:
    return _prefixed_message("warning", value)


def runtime_error_text(value: object) -> str:
    detail = str(value).strip() or value.__class__.__name__
    lowered = detail.lower()
    prefix = "timeout" if "timeout" in lowered or "timed out" in lowered else "error"
    return _prefixed_message(prefix, detail)


def _prefixed_message(prefix: str, value: object) -> str:
    detail = str(value).strip()
    for existing in ("error:", "warning:", "timeout:"):
        if detail.lower().startswith(existing):
            detail = detail[len(existing) :].strip()
            break
    return f"{prefix}: {detail}" if detail else f"{prefix}: unknown"


def _styled(text: str, prefix: str) -> str:
    if not supports_ansi():
        return text
    return f"{prefix}{text}{RESET}"
