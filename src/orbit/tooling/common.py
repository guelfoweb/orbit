from __future__ import annotations

from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    pass


def coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ToolError("boolean is not a valid integer value")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid integer value") from exc


def resolve_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    full = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        full.relative_to(root)
    except ValueError as exc:
        raise ToolError(f"path escapes workdir: {raw_path}") from exc
    return full
