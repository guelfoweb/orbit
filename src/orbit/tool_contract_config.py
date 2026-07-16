from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


TOOL_CALL_CANONICAL_GATE_ENV = "ORBIT_TOOL_CALL_CANONICAL_GATE"


@dataclass(frozen=True)
class ToolCallCanonicalGateConfig:
    enabled: bool
    source: str
    validation_error: str | None


def resolve_tool_call_canonical_gate(
    environ: Mapping[str, str] | None = None,
) -> ToolCallCanonicalGateConfig:
    env = os.environ if environ is None else environ
    if TOOL_CALL_CANONICAL_GATE_ENV not in env:
        return ToolCallCanonicalGateConfig(True, "default", None)
    value = env.get(TOOL_CALL_CANONICAL_GATE_ENV, "")
    if value not in {"0", "1"}:
        return ToolCallCanonicalGateConfig(False, "stable", "invalid_canonical_gate_value")
    return ToolCallCanonicalGateConfig(value == "1", "stable", None)
