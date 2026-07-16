from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


TOOL_CALL_HEALING_SHADOW_ENV = "ORBIT_TOOL_CALL_HEALING_SHADOW"
TOOL_CALL_HEALING_ENV = "ORBIT_TOOL_CALL_HEALING"


@dataclass(frozen=True)
class ToolCallHealingShadowConfig:
    enabled: bool
    source: str
    validation_error: str | None


@dataclass(frozen=True)
class ToolCallHealingConfig:
    enabled: bool
    source: str
    validation_error: str | None


def resolve_tool_call_healing_shadow(
    environ: Mapping[str, str] | None = None,
) -> ToolCallHealingShadowConfig:
    env = os.environ if environ is None else environ
    if TOOL_CALL_HEALING_SHADOW_ENV not in env:
        return ToolCallHealingShadowConfig(enabled=False, source="default", validation_error=None)
    value = env.get(TOOL_CALL_HEALING_SHADOW_ENV, "")
    if value not in {"0", "1"}:
        return ToolCallHealingShadowConfig(
            enabled=False,
            source="stable",
            validation_error="invalid_shadow_value",
        )
    return ToolCallHealingShadowConfig(
        enabled=value == "1",
        source="stable",
        validation_error=None,
    )


def resolve_tool_call_healing(
    environ: Mapping[str, str] | None = None,
) -> ToolCallHealingConfig:
    env = os.environ if environ is None else environ
    if TOOL_CALL_HEALING_ENV not in env:
        return ToolCallHealingConfig(enabled=True, source="default", validation_error=None)
    value = env.get(TOOL_CALL_HEALING_ENV, "")
    if value not in {"0", "1"}:
        return ToolCallHealingConfig(
            enabled=False,
            source="stable",
            validation_error="invalid_healing_value",
        )
    return ToolCallHealingConfig(
        enabled=value == "1",
        source="stable",
        validation_error=None,
    )
