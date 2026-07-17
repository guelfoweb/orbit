from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


TOOL_PLAN_SHADOW_ENV = "ORBIT_TOOL_PLAN_SHADOW"


@dataclass(frozen=True)
class ToolPlanShadowConfig:
    enabled: bool
    source: str
    validation_error: str | None = None


def resolve_tool_plan_shadow(env: Mapping[str, str] | None = None) -> ToolPlanShadowConfig:
    values = os.environ if env is None else env
    if TOOL_PLAN_SHADOW_ENV not in values:
        return ToolPlanShadowConfig(False, "default")
    raw = values.get(TOOL_PLAN_SHADOW_ENV, "")
    if raw == "1":
        return ToolPlanShadowConfig(True, "stable")
    if raw == "0":
        return ToolPlanShadowConfig(False, "stable")
    return ToolPlanShadowConfig(False, "stable", "invalid_boolean")
