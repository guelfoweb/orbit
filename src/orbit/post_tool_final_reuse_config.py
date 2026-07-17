from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


POST_TOOL_FINAL_REUSE_ENV = "ORBIT_POST_TOOL_FINAL_REUSE"


@dataclass(frozen=True)
class PostToolFinalReuseConfig:
    enabled: bool
    source: str
    validation_error: str | None = None


def resolve_post_tool_final_reuse(
    environ: Mapping[str, str] | None = None,
) -> PostToolFinalReuseConfig:
    env = os.environ if environ is None else environ
    if POST_TOOL_FINAL_REUSE_ENV not in env:
        return PostToolFinalReuseConfig(True, "default")
    value = env.get(POST_TOOL_FINAL_REUSE_ENV, "")
    if value == "1":
        return PostToolFinalReuseConfig(True, "stable")
    if value == "0":
        return PostToolFinalReuseConfig(False, "stable")
    return PostToolFinalReuseConfig(False, "stable", "invalid_boolean")
