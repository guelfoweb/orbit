from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


INFERENCE_AUDIT_SHADOW_ENV = "ORBIT_INFERENCE_AUDIT_SHADOW"


@dataclass(frozen=True)
class InferenceAuditShadowConfig:
    enabled: bool
    source: str
    validation_error: str | None = None


def resolve_inference_audit_shadow(
    environ: Mapping[str, str] | None = None,
) -> InferenceAuditShadowConfig:
    env = os.environ if environ is None else environ
    if INFERENCE_AUDIT_SHADOW_ENV not in env:
        return InferenceAuditShadowConfig(False, "default")
    value = env.get(INFERENCE_AUDIT_SHADOW_ENV, "")
    if value == "1":
        return InferenceAuditShadowConfig(True, "stable")
    if value == "0":
        return InferenceAuditShadowConfig(False, "stable")
    return InferenceAuditShadowConfig(False, "stable", "invalid_boolean")
