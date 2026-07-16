from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Mapping


FINAL_PREFIX_REUSE_ENV = "ORBIT_FINAL_PREFIX_REUSE"
FINAL_PREFIX_EXPERIMENT_ENV = "ORBIT_FINAL_PREFIX_EXPERIMENT"
FINAL_PREFIX_TOKEN_COUNT = 64


@dataclass(frozen=True)
class FinalPrefixReuseConfig:
    enabled: bool
    source: Literal["stable", "legacy", "default"]
    raw_value: str | None
    validation_error: str | None
    legacy_detected: bool


def resolve_final_prefix_reuse(environ: Mapping[str, str] | None = None) -> FinalPrefixReuseConfig:
    env = os.environ if environ is None else environ
    legacy_detected = FINAL_PREFIX_EXPERIMENT_ENV in env
    if FINAL_PREFIX_REUSE_ENV in env:
        return _resolve_value(
            env.get(FINAL_PREFIX_REUSE_ENV, ""),
            source="stable",
            error="invalid_stable_value",
            legacy_detected=legacy_detected,
        )
    if legacy_detected:
        return _resolve_value(
            env.get(FINAL_PREFIX_EXPERIMENT_ENV, ""),
            source="legacy",
            error="invalid_legacy_value",
            legacy_detected=True,
        )
    return FinalPrefixReuseConfig(
        enabled=True,
        source="default",
        raw_value=None,
        validation_error=None,
        legacy_detected=False,
    )


def _resolve_value(
    value: str,
    *,
    source: Literal["stable", "legacy"],
    error: str,
    legacy_detected: bool,
) -> FinalPrefixReuseConfig:
    valid = value in {"0", "1"}
    return FinalPrefixReuseConfig(
        enabled=value == "1" if valid else False,
        source=source,
        raw_value=_bounded_raw_value(value),
        validation_error=None if valid else error,
        legacy_detected=legacy_detected,
    )


def _bounded_raw_value(value: str) -> str | None:
    if len(value) > 16 or not value.isascii() or not value.isprintable():
        return None
    return value
