"""One declaration of the existing qualified cache-profile memberships.

This module owns no native state, cache key, token boundary or transition.
Callers retain every identity, configuration, thinking and MTP guard. A new
model gets no optimization merely because its architecture resembles one here.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .model_profiles import (
    ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID, QWEN38_FLASH_NEXT_PROFILE_ID, INKLING2_PROFILE_ID,
)


@dataclass(frozen=True)
class CacheProfilePolicy:
    aligned_prompt_checkpoint: bool = False
    jinja_route_prefill: bool = False
    rolling_route: bool = False
    analysis_prefix: bool = False
    rolling_analysis: bool = False
    post_final_shadow: bool = False
    preserve_fixed_on_mode_switch: bool = False
    preserve_fixed_on_session_reset: bool = False
    attempt_analysis_prewarm_after_route: bool = False

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in self.__dict__.values()):
            raise ValueError("cache policy fields must be booleans")
        if self.post_final_shadow and not self.rolling_route:
            raise ValueError("route shadow requires a qualified rolling route")
        if (self.preserve_fixed_on_mode_switch or self.preserve_fixed_on_session_reset
                or self.attempt_analysis_prewarm_after_route) and not self.jinja_route_prefill:
            raise ValueError("fixed-prefix behavior requires a qualified Jinja prefix")


_NONE = CacheProfilePolicy()
_POLICIES = MappingProxyType({
    INKLING2_PROFILE_ID: CacheProfilePolicy(aligned_prompt_checkpoint=True),
    QWEN3_CODER_PROFILE_ID: CacheProfilePolicy(
        jinja_route_prefill=True, preserve_fixed_on_mode_switch=True,
        preserve_fixed_on_session_reset=True,
        # Preserve the existing call to the analysis prewarm function: it
        # deliberately returns a model-ineligible diagnostic for this profile.
        attempt_analysis_prewarm_after_route=True,
    ),
    ORNITH15_PROFILE_ID: CacheProfilePolicy(
        jinja_route_prefill=True, rolling_route=True, analysis_prefix=True,
        rolling_analysis=True, attempt_analysis_prewarm_after_route=True,
    ),
    QWEN38_FLASH_NEXT_PROFILE_ID: CacheProfilePolicy(
        jinja_route_prefill=True, rolling_route=True, post_final_shadow=True,
    ),
})


def cache_policy_for_profile(profile_id: object) -> CacheProfilePolicy:
    return _POLICIES.get(profile_id, _NONE) if isinstance(profile_id, str) else _NONE


# Re-exported by client.py for its existing public test/diagnostic consumers.
ROLLING_ROUTE_PROFILE_IDS = frozenset(key for key, value in _POLICIES.items() if value.rolling_route)
POST_FINAL_ROUTE_SHADOW_PROFILE_IDS = frozenset(key for key, value in _POLICIES.items() if value.post_final_shadow)
