"""The Ornith ANALYSIS prewarm prefix: same derivation, its own identity.

An analysis step opens with a fixed contract -- the ANALYSIS system prompt and
the `execute_analysis` schema -- and only then names the artifact and the
analyst's instruction. That opening is identical for every session on a given
profile, so it can be prefilled once and restored instead of re-evaluated.

This is the CHAT route prefix mechanism with different content, not a second
mechanism: the same `derive_qwen_route_prefix_spec`, the same
`PrefixAnchorState`, the same capture and restore. Two things differ, and both
are deliberate.

The count is smaller. A whole first ANALYSIS request is around 540 tokens
against roughly 840 for a route request, so the route's 768 cannot fit; the
derivation would refuse it outright rather than shorten it. Measured on this
template the invariant region is 485 tokens, and 384 sits inside it with room
to spare.

The identity is its own. Sharing the route lineage's format version and
tokenizer identity would leave nothing but `model_id` separating an ANALYSIS
checkpoint from a CHAT one on the same model, and those two prefixes are
entirely different token sequences.
"""

from __future__ import annotations

import os
from typing import Callable, Mapping, Sequence

from .qwen_route_prefix import (
    QwenRoutePrefixConfig,
    QwenRoutePrefixSpec,
    QwenRoutePrefixStatus,
    derive_qwen_route_prefix_spec,
)


ORNITH_ANALYSIS_PREFIX_ENV = "ORBIT_ORNITH_ANALYSIS_PREFIX_REUSE"

# The lineage key. Every per-profile registry, identity and invalidation path
# in the client is keyed by a profile id, so the analysis prefix takes a key of
# its own rather than sharing Ornith's route key. It names a lineage, not a
# model: it never reaches a model profile lookup, and the backend still learns
# nothing about CHAT or ANALYSIS from it.
ORNITH_ANALYSIS_LINEAGE_ID = "orbit-ornith15-native-v1#analysis"

# 384: measured invariant region on this template is 485 tokens, so the
# captured prefix stops 101 tokens short of anything that varies. The whole
# request is ~540 tokens, which is why the route lineage's 768 is unusable
# here rather than merely wasteful.
#
# Re-measured after the input contract was reworded to say /workspace/input is
# the artifact file rather than a place it sits: the region grew 451 -> 485, so
# 384 was re-validated against a different token sequence rather than left
# alone. The count is derived, never trimmed to fit -- a prompt that shrank the
# region below it would be refused outright.
ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT = 384
ORNITH_ANALYSIS_PREFIX_FORMAT_VERSION = "ornith15-analysis-prefix-v1"
ORNITH_ANALYSIS_TOKENIZER_IDENTITY = "gpt2:qwen35"


def resolve_ornith_analysis_prefix_reuse(
    environ: Mapping[str, str] | None = None,
    *,
    default_enabled: bool = True,
) -> QwenRoutePrefixConfig:
    env = os.environ if environ is None else environ
    configured = env.get(ORNITH_ANALYSIS_PREFIX_ENV)
    if configured is None:
        return QwenRoutePrefixConfig(enabled=default_enabled, source="default")
    value = configured.strip()
    if value == "1":
        return QwenRoutePrefixConfig(enabled=True, source="stable")
    if value == "0":
        return QwenRoutePrefixConfig(enabled=False, source="stable")
    return QwenRoutePrefixConfig(
        enabled=False,
        source="stable",
        validation_error="invalid_ornith_analysis_prefix_reuse_value",
    )


def derive_ornith_analysis_prefix_spec(
    *,
    system_prompt: str,
    full_prompt: str,
    full_tokens: Sequence[int],
    render_reference: Callable[[str], str],
    tokenize: Callable[[str], list[int]],
) -> tuple[QwenRoutePrefixSpec | None, str | None]:
    return derive_qwen_route_prefix_spec(
        system_prompt=system_prompt,
        full_prompt=full_prompt,
        full_tokens=full_tokens,
        render_reference=render_reference,
        tokenize=tokenize,
        prefix_token_count=ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT,
    )


__all__ = [
    "ORNITH_ANALYSIS_LINEAGE_ID",
    "ORNITH_ANALYSIS_PREFIX_ENV",
    "ORNITH_ANALYSIS_PREFIX_FORMAT_VERSION",
    "ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT",
    "ORNITH_ANALYSIS_TOKENIZER_IDENTITY",
    "QwenRoutePrefixSpec",
    "QwenRoutePrefixStatus",
    "derive_ornith_analysis_prefix_spec",
    "resolve_ornith_analysis_prefix_reuse",
]
