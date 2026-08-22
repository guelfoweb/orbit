"""The Ornith route prefix: same derivation, its own identity.

Ornith renders through the GGUF's own ChatML template rather than the Gemma
renderer, which is why the startup prewarm never applied to it -- that path
builds Gemma markup and mismatches an Ornith request at the very first token.
Nothing about the derivation needs to change to fix that, only which template
does the rendering, so this module is the thin per-profile wrapper the Qwen
profiles already use.

The token count is fixed rather than measured. `derive_qwen_route_prefix_spec`
renders two reference requests that differ only in their user text, requires
the fixed count to fall strictly inside what those two share, and then checks
the result against the real production prompt. A count that is not invariant
is rejected outright; it is never trimmed to whatever happened to match.
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


ORNITH_ROUTE_PREFIX_ENV = "ORBIT_ORNITH_ROUTE_PREFIX_REUSE"

# 768, as for the other profiles. Measured headroom on this template: the
# rendered route system turn ends at token 825, and the two reference renders
# share 828, so the captured prefix stops well inside the invariant region and
# never reaches the user turn.
ORNITH_ROUTE_PREFIX_TOKEN_COUNT = 768
ORNITH_ROUTE_PREFIX_FORMAT_VERSION = "ornith15-route-prefix-v1"
ORNITH_ROUTE_TOKENIZER_IDENTITY = "gpt2:qwen35"


def resolve_ornith_route_prefix_reuse(
    environ: Mapping[str, str] | None = None,
    *,
    default_enabled: bool = True,
) -> QwenRoutePrefixConfig:
    env = os.environ if environ is None else environ
    configured = env.get(ORNITH_ROUTE_PREFIX_ENV)
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
        validation_error="invalid_ornith_route_prefix_reuse_value",
    )


def derive_ornith_route_prefix_spec(
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
        prefix_token_count=ORNITH_ROUTE_PREFIX_TOKEN_COUNT,
    )


__all__ = [
    "ORNITH_ROUTE_PREFIX_ENV",
    "ORNITH_ROUTE_PREFIX_FORMAT_VERSION",
    "ORNITH_ROUTE_PREFIX_TOKEN_COUNT",
    "ORNITH_ROUTE_TOKENIZER_IDENTITY",
    "QwenRoutePrefixSpec",
    "QwenRoutePrefixStatus",
    "derive_ornith_route_prefix_spec",
    "resolve_ornith_route_prefix_reuse",
]
