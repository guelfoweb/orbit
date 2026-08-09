from __future__ import annotations

import os
from typing import Callable, Mapping, Sequence

from .qwen_route_prefix import (
    QwenRoutePrefixConfig,
    QwenRoutePrefixSpec,
    QwenRoutePrefixStatus,
    derive_qwen_route_prefix_spec,
)


QWEN3_CODER_ROUTE_PREFIX_ENV = "ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE"
QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT = 768
QWEN3_CODER_ROUTE_PREFIX_FORMAT_VERSION = "qwen3-coder-route-prefix-v1"
QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY = "gpt2:qwen2"


def resolve_qwen3_coder_route_prefix_reuse(
    environ: Mapping[str, str] | None = None,
    *,
    default_enabled: bool = True,
) -> QwenRoutePrefixConfig:
    env = os.environ if environ is None else environ
    configured = env.get(QWEN3_CODER_ROUTE_PREFIX_ENV)
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
        validation_error="invalid_qwen3_coder_route_prefix_reuse_value",
    )


def derive_qwen3_coder_route_prefix_spec(
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
        prefix_token_count=QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT,
    )


__all__ = [
    "QWEN3_CODER_ROUTE_PREFIX_ENV",
    "QWEN3_CODER_ROUTE_PREFIX_FORMAT_VERSION",
    "QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT",
    "QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY",
    "QwenRoutePrefixSpec",
    "QwenRoutePrefixStatus",
    "derive_qwen3_coder_route_prefix_spec",
    "resolve_qwen3_coder_route_prefix_reuse",
]
