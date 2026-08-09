from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Callable, Mapping, Sequence


QWEN_ROUTE_PREFIX_ENV = "ORBIT_QWEN_ROUTE_PREFIX_REUSE"
QWEN_ROUTE_PREFIX_TOKEN_COUNT = 768
QWEN_ROUTE_PREFIX_FORMAT_VERSION = "qwen36-route-prefix-v1"
QWEN_ROUTE_TOKENIZER_IDENTITY = "gpt2:qwen35"


@dataclass(frozen=True)
class QwenRoutePrefixConfig:
    enabled: bool
    source: str
    validation_error: str | None = None


@dataclass(frozen=True)
class QwenRoutePrefixSpec:
    prefix_tokens: tuple[int, ...]
    prefix_token_hash: str
    invariant_text_hash: str
    system_prompt_hash: str
    invariant_token_count: int
    next_boundary_token: int


@dataclass
class QwenRoutePrefixStatus:
    initialized: bool = False
    prefix_tokens: int = 0
    capture_count: int = 0
    restore_count: int = 0
    fallback_count: int = 0
    invalidation_count: int = 0
    failure_reason: str | None = None
    last_used: str | None = None


def resolve_qwen_route_prefix_reuse(
    environ: Mapping[str, str] | None = None,
    *,
    default_enabled: bool = True,
) -> QwenRoutePrefixConfig:
    env = os.environ if environ is None else environ
    configured = env.get(QWEN_ROUTE_PREFIX_ENV)
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
        validation_error="invalid_qwen_route_prefix_reuse_value",
    )


def derive_qwen_route_prefix_spec(
    *,
    system_prompt: str,
    full_prompt: str,
    full_tokens: Sequence[int],
    render_reference: Callable[[str], str],
    tokenize: Callable[[str], list[int]],
    prefix_token_count: int = QWEN_ROUTE_PREFIX_TOKEN_COUNT,
) -> tuple[QwenRoutePrefixSpec | None, str | None]:
    if not system_prompt:
        return None, "missing_route_system_prompt"
    if len(full_tokens) <= prefix_token_count:
        return None, "route_prompt_too_short"

    first = render_reference("A-orbit-qwen-route-boundary")
    second = render_reference("Z-orbit-qwen-route-boundary")
    first_tokens = tokenize(first)
    second_tokens = tokenize(second)
    token_lcp = _longest_common_prefix(first_tokens, second_tokens)
    if token_lcp <= prefix_token_count:
        return None, "stable_token_boundary_unavailable"

    prefix = tuple(first_tokens[:prefix_token_count])
    if tuple(second_tokens[:prefix_token_count]) != prefix:
        return None, "reference_prefix_mismatch"
    if tuple(full_tokens[:prefix_token_count]) != prefix:
        return None, "production_prefix_mismatch"

    char_lcp = _longest_common_prefix_text(first, second)
    if char_lcp <= 0 or not full_prompt.startswith(first[:char_lcp]):
        return None, "invariant_text_boundary_mismatch"

    return (
        QwenRoutePrefixSpec(
            prefix_tokens=prefix,
            prefix_token_hash=hash_tokens(prefix),
            invariant_text_hash=hash_text(first[:char_lcp]),
            system_prompt_hash=hash_text(system_prompt),
            invariant_token_count=token_lcp,
            next_boundary_token=int(first_tokens[prefix_token_count]),
        ),
        None,
    )


def hash_tokens(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, byteorder="little", signed=True))
    return digest.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _longest_common_prefix(first: Sequence[int], second: Sequence[int]) -> int:
    common = 0
    limit = min(len(first), len(second))
    while common < limit and first[common] == second[common]:
        common += 1
    return common


def _longest_common_prefix_text(first: str, second: str) -> int:
    common = 0
    limit = min(len(first), len(second))
    while common < limit and first[common] == second[common]:
        common += 1
    return common
