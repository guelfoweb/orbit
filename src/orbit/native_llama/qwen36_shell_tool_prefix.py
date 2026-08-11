from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Callable, Mapping, Sequence

from .qwen_route_prefix import hash_text, hash_tokens


QWEN36_SHELL_TOOL_PREFIX_ENV = "ORBIT_QWEN36_SHELL_TOOL_PREFIX_REUSE"
QWEN36_SHELL_TOOL_PREFIX_FORMAT_VERSION = "qwen36-shell-tool-prefix-v1"
QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT = 384
QWEN36_SHELL_TOOL_INVARIANT_TOKEN_COUNT = 439
QWEN36_SHELL_TOOL_TOKENIZER_IDENTITY = "gpt2:qwen35"
QWEN36_SHELL_TOOL_PREFIX_TOKEN_HASH = "4428fc45a4eded6214ba9c7dbab6e36ca90686c381494776e776f286bea9b228"
QWEN36_SHELL_TOOL_RENDERED_PREFIX_HASH = "84cd1ab78799b6511159f660e958388f87ebc5e2dfccb102a8e1ed616c546bfd"
QWEN36_SHELL_TOOL_SCHEMA_HASH = "a2669f863cd2f569bc6e5b009ef72dd8fb6f31a66d83c29d4861e1f510071a68"


@dataclass(frozen=True)
class Qwen36ShellToolPrefixConfig:
    enabled: bool
    source: str
    validation_error: str | None = None


@dataclass(frozen=True)
class Qwen36ShellToolPrefixSpec:
    prefix_tokens: tuple[int, ...]
    prefix_token_hash: str
    invariant_text_hash: str
    tool_schema_hash: str
    invariant_token_count: int
    next_boundary_token: int


def resolve_qwen36_shell_tool_prefix_reuse(
    environ: Mapping[str, str] | None = None,
    *,
    default_enabled: bool = True,
) -> Qwen36ShellToolPrefixConfig:
    env = os.environ if environ is None else environ
    configured = env.get(QWEN36_SHELL_TOOL_PREFIX_ENV)
    if configured is None:
        return Qwen36ShellToolPrefixConfig(enabled=default_enabled, source="default")
    value = configured.strip()
    if value == "1":
        return Qwen36ShellToolPrefixConfig(enabled=True, source="stable")
    if value == "0":
        return Qwen36ShellToolPrefixConfig(enabled=False, source="stable")
    return Qwen36ShellToolPrefixConfig(
        enabled=False,
        source="stable",
        validation_error="invalid_qwen36_shell_tool_prefix_reuse_value",
    )


def exact_qwen36_shell_tool_schema(tools: Sequence[object] | None) -> bool:
    if not isinstance(tools, list) or len(tools) != 1:
        return False
    try:
        payload = json.dumps(tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return False
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() == QWEN36_SHELL_TOOL_SCHEMA_HASH


def derive_qwen36_shell_tool_prefix_spec(
    *,
    tools: Sequence[object] | None,
    full_prompt: str,
    full_tokens: Sequence[int],
    render_reference: Callable[[str], str],
    tokenize: Callable[[str], list[int]],
) -> tuple[Qwen36ShellToolPrefixSpec | None, str | None]:
    if not exact_qwen36_shell_tool_schema(tools):
        return None, "shell_tool_schema_mismatch"
    if len(full_tokens) <= QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT:
        return None, "shell_tool_prompt_too_short"

    first = render_reference("A-orbit-qwen36-shell-boundary")
    second = render_reference("Z-orbit-qwen36-shell-boundary")
    first_tokens = tokenize(first)
    second_tokens = tokenize(second)
    token_lcp = _longest_common_prefix(first_tokens, second_tokens)
    if token_lcp != QWEN36_SHELL_TOOL_INVARIANT_TOKEN_COUNT:
        return None, "qualified_invariant_token_count_changed"

    prefix = tuple(first_tokens[:QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT])
    if tuple(second_tokens[:QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT]) != prefix:
        return None, "reference_prefix_mismatch"
    if tuple(full_tokens[:QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT]) != prefix:
        return None, "production_prefix_mismatch"
    prefix_hash = hash_tokens(prefix)
    if prefix_hash != QWEN36_SHELL_TOOL_PREFIX_TOKEN_HASH:
        return None, "qualified_prefix_token_hash_changed"

    char_lcp = _longest_common_prefix_text(first, second)
    if char_lcp <= 0 or not full_prompt.startswith(first[:char_lcp]):
        return None, "invariant_text_boundary_mismatch"
    invariant_text_hash = hash_text(first[:char_lcp])
    if invariant_text_hash != QWEN36_SHELL_TOOL_RENDERED_PREFIX_HASH:
        return None, "qualified_rendered_prefix_hash_changed"

    return (
        Qwen36ShellToolPrefixSpec(
            prefix_tokens=prefix,
            prefix_token_hash=prefix_hash,
            invariant_text_hash=invariant_text_hash,
            tool_schema_hash=QWEN36_SHELL_TOOL_SCHEMA_HASH,
            invariant_token_count=token_lcp,
            next_boundary_token=int(first_tokens[QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT]),
        ),
        None,
    )


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
