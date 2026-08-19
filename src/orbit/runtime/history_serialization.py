"""Shared profile message-serialization contracts.

A verified model profile may declare a `history_serialization` contract that
constrains the message shape a template accepts. The contract belongs to the
profile, not to any one backend, so both the native client and an external
llama-server request path apply it from here.

This module is deliberately model-agnostic: it keys on the declared contract
string only, never on a model or family name.
"""

from __future__ import annotations

from typing import Any


LEADING_SYSTEM_ONLY = "qwen-leading-system-only"

# Roles the contract confines to the leading position. Orbit's context manager
# and session memory treat `developer` as a peer of `system`, so a contract
# that demoted only `system` would let a trailing developer message reach a
# template that discards it.
PREFIX_ONLY_ROLES = frozenset({"system", "developer"})


def serialize_profile_messages(
    messages: Any,
    *,
    history_serialization: str | None,
) -> Any:
    """Apply a profile's history-serialization contract to a message list.

    Unknown or absent contracts return the messages unchanged, so a backend
    that cannot identify a verified profile never gets model-specific
    normalization applied speculatively.
    """

    if history_serialization != LEADING_SYSTEM_ONLY:
        # No contract to apply: hand back the caller's own list so identity is
        # preserved for paths that rely on it.
        return messages
    items = [dict(message) for message in messages]
    for index, item in enumerate(items):
        if item.get("role") in PREFIX_ONLY_ROLES and index != 0:
            # The reviewed templates accept these roles only at the beginning:
            # one raises on a later occurrence, another drops it silently.
            # Preserve later Orbit evidence cards in place as a normal input
            # turn instead of reordering or dropping content.
            item["role"] = "user"
    return items
