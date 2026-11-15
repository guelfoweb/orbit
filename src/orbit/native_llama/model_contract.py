"""Technical model contracts, independent of measured qualification results."""
from __future__ import annotations

from enum import Enum


class ReasoningBehavior(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    ALWAYS = "always"


TEXT_CAPABILITIES = frozenset({
    "chat", "tools", "tool_history_results", "write_artifact",
    "verify_artifact", "full_document_analysis",
})


def reasoning_enabled(behavior: ReasoningBehavior, requested: bool) -> bool:
    return behavior == ReasoningBehavior.ALWAYS or (
        behavior == ReasoningBehavior.OPTIONAL and requested
    )
