from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquiredEvidence:
    tool_name: str
    arguments: dict[str, object]
    content: str
    source: str
