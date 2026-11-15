"""Hash-bound observations. Support, execution and semantic review are distinct.

These records are reports, never runtime admission or model-promotion input.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re


MODEL_QUALIFICATION_CAPABILITIES = (
    "load", "chat", "tool_intent", "analysis_semantics",
)
RESULT_STATUSES = frozenset({"PASS", "FAIL", "MANUAL_CHECK", "NOT_RUN"})


def require_sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ModelQualificationResult:
    artifact_sha256: str
    backend_sha256: str
    configuration_sha256: str
    capability: str
    status: str
    scope: str
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        for value in (self.artifact_sha256, self.backend_sha256, self.configuration_sha256):
            require_sha256(value)
        if self.capability not in MODEL_QUALIFICATION_CAPABILITIES:
            raise ValueError("unknown qualification capability")
        if self.status not in RESULT_STATUSES:
            raise ValueError("unknown qualification status")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("qualification requires an explicit scope")
        if self.status == "NOT_RUN":
            if self.evidence_sha256 is not None:
                raise ValueError("NOT_RUN cannot claim measured evidence")
        else:
            require_sha256(self.evidence_sha256)

    def as_dict(self) -> dict:
        return asdict(self)
