"""Finalization handoff: answering from verified evidence, not from history.

An investigation can consume its entire working context. When it does, a
same-session final turn has no room left to generate, and the user receives
nothing despite the evidence having been gathered successfully.

Finalization is therefore a separate phase with its own bounded context. It
consumes the durable evidence the investigation produced rather than the
conversation that produced it, so the size of the investigation no longer
determines whether an answer can be returned.

Construction is deterministic: evidence is deduplicated by exact content
identity, copied byte-for-byte, and carries its provenance. Nothing is
summarised, rewritten or inferred, and admission is decided by exact token
count before any generation is attempted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


# Reserve for template framing and tokenizer drift between counting and
# generation. Small: admission is exact, this only absorbs rendering overhead.
FINALIZATION_SAFETY_TOKENS = 256

# Upper bound on a final answer. The investigation's own per-call budget is
# unrelated to how much room a report needs, so it is never inherited: a
# 2,048-token cap truncated a report mid-section while 7,072 tokens of context
# sat unused.
FINALIZATION_FINAL_MAX_TOKENS = 4096

FINAL_ONLY_INSTRUCTION = (
    "Produce the final answer using only the verified evidence supplied below. "
    "Support each claim with the evidence identifier it comes from. If a "
    "requested claim is not supported by that evidence, mark it UNRESOLVED. Do "
    "not request tools and do not continue investigating."
)


@dataclass(frozen=True)
class BundleEntry:
    """One unique piece of verified evidence, with its provenance."""

    evidence_id: str
    kind: str
    sha256: str
    chars: int
    content: str


@dataclass(frozen=True)
class FinalizationAdmission:
    prompt_tokens: int
    output_budget: int
    safety_tokens: int
    context_tokens: int
    admitted: bool
    reason: str | None

    @property
    def headroom(self) -> int:
        return self.context_tokens - (
            self.prompt_tokens + self.output_budget + self.safety_tokens
        )


def content_digest(content: str) -> str:
    """Content identity: the SHA-256 of the exact bytes, computed here."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def deduplicate_evidence(entries) -> list[BundleEntry]:
    """Collapse byte-identical evidence, keeping first-occurrence identity.

    A long investigation re-reports the same artifacts as it revisits them, so
    the same bytes recur under later identifiers. The key is recomputed from
    the content rather than taken from the entry: a stale or wrong digest would
    otherwise merge findings that differ, and losing a distinct result silently
    is worse than carrying a duplicate. An entry whose declared digest does not
    match its bytes is dropped, because its identity cannot be trusted.
    """
    seen: set[str] = set()
    unique: list[BundleEntry] = []
    for entry in entries:
        digest = content_digest(entry.content)
        if entry.sha256 and entry.sha256 != digest:
            continue
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(entry)
    return unique


def entries_from_store(store, evidence_ids) -> list[BundleEntry]:
    """Build entries from the EvidenceStore, re-attesting every record.

    Composing with the store rather than accepting caller-supplied content is
    what makes the bundle trustworthy: `reattest_exact` re-reads the durable
    bytes and re-checks identity and provenance, so an entry can only exist
    here if its evidence still verifies. A record that fails attestation is
    omitted rather than reported, since a final answer must not cite evidence
    the runtime can no longer stand behind.
    """
    entries: list[BundleEntry] = []
    for evidence_id in evidence_ids:
        content = store.reattest_exact(evidence_id)
        if content is None:
            continue
        record = store.records.get(evidence_id)
        entries.append(
            BundleEntry(
                evidence_id=evidence_id,
                kind=getattr(record, "kind", "evidence"),
                sha256=content_digest(content),
                chars=len(content),
                content=content,
            )
        )
    return entries


def render_bundle(task: str, entries: list[BundleEntry]) -> str:
    """Render the finalization request. Exact bytes, no summarisation."""
    lines = [task.strip(), "", FINAL_ONLY_INSTRUCTION, "", "Verified evidence:"]
    for entry in entries:
        lines.append(
            f"[{entry.evidence_id}] {entry.kind}, {entry.chars} chars, "
            f"sha256={entry.sha256[:16]}"
        )
        lines.append(entry.content)
        lines.append("")
    return "\n".join(lines)


def resolve_output_budget(
    prompt_tokens: int,
    context_tokens: int,
    *,
    cap: int = FINALIZATION_FINAL_MAX_TOKENS,
    safety: int = FINALIZATION_SAFETY_TOKENS,
) -> int:
    """Room for the answer, from what the context actually has left."""
    available = context_tokens - prompt_tokens - safety
    if available <= 0:
        return 0
    return min(cap, available)


def admit_finalization(
    prompt_tokens: int,
    context_tokens: int,
    *,
    cap: int = FINALIZATION_FINAL_MAX_TOKENS,
    safety: int = FINALIZATION_SAFETY_TOKENS,
    minimum_output: int = 256,
) -> FinalizationAdmission:
    """Decide admission by exact token count, before any generation.

    Overflow must never be discovered by the backend mid-generation: that
    yields a decode failure and no answer at all. A bundle that cannot fit is
    refused here, with the durable evidence left untouched for a caller that
    wants to retry differently.
    """
    budget = resolve_output_budget(
        prompt_tokens, context_tokens, cap=cap, safety=safety
    )
    if prompt_tokens < 0 or prompt_tokens >= context_tokens:
        budget = 0
    if budget <= 0 or budget < minimum_output:
        return FinalizationAdmission(
            prompt_tokens=prompt_tokens,
            output_budget=budget,
            safety_tokens=safety,
            context_tokens=context_tokens,
            admitted=False,
            reason="finalization_bundle_exceeds_context",
        )
    return FinalizationAdmission(
        prompt_tokens=prompt_tokens,
        output_budget=budget,
        safety_tokens=safety,
        context_tokens=context_tokens,
        admitted=True,
        reason=None,
    )
