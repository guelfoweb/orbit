"""Which evidence a grounded report may treat as authoritative.

An analysis that corrects itself leaves both versions in the store: the record
that wrote an artifact and the record that rewrote it. That is the right thing
for an append-only, re-attestable store -- nothing is deleted, and the history
of a correction is itself evidence. It is the wrong thing to hand a finalizer,
which then sees a value and its correction side by side with equal standing and
can quote either.

This module decides standing, never content. It reads the provenance the store
already records -- the durable handle an action wrote and the digest it wrote
there -- and marks an earlier version of a handle SUPERSEDED once a later one
exists. It never compares observations, never prefers a later string, and never
knows what any value means: two records that share no artifact handle cannot
supersede one another however much they disagree.

Supersession is about versions of a thing, not about who is right. A record
that argues with an earlier finding in prose stays ACTIVE, and so does the one
it argues with, because resolving that is the report's job and the disagreement
is exactly what a reader needs to see.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTIVE = "ACTIVE"
SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True)
class EvidenceStanding:
    """What standing one record has, and why."""

    evidence_id: str
    status: str
    superseded_by: str | None = None
    handle: str | None = None

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE


_DEFAULT_ACTIVE = EvidenceStanding("", ACTIVE)


def _artifact_versions(record: object) -> list[tuple[str, str]]:
    """(handle, digest) pairs this record durably wrote. Provenance only."""
    metadata = getattr(record, "metadata", None) or {}
    entries = metadata.get("artifacts") if isinstance(metadata, dict) else None
    if not isinstance(entries, list):
        return []
    versions: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        handle = entry.get("handle") or entry.get("name")
        digest = entry.get("sha256")
        if handle and digest:
            versions.append((str(handle), str(digest)))
    return versions


def evaluate_standing(records: list) -> dict[str, EvidenceStanding]:
    """Standing for each record, in the order the store produced them.

    A record is SUPERSEDED when every artifact version it contributed has since
    been rewritten at the same handle with a different digest. "Every" matters:
    a record that wrote two artifacts and had only one replaced still holds
    something current, and demoting it would lose that.

    A record that wrote nothing durable is always ACTIVE. It cannot be a stale
    version of anything, because it never claimed to be a version of anything --
    which is what keeps a correction that merely quotes an old value from being
    mistaken for the old value itself.
    """
    # Which version of a handle is current is decided by the order the store
    # recorded them, never by the order this list happens to arrive in. The
    # caller passes `records.values()`, which is insertion-ordered today and so
    # would usually be right -- but a reversed or re-serialized index would
    # silently invert every verdict and mark the corrected version stale, which
    # is the exact failure this module exists to prevent. `evidence_sequence`
    # is the store's own answer and every record carries it.
    # A record the store never sequenced cannot be shown to be newer than one
    # it did, so it must not displace it. Sorting the unsequenced ones last
    # would let exactly that happen -- a stale version from a legacy or damaged
    # index would supersede the correction that replaced it, which is this
    # module's own failure mode inverted. They sort first, where they may be
    # superseded but can supersede nothing that carries a sequence.
    def _order(record: object) -> tuple[int, int, str]:
        # The id breaks ties. Two records can share a key -- both unsequenced,
        # or holding the same number after a discard freed it, since the store
        # numbers by max+1 over what it currently holds. `sorted` is stable, so
        # without a tiebreak those would resolve by arrival order, and the
        # verdict would flip when the same records were read back in a
        # different order. That is precisely the property this ordering exists
        # to remove, so it must not survive at the margin.
        rid = str(getattr(record, "evidence_id", "") or "")
        sequence = getattr(record, "evidence_sequence", None)
        try:
            # A sequence that is not a number is not a sequence. The store
            # normalises junk away on load, but this module takes any record
            # shaped like one, and an unreadable number should demote a record
            # to unsequenced rather than raise from a read-only audit.
            numbered = int(sequence)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Unsequenced first: it may be superseded, but it can supersede
            # nothing the store actually numbered.
            return (0, 0, rid)
        return (1, numbered, rid)

    ordered = sorted(
        (r for r in records if getattr(r, "evidence_id", None)), key=_order
    )

    latest: dict[str, tuple[str, str]] = {}
    for record in ordered:
        rid = record.evidence_id
        for handle, digest in _artifact_versions(record):
            latest[handle] = (digest, rid)

    standing: dict[str, EvidenceStanding] = {}
    for record in records:
        rid = getattr(record, "evidence_id", None)
        if not rid:
            continue
        versions = _artifact_versions(record)
        if not versions:
            standing[rid] = EvidenceStanding(rid, ACTIVE)
            continue
        stale: list[tuple[str, str]] = []
        for handle, digest in versions:
            current_digest, current_id = latest.get(handle, (digest, rid))
            if current_digest != digest and current_id != rid:
                stale.append((handle, current_id))
        if stale and len(stale) == len(versions):
            handle, current_id = stale[0]
            standing[rid] = EvidenceStanding(rid, SUPERSEDED, current_id, handle)
        else:
            standing[rid] = EvidenceStanding(rid, ACTIVE)
    return standing


def active_records(records: list) -> list:
    """The records a grounded report may cite as authoritative."""
    standing = evaluate_standing(records)
    # A record `evaluate_standing` skipped has no entry; treat it as active
    # rather than raising. The two functions must agree about how defensive to
    # be, or a record one of them tolerates becomes a KeyError in the other.
    return [
        record
        for record in records
        if standing.get(getattr(record, "evidence_id", ""), _DEFAULT_ACTIVE).is_active
    ]


__all__ = [
    "ACTIVE",
    "SUPERSEDED",
    "EvidenceStanding",
    "active_records",
    "evaluate_standing",
]
