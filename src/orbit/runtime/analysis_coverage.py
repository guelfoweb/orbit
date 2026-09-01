"""Deterministic source coverage: hand the model every artifact byte, once.

The failure this addresses is not that the model reasons badly. It is that it
spends actions *acquiring* text Orbit already holds -- reading the artifact
raw, then numbered, then as a repr, then in slices -- because nothing ever
told it the source was already available. Prompt guidance did not change that.
So the source stops being something to fetch and becomes something supplied.

What this module guarantees is narrow and stated precisely:

    SOURCE_COVERED -- every eligible source byte has been presented to the
                      model, exactly, in order, exactly once.

It is emphatically **not** ANALYSIS_COMPLETE. Coverage says the model has seen
the bytes; it says nothing about whether the model understood them, whether the
behaviour is resolved, or whether the analysis may stop. The pinned sample makes
the distinction concrete: `Win32_Process` appears nowhere in its source bytes --
it exists only inside a decoded stage -- so full source coverage of that
artifact still leaves a real behaviour unaccounted for by coverage alone. Any
code that reads coverage as proof that analysis is finished is a defect.

Ranges are byte ranges of the immutable snapshot, so provenance survives: a
chunk is identified by what it covers, not by the text it happened to render.
Splits fall on UTF-8 code-point boundaries, so every chunk decodes on its own,
and concatenating the covered ranges reproduces the artifact byte for byte.

Sizing is done by asking the real tokenizer, never by counting characters. A
character cap cannot bound tokens -- this repo measured 1.123 chars/token on
exactly the obfuscated content it would need to bound, and rare codepoints run
the other way -- so the admission oracle is the backend's own count, and a
chunk that will not fit is made smaller rather than sent hopefully.

What chunking does and does not buy, because it is easy to assume the wrong
thing. ANALYSIS history is append-only: every COVER turn stays resident, so the
FINAL call carries the whole artifact regardless of how many parts it arrived
in. Splitting therefore bounds what any single *message* adds, not what the
context must ultimately hold -- an artifact whose bytes do not fit the input
budget cannot be covered at all, in one part or in ten. The planner discovers
this honestly, by measuring each part against the prefix that will really
precede it, and refuses rather than sending bytes it cannot follow through.
That refusal is the fail-closed path: the ordinary workflow runs, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


# Coverage status. `COVERAGE_COMPLETE` is about bytes presented, never about
# analysis being finished -- see the module docstring.
COVERAGE_COMPLETE = "complete"
COVERAGE_NOT_ELIGIBLE = "not_eligible"
COVERAGE_BUDGET_EXCEEDED = "budget_exceeded"
COVERAGE_UNADMISSIBLE = "unadmissible"


@dataclass(frozen=True)
class SourceChunk:
    """One ordered, non-overlapping byte range of the artifact.

    `start`/`end` index the snapshot bytes; `text` is that exact slice decoded.
    Keeping both is what makes provenance checkable: the text can be verified
    against the range, and the ranges against the whole artifact.
    """

    index: int
    total: int
    start: int
    end: int
    text: str

    @property
    def is_final(self) -> bool:
        return self.index == self.total


@dataclass(frozen=True)
class CoveragePlan:
    """A complete plan, or a refusal with the reason it could not be made.

    There is no partial plan. Covering some of an artifact and calling it
    coverage would be worse than not covering it at all: it would tell the
    model it had seen everything when it had not. When a complete plan cannot
    be built the plan is empty and `status` says why, and the caller falls back
    to the ordinary autonomous workflow.
    """

    chunks: tuple[SourceChunk, ...]
    status: str
    sha256: str = ""
    size_bytes: int = 0

    @property
    def covered(self) -> bool:
        return self.status == COVERAGE_COMPLETE

    @property
    def covered_bytes(self) -> int:
        return sum(chunk.end - chunk.start for chunk in self.chunks)

    def attest(self) -> "CoverageAttestation":
        return attest_coverage(self)


@dataclass(frozen=True)
class CoverageAttestation:
    """What the runtime can prove about a plan, computed from the plan itself.

    Never inferred from model output: the model is not asked whether it read
    the source, because an answer to that question is not evidence.
    """

    sha256: str
    size_bytes: int
    ranges: tuple[tuple[int, int], ...]
    covered_bytes: int
    gaps: tuple[tuple[int, int], ...]
    overlaps: tuple[tuple[int, int], ...]
    status: str

    @property
    def complete(self) -> bool:
        """Every byte covered exactly once, and the plan says so.

        The gap and overlap checks are load-bearing as a pair rather than
        individually: on a fixed artifact size a gap forces a compensating
        overlap and vice versa, so the byte total plus either one already
        decides every case. All three conditions stay because the pair is what
        makes the byte total insufficient to fake -- a duplicated range and a
        missing one of the same size leave `covered_bytes == size_bytes`.
        `test_gap_and_overlap_checks_are_jointly_load_bearing` pins exactly
        that case.
        """
        return (
            self.status == COVERAGE_COMPLETE
            and not self.gaps
            and not self.overlaps
            and self.covered_bytes == self.size_bytes
        )


def decode_artifact(raw: bytes) -> str | None:
    """The artifact as text, or None when it is not text this can cover.

    Strict UTF-8 and only strict. Decoding with `errors="replace"` would put
    U+FFFD into text then presented as the artifact's own bytes, which is a
    character the artifact does not contain -- the same reasoning that governs
    indicator extraction. An embedded NUL means this is not the encoding it
    appears to be, and a binary artifact simply is not covered: existing
    behaviour stands for it.
    """
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    if text.encode("utf-8") != raw:
        # Defensive: strict UTF-8 round-trips by construction. If it ever does
        # not, the text is not the bytes and must not be presented as them.
        return None
    return text


def _boundary_at_or_before(raw: bytes, end: int, floor: int) -> int:
    """Largest index in (floor, end] that does not split a UTF-8 code point.

    `len(raw)` is always a boundary -- there is no byte there to be a
    continuation of anything -- so it is returned unchanged rather than
    indexed. Returns `floor` when no boundary exists above it, which the
    caller reads as "this candidate length is unusable".
    """
    end = min(end, len(raw))
    while end > floor and end < len(raw) and (raw[end] & 0xC0) == 0x80:
        end -= 1
    return end


def plan_coverage(
    raw: bytes,
    *,
    fits: "callable",
    sha256: str,
    max_chunks: int,
) -> CoveragePlan:
    """Build a complete ordered byte cover, or refuse.

    `fits(text, index, preceding)` is the admission oracle and must be the real
    tokenizer's answer for the message that would actually be sent. `preceding`
    is the list of chunk texts already planned, because each COVER turn joins
    the history the next one is admitted against: sizing every chunk as if it
    were the first is exactly how a plan that fits becomes a call that is
    refused halfway through, with bytes already sent that cannot be followed.

    Chunks are therefore planned in order, each against the real prefix before
    it, and the largest admissible slice is taken each time by binary search --
    so the plan spends as few model calls as the budget allows.

    `max_chunks` bounds how much of the model-call ceiling coverage may spend.
    Exceeding it is a refusal, never a truncation: an artifact too large to
    cover completely within the permitted calls falls back to the ordinary
    workflow with its source unread rather than half-read.
    """
    text = decode_artifact(raw)
    if text is None:
        return CoveragePlan((), COVERAGE_NOT_ELIGIBLE, sha256, len(raw))

    total = len(raw)
    spans: list[tuple[int, int]] = []
    planned: list[str] = []
    start = 0
    while start < total:
        if len(spans) >= max_chunks:
            return CoveragePlan((), COVERAGE_BUDGET_EXCEEDED, sha256, total)
        # Binary search over candidate END POSITIONS, not lengths: every probe
        # is snapped to a code-point boundary first, so the value compared and
        # the value kept are the same one and the interval always shrinks.
        lo, hi = start + 1, total
        best: int | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            end = _boundary_at_or_before(raw, mid, start)
            if end <= start:
                # Everything at or below `mid` lands mid-codepoint; the only
                # way forward is a longer slice.
                lo = mid + 1
                continue
            if fits(raw[start:end].decode("utf-8"), len(spans) + 1, tuple(planned)):
                best = max(best or 0, end)
                # `end` may be below `mid` after snapping, so advance past
                # `mid` -- not past `end` -- or the interval can fail to shrink
                # and the search will not terminate.
                lo = max(end, mid) + 1
            else:
                hi = min(end, mid) - 1
        if best is None:
            # Nothing further is admissible. Which refusal this is depends on
            # whether anything had been planned: with parts already placed, the
            # artifact simply outgrew the budget partway through; with none, not
            # even one code point ever fit. Both refuse, and the distinction is
            # what an operator reads to tell "too big" from "cannot start".
            reason = (
                COVERAGE_BUDGET_EXCEEDED if spans else COVERAGE_UNADMISSIBLE
            )
            return CoveragePlan((), reason, sha256, total)
        spans.append((start, best))
        planned.append(raw[start:best].decode("utf-8"))
        start = best

    if not spans:
        return CoveragePlan((), COVERAGE_NOT_ELIGIBLE, sha256, total)

    count = len(spans)
    chunks = tuple(
        SourceChunk(
            index=position + 1,
            total=count,
            start=span_start,
            end=span_end,
            text=raw[span_start:span_end].decode("utf-8"),
        )
        for position, (span_start, span_end) in enumerate(spans)
    )
    plan = CoveragePlan(chunks, COVERAGE_COMPLETE, sha256, total)
    # The invariant is checked here rather than trusted: a plan that does not
    # actually cover the artifact must never be reported as coverage.
    if not plan.attest().complete:
        return CoveragePlan((), COVERAGE_UNADMISSIBLE, sha256, total)
    return plan


def attest_coverage(plan: CoveragePlan) -> CoverageAttestation:
    """Prove what a plan covers, by scanning the plan's own ranges."""
    ranges = tuple((chunk.start, chunk.end) for chunk in plan.chunks)
    gaps: list[tuple[int, int]] = []
    overlaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            gaps.append((cursor, start))
        elif start < cursor:
            overlaps.append((start, cursor))
        cursor = max(cursor, end)
    if cursor < plan.size_bytes:
        gaps.append((cursor, plan.size_bytes))
    return CoverageAttestation(
        sha256=plan.sha256,
        size_bytes=plan.size_bytes,
        ranges=ranges,
        covered_bytes=plan.covered_bytes,
        gaps=tuple(gaps),
        overlaps=tuple(overlaps),
        status=plan.status,
    )


def reconstruct(plan: CoveragePlan, raw: bytes) -> bytes:
    """The artifact rebuilt from the covered ranges, for verification."""
    return b"".join(raw[chunk.start : chunk.end] for chunk in plan.chunks)
