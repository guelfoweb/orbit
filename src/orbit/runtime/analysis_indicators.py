"""Exact indicators, read from bytes Orbit already holds.

An indicator is worth reporting only if it is right to the character. A model
asked to repeat a 38-character URL from an evidence card it read several steps
ago will occasionally get one character wrong -- and a hostname differing by
one letter is not a weaker finding, it is a different finding that sends an
analyst to the wrong place.

So the runtime reads them instead. Everything here is a literal match over
bytes that are already on disk: the acquired artifact, and the output of a
deterministic transformation. Nothing is decoded here, nothing is fetched, and
nothing is inferred -- what an address is *for* is the analysis's conclusion,
and this file has no opinion about it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Absolute URIs, syntactically. Deliberately conservative about the closing
# boundary: an artifact is attacker-written, and a URI in a script is normally
# terminated by a quote, a pipe, whitespace or a closing bracket.
_URI = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9+.-]*://"
    # A bracketed IPv6 authority may appear first; its brackets are part of
    # the address rather than the punctuation that ends one.
    r"(?:\[[0-9A-Fa-f:.]*\])?"
    # Ends only at characters a URI cannot contain. `,` and `;` are legal --
    # a comma separates query values and a semicolon introduces a path
    # parameter -- so excluding them would silently shorten a real address
    # into a plausible wrong one, and the digest beside it would then attest
    # the truncation rather than the artifact.
    r"[^\s\"'<>{}\\|^`]*"
)

# Enough to name a host, path and query without becoming a URL parser: the
# authority runs to the first `/`, `?` or `#`.
# The authority runs to the first `/`, `?` or `#`. A bracketed IPv6 literal is
# matched as a unit first, because it legitimately contains the `:` that the
# unbracketed form uses to introduce a port.
_PARTS = re.compile(
    r"^(?P<scheme>[^:]+)://(?P<authority>\[[^\]]*\](?::\d+)?|[^/?#]+)"
    r"(?P<path>/[^?#]*)?(?P<query>\?[^#]*)?"
)

# A URI long enough to be an address rather than a fragment, short enough that
# a pathological artifact cannot turn the report into a transcript.
MAX_URI_CHARS = 2048
MAX_INDICATORS = 32


@dataclass(frozen=True)
class Indicator:
    """One exact string, and where in the evidence it was read from."""

    kind: str
    value: str
    evidence_id: str
    source: str
    line: int
    sha256: str

    @property
    def authority(self) -> str | None:
        """Everything between `://` and the path.

        Named `authority` rather than `host` because that is what it is: any
        userinfo and any port are part of it, and silently dropping them would
        report a different address from the one in the artifact. Splitting
        them out correctly means parsing URIs properly, which is more than an
        exact-string reader should take on -- so it reports the span verbatim.
        """
        match = _PARTS.match(self.value)
        return match.group("authority") if match else None

    @property
    def path(self) -> str | None:
        match = _PARTS.match(self.value)
        return (match.group("path") or "") if match else None

    @property
    def query(self) -> str | None:
        match = _PARTS.match(self.value)
        return (match.group("query") or "") if match else None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def uris_in(text: str) -> list[str]:
    """Absolute URIs in `text`, first-seen order, deduplicated.

    A trailing `.` or `)` is stripped, because prose and scripts alike end a
    sentence or close a call right after an address, and those characters are
    far more often punctuation than part of the URI.
    """
    found: list[str] = []
    for candidate in _URI.findall(text):
        # Only characters that end a sentence rather than an address, and only
        # at the very end. A comma or semicolon *inside* a URI is part of the
        # value -- a query separator, a path parameter -- so the character
        # class above keeps them; one in final position is prose far more
        # often than address, and is dropped here. The distinction is the
        # position, which is why this is a strip and not an exclusion.
        trimmed = candidate.rstrip(".,;:)")
        if not trimmed or len(trimmed) > MAX_URI_CHARS:
            continue
        if not _PARTS.match(trimmed):
            continue
        if trimmed not in found:
            found.append(trimmed)
    return found


def extract_indicators(
    sources: "list[tuple[str, str, str]]",
) -> list[Indicator]:
    """Exact indicators across `(label, evidence_id, text)` sources.

    Deduplicated by value, keeping the first source that carried it: the same
    address written twice is one indicator, and naming every place it appears
    would pad the report without adding a fact.
    """
    indicators: list[Indicator] = []
    seen: set[str] = set()
    for label, evidence_id, text in sources:
        if not isinstance(text, str):
            continue
        for value in uris_in(text):
            if value in seen:
                continue
            seen.add(value)
            indicators.append(
                Indicator(
                    kind="uri",
                    value=value,
                    evidence_id=evidence_id,
                    source=label,
                    line=text[: text.index(value)].count("\n") + 1,
                    sha256=_sha(value),
                )
            )
            if len(indicators) >= MAX_INDICATORS:
                return indicators
    return indicators


def render_indicators(indicators: "list[Indicator]") -> str:
    """The report section. Empty when nothing exact was found.

    Renders what the bytes say and nothing else. There is no `C2`, no
    `malicious`, no severity: an address recovered from an artifact is an
    address recovered from an artifact, and deciding what it means is the
    analysis's job, on the evidence, in its own words.
    """
    if not indicators:
        return ""
    lines = ["## Verified indicators", ""]
    for indicator in indicators:
        lines.append(f"- {indicator.kind}: {indicator.value}")
        authority = indicator.authority
        if authority:
            lines.append(f"  authority: {authority}")
        if indicator.path:
            lines.append(f"  path: {indicator.path}")
        if indicator.query:
            lines.append(f"  query: {indicator.query}")
        # An artifact reading is named by digest, not by an evidence id, and
        # the two are kept in different slots: the report instruction tells
        # the model to cite `evidence:<id>`, and a digest offered in that slot
        # invites a citation that would never resolve.
        if indicator.evidence_id.startswith("sha256:"):
            lines.append(
                f"  artifact: {indicator.evidence_id} "
                f"({indicator.source}, line {indicator.line})"
            )
        else:
            lines.append(
                f"  evidence: {indicator.evidence_id} "
                f"({indicator.source}, line {indicator.line})"
            )
        lines.append(f"  sha256: {indicator.sha256}")
    return "\n".join(lines)
