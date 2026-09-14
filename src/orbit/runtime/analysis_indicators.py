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

# An XML / XHTML namespace declaration: `xmlns="URI"` or `xmlns:prefix="URI"`.
# The URI there names a vocabulary, not a network endpoint -- it is document
# metadata, and publishing it as a verified indicator sends an analyst at a W3C
# specification. Recognised by SYNTAX (the xmlns attribute immediately before
# the value), never by a domain list: the SAME URI appearing OUTSIDE this syntax
# -- a real link in a script -- is a legitimate indicator and is kept.
#
# Two boundaries make this precise rather than a substring match:
#   - `(?<![\w.\-:])` before `xmlns`: the attribute name is the WHOLE token, so a
#     longer name that merely ENDS in `xmlns` (`data-xmlns`, `myxmlns`) is not a
#     namespace declaration and its value stays an indicator. Attacker-written
#     markup cannot evade extraction by embedding the URL in such an attribute.
#   - a required QUOTE immediately before the URI (`["']\Z`): a real declaration
#     is always quoted with the value flush against the quote, so `xmlns= http://x`
#     (a stray token then a URL) is not mistaken for one.
_XMLNS_DECL = re.compile(r"""(?<![\w.\-:])xmlns(?::[A-Za-z_][\w.\-]*)?\s*=\s*["']\Z""")
# How far back to look for the attribute. An attribute name plus separators and
# a quote is a handful of characters; bounding the look-behind keeps the check
# cheap and stops it spanning unrelated markup.
_XMLNS_LOOKBACK = 64


def _is_namespace_declaration(text: str, start: int) -> bool:
    """Whether the URI at `start` is the value of an xmlns declaration.

    Context only, by syntax: the bytes immediately before the URI must be a whole
    `xmlns`/`xmlns:prefix` attribute, its `=`, and its opening quote. A URI that
    merely looks like a namespace but sits anywhere else is not matched, and a
    longer attribute that ends in `xmlns` is not a declaration.
    """
    lo = max(0, start - _XMLNS_LOOKBACK)
    match = _XMLNS_DECL.search(text[lo:start])
    if match is None:
        return False
    # If `xmlns` sits at the very start of a TRUNCATED look-back window, the
    # character before it is unseen and the left boundary cannot be trusted --
    # fail closed toward KEEPING the indicator (dropping a real IOC is the worse
    # error). A window that reaches the document start (`lo == 0`) is exact.
    if match.start() == 0 and lo != 0:
        return False
    return True


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
    for match in _URI.finditer(text):
        # A URI that is the value of an xmlns declaration is document metadata,
        # not an indicator -- decided by the syntax immediately before it, so a
        # real endpoint written the same way anywhere else is untouched.
        if _is_namespace_declaration(text, match.start()):
            continue
        candidate = match.group(0)
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
        # Named for its subject: this is the sha256 of the INDICATOR STRING
        # above, not of any file the address might refer to. Left unqualified it
        # read as "the endpoint's hash", and a report once relabelled it as the
        # downloaded payload's digest -- a file whose bytes Orbit never fetched.
        lines.append(f"  sha256 of this indicator string: {indicator.sha256}")
    return "\n".join(lines)
