"""Prove an observation is the covered source plus only recomputable facts.

Exact equivalence -- `analysis_source_identity` -- catches an output that is
*only* the source. The live run showed that is rarely what happens: the model
prints the source and then one deterministic fact about it, `LEN: 1495` or
`TOTAL_LINES: 46` or a SHA. Each such observation carries nothing the session
does not already hold, because every one of those values is computable from
bytes Orbit already supplied, but exact equivalence refuses them all -- the
output is not the source, it is the source plus a line.

This module decides the narrower question:

    Is this output the covered source, plus properties Orbit can independently
    recompute from that same source, and nothing else at all?

"Recompute" is the whole boundary. A property qualifies only if Orbit can
derive it from the artifact's own bytes without interpreting the program:
how long it is, how many lines, its digest, the hex of an explicitly named
range. A count of functions, imports or calls does not qualify however
trivially true it is -- deriving it means parsing the language, which is
analysis, and analysis is what the model is for.

The proof is total. The observation is parsed into components; every component
is either the source in a supported representation or a property whose value is
recomputed and compared exactly; and no byte may remain unexplained. One
unrecognised line, one value that is off by one, one marker the grammar does not
name, and the answer is USEFUL. There is no tolerant parsing, no ignored
suffix, and no normalisation beyond the separators the grammars themselves
require.

Nothing here executes, imports or evaluates the text it inspects.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from orbit.runtime.analysis_source_identity import (
    NUMBERED,
    RAW,
    REPR,
    strip_line_numbers,
)

# Bound on what is worth attempting to prove, in characters. A dominated
# observation is the source plus a handful of short lines; anything vastly
# larger cannot be one, and refusing early keeps a hostile megabyte from being
# scanned line by line.
MAX_CANDIDATE_RATIO = 8
MAX_CANDIDATE_CHARS = 4_000_000
# The most property lines a dominated observation may carry. The live outputs
# carried one or two; this admits a generous margin while bounding the work.
MAX_PROPERTIES = 12

SOURCE_DOMINATED = "source_dominated"

# Property kinds, named so provenance can say exactly what was verified.
BYTE_LENGTH = "byte_length"
TEXT_LENGTH = "text_length"
LINE_COUNT = "line_count"
SHA256 = "sha256"
HEX_PREFIX = "hex_prefix"


@dataclass(frozen=True)
class SourceDominance:
    """Why an observation was judged to add nothing to the covered source."""

    representation: str
    properties: "tuple[str, ...]"

    @property
    def detail(self) -> str:
        named = ", ".join(self.properties)
        return f"{self.representation} of the covered source plus {named}"


def _too_large(candidate: str, source: str) -> bool:
    return (
        len(candidate) > MAX_CANDIDATE_CHARS
        or len(candidate) > max(4096, len(source) * MAX_CANDIDATE_RATIO)
    )


# --- properties -----------------------------------------------------------
#
# Each entry is a label the model actually printed, mapped to the function that
# recomputes the value from the covered source. The labels are exhaustive by
# design: a label not listed here is an unexplained component, and an
# observation containing one is USEFUL. Adding a label is a deliberate act that
# has to be justified by an observed output form, not a convenience.
#
# `len(str)` and `len(bytes)` are both accepted because a program can print
# either and both are recomputable. They coincide for ASCII and differ for
# anything else, so the value is checked against the specific one its label
# names rather than against whichever happens to match.


def _byte_length(source: str) -> str:
    return str(len(source.encode("utf-8")))


def _text_length(source: str) -> str:
    return str(len(source))


def _line_count_split(source: str) -> str:
    """`len(data.split("\\n"))` -- what the live outputs actually computed.

    Deliberately not `splitlines()`: they differ by one on text ending in a
    newline, and by more on text containing form feeds or U+2028. The label
    says which count was taken, and only the matching computation is accepted.
    """
    return str(len(source.split("\n")))


def _line_count_splitlines(source: str) -> str:
    """`len(data.splitlines())` -- the other rule the same label can mean.

    Both spellings are accepted for one label, and that is not a loosening:
    both are values Orbit recomputes from the artifact's own bytes, so a model
    printing either has told the session nothing it could not derive. What is
    NOT accepted is anything else -- on text containing U+2028 the two rules
    differ by two, and the values between them are refused along with every
    other number.
    """
    return str(len(source.splitlines()))


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# Labels observed in the live run, plus the obvious spellings of the same
# quantities. Every value is compared case-sensitively against the exact string
# the recomputation produces.
_PROPERTY_LABELS: "dict[str, tuple[str, tuple]]" = {
    "LEN": (TEXT_LENGTH, (_text_length, _byte_length)),
    "LENGTH": (TEXT_LENGTH, (_text_length, _byte_length)),
    "len_str": (TEXT_LENGTH, (_text_length,)),
    "len_bytes": (BYTE_LENGTH, (_byte_length,)),
    "BYTES": (BYTE_LENGTH, (_byte_length,)),
    "SIZE": (BYTE_LENGTH, (_byte_length, _text_length)),
    "TOTAL_LINES": (LINE_COUNT, (_line_count_split, _line_count_splitlines)),
    "LINES": (LINE_COUNT, (_line_count_split, _line_count_splitlines)),
    "LINE_COUNT": (LINE_COUNT, (_line_count_split, _line_count_splitlines)),
    "sha256": (SHA256, (_sha256,)),
    "SHA256": (SHA256, (_sha256,)),
    "SHA-256": (SHA256, (_sha256,)),
    "SHA": (SHA256, (_sha256,)),
}

# `LABEL: value`, with the separator the observed outputs used. `print("X:", v)`
# emits exactly one space after the colon, and nothing else is accepted: a
# different separator is a different output that this grammar does not name.
_PROPERTY_LINE = re.compile(r"\A(?P<label>[A-Za-z][A-Za-z0-9_-]{0,31}): (?P<value>.*)\Z")

# `HEXnnn: <hex>` -- the live form. The number in the label is what makes the
# range explicit, and it is the ONLY thing that makes this suppressible: the
# range is read from the label, never guessed. `HEX200` means the first 200
# characters, which is what `data[:200].encode().hex()` computed.
_HEX_PREFIX_LINE = re.compile(r"\AHEX(?P<count>\d{1,9}): (?P<value>[0-9a-f]*)\Z")


def _verify_property(label: str, value: str, source: str) -> str | None:
    """The property kind this line proves, or None if it proves nothing.

    None covers three different failures deliberately treated the same: a label
    outside the allow-list, a value that does not match any recomputation for
    that label, and a malformed line. All three mean a component is unexplained,
    and an unexplained component makes the whole observation useful.
    """
    entry = _PROPERTY_LABELS.get(label)
    if entry is None:
        return None
    kind, computations = entry
    for compute in computations:
        if value == compute(source):
            return kind
    return None


def _verify_hex_prefix(line: str, source: str) -> str | None:
    """A `HEXnnn:` line whose hex is exactly that prefix of the source.

    The count in the label names the range, so nothing is guessed. The
    recomputation uses the same slice-then-encode order the observed code used
    -- `data[:n].encode("utf-8").hex()` -- because slicing characters and
    slicing bytes are different ranges on non-ASCII text, and only the one the
    label describes is accepted.

    A count longer than the source is refused rather than clamped: a program
    asking for more bytes than exist got a shorter string than its label claims,
    and quietly treating that as a match would accept a range nobody proved.
    """
    match = _HEX_PREFIX_LINE.match(line)
    if match is None:
        return None
    count = int(match.group("count"))
    if count == 0 or count > len(source):
        return None
    if match.group("value") != source[:count].encode("utf-8").hex():
        return None
    return HEX_PREFIX


def _split_representation(candidate: str, source: str) -> "tuple[str, str] | None":
    """Peel the leading source representation off, returning (kind, rest).

    Only the representations `analysis_source_identity` already proves are
    accepted, and each is matched at the START of the candidate so what follows
    can be parsed as properties. The raw form is tried longest-first: the source
    may itself end in a newline, and both spellings have to be considered
    before concluding the output is not the source.
    """
    if candidate.startswith(source):
        return RAW, candidate[len(source) :]
    representation = repr(source)
    if candidate.startswith(representation):
        rest = candidate[len(representation) :]
        # Guard against a longer literal that merely starts with this one.
        if not rest or rest.startswith("\n"):
            return REPR, rest
    return None


def _split_numbered(candidate: str, source: str) -> "tuple[str, str] | None":
    """Peel a complete numbered listing off the front, if there is one.

    The listing is exactly as many lines as the source has, so the boundary is
    computed rather than searched for: both spellings of the source's line count
    are tried, each in constant time, and the prefix is stripped once. An
    earlier version tried successively shorter prefixes and re-stripped each
    one, which is quadratic -- 4000 numbered lines of ordinary model output
    cost ten seconds of host CPU after the sandbox had already returned, with
    no timeout around it.

    Trailing newlines are dropped one at a time rather than with `rstrip`,
    which would also eat form feeds, carriage returns and spaces: those are
    ordinary source bytes, and treating them as absent would let a listing that
    omits real trailing lines pass as complete.
    """
    lines = candidate.split("\n")
    for expected in _candidate_line_counts(source):
        if expected < 2 or expected > len(lines):
            continue
        head = "\n".join(lines[:expected])
        stripped = strip_line_numbers(head)
        if stripped is None:
            continue
        if stripped == source or stripped == _drop_one_newline(source):
            # The remainder keeps its leading newline, exactly as the raw and
            # repr paths leave it, so the caller sees one shape rather than
            # two and cannot tolerate a blank line on one path only.
            return NUMBERED, candidate[len(head) :]
    return None


def _drop_one_newline(text: str) -> str:
    """Exactly one trailing newline, the one a line-wise print drops.

    Deliberately not `rstrip`: `\x0c`, `\r`, `\x0b`, spaces and tabs are
    ordinary bytes of a source file, and removing them would accept a listing
    that never showed the lines they belong to.
    """
    return text[:-1] if text.endswith("\n") else text


def _candidate_line_counts(source: str) -> "tuple[int, ...]":
    """How many lines a complete listing of this source would have.

    The two counting rules a listing can be built from, de-duplicated. Both are
    recomputable from the source, so trying both proves nothing extra -- it
    only avoids guessing which one the program used.
    """
    counts = (len(source.split("\n")), len(source.splitlines()))
    return counts if counts[0] != counts[1] else (counts[0],)


def classify_dominated(candidate: str, source: str) -> SourceDominance | None:
    """Prove `candidate` is `source` plus only recomputable properties.

    Returns None -- meaning USEFUL -- for anything not proven in full. Every
    component must be explained: the leading representation must be an exact
    match for the source, each remaining line must be a property whose value
    Orbit recomputes to exactly what was printed, and nothing may be left over.
    """
    if not candidate or not source:
        return None
    if _too_large(candidate, source):
        return None

    split = _split_representation(candidate, source) or _split_numbered(
        candidate, source
    )
    if split is None:
        return None
    representation, rest = split

    # Nothing after the source is exact equivalence, which is the other
    # module's answer -- this one exists for the case where properties follow.
    if not rest.strip():
        return None

    # A property must begin on its own line, and exactly one newline may
    # separate it from the source. Two spellings produce that: `print(data)`
    # appends a newline of its own, and `sys.stdout.write(data)` on a source
    # that already ends in one does not -- both leave the property at the start
    # of a line, and requiring the extra newline would silently miss the second,
    # which is a form the model actually uses. Anything else, a space or a tab
    # or no separator at all, leaves the boundary to whatever the text allows.
    if rest.startswith("\n"):
        body = rest[1:]
    elif source.endswith("\n"):
        body = rest
    else:
        return None
    if body.endswith("\n"):
        body = body[:-1]  # the newline the last `print` appended
    lines = body.split("\n")
    if not lines or any(not line for line in lines):
        return None
    if len(lines) > MAX_PROPERTIES:
        return None

    kinds: list[str] = []
    for line in lines:
        hex_kind = _verify_hex_prefix(line, source)
        if hex_kind is not None:
            kinds.append(hex_kind)
            continue
        match = _PROPERTY_LINE.match(line)
        if match is None:
            return None  # an unexplained component
        kind = _verify_property(match.group("label"), match.group("value"), source)
        if kind is None:
            return None
        kinds.append(kind)

    # Ordered, de-duplicated, for a stable provenance string.
    seen: list[str] = []
    for kind in kinds:
        if kind not in seen:
            seen.append(kind)
    return SourceDominance(representation, tuple(seen))
