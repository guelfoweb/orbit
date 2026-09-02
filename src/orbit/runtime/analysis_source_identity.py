"""Prove that an action's output is only the source Orbit already supplied.

After COVER hands the model the complete artifact, the observed behaviour is
that it reads the same bytes back anyway -- as a repr, as plain text, as a
numbered listing. Those observations carry nothing the session does not already
hold, and treating them as new evidence is what lets a run spend its budget
re-reading what it was given.

This module decides one question and refuses to guess at it:

    Is this output, exactly and reversibly, the covered source and nothing else?

"Exactly" is the whole design. Every recognizer here reconstructs candidate
bytes and compares them to the artifact's own bytes; a single differing byte,
one extra line, one reordered line, one normalised newline, and the answer is
no. There is no similarity, no distance, no normalisation that could make two
different artifacts compare equal, and no model judgement. The failure mode
this must never have is suppressing something new, so every ambiguity resolves
to "not proven" and the observation stays ordinary evidence.

What is deliberately NOT recognised, because each carries something the source
alone does not: a partial range, the source plus any computed line, a grep or
search result, a decoded value, a hash, a summary, or output that merely looks
like the source. The live run that motivated this is the case in point -- of
its four source reads, the fourth also printed a line count, a `def` count and
an import list, and is therefore a real observation that must stay.

Nothing here executes, imports or evaluates the text it inspects. The repr
recognizer is a narrow literal decoder over a documented grammar, not `eval`:
artifact bytes are attacker-controlled, and a parser that could be steered by
them would be a worse problem than the one this solves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Bound on what is worth attempting to prove. A candidate cannot be the source
# if it is wildly larger than the source: numbering adds a prefix per line and
# a repr adds escapes, so a generous multiple still admits every real form
# while keeping a hostile megabyte of text from being scanned line by line.
MAX_CANDIDATE_RATIO = 8
MAX_CANDIDATE_CHARS = 4_000_000

# How a proven reacquisition is named in provenance and in the model-facing
# note. One string, so the record and the message cannot drift apart.
SOURCE_REACQUISITION = "source_reacquisition"

RAW = "raw"
REPR = "repr"
NUMBERED = "numbered"
ARTIFACT = "artifact"


@dataclass(frozen=True)
class SourceEquivalence:
    """Why an output was judged to be only the already-covered source."""

    recognizer: str
    detail: str = ""


def _too_large(candidate: str, source: str) -> bool:
    return (
        len(candidate) > MAX_CANDIDATE_CHARS
        or len(candidate) > max(1024, len(source) * MAX_CANDIDATE_RATIO)
    )


def _strip_trailing_newline(text: str) -> str:
    """Drop exactly one trailing newline, the one `print` itself adds.

    Only one, and only at the end: `print(x)` emits `x` followed by `\\n`, so
    without this every plain read would fail by a single byte. Removing more
    would start normalising, which is the thing this module must not do.
    """
    return text[:-1] if text.endswith("\n") else text


def _match_raw(candidate: str, source: str) -> bool:
    """The output IS the source, byte for byte.

    Two spellings are accepted and no others: the source exactly, and the
    source plus the single newline `print` appends. A file that already ends
    in a newline therefore matches both `print(data)` and a bare write of the
    same bytes -- which is why the candidate is compared as-is first rather
    than always having a newline stripped from it, since stripping would turn
    an exact match into a one-byte-short mismatch.
    """
    return candidate == source or candidate == source + "\n"


# A Python string literal, single- or double-quoted, with no adjacent
# concatenation and no prefix (no f, r, b, u): exactly what `repr()` of a
# `str` produces. Anything else is not a repr this will attempt.
# `\A` and `\Z`, never `^`/`$`: with `re.S` a `$` also matches just BEFORE a
# final newline, so `repr(x) + "\n"` would match with the newline silently
# tolerated -- accepting a candidate that is the repr plus something else.
_REPR_LITERAL = re.compile(r"\A(?P<quote>'|\")(?P<body>.*)(?P=quote)\Z", re.S)

# The escapes `repr` actually emits for a `str`, and nothing else. A form this
# does not list is not decoded -- it is a reason to refuse.
_SIMPLE_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "0": "\0",
}


def decode_repr_literal(text: str) -> str | None:
    """Decode a `repr()`-shaped string literal, or None.

    A deliberate, bounded re-implementation rather than `eval` or
    `ast.literal_eval`. `eval` would execute artifact-controlled text, and even
    `literal_eval` accepts a far larger grammar than `repr(str)` produces --
    tuples, dicts, concatenation, numeric towers -- which is surface this does
    not need and cannot audit. Only the escapes CPython's `repr` emits are
    understood; anything else returns None and the caller treats the output as
    ordinary evidence.

    The result is verified against the source by the caller, so a decoding
    error can only ever cause a missed suppression, never a false one.
    """
    match = _REPR_LITERAL.match(text)
    if match is None:
        return None
    body = match.group("body")
    quote = match.group("quote")
    out: list[str] = []
    index = 0
    length = len(body)
    while index < length:
        char = body[index]
        if char == quote:
            # An unescaped quote of the enclosing kind cannot appear inside a
            # real repr: it would have ended the literal.
            return None
        if char != "\\":
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= length:
            return None  # trailing backslash: not a complete literal
        marker = body[index]
        if marker in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[marker])
            index += 1
            continue
        if marker == "x":
            digits = body[index + 1 : index + 3]
            if len(digits) != 2 or any(
                d not in "0123456789abcdefABCDEF" for d in digits
            ):
                return None
            out.append(chr(int(digits, 16)))
            index += 3
            continue
        if marker in ("u", "U"):
            width = 4 if marker == "u" else 8
            digits = body[index + 1 : index + 1 + width]
            if len(digits) != width or any(
                d not in "0123456789abcdefABCDEF" for d in digits
            ):
                return None
            value = int(digits, 16)
            if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
                return None  # not a scalar value; repr never emits these
            out.append(chr(value))
            index += 1 + width
            continue
        # `\N{...}` and octal escapes are valid Python but are not emitted by
        # `repr`, so they are refused rather than decoded.
        return None
    return "".join(out)


def _match_repr(candidate: str, source: str) -> bool:
    """The output is `repr(source)` and nothing else."""
    decoded = decode_repr_literal(_strip_trailing_newline(candidate))
    return decoded is not None and decoded == source


# One numbered line: an optional run of spaces, digits, a separator, then the
# line. The separator forms are fixed here rather than inferred, so a file
# whose own lines begin with digits cannot make an arbitrary prefix look like
# numbering -- the reconstruction below is what actually decides.
#
# The leading `[ \t]*` absorbs the right-alignment padding a formatter emits
# (`f"{i:3}"`), and only that: it is whitespace that belongs to the number, not
# to the line. Any non-whitespace character before the digits stops the line
# matching at all, so nothing with real content can hide in front of a
# listing -- verified for prefixes like `x`, `0`, `#` and ` 0: `, each of which
# leaves the observation as ordinary evidence.
# The digit run is bounded. A line number is at most as large as the file has
# lines, so a long one cannot be a real listing -- and `int()` on a very long
# digit string raises rather than returning, which is artifact-controlled input
# crashing the runtime. Nine digits admits any file this could plausibly cover
# and keeps the conversion total.
_MAX_LINE_NUMBER_DIGITS = 9
_NUMBERED_LINE = re.compile(
    r"\A[ \t]*(?P<number>\d{1,%d})(?P<sep>: | \| |\t|: |\. |  | )"
    % _MAX_LINE_NUMBER_DIGITS
)


def strip_line_numbers(candidate: str) -> str | None:
    """Remove a uniform line-number prefix, or None if it is not uniform.

    Reversibility is the requirement: the numbers must run consecutively from
    a single start (0 or 1), every line must carry one, and every line must use
    the same separator. Anything else -- a gap, a repeat, a mixed separator, a
    line without a number -- means this is not a full numbered listing of a
    file, and could be a filtered or annotated view that carries real
    information. Those return None and are never suppressed.
    """
    lines = candidate.split("\n")
    if len(lines) < 2:
        return None
    numbers: list[int] = []
    bodies: list[str] = []
    separator: str | None = None
    for line in lines:
        match = _NUMBERED_LINE.match(line)
        if match is None:
            return None
        if separator is None:
            separator = match.group("sep")
        elif match.group("sep") != separator:
            return None  # mixed separators: not one uniform listing
        numbers.append(int(match.group("number")))
        bodies.append(line[match.end() :])
    if numbers[0] not in (0, 1):
        return None
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return None  # gaps, repeats or reordering: not a complete listing
    return "\n".join(bodies)


def _match_numbered(candidate: str, source: str) -> bool:
    """The output is a complete consecutive numbered listing of the source.

    `splitlines()` is what produces such a listing, and it drops the final
    newline, so the source is compared both as-is and without its trailing
    newline. Nothing else about the bytes is adjusted.
    """
    stripped = strip_line_numbers(_strip_trailing_newline(candidate))
    if stripped is None:
        return False
    return stripped == source or stripped == _strip_trailing_newline(source)


def classify_output(candidate: str, source: str) -> SourceEquivalence | None:
    """Prove `candidate` is only `source`, or return None.

    Tried in the order the forms actually occur, cheapest first. Every path
    ends in an exact comparison against the source's own bytes.
    """
    if not candidate or not source:
        return None
    if _too_large(candidate, source):
        return None
    if _match_raw(candidate, source):
        return SourceEquivalence(RAW, "output is the covered source verbatim")
    if _match_repr(candidate, source):
        return SourceEquivalence(REPR, "output is repr() of the covered source")
    if _match_numbered(candidate, source):
        return SourceEquivalence(
            NUMBERED, "output is a numbered listing of the covered source"
        )
    return None


def classify_artifacts(
    artifacts: "list[object]", source_sha256: str, source_bytes: int
) -> SourceEquivalence | None:
    """Prove a produced artifact is a copy of the covered source.

    Identity is the digest, not the name: a file called anything at all whose
    bytes hash to the artifact's own digest is a copy of it. More than one
    artifact means the action produced something besides a copy, so it is not
    suppressible.
    """
    if len(artifacts) != 1:
        return None
    only = artifacts[0]
    digest = getattr(only, "sha256", None)
    size = getattr(only, "size_bytes", None)
    if digest != source_sha256 or size != source_bytes:
        return None
    return SourceEquivalence(
        ARTIFACT, f"artifact {getattr(only, 'name', '?')!r} is a copy of the source"
    )
