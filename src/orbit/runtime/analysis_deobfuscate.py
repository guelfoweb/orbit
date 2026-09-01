"""Exact literal deobfuscation. Parsing and arithmetic, never execution.

Some obfuscation is not a question. When a script hands a decoder a string
literal, an integer literal and a delimiter literal, the result is already
determined by the bytes on disk: no analysis decides it, and asking a model to
write the loop that computes it only introduces a way to get it wrong. Three
real runs did get it wrong -- once by omitting numeric coercion, once by
passing an evidence id to `open()`, once by not attempting it at all -- while
a full traceback and a dedicated repair instruction sat in front of the model.

So the runtime computes those, and only those. Everything here is a literal
match followed by pure arithmetic. Nothing is executed, imported, evaluated or
fetched; a decoded stage is inert text that may be scanned for another literal
transformation and is never run.

The line this draws is *ambiguity*, not usefulness. A call whose key, input or
delimiter is any expression other than a literal is left alone, because
resolving it would mean interpreting the program rather than reading it -- and
that is the model's work, on evidence, with the sandbox it already has.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Bounds. Small, explicit, and about the artifact rather than the machine: a
# literal chain deep or large enough to exceed these is not the unambiguous
# case this handles, and stopping is the honest answer.
MAX_DEPTH = 4
MAX_INPUT_CHARS = 262_144
MAX_OUTPUT_CHARS = 262_144
# Enough for a real layered chain -- the deepest observed artifact yields five
# -- and small enough that a crafted file full of decoy call sites cannot turn
# the preamble into the prompt. The bound is on what a model is told about,
# not on what may be decoded: a file that legitimately exceeds it is unusual
# enough that stopping is the honest answer.
MAX_STAGES = 16

# JScript's String.fromCharCode takes a UTF-16 code unit; values are taken
# modulo 2**16 exactly as the spec's ToUint16 does.
_UINT16 = 0x1_0000

JSCRIPT_XOR = "jscript_numeric_xor"
POWERSHELL_XOR = "powershell_numeric_xor"


@dataclass(frozen=True)
class TransformStage:
    """One decoded stage, with everything needed to reproduce it."""

    kind: str
    key: int
    delimiter: str
    line: int
    offset: int
    depth: int
    encoded: str
    output: str
    input_sha256: str
    output_sha256: str

    @property
    def summary(self) -> str:
        return (
            f"{self.kind} key={self.key} delimiter={self.delimiter!r} "
            f"line={self.line} depth={self.depth} "
            f"({len(self.output)} chars, sha256 {self.output_sha256[:16]})"
        )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _to_number(token: str) -> int | None:
    """JScript's ToNumber, restricted to what a decoder token can be.

    `String.fromCharCode("94" ^ 46)` works because `^` coerces its operands
    through ToInt32, and a decimal string coerces to its value. The cases that
    matter and differ from Python's `int()`:

      - surrounding whitespace is ignored (`" 94 "` is 94);
      - the empty or all-whitespace string is 0, not an error;
      - anything else -- a letter, a float, a sign in the wrong place -- is
        NaN, and NaN through ToInt32 is 0.

    Returning None for the NaN case rather than 0 is deliberate: a token that
    is not a number means the literal is not the token list this claims to
    recognise, and the whole call site is rejected instead of silently
    decoding to a run of `chr(key)`.
    """
    stripped = token.strip()
    if not stripped:
        # An empty token is a real 0 in JScript, produced by a trailing
        # delimiter. It decodes to chr(key) and is kept.
        return 0
    if not stripped.isdigit():
        return None
    return int(stripped)


def _decode_tokens(encoded: str, key: int, delimiter: str) -> str | None:
    """split -> ToNumber -> XOR -> fromCharCode -> concat. Pure."""
    if not delimiter:
        return None
    parts = encoded.split(delimiter)
    if len(parts) < 2:
        return None
    out: list[str] = []
    for token in parts:
        value = _to_number(token)
        if value is None:
            return None
        out.append(chr((value ^ key) % _UINT16))
    return "".join(out)


# --- JScript ---------------------------------------------------------------
#
# Matched structurally, never by name. What identifies the decoder is what its
# body does: split the first parameter by the third, XOR each token against
# the second, and pass that to String.fromCharCode. Any names, any whitespace.

_FUNCTION = re.compile(
    r"function\s+([A-Za-z_$][\w$]*)\s*\(\s*"
    r"([A-Za-z_$][\w$]*)\s*,\s*([A-Za-z_$][\w$]*)\s*,\s*([A-Za-z_$][\w$]*)\s*\)",
)

# A string literal in either quote style, with escapes tolerated.
_STR = r"(?:\"((?:[^\"\\]|\\.)*)\"|'((?:[^'\\]|\\.)*)')"
# An argument that is a literal, optionally concatenated with identifiers that
# the file defines as the empty string. The concatenation is what real
# obfuscators emit; it changes nothing and is resolved before matching.
_ARG = rf"{_STR}((?:\s*\+\s*[A-Za-z_$][\w$]*)*)"


def _is_hex(text: str) -> bool:
    """A complete hex escape payload.

    Checked rather than attempted: a malformed `\\xZZ` anywhere in the file
    -- in a string this pass has no interest in -- would otherwise raise out
    of the whole scan, and the caller's guard would discard every stage the
    artifact really did determine. One bad byte must not disable the feature
    on exactly the hostile input it exists for. JScript treats an incomplete
    escape as the literal character, which is what falling through does.
    """
    return len(text) == len(text.strip()) and bool(text) and all(
        c in "0123456789abcdefABCDEF" for c in text
    ) and len(text) in (2, 4)


def _unescape(text: str) -> str:
    """Resolve the escapes a JScript string literal may carry."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\" or i + 1 >= len(text):
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1]
        simple = {"n": "\n", "t": "\t", "r": "\r", "0": "\0",
                  "\\": "\\", '"': '"', "'": "'", "/": "/", "b": "\b", "f": "\f"}
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt == "x" and _is_hex(text[i + 2:i + 4]):
            out.append(chr(int(text[i + 2:i + 4], 16)))
            i += 4
        elif nxt == "u" and _is_hex(text[i + 2:i + 6]):
            out.append(chr(int(text[i + 2:i + 6], 16)))
            i += 6
        else:
            out.append(nxt)
            i += 2
    return "".join(out)


def _assigned_once(name: str, source: str) -> bool:
    """Whether `source` writes `name` exactly once, however it is spelled.

    A second write -- another declaration, a bare `=`, a compound `+=`, an
    update through an index or property -- means the value at a given call
    site depends on control flow, and control flow is interpretation. Such a
    name is unusable here, whatever it was first assigned.
    """
    writes = re.findall(
        rf"(?<![\w$.]){re.escape(name)}\s*(?:\[[^\]]*\])?\s*(?:\+=|=(?!=))", source
    )
    return len(writes) == 1


def _empty_string_names(source: str) -> set[str]:
    """Identifiers that are provably the empty string at every point.

    Obfuscated calls concatenate one of these onto every argument. Folding it
    away is only sound if the name really is empty *at the call site*, so a
    name written more than once is refused exactly as `_string_variables`
    refuses one: reading the declaration of a variable that is later made
    non-empty would decode a truncated prefix, and a plausible-looking partial
    result is worse than no result at all.
    """
    names: set[str] = set()
    for match in re.finditer(
        r"\b(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:\"\"|'')\s*[;\n]", source
    ):
        name = match.group(1)
        if _assigned_once(name, source):
            names.add(name)
    return names


def _string_variables(source: str) -> dict[str, str]:
    """Identifiers assigned a string literal exactly once, and never rewritten.

    A name is usable only if the whole file agrees on its value. Any second
    assignment -- another declaration, a bare `x = ...`, a compound `+=`, or
    an update through an index or property -- means the value at a given call
    site depends on control flow, and control flow is interpretation. Such a
    name is dropped rather than guessed at.

    Getting this wrong would be worse than missing the stage: reading the
    first of two literals would decode confidently to a value the program
    never uses, and that is indistinguishable from a real result.
    """
    values: dict[str, str] = {}
    rejected: set[str] = set()
    for match in re.finditer(
        rf"\b(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*{_STR}\s*;", source
    ):
        name = match.group(1)
        if name in values:
            rejected.add(name)  # declared twice: ambiguous
            continue
        raw = match.group(2) if match.group(2) is not None else match.group(3)
        values[name] = _unescape(raw or "")

    for name in list(values):
        if not _assigned_once(name, source):
            rejected.add(name)
    for name in rejected:
        values.pop(name, None)
    return values


def _decoder_names(source: str) -> list[str]:
    """Every function whose body is a split/XOR/fromCharCode decoder."""
    found: list[str] = []
    for match in _FUNCTION.finditer(source):
        name, first, second, third = match.groups()
        body = source[match.end(): match.end() + 4000]
        # Structural, in the decoder's own parameter names: the first is split
        # by the third, and the second is the XOR operand under fromCharCode.
        splits = re.search(
            rf"\b{re.escape(first)}\s*\.\s*split\s*\(\s*{re.escape(third)}\s*\)", body
        )
        # The XOR must be *inside* the fromCharCode call, not merely somewhere
        # in the same function. A checksum helper that splits its input, folds
        # it with `^ seed`, and separately builds a character from the result
        # satisfies both conditions independently while being no decoder at
        # all -- and would be "decoded" into noise.
        xors = any(
            re.search(rf"\^\s*{re.escape(second)}\b", call.group(1))
            or re.search(rf"{re.escape(second)}\s*\^", call.group(1))
            for call in re.finditer(
                r"String\s*\.\s*fromCharCode\s*\(([^;]*?)\)", body
            )
        )
        if splits and xors:
            found.append(name)
    return found


def _resolve_argument(
    literal: str | None,
    alt: str | None,
    tail: str,
    empties: set[str],
) -> str | None:
    """A literal argument, with empty-string concatenations folded away."""
    if literal is None and alt is None:
        return None
    for name in re.findall(r"\+\s*([A-Za-z_$][\w$]*)", tail or ""):
        if name not in empties:
            # Concatenated with something that is not provably empty: the
            # value is not determined by the literal alone.
            return None
    return _unescape(literal if literal is not None else (alt or ""))


def find_jscript_stages(
    source: str, depth: int = 0, truncation: "_Truncation | None" = None
) -> list[TransformStage]:
    """Every call site of a structural decoder whose arguments are literal."""
    stages: list[TransformStage] = []
    names = _decoder_names(source)
    if not names:
        return stages
    empties = _empty_string_names(source)
    variables = _string_variables(source)
    for name in names:
        pattern = re.compile(
            re.escape(name) + r"\s*\(\s*"
            rf"(?:{_ARG}|([A-Za-z_$][\w$]*))\s*,\s*"      # encoded: literal or var
            r"(\d+)\s*,\s*"                                # key: integer literal only
            rf"{_ARG}\s*\)",                               # delimiter: literal
            re.S,
        )
        for match in pattern.finditer(source):
            enc_lit, enc_alt, enc_tail, enc_var, key_text, dl_lit, dl_alt, dl_tail = (
                match.groups()
            )
            if enc_var is not None:
                encoded = variables.get(enc_var)
                if encoded is None:
                    continue  # not a literal string variable: ambiguous
            else:
                encoded = _resolve_argument(enc_lit, enc_alt, enc_tail, empties)
            delimiter = _resolve_argument(dl_lit, dl_alt, dl_tail, empties)
            if encoded is None or delimiter is None:
                continue
            if len(encoded) > MAX_INPUT_CHARS:
                # A call site this pass could have decoded and did not.
                if truncation is not None:
                    truncation.dropped_input = True
                continue
            decoded = _decode_tokens(encoded, int(key_text), delimiter)
            if decoded is None:
                continue
            if len(decoded) > MAX_OUTPUT_CHARS:
                if truncation is not None:
                    truncation.dropped_output = True
                continue
            stages.append(
                TransformStage(
                    kind=JSCRIPT_XOR,
                    key=int(key_text),
                    delimiter=delimiter,
                    line=source[: match.start()].count("\n") + 1,
                    offset=match.start(),
                    depth=depth,
                    encoded=encoded,
                    output=decoded,
                    input_sha256=_sha(encoded),
                    output_sha256=_sha(decoded),
                )
            )
    return stages


# --- PowerShell ------------------------------------------------------------
#
# The same shape in another language, and matched the same way: a quoted list
# of decimal tokens, an integer assigned to a variable, and a `-bxor` against
# that variable. Nothing is executed; the tokens and the key are read.

_PS_LIST = re.compile(r"\$(\w+)\s*=\s*'((?:\s*\d+\s*,)+\s*\d+\s*)'")
_PS_INT = re.compile(r"\$(\w+)\s*=\s*(\d+)\s*[;\r\n]")
_PS_SPLIT = re.compile(r"\$(\w+)\s*-split\s*'([^']*)'")
_PS_BXOR_VAR = re.compile(r"-bxor\s*\$(\w+)")
_PS_BXOR_LIT = re.compile(r"-bxor\s*(\d+)")


def _reaches_bxor(name: str, source: str) -> bool:
    """Whether `$name` is connected by name to the `-bxor` in this script.

    Deliberately shallow: it follows assignments of the form `$b = $a ...`
    from the list variable, and asks whether any name reached that way is
    split or XORed. That is enough to separate a token list a loop consumes
    from an unrelated literal, and it stops well short of interpreting
    PowerShell -- an ambiguous case simply fails the check and is skipped.
    """
    reached = {name}
    for _ in range(4):  # a short chain; deeper aliasing is not the exact case
        grew = False
        for target, expression in re.findall(r"\$(\w+)\s*=([^\r\n;]*)", source):
            if target in reached:
                continue
            if any(re.search(rf"\${re.escape(n)}\b", expression) for n in reached):
                reached.add(target)
                grew = True
        if not grew:
            break
    for reference in reached:
        near = rf"\${re.escape(reference)}\b"
        if re.search(rf"{near}[^\r\n]*-bxor", source):
            return True
        if re.search(rf"{near}\s*-split", source):
            # Split then consumed elementwise by the operator: the loop
            # variable is not the list name, so the split is the link.
            return bool(re.search(r"-bxor", source))
    return False


def find_powershell_stages(
    source: str, depth: int = 0, truncation: "_Truncation | None" = None
) -> list[TransformStage]:
    """Literal `-bxor` reconstructions over a quoted numeric token list."""
    stages: list[TransformStage] = []
    # Matches rather than groups, so each list keeps the position it was
    # actually found at. `source.index(tokens)` would report the first place
    # the same digits appear anywhere -- a comment, another string -- which is
    # provenance pointing at bytes that did not produce the value.
    matches = list(_PS_LIST.finditer(source))
    if not matches:
        return stages
    integers = {name: int(value) for name, value in _PS_INT.findall(source)}
    # The delimiter is whatever `-split` names, defaulting to the comma the
    # token list is already written with.
    delimiters = {name: value for name, value in _PS_SPLIT.findall(source)}

    keys: list[int] = [
        integers[name] for name in _PS_BXOR_VAR.findall(source) if name in integers
    ]
    keys.extend(int(value) for value in _PS_BXOR_LIT.findall(source))
    if len(set(keys)) != 1:
        # No key, or more than one candidate: which applies to this list is not
        # determined by the text, so it is left alone.
        return stages
    key = keys[0]

    for match in matches:
        name, tokens = match.group(1), match.group(2)
        if len(tokens) > MAX_INPUT_CHARS:
            if truncation is not None:
                truncation.dropped_input = True
            continue
        if not _reaches_bxor(name, source):
            # This list is never fed to the operator. A version string or a
            # port list is not ciphertext, and XORing it produces noise that
            # would be recorded with the authority of arithmetic -- worse than
            # recording nothing, because it invites the analysis to interpret
            # it. Deciding which key applies is not the same as deciding that
            # a given list is an input at all.
            continue
        delimiter = delimiters.get(name, ",")
        decoded = _decode_tokens(tokens, key, delimiter)
        if decoded is None:
            continue
        if len(decoded) > MAX_OUTPUT_CHARS:
            if truncation is not None:
                truncation.dropped_output = True
            continue
        offset = match.start(2)
        stages.append(
            TransformStage(
                kind=POWERSHELL_XOR,
                key=key,
                delimiter=delimiter,
                line=source[:offset].count("\n") + 1,
                offset=offset,
                depth=depth,
                encoded=tokens,
                output=decoded,
                input_sha256=_sha(tokens),
                output_sha256=_sha(decoded),
            )
        )
    return stages


# Why a traversal stopped.
#
# The distinction that matters is between a walk that ended because there was
# nothing left, and one that ended because a ceiling stopped it -- the second
# leaves the artifact holding a transformation nobody ran, and a caller
# reasoning about completeness must be able to tell them apart.
#
# Refusing an ambiguous call site is neither. A dynamic key, a dynamic
# delimiter, a reassigned variable, a malformed token, an ambiguous operand:
# this pass handles literals and declines everything else, and declining is
# the contract rather than a limit. Those stay COMPLETE, because nothing ran
# out. So does a cycle: its output is already recovered, so nothing is
# missing. Only a call site this pass could have decoded, and did not because
# of a size ceiling, is a truncation.
COMPLETE = "complete"
STAGE_LIMIT = "stage_limit"
DEPTH_LIMIT = "depth_limit"
INPUT_LIMIT = "input_limit"
OUTPUT_LIMIT = "output_limit"


class _Truncation:
    """Whether a finder dropped a call site for exceeding a per-stage bound.

    The finders return a list, which has no room to say "and one more was
    skipped". A skipped call site is a real gap -- the artifact holds a
    transformation nobody ran -- and a traversal that reported a fixed point
    while one existed would be calling a truncation a completed result.

    Deliberately a small mutable passed down rather than a changed return
    type: both finders are called directly elsewhere, and widening their
    contract to carry a flag would complicate every caller for the benefit of
    the one that needs it.
    """

    def __init__(self) -> None:
        self.dropped_input = False
        self.dropped_output = False

    @property
    def dropped(self) -> bool:
        return self.dropped_input or self.dropped_output

    @property
    def status(self) -> str:
        """Which ceiling to name. Input first: it is the earlier refusal."""
        return INPUT_LIMIT if self.dropped_input else OUTPUT_LIMIT


@dataclass(frozen=True)
class TransformResult:
    """Stages, and whether the traversal that found them finished."""

    stages: "list[TransformStage]"
    status: str

    @property
    def complete(self) -> bool:
        return self.status == COMPLETE


def deobfuscate_with_status(source: str) -> TransformResult:
    """`deobfuscate`, plus why it stopped.

    Separate from `deobfuscate` rather than replacing it: every existing
    caller wants the stages and nothing else, and the status only matters to
    a decision about whether the recovered chain is the whole chain.

    A bound reached is not a failure -- the stages found are still exact --
    but it means something was not looked at, and a caller reasoning about
    completeness has to be able to tell the difference.
    """
    if len(source) > MAX_INPUT_CHARS:
        return TransformResult([], INPUT_LIMIT)
    stages: list[TransformStage] = []
    seen: set[str] = {_sha(source)}
    frontier = [(source, 0)]
    status = COMPLETE
    # Text a ceiling stopped the walk from scanning. Whether that is a
    # truncation or an irrelevance is decided after the loop, by asking what
    # it would have produced -- a bound reached over inert text cost nothing,
    # and reporting it as incomplete would be the mirror of the bug this
    # function exists to prevent.
    unexamined: list[str] = []
    truncation = _Truncation()
    while frontier and len(stages) < MAX_STAGES:
        text, depth = frontier.pop(0)
        if depth >= MAX_DEPTH:
            # Not followed, and not yet known to matter: whether the ceiling
            # cost anything depends on what this text would have yielded, and
            # that is settled once, below.
            unexamined.append(text)
            continue
        found = find_jscript_stages(
            text, depth, truncation
        ) + find_powershell_stages(text, depth, truncation)
        for stage in found:
            if len(stages) >= MAX_STAGES:
                # A stage this walk found and could not record: unlike an
                # unscanned output, this one is known to exist.
                status = STAGE_LIMIT
                break
            if stage.output_sha256 in seen:
                continue
            seen.add(stage.output_sha256)
            stages.append(stage)
            # Decoded text is inert data that may itself contain a literal
            # stage. It is scanned, never executed.
            frontier.append((stage.output, depth + 1))
    if status == COMPLETE:
        # Anything the ceilings left unscanned: outputs still on the frontier
        # when the stage ceiling stopped the walk, plus text refused for
        # depth. Scanning them here does not extend the result -- no stage is
        # added -- it only answers whether stopping cost anything. A ceiling
        # reached over text that yields nothing is not a truncation, and a
        # traversal that ended with nothing left to find is complete however
        # close to a bound it came.
        for text in [entry for entry, _depth in frontier] + unexamined:
            probe = _Truncation()
            if find_jscript_stages(text, 0, probe) or find_powershell_stages(
                text, 0, probe
            ):
                status = STAGE_LIMIT if frontier else DEPTH_LIMIT
                break
            if probe.dropped:
                status = probe.status
                break
    if truncation.dropped and status == COMPLETE:
        # A call site was recognised and not decoded because its input or its
        # output exceeded a per-stage bound. The frontier may have drained and
        # no traversal bound may have tripped, but the artifact still holds a
        # transformation nobody ran -- and a result with a hole in it is not a
        # fixed point, whatever the loop looked like from the outside.
        status = truncation.status
    return TransformResult(stages, status)


def deobfuscate(source: str) -> list[TransformStage]:
    """Every exact literal stage reachable from `source`, breadth-first.

    Bounded on all four axes that could otherwise run away: depth, stage
    count, per-stage size, and repetition. Cycles are detected by content
    digest rather than by shape, so a chain that decodes back to something
    already seen stops whatever route it took to get there.

    Kept as the plain form because every caller of it wants the stages and
    nothing else. It delegates rather than repeating the traversal: two copies
    of a bounded breadth-first walk would be two places for a bound to drift.
    """
    return deobfuscate_with_status(source).stages
