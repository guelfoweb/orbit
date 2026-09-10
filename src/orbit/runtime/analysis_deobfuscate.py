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
        # The key/delimiter describe the XOR and Chr passes; a folded value has
        # neither, so it is summarised by what it is and where it came from.
        if self.kind == JS_STRINGARRAY_FOLD:
            return (
                f"{self.kind} line={self.line} depth={self.depth} "
                f"(ActiveXObject ProgID {self.output!r} folded from {self.encoded!r})"
            )
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


# --- VBScript / PowerShell Chr-offset ---------------------------------------
#
# A third shape, generic across the two languages that use it: a literal array
# of integers, each mapped through a character function after subtracting a
# literal constant, concatenated. In VBScript:
#
#     Function d(a)
#         c = 29450
#         For Each e In a
#             s = s & Chr(e - c)
#         Next
#         d = s
#     End Function
#     ... d(Array(29562, 29561, ...))
#
# and inside PowerShell the same primitive written `[char]($e - $h)` over
# `@(22670, ...)` with `$h = 22624`. Matched STRUCTURALLY, never by name: what
# identifies the decoder is a loop that builds a string from `Chr`/`[char]` of
# `(<loop var> - <literal int>)`, and the active input is a literal numeric
# array actually passed to it (VBScript) or iterated by it (PowerShell). Like
# the XOR passes, this reads literals and arithmetic only -- nothing is
# executed, no VBScript/PowerShell engine is invoked -- and it declines (fails
# closed) on any operand it cannot prove is a literal.
VBSCRIPT_CHR = "vbscript_chr_offset"
POWERSHELL_CHR = "powershell_chr_offset"

# The character-conversion upper bounds, matched to the real function's domain
# so a decode never produces a character the interpreter itself would refuse.
# VBScript `Chr(n)` is single-byte: n outside 0..255 raises at runtime, so a
# value past 255 means this Array is not a `Chr` input and the stage is refused
# rather than fabricated. `ChrW(n)` and PowerShell `[char]n` take a UTF-16 code
# unit, 0..65535. Each value is range-checked individually; out-of-range fails
# closed (never wrapped).
_CHR_MAX = 0xFF        # VBScript Chr
_CHRW_MAX = 0xFFFF     # VBScript ChrW / PowerShell [char]

# A VBScript decoder body: `For Each <v> In <arr>` ... `Chr(<v> - <const>)`
# accumulated with `&`. The constant is read from a literal assigned to the
# loop-arithmetic variable. Whitespace/newlines tolerated; names are free.
#
# The gap between the loop header and the Chr expression is BOUNDED, not `.*?`
# to end of file. A decoder's body is a handful of lines; an unbounded gap over
# a 260 KB artifact packed with bare `For Each` headers that never reach a
# matching `Chr(v - k)` scans to EOF per header -- measured at ~1 s for 2000
# headers, and an artifact is attacker-controlled, so the bound turns that into
# a fixed small window. A real decoder well inside it; anything past it is not
# this tight loop-to-Chr construct.
_DECODER_BODY_GAP = 400
_VBS_DECODER = re.compile(
    r"For\s+Each\s+([A-Za-z_]\w*)\s+In\s+([A-Za-z_]\w*)\b"  # loop var, source var
    rf".{{0,{_DECODER_BODY_GAP}}}?"
    r"\b(Chr|ChrW|ChrB)\(\s*\1\s*-\s*([A-Za-z_]\w*)\s*\)",  # Chr*(loopvar - constvar)
    re.S | re.I,
)
# A literal integer assigned to a name: `name = 29450` (decimal only; a hex or
# expression operand is not the proven-literal case this handles). A trailing
# operator (`k = 100 + x`) means the value is not a bare literal -- the negative
# lookahead rejects it so `_int_constants` never reads a computed value as a
# constant.
_VBS_INT_ASSIGN = re.compile(
    r"(?<![\w$.])([A-Za-z_]\w*)\s*=\s*(\d+)(?!\s*[\d.&+\-*/^])\s*(?:[\r\n']|$)"
)
# An `Array(<ints>)` literal. Captured whole so its position is exact.
_VBS_ARRAY = re.compile(r"\bArray\(\s*((?:\d+\s*,\s*)*\d+)\s*\)")
# The PowerShell form: `[char]($v - $h)` inside `foreach($v in $z)`, with
# `$h = 22624` and the active list an `@( ints )` literal. Same bounded gap.
_PS_CHAR = re.compile(
    r"foreach\s*\(\s*\$([A-Za-z_]\w*)\s+in\s+\$([A-Za-z_]\w*)\s*\)"
    rf".{{0,{_DECODER_BODY_GAP}}}?"
    r"\[char\]\(\s*\$\1\s*-\s*\$([A-Za-z_]\w*)\s*\)",
    re.S | re.I,
)
_PS_AT_ARRAY = re.compile(r"@\(\s*((?:\d+\s*,\s*)*\d+)\s*\)")


def _decode_chr_offset(
    numbers: "list[int]", constant: int, char_max: int = _CHRW_MAX
) -> str | None:
    """`Chr(n - constant)` for each n, concatenated. Pure arithmetic.

    `char_max` is the conversion function's own domain ceiling (255 for `Chr`,
    65535 for `ChrW`/`[char]`). Returns None (fail closed) if any value falls
    outside 0..char_max after subtraction -- that means this array is not an
    input to this decoder, and a silent wrap would manufacture characters the
    source never produced and the interpreter itself would not.
    """
    out: list[str] = []
    for n in numbers:
        value = n - constant
        if value < 0 or value > char_max:
            return None
        out.append(chr(value))
    return "".join(out)


def _int_constants(source: str) -> "dict[str, int]":
    """Names assigned exactly one literal integer. A name assigned more than
    once is ambiguous (its value at the decode site is not proven) and is
    dropped, exactly as the XOR passes drop a reassigned string variable."""
    counts: dict[str, int] = {}
    values: dict[str, int] = {}
    for name, value in _VBS_INT_ASSIGN.findall(source):
        counts[name] = counts.get(name, 0) + 1
        values[name] = int(value)
    return {name: values[name] for name, n in counts.items() if n == 1}


# The function that encloses a decoder body, so a call to it can be found.
_VBS_FUNCTION = re.compile(
    r"\bFunction\s+([A-Za-z_]\w*)\s*\(", re.I
)


def _enclosing_vbs_function(source: str, position: int) -> str | None:
    """Name of the `Function` whose definition most recently opened before
    `position`. The decoder body matched at `position` lives inside it, and
    that name is how the array that feeds the decoder is linked to it."""
    name = None
    for m in _VBS_FUNCTION.finditer(source):
        if m.start() > position:
            break
        name = m.group(1)
    return name


def _vbs_decoder_inputs(source: str, decoder_names: "set[str]") -> "list[re.Match]":
    """Array(...) literals that are ACTUALLY passed to a named decoder.

    Two forms, and only these -- a literal array elsewhere in the file (a lookup
    table, an RGB triple, an opcode list) is never decoded just because its
    values happen to fall in range:

      decoder(Array(...))                direct argument
      v = Array(...) ... decoder(v)      assigned then passed

    Returns the `_VBS_ARRAY` matches that are linked, so each keeps its exact
    source position for provenance.
    """
    if not decoder_names:
        return []
    names = "|".join(re.escape(n) for n in decoder_names)
    linked: list[re.Match] = []
    # Direct: a decoder call whose argument is an Array literal. The Array match
    # is recovered by position so provenance points at the array, not the call.
    direct_call = re.compile(rf"(?:{names})\s*\(\s*(Array\(\s*(?:\d+\s*,\s*)*\d+\s*\))", re.I)
    linked_spans: set[int] = set()
    for m in direct_call.finditer(source):
        linked_spans.add(m.start(1))
    # Indirect: `v = Array(...)` where v is later passed to a decoder. Only a
    # variable assigned an array literal exactly once, and named in a decoder
    # call, qualifies -- a reassigned variable is ambiguous and dropped.
    assign = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(Array\(\s*(?:\d+\s*,\s*)*\d+\s*\))", re.I)
    assigns: dict[str, list[int]] = {}
    for m in assign.finditer(source):
        assigns.setdefault(m.group(1), []).append(m.start(2))
    passed = re.compile(rf"(?:{names})\s*\(\s*([A-Za-z_]\w*)\s*\)", re.I)
    passed_vars = {m.group(1) for m in passed.finditer(source)}
    for var, positions in assigns.items():
        if var in passed_vars and len(positions) == 1:
            linked_spans.add(positions[0])
    for m in _VBS_ARRAY.finditer(source):
        if m.start() in linked_spans:
            linked.append(m)
    return linked


def find_vbscript_chr_stages(
    source: str, depth: int = 0, truncation: "_Truncation | None" = None
) -> list[TransformStage]:
    """Literal `Chr(e - const)`-over-`Array(...)` reconstructions (VBScript).

    Only arrays syntactically passed to a recognised decoder are decoded: a
    detection of the decoder shape is not licence to decode every in-range
    literal array in the file, which would turn an innocent lookup table into a
    fabricated, provenance-stamped finding.
    """
    stages: list[TransformStage] = []
    decoders = list(_VBS_DECODER.finditer(source))
    if not decoders:
        return stages
    constants = _int_constants(source)
    # The distinct constants every recognised decoder subtracts (group 4 is the
    # constant variable; group 3 is the Chr/ChrW function it feeds). When the
    # text names exactly one constant it applies to the linked arrays; more
    # than one is ambiguous and declined, like the PowerShell XOR key.
    active = [m for m in decoders if m.group(4) in constants]
    offsets = {constants[m.group(4)] for m in active}
    if len(offsets) != 1:
        return stages
    constant = next(iter(offsets))
    # The conversion function's domain: narrow `Chr`/`ChrB` to a single byte,
    # `ChrW` to a UTF-16 code unit. When decoders disagree, taking the WIDER
    # bound would admit a value the narrower function rejects, so take the
    # NARROWEST any matched decoder uses -- a value that function would refuse
    # is refused here too.
    char_max = min(
        (_CHR_MAX if m.group(3).lower() in ("chr", "chrb") else _CHRW_MAX)
        for m in active
    )
    # The decoder function names, so only arrays actually fed to them decode.
    decoder_names = {
        name
        for m in active
        if (name := _enclosing_vbs_function(source, m.start())) is not None
    }
    for match in _vbs_decoder_inputs(source, decoder_names):
        tokens = match.group(1)
        if len(tokens) > MAX_INPUT_CHARS:
            if truncation is not None:
                truncation.dropped_input = True
            continue
        try:
            numbers = [int(t.strip()) for t in tokens.split(",")]
        except ValueError:
            continue
        # A real decoder input is more than a one- or two-element array; a short
        # literal array elsewhere in the program is not ciphertext. The floor is
        # deliberately small but non-trivial so a stray `Array(1,2)` is not
        # decoded into two control characters and recorded as a finding.
        if len(numbers) < 3:
            continue
        decoded = _decode_chr_offset(numbers, constant, char_max)
        if decoded is None:
            continue
        if len(decoded) > MAX_OUTPUT_CHARS:
            if truncation is not None:
                truncation.dropped_output = True
            continue
        offset = match.start(1)
        stages.append(
            TransformStage(
                kind=VBSCRIPT_CHR,
                key=constant,
                delimiter=",",
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


# The PowerShell function that encloses a decoder body. PowerShell names are
# commonly `Verb-Noun`, so a hyphen is part of the name (but not a leading or
# trailing one). The name is later `re.escape`d into the call patterns.
_PS_FUNCTION = re.compile(r"\bfunction\s+([A-Za-z_]\w*(?:-\w+)*)\s*[\({]", re.I)
# A `$h = 22624` assignment, refused when the value is part of a larger
# expression (`$h = 22624 + 1`, `$h = 226.24`) -- the negative lookahead keeps
# a computed value from being read as a proven literal, exactly as the VBScript
# path does. (MAJOR-2: the PowerShell path was previously fail-open here.)
# Two exclusions, for two ways a digit run is not a bare decimal literal:
#  - an IMMEDIATE continuation char (no space): `x`/`X` (hex `0x10`), another
#    digit, or `.` (float) -- so `0x10` is not read as `0`;
#  - a SPACED binary operator: `+ - * / % & ^ |` after optional whitespace, so
#    `$h = 500 + 1` / `$h = 22624 | Out-Null` are refused rather than read as
#    their first term. `e`/`E` alone are not excluded: a bare line after the
#    number may begin with them, and PS has no `500e3` literal in this grammar.
_PS_INT_ASSIGN = re.compile(
    r"\$([A-Za-z_]\w*)\s*=\s*(\d+)(?![\dxX.])(?!\s*[.&+\-*/%^|])"
)


def _enclosing_ps_function(source: str, position: int) -> str | None:
    name = None
    for m in _PS_FUNCTION.finditer(source):
        if m.start() > position:
            break
        name = m.group(1)
    return name


def find_powershell_chr_stages(
    source: str, depth: int = 0, truncation: "_Truncation | None" = None
) -> list[TransformStage]:
    """Literal `[char]($e - $h)`-over-`@(...)` reconstructions (PowerShell).

    Only `@(...)` arrays syntactically passed to a recognised decoder function
    are decoded -- an unrelated in-range literal list is not a finding."""
    stages: list[TransformStage] = []
    decoders = list(_PS_CHAR.finditer(source))
    if not decoders:
        return stages
    # `$h = 22624` style assignments, reading ONLY a bare integer literal (a
    # computed value is refused by the lookahead, not read as its first term).
    ps_ints: dict[str, int] = {}
    ps_counts: dict[str, int] = {}
    for name, value in _PS_INT_ASSIGN.findall(source):
        ps_counts[name] = ps_counts.get(name, 0) + 1
        ps_ints[name] = int(value)
    ps_ints = {n: v for n, v in ps_ints.items() if ps_counts[n] == 1}
    active = [m for m in decoders if m.group(3) in ps_ints]
    offsets = {ps_ints[m.group(3)] for m in active}
    if len(offsets) != 1:
        return stages
    constant = next(iter(offsets))
    # An `@(...)` array is linked to a decoder only by being PASSED to the
    # decoder function -- never by sharing a variable name with it. Linking by
    # the foreach-iterated variable name was unsafe: that name is usually the
    # decoder's PARAMETER, so a benign module-scope global of the same name
    # (`$a` is the conventional parameter) would be decoded into a fabricated
    # finding though it is never passed. Only an array that actually reaches the
    # decoder is decoded. Two call spellings, plus assign-then-pass:
    #   fn @(...)      fn(@(...))                 direct argument (bare or paren)
    #   $v = @(...) ... fn $v   /   fn($v)        assigned once, then passed
    linked: set[int] = set()
    decoder_names = {
        name
        for m in active
        if (name := _enclosing_ps_function(source, m.start())) is not None
    }
    if not decoder_names:
        return stages
    names = "|".join(re.escape(n) for n in decoder_names)
    # A decoder NAME also appears at its own `function <name>(...)` definition.
    # A "call" that is actually that definition must be ignored, or the
    # parameter list reads as an argument and `function Convert-Data($a)` would
    # "link" a global `$a` never passed to it -- the MAJOR-1 harm through the
    # back door. The definition sites are found exactly (via `_PS_FUNCTION`, so
    # any spacing/name form resolves identically to the enclosing-function
    # logic) and any candidate starting at one is dropped. Position-based rather
    # than a fixed-width look-behind, which `function  Name` (extra space) slips
    # past.
    # A name occurrence is a DEFINITION (not a call) when its position is the
    # name of a `function <name>` definition. Matched EXACTLY by position via
    # `_PS_FUNCTION` (whose group-1 start is the name, whatever whitespace or
    # newlines separate it from the keyword), not by a bounded text window --
    # a long whitespace run between `function` and the name would slip a
    # definition's default-parameter `@(...)` past a windowed guard and
    # fabricate a stage from an array never passed to anything.
    definition_name_starts = {m.start(1) for m in _PS_FUNCTION.finditer(source)}

    def _is_definition(pos: int) -> bool:
        return pos in definition_name_starts

    # Direct: `fn @(...)` or `fn(@(...))` -- an optional `(` between name and @.
    direct = re.compile(rf"({names})\s*\(?\s*(@\(\s*(?:\d+\s*,\s*)*\d+\s*\))", re.I)
    for m in direct.finditer(source):
        if not _is_definition(m.start(1)):
            linked.add(m.start(2))
    # Assign-then-pass: `$v = @(...)` where $v is later passed to a decoder, and
    # $v is assigned exactly once (a reassigned source is ambiguous, dropped).
    assign = re.compile(r"\$([A-Za-z_]\w*)\s*=\s*(@\(\s*(?:\d+\s*,\s*)*\d+\s*\))", re.I)
    assigns: dict[str, list[int]] = {}
    for m in assign.finditer(source):
        assigns.setdefault(m.group(1), []).append(m.start(2))
    passed = re.compile(rf"({names})\s*\(?\s*\$([A-Za-z_]\w*)", re.I)
    passed_vars = {
        m.group(2) for m in passed.finditer(source) if not _is_definition(m.start(1))
    }
    for var, positions in assigns.items():
        if var in passed_vars and len(positions) == 1:
            linked.add(positions[0])
    for match in _PS_AT_ARRAY.finditer(source):
        if match.start() not in linked:
            continue
        tokens = match.group(1)
        if len(tokens) > MAX_INPUT_CHARS:
            if truncation is not None:
                truncation.dropped_input = True
            continue
        try:
            numbers = [int(t.strip()) for t in tokens.split(",")]
        except ValueError:
            continue
        if len(numbers) < 3:
            continue
        decoded = _decode_chr_offset(numbers, constant)
        if decoded is None:
            continue
        if len(decoded) > MAX_OUTPUT_CHARS:
            if truncation is not None:
                truncation.dropped_output = True
            continue
        offset = match.start(1)
        stages.append(
            TransformStage(
                kind=POWERSHELL_CHR,
                key=constant,
                delimiter=",",
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


# --- JavaScript string-array + base64 + RC4, with helper folding -----------
#
# A different obfuscation family from the XOR and Chr passes above, handled the
# same way: structurally, never by name, and fail-closed. `javascript-obfuscator`
# hides operational strings in a string array, behind a base64+RC4 decoder, and
# assembles the strings a program actually uses (ActiveX ProgIDs such as
# `MSXML2.XMLHTTP`) through PURE helper functions -- `f(x)=x(y)` (apply) and
# `f(x,y)=x+y` (concat) -- over constant indices into a self-defending
# array-returning function. None of that is runtime-dependent: the indices are
# literals, the helpers are pure, the decoder is a fixed transform, and the
# self-defending guard is a comparison of a constant with itself. So the final
# operational strings are determined by the bytes on disk, exactly like the XOR
# and Chr decodes -- and like them, a model asked to reconstruct them by hand
# has a dozen ways to get it wrong.
#
# This pass recovers them by a BOUNDED EXPRESSION FOLDER, not a JavaScript
# engine. It evaluates a tiny grammar (string/number literals, `a+b`, `a[b]`,
# and calls of functions it has PROVEN are the decoder, the array accessor, or a
# pure apply/concat/eq helper) against an environment it builds only when the
# whole family shape is structurally proven. Anything outside that grammar --
# an impure helper, a dynamic key, a mutated array, a guard that is not
# constant-true, a decoder that is not this base64+RC4 -- folds to UNKNOWN and
# nothing is emitted. It never executes, imports, or interprets the script.
JS_STRINGARRAY_FOLD = "js_stringarray_fold"

# The standard base64 glyph set. The family's decoder uses these glyphs in a
# nonstandard ORDER (lowercase first); the alphabet is identified by being a
# 60+ char string drawn from exactly this set, not by a fixed literal.
_B64_GLYPHS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="
)

# Bounds on the fold. Small and about the artifact: an expression or array that
# exceeds them is not the unambiguous case this handles.
_FOLD_MAX_NODES = 4096      # expression nodes evaluated per top-level fold
_FOLD_MAX_DEPTH = 64        # fold recursion depth
_FOLD_MAX_ARRAY = 20_000    # string-array element count
_FOLD_MAX_Q = 4_000         # accessor-array element count
_FOLD_MAX_SOURCE = MAX_INPUT_CHARS


class _Unknown:
    """The fold's bottom value: a subexpression that is not statically
    determined. It is distinct from every real string/number, and it propagates
    -- any UNKNOWN inside an expression makes the whole expression UNKNOWN, so a
    partially-folded value is never emitted."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNKNOWN"


_UNKNOWN = _Unknown()


def _fold_b64_custom(encoded: str) -> str | None:
    """Base64-decode with the family's custom-ORDER alphabet, then read the
    bytes back as UTF-8 (the source percent-encodes each byte and calls
    decodeURIComponent). Pure; mirrors the decoder's own loop. Returns None on
    any invalid digit -- fail closed."""
    order = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="
    out: list[int] = []
    bits = 0
    val = 0
    for ch in encoded:
        if ch == "=":
            break
        idx = order.find(ch)
        if idx < 0:
            return None
        val = (val << 6) | idx
        bits += 6
        if bits >= 8:
            bits -= 8
            out.append((val >> bits) & 0xFF)
    try:
        return bytes(out).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _fold_rc4(data: str, key: str) -> str:
    """Standard RC4 over the code units of `data` with string `key`, returning
    fromCharCode of each XOR byte -- exactly what the family's `p(C,S)` does."""
    s = list(range(256))
    j = 0
    klen = len(key)
    for i in range(256):
        j = (j + s[i] + ord(key[i % klen])) % 256
        s[i], s[j] = s[j], s[i]
    i = j = 0
    out: list[str] = []
    for ch in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(chr(ord(ch) ^ s[(s[i] + s[j]) % 256]))
    return "".join(out)


def _fold_decode(array: "list[str]", index: int, key: str) -> str | None:
    """The decoder `v(index, key)` = RC4(base64_custom(array[index]), key).
    Pure arithmetic; None (fail closed) on any step that does not resolve."""
    if not (0 <= index < len(array)):
        return None
    if not key:
        # RC4's key schedule indexes `key[i % len(key)]`; an empty key is not a
        # call this decoder family makes, and evaluating it would divide by
        # zero. Refuse rather than invent a value.
        return None
    decoded = _fold_b64_custom(array[index])
    if decoded is None:
        return None
    return _fold_rc4(decoded, key)


# --- the bounded expression folder -----------------------------------------
#
# Grammar (the only syntax it will evaluate):
#   expr    := concat
#   concat  := postfix ('+' postfix)*
#   postfix := primary ( '[' expr ']' | '.' IDENT | '(' args ')' )*
#   primary := STRING | NUMBER | IDENT | '(' expr ')'
# Any other character or shape makes the parse return None, which folds to
# UNKNOWN. This is a calculator over proven-pure operations, not an interpreter.


def _fold_tokenize(expr: str):
    toks: list[tuple] = []
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c in "'\"":
            q = c
            j = i + 1
            buf: list[str] = []
            while j < n:
                d = expr[j]
                if d == "\\" and j + 1 < n:
                    buf.append(d)
                    buf.append(expr[j + 1])
                    j += 2
                    continue
                if d == q:
                    break
                buf.append(d)
                j += 1
            if j >= n:
                return None  # unterminated string
            toks.append(("str", _unescape("".join(buf))))
            i = j + 1
            continue
        if c.isdigit() or (c == "0" and i + 1 < n and expr[i + 1] in "xX"):
            j = i
            if expr[j:j + 2].lower() == "0x":
                j += 2
                while j < n and expr[j] in "0123456789abcdefABCDEF":
                    j += 1
                toks.append(("num", int(expr[i:j], 16)))
            else:
                while j < n and expr[j].isdigit():
                    j += 1
                toks.append(("num", int(expr[i:j])))
            i = j
            continue
        if c.isalpha() or c in "_$":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] in "_$"):
                j += 1
            toks.append(("id", expr[i:j]))
            i = j
            continue
        if c in "()[],+.":
            toks.append((c, c))
            i += 1
            continue
        return None  # outside the grammar
    return toks


class _FoldParser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def advance(self):
        tok = self.peek()
        self.i += 1
        return tok


def _fold_parse(expr: str):
    toks = _fold_tokenize(expr)
    if toks is None:
        return None
    p = _FoldParser(toks)
    node = _fold_concat(p)
    if node is None or p.i != len(toks):
        return None  # trailing garbage
    return node


def _fold_concat(p):
    left = _fold_postfix(p)
    if left is None:
        return None
    while p.peek()[0] == "+":
        p.advance()
        right = _fold_postfix(p)
        if right is None:
            return None
        left = ("add", left, right)
    return left


def _fold_postfix(p):
    node = _fold_primary(p)
    if node is None:
        return None
    while True:
        k = p.peek()[0]
        if k == "[":
            p.advance()
            idx = _fold_concat(p)
            if idx is None or p.peek()[0] != "]":
                return None
            p.advance()
            node = ("index", node, idx)
        elif k == ".":
            p.advance()
            kk, name = p.advance()
            if kk != "id":
                return None
            node = ("index", node, ("str", name))
        elif k == "(":
            p.advance()
            args = []
            if p.peek()[0] != ")":
                while True:
                    a = _fold_concat(p)
                    if a is None:
                        return None
                    args.append(a)
                    if p.peek()[0] == ",":
                        p.advance()
                        continue
                    break
            if p.peek()[0] != ")":
                return None
            p.advance()
            node = ("call", node, args)
        else:
            break
    return node


def _fold_primary(p):
    kind, val = p.peek()
    if kind == "str":
        p.advance()
        return ("str", val)
    if kind == "num":
        p.advance()
        return ("num", val)
    if kind == "(":
        p.advance()
        node = _fold_concat(p)
        if node is None or p.peek()[0] != ")":
            return None
        p.advance()
        return node
    if kind == "id":
        p.advance()
        return ("id", val)
    return None


@dataclass
class _FoldEnv:
    """The proven environment a folder evaluates against. Every field is built
    only when the family shape is structurally proven, so each function named
    here has been checked to do exactly what its role says."""

    string_array: "list[str]"      # the decoder's backing array
    decoder_names: set             # names proven to be the base64+RC4 decoder
    decoder_offset: int            # the `a = a - OFFSET` index normalization
    accessor_names: set            # names proven to index the accessor array q
    q_values: list                 # q, pre-resolved to final str / UNKNOWN
    helpers: dict                  # member -> ('apply'|'concat'|'eq',) | ('const', expr)
    objects: dict                  # value-object name -> {key: expr-string}
    helper_objects: set            # names of objects whose members are helpers


class _Folder:
    def __init__(self, env: "_FoldEnv"):
        self.env = env
        self._budget = _FOLD_MAX_NODES

    def fold(self, expr: str):
        self._budget = _FOLD_MAX_NODES
        node = _fold_parse(expr)
        if node is None:
            return _UNKNOWN
        return self._ev(node, 0)

    def _ev(self, node, depth):
        if depth > _FOLD_MAX_DEPTH:
            return _UNKNOWN
        self._budget -= 1
        if self._budget <= 0:
            return _UNKNOWN
        tag = node[0]
        if tag == "str":
            return node[1]
        if tag == "num":
            return node[1]
        if tag == "add":
            a = self._ev(node[1], depth + 1)
            b = self._ev(node[2], depth + 1)
            if isinstance(a, str) and isinstance(b, str):
                return a + b
            return _UNKNOWN
        if tag == "index":
            base = self._ev(node[1], depth + 1)
            idx = self._ev(node[2], depth + 1)
            return self._index(base, idx, depth)
        if tag == "id":
            name = node[1]
            if name in self.env.objects:
                return ("obj", name)
            if name in self.env.helper_objects:
                return ("helperobj", name)
            if name in self.env.decoder_names:
                return ("fn", "decoder")
            if name in self.env.accessor_names:
                return ("fn", "accessor")
            if name in self.env.helpers:
                role = self.env.helpers[name]
                if role[0] == "const":
                    return self._fold_sub(role[1], depth)
                return ("fn", role[0])
            return _UNKNOWN
        if tag == "call":
            callee = self._ev(node[1], depth + 1)
            return self._invoke(callee, node[2], depth)
        return _UNKNOWN

    def _invoke(self, callee, argnodes, depth):
        if not (isinstance(callee, tuple) and callee and callee[0] == "fn"):
            return _UNKNOWN
        role = callee[1]
        if role == "decoder":
            if len(argnodes) != 2:
                return _UNKNOWN
            idx = self._ev(argnodes[0], depth + 1)
            key = self._ev(argnodes[1], depth + 1)
            if not isinstance(idx, int) or not isinstance(key, str):
                return _UNKNOWN
            out = _fold_decode(self.env.string_array, idx - self.env.decoder_offset, key)
            return out if out is not None else _UNKNOWN
        if role == "accessor":
            if len(argnodes) != 1:
                return _UNKNOWN
            idx = self._ev(argnodes[0], depth + 1)
            if not isinstance(idx, int):
                return _UNKNOWN
            if not (0 <= idx < len(self.env.q_values)):
                return _UNKNOWN
            val = self.env.q_values[idx]
            return val if isinstance(val, str) else _UNKNOWN
        if role == "apply":
            # apply(f, x) == f(x): arg0 must resolve to a fn marker
            if len(argnodes) != 2:
                return _UNKNOWN
            f = self._ev(argnodes[0], depth + 1)
            return self._invoke(f, [argnodes[1]], depth + 1)
        if role == "concat":
            if len(argnodes) != 2:
                return _UNKNOWN
            a = self._ev(argnodes[0], depth + 1)
            b = self._ev(argnodes[1], depth + 1)
            if isinstance(a, str) and isinstance(b, str):
                return a + b
            return _UNKNOWN
        return _UNKNOWN

    def _index(self, base, idx, depth):
        if not isinstance(idx, str):
            return _UNKNOWN
        if isinstance(base, tuple) and base and base[0] == "obj":
            obj = self.env.objects.get(base[1])
            if obj is None or idx not in obj:
                return _UNKNOWN
            return self._fold_sub(obj[idx], depth)
        if isinstance(base, tuple) and base and base[0] == "helperobj":
            role = self.env.helpers.get(idx)
            if role is None:
                return _UNKNOWN
            if role[0] == "const":
                return self._fold_sub(role[1], depth)
            return ("fn", role[0])
        return _UNKNOWN

    def _fold_sub(self, expr, depth):
        node = _fold_parse(expr)
        if node is None:
            return _UNKNOWN
        return self._ev(node, depth + 1)


# --- structural recognition (builds the environment, or refuses) ------------
#
# Regex LOCATES candidates; balanced, string-aware scanning is the authority
# for nested syntax. Every proof is structural and local; the only cross-scope
# reasoning is a bounded alias-chain closure, the same idiom the PowerShell
# `-bxor` pass already uses. If any piece is missing or ambiguous, the whole
# family is refused and nothing is folded.


def _balanced(src: str, startpos: int, op: str, cl: str):
    """The balanced `op..cl` slice starting at the first `op` at/after
    `startpos`, skipping string literals. Returns (text, start, end) or None."""
    try:
        st = src.index(op, startpos)
    except ValueError:
        return None
    i = st
    depth = 0
    instr = None
    esc = False
    while i < len(src):
        c = src[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == instr:
                instr = None
        else:
            if c in "'\"":
                instr = c
            elif c == op:
                depth += 1
            elif c == cl:
                depth -= 1
                if depth == 0:
                    return src[st:i + 1], st, i
        i += 1
    return None


def _split_top(inner: str) -> "list[str]":
    """Top-level comma split, string/bracket aware."""
    segs: list[str] = []
    depth = 0
    instr = None
    esc = False
    cur: list[str] = []
    for ch in inner:
        if instr:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == instr:
                instr = None
        else:
            if ch in "'\"":
                instr = ch
                cur.append(ch)
            elif ch in "([{":
                depth += 1
                cur.append(ch)
            elif ch in ")]}":
                depth -= 1
                cur.append(ch)
            elif ch == "," and depth == 0:
                segs.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
    if cur:
        segs.append("".join(cur))
    return segs


def _string_array_elems(seg: str):
    """If `seg` is exactly `['...','...']` of string literals, return the list;
    otherwise None (an element that is not a string literal rejects it)."""
    seg = seg.strip()
    if not (seg.startswith("[") and seg.endswith("]")):
        return None
    out: list[str] = []
    for e in _split_top(seg[1:-1]):
        e = e.strip()
        m = re.fullmatch(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", e)
        if not m:
            return None
        out.append(_unescape(m.group(1) if m.group(1) is not None else m.group(2)))
    return out


def _find_string_array_fn(source: str):
    """`function N(){ ... ['...', ...] ... }` whose body returns the string
    array (directly or via the `N=function(){return ARR}` self-reassignment the
    obfuscator emits). Returns (name, elems) or None. The decoder-linkage check
    downstream is what actually qualifies it."""
    for m in re.finditer(r"function\s+([A-Za-z_$][\w$]*)\s*\(\s*\)\s*\{", source):
        name = m.group(1)
        body = _balanced(source, m.end() - 1, "{", "}")
        if not body:
            continue
        btext = body[0]
        # The data variable that holds the array literal, e.g. `var data=[...]`.
        dm = re.search(r"(?:var|let|const)?\s*([A-Za-z_$][\w$]*)\s*=\s*\[", btext)
        if not dm:
            continue
        datavar = dm.group(1)
        br = _balanced(btext, dm.end() - 1, "[", "]")
        if not br:
            continue
        elems = _string_array_elems(br[0])
        if not elems or not (1 <= len(elems) <= _FOLD_MAX_ARRAY):
            continue
        # The array variable must be assigned exactly once: a second write means
        # the value at the call site depends on control flow, which is
        # interpretation, not reading. Refuse rather than read a stale literal.
        writes = re.findall(
            rf"(?<![\w$.]){re.escape(datavar)}\s*(?:\[[^\]]*\])?\s*(?:\+=|=(?!=))", btext
        )
        if len(writes) != 1:
            continue
        # The array this var holds must be the one the function actually RETURNS,
        # or the decoy-first-array case reads a literal the program never uses.
        # Two spellings return it: `return datavar;` directly, or the
        # self-reassignment `N=function(){return datavar}` the obfuscator emits.
        returns_datavar = re.search(
            rf"return\s+{re.escape(datavar)}\s*;", btext
        ) or re.search(
            rf"{re.escape(name)}\s*=\s*function\s*\(\s*\)\s*\{{\s*return\s+"
            rf"{re.escape(datavar)}\s*;",
            btext,
        )
        if returns_datavar:
            return name, elems
    return None


def _find_fold_decoder(source: str, array_fn: str):
    """`function D(a,q){ a=a-OFF; ... <custom b64 alphabet> ... RC4 ... }`,
    matched structurally: (1) the first param normalized by subtraction, (2) a
    60+ char base64 alphabet drawn from the standard glyph set (both cases and
    digits), (3) an RC4 element swap `S[i]=S[j]` and a 256 modulus, and (4) a
    reference to the string-array function. Returns (name, offset) or None."""
    for m in re.finditer(
        r"function\s+([A-Za-z_$][\w$]*)\s*\(\s*([A-Za-z_$][\w$]*)\s*,\s*"
        r"([A-Za-z_$][\w$]*)\s*\)\s*\{",
        source,
    ):
        name, p0 = m.group(1), m.group(2)
        body = _balanced(source, m.end() - 1, "{", "}")
        if not body:
            continue
        b = body[0]
        off_m = re.search(
            rf"{re.escape(p0)}\s*=\s*{re.escape(p0)}\s*-\s*(0x[0-9a-fA-F]+|\d+)", b
        )
        if not off_m:
            continue
        alpha_ok = False
        for sm in re.finditer(r"'((?:[^'\\]|\\.){60,})'|\"((?:[^\"\\]|\\.){60,})\"", b):
            lit = _unescape(sm.group(1) if sm.group(1) is not None else sm.group(2))
            if (len(lit) >= 60 and set(lit) <= _B64_GLYPHS
                    and any(c.islower() for c in lit)
                    and any(c.isupper() for c in lit)
                    and any(c.isdigit() for c in lit)):
                alpha_ok = True
                break
        if not alpha_ok:
            continue
        rc4_mod = re.search(r"%\s*(?:0x100|256)\b", b)
        swap = re.search(
            r"[A-Za-z_$][\w$]*\s*\[[^\]]+\]\s*=\s*[A-Za-z_$][\w$]*\s*\[[^\]]+\]", b
        )
        uses_array = re.search(rf"{re.escape(array_fn)}\s*\(\s*\)", b)
        if rc4_mod and swap and uses_array:
            digits = off_m.group(1)
            off = int(digits, 16) if digits.lower().startswith("0x") else int(digits)
            return name, off
    return None


def _top_level_write_counts(source: str) -> dict:
    """How many times each name is WRITTEN (`=`, `+=`, or through an index) at
    brace-depth zero -- outside every function body. String-aware.

    Assignments inside a function body are that function's business and may
    reuse a name as a local (a minified RC4 loop counter reusing `R`, say);
    only writes at the top level, where the decoder/accessor aliases are bound,
    decide whether a fold name's value is agreed on across the file. A name
    written more than once at the top level is ambiguous at its call sites --
    reassigned to a call, array, object or other function -- and treating it as
    the decoder/accessor would fold a call that at runtime is something else.
    """
    counts: dict = {}
    depth = 0
    instr = None
    esc = False
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == instr:
                instr = None
            i += 1
            continue
        if c in "'\"":
            instr = c
            i += 1
            continue
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            i += 1
            continue
        if depth == 0:
            wm = re.match(
                r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*(?:\[[^\]]*\])?\s*(?:\+=|=(?!=))",
                source[i:],
            )
            if wm:
                counts[wm.group(1)] = counts.get(wm.group(1), 0) + 1
                i += wm.end()
                continue
        i += 1
    return counts


def _fold_aliases_of(source: str, target: str) -> set:
    """All names bound to `target` through a chain of plain `NAME = OTHERNAME`
    assignments (no call, member, or index on the RHS). A bounded transitive
    closure -- the alias-chain idiom the `-bxor` recogniser uses -- so a 2-hop
    `XX=qZ; qZ=v` resolves `XX`. A closure over simple bindings; it interprets
    no scope or control flow.

    A name with a CONFLICTING simple binding is EXCLUDED: if `NAME = OTHER`
    appears anywhere with `OTHER` not itself an alias of `target`, the name's
    value at a call site is ambiguous, and treating it as the decoder/accessor
    could fold a call that at runtime is something else. (A reused single-letter
    loop local such as `R=U%4...` is an arithmetic write, not a simple-identifier
    binding, so it does not make `R=H` ambiguous -- only a competing
    `NAME=OTHERNAME` does.) `target` was proven structurally and is always in.
    """
    # A simple alias binding is `NAME = OTHER` where OTHER is a bare identifier
    # that ENDS the right-hand side -- the next token closes the statement or
    # the comma-chain (`,`, `;`, `)`, `}`, newline, or end). `R = U % 4` is not
    # a simple binding (U is followed by an operator); it is arithmetic, and the
    # trailing guard rejects it, so a reused loop local does not masquerade as
    # an alias.
    binds = re.findall(
        r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*=\s*([A-Za-z_$][\w$]*)\s*(?=[,;)}\r\n]|$)",
        source,
    )
    # Every name that receives a simple-identifier binding, and the set of RHS
    # identifiers it is bound to. A name bound to two different identifiers is
    # ambiguous regardless of what those identifiers turn out to be.
    rhs_of: dict = {}
    for name, rhs in binds:
        rhs_of.setdefault(name, set()).add(rhs)
    # Top-level write counts catch the reassignment a simple-binding check
    # cannot see: `X = makeSomethingElse()`, `X = [..]`, `X = {..}`, `X = new ..`
    # are not simple-identifier bindings, so a name reassigned that way would
    # otherwise stay an alias. More than one top-level write -> ambiguous.
    top_writes = _top_level_write_counts(source)
    reached = {target}
    for _ in range(8):
        grew = False
        for name, rhs in binds:
            if name in reached or rhs not in reached:
                continue
            # Exclude a name bound to more than one distinct identifier, or
            # written more than once at the top level: its value is not agreed.
            if len(rhs_of.get(name, ())) != 1:
                continue
            if top_writes.get(name, 0) > 1:
                continue
            reached.add(name)
            grew = True
        if not grew:
            break
    return reached


def _extract_helper_members(objtext: str) -> dict:
    """Map an object literal's pure-function members to fold roles:
       ('apply',) for function(x,y){return x(y)}
       ('concat',) for function(x,y){return x+y}
       ('eq',) for function(x,y){return x===y}
       ('const', expr) for a decoder-call or string-literal member.
    Any member that is none of these is omitted, so a reference to it folds to
    UNKNOWN -- an impure or unexpected member never becomes a fold operation."""
    members: dict = {}
    for mm in re.finditer(
        r"'([A-Za-z_$][\w$]*)'\s*:\s*function\(([^)]*)\)\{return ([^;]+);\}", objtext
    ):
        nm, params, expr = mm.group(1), mm.group(2), mm.group(3).strip()
        ps = [p.strip() for p in params.split(",")]
        if len(ps) == 2 and expr == f"{ps[0]}({ps[1]})":
            members[nm] = ("apply",)
        elif len(ps) == 2 and expr == f"{ps[0]}+{ps[1]}":
            members[nm] = ("concat",)
        elif len(ps) == 2 and expr == f"{ps[0]}==={ps[1]}":
            members[nm] = ("eq",)
    for mm in re.finditer(
        r"'([A-Za-z_$][\w$]*)'\s*:\s*"
        r"([A-Za-z_$][\w$]*\((?:0x[0-9a-fA-F]+|\d+),'(?:[^'\\]|\\.)*'\))",
        objtext,
    ):
        members.setdefault(mm.group(1), ("const", mm.group(2)))
    for mm in re.finditer(r"'([A-Za-z_$][\w$]*)'\s*:\s*'((?:[^'\\]|\\.)*)'", objtext):
        members.setdefault(mm.group(1), ("const", "'" + mm.group(2) + "'"))
    return members


def _guard_folds_true(guard: str, folder: "_Folder") -> bool:
    """True only when the self-defending guard is PROVEN statically true: a
    direct `X===X` with textually-identical operands, or `CALLEE(A,B)` where
    CALLEE folds to an `eq` (===) helper and A,B fold to the same value (or are
    textually identical). The comparator key may itself be a decoder call. Any
    other guard -> False (refusal)."""
    g = guard.strip()
    parts = re.split(r"(?<![=!<>])===", g)
    if len(parts) == 2 and parts[0].strip() == parts[1].strip():
        return True
    if not g.endswith(")"):
        return False
    depth = 0
    instr = None
    esc = False
    open_at = None
    for i, c in enumerate(g):
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == instr:
                instr = None
        else:
            if c in "'\"":
                instr = c
            elif c == "(":
                if depth == 0:
                    open_at = i
                depth += 1
            elif c == ")":
                depth -= 1
    if open_at is None:
        return False
    callee = g[:open_at].strip()
    args = _split_top(g[open_at + 1:-1])
    if len(args) != 2:
        return False
    cv = folder.fold(callee)
    if not (isinstance(cv, tuple) and len(cv) >= 2 and cv[0] == "fn" and cv[1] == "eq"):
        return False
    if args[0].strip() == args[1].strip():
        return True
    a = folder.fold(args[0].strip())
    b = folder.fold(args[1].strip())
    return not isinstance(a, _Unknown) and a == b


def _resolve_self_defending(source, xname, string_array, decoder_names, decoder_offset):
    """Prove `function xname(){...}` statically returns its local array. The
    family emits a small, enumerable set of spellings, all with one semantics: a
    statically-true guard (or none) governs the function so it returns the
    array-local RET, and any other return lies on the dead branch. Requires RET
    assigned exactly one array literal and never rewritten; a guard, if present,
    that FOLDS to true. Returns (q_elems, x_helper_obj, x_helpers) or None."""
    fm = re.search(rf"function\s+{re.escape(xname)}\s*\(\s*\)\s*\{{", source)
    if not fm:
        return None
    body = _balanced(source, fm.end() - 1, "{", "}")
    if not body:
        return None
    b = body[0]
    # The X-local helper object is optional (a bare `x===x` guard needs none).
    xobj_name = None
    xhelpers: dict = {}
    om = re.search(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*=\s*\{", b)
    if om:
        xobj = _balanced(b, om.end() - 1, "{", "}")
        if xobj:
            hm = _extract_helper_members(xobj[0])
            if hm:
                xobj_name = om.group(1)
                xhelpers = hm
    # Determine the single identifier the function yields on the LIVE path, then
    # require it to be a single-assignment array local. This is what makes the
    # fold read the array the program actually uses, not a decoy on a dead
    # branch. The guard's true branch decides:
    #   * no guard            -> the body's `return RET;`
    #   * if(GUARD) return RET;                       -> RET (the consequent)
    #   * if(GUARD){ ...no return... } return RET;    -> RET (fall-through)
    #   * if(GUARD){ return RET; } ...                -> RET (consequent block)
    # A consequent (or fall-through) that returns anything OTHER than a single
    # array local refuses the function -- including the poison
    # `if(true){return other;} return ret;`, whose live return is `other`.
    gm = re.search(r"if\s*\(", b)
    guard = None
    retname = None
    if gm:
        gs = _balanced(b, gm.start(), "(", ")")
        if not gs:
            return None
        guard = gs[0][1:-1].strip()
        after = b[gs[2] + 1:].lstrip()
        if after.startswith("return"):
            rm = re.match(r"return\s+([A-Za-z_$][\w$]*)\s*;", after)
            if not rm:
                return None
            retname = rm.group(1)
        elif after.startswith("{"):
            blk = _balanced(after, 0, "{", "}")
            if not blk:
                return None
            inner_ret = re.search(r"return\s+([A-Za-z_$][\w$]*)\s*;", blk[0])
            if inner_ret:
                retname = inner_ret.group(1)          # true branch returns this
            else:
                # Empty/return-free true branch: control falls through past an
                # optional (dead) `else { ... }` to the next `return RET;`.
                tail = after[blk[2] + 1:].lstrip()
                if tail.startswith("else"):
                    eb = _balanced(tail, 0, "{", "}")
                    if not eb:
                        return None
                    tail = tail[eb[2] + 1:].lstrip()
                rm = re.match(r"return\s+([A-Za-z_$][\w$]*)\s*;", tail)
                if not rm:
                    return None
                retname = rm.group(1)
        else:
            return None
    else:
        rm = re.search(r"return\s+([A-Za-z_$][\w$]*)\s*;", b)
        if not rm:
            return None
        retname = rm.group(1)

    # RET must be a single-assignment array local, never rewritten.
    asg = list(re.finditer(rf"(?<![\w$.]){re.escape(retname)}\s*=\s*\[", b))
    writes = re.findall(
        rf"(?<![\w$.]){re.escape(retname)}\s*(?:\[[^\]]*\])?\s*(?:\+=|=(?!=))", b
    )
    if len(asg) != 1 or len(writes) != 1:
        return None
    asg_m = asg[0]
    arr = _balanced(b, asg_m.end() - 1, "[", "]")
    if not arr:
        return None
    q_elems = [e.strip() for e in _split_top(arr[0][1:-1])]
    if not q_elems or len(q_elems) > _FOLD_MAX_Q:
        return None
    if guard is not None:
        env = _FoldEnv(
            string_array=string_array, decoder_names=decoder_names,
            decoder_offset=decoder_offset, accessor_names=set(), q_values=[],
            helpers=xhelpers, objects={},
            helper_objects=({xobj_name} if xobj_name else set()),
        )
        if not _guard_folds_true(guard, _Folder(env)):
            return None
    return q_elems, xobj_name, xhelpers


def _find_fold_accessor(source, string_array, decoder_names, decoder_offset):
    """`function H(d[,F]){ ... C=X(); ... return C[d] }` (either `var C=X(),
    S=C[d]; return S` or `var C=X(); return C[d]`) where X is a self-defending
    array-returning function. Returns (accessor_name, q_elems, x_helper_obj,
    x_helpers) or None."""
    for m in re.finditer(
        r"function\s+([A-Za-z_$][\w$]*)\s*\(\s*([A-Za-z_$][\w$]*)\s*"
        r"(?:,\s*[A-Za-z_$][\w$]*\s*)?\)\s*\{",
        source,
    ):
        d = m.group(2)
        body = _balanced(source, m.end() - 1, "{", "}")
        if not body:
            continue
        b = body[0]
        am = re.search(
            rf"([A-Za-z_$][\w$]*)\s*=\s*([A-Za-z_$][\w$]*)\s*\(\s*\)\s*,\s*"
            rf"([A-Za-z_$][\w$]*)\s*=\s*\1\s*\[\s*{re.escape(d)}\s*\]\s*;?\s*return\s+\3",
            b,
        )
        if am:
            xname = am.group(2)
        else:
            am = re.search(
                rf"([A-Za-z_$][\w$]*)\s*=\s*([A-Za-z_$][\w$]*)\s*\(\s*\)\s*;\s*"
                rf"return\s+\1\s*\[\s*{re.escape(d)}\s*\]",
                b,
            )
            if not am:
                continue
            xname = am.group(2)
        res = _resolve_self_defending(
            source, xname, string_array, decoder_names, decoder_offset
        )
        if res is None:
            continue
        q_elems, xobj_name, xhelpers = res
        return m.group(1), q_elems, xobj_name, xhelpers
    return None


def _find_activex_exprs(source: str):
    """Every `new ActiveXObject(<expr>)` argument expression, with its offset."""
    out = []
    for m in re.finditer(r"new\s+ActiveXObject\s*\(", source):
        arg = _balanced(source, m.end() - 1, "(", ")")
        if arg:
            out.append((m.start(), arg[0][1:-1].strip()))
    return out


def _recognize_stringarray_family(source: str):
    """Build a `_FoldEnv` and the ActiveX expressions to fold, but ONLY when the
    complete supported family is structurally proven. Any missing or ambiguous
    piece returns None -- the pass then emits nothing for this artifact."""
    if not source or len(source) > _FOLD_MAX_SOURCE:
        return None
    sa = _find_string_array_fn(source)
    if not sa:
        return None
    array_fn, elems = sa
    dec = _find_fold_decoder(source, array_fn)
    if not dec:
        return None
    dec_name, dec_off = dec

    acc = _find_fold_accessor(source, elems, _fold_aliases_of(source, dec_name), dec_off)
    if not acc:
        return None
    acc_name, q_raw, xobj_name, xhelpers = acc

    # The decoder and accessor are named by a `function NAME(){}` declaration,
    # not an assignment. If either base name is ALSO reassigned at the top level
    # (`dec = wrap(dec)`), its value at a call site is no longer that function,
    # and folding a call through it would invent a string. Refuse.
    top_writes = _top_level_write_counts(source)
    if top_writes.get(dec_name, 0) > 0 or top_writes.get(acc_name, 0) > 0:
        return None

    decoder_names = _fold_aliases_of(source, dec_name)
    accessor_names = _fold_aliases_of(source, acc_name)
    if decoder_names & accessor_names:
        return None  # a name cannot be both the decoder and the accessor

    # Resolve q in the self-defending function's own scope (decoder + its local
    # helper object available), to final strings.
    q_env = _FoldEnv(
        string_array=elems, decoder_names=decoder_names, decoder_offset=dec_off,
        accessor_names=set(), q_values=[], helpers=xhelpers, objects={},
        helper_objects=({xobj_name} if xobj_name else set()),
    )
    q_folder = _Folder(q_env)
    q_values = []
    for e in q_raw:
        v = q_folder.fold(e)
        q_values.append(v if isinstance(v, str) else _UNKNOWN)

    # Outer scope: every object literal contributes its pure-function members to
    # the shared helper map and its non-function members to its own value map.
    # A member whose helper role conflicts across objects, or an object name
    # declared twice with different value members, is dropped/poisoned so the
    # lookup fails closed rather than pick an arbitrary scope.
    helper_members: dict = {}
    conflicts: set = set()
    helper_obj_names: set = set()
    value_objects: dict = {}
    for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*=\s*\{", source):
        name = m.group(1)
        obj = _balanced(source, m.end() - 1, "{", "}")
        if not obj:
            continue
        text = obj[0]
        hm = _extract_helper_members(text)
        if hm:
            helper_obj_names.add(name)
            for k, role in hm.items():
                if k in helper_members and helper_members[k] != role:
                    conflicts.add(k)
                helper_members[k] = role
        pairs = {}
        for seg in _split_top(text[1:-1]):
            seg = seg.strip()
            km = re.match(r"^'([A-Za-z_$][\w$]*)'\s*:(.*)$", seg, re.S)
            if km and not km.group(2).strip().startswith("function"):
                pairs[km.group(1)] = km.group(2).strip()
        if pairs:
            if name in value_objects and value_objects[name] != pairs:
                value_objects[name] = None
            elif name not in value_objects:
                value_objects[name] = pairs
    for k in conflicts:
        helper_members.pop(k, None)
    value_objects = {k: v for k, v in value_objects.items() if v}

    env = _FoldEnv(
        string_array=elems, decoder_names=decoder_names, decoder_offset=dec_off,
        accessor_names=accessor_names, q_values=q_values, helpers=helper_members,
        objects=value_objects, helper_objects=helper_obj_names,
    )
    return env, _find_activex_exprs(source)


def find_js_stringarray_fold_stages(
    source: str, depth: int = 0, truncation: "_Truncation | None" = None
) -> list[TransformStage]:
    """Every `new ActiveXObject(<expr>)` whose ProgID the bounded folder can
    prove from the string-array+base64+RC4 family. Emits nothing unless the
    whole family shape is recognised; a ProgID that does not fold to a string is
    skipped (fail closed), never guessed."""
    stages: list[TransformStage] = []
    recognized = _recognize_stringarray_family(source)
    if recognized is None:
        return stages
    env, activex = recognized
    folder = _Folder(env)
    seen: set[str] = set()
    for offset, expr in activex:
        if len(expr) > MAX_INPUT_CHARS:
            if truncation is not None:
                truncation.dropped_input = True
            continue
        value = folder.fold(expr)
        if not isinstance(value, str) or not value:
            continue  # not statically determined -> not emitted
        if len(value) > MAX_OUTPUT_CHARS:
            if truncation is not None:
                truncation.dropped_output = True
            continue
        if value in seen:
            continue
        seen.add(value)
        stages.append(
            TransformStage(
                kind=JS_STRINGARRAY_FOLD,
                key=0,
                delimiter="",
                line=source[:offset].count("\n") + 1,
                offset=offset,
                depth=depth,
                encoded=expr,
                output=value,
                input_sha256=_sha(expr),
                output_sha256=_sha(value),
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
        found = (
            find_jscript_stages(text, depth, truncation)
            + find_powershell_stages(text, depth, truncation)
            + find_vbscript_chr_stages(text, depth, truncation)
            + find_powershell_chr_stages(text, depth, truncation)
            + find_js_stringarray_fold_stages(text, depth, truncation)
        )
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
            if (
                find_jscript_stages(text, 0, probe)
                or find_powershell_stages(text, 0, probe)
                or find_vbscript_chr_stages(text, 0, probe)
                or find_powershell_chr_stages(text, 0, probe)
                or find_js_stringarray_fold_stages(text, 0, probe)
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
