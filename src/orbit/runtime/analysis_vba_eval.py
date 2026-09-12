"""A tiny, bounded, fail-closed static evaluator for the VBA constant
byte/character-offset + StrReverse obfuscation family.

STATIC ONLY. Nothing here executes VBA, Office, an interpreter, or a macro: it
evaluates its OWN whitelisted representation of the extracted VBA text, which is
hostile inert data. Every primitive is pure arithmetic or pure string algebra.

It exists to recover the hidden command a maldoc's macro would have passed to
`Shell`, deterministically and with provenance, so the analysis pipeline can
treat that command as exact evidence -- without running anything.

Design: NOT a VBA interpreter. It supports only a measured subset:
  - integer and string literals (decimal and &h hex integers);
  - `Chr`/`ChrB` (0..255), `ChrW` (0..65535);
  - `StrReverse(s)`;
  - `Replace(s, find, repl)` with all-static operands;
  - `Split(s, delim)(index)` with static delimiter and in-bounds static index;
  - `+`/`-` integer arithmetic, unary sign, parentheses;
  - `&` string concatenation (and `+` on two strings);
  - a recognised byte-offset decoder Function: a body that builds a string by
    `&`-accumulating string literals, converts it to a byte array with
    `StrConv(.., vbFromUnicode)`, subtracts (or adds) a constant OFFSET taken
    from the function's single argument to every byte, and converts back with
    `StrConv(.., vbUnicode)`. The offset is the call argument -- never hardcoded.
  - variable references resolved ONLY through a unique static reaching
    definition (a name assigned exactly once in scope).

Anything outside this subset -- a reassigned variable, an unknown function, a
runtime-dependent value, a dynamic index, an unbalanced construct -- makes the
whole evaluation fail closed (returns UNKNOWN or raises VbaRefuse). A partial or
"best effort" decode is never produced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Sentinel for a value that could not be statically proven. It propagates: any
#: operation with an UNKNOWN operand is UNKNOWN, so a single unknown input can
#: never be silently dropped to yield a confident-looking wrong answer.
UNKNOWN = object()

# Hostile-input bounds. An attacker controls the macro text, so every axis that
# could otherwise run away is capped; exceeding any cap fails closed.
MAX_EVAL_STEPS = 200_000
MAX_STRING_CHARS = 262_144
MAX_EXPR_DEPTH = 64
MAX_RESOLVE_DEPTH = 64
MAX_SPLIT_FIELDS = 100_000
MAX_HELPER_ENCODED = 262_144
# Python refuses to parse an int string longer than ~4300 digits (raising a
# ValueError, not our VbaRefuse); a VBA numeric literal that long is not a real
# constant anyway. Cap it well under that limit so an over-long literal fails
# CLOSED here rather than throwing an uncaught exception out of the evaluator.
MAX_INT_LITERAL_DIGITS = 64

_CHR_MAX = 0xFF
_CHRW_MAX = 0xFFFF


class VbaRefuse(Exception):
    """Raised to fail closed: an unsupported, ambiguous, or out-of-bounds
    construct. The caller treats a refusal as "no deterministic stage", never
    as a partial result."""


@dataclass(frozen=True)
class ByteOffsetHelper:
    """A recognised byte-offset decoder Function. `encoded` is the fully
    reconstructed accumulator string (static literals concatenated); `arg_name`
    is the parameter that carries the per-byte offset; `sign` is +1 if the body
    ADDS the offset, -1 if it SUBTRACTS it. Decoding is `(ord(c) + sign*-offset)`
    ... see `decode`."""

    name: str
    arg_name: str
    encoded: str
    sign: int  # +1 if body does `byte = byte + arg`, -1 if `byte = byte - arg`

    def decode(self, offset: int) -> str:
        # The body applies `byte <sign> offset`; decoding reproduces exactly
        # that arithmetic on each byte of the accumulator, modulo 256 (a VBA
        # Byte is 0..255 and the assignment wraps). StrConv(vbFromUnicode) takes
        # the low byte of each char; our accumulator is ASCII so ord == byte.
        out = []
        for ch in self.encoded:
            b = ord(ch) & 0xFF
            out.append(chr((b + self.sign * offset) & 0xFF))
        return "".join(out)


# --- helper recognition ------------------------------------------------------
_HELPER_RE = re.compile(
    r"\b(?:Public\s+|Private\s+)?Function\s+(\w+)\s*\(\s*(\w+)\s*\)"
    r"(?P<body>[\s\S]*?)\bEnd\s+Function",
    re.I,
)
#: Every assignment to a variable (`name = rhs`), to replay the accumulator's
#: full construction in source order rather than only its `&`-append lines.
_ACC_ASSIGN = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*$", re.M)
#: An append RHS `acc & <rest>` (the left operand is the accumulator itself).
_ACC_APPEND_RHS = re.compile(r"^(\w+)\s*&\s*(.+)$")
_STRCONV_FROM = re.compile(r"(\w+)\s*=\s*StrConv\(\s*(\w+)\s*,\s*vbFromUnicode\s*\)", re.I)
_STRCONV_TO = re.compile(r"(\w+)\s*=\s*StrConv\(\s*(\w+)\s*,\s*vbUnicode\s*\)", re.I)
_BYTE_OP = re.compile(r"(\w+)\(\s*\w+\s*\)\s*=\s*\1\(\s*\w+\s*\)\s*([+\-])\s*(\w+)")
_NOOP_REVERSE = re.compile(r"StrReverse\(\s*StrReverse\((.*)\)\s*\)")
_STR_LITERAL = re.compile(r'^"((?:[^"]|"")*)"$')


def _strip_noop_reverse(expr: str) -> str:
    """`StrReverse(StrReverse(x))` is an identity wrapper the obfuscator uses as
    noise; fold it away before parsing so the accumulator literals are visible.
    Bounded: each pass strictly shrinks the string."""
    prev = None
    while prev != expr:
        prev = expr
        expr = _NOOP_REVERSE.sub(r"\1", expr)
    return expr


def find_byte_offset_helpers(source: str) -> dict[str, ByteOffsetHelper]:
    """Every well-formed byte-offset decoder Function in the module.

    A function qualifies only if its body describes ONE coherent data-flow chain:
      byte_arr = StrConv(acc, vbFromUnicode)   -- acc is the accumulator
      byte_arr(i) = byte_arr(i) +/- <param>    -- per-byte offset, param = the arg
      result = StrConv(byte_arr, vbUnicode)     -- same byte array back to string
    with `acc` built ONLY from ASCII string literals. Every link is checked to be
    the SAME variable, and there must be EXACTLY ONE of each of the three
    operations -- otherwise an attacker could plant a decoy `StrConv(vbFromUnicode)`
    over an attacker-chosen literal and make the recogniser reconstruct that
    instead of the string the byte-op actually decodes, emitting an
    attacker-controlled command as exact evidence. A body that is ambiguous,
    reads a runtime value into the accumulator, or uses a non-ASCII accumulator
    byte (whose ANSI encoding this decoder does not model) does not qualify and
    is absent from the map (fail closed)."""
    helpers: dict[str, ByteOffsetHelper] = {}
    for m in _HELPER_RE.finditer(source):
        name, arg, body = m.group(1), m.group(2), m.group("body")
        cfs = list(_STRCONV_FROM.finditer(body))
        cts = list(_STRCONV_TO.finditer(body))
        bops = list(_BYTE_OP.finditer(body))
        # Exactly one of each: more than one is an ambiguous chain a decoy could
        # exploit; zero means this is not a byte-offset decoder.
        if len(cfs) != 1 or len(cts) != 1 or len(bops) != 1:
            continue
        cf, ct, bop = cfs[0], cts[0], bops[0]
        byte_arr = cf.group(1)   # variable assigned StrConv(acc, vbFromUnicode)
        acc_var = cf.group(2)    # the accumulator string
        # The whole chain must thread the SAME byte array: the byte op must
        # modify it, and the vbUnicode conversion must read it back.
        if bop.group(1) != byte_arr or ct.group(2) != byte_arr:
            continue
        # the byte op must use THIS function's parameter as the offset
        if bop.group(3) != arg:
            continue
        sign = -1 if bop.group(2) == "-" else 1
        # Reconstruct the accumulator by replaying EVERY assignment to acc_var in
        # source order. Each must be one of: `acc = ""` (init), `acc = "<lit>"`
        # (direct literal), or `acc = acc & "<lit>"` (append). Any other form --
        # an alias (`acc = other`), a runtime value, or an append of a non-literal
        # -- makes the reconstruction incomplete, so the whole helper is refused.
        # Scanning only the `&`-append lines (as before) silently dropped a
        # leading `acc = "<lit>"`, decoding a truncated string: a wrong-but-
        # confident result the fail-closed contract forbids.
        ok = True
        value = ""
        for am in _ACC_ASSIGN.finditer(body):
            if am.group(1) != acc_var:
                continue
            rhs = _strip_noop_reverse(am.group(2).strip())
            append = _ACC_APPEND_RHS.match(rhs)
            if append and append.group(1) == acc_var:
                lit = _STR_LITERAL.match(append.group(2).strip())
                if not lit:
                    ok = False
                    break
                value += lit.group(1).replace('""', '"')
            elif _STR_LITERAL.match(rhs):
                value = _STR_LITERAL.match(rhs).group(1).replace('""', '"')
            else:
                ok = False  # alias / runtime / unsupported form -> refuse
                break
        if not ok or value == "":
            continue
        encoded = value
        if len(encoded) > MAX_HELPER_ENCODED:
            continue
        # The decoder models StrConv(vbFromUnicode) as the ASCII low byte; a
        # non-ASCII accumulator char has an ANSI-codepage encoding this does not
        # model, so refuse rather than decode it wrong.
        if any(ord(c) > 0x7F for c in encoded):
            continue
        helpers[name.lower()] = ByteOffsetHelper(name, arg, encoded, sign)
    return helpers


# --- reaching definitions ----------------------------------------------------
_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$")
_KEYWORDS = {
    "if", "for", "while", "next", "do", "else", "elseif", "loop", "wend",
    "end", "then", "function", "sub", "dim", "set", "const", "exit", "select",
    "case", "with", "on", "goto", "return", "redim",
}


def _split_statements(line: str) -> list[str]:
    """Split a VBA line into statements at top-level `:` separators, without
    splitting inside a string literal. VBA allows `a = 1 : b = 2` on one line,
    so a reassignment hidden after a colon must be seen -- otherwise a variable
    reassigned there would look uniquely defined and decode to a stale value.
    A `""` inside a string is the VBA escaped quote and does not end the string.
    """
    parts: list[str] = []
    buf: list[str] = []
    in_str = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == '"':
            # a doubled quote inside a string is an escaped quote, not a close
            if in_str and i + 1 < n and line[i + 1] == '"':
                buf.append('""')
                i += 2
                continue
            in_str = not in_str
            buf.append(ch)
        elif ch == ":" and not in_str:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


#: A procedure header (`Sub`/`Function`, optionally Public/Private/Friend/Static)
#: and the matching `End Sub`/`End Function`. Used to scope reaching definitions:
#: a name in one procedure is a different local from the same name in another.
_PROC_OPEN = re.compile(
    r"^\s*(?:Public\s+|Private\s+|Friend\s+|Static\s+)*"
    r"(?:Sub|Function)\s+\w+", re.I,
)
_PROC_CLOSE = re.compile(r"^\s*End\s+(?:Sub|Function)\b", re.I)


def _scope_bounds(source: str, position: int) -> tuple[int, int] | None:
    """Character span [start, end) of the procedure enclosing `position`, or
    None if `position` is not inside any procedure.

    VBA scopes locals to a procedure, so a `cmd` in one Sub is unrelated to a
    `cmd` in another. Resolving across that boundary is what let a Shell site
    decode a value defined in a different procedure. A `position` that is NOT
    inside a procedure (module-declarations level) returns None: a real `Shell`
    statement only executes inside a procedure, and scoping it to the whole
    module would let it borrow another procedure's locals -- the same
    cross-scope leak, reintroduced at module level."""
    lines = source.splitlines(keepends=True)
    # map char position -> line index
    offsets = []
    acc = 0
    for ln in lines:
        offsets.append(acc)
        acc += len(ln)
    target_line = 0
    for i, off in enumerate(offsets):
        if off <= position:
            target_line = i
        else:
            break
    # walk up to the nearest procedure header not already closed
    start_line = None
    for i in range(target_line, -1, -1):
        if _PROC_CLOSE.match(lines[i]) and i < target_line:
            # a close between us and a header above means we are not in that proc
            break
        if _PROC_OPEN.match(lines[i]):
            start_line = i
            break
    if start_line is None:
        return None
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        if _PROC_CLOSE.match(lines[i]):
            end_line = i + 1
            break
    start = offsets[start_line]
    end = offsets[end_line] if end_line < len(offsets) else len(source)
    return (start, end)


def collect_reaching_defs(
    source: str, *, position: int | None = None
) -> dict[str, list[str]]:
    """name -> list of RHS expression texts assigned to it, in source order.

    A name with exactly one entry has a unique static reaching definition and
    may be resolved; a name with more than one is ambiguous and refused. Lines
    that are not simple `name = rhs` assignments (control flow, `Dim`, calls)
    are ignored. Dead write-only variables are harmless: nothing resolves them.

    Each source line is first split into `:`-separated statements, so a
    reassignment hidden after a colon (`DoStuff : cmd = "evil"`) counts as a
    second definition and makes the variable ambiguous -- decoding a stale value
    would be a wrong-but-confident result, which the fail-closed contract forbids.

    When `position` is given, definitions are scoped to the procedure enclosing
    that position and to assignments that TEXTUALLY PRECEDE it: a `cmd` used by a
    `Shell` in one Sub must not resolve to a `cmd` in another Sub, nor to a
    forward assignment that runs after the use. Without a position (used by the
    helper recogniser and tests) the whole module is scanned.
    """
    if position is not None:
        bounds = _scope_bounds(source, position)
        if bounds is None:
            # not inside any procedure: a real Shell cannot run here, and the
            # module scope would leak another procedure's locals -> resolve
            # nothing, so the site fails closed.
            return {}
        start, end = bounds
        scope = source[start:end]
        limit = position - start  # only assignments before the use in this scope
    else:
        scope = source
        limit = len(source)
    defs: dict[str, list[str]] = {}
    cursor = 0
    for raw in scope.splitlines(keepends=True):
        line_start = cursor
        cursor += len(raw)
        if line_start >= limit:
            break
        for stmt in _split_statements(raw):
            s = stmt.strip()
            m = _ASSIGN.match(s)
            if not m:
                continue
            name, rhs = m.group(1), m.group(2).strip()
            if name.lower() in _KEYWORDS:
                continue
            # an `x = x & ..` accumulator is handled by the helper recogniser,
            # not here; recording it as a def would make x look assigned twice.
            defs.setdefault(name, []).append(rhs)
    return defs


# --- expression evaluator ----------------------------------------------------
_TOKEN = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<str>"(?:[^"]|"")*")
    | (?P<num>&[hH][0-9A-Fa-f]+|\d+)
    | (?P<name>[A-Za-z_]\w*\$?)
    | (?P<op>[()+\-&,])
    """,
    re.VERBOSE,
)

_VB_CONSTS = {"vbunicode": "vbUnicode", "vbfromunicode": "vbFromUnicode"}


class VbaEvaluator:
    """Evaluates a whitelisted VBA expression to a str, int, list[str], or
    UNKNOWN. Resolves bare names through unique reaching definitions and folds
    recognised byte-offset helpers. Fail-closed on everything else."""

    def __init__(
        self,
        defs: dict[str, list[str]],
        helpers: dict[str, ByteOffsetHelper],
    ) -> None:
        self.defs = defs
        self.helpers = helpers
        self.steps = 0

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > MAX_EVAL_STEPS:
            raise VbaRefuse("evaluation step budget exceeded")

    def evaluate(self, text: str, depth: int = 0):
        if depth > MAX_RESOLVE_DEPTH:
            raise VbaRefuse("resolution depth exceeded")
        toks = self._tokenize(_strip_noop_reverse(text))
        parser = _Parser(toks, self, depth)
        value = parser.parse_expr(0)
        if not parser.at_end():
            raise VbaRefuse("trailing tokens after expression")
        return value

    def _tokenize(self, s: str) -> list[tuple[str, str]]:
        toks: list[tuple[str, str]] = []
        i = 0
        n = len(s)
        while i < n:
            m = _TOKEN.match(s, i)
            if not m:
                raise VbaRefuse(f"lexical error at offset {i}")
            i = m.end()
            if m.lastgroup == "ws":
                continue
            toks.append((m.lastgroup, m.group()))
        return toks

    # value helpers
    @staticmethod
    def _as_str(v) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return str(v)
        raise VbaRefuse("cannot coerce to string")

    def resolve_name(self, name: str, depth: int):
        lower = name.lower()
        if lower in _VB_CONSTS:
            return _VB_CONSTS[lower]
        rhss = self.defs.get(name)
        if not rhss:
            raise VbaRefuse(f"unresolved name {name!r}")
        if len(rhss) != 1:
            raise VbaRefuse(f"ambiguous name {name!r} ({len(rhss)} assignments)")
        return self.evaluate(rhss[0], depth + 1)

    def apply_call(self, fn: str, args: list, depth: int):
        self._tick()
        base = fn[:-1] if fn.endswith("$") else fn
        f = base.lower()
        if f in ("chr", "chrb", "chrw"):
            if len(args) != 1:
                raise VbaRefuse(f"{fn} arity")
            n = args[0]
            if n is UNKNOWN:
                return UNKNOWN
            if not isinstance(n, int):
                raise VbaRefuse(f"{fn} non-integer argument")
            if f == "chrw":
                if not (0 <= n <= _CHRW_MAX):
                    raise VbaRefuse(f"{fn} argument {n} out of domain")
                return chr(n)  # ChrW takes a Unicode code point: chr() is exact
            # Chr/ChrB map 0..127 to ASCII exactly, but 128..255 depend on the
            # process ANSI codepage (e.g. cp1252 Chr(128) = U+20AC, not U+0080).
            # Modelling that requires a codepage this evaluator does not carry,
            # so refuse the high range rather than return a wrong character.
            if not (0 <= n <= 0x7F):
                raise VbaRefuse(f"{fn} argument {n} outside modelled ASCII range")
            return chr(n)
        if f == "strreverse":
            if len(args) != 1:
                raise VbaRefuse("StrReverse arity")
            if args[0] is UNKNOWN:
                return UNKNOWN
            return self._as_str(args[0])[::-1]
        if f == "replace":
            if len(args) != 3:
                raise VbaRefuse("Replace arity")
            if any(a is UNKNOWN for a in args):
                return UNKNOWN
            s, find, repl = (self._as_str(a) for a in args)
            if find == "":
                # VBA Replace with an empty Find returns the source unchanged.
                return s
            out = s.replace(find, repl)
            if len(out) > MAX_STRING_CHARS:
                raise VbaRefuse("Replace output too large")
            return out
        if f == "split":
            if len(args) != 2:
                raise VbaRefuse("Split arity")
            if any(a is UNKNOWN for a in args):
                return UNKNOWN
            s, delim = self._as_str(args[0]), self._as_str(args[1])
            if delim == "":
                raise VbaRefuse("Split with empty delimiter")
            parts = s.split(delim)
            if len(parts) > MAX_SPLIT_FIELDS:
                raise VbaRefuse("Split produced too many fields")
            return parts
        # Helpers are keyed by lower-cased name (VBA is case-insensitive), so a
        # `Function Dd` called as `dd(1)` or `DD(1)` resolves.
        if f in self.helpers:
            if len(args) != 1:
                raise VbaRefuse("byte-offset helper arity")
            if args[0] is UNKNOWN:
                return UNKNOWN
            if not isinstance(args[0], int):
                raise VbaRefuse("byte-offset helper non-integer offset")
            decoded = self.helpers[f].decode(args[0])
            if len(decoded) > MAX_STRING_CHARS:
                raise VbaRefuse("helper output too large")
            return decoded
        raise VbaRefuse(f"unsupported function {fn!r}")

    @staticmethod
    def index(seq, idx):
        if seq is UNKNOWN or idx is UNKNOWN:
            return UNKNOWN
        if not isinstance(seq, list):
            raise VbaRefuse("index on non-array")
        if not isinstance(idx, int):
            raise VbaRefuse("non-integer index")
        if idx < 0 or idx >= len(seq):
            raise VbaRefuse(f"index {idx} out of bounds (len {len(seq)})")
        return seq[idx]


class _Parser:
    """Recursive-descent parser/evaluator over a token list. Separated from the
    evaluator so each evaluate() call has its own cursor (reaching-definition
    resolution re-enters evaluate() for a referenced name)."""

    def __init__(self, toks, ev: VbaEvaluator, depth: int) -> None:
        self.toks = toks
        self.pos = 0
        self.ev = ev
        self.depth = depth

    def at_end(self) -> bool:
        return self.pos >= len(self.toks)

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)

    def _next(self):
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def _expect(self, value: str) -> None:
        k, v = self._peek()
        if v != value:
            raise VbaRefuse(f"expected {value!r}, found {v!r}")
        self.pos += 1

    def parse_expr(self, edepth: int):
        if edepth > MAX_EXPR_DEPTH:
            raise VbaRefuse("expression depth exceeded")
        left = self._term(edepth)
        while self._peek()[1] in ("&", "+", "-"):
            op = self._next()[1]
            right = self._term(edepth)
            left = self._binop(op, left, right)
        return left

    def _binop(self, op: str, a, b):
        if a is UNKNOWN or b is UNKNOWN:
            return UNKNOWN
        if op == "&":
            out = self.ev._as_str(a) + self.ev._as_str(b)
            if len(out) > MAX_STRING_CHARS:
                raise VbaRefuse("concatenation too large")
            return out
        if op == "+":
            if isinstance(a, int) and isinstance(b, int):
                return a + b
            if isinstance(a, str) and isinstance(b, str):
                out = a + b
                if len(out) > MAX_STRING_CHARS:
                    raise VbaRefuse("concatenation too large")
                return out
            raise VbaRefuse("ambiguous '+' operands (mixed int/string)")
        if op == "-":
            if isinstance(a, int) and isinstance(b, int):
                return a - b
            raise VbaRefuse("'-' on non-integer operands")
        raise VbaRefuse(f"unknown operator {op!r}")

    def _term(self, edepth: int):
        k, v = self._peek()
        if v in ("+", "-"):
            self._next()
            val = self._term(edepth)
            if val is UNKNOWN:
                return UNKNOWN
            if not isinstance(val, int):
                raise VbaRefuse("unary sign on non-integer")
            return -val if v == "-" else val
        return self._postfix(self._primary(edepth), edepth)

    def _postfix(self, value, edepth: int):
        # array indexing: <expr>(i) -- e.g. Split(...)(2)
        while self._peek()[1] == "(":
            args = self._arglist(edepth)
            if len(args) != 1:
                raise VbaRefuse("index arity")
            value = self.ev.index(value, args[0])
        return value

    def _primary(self, edepth: int):
        self.ev._tick()
        k, v = self._peek()
        if v == "(":
            self._next()
            val = self.parse_expr(edepth + 1)
            self._expect(")")
            return val
        if k == "str":
            self._next()
            return v[1:-1].replace('""', '"')
        if k == "num":
            self._next()
            digits = v[2:] if v[0] == "&" else v
            if len(digits) > MAX_INT_LITERAL_DIGITS:
                raise VbaRefuse("integer literal too long")
            return int(digits, 16) if v[0] == "&" else int(v)
        if k == "name":
            name = self._next()[1]
            if self._peek()[1] == "(":
                args = self._arglist(edepth)
                return self.ev.apply_call(name, args, self.depth)
            return self.ev.resolve_name(name, self.depth)
        raise VbaRefuse(f"unexpected token {v!r}")

    def _arglist(self, edepth: int) -> list:
        self._expect("(")
        args: list = []
        if self._peek()[1] == ")":
            self._next()
            return args
        while True:
            args.append(self.parse_expr(edepth + 1))
            k, v = self._peek()
            if v == ",":
                self._next()
                continue
            if v == ")":
                self._next()
                return args
            raise VbaRefuse("malformed argument list")
