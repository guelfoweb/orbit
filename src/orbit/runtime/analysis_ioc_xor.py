"""Exact pure split/XOR helper shapes used only after closed syntax validation."""
import re
from orbit.runtime.analysis_deobfuscate import _balanced, _fold_parse, _split_top, _decode_tokens

_ID = r"[A-Za-z_$][A-Za-z0-9_$]*"
def _literal(expr):
    if "\\" in expr:
        return None  # escape semantics are outside this producer's proof
    try:
        node = _fold_parse(expr.strip())
    except (ValueError, RecursionError):
        return None
    if not node or node[0] not in ('str', 'num'):
        return None
    if node[0] == 'str' and any(c in node[1] for c in '\r\n'):
        return None
    return node[1]


def _integer(expr, depth=0):
    if depth > 32 or len(expr) > 128:
        return None
    expr = expr.strip()
    while expr.startswith("("):
        part = _balanced(expr, 0, "(", ")")
        if part is None or part[2] != len(expr) - 1:
            break
        expr = expr[1:-1].strip()
    if expr.startswith("+"):
        return _integer(expr[1:], depth + 1)
    value = _literal(expr)
    if isinstance(value, int):
        return value if 0 <= value <= 0xFFFF else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,5}", value):
        return int(value) if int(value) <= 0xFFFF else None
    return None


def _provider(params, body):
    if len(params) != 1:
        return None
    m = re.fullmatch(r"\s*return\s*\{(.*)\}\s*;?\s*", body, re.S)
    if not m:
        return None
    roles = {}
    names = set()
    for part in _split_top(m[1]):
        prop = re.fullmatch(rf"\s*({_ID})\s*:\s*(.*?)\s*", part, re.S)
        if not prop or prop[1] in names:
            return None
        names.add(prop[1])
        split = re.fullmatch(
            rf"{re.escape(params[0])}\s*\.\s*split\s*\((.*?)\)\s*(\.\s*length)?",
            prop[2], re.S)
        if split and isinstance(_literal(split[1]), str):
            role = "count" if split[2] else "values"
            item = (prop[1], _literal(split[1]))
        else:
            role, item = "key", (prop[1], _integer(prop[2]))
        if role in roles or item[1] is None:
            return None
        roles[role] = item
    if (set(roles) != {"values", "count", "key"}
            or roles["values"][1] != roles["count"][1] or not roles["values"][1]):
        return None
    return roles


def _loop(params, body, roles):
    if len(params) != 3 or 'String' in params:
        return False
    fn, data, out = map(re.escape, params)
    call = rf"{fn}\s*\(\s*{data}\s*\)\s*\.\s*"
    count, values, key = (re.escape(roles[k][0]) for k in ("count", "values", "key"))
    # Match the ENTIRE body and all data dependencies, not proximity of three
    # API names. An extra statement, altered bound, stride or return refuses it.
    match = re.fullmatch(
        rf"\s*var\s+(?P<i>{_ID})\s*=\s*0\s*;\s*"
        rf"while\s*\(\s*(?P=i)\s*<\s*{call}{count}\s*\)\s*\{{\s*"
        rf"{out}\s*\+=\s*String\s*\.\s*fromCharCode\s*\(\s*"
        rf"\(\s*\+\s*{call}{values}\s*\[\s*(?P=i)\s*\]\s*\)\s*"
        rf"\^\s*{call}{key}\s*\)\s*;\s*(?P=i)\s*\+\+\s*;?\s*\}}\s*"
        rf"return\s+{out}\s*;\s*", body, re.S)
    return match is not None and match['i'] not in (*params, 'String')



def decode(data, roles):
    from orbit.runtime.analysis_ioc_proof import Unproved
    parts = data.split(roles['values'][1])
    if any(not re.fullmatch(r'[0-9]{0,10}', p.strip(' \t\r\n'))
           or int(p.strip(' \t\r\n') or '0') > 0xFFFFFFFF for p in parts):
        raise Unproved('numeric decoder input is not exact')
    value = _decode_tokens(data, roles['key'][1], roles['values'][1])
    if value is None or len(value) > 2048 or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise Unproved('decoded value outside supported bound')
    return value

provider = _provider
loop = _loop
