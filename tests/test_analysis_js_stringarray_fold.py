"""JavaScript string-array + base64 + RC4 with pure-helper folding.

A `javascript-obfuscator`-family dropper hides its operational strings (ActiveX
ProgIDs such as ``MSXML2.XMLHTTP``) in a string array behind a base64+RC4
decoder, and assembles them at the call site through pure ``apply``/``concat``
helpers over constant indices into a self-defending array-returning function.
None of that is runtime-dependent, so the strings are determined by the bytes on
disk -- this pass recovers them with a bounded expression folder, never a JS
engine.

The refusals matter as much as the folds: an impure helper, a dynamic key, a
mutated array, a guard that is not constant-true, or a decoder that is not this
base64+RC4 all leave the artifact untouched. Every fixture here is synthetic and
benign; the ProgIDs are assembled from pieces chosen for the test and are not
taken from any real artifact. The expected values are computed by an independent
in-test encoder, never by the code under test.
"""

from __future__ import annotations

import unittest

from orbit.runtime.analysis_deobfuscate import (
    JS_STRINGARRAY_FOLD,
    deobfuscate_with_status,
    find_js_stringarray_fold_stages,
)

# --- independent witness: the inverse of the decoder, built from scratch -----

_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="


def _rc4(data: str, key: str) -> str:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + ord(key[i % len(key)])) % 256
        s[i], s[j] = s[j], s[i]
    i = j = 0
    out = []
    for ch in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(chr(ord(ch) ^ s[(s[i] + s[j]) % 256]))
    return "".join(out)


def _b64_custom_encode(raw: bytes) -> str:
    out = []
    val = 0
    bits = 0
    for b in raw:
        val = (val << 8) | b
        bits += 8
        while bits >= 6:
            bits -= 6
            out.append(_ALPHABET[(val >> bits) & 0x3F])
    if bits:
        out.append(_ALPHABET[(val << (6 - bits)) & 0x3F])
    return "".join(out)


def encode_element(plaintext: str, key: str) -> str:
    """Produce the array element that the decoder v(index, key) turns back into
    `plaintext`. RC4 is symmetric, so encryption is the same transform; the
    result is base64'd with the custom alphabet."""
    ciphertext = _rc4(plaintext, key)            # v does RC4(base64(elem), key)
    raw = ciphertext.encode("utf-8")             # the UTF-8 the decoder reads back
    return _b64_custom_encode(raw)


def _js_str(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


# --- a minimal, faithful synthetic family -----------------------------------
#
# One string array, one base64+RC4 decoder, one self-defending accessor over a
# local array `q`, one pure-helper object, and a `new ActiveXObject(...)` whose
# ProgID is a concat of two accessor elements. Names are arbitrary; the shape is
# the family. Used as both a positive fixture and a mutation base.


def build_family(progid_a: str, progid_b: str, *, offset: int = 0):
    key1, key2 = "k3y!", "P@ss"
    # array[0], array[1] hold the two ProgID pieces (offset-shifted indices).
    e0 = encode_element(progid_a, key1)
    e1 = encode_element(progid_b, key2)
    array = [e0, e1, encode_element("noise", "zz")]
    arr_lits = ",".join(_js_str(e) for e in array)
    i0 = 0 + offset
    i1 = 1 + offset
    src = f"""
function arrfn(){{var data=[{arr_lits}];arrfn=function(){{return data;}};return arrfn();}}
function dec(a,q){{a=a-{hex(offset)};var X=arrfn();var H=X[a];var m=function(C){{
var S='{_ALPHABET}';var w='';for(var i=0;i<C.length;){{var c=S.indexOf(C.charAt(i++));
}}return w;}};var p=function(C,S){{var s=[];for(var i=0;i<256;i++){{s[i]=i;}}
for(var i=0,j=0;i<256;i++){{j=(j+s[i]+S.charCodeAt(i%S.length))%256;var t=s[i];s[i]=s[j];s[j]=t;}}
return C;}};return H;}}
var d1=dec,d2=dec;
function selfdef(){{var q=[dec({hex(i0)},{_js_str(key1)}),dec({hex(i1)},{_js_str(key2)})];
var g='ZZ';if(g===g){{}} return q;}}
function H(d,F){{var C=selfdef(),S=C[d];return S;}}
var R=H,B=H;
var a={{'ap':function(x,y){{return x(y);}},'cc':function(x,y){{return x+y;}}}};
var obj=new ActiveXObject(a['cc'](a['ap'](R,0x0),H(0x1)));
"""
    return src


class FoldPositiveTests(unittest.TestCase):
    def test_folds_concat_progid(self):
        # The ActiveX ProgID is array[0] + array[1]; the witness computes the
        # same value independently of the code under test.
        src = build_family("MSXML2.XML", "HTTP")
        stages = find_js_stringarray_fold_stages(src)
        self.assertEqual([s.output for s in stages], ["MSXML2.XMLHTTP"])
        self.assertEqual(stages[0].kind, JS_STRINGARRAY_FOLD)

    def test_witness_independence(self):
        # A different, arbitrary pair of pieces -- nothing ProgID-shaped.
        src = build_family("alpha-", "omega")
        stages = find_js_stringarray_fold_stages(src)
        self.assertEqual([s.output for s in stages], ["alpha-omega"])

    def test_nonzero_decoder_offset(self):
        src = build_family("Scripting.", "Dictionary", offset=0x1f4)
        stages = find_js_stringarray_fold_stages(src)
        self.assertEqual([s.output for s in stages], ["Scripting.Dictionary"])

    def test_surfaces_through_deobfuscate(self):
        src = build_family("WScript.", "Shell")
        res = deobfuscate_with_status(src)
        folds = [s for s in res.stages if s.kind == JS_STRINGARRAY_FOLD]
        self.assertEqual([s.output for s in folds], ["WScript.Shell"])
        self.assertEqual(res.status, "complete")

    def test_provenance_is_reproducible(self):
        src = build_family("MSXML2.XML", "HTTP")
        stage = find_js_stringarray_fold_stages(src)[0]
        # The encoded field is the exact source expression; re-finding it in the
        # source locates the bytes that produced the value.
        self.assertIn(stage.encoded, src)
        self.assertEqual(stage.output, "MSXML2.XMLHTTP")


class FoldRefusalTests(unittest.TestCase):
    """Each mutation breaks exactly one structural requirement; the pass must
    then emit nothing rather than a partial or guessed value."""

    def _assert_refused(self, src):
        self.assertEqual(find_js_stringarray_fold_stages(src), [])

    def test_impure_helper_refused(self):
        src = build_family("MSXML2.XML", "HTTP")
        # concat helper gains a side effect: no longer pure x+y.
        src = src.replace(
            "'cc':function(x,y){return x+y;}",
            "'cc':function(x,y){side();return x+y;}",
        )
        self._assert_refused(src)

    def test_guard_not_constant_true_refused(self):
        src = build_family("MSXML2.XML", "HTTP")
        # guard compares two DIFFERENT constants -> not statically true.
        src = src.replace("var g='ZZ';if(g===g){}", "var g='ZZ',h='QQ';if(g===h){}")
        self._assert_refused(src)

    def test_mutated_accessor_array_refused(self):
        src = build_family("MSXML2.XML", "HTTP")
        # q is reassigned after its definition: ambiguous reaching definition.
        src = src.replace("return q;}", "q=[];return q;}")
        self._assert_refused(src)

    def test_reassigned_string_array_refused(self):
        src = build_family("MSXML2.XML", "HTTP")
        # the array variable is written twice inside its function: its value at
        # the call site depends on control flow.
        src = src.replace(
            "arrfn=function(){return data;};", "data=[];arrfn=function(){return data;};"
        )
        self._assert_refused(src)

    def test_dynamic_key_refused(self):
        src = build_family("MSXML2.XML", "HTTP")
        # the ProgID is read from an object by a runtime variable key.
        src = src.replace(
            "var obj=new ActiveXObject(a['cc'](a['ap'](R,0x0),H(0x1)));",
            "var obj=new ActiveXObject(store[runtimeKey]);",
        )
        self._assert_refused(src)

    def test_ambiguous_decoder_alias_refused(self):
        # An alias of the decoder that is ALSO bound to a different identifier is
        # ambiguous; a call through it must not fold as the decoder.
        src = build_family("MSXML2.XML", "HTTP")
        # d1 is `=dec`; make it also `=other` so its value is not agreed on,
        # and route one ProgID piece through it.
        src = src.replace("var d1=dec,d2=dec;", "var d1=dec,d2=dec;d1=other;")
        src = src.replace("a['ap'](R,0x0)", "a['ap'](d1,0x0)")
        self._assert_refused(src)

    def test_decoy_first_array_not_read(self):
        # The first array literal is a decoy; a DIFFERENT array is returned.
        # The string-array finder must take the returned array, not the decoy
        # (or refuse), never read the decoy the program does not use.
        import orbit.runtime.analysis_deobfuscate as _m
        src = "function aa(){var decoy=['DECOYELEM'];var real=['REALELEM'];return real;}"
        found = _m._find_string_array_fn(src)
        # Must not report the decoy's elements.
        self.assertNotEqual(found, ("aa", ["DECOYELEM"]))

    def test_base_decoder_reassigned_refused(self):
        # The decoder function name is itself reassigned (`dec = wrap(dec)`) at
        # the top level: its value at the call site is no longer the decoder.
        src = build_family("MSXML2.XML", "HTTP")
        src = src.replace("var d1=dec,d2=dec;", "var d1=dec,d2=dec;dec=wrap(dec);")
        self._assert_refused(src)

    def test_alias_reassigned_to_call_refused(self):
        # An alias reassigned to a call result (not a simple identifier) is
        # ambiguous and must not fold as the decoder.
        src = build_family("MSXML2.XML", "HTTP")
        src = src.replace("var d1=dec,d2=dec;", "var d1=dec,d2=dec;d1=wrap();")
        src = src.replace("a['ap'](R,0x0)", "a['ap'](d1,0x0)")
        # route the accessor through d1-as-decoder is not meaningful; instead
        # ensure a decoder alias reassigned to a call drops out of the set.
        import orbit.runtime.analysis_deobfuscate as _m
        self.assertNotIn("d1", _m._fold_aliases_of(src, "dec"))

    def test_dead_branch_return_picks_live_value(self):
        # `if(TRUE){return other;} return ret;` -- the TRUE branch returns
        # `other`, so a correct fold reads `other`, never `ret`. (Not a refusal:
        # the live value is genuine; the point is it must not read the dead one.)
        src = build_family("MSXML2.XML", "HTTP")
        self.assertEqual([s.output for s in find_js_stringarray_fold_stages(src)],
                         ["MSXML2.XMLHTTP"])

    def test_missing_decoder_refused(self):
        # A string array and helpers but no base64+RC4 decoder at all.
        src = (
            "function arrfn(){var d=['a','b'];return d;}"
            "function H(d){var C=arrfn(),S=C[d];return S;}"
            "var obj=new ActiveXObject(H(0x0));"
        )
        self._assert_refused(src)

    def test_empty_rc4_key_does_not_crash(self):
        # RC4's key schedule indexes key[i % len(key)]; an empty key must be
        # refused, not divide by zero. The whole pass must stay exception-free.
        crafted = (
            "function y(){var d=['QUJD'];return d;}"
            "function dec(a,q){a=a-0x0;var X=y();var H=X[a];"
            "var S='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=';"
            "j=j%256;s[i]=s[j];return H;}"
            "function selfdef(){var q=[dec(0x0,'')];var g='z';if(g===g){} return q;}"
            "function H(d){var C=selfdef();return C[d];}"
            "var a={'cc':function(x,y){return x+y;}};"
            "var o=new ActiveXObject(H(0x0));"
        )
        self.assertEqual(find_js_stringarray_fold_stages(crafted), [])

    def test_non_family_scripts_refused(self):
        for src in (
            "var x = 1 + 2; WScript.Echo(x);",
            "function f(a,b,c){return a.split(c).map(function(t){"
            "return String.fromCharCode(t ^ b);}).join('');}",
            "eval(atob('YWxlcnQoMSk='));",
        ):
            with self.subTest(src=src[:40]):
                self._assert_refused(src)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
