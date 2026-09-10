"""JavaScript String.fromCharCode over constant arithmetic.

A JScript dropper builds a script character by character with
`String.fromCharCode(<int expr>, ...)` over a constant offset, assigns it, and
runs it with eval. This pass computes the decoded string deterministically --
literals and integer arithmetic only, nothing executed -- so a model is handed
the value instead of adding hundreds of operands by hand.

Every expected value here is computed by independent in-test logic (`_fcc`),
never by the code under test. The refusals matter as much as the decodes: an
argument outside the small whitelist, or a constant whose value is not
statically unique, fails the whole call closed.
"""

from __future__ import annotations

import hashlib
import unittest

from orbit.runtime.analysis_deobfuscate import (
    JS_FROMCHARCODE_OFFSET,
    deobfuscate_with_status,
    find_js_fromcharcode_stages,
)


def _fcc(values):
    """Independent JS String.fromCharCode: each value is a UTF-16 code unit,
    ToUint16 (mod 2**16), including negatives."""
    return "".join(chr(v & 0xFFFF) for v in values)


def _decode(src):
    stages = find_js_fromcharcode_stages(src)
    return [s.output for s in stages]


class FromCharCodeDecodeTests(unittest.TestCase):
    # F1
    def test_single_offset(self):
        self.assertEqual(_decode("var K=65;String.fromCharCode(130-K)"), [_fcc([65])])

    # F2
    def test_multiple_arguments(self):
        self.assertEqual(
            _decode("var K=65;String.fromCharCode(130-K,131-K,132-K)"),
            [_fcc([65, 66, 67])],
        )

    # F3
    def test_different_constant_name(self):
        self.assertEqual(
            _decode("var offset=100;String.fromCharCode(165-offset,166-offset)"),
            [_fcc([65, 66])],
        )

    # F4
    def test_constant_defined_before_call(self):
        self.assertEqual(
            _decode("var K=64;var x=String.fromCharCode(K+1,K+2)"), [_fcc([65, 66])]
        )

    # F5
    def test_parenthesized_arithmetic(self):
        self.assertEqual(
            _decode("var K=10;String.fromCharCode((75-K),(76-K))"), [_fcc([65, 66])]
        )

    def test_nested_parentheses(self):
        self.assertEqual(
            _decode("var K=10;String.fromCharCode(((75)-K),(76-(K)))"),
            [_fcc([65, 66])],
        )

    # F6
    def test_hex_literals(self):
        self.assertEqual(
            _decode("var K=0x10;String.fromCharCode(0x51-K,0x52-K)"),
            [_fcc([0x51 - 0x10, 0x52 - 0x10])],
        )

    # F7
    def test_negative_and_intermediate_arithmetic(self):
        self.assertEqual(
            _decode("var K=200;String.fromCharCode(K-135,K-134)"), [_fcc([65, 66])]
        )

    def test_unary_minus_in_argument(self):
        # Unary minus is supported inside an argument expression over a positive
        # constant: -K + 130 == 130 - 65 == 65.
        self.assertEqual(
            _decode("var K=65;String.fromCharCode(130+-K)"), [_fcc([65])]
        )

    # F8: stress -- 600 operands over a constant offset.
    def test_large_argument_list(self):
        K = 1000
        vals = [(K + (i % 90) + 33) for i in range(600)]
        args = ",".join(f"{v}-K" for v in vals)
        src = f"var K={K};String.fromCharCode({args})"
        self.assertEqual(_decode(src), [_fcc([v - K for v in vals])])

    # F13
    def test_multiple_fromcharcode_calls(self):
        src = "var K=64;String.fromCharCode(K+1);String.fromCharCode(K+2,K+3)"
        self.assertEqual(_decode(src), [_fcc([65]), _fcc([66, 67])])

    # F14 + F9(execution relationship): assignment target and eval consumer.
    def test_assignment_target_and_eval_relationship(self):
        src = "var K=64;var payload=String.fromCharCode(K+1,K+2);eval(payload);"
        stages = find_js_fromcharcode_stages(src)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].delimiter, "payload|eval")

    def test_assignment_target_without_eval(self):
        src = "var K=64;var payload=String.fromCharCode(K+1);run(payload);"
        stages = find_js_fromcharcode_stages(src)
        self.assertEqual(stages[0].delimiter, "payload")  # no |eval

    def test_eval_in_comment_gives_no_relationship(self):
        # A dead eval (inside a comment) must not create the execution marker.
        src = "var K=64;var p=String.fromCharCode(K+1);/* eval(p) */"
        stages = find_js_fromcharcode_stages(src)
        self.assertEqual(stages[0].delimiter, "p")  # target only, no |eval

    def test_live_concatenated_call_decodes(self):
        # A fromCharCode concatenated into an expression is live code, decoded.
        self.assertEqual(
            _decode("var d='X'+String.fromCharCode(104,105)+'Y'"), [_fcc([104, 105])]
        )

    # F15
    def test_randomized_identifiers(self):
        src = "var qZ9=300;var Wv=String.fromCharCode(qZ9-235,qZ9-234);eval(Wv)"
        stages = find_js_fromcharcode_stages(src)
        self.assertEqual([s.output for s in stages], [_fcc([65, 66])])
        self.assertEqual(stages[0].delimiter, "Wv|eval")


class FromCharCodeRefusalTests(unittest.TestCase):
    def _assert_refused(self, src):
        self.assertEqual(find_js_fromcharcode_stages(src), [])

    # F9
    def test_unknown_constant(self):
        self._assert_refused("String.fromCharCode(165-K)")

    # F10
    def test_reassigned_constant(self):
        self._assert_refused("var K=100;K=101;String.fromCharCode(165-K)")

    def test_conditionally_reassigned_constant(self):
        self._assert_refused(
            "var K=100;if(x){K=101;}String.fromCharCode(165-K)"
        )

    # F11: a shadow is a second write to the name -> ambiguous -> refuse.
    def test_shadowed_constant(self):
        self._assert_refused(
            "var K=100;function f(){var K=200;}String.fromCharCode(165-K)"
        )

    def test_negative_literal_constant_refused(self):
        # A negative literal constant assignment (K=-65) is outside the
        # single-positive-literal constant grammar; refuse rather than guess.
        self._assert_refused("var K=-65;String.fromCharCode(-K)")

    # F12
    def test_unsupported_operator(self):
        self._assert_refused("var K=2;String.fromCharCode(130*K)")

    def test_unsupported_shift(self):
        self._assert_refused("var K=2;String.fromCharCode(65<<K)")

    def test_unsupported_property_arg(self):
        self._assert_refused("var K=2;String.fromCharCode(a.b-K)")

    def test_unsupported_call_arg(self):
        self._assert_refused("String.fromCharCode(getCode(0))")

    # F16
    def test_near_match_dynamic(self):
        self._assert_refused("String.fromCharCode(runtimeVar)")
        self._assert_refused("String.fromCharCode(arr[i])")

    # F17
    def test_malformed_syntax(self):
        self._assert_refused("var K=64;String.fromCharCode(K+")
        self._assert_refused("var K=64;String.fromCharCode(,)")
        self._assert_refused("var K=64;String.fromCharCode()")

    def test_empty_argument(self):
        self._assert_refused("var K=64;String.fromCharCode(K+1,,K+2)")

    # Review BLOCKER: runtime rebinding forms a plain-assignment check misses.
    def test_increment_decrement_invalidate_constant(self):
        for mut in ("K++", "++K", "K--", "--K"):
            with self.subTest(mut=mut):
                self._assert_refused(f"var K=64;{mut};String.fromCharCode(K+1)")

    def test_for_of_binding_invalidates_constant(self):
        self._assert_refused(
            "var K=104;for(K of [0]){}var p=String.fromCharCode(K+72);eval(p)"
        )

    def test_for_in_binding_invalidates_constant(self):
        self._assert_refused("var K=64;for(K in obj){}String.fromCharCode(K+1)")

    def test_array_destructuring_invalidates_constant(self):
        self._assert_refused("var K=64;[K]=[200];String.fromCharCode(K+1)")

    def test_object_destructuring_invalidates_constant(self):
        self._assert_refused("var K=64;({K}=obj);String.fromCharCode(K+1)")

    def test_unrelated_increment_does_not_invalidate(self):
        # A different variable's ++ must not invalidate a genuine constant.
        self.assertEqual(
            _decode("var K=64;var i=0;i++;String.fromCharCode(K+1)"), [_fcc([65])]
        )

    # Review MAJOR: fromCharCode inside a string literal or comment is dead text.
    def test_fromcharcode_inside_string_literal_ignored(self):
        self._assert_refused('var s="String.fromCharCode(104,105)";')
        self._assert_refused("var s='x'+'String.fromCharCode(104,105)';")

    def test_fromcharcode_inside_comment_ignored(self):
        self._assert_refused("// String.fromCharCode(104,105)\nvar x=1;")
        self._assert_refused("/* String.fromCharCode(104,105) */ var x=1;")

    def test_non_family_scripts(self):
        for src in (
            "var x = 1 + 2; alert(x);",
            "document.write('hello');",
            "String.prototype.foo = 1;",
        ):
            with self.subTest(src=src[:30]):
                self._assert_refused(src)


class FromCharCodeProvenanceAndBoundsTests(unittest.TestCase):
    def test_provenance_shas_are_exact(self):
        src = "var K=64;var p=String.fromCharCode(K+1,K+2);eval(p)"
        stages = find_js_fromcharcode_stages(src)
        self.assertEqual(len(stages), 1)
        st = stages[0]
        # output SHA is the exact SHA of the decoded output (M10).
        self.assertEqual(
            st.output_sha256, hashlib.sha256(st.output.encode()).hexdigest()
        )
        self.assertTrue(st.output_sha256)
        self.assertTrue(st.input_sha256)
        self.assertEqual(st.output, _fcc([65, 66]))
        # encoded provenance points back at the call.
        self.assertIn("fromCharCode", st.encoded)

    def test_argument_bound_refuses(self):
        # A call whose argument text stays under MAX_INPUT_CHARS but exceeds the
        # argument-count bound is refused (M12). Uses one-char args to keep the
        # inner text small while the count is large.
        from orbit.runtime.analysis_deobfuscate import _FCC_MAX_ARGS

        n = _FCC_MAX_ARGS + 1
        inner = ",".join(["0"] * n)  # ~2*n chars; keep under MAX_INPUT_CHARS
        if len(inner) > 262_144:
            self.skipTest("bound too large to exercise within input cap")
        src = f"String.fromCharCode({inner})"
        self.assertEqual(find_js_fromcharcode_stages(src), [])


class FromCharCodeSemanticsTests(unittest.TestCase):
    """Boundary vectors, checked against an independent oracle (_fcc)."""

    def test_boundary_values(self):
        for value in (0, 1, 65, 255, 256, 0x7FFF, 0xFFFF):
            with self.subTest(value=value):
                src = f"String.fromCharCode({value})"
                self.assertEqual(_decode(src), [_fcc([value])])

    def test_wraps_above_16_bits(self):
        # 65601 -> ToUint16 -> 65 ('A'); the source states it as a constant expr.
        self.assertEqual(_decode("var K=0;String.fromCharCode(65601-K)"), [_fcc([65601])])

    def test_wraps_negative(self):
        # -1 -> ToUint16 -> 0xFFFF.
        self.assertEqual(_decode("var K=66;String.fromCharCode(65-K)"), [_fcc([-1])])


class FromCharCodeIbanWitnessTests(unittest.TestCase):
    """The frozen IBAN case, decoded through the production pass and checked
    against the independently-established oracle SHA (a validation expectation,
    not a recognition rule)."""

    ORACLE_SHA = "5d51e7659955a754d55a83bce9157d8999864ee30a4f3cd5dc752ed0191a7de0"

    def test_iban_decodes_exactly(self):
        src = open("workdir/samples/IBAN.js", encoding="utf-8").read()
        stages = [
            s for s in deobfuscate_with_status(src).stages
            if s.kind == JS_FROMCHARCODE_OFFSET
        ]
        self.assertEqual(len(stages), 1)
        stage = stages[0]
        self.assertEqual(len(stage.output), 564)
        self.assertEqual(
            hashlib.sha256(stage.output.encode()).hexdigest(), self.ORACLE_SHA
        )
        self.assertEqual(stage.delimiter, "MMGCLZ|eval")
        # Oracle check (not recognition): the decoded chain is present.
        for token in (
            "MSXML2.XMLHTTP",
            "https://productoslili.cl/cv/cr2.exe",
            "Scripting.FileSystemObject",
            "ADODB.Stream",
            "WScript.Shell",
        ):
            self.assertIn(token, stage.output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
