"""VBScript/PowerShell Chr-offset deterministic decoder (V1-V15).

Proves the generic `array -> Chr(element - constant) -> concat` transform on
synthetic fixtures that do not use any mine.hta string, constant, name or SHA.
Expected decoded bytes are built INDEPENDENTLY by each test (a plain Python
char-offset over literal values), never by calling the production evaluator as
its own oracle (§14).
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import hashlib as _hashlib

from orbit.runtime.analysis_deobfuscate import (  # noqa: E402
    POWERSHELL_CHR,
    VBSCRIPT_CHR,
    deobfuscate,
    deobfuscate_with_status,
    find_vbscript_chr_stages,
    find_powershell_chr_stages,
)
from orbit.runtime.analysis_runtime import (  # noqa: E402
    DETERMINISTIC_ONLY_REPORT,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402


class _NullBackend:
    """A backend that would answer but is never called: the deterministic-only
    closing report spends no model call, which this proves by exploding if one
    is attempted."""
    thinking = False

    def supports_exact_context_admission(self):
        return True

    def chat_stream(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("no model call expected for a deterministic-only report")

    chat = chat_stream


def witness(values, constant):
    """Independent expected output: the same math, written plainly, not via the
    production decoder."""
    return "".join(chr(v - constant) for v in values)


def encode(text, constant):
    """Build a literal numeric array that decodes to `text` under `constant`."""
    return ", ".join(str(ord(c) + constant) for c in text)


def vbs_decoder(const_name="k", const=100, fn="dec", loop="e", arg="a"):
    """A generic VBScript Chr-offset decoder body (renamable)."""
    return (
        f"Function {fn}(ByVal {arg})\n"
        f"    Dim {const_name}\n"
        f"    {const_name} = {const}\n"
        f"    For Each {loop} In {arg}\n"
        f"        s = s & Chr({loop} - {const_name})\n"
        f"    Next\n"
        f"    {fn} = s\n"
        f"End Function\n"
    )


class VbsChrTests(unittest.TestCase):

    def _only(self, stages, kind=VBSCRIPT_CHR):
        return [s for s in stages if s.kind == kind]

    def test_v1_single_literal_chr(self) -> None:
        """Chr via a 3+ element literal array (the smallest real decoder input)."""
        expected = witness([65 + 100, 66 + 100, 67 + 100], 100)  # "ABC"
        src = vbs_decoder(const=100) + f"x = dec(Array({encode('ABC',100)}))\n"
        stages = self._only(find_vbscript_chr_stages(src))
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].output, "ABC")
        self.assertEqual(stages[0].output, expected)
        self.assertEqual(stages[0].key, 100)

    def test_v2_multiple_arrays_each_decoded(self) -> None:
        src = (vbs_decoder(const=50)
               + f"a = dec(Array({encode('hello',50)}))\n"
               + f"b = dec(Array({encode('world',50)}))\n")
        outs = sorted(s.output for s in self._only(find_vbscript_chr_stages(src)))
        self.assertEqual(outs, ["hello", "world"])

    def test_v3_array_loop_arithmetic(self) -> None:
        payload = "Wscript.Shell"  # generic API name, not a mine.hta constant
        src = vbs_decoder(const=29450) + f"o = dec(Array({encode(payload,29450)}))\n"
        stages = self._only(find_vbscript_chr_stages(src))
        self.assertEqual(stages[0].output, payload)
        self.assertEqual(stages[0].output, witness([ord(c)+29450 for c in payload], 29450))

    def test_v4_renamed_variables_equivalent(self) -> None:
        """Different identifiers, same construct -> same decode (no name dep)."""
        src = (vbs_decoder(const_name="zz9", const=7, fn="qWeRt", loop="ii", arg="buf")
               + f"r = qWeRt(Array({encode('payload',7)}))\n")
        stages = self._only(find_vbscript_chr_stages(src))
        self.assertEqual(stages[0].output, "payload")

    def test_v5_different_constant(self) -> None:
        src = vbs_decoder(const=1234) + f"x = dec(Array({encode('ABCDE',1234)}))\n"
        stages = self._only(find_vbscript_chr_stages(src))
        self.assertEqual(stages[0].output, "ABCDE")
        self.assertEqual(stages[0].key, 1234)

    def test_v6_two_sequential_layers(self) -> None:
        """VBScript layer decodes to text carrying a PowerShell [char] layer."""
        inner = "xABCx"
        inner_arr = ", ".join(str(ord(c) + 500) for c in inner)
        # The array is PASSED to the decoder function -- the only safe linkage.
        ps_text = (
            "function dd($z){$h=500;foreach($p in $z){$o+=[char]($p - $h)}}\n"
            f"x = dd @({inner_arr})\n"
        )
        # VBScript layer encodes the whole ps_text with constant 300.
        src = vbs_decoder(const=300) + f"c = dec(Array({encode(ps_text,300)}))\n"
        res = deobfuscate_with_status(src)
        kinds = [s.kind for s in res.stages]
        self.assertIn(VBSCRIPT_CHR, kinds)
        self.assertIn(POWERSHELL_CHR, kinds)
        outs = [s.output for s in res.stages]
        self.assertTrue(any(o == inner for o in outs), outs)

    def test_v7_decoy_array_unused_is_not_a_stage(self) -> None:
        """A short literal array not consumed by the decoder is not decoded.

        The decoder subtracts 100; a stray small Array of low integers would
        decode to control characters -- the 3-element floor and the
        character-range check keep it from being recorded as a finding."""
        src = (vbs_decoder(const=100)
               + f"good = dec(Array({encode('REALPAYLOAD',100)}))\n"
               + "decoy = Array(1, 2)\n")  # 2 elements, below floor
        outs = [s.output for s in self._only(find_vbscript_chr_stages(src))]
        self.assertEqual(outs, ["REALPAYLOAD"])

    def test_v7b_short_inrange_decoy_below_floor_not_a_stage(self) -> None:
        """A 2-element array whose values ARE in range must still be refused by
        the element-count floor -- this is the gate the range-check cannot cover
        (the decoy here decodes cleanly to two printable chars)."""
        # encode('Ab', 100) -> in-range 2-element array
        src = (vbs_decoder(const=100)
               + f"good = dec(Array({encode('REALPAYLOAD',100)}))\n"
               + f"decoy = Array({encode('Ab',100)})\n")  # 2 elems, in range
        outs = sorted(s.output for s in self._only(find_vbscript_chr_stages(src)))
        self.assertEqual(outs, ["REALPAYLOAD"])  # decoy "Ab" must NOT appear

    def test_v8_unknown_constant_refuses(self) -> None:
        """The subtraction operand is a name never assigned a literal -> refuse."""
        src = (
            "Function dec(a)\n"
            "    For Each e In a\n"
            "        s = s & Chr(e - unknownConst)\n"  # never assigned
            "    Next\n"
            "End Function\n"
            f"x = dec(Array({encode('ABC',100)}))\n"
        )
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_v9_reassigned_constant_is_ambiguous_refuses(self) -> None:
        """A constant assigned twice is not proven at the decode site -> refuse."""
        src = (
            "Function dec(a)\n"
            "    k = 100\n"
            "    k = 200\n"
            "    For Each e In a\n"
            "        s = s & Chr(e - k)\n"
            "    Next\n"
            "End Function\n"
            f"x = dec(Array({encode('ABC',100)}))\n"
        )
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_v9b_two_distinct_constants_is_ambiguous_refuses(self) -> None:
        """Two decoders subtracting DIFFERENT constants: which applies to the
        literal arrays is not determined by the text, so all are refused (the
        same discipline the PowerShell XOR pass applies to multiple keys)."""
        src = (
            vbs_decoder(const_name="k1", const=100, fn="d1", loop="e1", arg="a1")
            + vbs_decoder(const_name="k2", const=200, fn="d2", loop="e2", arg="a2")
            + f"x = d1(Array({encode('ABCDE',100)}))\n"
        )
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_v10_out_of_range_refuses(self) -> None:
        """An element that subtracts below 0 is not this scheme -> the whole
        array is refused, never wrapped."""
        src = vbs_decoder(const=100) + "x = dec(Array(50, 60, 70))\n"  # 50-100 < 0
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_v11_malformed_array_refuses(self) -> None:
        src = vbs_decoder(const=100) + "x = dec(Array(165, , 167))\n"  # malformed
        # The malformed array does not match the integer-list pattern; no stage.
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_v12_huge_array_bounded(self) -> None:
        """An array over the input bound is refused, not decoded."""
        from orbit.runtime.analysis_deobfuscate import MAX_INPUT_CHARS
        huge = ", ".join("165" for _ in range(MAX_INPUT_CHARS))  # > MAX_INPUT_CHARS chars
        src = vbs_decoder(const=100) + f"x = dec(Array({huge}))\n"
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_v13_cycle_terminates(self) -> None:
        """A decode whose output re-contains the same construct terminates by
        the sha-cycle guard rather than looping."""
        # Construct text that decodes to itself is impossible under subtraction,
        # so assert the bounded walk simply terminates on a self-referential-ish
        # nest without exceeding limits.
        src = vbs_decoder(const=1) + f"x = dec(Array({encode('abc',1)}))\n"
        res = deobfuscate_with_status(src)  # must return, not hang
        self.assertLessEqual(len(res.stages), 16)

    def test_v14_near_match_divide_not_decoded(self) -> None:
        """A loop that is structurally similar but NOT Chr(e - const) is ignored."""
        src = (
            "Function dec(a)\n"
            "    k = 100\n"
            "    For Each e In a\n"
            "        s = s & Chr(e + k)\n"  # addition, not the subtraction scheme
            "    Next\n"
            "End Function\n"
            f"x = dec(Array({encode('ABC',100)}))\n"
        )
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_v15_embedded_in_hta_container(self) -> None:
        """The same decoder inside an HTA/VBScript <script> container."""
        src = (
            '<html><head><script language="VBScript">\n'
            + vbs_decoder(const=77)
            + f"o = dec(Array({encode('Net.WebClient',77)}))\n"
            + "</script></head><body></body></html>\n"
        )
        stages = self._only(find_vbscript_chr_stages(src))
        self.assertEqual(stages[0].output, "Net.WebClient")

    def test_chr_rejects_above_255_but_chrw_allows(self) -> None:
        """Chr is single-byte (0..255): a value over 255 is not a Chr input and
        the stage is refused, never wrapped. ChrW takes a UTF-16 code unit, so
        the same values decode under a ChrW decoder."""
        # values that subtract to 300, 400, 500 -> out of Chr range
        chr_src = (
            "Function dec(a)\n    k = 100\n    For Each e In a\n"
            "        s = s & Chr(e - k)\n    Next\nEnd Function\n"
            "x = dec(Array(400, 500, 600))\n"  # -> 300,400,500 > 255
        )
        self.assertEqual(find_vbscript_chr_stages(chr_src), [])
        chrw_src = (
            "Function dec(a)\n    k = 100\n    For Each e In a\n"
            "        s = s & ChrW(e - k)\n    Next\nEnd Function\n"
            "x = dec(Array(400, 500, 600))\n"
        )
        stages = self._only(find_vbscript_chr_stages(chrw_src))
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].output, chr(300) + chr(400) + chr(500))

    def test_non_literal_constant_not_read(self) -> None:
        """A computed assignment (k = 100 + x) is not a proven literal; the
        decoder that subtracts k is refused."""
        src = (
            "Function dec(a)\n    k = 100 + z\n    For Each e In a\n"
            "        s = s & Chr(e - k)\n    Next\nEnd Function\n"
            f"x = dec(Array({encode('ABC',100)}))\n"
        )
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_unlinked_inrange_array_is_not_decoded(self) -> None:
        """An in-range literal array NOT passed to the decoder is not a stage.

        The classic false positive: a benign lookup table whose bytes happen to
        fall in range coexists with a real decoder. Detecting the decoder shape
        must not license decoding every array in the file -- only arrays
        syntactically fed to the decoder decode. (MAJOR-1.)"""
        src = (
            vbs_decoder(const=0)
            + "lookup = Array(104, 116, 116, 112, 58, 47, 47)\n"  # 'http://', unused
        )
        self.assertEqual(find_vbscript_chr_stages(src), [])
        # When the SAME array is actually passed, it decodes.
        self.assertEqual(
            find_vbscript_chr_stages(src + "x = dec(lookup)\n")[0].output, "http://"
        )

    def test_linked_via_inline_and_via_variable(self) -> None:
        """Both linkage forms decode: decoder(Array(...)) and v=Array(...);decoder(v)."""
        inline = vbs_decoder(const=100) + f"a = dec(Array({encode('INLINE',100)}))\n"
        self.assertEqual(find_vbscript_chr_stages(inline)[0].output, "INLINE")
        viavar = (vbs_decoder(const=100)
                  + f"v = Array({encode('VIAVAR',100)})\n"
                  + "r = dec(v)\n")
        self.assertEqual(find_vbscript_chr_stages(viavar)[0].output, "VIAVAR")

    def test_ps_computed_constant_is_not_read_as_literal(self) -> None:
        """PowerShell `$h = 22624 + 1` must not be read as 22624 (MAJOR-2)."""
        # A genuinely linked decoder (array passed to the function), refused
        # ONLY because its constant is a computed expression, not a literal.
        text = (
            "function atHcQU($z){foreach($p in $z){$o+=[char]($p - $h)}}\n"
            "$h = 500 + 1\n"  # computed -> not a proven literal -> refuse
            f"x = atHcQU @({', '.join(str(ord(c)+501) for c in 'ABC')})\n"
        )
        self.assertEqual(find_powershell_chr_stages(text), [])

    def test_ps_unpassed_global_sharing_param_name_not_decoded(self) -> None:
        """A benign PS global that merely SHARES the decoder's parameter name is
        not decoded -- linkage is by being passed, never by name collision.
        (Regression for the foreach-iterated-variable over-decode.)"""
        src = (
            "function Convert-Data($a){foreach($b in $a){$o+=[char]($b - $h)}}\n"
            "$h = 100\n"
            # a benign in-range global named $a, NEVER passed to Convert-Data
            f"$a = @({', '.join(str(ord(c)+100) for c in 'http://')})\n"
        )
        stages = [s for s in find_powershell_chr_stages(src) if s.kind == POWERSHELL_CHR]
        self.assertEqual(stages, [])
        # When $a IS passed, it decodes (linkage via assign-then-pass).
        linked = src + "x = Convert-Data $a\n"
        out = [s.output for s in find_powershell_chr_stages(linked) if s.kind == POWERSHELL_CHR]
        self.assertEqual(out, ["http://"])

    def test_ps_function_def_param_not_treated_as_call(self) -> None:
        """The function DEFINITION's parameter list is not a call: a global
        sharing the parameter name, with ANY spacing after `function`, is not
        linked. (Regression for the two-space `function  Name` look-behind gap.)"""
        for kw in ("function Convert", "function  Convert", "filter Convert"):
            src = (
                f"{kw}($a){{foreach($b in $a){{$o+=[char]($b - $h)}}}}\n"
                "$h = 100\n"
                f"$a = @({', '.join(str(ord(c)+100) for c in 'http://')})\n"
            )
            stages = [s for s in find_powershell_chr_stages(src) if s.kind == POWERSHELL_CHR]
            self.assertEqual(stages, [], f"{kw!r} def param leaked as a call")

    def test_ps_long_whitespace_def_not_treated_as_call(self) -> None:
        """A DEFINITION with a long whitespace/newline run between `function`
        and the name, and a default-parameter array, must not be decoded: the
        array is never passed to anything. (Regression for the bounded-window
        _is_definition leak -- the guard is now exact-position, not windowed.)"""
        for sep in (" " * 35, "\n    ", "\t\t", "\n\n\t  "):
            src = (
                f"function{sep}Invoke-Decode($a = @("
                f"{', '.join(str(ord(c)+100) for c in 'http://evil/x')}))"
                "{$s='';foreach($e in $a){$s+=[char]($e - $h)}}\n"
                "$h = 100\n"
            )
            stages = [s for s in find_powershell_chr_stages(src) if s.kind == POWERSHELL_CHR]
            self.assertEqual(stages, [], f"def with sep {sep!r} leaked as a call")

    def test_ps_hex_constant_not_read_as_decimal(self) -> None:
        """`$h = 0x10` must not be read as decimal 0 (MINOR-1)."""
        src = (
            "function dd($z){foreach($p in $z){$o+=[char]($p - $h)}}\n"
            "$h = 0x10\n"
            f"x = dd @({', '.join(str(ord(c)+16) for c in 'ABC')})\n"
        )
        self.assertEqual(
            [s for s in find_powershell_chr_stages(src) if s.kind == POWERSHELL_CHR], []
        )

    def test_ps_parenthesized_call_links(self) -> None:
        """`Dec(@(...))` (parenthesized) links as well as the bare form."""
        src = (
            "function dd($z){foreach($p in $z){$o+=[char]($p - $h)}}\n"
            "$h = 100\n"
            f"x = dd(@({', '.join(str(ord(c)+100) for c in 'OKOK')}))\n"
        )
        out = [s.output for s in find_powershell_chr_stages(src) if s.kind == POWERSHELL_CHR]
        self.assertEqual(out, ["OKOK"])

    def test_ps_linked_literal_constant_decodes(self) -> None:
        """The corrected PowerShell path still decodes a genuine literal case."""
        text = ("function dd($z){foreach($p in $z){$o+=[char]($p - $h)}}\n"
                "$h = 500\n"
                f"x = dd @({', '.join(str(ord(c)+500) for c in 'HELLO')})\n")
        stages = [s for s in find_powershell_chr_stages(text) if s.kind == POWERSHELL_CHR]
        self.assertEqual(stages[0].output, "HELLO")

    def test_property_assignment_not_read_as_constant(self) -> None:
        """`obj.k = 100` must not register a constant named `k` (MINOR-2)."""
        src = (
            "Function dec(a)\n    For Each e In a\n"
            "        s = s & Chr(e - k)\n    Next\nEnd Function\n"
            "obj.k = 100\n"                               # property, not a bare k
            f"x = dec(Array({encode('ABC',100)}))\n"
        )
        self.assertEqual(find_vbscript_chr_stages(src), [])

    def test_provenance_fields_exact(self) -> None:
        src = vbs_decoder(const=100) + f"x = dec(Array({encode('ABC',100)}))\n"
        s = self._only(find_vbscript_chr_stages(src))[0]
        self.assertEqual(s.key, 100)
        self.assertEqual(s.input_sha256, hashlib.sha256(s.encoded.encode()).hexdigest())
        self.assertEqual(s.output_sha256, hashlib.sha256(s.output.encode()).hexdigest())
        self.assertGreater(s.line, 0)
        self.assertEqual(s.kind, VBSCRIPT_CHR)

    def test_determinism_repeatable(self) -> None:
        src = vbs_decoder(const=100) + f"x = dec(Array({encode('determinism',100)}))\n"
        a = [(s.kind, s.output_sha256) for s in deobfuscate(src)]
        b = [(s.kind, s.output_sha256) for s in deobfuscate(src)]
        self.assertEqual(a, b)


class DeterministicOnlyReportTests(unittest.TestCase):
    """The closing report for an oversized source whose only evidence is the
    deterministic decode must present that evidence -- never claim 'no evidence'
    or 'source too large' when the runtime in fact recovered the hidden stages."""

    def _runtime(self, source_text: str) -> AnalysisRuntime:
        ws = AnalysisWorkspace.create()
        self.addCleanup(ws.close)
        data = source_text.encode()
        path = ws.source_root / "a.hta"
        path.write_bytes(data)
        return AnalysisRuntime(
            backend=_NullBackend(),
            source=AnalysisSource(
                snapshot_path=path, sha256=_hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(path),
            ),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )

    def test_oversized_with_decoder_reports_the_decoded_evidence(self) -> None:
        payload = "http://malware.example/stage2"
        # Oversized for the model context (~40 KB, well past the ~9.6 KB
        # head+tail bootstrap threshold), but under the 262 KB deobfuscate input
        # bound so the preflight runs -- the realistic shape, like mine.hta.
        body = (
            "Function dec(a)\n  k = 100\n  For Each e In a\n"
            "    s = s & Chr(e - k)\n  Next\n  dec=s\nEnd Function\n"
            f"x = dec(Array({encode(payload, 100)}))\n"
            + "'pad\n" * 8000  # ~40 KB, oversized for ctx, under MAX_INPUT_CHARS
        )
        rt = self._runtime(body)
        self.assertEqual(len(rt.transform_stages), 1)  # preflight decoded it
        rep = rt.report()
        self.assertEqual(rep.model_calls, 0)  # deterministic, no model call
        # NOT the "could not proceed / no evidence / source too large" wording.
        low = rep.text.lower()
        self.assertNotIn("could not proceed", low)
        self.assertNotIn("no evidence was collected", low)
        self.assertNotIn("so no evidence", low)
        # IS the deterministic-only opening, and carries the decoded value.
        self.assertTrue(
            rep.text.startswith(DETERMINISTIC_ONLY_REPORT.split("{", 1)[0])
        )
        self.assertIn(payload, rep.text)

    def test_oversized_without_decoder_still_says_coverage_limited(self) -> None:
        """No decoder, oversized source: the honest 'source too large' closing
        still stands (Part A) -- the new branch only fires when stages exist."""
        body = "Dim x\nx = 1\n" + "'inert\n" * 8000  # ~40 KB, no decoder
        rt = self._runtime(body)
        self.assertEqual(len(rt.transform_stages), 0)
        rep = rt.report()
        # With no transforms, it is NOT the deterministic-only opening.
        self.assertFalse(
            rep.text.startswith(DETERMINISTIC_ONLY_REPORT.split("{", 1)[0])
        )
        self.assertEqual(rep.model_calls, 0)


if __name__ == "__main__":
    unittest.main()
