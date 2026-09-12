"""Tests for the tiny VBA static evaluator and the byte-offset+StrReverse
transform finder. STATIC ONLY -- the evaluator folds its own whitelisted
representation of inert VBA text and executes nothing.

Synthetic fixtures are unrelated to any real malware; their expected outputs are
specified independently (computed by hand in the test, not by the production
evaluator). The frozen-sample expectations are validation constants established
by an independent witness.
"""
from __future__ import annotations

import hashlib
import unittest

from orbit.runtime.analysis_deobfuscate import (
    VBA_BYTEOFFSET,
    deobfuscate,
    deobfuscate_with_status,
    find_vba_byteoffset_stages,
)
from orbit.runtime.analysis_vba_eval import (
    UNKNOWN,
    VbaEvaluator,
    VbaRefuse,
    collect_reaching_defs,
    find_byte_offset_helpers,
)

SAMPLE = "workdir/samples/99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc"
# Independently established (hand witness) — validation expectation only.
ORACLE_CMD_SHA = "f1fa67e3f558443e9604b0f0b26715dc8065c53c92db7acbab2e6d32874e5c21"
ORACLE_URL = "http://185.189.58.222/x.exe"


def _enc(plain: str, offset: int) -> str:
    """The encoded accumulator for a byte-offset helper that SUBTRACTS `offset`:
    each plaintext byte is stored as (byte + offset), so decoding subtracts it
    back. Independent of the production evaluator."""
    return "".join(chr((ord(c) + offset) & 0xFF) for c in plain)


def _helper(name: str, arg: str, encoded: str, op: str = "-") -> str:
    # A minimal byte-offset decoder function. `encoded` is appended as a single
    # string literal (escaped for VBA doubled quotes).
    esc = encoded.replace('"', '""')
    return (
        f'Public Function {name}({arg})\r\n'
        f'    xAcc = ""\r\n'
        f'    xAcc = xAcc & "{esc}"\r\n'
        f'    Dim b() As Byte\r\n'
        f'    b = StrConv(xAcc, vbFromUnicode)\r\n'
        f'    i = 0\r\n'
        f'    While i <= UBound(b)\r\n'
        f'        b(i) = b(i) {op} {arg}\r\n'
        f'        i = i + 1\r\n'
        f'    Wend\r\n'
        f'    {name} = StrConv(b, vbUnicode)\r\n'
        f'End Function\r\n'
    )


class EvaluatorPrimitiveTests(unittest.TestCase):
    def _ev(self, defs=None, helpers=None):
        return VbaEvaluator(defs or {}, helpers or {})

    def test_chr_arithmetic(self):
        self.assertEqual(self._ev().evaluate("Chr(65 + 1)"), "B")
        self.assertEqual(self._ev().evaluate("Chr(100 - 1)"), "c")

    def test_chr_domain_refused(self):
        with self.assertRaises(VbaRefuse):
            self._ev().evaluate("Chr(300)")  # > 255

    def test_chrw_wide(self):
        self.assertEqual(self._ev().evaluate("ChrW(9731)"), "☃")

    def test_strreverse_exact(self):
        self.assertEqual(self._ev().evaluate('StrReverse("abc")'), "cba")
        self.assertEqual(self._ev().evaluate('StrReverse("")'), "")
        self.assertEqual(self._ev().evaluate('StrReverse("abcd")'), "dcba")
        self.assertEqual(self._ev().evaluate('StrReverse("a☃z")'), "z☃a")

    def test_replace_exact(self):
        self.assertEqual(self._ev().evaluate('Replace("axbxc", "x", "-")'), "a-b-c")
        # empty find returns source unchanged (VBA semantics)
        self.assertEqual(self._ev().evaluate('Replace("abc", "", "x")'), "abc")

    def test_split_index(self):
        self.assertEqual(self._ev().evaluate('Split("a|b|c", "|")(1)'), "b")
        self.assertEqual(self._ev().evaluate('Split("a|b|c", Chr(124))(2)'), "c")

    def test_split_out_of_range_refused(self):
        with self.assertRaises(VbaRefuse):
            self._ev().evaluate('Split("a|b", "|")(5)')

    def test_split_empty_delim_refused(self):
        with self.assertRaises(VbaRefuse):
            self._ev().evaluate('Split("abc", "")(0)')

    def test_concat(self):
        self.assertEqual(self._ev().evaluate('"po" & "wer"'), "power")
        self.assertEqual(
            self._ev().evaluate("Chr(104) + Chr(105)"), "hi"
        )  # + on two strings concatenates

    def test_unknown_name_refused(self):
        with self.assertRaises(VbaRefuse):
            self._ev().evaluate("someUndefinedName")

    def test_unsupported_function_refused(self):
        with self.assertRaises(VbaRefuse):
            self._ev().evaluate('Mid("abcdef", 2, 3)')  # Mid not in the subset

    def test_oversized_int_literal_fails_closed(self):
        # An over-long numeric literal must raise VbaRefuse, never an uncaught
        # ValueError from Python's int-string digit limit.
        for expr in ("9" * 5000, "Chr(" + "9" * 5000 + ")", "&h" + "F" * 5000):
            with self.assertRaises(VbaRefuse):
                self._ev().evaluate(expr)

    def test_deep_nesting_fails_closed(self):
        expr = "(" * 5000 + "1" + ")" * 5000
        with self.assertRaises(VbaRefuse):
            self._ev().evaluate(expr)

    def test_finder_never_propagates_on_adversarial_input(self):
        # The finder must fail closed on ANY crafted module, never propagate an
        # exception that would end a run.
        from orbit.runtime.analysis_deobfuscate import find_vba_byteoffset_stages
        helper = _helper("dd", "k", _enc("x", 1), op="-")
        adversarial = [
            helper + 'Sub Go()\r\n    cmd = Chr(' + "9" * 5000 + ')\r\n    Shell (cmd), 0\r\nEnd Sub\r\n',
            helper + 'Sub Go()\r\n    cmd = dd(1)\r\n    Shell (cmd), ' + "9" * 50 + '\r\nEnd Sub\r\n',
            helper + 'Sub Go()\r\n    Shell (undefinedvar), 0\r\nEnd Sub\r\n',
            'StrConv Shell ' * 1000,  # keyword soup, no real structure
        ]
        for src in adversarial:
            # must return a list (possibly empty), never raise
            self.assertIsInstance(find_vba_byteoffset_stages(src), list)

    def test_byte_decode_wraps_mod_256(self):
        # A byte pushed past 255 by the offset must wrap mod 256 exactly as a
        # VBA Byte assignment does -- not raise, not clamp. Pin the wrap so a
        # mutant that drops it is caught. offset +10 on a high byte.
        from orbit.runtime.analysis_vba_eval import ByteOffsetHelper
        # encoded byte 250; ADD sign, offset 10 -> (250+10)&255 = 4
        h = ByteOffsetHelper("h", "k", chr(250), sign=1)
        self.assertEqual(h.decode(10), chr(4))
        # SUBTRACT below 0 -> wraps up: byte 3, subtract 10 -> (3-10)&255 = 249
        h2 = ByteOffsetHelper("h", "k", chr(3), sign=-1)
        self.assertEqual(h2.decode(10), chr(249))

    def test_evaluation_step_budget_enforced(self):
        # A crafted expression that drives many evaluator steps must hit the
        # step budget and fail closed rather than run unbounded.
        import orbit.runtime.analysis_vba_eval as mod
        ev = self._ev()
        # each nested Chr/parenthesis costs ticks; build a deep chain.
        expr = "0"
        for _ in range(50):
            expr = f"({expr} + 0)"
        saved = mod.MAX_EVAL_STEPS
        try:
            mod.MAX_EVAL_STEPS = 5  # force the budget to bite
            ev2 = self._ev()
            with self.assertRaises(VbaRefuse):
                ev2.evaluate(expr)
        finally:
            mod.MAX_EVAL_STEPS = saved


class ReachingDefinitionTests(unittest.TestCase):
    def test_unique_definition_resolves(self):
        defs = collect_reaching_defs('x = "hello"\r\ny = StrReverse(x)\r\n')
        ev = VbaEvaluator(defs, {})
        self.assertEqual(ev.evaluate("y"), "olleh")

    def test_reassignment_refused(self):
        defs = collect_reaching_defs('x = "a"\r\nx = "b"\r\ny = StrReverse(x)\r\n')
        ev = VbaEvaluator(defs, {})
        with self.assertRaises(VbaRefuse):
            ev.evaluate("y")

    def test_colon_hidden_reassignment_refused(self):
        # A reassignment hidden after a `:` on a line that starts with something
        # else must still count -- otherwise the variable looks uniquely defined
        # and decodes to a stale value (a wrong-but-confident result).
        defs = collect_reaching_defs('cmd = "first"\r\nDoStuff : cmd = "second"\r\n')
        self.assertEqual(len(defs.get("cmd", [])), 2)
        ev = VbaEvaluator(defs, {})
        with self.assertRaises(VbaRefuse):
            ev.evaluate("cmd")

    def test_colon_inside_string_does_not_split(self):
        # A `:` inside a string literal (e.g. a URL) must not split the statement.
        defs = collect_reaching_defs('u = "http://host/x"\r\n')
        self.assertEqual(defs.get("u"), ['"http://host/x"'])
        self.assertEqual(VbaEvaluator(defs, {}).evaluate("u"), "http://host/x")


class ByteOffsetHelperTests(unittest.TestCase):
    def test_helper_recognised_and_decodes(self):
        src = _helper("dec", "k", _enc("ABC", 1), op="-")
        helpers = find_byte_offset_helpers(src)
        self.assertIn("dec", helpers)
        self.assertEqual(helpers["dec"].sign, -1)
        self.assertEqual(helpers["dec"].decode(1), "ABC")

    def test_helper_offset_is_the_argument_not_hardcoded(self):
        # offset 4, not 1: decode must track the call argument
        src = _helper("dec", "k", _enc("HELLO", 4), op="-")
        helpers = find_byte_offset_helpers(src)
        self.assertEqual(helpers["dec"].decode(4), "HELLO")
        self.assertNotEqual(helpers["dec"].decode(1), "HELLO")

    def test_helper_add_sign(self):
        src = _helper("dec", "k", _enc("ABC", -2), op="+")
        helpers = find_byte_offset_helpers(src)
        self.assertEqual(helpers["dec"].sign, 1)
        self.assertEqual(helpers["dec"].decode(2), "ABC")

    def test_nonliteral_accumulator_refused(self):
        # an accumulator fed from a runtime value is not reconstructable
        src = (
            'Public Function dec(k)\r\n'
            '    xAcc = xAcc & SomeRuntimeThing()\r\n'
            '    b = StrConv(xAcc, vbFromUnicode)\r\n'
            '    b(i) = b(i) - k\r\n'
            '    dec = StrConv(b, vbUnicode)\r\n'
            'End Function\r\n'
        )
        self.assertEqual(find_byte_offset_helpers(src), {})


class SyntheticFinderTests(unittest.TestCase):
    """§14 fixtures: each an independent synthetic macro, expected output
    specified by hand."""

    def _stage(self, src):
        stages = find_vba_byteoffset_stages(src)
        return stages

    def _module(self, body, helper_plain="ShellCmd.exe /q", offset=1, op="-"):
        # A module with a byte-offset helper and a Shell site consuming `cmd`.
        enc = _enc(helper_plain, offset)
        return _helper("dd", "k", enc, op=op) + body

    def test_v1_direct_decode_to_shell(self):
        # cmd = dd(1) ; Shell(cmd), 0   -> plaintext
        src = self._module(
            'Sub Go()\r\n    cmd = dd(1)\r\n    Shell (cmd), 0\r\nEnd Sub\r\n'
        )
        stages = self._stage(src)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].kind, VBA_BYTEOFFSET)
        self.assertEqual(stages[0].output, "ShellCmd.exe /q")
        self.assertEqual(stages[0].key, 0)  # window style
        self.assertEqual(stages[0].delimiter, "Shell|0")

    def test_v2_different_offset(self):
        src = self._module(
            'Sub Go()\r\n    cmd = dd(4)\r\n    Shell (cmd), 0\r\nEnd Sub\r\n',
            helper_plain="calc.exe", offset=4,
        )
        stages = self._stage(src)
        self.assertEqual(stages[0].output, "calc.exe")

    def test_v3_strreverse_and_split(self):
        # helper yields "x|calc.exe|y"; Shell runs Split(reverse(...),...)? Keep
        # it to reverse+split: cmd = Split(StrReverse(dd(1)), "|")(1)
        plain = "y|exe.clac|x"  # reversed -> "x|calc.exe|y"
        src = self._module(
            'Sub Go()\r\n'
            '    cmd = Split(StrReverse(dd(1)), "|")(1)\r\n'
            '    Shell (cmd), 0\r\n'
            'End Sub\r\n',
            helper_plain=plain,
        )
        stages = self._stage(src)
        self.assertEqual(stages[0].output, "calc.exe")

    def test_v4_randomized_names(self):
        src = self._helper_named("zQ_rnd", "off_x") + (
            'Sub AutoThing()\r\n'
            '    Qz7 = zQ_rnd(1)\r\n'
            '    Shell (Qz7), 0\r\n'
            'End Sub\r\n'
        )
        stages = self._stage(src)
        self.assertEqual(stages[0].output, "notepad.exe")

    def _helper_named(self, name, arg):
        return _helper(name, arg, _enc("notepad.exe", 1), op="-")

    def test_v5_window_style_preserved(self):
        src = self._module(
            'Sub Go()\r\n    cmd = dd(1)\r\n    Shell (cmd), 6\r\nEnd Sub\r\n'
        )
        self.assertEqual(self._stage(src)[0].key, 6)

    # --- refusal fixtures ---------------------------------------------------
    def test_v_dynamic_offset_refused(self):
        # offset comes from a runtime call, not a literal -> no decode
        src = self._module(
            'Sub Go()\r\n    cmd = dd(GetOffset())\r\n    Shell (cmd), 0\r\nEnd Sub\r\n'
        )
        self.assertEqual(self._stage(src), [])

    def test_v_nonint_offset_refused(self):
        # a RESOLVABLE but non-integer offset (a string) must be refused by the
        # helper's integer check, not coerced to a default -- pins that check.
        src = self._module(
            'Sub Go()\r\n    cmd = dd("x")\r\n    Shell (cmd), 0\r\nEnd Sub\r\n'
        )
        self.assertEqual(self._stage(src), [])

    def test_helper_nonint_offset_refused_directly(self):
        from orbit.runtime.analysis_vba_eval import VbaEvaluator
        src = self._helper_named("dd2", "k")
        helpers = find_byte_offset_helpers(src)
        ev = VbaEvaluator({}, helpers)
        with self.assertRaises(VbaRefuse):
            ev.evaluate('dd2("nope")')

    def test_v_reassigned_command_refused(self):
        src = self._module(
            'Sub Go()\r\n'
            '    cmd = dd(1)\r\n'
            '    cmd = "override"\r\n'
            '    Shell (cmd), 0\r\n'
            'End Sub\r\n'
        )
        self.assertEqual(self._stage(src), [])

    def test_v_unsupported_primitive_refused(self):
        src = self._module(
            'Sub Go()\r\n    cmd = Mid(dd(1), 2)\r\n    Shell (cmd), 0\r\nEnd Sub\r\n'
        )
        self.assertEqual(self._stage(src), [])

    def test_v_out_of_range_split_refused(self):
        src = self._module(
            'Sub Go()\r\n    cmd = Split(dd(1), "|")(9)\r\n    Shell (cmd), 0\r\nEnd Sub\r\n'
        )
        self.assertEqual(self._stage(src), [])

    def test_v_no_helper_no_decode(self):
        # a Shell site with no byte-offset helper in the module: not this family
        src = 'Sub Go()\r\n    cmd = "x"\r\n    Shell (cmd), 0\r\nEnd Sub\r\n'
        self.assertEqual(self._stage(src), [])


class ReviewHardeningTests(unittest.TestCase):
    """Regressions for the independent-review findings: a wrong-but-confident
    decode or a crash-the-run path is disqualifying."""

    def _stage(self, src):
        from orbit.runtime.analysis_deobfuscate import find_vba_byteoffset_stages
        return find_vba_byteoffset_stages(src)

    def test_blocker_decoy_accumulator_refused(self):
        # Two StrConv(vbFromUnicode) in one helper: a decoy over an
        # attacker-chosen URL, and the real one the byte-op actually decodes.
        # The recogniser must REFUSE (ambiguous chain), never emit the decoy.
        decoy = _enc("http://attacker.test/drop.exe", 1)   # not the real flow
        real = _enc("calc.exe", 1)
        src = (
            "Public Function dd(k)\r\n"
            '    c = ""\r\n'
            f'    c = c & "{decoy}"\r\n'
            "    junk = StrConv(c, vbFromUnicode)\r\n"   # decoy StrConv-from
            '    a = ""\r\n'
            f'    a = a & "{real}"\r\n'
            "    Dim b() As Byte\r\n"
            "    b = StrConv(a, vbFromUnicode)\r\n"        # real StrConv-from
            "    b(i) = b(i) - k\r\n"
            "    dd = StrConv(b, vbUnicode)\r\n"
            "End Function\r\n"
            "Sub Go()\r\n    cmd = dd(1)\r\n    Shell (cmd), 0\r\nEnd Sub\r\n"
        )
        self.assertEqual(find_byte_offset_helpers(src), {})  # no coherent chain
        self.assertEqual(self._stage(src), [])  # and no fabricated indicator

    def test_cross_scope_command_refused(self):
        h = _helper("dd", "k", _enc("evil.exe", 1))
        src = h + (
            "Sub A()\r\n    cmd = dd(1)\r\nEnd Sub\r\n"
            "Sub B()\r\n    Shell (cmd), 0\r\nEnd Sub\r\n"
        )
        self.assertEqual(self._stage(src), [])

    def test_forward_reference_refused(self):
        h = _helper("dd", "k", _enc("evil.exe", 1))
        src = h + "Sub C()\r\n    Shell (cmd), 0\r\n    cmd = dd(1)\r\nEnd Sub\r\n"
        self.assertEqual(self._stage(src), [])

    def test_module_level_shell_refused(self):
        # A `Shell` outside any procedure cannot run in real VBA and must not
        # borrow a procedure's locals (a cross-scope leak at module level).
        h = _helper("dd", "k", _enc("evil.exe", 1))
        src = h + "Sub A()\r\n    cmd = dd(1)\r\nEnd Sub\r\nShell (cmd), 0\r\n"
        self.assertEqual(self._stage(src), [])

    def test_chr_high_ansi_refused(self):
        # Chr(128..255) is codepage-dependent; the evaluator must refuse, not
        # return the Latin-1 character.
        ev = VbaEvaluator({}, {})
        for n in (128, 150, 255):
            with self.assertRaises(VbaRefuse):
                ev.evaluate(f"Chr({n})")
        # ChrW (Unicode) still works across the BMP
        self.assertEqual(ev.evaluate("ChrW(8364)"), "€")

    def test_nonascii_accumulator_helper_refused(self):
        # An accumulator with a non-ASCII byte cannot be decoded with the ASCII
        # low-byte model, so the helper must not be recognised.
        src = _helper("dd", "k", "abcé", op="-")  # é in the accumulator
        self.assertEqual(find_byte_offset_helpers(src), {})

    def test_helper_lookup_case_insensitive(self):
        src = _helper("Dd", "k", _enc("notepad.exe", 1))
        helpers = find_byte_offset_helpers(src)
        ev = VbaEvaluator({}, helpers)
        # VBA is case-insensitive: Dd, dd, DD all call the same function
        self.assertEqual(ev.evaluate("dd(1)"), "notepad.exe")
        self.assertEqual(ev.evaluate("DD(1)"), "notepad.exe")

    def test_incoherent_chain_single_of_each_refused(self):
        # Exactly ONE StrConv-from, ONE byte-op, ONE StrConv-to, but they thread
        # DIFFERENT variables: the count check passes, so only the same-variable
        # link check can reject it. `a` is StrConv'd from the decoy, but the
        # byte-op and StrConv-to use `x` (an unrelated array). No coherent chain.
        decoy = _enc("http://attacker.test/x.exe", 1)
        src = (
            "Public Function dd(k)\r\n"
            '    a = ""\r\n'
            f'    a = a & "{decoy}"\r\n'
            "    Dim bb() As Byte\r\n"
            "    bb = StrConv(a, vbFromUnicode)\r\n"   # from `a`
            "    x(i) = x(i) - k\r\n"                    # byte-op on `x`, not bb
            "    dd = StrConv(x, vbUnicode)\r\n"         # to from `x`, not bb
            "End Function\r\n"
        )
        self.assertEqual(find_byte_offset_helpers(src), {})

    def test_accumulator_direct_assign_then_append_full(self):
        # `a = "FIRST"` then `a = a & "SECOND"` must reconstruct the FULL string,
        # not just the appended tail -- scanning only `&`-append lines silently
        # truncated it (a wrong-but-confident decode).
        e1 = _enc("FIRST", 1)
        e2 = _enc("SECOND", 1)
        src = (
            "Public Function dd(k)\r\n"
            f'    a = "{e1}"\r\n'
            f'    a = a & "{e2}"\r\n'
            "    Dim b() As Byte\r\n"
            "    b = StrConv(a, vbFromUnicode)\r\n"
            "    b(i) = b(i) - k\r\n"
            "    dd = StrConv(b, vbUnicode)\r\n"
            "End Function\r\n"
        )
        helpers = find_byte_offset_helpers(src)
        self.assertIn("dd", helpers)
        self.assertEqual(helpers["dd"].decode(1), "FIRSTSECOND")

    def test_accumulator_alias_refused(self):
        # An accumulator fed by an alias (`a = c`), not literal assignments,
        # cannot be reconstructed -> refuse, never emit a partial/decoy value.
        ed = _enc("http://evil/x.exe", 1)
        src = (
            "Public Function dd(k)\r\n"
            f'    c = "{ed}"\r\n'
            "    a = c\r\n"
            "    Dim b() As Byte\r\n"
            "    b = StrConv(a, vbFromUnicode)\r\n"
            "    b(i) = b(i) - k\r\n"
            "    dd = StrConv(b, vbUnicode)\r\n"
            "End Function\r\n"
        )
        self.assertEqual(find_byte_offset_helpers(src), {})

    def test_ambiguous_chain_two_byteops_refused(self):
        # two byte-ops in one body is an ambiguous chain -> refuse
        src = (
            "Public Function dd(k)\r\n"
            '    a = ""\r\n'
            f'    a = a & "{_enc("calc.exe", 1)}"\r\n'
            "    Dim b() As Byte\r\n"
            "    b = StrConv(a, vbFromUnicode)\r\n"
            "    b(i) = b(i) - k\r\n"
            "    b(j) = b(j) - k\r\n"   # second byte-op -> ambiguous
            "    dd = StrConv(b, vbUnicode)\r\n"
            "End Function\r\n"
        )
        self.assertEqual(find_byte_offset_helpers(src), {})


class FrozenSampleTransformTests(unittest.TestCase):
    def setUp(self):
        from orbit.runtime.analysis_ole import extract_office_vba
        raw = open(SAMPLE, "rb").read()
        ex = extract_office_vba(raw)
        self.vba = ex.modules[0].source

    def test_frozen_decode_matches_oracle(self):
        stages = deobfuscate(self.vba)
        vba = [s for s in stages if s.kind == VBA_BYTEOFFSET]
        self.assertEqual(len(vba), 1)
        cmd = vba[0].output
        self.assertEqual(
            hashlib.sha256(cmd.encode("utf-8", "surrogatepass")).hexdigest(),
            ORACLE_CMD_SHA,
        )

    def test_frozen_window_style_hidden(self):
        vba = [s for s in deobfuscate(self.vba) if s.kind == VBA_BYTEOFFSET][0]
        self.assertEqual(vba.key, 0)  # vbHide
        self.assertEqual(vba.delimiter, "Shell|0")

    def test_frozen_decoded_url_is_extractable_ioc(self):
        from orbit.runtime.analysis_indicators import uris_in
        vba = [s for s in deobfuscate(self.vba) if s.kind == VBA_BYTEOFFSET][0]
        self.assertIn(ORACLE_URL, uris_in(vba.output))

    def test_frozen_status_complete(self):
        self.assertTrue(deobfuscate_with_status(self.vba).complete)


class RuntimeIntegrationTests(unittest.TestCase):
    """The transform must run over the EXTRACTED (decompressed) VBA source, not
    only the raw file bytes -- a real macro is MS-OVBA-compressed and never
    appears in the raw text."""

    def _runtime(self, sample_bytes, name="in.doc"):
        import tempfile
        from pathlib import Path
        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime, acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore
        tmp = tempfile.mkdtemp(prefix="orbit-vbaint-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        p = Path(tmp) / name
        p.write_bytes(sample_bytes)
        source = acquire_analysis_source(p, Path(tmp) / "owned")
        rt = AnalysisRuntime(
            backend=None, source=source,
            evidence_store=EvidenceStore(root=Path(tmp) / "evidence"),
        )
        self.addCleanup(rt.close)
        return rt

    def test_frozen_sample_yields_decoded_command_stage(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        vba = [
            (s, r) for s, r in rt.transform_stages if s.kind == VBA_BYTEOFFSET
        ]
        self.assertEqual(len(vba), 1)
        stage, record = vba[0]
        self.assertEqual(
            hashlib.sha256(stage.output.encode("utf-8", "surrogatepass")).hexdigest(),
            ORACLE_CMD_SHA,
        )
        # provenance points at the module, not the raw bytes
        self.assertEqual(
            record.metadata.get("transform_origin"), "Macros/VBA/ThisDocument"
        )

    def test_frozen_decoded_url_becomes_verified_indicator(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        self.assertIn(ORACLE_URL, rt.verified_indicators())

    def test_frozen_transform_appendix_carries_command(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        appendix = rt.transform_appendix()
        self.assertIn(VBA_BYTEOFFSET, appendix)
        self.assertIn("powershell", appendix.lower())


class NonSampleHardeningTests(unittest.TestCase):
    def test_transform_imports_no_execution_primitive(self):
        # The transform evaluates its own whitelisted representation of inert
        # text. It must never import an execution/eval primitive.
        import ast
        for mod in (
            "src/orbit/runtime/analysis_vba_eval.py",
            "src/orbit/runtime/analysis_deobfuscate.py",
        ):
            tree = ast.parse(open(mod, encoding="utf-8").read())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            for forbidden in ("subprocess", "os", "pty", "ctypes"):
                self.assertNotIn(
                    forbidden, imported, f"{mod} imports {forbidden}"
                )

    def test_no_sample_constants_in_production_source(self):
        # The IOC/command/SHA must never participate in recognition.
        for mod in (
            "src/orbit/runtime/analysis_vba_eval.py",
            "src/orbit/runtime/analysis_deobfuscate.py",
        ):
            text = open(mod, encoding="utf-8").read()
            for banned in (
                "185.189.58.222", "PHfW", "x.exe", ORACLE_CMD_SHA,
                "JtJlaLkoKk", "99eb1d90",
            ):
                self.assertNotIn(banned, text, f"{banned} leaked into {mod}")


if __name__ == "__main__":
    unittest.main()
