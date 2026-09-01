"""Exact literal deobfuscation: what it decodes, and what it refuses to.

The refusals matter as much as the decodes. Anything whose key, input or
delimiter is not a literal is left to the model and its sandbox, because
resolving it would mean interpreting the program rather than reading it.

Every fixture here is synthetic and benign; the outputs are unrelated to any
real artifact.
"""

from __future__ import annotations

import hashlib
import unittest

from orbit.runtime.analysis_deobfuscate import (
    JSCRIPT_XOR,
    MAX_DEPTH,
    POWERSHELL_XOR,
    deobfuscate,
    find_jscript_stages,
    find_powershell_stages,
)


def encode(text: str, key: int, delimiter: str) -> str:
    return delimiter.join(str(ord(c) ^ key) for c in text)


def decoder(name: str = "dec", a: str = "s", b: str = "k", c: str = "d") -> str:
    return (
        f"function {name}({a}, {b}, {c}) {{\n"
        f"    var out = '';\n"
        f"    var parts = {a}.split({c});\n"
        f"    for (var i = 0; i < parts.length; i++) {{\n"
        f"        out += String.fromCharCode(parts[i] ^ {b});\n"
        f"    }}\n"
        f"    return out;\n"
        f"}}\n"
    )


class JScriptMatcherTests(unittest.TestCase):
    def test_a_renamed_decoder_function_is_still_recognised(self) -> None:
        """Matched by what the body does, never by the name."""
        for name in ("dec", "GxQ9_zzT", "$a", "_0x4f2b"):
            with self.subTest(name=name):
                src = decoder(name) + f'{name}("{encode("HELLO", 9, "-")}", 9, "-");\n'
                stages = find_jscript_stages(src)
                self.assertEqual([s.output for s in stages], ["HELLO"])

    def test_renamed_parameters_are_still_recognised(self) -> None:
        src = decoder("d", a="zz1", b="zz2", c="zz3")
        src += f'd("{encode("PARAMS", 3, "|")}", 3, "|");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["PARAMS"])

    def test_a_different_key_decodes_correctly(self) -> None:
        for key in (1, 7, 60, 123, 255):
            with self.subTest(key=key):
                src = decoder() + f'dec("{encode("KEYED", key, ",")}", {key}, ",");\n'
                self.assertEqual([s.output for s in find_jscript_stages(src)], ["KEYED"])

    def test_a_different_delimiter_decodes_correctly(self) -> None:
        for delimiter in (",", "|", "<", ">", "[", ":", "~"):
            with self.subTest(delimiter=delimiter):
                src = decoder() + (
                    f'dec("{encode("DELIM", 5, delimiter)}", 5, "{delimiter}");\n'
                )
                self.assertEqual([s.output for s in find_jscript_stages(src)], ["DELIM"])

    def test_whitespace_and_multiline_formatting_are_tolerated(self) -> None:
        body = encode("SPACED", 11, ",")
        src = decoder() + f'dec(\n    "{body}",\n    11,\n    ","\n);\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["SPACED"])

    def test_every_qualifying_call_site_is_decoded(self) -> None:
        src = decoder()
        src += f'dec("{encode("ONE", 4, ",")}", 4, ",");\n'
        src += f'dec("{encode("TWO", 8, "|")}", 8, "|");\n'
        src += f'dec("{encode("THREE", 12, ">")}", 12, ">");\n'
        self.assertEqual(
            sorted(s.output for s in find_jscript_stages(src)), ["ONE", "THREE", "TWO"]
        )

    def test_source_location_is_recorded(self) -> None:
        src = decoder() + "\n\n" + f'dec("{encode("LOC", 2, ",")}", 2, ",");\n'
        stage = find_jscript_stages(src)[0]
        self.assertEqual(stage.line, src[: stage.offset].count("\n") + 1)
        self.assertEqual(src[stage.offset:stage.offset + 3], "dec")


class JScriptRejectionTests(unittest.TestCase):
    def test_a_malformed_numeric_token_rejects_the_call_site(self) -> None:
        """Not a token list; decoding it would invent a value."""
        src = decoder() + 'dec("72,NOPE,74", 0, ",");\n'
        self.assertEqual(find_jscript_stages(src), [])

    def test_a_dynamic_key_is_rejected(self) -> None:
        src = decoder() + f'dec("{encode("X", 5, ",")}", someKey, ",");\n'
        self.assertEqual(find_jscript_stages(src), [])

    def test_a_dynamic_delimiter_is_rejected(self) -> None:
        """Including one whose name is defined elsewhere as a string.

        A bare identifier must not be resolved here: the delimiter is only
        accepted as a literal at the call site, so a matcher that fell back to
        reading the variable would be interpreting the program.
        """
        src = decoder() + f'dec("{encode("XY", 5, ",")}", 5, sep);\n'
        self.assertEqual(find_jscript_stages(src), [])

        defined = 'var sep = ",";\n' + decoder()
        defined += f'dec("{encode("XY", 5, ",")}", 5, sep);\n'
        self.assertEqual(find_jscript_stages(defined), [])

    def test_a_dynamic_encoded_input_is_rejected(self) -> None:
        src = decoder() + 'dec(buildPayload(), 5, ",");\n'
        self.assertEqual(find_jscript_stages(src), [])

    def test_a_non_empty_concatenation_is_rejected(self) -> None:
        """Only provably-empty identifiers may be folded away."""
        src = "var pad = 'x';\n" + decoder()
        src += f'dec("{encode("X", 5, ",")}" + pad, 5, ",");\n'
        self.assertEqual(find_jscript_stages(src), [])

    def test_a_rewritten_variable_is_refused_rather_than_guessed(self) -> None:
        """A wrong decode is worse than a missing one.

        If a name holds one literal at its declaration and another by the time
        the decoder is called, reading the declaration would produce a
        confident value the program never uses -- indistinguishable, in the
        evidence, from a real result. Every form of second assignment is
        therefore disqualifying.
        """
        first = encode("FIRST", 5, ",")
        second = encode("SECOND", 5, ",")
        cases = {
            "reassigned": f'var p = "{first}";\np = "{second}";\ndec(p, 5, ",");\n',
            "declared twice": f'var p = "{first}";\nvar p = "{second}";\ndec(p, 5, ",");\n',
            "appended": f'var p = "{first}";\np += "99";\ndec(p, 5, ",");\n',
            "indexed write": f'var p = "{first}";\np[0] = "9";\ndec(p, 5, ",");\n',
        }
        for label, tail in cases.items():
            with self.subTest(case=label):
                self.assertEqual(find_jscript_stages(decoder() + tail), [])

    def test_a_variable_assigned_exactly_once_is_usable(self) -> None:
        """The refusal above must not swallow the ordinary case."""
        payload = encode("SINGLE", 5, ",")
        src = decoder() + f'var p = "{payload}";\ndec(p, 5, ",");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["SINGLE"])

    def test_a_similarly_named_variable_does_not_disqualify(self) -> None:
        """Matching must be on the name, not on a substring of one."""
        payload = encode("EXACT", 5, ",")
        src = decoder() + f'var p = "{payload}";\nvar padding = "x";\npadding = "y";\ndec(p, 5, ",");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["EXACT"])

    def test_decoy_numeric_strings_are_not_decoded(self) -> None:
        """A number list that no decoder consumes stays untouched."""
        src = 'var version = "1,2,3,4";\nvar ports = "80,443,8080";\n'
        self.assertEqual(find_jscript_stages(src), [])
        self.assertEqual(deobfuscate(src), [])

    def test_no_matching_decoder_yields_nothing(self) -> None:
        src = 'function notADecoder(a, b, c) { return a + b + c; }\nnotADecoder("1,2", 3, ",");\n'
        self.assertEqual(find_jscript_stages(src), [])


class CoercionSemanticsTests(unittest.TestCase):
    def test_whitespace_around_a_token_is_ignored(self) -> None:
        """JScript ToNumber trims; Python int() happens to agree, but the
        empty-token and non-numeric cases below do not, so all three are
        pinned together."""
        src = decoder() + 'dec(" 72 , 73 ", 0, ",");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["HI"])

    def test_an_empty_token_is_zero_not_an_error(self) -> None:
        """A trailing delimiter yields chr(key), exactly as JScript does."""
        src = decoder() + 'dec("72,73,", 0, ",");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["HI\x00"])

    def test_values_are_taken_modulo_2_16(self) -> None:
        """String.fromCharCode takes a UTF-16 code unit."""
        src = decoder() + 'dec("65536,65601", 0, ",");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["\x00A"])


class PowerShellMatcherTests(unittest.TestCase):
    def _script(self, text: str, key: int) -> str:
        tokens = ",".join(str(ord(c) ^ key) for c in text)
        return (
            f"$data='{tokens}';$k={key};$out='';\n"
            f"$parts=$data -split ',';\n"
            f"foreach($t in $parts){{$out=$out+[char]($t -bxor $k);}}\n"
        )

    def test_a_literal_bxor_reconstruction_is_decoded(self) -> None:
        stages = find_powershell_stages(self._script("PS-STAGE", 34))
        self.assertEqual([s.output for s in stages], ["PS-STAGE"])
        self.assertEqual(stages[0].kind, POWERSHELL_XOR)

    def test_a_literal_key_operand_is_accepted(self) -> None:
        script = (
            "$data='115,114,113';\n"
            "$parts=$data -split ',';\n"
            "foreach($t in $parts){$o=$o+[char]($t -bxor 55);}\n"
        )
        self.assertEqual([s.output for s in find_powershell_stages(script)], ["DEF"])

    def test_the_offset_names_the_assignment_not_an_earlier_lookalike(self) -> None:
        """Provenance must point at the bytes that produced the value.

        The same digits can appear earlier in a comment or another string;
        searching for the token text would report that position instead, and
        a reader following it would find nothing that explains the result.
        """
        tokens = ",".join(str(ord(c) ^ 34) for c in "AB")
        script = (
            f"# decoy {tokens}\n"
            f"$a='{tokens}';$k=34;\n"
            "$parts=$a -split ',';\n"
            "foreach($t in $parts){$o=$o+[char]($t -bxor $k);}\n"
        )
        stage = find_powershell_stages(script)[0]
        self.assertEqual(stage.line, 2)
        self.assertEqual(script[stage.offset:stage.offset + len(tokens)], tokens)
        self.assertEqual(script[stage.offset - 1], "'")

    def test_ambiguous_keys_are_rejected(self) -> None:
        """Two candidate keys: which applies is not determined by the text."""
        script = (
            "$data='1,2,3';$a=7;$b=9;\n"
            "$parts=$data -split ',';\n"
            "foreach($t in $parts){$x=($t -bxor $a);$y=($t -bxor $b);}\n"
        )
        self.assertEqual(find_powershell_stages(script), [])

    def test_no_key_yields_nothing(self) -> None:
        self.assertEqual(find_powershell_stages("$data='1,2,3';\n"), [])


class RecursionAndBoundTests(unittest.TestCase):
    def test_a_nested_stage_is_reached(self) -> None:
        """A JScript stage decoding to a PowerShell stage, and both recorded."""
        inner = "$data='" + ",".join(str(ord(c) ^ 34) for c in "NESTED-OK") + "';$k=34;\n"
        inner += "$parts=$data -split ',';\n"
        inner += "foreach($t in $parts){$o=$o+[char]($t -bxor $k);}\n"
        src = decoder() + f'dec("{encode(inner, 46, "<")}", 46, "<");\n'
        stages = deobfuscate(src)
        kinds = [s.kind for s in stages]
        self.assertIn(JSCRIPT_XOR, kinds)
        self.assertIn(POWERSHELL_XOR, kinds)
        self.assertIn("NESTED-OK", [s.output for s in stages])
        self.assertEqual(max(s.depth for s in stages), 1)

    def test_recursion_stops_at_the_depth_bound(self) -> None:
        """A chain deeper than the bound is followed exactly to the bound."""
        chain = "TERMINAL-VALUE"
        # Each layer decodes to a whole decoder plus its own call site, so the
        # nesting is real rather than a string that merely looks nested.
        for key in (11, 13, 17, 19, 23, 29):
            chain = decoder() + f'dec("{encode(chain, key, ",")}", {key}, ",");\n'

        stages = deobfuscate(chain)
        depths = sorted({s.depth for s in stages})

        self.assertEqual(depths, list(range(MAX_DEPTH)))
        self.assertEqual(len(stages), MAX_DEPTH)
        # The chain is longer than the bound, so the innermost value is out of
        # reach -- which is the bound doing its job, not a decode failure.
        self.assertNotIn("TERMINAL-VALUE", [s.output for s in stages])

    def test_a_repeated_output_is_not_recorded_twice(self) -> None:
        """Cycle detection is by content digest, not by shape."""
        payload = encode("SAME", 6, ",")
        src = decoder() + f'dec("{payload}", 6, ",");\n' + f'dec("{payload}", 6, ",");\n'
        self.assertEqual([s.output for s in deobfuscate(src)], ["SAME"])


class GenericityTests(unittest.TestCase):
    """No knowledge of any particular artifact may live in production."""

    ORACLE = (
        "smartmaket", "AA1789FF", "Fattura981033956",
        "GaYHJ7mHg1DKQlXezhTy8NwK0YN9ZGDm0ueVP3hviz",
    )

    def test_production_contains_no_sample_or_oracle_constant(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "orbit"
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for term in self.ORACLE:
                with self.subTest(path=path.name, term=term):
                    self.assertNotIn(term, text)

    def test_no_arbitrary_execution_primitive_is_used(self) -> None:
        from pathlib import Path

        import orbit.runtime.analysis_deobfuscate as module

        text = Path(module.__file__).read_text(encoding="utf-8")
        for token in ("eval(", "exec(", "__import__", "subprocess", "socket",
                      "urllib", "os.system", "popen"):
            with self.subTest(token=token):
                self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()


class RuntimeIntegrationTests(unittest.TestCase):
    """The pass as ANALYSIS runs it: once, before the model, as evidence."""

    def _runtime(self, source_text: str):
        import tempfile
        from pathlib import Path

        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime, acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmpdir = tempfile.TemporaryDirectory(prefix="orbit-deobf-")
        self.addCleanup(tmpdir.cleanup)
        tmp = Path(tmpdir.name)
        artifact = tmp / "artifact.js"
        artifact.write_text(source_text, encoding="utf-8")
        source = acquire_analysis_source(artifact, tmp / "owned")
        store = EvidenceStore(root=tmp / "evidence")
        runtime = AnalysisRuntime(backend=None, source=source, evidence_store=store)
        self.addCleanup(runtime.close)
        return runtime, store

    def _fixture(self) -> str:
        return decoder() + f'dec("{encode("SYNTHETIC-STAGE", 17, ",")}", 17, ",");\n'

    def test_each_stage_becomes_evidence_with_full_provenance(self) -> None:
        runtime, store = self._runtime(self._fixture())
        self.assertEqual(len(runtime.transform_stages), 1)
        stage, record = runtime.transform_stages[0]

        meta = record.metadata
        self.assertEqual(meta["analysis_source_sha256"], runtime.source.sha256)
        self.assertEqual(meta["transform_kind"], JSCRIPT_XOR)
        self.assertEqual(meta["transform_key"], 17)
        self.assertEqual(meta["transform_delimiter"], ",")
        self.assertEqual(meta["transform_depth"], 0)
        self.assertEqual(meta["input_sha256"], stage.input_sha256)
        self.assertEqual(meta["output_sha256"], stage.output_sha256)
        self.assertIn("transform_line", meta)
        self.assertIn("transform_offset", meta)
        # The digest recorded is the digest of what was stored.
        self.assertEqual(
            stage.output_sha256,
            hashlib.sha256(stage.output.encode("utf-8")).hexdigest(),
        )

    def test_transform_evidence_re_attests(self) -> None:
        runtime, store = self._runtime(self._fixture())
        _stage, record = runtime.transform_stages[0]
        self.assertEqual(store.reattest_exact(record.evidence_id), "SYNTHETIC-STAGE")

    def test_the_model_receives_references_not_a_transcript(self) -> None:
        runtime, _store = self._runtime(self._fixture())
        preamble = runtime.messages[-1]["content"]
        _stage, record = runtime.transform_stages[0]
        self.assertIn(record.evidence_id, preamble)
        self.assertIn("evidence:<evidence_id>", preamble)

    def test_the_pass_runs_once_per_snapshot(self) -> None:
        """Not rescanned on every step: guarded by the snapshot digest."""
        runtime, _store = self._runtime(self._fixture())
        before = list(runtime.transform_stages)
        for _ in range(3):
            runtime._run_transform_preflight()
        self.assertEqual(runtime.transform_stages, before)

    def test_an_artifact_with_nothing_to_decode_is_unchanged(self) -> None:
        """Byte-identical to a session built before this existed."""
        runtime, _store = self._runtime("var x = 1;\nconsole.log(x);\n")
        self.assertEqual(runtime.transform_stages, [])
        self.assertEqual(runtime.transform_appendix(), "")
        self.assertEqual([m["role"] for m in runtime.messages], ["system", "user"])

    def test_the_appendix_renders_exact_output_and_provenance(self) -> None:
        runtime, _store = self._runtime(self._fixture())
        appendix = runtime.transform_appendix()
        _stage, record = runtime.transform_stages[0]

        self.assertIn("## Deterministic transformations", appendix)
        self.assertIn("SYNTHETIC-STAGE", appendix)
        self.assertIn(record.evidence_id, appendix)
        self.assertIn(_stage.output_sha256, appendix)
        self.assertIn("key 17", appendix)

    def test_a_decoded_uri_appears_verbatim_even_from_a_long_stage(self) -> None:
        """A truncated indicator is not an indicator."""
        uri = "http://synthetic.invalid/path?token=ABC123"
        payload = ("filler " * 120) + uri
        source = decoder() + f'dec("{encode(payload, 23, ",")}", 23, ",");\n'
        runtime, _store = self._runtime(source)

        appendix = runtime.transform_appendix()
        stage, _record = runtime.transform_stages[0]
        self.assertGreater(len(stage.output), 400)
        self.assertNotIn(payload, appendix)          # not inlined whole
        self.assertIn(f"decoded URI: {uri}", appendix)  # but the URI survives

    def test_the_appendix_labels_nothing(self) -> None:
        """What a decoded string means is the analysis's conclusion."""
        uri = "http://synthetic.invalid/x"
        source = decoder() + f'dec("{encode(uri, 29, ",")}", 29, ",");\n'
        runtime, _store = self._runtime(source)
        appendix = runtime.transform_appendix().lower()
        for word in ("c2", "command and control", "malicious", "payload", "attacker"):
            with self.subTest(word=word):
                self.assertNotIn(word, appendix)


class WrongDecodeRefusalTests(unittest.TestCase):
    """The failure that matters most: a confident value the program never uses."""

    def test_a_rewritten_empty_string_concat_is_refused(self) -> None:
        """Folding away a concatenation is sound only if the name stays empty.

        Reading the declaration of a variable that is later made non-empty
        decodes a truncated prefix -- and a plausible-looking partial result
        is worse than no result, because nothing marks it as partial.
        """
        payload = encode("SAFE", 5, ",")
        src = (
            'var e = "";\n'
            'e = ",112,103,99,110";\n'
            + decoder()
            + f'dec("{payload}" + e, 5, ",");\n'
        )
        self.assertEqual(find_jscript_stages(src), [])

    def test_a_stable_empty_string_concat_is_still_folded(self) -> None:
        payload = encode("KEEP", 5, ",")
        src = 'var pad = "";\n' + decoder() + f'dec("{payload}" + pad, 5, ",");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["KEEP"])

    def test_a_non_decoder_that_merely_splits_and_xors_is_not_matched(self) -> None:
        """`fromCharCode` and the XOR must be the same expression.

        A checksum helper splits its input, folds it with `^ seed`, and
        separately builds a character from the result. Both conditions hold
        independently while the function decodes nothing at all.
        """
        helper = (
            "function mix(data, seed, sep){var acc=0;var p=data.split(sep);"
            "for(var i=0;i<p.length;i++){acc=acc^seed;}"
            "return String.fromCharCode(65+acc);}\n"
        )
        self.assertEqual(find_jscript_stages(helper + 'mix("10,20,30", 7, ",");\n'), [])


class MalformedInputTests(unittest.TestCase):
    def test_a_malformed_escape_does_not_discard_every_stage(self) -> None:
        """One bad byte must not disable the pass on hostile input.

        The escape appears in a string this pass has no interest in; raising
        out of the scan would lose every stage the artifact really determines.
        """
        good = decoder() + f'dec("{encode("OK", 5, ",")}", 5, ",");\n'
        for junk in (r'var j = "\uWXYZ";', r'var j = "\xZZ";', 'var j = "trailing\\";'):
            with self.subTest(junk=junk):
                stages = deobfuscate(junk + "\n" + good)
                self.assertEqual([s.output for s in stages], ["OK"])

    def test_a_valid_escape_is_still_resolved(self) -> None:
        src = decoder() + f'dec("{encode("AB", 0, ",")}", 0, "\\x2c");\n'
        self.assertEqual([s.output for s in find_jscript_stages(src)], ["AB"])


class PowerShellLinkageTests(unittest.TestCase):
    def test_a_list_the_operator_never_consumes_is_not_decoded(self) -> None:
        """Deciding which key applies is not deciding that a list is input.

        A version string or a port list XORed against a key found elsewhere
        produces noise, and noise recorded with the authority of arithmetic is
        worse than silence: it invites the analysis to interpret it.
        """
        tokens = ",".join(str(ord(c) ^ 34) for c in "REAL")
        script = (
            f"$payload='{tokens}';$k=34;$o='';\n"
            "$ports='1,2,3,4,5';\n"
            "$parts=$payload -split ',';\n"
            "foreach($t in $parts){$o=$o+[char]($t -bxor $k);}\n"
        )
        self.assertEqual([s.output for s in find_powershell_stages(script)], ["REAL"])

    def test_a_dead_code_key_decodes_nothing(self) -> None:
        script = "$versions='10,20,30';\n$x = 5 -bxor 7;\n"
        self.assertEqual(find_powershell_stages(script), [])


class ReportIntegrationTests(unittest.TestCase):
    """The appendix must reach the report, and cost no citation slots."""

    def _runtime(self, source_text: str):
        import tempfile
        from pathlib import Path

        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime, acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmpdir = tempfile.TemporaryDirectory(prefix="orbit-deobf-rep-")
        self.addCleanup(tmpdir.cleanup)
        tmp = Path(tmpdir.name)
        artifact = tmp / "artifact.js"
        artifact.write_text(source_text, encoding="utf-8")
        runtime = AnalysisRuntime(
            backend=None,
            source=acquire_analysis_source(artifact, tmp / "owned"),
            evidence_store=EvidenceStore(root=tmp / "evidence"),
        )
        self.addCleanup(runtime.close)
        return runtime

    def test_transform_records_do_not_consume_the_citation_budget(self) -> None:
        """The stage count comes from the artifact, so it must not be the
        artifact's to spend: a crafted file with many decoy call sites would
        otherwise fill the report's places before the analysis found anything
        of its own. The appendix renders every stage regardless.
        """
        from orbit.runtime.analysis_runtime import MAX_REPORT_EVIDENCE_RECORDS

        source = decoder() + "".join(
            f'dec("{encode(f"D{i}", 7, ",")}", 7, ",");\n'
            for i in range(MAX_REPORT_EVIDENCE_RECORDS + 4)
        )
        runtime = self._runtime(source)

        self.assertGreater(len(runtime.transform_stages), 0)
        self.assertEqual(runtime._reportable_records(), [])
        # Still fully rendered, so nothing is lost by excluding them.
        appendix = runtime.transform_appendix()
        for _stage, record in runtime.transform_stages:
            self.assertIn(record.evidence_id, appendix)

    def test_the_appendix_is_appended_to_the_report_text(self) -> None:
        """Evidence rendering, not model prose -- so it cannot be omitted."""
        from orbit.backend.base import ChatResult

        class _Backend:
            def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                            on_delta=None, on_progress=None):
                if on_delta:
                    on_delta("model prose about the artifact")
                return ChatResult(
                    content="model prose about the artifact", model="m",
                    finish_reason="stop", tool_calls=[], prompt_tokens=10,
                    completion_tokens=5, cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        uri = "http://synthetic.invalid/beacon?id=XYZ"
        runtime = self._runtime(decoder() + f'dec("{encode(uri, 29, ",")}", 29, ",");\n')
        runtime.backend = _Backend()
        # A finding of its own, so the report has something to cite.
        runtime.evidence_store.add(
            "execute_analysis", "an action finding",
            metadata={"tool_call_id": "c1", "user_turn_id": "t1",
                      "produced_by_phase": "analysis_action"},
        )

        report = runtime.report("summarise")
        self.assertIn("model prose about the artifact", report.text)
        self.assertIn("## Deterministic transformations", report.text)
        self.assertIn(uri, report.text)


class StageBudgetTests(unittest.TestCase):
    def test_a_crafted_artifact_cannot_grow_the_preamble_without_bound(self) -> None:
        from orbit.runtime.analysis_deobfuscate import MAX_STAGES

        src = decoder() + "".join(
            f'dec("{encode(f"D{i}", 7, ",")}", 7, ",");\n' for i in range(MAX_STAGES * 4)
        )
        self.assertEqual(len(deobfuscate(src)), MAX_STAGES)
