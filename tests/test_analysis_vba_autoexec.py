"""Tests for the deterministic Office/VBA auto-execution entrypoint recognizer.

STATIC source classification only -- nothing executes VBA or Office. Fixtures are
synthetic; the frozen-sample expectation (host=Word, ThisDocument.Document_Open)
is a validation constant established by inspection of the extracted source.
"""
from __future__ import annotations

import unittest

from orbit.runtime.analysis_vba_autoexec import (
    AUTOEXEC_CONTRACT,
    HOST_EXCEL,
    HOST_WORD,
    find_office_event_relationships,
    office_host,
)

SAMPLE = "workdir/samples/99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc"

WORD_INV = [("WordDocument", 4096), ("Macros/VBA/ThisDocument", 100)]
EXCEL_INV = [("Workbook", 4096), ("Macros/VBA/ThisWorkbook", 100)]
NO_HOST_INV = [("Foo", 10), ("Bar", 20)]


def _events(module, source, inv):
    return [
        (r.procedure, r.event)
        for r in find_office_event_relationships(module, source, inv, "ev_mod")
    ]


class HostDetectionTests(unittest.TestCase):
    def test_word_host(self):
        self.assertEqual(office_host(WORD_INV), HOST_WORD)

    def test_excel_host(self):
        self.assertEqual(office_host(EXCEL_INV), HOST_EXCEL)

    def test_no_host(self):
        self.assertIsNone(office_host(NO_HOST_INV))

    def test_ambiguous_both_hosts_refused(self):
        self.assertIsNone(office_host([("WordDocument", 1), ("Workbook", 1)]))


class FixtureTests(unittest.TestCase):
    """§11 fixtures, expected results specified independently."""

    def test_a1_thisdocument_document_open(self):
        src = "Private Sub Document_Open()\r\nEnd Sub\r\n"
        self.assertEqual(_events("ThisDocument", src, WORD_INV),
                         [("Document_Open", "document-open")])

    def test_a2_randomized_body(self):
        src = (
            "Private Sub Document_Open()\r\n"
            '    zQ = StrReverse("abc")\r\n'
            "    For i = 0 To 9: Next i\r\n"
            "End Sub\r\n"
        )
        self.assertEqual(_events("ThisDocument", src, WORD_INV),
                         [("Document_Open", "document-open")])

    def test_a3_wrong_module_refused(self):
        src = "Sub Document_Open()\r\nEnd Sub\r\n"
        self.assertEqual(_events("NewMacros", src, WORD_INV), [])

    def test_a4_name_in_comment_refused(self):
        src = "' Document_Open is mentioned here\r\nSub Harmless()\r\nEnd Sub\r\n"
        self.assertEqual(_events("ThisDocument", src, WORD_INV), [])

    def test_a4b_commented_out_declaration_refused(self):
        # A full declaration inside a comment (commented-out, or a comment that
        # quotes one) must not be recognised -- both the line-start anchor AND
        # the comment skip guard this.
        for src in (
            "'Private Sub Document_Open()\r\nSub Other()\r\nEnd Sub\r\n",
            "' see Private Sub Document_Open() below\r\nSub Other()\r\nEnd Sub\r\n",
            "  ' Private Sub Document_Open()\r\nEnd Sub\r\n",
        ):
            self.assertEqual(_events("ThisDocument", src, WORD_INV), [], src)

    def test_a5_name_in_string_refused(self):
        src = 'Sub X()\r\n    s = "Document_Open"\r\nEnd Sub\r\n'
        self.assertEqual(_events("ThisDocument", src, WORD_INV), [])

    def test_a6_similar_name_refused(self):
        for name in ("FooDocument_Open", "Document_OpenX", "Document_Opened"):
            src = f"Sub {name}()\r\nEnd Sub\r\n"
            self.assertEqual(_events("ThisDocument", src, WORD_INV), [], name)

    def test_a7_document_close(self):
        src = "Private Sub Document_Close()\r\nEnd Sub\r\n"
        self.assertEqual(_events("ThisDocument", src, WORD_INV),
                         [("Document_Close", "document-close")])

    def test_a8_autoopen_standard_module(self):
        # AutoOpen is a Word auto-macro: valid in a standard module, Word host.
        src = "Sub AutoOpen()\r\nEnd Sub\r\n"
        self.assertEqual(_events("NewMacros", src, WORD_INV),
                         [("AutoOpen", "document-open")])

    def test_a9_case_insensitive(self):
        src = "private sub document_open()\r\nend sub\r\n"
        self.assertEqual(_events("thisdocument", src, WORD_INV),
                         [("document_open", "document-open")])

    def test_a10_malformed_declaration_refused(self):
        # no parenthesis / not a Sub declaration
        for src in ("Document_Open\r\n", "Dim Document_Open\r\n",
                    "Call Document_Open\r\n"):
            self.assertEqual(_events("ThisDocument", src, WORD_INV), [], src)

    def test_function_not_treated_as_event_handler(self):
        # Office invokes handlers/auto-macros only as Subs; a Function of the
        # same name must NOT be classified an entrypoint.
        for src in (
            "Function Document_Open()\r\nEnd Function\r\n",
            "Public Function AutoOpen()\r\nEnd Function\r\n",
            "Property Get Document_Open()\r\nEnd Property\r\n",
        ):
            self.assertEqual(_events("ThisDocument", src, WORD_INV), [], src)

    def test_recognizer_is_total_on_malformed_input(self):
        # A malformed input must yield [] rather than raise -- the enrichment
        # must never crash a run.
        self.assertEqual(
            find_office_event_relationships(None, "x", WORD_INV, "ev"), [])
        self.assertEqual(
            find_office_event_relationships("ThisDocument", None, WORD_INV, "ev"), [])
        self.assertEqual(
            find_office_event_relationships("ThisDocument", "x", [("a",)], "ev"), [])
        self.assertEqual(
            find_office_event_relationships("ThisDocument", "x", [None], "ev"), [])

    def test_a11_duplicate_conservative_single(self):
        src = (
            "Private Sub Document_Open()\r\nEnd Sub\r\n"
            "Private Sub Document_Open()\r\nEnd Sub\r\n"
        )
        self.assertEqual(_events("ThisDocument", src, WORD_INV),
                         [("Document_Open", "document-open")])

    def test_a12_no_host_no_relationship(self):
        src = "Private Sub Document_Open()\r\nEnd Sub\r\n"
        self.assertEqual(_events("ThisDocument", src, NO_HOST_INV), [])

    def test_a13_workbook_open_excel(self):
        src = "Private Sub Workbook_Open()\r\nEnd Sub\r\n"
        self.assertEqual(_events("ThisWorkbook", src, EXCEL_INV),
                         [("Workbook_Open", "workbook-open")])

    def test_a13b_workbook_open_in_word_refused(self):
        src = "Private Sub Workbook_Open()\r\nEnd Sub\r\n"
        self.assertEqual(_events("ThisWorkbook", src, WORD_INV), [])

    def test_cross_document_open_in_excel_refused(self):
        src = "Private Sub Document_Open()\r\nEnd Sub\r\n"
        self.assertEqual(_events("ThisDocument", src, EXCEL_INV), [])

    def test_a14_benign_body_event_only_no_malicious_claim(self):
        src = (
            "Private Sub Document_Open()\r\n"
            '    MsgBox "hello"\r\n'
            "End Sub\r\n"
        )
        rels = find_office_event_relationships("ThisDocument", src, WORD_INV, "ev")
        self.assertEqual(len(rels), 1)
        r = rels[0]
        # the relationship states the event only, asserts no behaviour
        for banned in ("download", "malware", "payload", "network", "execute",
                       "persist", "C2"):
            self.assertNotIn(banned, r.description.lower())


class ProvenanceTests(unittest.TestCase):
    def test_relationship_carries_full_provenance(self):
        src = "Private Sub Document_Open()\r\nEnd Sub\r\n"
        r = find_office_event_relationships("ThisDocument", src, WORD_INV, "ev_abc")[0]
        self.assertEqual(r.host, HOST_WORD)
        self.assertEqual(r.module, "ThisDocument")
        self.assertEqual(r.procedure, "Document_Open")
        self.assertEqual(r.event, "document-open")
        self.assertEqual(r.module_evidence_id, "ev_abc")
        self.assertEqual(r.line, 1)
        self.assertEqual(r.contract, AUTOEXEC_CONTRACT)


class NonSampleHardeningTests(unittest.TestCase):
    def test_no_sample_specific_constants_in_production(self):
        text = open("src/orbit/runtime/analysis_vba_autoexec.py", encoding="utf-8").read()
        for banned in ("185.189", "PHfW", "99eb1d90", "d034bd83", "f1fa67e3"):
            self.assertNotIn(banned, text)

    def test_recognizer_imports_no_execution_primitive(self):
        import ast
        tree = ast.parse(open("src/orbit/runtime/analysis_vba_autoexec.py",
                              encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("subprocess", "os", "socket", "ctypes"):
            self.assertNotIn(forbidden, imported)


class FrozenSampleTests(unittest.TestCase):
    def test_frozen_document_open_recognized(self):
        from orbit.runtime.analysis_ole import extract_office_vba
        raw = open(SAMPLE, "rb").read()
        ex = extract_office_vba(raw)
        m = ex.modules[0]
        rels = find_office_event_relationships(
            m.name, m.source, ex.stream_inventory, "ev_mod"
        )
        self.assertEqual(len(rels), 1)
        r = rels[0]
        self.assertEqual(r.host, HOST_WORD)
        self.assertEqual(r.module, "ThisDocument")
        self.assertEqual(r.procedure, "Document_Open")
        self.assertEqual(r.event, "document-open")
        self.assertEqual(r.line, 256)


class RuntimeIntegrationTests(unittest.TestCase):
    def _runtime(self, sample_bytes, name="in.doc"):
        import tempfile
        from pathlib import Path
        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime, acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore
        tmp = tempfile.mkdtemp(prefix="orbit-autoexec-")
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

    def test_frozen_event_in_office_events_and_appendix(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        self.assertEqual(len(rt.office_events), 1)
        rel = rt.office_events[0]
        self.assertEqual(rel.procedure, "Document_Open")
        # references the module evidence, does not duplicate the command
        module_ids = {rec.evidence_id for _m, rec in rt.office_modules}
        self.assertIn(rel.module_evidence_id, module_ids)
        appendix = rt.office_events_appendix()
        self.assertIn("Document_Open", appendix)
        self.assertIn("does not establish", appendix)
        self.assertIn("Office auto-execution entrypoints", rt.deterministic_sections())

    def test_frozen_appendix_makes_no_behavioural_claim(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        appendix = rt.office_events_appendix()
        low = appendix.lower()
        # the ENTRYPOINT section itself must not assert behaviour
        for banned in ("downloadfile", "powershell", "start-process", "185.189"):
            self.assertNotIn(banned, low)
        # nor may it claim the macro actually ran / the victim opened the file
        for banned in ("was executed", "victim opened", "when the victim",
                       "the macro ran", "was run"):
            self.assertNotIn(banned, low)
        # it MUST carry the narrow-semantics disclaimer
        self.assertIn("when macros are permitted", low)
        self.assertIn("does not establish", low)

    def test_non_ole_no_events(self):
        rt = self._runtime(b"var x = 1;\n", name="a.js")
        self.assertEqual(rt.office_events, [])
        self.assertEqual(rt.office_events_appendix(), "")


if __name__ == "__main__":
    unittest.main()
