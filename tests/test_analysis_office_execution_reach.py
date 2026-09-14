"""Static execution-reach inside an Office auto-execution procedure.

OFFICE-EXTRACTED-VBA-EVIDENCE-CLOSURE-1, defect 4/5. The recognised
`Document_Open` handler's own body statically contains a `Shell` call, so the
entrypoint reaches an execution path -- a fact recovered from the exact VBA
source, not the model's memory. It is never a claim the macro ran or the
document was opened.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_ole import extract_office_vba  # noqa: E402
from orbit.runtime.analysis_vba_autoexec import (  # noqa: E402
    find_office_event_relationships,
    find_office_execution_reach,
)
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

DOC = ROOT / "workdir" / "samples" / (
    "99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc"
)
MODULE_SHA = "d034bd8381f4663af80a7f516cf104ae9fd2a49581379d6a7683bfdf1c8713f0"
STAGE_SHA = "f1fa67e3f558443e9604b0f0b26715dc8065c53c92db7acbab2e6d32874e5c21"
C2 = "http://185.189.58.222/x.exe"

WORD_INVENTORY = [("WordDocument", 100), ("Macros/VBA/ThisDocument", 200)]


def _rel(source: str):
    return find_office_event_relationships("ThisDocument", source, WORD_INVENTORY, "ev_m")


class DetectorTests(unittest.TestCase):
    def test_shell_in_autoexec_body_is_a_reach(self) -> None:
        src = "Private Sub Document_Open()\n    Shell (cmd), 0\nEnd Sub\n"
        reach = find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m")
        self.assertEqual(len(reach), 1)
        self.assertEqual((reach[0].procedure, reach[0].sink, reach[0].sink_line),
                         ("Document_Open", "Shell", 2))

    def test_run_method_is_a_reach(self) -> None:
        src = 'Private Sub Document_Open()\n wsh.Run "x"\nEnd Sub\n'
        reach = find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m")
        self.assertEqual([(r.sink) for r in reach], [".Run"])

    def test_no_execution_call_yields_no_reach(self) -> None:
        src = "Private Sub Document_Open()\n x = 1\nEnd Sub\n"
        self.assertEqual(find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m"), [])

    def test_shell_outside_the_autoexec_body_is_not_a_reach(self) -> None:
        """A Shell in a DIFFERENT procedure is not the entry's reach."""
        src = ("Private Sub Document_Open()\n x = 1\nEnd Sub\n"
               "Sub Other()\n Shell (y), 0\nEnd Sub\n")
        self.assertEqual(find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m"), [])

    def test_shell_in_a_comment_or_string_is_not_a_reach(self) -> None:
        for body in ("' Shell (cmd), 0", 'x = "Shell (cmd)"'):
            src = f"Private Sub Document_Open()\n {body}\nEnd Sub\n"
            self.assertEqual(
                find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m"), [],
                body,
            )

    def test_shell_as_a_variable_name_is_not_a_reach(self) -> None:
        src = "Private Sub Document_Open()\n myShell = 1\nEnd Sub\n"
        self.assertEqual(find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m"), [])

    def test_colon_packed_entry_does_not_borrow_next_procedure_sink(self) -> None:
        """A colon-packed/unclosed entry must not attribute a FOLLOWING
        procedure's Shell to itself (body bounded by the next declaration)."""
        src = ("Private Sub Document_Open() : x = 1 : End Sub\n"
               "Sub Helper()\n Shell \"evil\"\nEnd Sub\n")
        self.assertEqual(find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m"), [])

    def test_unclosed_entry_does_not_borrow_next_procedure_sink(self) -> None:
        src = ("Private Sub Document_Open()\n x = 1\n"
               "Sub Helper()\n Shell \"evil\"\nEnd Sub\n")
        self.assertEqual(find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m"), [])

    def test_sink_before_next_declaration_is_still_a_reach(self) -> None:
        """The bound does not hide a genuine in-body sink."""
        src = ("Private Sub Document_Open()\n Shell \"x\"\nEnd Sub\n"
               "Sub Helper()\n y = 1\nEnd Sub\n")
        reach = find_office_execution_reach("ThisDocument", src, _rel(src), "ev_m")
        self.assertEqual([(r.procedure, r.sink) for r in reach], [("Document_Open", "Shell")])


@unittest.skipUnless(DOC.exists(), "Office sample required")
class RealSampleTests(unittest.TestCase):
    def _runtime(self) -> AnalysisRuntime:
        data = DOC.read_bytes()
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "s.doc"
        p.write_bytes(data)
        rt = AnalysisRuntime(
            backend=None,
            source=AnalysisSource(
                snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(p),
            ),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )
        self.addCleanup(rt.close)
        return rt

    def test_module_and_stage_and_chain_before_plan(self) -> None:
        rt = self._runtime()
        # T1: extracted VBA source present before any model call.
        self.assertEqual(len(rt.office_modules), 1)
        module, _rec = rt.office_modules[0]
        self.assertEqual(
            hashlib.sha256(module.source.encode("utf-8", "surrogatepass")).hexdigest(),
            MODULE_SHA,
        )
        # T7/T8/T9: exact stage sha, C2, TEMP/PHfW, Start-Process.
        self.assertEqual(len(rt.transform_stages), 1)
        stage, _ = rt.transform_stages[0]
        self.assertEqual(stage.output_sha256, STAGE_SHA)
        self.assertIn(C2, stage.output)
        self.assertIn("PHfW.exe", stage.output)
        self.assertIn("Start-Process", stage.output)
        # T6: Document_Open -> Shell execution reach from exact source.
        self.assertEqual(
            [(r.procedure, r.sink) for r in rt.office_exec_reach],
            [("Document_Open", "Shell")],
        )

    def test_chain_is_in_report_grounding(self) -> None:
        rt = self._runtime()
        grounding = rt.deterministic_sections()
        self.assertIn("Document_Open", grounding)
        self.assertIn("reaches a Shell execution call", grounding)
        # The entrypoint event and the payload both present for the report.
        self.assertIn("document-open", grounding)
        self.assertIn(C2, grounding)
        self.assertIn("PHfW.exe", grounding)

    def test_grounding_does_not_claim_the_sink_argument_is_the_stage(self) -> None:
        """The detector finds an execution TOKEN, not its resolved argument, so
        grounding must not assert the call passes the decoded stage."""
        rt = self._runtime()
        grounding = rt.deterministic_sections().lower()
        self.assertNotIn("the executed command is the decoded stage", grounding)
        self.assertNotIn("the command it passes is the decoded stage", grounding)

    def test_chain_is_in_plan_bootstrap(self) -> None:
        rt = self._runtime()
        plan_text = "\n".join(
            m["content"] for m in rt.messages if m.get("role") == "user"
        )
        self.assertIn("reaches a Shell execution call", plan_text)
        self.assertIn("Document_Open", plan_text)


class ContractTests(unittest.TestCase):
    """T4/T5: the sandbox/evidence contract is unchanged and complete."""

    def test_no_evidence_reading_tool_is_exposed(self) -> None:
        # T4/T5: the sandbox offers read_file only; no get_evidence/read_evidence
        # is advertised, so the qualified path never needs a nonexistent tool.
        from orbit.runtime.analysis_runtime import (
            ANALYSIS_SYSTEM_PROMPT,
            ANALYSIS_TOOL_SCHEMA,
        )
        prompt = ANALYSIS_SYSTEM_PROMPT.lower()
        self.assertNotIn("get_evidence", prompt)
        self.assertNotIn("read_evidence", prompt)
        schema = str(ANALYSIS_TOOL_SCHEMA).lower()
        self.assertNotIn("get_evidence", schema)
        self.assertNotIn("read_evidence", schema)


@unittest.skipUnless(
    (ROOT / "workdir" / "samples" / "IBAN.js").exists(), "IBAN sample required"
)
class UnrelatedCorpusTests(unittest.TestCase):
    """T12: a non-Office sample gets no execution-reach and is unchanged."""

    def test_non_office_sample_has_no_office_exec_reach(self) -> None:
        data = (ROOT / "workdir" / "samples" / "IBAN.js").read_bytes()
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "IBAN.js"
        p.write_bytes(data)
        rt = AnalysisRuntime(
            backend=None,
            source=AnalysisSource(
                snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(p),
            ),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )
        self.addCleanup(rt.close)
        self.assertEqual(rt.office_exec_reach, [])
        self.assertEqual(rt.office_modules, [])
        # The IBAN decode stage is still present and unchanged.
        self.assertEqual(len(rt.transform_stages), 1)


if __name__ == "__main__":
    unittest.main()
