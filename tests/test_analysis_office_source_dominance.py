"""An extracted VBA module source is authoritative; re-reading it adds nothing.

OFFICE-EXTRACTED-VBA-EVIDENCE-CLOSURE-1, defect 2. The Office preflight extracts
the macro source exactly, as module evidence. A sandbox action that only
reproduces that extracted text establishes nothing new -- the same "establishes
nothing new" suppression the deterministic stages already get, one source
further out -- so raw-binary re-reads to obtain source Orbit already holds are
not the preferred path. The raw binary itself is never an authority here, so
legitimate binary inspection is untouched, and a failed read (the UTF-8 decode
error on a binary .doc) is never suppressed -- it reaches the model as a failure.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    TRANSFORM_REACQUISITION,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
    _transform_reacquisition,
)
from orbit.runtime.analysis_sandbox import AnalysisResult, DerivedArtifact  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

MODULE_SOURCE = (
    "Private Sub Document_Open()\n"
    "    cmd = BuildCommand()\n"
    "    Shell (cmd), 0\n"
    "End Sub\n"
)


class _Module:
    def __init__(self, source: str) -> None:
        self.source = source


class _Rec:
    def __init__(self, eid: str) -> None:
        self.evidence_id = eid


def _result(stdout="", stderr="", **kw):
    return AnalysisResult(
        status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
        stdout=stdout, stderr=stderr, exit_status=0, duration_seconds=0.1, **kw,
    )


class ReacquisitionUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.office = [(_Module(MODULE_SOURCE), _Rec("ev_mod"))]

    def test_reproducing_module_source_is_suppressed(self) -> None:
        verdict = _transform_reacquisition(_result(MODULE_SOURCE), [], self.office)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict[1].evidence_id, "ev_mod")

    def test_module_plus_recomputable_property_is_suppressed(self) -> None:
        verdict = _transform_reacquisition(
            _result(f"{MODULE_SOURCE}\nLEN: {len(MODULE_SOURCE)}"), [], self.office
        )
        self.assertIsNotNone(verdict)

    def test_novel_semantic_output_stays_useful(self) -> None:
        self.assertIsNone(
            _transform_reacquisition(
                _result(f"{MODULE_SOURCE}\nSUBS: 1"), [], self.office
            )
        )

    def test_a_failed_binary_read_is_never_suppressed(self) -> None:
        """The UnicodeDecodeError path: a failure reaches the model as one."""
        self.assertIsNone(
            _transform_reacquisition(
                _result("", stderr="UnicodeDecodeError: 'utf-8' codec can't decode"),
                [], self.office,
            )
        )

    def test_without_office_modules_nothing_is_suppressed(self) -> None:
        self.assertIsNone(_transform_reacquisition(_result(MODULE_SOURCE), [], []))

    def test_an_artifact_alongside_output_defeats_the_proof(self) -> None:
        art = DerivedArtifact(name="x", size_bytes=1, sha256="d" * 64)
        self.assertIsNone(
            _transform_reacquisition(
                _result(MODULE_SOURCE, artifacts=(art,)), [], self.office
            )
        )


CTX = 8192


class _Backend:
    thinking = False

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.code = "import orbit_tools\nprint(orbit_tools.read_file(0, 999999))"

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _I:
            context_length = CTX
        return _I()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * 0.25), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        self.chat_calls.append({})
        if not kwargs.get("tools"):
            return self._r("noted", [])
        import json
        return self._r("run", [{"id": f"c{len(self.chat_calls)}", "type": "function",
                                "function": {"name": "execute_analysis",
                                             "arguments": json.dumps({"code": self.code})}}])

    def _r(self, content, calls):
        return ChatResult(content=content, model="m", finish_reason="stop", tool_calls=calls,
                          prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                          prompt_tokens_per_second=None, generation_tokens_per_second=None)

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


class ReadFileBinaryContractTests(unittest.TestCase):
    """T3: reading a binary artifact as UTF-8 yields an actionable error that
    names the extracted-source path, not a bare UnicodeDecodeError -- so the
    model stops re-reading the binary and uses the evidence instead. The bytes
    are never silently decoded, and UTF-8 text is unaffected."""

    def _tools(self):
        import types
        from orbit.runtime.analysis_tools_shim import ORBIT_TOOLS_SOURCE
        mod = types.ModuleType("orbit_tools")
        exec(compile(ORBIT_TOOLS_SOURCE, "orbit_tools.py", "exec"), mod.__dict__)
        return mod

    def test_binary_read_raises_actionable_error(self) -> None:
        import os
        import tempfile
        mod = self._tools()
        d = tempfile.mkdtemp()
        path = os.path.join(d, "b.doc")
        with open(path, "wb") as fh:
            fh.write(b"\xd0\xcf\x11\xe0binary-office-doc")
        mod._safe_path = lambda v: path
        with self.assertRaises(ValueError) as ctx:
            mod.read_file(path)
        msg = str(ctx.exception)
        self.assertIn("not UTF-8 text", msg)
        self.assertIn("evidence:<evidence_id>", msg)
        self.assertNotIn("�", msg)  # no silent replacement decode

    def test_utf8_text_still_reads(self) -> None:
        import os
        import tempfile
        mod = self._tools()
        d = tempfile.mkdtemp()
        path = os.path.join(d, "t.txt")
        with open(path, "wb") as fh:
            fh.write(b"macro source here")
        mod._safe_path = lambda v: path
        self.assertEqual(mod.read_file(path), "macro source here")


class ReacquisitionRuntimeTests(unittest.TestCase):
    def test_module_reproduction_suppressed_through_step(self) -> None:
        data = b"x\n"
        backend = _Backend()
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "a.doc"
        p.write_bytes(data)
        rt = AnalysisRuntime(
            backend=backend,
            source=AnalysisSource(snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
                                  size_bytes=len(data), original_path=str(p)),
            evidence_store=EvidenceStore(root=ws.root / "evidence"), workspace=ws,
        )
        self.addCleanup(rt.close)
        rec = rt.evidence_store.add("execute_analysis", MODULE_SOURCE,
                                    metadata={"produced_by_phase": "analysis_office"})
        rt.office_modules.append((_Module(MODULE_SOURCE), rec))
        before = rt.actions_executed
        with mock.patch.object(module, "execute_analysis",
                               lambda **kw: _result(MODULE_SOURCE)):
            step = rt.step("go")
        self.assertIsNotNone(step.suppressed_duplicate_of)
        self.assertFalse(step.action_executed)
        self.assertEqual(rt.actions_executed, before)
        tool_msgs = [m for m in rt.messages if m.get("role") == "tool"]
        self.assertIn(TRANSFORM_REACQUISITION.upper(), tool_msgs[-1]["content"])


if __name__ == "__main__":
    unittest.main()
