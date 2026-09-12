"""Provenance-fidelity tests: a sandboxed static-analysis result must never be
represented as a network fetch (`kind: fetch`). The execute_analysis observation
leads with a `status:` line, which the content sniffer would otherwise read as a
fetch -- its kind is now decided by the tool, not the bytes.

This is a label-only change: identity, digest, raw_ref, status, admission,
enrichment and rendering are unchanged -- only the displayed `kind` differs.
"""
from __future__ import annotations

import hashlib
import unittest

from orbit.runtime.evidence import (
    ANALYSIS_ACTION_KIND,
    build_evidence_record,
    route_card,
)

# A realistic execute_analysis observation and raw-output record.
_OBS = "status: ok\nstdout:\ndecoded 42 bytes from the artifact\nstderr:\n"
_RAW = "status: ok\nstdout:\ndecoded 42 bytes from the artifact\nstderr:\n"
_MD = {
    "tool_call_id": "call_1",
    "user_turn_id": "turn_1",
    "producer_model_call_id": "m_1",
    "status": "ok",
}


def _rec(tool, content, extra=None):
    md = dict(_MD)
    if extra:
        md.update(extra)
    return build_evidence_record(tool, content, metadata=md)


class ExecuteAnalysisKindTests(unittest.TestCase):
    def test_t1_execute_analysis_never_fetch(self):
        rec = _rec("execute_analysis", _OBS)
        self.assertEqual(rec.kind, ANALYSIS_ACTION_KIND)
        self.assertNotEqual(rec.kind, "fetch")

    def test_t1_execute_analysis_raw_never_fetch(self):
        rec = _rec("execute_analysis_raw", _RAW, extra={"kind": "raw_action_output"})
        self.assertEqual(rec.kind, ANALYSIS_ACTION_KIND)
        self.assertNotEqual(rec.kind, "fetch")

    def test_status_line_does_not_force_fetch(self):
        # even an observation that is ONLY a status line is not a fetch
        rec = _rec("execute_analysis", "status: error\n")
        self.assertEqual(rec.kind, ANALYSIS_ACTION_KIND)

    def test_t2_genuine_fetch_url_still_fetch(self):
        # a real fetch (CHAT, outside ANALYSIS) keeps the fetch kind -- via the
        # tool name, with content that carries NO status line, so this pins the
        # `tool_name == "fetch_url"` branch, not the content heuristic.
        rec = _rec("fetch_url", "<html><body>hello</body></html>")
        self.assertEqual(rec.kind, "fetch")

    def test_status_heuristic_preserved_for_other_tools(self):
        # the status-line heuristic still applies to a non-analysis tool whose
        # output genuinely represents a fetch (e.g. a shell curl).
        rec = build_evidence_record(
            "exec_shell_full_command", "status: 200\nbody",
            metadata={**_MD, "command": "curl http://x"},
        )
        self.assertEqual(rec.kind, "fetch")


class ProvenanceFidelityTests(unittest.TestCase):
    """§3: only the label changes; identity/digest/raw_ref/status/enrichment are
    unchanged. Compared against what a `fetch`-labelled record carried."""

    def test_t6_identity_digest_rawref_unchanged(self):
        rec = _rec("execute_analysis", _OBS)
        self.assertEqual(
            rec.raw_sha256, hashlib.sha256(_OBS.encode("utf-8")).hexdigest()
        )
        self.assertTrue(rec.raw_ref.startswith("evidence:"))
        self.assertEqual(rec.raw_chars, len(_OBS))
        self.assertTrue(rec.evidence_id.startswith("ev_"))

    def test_status_from_metadata_not_content(self):
        # status comes from the metadata field, never forged from the content
        rec = _rec("execute_analysis", _OBS)
        self.assertEqual(rec.status, "ok")

    def test_no_shell_enrichment_added(self):
        # the label change must not add structured stdout/stderr fields that a
        # `fetch`-labelled record did not have (rendering unchanged).
        rec = _rec("execute_analysis", _OBS)
        self.assertNotIn("stdout_excerpt", rec.metadata)
        self.assertNotIn("stderr_excerpt", rec.metadata)
        self.assertNotIn("exit_code", rec.metadata)

    def test_t3_route_card_renders_new_label(self):
        rec = _rec("execute_analysis", _OBS)
        card = route_card(rec)
        self.assertIn(f"kind: {ANALYSIS_ACTION_KIND}", card)
        self.assertNotIn("kind: fetch", card)
        # identity still rendered
        self.assertIn("tool: execute_analysis", card)
        self.assertIn("raw_ref: evidence:", card)

    def test_t4_no_fetch_semantics_in_card(self):
        # the rendered card must not imply a network fetch happened
        rec = _rec("execute_analysis", _OBS)
        card = route_card(rec).lower()
        self.assertNotIn("fetch", card)
        self.assertNotIn("download", card)
        self.assertNotIn("http://", card)


class RuntimeIntegrationTests(unittest.TestCase):
    """The execute_analysis evidence a real analysis run records must carry the
    analysis_action kind end to end."""

    def test_recorded_action_evidence_kind(self):
        import tempfile
        from pathlib import Path
        from orbit.runtime.evidence import EvidenceStore

        tmp = tempfile.mkdtemp(prefix="orbit-kind-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        store = EvidenceStore(root=Path(tmp) / "ev")
        # mirror the runtime's action-evidence add (observation leads with status:)
        rec = store.add(
            "execute_analysis",
            _OBS,
            metadata={**_MD, "analysis_source_sha256": "deadbeef"},
        )
        raw = store.add(
            "execute_analysis_raw",
            _RAW,
            metadata={**_MD, "kind": "raw_action_output"},
        )
        self.assertEqual(rec.kind, ANALYSIS_ACTION_KIND)
        self.assertEqual(raw.kind, ANALYSIS_ACTION_KIND)
        self.assertNotEqual(rec.kind, "fetch")


if __name__ == "__main__":
    unittest.main()
