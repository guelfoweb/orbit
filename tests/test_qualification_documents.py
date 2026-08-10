from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from orbit.backend.base import ChatResult, TokenCount
from orbit.qualification.fixtures import load_fixture_set
from orbit.qualification.runner import QualificationRunner, RuntimeFixtureExecutor
from orbit.qualification.schema import DocumentEvidence, LifecycleOutcome, RunProvenance, Status
from orbit.qualification.validators import validate_observation
from orbit.runtime.full_document import identify_full_document_request


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = load_fixture_set(ROOT / "qualification/fixtures/documents-v1.json")


class DocumentBackend:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.calls += 1
        if self.mode == "inert":
            content = "ORBIT_INERT_DOCUMENT_OK"
        elif self.calls == 1:
            content = '{"command":"cat note.txt"}'
        else:
            content = "ORBIT_FULL_DOCUMENT_FIT_OK"
        return ChatResult(content, "fake", "stop", [], 100, 4, 0, 20.0, 8.0)

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        tokens = 8000 if self.mode == "oversized" else 100
        return TokenCount(tokens, 8192, "a" * 64, "b" * 64)

    def count_text_tokens(self, text):
        return TokenCount(max(1, len(text) // 4), 8192)


def execute(name: str):
    fixture = next(item for item in FIXTURES.fixtures if item.name == name)
    mode = name.removesuffix("_full_document").removesuffix("_full_document_phrase")
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        assert fixture.workspace is not None
        for item in fixture.workspace.files:
            (workdir / item.path).write_text(item.content, encoding="utf-8", newline="")
        observation = RuntimeFixtureExecutor(DocumentBackend(mode)).execute(fixture, workdir)
        return fixture, observation, validate_observation(fixture, observation, workdir=workdir)


class QualificationDocumentTests(unittest.TestCase):
    def test_oversized_document_fails_closed_without_inference(self) -> None:
        _fixture, observation, result = execute("oversized_full_document")
        self.assertEqual(result.status, Status.PASS)
        self.assertEqual([call.phase for call in observation.calls], ["route"])
        self.assertEqual((observation.tool_calls, observation.executed_tools), ((), ()))
        self.assertEqual(observation.document.coverage, "none")  # type: ignore[union-attr]
        self.assertIn("No document content was sent", observation.final_output)
        appended = replace(observation, final_output=observation.final_output + "\nA fabricated thesis.")
        wrong_phase = replace(observation, calls=(replace(observation.calls[0], phase="chat_final"),))
        self.assertEqual(validate_observation(_fixture, appended).reason.code, "partial_document_inference")
        self.assertEqual(validate_observation(_fixture, wrong_phase).reason.code, "document_phase_mismatch")

    def test_fit_document_has_complete_coverage_and_cleanup(self) -> None:
        _fixture, observation, result = execute("fit_full_document")
        self.assertEqual(result.status, Status.PASS)
        self.assertEqual([call.phase for call in observation.calls], ["route", "full_document"])
        self.assertTrue(observation.document.snapshot_clean)  # type: ignore[union-attr]
        self.assertIn("Document coverage: complete", observation.final_output)

    def test_inert_phrase_keeps_normal_chat_behavior(self) -> None:
        _fixture, observation, result = execute("inert_full_document_phrase")
        self.assertEqual(result.status, Status.PASS)
        self.assertEqual((observation.route, observation.final_output), ("CHAT", "ORBIT_INERT_DOCUMENT_OK"))
        self.assertIsNone(observation.document.coverage)  # type: ignore[union-attr]
        self.assertNotIn("full_document", [call.phase for call in observation.calls])
        unexpected = replace(observation, document=DocumentEvidence(None, True, None, None, True))
        self.assertEqual(validate_observation(_fixture, unexpected).reason.code, "document_analysis_mismatch")

    def test_partial_and_search_requests_do_not_enter_full_analysis(self) -> None:
        self.assertIsNone(identify_full_document_request("Read the first 20 lines of note.txt"))
        self.assertIsNone(identify_full_document_request("Search the whole document note.txt for alpha"))

    def test_missing_and_contradictory_document_evidence_fail_closed(self) -> None:
        fixture, observation, _result = execute("oversized_full_document")
        missing = validate_observation(fixture, replace(observation, document=None))
        missing_coverage = validate_observation(
            fixture, replace(observation, document=DocumentEvidence(None, False, None, None, True)))
        wrong_coverage = validate_observation(
            fixture, replace(observation, document=DocumentEvidence("complete", False, None, None, True)))
        contradictory = validate_observation(
            fixture, replace(observation, document=DocumentEvidence("none", False, 8192, 8192, True)))
        self.assertEqual((missing.status, missing.reason.code), (Status.TECHNICAL_STOP, "document_evidence_missing"))
        self.assertEqual(missing_coverage.status, Status.TECHNICAL_STOP)
        self.assertEqual(wrong_coverage.reason.code, "document_coverage_mismatch")
        self.assertEqual(contradictory.reason.code, "document_context_contradiction")

    def test_snapshot_cleanup_failure_is_rejected(self) -> None:
        fixture, observation, _result = execute("fit_full_document")
        dirty = DocumentEvidence("complete", True, None, None, False)
        changed = replace(observation, document=dirty, lifecycle=LifecycleOutcome(True, "clean"))
        self.assertEqual(validate_observation(fixture, changed).reason.code, "document_cleanup_failed")

    def test_unsupported_document_capability_is_not_applicable(self) -> None:
        fixture = FIXTURES.fixtures[0]
        profile_id = fixture.profiles[0]
        provenance = RunProvenance(
            1, FIXTURES.content_hash, None, profile_id, None, None, None, None, None, {}, {}, {})

        class NeverExecutor:
            def execute(self, fixture, workdir):
                raise AssertionError("not applicable fixture executed")

        with tempfile.TemporaryDirectory() as directory:
            run = QualificationRunner(
                FIXTURES,
                {"compatibility_profile": profile_id, "verified": True,
                 "capabilities": {"full_document_analysis": False}},
                provenance, NeverExecutor(), Path(directory),
            ).run((fixture.name,))
        self.assertEqual(run.fixtures[0].status, Status.NOT_APPLICABLE)


if __name__ == "__main__":
    unittest.main()
