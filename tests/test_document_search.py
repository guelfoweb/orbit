from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, TokenCount
from orbit.runtime import ChatRuntime
from orbit.runtime.document_search import (
    DocumentSearchPlan,
    concept_document_search_answer,
    identify_document_search_request,
    parse_document_concept_verification,
    parse_document_search_plan,
    search_document_snapshot,
    stratified_language_samples,
)
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.file_tools import read_full_document_snapshot
from orbit.runtime.full_document import parse_full_document_snapshot
from orbit.runtime.tools import tool_definitions


def _chat_result(content: str, *, finish_reason: str = "stop") -> ChatResult:
    return ChatResult(
        content=content,
        model="fake",
        finish_reason=finish_reason,
        tool_calls=[],
        prompt_tokens=100,
        completion_tokens=10,
        cached_tokens=0,
        prompt_tokens_per_second=None,
        generation_tokens_per_second=None,
    )


class SearchBackend:
    def __init__(self, responses: list[ChatResult], *, prompt_tokens: int = 100, context_tokens: int = 8192) -> None:
        self.responses = list(responses)
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens
        self.calls = 0
        self.messages_by_call: list[list[dict[str, object]]] = []
        self.tools_by_call: list[object] = []

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.calls += 1
        self.messages_by_call.append(messages)
        self.tools_by_call.append(tools)
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        if on_delta is not None and result.content:
            on_delta(result.content)
        return result

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        return TokenCount(
            tokens=self.prompt_tokens,
            context_tokens=self.context_tokens,
            rendered_hash="a" * 64,
            token_hash="b" * 64,
        )

    def count_text_tokens(self, text):
        return TokenCount(tokens=max(1, len(text) // 4), context_tokens=self.context_tokens)


class DocumentSearchPrimitiveTests(unittest.TestCase):
    def _snapshot(self, content: str):
        temporary = tempfile.TemporaryDirectory()
        workdir = Path(temporary.name)
        (workdir / "note.txt").write_text(content, encoding="utf-8")
        snapshot = parse_full_document_snapshot(read_full_document_snapshot("note.txt", workdir=workdir))
        self.addCleanup(temporary.cleanup)
        assert snapshot is not None
        return workdir, snapshot

    def test_literal_matches_beginning_after_100_and_last_line(self) -> None:
        lines = ["target first\n", *[f"line {index}\n" for index in range(2, 151)], "target last\n"]
        _, snapshot = self._snapshot("".join(lines))

        result = search_document_snapshot(snapshot, mode="literal", terms=("target",), context_before=0, context_after=0)

        self.assertEqual(result.total_matches, 2)
        self.assertEqual(result.line_ranges, ("1-1", "151-151"))
        self.assertEqual(result.evidence_line_ranges, ("1-1", "151-151", "1", "151"))
        self.assertEqual(result.scan_coverage, "complete")
        self.assertEqual(result.semantic_coverage, "exact")

    def test_literal_absence_is_exact_complete_scan(self) -> None:
        _, snapshot = self._snapshot("alpha\nbeta\n")

        result = search_document_snapshot(snapshot, mode="literal", terms=("gamma",))

        self.assertEqual(result.total_matches, 0)
        self.assertEqual(result.file_coverage, "none")
        self.assertEqual(result.scan_coverage, "complete")
        self.assertEqual(result.semantic_coverage, "exact")

    def test_phrase_casefold_unicode_and_nfkc(self) -> None:
        _, snapshot = self._snapshot("ACCESS CONTROL\nCaffè sicuro\nＡＩ systems\n")

        phrase = search_document_snapshot(snapshot, mode="literal", terms=("access control",), whole_word=False)
        accent = search_document_snapshot(snapshot, mode="literal", terms=("CAFFÈ",))
        nfkc = search_document_snapshot(snapshot, mode="literal", terms=("AI",))

        self.assertEqual((phrase.total_matches, accent.total_matches, nfkc.total_matches), (1, 1, 1))

    def test_whole_word_does_not_match_substring(self) -> None:
        _, snapshot = self._snapshot("inferno infernale superinferno\n")

        word = search_document_snapshot(snapshot, mode="literal", terms=("inferno",), whole_word=True)
        substring = search_document_snapshot(snapshot, mode="literal", terms=("inferno",), whole_word=False)

        self.assertEqual(word.total_matches, 1)
        self.assertEqual(substring.total_matches, 2)

    def test_overlapping_windows_are_merged(self) -> None:
        _, snapshot = self._snapshot("one\ntarget\nmiddle\ntarget\nfive\n")

        result = search_document_snapshot(snapshot, mode="literal", terms=("target",), context_before=1, context_after=1)

        self.assertEqual(result.total_matches, 2)
        self.assertEqual(len(result.windows), 1)
        self.assertEqual(result.line_ranges, ("1-5",))

    def test_total_count_survives_window_truncation(self) -> None:
        content = "".join(f"target {index}\nfiller\n" for index in range(20))
        _, snapshot = self._snapshot(content)

        result = search_document_snapshot(
            snapshot,
            mode="literal",
            terms=("target",),
            context_before=0,
            context_after=0,
            max_windows=3,
        )

        self.assertEqual(result.total_matches, 20)
        self.assertEqual(len(result.windows), 3)
        self.assertTrue(result.results_truncated)

    def test_truncated_long_line_keeps_the_match_visible(self) -> None:
        _, snapshot = self._snapshot("a" * 8_000 + " TARGET " + "z" * 8_000 + "\n")

        result = search_document_snapshot(
            snapshot,
            mode="literal",
            terms=("target",),
            context_before=0,
            context_after=0,
        )

        self.assertEqual(result.total_matches, 1)
        self.assertEqual(len(result.windows), 1)
        self.assertTrue(result.windows[0].text_truncated)
        self.assertTrue(result.results_truncated)
        self.assertIn("TARGET", result.windows[0].text)
        self.assertGreater(result.windows[0].text_char_start, 0)

    def test_truncated_window_counts_only_visible_matches_as_returned(self) -> None:
        _, snapshot = self._snapshot("TARGET " + "x" * 8_000 + " TARGET\n")

        result = search_document_snapshot(
            snapshot,
            mode="literal",
            terms=("target",),
            context_before=0,
            context_after=0,
        )

        self.assertEqual(result.total_matches, 2)
        self.assertEqual(result.returned_match_count, 1)
        self.assertEqual(result.windows[0].match_count, 1)
        self.assertTrue(result.results_truncated)

    def test_concept_terms_are_deduplicated_before_search(self) -> None:
        _, snapshot = self._snapshot("damnation and hell\n")

        result = search_document_snapshot(snapshot, mode="concept", terms=("hell", "HELL", "damnation"))

        self.assertEqual(result.searched_terms, ("hell", "damnation"))
        self.assertEqual(result.total_matches, 2)
        self.assertEqual(result.semantic_coverage, "partial")

    def test_language_samples_are_stratified(self) -> None:
        _, snapshot = self._snapshot("".join(f"line {index}\n" for index in range(300)))

        samples = stratified_language_samples(snapshot, max_chars=80)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].start_line, 1)
        self.assertLess(samples[0].end_line, samples[1].start_line)
        self.assertGreater(samples[-1].end_line, 290)


class DocumentSearchPlanTests(unittest.TestCase):
    def _plan(self, **overrides) -> str:
        value = {
            "mode": "concept",
            "query_language": "it",
            "document_languages": ["en"],
            "language_confidence": "high",
            "terms_by_language": {"en": ["hell", "damnation", "underworld"]},
        }
        value.update(overrides)
        return json.dumps(value)

    def test_italian_question_english_document_plan(self) -> None:
        plan, error = parse_document_search_plan(self._plan())

        self.assertIsNone(error)
        assert plan is not None
        self.assertEqual(plan.query_language, "it")
        self.assertEqual(plan.document_languages, ("en",))
        self.assertEqual(plan.terms, ("hell", "damnation", "underworld"))

    def test_english_question_italian_document_plan(self) -> None:
        plan, error = parse_document_search_plan(
            self._plan(
                query_language="en",
                document_languages=["it"],
                terms_by_language={"it": ["inferno", "dannazione"]},
            )
        )

        self.assertIsNone(error)
        assert plan is not None
        self.assertEqual(plan.document_languages, ("it",))
        self.assertIn("dannazione", plan.terms)

    def test_multilingual_plan(self) -> None:
        plan, error = parse_document_search_plan(
            self._plan(
                document_languages=["it", "en"],
                terms_by_language={"it": ["dannazione"], "en": ["damnation"]},
            )
        )

        self.assertIsNone(error)
        assert plan is not None
        self.assertEqual(plan.document_languages, ("it", "en"))

    def test_uncertain_language_requires_english_fallback(self) -> None:
        plan, error = parse_document_search_plan(
            self._plan(
                document_languages=["it"],
                language_confidence="uncertain",
                terms_by_language={"it": ["inferno"]},
            )
        )

        self.assertIsNone(plan)
        self.assertEqual(error, "uncertain_plan_missing_english")

    def test_invalid_json_duplicate_keys_and_schema_fail_closed(self) -> None:
        values = (
            "not json",
            '{"mode":"concept","mode":"concept"}',
            json.dumps({"mode": "concept"}),
        )
        for value in values:
            with self.subTest(value=value):
                plan, error = parse_document_search_plan(value)
                self.assertIsNone(plan)
                self.assertIsNotNone(error)

    def test_regex_glob_path_and_shell_terms_are_rejected(self) -> None:
        for term in ("hell.*", "*.txt", "../secret", "rm -rf /"):
            with self.subTest(term=term):
                plan, error = parse_document_search_plan(self._plan(terms_by_language={"en": [term]}))
                self.assertIsNone(plan)
                self.assertEqual(error, "unsafe_plan_term")

    def test_more_than_twelve_terms_is_rejected(self) -> None:
        terms = [f"term{index}" for index in range(13)]

        plan, error = parse_document_search_plan(self._plan(terms_by_language={"en": terms}))

        self.assertIsNone(plan)
        self.assertEqual(error, "too_many_plan_terms")

    def test_cross_language_duplicate_terms_are_deduplicated(self) -> None:
        plan, error = parse_document_search_plan(
            self._plan(
                document_languages=["it", "en"],
                terms_by_language={"it": ["Inferno"], "en": ["inferno", "hell"]},
            )
        )

        self.assertIsNone(error)
        assert plan is not None
        self.assertEqual(plan.terms, ("Inferno", "hell"))

    def test_no_terms_after_validation_is_rejected(self) -> None:
        plan, error = parse_document_search_plan(self._plan(terms_by_language={"en": []}))

        self.assertIsNone(plan)
        self.assertEqual(error, "invalid_plan_terms")

    def test_verification_requires_existing_evidence_range(self) -> None:
        valid = json.dumps({"decision": "supported", "answer": "Relevant evidence.", "line_ranges": ["10-12"]})
        invalid = json.dumps({"decision": "supported", "answer": "Invented.", "line_ranges": ["99-100"]})

        accepted, accepted_error = parse_document_concept_verification(valid, valid_line_ranges=("10-12",))
        rejected, rejected_error = parse_document_concept_verification(invalid, valid_line_ranges=("10-12",))

        self.assertIsNone(accepted_error)
        self.assertIsNotNone(accepted)
        self.assertIsNone(rejected)
        self.assertEqual(rejected_error, "invalid_verification_ranges")

    def test_verification_accepts_an_exact_matched_line(self) -> None:
        content = json.dumps({"decision": "supported", "answer": "Relevant evidence.", "line_ranges": ["10"]})

        accepted, error = parse_document_concept_verification(
            content,
            valid_line_ranges=("8-12", "10"),
        )

        self.assertIsNone(error)
        self.assertIsNotNone(accepted)


class DocumentSearchRoutingTests(unittest.TestCase):
    def test_clear_literal_and_concept_requests_are_distinguished(self) -> None:
        cases = {
            "Compare esattamente la parola inferno in note.txt?": ("literal", "inferno"),
            "Quante volte compare la frase access control in note.txt?": ("literal", "access control"),
            "Does note.txt contain exactly the word inferno?": ("literal", "inferno"),
            "Nel testo note.txt si parla di inferno?": ("concept", None),
            "Does the document note.txt discuss cybersecurity?": ("concept", None),
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                intent = identify_document_search_request(prompt)
                self.assertIsNotNone(intent)
                assert intent is not None
                self.assertEqual((intent.mode, intent.literal_term), expected)

    def test_range_full_mutation_and_non_document_requests_do_not_intercept(self) -> None:
        prompts = (
            "Mostra le prime 100 righe di note.txt.",
            "Read note.txt completely and summarize it.",
            "Find the word alpha in note.txt and replace it.",
            "Tell me about inferno.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIsNone(identify_document_search_request(prompt))

    def test_ambiguous_files_do_not_select_one(self) -> None:
        self.assertIsNone(identify_document_search_request("Does a.txt or b.txt discuss cybersecurity?"))


class DocumentSearchRuntimeTests(unittest.TestCase):
    def _runtime(self, workdir: Path, backend: SearchBackend) -> ChatRuntime:
        return ChatRuntime(
            backend=backend,
            system_prompt=None,
            evidence_store=EvidenceStore(workdir / ".evidence"),
        )

    def test_literal_path_scans_without_route_or_cat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("".join(["start\n"] * 120 + ["inferno\n"]), encoding="utf-8")
            backend = SearchBackend([])
            runtime = self._runtime(workdir, backend)
            tools: list[tuple[str, str]] = []

            result = runtime.ask_auto(
                "Compare esattamente la parola inferno in note.txt?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
                on_tool_call=lambda name, arguments: tools.append((name, arguments)),
            )

        self.assertEqual(backend.calls, 0)
        self.assertEqual(tools, [])
        self.assertIn("119-121", result.content)
        self.assertIn("scan_coverage=complete", result.content)

    def test_literal_missing_is_definitive_only_for_exact_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            runtime = self._runtime(workdir, SearchBackend([]))

            result = runtime.ask_auto(
                "Compare esattamente la parola inferno in note.txt?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertIn("does not occur in the complete document", result.content)
        self.assertIn("semantic_coverage=exact", result.content)

    def test_multilingual_concept_plan_and_positive_verification(self) -> None:
        plan = _chat_result(
            json.dumps(
                {
                    "mode": "concept",
                    "query_language": "it",
                    "document_languages": ["en"],
                    "language_confidence": "high",
                    "terms_by_language": {"en": ["hell", "damnation"]},
                }
            )
        )
        verification = _chat_result(
            json.dumps(
                {
                    "decision": "supported",
                    "answer": "The passage explicitly discusses damnation.",
                    "line_ranges": ["128-132"],
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            lines = ["neutral\n"] * 129 + ["eternal damnation is discussed here\n"] + ["neutral\n"] * 3
            (workdir / "note.txt").write_text("".join(lines), encoding="utf-8")
            backend = SearchBackend([plan, verification])
            runtime = self._runtime(workdir, backend)

            result = runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.tools_by_call, [None, None])
        self.assertIn("document_languages=[\"en\"]", result.content)
        self.assertIn("Supporting source ranges: 128-132", result.content)
        self.assertIn("eternal damnation is discussed here", result.content)
        self.assertNotIn("cat ", str(backend.messages_by_call))

    def test_lexical_false_positive_cannot_become_definitive_negative(self) -> None:
        plan = _chat_result(
            json.dumps(
                {
                    "mode": "concept",
                    "query_language": "it",
                    "document_languages": ["en"],
                    "language_confidence": "high",
                    "terms_by_language": {"en": ["hell"]},
                }
            )
        )
        verification = _chat_result(
            json.dumps(
                {
                    "decision": "lexical_only",
                    "answer": "The phrase is merely an exclamation.",
                    "line_ranges": ["1-1"],
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("what the hell\n", encoding="utf-8")
            runtime = self._runtime(workdir, SearchBackend([plan, verification]))

            result = runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertIn("did not establish", result.content)
        self.assertIn("does not exclude indirect formulations", result.content)

    def test_length_truncated_verification_cannot_support_a_concept(self) -> None:
        plan = _chat_result(
            json.dumps(
                {
                    "mode": "concept",
                    "query_language": "it",
                    "document_languages": ["en"],
                    "language_confidence": "high",
                    "terms_by_language": {"en": ["hell"]},
                }
            )
        )
        truncated = _chat_result(
            json.dumps(
                {
                    "decision": "supported",
                    "answer": "The passage discusses hell.",
                    "line_ranges": ["1"],
                }
            ),
            finish_reason="length",
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hell is discussed here\n", encoding="utf-8")
            runtime = self._runtime(workdir, SearchBackend([plan, truncated]))

            result = runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertEqual(result.finish_reason, "stop")
        self.assertNotIn("Supporting source ranges", result.content)
        self.assertIn("did not establish", result.content)
        self.assertIn("verification_finish_reason:length", result.content)

    def test_no_match_outside_context_is_prudent(self) -> None:
        plan = _chat_result(
            json.dumps(
                {
                    "mode": "concept",
                    "query_language": "it",
                    "document_languages": ["en"],
                    "language_confidence": "high",
                    "terms_by_language": {"en": ["hell", "damnation"]},
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("neutral\n" * 200, encoding="utf-8")
            backend = SearchBackend([plan], prompt_tokens=9000, context_tokens=8192)
            runtime = self._runtime(workdir, backend)

            result = runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 1)
        self.assertIn("No lexical references were found", result.content)
        self.assertIn("does not exclude indirect formulations", result.content)
        self.assertNotIn("does not discuss", result.content)

    def test_no_match_that_fits_escalates_once_to_complete_document(self) -> None:
        plan = _chat_result(
            json.dumps(
                {
                    "mode": "concept",
                    "query_language": "en",
                    "document_languages": ["it"],
                    "language_confidence": "high",
                    "terms_by_language": {"it": ["sicurezza informatica"]},
                }
            )
        )
        complete = _chat_result("The complete document does not discuss cybersecurity.")
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("testo neutro\n", encoding="utf-8")
            backend = SearchBackend([plan, complete], prompt_tokens=100, context_tokens=8192)
            runtime = self._runtime(workdir, backend)

            result = runtime.ask_auto(
                "Does note.txt discuss cybersecurity?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 2)
        self.assertIn("escalation=full_document", result.content)
        self.assertIn("semantic_coverage=complete", result.content)
        self.assertIn("Document coverage: complete", result.content)
        self.assertIn("does not discuss cybersecurity", result.content)

    def test_invalid_plan_fails_closed_without_shell_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("neutral\n", encoding="utf-8")
            backend = SearchBackend([_chat_result("not-json")])
            runtime = self._runtime(workdir, backend)

            result = runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 1)
        self.assertIn("invalid_plan_json", result.content)
        self.assertIn("scan_coverage=none", result.content)

    def test_nonexistent_symlink_and_fifo_fail_before_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "target.txt").write_text("inferno\n", encoding="utf-8")
            (workdir / "link.txt").symlink_to(workdir / "target.txt")
            os.mkfifo(workdir / "pipe.txt")
            for path in ("missing.txt", "link.txt", "pipe.txt"):
                with self.subTest(path=path):
                    backend = SearchBackend([])
                    runtime = self._runtime(workdir, backend)
                    result = runtime.ask_auto(
                        f"Compare esattamente la parola inferno in {path}?",
                        temperature=0,
                        max_tokens=256,
                        workdir=workdir,
                    )
                    self.assertEqual(backend.calls, 0)
                    self.assertIn("scan_coverage=none", result.content)

    def test_source_change_discards_literal_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("inferno\n", encoding="utf-8")
            runtime = self._runtime(workdir, SearchBackend([]))
            with patch(
                "orbit.runtime.environments.attest_full_document_snapshot",
                side_effect=[None, "source_identity_changed"],
            ):
                result = runtime.ask_auto(
                    "Compare esattamente la parola inferno in note.txt?",
                    temperature=0,
                    max_tokens=256,
                    workdir=workdir,
                )

        self.assertIn("source_identity_changed", result.content)
        self.assertNotIn("occurs 1 time", result.content)

    def test_ephemeral_snapshot_is_cleaned_on_success_and_planner_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("inferno\n", encoding="utf-8")
            store = EvidenceStore(workdir / ".evidence")
            literal_runtime = ChatRuntime(backend=SearchBackend([]), system_prompt=None, evidence_store=store)
            literal_runtime.ask_auto(
                "Compare esattamente la parola inferno in note.txt?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )
            self.assertEqual(list(store.root.glob("*.txt")), [])

            failed_runtime = ChatRuntime(
                backend=SearchBackend([_chat_result("invalid")]),
                system_prompt=None,
                evidence_store=store,
            )
            failed_runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )
            self.assertEqual(list(store.root.glob("*.txt")), [])

    def test_cancelled_planner_cleans_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("neutral\n", encoding="utf-8")
            store = EvidenceStore(workdir / ".evidence")
            runtime = ChatRuntime(
                backend=SearchBackend([_chat_result("", finish_reason="cancelled")]),
                system_prompt=None,
                evidence_store=store,
            )

            result = runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

        self.assertEqual(result.finish_reason, "cancelled")
        self.assertEqual(list(store.root.glob("*.txt")), [])

    def test_planner_timeout_cleans_snapshot(self) -> None:
        class TimeoutBackend(SearchBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                raise TimeoutError("timed out")

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("neutral\n", encoding="utf-8")
            store = EvidenceStore(workdir / ".evidence")
            runtime = ChatRuntime(backend=TimeoutBackend([]), system_prompt=None, evidence_store=store)

            with self.assertRaises(TimeoutError):
                runtime.ask_auto(
                    "Nel testo note.txt si parla di inferno?",
                    temperature=0,
                    max_tokens=256,
                    workdir=workdir,
                )

        self.assertEqual(list(store.root.glob("*.txt")), [])

    def test_cancelled_verification_cleans_snapshot(self) -> None:
        plan = _chat_result(
            json.dumps(
                {
                    "mode": "concept",
                    "query_language": "it",
                    "document_languages": ["en"],
                    "language_confidence": "high",
                    "terms_by_language": {"en": ["hell"]},
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hell\n", encoding="utf-8")
            store = EvidenceStore(workdir / ".evidence")
            runtime = ChatRuntime(
                backend=SearchBackend([plan, _chat_result("", finish_reason="cancelled")]),
                system_prompt=None,
                evidence_store=store,
            )

            result = runtime.ask_auto(
                "Nel testo note.txt si parla di inferno?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

            self.assertEqual(result.finish_reason, "cancelled")
            self.assertEqual(list(store.root.glob("*.txt")), [])

    def test_verification_timeout_cleans_snapshot(self) -> None:
        class VerificationTimeoutBackend(SearchBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                if self.calls == 1:
                    raise TimeoutError("timed out")
                return super().chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)

        plan = _chat_result(
            json.dumps(
                {
                    "mode": "concept",
                    "query_language": "it",
                    "document_languages": ["en"],
                    "language_confidence": "high",
                    "terms_by_language": {"en": ["hell"]},
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hell\n", encoding="utf-8")
            store = EvidenceStore(workdir / ".evidence")
            runtime = ChatRuntime(
                backend=VerificationTimeoutBackend([plan]),
                system_prompt=None,
                evidence_store=store,
            )

            with self.assertRaises(TimeoutError):
                runtime.ask_auto(
                    "Nel testo note.txt si parla di inferno?",
                    temperature=0,
                    max_tokens=256,
                    workdir=workdir,
                )

            self.assertEqual(list(store.root.glob("*.txt")), [])

    def test_reset_has_no_document_search_state_to_retain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("inferno\n", encoding="utf-8")
            store = EvidenceStore(workdir / ".evidence")
            runtime = ChatRuntime(backend=SearchBackend([]), system_prompt=None, evidence_store=store)
            runtime.ask_auto(
                "Compare esattamente la parola inferno in note.txt?",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
            )

            runtime.reset()

        self.assertEqual(runtime.messages, [])
        self.assertEqual(store.records, {})
        self.assertEqual(list(store.root.glob("*.txt")), [])

    def test_diagnostics_are_bounded_and_do_not_log_terms_or_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            log_path = workdir / "diag.jsonl"
            secret = "private-inferno-marker"
            (workdir / "note.txt").write_text(secret + "\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)},
                clear=False,
            ):
                runtime = self._runtime(workdir, SearchBackend([]))
                runtime.ask_auto(
                    "Compare esattamente la parola private-inferno-marker in note.txt?",
                    temperature=0,
                    max_tokens=256,
                    workdir=workdir,
                )
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            search_events = [event for event in events if event.get("event") == "document_search"]

        self.assertEqual(len(search_events), 1)
        serialized = json.dumps(search_events[0])
        self.assertNotIn(secret, serialized)
        self.assertEqual(search_events[0]["term_count"], 1)
        self.assertEqual(search_events[0]["total_matches"], 1)

    def test_production_schema_remains_unchanged(self) -> None:
        self.assertEqual(
            [definition["function"]["name"] for definition in tool_definitions()],
            [
                "exec_shell_full_command",
                "fetch_url",
                "list_directory",
                "system_info",
                "write_artifact",
            ],
        )
        self.assertNotIn("document_search", str(tool_definitions()))
        self.assertNotIn("verify_artifact", str(tool_definitions()))


if __name__ == "__main__":
    unittest.main()
