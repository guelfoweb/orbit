"""Live wiring for the grounded finalization handoff.

An evidence-backed tool workflow answers from its durable evidence in a fresh
session rather than from the conversation that produced it. These cover the
transition through the real answer() path: that the finalizer is reached
automatically, that the saturated investigation session is not carried into it,
and that the evidence is re-attested rather than trusted.

The trigger is narrow: a turn is rescued only once the ordinary window has been
refused admission, so a healthy turn keeps the ordinary path and every recovery
behaviour that comes with it.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from orbit.runtime.chat import ChatRuntime
from orbit.runtime.completion_budget import FINALIZATION_FINAL_MAX_TOKENS
from orbit.runtime.environments import FinalFromToolEnvironment, TransportEnvironment
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.full_document import FILE_DISPLAY_MARKER
from orbit.runtime.finalization import (
    BundleEntry,
    admit_finalization,
    deduplicate_evidence,
    entries_from_store,
)
from orbit.backend.base import ChatResult, TokenCount


ORACLE_TERMS = (
    "6000",
    "Sleep",
    "Invoke-RestMethod",
    "smartmaket",
    "AA1789FF",
    "Win32_Process",
    "winmgmts",
    "ShowWindow",
    "iex",
    "irm",
    "cimv2",
)


def _result(content: str, *, finish_reason: str = "stop") -> ChatResult:
    return ChatResult(
        content=content,
        model="fake",
        finish_reason=finish_reason,
        tool_calls=[],
        prompt_tokens=1,
        completion_tokens=1,
        cached_tokens=0,
        prompt_tokens_per_second=None,
        generation_tokens_per_second=None,
    )


class RecordingBackend:
    """Backend that counts tokens exactly and records how it was called."""

    def __init__(self, *, context_tokens: int, tokens_for: dict | None = None) -> None:
        self.context_tokens = context_tokens
        self._tokens_for = tokens_for or {}
        self.chat_calls: list[dict] = []
        self.resets = 0
        self.count_calls = 0

    def _tokens(self, messages) -> int:
        text = "".join(str(m.get("content", "")) for m in messages)
        # The bundle restates the task, so it matches the saturation needle too.
        # Price the bundle first: otherwise it is charged the saturated cost and
        # admission refuses a prompt that in fact fits.
        if "Verified evidence" in text and "Verified evidence" in self._tokens_for:
            return self._tokens_for["Verified evidence"]
        for needle in sorted(self._tokens_for, key=len, reverse=True):
            if needle and needle in text:
                return self._tokens_for[needle]
        if "" in self._tokens_for:
            return self._tokens_for[""]
        return max(1, len(text) // 4)

    def count_chat_tokens(self, messages, *, tools=None, thinking=False) -> TokenCount:
        self.count_calls += 1
        tokens = self._tokens(messages)
        digest = hashlib.sha256(
            "".join(str(m.get("content", "")) for m in messages).encode("utf-8")
        ).hexdigest()
        return TokenCount(
            tokens=tokens,
            context_tokens=self.context_tokens,
            rendered_hash=digest,
            token_hash=digest,
        )

    def reset_session_state(self) -> None:
        self.resets += 1

    def chat(self, messages, *, temperature, max_tokens, tools=None) -> ChatResult:
        self.chat_calls.append(
            {"messages": messages, "max_tokens": max_tokens, "tools": tools}
        )
        return _result("FINAL: grounded answer")

    def chat_stream(self, messages, *, temperature, max_tokens, on_delta, on_progress, tools=None) -> ChatResult:
        self.chat_calls.append(
            {"messages": messages, "max_tokens": max_tokens, "tools": tools}
        )
        on_delta("FINAL: grounded answer")
        return _result("FINAL: grounded answer")


def _store_with(tmp: str, payloads: list[str]) -> tuple[EvidenceStore, list[str]]:
    store = EvidenceStore(Path(tmp) / "evidence")
    ids = []
    for index, payload in enumerate(payloads, start=1):
        record = store.add(
            "exec_shell_full_command",
            payload,
            metadata={
                "tool_call_id": f"call-{index}",
                "user_turn_id": f"turn-{index}",
                "produced_by_phase": "tool_call",
            },
        )
        ids.append(record.evidence_id)
    return store, ids


def _environment(backend, store, *, messages, context_tokens):
    runtime = ChatRuntime(
        backend=backend,
        system_prompt="system",
        messages=messages,
        context_tokens=context_tokens,
        evidence_store=store,
    )
    return FinalFromToolEnvironment(
        runtime=runtime, transport=TransportEnvironment(runtime=runtime)
    ), runtime






class LiveFinalizationWiringTests(unittest.TestCase):
    """The runtime must actually reach the PR #211 finalizer."""

    def test_saturated_session_reaches_grounded_finalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["verified payload " + "p" * 200])
            env, runtime = _environment(
                RecordingBackend(
                    context_tokens=1000,
                    tokens_for={"saturated": 990, "Verified evidence": 300},
                ),
                store,
                messages=[{"role": "user", "content": "analyse this saturated session"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNotNone(out, "finalizer was not reached from the live path")
            assert out is not None
            self.assertEqual(out.result.finish_reason, "stop")

    def test_no_evidence_means_no_grounded_finalization(self) -> None:
        """Without verified evidence there is nothing to finalize from.

        This is what keeps the phase scoped: an empty store yields None and the
        ordinary path runs, so a turn that never produced evidence is untouched.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence")
            env, _ = _environment(
                RecordingBackend(context_tokens=100_000),
                store,
                messages=[{"role": "user", "content": "short question"}],
                context_tokens=100_000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out, "diverted with no evidence to cite")

    def test_finalization_disables_tools_structurally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["verified payload " + "p" * 200])
            backend = RecordingBackend(
                context_tokens=1000,
                tokens_for={"saturated": 990, "Verified evidence": 300},
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertTrue(backend.chat_calls)
            for call in backend.chat_calls:
                self.assertIsNone(call["tools"], "tools reached FINAL_ONLY")

    def test_finalization_resets_saturated_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["verified payload " + "p" * 200])
            backend = RecordingBackend(
                context_tokens=1000,
                tokens_for={"saturated": 990, "Verified evidence": 300},
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertEqual(backend.resets, 1, "saturated KV was reused")

    def test_output_budget_stays_at_the_qualified_cap(self) -> None:
        self.assertEqual(FINALIZATION_FINAL_MAX_TOKENS, 4096)
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["verified payload " + "p" * 200])
            backend = RecordingBackend(
                context_tokens=32_000,
                tokens_for={"saturated": 31_990, "Verified evidence": 300},
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=32_000,
            )
            env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertEqual(backend.chat_calls[-1]["max_tokens"], 4096)

    def test_admission_is_exact_and_precedes_decode(self) -> None:
        """A bundle that cannot fit must refuse before any generation."""
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["x" * 40_000])
            backend = RecordingBackend(
                context_tokens=1000,
                tokens_for={"saturated": 990, "Verified evidence": 5000},
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out)
            self.assertEqual(backend.chat_calls, [], "decoded despite failed admission")

    def test_all_backend_count_shapes_are_understood(self) -> None:
        """A backend whose count shape is unread would disconnect the finalizer."""
        shape = FinalFromToolEnvironment._exact_token_total

        class Counted:
            tokens = 42

        self.assertEqual(shape(7), 7)
        self.assertEqual(shape(Counted()), 42)
        self.assertEqual(shape({"tokens": 9}), 9)
        self.assertIsNone(shape(None))
        self.assertIsNone(shape(True))
        self.assertIsNone(shape("120"))

    def test_missing_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence")
            env, _ = _environment(
                RecordingBackend(context_tokens=1000, tokens_for={"saturated": 990}),
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out)

    def test_uncountable_backend_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])

            class Uncountable(RecordingBackend):
                def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                    raise RuntimeError("tokenizer unavailable")

            backend = Uncountable(context_tokens=1000)
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out)
            self.assertEqual(backend.chat_calls, [])




class LiveCallPathReachabilityTests(unittest.TestCase):
    """The hook must be reachable through the real answer() entry point.

    Calling the helper directly proves it works; it does not prove anything
    calls it. These go through FinalFromToolEnvironment.answer so that severing
    the hook fails the suite.
    """

    def _answer(self, backend, store, *, context_tokens, use_tool_prompt=False):
        messages = [
            {"role": "user", "content": "analyse this saturated investigation"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "saturated investigation output",
            },
        ]
        env, _ = _environment(
            backend, store, messages=messages, context_tokens=context_tokens
        )
        return env.answer(
            temperature=0.0,
            max_tokens=2048,
            on_final_delta=None,
            on_progress=None,
            on_model_step=None,
            on_phase_start=None,
            loop=1,
            use_tool_prompt=use_tool_prompt,
            workdir=None,
        )

    def test_healthy_workflow_stays_on_the_ordinary_path(self) -> None:
        """A turn the ordinary path can complete must keep it.

        The ordinary path owns retry, repair, compact retry and the non-empty
        fallback. Diverting a healthy turn to the bundle would hand it to a
        thinner controller that has none of that, so the rescue must stay a
        rescue: with room to answer normally, nothing here should fire.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])
            backend = RecordingBackend(context_tokens=100_000)
            self._answer(backend, store, context_tokens=100_000)
            self.assertTrue(backend.chat_calls)
            sent = backend.chat_calls[-1]["messages"]
            self.assertGreater(
                len(sent), 1, "healthy turn was diverted to the evidence bundle"
            )
            self.assertEqual(backend.resets, 0, "healthy turn reset the session")


class ScopeAndResilienceTests(unittest.TestCase):
    """Paths with their own answer shaping keep it; failures stay recoverable."""

    def _web_search_store(self, tmp):
        store = EvidenceStore(Path(tmp) / "evidence")
        store.add(
            "exec_shell_full_command",
            "web_search_results: true\nresults: none",
            metadata={
                "tool_call_id": "call-1",
                "user_turn_id": "turn-1",
                "produced_by_phase": "tool_call",
            },
        )
        return store

    def test_web_final_view_keeps_its_curated_framing(self) -> None:
        """A search that found nothing keeps its curated framing.

        The web-final view exists to say honestly what the search returned, and
        the bundle cannot reproduce that shaping. Because the rescue only fires
        when the ordinary window cannot be completed, a healthy web-final turn
        keeps its own final call; if that window genuinely will not fit, the
        rescue still runs, since the alternative there is no answer at all.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = self._web_search_store(tmp)
            backend = RecordingBackend(
                context_tokens=100_000, tokens_for={"": 200, "Verified evidence": 300}
            )
            messages = [
                {"role": "user", "content": "search for something"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "exec_shell_full_command",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "no results"},
            ]
            env, runtime = _environment(
                backend, store, messages=messages, context_tokens=100_000
            )
            self.assertTrue(
                runtime._should_use_web_final_view(use_tool_prompt=False),
                "fixture no longer selects the web-final view",
            )
            env.answer(
                temperature=0.0,
                max_tokens=2048,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                use_tool_prompt=False,
                workdir=None,
            )
            sent = backend.chat_calls[-1]["messages"]
            self.assertGreater(
                len(sent), 1, "web-final curated view was replaced by the bundle"
            )
            self.assertEqual(backend.resets, 0, "web-final path reset the session")

    def test_backend_failure_does_not_cost_the_turn(self) -> None:
        """A failure inside finalization must fall back, not propagate.

        The session has already been reset by this point, so propagating would
        leave the caller with neither an answer nor a usable session.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])

            class Failing(RecordingBackend):
                def chat(self, messages, *, temperature, max_tokens, tools=None):
                    raise RuntimeError("backend exploded")

            backend = Failing(
                context_tokens=1000, tokens_for={"": 5000, "Verified evidence": 300}
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out, "backend failure propagated instead of declining")

    def test_streaming_backend_failure_does_not_cost_the_turn(self) -> None:
        """The streaming half of the guard needs its own test.

        Only overriding `chat` leaves `chat_stream` unguarded: the guard around
        it could be deleted and every test would still pass.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])

            class FailingStream(RecordingBackend):
                def chat_stream(
                    self, messages, *, temperature, max_tokens, on_delta,
                    on_progress, tools=None,
                ):
                    raise RuntimeError("stream exploded")

            backend = FailingStream(
                context_tokens=1000,
                tokens_for={"saturated": 990, "Verified evidence": 300},
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=lambda _chunk: None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out, "streaming failure propagated instead of declining")

    def test_failed_reset_declines_rather_than_finalizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])

            class BadReset(RecordingBackend):
                def reset_session_state(self):
                    raise RuntimeError("native client not loaded")

            backend = BadReset(
                context_tokens=1000, tokens_for={"": 5000, "Verified evidence": 300}
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out)
            self.assertEqual(backend.chat_calls, [], "generated after a failed reset")


class CoverageNoticeTests(unittest.TestCase):
    """The coverage caveat must reach the user exactly once.

    The fixture builds a genuine partial-coverage `read` record, so
    `response_prefix` is really non-empty: with an empty prefix the guard under
    test never executes and the assertions measure the answer instead of the
    caveat, which is how an earlier version of this test passed while the
    prefix was being emitted twice.
    """

    def _partial_read_store(self, tmp):
        body = "alpha\nbeta\ngamma\n"
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        raw = (
            f"{FILE_DISPLAY_MARKER}\n"
            "path: /workspace/a.txt\n"
            f"bytes: {len(body)}\n"
            "lines: 3\n"
            f"sha256: {digest}\n"
            "coverage: partial\n"
            "line_range: 1-2\n"
            "content:\n"
            "alpha\nbeta\n"
        )
        store = EvidenceStore(Path(tmp) / "evidence")
        store.add(
            "read_file",
            raw,
            metadata={
                "tool_call_id": "call-1",
                "user_turn_id": "turn-1",
                "produced_by_phase": "tool_call",
            },
        )
        return store

    def _saturated_answer(self, *, streaming: bool):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._partial_read_store(tmp)
            backend = RecordingBackend(
                context_tokens=1000, tokens_for={"": 5000, "Verified evidence": 300}
            )
            messages = [
                {"role": "user", "content": "show me the file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "read"},
            ]
            env, runtime = _environment(
                backend, store, messages=messages, context_tokens=1000
            )
            prefix = env._file_display_prefix()
            self.assertTrue(prefix, "fixture no longer produces a coverage notice")
            streamed: list[str] = []
            out = env.answer(
                temperature=0.0,
                max_tokens=2048,
                on_final_delta=streamed.append if streaming else None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                use_tool_prompt=False,
                workdir=None,
            )
            return prefix, out.result.content, "".join(streamed), runtime.messages

    def test_notice_is_streamed_exactly_once(self) -> None:
        prefix, content, streamed, messages = self._saturated_answer(streaming=True)
        self.assertEqual(streamed.count(prefix), 1, "coverage notice was duplicated")
        self.assertEqual(content.count(prefix), 1)
        self.assertEqual(str(messages[-1]["content"]).count(prefix), 1)

    def test_notice_reaches_content_without_streaming(self) -> None:
        prefix, content, streamed, messages = self._saturated_answer(streaming=False)
        self.assertEqual(streamed, "")
        self.assertEqual(content.count(prefix), 1, "coverage notice lost")
        self.assertEqual(str(messages[-1]["content"]).count(prefix), 1)


class ExactAccountingPreconditionTests(unittest.TestCase):
    """The rescue needs exact token accounting, and declines without it.

    On a backend that cannot report an exact count -- a plain llama.cpp server,
    or a session with no known context size -- upstream admission is skipped, so
    no ContextAdmissionError is raised and a saturated turn fails at decode as
    it did before this phase existed. That limit is real: without an exact count
    there is no way to promise the bundle fits either. These pin the decline so
    the limitation is visible rather than discovered in production.
    """

    def test_unknown_context_size_declines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])
            backend = RecordingBackend(context_tokens=1000)
            env, runtime = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            runtime.context_tokens = None
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out)
            self.assertEqual(backend.chat_calls, [], "generated without a known ctx")

    def test_backend_without_exact_counting_declines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])

            class NoExactCount(RecordingBackend):
                def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                    return None  # what a non-native server reports

            backend = NoExactCount(context_tokens=1000)
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNone(out)
            self.assertEqual(
                backend.chat_calls, [], "generated without an exact token count"
            )


class ProductionTrustBoundaryTests(unittest.TestCase):
    """The bundle the model sees must come from re-attested evidence.

    Testing `entries_from_store` in isolation proves the helper works; it does
    not prove the production path calls it. These assert on what actually
    reaches the prompt, so bypassing re-attestation or deduplication in the
    wiring fails here even though the library itself is still correct.
    """

    def _finalize(self, backend, store, *, context_tokens=1000):
        messages = [
            {"role": "user", "content": "analyse"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "output"},
        ]
        env, _ = _environment(
            backend, store, messages=messages, context_tokens=context_tokens
        )
        env.answer(
            temperature=0.0,
            max_tokens=2048,
            on_final_delta=None,
            on_progress=None,
            on_model_step=None,
            on_phase_start=None,
            loop=1,
            use_tool_prompt=False,
            workdir=None,
        )
        return "".join(
            str(m.get("content", "")) for m in backend.chat_calls[-1]["messages"]
        )

    def test_tampered_evidence_never_reaches_the_prompt(self) -> None:
        """Bytes edited on disk after attestation must not be cited.

        A second, intact record is seeded deliberately: with only the tampered
        one the finalizer would decline for lack of evidence and the assertion
        would pass without a prompt ever being built. Keeping one good record
        forces a real bundle, so the tampered bytes have somewhere to leak into.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store, ids = _store_with(
                tmp,
                ["genuine payload " + "g" * 200, "second intact record " + "s" * 200],
            )
            (store.root / f"{ids[0]}.txt").write_text(
                "TAMPERED payload " + "x" * 200, encoding="utf-8"
            )
            store.raw_cache.pop(ids[0], None)
            backend = RecordingBackend(
                context_tokens=1000, tokens_for={"": 5000, "Verified evidence": 300}
            )
            env, _ = _environment(
                backend,
                store,
                messages=[{"role": "user", "content": "analyse"}],
                context_tokens=1000,
            )
            out = env._grounded_finalization(
                temperature=0.0,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                on_phase_start=None,
                loop=1,
                response_prefix="",
            )
            self.assertIsNotNone(out, "no bundle was built, so nothing was proven")
            self.assertTrue(backend.chat_calls, "finalization never called the model")
            sent = "".join(
                str(m.get("content", "")) for m in backend.chat_calls[-1]["messages"]
            )
            self.assertIn("second intact record", sent, "intact evidence missing")
            self.assertNotIn("TAMPERED", sent, "tampered bytes reached the model")

    def test_duplicate_evidence_is_collapsed_in_the_prompt(self) -> None:
        """Byte-identical records must appear once, via the library's dedup."""
        payload = "repeated finding " + "r" * 200
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, [payload, payload, payload])
            backend = RecordingBackend(
                context_tokens=1000, tokens_for={"": 5000, "Verified evidence": 300}
            )
            sent = self._finalize(backend, store)
            self.assertEqual(
                sent.count("repeated finding"), 1, "duplicates were not collapsed"
            )

    def test_prompt_carries_recomputed_digests(self) -> None:
        """The header hash must be of the bytes actually sent."""
        with tempfile.TemporaryDirectory() as tmp:
            store, ids = _store_with(tmp, ["payload " + "p" * 200])
            backend = RecordingBackend(
                context_tokens=1000, tokens_for={"": 5000, "Verified evidence": 300}
            )
            sent = self._finalize(backend, store)
            expected = hashlib.sha256(
                ("payload " + "p" * 200).encode("utf-8")
            ).hexdigest()
            self.assertIn(expected[:16], sent)


class EvidenceTrustChainTests(unittest.TestCase):
    def test_bundle_sources_only_reattested_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, ids = _store_with(tmp, ["alpha " + "a" * 100, "beta " + "b" * 100])
            entries = entries_from_store(store, ids)
            self.assertEqual(len(entries), 2)
            # Corrupt the durable bytes: attestation must now drop the record.
            (store.root / f"{ids[0]}.txt").write_text("tampered", encoding="utf-8")
            store.raw_cache.pop(ids[0], None)
            surviving = entries_from_store(store, ids)
            self.assertEqual([e.evidence_id for e in surviving], [ids[1]])

    def test_artifact_sha_is_exact_and_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, ids = _store_with(tmp, ["payload " + "p" * 100])
            entry = entries_from_store(store, ids)[0]
            expected = hashlib.sha256(
                ("payload " + "p" * 100).encode("utf-8")
            ).hexdigest()
            self.assertEqual(entry.sha256, expected)
            self.assertEqual(entry.sha256, store.records[ids[0]].raw_sha256)

    def test_zero_budget_is_refused(self) -> None:
        self.assertFalse(admit_finalization(1000, 1000).admitted)
        self.assertFalse(admit_finalization(999, 1000).admitted)


if __name__ == "__main__":
    unittest.main()
