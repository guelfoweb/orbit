"""Live wiring for the grounded finalization handoff.

An evidence-backed tool workflow answers from its durable evidence in a fresh
session rather than from the conversation that produced it. These cover the
transition through the real answer() path: that the finalizer is reached
automatically, that the saturated investigation session is not carried into it,
and that the evidence is re-attested rather than trusted.

The trigger is the evidence-backed boundary itself, not context pressure -- the
whole point of a separate phase is that the answer no longer depends on how
much room the investigation left.
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

    def test_answer_reaches_grounded_finalization_when_saturated(self) -> None:
        """Every ordinary window overflows; only the bundle can be admitted.

        Compaction can rescue many large turns on its own, so the scenario has
        to be one it cannot rescue -- otherwise the test passes with the
        finalizer entirely disconnected.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["verified payload " + "p" * 200])
            backend = RecordingBackend(
                context_tokens=1000,
                tokens_for={"": 2000, "Verified evidence": 300},
            )
            result = self._answer(backend, store, context_tokens=1000)
            self.assertEqual(result.result.finish_reason, "stop")
            self.assertTrue(backend.chat_calls)
            # The bundle, not the conversation, is what was sent.
            sent = backend.chat_calls[-1]["messages"]
            self.assertEqual(len(sent), 1)
            self.assertIn("Verified evidence", str(sent[0]["content"]))
            self.assertIsNone(backend.chat_calls[-1]["tools"])
            self.assertEqual(backend.resets, 1)

    def test_roomy_session_still_finalizes_from_evidence(self) -> None:
        """The trigger is the evidence-backed boundary, not context pressure.

        An earlier version diverted only when the ordinary window could not
        fit, which made grounded finalization a rescue for saturated sessions
        rather than how an investigation normally reports. With plenty of
        context left the same handoff must still apply, because the answer is
        not supposed to depend on what the investigation happened to leave.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store, _ids = _store_with(tmp, ["payload " + "p" * 200])
            backend = RecordingBackend(context_tokens=100_000)
            self._answer(backend, store, context_tokens=100_000)
            self.assertTrue(backend.chat_calls)
            sent = backend.chat_calls[-1]["messages"]
            self.assertEqual(len(sent), 1, "did not finalize from the bundle")
            self.assertIn("Verified evidence", str(sent[0]["content"]))
            self.assertIsNone(backend.chat_calls[-1]["tools"])
            self.assertEqual(backend.resets, 1)


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
