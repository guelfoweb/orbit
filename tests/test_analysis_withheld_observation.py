"""A large observation says so, so a transformation stays selectable.

An analysis step that prints a big encoded blob -- exactly the step that
precedes a deterministic decode -- renders above `COMPAT_INLINE_CHARS` as a
canonical reference carrying no excerpt. Inherited from CHAT, that rendering
says nothing about the omission, so the cheapest apparent next move is to
observe something smaller rather than transform what was just produced.

These cover the added sentence and, more importantly, what it must not become:
it carries no bytes, it fires only when bytes were actually withheld, and it
never displaces the evidence identity that compaction depends on.
"""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from orbit.runtime.analysis_runtime import AnalysisRuntime, acquire_analysis_source
from orbit.runtime.context_manager import conversation_structure_error
from orbit.runtime.evidence import COMPAT_INLINE_CHARS, EvidenceStore, tool_evidence_ref

from tests.test_analysis_runtime import ScriptedBackend, prose_response, tool_response


LARGE = COMPAT_INLINE_CHARS + 400
SMALL = 40

READ = "import orbit_tools; print(orbit_tools.read_file('/workspace/input')[:80])"


def _emit(payload: str) -> str:
    return f"print({payload!r})"


class WithheldTestBase(unittest.TestCase):
    """A session over source text the test chooses."""

    SOURCE = "alpha\nbeta\ngamma\n"

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="orbit-withheld-")
        self.addCleanup(self._dir.cleanup)
        self.tmp = Path(self._dir.name)
        original = self.tmp / "artifact.txt"
        original.write_text(self.SOURCE, encoding="utf-8")
        self.source = acquire_analysis_source(original, self.tmp / "owned")
        self.store = EvidenceStore(root=self.tmp / "evidence")

    def with_source(self, text: str) -> None:
        """Rebuild this session over `text`.

        Replaces the fixture built in `setUp` rather than calling `setUp`
        again: a second setup would register a second cleanup and orphan the
        first workspace and store for the life of the test.
        """
        original = self.tmp / "artifact-2.txt"
        original.write_text(text, encoding="utf-8")
        self.source = acquire_analysis_source(original, self.tmp / "owned-2")
        self.store = EvidenceStore(root=self.tmp / "evidence-2")

    def runtime(self, backend) -> AnalysisRuntime:
        built = AnalysisRuntime(
            backend=backend, source=self.source, evidence_store=self.store
        )
        self.addCleanup(built.close)
        return built

    def last_tool_message(self, runtime: AnalysisRuntime) -> dict:
        tools = [m for m in runtime.messages if m.get("role") == "tool"]
        self.assertTrue(tools, "expected a tool result")
        return tools[-1]


class WithheldPredicateTests(WithheldTestBase):
    def test_withheld_exactly_when_the_reference_carries_no_excerpt(self) -> None:
        """A reference with no excerpt says so; one with bytes stays silent."""
        big = self.store.add("execute_analysis", "x" * LARGE, metadata={})
        small = self.store.add("execute_analysis", "x" * SMALL, metadata={})

        self.assertNotIn("compat_excerpt:", tool_evidence_ref(big))
        self.assertIn("content_withheld: true", tool_evidence_ref(big))
        self.assertIn("compat_excerpt:", tool_evidence_ref(small))
        self.assertNotIn("content_withheld", tool_evidence_ref(small))

    def test_a_whitespace_result_at_the_bound_claims_nothing(self) -> None:
        """`>` not `>=`: at exactly the bound nothing was withheld.

        A whitespace-only result has no excerpt (`_text_excerpts` strips it),
        so it reaches the same branch as a large one. At exactly
        COMPAT_INLINE_CHARS nothing is held back, and saying otherwise would
        assert archived bytes that do not exist.
        """
        record = self.store.add(
            "execute_analysis", " " * COMPAT_INLINE_CHARS, metadata={}
        )
        self.assertEqual(record.raw_chars, COMPAT_INLINE_CHARS)
        self.assertNotIn("compat_excerpt:", tool_evidence_ref(record))
        self.assertNotIn("content_withheld", tool_evidence_ref(record))

    def test_an_empty_result_never_claims_withheld_bytes(self) -> None:
        """No excerpt is not the same as bytes held back.

        `_compat_excerpt` also returns "" for a result with no text at all, so
        a bare `else` here would tell the model that archived content exists
        for an action that printed nothing.
        """
        empty = self.store.add("execute_analysis", "", metadata={})
        self.assertEqual(empty.raw_chars, 0)
        self.assertNotIn("content_withheld", tool_evidence_ref(empty))

    def test_predicate_tracks_the_renderer_rather_than_a_copied_constant(self) -> None:
        """Just under the bound keeps its bytes; just over loses them."""
        under = self.store.add("execute_analysis", "x" * (COMPAT_INLINE_CHARS - 1), metadata={})
        over = self.store.add("execute_analysis", "x" * (COMPAT_INLINE_CHARS + 1), metadata={})
        self.assertNotIn("content_withheld", tool_evidence_ref(under))
        self.assertIn("content_withheld: true", tool_evidence_ref(over))


class WithheldNoticeTests(WithheldTestBase):
    def test_large_observation_says_its_bytes_were_withheld(self) -> None:
        backend = ScriptedBackend(tool_response(_emit("A" * LARGE)))
        runtime = self.runtime(backend)
        runtime.step("inspect")
        message = self.last_tool_message(runtime)

        self.assertIn("content_withheld: true", message["content"])
        # It must name the retrieval that already works, by the id actually used.
        self.assertIn(f"evidence:{message['evidence_id']}", message["content"])
        self.assertIn("archived, not lost", message["content"])

    def test_small_observation_keeps_its_bytes_and_says_nothing(self) -> None:
        backend = ScriptedBackend(tool_response(_emit("B" * SMALL)))
        runtime = self.runtime(backend)
        runtime.step("inspect")
        message = self.last_tool_message(runtime)

        self.assertIn("compat_excerpt:", message["content"])
        self.assertNotIn("content_withheld", message["content"])

    def test_the_notice_carries_no_content_of_its_own(self) -> None:
        """The point is to name the omission, never to smuggle a second excerpt."""
        backend = ScriptedBackend(tool_response(_emit("Z" * LARGE)))
        runtime = self.runtime(backend)
        runtime.step("inspect")
        content = self.last_tool_message(runtime)["content"]

        self.assertNotIn("Z" * 40, content)
        # A bounded sentence, not a rendering path that grows with the output.
        self.assertLess(len(content), 600)

    def test_notice_states_the_true_size(self) -> None:
        """The whole line, against a payload whose line count cannot alias it.

        `assertIn("1 chars", "1601 chars ...")` passes, so a substring check on
        a single-line payload would let a regression rendering `raw_lines` as
        `raw_chars` ship green.
        """
        payload = "\n".join("Q" * 40 for _ in range(LARGE // 40))
        backend = ScriptedBackend(tool_response(_emit(payload)))
        runtime = self.runtime(backend)
        runtime.step("inspect")
        message = self.last_tool_message(runtime)
        record = self.store.records[message["evidence_id"]]

        self.assertGreater(record.raw_lines, 1)
        self.assertNotEqual(record.raw_lines, record.raw_chars)
        self.assertIn(
            f"exact_content_ref: {record.raw_ref} "
            f"({record.raw_chars} chars archived, not lost)",
            message["content"],
        )


class IdentityAndProtocolTests(WithheldTestBase):
    def test_evidence_identity_survives_the_notice(self) -> None:
        """Compaction keys off `evidence_id`; the notice must not disturb it."""
        backend = ScriptedBackend(tool_response(_emit("A" * LARGE)))
        runtime = self.runtime(backend)
        runtime.step("inspect")
        message = self.last_tool_message(runtime)

        self.assertIn("evidence_id", message)
        self.assertIn(message["evidence_id"], self.store.records)
        self.assertIsNone(conversation_structure_error(runtime.messages))

    def test_exact_bytes_are_still_re_attestable(self) -> None:
        backend = ScriptedBackend(tool_response(_emit("A" * LARGE)))
        runtime = self.runtime(backend)
        runtime.step("inspect")
        message = self.last_tool_message(runtime)

        raw = self.store.reattest_exact(message["evidence_id"])
        self.assertIsNotNone(raw)
        self.assertIn("A" * 100, raw)

    def test_refused_action_carries_neither_identity_nor_notice(self) -> None:
        """No record means no claim of evidence, and nothing to say about it."""
        backend = ScriptedBackend(tool_response("import os\nos.nope("))
        runtime = self.runtime(backend)
        runtime.step("inspect")
        message = self.last_tool_message(runtime)

        self.assertNotIn("evidence_id", message)
        self.assertNotIn("content_withheld", message["content"])


class PersistedReferenceCompatibilityTests(WithheldTestBase):
    """A session written before the notice existed must still compact.

    `reattest_exact` compares the persisted message content byte-for-byte
    against the current rendering, and CHAT reloads saved tool messages
    verbatim -- the stored string is never rewritten. Without tolerance for the
    prior rendering, resuming an older session would silently drop that turn
    from `available`, pinning in context exactly the large output this change
    is about, and failing closed only later as an admission error.
    """

    def _pre_notice_reference(self, record) -> str:
        return "\n".join(
            line
            for line in tool_evidence_ref(record).splitlines()
            if not line.startswith(("content_withheld:", "exact_content_ref:"))
        )

    def test_reference_persisted_before_the_notice_still_re_attests(self) -> None:
        record = self.store.add(
            "execute_analysis",
            "x" * LARGE,
            metadata={
                "tool_call_id": "call_1",
                "user_turn_id": "turn_1",
                "produced_by_phase": "analysis_action",
            },
        )
        stored = self._pre_notice_reference(record)
        self.assertNotIn("content_withheld", stored)

        self.assertIsNotNone(
            self.store.reattest_exact(record.evidence_id, expected_reference=stored)
        )
        self.assertIsNotNone(
            self.store.reattest_exact(
                record.evidence_id, expected_reference=tool_evidence_ref(record)
            )
        )

    def test_a_reference_without_a_notice_is_never_truncated_to_match(self) -> None:
        """The tolerance applies only where a notice was actually rendered.

        Without that guard the matcher would blind-truncate its own rendering
        by the notice length for every record, and accept the truncated string
        for a small result that never carried a notice at all.
        """
        record = self.store.add(
            "execute_analysis",
            "small enough to inline",
            metadata={
                "tool_call_id": "call_1",
                "user_turn_id": "turn_1",
                "produced_by_phase": "analysis_action",
            },
        )
        current = tool_evidence_ref(record)
        self.assertNotIn("content_withheld", current)
        notice_len = len(
            "content_withheld: true\n"
            f"exact_content_ref: {record.raw_ref} "
            f"({record.raw_chars} chars archived, not lost)"
        )
        truncated = current[: -(notice_len + 1)]

        self.assertIsNotNone(
            self.store.reattest_exact(record.evidence_id, expected_reference=current)
        )
        self.assertIsNone(
            self.store.reattest_exact(record.evidence_id, expected_reference=truncated)
        )

    def test_tolerance_does_not_accept_an_arbitrary_reference(self) -> None:
        """Accepting the prior rendering must not accept anything else."""
        record = self.store.add(
            "execute_analysis",
            "x" * LARGE,
            metadata={
                "tool_call_id": "call_1",
                "user_turn_id": "turn_1",
                "produced_by_phase": "analysis_action",
            },
        )
        current = tool_evidence_ref(record)
        for forged in (
            "tool_evidence_ref: true",
            current.replace(record.evidence_id, "ev_deadbeef_0000000000000000"),
            current + "\nextra: line",
            current[:-1],
            self._pre_notice_reference(record) + "\ncontent_withheld: true",
            # Prefixes other than the exact pre-notice rendering: tolerating
            # the older form must not become "any leading substring".
            "\n".join(current.splitlines()[:2]),
            "\n".join(current.splitlines()[:-3]),
            current.split("\n")[0],
        ):
            with self.subTest(forged=forged[:40]):
                self.assertIsNone(
                    self.store.reattest_exact(
                        record.evidence_id, expected_reference=forged
                    )
                )


class DeterministicTransformationTests(WithheldTestBase):
    """Observe, transform, reason from the transformed output.

    Two structurally different transformations, because the affordance claims
    to be general. Neither is a decoder the runtime knows about: both are
    ordinary Python the model chose to write.
    """

    SECRET = "STAGE-TWO-COMMAND"
    XOR_SRC = (
        "import orbit_tools\n"
        "src = orbit_tools.read_file('/workspace/input')\n"
        "blob = src.split('payload=')[1].strip()\n"
        "print(''.join(chr(int(t) ^ 7) for t in blob.split('-')))\n"
    )
    B64_SRC = (
        "import base64, orbit_tools\n"
        "src = orbit_tools.read_file('/workspace/input')\n"
        "blob = src.split('b64=')[1].strip()\n"
        "print(base64.b64decode(blob).decode())\n"
    )

    def test_numeric_xor_transformation_executes_and_becomes_evidence(self) -> None:
        encoded = "-".join(str(ord(c) ^ 7) for c in self.SECRET)
        self.with_source(f"var x = 1;\npayload={encoded}\n")

        backend = ScriptedBackend(tool_response(READ), tool_response(self.XOR_SRC))
        runtime = self.runtime(backend)

        first = runtime.step("inspect")
        second = runtime.step("continue")

        self.assertTrue(second.action_executed)
        self.assertIsNone(second.suppressed_duplicate_of)
        raw = self.store.reattest_exact(second.evidence.evidence_id)
        self.assertIn(self.SECRET, raw)
        self.assertNotEqual(first.evidence.evidence_id, second.evidence.evidence_id)

    def test_base64_transformation_executes_and_becomes_evidence(self) -> None:
        secret = "second stage marker"
        encoded = base64.b64encode(secret.encode()).decode()
        self.with_source(f"var x = 1;\nb64={encoded}\n")

        backend = ScriptedBackend(tool_response(READ), tool_response(self.B64_SRC))
        runtime = self.runtime(backend)

        runtime.step("inspect")
        second = runtime.step("continue")

        self.assertTrue(second.action_executed)
        self.assertIn(secret, self.store.reattest_exact(second.evidence.evidence_id))

    def test_a_transformation_is_never_suppressed_as_a_duplicate_read(self) -> None:
        """Different program, same source: distinct experiment, must execute."""
        encoded = "-".join(str(ord(c) ^ 7) for c in self.SECRET)
        self.with_source(f"var x = 1;\npayload={encoded}\n")

        backend = ScriptedBackend(
            tool_response(READ),
            tool_response(READ),           # exact repeat -> suppressed
            tool_response(self.XOR_SRC),   # transformation -> must run
        )
        runtime = self.runtime(backend)

        runtime.step("a")
        duplicate = runtime.step("b")
        transform = runtime.step("c")

        self.assertIsNotNone(duplicate.suppressed_duplicate_of)
        self.assertFalse(duplicate.action_executed)
        self.assertTrue(transform.action_executed)
        self.assertIsNone(transform.suppressed_duplicate_of)
        self.assertIn(
            self.SECRET, self.store.reattest_exact(transform.evidence.evidence_id)
        )


if __name__ == "__main__":
    unittest.main()
