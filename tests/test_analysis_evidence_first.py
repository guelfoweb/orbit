"""An autonomous run starts from what the runtime already established.

The deterministic preamble lists what exists -- ids, kinds, sizes, digests --
but not what any of it says. A model told that five facts exist, without
knowing any of them, has one way to learn something: read the source. That is
what a real run did, in one rendering after another, until the action budget
stopped it, while the answer sat in evidence it had been handed.

These cover the instruction that closes that gap, and the four things it must
not disturb: a session with no deterministic evidence, guided mode, the
model's freedom to ask a real question, and the budgets.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orbit.runtime.analysis_runtime import (
    AUTONOMOUS_CONTINUATION_MESSAGE,
    MAX_EVIDENCE_FIRST_CHARS,
    MAX_AUTONOMOUS_ACTIONS,
    MAX_AUTONOMOUS_MODEL_CALLS,
    AnalysisRuntime,
    _evidence_first_ids,
    _evidence_first_instruction,
    acquire_analysis_source,
)
from orbit.runtime.evidence import EvidenceStore

from tests.test_analysis_runtime import ScriptedBackend, prose_response, tool_response

OPENING = "Analyse this artifact and report what it does."

DECODER = (
    "function dec(s, k, d) {\n"
    "    var out = '';\n"
    "    var parts = s.split(d);\n"
    "    for (var i = 0; i < parts.length; i++) {\n"
    "        out += String.fromCharCode(parts[i] ^ k);\n"
    "    }\n"
    "    return out;\n"
    "}\n"
)
SECRET = "STAGE-TWO-COMMAND"


def encode(text: str, key: int, delimiter: str) -> str:
    return delimiter.join(str(ord(c) ^ key) for c in text)


def decodable(secret: str = SECRET) -> str:
    return DECODER + f'dec("{encode(secret, 19, ",")}", 19, ",");\n'


class RecordingBackend(ScriptedBackend):
    """Records the analyst line that drove each call."""

    def __init__(self, *responses) -> None:
        super().__init__(*responses)
        self.instructions: list[str] = []

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                    on_delta, on_progress=None):
        users = [m for m in messages if m.get("role") == "user"]
        self.instructions.append(users[-1]["content"] if users else "")
        return super().chat_stream(
            messages, temperature=temperature, max_tokens=max_tokens,
            tools=tools, on_delta=on_delta, on_progress=on_progress,
        )


class EvidenceFirstTestBase(unittest.TestCase):
    def runtime(self, backend, source_text: str) -> AnalysisRuntime:
        tmpdir = tempfile.TemporaryDirectory(prefix="orbit-evfirst-")
        self.addCleanup(tmpdir.cleanup)
        tmp = Path(tmpdir.name)
        artifact = tmp / "artifact.js"
        artifact.write_text(source_text, encoding="utf-8")
        built = AnalysisRuntime(
            backend=backend,
            source=acquire_analysis_source(artifact, tmp / "owned"),
            evidence_store=EvidenceStore(root=tmp / "evidence"),
        )
        self.addCleanup(built.close)
        return built


class InstructionShapeTests(unittest.TestCase):
    def test_the_instruction_names_ids_not_values(self) -> None:
        """Naming an id is what the rehydration path answers to. Copying the
        output instead would put the evidence in the prompt twice and grow it
        with the size of what was decoded."""
        text = _evidence_first_instruction(OPENING, ["ev_a_1", "ev_b_2"])

        self.assertIn("evidence:ev_a_1", text)
        self.assertIn("evidence:ev_b_2", text)
        self.assertIn(OPENING, text)
        self.assertLess(len(text), 600, "the instruction is a sentence, not a payload")

    def test_an_empty_id_list_yields_the_analyst_line_unchanged(self) -> None:
        """Reachable: every stage can exceed the budget on its own, leaving
        stages present but nothing named. Claiming evidence is available and
        then naming none would be worse than saying nothing."""
        self.assertEqual(_evidence_first_instruction(OPENING, []), OPENING)

    def test_the_instruction_permits_a_further_action(self) -> None:
        """Deterministic evidence establishes what a transformation produced,
        never how the artifact uses it."""
        text = _evidence_first_instruction(OPENING, ["ev_a_1"]).lower()
        self.assertIn("take an action only for a concrete question", text)
        self.assertNotIn("do not take", text)

    def test_the_instruction_names_no_technique_or_artifact(self) -> None:
        text = _evidence_first_instruction(OPENING, ["ev_a_1"]).lower()
        for term in ("xor", "base64", "malware", "smartmaket", "powershell",
                     "jscript", "decode", "decoder"):
            with self.subTest(term=term):
                self.assertNotIn(term, text)

    def test_prompt_cost_does_not_grow_with_evidence_size(self) -> None:
        small = _evidence_first_instruction(OPENING, ["ev_a_1"])
        many = _evidence_first_instruction(OPENING, [f"ev_{i}_x" for i in range(5)])
        # Grows by ids only -- a bounded per-stage cost, not by output bytes.
        self.assertLess(len(many) - len(small), 200)


class EvidenceBudgetTests(EvidenceFirstTestBase):
    """What arrives unasked is bounded by the runtime, not by the artifact."""

    def _many_stages(self, count: int, size: int) -> str:
        return DECODER + "".join(
            f'dec("{encode("X" * size + str(i), 19, ",")}", 19, ",");\n'
            for i in range(count)
        )

    def test_a_crafted_artifact_cannot_flood_the_first_call(self) -> None:
        """Naming an id restores its bytes, and the size of those bytes is the
        artifact's to choose. Unbounded, a run that was merely slow becomes one
        that cannot start."""
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, self._many_stages(16, 4000))

        ids = _evidence_first_ids(runtime.transform_stages)
        restored = sum(
            len(stage.output)
            for stage, record in runtime.transform_stages
            if record.evidence_id in ids
        )

        self.assertGreater(len(runtime.transform_stages), len(ids))
        self.assertLessEqual(restored, MAX_EVIDENCE_FIRST_CHARS)

    def test_a_small_artifact_has_every_stage_named(self) -> None:
        """The bound must not cost anything in the ordinary case."""
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())

        self.assertEqual(
            len(_evidence_first_ids(runtime.transform_stages)),
            len(runtime.transform_stages),
        )

    def test_smaller_stages_are_preferred(self) -> None:
        """A short decoded string is usually the fact an analysis turns on;
        one large stage would crowd out several that answer more."""
        backend = RecordingBackend(prose_response("done"))
        source = DECODER
        source += f'dec("{encode("SHORT", 19, ",")}", 19, ",");\n'
        # Sized to fit alone but to leave no room for anything else: taking it
        # first would exclude both short stages, which is the ordering this
        # pins. A stage larger than the whole budget would be skipped either
        # way and prove nothing.
        source += f'dec("{encode("Y" * (MAX_EVIDENCE_FIRST_CHARS - 8), 19, ",")}", 19, ",");\n'
        source += f'dec("{encode("ALSO-SHORT", 19, ",")}", 19, ",");\n'
        runtime = self.runtime(backend, source)

        ids = set(_evidence_first_ids(runtime.transform_stages))
        by_id = {record.evidence_id: stage for stage, record in runtime.transform_stages}
        named = {by_id[i].output for i in ids}

        self.assertIn("SHORT", named)
        self.assertIn("ALSO-SHORT", named)
        # And the large one is what gives way: taking it first would spend the
        # whole budget on one stage and exclude both short ones.
        large = "Y" * (MAX_EVIDENCE_FIRST_CHARS - 8)
        self.assertNotIn(large, named)
        self.assertEqual(len(named), 2)

    def test_stages_past_the_budget_stay_available_on_request(self) -> None:
        """Excluded is not hidden: the preamble still lists them and the model
        can name any id it wants."""
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, self._many_stages(16, 4000))
        ids = set(_evidence_first_ids(runtime.transform_stages))

        preamble = runtime.messages[2]["content"]
        for _stage, record in runtime.transform_stages:
            with self.subTest(evidence=record.evidence_id):
                self.assertIn(record.evidence_id, preamble)
                self.assertIsNotNone(
                    runtime.evidence_store.reattest_exact(record.evidence_id)
                )
        self.assertLess(len(ids), len(runtime.transform_stages))


class AdmissionFallbackTests(EvidenceFirstTestBase):
    """A head start must never cost the ability to start.

    Naming an id restores its decoded bytes, and how many tokens those bytes
    are is the artifact's to decide: density on this tokenizer runs from about
    4 characters per token down to 1.1 on exactly the obfuscated content this
    decodes. A character budget cannot bound that, so the opening is attempted
    and a refusal withdraws it.
    """

    CTX = 8192

    def _dense_backend(self, density: float):
        from orbit.backend.base import TokenCount

        class _Dense(RecordingBackend):
            thinking = False

            def supports_exact_context_admission(self) -> bool:
                return True

            def model_info(self):
                class _Info:
                    context_length = AdmissionFallbackTests.CTX

                return _Info()

            def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                chars = sum(len(str(m.get("content", ""))) for m in messages)
                return TokenCount(
                    tokens=int(chars / density),
                    context_tokens=AdmissionFallbackTests.CTX,
                    rendered_hash="a" * 64,
                    token_hash="b" * 64,
                )

        return _Dense(prose_response("done"))

    def _dense_artifact(self, size: int) -> str:
        return DECODER + f'dec("{encode("P" * size, 19, ",")}", 19, ",");\n'

    def test_a_prompt_that_fits_keeps_the_evidence_first_opening(self) -> None:
        backend = self._dense_backend(1.123)
        runtime = self.runtime(backend, self._dense_artifact(1000))
        runtime.context_tokens = self.CTX

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(run.model_calls, 1)
        self.assertIn("Verified deterministic evidence", backend.instructions[0])

    def test_a_prompt_that_would_not_fit_withdraws_it_and_still_runs(self) -> None:
        """The alternative is an analysis that cannot begin at all on an
        artifact it used to handle."""
        backend = self._dense_backend(1.123)
        runtime = self.runtime(backend, self._dense_artifact(6000))
        runtime.context_tokens = self.CTX

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(run.model_calls, 1, "the run still starts")
        self.assertEqual(backend.instructions[0], OPENING, "the opening was withdrawn")
        self.assertNotIn("Verified deterministic evidence", backend.instructions[0])

    def test_the_withdrawal_costs_no_model_call(self) -> None:
        backend = self._dense_backend(1.123)
        runtime = self.runtime(backend, self._dense_artifact(6000))
        runtime.context_tokens = self.CTX

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(backend.calls, 1, "the refused attempt reached no backend call")
        self.assertEqual(len(run.steps), 1)

    def test_a_refusal_on_the_plain_line_still_ends_the_run(self) -> None:
        """Withdrawal is for the opening only. Retrying a request already known
        not to fit would spend the ceiling on it."""
        backend = self._dense_backend(0.05)  # nothing fits at this density
        runtime = self.runtime(backend, self._dense_artifact(1000))
        runtime.context_tokens = self.CTX

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(run.model_calls, 0)
        self.assertIn("ContextAdmissionError", run.stop_reason)


class SufficientEvidenceTests(EvidenceFirstTestBase):
    """CASE A: the evidence already answers the question."""

    def test_the_run_can_finish_without_a_single_observation(self) -> None:
        backend = RecordingBackend(prose_response("The artifact decodes a second stage."))
        runtime = self.runtime(backend, decodable())

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(run.actions_executed, 0, "no observation was needed")
        self.assertEqual(run.model_calls, 1)
        self.assertEqual(run.stop_reason, "model returned prose with no action")

    def test_the_first_instruction_carries_the_evidence_ids(self) -> None:
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())

        runtime.run_autonomous(OPENING, finalize=False)

        first = backend.instructions[0]
        for _stage, record in runtime.transform_stages:
            self.assertIn(f"evidence:{record.evidence_id}", first)

    def test_naming_the_ids_restores_their_exact_bytes(self) -> None:
        """The point of naming ids rather than copying values.

        Asserted against the rehydration seam directly: `_admit` only reaches
        it on a backend that attests exact tokens, so a scripted backend would
        skip the very step under test and pass vacuously.
        """
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())
        ids = [record.evidence_id for _stage, record in runtime.transform_stages]

        messages = runtime.messages + [
            {"role": "user", "content": _evidence_first_instruction(OPENING, ids)}
        ]
        rehydrated, used = runtime._with_evidence_rehydration(messages)

        self.assertEqual(len(used), len(ids), "every named id is restored")
        restored = "".join(
            str(m.get("content", "")) for m in rehydrated[len(messages) - 1 :]
        )
        self.assertIn(SECRET, restored)

    def test_the_opening_alone_would_restore_nothing(self) -> None:
        """Without the ids there is nothing for rehydration to answer, which
        is the state a real run was stuck in: five facts named, none readable.
        """
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())

        messages = runtime.messages + [{"role": "user", "content": OPENING}]
        _rehydrated, used = runtime._with_evidence_rehydration(messages)

        self.assertEqual(used, ())

    def test_the_deterministic_appendix_survives_an_immediate_finish(self) -> None:
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())
        runtime.run_autonomous(OPENING, finalize=False)

        appendix = runtime.deterministic_sections()
        self.assertIn("## Deterministic transformations", appendix)
        self.assertIn(SECRET, appendix)


class UnresolvedFactTests(EvidenceFirstTestBase):
    """CASE B / E: a real question still earns an action."""

    def test_one_legitimate_observation_executes_and_completes(self) -> None:
        backend = RecordingBackend(
            tool_response("import orbit_tools; print(orbit_tools.read_file('/workspace/input')[:60])"),
            prose_response("the decoded command is invoked at line 3"),
        )
        runtime = self.runtime(backend, decodable())

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.model_calls, 2)
        self.assertTrue(run.steps[0].action_executed)

    def test_the_second_step_uses_the_ordinary_continuation(self) -> None:
        """Evidence-first is an opening, not a mode: once the run is going the
        loop says what it always said."""
        backend = RecordingBackend(
            tool_response("print('a')"), prose_response("done")
        )
        runtime = self.runtime(backend, decodable())
        runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(backend.instructions[1], AUTONOMOUS_CONTINUATION_MESSAGE)


class NoPreflightTests(EvidenceFirstTestBase):
    """CASE C: nothing established means nothing changes."""

    PLAIN = "var x = 1;\nconsole.log(x);\n"

    def test_the_opening_is_sent_verbatim(self) -> None:
        backend = RecordingBackend(prose_response("nothing to decode"))
        runtime = self.runtime(backend, self.PLAIN)

        runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(runtime.transform_stages, [])
        self.assertEqual(backend.instructions[0], OPENING)
        self.assertNotIn("evidence:", backend.instructions[0])

    def test_an_ordinary_autonomous_run_is_unaffected(self) -> None:
        backend = RecordingBackend(
            tool_response("print('a')"), tool_response("print('b')"), prose_response("done")
        )
        runtime = self.runtime(backend, self.PLAIN)

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(backend.instructions[0], OPENING)


class GuidedModeTests(EvidenceFirstTestBase):
    """CASE D: a guided step is one step, whatever the runtime established."""

    def test_a_guided_step_receives_the_analyst_line_unchanged(self) -> None:
        backend = RecordingBackend(prose_response("answered"))
        runtime = self.runtime(backend, decodable())

        runtime.step("what does line 3 do?")

        self.assertEqual(backend.instructions[0], "what does line 3 do?")
        self.assertNotIn("Verified deterministic evidence", backend.instructions[0])

    def test_a_guided_step_does_not_start_a_run(self) -> None:
        backend = RecordingBackend(prose_response("answered"))
        runtime = self.runtime(backend, decodable())

        result = runtime.step("what does line 3 do?")

        self.assertEqual(result.model_calls, 1)
        self.assertEqual(backend.calls, 1)


class BudgetTests(EvidenceFirstTestBase):
    """The instruction spends no action and moves no ceiling."""

    def test_deterministic_evidence_consumes_no_action_slot(self) -> None:
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())

        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(runtime.analyst_turns, 1, "one analyst turn, not two")

    def test_the_ceilings_are_unchanged(self) -> None:
        self.assertEqual(MAX_AUTONOMOUS_ACTIONS, 12)
        self.assertEqual(MAX_AUTONOMOUS_MODEL_CALLS, 18)

    def test_no_synthetic_action_is_recorded(self) -> None:
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())
        run = runtime.run_autonomous(OPENING, finalize=False)

        self.assertFalse(any(step.action_attempted for step in run.steps))

    def test_tools_remain_available_on_the_first_call(self) -> None:
        """Starting from evidence must not mean starting without tools."""
        backend = RecordingBackend(prose_response("done"))
        runtime = self.runtime(backend, decodable())
        runtime.run_autonomous(OPENING, finalize=False)

        self.assertEqual(backend.seen_tools[0], ["execute_analysis"])


if __name__ == "__main__":
    unittest.main()
