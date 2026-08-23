"""One analyst step must end at the analyst, provably.

The property under test is an absence: after an action runs, nothing calls
the model again. Absences are easy to assert vacuously, so the backend here
raises the moment it is invoked once more than the script allows. A runtime
that quietly continued -- to interpret the result, repair a bad call, or
decide what to do next -- fails these tests by exploding, not by returning
a wrong value.
"""

from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime.analysis_runtime import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_TOOL_NAME,
    ANALYSIS_TOOL_SCHEMA,
    MAX_EVIDENCE_CHARS,
    SESSION_CAPACITY_EXHAUSTED,
    AnalysisRuntime,
    acquire_analysis_source,
)
from orbit.runtime.evidence import EvidenceStore

FIXTURE = "alpha\nbeta\ngamma\n"


class ExhaustedBackend(Exception):
    """Raised when the runtime calls the model more times than scripted."""


class ScriptedBackend:
    """Serves scripted responses; explodes on an unscripted call."""

    def __init__(self, *responses: ChatResult) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.seen_tools: list[list[str]] = []
        self.seen_messages: list[list[dict]] = []

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta, on_progress=None):
        if self.calls >= len(self._responses):
            raise ExhaustedBackend(
                f"model invoked {self.calls + 1} times; only {len(self._responses)} scripted "
                "-- the analyst boundary was crossed"
            )
        self.seen_messages.append([dict(m) for m in messages])
        self.seen_tools.append([t["function"]["name"] for t in (tools or [])])
        response = self._responses[self.calls]
        self.calls += 1
        if response.content:
            on_delta(response.content)
        return response


def tool_response(code: str, *, name: str = ANALYSIS_TOOL_NAME, count: int = 1, text: str = "") -> ChatResult:
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({"code": code})},
        }
        for i in range(count)
    ]
    return ChatResult(
        content=text, model="m", finish_reason="stop", tool_calls=calls,
        prompt_tokens=10, completion_tokens=5, cached_tokens=0,
        prompt_tokens_per_second=None, generation_tokens_per_second=None,
    )


def prose_response(text: str) -> ChatResult:
    return ChatResult(
        content=text, model="m", finish_reason="stop", tool_calls=[],
        prompt_tokens=10, completion_tokens=5, cached_tokens=0,
        prompt_tokens_per_second=None, generation_tokens_per_second=None,
    )


class AnalysisRuntimeTestBase(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._dir = tempfile.TemporaryDirectory(prefix="orbit-analysis-rt-")
        self.addCleanup(self._dir.cleanup)
        self.tmp = Path(self._dir.name)
        original = self.tmp / "artifact.txt"
        original.write_text(FIXTURE, encoding="utf-8")
        self.original = original
        self.source = acquire_analysis_source(original, self.tmp / "owned")
        self.store = EvidenceStore(root=self.tmp / "evidence")

    def runtime(self, backend) -> AnalysisRuntime:
        # Registered here rather than in each test: a session workspace that
        # outlives its test leaves a directory behind, and 30-odd of those per
        # run is how a suite quietly fills /tmp.
        built = AnalysisRuntime(backend=backend, source=self.source, evidence_store=self.store)
        self.addCleanup(built.close)
        return built


class AnalystBoundaryTest(AnalysisRuntimeTestBase):
    def test_one_step_makes_one_call_runs_one_action_and_stops(self) -> None:
        backend = ScriptedBackend(
            tool_response("import orbit_tools\nprint(orbit_tools.read_file('/workspace/input'), end='')")
        )
        runtime = self.runtime(backend)

        result = runtime.step("inspect the artifact")

        self.assertEqual(backend.calls, 1, "exactly one model call per analyst step")
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.action_executed)
        self.assertEqual(runtime.actions_executed, 1)
        self.assertIsNotNone(result.evidence)
        self.assertTrue(result.control_returned)
        # The action really ran in the sandbox against the snapshot.
        self.assertIn("alpha", result.result.stdout)

    def test_tool_result_is_in_history_before_control_returns(self) -> None:
        backend = ScriptedBackend(tool_response("print('recorded')"))
        runtime = self.runtime(backend)
        runtime.step("inspect")

        roles = [m["role"] for m in runtime.messages]
        self.assertEqual(roles[-3:], ["user", "assistant", "tool"])
        self.assertIn("recorded", runtime.messages[-1]["content"])

    def test_continuation_requires_a_new_analyst_message(self) -> None:
        backend = ScriptedBackend(
            tool_response("print('first')"),
            tool_response("print('second')"),
        )
        runtime = self.runtime(backend)

        runtime.step("first instruction")
        self.assertEqual(backend.calls, 1, "the runtime must not continue on its own")

        runtime.step("continue")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(runtime.actions_executed, 2)

    def test_history_grows_append_only_across_steps(self) -> None:
        backend = ScriptedBackend(tool_response("print('one')"), tool_response("print('two')"))
        runtime = self.runtime(backend)
        runtime.step("first")
        after_first = [dict(m) for m in runtime.messages]
        runtime.step("verify the length of the recovered text")

        self.assertEqual(
            runtime.messages[: len(after_first)],
            after_first,
            "step 2 history must extend step 1 history unchanged",
        )

    def test_analyst_message_becomes_model_visible_history(self) -> None:
        # Steering only works if what the analyst said is actually in the
        # history the next call sees -- otherwise the model is steered by
        # nothing and the boundary is decorative.
        backend = ScriptedBackend(prose_response("ok"), prose_response("ok2"))
        runtime = self.runtime(backend)
        runtime.step("decode the second stage")

        user_texts = [m["content"] for m in runtime.messages if m["role"] == "user"]
        self.assertIn("decode the second stage", user_texts)

        runtime.step("verify the length")
        sent = backend.seen_messages[1]
        self.assertIn(
            "verify the length",
            [m.get("content") for m in sent],
            "the second call must see the new analyst instruction",
        )

    def test_prose_response_returns_without_any_action(self) -> None:
        backend = ScriptedBackend(prose_response("The artifact is plain text."))
        runtime = self.runtime(backend)
        result = runtime.step("summarise")

        self.assertEqual(backend.calls, 1)
        self.assertFalse(result.action_attempted)
        self.assertFalse(result.action_executed)
        self.assertEqual(result.assistant_text, "The artifact is plain text.")


class StructuralRejectionTest(AnalysisRuntimeTestBase):
    def test_two_tool_calls_execute_nothing(self) -> None:
        backend = ScriptedBackend(tool_response("print('x')", count=2))
        runtime = self.runtime(backend)
        result = runtime.step("do two things")

        self.assertEqual(backend.calls, 1, "no repair call may follow a rejection")
        self.assertTrue(result.action_attempted)
        self.assertFalse(result.action_executed, "neither call may run")
        self.assertEqual(runtime.actions_executed, 0)
        self.assertIn("at most one action", result.rejection)

    def test_unsupported_tool_executes_nothing(self) -> None:
        backend = ScriptedBackend(tool_response("print('x')", name="exec_shell_full_command"))
        runtime = self.runtime(backend)
        result = runtime.step("run a shell command")

        self.assertFalse(result.action_executed)
        self.assertEqual(runtime.actions_executed, 0)
        self.assertIn("unsupported tool", result.rejection)
        self.assertEqual(backend.calls, 1)

    def test_malformed_arguments_execute_nothing(self) -> None:
        bad = ChatResult(
            content="", model="m", finish_reason="stop",
            tool_calls=[{"id": "c", "type": "function",
                         "function": {"name": ANALYSIS_TOOL_NAME, "arguments": "{not json"}}],
            prompt_tokens=1, completion_tokens=1, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )
        backend = ScriptedBackend(bad)
        runtime = self.runtime(backend)
        result = runtime.step("go")

        self.assertFalse(result.action_executed)
        self.assertIn("not valid JSON", result.rejection)

    def test_oversized_code_is_refused_without_a_repair_call(self) -> None:
        backend = ScriptedBackend(tool_response("#" * (64 * 1024)))
        runtime = self.runtime(backend)
        result = runtime.step("go")

        self.assertEqual(backend.calls, 1)
        self.assertFalse(result.action_executed)
        self.assertIsNotNone(result.rejection)

    def test_rejection_is_recorded_in_history_for_the_analyst(self) -> None:
        backend = ScriptedBackend(tool_response("print('x')", count=2))
        runtime = self.runtime(backend)
        runtime.step("do two things")
        # Recorded in the assistant turn rather than a tool turn: a tool turn
        # would have to answer a `tool_calls` entry, and committing the refused
        # call is exactly what makes the history unrenderable afterwards.
        last = runtime.messages[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIn("refused", last["content"])
        self.assertNotIn("tool_calls", last)


class SourceOwnershipTest(AnalysisRuntimeTestBase):
    def test_snapshot_matches_acquired_bytes_and_original_is_untouched(self) -> None:
        import hashlib

        self.assertEqual(
            self.source.sha256, hashlib.sha256(FIXTURE.encode()).hexdigest()
        )
        self.assertEqual(self.source.snapshot_path.read_text(encoding="utf-8"), FIXTURE)
        self.assertEqual(self.original.read_text(encoding="utf-8"), FIXTURE)

    def test_analysis_uses_the_snapshot_not_the_mutable_original(self) -> None:
        backend = ScriptedBackend(tool_response("print(open('/workspace/input').read(), end='')"))
        runtime = self.runtime(backend)
        # Swapping the original after acquisition must not change what runs.
        self.original.write_text("SWAPPED", encoding="utf-8")
        result = runtime.step("read it")

        self.assertIn("alpha", result.result.stdout)
        self.assertNotIn("SWAPPED", result.result.stdout)

    def test_identity_is_content_not_path(self) -> None:
        self.assertEqual(self.source.analysis_id, self.source.sha256[:16])


class EvidenceBoundTest(AnalysisRuntimeTestBase):
    def test_large_output_is_bounded_for_the_prompt_but_recorded_in_full(self) -> None:
        backend = ScriptedBackend(tool_response("print('A' * 60000)"))
        runtime = self.runtime(backend)
        result = runtime.step("emit a lot")

        observation = runtime.messages[-1]["content"]
        self.assertLessEqual(
            len(observation), MAX_EVIDENCE_CHARS, "model-facing evidence must stay bounded"
        )
        self.assertIn("truncated for prompt", observation)
        metadata = result.evidence.metadata
        self.assertTrue(metadata["observation_truncated"])
        self.assertGreater(metadata["observation_full_chars"], MAX_EVIDENCE_CHARS)

    def test_evidence_carries_full_provenance(self) -> None:
        backend = ScriptedBackend(tool_response("open('/workspace/work/out.txt','w').write('d')\nprint('ok')"))
        runtime = self.runtime(backend)
        result = runtime.step("derive something")

        metadata = result.evidence.metadata
        self.assertEqual(metadata["analysis_source_sha256"], self.source.sha256)
        self.assertEqual(metadata["input_sha256"], self.source.sha256)
        self.assertEqual(metadata["status"], "ok")
        self.assertTrue(metadata["code_sha256"])
        self.assertEqual(metadata["artifacts"][0]["name"], "out.txt")

    def test_evidence_survives_across_steps(self) -> None:
        backend = ScriptedBackend(tool_response("print('first')"), tool_response("print('second')"))
        runtime = self.runtime(backend)
        first = runtime.step("one")
        runtime.step("continue")
        # Each action now records two entries: the bounded observation and the
        # durable raw output. Assert the property -- earlier evidence survives
        # and both actions are represented -- rather than a fixed count.
        self.assertIn(first.evidence.evidence_id, self.store.records)
        self.assertIn(first.raw_output_evidence_id, self.store.records)
        bounded = [r for r in self.store.records.values() if r.tool_name == "execute_analysis"]
        self.assertEqual(len(bounded), 2, "one bounded record per action")


class AnalysisContractTest(AnalysisRuntimeTestBase):
    def test_only_the_analysis_tool_is_offered(self) -> None:
        backend = ScriptedBackend(prose_response("ok"))
        self.runtime(backend).step("hello")
        self.assertEqual(backend.seen_tools[0], [ANALYSIS_TOOL_NAME])

    def test_chat_tools_are_absent_from_the_analysis_surface(self) -> None:
        from orbit.runtime.tools import TOOL_NAMES

        self.assertNotIn(ANALYSIS_TOOL_NAME, TOOL_NAMES)
        offered = ANALYSIS_TOOL_SCHEMA["function"]["name"]
        self.assertEqual(offered, ANALYSIS_TOOL_NAME)

    def test_stable_prefix_holds_no_volatile_identity(self) -> None:
        # A future exact-prefix prewarm can only cover text identical across
        # every analysis, so the source hash and path must sit after it.
        for volatile in (self.source.sha256, self.source.original_path, "workspace/owned"):
            self.assertNotIn(volatile, ANALYSIS_SYSTEM_PROMPT)

    def test_volatile_identity_appears_after_the_stable_prefix(self) -> None:
        backend = ScriptedBackend(prose_response("ok"))
        runtime = self.runtime(backend)
        self.assertEqual(runtime.messages[0]["content"], ANALYSIS_SYSTEM_PROMPT)
        self.assertIn(self.source.sha256, runtime.messages[1]["content"])


if __name__ == "__main__":
    unittest.main()


class DurableEvidenceTest(AnalysisRuntimeTestBase):
    """A recorded hash must name bytes somebody can still produce."""

    def test_full_stdout_survives_truncated_observation(self) -> None:
        backend = ScriptedBackend(tool_response("print('A' * 60000)"))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        result = runtime.step("emit a lot")

        observation = runtime.messages[-1]["content"]
        self.assertLessEqual(len(observation), MAX_EVIDENCE_CHARS)
        self.assertTrue(result.evidence.metadata["observation_truncated"])

        raw = self.store.load_raw(result.raw_output_evidence_id)
        self.assertGreaterEqual(raw.count("A"), 60000, "full output must remain retrievable")

    def test_derived_artifact_survives_the_step_and_reattests(self) -> None:
        import hashlib

        backend = ScriptedBackend(
            tool_response("open('/workspace/work/derived.txt','w').write('D' * 40000)\nprint('done')")
        )
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        result = runtime.step("derive")

        artifact = result.evidence.metadata["artifacts"][0]
        on_disk = (runtime.workspace.scratch_root / "derived.txt").read_bytes()
        self.assertEqual(
            hashlib.sha256(on_disk).hexdigest(),
            artifact["sha256"],
            "the recorded SHA must match bytes that still exist",
        )
        self.assertEqual(artifact["handle"], "/workspace/work/derived.txt")

    def test_later_step_reopens_the_previous_artifact(self) -> None:
        backend = ScriptedBackend(
            tool_response("open('/workspace/work/derived.txt','w').write('D' * 40000)\nprint('written')"),
            tool_response(
                "import orbit_tools\n"
                "print('REOPENED', len(orbit_tools.read_file('/workspace/work/derived.txt')))"
            ),
        )
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        runtime.step("derive")
        second = runtime.step("verify the derived file")

        self.assertIn("REOPENED 40000", second.result.stdout)

    def test_large_output_does_not_enter_the_next_prompt(self) -> None:
        backend = ScriptedBackend(tool_response("print('A' * 60000)"), prose_response("ok"))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        runtime.step("emit a lot")
        runtime.step("continue")

        sent = backend.seen_messages[1]
        for message in sent:
            self.assertLessEqual(
                len(message.get("content") or ""),
                MAX_EVIDENCE_CHARS,
                "durable raw output must not leak into the next prompt",
            )

    def test_truncation_metadata_keeps_full_size_and_raw_handle(self) -> None:
        backend = ScriptedBackend(tool_response("print('A' * 60000)"))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        metadata = runtime.step("emit").evidence.metadata

        self.assertGreater(metadata["observation_full_chars"], MAX_EVIDENCE_CHARS)
        self.assertTrue(metadata["raw_output_evidence_id"])

    def test_sessions_do_not_share_scratch(self) -> None:
        first = self.runtime(ScriptedBackend(tool_response("open('/workspace/work/a.txt','w').write('x')\nprint('1')")))
        self.addCleanup(first.close)
        first.step("one")

        second = self.runtime(ScriptedBackend(tool_response("import os\nprint('LEAK', os.path.exists('/workspace/work/a.txt'))")))
        self.addCleanup(second.close)
        result = second.step("two")

        self.assertNotEqual(first.workspace.root, second.workspace.root)
        self.assertIn("LEAK False", result.result.stdout)

    def test_close_removes_the_workspace_and_is_idempotent(self) -> None:
        runtime = self.runtime(ScriptedBackend(tool_response("print('x')")))
        runtime.step("one")
        root = runtime.workspace.root
        self.assertTrue(root.exists())
        runtime.close()
        self.assertFalse(root.exists())
        runtime.close()  # must not raise

    def test_host_paths_never_reach_model_visible_history(self) -> None:
        backend = ScriptedBackend(tool_response("open('/workspace/work/x.txt','w').write('y')\nprint('ok')"))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        runtime.step("derive")

        host_root = str(runtime.workspace.root)
        for message in runtime.messages:
            self.assertNotIn(host_root, str(message.get("content") or ""))


class CumulativeWedgeRegressionTest(AnalysisRuntimeTestBase):
    """MAJOR-1: retained artifacts must not fail a later compliant action.

    The historical 8 MiB / 32-file allowance was written for a scratch
    directory that lived and died with one action. The session workspace
    outlives the action, so scanning it whole charges every step for every
    file every earlier step left behind. Eight 60 KiB files is a legal
    action; doing it repeatedly must stay legal.
    """

    def _writer(self, step: int) -> str:
        return (
            "for j in range(8):\n"
            f"    open('/workspace/work/s{step}_%d.bin' % j, 'w').write('Z' * 60000)\n"
            f"print('step {step} done')"
        )

    def test_repeated_compliant_actions_are_not_charged_for_history(self) -> None:
        backend = ScriptedBackend(*[tool_response(self._writer(i)) for i in range(5)])
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)

        for step in range(5):
            result = runtime.step(f"step {step}")
            self.assertEqual(
                result.result.status,
                "ok",
                f"step {step} writes 8 files / 480 KiB -- within the per-action "
                f"allowance -- but was judged {result.result.status!r} "
                f"({result.result.bound_exceeded!r})",
            )
            self.assertIsNone(result.result.bound_exceeded)
            self.assertEqual(
                len(result.evidence.metadata["artifacts"]),
                8,
                f"step {step} must record the 8 files it created",
            )

    def test_capacity_is_never_silently_lossy(self) -> None:
        """A bounded action must never look like a success that lost artifacts.

        Driven by an action that genuinely trips the bound, so the assertions
        actually execute: a version of this test that only ran compliant work
        would assert nothing at all on the happy path.
        """
        backend = ScriptedBackend(
            tool_response(
                "for j in range(40):\n"
                "    open('/workspace/work/many_%d.bin' % j, 'w').write('Z')\n"
                "print('LOOKS SUCCESSFUL')"
            )
        )
        runtime = self.runtime(backend)
        result = runtime.step("overrun the action bound")

        self.assertEqual(result.result.status, "bounded", "this action must trip the bound")
        self.assertIsNotNone(result.result.bound_exceeded, "it must say which bound it hit")
        self.assertEqual(result.evidence.metadata["artifacts"], [])
        # The give-away the original defect had: stdout reads like a success.
        self.assertIn("LOOKS SUCCESSFUL", result.result.stdout)
        raw = self.store.load_raw(result.raw_output_evidence_id)
        self.assertIn(
            result.result.bound_exceeded,
            raw,
            "the recorded evidence must state the bound, not just the happy stdout",
        )
        self.assertIn(
            f"status: {result.result.status}",
            raw,
            "the durable record must state the status it is attesting to",
        )


class DirectoryAccountingTest(AnalysisRuntimeTestBase):
    """Directories must be charged like files: once, to the action that made them.

    The flat-file regression alone missed this. `work/tmp` exists from the
    first action onward because bwrap creates it, so a directory recharged
    every step wedges a session that only ever writes into subdirectories.
    """

    def test_nested_actions_do_not_accumulate_a_directory_charge(self) -> None:
        steps = 34  # past the 32-entry per-action allowance
        backend = ScriptedBackend(
            *[
                tool_response(
                    f"import os\n"
                    f"os.makedirs('/workspace/work/d{i}', exist_ok=True)\n"
                    f"open('/workspace/work/d{i}/f.txt','w').write('x')\n"
                    f"print('ok {i}')"
                )
                for i in range(steps)
            ]
        )
        runtime = self.runtime(backend)

        for index in range(steps):
            result = runtime.step(f"step {index}")
            self.assertEqual(
                result.result.status,
                "ok",
                f"step {index} created one directory and one file -- well inside the "
                f"per-action allowance -- but was judged {result.result.bound_exceeded!r}",
            )
            self.assertEqual(len(result.evidence.metadata["artifacts"]), 1)

    def test_session_usage_counts_real_bytes_not_directory_sentinels(self) -> None:
        """Directories must not discount the retained byte total."""
        backend = ScriptedBackend(
            *[
                tool_response(
                    f"import os\n"
                    f"os.makedirs('/workspace/work/d{i}', exist_ok=True)\n"
                    f"open('/workspace/work/d{i}/f.txt','w').write('Z' * 1000)\n"
                    f"print('ok')"
                )
                for i in range(3)
            ]
        )
        runtime = self.runtime(backend)

        seen = []
        for index in range(3):
            runtime.step(f"step {index}")
            used_bytes, used_files = runtime.session_usage()
            self.assertGreater(used_bytes, 0, "directories must not drive the total negative")
            self.assertEqual(
                used_files,
                index + 1,
                "only real files count -- one per step here, whatever the directory count",
            )
            seen.append(used_bytes)

        self.assertEqual(seen, [1000, 2000, 3000], "retained bytes must grow with real content")

    def test_empty_directories_alone_do_not_exhaust_the_session(self) -> None:
        """Directories hold no data, so they must not consume the file cap."""
        steps = 6
        backend = ScriptedBackend(
            *[
                tool_response(
                    f"import os\n"
                    f"for j in range(30):\n"
                    f"    os.makedirs('/workspace/work/s{i}_%d' % j, exist_ok=True)\n"
                    f"print('dirs')"
                )
                for i in range(steps)
            ]
        )
        runtime = self.runtime(backend)

        for index in range(steps):
            result = runtime.step(f"step {index}")
            used_bytes, used_files = runtime.session_usage()
            self.assertTrue(
                result.action_executed,
                f"step {index} was refused with {used_files} files / {used_bytes} bytes "
                f"retained, but the workspace holds no file at all",
            )
            self.assertEqual(used_files, 0)
            self.assertEqual(used_bytes, 0)

    def test_an_action_creating_too_many_directories_is_still_bounded(self) -> None:
        backend = ScriptedBackend(
            tool_response(
                "import os\n"
                "for j in range(40):\n"
                "    os.makedirs('/workspace/work/many_%d' % j, exist_ok=True)\n"
                "print('made')"
            )
        )
        runtime = self.runtime(backend)
        result = runtime.step("make too many directories at once")

        self.assertEqual(result.result.status, "bounded")
        self.assertEqual(
            result.result.bound_exceeded,
            "scratch bound exceeded",
            "it must be the per-action allowance that fired, not some other bound",
        )


class HardLinkRecoveryTest(AnalysisRuntimeTestBase):
    """A hard link is refused as a bound, not raised as an exception.

    `os.link` is ordinary Python available to model-authored code. Raising at
    capture time would leave the link in the session workspace and brick every
    later step, so the entry is rejected while the action is being bounded.
    """

    def test_hard_link_is_bounded_and_the_session_recovers(self) -> None:
        backend = ScriptedBackend(
            tool_response(
                "import os\n"
                "open('/workspace/work/a.txt','w').write('x')\n"
                "os.link('/workspace/work/a.txt','/workspace/work/b.txt')\n"
                "print('linked')"
            ),
            tool_response("import os\nos.unlink('/workspace/work/b.txt')\nprint('cleaned')"),
            tool_response("print('still usable')"),
        )
        runtime = self.runtime(backend)

        linked = runtime.step("make a hard link")
        self.assertEqual(linked.result.status, "bounded")
        self.assertEqual(linked.result.bound_exceeded, "scratch contains a hard link")
        self.assertEqual(linked.evidence.metadata["artifacts"], [])

        self.assertEqual(runtime.step("remove the link").result.status, "ok")
        self.assertEqual(runtime.step("carry on").result.status, "ok")


class UnsafeEntryRecoveryTest(AnalysisRuntimeTestBase):
    """Entries the capture step cannot read are bounded, never raised.

    The workspace persists, so an exception escaping `step()` would repeat on
    every later step and end the session. Each case here must therefore be a
    clean bound that the analyst can undo.
    """

    def _case(self, make: str, undo: str, expected: str) -> None:
        backend = ScriptedBackend(
            tool_response(make), tool_response(undo), tool_response("print('fine')")
        )
        runtime = self.runtime(backend)

        first = runtime.step("create the unsafe entry")
        self.assertEqual(first.result.status, "bounded")
        self.assertEqual(first.result.bound_exceeded, expected)
        self.assertEqual(runtime.step("undo it").result.status, "ok")
        self.assertEqual(runtime.step("carry on").result.status, "ok")

    def test_unreadable_file_is_bounded_and_recoverable(self) -> None:
        self._case(
            "import os\n"
            "open('/workspace/work/a','w').write('x')\n"
            "os.chmod('/workspace/work/a', 0o000)\n"
            "print('locked')",
            "import os\nos.chmod('/workspace/work/a', 0o600)\nprint('unlocked')",
            "scratch contains an unreadable entry",
        )

    def test_fifo_is_bounded_and_recoverable(self) -> None:
        self._case(
            "import os\nos.mkfifo('/workspace/work/p')\nprint('fifo')",
            "import os\nos.unlink('/workspace/work/p')\nprint('removed')",
            "scratch contains an unsupported entry",
        )

    def test_undecodable_name_is_bounded_and_recoverable(self) -> None:
        self._case(
            "import os\n"
            "os.close(os.open(b'/workspace/work/' + bytes([0xff, 0xfe]),"
            " os.O_CREAT | os.O_WRONLY, 0o600))\n"
            "print('made')",
            "import os\n"
            "for name in os.listdir(b'/workspace/work'):\n"
            "    path = os.path.join(b'/workspace/work', name)\n"
            "    if not os.path.isdir(path):\n"
            "        os.unlink(path)\n"
            "print('removed')",
            "scratch contains an undecodable name",
        )

    def test_unreadable_directory_is_bounded_and_recoverable(self) -> None:
        """A directory os.walk cannot enter must not hide its contents.

        Left unbounded, the file inside goes uncharged and uncaptured, then is
        attributed to whichever later action happens to unlock the directory.
        """
        self._case(
            "import os\n"
            "os.makedirs('/workspace/work/d', exist_ok=True)\n"
            "open('/workspace/work/d/f','w').write('x')\n"
            "os.chmod('/workspace/work/d', 0o000)\n"
            "print('locked')",
            "import os\nos.chmod('/workspace/work/d', 0o700)\nprint('unlocked')",
            "scratch contains an unreadable entry",
        )

    def test_symlink_is_bounded_and_recoverable(self) -> None:
        self._case(
            "import os\nos.symlink('/etc/passwd','/workspace/work/s')\nprint('sym')",
            "import os\nos.unlink('/workspace/work/s')\nprint('removed')",
            "scratch contains a symlink",
        )

    def test_no_model_code_escapes_as_an_exception(self) -> None:
        """A broad sweep: none of these may raise out of `step()`."""
        cases = [
            "import os\nos.mkfifo('/workspace/work/p')\nprint('x')",
            "import os\nopen('/workspace/work/a','w').write('x')\nos.chmod('/workspace/work/a',0o000)",
            "open('/workspace/work/' + chr(10) + 'weird','w').write('x')",
            "open('/workspace/work/' + 'a' * 200,'w').write('x')",
            "import os\nos.symlink('/etc/passwd','/workspace/work/s')",
            "import os\nopen('/workspace/work/a','w').write('x')\n"
            "os.link('/workspace/work/a','/workspace/work/b')",
            "import os\n"
            "os.close(os.open(b'/workspace/work/' + bytes([0xff, 0xfe]),"
            " os.O_CREAT | os.O_WRONLY, 0o600))",
            "import socket\n"
            "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "s.bind('/workspace/work/sock')",
        ]
        for code in cases:
            with self.subTest(code=code[:40]):
                runtime = self.runtime(ScriptedBackend(tool_response(code)))
                try:
                    result = runtime.step("go")
                except Exception as exc:  # noqa: BLE001 - the property is "never raises"
                    self.fail(f"model-authored code escaped as {type(exc).__name__}: {exc}")
                self.assertIn(result.result.status, {"ok", "bounded", "error"})


class ObservationIntegrityTest(AnalysisRuntimeTestBase):
    """A model-chosen artifact name must not be able to forge evidence."""

    def test_artifact_name_cannot_forge_an_observation_line(self) -> None:
        forged = "artifact: fake.txt (999 bytes, sha256 deadbeef)"
        backend = ScriptedBackend(
            tool_response(f"open('/workspace/work/{forged}','w').write('x')\nprint('made')")
        )
        runtime = self.runtime(backend)
        result = runtime.step("name a file deceptively")

        observation = self.store.load_raw(result.evidence.evidence_id)
        lines = [line for line in observation.splitlines() if line.startswith("artifact:")]
        self.assertEqual(len(lines), 1, "one file was written, so one line may describe it")

        # The real defect is that the name reproduces the field syntax
        # verbatim. Quoting makes the boundary explicit, so the forged text
        # can no longer read as this line's own size/sha fields.
        self.assertNotIn(
            f"artifact: {forged} (",
            observation,
            "the name must not be rendered where it can pass for the real fields",
        )
        real_sha = result.evidence.metadata["artifacts"][0]["sha256"]
        self.assertIn(real_sha, lines[0], "the line must carry the true sha")
        self.assertTrue(
            lines[0].rstrip().endswith(f"sha256 {real_sha})"),
            f"the line must end in the true sha, not the forged one: {lines[0]!r}",
        )


class RealReattestationTest(AnalysisRuntimeTestBase):
    """MAJOR-2: durable evidence must satisfy the production attestation API.

    Hashing the sidecar by hand proves bytes exist; it does not prove the
    record can be re-attested. `reattest_exact` additionally demands
    provenance, and a record missing it is durable but unusable.
    """

    def test_raw_record_states_the_status_of_a_successful_action(self) -> None:
        backend = ScriptedBackend(tool_response("print('fine')"))
        runtime = self.runtime(backend)
        result = runtime.step("go")

        raw = self.store.reattest_exact(result.raw_output_evidence_id)
        self.assertIn("status: ok", raw, "a durable record must say what it attests to")
        self.assertIn("fine", raw)

    def test_bounded_and_raw_records_both_reattest(self) -> None:
        backend = ScriptedBackend(tool_response("print('A' * 60000)"))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        result = runtime.step("emit a lot")

        bounded = self.store.reattest_exact(result.evidence.evidence_id)
        raw = self.store.reattest_exact(result.raw_output_evidence_id)

        self.assertIsNotNone(bounded, "bounded record must pass production reattestation")
        self.assertIsNotNone(raw, "raw record must pass production reattestation")
        # These must be the stored evidence itself, not a digest that merely
        # looks like proof: a hex hash would satisfy every assertion below.
        self.assertEqual(bounded, self.store.load_raw(result.evidence.evidence_id))
        self.assertEqual(raw, self.store.load_raw(result.raw_output_evidence_id))
        self.assertEqual(len(raw), self.store.records[result.raw_output_evidence_id].raw_chars)
        self.assertEqual(len(bounded), self.store.records[result.evidence.evidence_id].raw_chars)
        self.assertLessEqual(len(bounded), MAX_EVIDENCE_CHARS)
        self.assertIn("A" * 60000, raw, "raw reattestation returns the full output")
        self.assertNotEqual(bounded, raw, "raw and bounded must not be confusable")

    def test_reattestation_survives_a_later_step(self) -> None:
        backend = ScriptedBackend(
            tool_response("print('A' * 60000)"),
            tool_response("print('later')"),
        )
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        first = runtime.step("emit a lot")
        runtime.step("do something else")

        self.assertIsNotNone(self.store.reattest_exact(first.evidence.evidence_id))
        raw = self.store.reattest_exact(first.raw_output_evidence_id)
        self.assertIsNotNone(raw)
        self.assertIn("A" * 60000, raw)

    def test_records_carry_the_models_own_call_id(self) -> None:
        backend = ScriptedBackend(tool_response("print('x')"))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        result = runtime.step("go")

        # `tool_response` emits id "call_0"; the record must carry that exact
        # value rather than a placeholder minted to satisfy the check.
        emitted = "call_0"
        for record_id in (result.evidence.evidence_id, result.raw_output_evidence_id):
            record = self.store.records[record_id]
            self.assertEqual(record.tool_call_id, emitted, "must be the model's id, not a synthetic one")
            self.assertEqual(record.user_turn_id, "turn_1")
            self.assertTrue(record.produced_by_phase)


class NegativeReattestationTest(AnalysisRuntimeTestBase):
    """Reattestation must refuse anything it cannot vouch for."""

    def _step(self):
        backend = ScriptedBackend(
            tool_response("open('/workspace/work/d.txt','w').write('D' * 500)\nprint('done')")
        )
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        return runtime, runtime.step("derive")

    def test_missing_provenance_field_fails(self) -> None:
        import dataclasses

        _, result = self._step()
        rid = result.evidence.evidence_id
        self.assertIsNotNone(self.store.reattest_exact(rid))
        self.store.records[rid] = dataclasses.replace(self.store.records[rid], tool_call_id=None)
        self.assertIsNone(self.store.reattest_exact(rid), "no provenance, no attestation")

    def test_modified_sidecar_content_fails(self) -> None:
        _, result = self._step()
        rid = result.raw_output_evidence_id
        self.assertIsNotNone(self.store.reattest_exact(rid))
        path = self.store.root / f"{rid}.txt"
        path.write_text(path.read_text() + "tampered", encoding="utf-8")
        self.assertIsNone(self.store.reattest_exact(rid), "content hash must be checked")

    def test_wrong_expected_reference_fails(self) -> None:
        _, result = self._step()
        rid = result.evidence.evidence_id
        self.assertIsNone(
            self.store.reattest_exact(rid, expected_reference="ev_not_this_one"),
            "a mismatched reference must not attest",
        )

    def test_raw_and_bounded_are_not_interchangeable(self) -> None:
        _, result = self._step()
        bounded = self.store.reattest_exact(result.evidence.evidence_id)
        raw = self.store.reattest_exact(result.raw_output_evidence_id)
        self.assertNotEqual(bounded, raw)
        self.assertNotEqual(result.evidence.evidence_id, result.raw_output_evidence_id)

    def test_changed_artifact_bytes_break_the_recorded_sha(self) -> None:
        import hashlib

        runtime, result = self._step()
        artifact = result.evidence.metadata["artifacts"][0]
        path = runtime.workspace.scratch_root / "d.txt"
        path.write_bytes(b"E" * 500)
        self.assertNotEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            artifact["sha256"],
            "the recorded SHA must not silently describe rewritten bytes",
        )


class SessionCapacityTest(AnalysisRuntimeTestBase):
    """The session bound is separate from the per-action bound.

    Tests here shrink the session cap rather than write 64 MiB, so the policy
    is exercised without the runtime becoming slow enough that nobody runs it.
    """

    def _shrink(self, runtime, *, files=None, byte_cap=None):
        import orbit.runtime.analysis_runtime as module

        if files is not None:
            self.enterContext(unittest.mock.patch.object(module, "MAX_SESSION_SCRATCH_FILES", files))
        if byte_cap is not None:
            self.enterContext(unittest.mock.patch.object(module, "MAX_SESSION_SCRATCH_BYTES", byte_cap))
        return runtime

    def _writer(self, step: int, count: int = 8, size: int = 60000) -> str:
        return (
            f"for j in range({count}):\n"
            f"    open('/workspace/work/s{step}_%d.bin' % j, 'w').write('Z' * {size})\n"
            f"print('step {step}')"
        )

    def test_a_action_exceeding_the_file_allowance_is_bounded(self) -> None:
        backend = ScriptedBackend(tool_response(self._writer(0, count=40, size=10)))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        result = runtime.step("write too many files at once")

        self.assertEqual(result.result.status, "bounded")
        self.assertEqual(result.result.bound_exceeded, "scratch bound exceeded")
        self.assertEqual(result.evidence.metadata["artifacts"], [])

    def test_b_action_exceeding_the_byte_allowance_is_bounded(self) -> None:
        """Drive the byte bound specifically, with the file count kept legal.

        RLIMIT_FSIZE caps any single file at 64 KiB, so exceeding 8 MiB needs
        many files -- but 20 stays well under the 32-file allowance, so a
        failure here can only be the byte path. Checked directly against
        `_scratch_bound_error` too, since a status alone would not say which
        of the two bounds fired.
        """
        from orbit.runtime.analysis_sandbox import (
            MAX_SCRATCH_BYTES,
            MAX_SCRATCH_FILES,
            _scratch_bound_error,
        )

        scratch = self.tmp / "bytes-probe"
        scratch.mkdir()
        for index in range(20):
            (scratch / f"f{index}.bin").write_bytes(b"Z" * (MAX_SCRATCH_BYTES // 10))

        self.assertLess(20, MAX_SCRATCH_FILES, "the file count must stay legal")
        self.assertEqual(
            _scratch_bound_error(scratch),
            "scratch bound exceeded",
            "20 files totalling 2x the byte allowance must trip the byte bound",
        )

    def test_b2_file_allowance_is_enforced_independently_of_bytes(self) -> None:
        from orbit.runtime.analysis_sandbox import MAX_SCRATCH_FILES, _scratch_bound_error

        scratch = self.tmp / "count-probe"
        scratch.mkdir()
        for index in range(MAX_SCRATCH_FILES + 2):
            (scratch / f"f{index}.bin").write_bytes(b"Z")

        self.assertEqual(_scratch_bound_error(scratch), "scratch bound exceeded")

    def test_c_many_legal_actions_reach_an_explicit_session_cap(self) -> None:
        backend = ScriptedBackend(*[tool_response(self._writer(i)) for i in range(4)])
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        self._shrink(runtime, files=20)

        outcomes = [runtime.step(f"step {i}") for i in range(4)]
        refusals = [r for r in outcomes if r.rejection and SESSION_CAPACITY_EXHAUSTED in r.rejection]

        self.assertTrue(refusals, "the session cap must eventually be reported")
        for refused in refusals:
            self.assertFalse(refused.action_executed, "capacity is refused before running code")
            self.assertIsNone(refused.result)

    def test_d_evidence_stays_readable_after_the_cap(self) -> None:
        backend = ScriptedBackend(*[tool_response(self._writer(i)) for i in range(3)])
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        self._shrink(runtime, files=10)

        first = runtime.step("step 0")
        self.assertTrue(first.action_executed)
        later = [runtime.step(f"step {i}") for i in (1, 2)]
        self.assertTrue(any(r.rejection and SESSION_CAPACITY_EXHAUSTED in r.rejection for r in later))

        # Nothing was evicted to make room.
        self.assertIsNotNone(self.store.reattest_exact(first.evidence.evidence_id))
        self.assertIsNotNone(self.store.reattest_exact(first.raw_output_evidence_id))
        for artifact in first.evidence.metadata["artifacts"]:
            self.assertTrue((runtime.workspace.scratch_root / artifact["name"]).is_file())
        runtime.close()
        self.assertFalse(runtime.workspace.root.exists())

    def test_f_byte_session_cap_is_enforced_independently(self) -> None:
        """Exhaust the session by bytes while the file count stays legal."""
        backend = ScriptedBackend(*[tool_response(self._writer(i, count=4)) for i in range(4)])
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        # 4 files x 60 KiB per step; cap bytes low, leave the file cap high.
        self._shrink(runtime, files=10_000, byte_cap=300_000)

        outcomes = [runtime.step(f"step {i}") for i in range(4)]
        refusals = [r for r in outcomes if r.rejection and SESSION_CAPACITY_EXHAUSTED in r.rejection]

        self.assertTrue(refusals, "the session BYTE cap must be enforced on its own")
        used_bytes, used_files = runtime.session_usage()
        self.assertLess(used_files, 10_000, "the file cap must not be what fired")
        for refused in refusals:
            self.assertFalse(refused.action_executed)

    def test_e_capacity_refusal_makes_no_extra_model_call(self) -> None:
        backend = ScriptedBackend(tool_response(self._writer(0)), tool_response(self._writer(1)))
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)
        self._shrink(runtime, files=4)

        runtime.step("step 0")
        runtime.step("step 1")
        self.assertEqual(backend.calls, 2, "one model call per analyst step, cap or no cap")


class ModifiedArtifactProvenanceTest(AnalysisRuntimeTestBase):
    """Rewriting a derived file must produce a new, distinguishable version."""

    def test_rewrite_is_recorded_as_a_new_version(self) -> None:
        backend = ScriptedBackend(
            tool_response("open('/workspace/work/d.txt','w').write('A' * 100)\nprint('v1')"),
            tool_response("open('/workspace/work/d.txt','w').write('B' * 200)\nprint('v2')"),
        )
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)

        first = runtime.step("write it")
        second = runtime.step("rewrite it")

        v1 = first.evidence.metadata["artifacts"][0]
        v2 = second.evidence.metadata["artifacts"][0]
        self.assertEqual(v1["name"], v2["name"])
        self.assertNotEqual(v1["sha256"], v2["sha256"], "a rewrite must get a new hash")
        self.assertEqual(v1["size_bytes"], 100)
        self.assertEqual(v2["size_bytes"], 200)
        # The superseded record still attests to what it actually described.
        self.assertIsNotNone(self.store.reattest_exact(first.evidence.evidence_id))
        self.assertNotEqual(first.evidence.evidence_id, second.evidence.evidence_id)

    def test_untouched_artifacts_are_not_reattributed(self) -> None:
        backend = ScriptedBackend(
            tool_response("open('/workspace/work/keep.txt','w').write('K' * 50)\nprint('made')"),
            tool_response("print('touched nothing')"),
        )
        runtime = self.runtime(backend)
        self.addCleanup(runtime.close)

        runtime.step("create")
        second = runtime.step("do nothing to the file")

        self.assertEqual(
            second.evidence.metadata["artifacts"], [],
            "an action that wrote nothing must not claim an earlier file",
        )


class InputMountContractTest(AnalysisRuntimeTestBase):
    """`/workspace/input` is the artifact file, and every surface says so.

    A production run read `/workspace/input/<original filename>` and was
    correctly refused: the sandbox binds the artifact *as* that path, so a
    subpath under it cannot exist. The wording invited the mistake by
    describing the artifact as "mounted at" a bare path, which reads as a
    directory. These pin the corrected contract on all three model-facing
    surfaces, because one of them reverting would reintroduce the ambiguity.
    """

    def _surfaces(self) -> dict[str, str]:
        runtime = self.runtime(ScriptedBackend(prose_response("ok")))
        return {
            "system prompt": ANALYSIS_SYSTEM_PROMPT,
            "tool schema": ANALYSIS_TOOL_SCHEMA["function"]["description"],
            "first user turn": runtime.messages[1]["content"],
        }

    def test_every_surface_calls_it_a_file(self) -> None:
        for name, text in self._surfaces().items():
            with self.subTest(surface=name):
                lowered = text.lower()
                self.assertIn("/workspace/input", lowered)
                self.assertIn("file", lowered, "the surface must name it a file")

    # Prepositions that put the artifact *inside* the path rather than
    # equating it with the path. Checking the literal old sentence was not
    # enough: "the artifact file lives read-only under /workspace/input"
    # keeps every required keyword and still says directory.
    CONTAINER_FRAMINGS = (
        "mounted at /workspace/input",
        "at /workspace/input",
        "under /workspace/input",
        "in /workspace/input",
        "inside /workspace/input",
        "into /workspace/input",
        "from /workspace/input/",
        "within /workspace/input",
        "/workspace/input directory",
        "/workspace/input folder",
        "/workspace/input area",
        "/workspace/input/",
    )

    def test_no_surface_describes_it_as_a_container(self) -> None:
        # Not a keyword check: these are the shapes that make the path read as
        # somewhere the artifact lives rather than as the artifact.
        for name, text in self._surfaces().items():
            lowered = text.lower()
            for framing in self.CONTAINER_FRAMINGS:
                with self.subTest(surface=name, framing=framing):
                    self.assertNotIn(
                        framing,
                        lowered,
                        f"{name} frames /workspace/input as a container",
                    )

    def test_each_surface_equates_the_path_with_the_file(self) -> None:
        """Positive form: the path must be *identified* with the artifact.

        The prohibition above can be satisfied by saying nothing at all, so
        each surface also has to make the equation explicitly.
        """
        equations = (
            "/workspace/input is the artifact file",
            "/workspace/input is the read-only artifact file",
            "the file /workspace/input",
        )
        for name, text in self._surfaces().items():
            lowered = text.lower()
            with self.subTest(surface=name):
                self.assertTrue(
                    any(phrase in lowered for phrase in equations),
                    f"{name} never equates /workspace/input with the file",
                )

    def test_the_system_prompt_forbids_appending_a_path(self) -> None:
        lowered = ANALYSIS_SYSTEM_PROMPT.lower()
        self.assertIn("read exactly", lowered)
        self.assertIn("never append", lowered)
        self.assertIn("subpath", lowered)

    def test_the_tool_schema_says_it_is_not_a_directory(self) -> None:
        description = ANALYSIS_TOOL_SCHEMA["function"]["description"].lower()
        self.assertIn("not a directory", description)

    def test_derived_files_still_belong_under_the_work_root(self) -> None:
        for name, text in self._surfaces().items():
            if "/workspace/work" not in text:
                continue
            with self.subTest(surface=name):
                self.assertIn("/workspace/work", text)
        self.assertIn("/workspace/work", ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("/workspace/work", ANALYSIS_TOOL_SCHEMA["function"]["description"])

    def test_the_contract_names_no_sample_and_no_malware_hint(self) -> None:
        # The fix must generalise; a filename from the failing run would make
        # it advice about one artifact.
        for name, text in self._surfaces().items():
            with self.subTest(surface=name):
                lowered = text.lower()
                for banned in ("fattura", ".js", "malware", "dropper", "sample"):
                    self.assertNotIn(banned, lowered)

    def test_the_shim_exposes_the_same_path_it_accepts(self) -> None:
        # The prompt points at `orbit_tools.SOURCE_PATH`; that has to be the
        # one path `_safe_path` will allow, or the hint sends the model wrong.
        from orbit.runtime.analysis_tools_shim import ORBIT_TOOLS_SOURCE, SOURCE_MOUNT
        from orbit.runtime.analysis_sandbox import SOURCE_MOUNT as SANDBOX_MOUNT

        self.assertEqual(SOURCE_MOUNT, SANDBOX_MOUNT)
        self.assertIn(f'SOURCE_PATH = "{SOURCE_MOUNT}"', ORBIT_TOOLS_SOURCE)
        self.assertIn("SOURCE_PATH", ANALYSIS_SYSTEM_PROMPT)

    def test_the_stable_prefix_still_holds_no_volatile_identity(self) -> None:
        # The wording grew; it must not have grown to include session data.
        for volatile in (self.source.sha256, self.source.original_path):
            self.assertNotIn(volatile, ANALYSIS_SYSTEM_PROMPT)
            self.assertNotIn(volatile, ANALYSIS_TOOL_SCHEMA["function"]["description"])

    def test_the_first_turn_never_shows_the_original_filename(self) -> None:
        """The analyst's own path must not be echoed back at the model.

        Naming `samples/<file>` beside the mount is exactly what invites
        appending it to `/workspace/input`. The size and hash identify the
        artifact without handing the model a path to concatenate.
        """
        runtime = self.runtime(ScriptedBackend(prose_response("ok")))
        first_turn = runtime.messages[1]["content"]

        self.assertNotIn(self.source.original_path, first_turn)
        self.assertNotIn(Path(self.source.original_path).name, first_turn)
        self.assertIn(self.source.sha256, first_turn)

    def test_chat_prompts_are_untouched_by_the_analysis_contract(self) -> None:
        from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT

        for prompt in (CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT):
            self.assertNotIn("/workspace/input", prompt)


class ProgressSeamTest(AnalysisRuntimeTestBase):
    """A step must be observable while it runs, and cost nothing to observe.

    A single analysis call can occupy minutes. Without a signal the terminal
    shows nothing until it returns, which is indistinguishable from a hang.
    """

    class _ProgressBackend(ScriptedBackend):
        """Emits a prefill and a generation update, as the real stream does."""

        def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                        on_delta, on_progress=None):
            from orbit.backend.base import StreamProgress

            if on_progress is not None:
                on_progress(StreamProgress(
                    phase="prefill", current=200, total=400, percent=50,
                    evaluated_current=200, evaluated_total=400, cached_tokens=768,
                ))
                on_progress(StreamProgress(
                    phase="generation", current=12, total=0, percent=0,
                    elapsed_seconds=1.5, tokens_per_second=11.0,
                ))
            return super().chat_stream(
                messages, temperature=temperature, max_tokens=max_tokens,
                tools=tools, on_delta=on_delta, on_progress=on_progress,
            )

    def test_progress_arrives_before_the_step_returns(self) -> None:
        seen: list[str] = []
        backend = self._ProgressBackend(prose_response("done"))
        runtime = self.runtime(backend)

        result = runtime.step("look", on_progress=lambda u: seen.append(u.phase))

        self.assertEqual(seen, ["prefill", "generation"])
        self.assertIsNotNone(result)

    def test_observing_adds_no_model_call(self) -> None:
        backend = self._ProgressBackend(prose_response("done"))
        runtime = self.runtime(backend)

        runtime.step("look", on_progress=lambda u: None, on_delta=lambda t: None)

        self.assertEqual(backend.calls, 1, "one analyst line is one model call")

    def test_progress_text_never_enters_the_conversation(self) -> None:
        backend = self._ProgressBackend(prose_response("done"))
        runtime = self.runtime(backend)

        runtime.step("look", on_progress=lambda u: None)

        sent = json.dumps(backend.seen_messages, default=str)
        for marker in ("prefill", "generation", "tok/s", "analysis ·"):
            self.assertNotIn(marker, sent)
        for message in runtime.messages:
            self.assertNotIn("prefill", str(message.get("content") or ""))

    def test_prose_streams_but_tool_arguments_never_do(self) -> None:
        """The delta seam carries assistant text only.

        A tool call is generated as JSON that is only valid once complete, so
        streaming it would put unparsed model output on the analyst's screen.
        """
        streamed: list[str] = []
        marker = "SENTINEL_IN_TOOL_CODE"
        code = f"import orbit_tools\nprint('{marker}')"
        backend = ScriptedBackend(tool_response(code, text="thinking about it"))
        runtime = self.runtime(backend)

        runtime.step("look", on_delta=streamed.append)

        joined = "".join(streamed)
        self.assertIn("thinking about it", joined)
        # The delta seam must carry prose and nothing else. Checked on a
        # marker rather than the whole code string: the arguments are
        # JSON-encoded, so escaping alone would hide a literal-substring
        # check while the payload still reached the screen.
        self.assertNotIn(marker, joined)
        self.assertNotIn("orbit_tools", joined)
        self.assertNotIn("{", joined)
        self.assertEqual(joined, "thinking about it")

    def test_a_refused_call_streams_no_arguments_either(self) -> None:
        """The unparseable case is the one that must never be shown."""
        streamed: list[str] = []
        backend = ScriptedBackend(tool_response("x", text="working"))
        backend._responses[0].tool_calls[0]["function"]["arguments"] = "{BROKEN_SENTINEL"
        runtime = self.runtime(backend)

        result = runtime.step("look", on_delta=streamed.append)

        self.assertIsNotNone(result.rejection)
        self.assertNotIn("BROKEN_SENTINEL", "".join(streamed))

    def test_callbacks_are_optional(self) -> None:
        backend = ScriptedBackend(prose_response("done"))
        runtime = self.runtime(backend)

        result = runtime.step("look")

        self.assertEqual(result.model_calls, 1)


class StepDiagnosticsTest(AnalysisRuntimeTestBase):
    """Every step records what its model call cost -- refusals included.

    A refused step writes no evidence, so it used to leave no trace at all of
    how long the model ran or how much it generated. That is precisely the
    turn someone needs to diagnose later.
    """

    def test_a_successful_action_records_its_cost(self) -> None:
        backend = ScriptedBackend(tool_response("import orbit_tools\nprint('x')"))
        result = self.runtime(backend).step("look")

        diag = result.diagnostics
        self.assertIsNotNone(diag)
        assert diag is not None
        self.assertEqual(diag.prompt_tokens, 10)
        self.assertEqual(diag.output_tokens, 5)
        self.assertEqual(diag.tool_call_count, 1)
        self.assertGreater(diag.tool_argument_chars, 0)
        self.assertIsNone(diag.refusal)
        self.assertIsNotNone(diag.duration_seconds)

    def test_a_refused_step_still_records_its_cost(self) -> None:
        backend = ScriptedBackend(tool_response("x", name=ANALYSIS_TOOL_NAME))
        # Replace the arguments with something unparseable.
        backend._responses[0].tool_calls[0]["function"]["arguments"] = "{not json"
        result = self.runtime(backend).step("look")

        self.assertIsNotNone(result.rejection)
        self.assertFalse(result.action_executed)
        diag = result.diagnostics
        assert diag is not None
        self.assertEqual(diag.refusal, result.rejection)
        self.assertEqual(diag.tool_argument_chars, len("{not json"))
        self.assertEqual(diag.output_tokens, 5)
        self.assertIsNotNone(diag.duration_seconds)

    def test_diagnostics_carry_no_model_output(self) -> None:
        secret = "SENTINEL-MODEL-TEXT"
        backend = ScriptedBackend(tool_response("y", text=secret))
        backend._responses[0].tool_calls[0]["function"]["arguments"] = "{" + secret
        result = self.runtime(backend).step("look")

        blob = json.dumps(result.diagnostics.as_log_fields(), default=str)
        self.assertNotIn(secret, blob, "diagnostics must record sizes, not content")

    def test_evaluated_tokens_are_derived_not_guessed(self) -> None:
        from orbit.runtime.analysis_runtime import StepDiagnostics

        self.assertEqual(
            StepDiagnostics(prompt_tokens=1077, reused_tokens=768).evaluated_tokens, 309
        )
        self.assertIsNone(StepDiagnostics().evaluated_tokens)

    def test_every_forensic_field_is_plumbed_from_the_response(self) -> None:
        """The fields the post-mortem depends on must come from the response.

        `getattr(response, name, None)` turns a renamed `ChatResult` field into
        a silent `None` rather than an error, and fixtures that leave these
        unset cannot tell a wired field from an absent one. This pins the exact
        shape of the failure being investigated: a step that hit the token
        ceiling after a long decode.
        """
        response = ChatResult(
            content="", model="m", finish_reason="length",
            tool_calls=[{
                "id": "c", "type": "function",
                "function": {"name": ANALYSIS_TOOL_NAME, "arguments": "{BROKEN"},
            }],
            prompt_tokens=1077, completion_tokens=1024, cached_tokens=768,
            prompt_tokens_per_second=37.0, generation_tokens_per_second=11.3,
        )
        result = self.runtime(ScriptedBackend(response)).step("look")

        diag = result.diagnostics
        assert diag is not None
        self.assertEqual(diag.finish_reason, "length")
        self.assertEqual(diag.prompt_tokens, 1077)
        self.assertEqual(diag.reused_tokens, 768)
        self.assertEqual(diag.evaluated_tokens, 309)
        self.assertEqual(diag.output_tokens, 1024)
        self.assertEqual(diag.generation_tokens_per_second, 11.3)
        self.assertEqual(diag.tool_argument_chars, len("{BROKEN"))
        self.assertIsNotNone(diag.refusal)

    def test_duration_measures_only_the_model_call(self) -> None:
        """Not the sandbox, not evidence capture -- the generation itself."""
        import time as _time

        from orbit.runtime import analysis_runtime as module

        ticks = iter([100.0, 142.5])
        with unittest.mock.patch.object(
            module.time, "monotonic", side_effect=lambda: next(ticks)
        ):
            result = self.runtime(
                ScriptedBackend(prose_response("done"))
            ).step("look")

        assert result.diagnostics is not None
        self.assertEqual(result.diagnostics.duration_seconds, 42.5)

    def test_the_session_survives_a_refused_step(self) -> None:
        backend = ScriptedBackend(
            tool_response("x"), tool_response("import orbit_tools\nprint('ok')")
        )
        backend._responses[0].tool_calls[0]["function"]["arguments"] = "{not json"
        runtime = self.runtime(backend)

        refused = runtime.step("first")
        self.assertIsNotNone(refused.rejection)
        self.assertEqual(runtime.actions_executed, 0)

        recovered = runtime.step("second")
        self.assertTrue(recovered.action_executed)
        self.assertEqual(backend.calls, 2)
        json.dumps(runtime.messages, default=str)
