"""Malformed model tool output must never enter the analysis history.

The history an analysis session keeps is append-only and is re-rendered whole
on every later step, so one `tool_calls` entry the template cannot parse ends
the session: not at the step that produced it, but at the next one, when the
prompt fails to render. A tool call the model got wrong therefore has to be
judged before the turn is committed, and reported as prose instead.

The scripted backend here fails the test if a step makes a second model call,
because the tempting fix -- asking the model to repair its own output -- is
exactly the human boundary this runtime exists to hold.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime.analysis_runtime import (
    ANALYSIS_TOOL_NAME,
    AnalysisRuntime,
    AnalysisWorkspace,
    acquire_analysis_source,
)
from orbit.runtime.evidence import EvidenceStore

# Verbatim from the preserved reproduction: the model was cut off mid-string by
# a small output budget, so the arguments JSON has no closing quote or brace.
# Rendering a history containing this raises from the C++ template parser:
#   RuntimeError: Failed to parse tool call arguments as JSON: ... parse_error.101
PRESERVED_TRUNCATED_ARGS = (
    '{"code": "import orbit_tools\\ndata = orbit_tools.read_file(\\"/workspace/input\\")\\n'
    'print(\\"length:\\", len(data))\\nprint(\\"repr:\\", repr(data))\\n'
)


def call(name: str = ANALYSIS_TOOL_NAME, arguments: str = '{"code": "print(1)"}', call_id: str = "call_0"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class SingleCallBackend:
    """Scripted, and hostile to a hidden second model call within one step."""

    def __init__(self, responses: list[tuple[str, list[dict]]]) -> None:
        self._responses = responses
        self.calls = 0
        self.calls_this_step = 0

    def begin_step(self) -> None:
        self.calls_this_step = 0

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        self.calls += 1
        self.calls_this_step += 1
        if self.calls_this_step > 1:
            raise AssertionError("a step made a second model call")
        content, tool_calls = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if on_delta:
            on_delta(content)
        return ChatResult(content, "scripted", "stop", list(tool_calls), 1, 1, 0, None, None)

    def chat(self, *args, **kwargs):
        return self.chat_stream(*args, **kwargs)


class MalformedToolOutputTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="orbit-malformed-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.artifact = self.tmp / "note.txt"
        self.artifact.write_text("orbit canary fixture\n", encoding="utf-8")

    def runtime(self, responses):
        backend = SingleCallBackend(responses)
        workspace = AnalysisWorkspace.create()
        source = acquire_analysis_source(self.artifact, workspace.source_root)
        built = AnalysisRuntime(
            backend=backend,
            source=source,
            evidence_store=EvidenceStore(root=self.tmp / f"ev{id(workspace)}"),
            workspace=workspace,
        )
        self.addCleanup(built.close)
        return built, backend

    def step(self, runtime, backend, text):
        backend.begin_step()
        return runtime.step(text)

    def assert_history_renderable(self, runtime) -> None:
        """Every tool_calls entry must parse, as the real template requires."""
        for message in runtime.messages:
            for entry in message.get("tool_calls") or []:
                raw = entry.get("function", {}).get("arguments")
                json.loads(raw)  # raises exactly where the bridge would

    def assert_history_serializable(self, runtime) -> None:
        json.dumps(runtime.messages)


class PreservedReproductionTest(MalformedToolOutputTestBase):
    """Case 1: the exact truncated JSON from the preserved reproduction."""

    def responses(self):
        return [("reading the file", [call(arguments=PRESERVED_TRUNCATED_ARGS)]), ("done", [])]

    def test_the_preserved_truncated_call_is_refused(self) -> None:
        runtime, backend = self.runtime(self.responses())

        result = self.step(runtime, backend, "read the file and report its length")

        self.assertEqual(result.rejection, "tool arguments are not valid JSON")
        self.assertTrue(result.action_attempted)
        self.assertFalse(result.action_executed)
        self.assertTrue(result.control_returned)

    def test_history_stays_renderable_after_the_truncated_call(self) -> None:
        runtime, backend = self.runtime(self.responses())

        self.step(runtime, backend, "read it")

        self.assert_history_renderable(runtime)
        self.assert_history_serializable(runtime)

    def test_no_poisoned_assistant_turn_is_appended(self) -> None:
        runtime, backend = self.runtime(self.responses())

        self.step(runtime, backend, "read it")

        for message in runtime.messages:
            self.assertNotIn("tool_calls", message, "the refused call must not be committed")

    def test_the_analyst_can_continue_afterwards(self) -> None:
        runtime, backend = self.runtime(self.responses())
        self.step(runtime, backend, "read it")

        result = self.step(runtime, backend, "continue")

        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.control_returned)
        self.assert_history_renderable(runtime)

    def test_zero_tools_executed(self) -> None:
        runtime, backend = self.runtime(self.responses())
        self.step(runtime, backend, "read it")
        self.assertEqual(runtime.actions_executed, 0)

    def test_only_one_model_call_for_the_refused_step(self) -> None:
        runtime, backend = self.runtime(self.responses())
        self.step(runtime, backend, "read it")
        self.assertEqual(backend.calls, 1)


class MalformedShapeTest(MalformedToolOutputTestBase):
    """Cases 2-4: every structural defect refuses without committing."""

    CASES = {
        "truncated_json": (PRESERVED_TRUNCATED_ARGS, "tool arguments are not valid JSON"),
        "not_json_at_all": ("not json", "tool arguments are not valid JSON"),
        "empty_arguments": ("", "tool arguments are not valid JSON"),
        "json_but_not_an_object": ('"a string"', "tool arguments must supply a 'code' string"),
        "missing_code_key": ('{"source": "print(1)"}', "tool arguments must supply a 'code' string"),
        "code_not_a_string": ('{"code": 17}', "tool arguments must supply a 'code' string"),
    }

    def test_each_malformed_shape_refuses_and_leaves_history_clean(self) -> None:
        for name, (arguments, expected) in self.CASES.items():
            with self.subTest(case=name):
                runtime, backend = self.runtime([("trying", [call(arguments=arguments)])])

                result = self.step(runtime, backend, "go")

                self.assertEqual(result.rejection, expected)
                self.assertFalse(result.action_executed)
                self.assertEqual(runtime.actions_executed, 0)
                self.assert_history_renderable(runtime)
                for message in runtime.messages:
                    self.assertNotIn("tool_calls", message)

    def test_unknown_tool_is_refused_without_committing(self) -> None:
        runtime, backend = self.runtime([("trying", [call(name="rm_rf")])])

        result = self.step(runtime, backend, "go")

        self.assertIn("unsupported tool", result.rejection or "")
        self.assertEqual(runtime.actions_executed, 0)
        for message in runtime.messages:
            self.assertNotIn("tool_calls", message)

    def test_multiple_tool_calls_are_refused_without_committing(self) -> None:
        runtime, backend = self.runtime(
            [("trying", [call(call_id="call_0"), call(call_id="call_1")])]
        )

        result = self.step(runtime, backend, "do two things")

        self.assertIn("at most one action per step", result.rejection or "")
        self.assertEqual(runtime.actions_executed, 0)
        for message in runtime.messages:
            self.assertNotIn("tool_calls", message)


class UnrenderableButParseableTest(MalformedToolOutputTestBase):
    """Valid JSON is not enough: the turn must also survive being rendered.

    The bridge serializes the history with `ensure_ascii=False` and encodes it
    as UTF-8, so a lone surrogate parses fine and then breaks every later step
    exactly like the truncated call did.
    """

    def test_a_lone_surrogate_in_code_is_refused(self) -> None:
        runtime, backend = self.runtime(
            [("trying", [call(arguments=json.dumps({"code": "print(1)"})[:-2] + '\\ud800"}')])]
        )
        result = self.step(runtime, backend, "go")
        self.assertIsNotNone(result.rejection)

    def test_a_surrogate_anywhere_in_the_call_is_refused(self) -> None:
        for field in ("code", "id", "name"):
            with self.subTest(field=field):
                entry = call()
                if field == "code":
                    entry["function"]["arguments"] = '{"code": "\ud800"}'
                elif field == "id":
                    entry["id"] = "\ud800"
                else:
                    entry["function"]["name"] = "\ud800"
                runtime, backend = self.runtime([("trying", [entry])])

                result = self.step(runtime, backend, "go")

                self.assertIsNotNone(result.rejection, f"{field} surrogate must be refused")
                self.assertEqual(runtime.actions_executed, 0)

    def test_committed_history_always_survives_the_bridge_encoding(self) -> None:
        entry = call()
        entry["function"]["arguments"] = '{"code": "\ud800"}'
        runtime, backend = self.runtime([("trying", [entry]), ("ok", [])])
        self.step(runtime, backend, "go")

        # Exactly what bindings.render does before the C++ call.
        json.dumps(runtime.messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def test_a_non_object_tool_call_is_refused_not_raised(self) -> None:
        runtime, backend = self.runtime([("trying", ["not an object"])])

        result = self.step(runtime, backend, "go")

        self.assertIn("not an object", result.rejection or "")
        self.assertEqual(runtime.actions_executed, 0)
        for message in runtime.messages:
            self.assertNotIn("tool_calls", message)

    def test_a_call_without_a_function_object_is_refused(self) -> None:
        for entry in ({"id": "c0", "type": "function"}, {"id": "c0", "function": None}):
            with self.subTest(entry=str(entry)):
                runtime, backend = self.runtime([("trying", [entry])])
                result = self.step(runtime, backend, "go")
                self.assertIsNotNone(result.rejection)
                self.assertEqual(runtime.actions_executed, 0)


class TemplateShapeRequirementsTest(MalformedToolOutputTestBase):
    """The template rejects more shapes than json.loads does."""

    def test_a_non_function_call_type_is_refused(self) -> None:
        entry = call()
        entry["type"] = "custom"
        runtime, backend = self.runtime([("trying", [entry])])

        result = self.step(runtime, backend, "go")

        self.assertIn("unsupported tool call type", result.rejection or "")
        self.assertEqual(runtime.actions_executed, 0)
        for message in runtime.messages:
            self.assertNotIn("tool_calls", message)

    def test_a_missing_type_is_still_accepted(self) -> None:
        # The template only objects to a *wrong* type; omitting it is fine and
        # must not become a spurious refusal.
        entry = call()
        del entry["type"]
        runtime, backend = self.runtime([("trying", [entry])])

        result = self.step(runtime, backend, "go")

        self.assertIsNone(result.rejection)

    def test_a_missing_function_object_is_refused_by_name(self) -> None:
        runtime, backend = self.runtime([("trying", [{"id": "c0", "type": "function"}])])

        result = self.step(runtime, backend, "go")

        self.assertIn("no function object", result.rejection or "")

    def test_unencodable_assistant_content_never_reaches_history(self) -> None:
        # Practically unreachable -- decoding uses errors="replace" -- but the
        # consequence would be the same unrenderable history.
        runtime, backend = self.runtime([("bad \ud800 text", [])])

        self.step(runtime, backend, "go")

        json.dumps(runtime.messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class ValidOutputParityTest(MalformedToolOutputTestBase):
    """Cases 5-6: nothing changes for output that is actually well formed."""

    def test_a_valid_single_call_is_still_committed_with_its_tool_calls(self) -> None:
        runtime, backend = self.runtime([("running", [call()])])

        result = self.step(runtime, backend, "go")

        self.assertIsNone(result.rejection)
        self.assertTrue(result.action_attempted)
        assistant = [m for m in runtime.messages if m["role"] == "assistant"][-1]
        self.assertIn("tool_calls", assistant, "a valid call must still reach the history")
        self.assert_history_renderable(runtime)

    def test_plain_prose_is_committed_unchanged(self) -> None:
        runtime, backend = self.runtime([("just thinking out loud", [])])

        result = self.step(runtime, backend, "what do you see?")

        self.assertIsNone(result.rejection)
        self.assertFalse(result.action_attempted)
        assistant = [m for m in runtime.messages if m["role"] == "assistant"][-1]
        self.assertEqual(assistant["content"], "just thinking out loud")
        self.assertNotIn("tool_calls", assistant)

    def test_the_refusal_is_visible_to_the_analyst_in_history(self) -> None:
        runtime, backend = self.runtime([("trying", [call(arguments="not json")])])

        self.step(runtime, backend, "go")

        assistant = [m for m in runtime.messages if m["role"] == "assistant"][-1]
        self.assertIn("refused", assistant["content"])
        self.assertIn("not valid JSON", assistant["content"])

    def test_the_refused_arguments_are_not_echoed_into_history(self) -> None:
        marker = '{"code": "SECRETMARKER_UNTERMINATED'
        runtime, backend = self.runtime([("trying", [call(arguments=marker)])])

        self.step(runtime, backend, "go")

        self.assertNotIn("SECRETMARKER_UNTERMINATED", json.dumps(runtime.messages))


class AppendOnlyRecoveryTest(MalformedToolOutputTestBase):
    """Cases 9, 11: recovery must not rewrite what came before."""

    def test_history_remains_append_only_across_a_refusal(self) -> None:
        runtime, backend = self.runtime(
            [("ok", [call()]), ("trying", [call(arguments="not json")]), ("fine", [])]
        )
        self.step(runtime, backend, "first")
        prefix = [dict(m) for m in runtime.messages]

        self.step(runtime, backend, "second")
        self.assertEqual(runtime.messages[: len(prefix)], prefix)

        self.step(runtime, backend, "third")
        self.assertEqual(runtime.messages[: len(prefix)], prefix)

    def test_a_refusal_does_not_advance_the_action_count(self) -> None:
        runtime, backend = self.runtime(
            [("ok", [call()]), ("trying", [call(arguments="not json")])]
        )
        self.step(runtime, backend, "first")
        after_valid = runtime.actions_executed

        self.step(runtime, backend, "second")

        self.assertEqual(runtime.actions_executed, after_valid)

    def test_several_consecutive_refusals_keep_the_session_usable(self) -> None:
        runtime, backend = self.runtime([("trying", [call(arguments=PRESERVED_TRUNCATED_ARGS)])])

        for _ in range(3):
            result = self.step(runtime, backend, "try again")
            self.assertIsNotNone(result.rejection)

        self.assert_history_renderable(runtime)
        self.assertEqual(backend.calls, 3, "one model call per analyst step")

    def test_analyst_turn_count_matches_the_steps_taken(self) -> None:
        runtime, backend = self.runtime([("trying", [call(arguments="not json")])])
        for _ in range(2):
            self.step(runtime, backend, "go")
        self.assertEqual(runtime.analyst_turns, 2)


class NoHiddenRepairCallTest(MalformedToolOutputTestBase):
    """Case 10: refusing must never trigger a repair or finalization call."""

    def test_a_refused_step_makes_exactly_one_model_call(self) -> None:
        runtime, backend = self.runtime([("trying", [call(arguments="not json")])])

        result = self.step(runtime, backend, "go")

        self.assertEqual(result.model_calls, 1)
        self.assertEqual(backend.calls_this_step, 1)

    def test_the_backend_would_fail_the_test_on_a_second_call(self) -> None:
        # Guards the guard: the harness must actually detect a repair call.
        runtime, backend = self.runtime([("trying", [call(arguments="not json")])])
        backend.begin_step()
        backend.chat_stream([], temperature=0.0, max_tokens=1)

        with self.assertRaises(AssertionError):
            backend.chat_stream([], temperature=0.0, max_tokens=1)


class InternalFailuresStillPropagateTest(MalformedToolOutputTestBase):
    """The fix must not become a broad exception swallow."""

    def test_a_backend_failure_is_not_disguised_as_a_rejection(self) -> None:
        class Failing(SingleCallBackend):
            def chat_stream(self, *args, **kwargs):
                raise RuntimeError("backend exploded")

        runtime, backend = self.runtime([("x", [])])
        runtime.backend = Failing([])

        with self.assertRaises(RuntimeError):
            runtime.step("go")

    def test_step_does_not_wrap_the_model_call_in_a_try(self) -> None:
        source = (SRC / "orbit" / "runtime" / "analysis_runtime.py").read_text(encoding="utf-8")
        start = source.index("def step(self, analyst_message")
        head = source[start : source.index("self.model_calls += 1", start)]

        self.assertNotIn("try:", head, "the model call must not be broadly guarded")

    def test_structural_rejection_only_inspects_parsed_output(self) -> None:
        source = (SRC / "orbit" / "runtime" / "analysis_runtime.py").read_text(encoding="utf-8")
        start = source.index("def _structural_rejection")
        block = source[start : source.index("    def ", start + 50)]

        # Narrow, structural checks only -- no blanket `except Exception`.
        self.assertIn("except (TypeError, json.JSONDecodeError)", block)
        self.assertNotIn("except Exception", block)


class StrictPrefixAfterRefusalTest(MalformedToolOutputTestBase):
    """Case 14: KV reuse must survive a refused step."""

    def test_each_step_extends_the_previous_history_exactly(self) -> None:
        runtime, backend = self.runtime(
            [("ok", [call()]), ("trying", [call(arguments="not json")]), ("more", [])]
        )
        snapshots = []
        for text in ("first", "second", "third"):
            self.step(runtime, backend, text)
            snapshots.append([dict(m) for m in runtime.messages])

        for earlier, later in zip(snapshots, snapshots[1:]):
            self.assertEqual(later[: len(earlier)], earlier)

    def test_the_prompt_sent_after_a_refusal_extends_the_one_before_it(self) -> None:
        seen: list[list[dict]] = []

        class Recording(SingleCallBackend):
            def chat_stream(self, messages, **kwargs):
                seen.append([dict(m) for m in messages])
                return super().chat_stream(messages, **kwargs)

        runtime, backend = self.runtime([("x", [])])
        recording = Recording(
            [("trying", [call(arguments=PRESERVED_TRUNCATED_ARGS)]), ("after", [])]
        )
        runtime.backend = recording

        recording.begin_step()
        runtime.step("first")
        recording.begin_step()
        runtime.step("second")

        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1][: len(seen[0])], seen[0], "strict prefix must survive a refusal")


if __name__ == "__main__":
    unittest.main()
