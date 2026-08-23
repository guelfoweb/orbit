from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.analysis_runtime import ANALYSIS_TOOL_NAME, AnalysisRuntime
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.sessions import SessionStore
from orbit.runtime.tools import TOOL_NAMES
from orbit.runtime.workflow_mode import (
    DEFAULT_WORKFLOW_MODE,
    RESUMABLE_WORKFLOW_MODES,
    WorkflowMode,
    parse_workflow_mode,
    restored_workflow_mode,
)
from orbit.terminal.analysis_mode import AnalysisModeError, open_analysis_session
from orbit.terminal.config import AppConfig
from orbit.terminal.repl import Repl

# Pinned literal. The route language gained an ANALYSIS form in the
# automatic-recognition mission; the pin is updated only alongside the
# prewarm requalification that a changed route prompt forces.
ROUTE_PROMPT_SHA256 = "d38e293a1d8fc0efb5371cff08bb5870ffc4faa6b96b889ff2af54ba2b66a38d"


class ScriptedBackend:
    """A backend that records what it was asked and never infers anything."""

    def __init__(self, tool_calls: list[dict] | None = None) -> None:
        self.calls = 0
        self.tools_seen: list[object] = []
        self.messages_seen: list[list[Message]] = []
        self._tool_calls = tool_calls or []

    def _result(self) -> ChatResult:
        return ChatResult(
            content="observing",
            model="scripted",
            finish_reason="stop",
            tool_calls=list(self._tool_calls),
            prompt_tokens=1,
            completion_tokens=1,
            cached_tokens=0,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def chat(self, messages, *, temperature, max_tokens, tools=None) -> ChatResult:
        self.calls += 1
        self.tools_seen.append(tools)
        self.messages_seen.append([dict(m) for m in messages])
        return self._result()

    def chat_stream(
        self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None
    ) -> ChatResult:
        self.calls += 1
        self.tools_seen.append(tools)
        self.messages_seen.append([dict(m) for m in messages])
        if on_delta:
            on_delta("observing")
        return self._result()

    def server_tools(self):
        return []

    def display_model_name(self):
        return "scripted"


class ModeTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="orbit-mode-test-"))
        self.addCleanup(self._cleanup)
        self.artifact = self.tmp / "sample.js"
        self.artifact.write_text("var a = 1;\n", encoding="utf-8")

    def _cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def repl(self, backend: ScriptedBackend | None = None, **kwargs) -> Repl:
        backend = backend or ScriptedBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        runtime.evidence_store = EvidenceStore(root=self.tmp / "evidence")
        built = Repl(
            runtime=runtime,
            backend=backend,
            config=AppConfig(workdir=self.tmp),
            **kwargs,
        )
        # Every test closes its workspace, so a failure cannot leak one.
        self.addCleanup(built._close_analysis)
        return built

    def run_command(self, repl: Repl, command: str) -> str:
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            repl._handle_command(command)
        return out.getvalue() + err.getvalue()

    def run_prompt(self, repl: Repl, prompt: str) -> str:
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            repl._ask(prompt)
        return out.getvalue() + err.getvalue()


class DefaultModeTest(ModeTestBase):
    def test_session_starts_in_chat(self) -> None:
        self.assertIs(self.repl().workflow_mode, WorkflowMode.CHAT)
        self.assertIs(DEFAULT_WORKFLOW_MODE, WorkflowMode.CHAT)

    def test_chat_input_reaches_chat_runtime_not_analysis(self) -> None:
        repl = self.repl()
        seen: list[str] = []
        repl.runtime.ask_auto = lambda prompt, **kw: seen.append(prompt) or ChatResult(
            "ok", "scripted", "stop", [], 1, 1, 0, None, None
        )
        repl.tools_mode = "on"
        self.run_prompt(repl, "what is a packer?")

        self.assertEqual(seen, ["what is a packer?"])
        self.assertIsNone(repl.analysis)


class EnterAnalysisTest(ModeTestBase):
    def test_valid_path_enters_analysis_without_a_model_call(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)

        output = self.run_command(repl, f"/analysis {self.artifact}")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertIsInstance(repl.analysis, AnalysisRuntime)
        self.assertEqual(backend.calls, 0)
        self.assertIn("mode: ANALYSIS", output)
        self.assertIn(self.artifact.name, output)

    def test_relative_path_resolves_against_workdir(self) -> None:
        repl = self.repl()
        self.run_command(repl, "/analysis sample.js")
        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)

    def test_source_is_snapshotted_by_content(self) -> None:
        import hashlib

        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        digest = hashlib.sha256(self.artifact.read_bytes()).hexdigest()

        self.assertEqual(repl.analysis.source.sha256, digest)
        # Editing the original afterwards must not change what is analysed.
        self.artifact.write_text("var a = 2;\n", encoding="utf-8")
        self.assertEqual(repl.analysis.source.sha256, digest)

    def test_missing_path_fails_safely_and_stays_in_chat(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)

        output = self.run_command(repl, "/analysis /nonexistent/zzz.js")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)
        self.assertEqual(backend.calls, 0)
        self.assertIn("no such artifact", output)

    def test_directory_is_refused(self) -> None:
        repl = self.repl()
        output = self.run_command(repl, f"/analysis {self.tmp}")
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIn("not a file", output)

    def test_empty_argument_is_refused(self) -> None:
        repl = self.repl()
        output = self.run_command(repl, "/analysis")
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIn("usage", output)

    def test_bad_path_while_in_analysis_keeps_current_session(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        first = repl.analysis

        self.run_command(repl, "/analysis /nonexistent/zzz.js")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertIs(repl.analysis, first)

    def test_failed_open_leaves_no_workspace_behind(self) -> None:
        before = set(Path(tempfile.gettempdir()).glob("orbit-analysis-session-*"))
        with self.assertRaises(AnalysisModeError):
            open_analysis_session(
                "/nonexistent/zzz.js",
                backend=ScriptedBackend(),
                workdir=self.tmp,
                evidence_store_factory=lambda root: EvidenceStore(root=root / "evidence"),
            )
        after = set(Path(tempfile.gettempdir()).glob("orbit-analysis-session-*"))
        self.assertEqual(before, after)


class AnalysisDispatchTest(ModeTestBase):
    def test_analyst_line_goes_to_analysis_runtime_not_chat(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        chat_calls: list[str] = []
        repl.runtime.ask_auto = lambda prompt, **kw: chat_calls.append(prompt)
        repl.runtime.ask_chat = lambda prompt, **kw: chat_calls.append(prompt)

        self.run_prompt(repl, "decode the payload")

        self.assertEqual(chat_calls, [])
        self.assertEqual(repl.analysis.analyst_turns, 1)

    def test_continue_is_one_analysis_step_with_verbatim_text(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")

        self.run_prompt(repl, "continue")

        user_messages = [m for m in repl.analysis.messages if m["role"] == "user"]
        self.assertEqual(user_messages[-1]["content"], "continue")
        self.assertEqual(repl.analysis.analyst_turns, 1)

    def test_steering_stays_in_analysis(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")

        for text in ("look at the strings", "continue", "now dump the header"):
            self.run_prompt(repl, text)
            self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)

        self.assertEqual(repl.analysis.analyst_turns, 3)

    def test_one_model_call_per_analyst_line(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")

        for expected in (1, 2, 3):
            self.run_prompt(repl, "continue")
            self.assertEqual(backend.calls, expected)

    def test_analysis_history_is_append_only(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "first")
        prefix = [dict(m) for m in repl.analysis.messages]

        self.run_prompt(repl, "second")

        self.assertEqual(repl.analysis.messages[: len(prefix)], prefix)


class ToolIsolationTest(ModeTestBase):
    def test_analysis_offers_only_execute_analysis(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")

        self.run_prompt(repl, "continue")

        offered = backend.tools_seen[-1]
        names = [tool["function"]["name"] for tool in offered]
        self.assertEqual(names, [ANALYSIS_TOOL_NAME])
        for chat_tool in TOOL_NAMES:
            self.assertNotIn(chat_tool, names)

    def test_chat_tool_registry_never_contains_execute_analysis(self) -> None:
        from orbit.runtime.tools import tool_definitions, tool_names

        self.assertNotIn(ANALYSIS_TOOL_NAME, tool_names())
        names = [tool["function"]["name"] for tool in tool_definitions()]
        self.assertNotIn(ANALYSIS_TOOL_NAME, names)

    def test_execute_analysis_is_not_executable_through_the_chat_tool_path(self) -> None:
        from orbit.runtime.tools import execute_tool

        result = execute_tool(ANALYSIS_TOOL_NAME, "{}", workdir=self.tmp)
        self.assertIn("unknown tool", result.content)

    def test_chat_surface_modules_do_not_reference_execute_analysis(self) -> None:
        for name in ("chat.py", "tools.py", "tool_loop.py"):
            text = (SRC / "orbit" / "runtime" / name).read_text(encoding="utf-8")
            self.assertNotIn(ANALYSIS_TOOL_NAME, text, name)

    def test_analysis_runtime_does_not_reference_chat_tools(self) -> None:
        text = (SRC / "orbit" / "runtime" / "analysis_runtime.py").read_text(encoding="utf-8")
        for chat_tool in TOOL_NAMES:
            self.assertNotIn(chat_tool, text)


class ReturnToChatTest(ModeTestBase):
    def test_chat_command_returns_to_chat_without_a_model_call(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        calls_before = backend.calls

        output = self.run_command(repl, "/chat")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertEqual(backend.calls, calls_before)
        self.assertIn("mode: CHAT", output)

    def test_chat_keeps_the_analysis_session_alive(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        workspace_root = repl.analysis.workspace.root

        self.run_command(repl, "/chat")

        self.assertIsNotNone(repl.analysis)
        self.assertTrue(workspace_root.exists())

    def test_after_chat_input_goes_back_to_chat_runtime(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_command(repl, "/chat")
        seen: list[str] = []
        repl.runtime.ask_auto = lambda prompt, **kw: seen.append(prompt) or ChatResult(
            "ok", "scripted", "stop", [], 1, 1, 0, None, None
        )
        repl.tools_mode = "on"

        self.run_prompt(repl, "explain XOR")

        self.assertEqual(seen, ["explain XOR"])
        self.assertEqual(repl.analysis.analyst_turns, 0)

    def test_no_flapping_a_conversational_line_does_not_switch_mode(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")

        for text in ("thanks", "what do you think?", "hello"):
            self.run_prompt(repl, text)
            self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)

    def test_chat_twice_is_idempotent(self) -> None:
        repl = self.repl()
        self.run_command(repl, "/chat")
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.run_command(repl, "/chat")
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)

    def test_chat_rejects_arguments(self) -> None:
        repl = self.repl()
        output = self.run_command(repl, "/chat now")
        self.assertIn("usage", output)


class LifecycleTest(ModeTestBase):
    def test_reset_closes_the_workspace_and_returns_to_chat(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        workspace_root = repl.analysis.workspace.root

        self.run_command(repl, "/reset")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)
        self.assertFalse(workspace_root.exists())

    def test_exit_closes_the_workspace(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        workspace_root = repl.analysis.workspace.root

        with contextlib.redirect_stdout(io.StringIO()):
            repl._finish_interactive_session(0)

        self.assertFalse(workspace_root.exists())

    def test_starting_a_second_analysis_releases_the_first(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        first_root = repl.analysis.workspace.root
        other = self.tmp / "other.js"
        other.write_text("var b = 2;\n", encoding="utf-8")

        self.run_command(repl, f"/analysis {other}")

        self.assertFalse(first_root.exists())
        self.assertTrue(repl.analysis.workspace.root.exists())
        self.assertNotEqual(repl.analysis.workspace.root, first_root)

    def test_close_analysis_is_idempotent(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        repl._close_analysis()
        repl._close_analysis()
        self.assertIsNone(repl.analysis)

    def test_unexpected_error_releases_the_workspace_before_propagating(self) -> None:
        class Exploding(ScriptedBackend):
            def chat_stream(self, *args, **kwargs):
                raise RuntimeError("backend exploded")

        backend = Exploding()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        workspace_root = repl.analysis.workspace.root

        with self.assertRaises(RuntimeError):
            with contextlib.redirect_stdout(io.StringIO()):
                repl._ask("go")

        # Crashing still ends the process as it does in CHAT, but a temporary
        # workspace must not be what survives it.
        self.assertFalse(workspace_root.exists())
        self.assertIsNone(repl.analysis)

    def test_recoverable_backend_error_keeps_the_session(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        class Failing(ScriptedBackend):
            def chat_stream(self, *args, **kwargs):
                raise LlamaServerError("upstream down")

        backend = Failing()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        workspace_root = repl.analysis.workspace.root

        self.run_prompt(repl, "go")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertIsNotNone(repl.analysis)
        self.assertTrue(workspace_root.exists())

    def test_no_workspace_leaks_across_a_full_cycle(self) -> None:
        pattern = "orbit-analysis-session-*"
        before = set(Path(tempfile.gettempdir()).glob(pattern))
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "continue")
        self.run_command(repl, "/chat")
        with contextlib.redirect_stdout(io.StringIO()):
            repl._finish_interactive_session(0)
        after = set(Path(tempfile.gettempdir()).glob(pattern))
        self.assertEqual(before, after)


class PersistenceTest(ModeTestBase):
    def store(self) -> SessionStore:
        return SessionStore(self.tmp / "session.json")

    def test_chat_mode_is_saved_and_restored(self) -> None:
        store = self.store()
        store.save(
            messages=[{"role": "user", "content": "hi"}],
            workdir=self.tmp,
            model="m",
            base_url="u",
            workflow_mode="CHAT",
        )
        mode, warning = restored_workflow_mode(store.load_workflow_mode())
        self.assertIs(mode, WorkflowMode.CHAT)
        self.assertIsNone(warning)

    def test_analysis_is_not_falsely_resumable(self) -> None:
        store = self.store()
        store.save(
            messages=[{"role": "user", "content": "hi"}],
            workdir=self.tmp,
            model="m",
            base_url="u",
            workflow_mode="ANALYSIS",
        )
        mode, warning = restored_workflow_mode(store.load_workflow_mode())

        self.assertIs(mode, WorkflowMode.CHAT)
        self.assertIsNotNone(warning)
        self.assertIn("cannot be resumed", warning)
        self.assertNotIn(WorkflowMode.ANALYSIS, RESUMABLE_WORKFLOW_MODES)

    def test_unknown_mode_falls_back_to_chat_with_a_warning(self) -> None:
        mode, warning = restored_workflow_mode("SUPERVISOR")
        self.assertIs(mode, WorkflowMode.CHAT)
        self.assertIn("unknown workflow mode", warning)

    def test_missing_mode_is_chat_without_a_warning(self) -> None:
        store = self.store()
        store.save(messages=[{"role": "user", "content": "hi"}], workdir=self.tmp, model="m", base_url="u")
        self.assertIsNone(store.load_workflow_mode())
        mode, warning = restored_workflow_mode(store.load_workflow_mode())
        self.assertIs(mode, WorkflowMode.CHAT)
        self.assertIsNone(warning)

    def test_non_string_mode_is_rejected(self) -> None:
        for value in (17, [], {}, True):
            self.assertIsNone(parse_workflow_mode(value))
            mode, warning = restored_workflow_mode(value)
            self.assertIs(mode, WorkflowMode.CHAT)
            self.assertIsNotNone(warning)

    def test_omitting_mode_keeps_the_payload_shape_unchanged(self) -> None:
        store = self.store()
        store.save(messages=[{"role": "user", "content": "hi"}], workdir=self.tmp, model="m", base_url="u")
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        self.assertNotIn("workflow_mode", payload)
        self.assertEqual(payload["version"], 1)

    def test_saved_mode_round_trips_through_the_repl(self) -> None:
        repl = self.repl(session=self.store())
        self.run_command(repl, f"/analysis {self.artifact}")
        repl._save_session()
        payload = json.loads(repl.session.path.read_text(encoding="utf-8"))

        self.assertEqual(payload["workflow_mode"], "ANALYSIS")
        # Recorded, but deliberately not resumable.
        mode, warning = restored_workflow_mode(payload["workflow_mode"])
        self.assertIs(mode, WorkflowMode.CHAT)
        self.assertIsNotNone(warning)

    def test_corrupt_session_file_yields_no_mode(self) -> None:
        store = self.store()
        store.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(store.load_workflow_mode())

    def test_repl_honours_an_injected_restored_mode(self) -> None:
        repl = self.repl(workflow_mode=WorkflowMode.CHAT)
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)


class ModeInvariantTest(ModeTestBase):
    def test_analysis_mode_without_a_runtime_refuses_instead_of_answering_as_chat(self) -> None:
        repl = self.repl()
        seen: list[str] = []
        repl.runtime.ask_auto = lambda prompt, **kw: seen.append(prompt)
        repl.workflow_mode = WorkflowMode.ANALYSIS
        repl.analysis = None

        output = self.run_prompt(repl, "decode it")

        self.assertEqual(seen, [])
        self.assertIn("analysis session unavailable", output)
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)


class EvidenceSeparationTest(ModeTestBase):
    """CHAT reads its evidence store to make decisions, so ANALYSIS keeps its own."""

    def _analysis_record(self, repl: Repl):
        self.run_prompt(repl, "look")
        records = list(repl.analysis.evidence_store.records.values())
        self.assertTrue(records, "the step should have recorded evidence")
        return records

    def repl_with_action(self):
        backend = ScriptedBackend(
            tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": ANALYSIS_TOOL_NAME, "arguments": json.dumps({"code": "print(1)"})},
                }
            ]
        )
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        return repl, backend

    def test_analysis_uses_a_store_of_its_own(self) -> None:
        repl = self.repl()
        chat_store = repl.runtime.evidence_store
        self.run_command(repl, f"/analysis {self.artifact}")

        self.assertIsNot(repl.analysis.evidence_store, chat_store)

    def test_analysis_evidence_lives_inside_the_analysis_workspace(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        store_root = repl.analysis.evidence_store.root

        self.assertTrue(str(store_root).startswith(str(repl.analysis.workspace.root)))

    def test_analysis_never_writes_into_the_chat_store(self) -> None:
        repl, _ = self.repl_with_action()
        chat_store = repl.runtime.evidence_store
        self.run_prompt(repl, "look")

        self.assertEqual(chat_store.records, {})

    def test_chat_route_window_is_unaffected_by_an_analysis_action(self) -> None:
        repl, _ = self.repl_with_action()
        repl.runtime.messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "my name is G"},
            {"role": "assistant", "content": "Hi G."},
            {"role": "user", "content": "what is my name?"},
        ]
        before = repl.runtime._route_messages()

        self.run_prompt(repl, "look")
        self.run_command(repl, "/chat")
        after = repl.runtime._route_messages()

        # The prior conversation must still be in the route window, and the
        # analysis tool must not be advertised inside it.
        self.assertEqual(len(after), len(before))
        self.assertEqual(after, before)
        for message in after:
            self.assertNotIn(ANALYSIS_TOOL_NAME, str(message["content"]))

    def test_chat_citation_gate_cannot_authorize_analysis_evidence(self) -> None:
        repl, _ = self.repl_with_action()
        self.run_prompt(repl, "look")
        self.run_command(repl, "/chat")

        # Both runtimes mint "turn_N" from independent counters, so the only
        # safe separation is that CHAT's store never holds analysis records.
        repl.runtime.current_user_turn_id = "turn_1"
        authorized = [
            record.tool_name
            for record in (repl.runtime.evidence_store.records.values())
            if record.user_turn_id == "turn_1"
        ]
        self.assertNotIn(ANALYSIS_TOOL_NAME, authorized)

    def test_chat_reset_does_not_clear_a_live_analysis_store(self) -> None:
        repl, _ = self.repl_with_action()
        self.run_prompt(repl, "look")
        analysis_store = repl.analysis.evidence_store
        count = len(analysis_store.records)

        repl.runtime.reset()

        self.assertEqual(len(analysis_store.records), count)


class FailedStepRollbackTest(ModeTestBase):
    """A step that produced nothing must leave no trace in the record."""

    def _failing_repl(self, exc: BaseException):
        class Failing(ScriptedBackend):
            def chat_stream(self, *args, **kwargs):
                raise exc

        backend = Failing()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        return repl

    def test_cancelled_step_leaves_no_dangling_user_message(self) -> None:
        repl = self._failing_repl(KeyboardInterrupt())
        before = [dict(m) for m in repl.analysis.messages]

        self.run_prompt(repl, "decode it")

        self.assertEqual(repl.analysis.messages, before)
        self.assertEqual(repl.analysis.analyst_turns, 0)

    def test_failed_step_leaves_no_dangling_user_message(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        repl = self._failing_repl(LlamaServerError("upstream down"))
        before = [dict(m) for m in repl.analysis.messages]

        self.run_prompt(repl, "decode it")

        self.assertEqual(repl.analysis.messages, before)
        self.assertEqual(repl.analysis.analyst_turns, 0)

    def test_no_run_of_unanswered_user_turns_after_retries(self) -> None:
        repl = self._failing_repl(KeyboardInterrupt())
        for _ in range(3):
            self.run_prompt(repl, "decode it")

        roles = [m["role"] for m in repl.analysis.messages]
        for first, second in zip(roles, roles[1:]):
            self.assertFalse(first == "user" and second == "user", roles)

    def test_analyst_turn_ids_only_count_steps_that_ran(self) -> None:
        from orbit.backend.llama_server import LlamaServerError

        class Flaky(ScriptedBackend):
            def __init__(self) -> None:
                super().__init__()
                self.fail_next = True

            def chat_stream(self, *args, **kwargs):
                if self.fail_next:
                    self.fail_next = False
                    raise LlamaServerError("transient")
                return super().chat_stream(*args, **kwargs)

        repl = self.repl(Flaky())
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "first attempt")
        self.run_prompt(repl, "second attempt")

        self.assertEqual(repl.analysis.analyst_turns, 1)


class SandboxRefusalTest(ModeTestBase):
    """A sandbox refusal is an answer to the analyst, not the end of a session."""

    def _repl_raising(self, exc: BaseException):
        backend = ScriptedBackend(
            tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": ANALYSIS_TOOL_NAME, "arguments": json.dumps({"code": "pass"})},
                }
            ]
        )
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        import orbit.runtime.analysis_runtime as module

        def boom(**kwargs):
            raise exc

        self.enterContext(mock.patch.object(module, "execute_analysis", boom))
        return repl

    def test_unsafe_scratch_entry_is_a_refusal_not_a_dead_session(self) -> None:
        repl = self._repl_raising(RuntimeError("unsafe scratch entry"))

        output = self.run_prompt(repl, "run it")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertIsNotNone(repl.analysis)
        self.assertTrue(repl.analysis.workspace.root.exists())
        self.assertIn("action refused", output)

    def test_os_error_from_the_sandbox_is_a_refusal(self) -> None:
        repl = self._repl_raising(OSError("scratch vanished"))

        self.run_prompt(repl, "run it")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertIsNotNone(repl.analysis)

    def test_a_refusal_still_returns_control_with_one_model_call(self) -> None:
        repl = self._repl_raising(RuntimeError("read-only input changed during analysis"))
        result = repl.analysis.step("run it")

        self.assertEqual(result.model_calls, 1)
        self.assertFalse(result.action_executed)
        self.assertTrue(result.control_returned)
        self.assertIn("read-only input changed", result.rejection)

    def test_mode_and_runtime_never_desynchronize_on_a_refusal(self) -> None:
        repl = self._repl_raising(RuntimeError("unsafe scratch entry"))
        self.run_prompt(repl, "run it")

        # The bug this guards: an escaping RuntimeError closed the session
        # while the REPL still believed it was in ANALYSIS.
        self.assertEqual(
            repl.workflow_mode is WorkflowMode.ANALYSIS,
            repl.analysis is not None,
        )


class EvidenceOrderingTest(ModeTestBase):
    def test_an_executed_action_leaves_evidence_and_tool_result_together(self) -> None:
        backend = ScriptedBackend(
            tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": ANALYSIS_TOOL_NAME, "arguments": json.dumps({"code": "print(1)"})},
                }
            ]
        )
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "run it")

        # Two records (bounded + raw) and exactly one tool message for them.
        records = list(repl.analysis.evidence_store.records.values())
        tool_messages = [m for m in repl.analysis.messages if m["role"] == "tool"]
        self.assertEqual(len(records), 2)
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(repl.analysis.messages[-1]["role"], "tool")


class SessionsClearLifecycleTest(ModeTestBase):
    def test_sessions_clear_closes_the_analysis_session(self) -> None:
        import unittest.mock as mock

        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        workspace_root = repl.analysis.workspace.root

        with mock.patch("orbit.terminal.repl._confirm_clear_sessions", return_value=True):
            self.run_command(repl, "/sessions clear")

        self.assertFalse(workspace_root.exists())
        self.assertIsNone(repl.analysis)
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)


class BackendAgnosticTest(ModeTestBase):
    def test_backend_never_receives_a_mode_argument(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "continue")

        for messages in backend.messages_seen:
            for message in messages:
                self.assertNotIn("workflow_mode", message)
                self.assertNotIn("mode", message)

    def test_mode_is_not_written_into_the_analysis_prompt(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "continue")

        system = repl.analysis.messages[0]["content"]
        for token in ("ANALYSIS mode", "workflow_mode", "WorkflowMode", "/analysis", "/chat"):
            self.assertNotIn(token, system)

    def test_backend_module_does_not_branch_on_workflow_mode(self) -> None:
        for path in (SRC / "orbit" / "backend").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("WorkflowMode", text, path.name)
            self.assertNotIn("workflow_mode", text, path.name)


class StrictPrefixTest(ModeTestBase):
    def test_analysis_stable_prefix_is_free_of_volatile_state(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        system = repl.analysis.messages[0]["content"]

        for volatile in (
            str(self.tmp),
            str(self.artifact),
            repl.analysis.source.sha256,
            str(repl.analysis.workspace.root),
            "turn_",
        ):
            self.assertNotIn(volatile, system)

    def test_step_messages_extend_the_previous_step_exactly(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "one")
        first = [dict(m) for m in repl.analysis.messages]
        self.run_prompt(repl, "two")
        second = repl.analysis.messages

        self.assertGreater(len(second), len(first))
        self.assertEqual(second[: len(first)], first)

    def test_chat_and_analysis_prompts_are_disjoint(self) -> None:
        from orbit.runtime.analysis_runtime import ANALYSIS_SYSTEM_PROMPT
        from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT

        self.assertNotIn(ANALYSIS_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT)
        self.assertNotIn(ANALYSIS_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT)
        self.assertNotIn(ROUTE_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT)


class ChatNonRegressionTest(ModeTestBase):
    def test_route_prompt_matches_the_qualified_pin(self) -> None:
        import hashlib

        from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT

        # Pinned so a future mode change cannot quietly edit route language,
        # which is what the deferred recognition mission has to qualify.
        self.assertEqual(
            hashlib.sha256(ROUTE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            ROUTE_PROMPT_SHA256,
        )

    def test_chat_five_tool_surface_is_unchanged(self) -> None:
        self.assertEqual(
            TOOL_NAMES,
            (
                "exec_shell_full_command",
                "fetch_url",
                "list_directory",
                "system_info",
                "write_artifact",
            ),
        )

    def test_chat_turn_still_uses_ask_auto_with_chat_tools(self) -> None:
        repl = self.repl()
        repl.tools_mode = "on"
        seen: dict[str, object] = {}

        def fake_ask_auto(prompt, **kwargs):
            seen["prompt"] = prompt
            seen["allowed"] = kwargs.get("allowed_tool_names")
            return ChatResult("ok", "scripted", "stop", [], 1, 1, 0, None, None)

        repl.runtime.ask_auto = fake_ask_auto
        self.run_prompt(repl, "list the files")

        self.assertEqual(seen["prompt"], "list the files")
        self.assertNotIn(ANALYSIS_TOOL_NAME, seen["allowed"] or ())


if __name__ == "__main__":
    unittest.main()


class PromptMarkerTest(ModeTestBase):
    """The marker names the runtime that owns the next line.

    It is display only. These pin both halves of that: the text follows
    `workflow_mode`, and it never becomes something the model can read.
    """

    def test_chat_is_the_default_marker(self) -> None:
        self.assertEqual(self.repl()._prompt_label(), "chat")

    def test_explicit_analysis_switches_the_marker(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertEqual(repl._prompt_label(), "analysis")

    def test_slash_chat_restores_the_marker(self) -> None:
        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_command(repl, "/chat")
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertEqual(repl._prompt_label(), "chat")

    def test_the_marker_follows_the_mode_rather_than_being_stored(self) -> None:
        # No second variable to drift: setting the authoritative state is
        # enough to change what is displayed.
        repl = self.repl()
        repl.workflow_mode = WorkflowMode.ANALYSIS
        self.assertEqual(repl._prompt_label(), "analysis")
        repl.workflow_mode = WorkflowMode.CHAT
        self.assertEqual(repl._prompt_label(), "chat")

    def test_switching_mode_costs_no_model_call(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        before = backend.calls
        self.run_command(repl, f"/analysis {self.artifact}")
        self.assertEqual(repl._prompt_label(), "analysis")
        self.run_command(repl, "/chat")
        self.assertEqual(repl._prompt_label(), "chat")
        self.assertEqual(backend.calls, before)

    def test_the_marker_never_reaches_the_backend(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_command(repl, "/chat")
        self.run_prompt(repl, "hello")
        sent = json.dumps(backend.messages_seen, default=str)
        for marker in ("chat> ", "analysis> "):
            self.assertNotIn(marker, sent)
        for message in repl.runtime.messages:
            self.assertNotIn("analysis> ", str(message.get("content") or ""))

    def test_rendered_prompt_carries_the_label(self) -> None:
        from orbit.terminal.repl_input import input_prompt

        with mock.patch("orbit.terminal.repl_input.sys.stdout") as stdout:
            stdout.isatty.return_value = False
            self.assertEqual(input_prompt("chat"), "chat> ")
            self.assertEqual(input_prompt("analysis"), "analysis> ")
        with mock.patch("orbit.terminal.repl_input.sys.stdout") as stdout:
            stdout.isatty.return_value = True
            self.assertIn("analysis> ", input_prompt("analysis"))


class ActionCauseRenderingTest(unittest.TestCase):
    """A failed action has to say what went wrong, not only where to look.

    The cause is read from what the sandbox already reported. Nothing is
    inferred, no model is asked, and the full text stays in evidence.
    """

    def _result(self, **overrides):
        from orbit.runtime.analysis_sandbox import AnalysisResult

        base = dict(
            status="error",
            code_sha256="c" * 64,
            input_sha256="i" * 64,
            stdout="",
            stderr="",
            exit_status=1,
            duration_seconds=0.4,
        )
        base.update(overrides)
        return AnalysisResult(**base)

    def test_the_real_recorded_failure_renders_its_exception(self) -> None:
        # Verbatim stderr from the observed run whose summary showed only ids.
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/program/main.py", line 10, in <module>\n'
            '    data = orbit_tools.read_file("/workspace/input/samples/x.js")\n'
            "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
            '  File "orbit_tools.py", line 29, in read_file\n'
            '  File "orbit_tools.py", line 19, in _safe_path\n'
            "PermissionError: path is outside the analyst workspace\n"
        )
        from orbit.terminal.analysis_mode import _action_cause

        self.assertEqual(
            _action_cause(self._result(stderr=stderr)),
            "PermissionError: path is outside the analyst workspace",
        )

    def test_a_timeout_says_so(self) -> None:
        from orbit.terminal.analysis_mode import _action_cause

        cause = _action_cause(self._result(status="timeout", duration_seconds=30.0))
        self.assertEqual(cause, "sandbox timeout after 30.0s")

    def test_a_bare_nonzero_exit_is_reported(self) -> None:
        from orbit.terminal.analysis_mode import _action_cause

        self.assertEqual(
            _action_cause(self._result(stderr="", exit_status=3)),
            "Python exited with status 3",
        )

    def test_a_successful_action_has_no_cause(self) -> None:
        from orbit.terminal.analysis_mode import _action_cause

        self.assertIsNone(_action_cause(self._result(status="ok", exit_status=0)))

    def test_enormous_stderr_stays_bounded(self) -> None:
        from orbit.terminal.analysis_mode import MAX_ACTION_CAUSE_CHARS, _action_cause

        noise = "\n".join(f"  File \"/host/private/{i}.py\", line {i}" for i in range(5000))
        stderr = f"Traceback (most recent call last):\n{noise}\nValueError: {'x' * 20000}\n"
        cause = _action_cause(self._result(stderr=stderr))
        assert cause is not None
        self.assertLessEqual(len(cause), MAX_ACTION_CAUSE_CHARS)
        self.assertTrue(cause.startswith("ValueError: "))
        self.assertNotIn("/host/private", cause)

    def test_host_frames_are_not_echoed(self) -> None:
        from orbit.terminal.analysis_mode import _action_cause

        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/home/someone/secret/main.py", line 1, in <module>\n'
            "RuntimeError: boom\n"
        )
        cause = _action_cause(self._result(stderr=stderr))
        self.assertEqual(cause, "RuntimeError: boom")
        self.assertNotIn("/home/someone", cause or "")

    def test_summary_keeps_ids_and_adds_the_cause(self) -> None:
        from orbit.runtime.analysis_runtime import AnalysisStepResult
        from orbit.runtime.evidence import EvidenceRecord
        from orbit.terminal.analysis_mode import format_analysis_step

        action = self._result(stderr="PermissionError: nope\n")
        record = EvidenceRecord(
            evidence_id="ev_aaa_bbb",
            tool_name="execute_analysis",
            kind="fetch",
            raw_ref="evidence:ev_aaa_bbb",
            raw_sha256="d" * 64,
            raw_chars=10,
            raw_lines=1,
            status="ok",
            metadata={},
            route_card=None,
            final_card=None,
        )
        step = AnalysisStepResult(
            model_calls=1,
            action_attempted=True,
            action_executed=True,
            assistant_text="",
            result=action,
            evidence=record,
            raw_output_evidence_id="ev_ccc_ddd",
        )
        rendered = format_analysis_step(step)
        self.assertIn("action: error", rendered)
        self.assertIn("PermissionError: nope", rendered)
        # The ids moved onto their own lines once a preview can sit above
        # them; they are for copying, and a long joined tail is hard to select.
        self.assertIn("ev_aaa_bbb", rendered)
        self.assertIn("ev_ccc_ddd", rendered)

    def test_a_successful_summary_is_unchanged(self) -> None:
        from orbit.runtime.analysis_runtime import AnalysisStepResult
        from orbit.terminal.analysis_mode import format_analysis_step

        step = AnalysisStepResult(
            model_calls=1,
            action_attempted=True,
            action_executed=True,
            assistant_text="",
            result=self._result(status="ok", exit_status=0),
        )
        self.assertEqual(format_analysis_step(step), "action: ok")

    def test_a_multiline_exception_message_leaks_no_host_path(self) -> None:
        # The last non-frame line of a multi-line message is a continuation,
        # not the failure; reporting it printed a host path nobody asked for.
        from orbit.terminal.analysis_mode import _action_cause

        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/program/main.py", line 3, in <module>\n'
            "ValueError: line one\n"
            "of the message continues /home/someone/secret/path\n"
        )
        cause = _action_cause(self._result(stderr=stderr))
        self.assertEqual(cause, "ValueError: line one")
        self.assertNotIn("/home/someone", cause or "")

    def test_stderr_without_an_exception_falls_back_to_the_exit_status(self) -> None:
        from orbit.terminal.analysis_mode import _action_cause

        stderr = "some warning: not an exception\ntrailing /home/someone/noise\n"
        cause = _action_cause(self._result(stderr=stderr, exit_status=2))
        self.assertEqual(cause, "Python exited with status 2")
        self.assertNotIn("/home/someone", cause or "")

    def test_a_bounded_result_does_not_add_an_exit_status(self) -> None:
        # The bound is already named in the summary; the exit status would
        # describe the symptom rather than the reason.
        from orbit.terminal.analysis_mode import _action_cause

        cause = _action_cause(
            self._result(status="bounded", bound_exceeded="scratch_bytes", stderr="")
        )
        self.assertIsNone(cause)

    def test_whitespace_only_stderr_yields_the_exit_status(self) -> None:
        from orbit.terminal.analysis_mode import _action_cause

        self.assertEqual(
            _action_cause(self._result(stderr="   \n\n  \n", exit_status=1)),
            "Python exited with status 1",
        )


class PromptEchoLabelTest(ModeTestBase):
    """The echo must use the marker the line was displayed with.

    A command can change the mode between reading a line and erasing it, so
    re-deriving the label afterwards would erase the wrong number of rows.
    """

    def test_the_erase_width_counts_the_label(self) -> None:
        from orbit.terminal.repl_input import visual_row_count

        prompt = "x" * 74
        # 74 chars fits one 80-column row bare, but not behind "analysis> ".
        self.assertEqual(visual_row_count(f"> {prompt}", columns=80), 1)
        self.assertEqual(visual_row_count(f"analysis> {prompt}", columns=80), 2)

    def test_clear_input_echo_uses_the_supplied_label(self) -> None:
        from orbit.terminal import repl_input

        seen: list[str] = []
        with (
            mock.patch.object(repl_input.sys.stdout, "isatty", return_value=True),
            mock.patch.object(repl_input, "get_terminal_size", return_value=os.terminal_size((80, 20))),
            mock.patch("builtins.print", side_effect=lambda *a, **k: seen.append(str(a[0]))),
        ):
            repl_input.clear_input_echo("x" * 74, "analysis")
        # Two rows because the marker pushed the line over the column count.
        self.assertIn("2F", seen[0])

    def test_the_echo_label_is_the_displayed_one_not_the_post_command_one(self) -> None:
        repl = self.repl()
        displayed = repl._prompt_label()
        self.assertEqual(displayed, "chat")
        self.run_command(repl, f"/analysis {self.artifact}")
        # The mode changed, so re-deriving now would give the wrong marker for
        # the line that was typed while CHAT was displayed.
        self.assertEqual(repl._prompt_label(), "analysis")
        self.assertNotEqual(displayed, repl._prompt_label())


class AnalysisProgressRenderingTest(ModeTestBase):
    """The analyst sees the step working, and never sees raw tool JSON."""

    def test_the_step_receives_a_progress_and_delta_seam(self) -> None:
        """The Repl must hand the runtime somewhere to report to.

        Without this the terminal stays silent for the whole call, which is
        the behaviour that made a long step look like a hang.
        """
        import inspect

        from orbit.terminal.repl import Repl

        source = inspect.getsource(Repl._ask_analysis)
        self.assertIn("on_progress=", source)
        self.assertIn("on_delta=", source)
        self.assertIn("StreamRenderer(", source)

    def test_the_renderer_is_finished_on_every_exit_path(self) -> None:
        # A renderer left running keeps a timer thread and a stale status line.
        import inspect

        from orbit.terminal.repl import Repl

        source = inspect.getsource(Repl._ask_analysis)
        self.assertEqual(source.count("renderer.finish("), 4)

    def test_analysis_never_renders_reasoning(self) -> None:
        """Checked on the constructed renderer, not on the source text.

        The literal `thinking=False` also appears in a comment here, so a
        substring check passed even with the real argument flipped.
        """
        from orbit.terminal.streaming import StreamRenderer

        captured: list[dict] = []
        original = StreamRenderer.__init__

        def spy(self, *args, **kwargs):
            captured.append(dict(kwargs))
            original(self, *args, **kwargs)

        repl = self.repl()
        self.run_command(repl, f"/analysis {self.artifact}")
        with mock.patch.object(StreamRenderer, "__init__", spy):
            self.run_prompt(repl, "look at it")

        self.assertTrue(captured, "the analysis step must build a renderer")
        self.assertFalse(
            captured[-1].get("thinking", True),
            "ANALYSIS must never render reasoning",
        )

    def test_diagnostics_render_sizes_not_content(self) -> None:
        from orbit.runtime.analysis_runtime import StepDiagnostics
        from orbit.terminal.analysis_mode import format_step_diagnostics

        line = format_step_diagnostics(
            StepDiagnostics(
                prompt_tokens=1077, output_tokens=1024, reused_tokens=768,
                finish_reason="length", tool_argument_chars=9312,
                refusal="tool arguments are not valid JSON",
            )
        )
        self.assertIn("1077 in", line)
        self.assertIn("309 eval", line)
        self.assertIn("768 cache", line)
        self.assertIn("1024 out", line)
        self.assertIn("length", line)
        self.assertIn("tool args 9312 chars", line)

    def test_a_step_without_diagnostics_renders_nothing_extra(self) -> None:
        from orbit.terminal.analysis_mode import format_step_diagnostics

        self.assertEqual(format_step_diagnostics(None), "")

    def test_non_tty_progress_is_not_interactive(self) -> None:
        """Non-interactive output must stay plain: no timer, no escapes."""
        from orbit.backend.base import StreamProgress
        from orbit.terminal.streaming import StreamRenderer

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            renderer = StreamRenderer(thinking=False, interactive=False)
            renderer.set_activity("analysis")
            renderer.start()
            renderer.progress(StreamProgress(
                phase="prefill", current=1, total=2, percent=50,
                evaluated_current=1, evaluated_total=2,
            ))
            renderer.finish()
        rendered = out.getvalue()
        self.assertNotIn("\x1b[", rendered, "no ANSI in non-interactive output")


class EvidencePreviewTest(unittest.TestCase):
    """After an action the analyst must be able to see what it produced.

    An evidence id says where to look, not what happened. Without a preview
    the only way to choose the next step is to open the store by hand.
    """

    def _result(self, **overrides):
        from orbit.runtime.analysis_sandbox import AnalysisResult

        base = dict(
            status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
            stdout="", stderr="", exit_status=0, duration_seconds=1.0,
        )
        base.update(overrides)
        return AnalysisResult(**base)

    def _step(self, action, **overrides):
        from orbit.runtime.analysis_runtime import AnalysisStepResult
        from orbit.runtime.evidence import EvidenceRecord

        record = EvidenceRecord(
            evidence_id="ev_aaa_bbb", tool_name="execute_analysis", kind="fetch",
            raw_ref="evidence:ev_aaa_bbb", raw_sha256="d" * 64, raw_chars=10,
            raw_lines=1, status="ok", metadata={}, route_card=None, final_card=None,
        )
        base = dict(
            model_calls=1, action_attempted=True, action_executed=True,
            assistant_text="", result=action, evidence=record,
            raw_output_evidence_id="ev_ccc_ddd",
        )
        base.update(overrides)
        return AnalysisStepResult(**base)

    def test_the_preview_caps_are_pinned_and_independent(self) -> None:
        """Absolute values, not self-referential comparisons.

        Asserting a bound against the constant it bounds passes at any value.
        These are also deliberately separate from the model-facing budget:
        one is about prompt cost, the other about a readable terminal.
        """
        from orbit.runtime.analysis_runtime import MAX_EVIDENCE_CHARS
        from orbit.terminal.analysis_mode import MAX_PREVIEW_CHARS, MAX_PREVIEW_LINES

        self.assertEqual(MAX_PREVIEW_CHARS, 1200)
        self.assertEqual(MAX_PREVIEW_LINES, 24)
        self.assertLess(MAX_PREVIEW_CHARS, MAX_EVIDENCE_CHARS)

    def test_useful_stdout_is_visible(self) -> None:
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(stdout="strings found: 12\nWScript.Shell\n"))
        )
        self.assertIn("result:", rendered)
        self.assertIn("strings found: 12", rendered)
        self.assertIn("WScript.Shell", rendered)

    def test_multiline_output_is_bounded_by_lines(self) -> None:
        from orbit.terminal.analysis_mode import MAX_PREVIEW_LINES, format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(stdout="\n".join(f"line {i}" for i in range(500))))
        )
        shown = [ln for ln in rendered.splitlines() if ln.startswith("  line ")]
        self.assertLessEqual(len(shown), MAX_PREVIEW_LINES)
        self.assertIn("preview truncated", rendered)

    def test_a_single_enormous_line_is_bounded_by_chars(self) -> None:
        from orbit.terminal.analysis_mode import MAX_PREVIEW_CHARS, format_analysis_step

        rendered = format_analysis_step(self._step(self._result(stdout="Z" * 100000)))
        body = "\n".join(
            ln for ln in rendered.splitlines() if ln.startswith("  ") and "Z" in ln
        )
        self.assertLessEqual(len(body), MAX_PREVIEW_CHARS + 40)
        self.assertIn("preview truncated", rendered)

    def test_truncation_keeps_the_raw_evidence_id(self) -> None:
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(self._step(self._result(stdout="Q" * 90000)))
        self.assertIn("preview truncated", rendered)
        self.assertIn("ev_ccc_ddd", rendered)
        self.assertIn("full output in evidence", rendered)

    def test_artifacts_show_virtual_path_size_and_sha(self) -> None:
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(
                stdout="done\n",
                artifacts=(DerivedArtifact(name="stage2.js", size_bytes=4096, sha256="a" * 64),),
            ))
        )
        self.assertIn("artifacts:", rendered)
        self.assertIn("/workspace/work/stage2.js", rendered)
        self.assertIn("4.0 KiB", rendered)
        # Shortened for the terminal; the stored digest keeps all 64 chars.
        self.assertIn("sha256 " + "a" * 12 + "\u2026", rendered)

    def test_the_artifact_digest_is_shortened_for_reading(self) -> None:
        """Twelve hex characters and an ellipsis, not the whole digest.

        The full 64 characters pushed the line past 80 columns, so it wrapped
        and became harder to read than no digest at all. Twelve is enough to
        tell two artifacts apart and to grep the full value out of evidence.
        """
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import SHORT_SHA_CHARS, format_analysis_step

        digest = "ec8ccda0a41f" + "3b" * 26
        rendered = format_analysis_step(
            self._step(self._result(
                stdout="done\n",
                artifacts=(DerivedArtifact(name="stage1.txt", size_bytes=1008, sha256=digest),),
            ))
        )
        self.assertEqual(SHORT_SHA_CHARS, 12)
        self.assertIn("sha256 ec8ccda0a41f\u2026", rendered)
        self.assertNotIn(digest, rendered, "the full digest must not be on screen")

    def test_the_stored_digest_is_never_truncated(self) -> None:
        """Presentation only: what is recorded keeps all 64 characters."""
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import format_analysis_step

        digest = "ec8ccda0a41f" + "3b" * 26
        artifact = DerivedArtifact(name="stage1.txt", size_bytes=1008, sha256=digest)
        step = self._step(self._result(stdout="done\n", artifacts=(artifact,)))
        format_analysis_step(step)

        self.assertEqual(artifact.sha256, digest)
        self.assertEqual(len(artifact.sha256), 64)
        self.assertEqual(step.result.artifacts[0].sha256, digest)

    def test_two_artifacts_stay_distinguishable(self) -> None:
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import format_analysis_step

        first = "aaaaaaaaaaaa" + "0" * 52
        second = "bbbbbbbbbbbb" + "0" * 52
        rendered = format_analysis_step(
            self._step(self._result(
                stdout="done\n",
                artifacts=(
                    DerivedArtifact(name="one.bin", size_bytes=1, sha256=first),
                    DerivedArtifact(name="two.bin", size_bytes=2, sha256=second),
                ),
            ))
        )
        self.assertIn("aaaaaaaaaaaa\u2026", rendered)
        self.assertIn("bbbbbbbbbbbb\u2026", rendered)

    def test_a_digest_carrying_escapes_cannot_act_on_the_terminal(self) -> None:
        """The digest is model-adjacent data and gets the same treatment.

        Sanitising the artifact name but not its digest would leave the same
        hole one field to the right.
        """
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(
                stdout="x\n",
                artifacts=(DerivedArtifact(
                    name="a.txt", size_bytes=1, sha256="\x1b[31m" + "a" * 60),),
            ))
        )
        self.assertNotIn("\x1b", rendered)

    def test_a_malformed_digest_is_shown_as_it_is(self) -> None:
        """Never trimmed into something that merely looks well-formed."""
        from orbit.terminal.analysis_mode import _short_sha

        for value in ("", "abc", "not-a-hex-digest", "zz" * 32):
            with self.subTest(value=value[:12]):
                self.assertEqual(_short_sha(value), value)

    def test_stderr_is_still_not_shown_for_a_successful_action(self) -> None:
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(stdout="ok\n", stderr="warning: trailing data\n"))
        )
        self.assertNotIn("warning: trailing data", rendered)

    def test_the_artifact_handles_line_is_not_duplicated(self) -> None:
        """The preview already lists them with size and digest."""
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(
                self._result(
                    stdout="done\n",
                    artifacts=(DerivedArtifact(name="a.txt", size_bytes=1, sha256="a" * 64),),
                ),
                artifact_handles=("/workspace/work/a.txt",),
            )
        )
        self.assertEqual(rendered.count("/workspace/work/a.txt"), 1)

    def test_the_handles_line_still_renders_without_a_preview(self) -> None:
        """Delta B suppresses the duplicate, it does not remove the fallback.

        When the preview did not list artifacts there is nothing to duplicate,
        so the handles line must still appear -- otherwise suppressing it
        would lose the information instead of de-duplicating it.
        """
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(
                self._result(stdout="done\n"),  # no artifacts -> no preview list
                artifact_handles=("/workspace/work/kept.txt",),
            )
        )
        self.assertIn("artifacts: /workspace/work/kept.txt", rendered)

    def test_the_shortener_runs_before_the_sanitiser(self) -> None:
        """Order matters: sanitising first would expand, then trim mid-escape."""
        import inspect

        from orbit.terminal import analysis_mode

        source = inspect.getsource(analysis_mode._preview_block)
        self.assertIn("_sanitize(_short_sha(artifact.sha256))", source)

    def test_a_digest_at_the_boundary_is_not_marked_short(self) -> None:
        from orbit.terminal.analysis_mode import SHORT_SHA_CHARS, _short_sha

        exact = "a" * SHORT_SHA_CHARS
        self.assertEqual(_short_sha(exact), exact, "nothing was dropped")
        self.assertEqual(_short_sha(exact + "b"), exact + "\u2026")

    def test_host_paths_are_never_rendered(self) -> None:
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(
                stdout="done\n",
                artifacts=(DerivedArtifact(name="out.bin", size_bytes=10, sha256="b" * 64),),
            ))
        )
        for host in ("/tmp/orbit-analysis-session-", "/home/", "/tmp/"):
            self.assertNotIn(host, rendered)

    def test_binary_output_is_metadata_only(self) -> None:
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(stdout="\x00\x01\x02\x03\xff" * 400))
        )
        self.assertIn("non-text output", rendered)
        self.assertNotIn("\x00", rendered)
        self.assertIn("ev_ccc_ddd", rendered)

    def test_ansi_in_model_output_cannot_act_on_the_terminal(self) -> None:
        """A crafted action must not be able to forge Orbit's own output.

        Everything previewed here is model-authored. Cursor movement would let
        it overwrite the lines Orbit already printed -- including a fake
        `action: ok` above itself.
        """
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(self._result(stdout="\x1b[1A\x1b[2Kaction: ok | evidence ev_FAKE\n"))
        )
        self.assertNotIn("\x1b", rendered)
        self.assertIn("\\x1b[1A", rendered, "shown literally instead")

    def test_an_artifact_name_cannot_break_the_line_shape(self) -> None:
        from orbit.runtime.analysis_sandbox import DerivedArtifact
        from orbit.terminal.analysis_mode import format_analysis_step

        for name in ("\x1b[31mred", "two\nlines", "\r\ncarriage"):
            with self.subTest(name=name):
                rendered = format_analysis_step(
                    self._step(self._result(
                        stdout="x\n",
                        artifacts=(DerivedArtifact(name=name, size_bytes=1, sha256="a" * 64),),
                    ))
                )
                artifact_lines = [
                    ln for ln in rendered.splitlines() if ln.strip().startswith("- /workspace")
                ]
                self.assertEqual(len(artifact_lines), 1, "one artifact, one line")
                self.assertNotIn("\x1b", rendered)

    def test_tabs_in_output_survive(self) -> None:
        """Ordinary program output must not be mangled by the sanitiser."""
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(self._step(self._result(stdout="a\tb\n")))
        self.assertIn("a\tb", rendered)

    def test_an_action_with_no_output_stays_on_one_line(self) -> None:
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(self._step(self._result()))
        self.assertEqual(len(rendered.splitlines()), 1)
        self.assertIn("action: ok", rendered)
        self.assertIn("ev_aaa_bbb", rendered)

    def test_error_cause_rendering_is_unchanged(self) -> None:
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(
            self._step(
                self._result(status="error", exit_status=1,
                             stderr="Traceback...\nPermissionError: nope\n"),
                action_executed=True,
            )
        )
        self.assertIn("action: error", rendered)
        self.assertIn("PermissionError: nope", rendered)
        # The cause line already carries it; the preview must not repeat it.
        self.assertEqual(rendered.count("PermissionError: nope"), 1)

    def test_both_evidence_ids_survive(self) -> None:
        from orbit.terminal.analysis_mode import format_analysis_step

        rendered = format_analysis_step(self._step(self._result(stdout="hello\n")))
        self.assertIn("evidence: ev_aaa_bbb", rendered)
        self.assertIn("raw: ev_ccc_ddd", rendered)

    def test_the_preview_is_terminal_only(self) -> None:
        """Rendering must not mutate the step or reach a backend.

        Checked behaviourally rather than by scanning the source: the rendered
        text is a pure function of the result, so calling it twice on the same
        input gives the same answer and changes nothing.
        """
        from orbit.terminal.analysis_mode import format_analysis_step

        step = self._step(self._result(stdout="hello\nworld\n"))
        before = json.dumps(step.result.__dict__, default=str)
        first = format_analysis_step(step)
        second = format_analysis_step(step)

        self.assertEqual(first, second)
        self.assertEqual(json.dumps(step.result.__dict__, default=str), before)


class AmberAnalysisPromptTest(ModeTestBase):
    def test_analysis_is_amber_and_chat_is_not(self) -> None:
        from orbit.terminal import repl_input
        from orbit.terminal.theme import CYAN, YELLOW

        with mock.patch.object(repl_input.sys, "stdout") as stdout:
            stdout.isatty.return_value = True
            chat = repl_input.input_prompt("chat")
            analysis = repl_input.input_prompt("analysis")

        self.assertIn(YELLOW, analysis)
        self.assertNotIn(CYAN, analysis)
        self.assertIn(CYAN, chat)
        self.assertNotIn(YELLOW, chat)

    def test_analysis_is_not_red(self) -> None:
        from orbit.terminal import repl_input
        from orbit.terminal.theme import RED

        with mock.patch.object(repl_input.sys, "stdout") as stdout:
            stdout.isatty.return_value = True
            self.assertNotIn(RED, repl_input.input_prompt("analysis"))

    def test_no_ansi_in_non_tty(self) -> None:
        from orbit.terminal import repl_input

        with mock.patch.object(repl_input.sys, "stdout") as stdout:
            stdout.isatty.return_value = False
            for label in ("chat", "analysis"):
                self.assertEqual(repl_input.input_prompt(label), f"{label}> ")


class WorkdirReminderTest(ModeTestBase):
    def test_it_is_announced_once_and_not_repeated(self) -> None:
        repl = self.repl()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._announce_workdir()
            repl._announce_workdir()
            repl._announce_workdir()
        self.assertEqual(out.getvalue().count("workdir:"), 1)

    def test_it_is_announced_again_when_the_workdir_changes(self) -> None:
        import dataclasses

        repl = self.repl()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._announce_workdir()
            repl.config = dataclasses.replace(repl.config, workdir=self.tmp / "other")
            repl._announce_workdir()
        self.assertEqual(out.getvalue().count("workdir:"), 2)

    def test_run_announces_the_workdir_before_the_first_prompt(self) -> None:
        """The headline behaviour, driven through run() rather than the helper.

        Calling `_announce_workdir` directly proves the helper works; it does
        not prove anything calls it.
        """
        repl = self.repl()
        out = io.StringIO()
        with (
            mock.patch("orbit.terminal.repl.read_prompt_input", side_effect=EOFError),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            repl.run()
        self.assertIn("workdir:", out.getvalue())

    def test_the_workdir_is_not_repeated_on_every_turn(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        out = io.StringIO()
        prompts = iter(["hello", "again"])

        def fake_read(**_kwargs):
            try:
                return next(prompts)
            except StopIteration:
                raise EOFError from None

        with (
            mock.patch("orbit.terminal.repl.read_prompt_input", side_effect=fake_read),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            repl.run()
        self.assertEqual(out.getvalue().count("workdir:"), 1)

    def test_home_itself_renders_as_a_bare_tilde(self) -> None:
        from pathlib import Path

        from orbit.terminal.repl import _abbreviate_home

        self.assertEqual(_abbreviate_home(Path.home()), "~")

    def test_home_is_abbreviated(self) -> None:
        from pathlib import Path

        from orbit.terminal.repl import _abbreviate_home

        rendered = _abbreviate_home(Path.home() / "LAB" / "orbit" / "workdir")
        self.assertTrue(rendered.startswith("~/"))
        self.assertNotIn(str(Path.home()), rendered)

    def test_a_path_outside_home_is_left_alone(self) -> None:
        from pathlib import Path

        from orbit.terminal.repl import _abbreviate_home

        self.assertEqual(_abbreviate_home(Path("/opt/data")), "/opt/data")

    def test_the_analyst_preview_never_enters_model_history(self) -> None:
        """The bounded model observation is unchanged by this rendering."""
        from orbit.runtime.analysis_runtime import ANALYSIS_TOOL_NAME

        backend = ScriptedBackend([{
            "id": "c", "type": "function",
            "function": {"name": ANALYSIS_TOOL_NAME,
                         "arguments": json.dumps({"code": "print('x' * 50)"})},
        }])
        repl = self.repl(backend)
        self.run_command(repl, f"/analysis {self.artifact}")
        self.run_prompt(repl, "look")

        history = json.dumps(repl.analysis.messages, default=str)
        for marker in ("result:", "artifacts:", "preview truncated", "evidence: ev_"):
            self.assertNotIn(marker, history)

    def test_the_workdir_never_reaches_the_backend(self) -> None:
        backend = ScriptedBackend()
        repl = self.repl(backend)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._announce_workdir()
        self.run_prompt(repl, "hello")
        sent = json.dumps(backend.messages_seen, default=str)
        self.assertNotIn("workdir:", sent)
