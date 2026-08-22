"""Automatic CHAT -> ANALYSIS recognition, decided by the route call itself.

The distinction has to come from the model's route decision, not from the
runtime inspecting the user's words. So the route language gained one form,
`{"route":"ANALYSIS","artifact":"..."}`, and everything here checks that the
form is honoured exactly: that it carries a path (a session cannot open
without one), that it never becomes a catch-all for reads and summaries, and
that recognising it costs one route call and no classifier of its own.

The scripted backends count their calls, because the failure this design most
needs to avoid is a second model call quietly appearing per turn.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime import ChatRuntime
from orbit.runtime.command_request import (
    RouteDecision,
    ToolRoute,
    decision_tool_names,
    parse_command_decision,
)
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT
from orbit.runtime.workflow_mode import WorkflowMode
from orbit.terminal.config import AppConfig
from orbit.terminal.repl import Repl

# The qualified route prompt. Changing the route language forces the Ornith
# CHAT prewarm to be re-derived, so this pin and that requalification move
# together or not at all.
ROUTE_PROMPT_SHA256 = "d38e293a1d8fc0efb5371cff08bb5870ffc4faa6b96b889ff2af54ba2b66a38d"


class RouteLanguageTest(unittest.TestCase):
    """The parser side: one new form, nothing else disturbed."""

    def test_the_analysis_form_parses_with_its_artifact(self) -> None:
        decision = parse_command_decision('{"route":"ANALYSIS","artifact":"samples/foo.js"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.ANALYSIS)
        self.assertEqual(decision.artifact, "samples/foo.js")

    def test_analysis_without_an_artifact_is_not_a_decision(self) -> None:
        # A session is bound to one file at construction, so a route that
        # names none cannot be honoured and must not masquerade as a bare
        # route either.
        for text in ('{"route":"ANALYSIS"}', '{"route":"ANALYSIS","artifact":"   "}',
                     '{"route":"ANALYSIS","artifact":null}', '{"route":"ANALYSIS","artifact":7}'):
            with self.subTest(text=text):
                self.assertIsNone(parse_command_decision(text))

    def test_the_artifact_is_carried_verbatim(self) -> None:
        for path in ("samples/foo.js", "/abs/path/x.bin", "deeply/nested/name.dat", "a"):
            with self.subTest(path=path):
                decision = parse_command_decision(json.dumps({"route": "ANALYSIS", "artifact": path}))
                assert decision is not None
                self.assertEqual(decision.artifact, path)

    def test_analysis_offers_no_chat_tools(self) -> None:
        decision = RouteDecision(ToolRoute.ANALYSIS, (), "x.js")
        self.assertEqual(decision_tool_names(decision), ())

    def test_every_other_route_still_carries_no_artifact(self) -> None:
        for text in ('{"route":"CHAT"}', '{"command":"cat README.md"}', '{"url":"https://x"}',
                     '{"path":".","recursive":false}'):
            with self.subTest(text=text):
                decision = parse_command_decision(text)
                assert decision is not None
                self.assertIsNone(decision.artifact)


class ExistingRouteParityTest(unittest.TestCase):
    """Case: nothing that routed correctly before may route differently now."""

    CORPUS = {
        '{"route":"CHAT"}': (ToolRoute.CHAT, ()),
        '{"command":"cat README.md"}': (ToolRoute.FILESYSTEM, ("exec_shell_full_command",)),
        '{"command":"ls -la"}': (ToolRoute.FILESYSTEM, ("exec_shell_full_command",)),
        '{"command":"orbit-web-search \\"x\\""}': (ToolRoute.FILESYSTEM, ("exec_shell_full_command",)),
        '{"url":"https://example.com"}': (ToolRoute.FILESYSTEM, ("fetch_url",)),
        '{"path":".","recursive":false}': (ToolRoute.FILESYSTEM, ("list_directory",)),
        '{"include_cpu":true,"include_memory":true,"include_disks":true,"include_os":true}':
            (ToolRoute.FILESYSTEM, ("system_info",)),
        '{"tool":"write_artifact","arguments":{"path":"x","overwrite":false,"create_parents":true}}':
            (ToolRoute.FILESYSTEM, ("write_artifact",)),
    }

    def test_the_existing_corpus_routes_exactly_as_before(self) -> None:
        for text, (route, tools) in self.CORPUS.items():
            with self.subTest(text=text[:40]):
                decision = parse_command_decision(text)
                assert decision is not None, text
                self.assertEqual(decision.route, route)
                self.assertEqual(decision_tool_names(decision), tools)
                self.assertNotEqual(decision.route, ToolRoute.ANALYSIS)

    def test_a_file_read_is_never_analysis(self) -> None:
        # The failure mode this guards: ANALYSIS quietly swallowing every
        # request that mentions a file.
        for text in ('{"command":"cat README.md"}', '{"command":"head -20 config.json"}',
                     '{"command":"strings sample.bin"}'):
            with self.subTest(text=text):
                decision = parse_command_decision(text)
                assert decision is not None
                self.assertEqual(decision.route, ToolRoute.FILESYSTEM)

    def test_a_hedged_output_keeps_its_existing_filesystem_decision(self) -> None:
        """Precedence: filesystem key-shapes win over a hedged ANALYSIS route.

        The single most load-bearing decision for parity is that
        `_has_analysis_route` is checked AFTER the filesystem predicates. A
        model that hedges -- naming a route and a command in one object --
        must keep doing exactly what it did before this mission.
        """
        cases = {
            '{"route":"ANALYSIS","artifact":"x.js","command":"cat y"}':
                (ToolRoute.FILESYSTEM, ("exec_shell_full_command",)),
            '{"route":"ANALYSIS","artifact":"x.js","path":".","recursive":false}':
                (ToolRoute.FILESYSTEM, ("list_directory",)),
            '{"route":"ANALYSIS","artifact":"x.js","url":"https://example.com"}':
                (ToolRoute.FILESYSTEM, ("fetch_url",)),
        }
        for text, (route, tools) in cases.items():
            with self.subTest(text=text[:48]):
                decision = parse_command_decision(text)
                assert decision is not None
                self.assertEqual(decision.route, route, "filesystem shapes must win")
                self.assertEqual(decision_tool_names(decision), tools)
                self.assertIsNone(decision.artifact)

    def test_a_chat_route_carrying_an_artifact_key_stays_chat(self) -> None:
        decision = parse_command_decision('{"route":"CHAT","artifact":"x.js"}')

        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.CHAT)
        self.assertIsNone(decision.artifact)

    def test_malformed_route_output_is_still_no_decision(self) -> None:
        for text in ("", "not json", "{", '{"route":"NONSENSE"}', '{"artifact":"x.js"}'):
            with self.subTest(text=text):
                self.assertIsNone(parse_command_decision(text))


class RoutePromptTest(unittest.TestCase):
    def test_the_route_prompt_matches_the_qualified_pin(self) -> None:
        self.assertEqual(
            hashlib.sha256(ROUTE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            ROUTE_PROMPT_SHA256,
        )

    def test_the_prompt_teaches_the_analysis_form_and_its_boundary(self) -> None:
        self.assertIn('{"route":"ANALYSIS","artifact":', ROUTE_SYSTEM_PROMPT)
        # And says plainly what must NOT take it.
        self.assertIn("never ANALYSIS", ROUTE_SYSTEM_PROMPT)

    def test_the_prompt_still_teaches_the_existing_forms(self) -> None:
        for form in ('{"route":"CHAT"}', '{"command":"cat README.md"}', '{"url":"https://example.com"}'):
            self.assertIn(form, ROUTE_SYSTEM_PROMPT)


class CountingBackend:
    """Answers the route call, then any analysis call. Counts both."""

    def __init__(self, route_payload: str, analysis_text: str = "looking at it") -> None:
        self.route_payload = route_payload
        self.analysis_text = analysis_text
        self.route_calls = 0
        self.analysis_calls = 0

    def _is_analysis(self, tools) -> bool:
        return bool(tools) and tools[0].get("function", {}).get("name") == "execute_analysis"

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        if self._is_analysis(tools):
            self.analysis_calls += 1
            content = self.analysis_text
        else:
            self.route_calls += 1
            content = self.route_payload
        if on_delta:
            on_delta(content)
        return ChatResult(content, "scripted", "stop", [], 1, 1, 0, None, None)

    def chat(self, *args, **kwargs):
        return self.chat_stream(*args, **kwargs)

    def server_tools(self):
        return []

    def display_model_name(self):
        return "scripted"


class TransitionTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="orbit-autoroute-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.artifact = self.tmp / "foo.js"
        self.artifact.write_text("var a = 1;\n", encoding="utf-8")

    def repl(self, backend):
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        runtime.evidence_store = EvidenceStore(root=self.tmp / "evidence")
        built = Repl(runtime=runtime, backend=backend, config=AppConfig(workdir=self.tmp))
        built.tools_mode = "on"
        self.addCleanup(built._close_analysis)
        return built

    def ask(self, repl, prompt):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            repl._ask(prompt)
        return out.getvalue()

    def analysis_payload(self, artifact=None):
        return json.dumps({"route": "ANALYSIS", "artifact": str(artifact or self.artifact)})


class AutomaticTransitionTest(TransitionTestBase):
    def test_an_analysis_route_enters_analysis(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)

        self.ask(repl, "analyze foo.js as an artifact and extract indicators")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertIsNotNone(repl.analysis)

    def test_the_original_request_becomes_the_first_analyst_step(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)
        request = "extract indicators and reconstruct transformations from foo.js"

        self.ask(repl, request)

        user_turns = [m for m in repl.analysis.messages if m["role"] == "user"]
        self.assertEqual(user_turns[-1]["content"], request, "the request must reach step 1 unchanged")
        self.assertEqual(repl.analysis.analyst_turns, 1)

    def test_exactly_one_route_call_and_one_analysis_call(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)

        self.ask(repl, "analyze foo.js as an artifact")

        self.assertEqual(backend.route_calls, 1, "no second classifier call")
        self.assertEqual(backend.analysis_calls, 1, "exactly one analysis step")

    def test_no_chat_tool_runs_before_the_transition(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)

        self.ask(repl, "analyze foo.js as an artifact")

        self.assertFalse(
            any(m.get("role") == "tool" for m in repl.runtime.messages),
            "a filesystem tool must not answer first",
        )

    def test_the_route_turn_answers_nothing_and_runs_no_tool(self) -> None:
        """The runtime must stop at the route, not fall through to CHAT work.

        Deleting the early return leaves the ANALYSIS decision falling into the
        filesystem branch, which both answers the wrong question and spends the
        model call the analysis step needs.
        """
        calls: list[str] = []

        class Watching(CountingBackend):
            def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                            on_delta=None, on_progress=None):
                calls.append("analysis" if self._is_analysis(tools) else "route")
                return super().chat_stream(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    tools=tools, on_delta=on_delta, on_progress=on_progress,
                )

        backend = Watching(self.analysis_payload())
        repl = self.repl(backend)

        self.ask(repl, "analyze foo.js as an artifact")

        # Exactly the route call, then exactly the analysis step. Anything in
        # between is CHAT work that should never have happened.
        self.assertEqual(calls, ["route", "analysis"])

    def test_no_assistant_answer_is_appended_to_chat_history(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)

        self.ask(repl, "analyze foo.js as an artifact")

        # The route turn produced no answer: the reply is owed to the analysis
        # step, so CHAT history must not carry a finalized assistant answer.
        assistants = [m for m in repl.runtime.messages if m.get("role") == "assistant"]
        self.assertEqual(assistants, [], "the route turn must not answer in CHAT")

    def test_the_session_is_bound_to_the_routed_artifact(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)

        self.ask(repl, "analyze foo.js")

        self.assertEqual(repl.analysis.source.original_path, str(self.artifact))

    def test_the_human_boundary_holds_on_the_first_step(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)

        self.ask(repl, "analyze foo.js")

        self.assertEqual(repl.analysis.model_calls, 1)
        self.assertEqual(repl.analysis.actions_executed, 0)


class NoTransitionTest(TransitionTestBase):
    """Ordinary CHAT turns must be untouched."""

    def test_a_chat_route_stays_in_chat(self) -> None:
        backend = CountingBackend('{"route":"CHAT"}')
        repl = self.repl(backend)
        repl.runtime.ask_auto = lambda prompt, **kw: ChatResult("ok", "s", "stop", [], 1, 1, 0, None, None)

        self.ask(repl, "explain XOR")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)

    def test_a_filesystem_route_stays_in_chat(self) -> None:
        seen: list[str] = []
        backend = CountingBackend('{"command":"cat README.md"}')
        repl = self.repl(backend)
        repl.runtime.ask_auto = lambda prompt, **kw: (
            seen.append(prompt) or ChatResult("ok", "s", "stop", [], 1, 1, 0, None, None)
        )

        self.ask(repl, "summarize README.md")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)
        self.assertEqual(seen, ["summarize README.md"])

    def test_the_signal_is_cleared_between_turns(self) -> None:
        backend = CountingBackend('{"route":"CHAT"}')
        repl = self.repl(backend)
        repl.runtime.last_analysis_request = "stale/path.js"
        repl.runtime.ask_auto = lambda prompt, **kw: ChatResult("ok", "s", "stop", [], 1, 1, 0, None, None)

        self.ask(repl, "hello")

        # ask_auto is stubbed here, so assert the runtime clears it itself.
        repl.runtime._begin_user_turn()
        self.assertIsNone(repl.runtime.last_analysis_request)


class InvalidArtifactTest(TransitionTestBase):
    def test_a_missing_artifact_path_does_not_enter_analysis(self) -> None:
        backend = CountingBackend(self.analysis_payload(self.tmp / "nope.js"))
        repl = self.repl(backend)

        output = self.ask(repl, "analyze nope.js")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)
        self.assertIn("refusing analysis", output)

    def test_a_directory_artifact_is_refused(self) -> None:
        backend = CountingBackend(self.analysis_payload(self.tmp))
        repl = self.repl(backend)

        output = self.ask(repl, "analyze that directory")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIn("refusing analysis", output)

    def test_a_refused_transition_makes_no_second_model_call(self) -> None:
        backend = CountingBackend(self.analysis_payload(self.tmp / "nope.js"))
        repl = self.repl(backend)

        self.ask(repl, "analyze nope.js")

        self.assertEqual(backend.route_calls, 1)
        self.assertEqual(backend.analysis_calls, 0)

    def test_a_refused_transition_leaves_no_unanswered_turn(self) -> None:
        """The route turn answered nothing, so its user turn must be rewound.

        Left in place it would sit unanswered in CHAT history and be resent on
        every later turn, producing two consecutive user turns.
        """
        backend = CountingBackend(self.analysis_payload(self.tmp / "nope.js"))
        repl = self.repl(backend)

        self.ask(repl, "analyze nope.js")

        roles = [m.get("role") for m in repl.runtime.messages]
        self.assertNotIn("user", roles, "the unanswered user turn must be rewound")
        for earlier, later in zip(roles, roles[1:]):
            self.assertFalse(earlier == "user" and later == "user")

    def test_a_refused_transition_clears_the_signal(self) -> None:
        backend = CountingBackend(self.analysis_payload(self.tmp / "nope.js"))
        repl = self.repl(backend)

        self.ask(repl, "analyze nope.js")

        self.assertIsNone(repl.runtime.last_analysis_request)

    def test_chat_still_works_after_a_refused_transition(self) -> None:
        backend = CountingBackend(self.analysis_payload(self.tmp / "nope.js"))
        repl = self.repl(backend)
        self.ask(repl, "analyze nope.js")

        seen: list[str] = []
        repl.runtime.ask_auto = lambda prompt, **kw: (
            seen.append(prompt) or ChatResult("ok", "s", "stop", [], 1, 1, 0, None, None)
        )
        self.ask(repl, "what is 2+2")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertEqual(seen, ["what is 2+2"])

    def test_a_refused_transition_leaks_no_workspace(self) -> None:
        pattern = "orbit-analysis-session-*"
        before = set(Path(tempfile.gettempdir()).glob(pattern))
        backend = CountingBackend(self.analysis_payload(self.tmp / "nope.js"))
        repl = self.repl(backend)

        self.ask(repl, "analyze nope.js")

        self.assertEqual(set(Path(tempfile.gettempdir()).glob(pattern)) - before, set())


class ConfinedAcquisitionTest(TransitionTestBase):
    """A model-chosen artifact is acquired from an opened object, not a name.

    Validating a pathname and letting the session reopen it left a window in
    which the name could be repointed between the two -- reproducibly, on a
    few racing turns in a hundred. So the file is opened once under the
    workdir, following no symlink on any component, and the bytes come from
    that descriptor. Symlinks are refused outright here, even ones currently
    pointing inside: for model-chosen input, following a link is the thing
    that must not happen.
    """

    def setUp(self) -> None:
        super().setUp()
        self.outside = Path(tempfile.mkdtemp(prefix="orbit-outside-"))
        self.addCleanup(shutil.rmtree, self.outside, ignore_errors=True)
        self.secret = self.outside / "secret.txt"
        self.secret.write_text("OUTSIDE\n", encoding="utf-8")
        (self.tmp / "samples").mkdir(exist_ok=True)
        self.inside = self.tmp / "samples" / "inner.js"
        self.inside.write_text("INSIDE\n", encoding="utf-8")

    def transition(self, artifact, *, on_acquired=None):
        backend = CountingBackend(self.analysis_payload(artifact))
        repl = self.repl(backend)
        if on_acquired is not None:
            repl._analysis_acquired_hook = on_acquired
        output = self.ask(repl, "analyze it")
        return repl, backend, output

    # --- the deterministic TOCTOU proof --------------------------------
    def test_a_swap_after_acquisition_cannot_change_the_snapshot(self) -> None:
        """The primary proof: no racing, the swap is forced at the seam."""
        target = self.tmp / "art.js"
        target.write_text("ORIGINAL\n", encoding="utf-8")
        swapped: list[bool] = []

        def swap() -> None:
            target.unlink()
            target.symlink_to(self.secret)
            swapped.append(True)

        repl, _backend, _out = self.transition("art.js", on_acquired=swap)

        self.assertEqual(swapped, [True], "the swap must actually have happened")
        # No escape hatch: the swap lands after acquisition, so the session
        # must have opened. Allowing `None` here would let a future change
        # that refuses everything leave this test vacuously green.
        self.assertIsNotNone(repl.analysis, "acquisition must succeed before the swap")
        snapshot = repl.analysis.source.snapshot_path.read_bytes()
        self.assertEqual(snapshot, b"ORIGINAL\n")
        self.assertNotIn(b"OUTSIDE", snapshot)

    def test_no_workspace_ever_holds_outside_content(self) -> None:
        target = self.tmp / "art.js"
        target.write_text("ORIGINAL\n", encoding="utf-8")

        def swap() -> None:
            target.unlink()
            target.symlink_to(self.secret)

        repl, _backend, _out = self.transition("art.js", on_acquired=swap)

        self.assertIsNotNone(repl.analysis, "acquisition must succeed before the swap")
        for path in repl.analysis.workspace.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"OUTSIDE", path.read_bytes(), str(path))

    def test_an_oversize_artifact_is_refused_before_it_is_read(self) -> None:
        from orbit.runtime.confined_acquire import (
            ConfinedAcquireError,
            acquire_confined_bytes,
        )

        big = self.tmp / "big.bin"
        big.write_bytes(b"x" * 5000)

        with self.assertRaises(ConfinedAcquireError) as caught:
            acquire_confined_bytes("big.bin", workdir=self.tmp, max_bytes=1000)

        self.assertIn("too large", str(caught.exception))

    def test_an_oversize_artifact_is_rejected_without_being_read(self) -> None:
        """The `st_size` pre-check must fire before any byte is read.

        The bounded read loop would also stop an oversize file, so this
        asserts the cheaper guard specifically: no `os.read` may happen at
        all. Without it, a caller would pull the whole file into memory
        before rejecting it.
        """
        from unittest import mock

        from orbit.runtime.confined_acquire import (
            ConfinedAcquireError,
            acquire_confined_bytes,
        )

        big = self.tmp / "huge.bin"
        big.write_bytes(b"z" * 8192)

        real_read = os.read
        reads: list[int] = []

        def counting_read(fd, size):
            reads.append(size)
            return real_read(fd, size)

        with mock.patch("orbit.runtime.confined_acquire.os.read", counting_read):
            with self.assertRaises(ConfinedAcquireError) as caught:
                acquire_confined_bytes("huge.bin", workdir=self.tmp, max_bytes=100)

        self.assertIn("too large", str(caught.exception))
        self.assertEqual(reads, [], "an oversize file must be refused before reading")

    def test_a_swap_before_acquisition_is_refused(self) -> None:
        # Already a symlink when the route names it: refused outright.
        target = self.tmp / "art.js"
        target.symlink_to(self.secret)

        repl, backend, output = self.transition("art.js")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)
        self.assertEqual(backend.analysis_calls, 0)
        self.assertIn("refusing analysis", output)

    # --- accepted -------------------------------------------------------
    def test_a_relative_file_inside_the_workdir_is_accepted(self) -> None:
        repl, backend, _out = self.transition("samples/inner.js")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertEqual(repl.analysis.source.snapshot_path.read_bytes(), b"INSIDE\n")
        self.assertEqual(backend.route_calls, 1)
        self.assertEqual(backend.analysis_calls, 1)
        self.assertEqual(repl.analysis.analyst_turns, 1)

    def test_a_nested_internal_file_is_accepted(self) -> None:
        nested = self.tmp / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "deep.js").write_text("DEEP\n", encoding="utf-8")

        repl, _backend, _out = self.transition("a/b/deep.js")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertEqual(repl.analysis.source.snapshot_path.read_bytes(), b"DEEP\n")

    def test_an_absolute_internal_path_is_accepted(self) -> None:
        repl, _backend, _out = self.transition(self.inside)

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertEqual(repl.analysis.source.snapshot_path.read_bytes(), b"INSIDE\n")

    def test_the_identity_is_the_acquired_bytes(self) -> None:
        import hashlib

        repl, _backend, _out = self.transition("samples/inner.js")

        self.assertEqual(
            repl.analysis.source.sha256, hashlib.sha256(b"INSIDE\n").hexdigest()
        )
        self.assertEqual(repl.analysis.source.size_bytes, len(b"INSIDE\n"))

    # --- rejected -------------------------------------------------------
    def _assert_refused(self, artifact, marker=None):
        repl, backend, output = self.transition(artifact)
        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertIsNone(repl.analysis)
        self.assertEqual(backend.route_calls, 1)
        self.assertEqual(backend.analysis_calls, 0)
        self.assertNotIn("user", [m.get("role") for m in repl.runtime.messages])
        self.assertIsNone(repl.runtime.last_analysis_request)
        self.assertIn("refusing analysis", output)
        if marker:
            self.assertIn(marker, output)
        return repl

    def test_every_unsafe_shape_is_refused(self) -> None:
        (self.tmp / "dir").mkdir()
        (self.tmp / "final_link").symlink_to(self.secret)
        (self.tmp / "inside_link").symlink_to(self.inside)
        (self.tmp / "dangling").symlink_to(self.tmp / "missing")
        (self.tmp / "dirlink").symlink_to(self.outside)
        os.mkfifo(self.tmp / "fifo")
        sibling = Path(str(self.tmp) + "-evil")
        sibling.mkdir()
        self.addCleanup(shutil.rmtree, sibling, ignore_errors=True)
        (sibling / "x.js").write_text("SIB\n", encoding="utf-8")

        cases = {
            "parent escape": "../../etc/passwd",
            "absolute outside": str(self.secret),
            "final symlink": "final_link",
            "symlink pointing inside": "inside_link",
            "intermediate symlink": "dirlink/secret.txt",
            "dangling symlink": "dangling",
            "missing": "nope.js",
            "directory": "dir",
            "fifo": "fifo",
            "sibling prefix": str(sibling / "x.js"),
            "home relative": "~/.bashrc",
            "unknown user": "~nosuchuser/x",
            "nul byte": "a\x00b",
        }
        for label, artifact in cases.items():
            with self.subTest(case=label):
                self._assert_refused(artifact)

    def test_a_symlink_pointing_inside_is_still_refused(self) -> None:
        # Stricter than `/analysis` on purpose: the target is fine today and
        # can be repointed tomorrow, and the model chose the name.
        (self.tmp / "ok_link").symlink_to(self.inside)

        self._assert_refused("ok_link")

    def test_chat_still_works_after_a_refusal(self) -> None:
        repl = self._assert_refused(str(self.secret))
        seen: list[str] = []
        repl.runtime.ask_auto = lambda prompt, **kw: (
            seen.append(prompt) or ChatResult("ok", "s", "stop", [], 1, 1, 0, None, None)
        )

        self.ask(repl, "what is 2+2")

        self.assertEqual(seen, ["what is 2+2"])

    def test_a_refusal_leaks_no_workspace(self) -> None:
        pattern = "orbit-analysis-session-*"
        before = set(Path(tempfile.gettempdir()).glob(pattern))

        self._assert_refused(str(self.secret))

        self.assertEqual(set(Path(tempfile.gettempdir()).glob(pattern)) - before, set())

    # --- explicit command parity ----------------------------------------
    def test_explicit_analysis_may_still_name_an_outside_path(self) -> None:
        backend = CountingBackend('{"route":"CHAT"}')
        repl = self.repl(backend)

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            repl._handle_command(f"/analysis {self.secret}")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertEqual(repl.analysis.source.original_path, str(self.secret))

    def test_explicit_analysis_may_still_name_a_symlink(self) -> None:
        link = self.tmp / "explicit_link"
        link.symlink_to(self.inside)
        backend = CountingBackend('{"route":"CHAT"}')
        repl = self.repl(backend)

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            repl._handle_command(f"/analysis {link}")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS, "explicit policy unchanged")


class NoReRoutingInAnalysisTest(TransitionTestBase):
    """Once in ANALYSIS, steering must not be re-classified."""

    def test_continue_does_not_route_again(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)
        self.ask(repl, "analyze foo.js")
        route_calls_after_entry = backend.route_calls

        self.ask(repl, "continue")

        self.assertEqual(backend.route_calls, route_calls_after_entry, "no re-routing in ANALYSIS")
        self.assertEqual(backend.analysis_calls, 2)
        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)

    def test_steering_stays_in_analysis_without_routing(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)
        self.ask(repl, "analyze foo.js")

        for text in ("look at the strings", "continue", "now the header"):
            self.ask(repl, text)
            self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)

        self.assertEqual(backend.route_calls, 1, "the route ran once, at entry")
        self.assertEqual(backend.analysis_calls, 4)

    def test_explicit_chat_returns_and_routing_resumes(self) -> None:
        backend = CountingBackend(self.analysis_payload())
        repl = self.repl(backend)
        self.ask(repl, "analyze foo.js")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._handle_command("/chat")

        self.assertIs(repl.workflow_mode, WorkflowMode.CHAT)
        self.assertEqual(backend.route_calls, 1, "/chat makes no model call")


class ExplicitCommandUnchangedTest(TransitionTestBase):
    def test_explicit_analysis_still_works_without_any_route_call(self) -> None:
        backend = CountingBackend('{"route":"CHAT"}')
        repl = self.repl(backend)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._handle_command(f"/analysis {self.artifact}")

        self.assertIs(repl.workflow_mode, WorkflowMode.ANALYSIS)
        self.assertEqual(backend.route_calls, 0)
        self.assertEqual(backend.analysis_calls, 0)

    def test_neither_entry_path_persists_a_chat_session_on_transition(self) -> None:
        """Parity, and deliberate: ANALYSIS is not resumable.

        The routed transition skips `_save_session` exactly as `/analysis`
        does. Pinned so the two paths cannot quietly diverge.
        """
        from orbit.runtime.sessions import SessionStore

        routed_store = SessionStore(self.tmp / "routed.json")
        routed = self.repl(CountingBackend(self.analysis_payload()))
        routed.session = routed_store
        self.ask(routed, "analyze foo.js as an artifact")

        explicit_store = SessionStore(self.tmp / "explicit.json")
        explicit = self.repl(CountingBackend('{"route":"CHAT"}'))
        explicit.session = explicit_store
        with contextlib.redirect_stdout(io.StringIO()):
            explicit._handle_command(f"/analysis {self.artifact}")

        self.assertEqual(routed_store.path.exists(), explicit_store.path.exists())
        self.assertFalse(routed_store.path.exists())

    def test_both_entry_paths_produce_the_same_session_shape(self) -> None:
        explicit_backend = CountingBackend('{"route":"CHAT"}')
        explicit = self.repl(explicit_backend)
        with contextlib.redirect_stdout(io.StringIO()):
            explicit._handle_command(f"/analysis {self.artifact}")

        routed_backend = CountingBackend(self.analysis_payload())
        routed = self.repl(routed_backend)
        self.ask(routed, "analyze foo.js")

        self.assertEqual(
            explicit.analysis.source.sha256, routed.analysis.source.sha256
        )
        self.assertEqual(explicit.analysis.messages[0], routed.analysis.messages[0])


class BackendStaysModeAgnosticTest(TransitionTestBase):
    def test_no_workflow_mode_reaches_the_backend(self) -> None:
        seen: list[list[dict]] = []

        class Recording(CountingBackend):
            def chat_stream(self, messages, **kwargs):
                seen.append([dict(m) for m in messages])
                return super().chat_stream(messages, **kwargs)

        backend = Recording(self.analysis_payload())
        repl = self.repl(backend)
        self.ask(repl, "analyze foo.js")

        blob = json.dumps(seen)
        for token in ("workflow_mode", "WorkflowMode", "last_analysis_request"):
            self.assertNotIn(token, blob)

    def test_the_backend_module_has_no_route_analysis_logic(self) -> None:
        # The backend may name ANALYSIS_STEP_PHASE -- that is the KV phase
        # label the analysis runtime already declares, and it predates this
        # mission. What it must not do is decide anything about the ANALYSIS
        # route: no ToolRoute.ANALYSIS, no artifact handling.
        for path in (SRC / "orbit" / "backend").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("ToolRoute", text, path.name)
            self.assertNotIn("last_analysis_request", text, path.name)
            self.assertNotIn('"ANALYSIS"', text, path.name)


if __name__ == "__main__":
    unittest.main()
