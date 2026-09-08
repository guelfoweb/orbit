"""Deterministic analysis progress on the terminal, without touching behaviour.

ANALYSIS-PROGRESS-1. The runtime emits one `AnalysisProgressEvent` at each
authoritative controller transition -- planning, a question becoming active,
a step being decided, a program handed to the sandbox, a step classified, a
completion checked, a question resolved or blocked, a replan, the stop, the
report -- through the optional `on_event` sink of `run_autonomous()`. The
terminal formats them; nothing in the runtime reads whether a sink exists.

The trajectory below is the live SC1 shape: PLAN with three questions; Q1's
first read delivers the source and is resolved; Q2 re-reads the same bytes
twice, is suppressed twice, and is blocked on the no-progress bound; Q3 then
gets its turn and resolves; the report is composed. Every witness is the
runtime's own result object, compared between a run with no sink and a run
with a recording sink.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, StreamProgress, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    ANALYSIS_TOOL_NAME,
    FINISH_TOOL_NAME,
    PLAN_TOOL_NAME,
    AnalysisProgressEvent,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402
import orbit.terminal.analysis_mode as analysis_mode  # noqa: E402
from orbit.terminal.analysis_mode import AnalysisProgressDisplay, format_progress_event  # noqa: E402
import orbit.terminal.streaming as streaming_module  # noqa: E402
from orbit.terminal.streaming import StreamRenderer  # noqa: E402

CTX = 8192
SOURCE = "import os\n\n\ndef handler(name):\n    return os.environ.get(name)\n"
SHA = hashlib.sha256(SOURCE.encode()).hexdigest()
QUESTIONS = [
    {"question": "What does the handler read from the environment?", "missing_fact": "the variable"},
    {"question": "Where does the value go?", "missing_fact": "the sink"},
    {"question": "Is anything written to disk?", "missing_fact": "file writes"},
]
# What the sandbox answers, in order. Q1: the full source (delivery). Q2: the
# same bytes twice (both suppressed, Q2 blocked). Q3: a finding.
OUTPUTS = [SOURCE + "\n", SOURCE + "\n", SOURCE + "\n", "WRITES: none\n"]


class _Model:
    """PLAN once; every STEP is a distinct program; FINISH per question."""

    def __init__(self) -> None:
        self.actions = 0
        self.calls: list[dict] = []
        self.decisions = {"Q1": "resolved", "Q3": "resolved"}
        self.raise_on_step: int | None = None

    def reply(self, tools, active_question: str | None):
        names = [t["function"]["name"] for t in (tools or [])]
        if PLAN_TOOL_NAME in names:
            return self._call(PLAN_TOOL_NAME, {"questions": QUESTIONS})
        if FINISH_TOOL_NAME in names:
            status = self.decisions.get(active_question or "", "still_open")
            return self._call(FINISH_TOOL_NAME, {"status": status, "answer_summary": "done"})
        if ANALYSIS_TOOL_NAME in names:
            self.actions += 1
            if self.raise_on_step == self.actions:
                raise KeyboardInterrupt
            return self._call(ANALYSIS_TOOL_NAME, {"code": f"# action {self.actions}\nprint({self.actions})"})
        return None

    def _call(self, name, arguments):
        return [{"id": f"c{len(self.calls)}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}]


class _Backend:
    thinking = False

    def __init__(self, model: _Model, runtime_ref: dict) -> None:
        self.model = model
        self.runtime_ref = runtime_ref
        self.chat_calls: list[dict] = []

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * 0.25), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def count_text_tokens(self, text: str):
        return TokenCount(tokens=len(text.split()), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        tools = kwargs.get("tools")
        self.chat_calls.append({"tools": tools, "n": len(messages)})
        runtime = self.runtime_ref.get("runtime")
        active = getattr(runtime, "_diagnostic_active_question", None)
        calls = self.model.reply(tools, active) or []
        self.model.calls.extend(calls)
        content = "" if tools else "REPORT: the handler reads one variable."
        return ChatResult(content=content, model="m", finish_reason="stop", tool_calls=calls,
                          prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                          prompt_tokens_per_second=None, generation_tokens_per_second=None)

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


def run_trajectory(*, on_event=None, outputs=None, finalize=True, raise_on_step=None):
    """The SC1-shaped run; returns (run, runtime, dispatched, backend)."""
    model = _Model()
    model.raise_on_step = raise_on_step
    ref: dict = {}
    backend = _Backend(model, ref)
    workspace = AnalysisWorkspace.create()
    path = workspace.source_root / "artifact.py"
    path.write_bytes(SOURCE.encode())
    runtime = AnalysisRuntime(
        backend=backend,
        source=AnalysisSource(snapshot_path=path, sha256=SHA, size_bytes=len(SOURCE.encode()),
                              original_path=str(path)),
        evidence_store=EvidenceStore(root=workspace.root / "evidence"),
        workspace=workspace,
    )
    ref["runtime"] = runtime
    queue = list(OUTPUTS if outputs is None else outputs)
    dispatched: list[str] = []

    def sandbox(**kw):
        out = queue.pop(0) if queue else f"FINDING {len(dispatched)}\n"
        dispatched.append(out)
        return AnalysisResult(status="ok", code_sha256=hashlib.sha256(kw["code"].encode()).hexdigest(),
                              input_sha256="i" * 64, stdout=out, stderr="", exit_status=0, duration_seconds=0.1)

    try:
        with mock.patch.object(module, "execute_analysis", sandbox):
            run = runtime.run_autonomous("Analyse it.", cover=False, finalize=finalize,
                                         max_model_calls=18, max_actions=8, on_event=on_event)
    finally:
        runtime.close()
    return run, runtime, dispatched, backend


def fingerprint(run, runtime, dispatched, backend) -> dict:
    """Everything a presentation change must leave untouched."""
    return {
        "model_calls": run.model_calls,
        "actions": run.actions_executed,
        "suppressed": run.suppressed_duplicates,
        "steps": [(s.action_attempted, s.action_executed, bool(s.suppressed_duplicate_of)) for s in run.steps],
        "progress": [p.classification for p in run.progress],
        "stop": run.stop_reason,
        "resolved": tuple(run.resolved_questions),
        "open": tuple(run.open_questions),
        "dispatched": list(dispatched),
        "backend_calls": [c["n"] for c in backend.chat_calls],
        "tools_per_call": [tuple(t["function"]["name"] for t in (c["tools"] or [])) for c in backend.chat_calls],
        "evidence": len(runtime.evidence_store.records),
        "report_text": run.final_report.text if run.final_report is not None else None,
        "cancelled": run.cancelled,
    }


class _Recorder:
    def __init__(self) -> None:
        self.events: list[AnalysisProgressEvent] = []

    def __call__(self, event: AnalysisProgressEvent) -> None:
        self.events.append(event)

    def names(self) -> list[str]:
        return [e.event for e in self.events]

    def lines(self) -> list[str]:
        return [f"{e.event}:{e.question_id}:{e.detail}" for e in self.events]


# --------------------------------------------------------------------------
class EventStreamTests(unittest.TestCase):
    """T1-T8: the runtime tells the sink what the controller decided."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rec = _Recorder()
        cls.result, cls.runtime, cls.dispatched, cls.backend = run_trajectory(on_event=cls.rec)

    def _events(self, name: str):
        return [e for e in self.rec.events if e.event == name]

    def test_t1_plan_emits_planning(self) -> None:
        names = self.rec.names()
        self.assertEqual(names[0], "planning")
        self.assertEqual(self._events("planning")[0].phase, module.ANALYSIS_PLAN_PHASE)

    def test_t2_the_first_active_question_is_q1_of_3(self) -> None:
        first = self._events("question")[0]
        self.assertEqual((first.question_id, first.question_index, first.question_total), ("Q1", 1, 3))
        self.assertEqual(first.detail, QUESTIONS[0]["question"])

    def test_t3_each_dispatched_program_is_one_action_event(self) -> None:
        self.assertEqual(len(self._events("action")), len(self.dispatched))
        self.assertEqual({e.detail for e in self._events("action")}, {"sandboxed analysis"})

    def test_t4_each_step_is_classified_exactly_once(self) -> None:
        self.assertEqual([e.detail for e in self._events("classified")],
                         [p.classification for p in self.result.progress])
        self.assertEqual(len(self._events("investigating")), len(self.result.steps))

    def test_t5_the_first_no_progress_is_followed_by_a_replan(self) -> None:
        names = self.rec.names()
        first_no_progress = next(i for i, e in enumerate(self.rec.events)
                                 if e.event == "classified" and e.detail == "NO_PROGRESS")
        self.assertIn("replanning", names[first_no_progress:first_no_progress + 3])
        self.assertEqual(self.rec.events[first_no_progress].question_id, "Q2")
        self.assertEqual(self.result.replans, 1)

    def test_t6_a_repeated_strategy_blocks_q2_and_q3_becomes_visible(self) -> None:
        blocked = self._events("blocked")
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].question_id, "Q2")
        self.assertIn("strategy repeated", blocked[0].detail)
        questions = [e.question_id for e in self._events("question")]
        self.assertEqual(questions, ["Q1", "Q2", "Q3"], "each question announced once, in order")
        self.assertGreater(self.rec.events.index(self._events("question")[2]), self.rec.events.index(blocked[0]))
        self.assertEqual(tuple(self.result.resolved_questions), ("Q1", "Q3"))

    def test_t7_q1_resolved_then_q2_announced(self) -> None:
        resolved = self._events("resolved")
        self.assertEqual([e.question_id for e in resolved], ["Q1", "Q3"])
        checking = self._events("checking")
        self.assertEqual([e.question_id for e in checking], ["Q1", "Q3"], "no completion is checked for a suppressed step")
        order = self.rec.names()
        i_res = self.rec.events.index(resolved[0])
        i_q2 = self.rec.events.index(self._events("question")[1])
        self.assertLess(i_res, i_q2)
        self.assertEqual(order[i_res - 1], "checking")

    def test_t8_the_report_is_announced_after_the_stop(self) -> None:
        names = self.rec.names()
        self.assertEqual(names[-2:], ["stopped", "report"])
        self.assertEqual(self._events("stopped")[0].detail, self.result.stop_reason)
        self.assertEqual(self._events("report")[0].phase, module.ANALYSIS_REPORT_PHASE)

    def test_suppressed_reads_are_skipped_events_not_actions(self) -> None:
        skipped = self._events("skipped")
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(e.question_id == "Q2" for e in skipped))
        self.assertTrue(all("already-known source" in e.detail for e in skipped))

    def test_no_event_carries_model_prose_or_program_text(self) -> None:
        for e in self.rec.events:
            for banned in ("print(", "# action", "REPORT:", "import os"):
                self.assertNotIn(banned, str(e.detail or ""), e)


# --------------------------------------------------------------------------
class ZeroDeltaTests(unittest.TestCase):
    """T11-T14: presentation cannot change the run."""

    def test_t14_recording_sink_versus_no_sink_are_identical(self) -> None:
        base = fingerprint(*run_trajectory())
        rec = _Recorder()
        with_sink = fingerprint(*run_trajectory(on_event=rec))
        self.assertEqual(base, with_sink)
        self.assertTrue(rec.events)

    def test_t13_report_text_is_byte_identical(self) -> None:
        base = run_trajectory()[0].final_report.text
        with_sink = run_trajectory(on_event=_Recorder())[0].final_report.text
        self.assertEqual(base, with_sink)
        self.assertIn("REPORT:", base)

    def test_t11_a_failing_sink_changes_nothing(self) -> None:
        def broken(event):
            raise RuntimeError("renderer exploded")

        base = fingerprint(*run_trajectory())
        self.assertEqual(base, fingerprint(*run_trajectory(on_event=broken)))

    def test_t12_a_sink_raising_keyboard_interrupt_is_not_swallowed(self) -> None:
        def interrupt(event):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            module._notify(interrupt, module.ANALYSIS_STEP_PHASE, "planning")

    def test_t12_cancellation_during_a_step_propagates_exactly_as_baseline(self) -> None:
        base = fingerprint(*run_trajectory(raise_on_step=2))
        with_sink = fingerprint(*run_trajectory(raise_on_step=2, on_event=_Recorder()))
        self.assertEqual(base, with_sink)
        self.assertTrue(base["cancelled"])

    def test_the_sink_is_never_consulted_by_the_controller(self) -> None:
        """`on_event` appears only as a parameter, a pass-through, or a `_notify` argument."""
        import inspect
        source = inspect.getsource(module.AnalysisRuntime.run_autonomous) + inspect.getsource(module.AnalysisRuntime.step)
        allowed = (
            r"^on_event: ",              # the parameter
            r"^on_event=on_event,?$",    # passed through to step()
            r"^on_event, [A-Z_]+,",      # the first argument line of a _notify call
            r"^_notify\(on_event, ",     # a one-line _notify call
            r"^`on_event`",              # docstring
            r"^#",                       # comment
        )
        offending = [
            line.strip() for line in source.splitlines()
            if "on_event" in line and not any(re.match(p, line.strip()) for p in allowed)
        ]
        self.assertEqual(offending, [], "on_event must not be read by anything but _notify")


# --------------------------------------------------------------------------
class FormattingTests(unittest.TestCase):
    def test_lines_are_concise_and_positioned(self) -> None:
        e = AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "question", "Q2", 2, 3, "Where does the value go?")
        self.assertEqual(format_progress_event(e), "[analysis] Q2/3 Investigating: Where does the value go?")
        e = AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "blocked", "Q2", 2, 3, "no new evidence: strategy repeated")
        self.assertEqual(format_progress_event(e), "[analysis] Q2/3 Blocked: no new evidence: strategy repeated")
        e = AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "classified", "Q1", 1, 3, "NEW_CONTENT")
        self.assertEqual(format_progress_event(e), "[analysis] Q1/3 New evidence recorded")
        e = AnalysisProgressEvent(module.ANALYSIS_PLAN_PHASE, "planning")
        self.assertEqual(format_progress_event(e), "[analysis] Planning investigation")
        e = AnalysisProgressEvent(module.ANALYSIS_REPORT_PHASE, "report")
        self.assertEqual(format_progress_event(e), "[analysis] Composing report")

    def test_question_text_is_sanitised_and_bounded(self) -> None:
        text = "evil\x1b[2J\r" + "x" * 200
        e = AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "question", "Q1", 1, 1, text)
        line = format_progress_event(e)
        self.assertNotIn("\x1b", line)
        self.assertNotIn("\r", line)
        self.assertLessEqual(len(line), len("[analysis] Q1/1 Investigating: ") + analysis_mode._PROGRESS_QUESTION_CHARS)
        self.assertTrue(line.endswith("…"))

    def test_an_unknown_event_prints_nothing(self) -> None:
        self.assertIsNone(format_progress_event(AnalysisProgressEvent("p", "no-such-event")))
        self.assertIsNone(format_progress_event(AnalysisProgressEvent("p", "classified", detail="WEIRD")))


class _FakeRenderer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.kwargs: list[dict] = []
        self.settled = 0

    def settle_progress_line(self) -> None:
        self.settled += 1

    def event(self, text, **kwargs) -> None:
        self.lines.append(text)
        self.kwargs.append(kwargs)


class DisplayTests(unittest.TestCase):
    """T9, T10 and deduplication, through the display and a real renderer."""

    def _events(self):
        rec = _Recorder()
        run_trajectory(on_event=rec)
        return rec.events

    def test_repeated_lines_are_printed_once(self) -> None:
        renderer = _FakeRenderer()
        display = AnalysisProgressDisplay(renderer, interactive=True)
        e = AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "investigating", "Q1", 1, 3)
        display(e); display(e); display(e)
        self.assertEqual(len(renderer.lines), 1)

    def test_a_full_run_prints_one_line_per_transition(self) -> None:
        renderer = _FakeRenderer()
        display = AnalysisProgressDisplay(renderer, interactive=True)
        for e in self._events():
            display(e)
        joined = "\n".join(renderer.lines)
        self.assertEqual(sum(1 for l in renderer.lines if "Deciding the next action" in l), 1,
                         "only Q2's second step earns the line; the first step of a question is the question line")
        for expected in ("[analysis] Planning investigation", "[analysis] Q1/3 Investigating:",
                         "[analysis] Q1/3 Running: sandboxed analysis",
                         "[analysis] Q1/3 Checking completion", "[analysis] Q1/3 Resolved",
                         "[analysis] Q2/3 Investigating:", "[analysis] Q2/3 Skipped: already-known source",
                         "[analysis] Q2/3 Replanning after no progress",
                         "[analysis] Q2/3 Blocked: no new evidence: strategy repeated",
                         "[analysis] Q3/3 Investigating:", "[analysis] Q3/3 Resolved",
                         "[analysis] Stopped:", "[analysis] Composing report"):
            self.assertIn(expected, joined, expected)
        self.assertLess(len(renderer.lines), 40, "one line per transition, never per token")
        # The ledger's verdict is folded when an outcome line already said it.
        self.assertNotIn("[analysis] Q1/3 New evidence recorded", renderer.lines)
        self.assertNotIn("[analysis] Q2/3 No new evidence", renderer.lines)

    def test_t4_a_classification_without_an_outcome_line_is_printed_once(self) -> None:
        renderer = _FakeRenderer()
        display = AnalysisProgressDisplay(renderer, interactive=True)
        display(AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "investigating", "Q1", 1, 1))
        display(AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "action", "Q1", 1, 1, "sandboxed analysis"))
        display(AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "classified", "Q1", 1, 1, "NEW_CONTENT"))
        display(AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "classified", "Q1", 1, 1, "NEW_CONTENT"))
        self.assertEqual(renderer.lines.count("[analysis] Q1/1 New evidence recorded"), 1)
        # ERROR is never folded, even after an outcome line.
        display(AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "investigating", "Q1", 1, 1))
        display(AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "still_open", "Q1", 1, 1))
        display(AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "classified", "Q1", 1, 1, "ERROR"))
        self.assertIn("[analysis] Q1/1 Action failed", renderer.lines)
        self.assertEqual(len(renderer.lines), len(set(zip(range(len(renderer.lines)), renderer.lines))))

    def test_t9_off_a_terminal_nothing_is_printed(self) -> None:
        renderer = _FakeRenderer()
        with mock.patch.object(analysis_mode, "is_tty", return_value=False):
            display = AnalysisProgressDisplay(renderer)
        for e in self._events():
            display(e)
        self.assertEqual(renderer.lines, [])

    def test_t9_redirected_stdout_keeps_its_bytes(self) -> None:
        """Through the real renderer, off a TTY: no progress line reaches stdout."""
        out = io.StringIO()
        with mock.patch.object(streaming_module, "is_tty", return_value=False), \
                mock.patch.object(streaming_module, "supports_ansi", return_value=False), \
                mock.patch.object(analysis_mode, "is_tty", return_value=False), \
                contextlib.redirect_stdout(out):
            renderer = StreamRenderer(thinking=False)
            renderer.start()
            display = AnalysisProgressDisplay(renderer)
            for e in self._events():
                display(e)
            renderer.finish()
        self.assertEqual(out.getvalue(), "")

    def test_t10_colours_disabled_means_no_ansi(self) -> None:
        out = io.StringIO()
        with mock.patch.object(streaming_module, "is_tty", return_value=True), \
                mock.patch.object(streaming_module, "supports_ansi", return_value=False), \
                mock.patch("orbit.terminal.theme.supports_ansi", return_value=False), \
                contextlib.redirect_stdout(out):
            renderer = StreamRenderer(thinking=False, interactive=True)
            display = AnalysisProgressDisplay(renderer, interactive=True)
            for e in self._events():
                display(e)
        text = out.getvalue()
        self.assertIn("[analysis] Q1/3 Resolved", text)
        self.assertNotIn("\x1b", text)

    def test_on_a_terminal_lines_are_dim_and_complete(self) -> None:
        out = io.StringIO()
        with mock.patch.object(streaming_module, "is_tty", return_value=True), \
                mock.patch.object(streaming_module, "supports_ansi", return_value=True), \
                mock.patch("orbit.terminal.theme.supports_ansi", return_value=True), \
                contextlib.redirect_stdout(out):
            renderer = StreamRenderer(thinking=False, interactive=True)
            display = AnalysisProgressDisplay(renderer, interactive=True)
            for e in self._events():
                display(e)
        visible = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue())
        self.assertIn("[analysis] Q2/3 Blocked: no new evidence: strategy repeated\n", visible)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
def generation(elapsed: float = 61.0, *, tokens: int = 100, rate: float = 4.0) -> StreamProgress:
    return StreamProgress(phase="generation", current=tokens, total=0, percent=0,
                          tokens_per_second=rate, elapsed_seconds=elapsed)


def run_trajectory_with_generation(*, on_event=None, on_step=None, on_progress=None):
    """The SC1 trajectory with a backend that reports a generation line on every call."""
    model = _Model()
    ref: dict = {}
    backend = _Backend(model, ref)
    inner = backend.chat_stream

    def chat_stream(messages, **kwargs):
        progress = kwargs.get("on_progress")
        if progress is not None:
            progress(generation())
        return inner(messages, **kwargs)

    backend.chat_stream = chat_stream  # type: ignore[method-assign]
    workspace = AnalysisWorkspace.create()
    path = workspace.source_root / "artifact.py"
    path.write_bytes(SOURCE.encode())
    runtime = AnalysisRuntime(
        backend=backend,
        source=AnalysisSource(snapshot_path=path, sha256=SHA, size_bytes=len(SOURCE.encode()),
                              original_path=str(path)),
        evidence_store=EvidenceStore(root=workspace.root / "evidence"),
        workspace=workspace,
    )
    ref["runtime"] = runtime
    queue = list(OUTPUTS)

    def sandbox(**kw):
        out = queue.pop(0) if queue else "FINDING\n"
        return AnalysisResult(status="ok", code_sha256=hashlib.sha256(kw["code"].encode()).hexdigest(),
                              input_sha256="i" * 64, stdout=out, stderr="", exit_status=0, duration_seconds=0.1)

    try:
        with mock.patch.object(module, "execute_analysis", sandbox):
            return runtime.run_autonomous("Analyse it.", cover=False, finalize=True, max_model_calls=18,
                                          max_actions=8, on_event=on_event, on_step=on_step,
                                          on_progress=on_progress)
    finally:
        runtime.close()


def _screen(raw: str) -> list[str]:
    """What a terminal shows: each `\\r` rewinds the row; escapes stripped."""
    rows: list[str] = []
    for physical in raw.split("\n"):
        visible = re.sub(r"\x1b\[[0-9;]*m", "", physical)
        rows.append(visible.split("\r")[-1].rstrip())
    return rows


class RealRendererWithStepBlocksTests(unittest.TestCase):
    """The reviewer's collision: the display, the wait line and `on_step` together.

    This is the REPL's exact wiring -- `show()` copied from `_ask_analysis` --
    with a backend that draws a generation line during every model call, on a
    faked interactive terminal. Two things have to hold that a display-only
    test cannot see: the finished generation line is committed (kept on
    screen), not erased by the next progress line; and the step block printed
    by `on_step` starts at column 0, never on a row a fresh wait tick drew.
    """

    def _capture(self, *, display: bool) -> tuple[str, object]:
        out = io.StringIO()
        with mock.patch.object(streaming_module, "is_tty", return_value=True), \
                mock.patch.object(streaming_module, "supports_ansi", return_value=True), \
                mock.patch("orbit.terminal.theme.supports_ansi", return_value=True), \
                mock.patch.object(streaming_module, "_terminal_columns", return_value=100), \
                mock.patch.object(analysis_mode, "is_tty", return_value=True), \
                contextlib.redirect_stdout(out):
            renderer = StreamRenderer(thinking=False, render_markdown_mode="plain", interval=3600)
            renderer.set_activity("analysis")
            renderer.start()

            def show(step, record) -> None:   # verbatim from repl._ask_analysis
                renderer.settle_progress_line()
                block = analysis_mode.format_analysis_step(step, prose_already_shown=renderer.rendered_visible_text)
                if renderer.rendered_visible_text:
                    print(flush=True)
                if block:
                    print(block, flush=True)
                renderer.reset_visible_text()

            sink = AnalysisProgressDisplay(renderer) if display else None
            run = run_trajectory_with_generation(on_event=sink, on_step=show, on_progress=renderer.progress)
            renderer.finish()
        return out.getvalue(), run

    def test_step_blocks_start_at_column_zero_and_generation_lines_are_kept(self) -> None:
        base_raw, base_run = self._capture(display=False)
        cand_raw, cand_run = self._capture(display=True)
        base_rows, cand_rows = _screen(base_raw), _screen(cand_raw)
        # every block row is a whole row of its own
        for row in cand_rows:
            if row.startswith("[analysis]"):
                continue
            if "action:" in row or "no new evidence:" in row:
                self.assertTrue(row.startswith(("action:", "no new evidence:")), f"block row corrupted: {row!r}")
                self.assertNotIn("analysis ·", row)
        # the per-call generation line survives exactly as often as before
        committed = lambda rows: sum(1 for r in rows if r.startswith("analysis · generating"))
        self.assertGreaterEqual(committed(cand_rows), committed(base_rows), "no generation line erased")
        # One per model call whose reply was not streamed as prose: the
        # baseline kept only the STEP lines (the display now settles before
        # PLAN's and FINISH's lines too); the report streams and is erased.
        self.assertEqual(committed(cand_rows), cand_run.model_calls - 1)
        # and the progress lines are there, whole, once each
        self.assertEqual(sum(1 for r in cand_rows if r == "[analysis] Q1/3 Resolved"), 1)
        self.assertEqual(cand_run.model_calls, base_run.model_calls)
        self.assertEqual(cand_run.final_report.text, base_run.final_report.text)

    def test_no_wait_row_is_drawn_between_an_outcome_line_and_its_block(self) -> None:
        cand_raw, _ = self._capture(display=True)
        rows = _screen(cand_raw)
        for i, row in enumerate(rows):
            if row.startswith("[analysis]") and any(k in row for k in ("Resolved", "Blocked", "Skipped")):
                following = [r for r in rows[i + 1:i + 3] if r]
                self.assertTrue(following, row)
                self.assertFalse(following[0].startswith("analysis · working"), f"wait row under {row!r}")

    def test_an_interrupt_while_printing_the_outcome_is_not_billed_as_a_finish_call(self) -> None:
        """MAJOR-2: the outcome emit lives outside the completion call's try."""
        seen: list[str] = []

        def sink(event):
            seen.append(event.event)
            if event.event == "resolved":
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            run_trajectory_with_generation(on_event=sink)
        self.assertIn("resolved", seen)


class ReplWiringTests(unittest.TestCase):
    """`_ask_analysis` attaches the display to the run it starts."""

    def test_the_repl_passes_a_display_bound_to_its_renderer(self) -> None:
        from orbit.terminal import repl as repl_module

        seen: dict = {}

        class _Analysis:
            def run_autonomous(self, message, **kwargs):
                seen.update(kwargs)
                raise KeyboardInterrupt

        repl = repl_module.Repl.__new__(repl_module.Repl)
        repl.analysis = _Analysis()
        repl.autonomous_analysis = True
        repl._analysis_checkpoint = lambda: object()  # type: ignore[method-assign]
        repl._restore_analysis_checkpoint = lambda checkpoint: None  # type: ignore[method-assign]
        out = io.StringIO()
        with mock.patch.object(streaming_module, "is_tty", return_value=False), \
                mock.patch.object(analysis_mode, "is_tty", return_value=False), \
                contextlib.redirect_stdout(out):
            repl._ask_analysis("go")
        self.assertIsInstance(seen.get("on_event"), AnalysisProgressDisplay)
        self.assertIs(seen["on_event"]._renderer, seen["on_progress"].__self__,
                      "the display prints through the renderer the run streams to")
        self.assertIn("on_step", seen)
        self.assertIn("cancelled", out.getvalue())


class WidthTests(unittest.TestCase):
    def test_a_progress_line_never_exceeds_the_terminal_width(self) -> None:
        long_question = "Where does the decoded payload go after the second stage runs on the host? " * 3
        e = AnalysisProgressEvent(module.ANALYSIS_STEP_PHASE, "question", "Q10", 10, 12, long_question)
        renderer = _FakeRenderer()
        with mock.patch.object(streaming_module, "_terminal_columns", return_value=40):
            AnalysisProgressDisplay(renderer, interactive=True)(e)
        self.assertEqual(len(renderer.lines), 1)
        self.assertLessEqual(len(renderer.lines[0]), 39)
        self.assertTrue(renderer.lines[0].endswith("…"))
        self.assertTrue(renderer.lines[0].startswith("[analysis] Q10/12 Investigating: "))
