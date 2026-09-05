"""The deterministic appendix must reach the analyst on both report paths.

`report()` streams the model's prose through `on_delta` and attaches the
appendix to `report.text` afterwards. A terminal that prints `report.text`
only when no model call happened therefore shows the appendix on the empty
path and silently drops it on the ordinary one -- losing exactly the half of
the report that does not depend on the model having mentioned anything.

These cover both paths, and the two failure modes a naive fix invites: the
prose printed twice, or the appendix printed twice.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from orbit.backend.base import ChatResult
from orbit.runtime.analysis_runtime import (
    REPORT_NOT_COMPOSED_PREFIX,
    AnalysisRuntime,
    acquire_analysis_source,
)
from orbit.runtime.evidence import EvidenceStore
from orbit.terminal.config import AppConfig
from orbit.terminal.repl import Repl
from orbit.runtime.workflow_mode import WorkflowMode

PROSE = "The artifact drops a hidden second stage."
HEADING = "## Deterministic transformations"

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


def encode(text: str, key: int, delimiter: str) -> str:
    return delimiter.join(str(ord(c) ^ key) for c in text)


def _control_reply(offered: "list[str]") -> "ChatResult | None":
    """A valid answer to a control tool, or None when none was offered.

    These doubles stream prose for every call, which the structured
    controller correctly reads as "the model cannot use the protocol": PLAN
    is attempted twice and the run reports itself unsupported before any
    action, so nothing reaches the report these tests are about. Answering
    the control tools here restores the run without touching what the
    doubles say on the paths the suite actually measures -- the step prose
    and the closing report.
    """
    import json

    if "submit_analysis_plan" in offered:
        arguments = {
            "questions": [
                {
                    "question": f"Report fixture question {i + 1}",
                    "missing_fact": "needs execution",
                }
                for i in range(6)
            ]
        }
        name = "submit_analysis_plan"
    elif "finish_analysis_question" in offered:
        arguments = {"status": "still_open", "answer_summary": "more to do"}
        name = "finish_analysis_question"
    else:
        return None
    return ChatResult(
        content="", model="m", finish_reason="stop",
        tool_calls=[{
            "id": "control_call", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
        prompt_tokens=10, completion_tokens=5, cached_tokens=0,
        prompt_tokens_per_second=None, generation_tokens_per_second=None,
    )


class _ReportBackend:
    """Streams prose the way the real backend does, then returns it."""

    def __init__(self, prose: str = PROSE) -> None:
        self.prose = prose
        self.calls = 0

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                    on_delta=None, on_progress=None):
        control = _control_reply([t["function"]["name"] for t in (tools or [])])
        if control is not None:
            return control
        self.calls += 1
        if on_delta:
            on_delta(self.prose)
        return ChatResult(
            content=self.prose, model="m", finish_reason="stop", tool_calls=[],
            prompt_tokens=10, completion_tokens=5, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )


class _StubBackend:
    """The Repl configures `thinking` on its backend at construction."""

    thinking = False


class _StubRuntime:
    """The chat runtime the Repl requires but this path never uses."""

    def __init__(self) -> None:
        self.messages: list = []
        self.context_tokens = 8192
        self.evidence_store = None
        self.last_memory_refresh = None

    def can_continue_last_response(self) -> bool:
        return False


class ReportVisibilityTestBase(unittest.TestCase):
    def _analysis(self, source_text: str, backend=None) -> AnalysisRuntime:
        tmpdir = tempfile.TemporaryDirectory(prefix="orbit-report-vis-")
        self.addCleanup(tmpdir.cleanup)
        tmp = Path(tmpdir.name)
        artifact = tmp / "artifact.js"
        artifact.write_text(source_text, encoding="utf-8")
        runtime = AnalysisRuntime(
            backend=backend,
            source=acquire_analysis_source(artifact, tmp / "owned"),
            evidence_store=EvidenceStore(root=tmp / "evidence"),
        )
        self.addCleanup(runtime.close)
        return runtime

    def _repl(self, analysis: AnalysisRuntime) -> Repl:
        stub = _StubRuntime()
        repl = Repl(runtime=stub, backend=_StubBackend(), config=AppConfig(workdir=Path(".")))
        repl.analysis = analysis
        repl.workflow_mode = WorkflowMode.ANALYSIS
        return repl

    def _render(self, analysis: AnalysisRuntime) -> str:
        """What the analyst actually sees for `/report`."""
        repl = self._repl(analysis)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._handle_report_command("summarise")
        return out.getvalue()

    def _finding(self, analysis: AnalysisRuntime) -> None:
        """One ordinary action finding, so a model report has something to cite."""
        analysis.evidence_store.add(
            "execute_analysis", "an action finding",
            metadata={"tool_call_id": "c1", "user_turn_id": "t1",
                      "produced_by_phase": "analysis_action"},
        )


class StreamedPathTests(ReportVisibilityTestBase):
    """model_calls > 0: prose streamed, appendix printed after."""

    def _source(self, secret: str = "STAGE-TWO") -> str:
        return DECODER + f'dec("{encode(secret, 19, ",")}", 19, ",");\n'

    def test_prose_and_appendix_each_appear_exactly_once(self) -> None:
        analysis = self._analysis(self._source(), backend=_ReportBackend())
        self._finding(analysis)

        output = self._render(analysis)

        self.assertEqual(output.count(PROSE), 1, "model prose must not repeat")
        self.assertEqual(output.count(HEADING), 1, "appendix must not repeat")
        self.assertIn("STAGE-TWO", output)

    def test_the_appendix_follows_the_model_prose(self) -> None:
        analysis = self._analysis(self._source(), backend=_ReportBackend())
        self._finding(analysis)

        output = self._render(analysis)

        self.assertLess(
            output.index(PROSE), output.index(HEADING),
            "deterministic evidence is rendered after the reasoning about it",
        )

    def test_the_whole_report_text_is_not_printed_again(self) -> None:
        """Printing `report.text` here would repeat prose already streamed."""
        analysis = self._analysis(self._source(), backend=_ReportBackend())
        self._finding(analysis)

        output = self._render(analysis)
        self.assertEqual(output.count(PROSE), 1)

    def test_prose_resembling_the_appendix_is_not_truncated_or_split(self) -> None:
        """No slicing of the report text: the two halves are never separated
        by searching for a heading that the model may itself have written."""
        mimic = f"I will now describe them.\n{HEADING}\nnothing real here."
        analysis = self._analysis(self._source(), backend=_ReportBackend(mimic))
        self._finding(analysis)

        output = self._render(analysis)

        self.assertIn("I will now describe them.", output)
        self.assertIn("nothing real here.", output)
        # The model's imitation plus the real one.
        self.assertEqual(output.count(HEADING), 2)
        self.assertIn("STAGE-TWO", output)

    def test_multiple_stages_all_render(self) -> None:
        source = DECODER + "".join(
            f'dec("{encode(f"STAGE-{i}", 7, ",")}", 7, ",");\n' for i in range(4)
        )
        analysis = self._analysis(source, backend=_ReportBackend())
        self._finding(analysis)

        output = self._render(analysis)

        self.assertEqual(len(analysis.transform_stages), 4)
        for index in range(4):
            with self.subTest(stage=index):
                self.assertIn(f"STAGE-{index}", output)
        for _stage, record in analysis.transform_stages:
            self.assertIn(record.evidence_id, output)

    def test_a_decoded_uri_is_rendered_verbatim(self) -> None:
        uri = "http://synthetic.invalid/beacon?id=VISIBLE"
        analysis = self._analysis(
            DECODER + f'dec("{encode(uri, 29, ",")}", 29, ",");\n',
            backend=_ReportBackend(),
        )
        self._finding(analysis)

        self.assertIn(uri, self._render(analysis))


class ZeroModelCallPathTests(ReportVisibilityTestBase):
    """model_calls == 0: the whole report is printed, and only once."""

    def test_the_appendix_appears_once_with_no_findings(self) -> None:
        analysis = self._analysis(
            DECODER + f'dec("{encode("ONLY-TRANSFORM", 11, ",")}", 11, ",");\n'
        )
        output = self._render(analysis)

        self.assertEqual(output.count(HEADING), 1)
        self.assertIn("ONLY-TRANSFORM", output)

    def test_the_no_evidence_notice_reaches_the_terminal(self) -> None:
        """The zero-call branch carries the whole report, not just the
        appendix: a session with nothing to say must still say it."""
        from orbit.runtime.analysis_runtime import NO_EVIDENCE_REPORT

        analysis = self._analysis("var x = 1;\n")
        output = self._render(analysis)

        self.assertIn(NO_EVIDENCE_REPORT, output)

    def test_the_no_evidence_notice_accompanies_the_appendix(self) -> None:
        """Both halves, when an artifact decodes but produced no findings."""
        from orbit.runtime.analysis_runtime import NO_EVIDENCE_REPORT

        analysis = self._analysis(
            DECODER + f'dec("{encode("NOTICE-AND-STAGE", 5, ",")}", 5, ",");\n'
        )
        output = self._render(analysis)

        self.assertIn(NO_EVIDENCE_REPORT, output)
        self.assertEqual(output.count(HEADING), 1)
        self.assertIn("NOTICE-AND-STAGE", output)

    def test_a_refusal_notice_reaches_the_terminal(self) -> None:
        """A refused report must say so, not silently print evidence."""
        from orbit.runtime.context_manager import ContextAdmissionError

        analysis = self._analysis(
            DECODER + f'dec("{encode("REFUSAL-TEXT", 9, ",")}", 9, ",");\n',
            backend=_ReportBackend(),
        )
        self._finding(analysis)

        def _refuse(*_args, **_kwargs):
            raise ContextAdmissionError("required-context-does-not-fit")

        analysis._admit = _refuse
        output = self._render(analysis)

        self.assertIn("could not be composed", output)
        self.assertIn("REFUSAL-TEXT", output)

    def test_a_refused_report_still_shows_the_appendix_once(self) -> None:
        from orbit.runtime.context_manager import ContextAdmissionError

        analysis = self._analysis(
            DECODER + f'dec("{encode("REFUSED-BUT-SHOWN", 23, ",")}", 23, ",");\n',
            backend=_ReportBackend(),
        )
        self._finding(analysis)

        def _refuse(*_args, **_kwargs):
            raise ContextAdmissionError("required-context-does-not-fit")

        analysis._admit = _refuse
        output = self._render(analysis)

        self.assertEqual(output.count(HEADING), 1)
        self.assertIn("REFUSED-BUT-SHOWN", output)


class NoTransformTests(ReportVisibilityTestBase):
    """An artifact with nothing to decode emits no appendix at all."""

    def test_no_phantom_appendix_on_the_streamed_path(self) -> None:
        analysis = self._analysis("var x = 1;\n", backend=_ReportBackend())
        self._finding(analysis)

        output = self._render(analysis)

        self.assertEqual(analysis.transform_stages, [])
        self.assertIn(PROSE, output)
        self.assertNotIn(HEADING, output)
        self.assertNotIn("Deterministic", output)
        # Nor blank separators for an appendix that does not exist: the guard
        # decides whether anything is emitted at all, not merely what. Without
        # it the branch still prints its separator, leaving trailing blank
        # lines after the prose that no content ever follows.
        self.assertEqual(output.rstrip("\n").count("\n\n\n"), 0)
        self.assertFalse(
            output.endswith("\n\n\n"),
            "an absent appendix must emit no separator of its own",
        )

    def test_no_phantom_appendix_on_the_zero_call_path(self) -> None:
        analysis = self._analysis("var x = 1;\n")
        output = self._render(analysis)
        self.assertNotIn(HEADING, output)


REPORT_PROSE = "Closing report prose, distinct from any step."


class _TwoPhaseBackend:
    """Step prose first, then a distinct closing-report prose.

    The autonomous path makes several model calls; reusing one string for all
    of them would make "printed once" unfalsifiable.
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                    on_delta=None, on_progress=None):
        control = _control_reply([t["function"]["name"] for t in (tools or [])])
        if control is not None:
            return control
        self.calls += 1
        text = REPORT_PROSE if tools == [] else PROSE
        if on_delta:
            on_delta(text)
        return ChatResult(
            content=text, model="m", finish_reason="stop", tool_calls=[],
            prompt_tokens=10, completion_tokens=5, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )


class AutonomousRunTests(ReportVisibilityTestBase):
    """The closing report of an autonomous run has the same two paths.

    This is where a run produces the most evidence, so an appendix dropped
    here is the whole defect again on the path that matters most: the address
    is computed, attested, and never told to anyone.
    """

    def _autonomous_repl(self, analysis: AnalysisRuntime) -> Repl:
        repl = self._repl(analysis)
        repl.autonomous_analysis = True
        return repl

    def _run(self, analysis: AnalysisRuntime) -> str:
        repl = self._autonomous_repl(analysis)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._ask_analysis("analyse this artifact")
        return out.getvalue()

    def _source(self, secret: str) -> str:
        return DECODER + f'dec("{encode(secret, 21, ",")}", 21, ",");\n'

    def _run_with_finding(self, analysis: AnalysisRuntime) -> str:
        """A finding, so the closing report really makes a model call.

        Without one `report()` short-circuits to NO_EVIDENCE_REPORT with
        `model_calls == 0` -- the other branch entirely, which would leave the
        streamed path untested while appearing to pass.
        """
        self._finding(analysis)
        return self._run(analysis)

    def test_the_appendix_reaches_the_terminal_when_the_report_streams(self) -> None:
        uri = "http://synthetic.invalid/c2-AUTONOMOUS"
        analysis = self._analysis(self._source(uri), backend=_ReportBackend())

        output = self._run_with_finding(analysis)

        self.assertIn(PROSE, output, "the closing report must have streamed")
        self.assertEqual(output.count(HEADING), 1)
        self.assertIn(uri, output)

    def test_the_closing_prose_is_not_repeated_when_the_report_streams(self) -> None:
        """Printing `final_report.text` here would show it a second time."""
        analysis = self._analysis(self._source("AUTO-STAGE"), backend=_TwoPhaseBackend())
        output = self._run_with_finding(analysis)

        self.assertEqual(output.count(REPORT_PROSE), 1)
        self.assertEqual(output.count(HEADING), 1)
        self.assertIn("AUTO-STAGE", output)

    def test_decoded_control_sequences_are_neutralised_on_this_path_too(self) -> None:
        hostile = "\x1b[2Jforged\r\x07"
        analysis = self._analysis(
            DECODER + f'dec("{encode(hostile, 21, ",")}", 21, ",");\n',
            backend=_ReportBackend(),
        )
        output = self._run_with_finding(analysis)

        self.assertIn(HEADING, output)
        for char in ("\x1b", "\r", "\x07"):
            with self.subTest(control=repr(char)):
                self.assertNotIn(char, output)

    def test_no_phantom_appendix_when_nothing_decodes(self) -> None:
        analysis = self._analysis("var x = 1;\n", backend=_ReportBackend())
        output = self._run_with_finding(analysis)
        self.assertIn(PROSE, output)
        self.assertNotIn(HEADING, output)


class TerminalSafetyTests(ReportVisibilityTestBase):
    """Decoded bytes are attacker-controlled and reach the terminal here.

    The appendix renders content the artifact chose. Printing it unsanitised
    would let a decoded payload clear the screen, forge Orbit's own output,
    set the window title, or write the clipboard through OSC 52 -- so the
    sanitiser on this path is a security control, and these pin it.
    """

    HOSTILE = (
        "\x1b[2J\x1b[H"          # clear screen, home cursor
        "\x1b[31mforged\x1b[0m"  # colour, imitating runtime output
        "\r overwrite"           # carriage return
        "\x1b]0;title\x07"       # OSC window title + BEL
        "\x1b]52;c;ZXZpbA==\x07"  # OSC 52 clipboard write
        "\x9b2J"                 # 8-bit C1 CSI
    )

    def _hostile_source(self) -> str:
        return DECODER + f'dec("{encode(self.HOSTILE, 3, ",")}", 3, ",");\n'

    def _assert_inert(self, output: str) -> None:
        for name, char in (
            ("ESC", "\x1b"), ("CR", "\r"), ("BEL", "\x07"),
            ("C1-CSI", "\x9b"), ("NUL", "\x00"),
        ):
            with self.subTest(control=name):
                self.assertNotIn(char, output)

    def test_the_streamed_path_neutralises_decoded_control_sequences(self) -> None:
        analysis = self._analysis(self._hostile_source(), backend=_ReportBackend())
        self._finding(analysis)

        raw = analysis.transform_appendix()
        self.assertIn("\x1b", raw, "the fixture must really carry escapes")

        output = self._render(analysis)
        self.assertIn(HEADING, output)
        self._assert_inert(output)

    def test_the_zero_call_path_neutralises_decoded_control_sequences(self) -> None:
        analysis = self._analysis(self._hostile_source())
        output = self._render(analysis)

        self.assertIn(HEADING, output)
        self._assert_inert(output)

    def test_the_appendix_keeps_its_line_structure(self) -> None:
        """`allow_newlines=True`: without it the whole appendix collapses into
        one unreadable line and every entry runs together."""
        source = DECODER + "".join(
            f'dec("{encode(f"LINE-{i}", 7, ",")}", 7, ",");\n' for i in range(3)
        )
        analysis = self._analysis(source, backend=_ReportBackend())
        self._finding(analysis)

        output = self._render(analysis)
        appendix_lines = [
            line for line in output.splitlines() if line.startswith("  output: ")
        ]
        self.assertEqual(len(appendix_lines), 3)


class ReportObjectTests(ReportVisibilityTestBase):
    """The report object an API consumer receives is unchanged."""

    def test_report_text_still_carries_prose_and_appendix(self) -> None:
        analysis = self._analysis(
            DECODER + f'dec("{encode("API-STAGE", 17, ",")}", 17, ",");\n',
            backend=_ReportBackend(),
        )
        self._finding(analysis)

        report = analysis.report("summarise")

        self.assertEqual(report.model_calls, 1)
        self.assertIn(PROSE, report.text)
        self.assertIn(HEADING, report.text)
        self.assertIn("API-STAGE", report.text)

    def test_rendering_does_not_mutate_the_store_or_the_stages(self) -> None:
        analysis = self._analysis(
            DECODER + f'dec("{encode("STABLE", 13, ",")}", 13, ",");\n',
            backend=_ReportBackend(),
        )
        self._finding(analysis)
        before_stages = list(analysis.transform_stages)
        before_ids = set(analysis.evidence_store.records)

        self._render(analysis)

        self.assertEqual(analysis.transform_stages, before_stages)
        self.assertEqual(set(analysis.evidence_store.records), before_ids)
        for _stage, record in analysis.transform_stages:
            self.assertIsNotNone(
                analysis.evidence_store.reattest_exact(record.evidence_id)
            )


class RealSampleRenderingTests(ReportVisibilityTestBase):
    """The pinned artifact, rendered with no model inference.

    The expected URL is used here as a post-generation oracle: the
    deterministic pass recovered it independently, and this asserts only that
    what was recovered actually reaches the analyst.
    """

    SAMPLE = Path(__file__).resolve().parents[1] / "workdir" / "samples" / "Fattura981033956.js"
    SAMPLE_SHA = "b7cfd5fdeb16d7b5ecea1063419bdad6ad280ed9b73c636707874c3f4001dc0c"
    EXPECTED_URI = (
        "http://smartmaket.com/1.php?s=AA1789FF-522F-4D9A-94E9-C9BE2BA3A1D3"
    )

    def _real_analysis(self) -> AnalysisRuntime:
        import hashlib

        if not self.SAMPLE.exists():
            self.skipTest("pinned sample not present")
        data = self.SAMPLE.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.SAMPLE_SHA:
            self.skipTest("sample is not the pinned artifact")

        tmpdir = tempfile.TemporaryDirectory(prefix="orbit-real-render-")
        self.addCleanup(tmpdir.cleanup)
        tmp = Path(tmpdir.name)
        artifact = tmp / "sample.js"
        shutil.copy(self.SAMPLE, artifact)
        runtime = AnalysisRuntime(
            backend=_ReportBackend(),
            source=acquire_analysis_source(artifact, tmp / "owned"),
            evidence_store=EvidenceStore(root=tmp / "evidence"),
        )
        self.addCleanup(runtime.close)
        return runtime

    def test_every_stage_reaches_the_terminal(self) -> None:
        analysis = self._real_analysis()
        self._finding(analysis)

        output = self._render(analysis)

        self.assertEqual(len(analysis.transform_stages), 5)
        self.assertEqual(output.count(HEADING), 1)
        # S1-S3: exact short outputs.
        self.assertIn(r"winmgmts:\\.\root\cimv2", output)
        self.assertIn("Win32_ProcessStartup", output)
        self.assertIn(r"winmgmts:\\.\root\cimv2:Win32_Process", output)
        # S4: too long to inline, so identity, length and digest.
        self.assertIn("1008 chars", output)
        self.assertIn(
            "ec8ccda0cbdce79a76748c0e32c1fb788276c762abc5fd8c6f77609a0c8f58f1", output
        )
        # S5: the recovered address, verbatim.
        self.assertIn(self.EXPECTED_URI, output)

    def test_the_appendix_is_rendered_exactly_once(self) -> None:
        """Once as a block. The address appears twice inside it -- as the
        stage's own short output and again on the `decoded URI` line, which
        exists so an indicator survives a stage too long to inline -- and that
        redundancy is the rendering rule rather than a repeated appendix.
        """
        analysis = self._real_analysis()
        self._finding(analysis)
        output = self._render(analysis)

        self.assertEqual(output.count(HEADING), 1)
        self.assertEqual(output.count(f"decoded URI: {self.EXPECTED_URI}"), 1)
        self.assertEqual(output.count(PROSE), 1)


if __name__ == "__main__":
    unittest.main()


class ReportDossierAdmissionTests(unittest.TestCase):
    """The report prompt is bounded by its cards, not by raw evidence bytes.

    A live Fattura run reached a correct analysis -- exact C2, five
    deterministic stages -- and then could not compose its narrative: the
    report prompt carried every reportable record's full re-attested body, so
    it grew with the artifact rather than with what the report had to say.
    Measured on that shape at ctx 8192: 44,287 tokens against an 8,192 limit,
    and 9,129 for a single 35 KB observation.
    """

    C2 = "http://185.234.72.19/gate.php"
    SOURCE = (
        "// padding line with some javascript content here\n" * 700
        + f"var c2 = '{C2}';\n"
    )
    CTX = 8192

    class _Ctx8192:
        """Exact-admission backend with a deterministic ~4 char/token count."""

        thinking = False

        def __init__(self, context_length: int = 8192) -> None:
            self.context_length = context_length
            self.calls = 0
            self.admitted: list | None = None

        def health(self):
            return True

        def supports_exact_context_admission(self):
            return True

        def model_info(self):
            class _Info:
                pass

            info = _Info()
            info.context_length = self.context_length
            return info

        def count_chat_tokens(self, messages, *, tools=None, thinking=False):
            from orbit.backend.base import TokenCount

            chars = sum(len(str(m.get("content") or "")) for m in messages)
            return TokenCount(
                tokens=int(chars / 4), context_tokens=self.context_length,
                rendered_hash="a" * 64, token_hash="b" * 64,
            )

        def count_text_tokens(self, text):
            from orbit.backend.base import TokenCount

            return TokenCount(
                tokens=int(len(text) / 4), context_tokens=self.context_length
            )

        def chat_stream(self, messages, *, temperature=None, max_tokens=None,
                        tools=None, on_delta=None, on_progress=None):
            # Records what admission actually let through, so a test can
            # assert on the prompt the model was really sent rather than on
            # the one `_report_messages` built.
            self.calls += 1
            self.admitted = list(messages)
            text = "The artifact drops a second stage."
            if on_delta:
                on_delta(text)
            return ChatResult(
                content=text, model="m", finish_reason="stop", tool_calls=[],
                prompt_tokens=10, completion_tokens=5, cached_tokens=0,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
            )

    def _runtime(self, *, context_length: int = 8192):
        import hashlib

        from orbit.runtime.analysis_runtime import AnalysisRuntime, AnalysisSource

        tmp = Path(tempfile.mkdtemp(prefix="orbit-dossier-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        artifact = tmp / "artifact.js"
        artifact.write_text(self.SOURCE, encoding="utf-8")
        store = EvidenceStore(root=tmp / "evidence")
        runtime = AnalysisRuntime(
            backend=self._Ctx8192(context_length),
            source=AnalysisSource(
                snapshot_path=artifact,
                sha256=hashlib.sha256(self.SOURCE.encode()).hexdigest(),
                size_bytes=len(self.SOURCE),
                original_path=str(artifact),
            ),
            evidence_store=store,
        )
        self.addCleanup(runtime.close)
        return runtime, store

    def _observe(self, store, count, *, body=None):
        for index in range(count):
            store.add(
                "execute_analysis", self.SOURCE if body is None else body,
                metadata={
                    "tool_call_id": f"call_{index}", "user_turn_id": "turn_0",
                    "status": "ok", "produced_by_phase": "analysis_step",
                },
            )

    def _compose(self, runtime):
        """Drive the REAL report path and return (report, admitted tokens).

        Deliberately not `_report_messages` + `count_chat_tokens`: the first
        version of this fix was measured that way and looked correct while
        the defect was untouched. `_admit` rehydrates evidence references,
        so a dossier that is bounded on the way in can be unbounded by the
        time it is sent. Only the admitted prompt says what happened.
        """
        report = runtime.report("What does it do?")
        admitted = runtime.backend.admitted
        tokens = (
            runtime.backend.count_chat_tokens(admitted).tokens
            if admitted is not None else None
        )
        return report, tokens

    # -- the defect ---------------------------------------------------------
    def test_one_large_observation_still_fits(self) -> None:
        """The measured live shape started failing at a single observation."""
        runtime, store = self._runtime()
        self._observe(store, 1)
        report, tokens = self._compose(runtime)
        self.assertLess(tokens, self.CTX)
        # Composed, not merely admissible: the model call was dispatched.
        self.assertEqual(report.model_calls, 1)
        self.assertNotIn(REPORT_NOT_COMPOSED_PREFIX, report.text)

    def test_duplicate_observations_do_not_scale_the_prompt(self) -> None:
        """Raw copies may stay in the store; admission must not follow them.

        Deduplication is deliberately not the fix here -- the records remain
        -- so this asserts the prompt grows by CARD, which is a small
        constant, rather than by the 35 KB each copy holds.
        """
        seen = {}
        for count in (1, 3, 5):
            runtime, store = self._runtime()
            self._observe(store, count)
            report, seen[count] = self._compose(runtime)
            self.assertLess(seen[count], self.CTX)
            self.assertEqual(report.model_calls, 1)

        # Growth per extra copy is a card, not an artifact: well under a
        # tenth of the ~8,790 tokens each raw copy used to add.
        per_copy = (seen[5] - seen[1]) / 4
        self.assertLess(per_copy, 879)

    def test_the_record_bound_caps_the_dossier(self) -> None:
        """Past MAX_REPORT_EVIDENCE_RECORDS the prompt stops growing at all."""
        from orbit.runtime.analysis_runtime import MAX_REPORT_EVIDENCE_RECORDS

        runtime, store = self._runtime()
        self._observe(store, MAX_REPORT_EVIDENCE_RECORDS)
        _, at_bound = self._compose(runtime)
        self._observe(store, 8)
        _, beyond = self._compose(runtime)
        self.assertEqual(beyond, at_bound)
        self.assertLess(at_bound, self.CTX)

    # -- what must survive --------------------------------------------------
    def test_the_exact_c2_survives_in_the_dossier(self) -> None:
        runtime, store = self._runtime()
        self._observe(store, 5)
        dossier = "\n\n".join(
            runtime._evidence_card(record)
            for record in runtime._reportable_records()
        )
        self.assertIn(self.C2, dossier)

    def test_raw_evidence_remains_re_attestable(self) -> None:
        """Bounded in the prompt, complete in the store."""
        runtime, store = self._runtime()
        self._observe(store, 5)
        records = runtime._reportable_records()
        for record in records:
            restored = store.reattest_exact(record.evidence_id)
            self.assertIsNotNone(restored)
            self.assertEqual(len(restored), len(self.SOURCE))
            self.assertIn(self.C2, restored)

    def test_no_excerpt_is_presented_as_complete(self) -> None:
        """The card states the real size beside the reference that recovers it."""
        runtime, store = self._runtime()
        self._observe(store, 1)
        card = runtime._evidence_card(runtime._reportable_records()[0])
        self.assertLess(len(card), len(self.SOURCE))
        self.assertIn(f"size: {len(self.SOURCE)} chars", card)
        self.assertIn("raw_ref: evidence:", card)

    def test_small_evidence_is_still_quoted_whole(self) -> None:
        """No regression for the ordinary case, which is most of them."""
        runtime, store = self._runtime()
        finding = "small finding: port 4444 open"
        self._observe(store, 1, body=finding)
        card = runtime._evidence_card(runtime._reportable_records()[0])
        self.assertIn(finding, card)

    def test_a_dossier_that_still_cannot_fit_fails_honestly(self) -> None:
        """Bounded is not unbounded-enough: admission stays authoritative."""
        from orbit.runtime.analysis_runtime import ContextAdmissionError

        runtime, store = self._runtime(context_length=200)
        self._observe(store, 3)
        with self.assertRaises(ContextAdmissionError):
            runtime._admit(
                runtime._report_messages(
                    "q", runtime._reportable_records()
                ),
                max_tokens=64, tools=[], next_action_reserve=0,
            )

    # -- the design rule the first fix violated -----------------------------
    def test_a_dossier_reference_is_not_a_retrieval_request(self) -> None:
        """Citation is not a request, even when it names an evidence id.

        `_with_evidence_rehydration` states the rule: retrieval is something
        the model asks for, never something a reference triggers by existing.
        The report writes its own dossier and every card names `raw_ref` as
        provenance, so admission must not read those as requests -- the first
        version of this fix did, re-inlining every record it had just replaced
        with a citation and making the prompt LARGER than the unbounded one.
        """
        runtime, store = self._runtime()
        self._observe(store, 5)
        records = runtime._reportable_records()
        messages = runtime._report_messages("q", records)

        rehydrated, ids = runtime._with_evidence_rehydration(messages)
        # The references ARE recognisable as ids -- that is what makes the
        # suppression meaningful rather than accidental.
        self.assertEqual(len(ids), len(records))
        self.assertGreater(
            runtime.backend.count_chat_tokens(rehydrated).tokens,
            runtime.backend.count_chat_tokens(messages).tokens * 5,
        )

        # And the report path does not take that route.
        report, admitted_tokens = self._compose(runtime)
        self.assertEqual(report.model_calls, 1)
        self.assertLess(admitted_tokens, self.CTX)
        sent = "\n".join(
            str(m.get("content") or "") for m in runtime.backend.admitted
        )
        self.assertNotIn(self.SOURCE, sent)
        # Provenance survives: the analyst can still follow every citation.
        for record in records:
            self.assertIn(record.evidence_id, sent)

    def test_an_ordinary_request_still_rehydrates(self) -> None:
        """The default is unchanged, which is most of the system."""
        runtime, store = self._runtime()
        finding = "finding: beacon to 10.0.0.5:4444"
        self._observe(store, 1, body=finding)
        record = runtime._reportable_records()[0]
        asked = [
            {"role": "system", "content": "s"},
            {"role": "user",
             "content": f"show me evidence:{record.evidence_id} again"},
        ]

        admitted = runtime._admit(asked, max_tokens=256, tools=None)
        self.assertIn(finding, "\n".join(
            str(m.get("content") or "") for m in admitted
        ))

        withheld = runtime._admit(
            asked, max_tokens=256, tools=None, rehydrate_evidence=False
        )
        body = "\n".join(str(m.get("content") or "") for m in withheld)
        self.assertNotIn(finding, body)
        # Still cited, just not fetched.
        self.assertIn(record.evidence_id, body)
