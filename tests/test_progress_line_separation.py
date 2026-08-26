from __future__ import annotations

import contextlib
import io
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.terminal.streaming as streaming_module
from orbit.backend.base import StreamProgress
from orbit.terminal.streaming import StreamRenderer


def screen(raw: str) -> list[str]:
    """Resolve a stream of terminal bytes the way a terminal would.

    The wait line is redrawn with a bare carriage return, so the bytes on the
    wire are not what the analyst reads. Asserting on the raw string would let
    a line that is overwritten -- or one that collides with the next print --
    pass as though it were visible. Applying `\\r` here means every assertion
    below is about what actually ends up on screen.
    """
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    lines: list[str] = []
    current = ""
    for ch in raw:
        if ch == "\n":
            lines.append(current.rstrip())
            current = ""
        elif ch == "\r":
            current = ""
        else:
            current += ch
    lines.append(current.rstrip())
    return lines


def generation(elapsed: float, *, tokens: int = 552, rate: float = 4.8) -> StreamProgress:
    return StreamProgress(
        phase="generation",
        current=tokens,
        total=0,
        percent=0,
        tokens_per_second=rate,
        elapsed_seconds=elapsed,
    )


class ProgressLineHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            mock.patch("orbit.terminal.streaming.is_tty", return_value=True),
            mock.patch("orbit.terminal.streaming.supports_ansi", return_value=True),
            mock.patch("orbit.terminal.theme.supports_ansi", return_value=True),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()

    def run_step(
        self,
        *,
        elapsed: float = 114.0,
        before: str | None = "raw: ev_91adbf966e59_dc809e8533424e29",
        after: str | None = "action: ok",
        steps: int = 1,
        via_finish: bool = False,
        interactive: bool = True,
    ) -> list[str]:
        """Drive one or more analysis steps and return the resulting screen."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=interactive)
            renderer.set_activity("analysis")
            if before:
                print(before, flush=True)
            renderer.start()
            for _ in range(steps):
                renderer.progress(generation(elapsed))
                # One timer tick. The real timer thread would do this; calling
                # it directly keeps the test free of sleeps.
                if interactive:
                    renderer._render_wait_line()
                if via_finish:
                    # What a guided step does: ask for the line, then stop.
                    renderer.keep_progress_line()
                    renderer.finish()
                else:
                    renderer.settle_progress_line()
                if after:
                    print(after, flush=True)
        return screen(buffer.getvalue())


class LiveToCompletedTests(ProgressLineHarness):
    def test_the_completed_line_survives_the_next_print(self) -> None:
        """The reported defect: the next line landed on the same row.

        The wait line carries no newline of its own, so `action: ok` used to
        be printed into the middle of it -- separated only by the padding
        spaces, which read as one ragged line.
        """
        lines = self.run_step()

        self.assertIn(
            "analysis · generating · 552 tok · 4.8 tok/s · 1m 54s",
            lines,
            "the completed progress line must stand on a row of its own",
        )
        for line in lines:
            self.assertNotIn(
                "action: ok",
                line if "generating" in line else "",
                "action must not share the progress row",
            )

    def test_the_line_is_not_erased_on_completion(self) -> None:
        """Clearing the line would take the elapsed duration with it."""
        lines = self.run_step()
        self.assertTrue(
            any("generating" in line for line in lines),
            "the progress line was cleared instead of committed",
        )


class VerticalSeparationTests(ProgressLineHarness):
    def test_one_blank_line_before_and_after(self) -> None:
        lines = self.run_step()
        index = next(i for i, line in enumerate(lines) if "generating" in line)

        self.assertEqual(lines[index - 1], "", "no blank line before the progress line")
        self.assertEqual(lines[index + 1], "", "no blank line after the progress line")
        self.assertIn("raw: ev_91adbf966e59_dc809e8533424e29", lines[index - 2])
        self.assertIn("action: ok", lines[index + 2])

    def test_no_duplicate_blank_lines(self) -> None:
        """One blank line, not two. A ragged gap is its own defect."""
        for steps in (1, 2, 3):
            with self.subTest(steps=steps):
                lines = self.run_step(steps=steps)
                doubled = [
                    i for i in range(len(lines) - 1) if lines[i] == "" and lines[i + 1] == ""
                ]
                self.assertEqual(doubled, [], f"duplicate blank lines at {doubled}")

    def test_separation_holds_with_no_preceding_output(self) -> None:
        lines = self.run_step(before=None)
        index = next(i for i, line in enumerate(lines) if "generating" in line)
        self.assertEqual(lines[index + 1], "")


class ElapsedPreservationTests(ProgressLineHarness):
    def test_minute_and_second_duration_is_kept(self) -> None:
        """`1m 54s` is the whole point: it must not be lost at completion."""
        lines = self.run_step(elapsed=114.0)
        self.assertTrue(
            any("1m 54s" in line for line in lines),
            "the minute+second elapsed duration was lost",
        )

    def test_seconds_only_duration_is_kept(self) -> None:
        lines = self.run_step(elapsed=42.0)
        self.assertTrue(any("42s" in line for line in lines))

    def test_a_range_of_durations_round_trips(self) -> None:
        for elapsed, expected in ((0.0, "0s"), (9.0, "9s"), (59.0, "59s"), (60.0, "1m 0s"), (114.0, "1m 54s"), (3599.0, "59m 59s")):
            with self.subTest(elapsed=elapsed):
                lines = self.run_step(elapsed=elapsed)
                self.assertTrue(
                    any(line.endswith(expected) for line in lines),
                    f"expected a line ending in {expected!r}",
                )

    def test_token_count_and_rate_are_kept(self) -> None:
        """Presentation only: the accounting shown must be what was measured."""
        lines = self.run_step()
        line = next(line for line in lines if "generating" in line)
        self.assertIn("552 tok", line)
        self.assertIn("4.8 tok/s", line)


class ActionOutcomeTests(ProgressLineHarness):
    def test_every_action_outcome_is_separated(self) -> None:
        for outcome in ("action: ok", "action: error | boom", "action: refused"):
            with self.subTest(outcome=outcome):
                lines = self.run_step(after=outcome)
                index = next(i for i, line in enumerate(lines) if "generating" in line)
                self.assertEqual(lines[index + 1], "")
                self.assertIn(outcome, lines[index + 2])


class BothCompletionPathsTests(ProgressLineHarness):
    def test_the_guided_path_keeps_its_line_through_finish(self) -> None:
        """A guided step asks for the line, then ends at `finish()`.

        It prints its block after the renderer has stopped, so it has no live
        line to settle over -- it requests the keep instead.
        """
        lines = self.run_step(via_finish=True)
        index = next(i for i, line in enumerate(lines) if "generating" in line)
        self.assertEqual(lines[index - 1], "")
        self.assertEqual(lines[index + 1], "")
        self.assertIn("1m 54s", lines[index])

    def test_repeated_autonomous_steps_each_get_their_own_line(self) -> None:
        lines = self.run_step(steps=3)
        rendered = [line for line in lines if "generating" in line]
        self.assertEqual(len(rendered), 3, "each completed step keeps its own line")
        for line in rendered:
            self.assertIn("1m 54s", line)


class NonAnsiTests(ProgressLineHarness):
    def test_separation_does_not_depend_on_colour(self) -> None:
        """A pipe, `NO_COLOR` and `TERM=dumb` all lose `dim()`.

        The blank lines and the line itself are plain text, so the separation
        has to survive that. If it depended on escape codes it would collapse
        exactly where a transcript is most often read back.
        """
        with mock.patch("orbit.terminal.theme.supports_ansi", return_value=False):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                renderer = StreamRenderer(interval=999, interactive=True)
                renderer.set_activity("analysis")
                print("raw: ev_abc", flush=True)
                renderer.start()
                renderer.progress(generation(114.0))
                renderer._render_wait_line()
                renderer.settle_progress_line()
                print("action: ok", flush=True)
            raw = buffer.getvalue()

        self.assertNotIn("\x1b", raw, "no escape codes should be emitted")
        lines = screen(raw)
        index = next(i for i, line in enumerate(lines) if "generating" in line)
        self.assertEqual(lines[index - 1], "")
        self.assertEqual(lines[index + 1], "")
        self.assertIn("1m 54s", lines[index])

    def test_a_non_interactive_stream_still_prints_no_progress_line(self) -> None:
        """Unchanged behaviour: a pipe never gets the live line at all.

        The fix must not start emitting progress rows into redirected output
        that never had them.
        """
        lines = self.run_step(interactive=False)
        self.assertFalse(
            any("generating" in line for line in lines),
            "a non-interactive stream must not gain a progress line",
        )
        self.assertIn("raw: ev_91adbf966e59_dc809e8533424e29", lines[0])
        self.assertIn("action: ok", lines[1])


class TerminalWidthTests(ProgressLineHarness):
    def test_the_elapsed_duration_survives_a_narrow_terminal(self) -> None:
        """Truncation trims the detail, never the duration."""
        for columns in (40, 60, 80, 200):
            with self.subTest(columns=columns):
                with mock.patch.object(
                    streaming_module, "_terminal_columns", return_value=columns
                ):
                    lines = self.run_step()
                line = next(line for line in lines if "generating" in line)
                self.assertTrue(
                    line.endswith("1m 54s"),
                    f"elapsed lost at {columns} columns: {line!r}",
                )
                self.assertLessEqual(len(line), columns, "the committed line overflowed")

    def test_no_row_exceeds_the_terminal_width(self) -> None:
        with mock.patch.object(streaming_module, "_terminal_columns", return_value=40):
            lines = self.run_step()
        for line in lines:
            self.assertLessEqual(len(line), 40, f"row wider than the terminal: {line!r}")


class NonGenerationPhaseTests(ProgressLineHarness):
    def test_a_prefill_line_is_still_cleared(self) -> None:
        """Only a finished generation is worth keeping.

        Prefill and the bare "working" tick are scaffolding: committing them
        would leave rows the analyst has no use for once the step is over.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.set_activity("analysis")
            renderer.start()
            renderer.progress(
                StreamProgress(
                    phase="prefill",
                    current=0,
                    total=100,
                    percent=10,
                    evaluated_current=10,
                    evaluated_total=100,
                )
            )
            renderer._render_wait_line()
            renderer.finish()
            print("action: ok", flush=True)
        lines = screen(buffer.getvalue())

        self.assertFalse(
            any("prefill" in line for line in lines),
            "a prefill line should not be committed to the transcript",
        )


if __name__ == "__main__":
    unittest.main()


class ProductionWiringTests(unittest.TestCase):
    """The autonomous step renderer must settle before it prints.

    Every test above drives `settle_progress_line()` itself, which proves the
    renderer behaves -- and proves nothing about whether production calls it.
    Removing the call from `show()` leaves all of them passing while the
    reported defect returns in full, so the wiring is asserted here directly.
    """

    def test_the_autonomous_step_renderer_settles_first(self) -> None:
        import inspect

        from orbit.terminal import repl as repl_module

        source = inspect.getsource(repl_module.Repl._ask_analysis)
        show = source[source.index("def show(") :]
        show = show[: show.index("run = self.analysis.run_autonomous")]

        self.assertIn(
            "settle_progress_line()",
            show,
            "show() must end the live progress line before printing a step",
        )
        self.assertLess(
            show.index("settle_progress_line()"),
            show.index("print("),
            "the settle must come before the first print, not after",
        )

    def test_the_guided_path_asks_to_keep_its_line(self) -> None:
        """A guided step prints after `finish()`, so it must request the keep.

        Every guided test drives `keep_progress_line()` itself, which proves
        the renderer honours a request and nothing about whether production
        makes one. Removing the call would leave those tests green while the
        guided progress line silently disappeared again.
        """
        import inspect

        from orbit.terminal import repl as repl_module

        source = inspect.getsource(repl_module.Repl._ask_analysis)

        self.assertIn(
            "keep_progress_line()",
            source,
            "the guided path must ask for its finished line to be kept",
        )
        self.assertLess(
            source.index("keep_progress_line()"),
            source.rindex("renderer.finish()"),
            "the request must come before the renderer stops",
        )

    def test_the_seam_is_safe_when_nothing_is_pending(self) -> None:
        """Called on every step, including ones that never drew a line."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.settle_progress_line()
            renderer.settle_progress_line()

        self.assertEqual(buffer.getvalue(), "", "settling nothing must print nothing")


class MultiStepStateTests(ProgressLineHarness):
    """Each committed step must carry its own counts, not the previous one's."""

    def test_each_step_keeps_its_own_measurements(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.set_activity("analysis")
            renderer.start()
            for tokens, rate, elapsed in ((100, 4.0, 10.0), (200, 5.0, 20.0)):
                renderer.progress(generation(elapsed, tokens=tokens, rate=rate))
                renderer._render_wait_line()
                renderer.settle_progress_line()
                print("action: ok", flush=True)
        lines = [line for line in screen(buffer.getvalue()) if "generating" in line]

        self.assertEqual(len(lines), 2)
        self.assertIn("100 tok", lines[0])
        self.assertIn("10s", lines[0])
        self.assertIn("200 tok", lines[1])
        self.assertIn("20s", lines[1])

    def test_a_tick_after_settling_does_not_recommit_stale_counts(self) -> None:
        """A timer tick between steps must not repeat the finished line.

        The timer keeps running after a step is committed. Without clearing
        the progress state, the next tick would redraw -- and then commit --
        the previous step's token count and duration as though they were new.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.set_activity("analysis")
            print("raw: ev_abc", flush=True)
            renderer.start()
            renderer.progress(generation(10.0, tokens=100, rate=4.0))
            renderer._render_wait_line()
            renderer.settle_progress_line()
            print("action: ok", flush=True)
            renderer._render_wait_line()
            renderer.finish()
        lines = screen(buffer.getvalue())

        self.assertEqual(
            len([line for line in lines if "100 tok" in line]),
            1,
            "the finished step was committed twice",
        )


class ChatPresentationTests(ProgressLineHarness):
    """CHAT must render exactly as it did before this change.

    Native streaming reports generation progress per token, so a progress line
    is always armed by the time the first delta arrives. Committing it there
    would put a `model · generating · …` row and two blank lines above every
    CHAT answer -- a regression in the one place this change was required not
    to touch.
    """

    def chat_turn(self) -> list[str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.start()
            renderer.progress(generation(0.0, tokens=1, rate=5.0))
            renderer._render_wait_line()
            renderer.write("The answer is 42.")
            renderer.finish()
            print()
            print("~ 80 tok · 12s", flush=True)
        return screen(buffer.getvalue())

    def test_a_streamed_answer_gains_no_progress_row(self) -> None:
        lines = self.chat_turn()
        self.assertFalse(
            any("generating" in line for line in lines),
            "a committed progress row appeared above a CHAT answer",
        )

    def test_the_answer_is_the_first_thing_printed(self) -> None:
        """No leading blank lines before the reply."""
        lines = self.chat_turn()
        self.assertEqual(lines[0], "The answer is 42.")

    def test_an_abandoned_line_never_surfaces_later(self) -> None:
        """A line dropped mid-stream must not be committed by a later stop."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.start()
            renderer.progress(generation(114.0))
            renderer._render_wait_line()
            renderer.write("prose")
            renderer.finish()
            renderer.finish()
        lines = screen(buffer.getvalue())

        self.assertFalse(any("generating" in line for line in lines))


class TimerInterferenceTests(ProgressLineHarness):
    """A tick must not draw a new line into the caller's print window.

    Serialising the timer and the committing thread is not sufficient on its
    own: a tick landing after the commit draws a fresh unterminated line, and
    the caller's output then collides with that one instead.
    """

    def test_no_tick_lands_between_the_commit_and_the_next_print(self) -> None:
        import time

        collisions = 0
        for _ in range(20):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                renderer = StreamRenderer(interval=0.002, interactive=True)
                renderer.set_activity("analysis")
                renderer.start()
                renderer.progress(generation(114.0))
                renderer.settle_progress_line()
                # The window `format_analysis_step` occupies in production.
                time.sleep(0.02)
                print("action: ok", flush=True)
                renderer.finish()
            for line in screen(buffer.getvalue()):
                if ("generating" in line or "working" in line) and "action:" in line:
                    collisions += 1

        self.assertEqual(collisions, 0, "a timer tick collided with the caller's output")

    def test_the_timer_resumes_for_the_following_step(self) -> None:
        """Suspending ticks must not silence the rest of the run."""
        lines = self.run_step(steps=2)
        rendered = [line for line in lines if "generating" in line]
        self.assertEqual(len(rendered), 2, "the second step drew no progress line")


class ToolCallPresentationTests(ProgressLineHarness):
    """A CHAT turn that only calls a tool must gain nothing either.

    Tool-call tokens never reach `write()` -- the native protocol delivers
    them as tool_call events, not deltas -- while the same model call still
    emits generation progress per token. A renderer that infers "the caller
    is finished" from streamed prose therefore sees a tool turn as a finished
    step, and commits a progress row above every tool call. That is why the
    commit is requested explicitly rather than deduced.
    """

    def tool_turn(self) -> list[str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.start()
            renderer.progress(generation(2.0, tokens=12))
            renderer._render_wait_line()
            renderer.event("tool: read_file(path=a.js)")
        return screen(buffer.getvalue())

    def test_a_tool_call_gains_no_progress_row(self) -> None:
        lines = self.tool_turn()
        self.assertFalse(
            any("generating" in line for line in lines),
            "a committed progress row appeared above a CHAT tool call",
        )

    def test_the_tool_event_is_the_first_thing_printed(self) -> None:
        lines = self.tool_turn()
        self.assertEqual(lines[0], "tool: read_file(path=a.js)")

    def test_a_turn_with_no_prose_at_all_gains_nothing(self) -> None:
        """`finish()` with neither prose nor a request keeps nothing."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.start()
            renderer.progress(generation(2.0, tokens=2))
            renderer._render_wait_line()
            renderer.finish()
        lines = screen(buffer.getvalue())

        self.assertFalse(
            any("generating" in line for line in lines),
            "a silent turn committed a progress row",
        )

    def test_an_unrequested_line_cannot_surface_from_a_later_stop(self) -> None:
        """An abandoned line must not be committed by a subsequent request."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.start()
            renderer.progress(generation(2.0, tokens=7))
            renderer._render_wait_line()
            renderer.event("tool: read_file(path=a.js)")
            # A later step asks to keep its own line; the abandoned one is gone.
            renderer.keep_progress_line()
            renderer.finish()
        lines = screen(buffer.getvalue())

        self.assertFalse(
            any("7 tok" in line for line in lines),
            "an abandoned line resurfaced on a later request",
        )


class CommitRequestInvariantTests(ProgressLineHarness):
    """A request must not outlive the line it was made for.

    `keep_progress_line()` can be called when nothing has settled yet. If the
    flag survives the drop path, a later line -- one nobody asked to keep --
    is committed instead. Production happens to call `keep` immediately before
    `finish()`, which hides this; the invariant is asserted directly so a
    reordering cannot reintroduce it.
    """

    def test_a_stale_request_does_not_commit_a_later_line(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.set_activity("analysis")
            renderer.start()
            renderer.keep_progress_line()
            # Nothing has settled, so this takes the drop path.
            renderer.event("tool: x")
            renderer.progress(generation(99.0, tokens=9, rate=5.0))
            renderer._render_wait_line()
            renderer.finish()
        lines = screen(buffer.getvalue())

        self.assertFalse(
            any("generating" in line for line in lines),
            "a stale keep request committed a line nobody asked for",
        )

    def test_a_request_is_consumed_by_the_commit_it_causes(self) -> None:
        """One request keeps one line, not every line after it."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.set_activity("analysis")
            renderer.start()
            renderer.progress(generation(10.0, tokens=11))
            renderer._render_wait_line()
            renderer.keep_progress_line()
            renderer.finish()
            # A second segment nobody asked about.
            renderer.progress(generation(20.0, tokens=22))
            renderer._render_wait_line()
            renderer.finish()
        lines = screen(buffer.getvalue())

        self.assertTrue(any("11 tok" in line for line in lines), "the kept line is missing")
        self.assertFalse(
            any("22 tok" in line for line in lines),
            "the request leaked into a later line",
        )


class PhaseSelectionTests(ProgressLineHarness):
    """Only a finished generation is worth keeping in the transcript."""

    def test_a_prefill_line_is_not_committed_on_the_guided_path(self) -> None:
        """A guided step that never generates must leave no scaffolding row.

        `test_a_prefill_line_is_still_cleared` covers the plain stop; this
        covers the requested one, which is the route the guided fix uses.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer = StreamRenderer(interval=999, interactive=True)
            renderer.set_activity("analysis")
            renderer.start()
            renderer.progress(
                StreamProgress(
                    phase="prefill",
                    current=0,
                    total=100,
                    percent=10,
                    evaluated_current=10,
                    evaluated_total=100,
                )
            )
            renderer._render_wait_line()
            renderer.keep_progress_line()
            renderer.finish()
            print("action: ok", flush=True)
        lines = screen(buffer.getvalue())

        self.assertFalse(
            any("prefill" in line for line in lines),
            "a prefill row was committed to the transcript",
        )


class TimerRestartTests(ProgressLineHarness):
    def test_a_restart_releases_suspended_ticks(self) -> None:
        """Restarting the timer resets line state, this included.

        A settle suspends ticks for the caller's print. If an `event()`
        restarts the timer while they are still suspended, the live line stays
        frozen for the whole next segment. Not reachable from today's single
        settle caller, which passes no event callback -- asserted so that
        wiring one in later cannot silently freeze the line.
        """
        renderer = StreamRenderer(interval=999, interactive=True)
        renderer.set_activity("analysis")
        renderer.start()
        renderer.progress(generation(5.0))
        renderer._render_wait_line()
        renderer.settle_progress_line()
        self.assertTrue(renderer._suspend_ticks, "settling should suspend ticks")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            renderer._restart_timer()

        self.assertFalse(
            renderer._suspend_ticks,
            "a restarted timer must not stay suspended",
        )
