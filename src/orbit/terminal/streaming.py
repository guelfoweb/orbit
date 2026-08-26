from __future__ import annotations

import math
import re
import sys
import threading
import time
from dataclasses import dataclass

from orbit.backend.base import StreamProgress
from orbit.terminal.theme import (
    CYAN,
    DIM,
    RESET,
    dim,
    is_tty,
    sanitize_terminal_text,
    supports_ansi,
)


PREFILL_COMPLETION_LABEL = "waiting for model..."
MARKDOWN_HEADING = "\033[1m" + CYAN
MARKDOWN_BOLD = "\033[1m"
MARKDOWN_BOLD_OFF = "\033[22m"
MARKDOWN_ITALIC = "\033[3m"
MARKDOWN_ITALIC_OFF = "\033[23m"
MARKDOWN_INLINE_CODE = CYAN
MARKDOWN_INLINE_CODE_OFF = "\033[39m"


@dataclass(frozen=True)
class WorkProgress:
    phase: str
    current: int
    total: int
    unit: str


class StreamRenderer:
    def __init__(
        self,
        *,
        interval: float = 1.0,
        prefill_estimate_seconds: float | None = None,
        prefill_estimate_tokens: int | None = None,
        thinking: bool = False,
        render_markdown_mode: str = "plain",
        interactive: bool | None = None,
    ) -> None:
        self.interval = interval
        self._prefill_estimate_seconds = prefill_estimate_seconds
        self._prefill_estimate_tokens = prefill_estimate_tokens
        self._thinking_filter = _ThinkingDisplayFilter() if thinking else None
        self._interactive = is_tty(sys.stdout) if interactive is None else interactive
        self._ansi = supports_ansi(sys.stdout) if interactive is None else interactive
        self._markdown_mode = render_markdown_mode if self._ansi else "plain"
        self._markdown_live = _LiveMarkdownRenderer(enabled=self._markdown_mode == "live")
        self._started = False
        self._first_delta = False
        # Whether model prose with visible content has reached the terminal
        # through this renderer. Rendering state, owned by the terminal: a
        # caller that also holds a final copy of the same prose uses it to
        # avoid printing it twice. Whitespace alone does not count -- it shows
        # the analyst nothing, and the final copy would still be worth having.
        # Never reset by `finish()`, because what was displayed stays displayed.
        self._rendered_visible_text = False
        self._timer_active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._progress: StreamProgress | WorkProgress | None = None
        self._thinking_started = False
        self._thinking_final_started = False
        self._thinking_dim_open = False
        self._phase_label: str | None = None
        self._activity_kind = "model"
        # The last generation line drawn in place, kept so it can be committed
        # to the transcript instead of erased. The wait line is written with a
        # bare `\r` and no newline, so whatever prints next lands on the same
        # row; and clearing it takes the elapsed duration with it. Generation
        # only -- a prefill or "working" tick is scaffolding the analyst does
        # not need once the step is over.
        self._settled_line: str | None = None
        # Held for every write to the wait-line row. The timer thread redraws
        # that row while the main thread may be committing or clearing it;
        # without this a redraw can land between a commit and the print that
        # follows, putting an unterminated line back under the caller's output
        # -- the exact collision this class exists to prevent.
        self._line_lock = threading.Lock()
        # Set while a caller is printing after a settle, so the timer does not
        # draw a new line underneath that output.
        self._suspend_ticks = False

    def start(self) -> None:
        self._started = True
        self._start_time = time.monotonic()
        if not self._interactive:
            return
        self._timer_active = True
        self._thread = threading.Thread(target=self._run_wait_timer, daemon=True)
        self._thread.start()

    def write(self, text: str) -> None:
        """Render one delta of model-authored text.

        Every streamed delta passes through here -- CHAT, an analysis step,
        and `/report` all hand theirs to this method -- so it is where model
        text stops being able to act on the terminal. It is not the only such
        boundary: `format_analysis_step` sanitizes the prose it renders when
        nothing was streamed. Only what is displayed is sanitized; what the
        runtime keeps in history, in the session file, and in evidence is the
        model's original.

        Sanitizing here, before the thinking filter and the markdown renderer,
        is deliberate: those run downstream and emit Orbit's own colour, which
        must not be stripped.
        """
        if not text:
            return
        text = sanitize_terminal_text(text, allow_newlines=True)
        if self._timer_active:
            self._first_delta = True
            self._stop_timer(clear=True)
        if self._thinking_filter is None:
            self._write_visible_text(text)
            return
        for fragment, dimmed in self._thinking_filter.write(text):
            if not fragment:
                continue
            self._print_thinking_fragment(fragment, dimmed=dimmed)

    def event(
        self,
        text: str,
        *,
        restart_timer: bool = True,
        trailing_blank_line: bool = False,
        next_activity: tuple[str, str | None] | None = None,
    ) -> None:
        self._flush_markdown_buffer(interrupted=False)
        self._stop_timer(clear=True)
        print(dim(text), flush=True)
        if trailing_blank_line:
            print(flush=True)
        if next_activity is not None:
            self.set_activity(*next_activity)
        if not restart_timer and next_activity is None:
            return
        self._restart_timer()

    def _restart_timer(self) -> None:
        self._stop.clear()
        self._start_time = time.monotonic()
        self._first_delta = False
        self._progress = None
        if not self._interactive or not self._started:
            return
        self._timer_active = True
        self._thread = threading.Thread(target=self._run_wait_timer, daemon=True)
        self._thread.start()

    def settle_progress_line(self) -> None:
        """Commit the live generation line before printing something else.

        A caller that prints while the timer is still running -- an autonomous
        run rendering each completed step -- would otherwise land on the row
        the wait line is redrawing, because that line carries no newline of
        its own. Calling this first ends the line properly and keeps the
        elapsed duration on screen. Safe to call when nothing is pending.
        """
        with self._line_lock:
            if self._settled_line is None:
                return
            self._commit_wait_line()
            self._progress = None
            # Silence the timer for the caller's print. Serialising the two
            # writers is not enough on its own: a tick that lands after the
            # commit draws a fresh unterminated line, and the caller's output
            # then collides with that one instead. The timer is restarted by
            # the next `progress()` or `event()`, as it is after any pause.
            self._suspend_ticks = True

    def progress(self, update: StreamProgress | WorkProgress) -> None:
        with self._line_lock:
            # Published together: releasing the timer before the new progress
            # is in place would let a tick draw from the previous step's.
            self._progress = update
            self._suspend_ticks = False
        if self._interactive and self._started and not self._first_delta:
            self._render_wait_line()

    @property
    def rendered_visible_text(self) -> bool:
        """Whether the analyst has already seen model prose from this renderer."""
        return self._rendered_visible_text

    def reset_visible_text(self) -> None:
        """Forget that prose was shown, so the next step is judged on its own.

        An autonomous run renders several steps through one renderer. Without
        this, one step that streamed prose would suppress the prose of every
        later step, because the flag would still be set from the earlier one.
        """
        self._rendered_visible_text = False

    def finish(self, *, interrupted: bool = False) -> None:
        if self._thinking_filter is not None:
            for fragment, dimmed in self._thinking_filter.finish():
                if fragment:
                    self._print_thinking_fragment(fragment, dimmed=dimmed)
            if self._thinking_dim_open and self._ansi:
                print(RESET, end="", flush=True)
                self._thinking_dim_open = False
        self._flush_markdown_buffer(interrupted=interrupted)
        if not self._started:
            return
        self._stop_timer(clear=not self._first_delta)

    def _print_thinking_fragment(self, fragment: str, *, dimmed: bool) -> None:
        if dimmed and not self._thinking_started:
            print(dim("Thinking...\n"), end="", flush=True)
            self._thinking_started = True
        if not dimmed and self._thinking_started and not self._thinking_final_started:
            if self._thinking_dim_open and self._ansi:
                print(RESET, end="", flush=True)
                self._thinking_dim_open = False
            if not fragment.startswith("\n"):
                print("\n\n", end="", flush=True)
            self._thinking_final_started = True
        if dimmed:
            if self._ansi and not self._thinking_dim_open:
                print(DIM, end="", flush=True)
                self._thinking_dim_open = True
            print(fragment, end="", flush=True)
            return
        self._write_visible_text(fragment)

    def _write_visible_text(self, text: str) -> None:
        # Set here rather than in `write()`: with a thinking filter active the
        # incoming delta may be entirely hidden reasoning, which the analyst
        # never sees and which must not suppress the final prose. Whitespace is
        # excluded for the same reason -- nothing legible was shown.
        self._rendered_visible_text = self._rendered_visible_text or bool(text.strip())
        try:
            if self._markdown_mode == "live":
                for chunk in self._markdown_live.write(text):
                    print(chunk, end="", flush=True)
                return
            print(text, end="", flush=True)
        except Exception:
            print(text, end="", flush=True)

    def _flush_markdown_buffer(self, *, interrupted: bool) -> None:
        if self._markdown_mode == "live":
            try:
                for chunk in self._markdown_live.finish():
                    print(chunk, end="", flush=True)
            except Exception:
                pass

    def _stop_timer(self, *, clear: bool) -> None:
        if not self._interactive:
            self._timer_active = False
            return
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self._thread = None
        self._timer_active = False
        # Taken after the join above, so the timer thread is already gone and
        # cannot be holding it.
        with self._line_lock:
            # Only a line the caller is finished with gets committed. Once
            # prose has started streaming the progress line has been
            # superseded by the answer itself, and `write()` asks for it to be
            # erased -- committing it there would put a generating row above
            # every CHAT reply.
            if self._settled_line is not None and not self._first_delta:
                self._commit_wait_line()
                return
            # Dropped rather than kept: a line abandoned mid-stream must not
            # surface later as though it had just finished.
            self._settled_line = None
            if clear:
                self._clear_wait_line()

    def _commit_wait_line(self) -> None:
        """Leave the finished generation line standing, set apart above and below.

        The live line is drawn with `\r` and no newline so each tick can
        overwrite the last. That is right while it is still moving, and wrong
        the moment it stops: the next thing printed lands on the same row, and
        clearing it instead loses the elapsed duration the analyst was
        watching accumulate. Committing it writes the newline the redraw loop
        deliberately omitted.

        A blank line above separates it from the evidence or prose it follows;
        one below separates it from the `action:` line that comes next. Both
        are plain newlines, so the separation holds where colour does not --
        a pipe, `NO_COLOR`, `TERM=dumb`.
        """
        line = self._settled_line
        self._settled_line = None
        if line is None:
            return
        # Overwrite the in-place copy rather than adding a second one.
        print("\r" + _pad_to_terminal_width(""), end="", flush=True)
        print(f"\r\n{dim(line)}\n", flush=True)

    def _run_wait_timer(self) -> None:
        while not self._stop.is_set():
            self._render_wait_line()
            self._stop.wait(self.interval)

    def _render_wait_line(self) -> None:
        with self._line_lock:
            if self._suspend_ticks:
                return
            self._render_wait_line_locked()

    def _render_wait_line_locked(self) -> None:
        elapsed = time.monotonic() - self._start_time
        detail = self._progress_activity_detail() or self._phase_label or "working"
        progress_elapsed = getattr(self._progress, "elapsed_seconds", None)
        shown_elapsed = (
            progress_elapsed
            if self._progress is not None and self._progress.phase == "generation" and _valid_metric(progress_elapsed)
            else elapsed
        )
        fitted = _fit_activity_line(self._activity_kind, detail, format_elapsed(shown_elapsed))
        if self._progress is not None and self._progress.phase == "generation":
            self._settled_line = fitted
        print(f"\r{_pad_to_terminal_width(dim(fitted))}", end="", flush=True)

    @staticmethod
    def _clear_wait_line() -> None:
        columns = _terminal_columns()
        print("\r" + (" " * max(1, columns - 1)) + "\r", end="", flush=True)

    def set_prefill_estimate(self, seconds: float | None, tokens: int | None = None) -> None:
        self._prefill_estimate_seconds = seconds
        self._prefill_estimate_tokens = tokens

    def set_phase_label(self, label: str | None) -> None:
        self._activity_kind = "model"
        self._phase_label = label.strip() if label else None
        self._progress = None

    def set_activity(self, kind: str, label: str | None = None) -> None:
        self._activity_kind = kind.strip() or "model"
        self._phase_label = label.strip() if label else None

    def set_final_output_mode(self, enabled: bool) -> None:
        if self._thinking_filter is None or not enabled:
            return
        self._thinking_filter.start_final_output()

    def _working_status(self, elapsed: float) -> str:
        parts = [format_elapsed(elapsed)]
        detail = self._progress_activity_detail()
        if detail is not None:
            parts.append(detail)
            return ", ".join(parts)
        if self._prefill_estimate_seconds and self._prefill_estimate_seconds >= 1:
            progress = max(1, int((elapsed / self._prefill_estimate_seconds) * 100))
            if self._prefill_estimate_tokens and self._prefill_estimate_tokens > 0:
                current = min(self._prefill_estimate_tokens, max(1, int((progress / 100) * self._prefill_estimate_tokens)))
                label = (
                    PREFILL_COMPLETION_LABEL
                    if current >= self._prefill_estimate_tokens
                    else f"prefill estimate ~{current}/{self._prefill_estimate_tokens} tk"
                )
            else:
                label = PREFILL_COMPLETION_LABEL if progress >= 95 else f"prefill estimate ~{progress}%"
            parts.append(label)
        return ", ".join(parts)

    def _working_phase_prefix(self) -> str:
        detail = self._working_phase_detail()
        if detail:
            return f" [{detail}]"
        return ""

    def _working_phase_detail(self) -> str | None:
        if self._progress is not None:
            prefix = f"{self._phase_label} " if self._phase_label else ""
            if self._progress.phase == "prefill":
                return f"{prefix}prefill"
            if self._progress.phase == "generation":
                return f"{prefix}generation"
            return f"{prefix}{self._progress.phase}"
        if self._prefill_estimate_seconds and self._prefill_estimate_seconds >= 1:
            return "prefill estimate"
        return None

    def _progress_activity_detail(self) -> str | None:
        if self._progress is None:
            return None
        progress = self._progress
        if isinstance(progress, WorkProgress):
            current = _valid_count(progress.current)
            total = _valid_count(progress.total)
            if current is None or total is None or total <= 0 or current > total:
                return progress.phase
            quantity = _format_progress_quantity(current, total, progress.unit)
            return f"{progress.phase} · {_percent(current, total)}% · {quantity}"
        if progress.phase == "prefill":
            current = _valid_count(progress.evaluated_current)
            total = _valid_count(progress.evaluated_total)
            if current is None or total is None or total <= 0 or current > total:
                return "prefill"
            parts = ["prefill", f"{_percent(current, total)}%", f"{current}/{total} eval"]
            cached = _valid_count(progress.cached_tokens)
            if cached is not None:
                parts.append(f"{cached} cached")
            rate = _format_rate(progress.tokens_per_second)
            if rate is not None:
                parts.append(f"{rate} tok/s")
            return " · ".join(parts)
        if progress.phase == "generation":
            current = _valid_count(progress.current)
            parts = ["generating", f"{current} tok"] if current is not None else ["generating"]
            rate = _format_rate(progress.tokens_per_second)
            if rate is not None:
                parts.append(f"{rate} tok/s")
            return " · ".join(parts)
        return progress.phase


def _valid_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _valid_metric(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0


def _format_rate(value: object) -> str | None:
    return f"{float(value):.1f}" if _valid_metric(value) else None


def _percent(current: int, total: int) -> int:
    return min(100, max(0, int(round((current / total) * 100)))) if total > 0 else 0


def _format_progress_quantity(current: int, total: int, unit: str | None) -> str:
    if unit == "bytes":
        divisor, label = (1024 * 1024, "MB") if total >= 1024 * 1024 else (1024, "KB")
        return f"{current / divisor:.1f}/{total / divisor:.1f} {label}"
    if unit:
        return f"{current}/{total} {unit}"
    return f"{current}/{total}"


def _terminal_columns() -> int:
    try:
        return max(20, int(__import__("shutil").get_terminal_size((80, 20)).columns))
    except Exception:
        return 80


def _fit_activity_line(kind: str, detail: str, elapsed: str) -> str:
    columns = _terminal_columns()
    prefix = f"{kind} · "
    suffix = f" · {elapsed}"
    available = max(1, columns - len(prefix) - len(suffix) - 1)
    if len(detail) > available:
        detail = detail[: max(0, available - 1)].rstrip() + "…"
    return f"{prefix}{detail}{suffix}"


def _pad_to_terminal_width(text: str) -> str:
    columns = _terminal_columns()
    return text + (" " * max(0, columns - _visible_len(text) - 1))


def _visible_len(text: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*m", "", text))


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, remaining = divmod(total, 60)
    return f"{minutes}m {remaining}s"


class _ThinkingDisplayFilter:
    _THOUGHT_START = "<|channel>thought\n"
    _CHANNEL_END = "<channel|>"
    _FINAL_MARKERS = (
        "**final answer:**",
        "final answer:",
        "the final answer is:",
        "the final answer:",
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._in_final = False
        self._in_channel_thought = False
        self._thought_text_parts: list[str] = []
        self._saw_final_output = False

    def write(self, text: str) -> list[tuple[str, bool]]:
        if not text:
            return []
        self._buffer += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, bool]]:
        return self._drain(final=True)

    def start_final_output(self) -> None:
        self._in_channel_thought = False
        self._in_final = True

    def _drain(self, *, final: bool) -> list[tuple[str, bool]]:
        emitted: list[tuple[str, bool]] = []
        while self._buffer:
            if self._in_channel_thought:
                end = self._buffer.find(self._CHANNEL_END)
                if end < 0:
                    emit_len = len(self._buffer) if final else self._safe_emit_length_with_channel()
                    if emit_len <= 0:
                        break
                    text = _strip_channel_markup(self._buffer[:emit_len])
                    if text:
                        self._remember(text, dimmed=True)
                        emitted.append((text, True))
                    self._buffer = self._buffer[emit_len:]
                    continue
                text = _strip_channel_markup(self._buffer[:end])
                if text:
                    self._remember(text, dimmed=True)
                    emitted.append((text, True))
                self._buffer = self._buffer[end + len(self._CHANNEL_END) :]
                self._in_channel_thought = False
                self._in_final = True
                continue
            thought_idx = self._buffer.find(self._THOUGHT_START)
            if thought_idx >= 0:
                if thought_idx > 0:
                    text = _strip_channel_markup(self._buffer[:thought_idx])
                    if text:
                        emitted.append((text, True))
                self._buffer = self._buffer[thought_idx + len(self._THOUGHT_START) :]
                self._in_channel_thought = True
                continue
            if self._in_final:
                emit_len = len(self._buffer) if final else self._safe_emit_length()
                if emit_len <= 0:
                    break
                text = _strip_channel_markup(self._buffer[:emit_len])
                if text:
                    self._remember(text, dimmed=False)
                    emitted.append((text, False))
                self._buffer = self._buffer[emit_len:]
                continue
            match = _find_final_marker(self._buffer)
            if match is None:
                emit_len = len(self._buffer) if final else min(
                    self._safe_emit_length(),
                    self._safe_emit_length_with_thought_start(),
                )
                if emit_len <= 0:
                    break
                text = _strip_channel_markup(self._buffer[:emit_len])
                if text:
                    self._remember(text, dimmed=True)
                    emitted.append((text, True))
                self._buffer = self._buffer[emit_len:]
                continue
            start, end = match
            if start > 0:
                text = _strip_channel_markup(self._buffer[:start])
                if text:
                    self._remember(text, dimmed=True)
                    emitted.append((text, True))
            marker = _strip_channel_markup(self._buffer[start:end])
            if marker:
                self._remember(marker, dimmed=False)
                emitted.append((marker, False))
            self._buffer = self._buffer[end:]
            self._in_final = True
        return emitted

    def _remember(self, text: str, *, dimmed: bool) -> None:
        if dimmed:
            self._thought_text_parts.append(text)
            return
        self._saw_final_output = True

    def _safe_emit_length(self) -> int:
        keep = 0
        for marker in self._FINAL_MARKERS:
            max_prefix = min(len(marker) - 1, len(self._buffer))
            lowered = self._buffer.lower()
            for size in range(max_prefix, 0, -1):
                if marker.startswith(lowered[-size:]):
                    keep = max(keep, size)
                    break
        return max(0, len(self._buffer) - keep)

    def _safe_emit_length_with_thought_start(self) -> int:
        keep = self._max_suffix_prefix_overlap(self._buffer, self._THOUGHT_START)
        return max(0, len(self._buffer) - keep)

    def _safe_emit_length_with_channel(self) -> int:
        keep = self._max_suffix_prefix_overlap(self._buffer, self._CHANNEL_END)
        return max(0, len(self._buffer) - keep)

    @staticmethod
    def _max_suffix_prefix_overlap(text: str, marker: str) -> int:
        max_prefix = min(len(marker) - 1, len(text))
        for size in range(max_prefix, 0, -1):
            if marker.startswith(text[-size:]):
                return size
        return 0


def _find_final_marker(text: str) -> tuple[int, int] | None:
    lowered = text.lower()
    best: tuple[int, int] | None = None
    for marker in _ThinkingDisplayFilter._FINAL_MARKERS:
        idx = lowered.find(marker)
        if idx < 0:
            continue
        candidate = (idx, idx + len(marker))
        if best is None or idx < best[0]:
            best = candidate
    return best


def _strip_channel_markup(text: str) -> str:
    return (
        text.replace("<|channel>thought\n", "")
        .replace("<|channel>final\n", "")
        .replace("<channel|>", "")
    )


class _LiveMarkdownRenderer:
    _INLINE_BUFFER_LIMIT = 160

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._inside_code_fence = False
        self._start_of_line = True
        self._prefix = ""
        self._line_style: str | None = None
        self._style_open = False
        self._inline_buffer = ""
        self._discard_fence_line = False
        self._last_visible_char = ""
        self._trailing_backslashes = 0

    def write(self, text: str) -> list[str]:
        if not text:
            return []
        if not self.enabled:
            return [text]
        emitted: list[str] = []
        for ch in text:
            emitted.extend(self._write_char(ch))
        return emitted

    def finish(self) -> list[str]:
        if not self.enabled:
            return []
        tail: list[str] = []
        if self._prefix:
            tail.append(self._emit(self._prefix))
            self._prefix = ""
        if self._inline_buffer:
            tail.append(self._emit(self._inline_buffer))
            self._inline_buffer = ""
        if self._style_open:
            tail.append(RESET)
            self._style_open = False
        self._last_visible_char = ""
        self._trailing_backslashes = 0
        return tail

    def _write_char(self, ch: str) -> list[str]:
        if self._discard_fence_line:
            if ch == "\n":
                self._discard_fence_line = False
                self._start_of_line = True
            return []
        if self._start_of_line:
            return self._write_line_start(ch)
        if ch == "\n":
            chunks = self._flush_inline_buffer()
            chunk = self._emit(ch)
            if self._style_open:
                chunk += RESET
                self._style_open = False
            self._start_of_line = True
            self._line_style = None
            chunks.append(chunk)
            return chunks
        return self._write_inline_char(ch)

    def _write_line_start(self, ch: str) -> list[str]:
        self._prefix += ch
        if ch == "\n":
            chunk = self._emit(self._prefix)
            self._prefix = ""
            self._line_style = None
            self._start_of_line = True
            return [chunk]

        decision = self._line_start_decision()
        if decision is None:
            return []
        style, keep_start, visible_prefix = decision
        prefix = self._prefix
        self._prefix = ""
        self._line_style = style
        self._start_of_line = keep_start
        if visible_prefix is None:
            visible_prefix = prefix
        if visible_prefix.startswith(("**", "`")):
            return self._write_inline_text(visible_prefix)
        if not visible_prefix:
            return []
        return [self._emit(visible_prefix)]

    def _line_start_decision(self) -> tuple[str | None, bool, str | None] | None:
        prefix = self._prefix
        if prefix.startswith("```"):
            self._inside_code_fence = not self._inside_code_fence
            self._discard_fence_line = True
            return DIM if self._inside_code_fence else None, False, ""
        if prefix in {"#", "##", "###", "-", "*", "`", "``"}:
            return None
        if prefix.startswith("**") and len(prefix) >= 3:
            return None, False, prefix
        if prefix.startswith("*") and not prefix.startswith("* "):
            return None, False, prefix
        if prefix.startswith("-") and not prefix.startswith("- "):
            return None, False, prefix
        if re.fullmatch(r"\d+", prefix) or re.fullmatch(r"\d+\.", prefix):
            return None
        if prefix.startswith(("### ", "## ", "# ")):
            return MARKDOWN_HEADING, False, ""
        if prefix.startswith(("- ", "* ")) or re.fullmatch(r"\d+\. ", prefix):
            return CYAN, False, prefix
        if self._inside_code_fence and not prefix.startswith("```"):
            return DIM, False, prefix
        if len(prefix) >= 4 or prefix[0] not in "#-*`0123456789":
            return None, False, prefix
        if prefix.startswith("-") or prefix.startswith("*"):
            return None, False, prefix
        return None, False, prefix

    def _emit(self, text: str) -> str:
        if not text:
            return ""
        self._remember_visible_text(text)
        if not self._line_style:
            return text
        if self._style_open:
            return text
        self._style_open = True
        return f"{self._line_style}{text}"

    def _write_inline_char(self, ch: str) -> list[str]:
        if self._inside_code_fence:
            return [self._emit(ch)]
        self._inline_buffer += ch
        return self._drain_inline_buffer()

    def _write_inline_text(self, text: str) -> list[str]:
        emitted: list[str] = []
        for ch in text:
            emitted.extend(self._write_inline_char(ch))
        return emitted

    def _drain_inline_buffer(self) -> list[str]:
        emitted: list[str] = []
        while self._inline_buffer:
            start, marker = self._next_inline_marker()
            if start < 0:
                if self._inline_buffer.endswith(("*", "_")):
                    plain = self._inline_buffer[:-1]
                    if plain:
                        emitted.append(self._emit(plain))
                    self._inline_buffer = self._inline_buffer[-1]
                    break
                emitted.append(self._emit(self._inline_buffer))
                self._inline_buffer = ""
                break
            if start > 0:
                emitted.append(self._emit(self._inline_buffer[:start]))
                self._inline_buffer = self._inline_buffer[start:]
                continue
            end = self._find_closing_inline_marker(marker)
            if end < 0:
                if len(self._inline_buffer) > self._INLINE_BUFFER_LIMIT:
                    emitted.append(self._emit(self._inline_buffer))
                    self._inline_buffer = ""
                break
            content = self._inline_buffer[len(marker) : end]
            if content:
                emitted.append(self._emit(self._inline_style(content, marker=marker)))
            self._inline_buffer = self._inline_buffer[end + len(marker) :]
        return emitted

    def _flush_inline_buffer(self) -> list[str]:
        if not self._inline_buffer:
            return []
        chunk = self._emit(self._inline_buffer)
        self._inline_buffer = ""
        return [chunk]

    def _next_inline_marker(self) -> tuple[int, str]:
        candidates: list[tuple[int, str]] = []
        for marker in ("**", "*", "_", "`"):
            idx = self._find_inline_marker(marker)
            if idx >= 0:
                candidates.append((idx, marker))
        if not candidates:
            return -1, ""
        return min(candidates, key=lambda item: (item[0], -len(item[1])))

    def _find_inline_marker(self, marker: str) -> int:
        start = 0
        while True:
            idx = self._inline_buffer.find(marker, start)
            if idx < 0:
                return -1
            next_idx = idx + len(marker)
            if marker == "`" and self._backtick_is_escaped(idx):
                start = idx + 1
                continue
            if marker in {"*", "_"} and next_idx < len(self._inline_buffer) and self._inline_buffer[next_idx].isspace():
                start = idx + 1
                continue
            if marker == "_" and self._marker_inside_word(idx=idx, marker=marker):
                start = idx + 1
                continue
            return idx

    def _inline_style(self, text: str, *, marker: str) -> str:
        if marker == "**":
            open_style = MARKDOWN_BOLD
            close_style = MARKDOWN_BOLD_OFF
        elif marker == "`":
            open_style = MARKDOWN_INLINE_CODE
            close_style = MARKDOWN_INLINE_CODE_OFF
        else:
            open_style = MARKDOWN_ITALIC
            close_style = MARKDOWN_ITALIC_OFF
        if self._line_style and self._style_open:
            close_style = RESET + self._line_style
        return f"{open_style}{text}{close_style}"

    def _find_closing_inline_marker(self, marker: str) -> int:
        start = len(marker)
        while True:
            idx = self._inline_buffer.find(marker, start)
            if idx < 0:
                return -1
            if marker != "`" or not self._backtick_is_escaped(idx):
                return idx
            start = idx + 1

    def _backtick_is_escaped(self, idx: int) -> bool:
        if idx == 0:
            return self._trailing_backslashes % 2 == 1
        backslashes = 0
        cursor = idx - 1
        while cursor >= 0 and self._inline_buffer[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if cursor < 0:
            backslashes += self._trailing_backslashes
        return backslashes % 2 == 1

    def _marker_inside_word(self, *, idx: int, marker: str) -> bool:
        previous = self._inline_buffer[idx - 1] if idx > 0 else self._last_visible_char
        next_idx = idx + len(marker)
        next_char = self._inline_buffer[next_idx] if next_idx < len(self._inline_buffer) else ""
        return bool(previous and previous.isalnum() and next_char and next_char.isalnum())

    def _remember_visible_text(self, text: str) -> None:
        visible = re.sub(r"\x1b\[[0-9;]*m", "", text)
        if visible:
            self._last_visible_char = visible[-1]
            trailing = len(visible) - len(visible.rstrip("\\"))
            self._trailing_backslashes = (
                self._trailing_backslashes + trailing if trailing == len(visible) else trailing
            )
