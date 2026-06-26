from __future__ import annotations

import re
import threading
import time

from orbit.backend.base import StreamProgress
from orbit.terminal.theme import CYAN, DIM, RESET, dim


SPINNER_FRAMES = ("◐", "◓", "◑", "◒")
PREFILL_COMPLETION_LABEL = "waiting for model..."
MARKDOWN_HEADING = "\033[1m" + CYAN
MARKDOWN_BOLD = "\033[1m"
MARKDOWN_BOLD_OFF = "\033[22m"
MARKDOWN_ITALIC = "\033[3m"
MARKDOWN_ITALIC_OFF = "\033[23m"


class StreamRenderer:
    def __init__(
        self,
        *,
        interval: float = 1.0,
        prefill_estimate_seconds: float | None = None,
        prefill_estimate_tokens: int | None = None,
        thinking: bool = False,
        render_markdown_mode: str = "plain",
    ) -> None:
        self.interval = interval
        self._prefill_estimate_seconds = prefill_estimate_seconds
        self._prefill_estimate_tokens = prefill_estimate_tokens
        self._thinking_filter = _ThinkingDisplayFilter() if thinking else None
        self._markdown_mode = render_markdown_mode
        self._markdown_live = _LiveMarkdownRenderer(enabled=render_markdown_mode == "live")
        self._started = False
        self._first_delta = False
        self._timer_active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._frame_index = 0
        self._progress: StreamProgress | None = None
        self._generation_completed = 0
        self._generation_budget_completed = 0
        self._thinking_started = False
        self._thinking_final_started = False
        self._thinking_dim_open = False
        self._phase_label: str | None = None

    def start(self) -> None:
        self._started = True
        self._start_time = time.monotonic()
        self._frame_index = 0
        self._timer_active = True
        self._thread = threading.Thread(target=self._run_wait_timer, daemon=True)
        self._thread.start()

    def write(self, text: str) -> None:
        if not text:
            return
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

    def event(self, text: str, *, restart_timer: bool = True, trailing_blank_line: bool = False) -> None:
        self._flush_markdown_buffer(interrupted=False)
        self._stop_timer(clear=True)
        print(dim(text), flush=True)
        if trailing_blank_line:
            print(flush=True)
        if not restart_timer:
            return
        self._restart_timer()

    def _restart_timer(self) -> None:
        self._stop.clear()
        self._start_time = time.monotonic()
        self._frame_index = 0
        self._first_delta = False
        self._progress = None
        self._generation_completed = 0
        self._generation_budget_completed = 0
        self._timer_active = True
        self._thread = threading.Thread(target=self._run_wait_timer, daemon=True)
        self._thread.start()

    def progress(self, update: StreamProgress) -> None:
        self._progress = self._normalize_progress(update)
        if self._started and not self._first_delta:
            self._render_wait_line()

    def finish(self, *, interrupted: bool = False) -> None:
        if self._thinking_filter is not None:
            for fragment, dimmed in self._thinking_filter.finish():
                if fragment:
                    self._print_thinking_fragment(fragment, dimmed=dimmed)
            if self._thinking_dim_open:
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
            if self._thinking_dim_open:
                print(RESET, end="", flush=True)
                self._thinking_dim_open = False
            if not fragment.startswith("\n"):
                print("\n\n", end="", flush=True)
            self._thinking_final_started = True
        if dimmed:
            if not self._thinking_dim_open:
                print(DIM, end="", flush=True)
                self._thinking_dim_open = True
            print(fragment, end="", flush=True)
            return
        self._write_visible_text(fragment)

    def _write_visible_text(self, text: str) -> None:
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
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self._thread = None
        self._timer_active = False
        if clear:
            self._clear_wait_line()

    def _run_wait_timer(self) -> None:
        while not self._stop.is_set():
            self._render_wait_line()
            self._stop.wait(self.interval)

    def _render_wait_line(self) -> None:
        elapsed = time.monotonic() - self._start_time
        frame = SPINNER_FRAMES[self._frame_index % len(SPINNER_FRAMES)]
        self._frame_index += 1
        line = dim(f"{frame} Working{self._working_phase_prefix()} ({self._working_status(elapsed)} - Ctrl+C to interrupt)")
        print(f"\r{_pad_to_terminal_width(line)}", end="", flush=True)

    @staticmethod
    def _clear_wait_line() -> None:
        columns = _terminal_columns()
        print("\r" + (" " * max(1, columns - 1)) + "\r", end="", flush=True)

    def set_prefill_estimate(self, seconds: float | None, tokens: int | None = None) -> None:
        self._prefill_estimate_seconds = seconds
        self._prefill_estimate_tokens = tokens

    def set_phase_label(self, label: str | None) -> None:
        self._phase_label = label.strip() if label else None

    def set_final_output_mode(self, enabled: bool) -> None:
        if self._thinking_filter is None or not enabled:
            return
        self._thinking_filter.start_final_output()

    def _normalize_progress(self, update: StreamProgress) -> StreamProgress:
        if update.phase != "generation":
            return update
        if self._progress is not None and self._progress.phase == "generation":
            previous_current = self._progress.current - self._generation_completed
            previous_total = self._progress.total - self._generation_budget_completed
            if update.current < previous_current:
                self._generation_completed += max(0, previous_current)
                self._generation_budget_completed += max(0, previous_total)
        current = self._generation_completed + update.current
        total = self._generation_budget_completed + update.total
        percent = int((current / total) * 100) if total > 0 else 0
        return StreamProgress(phase=update.phase, current=current, total=total, percent=percent)

    def _working_status(self, elapsed: float) -> str:
        parts = [format_elapsed(elapsed)]
        if self._progress is not None:
            if self._progress.phase == "prefill":
                parts.append(f"{self._progress.current}/{self._progress.total} tk ({self._progress.percent}%)")
            elif self._progress.phase == "generation":
                parts.append(f"{self._progress.current}/{self._progress.total} tk ({self._progress.percent}%)")
            else:
                parts.append(f"{self._progress.current}/{self._progress.total} ({self._progress.percent}%)")
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
            if self._progress.phase == "prefill":
                return "prefill"
            if self._progress.phase == "generation":
                return "generation"
            return self._progress.phase
        if self._prefill_estimate_seconds and self._prefill_estimate_seconds >= 1:
            return "prefill estimate"
        return None


def _terminal_columns() -> int:
    try:
        return max(20, int(__import__("shutil").get_terminal_size((80, 20)).columns))
    except Exception:
        return 80


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
        if visible_prefix.startswith("**"):
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
            end = self._inline_buffer.find(marker, len(marker))
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
        for marker in ("**", "*", "_"):
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
            if marker != "**" and next_idx < len(self._inline_buffer) and self._inline_buffer[next_idx].isspace():
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
        else:
            open_style = MARKDOWN_ITALIC
            close_style = MARKDOWN_ITALIC_OFF
        if self._line_style and self._style_open:
            close_style = RESET + self._line_style
        return f"{open_style}{text}{close_style}"

    def _marker_inside_word(self, *, idx: int, marker: str) -> bool:
        previous = self._inline_buffer[idx - 1] if idx > 0 else self._last_visible_char
        next_idx = idx + len(marker)
        next_char = self._inline_buffer[next_idx] if next_idx < len(self._inline_buffer) else ""
        return bool(previous and previous.isalnum() and next_char and next_char.isalnum())

    def _remember_visible_text(self, text: str) -> None:
        visible = re.sub(r"\x1b\[[0-9;]*m", "", text)
        if visible:
            self._last_visible_char = visible[-1]
