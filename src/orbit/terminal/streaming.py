from __future__ import annotations

import threading
import time

from orbit.terminal.theme import dim


SPINNER_FRAMES = ("◐", "◓", "◑", "◒")


class StreamRenderer:
    def __init__(
        self,
        *,
        interval: float = 1.0,
        prefill_estimate_seconds: float | None = None,
        prefill_estimate_tokens: int | None = None,
    ) -> None:
        self.interval = interval
        self._prefill_estimate_seconds = prefill_estimate_seconds
        self._prefill_estimate_tokens = prefill_estimate_tokens
        self._started = False
        self._first_delta = False
        self._timer_active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._frame_index = 0

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
        print(text, end="", flush=True)

    def event(self, text: str, *, restart_timer: bool = True, trailing_blank_line: bool = False) -> None:
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
        self._timer_active = True
        self._thread = threading.Thread(target=self._run_wait_timer, daemon=True)
        self._thread.start()

    def finish(self) -> None:
        if not self._started:
            return
        self._stop_timer(clear=not self._first_delta)

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
            elapsed = time.monotonic() - self._start_time
            frame = SPINNER_FRAMES[self._frame_index % len(SPINNER_FRAMES)]
            self._frame_index += 1
            print(f"\r{dim(f'{frame} Working ({self._working_status(elapsed)} - Ctrl+C to interrupt)')}", end="", flush=True)
            self._stop.wait(self.interval)

    @staticmethod
    def _clear_wait_line() -> None:
        columns = _terminal_columns()
        print("\r" + (" " * max(1, columns - 1)) + "\r", end="", flush=True)

    def set_prefill_estimate(self, seconds: float | None, tokens: int | None = None) -> None:
        self._prefill_estimate_seconds = seconds
        self._prefill_estimate_tokens = tokens

    def _working_status(self, elapsed: float) -> str:
        parts = [format_elapsed(elapsed)]
        if self._prefill_estimate_seconds and self._prefill_estimate_seconds >= 1:
            progress = max(1, int((elapsed / self._prefill_estimate_seconds) * 100))
            if self._prefill_estimate_tokens and self._prefill_estimate_tokens > 0:
                current = min(self._prefill_estimate_tokens, max(1, int((progress / 100) * self._prefill_estimate_tokens)))
                label = (
                    "processing prompt"
                    if current >= self._prefill_estimate_tokens
                    else f"pf ~{current}/{self._prefill_estimate_tokens} tk"
                )
            else:
                label = "processing prompt" if progress >= 95 else f"pf ~{progress}%"
            parts.append(label)
        return ", ".join(parts)


def _terminal_columns() -> int:
    try:
        return max(20, int(__import__("shutil").get_terminal_size((80, 20)).columns))
    except Exception:
        return 80


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, remaining = divmod(total, 60)
    return f"{minutes}m {remaining}s"
