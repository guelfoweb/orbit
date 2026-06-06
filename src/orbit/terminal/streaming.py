from __future__ import annotations

import threading
import time

from orbit.terminal.theme import dim


SPINNER_FRAMES = ("◐", "◓", "◑", "◒")


class StreamRenderer:
    def __init__(self, *, interval: float = 1.0) -> None:
        self.interval = interval
        self._started = False
        self._first_delta = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._frame_index = 0

    def start(self) -> None:
        self._started = True
        self._start_time = time.monotonic()
        self._frame_index = 0
        self._thread = threading.Thread(target=self._run_wait_timer, daemon=True)
        self._thread.start()

    def write(self, text: str) -> None:
        if not text:
            return
        if not self._first_delta:
            self._first_delta = True
            self._stop.set()
            self._clear_wait_line()
        print(text, end="", flush=True)

    def event(self, text: str, *, restart_timer: bool = True, trailing_blank_line: bool = False) -> None:
        self._stop.set()
        self._clear_wait_line()
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
        self._thread = threading.Thread(target=self._run_wait_timer, daemon=True)
        self._thread.start()

    def finish(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if not self._first_delta:
            self._clear_wait_line()

    def _run_wait_timer(self) -> None:
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._start_time
            frame = SPINNER_FRAMES[self._frame_index % len(SPINNER_FRAMES)]
            self._frame_index += 1
            print(f"\r{dim(f'{frame} Working ({format_elapsed(elapsed)} - Ctrl+C to interrupt)')}", end="", flush=True)
            self._stop.wait(self.interval)

    @staticmethod
    def _clear_wait_line() -> None:
        columns = _terminal_columns()
        print("\r" + (" " * max(1, columns - 1)) + "\r", end="", flush=True)


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
