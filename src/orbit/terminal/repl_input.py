from __future__ import annotations

import re
import select
import sys
from collections.abc import Callable
from shutil import get_terminal_size

from orbit.terminal.prompt_preview import compact_prompt_preview, is_long_text_prompt
from orbit.terminal.theme import CYAN, RESET, accent, yellow_dim


PASTE_BADGE_PATTERN = re.compile(r"(\[text \d+ chars #[0-9a-f]{8}\])$")
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"
READLINE_IGNORE_START = "\001"
READLINE_IGNORE_END = "\002"


def read_prompt_input(*, redisplay: bool = False, label: str = "") -> str:
    clear_redisplay_hook = _install_redisplay_hook() if redisplay else None
    try:
        first_line = input(input_prompt(label))
    finally:
        if clear_redisplay_hook is not None:
            clear_redisplay_hook()
    return read_available_paste_tail(first_line)


def _install_redisplay_hook() -> Callable[[], None] | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    readline = sys.modules.get("readline")
    set_pre_input_hook = getattr(readline, "set_pre_input_hook", None)
    redisplay = getattr(readline, "redisplay", None)
    if not callable(set_pre_input_hook) or not callable(redisplay):
        return None

    def clear_hook() -> None:
        try:
            set_pre_input_hook(None)
        except Exception:
            pass

    def redisplay_once() -> None:
        clear_hook()
        try:
            redisplay()
        except Exception:
            pass

    try:
        set_pre_input_hook(redisplay_once)
    except Exception:
        return None
    return clear_hook


def input_prompt(label: str = "") -> str:
    """The marker the analyst types after.

    `label` names the runtime that owns the next line. It is display only:
    the caller passes the mode it already holds, so nothing here decides or
    stores what mode the session is in, and the string never reaches a model.
    """
    if not sys.stdout.isatty():
        return f"{label}> "
    return (
        f"{READLINE_IGNORE_START}{CYAN}{READLINE_IGNORE_END}{label}> "
        f"{READLINE_IGNORE_START}{RESET}{READLINE_IGNORE_END}"
    )


def replace_input_echo(prompt: str, label: str = "") -> None:
    if not should_replace_input_echo(prompt):
        return
    if not sys.stdout.isatty():
        return
    preview = compact_prompt_preview(prompt, multiline=True)
    rendered = colorize_user_prompt(f"{label}> {preview}")
    columns = max(20, get_terminal_size((80, 20)).columns)
    # The marker occupies columns on the first row, so the label has to be
    # counted or a wrapped line is erased one row short.
    visual_rows = visual_row_count(f"{label}> {prompt}", columns=columns)
    print(f"\x1b[{visual_rows}F\x1b[J{rendered}", flush=True)


def clear_input_echo(prompt: str, label: str = "") -> None:
    if not sys.stdout.isatty():
        return
    columns = max(20, get_terminal_size((80, 20)).columns)
    visual_rows = visual_row_count(f"{label}> {prompt}", columns=columns)
    print(f"\x1b[{visual_rows}F\x1b[J", end="", flush=True)


def should_replace_input_echo(prompt: str) -> bool:
    return is_long_text_prompt(prompt) or "\n" in prompt


def colorize_paste_preview(preview: str) -> str:
    return PASTE_BADGE_PATTERN.sub(lambda match: yellow_dim(match.group(1)), preview)


def colorize_user_prompt(text: str) -> str:
    match = PASTE_BADGE_PATTERN.search(text)
    if not match:
        return accent(text)
    return accent(text[: match.start(1)]) + yellow_dim(match.group(1))


def read_available_paste_tail(
    first_line: str,
    *,
    timeout: float = 0.04,
    idle_polls: int = 3,
    require_tty: bool = True,
) -> str:
    if require_tty and not sys.stdin.isatty():
        return strip_bracketed_paste_markers(first_line)
    try:
        fileno = sys.stdin.fileno()
    except (AttributeError, OSError):
        return first_line
    lines = [first_line]
    idle_count = 0
    while True:
        try:
            ready, _, _ = select.select([fileno], [], [], timeout)
        except (OSError, ValueError):
            break
        if not ready:
            idle_count += 1
            if idle_count >= idle_polls:
                break
            continue
        idle_count = 0
        line = sys.stdin.readline()
        if line == "":
            break
        lines.append(line.rstrip("\n"))
    return strip_bracketed_paste_markers("\n".join(lines))


def strip_bracketed_paste_markers(prompt: str) -> str:
    return prompt.replace(BRACKETED_PASTE_START, "").replace(BRACKETED_PASTE_END, "")


def visual_row_count(text: str, *, columns: int) -> int:
    return sum(max(1, (len(line) // columns) + 1) for line in text.split("\n"))
