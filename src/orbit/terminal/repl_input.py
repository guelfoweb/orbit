from __future__ import annotations

import re
import select
import sys
from collections.abc import Callable
from shutil import get_terminal_size

from orbit.terminal.prompt_preview import compact_prompt_preview, is_long_text_prompt
from orbit.terminal.theme import CYAN, GREEN, RED, RESET, YELLOW, accent, yellow_dim


PASTE_BADGE_PATTERN = re.compile(r"(\[text \d+ chars #[0-9a-f]{8}\])$")
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"
READLINE_IGNORE_START = "\001"
READLINE_IGNORE_END = "\002"


def read_prompt_input(
    *, redisplay: bool = False, label: str = "", autonomous: bool = False
) -> str:
    clear_redisplay_hook = _install_redisplay_hook() if redisplay else None
    try:
        first_line = input(input_prompt(label, autonomous=autonomous))
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


def prompt_marker(label: str = "", *, autonomous: bool = False) -> str:
    """The prompt's visible text, with no colour and no readline markers.

    The single definition of what the analyst sees, so the rendered prompt and
    the echo-clearing arithmetic cannot disagree about its width. They did:
    the prompt grew an `[auto:...]` segment while the row count was still
    computed from the bare mode, which erased one row too few whenever the
    difference pushed a typed line across the terminal edge.
    """
    return f"{label} [auto:{'on' if autonomous else 'off'}]> "


def input_prompt(label: str = "", *, autonomous: bool = False) -> str:
    """The marker the analyst types after, with the autonomy state beside it.

    `label` names the runtime that owns the next line and `autonomous` says
    how it will advance. Both are display only: the caller passes state it
    already holds, so nothing here decides or stores anything, and the string
    never reaches a model.

    Autonomy is shown because it is otherwise invisible. `/autonomous on`
    deliberately changes no mode, so without this the analyst types it, sees
    the prompt unchanged, and cannot tell whether it took effect.

    Every escape this emits is wrapped in readline's ignore markers, which is
    why the colouring lives here rather than in the label: readline counts the
    prompt's width to place the cursor, and an unwrapped escape makes it
    mis-count on every edited line. A caller that coloured its own label would
    put those bytes outside the markers.
    """
    state = "on" if autonomous else "off"
    if not sys.stdout.isatty():
        return prompt_marker(label, autonomous=autonomous)
    # ANALYSIS is amber, not red: the session is doing normal work under
    # different rules, and red belongs to failures. Chosen from the label the
    # caller passed, so it cannot disagree with the text.
    colour = YELLOW if label == "analysis" else CYAN
    state_colour = GREEN if autonomous else RED
    return (
        f"{READLINE_IGNORE_START}{colour}{READLINE_IGNORE_END}{label} [auto:"
        f"{READLINE_IGNORE_START}{state_colour}{READLINE_IGNORE_END}{state}"
        f"{READLINE_IGNORE_START}{colour}{READLINE_IGNORE_END}]> "
        f"{READLINE_IGNORE_START}{RESET}{READLINE_IGNORE_END}"
    )


def replace_input_echo(prompt: str, label: str = "", *, autonomous: bool = False) -> None:
    if not should_replace_input_echo(prompt):
        return
    if not sys.stdout.isatty():
        return
    marker = prompt_marker(label, autonomous=autonomous)
    preview = compact_prompt_preview(prompt, multiline=True)
    rendered = colorize_user_prompt(f"{marker}{preview}")
    columns = max(20, get_terminal_size((80, 20)).columns)
    # The marker occupies columns on the first row, so it has to be counted --
    # in the form actually displayed -- or a wrapped line is erased one row
    # short.
    visual_rows = visual_row_count(f"{marker}{prompt}", columns=columns)
    print(f"\x1b[{visual_rows}F\x1b[J{rendered}", flush=True)


def clear_input_echo(prompt: str, label: str = "", *, autonomous: bool = False) -> None:
    if not sys.stdout.isatty():
        return
    columns = max(20, get_terminal_size((80, 20)).columns)
    marker = prompt_marker(label, autonomous=autonomous)
    visual_rows = visual_row_count(f"{marker}{prompt}", columns=columns)
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
