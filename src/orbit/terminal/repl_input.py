from __future__ import annotations

import re
import select
import sys
from shutil import get_terminal_size

from orbit.terminal.prompt_preview import compact_prompt_preview, is_long_text_prompt
from orbit.terminal.theme import yellow_dim


PASTE_BADGE_PATTERN = re.compile(r"(\[text \d+ chars #[0-9a-f]{8}\])$")
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"


def read_prompt_input() -> str:
    first_line = input("> ")
    return read_available_paste_tail(first_line)


def replace_input_echo(prompt: str) -> None:
    if not should_replace_input_echo(prompt) or not sys.stdout.isatty():
        return
    preview = colorize_paste_preview(compact_prompt_preview(prompt, multiline=True))
    columns = max(20, get_terminal_size((80, 20)).columns)
    visual_rows = visual_row_count(f"> {prompt}", columns=columns)
    print(f"\x1b[{visual_rows}F\x1b[J> {preview}", flush=True)


def should_replace_input_echo(prompt: str) -> bool:
    return is_long_text_prompt(prompt) or "\n" in prompt


def colorize_paste_preview(preview: str) -> str:
    return PASTE_BADGE_PATTERN.sub(lambda match: yellow_dim(match.group(1)), preview)


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
