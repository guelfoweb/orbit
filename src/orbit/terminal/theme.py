from __future__ import annotations

import os
import sys
from typing import TextIO


DIM = "\033[2m"
CYAN = "\033[36m"
RED = "\033[31m"
GREEN = "\033[32m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def supports_ansi(stream: TextIO | None = None) -> bool:
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    return is_tty(stream)


def is_tty(stream: TextIO | None = None) -> bool:
    target = stream or sys.stdout
    try:
        return bool(target.isatty())
    except (AttributeError, OSError):
        return False


def accent(text: str) -> str:
    return _styled(text, CYAN)


def dim(text: str) -> str:
    return _styled(text, DIM)


def yellow_dim(text: str) -> str:
    return _styled(text, DIM + YELLOW)


def danger(text: str) -> str:
    return _styled(text, RED)


def on_off(enabled: object, *, on: str = "on", off: str = "off") -> str:
    """One rendering of a boolean state, coloured where colour is allowed.

    Used wherever a bare on/off state is shown -- the banner's tokens, the
    `/autonomous` state reply -- so those cannot drift into saying the same
    thing several ways. Green for on and red for off is the whole convention.

    Two callers deliberately do not use it. The interactive prompt renders its
    own, because every escape there must sit inside readline's ignore markers
    or the cursor arithmetic breaks. And a reply that explains a setting in a
    sentence keeps its plain word, because colouring one word mid-sentence
    reads as emphasis rather than as state.

    Colour is decided by `_styled`, which is the single place that asks
    whether ANSI is permitted -- so NO_COLOR, a dumb terminal and a
    redirected stream all produce bare `on`/`off` here without this function
    knowing why.
    """
    return _styled(on, GREEN) if enabled else _styled(off, RED)


def warning_text(value: object) -> str:
    return _prefixed_message("warning", value)


def runtime_error_text(value: object) -> str:
    detail = str(value).strip() or value.__class__.__name__
    lowered = detail.lower()
    prefix = "timeout" if "timeout" in lowered or "timed out" in lowered else "error"
    return _prefixed_message(prefix, detail)


def _prefixed_message(prefix: str, value: object) -> str:
    detail = str(value).strip()
    for existing in ("error:", "warning:", "timeout:"):
        if detail.lower().startswith(existing):
            detail = detail[len(existing) :].strip()
            break
    return f"{prefix}: {detail}" if detail else f"{prefix}: unknown"


def _styled(text: str, prefix: str) -> str:
    if not supports_ansi():
        return text
    return f"{prefix}{text}{RESET}"


# Tab is ordinary in program output and in prose; everything else in the
# control range is an instruction to the terminal rather than text.
_SAFE_CONTROL = {ord("\t")}
_SAFE_CONTROL_MULTILINE = {ord("\t"), ord("\n")}


def sanitize_terminal_text(text: str, *, allow_newlines: bool = False) -> str:
    """Neutralize control sequences in model-authored text before printing it.

    Everything the model writes reaches the terminal as text, so left as-is it
    can act on it instead: move the cursor, erase the screen, set the window
    title, drive an OSC 52 clipboard write, or return to the start of the line
    and overwrite what Orbit already printed -- which is how a crafted response
    forges its own status line above itself. Printable text of any script
    survives untouched; a control character becomes its visible escape.

    `allow_newlines` keeps `\\n` for multi-line prose. Carriage return is never
    kept: it is the line-overwrite primitive, and prose has no use for it.
    """
    safe = _SAFE_CONTROL_MULTILINE if allow_newlines else _SAFE_CONTROL
    return "".join(
        ch if ch.isprintable() or ord(ch) in safe else repr(ch)[1:-1]
        for ch in text
    )
