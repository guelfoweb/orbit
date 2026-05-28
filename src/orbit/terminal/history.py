from __future__ import annotations

import atexit

from ..paths import HISTORY_PATH, ensure_orbit_home

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None


HISTORY_LIMIT = 1000
BRACKETED_PASTE_SETTING = "set enable-bracketed-paste on"


def setup_history() -> None:
    if readline is None:
        return
    _enable_bracketed_paste()
    ensure_orbit_home()
    try:
        readline.read_history_file(HISTORY_PATH)
    except FileNotFoundError:
        pass
    except OSError:
        return
    readline.set_history_length(HISTORY_LIMIT)
    atexit.register(save_history)


def _enable_bracketed_paste() -> None:
    if readline is None:
        return
    try:
        readline.parse_and_bind(BRACKETED_PASTE_SETTING)
    except (AttributeError, OSError, TypeError, ValueError):  # pragma: no cover - best effort
        pass


def save_history() -> None:
    if readline is None:
        return
    try:
        ensure_orbit_home()
        readline.set_history_length(HISTORY_LIMIT)
        readline.write_history_file(HISTORY_PATH)
    except OSError:
        pass


def remember_input(user_input: str) -> None:
    if readline is None:
        return
    text = user_input.strip()
    if not text:
        return
    current_length = readline.get_current_history_length()
    if current_length > 0:
        previous = readline.get_history_item(current_length)
        if previous == text:
            return
    readline.add_history(text)
    save_history()
