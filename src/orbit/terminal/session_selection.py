from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from orbit.runtime.sessions import SessionStore
from orbit.terminal.theme import dim


def select_interactive_session(workdir: Path, *, root: Path | None = None) -> SessionStore:
    sessions = SessionStore.list_for_workdir(workdir, root=root) if root else SessionStore.list_for_workdir(workdir)
    if not sessions or not sys.stdin.isatty():
        return _new_session(workdir, root=root)
    print("saved sessions:")
    for index, summary in enumerate(sessions, start=1):
        print(f"{index}. [{display_datetime(summary.updated_at)}] {preview_prompt(summary.first_prompt)}")
    try:
        choice = input("Select session number, or press Enter for a new session: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return _new_session(workdir, root=root)
    if not choice:
        return _new_session(workdir, root=root)
    try:
        selected = int(choice)
    except ValueError:
        print(dim("invalid session selection; starting new session"), file=sys.stderr)
        return _new_session(workdir, root=root)
    if not 1 <= selected <= len(sessions):
        print(dim("invalid session selection; starting new session"), file=sys.stderr)
        return _new_session(workdir, root=root)
    return sessions[selected - 1].store


def display_datetime(value: str) -> str:
    if not value:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def preview_prompt(prompt: str, *, limit: int = 70) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _new_session(workdir: Path, *, root: Path | None) -> SessionStore:
    return SessionStore.new_for_workdir(workdir, root=root) if root else SessionStore.new_for_workdir(workdir)
