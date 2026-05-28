from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import os
from typing import Any

from .paths import SESSIONS_DIR as DEFAULT_SESSIONS_DIR, ensure_orbit_home


SESSION_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
SESSIONS_DIR = DEFAULT_SESSIONS_DIR


@dataclass(frozen=True)
class SessionData:
    name: str
    path: Path
    messages: list[dict[str, Any]]
    skill_ref: str | None = None
    workdir: Path | None = None
    updated_at: int | None = None


@dataclass(frozen=True)
class SessionSummary:
    name: str
    path: Path
    workdir: Path | None
    first_prompt: str
    updated_at: int | None = None


def derive_session_name(workdir: Path) -> str:
    digest = hashlib.sha1(str(workdir).encode("utf-8")).hexdigest()[:8]
    base = SESSION_NAME_RE.sub("-", workdir.name or "root").strip("-") or "root"
    return f"{base}-{digest}"


def ensure_sessions_dir() -> Path:
    ensure_orbit_home()
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return SESSIONS_DIR


def resolve_session_path(name: str) -> Path:
    ensure_sessions_dir()
    safe_name = SESSION_NAME_RE.sub("-", name).strip("-") or "session"
    return SESSIONS_DIR / f"{safe_name}.json"


def create_session_name(workdir: Path) -> str:
    base = derive_session_name(workdir)
    existing = {summary.name for summary in list_sessions_for_workdir(workdir)}
    if base not in existing:
        return base
    index = 2
    while True:
        candidate = f"{base}-{index}"
        if candidate not in existing:
            return candidate
        index += 1


def load_session(name: str) -> SessionData | None:
    path = resolve_session_path(name)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    messages = raw.get("messages")
    if not isinstance(messages, list):
        messages = []
    skill_ref = raw.get("skill_ref")
    if not isinstance(skill_ref, str):
        skill_ref = None
    workdir_value = raw.get("workdir")
    workdir = Path(workdir_value).resolve() if isinstance(workdir_value, str) and workdir_value.strip() else None
    updated_at = raw.get("updated_at")
    if not isinstance(updated_at, int):
        updated_at = None
    return SessionData(
        name=name,
        path=path,
        messages=messages,
        skill_ref=skill_ref,
        workdir=workdir,
        updated_at=updated_at,
    )


def list_sessions_for_workdir(workdir: Path) -> list[SessionSummary]:
    ensure_sessions_dir()
    base_name = derive_session_name(workdir)
    summaries: list[SessionSummary] = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        session_workdir = _parse_workdir(raw.get("workdir"))
        if session_workdir is not None:
            if session_workdir != workdir:
                continue
        elif not (name == base_name or name.startswith(f"{base_name}-")):
            continue
        messages = raw.get("messages")
        first_prompt = _extract_first_prompt(messages)
        updated_at = raw.get("updated_at")
        if not isinstance(updated_at, int):
            updated_at = None
        summaries.append(
            SessionSummary(
                name=name,
                path=path,
                workdir=session_workdir,
                first_prompt=first_prompt,
                updated_at=updated_at,
            )
        )
    summaries.sort(key=lambda item: (item.updated_at or 0, item.name), reverse=True)
    return summaries


def delete_sessions_for_workdir(workdir: Path) -> int:
    deleted = 0
    for summary in list_sessions_for_workdir(workdir):
        try:
            summary.path.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted


def save_session(name: str, messages: list[dict[str, Any]], skill_ref: str | None, workdir: Path) -> Path:
    path = resolve_session_path(name)
    payload = {
        "name": name,
        "messages": messages,
        "skill_ref": skill_ref,
        "workdir": str(workdir),
        "updated_at": int(time.time()),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    try:
        tmp_path.replace(path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return path


def _parse_workdir(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).resolve()
def _extract_first_prompt(messages: Any) -> str:
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")) != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                first_line = content.splitlines()[0].strip()
                return first_line[:117] + "..." if len(first_line) > 120 else first_line
    return "-"
