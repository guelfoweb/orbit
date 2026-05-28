from __future__ import annotations

from pathlib import Path


ORBIT_HOME = Path.home() / ".orbit"
HISTORY_PATH = ORBIT_HOME / "history"
SKILLS_DIR = ORBIT_HOME / "skills"
SESSIONS_DIR = ORBIT_HOME / "sessions"


def ensure_orbit_home() -> Path:
    try:
        ORBIT_HOME.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return ORBIT_HOME


def ensure_skills_dir() -> Path:
    ensure_orbit_home()
    try:
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return SKILLS_DIR


def ensure_sessions_dir() -> Path:
    ensure_orbit_home()
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return SESSIONS_DIR
