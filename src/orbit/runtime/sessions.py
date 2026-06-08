from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orbit.backend.base import Message


DEFAULT_SESSION_ROOT = Path.home() / ".orbit" / "sessions"


@dataclass(frozen=True)
class SessionStore:
    path: Path

    @classmethod
    def for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_SESSION_ROOT) -> "SessionStore":
        resolved = workdir.expanduser().resolve()
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        return cls(root / f"{resolved.name}-{digest}.json")

    def load(self) -> list[Message] | None:
        messages, _warning = self.load_with_warning()
        return messages

    def load_with_warning(self) -> tuple[list[Message] | None, str | None]:
        if not self.path.exists():
            return None, None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            return None, f"warning: cannot read session {self.path}: {exc}"
        except json.JSONDecodeError as exc:
            return None, f"warning: ignoring corrupt session {self.path}: {exc}"
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            return None, f"warning: ignoring invalid session {self.path}: missing messages"
        if not all(_is_message(message) for message in messages):
            return None, f"warning: ignoring invalid session {self.path}: malformed message"
        return messages, None

    def save(self, *, messages: list[Message], workdir: Path, model: str, base_url: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "workdir": str(workdir.expanduser().resolve()),
            "model": model,
            "base_url": base_url,
            "messages": messages,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return


def _is_message(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    role = value.get("role")
    return role in {"system", "user", "assistant"} and "content" in value
