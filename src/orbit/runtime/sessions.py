from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from orbit.backend.base import Message
from orbit.runtime.context_manager import conversation_structure_error


DEFAULT_SESSION_ROOT = Path.home() / ".orbit" / "sessions"


@dataclass(frozen=True)
class SessionSummary:
    store: "SessionStore"
    updated_at: str
    first_prompt: str


@dataclass(frozen=True)
class SessionStore:
    path: Path

    @classmethod
    def for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_SESSION_ROOT) -> "SessionStore":
        resolved = workdir.expanduser().resolve()
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        return cls(root / f"{resolved.name}-{digest}.json")

    @classmethod
    def new_for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_SESSION_ROOT) -> "SessionStore":
        resolved = workdir.expanduser().resolve()
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return cls(root / f"{resolved.name}-{digest}-{stamp}-{uuid4().hex[:8]}.json")

    @classmethod
    def list_for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_SESSION_ROOT) -> list[SessionSummary]:
        legacy = cls.for_workdir(workdir, root=root)
        prefix = legacy.path.stem
        summaries: list[SessionSummary] = []
        for path in sorted(root.glob(f"{prefix}*.json")):
            summary = cls(path)._summary()
            if summary is not None:
                summaries.append(summary)
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    @classmethod
    def clear_for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_SESSION_ROOT) -> int:
        legacy = cls.for_workdir(workdir, root=root)
        prefix = legacy.path.stem
        removed = 0
        for path in sorted(root.glob(f"{prefix}*.json")):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed += 1
        return removed

    def load(self) -> list[Message] | None:
        messages, _warning = self.load_with_warning()
        return messages

    def load_resumable(self) -> tuple[list[Message] | None, str | None]:
        """Load this session for use as active conversation history.

        Persisting something and being able to resume it are different
        questions. A session may legitimately hold state that is not a
        conversation at all -- a single attested tool result kept so an
        evidence card stays promptable -- and that must still load through the
        ordinary API. What cannot happen is such a history, or a genuinely
        malformed one, silently becoming the live conversation: admission
        parses history into turns before every call, so a sequence it rejects
        fails on the user's next prompt rather than at the point it was read.

        Structure is therefore checked here, on the path where persisted state
        becomes conversational, and not in `load_with_warning`. The stored file
        is never altered: it is the user's data, and rewriting a history so it
        parses would hide the problem instead of reporting it.
        """
        messages, warning = self.load_with_warning()
        if warning is not None or not messages:
            return messages, warning
        structure_error = conversation_structure_error(messages)
        if structure_error is not None:
            return None, (
                f"warning: session {self.path} cannot be resumed: "
                f"{structure_error}"
            )
        return messages, None

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

    def save(
        self,
        *,
        messages: list[Message],
        workdir: Path,
        model: str,
        base_url: str,
        workflow_mode: str | None = None,
    ) -> None:
        """Persist this session. `workflow_mode` records the mode it ended in.

        Recording a mode is not a promise that it can be resumed: what a mode
        needs in order to continue is decided when the file is read, by
        `load_workflow_mode`, and an ANALYSIS session has no durable workspace
        to come back to. Omitting the key keeps older files loadable.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "workdir": str(workdir.expanduser().resolve()),
            "model": model,
            "base_url": base_url,
            "messages": messages,
        }
        if workflow_mode is not None:
            payload["workflow_mode"] = workflow_mode
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return

    def load_workflow_mode(self) -> object | None:
        """Return the mode recorded in this file, unvalidated, or None.

        Interpreting it is the caller's job: this only reports what is stored,
        so a missing file and an unusable value stay distinguishable.
        """
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return data.get("workflow_mode")

    def _summary(self) -> SessionSummary | None:
        # The picker offers sessions to resume, so it asks the resumability
        # question rather than the persistence one: a stored object Orbit
        # already knows cannot become a conversation must not be presented as
        # one. The file stays where it is; only the offer is withheld.
        messages, warning = self.load_resumable()
        if warning or not messages:
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        updated_at = data.get("updated_at") if isinstance(data, dict) else None
        if not isinstance(updated_at, str):
            updated_at = ""
        return SessionSummary(store=self, updated_at=updated_at, first_prompt=_first_user_prompt(messages))


def _is_message(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    role = value.get("role")
    if role in {"system", "user", "assistant"}:
        return "content" in value
    if role == "tool":
        return (
            isinstance(value.get("tool_call_id"), str)
            and isinstance(value.get("name"), str)
            and "content" in value
        )
    return False


def _first_user_prompt(messages: list[Message]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return " ".join(content.split())
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            text = " ".join(" ".join(part.split()) for part in text_parts if part)
            return text or "[media prompt]"
    return "[no user prompt]"
