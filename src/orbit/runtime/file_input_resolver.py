from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from orbit.runtime.media import AudioInput, ImageInput, load_referenced_media


_LOCAL_DOCUMENT_ROUTE_RE = re.compile(r"(?:[\"'`](?:[^\"'`]+)[\"'`]|[A-Za-z0-9_./ -]+?\.(?:pdf|docx?|md|txt))", re.IGNORECASE)
_DOCUMENTISH_BYPASS_RE = re.compile(r"\b(?:pdf|document|doc|report|relazione|documento)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FileInputResolver:
    workdir: Path

    def resolve_media(self, prompt: str) -> tuple[list[ImageInput], list[AudioInput]]:
        return load_referenced_media(prompt, workdir=self.workdir)

    def should_bypass_tool_route(self, prompt: str, allowed_tool_names: tuple[str, ...] | None) -> bool:
        if allowed_tool_names is None or "exec_shell_full_command" not in allowed_tool_names:
            return False
        return bool(_LOCAL_DOCUMENT_ROUTE_RE.search(prompt) and _DOCUMENTISH_BYPASS_RE.search(prompt))
