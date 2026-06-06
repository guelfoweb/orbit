from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_HISTORY_ROOT = Path.home() / ".orbit" / "history"
LEGACY_HISTORY_NAME = "default.history"
MAX_HISTORY_ITEMS = 500


class PromptHistory:
    def __init__(self, *, path: Path, readline_module: object | None) -> None:
        self.path = path
        self.readline = readline_module

    @classmethod
    def for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_HISTORY_ROOT) -> PromptHistory:
        key = hashlib.sha256(str(workdir.expanduser().resolve()).encode("utf-8")).hexdigest()[:16]
        return cls(path=root / f"{key}.history", readline_module=_load_readline())

    def load(self) -> None:
        if self.readline is None:
            return
        if not self._ensure_parent_dir():
            return
        if not self.path.exists():
            return
        try:
            self.readline.read_history_file(str(self.path))
            self._dedupe_readline_history()
        except OSError:
            return

    def add(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt or prompt.startswith("/") or self.readline is None:
            return
        self._remove_existing(prompt)
        self.readline.add_history(prompt)
        self._trim_readline_history()

    def save(self) -> None:
        if self.readline is None:
            return
        if not self._ensure_parent_dir():
            return
        try:
            self._dedupe_readline_history()
            self.readline.write_history_file(str(self.path))
        except OSError:
            return

    def _ensure_parent_dir(self) -> bool:
        try:
            if self.path.parent.is_file():
                legacy_content = self.path.parent.read_text(encoding="utf-8", errors="replace")
                self.path.parent.unlink()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                legacy_path = self.path.parent / LEGACY_HISTORY_NAME
                if legacy_content and not legacy_path.exists():
                    legacy_path.write_text(legacy_content, encoding="utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return self.path.parent.is_dir()

    def _dedupe_readline_history(self) -> None:
        items = self._history_items()
        deduped = dedupe_prompts(items)[-MAX_HISTORY_ITEMS:]
        self.readline.clear_history()
        for item in deduped:
            self.readline.add_history(item)

    def _remove_existing(self, prompt: str) -> None:
        items = [item for item in self._history_items() if item != prompt]
        self.readline.clear_history()
        for item in items[-MAX_HISTORY_ITEMS:]:
            self.readline.add_history(item)

    def _trim_readline_history(self) -> None:
        items = self._history_items()
        if len(items) <= MAX_HISTORY_ITEMS:
            return
        self.readline.clear_history()
        for item in items[-MAX_HISTORY_ITEMS:]:
            self.readline.add_history(item)

    def _history_items(self) -> list[str]:
        length = self.readline.get_current_history_length()
        items: list[str] = []
        for index in range(1, length + 1):
            item = self.readline.get_history_item(index)
            if isinstance(item, str) and item:
                items.append(item)
        return items


def dedupe_prompts(prompts: list[str]) -> list[str]:
    seen: set[str] = set()
    result_reversed: list[str] = []
    for prompt in reversed(prompts):
        normalized = prompt.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result_reversed.append(normalized)
    result_reversed.reverse()
    return result_reversed


def _load_readline() -> object | None:
    try:
        import readline  # type: ignore
    except ImportError:
        return None
    try:
        readline.set_history_length(MAX_HISTORY_ITEMS)
    except Exception:
        pass
    return readline
