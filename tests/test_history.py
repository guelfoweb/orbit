from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.history import PromptHistory, dedupe_prompts
from orbit.terminal.prompt_preview import compact_prompt_preview, is_compact_prompt_preview


class FakeReadline:
    def __init__(self) -> None:
        self.items: list[str] = []

    def read_history_file(self, path: str) -> None:
        self.items = Path(path).read_text(encoding="utf-8").splitlines()

    def write_history_file(self, path: str) -> None:
        Path(path).write_text("\n".join(self.items) + ("\n" if self.items else ""), encoding="utf-8")

    def add_history(self, item: str) -> None:
        self.items.append(item)

    def clear_history(self) -> None:
        self.items.clear()

    def get_current_history_length(self) -> int:
        return len(self.items)

    def get_history_item(self, index: int) -> str | None:
        try:
            return self.items[index - 1]
        except IndexError:
            return None


class HistoryTests(unittest.TestCase):
    def test_dedupe_prompts_keeps_last_occurrence_order(self) -> None:
        prompts = ["a", "b", "a", "c", "b"]

        self.assertEqual(dedupe_prompts(prompts), ["a", "c", "b"])

    def test_prompt_history_skips_slash_commands_and_dedupes(self) -> None:
        readline = FakeReadline()
        with tempfile.TemporaryDirectory() as tmp:
            history = PromptHistory(path=Path(tmp) / "history", readline_module=readline)

            history.add("hello")
            history.add("/status")
            history.add("hello")
            history.add("summarize file")

        self.assertEqual(readline.items, ["hello", "summarize file"])

    def test_prompt_history_persists_deduped_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history"
            first = PromptHistory(path=path, readline_module=FakeReadline())
            first.add("one")
            first.add("two")
            first.add("one")
            first.save()

            readline = FakeReadline()
            second = PromptHistory(path=path, readline_module=readline)
            second.load()

        self.assertEqual(readline.items, ["two", "one"])

    def test_long_prompt_history_uses_preview_and_resolves_full_text(self) -> None:
        long_prompt = "Lorem ipsum dolor sit amet, " + ("x" * 900)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history"
            first_readline = FakeReadline()
            first = PromptHistory(path=path, readline_module=first_readline)
            first.add(long_prompt)
            first.save()

            preview = first_readline.items[0]
            second_readline = FakeReadline()
            second = PromptHistory(path=path, readline_module=second_readline)
            second.load()

            self.assertNotEqual(preview, long_prompt)
            self.assertIn("... ", preview)
            self.assertTrue(is_compact_prompt_preview(preview))
            self.assertEqual(second_readline.items, [preview])
            self.assertEqual(second.resolve(preview), long_prompt)

    def test_missing_sidecar_reports_unavailable_full_text(self) -> None:
        long_prompt = "Lorem ipsum dolor sit amet, " + ("x" * 900)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history"
            history = PromptHistory(path=path, readline_module=FakeReadline())
            history.add(long_prompt)
            history.save()
            preview = history.readline.items[0]
            history._full_history_path().unlink()

            resolution = history.resolve_prompt(preview)

        self.assertTrue(resolution.missing_full_text)
        self.assertEqual(resolution.prompt, preview)

    def test_corrupt_sidecar_records_are_ignored_safely(self) -> None:
        preview = compact_prompt_preview("Lorem ipsum dolor sit amet, " + ("x" * 900))
        with tempfile.TemporaryDirectory() as tmp:
            history = PromptHistory(path=Path(tmp) / "history", readline_module=FakeReadline())
            history._ensure_parent_dir()
            history.path.write_text(preview + "\n", encoding="utf-8")
            history._full_history_path().write_text(
                "\n".join(
                    [
                        "{not-json",
                        "{}",
                        '{"preview":"x"}',
                        '{"prompt":"x"}',
                    ]
                ),
                encoding="utf-8",
            )

            resolution = history.resolve_prompt(preview)

        self.assertTrue(resolution.missing_full_text)
        self.assertEqual(resolution.prompt, preview)

    def test_duplicate_prefix_and_length_previews_are_distinct(self) -> None:
        shared_prefix = "same prefix " + ("x" * 80)
        first = shared_prefix + ("a" * 820)
        second = shared_prefix + ("b" * 820)
        with tempfile.TemporaryDirectory() as tmp:
            readline = FakeReadline()
            history = PromptHistory(path=Path(tmp) / "history", readline_module=readline)
            history.add(first)
            history.add(second)

            first_preview, second_preview = readline.items

        self.assertNotEqual(first_preview, second_preview)
        self.assertTrue(is_compact_prompt_preview(first_preview))
        self.assertTrue(is_compact_prompt_preview(second_preview))

    def test_multiline_long_prompt_preview_resolves_full_text(self) -> None:
        long_prompt = "Header line\n" + "\n".join(f"line {index}" for index in range(200))
        with tempfile.TemporaryDirectory() as tmp:
            readline = FakeReadline()
            history = PromptHistory(path=Path(tmp) / "history", readline_module=readline)
            history.add(long_prompt)

            preview = readline.items[0]

            self.assertNotIn("\n", preview)
            self.assertEqual(history.resolve(preview), long_prompt)

    def test_restart_loads_preview_and_resolves_full_text(self) -> None:
        long_prompt = "Restarted paste " + ("x" * 900)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history"
            first = PromptHistory(path=path, readline_module=FakeReadline())
            first.add(long_prompt)
            first.save()

            second_readline = FakeReadline()
            second = PromptHistory(path=path, readline_module=second_readline)
            second.load()
            preview = second_readline.items[0]

            self.assertEqual(second.resolve(preview), long_prompt)

    def test_compact_prompt_preview_keeps_short_prompts_unchanged(self) -> None:
        self.assertEqual(compact_prompt_preview("short prompt"), "short prompt")

    def test_prompt_history_migrates_legacy_parent_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "history"
            parent.write_text("legacy", encoding="utf-8")
            history = PromptHistory(path=parent / "item.history", readline_module=FakeReadline())

            history.load()
            history.add("hello")
            history.save()

            self.assertTrue(parent.is_dir())
            self.assertEqual((parent / "default.history").read_text(encoding="utf-8"), "legacy")
            self.assertEqual((parent / "item.history").read_text(encoding="utf-8"), "hello\n")


if __name__ == "__main__":
    unittest.main()
