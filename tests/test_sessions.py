from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.sessions import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_save_and_load_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=root)
            messages = [{"role": "user", "content": "hello"}]

            store.save(messages=messages, workdir=workdir, model="model", base_url="http://localhost")

            self.assertEqual(store.load(), messages)

    def test_clear_removes_session_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.save(messages=[{"role": "user", "content": "hello"}], workdir=workdir, model="m", base_url="u")

            store.clear()

            self.assertIsNone(store.load())

    def test_new_for_workdir_creates_distinct_session_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            root = Path(tmp) / "sessions"

            first = SessionStore.new_for_workdir(workdir, root=root)
            second = SessionStore.new_for_workdir(workdir, root=root)

            self.assertNotEqual(first.path, SessionStore.for_workdir(workdir, root=root).path)
            self.assertNotEqual(first.path, second.path)

    def test_list_for_workdir_returns_saved_sessions_with_first_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            root = Path(tmp) / "sessions"
            legacy = SessionStore.for_workdir(workdir, root=root)
            current = SessionStore.new_for_workdir(workdir, root=root)
            legacy.save(messages=[{"role": "user", "content": "old prompt"}], workdir=workdir, model="m", base_url="u")
            current.save(messages=[{"role": "user", "content": "new prompt"}], workdir=workdir, model="m", base_url="u")

            summaries = SessionStore.list_for_workdir(workdir, root=root)

            self.assertEqual(len(summaries), 2)
            self.assertEqual(summaries[0].first_prompt, "new prompt")
            self.assertEqual(summaries[1].first_prompt, "old prompt")

    def test_list_for_workdir_ignores_corrupt_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            root = Path(tmp) / "sessions"
            valid = SessionStore.new_for_workdir(workdir, root=root)
            corrupt = SessionStore.for_workdir(workdir, root=root)
            valid.save(messages=[{"role": "user", "content": "hello"}], workdir=workdir, model="m", base_url="u")
            corrupt.path.parent.mkdir(parents=True, exist_ok=True)
            corrupt.path.write_text("{bad-json", encoding="utf-8")

            summaries = SessionStore.list_for_workdir(workdir, root=root)

            self.assertEqual([summary.first_prompt for summary in summaries], ["hello"])

    def test_clear_for_workdir_removes_all_matching_sessions_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            workdir = Path(tmp) / "work"
            other_workdir = Path(tmp) / "other"
            workdir.mkdir()
            other_workdir.mkdir()
            first = SessionStore.for_workdir(workdir, root=root)
            second = SessionStore.new_for_workdir(workdir, root=root)
            other = SessionStore.for_workdir(other_workdir, root=root)
            for store, prompt in ((first, "one"), (second, "two"), (other, "other")):
                store.save(messages=[{"role": "user", "content": prompt}], workdir=workdir, model="m", base_url="u")

            removed = SessionStore.clear_for_workdir(workdir, root=root)

            self.assertEqual(removed, 2)
            self.assertEqual(SessionStore.list_for_workdir(workdir, root=root), [])
            self.assertIsNotNone(other.load())

    def test_load_with_warning_reports_corrupt_session_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.path.parent.mkdir(parents=True)
            store.path.write_text("{not-json", encoding="utf-8")

            messages, warning = store.load_with_warning()

        self.assertIsNone(messages)
        self.assertIsNotNone(warning)
        self.assertIn("corrupt session", warning or "")

    def test_load_with_warning_reports_invalid_session_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.path.parent.mkdir(parents=True)
            store.path.write_text('{"messages": [{"role": "tool", "content": "x"}]}', encoding="utf-8")

            messages, warning = store.load_with_warning()

        self.assertIsNone(messages)
        self.assertIsNotNone(warning)
        self.assertIn("invalid session", warning or "")

    def test_load_with_warning_accepts_tool_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            messages = [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "list_files", "arguments": {"path": "."}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "name": "list_files", "content": "README.md"},
                {"role": "assistant", "content": "README.md"},
            ]
            store.save(messages=messages, workdir=workdir, model="m", base_url="u")

            loaded, warning = store.load_with_warning()

        self.assertEqual(loaded, messages)
        self.assertIsNone(warning)

    def test_load_with_warning_rejects_malformed_tool_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.path.parent.mkdir(parents=True)
            store.path.write_text('{"messages": [{"role": "tool", "content": "x"}]}', encoding="utf-8")

            messages, warning = store.load_with_warning()

        self.assertIsNone(messages)
        self.assertIsNotNone(warning)
        self.assertIn("invalid session", warning or "")


if __name__ == "__main__":
    unittest.main()
