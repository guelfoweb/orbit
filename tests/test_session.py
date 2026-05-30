from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit import session as session_module


class SessionTests(unittest.TestCase):
    def test_list_sessions_for_workdir_does_not_resolve_stored_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir)
            workdir = Path("/tmp/project")
            (sessions_dir / "project.json").write_text(
                (
                    "{"
                    '"name": "project", '
                    '"workdir": "/tmp/project", '
                    '"messages": [{"role": "user", "content": "hello"}], '
                    '"updated_at": 1'
                    "}"
                ),
                encoding="utf-8",
            )
            with (
                patch.object(session_module, "SESSIONS_DIR", sessions_dir),
                patch.object(session_module, "ensure_orbit_home"),
                patch("pathlib.Path.resolve", side_effect=AssertionError("resolve should not be called")),
            ):
                summaries = session_module.list_sessions_for_workdir(workdir)

        self.assertEqual([summary.name for summary in summaries], ["project"])


if __name__ == "__main__":
    unittest.main()
