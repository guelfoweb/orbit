from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_one_shot_status_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/status")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("base_url: http://127.0.0.1:18080", completed.stdout)
        self.assertIn("server:", completed.stdout)
        self.assertIn("messages: 1", completed.stdout)
        self.assertNotIn("model: fake", completed.stdout)

    def test_one_shot_tools_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/tools")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("llama-server:", completed.stdout)
        self.assertIn("orbit-only:", completed.stdout)

    def test_one_shot_max_tokens_command_does_not_call_model(self) -> None:
        completed = _run_cli("", "/max-tokens 2048")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("max_tokens: 2048", completed.stdout)

    def test_repl_status_command_does_not_call_model(self) -> None:
        completed = _run_cli("/status\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("orbit interactive mode", completed.stdout)
        self.assertIn("base_url: http://127.0.0.1:18080", completed.stdout)
        self.assertIn("server:", completed.stdout)
        self.assertIn("messages: 1", completed.stdout)
        self.assertIn("workdir:", completed.stdout)

    def test_repl_unknown_command_is_not_sent_to_model(self) -> None:
        completed = _run_cli("/unknown\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("unknown command: /unknown", completed.stderr)

    def test_repl_max_tokens_command_updates_status(self) -> None:
        completed = _run_cli("/max-tokens\n/max-tokens 2048\n/status\n/exit\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("max_tokens: 512", completed.stdout)
        self.assertIn("max_tokens: 2048", completed.stdout)


def _run_cli(stdin: str, *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as home:
        return subprocess.run(
            [sys.executable, "-m", "orbit.terminal.cli", *args],
            cwd=ROOT,
            input=stdin,
            text=True,
            capture_output=True,
            env={"PYTHONPATH": str(ROOT / "src"), "HOME": home},
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
