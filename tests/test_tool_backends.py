from __future__ import annotations

import shlex
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.llama_server import LlamaServerError
from orbit.runtime.edit_guardrails import apply_local_edit_file
from orbit.runtime.tool_backends import HybridToolExecutor


class FakeServerTools:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def server_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": "read_file",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "server read",
                        "parameters": {"type": "object"},
                    },
                },
            },
            {
                "tool": "grep_search",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "grep_search",
                        "description": "server grep",
                        "parameters": {"type": "object"},
                    },
                },
            },
            {
                "tool": "file_glob_search",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "file_glob_search",
                        "description": "server glob",
                        "parameters": {"type": "object"},
                    },
                },
            },
            {
                "tool": "write_file",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "server write",
                        "parameters": {"type": "object"},
                    },
                },
            },
            {
                "tool": "exec_shell_command",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "exec_shell_command",
                        "description": "raw shell",
                        "parameters": {"type": "object"},
                    },
                },
            },
            {
                "tool": "edit_file",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "description": "raw edit",
                        "parameters": {"type": "object"},
                    },
                },
            },
            {
                "tool": "apply_diff",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "apply_diff",
                        "description": "raw diff",
                        "parameters": {"type": "object"},
                    },
                },
            },
        ]

    def execute_server_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self.fail:
            raise LlamaServerError("server down")
        self.executed.append((name, arguments))
        return "server result"


class HybridToolExecutorTests(unittest.TestCase):
    def test_prefers_llama_server_for_available_safe_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("read_file",),
            )

            execution = executor.execute("read_file", {"path": "note.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertEqual(execution.result.content, "server result")
        self.assertEqual(backend.executed, [("read_file", {"path": str(Path(tmp, "note.txt").resolve())})])

    def test_bounds_large_llama_server_tool_results(self) -> None:
        class LargeResultServer(FakeServerTools):
            def execute_server_tool(self, name: str, arguments: dict[str, Any]) -> str:
                self.executed.append((name, arguments))
                return "x" * 2500

        with tempfile.TemporaryDirectory() as tmp:
            backend = LargeResultServer()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("grep_search",),
            )

            execution = executor.execute("grep_search", {"path": ".", "pattern": "x"}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertLess(len(execution.result.content), 1100)
        self.assertIn("server tool result truncated", execution.result.content)

    def test_grep_search_without_path_defaults_to_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("grep_search",),
            )

            execution = executor.execute("grep_search", {"pattern": "Virgilio"}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertEqual(backend.executed, [("grep_search", {"pattern": "Virgilio", "path": str(workdir.resolve())})])

    def test_falls_back_to_orbit_for_local_only_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("stat_path",),
            )

            execution = executor.execute("stat_path", {"path": "note.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("type: file", execution.result.content)

    def test_falls_back_to_orbit_when_server_tool_fails_and_orbit_can_handle_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=FakeServerTools(fail=True),
                workdir=workdir,
                allowed_tool_names=("read_file",),
            )

            execution = executor.execute("read_file", {"path": "note.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(execution.result.content, "hello")

    def test_prefers_orbit_for_large_read_file_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.md").write_text("x" * 9000, encoding="utf-8")
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("read_file",),
            )

            execution = executor.execute("read_file", {"path": "large.md"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(backend.executed, [])
        self.assertIn("truncated", execution.result.content)

    def test_blocks_tool_outside_turn_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("read_file",),
            )

            execution = executor.execute("write_file", {"path": "note.txt", "content": "x"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("tool not available", execution.result.content)

    def test_tool_definitions_merge_server_preferred_and_orbit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("read_file", "stat_path", "grep_search"),
            )

            names = [tool["function"]["name"] for tool in executor.tool_definitions()]

        self.assertEqual(names, ["read_file", "grep_search", "stat_path"])

    def test_tool_definitions_compact_server_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("read_file", "grep_search"),
            )

            definitions = {tool["function"]["name"]: tool["function"] for tool in executor.tool_definitions()}

        self.assertEqual(definitions["read_file"]["description"], "Read file text, optionally by 1-based lines.")
        self.assertEqual(definitions["grep_search"]["description"], "Search regex in files.")

    def test_exec_shell_definition_uses_orbit_guardrail_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            definition = executor.tool_definitions()[0]

        self.assertEqual(definition["function"]["name"], "exec_shell_command")
        self.assertIn("bounded read-only", definition["function"]["description"])
        self.assertNotEqual(definition["function"]["description"], "raw shell")

    def test_exec_shell_allowed_command_runs_through_server_inside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "pwd"}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertEqual(execution.result.content, "server result")
        self.assertEqual(backend.executed[0][0], "exec_shell_command")
        self.assertEqual(backend.executed[0][1]["command"], f"cd {shlex.quote(str(workdir.resolve()))} && pwd")
        self.assertEqual(backend.executed[0][1]["timeout"], 10)
        self.assertEqual(backend.executed[0][1]["max_output_size"], 12000)

    def test_exec_shell_allows_bounded_cat_inside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "cat note.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertEqual(backend.executed[0][1]["command"], f"cd {shlex.quote(str(workdir.resolve()))} && cat note.txt")

    def test_exec_shell_allows_free_memory_readout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "free -h"}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertEqual(backend.executed[0][1]["command"], f"cd {shlex.quote(str(workdir.resolve()))} && free -h")

    def test_exec_shell_allows_lscpu_readout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "lscpu"}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertEqual(backend.executed[0][1]["command"], f"cd {shlex.quote(str(workdir.resolve()))} && lscpu")

    def test_exec_shell_allows_short_chain_of_allowed_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute(
                "exec_shell_command",
                {"command": "pwd && pwd"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(backend.executed, [])
        self.assertNotIn("error:", execution.result.content)
        self.assertGreaterEqual(execution.result.content.count(str(workdir.resolve())), 2)

    def test_exec_shell_allows_read_only_system_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("exec_shell_command",),
            )

            commands = ["uname -a", "hostname", "uptime -p", "whoami", "id -u", "date -I", "lsblk -f", "df -h --total"]
            for command in commands:
                with self.subTest(command=command):
                    execution = executor.execute("exec_shell_command", {"command": command}, chunk_budget={})
                    self.assertEqual(execution.source, "llama-server")

        self.assertEqual([item[1]["command"] for item in backend.executed], [f"cd {shlex.quote(str(workdir.resolve()))} && {command}" for command in commands])

    def test_exec_shell_allows_bounded_process_and_network_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("exec_shell_command",),
            )

            commands = ["ps aux", "pgrep -a python", "ip -brief addr", "ip addr show", "ip route", "ss -tulpen"]
            for command in commands:
                with self.subTest(command=command):
                    execution = executor.execute("exec_shell_command", {"command": command}, chunk_budget={})
                    self.assertEqual(execution.source, "llama-server")

        self.assertEqual([item[1]["command"] for item in backend.executed], [f"cd {shlex.quote(str(workdir.resolve()))} && {command}" for command in commands])

    def test_exec_shell_blocks_shell_operators_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "printf x | wc -c"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell operators", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_exec_shell_blocks_grep_pipe_for_df_total_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "df -h --total | grep total"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell operators", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_exec_shell_blocks_single_ampersand_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "ls & pwd"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("only && is allowed", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_exec_shell_blocks_disallowed_commands_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "rm note.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("command not allowed", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_exec_shell_blocks_active_network_and_state_commands_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            commands = ["curl https://example.com", "wget https://example.com", "ping 127.0.0.1", "nc -z 127.0.0.1 80", "systemctl status"]
            for command in commands:
                with self.subTest(command=command):
                    execution = executor.execute("exec_shell_command", {"command": command}, chunk_budget={})
                    self.assertEqual(execution.source, "orbit")
                    self.assertIn("command not allowed", execution.result.content)

        self.assertEqual(backend.executed, [])

    def test_exec_shell_blocks_unsafe_diagnostic_forms_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            blocked = {
                "date +%s": "arguments not allowed for date",
                "find . -exec echo": "find option not allowed",
                "ip link set lo down": "ip allows only",
                "ip route get 1.1.1.1": "ip allows only",
                "ps -eo pid,cmd": "ps allows only",
                "pgrep -u root python": "flag not allowed",
                "pgrep ../../secret": "unsafe pgrep pattern",
                "ss -K dst 127.0.0.1": "flag not allowed",
                "ss --help": "flag not allowed",
                "ls -R": "flag not allowed",
            }
            for command, expected in blocked.items():
                with self.subTest(command=command):
                    execution = executor.execute("exec_shell_command", {"command": command}, chunk_budget={})
                    self.assertEqual(execution.source, "orbit")
                    self.assertIn(expected, execution.result.content)

        self.assertEqual(backend.executed, [])

    def test_exec_shell_blocks_paths_outside_workdir_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_command",),
            )

            execution = executor.execute("exec_shell_command", {"command": "wc -l /etc/passwd"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("path escapes workdir", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_server_read_file_blocks_symlink_escape_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "work"
            outside = root / "secret.txt"
            workdir.mkdir()
            outside.write_text("secret", encoding="utf-8")
            link = workdir / "link.txt"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink not available: {exc}")
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("read_file",),
            )

            execution = executor.execute("read_file", {"path": "link.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("path escapes workdir", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_server_file_tools_block_absolute_path_escape_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("read_file", "grep_search", "file_glob_search", "write_file"),
            )

            for name, arguments in {
                "read_file": {"path": "/etc/passwd"},
                "grep_search": {"path": "/etc", "pattern": "root"},
                "file_glob_search": {"path": "/etc", "include": "*.conf"},
                "write_file": {"path": "/tmp/outside.txt", "content": "x"},
            }.items():
                with self.subTest(name=name):
                    execution = executor.execute(name, arguments, chunk_budget={})
                    self.assertEqual(execution.source, "orbit")
                    self.assertIn("path escapes workdir", execution.result.content)

        self.assertEqual(backend.executed, [])

    def test_edit_file_definition_uses_orbit_guardrail_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("edit_file", "apply_diff"),
            )

            descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in executor.tool_definitions()}

        self.assertIn("existing UTF-8", descriptions["edit_file"])
        self.assertIn("in workdir", descriptions["apply_diff"])
        self.assertNotEqual(descriptions["edit_file"], "raw edit")
        self.assertNotEqual(descriptions["apply_diff"], "raw diff")

    def test_edit_file_allowed_change_runs_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("alpha\nbeta\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("edit_file",),
            )

            execution = executor.execute(
                "edit_file",
                {"path": "note.txt", "changes": [{"mode": "replace", "line_start": 2, "line_end": 2, "content": "BETA"}]},
                chunk_budget={},
            )
            content = target.read_text(encoding="utf-8")

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(backend.executed, [])
        self.assertEqual(content, "alpha\nBETA\n")

    def test_local_edit_file_keeps_original_when_atomic_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("alpha\nbeta\n", encoding="utf-8")

            with patch("orbit.runtime.edit_guardrails.os.replace", side_effect=OSError("simulated failure")):
                result = apply_local_edit_file(
                    {
                        "path": "note.txt",
                        "changes": [{"mode": "replace", "line_start": 2, "line_end": 2, "content": "BETA"}],
                    },
                    workdir=workdir,
                )

            content = target.read_text(encoding="utf-8")
            leftovers = [path for path in workdir.iterdir() if path.name != "note.txt"]

        self.assertIn("cannot write edited file atomically", result)
        self.assertEqual(content, "alpha\nbeta\n")
        self.assertEqual(leftovers, [])

    def test_edit_file_normalizes_append_after_last_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("edit_file",),
            )

            execution = executor.execute(
                "edit_file",
                {"path": "note.txt", "changes": [{"mode": "append", "line_start": 4, "content": "delta"}]},
                chunk_budget={},
            )
            content = target.read_text(encoding="utf-8")

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(backend.executed, [])
        self.assertEqual(content, "alpha\nbeta\ngamma\ndelta\n")

    def test_edit_file_normalizes_eof_append_end_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("alpha\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("edit_file",),
            )

            execution = executor.execute(
                "edit_file",
                {"path": "note.txt", "changes": [{"mode": "append", "line_start": -1, "line_end": -1, "content": "beta"}]},
                chunk_budget={},
            )
            content = target.read_text(encoding="utf-8")

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(backend.executed, [])
        self.assertEqual(content, "alpha\nbeta\n")

    def test_edit_file_blocks_absolute_path_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("edit_file",),
            )

            execution = executor.execute(
                "edit_file",
                {"path": "/etc/passwd", "changes": [{"mode": "replace", "line_start": 1, "line_end": 1, "content": "x"}]},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("path escapes workdir", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_apply_diff_rewrites_paths_under_server_cwd(self) -> None:
        workdir_root = Path.cwd() / "workdir"
        workdir_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workdir_root) as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("alpha\nbeta\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("apply_diff",),
            )
            diff = (
                "diff --git a/note.txt b/note.txt\n"
                "--- a/note.txt\n"
                "+++ b/note.txt\n"
                "@@ -1,2 +1,2 @@\n"
                " alpha\n"
                "-beta\n"
                "+BETA\n"
            )

            execution = executor.execute("apply_diff", {"diff": diff}, chunk_budget={})

        self.assertEqual(execution.source, "llama-server")
        self.assertEqual(backend.executed[0][0], "apply_diff")
        self.assertIn("workdir/", backend.executed[0][1]["diff"])
        self.assertNotIn("--- a/note.txt", backend.executed[0][1]["diff"])

    def test_apply_diff_blocks_delete_patch_before_server(self) -> None:
        workdir_root = Path.cwd() / "workdir"
        workdir_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workdir_root) as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("apply_diff",),
            )
            diff = (
                "diff --git a/note.txt b/note.txt\n"
                "deleted file mode 100644\n"
                "--- a/note.txt\n"
                "+++ /dev/null\n"
            )

            execution = executor.execute("apply_diff", {"diff": diff}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("rejects delete", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_apply_diff_blocks_workdir_outside_server_cwd_before_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=backend,
                workdir=workdir,
                allowed_tool_names=("apply_diff",),
            )
            diff = (
                "diff --git a/note.txt b/note.txt\n"
                "--- a/note.txt\n"
                "+++ b/note.txt\n"
                "@@ -1 +1 @@\n"
                "-alpha\n"
                "+ALPHA\n"
            )

            execution = executor.execute("apply_diff", {"diff": diff}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("under the llama-server working directory", execution.result.content)
        self.assertEqual(backend.executed, [])


if __name__ == "__main__":
    unittest.main()
