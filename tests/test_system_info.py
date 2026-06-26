from __future__ import annotations

import getpass
import os
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.system_info import execute_system_info, system_info_definition


class SystemInfoTests(unittest.TestCase):
    def test_default_output_includes_compact_core_sections(self) -> None:
        output = execute_system_info({})

        self.assertIn("system_info:", output)
        self.assertIn("OS:", output)
        self.assertIn("CPU:", output)
        self.assertIn("RAM:", output)
        self.assertIn("Disk:", output)
        self.assertIn("Python:", output)

    def test_include_flags_exclude_sections(self) -> None:
        output = execute_system_info({"include_cpu": False, "include_memory": False, "include_disks": False})

        self.assertIn("OS:", output)
        self.assertNotIn("CPU:", output)
        self.assertNotIn("RAM:", output)
        self.assertNotIn("Disk:", output)

    def test_output_avoids_common_sensitive_identifiers(self) -> None:
        output = execute_system_info({})
        username = getpass.getuser()
        hostname = socket.gethostname()

        if username:
            self.assertNotIn(username, output)
        if hostname:
            self.assertNotIn(hostname, output)
        self.assertNotIn("HOME=", output)
        self.assertNotIn("PATH=", output)
        self.assertNotRegex(output, r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
        self.assertNotRegex(output, r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def test_memory_fallback_when_proc_meminfo_unavailable(self) -> None:
        with patch("orbit.runtime.system_info._read_linux_meminfo", return_value={}):
            output = execute_system_info({"include_os": False, "include_cpu": False, "include_disks": False, "include_runtime": False})

        self.assertIn("RAM: unavailable", output)
        self.assertIn("RAM total unavailable", output)

    def test_cpu_fallback_when_proc_cpuinfo_unavailable(self) -> None:
        with patch("orbit.runtime.system_info._read_linux_cpuinfo", return_value={}):
            with patch("orbit.runtime.system_info.platform.processor", return_value="fallback-cpu"):
                with patch("orbit.runtime.system_info.os.cpu_count", return_value=8):
                    output = execute_system_info({"include_os": False, "include_memory": False, "include_disks": False, "include_runtime": False})

        self.assertIn("CPU: fallback-cpu", output)
        self.assertIn("8 logical cores", output)

    def test_disk_output_is_compact(self) -> None:
        output = execute_system_info({"include_os": False, "include_cpu": False, "include_memory": False, "include_runtime": False})
        lines = [line for line in output.splitlines() if line.strip()]

        self.assertLessEqual(len(lines), 3)
        self.assertIn("Disk:", output)
        self.assertIn("- /:", output)

    def test_definition_mentions_preferred_usage(self) -> None:
        definition = system_info_definition()
        description = definition["function"]["description"]

        self.assertIn("machine specifications", description.lower())
        self.assertIn("lscpu", description)


if __name__ == "__main__":
    unittest.main()
