from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.runtime_status import (
    AccelerationInfo,
    HostInfo,
    RuntimeStatus,
    _clean_machine_value,
    _linux_machine_model,
    _linux_meminfo_value,
    _machine_model,
    _os_name,
    collect_runtime_status,
    format_startup_banner,
    format_status_panel,
)
from orbit.terminal.config import AppConfig


class RuntimeStatusFormattingTests(unittest.TestCase):
    def test_startup_banner_cpu_only(self) -> None:
        banner = format_startup_banner(_status())

        self.assertIn("┌─ Orbit Runtime", banner)
        self.assertIn("│ Version      0.0.1", banner)
        self.assertIn("Gemma 4 12B", banner)
        self.assertIn("│ MTP          on, mmproj loaded", banner)
        self.assertIn("│ Tools        on", banner)
        self.assertIn("│ Think        off", banner)
        self.assertIn("│ Workdir      /tmp/orbit", banner)
        self.assertIn("│ Machine      Test Machine", banner)
        self.assertIn("│ OS           Linux test", banner)
        self.assertIn("│ CPU          8C/16T", banner)
        self.assertIn("│ Cores        8 physical / 16 logical", banner)
        self.assertIn("│ RAM          32 GB total, 21 GB free", banner)
        self.assertIn("│ Accel        CPU-only", banner)
        self.assertIn("Type /help for commands, /status for runtime details.", banner)

    def test_startup_banner_gpu_mock(self) -> None:
        status = _status(
            acceleration=AccelerationInfo(
                mode="CUDA",
                gpu="RTX 4090",
                vram_total="24 GB",
                vram_available="20 GB",
                offload="41 layers",
            )
        )

        banner = format_startup_banner(status)

        self.assertIn("│ Accel        CUDA, GPU RTX 4090, 24 GB, offload 41 la...", banner)

    def test_status_panel_contains_runtime_host_and_acceleration(self) -> None:
        panel = format_status_panel(_status())

        self.assertIn("┌─ Orbit Runtime", panel)
        self.assertIn("Version", panel)
        self.assertIn("Model", panel)
        self.assertIn("MTP", panel)
        self.assertIn("Host", panel)
        self.assertIn("Machine", panel)
        self.assertIn("Linux test", panel)
        self.assertIn("Acceleration", panel)
        self.assertIn("CPU-only", panel)
        self.assertIn("Mutations", panel)

    def test_status_panel_shows_package_when_git_version_differs(self) -> None:
        panel = format_status_panel(_status(version="v0.0.1-rc11", package_version="0.0.1"))

        self.assertIn("│ Version      v0.0.1-rc11", panel)
        self.assertIn("│ Package      0.0.1", panel)

    def test_collect_runtime_status_uses_exact_git_tag(self) -> None:
        def fake_run(command, **kwargs):
            if "--exact-match" in command:
                return SimpleNamespace(returncode=0, stdout="v0.0.1-rc11\n")
            return SimpleNamespace(returncode=0, stdout="ignored\n")

        with mock.patch("orbit.terminal.runtime_status.subprocess.run", side_effect=fake_run):
            status = collect_runtime_status(_Runtime(), AppConfig(workdir=ROOT), _Backend())

        self.assertEqual(status.version, "v0.0.1-rc11")
        self.assertEqual(status.package_version, "0.0.1")
        self.assertEqual(status.workdir, str(ROOT))

    def test_collect_runtime_status_falls_back_to_describe_then_package(self) -> None:
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if "--exact-match" in command:
                return SimpleNamespace(returncode=1, stdout="")
            return SimpleNamespace(returncode=0, stdout="v0.0.1-rc11-2-gabc123\n")

        with mock.patch("orbit.terminal.runtime_status.subprocess.run", side_effect=fake_run):
            status = collect_runtime_status(_Runtime(), AppConfig(workdir=ROOT), _Backend())

        self.assertEqual(status.version, "v0.0.1-rc11-2-gabc123")
        self.assertEqual(len(calls), 2)

        with mock.patch("orbit.terminal.runtime_status.subprocess.run", side_effect=OSError):
            fallback = collect_runtime_status(_Runtime(), AppConfig(workdir=ROOT), _Backend())

        self.assertEqual(fallback.version, "0.0.1")

    def test_startup_banner_truncates_long_workdir(self) -> None:
        banner = format_startup_banner(_status(workdir="/tmp/" + "very-long/" * 12))

        self.assertIn("│ Workdir      /tmp/very-long/very-long/very-long/very-...", banner)

    def test_unknown_values_do_not_fail(self) -> None:
        panel = format_status_panel(_status(host=HostInfo(), acceleration=AccelerationInfo()))

        self.assertIn("unknown", panel)

    def test_linux_machine_model_combines_vendor_and_product(self) -> None:
        values = {
            "sys_vendor": "Intel(R) Client Systems",
            "product_name": "NUC10i7FNH",
            "product_version": "K61360-306",
        }

        with mock.patch("orbit.terminal.runtime_status._read_first_line", side_effect=lambda path: values[path.name]):
            self.assertEqual(_linux_machine_model(), "Intel(R) Client Systems NUC10i7FNH")

    def test_linux_machine_model_filters_useless_values(self) -> None:
        values = {
            "sys_vendor": "To Be Filled By O.E.M.",
            "product_name": "System Product Name",
            "product_version": "Default string",
        }

        with mock.patch("orbit.terminal.runtime_status._read_first_line", side_effect=lambda path: values[path.name]):
            self.assertEqual(_linux_machine_model(), "unknown")

        self.assertIsNone(_clean_machine_value("None"))
        self.assertIsNone(_clean_machine_value(" "))

    def test_machine_model_macos_uses_hw_model(self) -> None:
        with (
            mock.patch("orbit.terminal.runtime_status.platform.system", return_value="Darwin"),
            mock.patch("orbit.terminal.runtime_status._sysctl_value", return_value="Mac15,3"),
        ):
            self.assertEqual(_machine_model(), "Mac15,3")

    def test_os_formatter_for_common_platforms(self) -> None:
        with (
            mock.patch("orbit.terminal.runtime_status.platform.system", return_value="Linux"),
            mock.patch("orbit.terminal.runtime_status.platform.release", return_value="6.8"),
            mock.patch("orbit.terminal.runtime_status.platform.machine", return_value="x86_64"),
        ):
            self.assertEqual(_os_name(), "Linux 6.8 x86_64")

        with (
            mock.patch("orbit.terminal.runtime_status.platform.system", return_value="Darwin"),
            mock.patch("orbit.terminal.runtime_status.platform.mac_ver", return_value=("15.5", ("", "", ""), "")),
            mock.patch("orbit.terminal.runtime_status.platform.release", return_value="24.5.0"),
            mock.patch("orbit.terminal.runtime_status.platform.machine", return_value="arm64"),
        ):
            self.assertEqual(_os_name(), "macOS 15.5 arm64")

        with (
            mock.patch("orbit.terminal.runtime_status.platform.system", return_value="Windows"),
            mock.patch("orbit.terminal.runtime_status.platform.release", return_value="11"),
            mock.patch("orbit.terminal.runtime_status.platform.machine", return_value="AMD64"),
        ):
            self.assertEqual(_os_name(), "Windows 11 AMD64")

    def test_meminfo_parser_reads_kb_values(self) -> None:
        path = ROOT / "tmp-test-meminfo"
        try:
            path.write_text("MemTotal:       1024 kB\nMemAvailable:    512 kB\n", encoding="utf-8")

            self.assertEqual(_linux_meminfo_value("MemTotal", path), 1024 * 1024)
            self.assertEqual(_linux_meminfo_value("MemAvailable", path), 512 * 1024)
        finally:
            path.unlink(missing_ok=True)


def _status(
    *,
    version: str = "0.0.1",
    package_version: str = "0.0.1",
    workdir: str = "/tmp/orbit",
    host: HostInfo | None = None,
    acceleration: AccelerationInfo | None = None,
) -> RuntimeStatus:
    return RuntimeStatus(
        version=version,
        package_version=package_version,
        workdir=workdir,
        model="Gemma 4 12B",
        backend="native",
        server="ok",
        mtp="on",
        mmproj="loaded",
        tools="on",
        think="off",
        max_tokens="192",
        temperature="0.0",
        messages="1",
        estimated_context_tokens="42",
        context_window="8192",
        model_tools="exec_shell_full_command",
        memory_refreshes="0",
        total_memory_tokens_saved="0",
        mutation_verifications="0",
        mutation_repairs="0",
        mutation_failures="0",
        host=host
        or HostInfo(
            machine="Test Machine",
            cpu="Test CPU",
            physical_cores="8",
            logical_cores="16",
            ram_total="32 GB",
            ram_available="21 GB",
            os_name="Linux test",
        ),
        acceleration=acceleration or AccelerationInfo(mode="CPU-only"),
    )


class _Runtime:
    messages = []
    context_tokens = None
    memory_refreshes = 0
    total_memory_tokens_saved = 0
    mutation_verifications = 0
    mutation_verification_repairs = 0
    mutation_verification_failures = 0


class _Backend:
    def model_info(self):
        return SimpleNamespace(id="gemma4:12b", context_length=8192)

    def display_model_name(self) -> str:
        return "gemma4:12b"

    def backend_props(self) -> dict[str, object]:
        return {"backend": "orbit-native", "mtp_enabled": True, "multimodal_available": True}

    def health(self) -> bool:
        return True
