from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any


def system_info_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": (
                "Return compact local machine specifications such as OS, CPU, RAM, disk capacity, "
                "and Python runtime. Prefer this over noisy shell commands like lscpu, free, df, "
                "uname, or cat /proc/* when the user asks for computer specs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_disks": {"type": "boolean", "default": True},
                    "include_cpu": {"type": "boolean", "default": True},
                    "include_memory": {"type": "boolean", "default": True},
                    "include_os": {"type": "boolean", "default": True},
                    "include_runtime": {"type": "boolean", "default": True},
                    "include_gpu": {"type": "boolean", "default": False},
                    "human_readable": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
        },
    }


def execute_system_info(arguments: dict[str, Any]) -> str:
    include_disks = _bool_arg(arguments.get("include_disks"), default=True)
    include_cpu = _bool_arg(arguments.get("include_cpu"), default=True)
    include_memory = _bool_arg(arguments.get("include_memory"), default=True)
    include_os = _bool_arg(arguments.get("include_os"), default=True)
    include_runtime = _bool_arg(arguments.get("include_runtime"), default=True)
    include_gpu = _bool_arg(arguments.get("include_gpu"), default=False)
    human_readable = _bool_arg(arguments.get("human_readable"), default=True)

    warnings: list[str] = []
    lines = ["system_info:"]

    if include_os:
        lines.append(f"OS: {_os_summary()}")
    if include_cpu:
        lines.append(f"CPU: {_cpu_summary(warnings=warnings)}")
    if include_memory:
        lines.append(f"RAM: {_memory_summary(human_readable=human_readable, warnings=warnings)}")
    if include_disks:
        lines.append("Disk:")
        lines.extend(f"- {line}" for line in _disk_summaries(human_readable=human_readable, warnings=warnings))
    if include_runtime:
        lines.append(f"Python: {platform.python_version()}")
    if include_gpu:
        lines.append("GPU: not detected by the standard-library system_info tool")
        warnings.append("GPU detection is limited; no external GPU tools were run.")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
    return "\n".join(lines)


def _os_summary() -> str:
    system = platform.system() or "unknown"
    release = platform.release() or "unknown"
    machine = platform.machine() or "unknown"
    return f"{system} {release} {machine}"


def _cpu_summary(*, warnings: list[str]) -> str:
    cpu = _read_linux_cpuinfo(Path("/proc/cpuinfo"))
    model = cpu.get("model") or platform.processor() or "unknown"
    logical = os.cpu_count()
    architecture = platform.machine() or "unknown"
    parts = [model]
    physical = cpu.get("physical_cores")
    if physical and logical:
        parts.append(f"{physical} physical cores / {logical} logical cores")
    elif logical:
        parts.append(f"{logical} logical cores")
    else:
        warnings.append("CPU core count unavailable.")
    parts.append(f"architecture {architecture}")
    return ", ".join(parts)


def _memory_summary(*, human_readable: bool, warnings: list[str]) -> str:
    memory = _read_linux_meminfo(Path("/proc/meminfo"))
    total = memory.get("MemTotal")
    available = memory.get("MemAvailable")
    if total is None:
        warnings.append("RAM total unavailable.")
        return "unavailable"
    if available is None:
        return f"{_format_bytes(total, human_readable=human_readable)} total, available unavailable"
    return (
        f"{_format_bytes(total, human_readable=human_readable)} total, "
        f"{_format_bytes(available, human_readable=human_readable)} available"
    )


def _disk_summaries(*, human_readable: bool, warnings: list[str]) -> list[str]:
    try:
        usage = shutil.disk_usage("/")
    except OSError as exc:
        warnings.append(f"Disk usage unavailable: {exc}")
        return ["unavailable"]
    percent = int(round((usage.used / usage.total) * 100)) if usage.total else 0
    return [
        (
            f"/: {_format_bytes(usage.total, human_readable=human_readable)} total, "
            f"{_format_bytes(usage.used, human_readable=human_readable)} used, "
            f"{_format_bytes(usage.free, human_readable=human_readable)} available, {percent}% used"
        )
    ]


def _read_linux_meminfo(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else ""
        values[key] = value * 1024 if unit == "kb" else value
    return values


def _read_linux_cpuinfo(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    model = ""
    physical_core_ids: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    for raw_line in content.splitlines() + [""]:
        line = raw_line.strip()
        if not line:
            model = model or current.get("model name", "")
            physical_id = current.get("physical id")
            core_id = current.get("core id")
            if physical_id is not None and core_id is not None:
                physical_core_ids.add((physical_id, core_id))
            current = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    result: dict[str, Any] = {}
    if model:
        result["model"] = model
    if physical_core_ids:
        result["physical_cores"] = len(physical_core_ids)
    return result


def _format_bytes(value: int, *, human_readable: bool) -> str:
    if not human_readable:
        return f"{value} B"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024 or unit == "PiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _bool_arg(value: object, *, default: bool) -> bool:
    return value if isinstance(value, bool) else default
