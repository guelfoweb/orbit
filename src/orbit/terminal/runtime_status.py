from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from orbit import __version__
from orbit.runtime.session_memory import DEFAULT_CONTEXT_TOKENS, SOFT_MEMORY_RATIO, estimate_message_tokens
from orbit.runtime.tools import tool_names
from orbit.terminal.config import AppConfig
from orbit.terminal.tool_mode import ToolSpec


UNKNOWN = "unknown"


class RuntimeLike(Protocol):
    messages: list[dict[str, object]]
    context_tokens: int | None
    memory_refreshes: int
    total_memory_tokens_saved: int
    mutation_verifications: int
    mutation_verification_repairs: int
    mutation_verification_failures: int


class BackendLike(Protocol):
    def health(self) -> bool: ...
    def model_info(self): ...
    def display_model_name(self) -> str: ...
    def backend_props(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class HostInfo:
    machine: str = UNKNOWN
    cpu: str = UNKNOWN
    physical_cores: str = UNKNOWN
    logical_cores: str = UNKNOWN
    ram_total: str = UNKNOWN
    ram_available: str = UNKNOWN
    os_name: str = UNKNOWN


@dataclass(frozen=True)
class AccelerationInfo:
    mode: str = UNKNOWN
    gpu: str = UNKNOWN
    vram_total: str = UNKNOWN
    vram_available: str = UNKNOWN
    offload: str = UNKNOWN


@dataclass(frozen=True)
class RuntimeStatus:
    version: str
    package_version: str
    workdir: str
    model: str
    backend: str
    server: str
    mtp: str
    mmproj: str
    tools: str
    think: str
    max_tokens: str
    temperature: str
    messages: str
    estimated_context_tokens: str
    context_window: str
    model_tools: str
    memory_refreshes: str
    total_memory_tokens_saved: str
    mutation_verifications: str
    mutation_repairs: str
    mutation_failures: str
    host: HostInfo
    acceleration: AccelerationInfo


def collect_host_info() -> HostInfo:
    return HostInfo(
        machine=_machine_model(),
        cpu=_short_cpu_name(_linux_cpu_model() or platform.processor() or UNKNOWN),
        physical_cores=_physical_core_count() or UNKNOWN,
        logical_cores=str(os.cpu_count()) if os.cpu_count() else UNKNOWN,
        ram_total=_format_gb(_linux_meminfo_value("MemTotal")),
        ram_available=_format_gb(_linux_meminfo_value("MemAvailable")),
        os_name=_os_name(),
    )


def collect_runtime_status(
    runtime: RuntimeLike,
    config: AppConfig,
    backend: BackendLike,
    *,
    tools_mode: ToolSpec | None = None,
    host_info: HostInfo | None = None,
) -> RuntimeStatus:
    info = _safe_call(getattr(backend, "model_info", None))
    props = _safe_call(getattr(backend, "backend_props", None)) or {}
    display_model = _model_name(info, backend)
    context_window = runtime.context_tokens or _model_context(info) or DEFAULT_CONTEXT_TOKENS
    display_version = _display_version(__version__, config.workdir)
    return RuntimeStatus(
        version=display_version,
        package_version=__version__,
        workdir=_workdir_display(config.workdir),
        model=display_model,
        backend=_backend_name(props),
        server="ok" if bool(_safe_call(getattr(backend, "health", None))) else "unavailable",
        mtp=_on_off(props.get("mtp_enabled")),
        mmproj=_loaded_missing(props.get("multimodal_available")),
        tools=_tools_mode(tools_mode if tools_mode is not None else config.tools),
        think="on" if config.think else "off",
        max_tokens=str(config.max_tokens),
        temperature=str(config.temperature),
        messages=str(len(runtime.messages)),
        estimated_context_tokens=str(estimate_message_tokens(runtime.messages)),
        context_window=str(context_window),
        model_tools=", ".join(tool_names()),
        memory_refreshes=str(runtime.memory_refreshes),
        total_memory_tokens_saved=str(runtime.total_memory_tokens_saved),
        mutation_verifications=str(runtime.mutation_verifications),
        mutation_repairs=str(runtime.mutation_verification_repairs),
        mutation_failures=str(runtime.mutation_verification_failures),
        host=host_info or collect_host_info(),
        acceleration=_acceleration_info(props),
    )


def format_startup_banner(status: RuntimeStatus) -> str:
    cpu = _cores_short(status.host.physical_cores, status.host.logical_cores)
    rows = [
        ("header", "Orbit Runtime"),
        ("Version", status.version),
        ("Model", status.model),
        ("Backend", status.backend),
        ("MTP", f"{status.mtp}, mmproj {status.mmproj}"),
        ("Tools", status.tools),
        ("Think", status.think),
        ("Max tokens", status.max_tokens),
        ("Workdir", status.workdir),
        ("separator", "Host"),
        ("Machine", status.host.machine),
        ("OS", status.host.os_name),
        ("CPU", cpu),
        ("Cores", f"{status.host.physical_cores} physical / {status.host.logical_cores} logical"),
        ("RAM", f"{status.host.ram_total} total, {status.host.ram_available} free"),
        ("Accel", _startup_accel_value(status.acceleration)),
    ]
    return "\n".join(
        [
            _box(rows, width=58),
            "Type /help for commands, /status for runtime details.",
        ]
    )


def _startup_accel_value(accel: AccelerationInfo) -> str:
    parts = [accel.mode]
    if accel.gpu != UNKNOWN:
        parts.append(f"GPU {accel.gpu}")
    if accel.vram_total != UNKNOWN:
        parts.append(accel.vram_total)
    if accel.offload != UNKNOWN:
        parts.append(f"offload {accel.offload}")
    return ", ".join(parts)


def format_status_panel(status: RuntimeStatus) -> str:
    rows = [
        ("header", "Orbit Runtime"),
        ("Version", status.version),
        *([("Package", status.package_version)] if status.version != status.package_version else []),
        ("Model", status.model),
        ("Backend", f"{status.backend}, server {status.server}"),
        ("MTP", f"{status.mtp}, mmproj {status.mmproj}"),
        ("Tools", status.tools),
        ("Think", status.think),
        ("Max tokens", status.max_tokens),
        ("Workdir", status.workdir),
        ("separator", "Host"),
        ("Machine", status.host.machine),
        ("CPU", status.host.cpu),
        ("Cores", f"{status.host.physical_cores} physical / {status.host.logical_cores} logical"),
        ("RAM", f"{status.host.ram_total} total / {status.host.ram_available} available"),
        ("OS", status.host.os_name),
        ("separator", "Acceleration"),
        ("Mode", status.acceleration.mode),
        ("GPU", status.acceleration.gpu),
        ("VRAM", f"{status.acceleration.vram_total} total / {status.acceleration.vram_available} available"),
        ("Offload", status.acceleration.offload),
        ("separator", "Runtime"),
        ("Messages", status.messages),
        ("Context", f"{status.estimated_context_tokens} estimated / {status.context_window} window"),
        ("Temperature", status.temperature),
        ("Model tools", status.model_tools),
        ("Memory", _memory_summary(status)),
        ("Mutations", _mutation_summary(status)),
    ]
    return _box(rows, width=80)


def _linux_cpu_model(path: Path = Path("/proc/cpuinfo")) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, value = line.split(":", 1)
                value = value.strip()
                if value:
                    return value
    except OSError:
        return None
    return None


def _physical_core_count(path: Path = Path("/proc/cpuinfo")) -> str | None:
    try:
        physical_id: str | None = None
        core_id: str | None = None
        cores: set[tuple[str, str]] = set()
        processor_count = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
            if not line.strip():
                if physical_id is not None and core_id is not None:
                    cores.add((physical_id, core_id))
                physical_id = None
                core_id = None
                continue
            if line.startswith("processor"):
                processor_count += 1
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
        if cores:
            return str(len(cores))
        return str(processor_count) if processor_count else None
    except OSError:
        return None


def _linux_meminfo_value(key: str, path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(f"{key}:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _format_gb(value: int | None) -> str:
    if value is None:
        return UNKNOWN
    return f"{value / 1024**3:.0f} GB"


def _format_bytes(value: object) -> str:
    if not isinstance(value, int):
        return UNKNOWN
    if value >= 1024**3:
        return f"{value / 1024**3:.0f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f} MB"
    return f"{value} B"


def _short_cpu_name(value: str) -> str:
    value = " ".join(value.split())
    if len(value) <= 48:
        return value
    return value[:45].rstrip() + "..."


def _os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        system = "macOS"
        release = platform.mac_ver()[0] or platform.release()
    elif system == "Windows":
        release = platform.release()
    else:
        release = platform.release()
    arch = platform.machine()
    name = " ".join(part for part in (system, release, arch) if part).strip()
    return name or UNKNOWN


def _machine_model() -> str:
    system = platform.system()
    if system == "Linux":
        return _linux_machine_model()
    if system == "Darwin":
        value = _sysctl_value("hw.model") or platform.machine()
        return _clean_machine_value(value) or UNKNOWN
    if system == "Windows":
        uname = platform.uname()
        values = [uname.node, uname.machine]
        return _clean_machine_value(" ".join(value for value in values if value)) or UNKNOWN
    return UNKNOWN


def _linux_machine_model() -> str:
    base = Path("/sys/devices/virtual/dmi/id")
    vendor = _clean_machine_value(_read_first_line(base / "sys_vendor"))
    product = _clean_machine_value(_read_first_line(base / "product_name"))
    version = _clean_machine_value(_read_first_line(base / "product_version"))
    if vendor and product and vendor.lower() not in product.lower():
        return _short_machine_name(f"{vendor} {product}")
    if product:
        return _short_machine_name(product)
    if vendor and version:
        return _short_machine_name(f"{vendor} {version}")
    if vendor:
        return _short_machine_name(vendor)
    return UNKNOWN


def _sysctl_value(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _read_first_line(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    except (OSError, IndexError):
        return None


def _clean_machine_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    useless = {
        "to be filled by o.e.m.",
        "default string",
        "system product name",
        "system version",
        "none",
    }
    if cleaned.lower() in useless:
        return None
    return cleaned


def _short_machine_name(value: str) -> str:
    if len(value) <= 48:
        return value
    return value[:45].rstrip() + "..."


def _display_version(package_version: str, cwd: Path) -> str:
    exact = _git_describe(["git", "describe", "--tags", "--exact-match", "HEAD"], cwd)
    if exact:
        return exact
    described = _git_describe(["git", "describe", "--tags", "--always", "--dirty"], cwd)
    return described or package_version


def _git_describe(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=_safe_cwd(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _safe_cwd(cwd: Path) -> Path:
    try:
        return Path(cwd).expanduser().resolve()
    except OSError:
        return Path(".")


def _workdir_display(workdir: Path) -> str:
    try:
        return str(Path(workdir).expanduser().resolve())
    except OSError:
        return UNKNOWN


def _model_name(info: object, backend: BackendLike) -> str:
    value = getattr(info, "id", None)
    if isinstance(value, str) and value:
        return _strip_long_path(value)
    display = _safe_call(getattr(backend, "display_model_name", None))
    if isinstance(display, str) and display:
        return _strip_long_path(display)
    return UNKNOWN


def _model_context(info: object) -> int | None:
    value = getattr(info, "context_length", None)
    return value if isinstance(value, int) else None


def _backend_name(props: dict[str, object]) -> str:
    value = props.get("backend") or props.get("backend_mode")
    if isinstance(value, str) and value:
        return value
    return UNKNOWN


def _acceleration_info(props: dict[str, object]) -> AccelerationInfo:
    mode = props.get("accel") or props.get("accelerator") or props.get("acceleration") or props.get("gpu_backend")
    if isinstance(mode, str) and mode:
        mode_text = mode
    elif any(_truthy(props.get(key)) for key in ("cuda", "metal", "rocm", "vulkan", "gpu_enabled")):
        mode_text = "unknown"
    else:
        mode_text = "CPU-only"
    return AccelerationInfo(
        mode=mode_text,
        gpu=_string_prop(props, "gpu_name", "gpu", "device_name"),
        vram_total=_format_bytes(_first_int_prop(props, "vram_total", "gpu_vram_total", "vram_total_bytes")),
        vram_available=_format_bytes(_first_int_prop(props, "vram_available", "gpu_vram_available", "vram_available_bytes")),
        offload=_offload_layers(props),
    )


def _first_int_prop(props: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = props.get(key)
        if isinstance(value, int):
            return value
    return None


def _string_prop(props: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = props.get(key)
        if isinstance(value, str) and value:
            return value
    return UNKNOWN


def _offload_layers(props: dict[str, object]) -> str:
    value = props.get("offload_layers") or props.get("gpu_layers") or props.get("n_gpu_layers")
    if isinstance(value, int):
        return f"{value} layers"
    if isinstance(value, str) and value:
        return value
    return UNKNOWN


def _tools_mode(mode: ToolSpec | None) -> str:
    return mode if isinstance(mode, str) and mode else UNKNOWN


def _on_off(value: object) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return UNKNOWN


def _loaded_missing(value: object) -> str:
    if isinstance(value, bool):
        return "loaded" if value else "missing"
    return UNKNOWN


def _truthy(value: object) -> bool:
    return bool(value) if isinstance(value, bool | int | str) else False


def _safe_call(fn):
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _strip_long_path(value: str) -> str:
    if "/" in value:
        value = Path(value).name
    return value if value else UNKNOWN


def _cores_short(physical: str, logical: str) -> str:
    if physical != UNKNOWN and logical != UNKNOWN:
        return f"{physical}C/{logical}T"
    if logical != UNKNOWN:
        return f"{logical}T"
    return UNKNOWN


def _memory_summary(status: RuntimeStatus) -> str:
    threshold = int(int(status.context_window) * SOFT_MEMORY_RATIO) if status.context_window.isdigit() else UNKNOWN
    return (
        f"{status.memory_refreshes} refreshes, {status.total_memory_tokens_saved} tokens saved, "
        f"threshold {threshold}/{status.context_window}"
    )


def _mutation_summary(status: RuntimeStatus) -> str:
    return (
        f"{status.mutation_verifications} verifications, "
        f"{status.mutation_repairs} repairs, {status.mutation_failures} failures"
    )


def _box(rows: list[tuple[str, str]], *, width: int) -> str:
    lines: list[str] = []
    for label, value in rows:
        if label == "header":
            lines.append(f"┌─ {value} " + "─" * max(0, width - len(value) - 3) + "┐")
        elif label == "separator":
            lines.append(f"├─ {value} " + "─" * max(0, width - len(value) - 3) + "┤")
        else:
            content = f"{label:<12} {_truncate(value, width - 15)}"
            lines.append(f"│ {content:<{width - 2}} │")
    lines.append("└" + "─" * width + "┘")
    return "\n".join(lines)


def _truncate(value: str, limit: int) -> str:
    value = str(value)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."
