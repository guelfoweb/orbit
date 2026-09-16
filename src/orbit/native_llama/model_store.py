"""The user-facing side of the models directory: `orbit config models-dir`,
its persistence, and the large-model filesystem advisory.

The resolution itself (`resolve_models_dir`) lives in `model_registry` beside
the historical default so that every model operation -- download, discovery,
server bootstrap, availability checks -- shares one resolver. This module only
writes the persisted value, explains the effective one, and warns (never
blocks) when a very large model is about to land on a filesystem known to
serve beyond-RAM models poorly.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping
from urllib.request import Request, urlopen

from orbit.native_llama.model_registry import (
    MODELS_DIR_CONFIG_KEY,
    MODELS_DIR_ENV,
    ModelsDirResolution,
    resolve_models_dir,
)

# Filesystems on which a memory-mapped model larger than RAM is known to
# perform poorly: eCryptfs keeps both the encrypted and the decrypted pages in
# the page cache, halving what a beyond-RAM model can keep resident.
UNFAVORABLE_FILESYSTEMS = {"ecryptfs"}

SOURCE_LABELS = {
    "cli": "--models-dir",
    "env": f"{MODELS_DIR_ENV} environment variable",
    "config": "orbit config models-dir",
    "default": "default",
}


class ModelsDirConfigError(RuntimeError):
    """A models-directory setting that cannot be applied, explained for a person."""


def persist_models_dir(
    value: "Path | str",
    *,
    config_path: Path | None = None,
    environ: "Mapping[str, str] | None" = None,
) -> Path:
    """Create the directory when possible and record it in the config file.

    Never escalates privileges: a directory whose parent needs them is reported
    with a clear message and nothing is written. The config file is merged, so
    other settings the user keeps there are preserved.
    """
    raw = str(value).strip()
    if not raw:
        raise ModelsDirConfigError("models-dir needs a directory path")
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    # The config file first: creating the models directory and then failing
    # to record it would leave a directory the user did not ask for.
    path = config_path or resolve_models_dir(environ=environ).config_path
    data = _read_config_object(path)
    _prepare_config_dir(path)
    _create_models_dir(target)
    data[MODELS_DIR_CONFIG_KEY] = str(target)
    _write_config_object(path, data)
    return target


def _create_models_dir(target: Path) -> None:
    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ModelsDirConfigError(
            f"cannot create {target}: permission denied.\n"
            "Choose a directory you can write to, or create it first with the "
            "privileges you already have (Orbit never runs sudo for you)."
        ) from None
    except FileExistsError:
        raise ModelsDirConfigError(f"cannot use {target}: a file with that name exists") from None
    except OSError as exc:
        raise ModelsDirConfigError(f"cannot create {target}: {exc.strerror or exc}") from None
    if not target.is_dir():
        raise ModelsDirConfigError(f"cannot use {target}: not a directory")
    if not os.access(target, os.W_OK | os.X_OK):
        raise ModelsDirConfigError(
            f"cannot use {target}: not writable by this user.\n"
            "Choose a directory you can write to (Orbit never runs sudo for you)."
        )


def _prepare_config_dir(path: Path) -> None:
    real = _config_write_target(path)
    try:
        real.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelsDirConfigError(f"cannot create {real.parent}: {exc.strerror or exc}") from None
    if not os.access(real.parent, os.W_OK | os.X_OK):
        raise ModelsDirConfigError(f"cannot write {real}: {real.parent} is not writable by this user")


def _config_write_target(path: Path) -> Path:
    """The file to replace: a symlinked config file is rewritten in place of
    its target, so a config kept in a dotfiles repository stays one file."""
    try:
        return path.resolve() if path.is_symlink() else path
    except OSError:
        return path


def _read_config_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelsDirConfigError(f"cannot read {path}: {exc.strerror or exc}") from None
    except ValueError as exc:
        raise ModelsDirConfigError(
            f"{path} is not valid JSON ({exc}); fix or remove it before changing models-dir"
        ) from None
    if not isinstance(data, dict):
        raise ModelsDirConfigError(f"{path}: root value must be an object")
    return data


def _write_config_object(path: Path, data: dict) -> None:
    real = _config_write_target(path)
    try:
        real.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=real.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, real)
    except OSError as exc:
        raise ModelsDirConfigError(f"cannot write {real}: {exc.strerror or exc}") from None


def describe_models_dir(resolution: ModelsDirResolution) -> str:
    lines = [
        f"models directory: {resolution.path}",
        f"source: {SOURCE_LABELS.get(resolution.source, resolution.source)}",
        f"exists: {'yes' if resolution.path.is_dir() else 'no (created on first download)'}",
    ]
    if resolution.config_error:
        lines.append(f"note: config file ignored ({resolution.config_error})")
    return "\n".join(lines)


# --- filesystem advisory ----------------------------------------------------


def filesystem_type(path: Path, *, mountinfo: str | None = None) -> str | None:
    """The filesystem type backing `path` (its nearest existing ancestor), or
    None where that cannot be determined reliably (non-Linux, unreadable
    mount table)."""
    probe = path.expanduser()
    if mountinfo is None:
        # Real probe: the nearest existing ancestor decides, symlinks followed.
        try:
            probe = probe.resolve()
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        except OSError:
            return None
    elif not probe.is_absolute():
        probe = Path.cwd() / probe
    best: tuple[int, str] | None = None
    for line in mountinfo.splitlines():
        if " - " not in line:
            continue
        head, _, tail = line.partition(" - ")
        fields = head.split()
        tail_fields = tail.split()
        if len(fields) < 5 or not tail_fields:
            continue
        mount_point = _unescape_mount(fields[4])
        fs_type = tail_fields[0]
        try:
            probe.relative_to(mount_point)
        except ValueError:
            continue
        if best is None or len(mount_point) > best[0]:
            best = (len(mount_point), fs_type)
    return best[1] if best else None


def _unescape_mount(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def total_memory_bytes(meminfo: str | None = None) -> int | None:
    try:
        text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def remote_content_length(url: str, *, timeout: float = 10.0, opener=urlopen) -> int | None:
    """Best-effort size of a download target via HEAD; None when unknown.

    `opener` is bound at definition time: tests inject it rather than patching
    `urlopen` on this module.
    """
    try:
        with opener(Request(url, method="HEAD"), timeout=timeout) as response:
            for header in ("x-linked-size", "Content-Length"):
                value = response.headers.get(header)
                if value and str(value).isdigit():
                    return int(value)
    except Exception:
        return None
    return None


def large_model_advisory(
    models_dir: Path,
    *,
    model_bytes: int | None,
    mem_total_bytes: int | None = None,
    filesystem: str | None = None,
    verbose: bool = False,
) -> str | None:
    """A concise, non-blocking warning, or None in a normal configuration.

    Shown only when the target directory sits on a known unfavorable
    filesystem AND the model is larger than the machine's RAM (or its size is
    unknown). Never raises; safe to print in scripts.
    """
    fs_type = filesystem if filesystem is not None else filesystem_type(models_dir)
    if fs_type not in UNFAVORABLE_FILESYSTEMS:
        return None
    mem_total = mem_total_bytes if mem_total_bytes is not None else total_memory_bytes()
    if model_bytes is not None and mem_total is not None and model_bytes <= mem_total:
        return None
    if model_bytes is None:
        first = "This model may be larger than the system RAM."
    else:
        first = "This model is larger than the system RAM."
    text = (
        f"{first}\n"
        f"The current model directory is on {fs_type}, which may reduce performance.\n"
        "\n"
        "Current model directory:\n"
        f"  {models_dir}\n"
        "\n"
        "You can choose another location with:\n"
        "  orbit config models-dir /path/on/another/filesystem\n"
    )
    if verbose:
        size = f"{model_bytes / 2**30:.1f} GiB" if model_bytes is not None else "unknown"
        ram = f"{mem_total / 2**30:.1f} GiB" if mem_total is not None else "unknown"
        text += f"(model {size}, RAM {ram}, filesystem {fs_type}; existing models are not moved)\n"
    return text.rstrip("\n")


def download_advisory(
    models_dir: Path,
    *,
    url: str,
    destination: Path | None = None,
    verbose: bool = False,
    size_probe=remote_content_length,
) -> str | None:
    """The advisory for one download, or None; never raises, never hangs a
    healthy configuration.

    The filesystem is checked FIRST and the size probe (an HTTP HEAD) is only
    issued when it is unfavorable and the file is not already present, so a
    download onto ext4 -- or a re-run for a model already on disk -- stays the
    purely local operation it always was.
    """
    try:
        fs_type = filesystem_type(models_dir)
        if fs_type not in UNFAVORABLE_FILESYSTEMS:
            return None
        if destination is not None and destination.exists():
            return None
        return large_model_advisory(
            models_dir, model_bytes=size_probe(url), filesystem=fs_type, verbose=verbose
        )
    except Exception:
        return None


# --- `orbit config` ---------------------------------------------------------


USAGE = """usage: orbit config models-dir [PATH]

  orbit config models-dir            show the effective models directory and its source
  orbit config models-dir PATH       use PATH for downloads, discovery and the server

Precedence: --models-dir > ORBIT_MODELS_DIR > orbit config models-dir > <orbit>/models"""


def main(argv: "list[str] | None" = None, *, environ: "Mapping[str, str] | None" = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "config":
        args = args[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] != "models-dir":
        print(f"error: unknown config setting: {args[0]}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if len(args) == 1:
        resolution = resolve_models_dir(environ=environ)
        print(describe_models_dir(resolution))
        return 0
    if len(args) > 2:
        print("error: models-dir takes one path", file=sys.stderr)
        return 2
    try:
        target = persist_models_dir(args[1], environ=environ)
    except ModelsDirConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    resolution = resolve_models_dir(environ=environ)
    print(f"models directory set to: {target}")
    print(f"recorded in: {resolution.config_path}")
    if resolution.source != "config":
        print(
            f"note: {SOURCE_LABELS[resolution.source]} currently overrides it "
            f"({resolution.path})",
        )
    return 0
