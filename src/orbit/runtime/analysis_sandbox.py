"""Run one bounded Python program against a read-only input, in isolation.

This is the execution primitive behind analyst-guided investigation: a caller
supplies an input file and a Python program, and gets back what that program
printed plus hashes of anything it wrote. The program runs under bubblewrap
with no network, no host filesystem, dropped capabilities, and hard resource
limits, so an investigation can transform data without the transforming code
reaching anything that matters.

Ported from the research harness that proved the approach, minus everything
specific to that experiment. This module knows nothing about what is being
analysed -- no file formats, no indicators, no notion of "malware". It runs a
program and reports facts about the run.

What isolation actually buys, stated precisely: the program cannot reach the
network or the host filesystem, cannot modify its input, and cannot exceed its
CPU/memory/output budget. It is arbitrary Python, so it *can* interpret the
bytes it reads however it likes -- including running them through an
interpreter it writes itself. Preventing that would require deciding what a
program means, which is not decidable and not attempted here. The guarantee is
containment of effects, not restriction of intent.

Nothing here is registered as a model-facing tool. Reaching this module is a
deliberate act by a caller that already decided code execution is warranted.
"""

from __future__ import annotations

import ast
import hashlib
import os
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from orbit.runtime.analysis_tools_shim import ORBIT_TOOLS_SOURCE

BWRAP = "/usr/bin/bwrap"
SOURCE_MOUNT = "/workspace/input"
WORK_MOUNT = "/workspace/work"

# Bounds carried over unchanged from the proven research configuration.
ACTION_TIMEOUT_SECONDS = 15.0
CPU_SECONDS = 14
ADDRESS_SPACE_BYTES = 768 * 1024 * 1024
FILE_SIZE_BYTES = 64 * 1024
OPEN_FILES = 96
EXTRA_PROCESSES = 32
MAX_CODE_CHARS = 32 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
HARD_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SCRATCH_BYTES = 8 * 1024 * 1024
MAX_SCRATCH_FILES = 32
MAX_SCRATCH_DEPTH = 4

# The recorded successful trajectory used one helper -- reading the input --
# and never invoked an external binary. Nothing else is mounted: an executable
# that no proven workflow needs is attack surface with no upside. Adding one
# back is a deliberate change with its own evidence, not a default.
ALLOWED_COMMANDS: frozenset[str] = frozenset()


class SandboxUnavailable(RuntimeError):
    """The isolation prerequisite is missing, so nothing was executed.

    Raised instead of degrading to a weaker sandbox. Running the program
    without isolation would be the one outcome worse than not running it.
    """


@dataclass(frozen=True)
class DerivedArtifact:
    """A file the program wrote, identified by content."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class AnalysisResult:
    """Facts about one execution. No interpretation of what was analysed."""

    status: str  # "ok" | "error" | "timeout" | "bounded"
    code_sha256: str
    input_sha256: str
    stdout: str
    stderr: str
    exit_status: int | None
    duration_seconds: float
    truncated: bool = False
    bound_exceeded: str | None = None
    artifacts: tuple[DerivedArtifact, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def sandbox_preflight(bwrap_path: str = BWRAP) -> None:
    """Fail closed unless real isolation is available.

    Checks that bubblewrap exists and that the exact flag set below actually
    works on this host, rather than assuming a version implies behaviour: a
    namespace the kernel refuses is a silent hole if nobody looks.
    """
    resolved = shutil.which(bwrap_path) or (bwrap_path if Path(bwrap_path).exists() else None)
    if resolved is None:
        raise SandboxUnavailable(
            "bubblewrap (bwrap) is required for sandboxed analysis and was not found"
        )
    probe = [
        resolved,
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--assert-userns-disabled",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "/bin/true",
    ]
    for root in ("/usr/lib", "/lib", "/lib64", "/bin", "/usr/bin"):
        if Path(root).exists():
            probe[-1:-1] = ["--ro-bind", root, root]
    try:
        completed = subprocess.run(
            probe, stdin=subprocess.DEVNULL, capture_output=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxUnavailable(f"bubblewrap could not start: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()[-240:]
        raise SandboxUnavailable(f"bubblewrap rejected the required isolation: {detail}")


def validate_code(code: object) -> str:
    """Accept only bounded, parseable UTF-8 Python. Returns the code."""
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    try:
        raw = code.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError("code is not valid UTF-8") from exc
    if not code.strip():
        raise ValueError("code is empty")
    if len(code) > MAX_CODE_CHARS or len(raw) > MAX_CODE_CHARS * 4:
        raise ValueError("code exceeds bounds")
    if "\x00" in code:
        raise ValueError("code contains a NUL byte")
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as exc:
        # Keep the interpreter's own wording: a caller repairing the program
        # needs the real diagnostic, not a paraphrase.
        location = f" (line {exc.lineno}, offset {exc.offset})" if exc.lineno else ""
        raise ValueError(f"Python syntax is malformed: {type(exc).__name__}: {exc.msg}{location}") from exc
    return code


def _program_with_tools(code: str) -> str:
    """Make `orbit_tools` importable without relaxing interpreter isolation.

    The sandbox runs Python with `-I`, which keeps the script's directory off
    `sys.path`; a mounted `orbit_tools.py` therefore would not import, and
    dropping `-I` to fix that would trade real isolation for convenience. So
    the module is registered in `sys.modules` ahead of the analysis code,
    which runs afterwards byte-for-byte as written.
    """
    prelude = (
        "import sys as _orbit_sys, types as _orbit_types\n"
        "_orbit_mod = _orbit_types.ModuleType('orbit_tools')\n"
        "exec(compile(_ORBIT_TOOLS_SRC, 'orbit_tools.py', 'exec'), _orbit_mod.__dict__)\n"
        "_orbit_sys.modules['orbit_tools'] = _orbit_mod\n"
        "del _orbit_sys, _orbit_types, _orbit_mod, _ORBIT_TOOLS_SRC\n"
    )
    return f"_ORBIT_TOOLS_SRC = {ORBIT_TOOLS_SOURCE!r}\n{prelude}\n{code}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _preexec_limits() -> None:
    import resource as _resource

    for kind, value in (
        (_resource.RLIMIT_CORE, 0),
        (_resource.RLIMIT_CPU, CPU_SECONDS),
        (_resource.RLIMIT_AS, ADDRESS_SPACE_BYTES),
        (_resource.RLIMIT_FSIZE, FILE_SIZE_BYTES),
        (_resource.RLIMIT_NOFILE, OPEN_FILES),
        (_resource.RLIMIT_STACK, 16 * 1024 * 1024),
    ):
        _resource.setrlimit(kind, (value, value))
    # NPROC is per-UID, not per-process: the ceiling has to clear every task
    # this user already owns, or bubblewrap cannot even fork to build the
    # namespaces and the sandbox fails before the program runs.
    _resource.setrlimit(
        _resource.RLIMIT_NPROC, (_uid_task_count() + EXTRA_PROCESSES,) * 2
    )


def _uid_task_count() -> int:
    uid = os.getuid()
    count = 0
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            lines = status.read_text(encoding="ascii", errors="ignore").splitlines()
        except OSError:
            continue
        owner = next((line for line in lines if line.startswith("Uid:")), "")
        if owner.split()[1:2] != [str(uid)]:
            continue
        try:
            count += sum(1 for _task in (status.parent / "task").iterdir())
        except OSError:
            count += 1
    return max(count, 1)


def _sandbox_command(source: Path, scratch: Path, program: Path, python: str) -> list[str]:
    args = [
        BWRAP,
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--assert-userns-disabled",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--cap-drop",
        "ALL",
        "--hostname",
        "orbit-analysis",
        "--tmpfs",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--dir",
        "/usr",
        "--dir",
        "/usr/bin",
        "--ro-bind",
        python,
        python,
    ]
    for root in ("/usr/lib", "/lib", "/lib64"):
        if Path(root).exists():
            args.extend(("--ro-bind", root, root))
    args.extend(("--dir", "/program", "--ro-bind", str(program), "/program/main.py"))
    args.extend(
        (
            "--dir",
            "/workspace",
            "--ro-bind",
            str(source),
            SOURCE_MOUNT,
            "--bind",
            str(scratch),
            WORK_MOUNT,
            "--dir",
            f"{WORK_MOUNT}/tmp",
            "--symlink",
            f"{WORK_MOUNT}/tmp",
            "/tmp",
            "--chdir",
            WORK_MOUNT,
            "--setenv",
            "PATH",
            "/nonexistent",
            "--setenv",
            "HOME",
            WORK_MOUNT,
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "PYTHONHASHSEED",
            "0",
            python,
            "-I",
            "-S",
            "/program/main.py",
        )
    )
    return args


def scratch_baseline(root: Path) -> dict[str, int]:
    """Record every entry already in `root`, keyed by relative path.

    A persistent workspace carries earlier steps' output into the next action.
    Charging an action for those is what wedges a session, so the caller
    snapshots first and the bound below is measured against the difference.
    An empty mapping (the default) reproduces the original whole-directory
    accounting, which is correct for a scratch directory that starts empty.

    Directories are recorded too, with size -1 to mark them as non-files.
    Counting them here but not there would recharge every directory an earlier
    step created to every later action, which is the same wedge one level up:
    the sandbox itself creates `work/tmp`, so it would start on the first run.
    """
    baseline: dict[str, int] = {}
    try:
        for base, dirs, files in os.walk(root, followlinks=False):
            for name in dirs:
                path = Path(base) / name
                if stat.S_ISDIR(path.lstat().st_mode):
                    baseline[str(path.relative_to(root))] = -1
            for name in files:
                path = Path(base) / name
                info = path.lstat()
                if stat.S_ISREG(info.st_mode):
                    baseline[str(path.relative_to(root))] = info.st_size
    except OSError:
        return {}
    return baseline


def _scratch_bound_error(root: Path, baseline: dict[str, int] | None = None) -> str | None:
    """Reject anything the capture step could not read safely.

    `total` and `count` measure THIS action's footprint: bytes it added and
    files it created. Growth of an existing file counts, because appending is
    writing; shrinking counts as zero rather than a credit, so deleting an old
    file cannot buy room for a new one. Unsafe entries are rejected wherever
    they sit -- a symlink left behind by an earlier step is still a symlink,
    and the capture step must never follow it.
    """
    prior = baseline or {}
    total = 0
    count = 0
    try:
        for base, dirs, files in os.walk(root, followlinks=False):
            if len(Path(base).relative_to(root).parts) > MAX_SCRATCH_DEPTH:
                return "scratch depth exceeded"
            for name in dirs + files:
                path = Path(base) / name
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    return "scratch contains a symlink"
                if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    return "scratch contains an unsupported entry"
                # Rejected here rather than at capture: an extra hard link means
                # the bytes hashed need not be the bytes written, and a session
                # workspace keeps the file, so raising during capture would
                # brick every later step instead of bounding this one.
                if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                    return "scratch contains a hard link"
                # Checked while bounding, for the same reason as the hard link:
                # a file the capture step cannot open would otherwise raise out
                # of `step()` and, because the workspace persists, keep raising.
                # Directories are checked too: os.walk cannot descend into an
                # unreadable one, so its contents would go uncharged and
                # uncaptured, then be attributed to whichever later action
                # happens to unlock it.
                readable = os.R_OK | (os.X_OK if stat.S_ISDIR(info.st_mode) else 0)
                if not os.access(path, readable):
                    return "scratch contains an unreadable entry"
                # A name the filesystem accepts but UTF-8 cannot represent
                # arrives as surrogates, which cannot be encoded again. It
                # would fail later while being written to evidence, by which
                # point the file is in the persistent workspace and every
                # later step would fail the same way.
                relative = str(path.relative_to(root))
                if any("\ud800" <= ch <= "\udfff" for ch in relative):
                    return "scratch contains an undecodable name"
                if stat.S_ISREG(info.st_mode):
                    if relative in prior and prior[relative] >= 0:
                        total += max(0, info.st_size - prior[relative])
                    else:
                        count += 1
                        total += info.st_size
                elif relative not in prior:
                    count += 1
                if count > MAX_SCRATCH_FILES or total > MAX_SCRATCH_BYTES:
                    return "scratch bound exceeded"
    except OSError:
        return "scratch inspection failed"
    return None


def _capture_artifacts(
    root: Path, baseline: dict[str, str] | None = None
) -> tuple[DerivedArtifact, ...]:
    """Hash the files this action produced.

    `baseline` maps relative path to the SHA the file had before the action.
    A file whose bytes are unchanged belongs to the step that made it and is
    not re-reported here; a file whose bytes differ is a new version and is
    reported with its new hash. Reporting the whole directory instead would
    credit every action with every artifact ever written.
    """
    prior = baseline or {}
    captured: list[DerivedArtifact] = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        # Opened with O_NOFOLLOW and re-checked: a symlink or extra hard link
        # would mean the bytes hashed are not the bytes that were written.
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("unsafe scratch entry")
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            # The bound check already rejects unreadable entries, so reaching
            # here means the state changed underneath us. Fail the capture
            # loudly rather than reporting a partial artifact set.
            raise RuntimeError("unsafe scratch entry") from exc
        try:
            data = b""
            while len(data) <= MAX_SCRATCH_BYTES:
                chunk = os.read(descriptor, min(64 * 1024, MAX_SCRATCH_BYTES - len(data) + 1))
                if not chunk:
                    break
                data += chunk
        finally:
            os.close(descriptor)
        relative = str(path.relative_to(root))
        digest = _sha256_bytes(data)
        if prior.get(relative) == digest:
            continue
        captured.append(
            DerivedArtifact(name=relative, size_bytes=len(data), sha256=digest)
        )
    return tuple(captured)


def execute_analysis(
    *,
    source_path: Path | str,
    code: str,
    scratch_dir: Path | str | None = None,
    python_executable: str | None = None,
    scratch_baseline_sizes: dict[str, int] | None = None,
    scratch_baseline_digests: dict[str, str] | None = None,
) -> AnalysisResult:
    """Run `code` with `source_path` mounted read-only, and report what happened.

    The input is hashed before and after: if the bytes differ, the run is
    rejected rather than reported, because every later claim about the input
    would be describing something that no longer exists.
    """
    source = Path(source_path).resolve()
    if not source.is_file():
        raise ValueError(f"source is not a file: {source}")
    validated = validate_code(code)
    sandbox_preflight()

    python = python_executable or f"/usr/bin/python{'.'.join(map(str, __import__('sys').version_info[:2]))}"
    if not Path(python).exists():
        raise SandboxUnavailable(f"sandbox interpreter not found: {python}")

    source_before = source.read_bytes()
    input_sha = _sha256_bytes(source_before)
    code_sha = _sha256_bytes(validated.encode("utf-8"))

    program_owner = tempfile.TemporaryDirectory(prefix="orbit-analysis-program-")
    scratch_owner = None
    if scratch_dir is None:
        scratch_owner = tempfile.TemporaryDirectory(prefix="orbit-analysis-work-")
        scratch = Path(scratch_owner.name)
    else:
        scratch = Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)

    program = Path(program_owner.name) / "main.py"
    program.write_text(_program_with_tools(validated), encoding="utf-8")
    os.chmod(program, 0o400)


    started = time.monotonic()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    bound_error: str | None = None
    timed_out = False
    exit_status: int | None = None

    try:
        process = subprocess.Popen(
            _sandbox_command(source, scratch, program, python),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_preexec_limits,
        )
    except OSError as exc:
        program_owner.cleanup()
        if scratch_owner is not None:
            scratch_owner.cleanup()
        raise SandboxUnavailable(f"sandbox failed to start: {exc}") from exc

    try:
        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            if time.monotonic() - started > ACTION_TIMEOUT_SECONDS:
                timed_out = True
                bound_error = bound_error or "analysis timeout"
            if sum(len(value) for value in buffers.values()) > HARD_OUTPUT_BYTES:
                bound_error = bound_error or "output hard limit exceeded"
            if bound_error is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            for key, _mask in selector.select(timeout=0.05):
                chunk = os.read(key.fileobj.fileno(), 8192)  # type: ignore[union-attr]
                if chunk:
                    buffers[key.data].extend(chunk)
                else:
                    selector.unregister(key.fileobj)
        exit_status = process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()

        if source.read_bytes() != source_before:
            raise RuntimeError("read-only input changed during analysis")

        scratch_error = _scratch_bound_error(scratch, scratch_baseline_sizes)
        bound_error = bound_error or scratch_error
        artifacts: tuple[DerivedArtifact, ...] = ()
        if scratch_error is None:
            artifacts = _capture_artifacts(scratch, scratch_baseline_digests)

        stdout_raw = bytes(buffers["stdout"])
        stderr_raw = bytes(buffers["stderr"])
        truncated = len(stdout_raw) + len(stderr_raw) > MAX_OUTPUT_BYTES
        stdout = stdout_raw[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")
        stderr = stderr_raw[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")

        if timed_out:
            status = "timeout"
        elif bound_error is not None:
            status = "bounded"
        elif exit_status == 0:
            status = "ok"
        else:
            status = "error"

        return AnalysisResult(
            status=status,
            code_sha256=code_sha,
            input_sha256=input_sha,
            stdout=stdout,
            stderr=stderr,
            exit_status=exit_status,
            duration_seconds=time.monotonic() - started,
            truncated=truncated,
            bound_exceeded=bound_error,
            artifacts=artifacts,
        )
    finally:
        program_owner.cleanup()
        if scratch_owner is not None:
            scratch_owner.cleanup()
