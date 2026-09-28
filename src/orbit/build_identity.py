"""Process-local Orbit identity for diagnostics, never an inference input.

Capture once, before serving requests: reading HEAD at /props time would label
an old process with the new checkout after a pull. Installed packages without
their source checkout expose their package version and an unknown commit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

from orbit import __version__


@dataclass(frozen=True)
class BuildIdentity:
    commit: str | None = None
    version: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _text(value: object) -> str | None:
    if not isinstance(value, str) or not 0 < len(value) <= 128:
        return None
    return value if value.isprintable() and value == value.strip() else None


def parse_build_identity(value: object) -> BuildIdentity:
    """Untrusted or missing server metadata is unknown, not a connection error."""
    if not isinstance(value, dict):
        return BuildIdentity()
    commit = _text(value.get("commit"))
    if commit is not None and (
        len(commit) not in (40, 64)
        or any(c not in "0123456789abcdefABCDEF" for c in commit)
    ):
        commit = None
    return BuildIdentity(
        commit=commit.lower() if commit else None,
        version=_text(value.get("version")),
        description=_text(value.get("description")),
    )


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            check=False, capture_output=True, text=True, timeout=0.5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _capture_build_identity(package_dir: Path) -> BuildIdentity:
    unknown = BuildIdentity(version=__version__)
    try:
        package_dir = package_dir.resolve()
        root = package_dir.parent.parent
        # Never borrow an analyst's workdir repository (or a containing repo
        # around site-packages) as the identity of the installed Orbit code.
        if package_dir != root / "src" / "orbit":
            return unknown
        if _git(root, "rev-parse", "--show-toplevel") != str(root):
            return unknown
        if _git(root, "ls-files", "--error-unmatch", "src/orbit/__init__.py") is None:
            return unknown
        commit = _git(root, "rev-parse", "--verify", "HEAD")
        description = _git(root, "describe", "--tags", "--always", "--dirty")
        return parse_build_identity({
            "commit": commit, "version": __version__, "description": description,
        })
    except OSError:
        return unknown


PROCESS_BUILD_IDENTITY = _capture_build_identity(Path(__file__).parent)
