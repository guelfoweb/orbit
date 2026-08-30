#!/usr/bin/env python3
"""Run a Python qualification command against the CURRENT source bytes.

During REF-6 a real test failure was first written off as a mutation artifact.
It was not. The source on disk said ``restored=True`` while the interpreter
executed ``captured=True``: the loaded module was stale bytecode.

CPython validates a cached ``.pyc`` against the source's *(mtime, size)* pair.
The edit that caused this replaced ``captured`` with ``restored`` -- the same
eight characters -- and landed inside the same filesystem-timestamp second, so
neither half of that pair changed and the cache was trusted. Mutation testing
produces exactly this shape of edit on purpose: a keyword, an operator or a
constant swapped for one of equal length, over and over, fast.

``inspect.getsource`` reads the *file*, so it reported the new source and
actively concealed the divergence. Source inspection is not proof of what runs.

This module launches the command in a fresh interpreter with a private,
per-run bytecode cache root, so no ``.pyc`` written before the current source
bytes can be read. Two properties are load-bearing and neither is sufficient
alone:

* **Fresh process.** A cache root cannot help a module already in
  ``sys.modules``. Mutating source inside a live interpreter and re-importing
  keeps executing the old code object.
* **Fresh root per run.** An isolated but *reused* root fails exactly like the
  worktree's: it is measured against the same unchanged ``(mtime, size)`` pair,
  so mutation N happily executes bytecode compiled for mutation N-1.

``-B`` and ``PYTHONDONTWRITEBYTECODE`` were rejected: both only suppress
*writing* a cache. A stale ``.pyc`` already on disk is still read and executed,
which is the failure being prevented. Deleting ``__pycache__`` across the
worktree was rejected as the primary mechanism: it mutates unrelated developer
state, races with concurrent runs, and fails open -- an incomplete or skipped
cleanup silently restores the hazard.

Known boundary: a ``.pyc`` whose ``.py`` no longer exists is imported
*sourceless*, and no cache setting can help -- there is no source to correspond
to. That is outside this invariant, which is about source-to-executed
correspondence, and it does not arise here: the worktree currently holds no
sourceless or orphaned bytecode.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_fresh(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    **kwargs: object,
) -> subprocess.CompletedProcess:
    """Run ``[sys.executable, *args]`` against the current source bytes.

    The private cache root is removed in ``finally``, so a failing command, a
    timeout or a spawned-process error cleans up just as a passing one does.
    The child's real exit code is returned untouched -- never re-derived from
    output text, and never taken from a shell pipeline, where it would report
    the last stage rather than the interpreter.
    """
    cache_root = tempfile.mkdtemp(prefix="orbit-qualify-pycache-")
    try:
        child_env = dict(os.environ if env is None else env)
        child_env["PYTHONPYCACHEPREFIX"] = cache_root
        child_env.setdefault("PYTHONPATH", str(SRC))
        return subprocess.run(
            [sys.executable, *args],
            cwd=str(ROOT if cwd is None else cwd),
            env=child_env,
            timeout=timeout,
            **kwargs,
        )
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: qualify_fresh.py <python-args>...\n"
            "  e.g. qualify_fresh.py -m unittest discover -s tests -q",
            file=sys.stderr,
        )
        return 2
    return run_fresh(args).returncode


if __name__ == "__main__":
    raise SystemExit(main())
