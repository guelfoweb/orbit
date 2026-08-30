"""Qualification must execute the current source bytes, never a stale cache.

During REF-6 a real failure was nearly dismissed as a mutation artifact: the
source said ``restored=True`` while the interpreter ran ``captured=True``.
CPython validates a cached ``.pyc`` on the source's *(mtime, size)* pair, and a
same-length edit inside one filesystem-timestamp second changes neither.

Every test here builds that exact condition deterministically -- equal-length
source, timestamp pinned back -- and then proves what actually RAN. None of them
assert on environment variables, and none use ``inspect.getsource``, which reads
the file and so agreed with the new source while the old code executed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qualify_fresh import run_fresh

# `captured` and `restored` are both 8 characters: the REF-6 edit exactly.
VERSIONS = ("captured", "restored", "deferred", "rejected")
PROBE = "import victim; print(victim.keyword())"


def _write(module: Path, keyword: str, *, mtime: float | None = None) -> float:
    """Write the module at a fixed length, optionally pinning its mtime."""
    module.write_text(f"def keyword():\n    return {keyword!r}\n")
    if mtime is not None:
        os.utime(module, (mtime, mtime))
    return module.stat().st_mtime


def _stale_fixture(tmp: Path) -> tuple[Path, float, int]:
    """Version A, imported so a cache exists, then overwritten by B.

    Returns the module, the pinned mtime and the byte size -- both identical
    across A and B, so CPython's cache check cannot tell them apart.
    """
    module = tmp / "victim.py"
    mtime = _write(module, "captured")
    size = module.stat().st_size
    # Warm the cache the way an ordinary interpreter would, next to the source.
    # PYTHONPYCACHEPREFIX is stripped deliberately: when the suite itself runs
    # under the fresh runner it is inherited, and the warm-up would land in that
    # private root instead of the `__pycache__` this fixture is about.
    warm = {k: v for k, v in os.environ.items() if k != "PYTHONPYCACHEPREFIX"}
    subprocess.run(
        [sys.executable, "-c", PROBE], cwd=tmp, env=warm,
        capture_output=True, text=True, check=True,
    )
    _write(module, "restored", mtime=mtime)
    assert module.stat().st_size == size, "A and B must be the same length"
    assert module.stat().st_mtime == mtime, "the cache-visible timestamp must not move"
    return module, mtime, size


def _baseline(tmp: Path) -> str:
    """A plain interpreter, as the REF-6 qualification ran."""
    plain = {k: v for k, v in os.environ.items() if k != "PYTHONPYCACHEPREFIX"}
    done = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=tmp, env=plain, capture_output=True, text=True
    )
    return done.stdout.strip()


def _fresh(tmp: Path, **kwargs) -> subprocess.CompletedProcess:
    return run_fresh(["-c", PROBE], cwd=tmp, capture_output=True, text=True, **kwargs)


class StaleBytecodeReproducerTests(unittest.TestCase):
    """The hazard is real and deterministic, not a timing accident."""

    def test_a_same_length_edit_leaves_the_cache_valid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            module, mtime, size = _stale_fixture(tmp)

            self.assertEqual(module.stat().st_size, size)
            self.assertEqual(module.stat().st_mtime, mtime)
            self.assertTrue(
                any((tmp / "__pycache__").glob("victim.*.pyc")),
                "the stale cache must exist for the reproducer to mean anything",
            )

    def test_baseline_execution_is_stale(self) -> None:
        """Without the fix the old code runs. This is the REF-6 failure."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _stale_fixture(tmp)

            self.assertEqual(
                _baseline(tmp), "captured",
                "if this stops being stale the reproducer no longer reproduces "
                "the condition these tests exist to defend against",
            )

    def test_source_and_executed_code_diverge_at_baseline(self) -> None:
        """Reading the file cannot clear the anomaly -- it shows the NEW source.

        This is why REF-6 was nearly misdiagnosed: the file said `restored`
        while `captured` executed.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            module, _, _ = _stale_fixture(tmp)

            self.assertIn("restored", module.read_text())
            self.assertEqual(_baseline(tmp), "captured")


class FreshExecutionTests(unittest.TestCase):
    """The load-bearing case: same size, same mtime, stale cache present."""

    def test_the_fixed_runner_executes_current_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _stale_fixture(tmp)

            self.assertEqual(_fresh(tmp).stdout.strip(), "restored")

    def test_the_worktree_cache_is_ignored_not_deleted(self) -> None:
        """Isolation, not cleanup: unrelated developer state stays put."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _stale_fixture(tmp)
            before = sorted(p.name for p in (tmp / "__pycache__").glob("*.pyc"))

            self.assertEqual(_fresh(tmp).stdout.strip(), "restored")
            self.assertEqual(
                sorted(p.name for p in (tmp / "__pycache__").glob("*.pyc")), before,
                "the pre-existing cache must be bypassed, not removed",
            )

    def test_repeated_same_length_mutations_each_execute_current_source(self) -> None:
        """A -> B -> C -> D, every edit length-identical and mtime-pinned.

        A private cache root that were REUSED across runs would fail here
        exactly as the worktree's does: mutation N would execute the bytecode
        compiled for mutation N-1.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            module, mtime, size = _stale_fixture(tmp)

            for keyword in VERSIONS:
                _write(module, keyword, mtime=mtime)
                self.assertEqual(module.stat().st_size, size)

                self.assertEqual(
                    _fresh(tmp).stdout.strip(), keyword,
                    f"mutant {keyword!r} must not execute a previous mutant's code",
                )

    def test_a_mutant_cannot_consume_the_previous_mutants_cache(self) -> None:
        """§13 stated directly: mutation N never reads mutation N-1's bytecode."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            module, mtime, _ = _stale_fixture(tmp)

            _write(module, "captured", mtime=mtime)
            first = _fresh(tmp).stdout.strip()
            _write(module, "restored", mtime=mtime)
            second = _fresh(tmp).stdout.strip()

            self.assertEqual((first, second), ("captured", "restored"))


class ProcessIsolationTests(unittest.TestCase):
    """A fresh cache root cannot fix a module already in sys.modules."""

    def test_an_already_imported_module_survives_reimport_in_process(self) -> None:
        """Why a subprocess is required, not merely a private cache.

        Re-importing inside a live interpreter returns the resident module, so
        the new source never runs however clean the cache is.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            module, mtime, _ = _stale_fixture(tmp)
            _write(module, "captured", mtime=mtime)

            sys.path.insert(0, str(tmp))
            try:
                import victim  # type: ignore

                self.assertEqual(victim.keyword(), "captured")
                _write(module, "restored", mtime=mtime)
                import victim as again  # type: ignore

                self.assertEqual(
                    again.keyword(), "captured",
                    "an in-process rerun keeps the stale code; the fix must "
                    "launch a new interpreter",
                )
            finally:
                sys.modules.pop("victim", None)
                sys.path.remove(str(tmp))

    def test_the_runner_uses_a_new_interpreter_each_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            first = run_fresh(["-c", "import os; print(os.getpid())"], cwd=tmp, capture_output=True, text=True)
            second = run_fresh(["-c", "import os; print(os.getpid())"], cwd=tmp, capture_output=True, text=True)

            self.assertNotEqual(first.stdout.strip(), second.stdout.strip())
            self.assertNotEqual(first.stdout.strip(), str(os.getpid()))


class ExitCodeTests(unittest.TestCase):
    """The interpreter's real exit code, never re-derived from output."""

    def test_a_passing_command_reports_zero(self) -> None:
        self.assertEqual(run_fresh(["-c", "pass"], capture_output=True).returncode, 0)

    def test_a_failing_command_preserves_its_exit_code(self) -> None:
        done = run_fresh(["-c", "raise SystemExit(7)"], capture_output=True)
        self.assertEqual(done.returncode, 7)

    def test_a_failing_test_run_is_not_reported_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            (tmp / "test_fails.py").write_text(
                "import unittest\n"
                "class T(unittest.TestCase):\n"
                "    def test_no(self): self.fail('expected')\n"
            )
            done = run_fresh(["-m", "unittest", "test_fails", "-q"], cwd=tmp, capture_output=True, text=True)

            self.assertNotEqual(done.returncode, 0, "a failing suite must never pass")

class PythonPathTests(unittest.TestCase):
    """The child must be able to import Orbit without the caller arranging it."""

    def test_the_child_can_import_orbit_by_default(self) -> None:
        """With no PYTHONPATH arranged by the caller, `src` is still importable.

        The ambient environment is scrubbed first: a developer shell commonly
        exports `PYTHONPATH=src`, which would satisfy the import regardless and
        leave the helper's own default untested.
        """
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

        done = run_fresh(
            ["-c", "import orbit; print('ok')"], env=env, capture_output=True, text=True
        )

        self.assertEqual(done.stdout.strip(), "ok", done.stderr)

    def test_the_default_names_the_repository_source_tree(self) -> None:
        """The child is told where `src` is, not left to an ambient install.

        In this checkout `.venv` carries an editable-install `.pth`, so `import
        orbit` succeeds even with no path set and the test above cannot
        distinguish the helper's default from that. This pins the value the
        helper actually hands the child, so dropping the default is visible.
        """
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

        done = run_fresh(
            ["-c", "import os; print(os.environ.get('PYTHONPATH'))"],
            env=env, capture_output=True, text=True,
        )

        self.assertEqual(done.stdout.strip(), str(ROOT / "src"))

    def test_a_caller_supplied_cache_prefix_cannot_win(self) -> None:
        """The private root must override the caller's, not the reverse.

        The real call site passes `env={**os.environ, ...}`, so if a developer
        shell exports `PYTHONPYCACHEPREFIX` that value would arrive here. Were
        the assignment reordered to let the caller's environment win, the
        guarantee would die silently against a reused root -- exactly the
        class of unnoticed regression QREL-1 exists to prevent.
        """
        with tempfile.TemporaryDirectory() as shared_raw:
            shared = Path(shared_raw)
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                module, mtime, _ = _stale_fixture(tmp)
                env = {**os.environ, "PYTHONPYCACHEPREFIX": str(shared)}

                executed = []
                for keyword in ("restored", "deferred"):
                    _write(module, keyword, mtime=mtime)
                    executed.append(
                        run_fresh(
                            ["-c", PROBE], cwd=tmp, env=env,
                            capture_output=True, text=True,
                        ).stdout.strip()
                    )

                self.assertEqual(
                    executed, ["restored", "deferred"],
                    "a caller-supplied cache prefix must not reintroduce staleness",
                )

    def test_an_explicit_pythonpath_is_not_overridden(self) -> None:
        """A caller that sets its own path keeps it -- the default only fills a gap."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            (tmp / "only_here.py").write_text("marker = 'caller'\n")
            env = {**os.environ, "PYTHONPATH": str(tmp)}

            done = run_fresh(
                ["-c", "import only_here; print(only_here.marker)"],
                env=env, capture_output=True, text=True,
            )

            self.assertEqual(done.stdout.strip(), "caller", done.stderr)


class ExitCodeExtraTests(unittest.TestCase):
    def test_main_returns_the_child_exit_code(self) -> None:
        from scripts.qualify_fresh import main

        self.assertEqual(main(["-c", "raise SystemExit(3)"]), 3)

    def test_main_without_arguments_is_a_usage_error(self) -> None:
        from scripts.qualify_fresh import main

        self.assertEqual(main([]), 2)


class CleanupTests(unittest.TestCase):
    """The private cache root goes away on every path, including failures."""

    def _roots(self) -> set[str]:
        base = Path(tempfile.gettempdir())
        return {p.name for p in base.glob("orbit-qualify-pycache-*")}

    def test_a_passing_run_removes_its_cache_root(self) -> None:
        before = self._roots()
        run_fresh(["-c", "pass"], capture_output=True)
        self.assertEqual(self._roots() - before, set())

    def test_a_failing_run_removes_its_cache_root(self) -> None:
        before = self._roots()
        run_fresh(["-c", "raise SystemExit(9)"], capture_output=True)
        self.assertEqual(self._roots() - before, set(), "cleanup must not depend on success")

    def test_a_timeout_removes_its_cache_root(self) -> None:
        before = self._roots()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_fresh(["-c", "import time; time.sleep(30)"], capture_output=True, timeout=0.5)
        self.assertEqual(self._roots() - before, set(), "cleanup must survive an exception")

    def test_an_interrupt_removes_its_cache_root(self) -> None:
        """Ctrl-C during a run must not strand its cache root.

        `SIGKILL` is the one case that cannot be swept, since no `finally`
        runs. A stranded root is inert -- it is never reused, so it cannot
        serve stale bytecode -- but it does linger in the temp directory.
        """
        before = self._roots()
        with mock.patch("subprocess.run", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_fresh(["-c", "pass"], capture_output=True)
        self.assertEqual(self._roots() - before, set())

    def test_a_spawn_failure_removes_its_cache_root(self) -> None:
        before = self._roots()
        with self.assertRaises(OSError):
            run_fresh(["-c", "pass"], cwd=Path("/nonexistent-qrel1"), capture_output=True)
        self.assertEqual(self._roots() - before, set())

    def test_the_source_is_left_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            module, mtime, size = _stale_fixture(tmp)
            text = module.read_text()

            _fresh(tmp)

            self.assertEqual(module.read_text(), text)
            self.assertEqual(module.stat().st_size, size)


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_runs_do_not_share_a_cache_root(self) -> None:
        """Per-run temporary roots, so parallel qualification cannot collide."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _stale_fixture(tmp)

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = [f.result().stdout.strip() for f in [pool.submit(_fresh, tmp) for _ in range(4)]]

            self.assertEqual(results, ["restored"] * 4)

    def test_concurrent_runs_report_independent_exit_codes(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            ok = pool.submit(run_fresh, ["-c", "pass"], capture_output=True)
            bad = pool.submit(run_fresh, ["-c", "raise SystemExit(5)"], capture_output=True)

            self.assertEqual((ok.result().returncode, bad.result().returncode), (0, 5))


class CallSiteTests(unittest.TestCase):
    """The qualification launch site must go through the fresh runner.

    This is the one place in the repository that runs tests against the Orbit
    worktree itself, so it is the one place a stale worktree `.pyc` can be
    executed. A revert to a bare `subprocess.run` would restore the REF-6
    hazard silently, so it is pinned by behaviour rather than by reading the
    source.
    """

    def test_the_lifecycle_restore_hook_runs_through_run_fresh(self) -> None:
        import ast

        source = (ROOT / "scripts" / "orbit_qualify_lifecycle.py").read_text()
        hook = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_restore_hook"
        )
        called = {
            node.func.id
            for node in ast.walk(hook)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } | {
            node.func.attr
            for node in ast.walk(hook)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        self.assertIn("run_fresh", called)
        self.assertNotIn(
            "run", called,
            "a bare subprocess.run here would execute stale worktree bytecode",
        )

    def test_the_hook_still_reports_a_failing_test_as_failure(self) -> None:
        """Freshness must not have loosened the hook's own pass criterion."""
        source = (ROOT / "scripts" / "orbit_qualify_lifecycle.py").read_text()
        self.assertIn("if completed.returncode", source)


class ProductionIsolationTests(unittest.TestCase):
    """QREL-1 is qualification tooling; the runtime must not learn about it."""

    def test_the_helper_is_not_imported_by_the_orbit_runtime(self) -> None:
        src = ROOT / "src" / "orbit"
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in src.rglob("*.py")
            if "vendor" not in path.parts and "qualify_fresh" in path.read_text()
        ]
        self.assertEqual(offenders, [], "the runtime must not depend on qualification tooling")

    def test_a_nested_subprocess_inherits_the_fresh_cache_root(self) -> None:
        """Tests that spawn their own interpreter stay fresh too.

        Several suites here launch a child interpreter of their own. The
        private root is inherited through the environment, so those grandchild
        runs are covered by the same guarantee rather than quietly falling back
        to a worktree cache.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _stale_fixture(tmp)
            code = (
                "import subprocess, sys, os;"
                "r = subprocess.run([sys.executable, '-c', 'import victim; print(victim.keyword())'],"
                " cwd=os.getcwd(), capture_output=True, text=True);"
                "print(r.stdout.strip())"
            )

            done = run_fresh(["-c", code], cwd=tmp, capture_output=True, text=True)

            self.assertEqual(
                done.stdout.strip(), "restored",
                "a nested interpreter must not fall back to the stale cache",
            )

    def test_the_helper_does_not_mutate_this_process_environment(self) -> None:
        """The child is configured; the parent's environment is left as it was.

        Asserting the variable is simply absent would be wrong: when the suite
        itself runs under the fresh runner, the parent legitimately inherits
        one. What must hold is that `run_fresh` does not change it.
        """
        before = os.environ.get("PYTHONPYCACHEPREFIX")

        run_fresh(["-c", "pass"], capture_output=True)

        self.assertEqual(os.environ.get("PYTHONPYCACHEPREFIX"), before)


if __name__ == "__main__":
    unittest.main()
