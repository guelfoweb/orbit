"""The analysis sandbox has to hold against the program it is running.

These cross the real bubblewrap boundary rather than mocking it: a mocked
sandbox proves the call shape and nothing about isolation, and isolation is
the entire point of the module. Every fixture is benign -- the adversarial
cases are ordinary Python doing things the sandbox must refuse.
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.analysis_sandbox import (
    ALLOWED_COMMANDS,
    MAX_CODE_CHARS,
    SandboxUnavailable,
    execute_analysis,
    sandbox_preflight,
    validate_code,
)

BWRAP_AVAILABLE = shutil.which("/usr/bin/bwrap") is not None
FIXTURE_TEXT = "alpha\nbeta\ngamma\nSECRET-MARKER-7\n"


def requires_bwrap(test):
    return unittest.skipUnless(BWRAP_AVAILABLE, "bubblewrap not installed")(test)


class SandboxTestBase(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._dir = tempfile.TemporaryDirectory(prefix="orbit-sandbox-test-")
        self.tmp = Path(self._dir.name)
        self.fixture = self.tmp / "input.txt"
        self.fixture.write_text(FIXTURE_TEXT, encoding="utf-8")
        self.addCleanup(self._dir.cleanup)

    def run_code(self, code: str, **kwargs):
        return execute_analysis(source_path=self.fixture, code=code, **kwargs)


class CodeValidationTest(unittest.TestCase):
    """Rejections that happen before anything is executed."""

    def test_valid_code_is_returned(self) -> None:
        self.assertEqual(validate_code("print(1)"), "print(1)")

    def test_oversized_code_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_code("#" * (MAX_CODE_CHARS + 1))

    def test_malformed_syntax_keeps_the_interpreter_diagnostic(self) -> None:
        with self.assertRaises(ValueError) as raised:
            validate_code("def (:")
        self.assertIn("SyntaxError", str(raised.exception))

    def test_empty_and_nul_and_non_string_rejected(self) -> None:
        for bad in ("", "   ", "print(1)\x00"):
            with self.subTest(code=repr(bad)):
                with self.assertRaises(ValueError):
                    validate_code(bad)
        with self.assertRaises(ValueError):
            validate_code(b"print(1)")  # type: ignore[arg-type]


class PreflightTest(unittest.TestCase):
    def test_missing_bwrap_executes_nothing(self) -> None:
        with self.assertRaises(SandboxUnavailable):
            sandbox_preflight(bwrap_path="/nonexistent/bwrap-does-not-exist")

    @requires_bwrap
    def test_preflight_passes_on_this_host(self) -> None:
        sandbox_preflight()

    def test_execute_refuses_when_sandbox_is_unavailable(self) -> None:
        import tempfile

        from orbit.runtime import analysis_sandbox

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "in.txt"
            src.write_text("x", encoding="utf-8")
            original = analysis_sandbox.BWRAP
            analysis_sandbox.BWRAP = "/nonexistent/bwrap-does-not-exist"
            try:
                with self.assertRaises(SandboxUnavailable):
                    analysis_sandbox.execute_analysis(source_path=src, code="print(1)")
            finally:
                analysis_sandbox.BWRAP = original


@requires_bwrap
class SandboxBehaviourTest(SandboxTestBase):
    def test_normal_transformation_succeeds(self) -> None:
        result = self.run_code(
            "data = open('/workspace/input').read()\n"
            "print('LINES:', len(data.splitlines()))\n"
        )
        self.assertEqual(result.status, "ok", result.stderr)
        self.assertIn("LINES: 4", result.stdout)
        self.assertEqual(result.exit_status, 0)

    def test_input_bytes_are_exact_and_hashed(self) -> None:
        import hashlib

        result = self.run_code("print(open('/workspace/input').read(), end='')")
        self.assertEqual(result.stdout, FIXTURE_TEXT)
        self.assertEqual(
            result.input_sha256, hashlib.sha256(FIXTURE_TEXT.encode()).hexdigest()
        )

    def test_input_is_read_only(self) -> None:
        result = self.run_code(
            "try:\n"
            "    open('/workspace/input', 'w').write('tampered')\n"
            "    print('WROTE')\n"
            "except OSError as exc:\n"
            "    print('DENIED', type(exc).__name__)\n"
        )
        self.assertIn("DENIED", result.stdout)
        self.assertNotIn("WROTE", result.stdout)
        # And the real file on the host is untouched.
        self.assertEqual(self.fixture.read_text(encoding="utf-8"), FIXTURE_TEXT)

    def test_workspace_is_writable_and_artifacts_are_hashed(self) -> None:
        import hashlib

        payload = "derived-content"
        result = self.run_code(
            f"open('/workspace/work/out.txt','w').write({payload!r})\nprint('OK')\n"
        )
        self.assertEqual(result.status, "ok", result.stderr)
        self.assertEqual(len(result.artifacts), 1)
        artifact = result.artifacts[0]
        self.assertEqual(artifact.name, "out.txt")
        self.assertEqual(artifact.sha256, hashlib.sha256(payload.encode()).hexdigest())

    def test_host_files_are_unreachable(self) -> None:
        result = self.run_code(
            "import os\n"
            "for probe in ('/etc/passwd', '/home', '/root'):\n"
            "    print(probe, os.path.exists(probe))\n"
        )
        self.assertIn("/etc/passwd False", result.stdout)
        self.assertIn("/home False", result.stdout)

    def test_project_files_are_unreachable(self) -> None:
        result = self.run_code(
            f"import os\nprint('PROJECT', os.path.exists({str(ROOT)!r}))\n"
        )
        self.assertIn("PROJECT False", result.stdout)

    def test_network_is_denied(self) -> None:
        result = self.run_code(
            "import socket\n"
            "try:\n"
            "    s = socket.socket()\n"
            "    s.settimeout(2)\n"
            "    s.connect(('1.1.1.1', 80))\n"
            "    print('CONNECTED')\n"
            "except OSError as exc:\n"
            "    print('DENIED', type(exc).__name__)\n"
        )
        self.assertIn("DENIED", result.stdout)
        self.assertNotIn("CONNECTED", result.stdout)

    def test_no_external_binaries_are_mounted(self) -> None:
        self.assertEqual(ALLOWED_COMMANDS, frozenset())
        result = self.run_code(
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['grep', '--version'], capture_output=True)\n"
            "    print('RAN')\n"
            "except Exception as exc:\n"
            "    print('DENIED', type(exc).__name__)\n"
        )
        self.assertIn("DENIED", result.stdout)

    def test_shell_escape_fails(self) -> None:
        result = self.run_code(
            "import os\nprint('RC', os.system('echo escaped > /tmp/escape.txt'))\n"
        )
        self.assertNotIn("RC 0", result.stdout)

    def test_writes_outside_workspace_never_reach_the_host(self) -> None:
        # The sandbox root is a tmpfs, so a write to "/" succeeds inside and is
        # then discarded. What must hold is that nothing lands on the host and
        # nothing becomes a captured artifact -- not that every write errors.
        result = self.run_code(
            "import os\n"
            "for target in ('/etc/x', '/usr/x', '/out.txt'):\n"
            "    try:\n"
            "        open(target, 'w').write('x')\n"
            "        print('WROTE', target)\n"
            "    except OSError:\n"
            "        print('DENIED', target)\n"
        )
        for target in (Path("/etc/x"), Path("/usr/x"), Path("/out.txt")):
            self.assertFalse(target.exists(), f"{target} escaped to the host")
        self.assertEqual(result.artifacts, (), "only /workspace/work may yield artifacts")

    def test_symlink_escape_is_rejected(self) -> None:
        result = self.run_code(
            "import os\nos.symlink('/etc/passwd', '/workspace/work/link')\nprint('LINKED')\n"
        )
        # Either the link cannot resolve to anything useful, or the scratch
        # scan refuses it -- but it must never be captured as an artifact.
        self.assertTrue(
            result.bound_exceeded is not None or not result.artifacts,
            f"symlink must not become an artifact: {result.artifacts}",
        )

    def test_timeout_is_enforced(self) -> None:
        result = self.run_code("import time\nwhile True:\n    time.sleep(0.1)\n")
        self.assertEqual(result.status, "timeout")
        self.assertLess(result.duration_seconds, 40)

    def test_cpu_spin_is_bounded(self) -> None:
        result = self.run_code("x = 0\nwhile True:\n    x += 1\n")
        self.assertIn(result.status, {"timeout", "bounded", "error"})

    def test_memory_is_bounded(self) -> None:
        result = self.run_code(
            "try:\n"
            "    blob = bytearray(4 * 1024 * 1024 * 1024)\n"
            "    print('ALLOCATED')\n"
            "except MemoryError:\n"
            "    print('DENIED MemoryError')\n"
        )
        self.assertNotIn("ALLOCATED", result.stdout)

    def test_excessive_file_creation_is_bounded(self) -> None:
        result = self.run_code(
            "for i in range(200):\n"
            "    open(f'/workspace/work/f{i}.txt','w').write('x')\n"
            "print('DONE')\n"
        )
        self.assertIsNotNone(result.bound_exceeded)

    def test_large_stdout_is_truncated_not_unbounded(self) -> None:
        result = self.run_code("print('A' * (200 * 1024))")
        self.assertTrue(result.truncated or result.bound_exceeded)
        self.assertLessEqual(len(result.stdout.encode()), 64 * 1024)

    def test_environment_is_cleared(self) -> None:
        result = self.run_code(
            "import os\nprint('KEYS', sorted(os.environ))\n"
        )
        self.assertNotIn("SSH_AUTH_SOCK", result.stdout)
        self.assertNotIn("AWS_", result.stdout)

    def test_scratch_does_not_leak_between_actions(self) -> None:
        first = self.run_code("open('/workspace/work/left.txt','w').write('x')\nprint('one')\n")
        self.assertEqual(first.status, "ok", first.stderr)
        second = self.run_code(
            "import os\nprint('LEFTOVER', os.path.exists('/workspace/work/left.txt'))\n"
        )
        self.assertIn("LEFTOVER False", second.stdout)

    def test_temporary_directories_do_not_accumulate(self) -> None:
        # Explicit cleanup is defence in depth: TemporaryDirectory also removes
        # itself via a GC finalizer, so removing the explicit call is not
        # observable from outside. What is worth pinning is the property that
        # matters -- actions leave nothing behind on the host.
        import tempfile

        root = Path(tempfile.gettempdir())
        pattern = "orbit-analysis-*"
        before = {p.name for p in root.glob(pattern)}
        for _ in range(3):
            self.run_code("print('tick')")
        after = {p.name for p in root.glob(pattern)}
        self.assertEqual(
            after - before, set(), "each action must remove its temporary directories"
        )

    def test_nonzero_exit_is_classified_as_error(self) -> None:
        result = self.run_code("raise SystemExit(3)")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.exit_status, 3)

    def test_code_hash_is_recorded(self) -> None:
        import hashlib

        code = "print('hash me')"
        result = self.run_code(code)
        self.assertEqual(result.code_sha256, hashlib.sha256(code.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()


class SandboxConfigurationContractTest(unittest.TestCase):
    """The security contract, asserted on the invocation Orbit builds.

    Behavioural tests prove the sandbox holds on *this* host. These prove the
    policy is still being requested at all -- a flag can be dropped while some
    unrelated protection masks the loss locally, and that must fail loudly
    here rather than survive until a host where nothing masks it.
    """

    REQUIRED_FLAGS = (
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--assert-userns-disabled",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    )

    def build(self):
        from orbit.runtime.analysis_sandbox import (
            SOURCE_MOUNT,
            WORK_MOUNT,
            _sandbox_command,
        )

        argv = _sandbox_command(
            Path("/srv/input.bin"),
            Path("/srv/scratch"),
            Path("/srv/program/main.py"),
            "/usr/bin/python3.12",
        )
        return argv, SOURCE_MOUNT, WORK_MOUNT

    def test_required_isolation_flags_are_all_requested(self) -> None:
        argv, _src, _work = self.build()
        missing = [flag for flag in self.REQUIRED_FLAGS if flag not in argv]
        self.assertEqual(missing, [], f"missing mandatory isolation flags: {missing}")

    def test_capabilities_are_dropped(self) -> None:
        argv, _src, _work = self.build()
        self.assertIn("--cap-drop", argv, "capability dropping is mandatory")
        self.assertEqual(
            argv[argv.index("--cap-drop") + 1],
            "ALL",
            "capabilities must be dropped wholesale, not selectively",
        )

    def test_source_is_mounted_read_only(self) -> None:
        argv, source_mount, _work = self.build()
        self.assertIn(source_mount, argv)
        index = argv.index(source_mount)
        # bwrap takes `<kind> <host> <sandbox>`, so the kind sits two back.
        self.assertEqual(
            argv[index - 2],
            "--ro-bind",
            "the analysed input must never be host-writable",
        )

    def test_only_scratch_is_host_writable(self) -> None:
        argv, _src, work_mount = self.build()
        writable = [
            argv[i + 2] for i, token in enumerate(argv) if token == "--bind"
        ]
        self.assertEqual(
            writable,
            [work_mount],
            f"exactly one host-backed writable mount is permitted, got {writable}",
        )

    def test_host_root_is_never_bound(self) -> None:
        argv, _src, _work = self.build()
        for i, token in enumerate(argv):
            if token in {"--bind", "--ro-bind"}:
                self.assertNotEqual(argv[i + 1], "/", "the host root must never be bound")
        self.assertIn("--tmpfs", argv)
        self.assertEqual(argv[argv.index("--tmpfs") + 1], "/", "root must be a tmpfs")

    def test_path_cannot_reach_host_executables(self) -> None:
        argv, _src, _work = self.build()
        path_value = argv[argv.index("PATH") + 1]
        self.assertNotIn("/usr/bin", path_value)
        self.assertNotIn("/bin", path_value)

    def test_command_allowlist_stays_empty(self) -> None:
        # No proven workflow needed an external binary; each one added back is
        # attack surface and must arrive with its own evidence.
        self.assertEqual(ALLOWED_COMMANDS, frozenset())

    def test_python_runs_isolated(self) -> None:
        argv, _src, _work = self.build()
        self.assertIn("-I", argv)
        self.assertIn("-S", argv)

    def test_no_unsandboxed_fallback_exists(self) -> None:
        import inspect

        from orbit.runtime import analysis_sandbox

        source = inspect.getsource(analysis_sandbox)
        self.assertIn("sandbox_preflight()", source, "preflight must gate execution")
        self.assertIn("raise SandboxUnavailable", source)

    def test_source_integrity_is_verified_after_the_run(self) -> None:
        import inspect

        from orbit.runtime import analysis_sandbox

        source = inspect.getsource(analysis_sandbox.execute_analysis)
        self.assertIn("source_before", source)
        self.assertIn("read-only input changed", source)

    def test_resource_limits_are_all_applied(self) -> None:
        import inspect

        from orbit.runtime import analysis_sandbox

        source = inspect.getsource(analysis_sandbox._preexec_limits)
        for limit in ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_NPROC"):
            self.assertIn(limit, source, f"{limit} must be enforced")

    def test_bounds_remain_finite(self) -> None:
        from orbit.runtime import analysis_sandbox as s

        self.assertLessEqual(s.ACTION_TIMEOUT_SECONDS, 60)
        self.assertLessEqual(s.CPU_SECONDS, 60)
        self.assertLessEqual(s.ADDRESS_SPACE_BYTES, 2 * 1024 * 1024 * 1024)
        self.assertLessEqual(s.HARD_OUTPUT_BYTES, 8 * 1024 * 1024)
        self.assertLessEqual(s.MAX_CODE_CHARS, 128 * 1024)

    def test_scratch_cleanup_is_wired(self) -> None:
        import inspect

        from orbit.runtime import analysis_sandbox

        source = inspect.getsource(analysis_sandbox.execute_analysis)
        self.assertIn("finally:", source)
        self.assertIn("cleanup()", source)

    def test_symlinks_and_extra_links_are_refused_when_capturing(self) -> None:
        import inspect

        from orbit.runtime import analysis_sandbox

        capture = inspect.getsource(analysis_sandbox._capture_artifacts)
        self.assertIn("O_NOFOLLOW", capture)
        self.assertIn("st_nlink", capture)
        scan = inspect.getsource(analysis_sandbox._scratch_bound_error)
        self.assertIn("S_ISLNK", scan)
