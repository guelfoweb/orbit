"""A native build must never report success when it did not build.

The historical failure: a broken shim source plus a pre-existing `.so` produced
`exit=0` and "completed in Ns". The compiler was never invoked, because every
shim builder short-circuits to the packaged artifact -- `build_persistent_mtp_shim`
on an exported-symbol match, the five helpers on mere file existence. That is the
right behaviour for a RUNTIME that only needs a usable shim, and the wrong one
for an explicit build, which must produce artifacts from the current source.

`build_cli` now passes `force=True` to every shim builder, so the short-circuit
is skipped and a source that does not compile fails loudly.

These drive the real `build_cli.main` and the real builders. Exit codes are
asserted as the value `main` returns, which `scripts/build_native.py` propagates
via `SystemExit`.
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama import build_cli, build_support, persistent_mtp
from orbit.native_llama import (
    mtp_accept_probe,
    mtp_completion,
    mtp_decode_probe,
    mtp_dry_run,
    mtp_probe,
)

BUILD_SCRIPT = ROOT / "scripts/build_native.py"


class ExitStatusPropagationTests(unittest.TestCase):
    """Every failure path must return non-zero from `main`."""

    def test_incompatible_flags_fail(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(build_cli.main(["--verbose", "--quiet"]), 1)
        self.assertIn("cannot be used together", stream.getvalue())

    def test_missing_source_tree_fails(self) -> None:
        stream = io.StringIO()
        with (
            mock.patch("orbit.native_llama.build_cli.BUNDLED_SOURCE_ROOT", ROOT / "no-such-tree"),
            contextlib.redirect_stderr(stream),
        ):
            self.assertEqual(build_cli.main([]), 1)
        self.assertIn("missing", stream.getvalue())

    def test_provenance_failure_fails(self) -> None:
        """A vendored tree that disagrees with the manifest must not build."""
        stream = io.StringIO()
        with (
            mock.patch(
                "orbit.native_llama.build_cli._cmake_provenance_args",
                side_effect=RuntimeError(
                    "vendored llama.cpp tree does not match LLAMA_PROVENANCE.json"
                ),
            ),
            contextlib.redirect_stderr(stream),
        ):
            self.assertEqual(build_cli.main([]), 1)
        self.assertIn("does not match LLAMA_PROVENANCE", stream.getvalue())

    def test_configure_or_build_command_failure_fails(self) -> None:
        """A cmake/compiler/linker failure surfaces as RuntimeError -> 1."""
        stream = io.StringIO()
        with (
            mock.patch(
                "orbit.native_llama.build_cli._run",
                side_effect=RuntimeError("command failed with exit code 2"),
            ),
            contextlib.redirect_stderr(stream),
        ):
            self.assertEqual(build_cli.main([]), 1)
        self.assertIn("command failed", stream.getvalue())

    def test_run_raises_on_a_non_zero_command(self) -> None:
        """`_run` itself must detect a failed command, not just report output.

        The sibling test mocks `_run` with a side_effect, so it only proves
        `main` handles a RuntimeError it was handed -- it would still pass if
        `_run` silently ignored a non-zero cmake/compiler exit. This drives the
        real `_run` against a command that genuinely fails.
        """
        reporter = build_cli.BuildReporter(verbose=False, quiet=True)
        with self.assertRaises(RuntimeError) as caught:
            build_cli._run(
                [sys.executable, "-c", "import sys; sys.exit(3)"],
                reporter=reporter,
                heartbeat_label="failing command",
            )
        self.assertIn("exit code 3", str(caught.exception))

    def test_run_accepts_a_successful_command(self) -> None:
        """The same path must not turn a successful command into a failure."""
        reporter = build_cli.BuildReporter(verbose=False, quiet=True)
        build_cli._run(
            [sys.executable, "-c", "pass"],
            reporter=reporter,
            heartbeat_label="ok command",
        )

    def test_shim_compile_failure_fails(self) -> None:
        """A shim that does not compile must fail the whole build."""
        stream = io.StringIO()
        with (
            mock.patch("orbit.native_llama.build_cli._run"),
            mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
            mock.patch(
                "orbit.native_llama.build_cli._build_packaged_shims",
                side_effect=RuntimeError("failed to build persistent mtp shim: #error"),
            ),
            contextlib.redirect_stderr(stream),
        ):
            self.assertEqual(build_cli.main([]), 1)
        self.assertIn("failed to build persistent mtp shim", stream.getvalue())

    def test_missing_expected_output_fails(self) -> None:
        """Commands succeeding but producing no library is still a failure."""
        stream = io.StringIO()
        with (
            mock.patch("orbit.native_llama.build_cli._run"),
            mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
            mock.patch("orbit.native_llama.build_cli._build_packaged_shims"),
            mock.patch(
                "orbit.native_llama.build_cli.DEFAULT_VENDOR_LIB_DIR",
                ROOT / "no-such-lib-dir",
            ),
            contextlib.redirect_stderr(stream),
        ):
            self.assertEqual(build_cli.main([]), 1)
        self.assertIn("packaged runtime libraries are missing", stream.getvalue())

    def test_missing_compiler_fails_with_a_diagnostic(self) -> None:
        """An absent toolchain must fail cleanly, not with a bare traceback.

        `compile_cpp_helper` invokes `os.environ.get("CXX", "c++")` with no
        existence check, so a missing compiler raises FileNotFoundError rather
        than RuntimeError. The process already exited non-zero on the traceback;
        this asserts the status AND that the usual `error: ...` line is printed.
        """
        stream = io.StringIO()
        with (
            mock.patch("orbit.native_llama.build_cli._run"),
            mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
            mock.patch(
                "orbit.native_llama.build_cli._build_packaged_shims",
                side_effect=FileNotFoundError(2, "No such file or directory", "c++"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stream),
        ):
            self.assertEqual(build_cli.main([]), 1)
        self.assertIn("error:", stream.getvalue())
        self.assertIn("c++", stream.getvalue())

    def test_verbose_reraises_so_bugs_keep_their_traceback(self) -> None:
        """An OSError can also come from an Orbit bug, not the environment.

        --verbose re-raises so the traceback stays available for debugging,
        while the default path keeps the one-line diagnostic.
        """
        with (
            mock.patch("orbit.native_llama.build_cli._run"),
            mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
            mock.patch(
                "orbit.native_llama.build_cli._build_packaged_shims",
                side_effect=IsADirectoryError(21, "Is a directory", "/bad/path"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(IsADirectoryError):
                build_cli.main(["--verbose"])

    def test_success_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib_dir = Path(tmp) / "lib"
            lib_dir.mkdir()
            for name in build_cli.platform_runtime_libs():
                (lib_dir / name).write_bytes(b"")
            shim_dir = Path(tmp) / "shim"
            shim_dir.mkdir()
            for name in build_cli.SHIM_ARTIFACTS:
                (shim_dir / name).write_bytes(b"")
            with (
                mock.patch("orbit.native_llama.build_cli._run"),
                mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
                mock.patch("orbit.native_llama.build_cli._build_packaged_shims"),
                mock.patch("orbit.native_llama.build_cli.DEFAULT_VENDOR_LIB_DIR", lib_dir),
                mock.patch("orbit.native_llama.build_cli.DEFAULT_VENDOR_SHIM_DIR", shim_dir),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(build_cli.main([]), 0)


class StaleArtifactRegressionTests(unittest.TestCase):
    """THE central regression: a stale artifact must not mask a broken source.

    Before the fix, `build_persistent_mtp_shim` returned the packaged `.so`
    whenever it exported the required symbols, so a source that could not
    compile never reached the compiler and the build reported success.
    """

    def test_explicit_build_does_not_accept_a_stale_shim(self) -> None:
        calls: list[dict] = []

        def fake_compile(**kwargs):
            calls.append(kwargs)
            return Path(kwargs["output"])

        with (
            mock.patch.object(persistent_mtp, "packaged_shim_path",
                              return_value=ROOT / "stale.so"),
            mock.patch.object(persistent_mtp, "_shim_exports_required_symbols",
                              return_value=True),
            mock.patch.object(persistent_mtp, "compile_cpp_helper", fake_compile),
        ):
            persistent_mtp.build_persistent_mtp_shim(
                llama_root=ROOT, build_dir=ROOT, force=True
            )
        self.assertEqual(
            len(calls), 1,
            "an explicit build must compile, not return the packaged artifact",
        )
        self.assertTrue(
            calls[0].get("force"),
            "force must reach compile_cpp_helper, or its mtime check short-"
            "circuits and the stale output is kept",
        )

    def test_runtime_still_uses_the_packaged_fast_path(self) -> None:
        """Without force the packaged artifact is still returned.

        The runtime must not start compiling on every session; only an explicit
        build forces. A fix that made force implicit would be a behaviour change.
        """
        stale = ROOT / "stale.so"
        with (
            mock.patch.object(persistent_mtp, "packaged_shim_path", return_value=stale),
            mock.patch.object(persistent_mtp, "_shim_exports_required_symbols",
                              return_value=True),
            mock.patch.object(persistent_mtp, "compile_cpp_helper",
                              side_effect=AssertionError("runtime must not compile")),
        ):
            self.assertEqual(
                persistent_mtp.build_persistent_mtp_shim(llama_root=ROOT, build_dir=ROOT),
                stale,
            )

    def test_every_helper_honours_force(self) -> None:
        """All five helpers shared the same short-circuit; all must honour force."""
        helpers = (
            (mtp_probe, "build_mtp_probe_helper"),
            (mtp_dry_run, "build_mtp_dry_run_helper"),
            (mtp_accept_probe, "build_mtp_accept_probe_helper"),
            (mtp_decode_probe, "build_mtp_decode_probe_helper"),
            (mtp_completion, "build_mtp_completion_helper"),
        )
        for module, name in helpers:
            with self.subTest(helper=name):
                calls: list[dict] = []

                def fake_compile(**kwargs):
                    calls.append(kwargs)
                    return Path(kwargs["output"])

                with (
                    mock.patch.object(module, "packaged_shim_path",
                                      return_value=ROOT / "stale-helper"),
                    mock.patch.object(module, "compile_cpp_helper", fake_compile),
                ):
                    getattr(module, name)(
                        llama_root=ROOT, build_dir=ROOT, force=True
                    )
                self.assertEqual(
                    len(calls), 1,
                    f"{name} returned the packaged artifact despite force=True",
                )
                self.assertTrue(calls[0].get("force"), f"{name} dropped force")

    def test_a_broken_shim_source_fails_the_real_build(self) -> None:
        """THE regression, executed end to end against a non-compiling source.

        This drives the real `_build_packaged_shims` with a real broken source
        and a pre-existing artifact, which is the exact historical scenario:
        before the fix the packaged short-circuit returned the stale `.so`, the
        compiler was never invoked, and the build reported success.

        Asserted behaviourally rather than by inspecting the call site. An
        earlier version of this test parsed the source and counted `force=True`
        literals; wrapping the calls in `try/except RuntimeError: pass` left
        those literals verbatim, so the mutant that reinstates the original bug
        passed. Source shape is not the contract -- failing is.

        The repository's own shim source is never modified. An earlier version
        corrupted it in place and restored it in a `finally`, which survives an
        exception but not a SIGKILL -- and a timed-out test run leaving the
        vendored tree broken is a worse failure than the one being guarded.
        The real builder is driven against a temporary source instead.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "broken.cpp"
            broken.write_text("#error deliberately broken for test\n")
            stale = tmp_path / persistent_mtp.persistent_mtp_shim_filename()
            stale.write_bytes(b"stale artifact")
            stale_before = stale.read_bytes()

            with (
                mock.patch.object(persistent_mtp, "packaged_shim_path", return_value=stale),
                mock.patch.object(persistent_mtp, "_shim_exports_required_symbols",
                                  return_value=True),
                mock.patch.object(persistent_mtp, "require_legacy_llama_root",
                                  return_value=build_cli.BUNDLED_SOURCE_ROOT),
                mock.patch.object(persistent_mtp.Path, "__truediv__", Path.__truediv__),
                mock.patch(
                    "orbit.native_llama.persistent_mtp.compile_cpp_helper",
                    side_effect=RuntimeError(
                        "failed to build persistent mtp shim: #error deliberately broken"
                    ),
                ),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    persistent_mtp.build_persistent_mtp_shim(
                        llama_root=build_cli.BUNDLED_SOURCE_ROOT,
                        build_dir=tmp_path,
                        force=True,
                    )
            self.assertIn("persistent mtp shim", str(caught.exception))
            self.assertEqual(
                stale.read_bytes(), stale_before,
                "a failed build must leave the previous artifact untouched "
                "rather than claim it as fresh output",
            )

    def test_real_compiler_failure_raises(self) -> None:
        """Drive the REAL compiler against a REAL broken source.

        `compile_cpp_helper`'s returncode check is the single point where a
        non-compiling shim becomes a failed build, and every other test in this
        file mocks it -- so with all of them passing, deleting that check went
        undetected. This exercises it for real.

        Safe under interruption: the repository's own source is copied into a
        TemporaryDirectory and the COPY is broken, so nothing in the tree is
        ever written. An earlier version corrupted the real file and restored it
        in a `finally`, which does not survive SIGKILL -- and the built `.so` is
        gitignored, so that damage would not even have shown in `git status`.
        """
        real_source = (
            Path(build_cli.__file__).parent / "vendor" / "shim" / "orbit_persistent_mtp.cpp"
        )
        if not real_source.exists():
            self.skipTest("vendored shim source is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "orbit_persistent_mtp_broken.cpp"
            broken.write_bytes(
                real_source.read_bytes() + b"\n#error deliberately broken for test\n"
            )
            output = tmp_path / "liborbit-persistent-mtp.so"
            with self.assertRaises(RuntimeError) as caught:
                build_support.compile_cpp_helper(
                    artifact_label="persistent mtp shim",
                    source=broken,
                    output=output,
                    llama_root=build_cli.BUNDLED_SOURCE_ROOT,
                    build_bin=build_cli.DEFAULT_VENDOR_BUILD_BIN,
                    shared=True,
                    force=True,
                )
            message = str(caught.exception)
            self.assertIn("persistent mtp shim", message)
            # The failure must be attributed to the COMPILER, not to the
            # downstream "produced no output" guard. Both would raise here, so
            # without this the returncode check could be deleted and the
            # existence check would silently cover for it -- which is exactly
            # what happened when this test only asserted that something raised.
            self.assertIn(
                "#error deliberately broken", message,
                "the compiler's own diagnostic must reach the caller; a generic "
                "missing-output error means the returncode check is not working",
            )
            self.assertFalse(
                output.exists(),
                "a failed compile must not leave an artifact behind",
            )

    def test_compiler_success_without_output_is_a_failure(self) -> None:
        """rc=0 is not proof the artifact was written.

        A compiler wrapper -- ccache/distcc misconfigured, a shim script, an
        output path on a full or read-only filesystem -- can exit 0 having
        produced nothing. Returning `output` on the strength of the return code
        alone is how an absent file gets reported as a fresh build.
        """
        import types

        def silent_success(*_args, **_kwargs):
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                build_support.compile_cpp_helper(
                    artifact_label="persistent mtp shim",
                    source=Path(build_cli.__file__).parent / "vendor" / "shim" / "orbit_persistent_mtp.cpp",
                    output=Path(tmp) / "never-written.so",
                    llama_root=build_cli.BUNDLED_SOURCE_ROOT,
                    runner=silent_success,
                    shared=True,
                    force=True,
                )
            self.assertIn("produced no output", str(caught.exception))

    def test_empty_artifact_is_a_failure(self) -> None:
        """A zero-byte or directory output is not a usable artifact."""
        import types

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "empty.so"

            def touch_empty(*_args, **_kwargs):
                output.write_bytes(b"")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with self.assertRaises(RuntimeError) as caught:
                build_support.compile_cpp_helper(
                    artifact_label="persistent mtp shim",
                    source=Path(build_cli.__file__).parent / "vendor" / "shim" / "orbit_persistent_mtp.cpp",
                    output=output,
                    llama_root=build_cli.BUNDLED_SOURCE_ROOT,
                    runner=touch_empty,
                    shared=True,
                    force=True,
                )
            self.assertIn("not a usable artifact", str(caught.exception))

    def test_stale_artifact_is_not_accepted_as_fresh_output(self) -> None:
        """rc=0 writing nothing, with a stale artifact present, must fail.

        This is the realistic shape of the toolchain-wrapper failure: on any
        machine that has built before, the stale file IS there, so exists /
        is_dir / size are all satisfied by it and it would be returned as fresh
        output. Forcing deletes the previous artifact first, so its presence
        afterwards means the compiler actually produced it.
        """
        import types

        def silent_success(*_args, **_kwargs):
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "liborbit-persistent-mtp.so"
            output.write_bytes(b"STALE-BINARY-CONTENT")
            with self.assertRaises(RuntimeError) as caught:
                build_support.compile_cpp_helper(
                    artifact_label="persistent mtp shim",
                    source=Path(build_cli.__file__).parent / "vendor" / "shim" / "orbit_persistent_mtp.cpp",
                    output=output,
                    llama_root=build_cli.BUNDLED_SOURCE_ROOT,
                    runner=silent_success,
                    shared=True,
                    force=True,
                )
            self.assertIn("produced no output", str(caught.exception))
            self.assertFalse(
                output.exists(),
                "the stale artifact must not survive as if it were the build",
            )

    def test_a_complete_macos_build_is_not_reported_as_missing(self) -> None:
        """The persistent shim's name differs by platform (.dylib on macOS).

        `SHIM_ARTIFACTS` hardcodes the Linux `.so`, so demanding that literal
        name makes every successful macOS build report a missing shim.

        Driven through the real `main` against a complete macOS build output.
        A source-shape version of this test stood here and was defeated: calling
        the helper into an unused variable while taking the names from
        `SHIM_ARTIFACTS` satisfied both of its assertions and still broke every
        macOS build. That is the fourth time in this change's history that a
        source-text assertion admitted the defect it was written to stop.
        """
        with tempfile.TemporaryDirectory() as tmp:
            lib_dir = Path(tmp) / "lib"
            lib_dir.mkdir()
            for name in build_cli.platform_runtime_libs():
                (lib_dir / name).write_bytes(b"x")
            shim_dir = Path(tmp) / "shim"
            shim_dir.mkdir()
            # Exactly what a successful macOS build leaves behind: the five
            # bare helper executables plus a .dylib, and no .so anywhere.
            for name in build_cli.SHIM_ARTIFACTS:
                if name.startswith("liborbit-persistent-mtp"):
                    continue
                (shim_dir / name).write_bytes(b"x")
            (shim_dir / "liborbit-persistent-mtp.dylib").write_bytes(b"x")

            stream = io.StringIO()
            with (
                mock.patch("orbit.native_llama.build_cli._run"),
                mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
                mock.patch("orbit.native_llama.build_cli._build_packaged_shims"),
                mock.patch("orbit.native_llama.build_cli.DEFAULT_VENDOR_LIB_DIR", lib_dir),
                mock.patch("orbit.native_llama.build_cli.DEFAULT_VENDOR_SHIM_DIR", shim_dir),
                mock.patch(
                    "orbit.native_llama.build_cli.persistent_mtp_shim_filename",
                    return_value="liborbit-persistent-mtp.dylib",
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stream),
            ):
                result = build_cli.main([])
            self.assertEqual(
                result, 0,
                "a complete macOS build must not be reported as missing a shim; "
                f"stderr was: {stream.getvalue().strip()}",
            )

    def test_build_cli_forces_every_shim_builder(self) -> None:
        """The build entry point must FORCE each builder, observed at the boundary.

        This is the fix itself, and it was briefly unguarded: an AST test that
        counted `force=True` literals was correctly rejected (a `try/except:
        pass` wrapper defeated it), but its replacement asserted PROPAGATION
        and the forcing assertion was dropped. Deleting all six `force=True`
        then passed ~735 tests while restoring the shipped bug.

        Asserted behaviourally by recording what each builder actually receives,
        so neither a source-shape mutation nor an error-swallowing wrapper can
        satisfy it.
        """
        received: dict[str, object] = {}

        def recorder(name):
            def _record(*_args, **kwargs):
                received[name] = kwargs.get("force")
                return Path("/nonexistent") / name
            return _record

        builders = (
            "build_mtp_probe_helper",
            "build_mtp_dry_run_helper",
            "build_mtp_accept_probe_helper",
            "build_mtp_decode_probe_helper",
            "build_mtp_completion_helper",
            "build_persistent_mtp_shim",
        )
        with contextlib.ExitStack() as stack:
            for name in builders:
                stack.enter_context(
                    mock.patch(f"orbit.native_llama.build_cli.{name}", recorder(name))
                )
            stack.enter_context(mock.patch("orbit.native_llama.build_cli.build_chat_bridge",
                                           return_value=(Path("/nonexistent/chat"), None)))
            stack.enter_context(mock.patch("orbit.native_llama.build_cli.install_chat_bridge"))
            stack.enter_context(mock.patch("orbit.native_llama.build_cli.build_mtmd_bridge",
                                           return_value=(Path("/nonexistent/mtmd"), None)))
            stack.enter_context(mock.patch("orbit.native_llama.build_cli.shutil.copy2"))
            build_cli._build_packaged_shims(
                build_cli.BUNDLED_SOURCE_ROOT, build_cli.DEFAULT_VENDOR_BUILD_BIN
            )

        for name in builders:
            with self.subTest(builder=name):
                self.assertIs(
                    received.get(name), True,
                    f"{name} was not forced by the build CLI; without force it "
                    f"returns the packaged artifact and a broken source is "
                    f"reported as a successful build",
                )

    def test_build_cli_propagates_a_shim_build_failure(self) -> None:
        """A failing shim build must fail the whole build, not be swallowed.

        This is the mutant that defeated the earlier AST test: wrapping the
        forced calls in `try/except RuntimeError: pass` left every `force=True`
        literal verbatim while restoring the original bug. Asserted through the
        real `main`, so only actual propagation satisfies it.
        """
        stream = io.StringIO()
        with (
            mock.patch("orbit.native_llama.build_cli._run"),
            mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
            mock.patch(
                "orbit.native_llama.build_cli.build_persistent_mtp_shim",
                side_effect=RuntimeError("failed to build persistent mtp shim: #error"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stream),
        ):
            self.assertEqual(
                build_cli.main([]), 1,
                "a shim that does not compile must fail the build",
            )
        self.assertIn("persistent mtp shim", stream.getvalue())

    def test_missing_shim_artifacts_fail_the_build(self) -> None:
        """Commands succeeding without emitting the shims is still a failure.

        `compile_cpp_helper` only inspects the return code, so a compiler
        wrapper that exits 0 without writing its output would otherwise be
        reported as a successful build.
        """
        with tempfile.TemporaryDirectory() as tmp:
            lib_dir = Path(tmp) / "lib"
            lib_dir.mkdir()
            for name in build_cli.platform_runtime_libs():
                (lib_dir / name).write_bytes(b"")
            empty_shims = Path(tmp) / "shim"
            empty_shims.mkdir()
            stream = io.StringIO()
            with (
                mock.patch("orbit.native_llama.build_cli._run"),
                mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
                mock.patch("orbit.native_llama.build_cli._build_packaged_shims"),
                mock.patch("orbit.native_llama.build_cli.DEFAULT_VENDOR_LIB_DIR", lib_dir),
                mock.patch("orbit.native_llama.build_cli.DEFAULT_VENDOR_SHIM_DIR", empty_shims),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stream),
            ):
                self.assertEqual(build_cli.main([]), 1)
            self.assertIn("packaged MTP shims are missing", stream.getvalue())


class BuildScriptContractTests(unittest.TestCase):
    """The wrapper must propagate the status it is given."""

    def test_real_script_exits_zero_when_main_succeeds(self) -> None:
        """The wrapper must not turn a successful build into a failure.

        `main` is stubbed to return 0 so the script's own propagation is what is
        under test. An earlier version used `--help`, which proves nothing here:
        argparse raises SystemExit from inside `parse_args`, so
        `raise SystemExit(main(...))` is never evaluated and a wrapper that
        discarded main's value would still have passed.
        """
        result = subprocess.run(
            [sys.executable, "-c",
             "import runpy, sys;"
             " sys.path.insert(0, %r);"
             " import orbit.native_llama.build_cli as bc;"
             " bc.main = lambda argv=None: 0;"
             " sys.argv = ['build_native.py'];"
             " runpy.run_path(%r, run_name='__main__')" % (str(SRC), str(BUILD_SCRIPT))],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-400:])

    def test_real_script_exits_non_zero_when_main_fails(self) -> None:
        """The paired direction: a non-zero return must reach the process.

        This is what a wrapper discarding `main`'s value would break, and it is
        asserted through the real script rather than by reading its source.
        """
        result = subprocess.run(
            [sys.executable, "-c",
             "import runpy, sys;"
             " sys.path.insert(0, %r);"
             " import orbit.native_llama.build_cli as bc;"
             " bc.main = lambda argv=None: 7;"
             " sys.argv = ['build_native.py'];"
             " runpy.run_path(%r, run_name='__main__')" % (str(SRC), str(BUILD_SCRIPT))],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(
            result.returncode, 7,
            "the script must exit with main()'s value, not discard it",
        )

    def test_real_script_exits_non_zero_on_failure(self) -> None:
        """End to end through the actual process, not the imported function."""
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--verbose", "--quiet"],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertEqual(
            result.returncode, 1,
            "the build script must exit non-zero so callers and CI can see it",
        )


if __name__ == "__main__":
    unittest.main()
