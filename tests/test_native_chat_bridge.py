from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from orbit.native_llama.bindings import ChatBridgeLibrary, LlamaLibrary
from orbit.native_llama.build_support import DEFAULT_VENDOR_BUILD_BIN
from orbit.native_llama.chat_bridge import CHAT_BRIDGE_API_VERSION, chat_bridge_filename, validate_chat_bridge_artifact
from orbit.native_llama.client import _resolve_chat_bridge_path
from orbit.native_llama.native_names import (
    mtmd_bridge_filename,
    platform_runtime_libs,
    runtime_library_filename,
)


class NativeChatBridgeTests(unittest.TestCase):
    def test_llama_runtime_loads_only_lower_level_dependencies_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            for name in platform_runtime_libs():
                (build_bin / name).touch()
            library = LlamaLibrary.__new__(LlamaLibrary)
            library.build_bin = build_bin
            library._handles = []
            loaded: list[str] = []

            def fake_load(path: Path, *, mode: int):
                del mode
                loaded.append(path.name)
                return object()

            with mock.patch("orbit.native_llama.bindings.load_native_cdll", side_effect=fake_load):
                library._load_library(runtime_library_filename("llama"))

        self.assertEqual(
            loaded,
            [
                runtime_library_filename("ggml-base"),
                runtime_library_filename("ggml-cpu"),
                runtime_library_filename("ggml"),
                runtime_library_filename("llama"),
            ],
        )

    def test_llama_runtime_missing_dependency_fails_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            for name in platform_runtime_libs():
                if name != runtime_library_filename("ggml-base"):
                    (build_bin / name).touch()

            with mock.patch("orbit.native_llama.bindings.load_native_cdll") as load:
                with self.assertRaisesRegex(RuntimeError, "incomplete native runtime family"):
                    LlamaLibrary(build_bin)

        load.assert_not_called()

    def test_llama_runtime_rejects_dependency_symlink_outside_family_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_bin = base / "runtime"
            external = base / "external"
            build_bin.mkdir()
            external.mkdir()
            escaped_name = runtime_library_filename("ggml-base")
            for name in platform_runtime_libs():
                target = external / name if name == escaped_name else build_bin / name
                target.touch()
                if name == escaped_name:
                    (build_bin / name).symlink_to(target)

            with mock.patch("orbit.native_llama.bindings.load_native_cdll") as load:
                with self.assertRaisesRegex(RuntimeError, "escapes family root"):
                    LlamaLibrary(build_bin)

        load.assert_not_called()

    def test_bridge_must_be_co_located_with_active_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            paths = SimpleNamespace(build_bin=build_bin)
            self.assertIsNone(_resolve_chat_bridge_path(paths))

            bridge = build_bin / chat_bridge_filename()
            bridge.write_bytes(b"bridge")
            self.assertEqual(_resolve_chat_bridge_path(paths), bridge)

    def test_missing_identity_fails_before_loading_native_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            bridge = build_bin / chat_bridge_filename()
            bridge.write_bytes(b"not-a-library")

            with self.assertRaisesRegex(RuntimeError, "chat bridge identity"):
                ChatBridgeLibrary(build_bin, bridge)

    def test_runtime_library_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            bridge = build_bin / chat_bridge_filename()
            bridge.write_bytes(b"bridge")
            libraries: dict[str, str] = {}
            for name in platform_runtime_libs():
                path = build_bin / name
                path.write_bytes(name.encode())
                libraries[name] = _sha256(path)
            first = next(iter(libraries))
            libraries[first] = "0" * 64
            identity = {"build": {"libraries": libraries}, "artifact_sha256": _sha256(bridge)}
            bridge.with_name(f"{bridge.name}.identity.json").write_text(json.dumps(identity), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "library identity mismatch"):
                validate_chat_bridge_artifact(build_bin, bridge)

    @unittest.skipUnless(
        (DEFAULT_VENDOR_BUILD_BIN / chat_bridge_filename()).exists(),
        "native chat bridge is not built",
    )
    def test_current_native_bridge_exports_supported_stable_api(self) -> None:
        bridge = ChatBridgeLibrary(DEFAULT_VENDOR_BUILD_BIN, DEFAULT_VENDOR_BUILD_BIN / chat_bridge_filename())

        self.assertEqual(bridge.lib.orbit_chat_bridge_api_version(), CHAT_BRIDGE_API_VERSION)
        self.assertEqual(bridge.build_identity["api_version"], CHAT_BRIDGE_API_VERSION)
        self.assertIn("upstream_commit", bridge.build_identity)

    @unittest.skipUnless(
        Path("/proc/self/maps").exists()
        and Path(__file__).resolve().parents[1].joinpath(
            "src/orbit/native_llama/vendor/lib",
            chat_bridge_filename(),
        ).exists(),
        "packaged native chat bridge or Linux process maps are unavailable",
    )
    def test_packaged_bridge_does_not_load_a_second_native_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime = root / "src/orbit/native_llama/vendor/lib"
        script = """
from pathlib import Path
from orbit.native_llama.bindings import ChatBridgeLibrary, LlamaLibrary
from orbit.native_llama.chat_bridge import chat_bridge_filename

runtime = Path(__import__('sys').argv[1])
LlamaLibrary(runtime)
ChatBridgeLibrary(runtime, runtime / chat_bridge_filename())
mapped = {
    line.rsplit(' ', 1)[-1]
    for line in Path('/proc/self/maps').read_text(encoding='utf-8').splitlines()
    if any(marker in line for marker in ('/libllama', '/libggml', '/libmtmd', '/liborbit-'))
}
outside = sorted(path for path in mapped if not path.startswith(str(runtime) + '/'))
if outside:
    raise SystemExit('mixed native runtime: ' + ','.join(outside))
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [sys.executable, "-c", script, str(runtime)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    @unittest.skipUnless(
        Path("/proc/self/maps").exists()
        and Path(__file__).resolve().parents[1].joinpath(
            "src/orbit/native_llama/vendor/lib",
            runtime_library_filename("llama"),
        ).exists(),
        "packaged native runtime or Linux process maps are unavailable",
    )
    @unittest.skipUnless(
        (DEFAULT_VENDOR_BUILD_BIN / mtmd_bridge_filename()).exists()
        and (DEFAULT_VENDOR_BUILD_BIN / runtime_library_filename("mtmd")).exists()
        and Path("/proc/self/maps").exists(),
        "native mtmd bridge or Linux process maps are unavailable",
    )
    def test_mtmd_bridge_cannot_introduce_a_second_runtime_family(self) -> None:
        """The mtmd bridge claims the family like every other entry point.

        `libmtmd` links against llama and ggml, so this bridge pulls a whole
        runtime family in behind it. Verifying the bridge's own artifact
        identity only says those files belong together -- not that they are
        the family this process already runs on. Without a claim a second,
        internally consistent family loads beside the first, which is the
        topology observed in the crashed processes.

        Exercised through `MtmdLibrary` rather than by inspecting the source,
        because the defect this replaces was invisible to tests that only ever
        constructed `LlamaLibrary`.
        """
        root = Path(__file__).resolve().parents[1]
        source = DEFAULT_VENDOR_BUILD_BIN
        script = """
from pathlib import Path
import shutil
import sys
import tempfile
from orbit.native_llama.bindings import LlamaLibrary, MtmdLibrary
from orbit.native_llama.native_names import (
    mtmd_bridge_filename,
    platform_runtime_load_order,
    runtime_library_filename,
)

source = Path(sys.argv[1])
bridge_name = mtmd_bridge_filename()
family = [*platform_runtime_load_order(), runtime_library_filename('mtmd'), bridge_name]
with tempfile.TemporaryDirectory(prefix='orbit-mtmd-family.') as tmp:
    second = Path(tmp) / 'second'
    second.mkdir()
    for name in family:
        # Copy the SONAME spellings too, not just the linker name: the loader
        # asks for `libmtmd.so.0`, so a directory holding only `libmtmd.so`
        # fails to load before the guard is ever consulted -- which would let
        # this test pass for the wrong reason.
        for candidate in sorted(source.glob(name + '*')):
            if candidate.is_file():
                shutil.copy2(candidate.resolve(), second / candidate.name)
        real = source / name
        if real.exists() and not (second / name).exists():
            shutil.copy2(real.resolve(), second / name)
    identity = source / (bridge_name + '.identity.json')
    if identity.exists():
        shutil.copy2(identity, second / identity.name)

    # First family: the one this process is entitled to.
    LlamaLibrary(source)
    try:
        MtmdLibrary(second, second / bridge_name)
    except RuntimeError as exc:
        if 'native runtime family conflict' not in str(exc):
            raise
    else:
        raise SystemExit('mtmd bridge accepted a second native runtime family')

    mapped = {
        line.rsplit(' ', 1)[-1]
        for line in Path('/proc/self/maps').read_text(encoding='utf-8').splitlines()
        if any(m in line for m in ('/libllama', '/libggml', '/libmtmd', '/liborbit-'))
    }
    if any(path.startswith(str(second) + '/') for path in mapped):
        raise SystemExit('second family was mapped before rejection')
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [sys.executable, "-c", script, str(source)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_a_path_alias_is_the_same_family(self) -> None:
        """Two names for one directory are one runtime, not two.

        The guard compares canonical paths. Comparing the paths as written
        would refuse a legitimate load whenever the caller reached the same
        directory through a symlink or a non-normalised path -- a false
        conflict that would look exactly like the real one.
        """
        from orbit.native_llama import bindings

        with tempfile.TemporaryDirectory(prefix="orbit-family-alias.") as tmp:
            real = Path(tmp) / "real"
            real.mkdir()
            alias = Path(tmp) / "alias"
            alias.symlink_to(real, target_is_directory=True)
            indirect = Path(tmp) / "." / "real"

            with mock.patch.object(bindings, "_RUNTIME_FAMILY_ROOT", None):
                bindings._claim_runtime_family(real)
                # Neither spelling is a second family.
                self.assertEqual(bindings._claim_runtime_family(alias), real.resolve())
                self.assertEqual(bindings._claim_runtime_family(indirect), real.resolve())

    def test_an_incomplete_family_is_refused_before_it_is_claimed(self) -> None:
        """A directory missing part of the runtime is not a family.

        Claiming it would bind the process to something that cannot satisfy
        its own dependencies, and the loader would then answer the missing
        pieces from wherever else it could find them.
        """
        from orbit.native_llama import bindings

        with tempfile.TemporaryDirectory(prefix="orbit-family-partial.") as tmp:
            partial = Path(tmp) / "partial"
            partial.mkdir()
            (partial / runtime_library_filename("mtmd")).write_bytes(b"")

            with mock.patch.object(bindings, "_RUNTIME_FAMILY_ROOT", None):
                with self.assertRaises(RuntimeError) as caught:
                    bindings._require_runtime_prefix(
                        partial.resolve(), runtime_library_filename("llama-common")
                    )
                self.assertIn("incomplete native runtime family", str(caught.exception))
                self.assertIsNone(
                    bindings._RUNTIME_FAMILY_ROOT,
                    "an incomplete family must not be claimed",
                )

    @unittest.skipUnless(
        (DEFAULT_VENDOR_BUILD_BIN / mtmd_bridge_filename()).exists()
        and (DEFAULT_VENDOR_BUILD_BIN / runtime_library_filename("mtmd")).exists()
        and Path("/proc/self/maps").exists(),
        "native mtmd bridge or Linux process maps are unavailable",
    )
    def test_mtmd_family_is_loaded_by_path_not_by_search(self) -> None:
        """A search path must not decide which family answers.

        `libmtmd` is an optional member of the family, so the mandatory prefix
        does not name it. Left to the loader, the bridge's own `DT_NEEDED`
        would be answered by whatever the search path offered first -- here a
        foreign copy -- which puts a second family in the process even though
        the claim succeeded.
        """
        root = Path(__file__).resolve().parents[1]
        script = """
from pathlib import Path
import shutil
import sys
import tempfile
from orbit.native_llama.bindings import MtmdLibrary
from orbit.native_llama.native_names import mtmd_bridge_filename, runtime_library_filename

source = Path(sys.argv[1])
foreign = Path(sys.argv[2])
MtmdLibrary(source, source / mtmd_bridge_filename())
mapped = {
    line.rsplit(' ', 1)[-1]
    for line in Path('/proc/self/maps').read_text(encoding='utf-8').splitlines()
    if any(m in line for m in ('/libllama', '/libggml', '/libmtmd', '/liborbit-'))
}
outside = sorted(path for path in mapped if path.startswith(str(foreign) + '/'))
if outside:
    raise SystemExit('foreign family answered the search path: ' + ','.join(outside))
"""
        with tempfile.TemporaryDirectory(prefix="orbit-mtmd-search.") as tmp:
            foreign = Path(tmp) / "foreign"
            foreign.mkdir()
            for candidate in DEFAULT_VENDOR_BUILD_BIN.glob(
                runtime_library_filename("mtmd") + "*"
            ):
                if candidate.is_file():
                    shutil.copy2(candidate.resolve(), foreign / candidate.name)

            env = dict(os.environ)
            env["PYTHONPATH"] = str(root / "src")
            # The condition under test: a search path that offers the same
            # SONAME from somewhere else.
            env["LD_LIBRARY_PATH"] = str(foreign)
            result = subprocess.run(
                [sys.executable, "-c", script, str(DEFAULT_VENDOR_BUILD_BIN), str(foreign)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_a_rejected_family_does_not_replace_the_claimed_one(self) -> None:
        """A refusal must not quietly rebind the process to the family it refused.

        If the conflict path recorded the rejected root, the first rejection
        would move the claim: the next load from the original family would
        then be refused instead, and a third from the rejected family would be
        accepted. The process would end up running the family it just said no
        to, and every later check would agree with it.

        Exercised through the guard directly because no artifact needs to
        exist for the ordering to be wrong.
        """
        from orbit.native_llama import bindings

        with tempfile.TemporaryDirectory(prefix="orbit-claim-order.") as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()

            with mock.patch.object(bindings, "_RUNTIME_FAMILY_ROOT", None):
                claimed = bindings._claim_runtime_family(first)
                self.assertEqual(claimed, first.resolve())

                with self.assertRaises(RuntimeError):
                    bindings._claim_runtime_family(second)

                self.assertEqual(
                    bindings._RUNTIME_FAMILY_ROOT,
                    first.resolve(),
                    "a rejected family replaced the claimed one",
                )
                # The original family is still the one this process may load.
                self.assertEqual(bindings._claim_runtime_family(first), first.resolve())
                with self.assertRaises(RuntimeError):
                    bindings._claim_runtime_family(second)

    def test_process_rejects_a_second_runtime_family(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = root / "src/orbit/native_llama/vendor/lib"
        script = """
from pathlib import Path
import shutil
import sys
import tempfile
from orbit.native_llama.bindings import LlamaLibrary
from orbit.native_llama.native_names import platform_runtime_load_order

source = Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix='orbit-runtime-family.') as tmp:
    roots = [Path(tmp) / 'first', Path(tmp) / 'second']
    for root in roots:
        root.mkdir()
        for name in platform_runtime_load_order()[:-1]:
            shutil.copy2(source / name, root / name)
    LlamaLibrary(roots[0])
    try:
        LlamaLibrary(roots[1])
    except RuntimeError as exc:
        if 'native runtime family conflict' not in str(exc):
            raise
    else:
        raise SystemExit('second native runtime family was accepted')
    mapped = {
        line.rsplit(' ', 1)[-1]
        for line in Path('/proc/self/maps').read_text(encoding='utf-8').splitlines()
        if any(marker in line for marker in ('/libllama', '/libggml', '/libmtmd', '/liborbit-'))
    }
    if any(path.startswith(str(roots[1]) + '/') for path in mapped):
        raise SystemExit('second native runtime family was mapped')
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [sys.executable, "-c", script, str(source)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
