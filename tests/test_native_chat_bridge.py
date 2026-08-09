from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
from orbit.native_llama.native_names import platform_runtime_libs, runtime_library_filename


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
    if '/libllama' in line or '/libggml' in line
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
        if '/libllama' in line or '/libggml' in line
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
