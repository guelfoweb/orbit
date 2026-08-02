from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from orbit.native_llama.bindings import ChatBridgeLibrary
from orbit.native_llama.build_support import DEFAULT_VENDOR_BUILD_BIN
from orbit.native_llama.chat_bridge import CHAT_BRIDGE_API_VERSION, chat_bridge_filename, validate_chat_bridge_artifact
from orbit.native_llama.client import _resolve_chat_bridge_path
from orbit.native_llama.native_names import platform_runtime_libs


class NativeChatBridgeTests(unittest.TestCase):
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
