"""The live context's thread counts are observable, not inferred.

`config.threads` is what Orbit resolved; `llama_n_threads(ctx)` is what the
context runs. They agree only if every retune along the startup path --
calibration candidates, the restore afterwards -- was undone, and until now
the only way to check was a benchmark that came out slow. These pin the
read-back seam: the client method, the server's `/props` fields, and the
startup log lines at each lifecycle stage, plus the tolerance that keeps a
diagnostic from ever failing a server start.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient, NativeLlamaPaths  # noqa: E402
from orbit.native_server import app as app_module  # noqa: E402
from orbit.native_server.app import OrbitNativeServer, _log_native_threads, _native_thread_counts  # noqa: E402


class _Lib:
    def __init__(self, threads=(6, 6), *, getters=True) -> None:
        self.threads = threads
        if getters:
            self.llama_n_threads = lambda ctx: self.threads[0]
            self.llama_n_threads_batch = lambda ctx: self.threads[1]


def _paths() -> NativeLlamaPaths:
    return NativeLlamaPaths(
        llama_root=pathlib.Path("/llama"), build_bin=pathlib.Path("/llama/build/bin"),
        library=pathlib.Path("/llama/build/bin/libllama.so"), model=pathlib.Path("/models/m.gguf"),
        mmproj_model=None, draft_mtp_model=None, multimodal_available=False,
        multimodal_fallback_reason=None, mtp_available=False,
        fallback_reason="draft-mtp-missing", model_id="m",
    )


def _client(lib, *, loaded=True) -> NativeLlamaClient:
    with mock.patch("orbit.native_llama.client.LlamaLibrary"):
        client = NativeLlamaClient(_paths(), NativeClientConfig(threads=6, threads_batch=6))
    client.lib = SimpleNamespace(lib=lib)
    client._session.ctx_tgt = 1 if loaded else 0
    return client


class ClientReadBackTests(unittest.TestCase):
    def test_reads_what_the_context_holds_not_what_the_config_says(self) -> None:
        client = _client(_Lib((8, 8)))
        self.assertEqual(client.config.threads, 6)
        self.assertEqual(client.native_thread_counts(), (8, 8))

    def test_none_before_the_context_exists(self) -> None:
        self.assertIsNone(_client(_Lib(), loaded=False).native_thread_counts())

    def test_none_on_a_backend_without_the_getters(self) -> None:
        self.assertIsNone(_client(_Lib(getters=False)).native_thread_counts())

    def test_the_bindings_declare_the_getters(self) -> None:
        source = inspect.getsource(sys.modules["orbit.native_llama.bindings"])
        self.assertIn("lib.llama_n_threads.argtypes = [c_void_p]", source)
        self.assertIn("lib.llama_n_threads_batch.argtypes = [c_void_p]", source)


class ServerReadBackTests(unittest.TestCase):
    def test_runtime_info_publishes_the_read_back_beside_the_config(self) -> None:
        server = OrbitNativeServer(client=_client(_Lib((8, 8))), model_alias="m")
        info = server.runtime_info()
        self.assertEqual((info["threads"], info["threads_batch"]), (6, 6))
        self.assertEqual((info["native_threads"], info["native_threads_batch"]), (8, 8))

    def test_runtime_info_publishes_none_when_unobservable(self) -> None:
        server = OrbitNativeServer(client=_client(_Lib(getters=False)), model_alias="m")
        info = server.runtime_info()
        self.assertIsNone(info["native_threads"])
        self.assertIsNone(info["native_threads_batch"])

    def test_props_carries_both_fields(self) -> None:
        """The handler copies runtime fields one by one; these must be among them."""
        source = inspect.getsource(app_module.OrbitNativeHandler.do_GET)
        self.assertIn('"native_threads": runtime["native_threads"]', source)
        self.assertIn('"native_threads_batch": runtime["native_threads_batch"]', source)

    def test_a_client_without_the_method_is_unobservable_not_an_error(self) -> None:
        self.assertIsNone(_native_thread_counts(SimpleNamespace()))

    def test_a_raising_or_malformed_read_back_is_unobservable_not_an_error(self) -> None:
        def boom():
            raise RuntimeError("ctypes")

        self.assertIsNone(_native_thread_counts(SimpleNamespace(native_thread_counts=boom)))
        self.assertIsNone(_native_thread_counts(SimpleNamespace(native_thread_counts=lambda: ("x", 1))))
        self.assertIsNone(_native_thread_counts(SimpleNamespace(native_thread_counts=lambda: None)))


class StartupLogTests(unittest.TestCase):
    def _log(self, client, stage) -> str:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _log_native_threads(client, stage)
        return err.getvalue()

    def test_the_line_names_the_counts_and_the_stage(self) -> None:
        line = self._log(SimpleNamespace(native_thread_counts=lambda: (6, 6)), "after model load")
        self.assertEqual(line.strip(), "orbit-server native threads: 6/6 (after model load)")

    def test_an_unobservable_client_is_logged_as_such(self) -> None:
        line = self._log(SimpleNamespace(), "after prewarm, before bind")
        self.assertIn("unobservable", line)
        self.assertIn("after prewarm, before bind", line)

    def test_every_lifecycle_stage_is_logged_in_order(self) -> None:
        """Load -> resolution (after restore) -> prewarm -> bind."""
        source = inspect.getsource(app_module.run_server)
        load = source.index("client.load()")
        after_load = source.index('_log_native_threads(client, "after model load")')
        restore = source.index("restore_threads(client")
        after_resolution = source.index('_log_native_threads(client, "after profile resolution")')
        prewarm = source.index("prewarm_startup_route_prefix(client)")
        before_bind = source.index('_log_native_threads(client, "after prewarm, before bind")')
        bind = source.index("ThreadingHTTPServer((args.host, args.port)")
        self.assertLess(load, after_load)
        self.assertLess(after_load, restore)
        self.assertLess(restore, after_resolution)
        self.assertLess(after_resolution, prewarm)
        self.assertLess(prewarm, before_bind)
        self.assertLess(before_bind, bind)


if __name__ == "__main__":
    unittest.main()
