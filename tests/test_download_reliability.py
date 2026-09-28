"""Small local HTTP transfers only: no model or external network required."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from unittest import mock

from orbit.native_llama import model_download as download
from orbit.native_llama.download_cli import _DownloadProgress

BODY = bytes(range(250)) * 400
CUT = 47_000
ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(body))}


class LocalHTTP:
    def __init__(self, mode="complete"):
        self.mode = mode
        self.requests = []
        self.sent = threading.Event()
        self.release = threading.Event()
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                state.requests.append(dict(self.headers))
                number = len(state.requests)
                if (state.mode == "404" or state.mode == "transient" and number < 3
                        or state.mode in {"drop_then_503", "ignore_drop_then_503"} and number > 1):
                    self.send_response(404 if state.mode == "404" else 503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = int(rng.removeprefix("bytes=").removesuffix("-")) if rng else 0
                if state.mode in {"ignore", "ignore_drop_then_503"}:
                    start = 0
                    rng = None
                if start >= len(BODY):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(BODY)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206 if rng else 200)
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{len(BODY)-1}/{len(BODY)}")
                self.send_header("Content-Length", str(len(BODY) - start))
                self.send_header("ETag", '"fixture-v1"')
                self.end_headers()
                try:
                    if number == 1 and state.mode in {"drop", "reset", "stall", "hold", "drop_then_503", "ignore_drop_then_503"}:
                        self.wfile.write(BODY[:CUT])
                        self.wfile.flush()
                        state.sent.set()
                        if state.mode in {"stall", "hold"}:
                            state.release.wait(4)
                        elif state.mode == "reset":
                            time.sleep(.02)
                            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                        self.close_connection = True
                        return
                    self.wfile.write(BODY[start:])
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)


class DownloadReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dest = self.root / "model.gguf"
        self.part = self.root / "model.gguf.part"
        for name, value in [("_READ_TIMEOUT", .15), ("_MAX_ATTEMPTS", 3), ("_RETRY_BACKOFF", .01)]:
            patch = mock.patch.object(download, name, value, create=True)
            patch.start()
            self.addCleanup(patch.stop)

    def child(self, server):
        code = (
            "import sys; from pathlib import Path; from orbit.native_llama import model_download as d; "
            "d.HF_RESOLVE_BASE=sys.argv[1]; d._READ_TIMEOUT=.15; d._MAX_ATTEMPTS=3; d._RETRY_BACKOFF=.01; "
            "d.download_model('owner/repo/model.gguf', models_dir=Path(sys.argv[2]))"
        )
        return subprocess.Popen([sys.executable, "-c", code, server.url, str(self.root)],
                                env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_complete_single_file_uses_real_http_and_atomic_publication(self):
        observed = []
        with LocalHTTP() as server, mock.patch.object(download, "HF_RESOLVE_BASE", server.url):
            final = self.root / "owner--repo/model.gguf"
            result = download.download_model("owner/repo/model.gguf", models_dir=self.root,
                progress=lambda n, total: observed.append((n, total, final.exists())))
        self.assertEqual(result.path.read_bytes(), BODY)
        self.assertTrue(result.downloaded)
        self.assertTrue(observed)
        self.assertTrue(all(not exists for _, _, exists in observed))
        self.assertEqual(observed[-1][:2], (len(BODY), len(BODY)))

    def test_resume_cross_process_after_47_percent(self):
        with LocalHTTP("drop_then_503") as server:
            first = self.child(server)
            _, stderr = first.communicate(timeout=5)
            self.assertNotEqual(first.returncode, 0, stderr)
            final = self.root / "owner--repo/model.gguf"
            part = final.with_name(final.name + ".part")
            self.assertFalse(final.exists())
            self.assertEqual(part.read_bytes(), BODY[:CUT])
            server.mode = "complete"
            server.requests.clear()
            second = self.child(server)
            _, stderr = second.communicate(timeout=5)
            self.assertEqual(second.returncode, 0, stderr)
            self.assertEqual(server.requests[0].get("Range"), f"bytes={CUT}-")
            self.assertEqual(server.requests[0].get("If-Range"), '"fixture-v1"')
            self.assertEqual(final.read_bytes(), BODY)
            self.assertFalse(part.exists())

    def test_signals_preserve_partial_between_processes(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig), LocalHTTP("hold") as server:
                final = self.root / "owner--repo/model.gguf"
                part = final.with_name(final.name + ".part")
                child = self.child(server)
                try:
                    self.assertTrue(server.sent.wait(3))
                    deadline = time.monotonic() + 2
                    while not part.exists() or part.stat().st_size != CUT:
                        if time.monotonic() > deadline:
                            self.fail("received prefix not persisted before stalled read")
                        time.sleep(.005)
                    child.send_signal(sig)
                    child.communicate(timeout=3)
                    self.assertNotEqual(child.returncode, 0)
                    self.assertFalse(final.exists())
                    self.assertEqual(part.read_bytes(), BODY[:CUT])
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.communicate()
                server.mode = "complete"
                resumed = self.child(server)
                _, stderr = resumed.communicate(timeout=5)
                self.assertEqual(resumed.returncode, 0, stderr)
                self.assertEqual(final.read_bytes(), BODY)
                final.unlink()

    def test_stall_is_bounded_and_resumed(self):
        with LocalHTTP("stall") as server:
            start = time.monotonic()
            download.fetch_resumable(server.url, self.dest)
            self.assertLess(time.monotonic() - start, 5)
            self.assertEqual(self.dest.read_bytes(), BODY)
            self.assertEqual(server.requests[1].get("Range"), f"bytes={CUT}-")

    def test_connection_reset_is_retried(self):
        with LocalHTTP("reset") as server:
            download.fetch_resumable(server.url, self.dest)
            self.assertEqual(self.dest.read_bytes(), BODY)
            self.assertEqual(len(server.requests), 2)
            self.assertTrue(server.requests[1].get("Range"))

    def test_ignored_range_restarts_without_append(self):
        self.part.write_bytes(b"wrong old prefix")
        with LocalHTTP("ignore") as server:
            download.fetch_resumable(server.url, self.dest)
            self.assertEqual(server.requests[0].get("Range"), "bytes=16-")
        self.assertEqual(self.dest.read_bytes(), BODY)

    def test_ignored_range_failure_preserves_old_partial_and_resumes_restart(self):
        self.part.write_bytes(b"old bytes")
        restart = self.part.with_name(self.part.name + ".restart")

        class Interrupted(Response):
            def read1(self, size):
                data = super().read1(size)
                if not data:
                    raise TimeoutError("stalled replacement")
                return data

        with mock.patch.object(download, "_MAX_ATTEMPTS", 1):
            for received in (b"", b"abcd"):
                with self.subTest(received=received):
                    opener = mock.Mock(return_value=Interrupted(received, headers={
                        "Content-Length": "6", "ETag": '"new"'}))
                    with self.assertRaises(download.DownloadIncomplete):
                        download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
                    self.assertEqual(self.part.read_bytes(), b"old bytes")
                    self.assertEqual(restart.read_bytes(), received)
                    self.assertFalse(self.dest.exists())
        opener = mock.Mock(return_value=Response(b"ef", 206, {
            "Content-Length": "2", "Content-Range": "bytes 4-5/6", "ETag": '"new"'}))
        download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
        self.assertEqual(opener.call_args.args[0].get_header("Range"), "bytes=4-")
        self.assertEqual(opener.call_args.args[0].get_header("If-range"), '"new"')
        self.assertEqual(self.dest.read_bytes(), b"abcdef")
        self.assertFalse(self.part.exists())
        self.assertFalse(restart.exists())

    def test_repeated_ignored_range_preserves_prefix_on_cancel_or_change(self):
        self.part.write_bytes(b"original prefix")
        restart = self.part.with_name(self.part.name + ".restart")
        restart.write_bytes(b"abcd")

        def cancel(n, total):
            if n:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            download.fetch_resumable("http://fixture.invalid", self.dest, chunk_size=2,
                                    opener=lambda *a, **k: Response(b"abcdef"), progress=cancel)
        self.assertEqual(self.part.read_bytes(), b"original prefix")
        self.assertEqual(restart.read_bytes(), b"abcd")
        with self.assertRaisesRegex(download.DownloadIncomplete, "representation changed"):
            download.fetch_resumable("http://fixture.invalid", self.dest,
                                    opener=lambda *a, **k: Response(b"xyzuvw"))
        self.assertEqual(restart.read_bytes(), b"abcd")
        self.assertFalse(self.dest.exists())
        # A stable server ignoring Range still works, with no corrupt append.
        download.fetch_resumable("http://fixture.invalid", self.dest, chunk_size=3,
                                opener=lambda *a, **k: Response(b"abcdef"))
        self.assertEqual(self.dest.read_bytes(), b"abcdef")
        self.assertFalse(self.part.exists())
        self.assertFalse(restart.exists())

    def test_restart_partial_resumes_in_another_process(self):
        with LocalHTTP("ignore_drop_then_503") as server:
            final = self.root / "owner--repo/model.gguf"
            final.parent.mkdir()
            part = final.with_name(final.name + ".part")
            part.write_bytes(b"previous representation")
            first = self.child(server)
            _, stderr = first.communicate(timeout=5)
            self.assertNotEqual(first.returncode, 0, stderr)
            self.assertEqual(part.read_bytes(), b"previous representation")
            restart = part.with_name(part.name + ".restart")
            self.assertEqual(restart.read_bytes(), BODY[:CUT])
            self.assertFalse(final.exists())
            server.mode = "complete"
            server.requests.clear()
            second = self.child(server)
            _, stderr = second.communicate(timeout=5)
            self.assertEqual(second.returncode, 0, stderr)
            self.assertEqual(server.requests[0].get("Range"), f"bytes={CUT}-")
            self.assertEqual(final.read_bytes(), BODY)
            self.assertFalse(part.exists())
            self.assertFalse(restart.exists())

    def test_complete_partial_is_finalized_with_416(self):
        self.part.write_bytes(BODY)
        with LocalHTTP() as server:
            download.fetch_resumable(server.url, self.dest)
            self.assertEqual(len(server.requests), 1)
        self.assertEqual(self.dest.read_bytes(), BODY)
        self.assertFalse(self.part.exists())

    def test_permanent_404_is_not_retried(self):
        self.part.write_bytes(BODY[:CUT])
        with LocalHTTP("404") as server:
            with self.assertRaises(HTTPError):
                download.fetch_resumable(server.url, self.dest)
            self.assertEqual(len(server.requests), 1)
        self.assertEqual(self.part.read_bytes(), BODY[:CUT])
        self.assertFalse(self.dest.exists())

    def test_transient_http_retries_are_bounded(self):
        with LocalHTTP("transient") as server, mock.patch.object(download.time, "sleep") as sleep:
            download.fetch_resumable(server.url, self.dest)
            self.assertEqual(len(server.requests), 3)
            self.assertEqual(sleep.call_args_list, [mock.call(.01), mock.call(.02)])
        self.assertEqual(self.dest.read_bytes(), BODY)

    def test_incoherent_206_is_rejected_without_changing_partial(self):
        for value, length in [(None, "3"), ("nonsense", "3"), ("bytes 2-5/6", "4"),
                              ("bytes 3-2/6", "0"), ("bytes 3-6/6", "4"),
                              ("bytes 3-5/6", "4"), ("bytes 3-5/*", "3")]:
            with self.subTest(range=value):
                self.part.write_bytes(b"abc")
                headers = {"Content-Length": length}
                if value is not None:
                    headers["Content-Range"] = value
                opener = mock.Mock(return_value=Response(b"def", 206, headers))
                with self.assertRaises(download.DownloadIncomplete):
                    download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
                self.assertEqual(self.part.read_bytes(), b"abc")
                self.assertFalse(self.dest.exists())
                self.assertEqual(opener.call_count, 1)

    def test_unavailable_size_is_not_published(self):
        with self.assertRaises(download.DownloadIncomplete):
            download.fetch_resumable("http://fixture.invalid", self.dest,
                                    opener=lambda *a, **k: Response(b"abc", headers={}))
        self.assertFalse(self.dest.exists())

    def test_changed_or_weak_etag_in_range_preserves_partial(self):
        for tag in ['"different"', 'W/"original"', "malformed"]:
            with self.subTest(etag=tag):
                self.part.write_bytes(b"abc")
                self.dest.with_name(self.dest.name + ".part.etag").write_text('"original"')
                opener = mock.Mock(return_value=Response(b"def", 206, {
                    "Content-Length": "3", "Content-Range": "bytes 3-5/6", "ETag": tag}))
                with self.assertRaises(download.DownloadIncomplete):
                    download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
                self.assertEqual(self.part.read_bytes(), b"abc")
                self.assertEqual(opener.call_count, 1)
                self.assertFalse(self.dest.exists())

    def test_short_valid_subranges_resume_until_total(self):
        self.part.write_bytes(b"ab")
        opener = mock.Mock(side_effect=[
            Response(b"cd", 206, {"Content-Range": "bytes 2-3/6", "Content-Length": "2"}),
            Response(b"ef", 206, {"Content-Range": "bytes 4-5/6", "Content-Length": "2"}),
        ])
        download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
        self.assertEqual(self.dest.read_bytes(), b"abcdef")
        self.assertEqual([call.args[0].get_header("Range") for call in opener.call_args_list],
                         ["bytes=2-", "bytes=4-"])

    def test_short_body_exhausts_budget_without_final(self):
        opener = mock.Mock(side_effect=lambda *a, **k: Response(b"ab", headers={"Content-Length": "6"}))
        with self.assertRaisesRegex(download.DownloadIncomplete, "3 attempts"):
            download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(self.part.read_bytes(), b"ab")
        self.assertFalse(self.dest.exists())

    def test_invalid_encoding_or_oversized_body_never_publishes(self):
        for headers in [{"Content-Length": "3", "Content-Encoding": "gzip"},
                        {"Content-Length": "2"}]:
            with self.subTest(headers=headers):
                opener = mock.Mock(return_value=Response(b"abc", headers=headers))
                with self.assertRaises(download.DownloadIncomplete):
                    download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
                self.assertFalse(self.dest.exists())
                self.assertEqual(opener.call_count, 1)

    def test_callback_exception_is_not_a_network_retry(self):
        opener = mock.Mock(return_value=Response(b"abc"))

        def progress(n, total):
            if n:
                raise ConnectionResetError("callback, not network")

        with self.assertRaisesRegex(ConnectionResetError, "callback"):
            download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener, progress=progress)
        self.assertEqual(self.part.read_bytes(), b"abc")
        self.assertEqual(opener.call_count, 1)
        self.assertFalse(self.dest.exists())

    def test_short_disk_write_is_not_published_or_retried(self):
        original_open = Path.open

        def short_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if path != self.part or not kwargs.get("buffering") == 0:
                return handle
            wrapped = mock.MagicMock(wraps=handle)
            wrapped.__enter__.return_value = wrapped
            wrapped.__exit__.side_effect = lambda *a: handle.close()
            wrapped.write.side_effect = lambda data: handle.write(data[:1])
            return wrapped

        opener = mock.Mock(return_value=Response(b"abc"))
        with mock.patch.object(Path, "open", short_open), self.assertRaisesRegex(OSError, "short file write"):
            download.fetch_resumable("http://fixture.invalid", self.dest, opener=opener)
        self.assertEqual(self.part.read_bytes(), b"a")
        self.assertEqual(opener.call_count, 1)
        self.assertFalse(self.dest.exists())

    def test_existing_final_is_reused_without_request(self):
        final = self.root / "owner--repo/model.gguf"
        final.parent.mkdir()
        final.write_bytes(b"existing")
        opener = mock.Mock(side_effect=AssertionError("network not allowed"))
        result = download.download_model("owner/repo/model.gguf", models_dir=self.root, opener=opener)
        self.assertFalse(result.downloaded)
        self.assertEqual(result.path.read_bytes(), b"existing")

    def test_progress_names_resume_and_uses_total_size(self):
        progress = _DownloadProgress()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            progress(11_107_221_504, 23_630_258_520)
            progress(23_630_258_520, 23_630_258_520)
            progress.finish()
        self.assertIn("resuming from 11.1 GB (47%)", output.getvalue())
        self.assertIn("100%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
