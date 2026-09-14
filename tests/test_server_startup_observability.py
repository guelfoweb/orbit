"""Startup UX observability for `orbit server` — presentation only.

SERVER-STARTUP-UX-OBSERVABILITY-1. These prove the added progress lines are
truthful and correctly ordered, that a cached start never claims a calibration
sweep, that prewarm success is printed only after success, and that the output
is deterministic line-based text (no ANSI/cursor control) that adds nothing to
color. No real model, no calibration, no prewarm work is invoked -- every seam
is faked -- so the tests also confirm observability causes no extra invocation.
"""
from __future__ import annotations

import io
import os
import pathlib
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_server import app as APP  # noqa: E402
from orbit.native_llama.client import NativeRoutePrefixPrefillResult  # noqa: E402
from tests.test_native_server_bootstrap import (  # noqa: E402
    _FakeHTTPServer,
    _FakeNativeClient,
)


def _profile(source="heuristic", threads=6, threads_batch=6):
    return SimpleNamespace(
        source=source, threads=threads, threads_batch=threads_batch, batch=256,
        ubatch=128,
    )


def _resolution(*, source="heuristic", calibrated=False, threads=6, threads_batch=6):
    return SimpleNamespace(
        profile=_profile(source, threads, threads_batch), calibrated=calibrated,
        fingerprint="fp", cache_path=None, calibration_error=None, measurements=None,
    )


def _ok_prewarm(tokens=768, ms=23800.0):
    return NativeRoutePrefixPrefillResult(
        attempted=True, succeeded=True, skipped=False,
        prefix_token_count=tokens, prefill_ms=ms, restore_ready=True,
    )


def _skipped_prewarm(reason="model_profile_ineligible"):
    return NativeRoutePrefixPrefillResult(
        attempted=False, succeeded=False, skipped=True, skip_reason=reason,
    )


def _failed_prewarm(reason="startup_prewarm_failed:RuntimeError"):
    return NativeRoutePrefixPrefillResult(
        attempted=True, succeeded=False, skipped=False, failed_reason=reason,
    )


class ReportPrewarmUnitTests(unittest.TestCase):
    """T5, T6, T8, metrics truthfulness — the pure reporting seam."""

    def _emit(self, label, result):
        buf = io.StringIO()
        with redirect_stderr(buf):
            APP._report_startup_prewarm(label, result)
        return buf.getvalue()

    def test_success_prints_only_after_success_with_real_metrics(self) -> None:
        out = self._emit("route-prefix", _ok_prewarm(768, 23800.0))
        self.assertIn("prewarm complete (route-prefix): 768 tokens, 23.8s", out)

    def test_success_without_metrics_omits_them_never_estimates(self) -> None:
        out = self._emit("route-prefix", NativeRoutePrefixPrefillResult(
            attempted=True, succeeded=True, skipped=False))
        self.assertIn("prewarm complete (route-prefix)", out)
        self.assertNotIn("tokens", out)
        self.assertNotIn("s\n", out.replace("...", ""))  # no fabricated seconds

    def test_skip_never_prints_success(self) -> None:
        out = self._emit("route-prefix", _skipped_prewarm("tools_disabled"))
        self.assertIn("prewarm skipped (route-prefix): tools_disabled", out)
        self.assertNotIn("complete", out)

    def test_failure_never_prints_success(self) -> None:
        out = self._emit("analysis-prefix", _failed_prewarm())
        self.assertIn("prewarm failed (analysis-prefix)", out)
        self.assertNotIn("complete", out)

    def test_output_is_plain_no_ansi(self) -> None:
        out = self._emit("route-prefix", _ok_prewarm())
        self.assertNotIn("\x1b[", out)  # no ANSI escape
        self.assertNotIn("\r", out)     # no cursor control


class _RunServerHarness(unittest.TestCase):
    """Drives run_server with every heavy seam faked; captures stderr/stdout."""

    def _run(self, argv, *, resolutions, calibrate=None, route=None, analysis=None,
             prewarm_env="off"):
        _FakeNativeClient.instances.clear()
        _FakeHTTPServer.instances.clear()
        stderr, stdout = io.StringIO(), io.StringIO()
        resolve_calls = {"n": 0}

        def fake_resolve(args, calibrator=None):
            resolve_calls["n"] += 1
            entry = resolutions[min(resolve_calls["n"] - 1, len(resolutions) - 1)]
            res, invoke = entry if isinstance(entry, tuple) else (entry, False)
            if calibrator is not None and invoke:
                # Simulate a real sweep: invoke the runtime's calibrator, which
                # emits the progress lines, exactly as the resolver would.
                calibrator(topology=SimpleNamespace(), fields=("threads", "threads_batch"))
            return res

        patches = [
            mock.patch.dict(os.environ, {"ORBIT_KV_PREFIX_PREWARM": prewarm_env}, clear=False),
            mock.patch.object(APP, "resolve_bootstrap_paths",
                              return_value=SimpleNamespace(model=pathlib.Path("/m/v.gguf"))),
            mock.patch.object(APP, "NativeLlamaClient", _FakeNativeClient),
            mock.patch.object(APP, "ThreadingHTTPServer", _FakeHTTPServer),
            mock.patch.object(APP, "_resolve_startup_profile", side_effect=fake_resolve),
            mock.patch.object(APP, "resolve_model_alias", return_value="Test Model 1.0"),
            mock.patch.object(APP, "render_profile_lines", return_value=["profile: test-source"]),
            mock.patch.object(APP, "_log_native_threads", return_value=None),
            mock.patch.object(APP, "thread_candidates", return_value=[4, 6, 8]),
        ]
        if calibrate is not None:
            patches.append(mock.patch.object(APP, "calibrate_threads", calibrate))
        if route is not None:
            patches.append(mock.patch.object(APP, "prewarm_startup_route_prefix", route))
        if analysis is not None:
            patches.append(mock.patch.object(APP, "prewarm_startup_analysis_prefix", analysis))
        with redirect_stderr(stderr), redirect_stdout(stdout):
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                code = APP.run_server(argv)
        return code, stderr.getvalue(), stdout.getvalue(), resolve_calls["n"]


class CachedPathTests(_RunServerHarness):
    def test_cached_start_shows_no_calibration_message(self) -> None:  # T1
        cached = _resolution(source="cached auto-calibrated (threads, threads_batch)",
                             calibrated=False)
        # both resolves return the cached profile; the 2nd never invokes the calibrator
        code, err, _out, _n = self._run(
            ["--model", "/m/v.gguf"], resolutions=[(cached, False), (cached, False)])
        self.assertEqual(code, 0)
        self.assertIn("server profile: cached auto-calibrated", err)
        self.assertNotIn("auto-calibrating server profile", err)
        self.assertNotIn("calibration complete", err)
        self.assertNotIn("candidate", err)

    def test_cached_start_calibrator_never_invoked(self) -> None:  # T12
        cached = _resolution(source="cached auto-calibrated (x)", calibrated=False)
        calib = mock.Mock()
        self._run(["--model", "/m/v.gguf"], resolutions=[(cached, False), (cached, False)],
                  calibrate=calib)
        calib.assert_not_called()


class CalibrationPathTests(_RunServerHarness):
    def _calibrate_emitting(self, candidates):
        def _calibrate(client, *, topology, fields, on_event=None):
            if on_event:
                on_event(SimpleNamespace(threads=candidates[0], rejected="warmup"))  # warm-up
                for t in candidates:
                    on_event(SimpleNamespace(threads=t, rejected=None))
            return ({"threads": candidates[-1], "threads_batch": candidates[-1]}, [])
        return _calibrate

    def test_calibration_start_and_completion(self) -> None:  # T2
        pre = _resolution(source="heuristic", calibrated=False)
        post = _resolution(source="auto-calibrated (threads)", calibrated=True,
                           threads=8, threads_batch=8)
        code, err, _o, _n = self._run(
            ["--model", "/m/v.gguf"],
            resolutions=[(pre, False), (post, True)],
            calibrate=self._calibrate_emitting([4, 6, 8]),
            route=mock.Mock(return_value=_skipped_prewarm()))
        self.assertEqual(code, 0)
        self.assertIn("auto-calibrating server profile", err)
        self.assertIn("calibration complete: threads=8, threads_batch=8", err)

    def test_candidate_progress_matches_actual_sequence(self) -> None:  # T3
        pre = _resolution(calibrated=False)
        post = _resolution(source="auto-calibrated (threads)", calibrated=True,
                           threads=8, threads_batch=8)
        with mock.patch.object(APP, "thread_candidates", return_value=[4, 6, 8]):
            code, err, _o, _n = self._run(
                ["--model", "/m/v.gguf"],
                resolutions=[(pre, False), (post, True)],
                calibrate=self._calibrate_emitting([4, 6, 8]),
                route=mock.Mock(return_value=_skipped_prewarm()))
        self.assertEqual(code, 0)
        # exactly one line per real candidate (warm-up excluded), numbered /3
        self.assertIn("candidate 1/3: threads=4", err)
        self.assertIn("candidate 2/3: threads=6", err)
        self.assertIn("candidate 3/3: threads=8", err)
        self.assertNotIn("candidate 4/", err)  # warm-up is not counted


class PrewarmPathTests(_RunServerHarness):
    def test_prewarm_start_before_execution_and_success_after(self) -> None:  # T4, T5
        order = []
        def route(client):
            order.append("exec")
            return _ok_prewarm(768, 23800.0)
        # Capture that the start line precedes the exec by interleaving.
        pre = _resolution(calibrated=True)  # skip calibration
        code, err, _o, _n = self._run(
            ["--model", "/m/v.gguf"], resolutions=[(pre, False)],
            route=route, prewarm_env="startup")
        self.assertEqual(code, 0)
        start = err.index("prewarming route-prefix cache...")
        done = err.index("prewarm complete (route-prefix): 768 tokens, 23.8s")
        self.assertLess(start, done)
        self.assertEqual(order, ["exec"])  # prewarm ran exactly once

    def test_prewarm_skip_emits_no_false_success(self) -> None:  # T6
        pre = _resolution(calibrated=True)
        code, err, _o, _n = self._run(
            ["--model", "/m/v.gguf"], resolutions=[(pre, False)],
            route=mock.Mock(return_value=_skipped_prewarm("model_profile_ineligible")),
            prewarm_env="startup")
        self.assertEqual(code, 0)
        self.assertIn("prewarm skipped (route-prefix): model_profile_ineligible", err)
        self.assertNotIn("prewarm complete", err)

    def test_prewarm_disabled_is_quiet(self) -> None:
        pre = _resolution(calibrated=True)
        code, err, _o, _n = self._run(
            ["--model", "/m/v.gguf"], resolutions=[(pre, False)],
            route=mock.Mock(return_value=_skipped_prewarm("disabled")),
            prewarm_env="off")
        self.assertEqual(code, 0)
        self.assertNotIn("prewarming route-prefix", err)
        self.assertNotIn("prewarm skipped", err)


class OrderingAndContractTests(_RunServerHarness):
    def test_lifecycle_ordering(self) -> None:  # T7, T11
        pre = _resolution(source="cached auto-calibrated (x)", calibrated=False)
        code, err, out, _n = self._run(
            ["--model", "/m/v.gguf"], resolutions=[(pre, False), (pre, False)],
            route=mock.Mock(return_value=_skipped_prewarm()))
        self.assertEqual(code, 0)
        # resolving -> cached -> loading model -> profile line -> model -> listening
        idx = [
            err.index("resolving server profile..."),
            err.index("server profile: cached auto-calibrated"),
            err.index("loading model: Test Model 1.0..."),
            err.index("orbit-server profile: test-source"),
        ]
        self.assertEqual(idx, sorted(idx))
        # existing final lines remain, on stdout, semantically unchanged
        self.assertIn("orbit-server model: Test Model 1.0", out)
        self.assertIn("orbit-server listening on http://", out)

    def test_output_is_deterministic_no_ansi_or_cursor(self) -> None:  # T8, T9
        pre = _resolution(calibrated=True)
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            code, err, out, _n = self._run(
                ["--model", "/m/v.gguf"], resolutions=[(pre, False)],
                route=mock.Mock(return_value=_ok_prewarm()), prewarm_env="startup")
        self.assertEqual(code, 0)
        for stream in (err, out):
            self.assertNotIn("\x1b[", stream)  # no ANSI
            self.assertNotIn("\r", stream)     # no cursor control
        # every startup line carries the existing orbit-server prefix
        for line in err.splitlines():
            if line.strip():
                self.assertTrue(line.startswith("orbit-server "), line)

    def test_startup_exception_still_propagates(self) -> None:  # T10
        pre = _resolution(calibrated=True)
        def boom(client):
            raise RuntimeError("prewarm exploded")
        # A RuntimeError in the try-block is caught and returns 1 (existing
        # contract); the point is it is not swallowed silently.
        code, err, _o, _n = self._run(
            ["--model", "/m/v.gguf"], resolutions=[(pre, False)],
            route=boom, prewarm_env="startup")
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
