"""Build diagnostics through the real HTTP /props and REPL, without inference."""
from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit import build_identity as identity
from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.native_server import app
from orbit.native_llama.client import NativeClientConfig
from orbit.terminal.config import AppConfig
from orbit.terminal.repl import Repl
from orbit.terminal import runtime_status
from tests.test_repl import CountingRuntime
from tests.test_runtime_status import _Backend, _Runtime, HostInfo
from tests.test_server_startup_observability import _RunServerHarness, _resolution

OLD = '1' * 40
NEW = '2' * 40
WARNING = 'warning: Orbit client/server build mismatch'


class BuildIdentityTests(unittest.TestCase):
    def test_capture_from_real_checkout_not_workdir_and_keep_process_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            package = root / 'src/orbit'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('__version__ = "0.0.1"\n')
            (package / 'build_identity.py').write_bytes(Path(identity.__file__).read_bytes())
            def git(*args):
                return subprocess.check_output(['git', '-C', td, *args], stderr=subprocess.DEVNULL, text=True).strip()
            git('init'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            git('add', '.'); git('commit', '-m', 'first'); git('tag', 'vfixture')
            first = git('rev-parse', 'HEAD')
            captured = identity._capture_build_identity(package)
            self.assertEqual(captured.commit, first)
            self.assertEqual(captured.description, 'vfixture')
            # A separate process imports the helper, then waits while HEAD changes.
            code = ('import json,sys; from orbit.build_identity import PROCESS_BUILD_IDENTITY as b; '
                    'print(json.dumps(b.to_dict()),flush=True); input(); '
                    'print(json.dumps(b.to_dict()),flush=True)')
            env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
            with subprocess.Popen(['python3', '-c', code], cwd='/', env=env, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                try:
                    before = json.loads(proc.stdout.readline())
                    git('commit', '--allow-empty', '-m', 'second')
                    after, err = proc.communicate('\n', timeout=10)
                    self.assertEqual(proc.returncode, 0, err)
                    self.assertEqual(json.loads(after), before)
                    self.assertEqual(before['commit'], first)
                    self.assertNotEqual(identity._capture_build_identity(package).commit, first)
                finally:
                    if proc.poll() is None: proc.kill(); proc.wait()
            (package / '__init__.py').write_text('__version__ = "changed"\n')
            self.assertTrue(identity._capture_build_identity(package).description.endswith('-dirty'))

    def test_installed_package_does_not_borrow_containing_repository(self):
        with mock.patch.object(identity, '_git') as git:
            found = identity._capture_build_identity(Path('/tmp/project/.venv/site-packages/orbit'))
        self.assertIsNone(found.commit)
        self.assertEqual(found.version, '0.0.1')
        git.assert_not_called()

    def test_foreign_repository_and_untracked_source_are_unknown(self):
        for replies in (['/elsewhere'], ['/tmp/repo', None]):
            with self.subTest(replies=replies), mock.patch.object(identity, '_git', side_effect=replies):
                self.assertIsNone(identity._capture_build_identity(Path('/tmp/repo/src/orbit')).commit)

    def test_git_absent_timeout_and_malformed_output_fail_soft(self):
        for failure in (OSError('missing'), subprocess.TimeoutExpired('git', .5), UnicodeError()):
            with self.subTest(failure=failure), mock.patch.object(identity.subprocess, 'run', side_effect=failure):
                self.assertIsNone(identity._capture_build_identity(Path('/tmp/repo/src/orbit')).commit)
        with mock.patch.object(identity, '_git', side_effect=['/tmp/repo', 'src/orbit/__init__.py', 'broken', None]):
            self.assertIsNone(identity._capture_build_identity(Path('/tmp/repo/src/orbit')).commit)

    def test_metadata_validation(self):
        for bad in (None, [], 'old', 1, {'commit': []}, {'commit': 'short'}, {'commit': 'z'*40},
                    {'commit': NEW+'\n'}, {'commit': False}, {'commit': 'a'*4000}):
            with self.subTest(bad=bad): self.assertIsNone(identity.parse_build_identity(bad).commit)
        value = identity.parse_build_identity({'commit': 'A'*64, 'version': '\x1b[31mBAD', 'description': ['bad']})
        self.assertEqual(value.commit, 'a'*64)
        self.assertIsNone(value.version)
        self.assertIsNone(value.description)


class StatusIdentityTests(unittest.TestCase):
    def status(self, server, client=NEW, props=None):
        backend = _Backend()
        backend.backend_props = lambda: props if props is not None else {
            'backend': 'orbit-native', 'orbit_build': server}
        with mock.patch.object(runtime_status, 'PROCESS_BUILD_IDENTITY', identity.BuildIdentity(client, '0.0.1')):
            return runtime_status.collect_runtime_status(_Runtime(), AppConfig(workdir=Path('/tmp')), backend,
                                                        host_info=HostInfo())

    def test_equal_commits_no_warning_even_different_tags(self):
        status = self.status({'commit': NEW, 'description': 'another-tag'})
        self.assertNotIn(WARNING, runtime_status.format_startup_banner(status))
        self.assertNotIn(WARNING, runtime_status.format_status_panel(status))

    def test_mismatch_visible_at_startup_and_status(self):
        status = self.status({'commit': OLD, 'version': '0.0.1'})
        for text in (runtime_status.format_startup_banner(status), runtime_status.format_status_panel(status)):
            self.assertIn(WARNING, text)
        panel = runtime_status.format_status_panel(status)
        self.assertIn('Client build '+NEW, panel)
        self.assertIn('Server build '+OLD, panel)

    def test_unknown_external_legacy_malformed_no_false_mismatch(self):
        for server in (None, {}, [], 'bad', {'version': 'other'}, {'commit': 'bad'}, {'commit': OLD+'\n'}):
            with self.subTest(server=server):
                status = self.status(server)
                self.assertIn('Server build unknown', runtime_status.format_status_panel(status))
                self.assertNotIn(WARNING, runtime_status.format_status_panel(status))
        self.assertNotIn(WARNING, runtime_status.format_status_panel(self.status({'commit': OLD}, client=None)))
        for props in ([], 'malformed', 7):
            self.assertIsNone(self.status(None, props=props).server_build.commit)

    def test_repl_warns_without_blocking_session_or_mutating_history(self):
        runtime = CountingRuntime()
        backend = runtime.backend
        backend.backend_props = lambda: {'backend': 'orbit-native', 'orbit_build': {'commit': OLD}}
        backend.health = lambda: True
        repl = Repl(runtime=runtime, backend=backend, config=AppConfig(workdir=Path('/tmp')))
        messages = list(runtime.messages)
        buf = io.StringIO()
        with mock.patch.object(runtime_status, 'PROCESS_BUILD_IDENTITY', identity.BuildIdentity(NEW, '0.0.1')), \
             mock.patch('builtins.input', return_value='/exit'), redirect_stdout(buf):
            self.assertEqual(repl.run(), 0)
            self.assertTrue(repl._handle_command('/status'))
        self.assertGreaterEqual(buf.getvalue().count(WARNING), 2)
        self.assertEqual(runtime.messages, messages)
        self.assertEqual(runtime.ask_calls, 0)

    def test_status_refreshes_server_identity_without_inference_cache_mutation(self):
        backend = LlamaServerBackend(base_url='http://fixture', timeout=1)
        backend._props_cache = {'backend': 'orbit-native', 'orbit_build': {'commit': NEW}, 'n_ctx': 8192}
        saved = dict(backend._props_cache)
        with mock.patch.object(backend, '_get_json', return_value={'orbit_build': {'commit': OLD}}) as get:
            self.assertEqual(backend.server_build_info(), {'commit': OLD})
            get.assert_called_once_with('/props')
        self.assertEqual(backend._props_cache, saved)
        with mock.patch.object(backend, '_get_json', side_effect=LlamaServerError('offline')):
            self.assertIsNone(backend.server_build_info())
        for props in ({}, [], 'bad', None):
            with mock.patch.object(backend, '_get_json', return_value=props):
                self.assertIsNone(backend.server_build_info())
        # Real collection must consume the fresh getter, not the stale cached identity.
        with mock.patch.object(backend, '_get_json', return_value={'orbit_build': {'commit': OLD}}), \
             mock.patch.object(backend, 'model_info', return_value=None), \
             mock.patch.object(backend, 'health', return_value=True), \
             mock.patch.object(runtime_status, 'PROCESS_BUILD_IDENTITY', identity.BuildIdentity(NEW)):
            status = runtime_status.collect_runtime_status(_Runtime(), AppConfig(), backend, host_info=HostInfo())
        self.assertIn(WARNING, runtime_status.format_status_panel(status))
        self.assertEqual(backend._props_cache, saved)


def _fake_state():
    """Only the model is substituted; do_GET and HTTP serialization are real."""
    probe = SimpleNamespace(enabled=False, initialized=False, error=None, success=False,
                            draft_tokens=0, accepted_tokens=0)
    client = SimpleNamespace(
        paths=SimpleNamespace(model=Path('/fixture.gguf'), mmproj_model=None, draft_mtp_model=None,
                              multimodal_available=False, multimodal_fallback_reason=None,
                              mtp_available=False, fallback_reason=None, model_id='fixture'),
        config=NativeClientConfig(), supports_vision=False, supports_audio=False,
        _persistent_mtp_runtime=None, mtp_probe=probe, mtp_dry_run=probe,
        mtp_accept_probe=probe, mtp_decode_probe=probe,
        last_mtp_completion=SimpleNamespace(enabled=False, success=False), mtp_fallback_reason=None,
        compatibility_diagnostics=lambda: {}, model_load_status=lambda: {},
        final_prefix_experiment_status=lambda: defaultdict(lambda: None),
        qwen_route_prefix_reuse_status=lambda: {}, qwen36_shell_tool_prefix_reuse_status=lambda: {},
        qwen3_coder_route_prefix_reuse_status=lambda: {}, ornith_route_prefix_reuse_status=lambda: {},
    )
    with mock.patch.object(app, 'safe_native_capability_manifest', return_value={}):
        state = app.OrbitNativeServer(client=client, model_alias='fixture')
    state.session_info = lambda: defaultdict(lambda: None, backend_mode='native')
    state.runtime_info = lambda: defaultdict(lambda: None)
    return state


class NativePropsTests(unittest.TestCase):
    def test_real_http_props_uses_frozen_process_identity(self):
        class QuietHandler(app.OrbitNativeHandler):
            def log_message(self, *args): pass
        frozen = identity.BuildIdentity(OLD, '0.0.1', 'vfixture')
        with mock.patch.object(app, 'PROCESS_BUILD_IDENTITY', frozen), \
             app.ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler) as server:
            server.orbit_state = _fake_state()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                backend = LlamaServerBackend(base_url=f'http://127.0.0.1:{server.server_port}', timeout=2)
                with mock.patch.object(identity, '_capture_build_identity', side_effect=AssertionError('must not reread Git')):
                    props = backend.backend_props()
                    self.assertEqual(props['orbit_build'], frozen.to_dict())
                    self.assertEqual(props['backend_mode'], 'native')
                    self.assertEqual(backend.server_build_info(), frozen.to_dict())
            finally:
                server.shutdown(); thread.join(timeout=3)


class StartupIdentityTests(_RunServerHarness):
    def test_startup_logs_same_identity_without_changing_lifecycle(self):
        with mock.patch.object(app, 'PROCESS_BUILD_IDENTITY', identity.BuildIdentity(OLD, '0.0.1', 'vfixture')):
            code, err, out, _ = self._run(['--model', '/m/v.gguf'], resolutions=[(_resolution(calibrated=True), False)])
        self.assertEqual(code, 0)
        self.assertIn('orbit-server build: '+OLD+' (vfixture)', out)
        self.assertIn('orbit-server listening on http://', out)
