"""MODEL-STORE-UX-20: one configurable models directory for every model operation.

Precedence is fixed and short: `--models-dir` > `ORBIT_MODELS_DIR` > the value
persisted by `orbit config models-dir` (in the terminal client's existing
`~/.orbit/config.json`) > the historical default `<orbit>/models`. Download,
discovery, the server bootstrap and availability checks all read it through
`model_registry.resolve_models_dir`; no component builds the path by hand.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import stat
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import download_cli, model_store  # noqa: E402
from orbit.native_llama.model_discovery import discover_models  # noqa: E402
from orbit.native_llama.model_download import DownloadResult, download_model  # noqa: E402
from orbit.native_llama.model_profiles import detect_native_model_profile  # noqa: E402
from orbit.native_llama.model_registry import (  # noqa: E402
    MODELS_DIR_CONFIG_KEY,
    MODELS_DIR_ENV,
    default_models_dir,
    effective_models_dir,
    get_manifest,
    local_model_path,
    orbit_config_path,
    resolve_model,
    resolve_models_dir,
)
from orbit.native_llama.model_store import (  # noqa: E402
    ModelsDirConfigError,
    download_advisory,
    filesystem_type,
    large_model_advisory,
    persist_models_dir,
)
from orbit.terminal import cli as terminal_cli  # noqa: E402
import subprocess  # noqa: E402
from orbit.terminal.config import DEFAULT_CONFIG_PATH, add_config_arguments, load_app_config  # noqa: E402
import argparse  # noqa: E402

ORNITH_ID = "ornith15-35b-a3b-q4-k-m"
GIB = 2**30


def _env(home: Path, **extra: str) -> dict[str, str]:
    env = {"HOME": str(home)}
    env.update(extra)
    return env


def _write_config(home: Path, data: dict) -> Path:
    path = home / ".orbit" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _inspector_for(target: Path):
    """A discovery inspector that verifies only the file placed as Ornith."""
    def inspect(path: Path):
        if path.resolve() == target.resolve():
            return _verified_ornith_profile()
        raise ValueError("unsupported file")
    return inspect


def _verified_ornith_profile():
    metadata = {
        "general.architecture": "qwen35moe", "general.name": "Ornith-1.5-35B", "general.file_type": "15",
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.bos_token_id": "248044", "tokenizer.ggml.eos_token_id": "248046",
        "qwen35moe.context_length": "262144", "qwen35moe.block_count": "41",
        "qwen35moe.expert_count": "256", "qwen35moe.expert_used_count": "8",
    }
    import hashlib
    template = "ornith-template"
    with mock.patch("orbit.native_llama.model_profiles.ORNITH15_OFFICIAL_TEMPLATE_SHA256",
                    hashlib.sha256(template.encode()).hexdigest()):
        profile = detect_native_model_profile(metadata, template)
    assert profile.verified
    return profile


class ResolverTests(unittest.TestCase):
    def test_A_no_configuration_keeps_the_existing_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            resolution = resolve_models_dir(environ=_env(home))
        self.assertEqual(resolution.path, default_models_dir())
        self.assertEqual(resolution.source, "default")
        self.assertIsNone(resolution.config_error)
        self.assertEqual(resolution.config_path, home / ".orbit" / "config.json")
        # The historical default is untouched: <orbit>/models inside a checkout.
        self.assertEqual(default_models_dir(ROOT / "src"), ROOT / "models")

    def test_B_persisted_models_dir_is_used_by_download_discovery_and_server_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; store = Path(tmp) / "store"; store.mkdir()
            _write_config(home, {MODELS_DIR_CONFIG_KEY: str(store)})
            env = _env(home)
            resolution = resolve_models_dir(environ=env)
            self.assertEqual((resolution.path, resolution.source), (store, "config"))
            with mock.patch.dict(os.environ, env, clear=False):
                os.environ.pop(MODELS_DIR_ENV, None)
                self.assertEqual(effective_models_dir(), store)
                # download writes there without any explicit directory
                retrieve = mock.Mock(side_effect=lambda url, dest, *a: Path(dest).write_bytes(b"gguf"))
                result = download_model("owner/repo/model.gguf", retrieve=retrieve)
                self.assertEqual(result.path, store / "owner--repo" / "model.gguf")
                self.assertTrue(result.path.is_file())
                # the server's registry resolution finds a model placed there
                manifest = get_manifest(ORNITH_ID)
                target = local_model_path(manifest.target, models_dir=store)
                target.parent.mkdir(parents=True); target.write_bytes(b"gguf")
                resolved = resolve_model(manifest, hf_cache=Path(tmp) / "hf")
                self.assertEqual(resolved.target_path, target)
                # discovery lists exactly that file as AVAILABLE
                discovered = discover_models(hf_cache=Path(tmp) / "hf", inspector=_inspector_for(target))
                rows = {row.model: row for row in discovered.rows}
                self.assertEqual(rows["Ornith 1.5 35B-A3B"].local, "AVAILABLE")
                self.assertEqual(Path(rows["Ornith 1.5 35B-A3B"].path_or_action), target.resolve())

    def test_C_environment_overrides_the_persisted_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; _write_config(home, {MODELS_DIR_CONFIG_KEY: str(Path(tmp) / "persisted")})
            resolution = resolve_models_dir(environ=_env(home, ORBIT_MODELS_DIR=str(Path(tmp) / "from-env")))
            self.assertEqual((resolution.path, resolution.source), (Path(tmp) / "from-env", "env"))
            blank = resolve_models_dir(environ=_env(home, ORBIT_MODELS_DIR="  "))
            self.assertEqual(blank.source, "config")  # a blank variable does not count

    def test_D_explicit_models_dir_beats_environment_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; _write_config(home, {MODELS_DIR_CONFIG_KEY: str(Path(tmp) / "persisted")})
            env = _env(home, ORBIT_MODELS_DIR=str(Path(tmp) / "from-env"))
            resolution = resolve_models_dir(Path(tmp) / "explicit", environ=env)
            self.assertEqual((resolution.path, resolution.source), (Path(tmp) / "explicit", "cli"))
            self.assertEqual(resolve_models_dir("", environ=env).source, "env")  # empty explicit is "unset"
            # the download CLI and the server pass their --models-dir through the same resolver
            with mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(effective_models_dir(Path(tmp) / "explicit"), Path(tmp) / "explicit")
                self.assertEqual(effective_models_dir(None), Path(tmp) / "from-env")

    def test_a_malformed_config_file_never_blocks_a_model_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            path = home / ".orbit" / "config.json"; path.parent.mkdir(parents=True); path.write_text("{not json")
            resolution = resolve_models_dir(environ=_env(home))
            self.assertEqual((resolution.path, resolution.source), (default_models_dir(), "default"))
            self.assertIn(str(path), resolution.config_error or "")
            _write_config(home, {MODELS_DIR_CONFIG_KEY: 42})
            self.assertIn("non-empty string", resolve_models_dir(environ=_env(home)).config_error or "")


class PersistenceTests(unittest.TestCase):
    def test_E_configured_directory_is_created_when_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; target = Path(tmp) / "nested" / "orbit-models"
            self.assertFalse(target.exists())
            written = persist_models_dir(target, environ=_env(home))
            self.assertEqual(written, target)
            self.assertTrue(target.is_dir())
            config = json.loads((home / ".orbit" / "config.json").read_text())
            self.assertEqual(config, {MODELS_DIR_CONFIG_KEY: str(target)})
            self.assertEqual(resolve_models_dir(environ=_env(home)).path, target)

    def test_E_a_relative_path_is_anchored_to_the_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with mock.patch("orbit.native_llama.model_store.Path.cwd", return_value=Path(tmp)):
                written = persist_models_dir("rel-models", environ=_env(home))
            self.assertEqual(written, Path(tmp) / "rel-models")
            self.assertTrue(written.is_dir())

    @unittest.skipIf(os.geteuid() == 0, "permission checks are meaningless as root")
    def test_F_an_unwritable_location_fails_clearly_and_never_runs_sudo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; existing = _write_config(home, {"timeout": 42})
            locked = Path(tmp) / "locked"; locked.mkdir(); locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
            forbidden = mock.Mock(side_effect=AssertionError("Orbit must never spawn a process here"))
            try:
                with mock.patch.object(subprocess, "Popen", forbidden), mock.patch.object(subprocess, "run", forbidden), \
                     mock.patch.object(os, "system", forbidden), mock.patch.object(os, "execvp", forbidden):
                    with self.assertRaises(ModelsDirConfigError) as ctx:
                        persist_models_dir(locked / "orbit-models", environ=_env(home))
            finally:
                locked.chmod(stat.S_IRWXU)
            forbidden.assert_not_called()
            message = str(ctx.exception)
            self.assertIn("permission denied", message)
            self.assertIn("never runs sudo", message)
            self.assertNotIn("sudo ", message.replace("runs sudo", ""))
            # nothing was written and the existing settings survive
            self.assertEqual(json.loads(existing.read_text()), {"timeout": 42})
            with self.assertRaisesRegex(ModelsDirConfigError, "a file with that name exists"):
                persist_models_dir(existing, environ=_env(home))

    @unittest.skipIf(os.geteuid() == 0, "permission checks are meaningless as root")
    def test_F_an_unwritable_config_location_creates_no_models_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; (home / ".orbit").mkdir(parents=True)
            (home / ".orbit").chmod(stat.S_IRUSR | stat.S_IXUSR)
            target = Path(tmp) / "store"
            try:
                with self.assertRaisesRegex(ModelsDirConfigError, "not writable"):
                    persist_models_dir(target, environ=_env(home))
            finally:
                (home / ".orbit").chmod(stat.S_IRWXU)
            self.assertFalse(target.exists())  # nothing half-done on disk

    def test_a_symlinked_config_file_is_rewritten_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; dotfiles = Path(tmp) / "dotfiles" / "orbit.json"
            dotfiles.parent.mkdir(); dotfiles.write_text(json.dumps({"timeout": 7}))
            (home / ".orbit").mkdir(parents=True); (home / ".orbit" / "config.json").symlink_to(dotfiles)
            persist_models_dir(Path(tmp) / "store", environ=_env(home))
            self.assertTrue((home / ".orbit" / "config.json").is_symlink())
            self.assertEqual(json.loads(dotfiles.read_text()), {"timeout": 7, MODELS_DIR_CONFIG_KEY: str(Path(tmp) / "store")})

    def test_a_relative_persisted_value_is_reported_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; _write_config(home, {MODELS_DIR_CONFIG_KEY: "rel/models"})
            resolution = resolve_models_dir(environ=_env(home))
            self.assertEqual((resolution.path, resolution.source), (default_models_dir(), "default"))
            self.assertIn("absolute path", resolution.config_error or "")

    def test_F_a_corrupt_config_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            path = home / ".orbit" / "config.json"; path.parent.mkdir(parents=True); path.write_text("{not json")
            with self.assertRaisesRegex(ModelsDirConfigError, "not valid JSON"):
                persist_models_dir(Path(tmp) / "store", environ=_env(home))
            self.assertEqual(path.read_text(), "{not json")

    def test_G_a_models_directory_that_is_a_symlink_keeps_working(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; real = Path(tmp) / "real-store"; link = Path(tmp) / "models"
            real.mkdir(); link.symlink_to(real, target_is_directory=True)
            manifest = get_manifest(ORNITH_ID)
            target = local_model_path(manifest.target, models_dir=link)
            target.parent.mkdir(parents=True); target.write_bytes(b"gguf")
            # persisted value points at the symlink and is kept verbatim
            persist_models_dir(link, environ=_env(home))
            self.assertEqual(resolve_models_dir(environ=_env(home)).path, link)
            with mock.patch.dict(os.environ, _env(home), clear=False):
                os.environ.pop(MODELS_DIR_ENV, None)
                resolved = resolve_model(manifest, hf_cache=Path(tmp) / "hf")
                self.assertEqual(resolved.target_path, target)
                discovered = discover_models(hf_cache=Path(tmp) / "hf", inspector=_inspector_for(target))
                row = {r.model: r for r in discovered.rows}["Ornith 1.5 35B-A3B"]
                self.assertEqual(row.local, "AVAILABLE")

    def test_I_unrelated_configuration_is_preserved_and_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            _write_config(home, {"timeout": 42.0, "base_url": "http://127.0.0.1:9999", "think": True})
            persist_models_dir(Path(tmp) / "store", environ=_env(home))
            data = json.loads((home / ".orbit" / "config.json").read_text())
            self.assertEqual(data["timeout"], 42.0); self.assertEqual(data["base_url"], "http://127.0.0.1:9999")
            self.assertIs(data["think"], True); self.assertEqual(data[MODELS_DIR_CONFIG_KEY], str(Path(tmp) / "store"))
            # the terminal client reads the same file and ignores the new key
            parser = argparse.ArgumentParser(); add_config_arguments(parser)
            args = parser.parse_args(["--config", str(home / ".orbit" / "config.json")])
            config = load_app_config(args)
            self.assertEqual((config.timeout, config.base_url, config.think), (42.0, "http://127.0.0.1:9999", True))
        self.assertEqual(DEFAULT_CONFIG_PATH, orbit_config_path())


MOUNTINFO = (
    "22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
    "40 22 0:38 / /home/user rw,nosuid,nodev,relatime shared:2 - ecryptfs /home/.ecryptfs/user/.Private rw\n"
    "41 22 0:5 / /dev rw,nosuid shared:3 - devtmpfs devtmpfs rw\n"
)


class AdvisoryTests(unittest.TestCase):
    def test_filesystem_type_uses_the_longest_matching_mount(self) -> None:
        self.assertEqual(filesystem_type(Path("/home/user/.orbit/models"), mountinfo=MOUNTINFO), "ecryptfs")
        self.assertEqual(filesystem_type(Path("/var/tmp/x"), mountinfo=MOUNTINFO), "ext4")
        self.assertIsNone(filesystem_type(Path("/var/tmp/x"), mountinfo="garbage"))

    def test_H_advisory_is_silent_in_good_configurations(self) -> None:
        self.assertIsNone(large_model_advisory(Path("/x"), model_bytes=80 * GIB, mem_total_bytes=32 * GIB, filesystem="ext4"))
        self.assertIsNone(large_model_advisory(Path("/x"), model_bytes=20 * GIB, mem_total_bytes=32 * GIB, filesystem="ecryptfs"))
        self.assertIsNone(large_model_advisory(Path("/x"), model_bytes=None, mem_total_bytes=32 * GIB, filesystem="btrfs"))

    def test_H_advisory_text_for_a_beyond_ram_model_on_ecryptfs(self) -> None:
        text = large_model_advisory(Path("/home/user/models"), model_bytes=80 * GIB, mem_total_bytes=32 * GIB, filesystem="ecryptfs")
        self.assertIn("This model is larger than the system RAM.", text)
        self.assertIn("on ecryptfs, which may reduce performance", text)
        self.assertIn("/home/user/models", text)
        self.assertIn("orbit config models-dir", text)
        for detail in ("page cache", "symlink", "/srv", "GiB"):
            self.assertNotIn(detail, text)
        unknown = large_model_advisory(Path("/home/user/models"), model_bytes=None, mem_total_bytes=32 * GIB, filesystem="ecryptfs")
        self.assertIn("may be larger", unknown)
        verbose = large_model_advisory(Path("/home/user/models"), model_bytes=80 * GIB, mem_total_bytes=32 * GIB, filesystem="ecryptfs", verbose=True)
        self.assertIn("80.0 GiB", verbose)

    def test_H_advisory_never_changes_what_the_download_cli_does(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            args = SimpleNamespace(spec="owner/repo/model.gguf", all=False, mmproj=False, models_dir=str(store))
            calls = []

            def fake_download(spec, *, models_dir, prefer, progress):
                calls.append((spec, models_dir, prefer)); return DownloadResult(path=models_dir / "m.gguf", downloaded=True, url="u")

            stderr = io.StringIO()
            with mock.patch.object(download_cli, "download_model", side_effect=fake_download), \
                 mock.patch.object(download_cli, "download_advisory", return_value="ADVISORY") as advisory, \
                 redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                self.assertEqual(download_cli._download(args), 0)
            self.assertEqual(calls, [("owner/repo/model.gguf", store, "target")])
            self.assertIn("ADVISORY", stderr.getvalue())
            self.assertEqual(advisory.call_count, 1)  # once per operation
            self.assertEqual(advisory.call_args.kwargs["destination"], store / "owner--repo" / "model.gguf")
            # and a failing helper is swallowed: the download still runs, silently
            calls.clear(); stderr = io.StringIO()
            with mock.patch.object(download_cli, "download_model", side_effect=fake_download), \
                 mock.patch.object(download_cli, "download_advisory", side_effect=RuntimeError("no network")), \
                 redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                self.assertEqual(download_cli._download(args), 0)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("ADVISORY", stderr.getvalue())

    def test_H_the_size_probe_only_runs_on_an_unfavorable_filesystem(self) -> None:
        probe = mock.Mock(return_value=80 * GIB)
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp); present = store / "present.gguf"; present.write_bytes(b"x")
            with mock.patch.object(model_store, "filesystem_type", return_value="ext4"), \
                 mock.patch.object(model_store, "total_memory_bytes", return_value=32 * GIB):
                self.assertIsNone(download_advisory(store, url="u", size_probe=probe))
            probe.assert_not_called()  # a healthy filesystem stays offline
            with mock.patch.object(model_store, "filesystem_type", return_value="ecryptfs"), \
                 mock.patch.object(model_store, "total_memory_bytes", return_value=32 * GIB):
                self.assertIsNone(download_advisory(store, url="u", destination=present, size_probe=probe))
                probe.assert_not_called()  # already present: nothing to advise, no network
                text = download_advisory(store, url="u", destination=store / "missing.gguf", size_probe=probe)
                self.assertIn("larger than the system RAM", text)
                probe.assert_called_once_with("u")
                broken = mock.Mock(side_effect=RuntimeError("dns hang"))
                self.assertIsNone(download_advisory(store, url="u", size_probe=broken))


class ConfigCliTests(unittest.TestCase):
    def _run(self, argv, env):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = model_store.main(argv, environ=env)
        return code, out.getvalue(), err.getvalue()

    def test_print_shows_effective_directory_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            code, out, _ = self._run(["models-dir"], _env(home))
            self.assertEqual(code, 0)
            self.assertIn(f"models directory: {default_models_dir()}", out)
            self.assertIn("source: default", out)
            store = Path(tmp) / "store"
            code, out, _ = self._run(["models-dir", str(store)], _env(home))
            self.assertEqual(code, 0)
            self.assertIn(f"models directory set to: {store}", out)
            self.assertIn(str(home / ".orbit" / "config.json"), out)
            code, out, _ = self._run(["config", "models-dir"], _env(home))
            self.assertIn(f"models directory: {store}", out)
            self.assertIn("source: orbit config models-dir", out)
            code, out, _ = self._run(["models-dir"], _env(home, ORBIT_MODELS_DIR=str(Path(tmp) / "env")))
            self.assertIn("source: ORBIT_MODELS_DIR", out)

    def test_orbit_cli_dispatches_and_documents_the_config_command(self) -> None:
        self.assertIn("orbit config models-dir [PATH]", terminal_cli.build_parser().format_help())
        with mock.patch("orbit.terminal.cli.config_main", return_value=5) as mocked:
            self.assertEqual(terminal_cli.main(["config", "models-dir", "/x"]), 5)
        mocked.assert_called_once_with(["models-dir", "/x"])

    def test_setting_while_the_environment_overrides_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            code, out, _ = self._run(["models-dir", str(Path(tmp) / "store")], _env(home, ORBIT_MODELS_DIR=str(Path(tmp) / "env")))
            self.assertEqual(code, 0)
            self.assertIn("note: ORBIT_MODELS_DIR environment variable currently overrides it", out)

    def test_errors_are_clear_and_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            code, _, err = self._run(["timeout", "5"], _env(home))
            self.assertEqual(code, 2); self.assertIn("unknown config setting", err)
            code, _, err = self._run(["models-dir", "a", "b"], _env(home))
            self.assertEqual(code, 2)
            code, _, err = self._run(["models-dir", "/proc/orbit-cannot-exist/models"], _env(home))
            self.assertEqual(code, 1); self.assertIn("error: cannot create", err)
            code, out, _ = self._run([], _env(home))
            self.assertEqual(code, 0); self.assertIn("orbit config models-dir", out)


if __name__ == "__main__":
    unittest.main()
