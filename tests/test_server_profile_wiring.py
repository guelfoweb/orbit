"""The server's side of profile resolution: the parser, and what reads it.

The policy module is tested against hand-built dicts, which cannot see the
thing most likely to break: `orbit server`'s own argparse defaults. They were
`6/6/256/128`, and the entire precedence contract rests on them being `None` --
with an int default, argparse cannot distinguish `--threads 6` from an
unsupplied flag, every field reports itself as `(cli)`, and calibration never
runs again. Reverting that one word would leave the policy suite green.

These also cover the helpers that run before the model loads, where a raised
exception costs a server start rather than a cache miss.
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_server import app as app_module  # noqa: E402
from orbit.native_server.server_profile import (  # noqa: E402
    FALLBACK_BATCH,
    FALLBACK_THREADS,
    FALLBACK_UBATCH,
)


class ParserDefaultTests(unittest.TestCase):
    """The regression the policy suite structurally cannot catch."""

    def parse(self, argv):
        return app_module.build_parser().parse_args(argv)

    def test_the_tuning_flags_default_to_none_not_to_numbers(self) -> None:
        args = self.parse([])
        for field in ("threads", "threads_batch", "batch", "ubatch"):
            self.assertIsNone(
                getattr(args, field),
                f"{field} must default to None so an unsupplied flag is "
                f"distinguishable from an explicit one",
            )

    def test_the_reference_numbers_are_still_reachable_explicitly(self) -> None:
        args = self.parse(
            ["--threads", str(FALLBACK_THREADS),
             "--threads-batch", str(FALLBACK_THREADS),
             "--batch", str(FALLBACK_BATCH), "--ubatch", str(FALLBACK_UBATCH)]
        )
        self.assertEqual(args.threads, FALLBACK_THREADS)
        self.assertEqual(args.batch, FALLBACK_BATCH)
        self.assertEqual(args.ubatch, FALLBACK_UBATCH)

    def test_ctx_keeps_its_default(self) -> None:
        """ctx is NOT auto-resolved: it is a capability, not tuning."""
        self.assertEqual(self.parse([]).ctx, 8192)

    def test_the_new_flags_exist_and_default_off(self) -> None:
        args = self.parse([])
        self.assertFalse(args.show_profile)
        self.assertFalse(args.recalibrate)
        self.assertTrue(self.parse(["--show-profile"]).show_profile)
        self.assertTrue(self.parse(["--recalibrate"]).recalibrate)

    def test_nothing_auto_enables_a_capability(self) -> None:
        args = self.parse([])
        self.assertFalse(args.enable_mtp_experimental)
        self.assertFalse(args.moe_expert_usage)
        self.assertFalse(args.low_memory)


class _Args(types.SimpleNamespace):
    """A parsed-args stand-in with the fields the helpers read."""

    def __init__(self, **kwargs) -> None:
        defaults = dict(
            threads=None, threads_batch=None, batch=None, ubatch=None,
            ctx=8192, model=None, model_id=None, models_dir=None,
            hf_cache=None, mmproj=None, alias=None, llama_root=None,
            low_memory=False, enable_mtp_experimental=False,
            moe_expert_usage=False, recalibrate=False, show_profile=False,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


class IdentityHelperTests(unittest.TestCase):
    """These run before the model loads. They may not raise."""

    def test_an_unresolvable_model_is_a_cache_miss_not_a_crash(self) -> None:
        identity, size = app_module._model_identity_for_profile(
            _Args(model="/nonexistent/definitely/not/here.gguf")
        )
        self.assertEqual((identity, size), ("", 0))

    def test_a_nonsense_model_value_is_a_cache_miss_not_a_crash(self) -> None:
        """A test double or an odd type must not take down a server start."""
        for value in (object(), 12345, [], {"a": 1}):
            identity, size = app_module._model_identity_for_profile(_Args(model=value))
            self.assertEqual((identity, size), ("", 0))

    def test_a_real_file_produces_a_size_bearing_identity(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
            handle.write(b"x" * 1024)
            name = handle.name
        self.addCleanup(lambda: pathlib.Path(name).unlink(missing_ok=True))
        identity, size = app_module._model_identity_for_profile(_Args(model=name))
        self.assertEqual(size, 1024)
        self.assertIn("1024", identity)

    def test_two_different_models_do_not_share_an_identity(self) -> None:
        import tempfile

        names = []
        for payload in (b"a" * 10, b"b" * 20):
            with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
                handle.write(payload)
                names.append(handle.name)
        self.addCleanup(
            lambda: [pathlib.Path(n).unlink(missing_ok=True) for n in names]
        )
        first = app_module._model_identity_for_profile(_Args(model=names[0]))[0]
        second = app_module._model_identity_for_profile(_Args(model=names[1]))[0]
        self.assertNotEqual(first, second)

    def test_the_backend_identity_is_a_string_and_never_raises(self) -> None:
        self.assertIsInstance(app_module._backend_identity(), str)

    def test_the_backend_identity_prefers_the_release_tag(self) -> None:
        payload = {"upstream_tag": "b9551", "upstream_commit": "379ac6673b5cd75c7b4e07d1521c50f1e093878c"}
        with mock.patch.object(pathlib.Path, "read_text", return_value=__import__("json").dumps(payload)):
            self.assertEqual(app_module._backend_identity(), "b9551")

    def test_an_untagged_pin_is_identified_by_its_commit(self) -> None:
        # 41abbfd has no upstream release tag; the calibration cache must still
        # key on THIS pin rather than on the shared word "untagged".
        payload = {"upstream_tag": "untagged", "upstream_commit": "41abbfd599fbdd3470fcae0a1fb6530ad8403cd7"}
        with mock.patch.object(pathlib.Path, "read_text", return_value=__import__("json").dumps(payload)):
            self.assertEqual(app_module._backend_identity(), "41abbfd599fb")


class ConfigCorrectionTests(unittest.TestCase):
    """The calibrated counts must reach `client.config`, not only the context.

    `/props` publishes this field, the smoke harness records it, and both the
    native-version and final-prefix identities hash it -- so a stale config
    means one process reports threads it is not using and two identical
    runtimes compute different checkpoint identities.
    """

    def test_a_frozen_config_still_accepts_the_correction(self) -> None:
        from dataclasses import FrozenInstanceError

        from orbit.native_llama.client import NativeClientConfig

        config = NativeClientConfig(context_tokens=8192, threads=16,
                                    threads_batch=16, batch_size=256,
                                    ubatch_size=128)
        # Normal assignment is still refused: the dataclass is genuinely frozen
        # and this is a deliberate reach-around, not an accident.
        with self.assertRaises(FrozenInstanceError):
            config.threads = 6
        for field, value in (("threads", 6), ("threads_batch", 6)):
            object.__setattr__(config, field, value)
        self.assertEqual((config.threads, config.threads_batch), (6, 6))

    def test_the_correction_is_what_props_would_publish(self) -> None:
        from orbit.native_llama.client import NativeClientConfig

        config = NativeClientConfig(context_tokens=8192, threads=16,
                                    threads_batch=16, batch_size=256,
                                    ubatch_size=128)
        object.__setattr__(config, "threads", 6)
        object.__setattr__(config, "threads_batch", 6)
        self.assertEqual(config.threads, 6)
        self.assertNotEqual(config.threads, 16)


class ConfigCorrectionCallSiteTests(unittest.TestCase):
    """The correction must be applied by run_server, not merely be possible.

    `ConfigCorrectionTests` proves the technique; this proves the call site
    still uses it. Without it a mutant deleting the loop passes every other
    test, and a stale config silently corrupts `/props`, the smoke-harness
    record, and two checkpoint identity hashes.

    Driven by reading the source rather than by booting a server: the block
    sits between a 20 GiB load and a socket bind, and neither belongs in a unit
    test. What is pinned is that the correction exists, names both fields, and
    runs after `restore_threads` rather than before it.
    """

    def _calibration_block(self) -> str:
        import inspect

        source = inspect.getsource(app_module.run_server)
        start = source.index("restore_threads(client")
        return source[start:start + 1400]

    def test_the_call_site_corrects_both_thread_fields(self) -> None:
        block = self._calibration_block()
        self.assertIn("object.__setattr__(client.config", block)
        self.assertIn('"threads"', block)
        self.assertIn('"threads_batch"', block)

    def test_the_correction_follows_the_context_restore(self) -> None:
        """Order matters: the context is the thing actually serving."""
        import inspect

        source = inspect.getsource(app_module.run_server)
        self.assertLess(
            source.index("restore_threads(client"),
            source.index("object.__setattr__(client.config"),
        )

    def test_the_correction_does_not_swallow_every_exception(self) -> None:
        """A structural change must surface, not leave /props quietly stale."""
        block = self._calibration_block()
        self.assertIn("except (AttributeError, TypeError)", block)
        self.assertNotIn("except Exception", block)


class ResolveStartupProfileTests(unittest.TestCase):
    """The bridge between parsed args and the policy module."""

    def setUp(self) -> None:
        import os
        import tempfile

        self.cache = tempfile.mkdtemp(prefix="orbit-wiring-test-")
        self.addCleanup(__import__("shutil").rmtree, self.cache, True)
        self._old = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = self.cache

        def restore():
            if self._old is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = self._old

        self.addCleanup(restore)

    def test_an_unsupplied_flag_is_resolved_not_frozen(self) -> None:
        resolution = app_module._resolve_startup_profile(_Args())
        self.assertGreaterEqual(resolution.profile.threads, 1)
        self.assertNotEqual(resolution.profile.field_sources["threads"], "cli")

    def test_an_explicit_flag_survives_resolution(self) -> None:
        resolution = app_module._resolve_startup_profile(_Args(threads=3))
        self.assertEqual(resolution.profile.threads, 3)
        self.assertEqual(resolution.profile.field_sources["threads"], "cli")

    def test_previewing_never_calibrates(self) -> None:
        """`--show-profile` must not load a model or spend a sweep."""
        resolution = app_module._resolve_startup_profile(_Args())
        self.assertFalse(resolution.calibrated)

    def test_previewing_with_recalibrate_does_not_delete_the_measurement(self) -> None:
        """A preview that mutated state would be worse than no preview.

        `--show-profile --recalibrate` used to discard the stored measurement
        and then report "no cached measurement for this machine yet" -- it had
        just destroyed one.
        """
        from orbit.native_server.server_profile import (
            cache_path_for, store_cached_profile,
        )

        seeded = app_module._resolve_startup_profile(_Args())
        path = store_cached_profile(
            seeded.fingerprint, {"threads": 6, "threads_batch": 6}
        )
        self.assertIsNotNone(path)
        self.assertTrue(pathlib.Path(path).is_file())

        app_module._resolve_startup_profile(_Args(recalibrate=True))
        self.assertTrue(
            pathlib.Path(path).is_file(),
            "a preview must not discard the stored measurement",
        )
        self.assertEqual(
            cache_path_for(seeded.fingerprint), pathlib.Path(path)
        )

    def test_a_recalibrate_preview_does_not_claim_the_cache_is_empty(self) -> None:
        """It deliberately did not read the cache, so it cannot say."""
        from orbit.native_server.server_profile import store_cached_profile

        seeded = app_module._resolve_startup_profile(_Args())
        store_cached_profile(seeded.fingerprint, {"threads": 6, "threads_batch": 6})
        preview = app_module._resolve_startup_profile(_Args(recalibrate=True))
        self.assertIsNone(preview.cache_path)
        self.assertFalse(preview.calibrated)

    def test_capability_flags_change_the_fingerprint(self) -> None:
        plain = app_module._resolve_startup_profile(_Args()).fingerprint
        for flag in ("low_memory", "enable_mtp_experimental", "moe_expert_usage"):
            other = app_module._resolve_startup_profile(
                _Args(**{flag: True})
            ).fingerprint
            self.assertNotEqual(plain, other, f"{flag} must be fingerprinted")

    def test_the_context_size_changes_the_fingerprint(self) -> None:
        self.assertNotEqual(
            app_module._resolve_startup_profile(_Args()).fingerprint,
            app_module._resolve_startup_profile(_Args(ctx=4096)).fingerprint,
        )


class ShowProfileModelResolutionTests(unittest.TestCase):
    """SERVER-SHOW-PROFILE-MODEL-RESOLUTION-1: `--show-profile` reports the
    profile for the model a real start would use, read-only and model-aware."""

    from orbit.native_llama.model_discovery import ModelDiscoveryRow as _Row

    def _tmp_gguf(self, payload: bytes = b"G" * 2048) -> str:
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
            handle.write(payload)
            name = handle.name
        self.addCleanup(lambda: pathlib.Path(name).unlink(missing_ok=True))
        return name

    def _run_show_profile(self, args) -> "tuple[int, str]":
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = app_module._show_profile(args)
        return rc, out.getvalue()

    # T3/T11: an explicit model is resolved exactly, with the shared fingerprint.
    def test_explicit_model_uses_its_own_identity(self) -> None:
        path = self._tmp_gguf()
        name, disp, identity, missing = app_module._resolve_preview_target(_Args(model=path))
        self.assertEqual(identity, app_module._model_identity_for_profile(_Args(model=path)))
        self.assertFalse(missing)
        self.assertIn(pathlib.Path(path).name, name)
        self.assertEqual(disp, path)

    def test_explicit_missing_model_is_reported_missing_not_calibrated(self) -> None:
        name, disp, identity, missing = app_module._resolve_preview_target(
            _Args(model="/nope/not/here.gguf")
        )
        self.assertTrue(missing)
        self.assertEqual(identity, ("", 0))
        self.assertIn("not present", disp)

    # T1/T2: no explicit model -> resolves through the shared interactive
    # selection, and the selected model's identity finds its cached profile.
    def test_interactive_selection_resolves_selected_model(self) -> None:
        from orbit.native_server.server_profile import store_cached_profile

        path = self._tmp_gguf()
        row = self._Row("Ornith 1.5 35B-A3B", "AVAILABLE", "VERIFIED", path, model_id="ornith")
        with mock.patch.object(app_module, "_interactive_model_selection_requested", return_value=True), \
             mock.patch.object(app_module, "_choose_verified_model", return_value=(row, pathlib.Path("/bin"))):
            args = _Args()
            name, disp, identity, missing = app_module._resolve_preview_target(args)
            self.assertEqual(name, "Ornith 1.5 35B-A3B")
            self.assertFalse(missing)
            self.assertEqual(identity, app_module._model_identity_for_profile(_Args(model=path)))
            self.assertEqual(args.model, pathlib.Path(path))  # resolved, for the real fingerprint
            # T2: seed a measured profile at that fingerprint; the preview shows it.
            from orbit.native_server.server_profile import cache_path_for
            fp = app_module._resolve_startup_profile(_Args(model=path)).fingerprint
            store_cached_profile(fp, {"threads": 6, "threads_batch": 6, "batch": 256, "ubatch": 128})
            self.addCleanup(lambda: cache_path_for(fp).unlink(missing_ok=True))
            rc, text = self._run_show_profile(_Args())  # fresh args, same interactive mocks
        self.assertEqual(rc, 0)
        self.assertIn("Ornith 1.5 35B-A3B", text)
        self.assertIn("threads: 6", text)

    def test_interactive_available_pick_runs_memory_mode_selection(self) -> None:
        # Memory mode is a fingerprint axis; a low-memory-capable pick must ask
        # here too, so the preview keys the same fingerprint a real low-memory
        # start would (not the standard-mode cache entry).
        path = self._tmp_gguf()
        row = self._Row("Ornith", "AVAILABLE", "VERIFIED", path, model_id="ornith", low_memory_supported=True)

        def fake_memory_mode(args, selected):
            args.low_memory = True  # the operator picks low memory
            return None

        with mock.patch.object(app_module, "_interactive_model_selection_requested", return_value=True), \
             mock.patch.object(app_module, "_choose_verified_model", return_value=(row, pathlib.Path("/bin"))), \
             mock.patch.object(app_module, "_select_memory_mode", side_effect=fake_memory_mode) as memory_mode:
            args = _Args()
            app_module._resolve_preview_target(args)
        memory_mode.assert_called_once()
        self.assertTrue(args.low_memory)
        self.assertEqual(
            app_module._resolve_startup_profile(args).fingerprint,
            app_module._resolve_startup_profile(_Args(model=path, low_memory=True)).fingerprint,
        )

    def test_interactive_missing_pick_never_downloads(self) -> None:
        row = self._Row("Gemma 4 26B-A4B", "MISSING", "VERIFIED", "orbit download x/y.gguf", model_id="g")
        with mock.patch.object(app_module, "_interactive_model_selection_requested", return_value=True), \
             mock.patch.object(app_module, "_choose_verified_model", return_value=(row, pathlib.Path("/bin"))), \
             mock.patch.object(app_module, "_download_selected_model", side_effect=AssertionError("no download in preview")):
            rc, text = self._run_show_profile(_Args())
        self.assertEqual(rc, 0)
        self.assertIn("Gemma 4 26B-A4B", text)
        self.assertIn("not present locally", text)

    def test_interactive_cancel_returns_exit_code(self) -> None:
        with mock.patch.object(app_module, "_interactive_model_selection_requested", return_value=True), \
             mock.patch.object(app_module, "_choose_verified_model", return_value=1):
            self.assertEqual(app_module._show_profile(_Args()), 1)

    # T4/T5/T6: precedence is preserved (partial CLI/env overrides fill only the
    # missing fields; explicit wins). These reuse the resolver the preview uses.
    def test_explicit_cli_override_is_authoritative_in_preview(self) -> None:
        rc, text = self._run_show_profile(_Args(threads=3))
        self.assertEqual(rc, 0)
        self.assertIn("threads: 3 (cli)", text)

    def test_partial_cli_override_fills_only_that_field(self) -> None:
        res = app_module._resolve_startup_profile(_Args(batch=64))
        self.assertEqual(res.profile.batch, 64)
        self.assertEqual(res.profile.field_sources["batch"], "cli")
        self.assertNotEqual(res.profile.field_sources["threads"], "cli")

    # T7: no matching measured cache -> honest heuristic/fallback.
    def test_no_cached_profile_reports_heuristic_honestly(self) -> None:
        rc, text = self._run_show_profile(_Args(model="/nope/not/here.gguf"))
        self.assertEqual(rc, 0)
        self.assertIn("heuristic", text)
        self.assertNotIn("cached", text.split("note:")[0])

    # T9/T12: the preview never constructs an inference client / loads a model.
    def test_preview_never_builds_a_client(self) -> None:
        with mock.patch.object(app_module, "NativeLlamaClient", side_effect=AssertionError("no load in preview")):
            rc, text = self._run_show_profile(_Args(model=self._tmp_gguf()))
        self.assertEqual(rc, 0)
        self.assertIn("model:", text)

    # T10: the preview never creates or mutates a cache entry.
    def test_preview_does_not_create_cache_entry(self) -> None:
        from orbit.native_server.server_profile import cache_path_for

        args = _Args(model=self._tmp_gguf())
        fp = app_module._resolve_startup_profile(args).fingerprint
        cache = cache_path_for(fp)
        self.assertFalse(cache.is_file())
        self._run_show_profile(_Args(model=args.model))
        self.assertFalse(cache.is_file(), "a preview must not create a cache entry")

    # T13: non-interactive is safe, names the default, and is deterministic.
    def test_non_interactive_default_is_named_and_deterministic(self) -> None:
        with mock.patch.object(app_module, "_interactive_model_selection_requested", return_value=False):
            rc1, a = self._run_show_profile(_Args())
            rc2, b = self._run_show_profile(_Args())
        self.assertEqual((rc1, rc2), (0, 0))
        self.assertEqual(a, b)
        self.assertIn("model:", a)
        self.assertNotIn("\x1b[", a)

    # T5: an ORBIT_* env override remains authoritative in the preview (the
    # preview uses the same resolver, which reads os.environ).
    def test_env_override_is_authoritative_in_preview(self) -> None:
        import os

        with mock.patch.dict(os.environ, {"ORBIT_THREADS": "7"}):
            rc, text = self._run_show_profile(_Args(model="/nope/not/here.gguf"))
        self.assertEqual(rc, 0)
        self.assertIn("threads: 7 (env)", text)

    # T12: the refactor left normal startup selection intact -- an AVAILABLE
    # pick sets the model and proceeds; a MISSING pick still downloads.
    def test_select_startup_model_sets_available_pick(self) -> None:
        row = self._Row("Ornith", "AVAILABLE", "VERIFIED", "/models/ornith.gguf", model_id="ornith")
        args = _Args()
        with mock.patch.object(app_module, "_choose_verified_model", return_value=(row, pathlib.Path("/bin"))), \
             mock.patch.object(app_module, "_select_memory_mode", return_value=None):
            rc = app_module._select_startup_model(args)
        self.assertIsNone(rc)
        self.assertEqual(args.model, pathlib.Path("/models/ornith.gguf"))
        self.assertEqual(args.model_id, "ornith")

    def test_select_startup_model_downloads_missing_pick(self) -> None:
        row = self._Row("Gemma", "MISSING", "VERIFIED", "orbit download x/y.gguf", model_id="g")
        with mock.patch.object(app_module, "_choose_verified_model", return_value=(row, pathlib.Path("/bin"))), \
             mock.patch.object(app_module, "_download_selected_model", return_value=0) as download:
            rc = app_module._select_startup_model(_Args())
        download.assert_called_once()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
