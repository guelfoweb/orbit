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

import pathlib
import sys
import types
import unittest

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


if __name__ == "__main__":
    unittest.main()
