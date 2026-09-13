"""Startup profile resolution: precedence, fingerprinting, and failing safe.

The contract these pin is short and absolute: a value the operator supplied is
never replaced, never measured, and never cached over. Everything else here --
the fingerprint, the atomic write, the memory arithmetic, the fallbacks -- exists
so that Orbit choosing the rest cannot make a machine worse than the fixed
numbers it used to ship.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_server.server_profile import (  # noqa: E402
    ENV_BATCH,
    ENV_CACHE_RAM,
    ENV_THREADS,
    ENV_THREADS_BATCH,
    ENV_UBATCH,
    FALLBACK_BATCH,
    FALLBACK_THREADS,
    FALLBACK_UBATCH,
    MAX_THREADS,
    MIN_CACHE_RAM_MIB,
    PROFILE_FORMAT_VERSION,
    HostTopology,
    ServerProfile,
    cache_path_for,
    clamp_profile,
    compute_cache_ram_mib,
    detect_topology,
    discard_cached_profile,
    env_overrides,
    fallback_profile,
    heuristic_profile,
    load_cached_profile,
    profile_fingerprint,
    render_profile_lines,
    resolve_profile,
    store_cached_profile,
    thread_candidates,
)

GIB = 1024

# A 16-core hybrid laptop with 31 GiB: the machine whose measurement showed
# decode falling from 19 to 5 tok/s as threads rose, which is why any of this
# measures rather than counts.
DELL = HostTopology(
    cpu_model="Intel(R) Core(TM) Ultra 7 366H",
    physical_cores=16, logical_cpus=16,
    total_ram_mib=31 * GIB, available_ram_mib=11 * GIB, swap_total_mib=2 * GIB,
)
# The reference NUC: 6 physical, 12 logical, 64 GiB.
NUC = HostTopology(
    cpu_model="Intel(R) Core(TM) i7-10710U",
    physical_cores=6, logical_cpus=12,
    total_ram_mib=64 * GIB, available_ram_mib=40 * GIB, swap_total_mib=8 * GIB,
)
TINY = HostTopology(
    cpu_model="tiny", physical_cores=2, logical_cpus=2,
    total_ram_mib=4 * GIB, available_ram_mib=1 * GIB, swap_total_mib=0,
)

NO_CLI = {
    "threads": None, "threads_batch": None,
    "batch": None, "ubatch": None, "cache_ram_mib": None,
}


def _cli(**kwargs):
    values = dict(NO_CLI)
    values.update(kwargs)
    return values


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.cache = tempfile.mkdtemp(prefix="orbit-profile-test-")
        self.addCleanup(__import__("shutil").rmtree, self.cache, True)
        # XDG_CACHE_HOME redirects the profile store, so no test can read or
        # write the developer's real cache.
        self.env = {"XDG_CACHE_HOME": self.cache}

    def resolve(self, **kwargs):
        params = {
            "cli": NO_CLI, "topology": DELL, "environ": self.env,
            "model_sha256": "model-a", "backend_id": "b9551",
        }
        params.update(kwargs)
        return resolve_profile(**params)


class PrecedenceTests(_Case):
    """T1/T2/T3. Explicit beats everything, field by field."""

    def test_cli_beats_every_other_level(self) -> None:
        env = dict(self.env, **{ENV_THREADS: "9", ENV_BATCH: "1024"})
        resolution = self.resolve(
            cli=_cli(threads=3, batch=64), environ=env,
            calibrator=lambda **_: ({"threads": 11, "threads_batch": 11}, []),
        )
        self.assertEqual(resolution.profile.threads, 3)
        self.assertEqual(resolution.profile.batch, 64)
        self.assertEqual(resolution.profile.field_sources["threads"], "cli")

    def test_env_beats_cache_calibration_and_heuristic(self) -> None:
        env = dict(self.env, **{ENV_THREADS: "9", ENV_UBATCH: "32"})
        resolution = self.resolve(
            environ=env,
            calibrator=lambda **_: ({"threads": 11, "threads_batch": 11}, []),
        )
        self.assertEqual(resolution.profile.threads, 9)
        self.assertEqual(resolution.profile.ubatch, 32)
        self.assertEqual(resolution.profile.field_sources["threads"], "env")

    def test_a_partial_override_leaves_the_rest_resolvable(self) -> None:
        """T3. Naming threads must not freeze batch, ubatch or cache_ram."""
        resolution = self.resolve(cli=_cli(threads=5))
        sources = resolution.profile.field_sources
        self.assertEqual(resolution.profile.threads, 5)
        self.assertEqual(sources["threads"], "cli")
        for field in ("threads_batch", "batch", "ubatch", "cache_ram_mib"):
            self.assertNotEqual(sources[field], "cli")

    def test_a_named_field_is_never_offered_to_the_calibrator(self) -> None:
        """An explicit field must not even be ASKED about.

        It is not enough that the answer would be discarded: measuring costs a
        ~47-second sweep, and spending it to re-answer a question the operator
        already answered is the same defect as overriding them, priced
        differently.
        """
        asked: list[tuple] = []

        def calibrator(*, topology, fields):
            asked.append(fields)
            return {"threads": 11, "threads_batch": 11}, []

        self.resolve(cli=_cli(threads=5, threads_batch=5), calibrator=calibrator)
        for fields in asked:
            self.assertNotIn("threads", fields)
            self.assertNotIn("threads_batch", fields)

    def test_all_explicit_measurable_fields_skip_calibration_entirely(self) -> None:
        asked: list[tuple] = []

        def calibrator(*, topology, fields):
            asked.append(fields)
            return {"threads": 11, "threads_batch": 11}, []

        self.resolve(
            cli=_cli(threads=5, threads_batch=5, batch=256, ubatch=128),
            calibrator=calibrator,
        )
        self.assertEqual(asked, [])

    def test_a_user_profile_sits_below_env_and_above_the_cache(self) -> None:
        env = dict(self.env, **{ENV_THREADS: "9"})
        resolution = self.resolve(
            environ=env, user_profile={"threads": 7, "batch": 512},
        )
        self.assertEqual(resolution.profile.threads, 9)          # env wins
        self.assertEqual(resolution.profile.batch, 512)          # user profile
        self.assertEqual(resolution.profile.field_sources["batch"], "user-profile")

    def test_a_fully_explicit_invocation_reports_itself_as_explicit(self) -> None:
        resolution = self.resolve(
            cli=_cli(threads=6, threads_batch=6, batch=256, ubatch=128,
                     cache_ram_mib=8192)
        )
        self.assertEqual(resolution.profile.source, "explicit")
        self.assertFalse(resolution.calibrated)

    def test_a_malformed_env_value_is_ignored_not_defaulted(self) -> None:
        env = dict(self.env, **{ENV_THREADS: "fast", ENV_BATCH: "-4"})
        self.assertEqual(env_overrides(env), {})
        resolution = self.resolve(environ=env)
        self.assertNotEqual(resolution.profile.field_sources["threads"], "env")


class QualifiedConfigTests(_Case):
    """T11. The qualified invocation must resolve to the qualified numbers."""

    def test_the_qualified_flags_survive_unchanged(self) -> None:
        resolution = self.resolve(
            cli=_cli(threads=6, threads_batch=6, batch=256, ubatch=128),
            calibrator=lambda **_: ({"threads": 16, "threads_batch": 16}, []),
        )
        profile = resolution.profile
        self.assertEqual(
            (profile.threads, profile.threads_batch, profile.batch, profile.ubatch),
            (6, 6, 256, 128),
        )


class CalibrationTests(_Case):
    """T6. A measurement is an optimisation, never a dependency."""

    def test_a_successful_calibration_is_used_and_stored(self) -> None:
        resolution = self.resolve(
            calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, [{"x": 1}]),
        )
        self.assertTrue(resolution.calibrated)
        self.assertEqual(resolution.profile.threads, 6)
        self.assertTrue(resolution.profile.source.startswith("auto-calibrated"))
        self.assertIn("threads", resolution.profile.source)
        self.assertIsNotNone(resolution.cache_path)
        self.assertTrue(pathlib.Path(resolution.cache_path).is_file())

    def test_a_raising_calibrator_falls_back_and_records_why(self) -> None:
        def boom(**_):
            raise MemoryError("no room")

        resolution = self.resolve(calibrator=boom)
        self.assertFalse(resolution.calibrated)
        self.assertIn("MemoryError", resolution.calibration_error or "")
        # Still a usable profile, from the heuristic.
        self.assertGreaterEqual(resolution.profile.threads, 1)
        self.assertEqual(resolution.profile.source, "heuristic")

    def test_a_calibrator_returning_none_falls_back(self) -> None:
        resolution = self.resolve(calibrator=lambda **_: None)
        self.assertFalse(resolution.calibrated)
        self.assertIsNotNone(resolution.calibration_error)
        self.assertEqual(resolution.profile.source, "heuristic")

    def test_calibration_is_skipped_when_disallowed(self) -> None:
        called: list[int] = []

        def calibrator(**_):
            called.append(1)
            return {"threads": 11, "threads_batch": 11}, []

        resolution = self.resolve(calibrator=calibrator, allow_calibration=False)
        self.assertEqual(called, [])
        self.assertFalse(resolution.calibrated)

    def test_cache_ram_is_never_measured(self) -> None:
        """§4 is arithmetic. A calibrator must not be asked for it."""
        asked: list[tuple] = []

        def calibrator(*, topology, fields):
            asked.append(fields)
            return {"threads": 6, "threads_batch": 6}, []

        self.resolve(calibrator=calibrator)
        self.assertTrue(asked)
        for fields in asked:
            self.assertNotIn("cache_ram_mib", fields)


class CacheTests(_Case):
    """T4/T5/T10. A stored measurement, and when it stops applying."""

    def test_a_cached_profile_is_reused_without_measuring(self) -> None:
        first = self.resolve(
            calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []),
        )
        self.assertTrue(first.calibrated)
        calls: list[int] = []

        def calibrator(**_):
            calls.append(1)
            return {"threads": 16, "threads_batch": 16}, []

        second = self.resolve(calibrator=calibrator)
        self.assertEqual(calls, [])
        self.assertEqual(second.profile.threads, 6)
        self.assertTrue(second.profile.source.startswith("cached auto-calibrated"))

    def test_a_different_model_invalidates_the_cache(self) -> None:
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        calls: list[int] = []

        def calibrator(**_):
            calls.append(1)
            return {"threads": 8, "threads_batch": 8}, []

        other = self.resolve(model_sha256="model-b", calibrator=calibrator)
        self.assertEqual(len(calls), 1)
        self.assertEqual(other.profile.threads, 8)

    def test_a_different_backend_invalidates_the_cache(self) -> None:
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        calls: list[int] = []
        self.resolve(
            backend_id="b9999",
            calibrator=lambda **_: (calls.append(1), ({"threads": 8, "threads_batch": 8}, []))[1],
        )
        self.assertEqual(len(calls), 1)

    def test_different_hardware_invalidates_the_cache(self) -> None:
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        calls: list[int] = []
        self.resolve(
            topology=NUC,
            calibrator=lambda **_: (calls.append(1), ({"threads": 4, "threads_batch": 4}, []))[1],
        )
        self.assertEqual(len(calls), 1)

    def test_core_count_alone_invalidates_the_cache(self) -> None:
        """Same CPU string, same RAM, fewer cores -- still a different machine.

        A container with a narrowed CPU affinity, or a host that lost a core to
        an offline sibling, reports the same model name and the same MemTotal.
        If the core counts were not fingerprinted, such a box would silently
        inherit a thread count measured against hardware it no longer has.
        """
        narrowed = HostTopology(
            cpu_model=DELL.cpu_model, physical_cores=4, logical_cpus=4,
            total_ram_mib=DELL.total_ram_mib,
            available_ram_mib=DELL.available_ram_mib,
            swap_total_mib=DELL.swap_total_mib,
        )
        self.assertNotEqual(
            profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551"),
            profile_fingerprint(narrowed, model_sha256="model-a", backend_id="b9551"),
        )
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        calls: list[int] = []
        self.resolve(
            topology=narrowed,
            calibrator=lambda **_: (calls.append(1), ({"threads": 4, "threads_batch": 4}, []))[1],
        )
        self.assertEqual(len(calls), 1)

    def test_low_memory_mode_invalidates_the_cache(self) -> None:
        """Same machine, same model, >10 GiB different resident set.

        `--low-memory` is an inference-shape flag, not a cosmetic one: it moved
        peak RSS from ~31 GiB to ~18 GiB on a verified profile. A thread count
        and a cache size measured in one mode are not measurements of the other.
        """
        self.assertNotEqual(
            profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551"),
            profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551",
                                low_memory=True),
        )
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        calls: list[int] = []
        self.resolve(
            low_memory=True,
            calibrator=lambda **_: (calls.append(1), ({"threads": 8, "threads_batch": 8}, []))[1],
        )
        self.assertEqual(len(calls), 1)

    def test_mtp_and_expert_usage_invalidate_the_cache(self) -> None:
        """Both change what the benchmark would have measured.

        MTP alters KV allocation at context creation; expert-usage counters add
        per-token instrumentation inside the CPU backend -- inside the thing
        being timed.
        """
        base = profile_fingerprint(DELL, model_sha256="m", backend_id="b")
        self.assertNotEqual(
            base, profile_fingerprint(DELL, model_sha256="m", backend_id="b",
                                      mtp_enabled=True))
        self.assertNotEqual(
            base, profile_fingerprint(DELL, model_sha256="m", backend_id="b",
                                      expert_usage_enabled=True))

    def test_a_zero_or_negative_explicit_value_is_passed_over(self) -> None:
        """One validation policy for every level, not two."""
        resolution = self.resolve(cli=_cli(threads=0, batch=-8))
        self.assertNotEqual(resolution.profile.field_sources["threads"], "cli")
        self.assertNotEqual(resolution.profile.field_sources["batch"], "cli")
        self.assertGreaterEqual(resolution.profile.threads, 1)

    def test_a_different_ctx_invalidates_the_cache(self) -> None:
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        calls: list[int] = []
        self.resolve(
            ctx_tokens=4096,
            calibrator=lambda **_: (calls.append(1), ({"threads": 4, "threads_batch": 4}, []))[1],
        )
        self.assertEqual(len(calls), 1)

    def test_recalibrate_discards_the_stored_measurement(self) -> None:
        first = self.resolve(
            calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []),
        )
        self.assertTrue(pathlib.Path(first.cache_path).is_file())
        calls: list[int] = []
        second = self.resolve(
            recalibrate=True,
            calibrator=lambda **_: (calls.append(1), ({"threads": 8, "threads_batch": 8}, []))[1],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(second.profile.threads, 8)

    def test_only_measured_fields_are_written_to_the_cache(self) -> None:
        """The BLOCKER this replaced: a cache entry is a measurement record.

        Storing the resolved profile persisted the operator's one-off
        `--batch 1024` into ~/.cache and replayed it on every later start,
        labelled `cached auto-calibrated` -- an experiment typed once becoming
        a permanent setting that reported itself as a benchmark result.
        """
        resolution = self.resolve(
            cli=_cli(batch=1024, ubatch=512),
            calibrator=lambda **_: ({"threads": 11, "threads_batch": 11}, []),
        )
        payload = json.loads(pathlib.Path(resolution.cache_path).read_text())
        self.assertEqual(payload["threads"], 11)
        for leaked in ("batch", "ubatch", "cache_ram_mib", "source"):
            self.assertNotIn(leaked, payload)

    def test_an_explicit_thread_count_is_not_stored_as_a_measurement(self) -> None:
        """The narrow case the field list alone does not cover.

        `threads` IS a measurable field, so restricting the write to measurable
        fields is not by itself enough: an operator who sets `--threads 8` and
        leaves `--threads-batch` unset would still have their 8 written to the
        cache and replayed later as though a benchmark chose it. Only the
        `sources == "calibrated"` filter prevents that.
        """
        resolution = self.resolve(
            cli=_cli(threads=8),
            calibrator=lambda **_: ({"threads_batch": 12}, []),
        )
        payload = json.loads(pathlib.Path(resolution.cache_path).read_text())
        self.assertNotIn("threads", payload)
        self.assertEqual(payload["threads_batch"], 12)
        # And the next plain run must not inherit the 8.
        plain = self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        self.assertNotEqual(plain.profile.field_sources.get("threads"), "cached")

    def test_an_explicit_value_does_not_come_back_on_the_next_run(self) -> None:
        self.resolve(
            cli=_cli(batch=1024),
            calibrator=lambda **_: ({"threads": 11, "threads_batch": 11}, []),
        )
        plain = self.resolve()
        self.assertNotEqual(plain.profile.batch, 1024)
        self.assertEqual(plain.profile.field_sources["batch"], "heuristic")

    def test_a_heuristic_change_still_reaches_a_calibrated_machine(self) -> None:
        """Freezing the heuristic into the cache would make it unfixable."""
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        second = self.resolve()
        self.assertEqual(second.profile.field_sources["threads"], "cached")
        self.assertEqual(second.profile.field_sources["batch"], "heuristic")

    def test_a_partial_measurement_is_readable_back(self) -> None:
        """The regression the B1 fix introduced.

        Only calibrated fields are written, so pinning `--threads` and letting
        Orbit measure `threads_batch` stores exactly one key. An all-or-nothing
        read rejected that file forever: the write was a no-op and the full
        sweep ran again on every start, silently, for as long as the operator
        kept the flag.
        """
        first = self.resolve(
            cli=_cli(threads=8),
            calibrator=lambda **_: ({"threads_batch": 12}, []),
        )
        payload = json.loads(pathlib.Path(first.cache_path).read_text())
        self.assertEqual(set(payload) & set(("threads", "threads_batch")),
                         {"threads_batch"})
        calls: list[int] = []
        second = self.resolve(
            cli=_cli(threads=8),
            calibrator=lambda **_: (calls.append(1), ({"threads_batch": 12}, []))[1],
        )
        self.assertEqual(calls, [], "a stored measurement must not be re-measured")
        self.assertEqual(second.profile.threads_batch, 12)
        self.assertEqual(second.profile.field_sources["threads_batch"], "cached")

    def test_a_present_but_unusable_value_still_condemns_the_file(self) -> None:
        fingerprint = profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551")
        path = cache_path_for(fingerprint, self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        for junk in ("abc", -1, 0, None):
            path.write_text(json.dumps({
                "format_version": PROFILE_FORMAT_VERSION,
                "fingerprint": fingerprint, "threads": junk,
            }), encoding="utf-8")
            self.assertIsNone(load_cached_profile(fingerprint, self.env))

    def test_a_stale_format_version_evicts_the_file(self) -> None:
        fingerprint = profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551")
        path = cache_path_for(fingerprint, self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format_version": "orbit-server-profile-v0",
            "fingerprint": fingerprint, "threads": 6, "threads_batch": 6,
        }), encoding="utf-8")
        self.assertIsNone(
            load_cached_profile(fingerprint, self.env, evict_stale=False)
        )
        self.assertTrue(path.is_file(), "a preview must not delete anything")
        self.assertIsNone(load_cached_profile(fingerprint, self.env))
        self.assertFalse(path.is_file(), "a stale entry should not accumulate")

    def test_a_previous_version_entry_under_another_key_is_evicted(self) -> None:
        """The version is part of the key, so a bump orphans the old file.

        Evicting only the file at the current key can never fire after a real
        bump; the sweep is what removes the pre-bump entry -- and it runs even
        when nothing exists under the current key yet, which is the ordinary
        state right after an upgrade.
        """
        fingerprint = profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551")
        current = cache_path_for(fingerprint, self.env)
        current.parent.mkdir(parents=True, exist_ok=True)
        old = current.parent / ("0" * 64 + ".json")
        old.write_text(json.dumps({
            "format_version": "orbit-server-profile-v2",
            "fingerprint": "0" * 64, "threads": 8, "threads_batch": 8,
        }), encoding="utf-8")
        unrelated = current.parent / "notes.txt"
        unrelated.write_text("keep", encoding="utf-8")
        # A preview leaves everything alone.
        self.assertIsNone(load_cached_profile(fingerprint, self.env, evict_stale=False))
        self.assertTrue(old.is_file())
        # A calibrating load with no current entry sweeps the orphan.
        self.assertIsNone(load_cached_profile(fingerprint, self.env))
        self.assertFalse(old.is_file(), "the pre-bump entry must not outlive the bump")
        self.assertTrue(unrelated.is_file())

    def test_the_sweep_leaves_current_and_unreadable_entries_alone(self) -> None:
        from orbit.native_server.server_profile import evict_foreign_versions

        directory = cache_path_for("x" * 64, self.env).parent
        directory.mkdir(parents=True, exist_ok=True)
        current = directory / ("1" * 64 + ".json")
        current.write_text(json.dumps({
            "format_version": PROFILE_FORMAT_VERSION, "fingerprint": "1" * 64, "threads": 6,
        }), encoding="utf-8")
        broken = directory / ("2" * 64 + ".json")
        broken.write_text("{not json", encoding="utf-8")
        stale = directory / ("3" * 64 + ".json")
        stale.write_text(json.dumps({"format_version": "orbit-server-profile-v1"}), encoding="utf-8")
        self.assertEqual(evict_foreign_versions(self.env), 1)
        self.assertTrue(current.is_file())
        self.assertTrue(broken.is_file(), "not this Orbit's to judge")
        self.assertFalse(stale.is_file())

    def test_a_preview_resolution_does_not_evict_a_stale_entry(self) -> None:
        """Through `resolve_profile`, not just the helper.

        `--show-profile` resolves with `allow_calibration=False`; that is the
        path that must leave the filesystem alone, and testing the helper
        directly does not exercise it.
        """
        fingerprint = profile_fingerprint(
            DELL, model_sha256="model-a", backend_id="b9551")
        path = cache_path_for(fingerprint, self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format_version": "orbit-server-profile-v0",
            "fingerprint": fingerprint, "threads": 6, "threads_batch": 6,
        }), encoding="utf-8")
        self.resolve(allow_calibration=False)
        self.assertTrue(path.is_file(), "a preview must not evict")
        # A real start may.
        self.resolve(calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []))
        self.assertTrue(path.is_file())   # rewritten by the new measurement

    def test_a_calibrator_measuring_nothing_is_not_a_calibration(self) -> None:
        resolution = self.resolve(calibrator=lambda **_: ({}, [{"x": 1}]))
        self.assertFalse(resolution.calibrated)
        self.assertIsNotNone(resolution.calibration_error)
        self.assertNotIn("auto-calibrated", resolution.profile.source)

    def test_a_corrupt_cache_file_reads_as_absent(self) -> None:
        """T10. Corruption costs a recalibration, never a crash or a wrong value."""
        fingerprint = profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551")
        path = cache_path_for(fingerprint, self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        for junk in ("", "{", "null", "[]", '{"format_version": "other"}',
                     '{"format_version": "%s"}' % PROFILE_FORMAT_VERSION):
            path.write_text(junk, encoding="utf-8")
            self.assertIsNone(load_cached_profile(fingerprint, self.env))

    def test_a_cache_entry_for_another_fingerprint_is_refused(self) -> None:
        fingerprint = profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551")
        path = cache_path_for(fingerprint, self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format_version": PROFILE_FORMAT_VERSION, "fingerprint": "somethingelse",
            "threads": 6, "threads_batch": 6,
        }), encoding="utf-8")
        self.assertIsNone(load_cached_profile(fingerprint, self.env))

    def test_the_write_is_atomic_and_leaves_no_temp_files(self) -> None:
        fingerprint = "f" * 64
        path = store_cached_profile(
            fingerprint, {"threads": 6, "threads_batch": 6}, environ=self.env
        )
        self.assertIsNotNone(path)
        directory = pathlib.Path(path).parent
        self.assertEqual([p.name for p in directory.glob("*.tmp")], [])
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self.assertEqual(payload["fingerprint"], fingerprint)

    def test_an_unwritable_cache_directory_is_survivable(self) -> None:
        path = store_cached_profile(
            "a" * 64, {"threads": 6, "threads_batch": 6},
            environ={"XDG_CACHE_HOME": "/proc/nonexistent/cannot-create"},
        )
        self.assertIsNone(path)

    def test_a_cached_profile_is_clamped_on_load(self) -> None:
        fingerprint = profile_fingerprint(DELL, model_sha256="model-a", backend_id="b9551")
        path = cache_path_for(fingerprint, self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format_version": PROFILE_FORMAT_VERSION, "fingerprint": fingerprint,
            "threads": 9999, "threads_batch": 12,
        }), encoding="utf-8")
        loaded = load_cached_profile(fingerprint, self.env)
        self.assertLessEqual(loaded["threads"], MAX_THREADS)
        self.assertGreaterEqual(loaded["threads_batch"], 1)

    def test_discard_is_quiet_when_there_is_nothing_to_discard(self) -> None:
        self.assertFalse(discard_cached_profile("b" * 64, self.env))


class CacheRamTests(_Case):
    """T8. Arithmetic that fails toward a smaller cache."""

    def test_the_model_is_subtracted(self) -> None:
        without = compute_cache_ram_mib(DELL, model_bytes=0)
        with_model = compute_cache_ram_mib(DELL, model_bytes=20 * GIB * 1024 * 1024)
        self.assertLess(with_model, without)

    def test_it_never_exceeds_what_is_left_after_the_model(self) -> None:
        model_mib = 20 * GIB
        value = compute_cache_ram_mib(DELL, model_bytes=model_mib * 1024 * 1024)
        self.assertLess(value, DELL.total_ram_mib - model_mib)

    def test_a_model_larger_than_ram_yields_the_floor(self) -> None:
        """T9's sibling: the swap-inducing answer is arithmetically unreachable."""
        value = compute_cache_ram_mib(DELL, model_bytes=64 * GIB * 1024 * 1024)
        self.assertEqual(value, MIN_CACHE_RAM_MIB)

    def test_unknown_memory_yields_the_floor(self) -> None:
        blind = HostTopology(total_ram_mib=0)
        self.assertEqual(compute_cache_ram_mib(blind), MIN_CACHE_RAM_MIB)

    def test_a_bigger_context_never_yields_a_bigger_cache(self) -> None:
        small = compute_cache_ram_mib(DELL, ctx_tokens=8192)
        large = compute_cache_ram_mib(DELL, ctx_tokens=131072)
        self.assertLessEqual(large, small)


class HeuristicTests(_Case):
    """The floor under everything, on machines we cannot measure."""

    def test_threads_track_physical_cores_not_logical(self) -> None:
        self.assertEqual(heuristic_profile(NUC).threads, 6)

    def test_a_memory_poor_machine_is_capped_harder(self) -> None:
        profile = heuristic_profile(TINY)
        self.assertLessEqual(profile.threads, 2)
        self.assertLessEqual(profile.batch, 256)

    def test_the_qualified_pair_is_the_default_batch_shape(self) -> None:
        profile = heuristic_profile(DELL)
        self.assertEqual((profile.batch, profile.ubatch), (FALLBACK_BATCH, FALLBACK_UBATCH))

    def test_the_fallback_is_the_reference_numbers(self) -> None:
        profile = fallback_profile(DELL)
        self.assertEqual(
            (profile.threads, profile.batch, profile.ubatch),
            (FALLBACK_THREADS, FALLBACK_BATCH, FALLBACK_UBATCH),
        )

    def test_every_resolution_is_within_bounds(self) -> None:
        for topology in (DELL, NUC, TINY, HostTopology()):
            profile = clamp_profile(heuristic_profile(topology))
            self.assertGreaterEqual(profile.threads, 1)
            self.assertLessEqual(profile.threads, MAX_THREADS)
            self.assertLessEqual(profile.ubatch, profile.batch)


class CandidateTests(_Case):
    """The bounded set, built from topology rather than named."""

    def test_the_dell_candidates_are_bounded_and_sorted(self) -> None:
        candidates = thread_candidates(DELL)
        self.assertEqual(candidates, sorted(set(candidates)))
        self.assertLessEqual(len(candidates), 5)
        self.assertIn(DELL.physical_cores, candidates)

    def test_a_uniform_machine_measures_fewer_candidates(self) -> None:
        self.assertLessEqual(len(thread_candidates(TINY)), 3)

    def test_candidates_never_exceed_the_thread_ceiling(self) -> None:
        huge = HostTopology(physical_cores=256, logical_cpus=512, total_ram_mib=1024 * GIB)
        self.assertTrue(all(c <= MAX_THREADS for c in thread_candidates(huge)))

    def test_candidates_are_never_empty(self) -> None:
        self.assertTrue(thread_candidates(HostTopology()))


class CapabilityTests(_Case):
    """T7. Tuning is not consent to enable a capability."""

    def test_the_profile_carries_no_capability_switch(self) -> None:
        resolution = self.resolve(
            calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []),
        )
        keys = set(asdict(resolution.profile))
        self.assertEqual(
            keys,
            {"threads", "threads_batch", "batch", "ubatch", "cache_ram_mib",
             "source", "field_sources"},
        )
        for forbidden in ("mtp", "gpu", "gpu_layers", "ctx", "context_tokens",
                          "model", "quantization", "low_memory"):
            self.assertNotIn(forbidden, keys)

    def test_the_stored_payload_carries_no_capability_switch(self) -> None:
        resolution = self.resolve(
            calibrator=lambda **_: ({"threads": 6, "threads_batch": 6}, []),
        )
        payload = json.loads(pathlib.Path(resolution.cache_path).read_text())
        for forbidden in ("mtp", "gpu", "ctx", "model_path", "low_memory"):
            self.assertNotIn(forbidden, payload)


class TopologyTests(_Case):
    """Reading the machine, including when it will not be read."""

    def test_meminfo_and_cpuinfo_are_parsed(self) -> None:
        topology = detect_topology(
            meminfo_text="MemTotal:       32505856 kB\nMemAvailable:   11567104 kB\n"
                         "SwapTotal:       2097148 kB\n",
            cpuinfo_text="model name\t: Test CPU\nphysical id\t: 0\ncore id\t: 0\n"
                         "physical id\t: 0\ncore id\t: 1\n",
        )
        self.assertEqual(topology.cpu_model, "Test CPU")
        self.assertEqual(topology.physical_cores, 2)
        self.assertEqual(topology.total_ram_mib, 31744)
        self.assertEqual(topology.swap_total_mib, 2047)

    def test_an_unreadable_machine_degrades_to_zero_not_to_a_guess(self) -> None:
        topology = detect_topology(meminfo_text="", cpuinfo_text="")
        self.assertEqual(topology.total_ram_mib, 0)
        self.assertGreaterEqual(topology.physical_cores, 1)


class RenderTests(_Case):
    """What the operator sees at startup."""

    def test_every_field_is_shown_with_its_source(self) -> None:
        resolution = self.resolve(cli=_cli(threads=5))
        lines = render_profile_lines(resolution)
        joined = "\n".join(lines)
        for label in ("profile:", "threads:", "threads_batch:", "batch:",
                      "ubatch:", "cache_ram:"):
            self.assertIn(label, joined)
        self.assertIn("threads: 5 (cli)", joined)

    def test_a_calibration_failure_is_reported(self) -> None:
        def boom(**_):
            raise RuntimeError("nope")

        lines = render_profile_lines(self.resolve(calibrator=boom))
        self.assertTrue(any("calibration: skipped" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
