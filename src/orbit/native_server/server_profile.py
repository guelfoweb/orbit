"""Where `orbit server` gets its threads, batch and ubatch.

One caveat up front, because it would otherwise be discovered the hard way:
`cache_ram_mib` is RESOLVED AND REPORTED BUT NOT CONSUMED by the native server.
Orbit's native path has no prompt-cache-RAM knob to pass it to -- `--cache-ram`
is a `llama-server` option, and wiring a new inference parameter is not what
this change is. It is computed here so there is one conservative answer for it,
shown by `--show-profile`, stored with the measurement, and printed by
`scripts/suggest-server-profile.sh` for an operator running an external
llama-server. When the native path grows such a knob, this is the value to feed
it; until then, treat it as advice rather than as configuration.

Orbit shipped one set of numbers for every machine: `--threads 6 --batch 256
--ubatch 128`, measured on the reference NUC and correct there. On any other
box they are a guess, and `scripts/suggest-server-profile.sh` existed to tell an
operator what to export instead -- except `orbit server` read no environment at
all, so the exports it printed were inert. This module is the source of truth
those two never shared.

What it does NOT do is decide anything the operator has decided. Resolution is a
precedence chain, and an explicitly supplied value wins outright at every level:

    explicit CLI
      > explicit environment
        > explicit stored user profile   (reserved: the level exists and is
                                         honoured, but no loader or file
                                         format is wired yet)
          > cached auto-calibrated profile
            > bounded auto-calibration
              > conservative heuristic

Each level fills only the fields still unset, so `--threads 8` alone leaves
batch, ubatch and cache_ram to be resolved and never touches threads again. A
field an operator named is never measured, never cached over, and never reported
as auto.

Two deliberate non-goals. Nothing here enables MTP, a GPU, a different context
size, a different model or any experimental backend path: those are capability
decisions, not tuning, and a machine that happens to be fast is not consent.
And nothing here adapts during a session -- the profile is resolved once, before
the model loads, and then it is a fact about the process.

Measurement is the point of the exercise. Core count is not the answer on a
heterogeneous CPU: a package with performance and efficiency cores reports every
one of them, ggml waits on the slowest thread in the batch, and "use all 16"
can be slower than "use 8". So the thread candidates come from topology but the
winner comes from a bounded benchmark, and `calibrate_threads` lives beside this
policy rather than inside it so that policy stays pure and testable.
"""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass, asdict, replace, fields
from pathlib import Path
from typing import Mapping

# The reference numbers. They remain the fallback because they are the only
# values with a qualified corpus behind them, and a machine we cannot measure
# is better served by a profile that is known to work somewhere than by one
# derived from arithmetic alone.
FALLBACK_THREADS = 6
FALLBACK_BATCH = 256
FALLBACK_UBATCH = 128

# Environment names. Deliberately `ORBIT_`-prefixed: the suggest script used to
# print `export THREADS=...`, and reading a name that generic out of an ambient
# environment would let an unrelated shell variable retune an inference server.
ENV_THREADS = "ORBIT_THREADS"
ENV_THREADS_BATCH = "ORBIT_THREADS_BATCH"
ENV_BATCH = "ORBIT_BATCH"
ENV_UBATCH = "ORBIT_UBATCH"
ENV_CACHE_RAM = "ORBIT_CACHE_RAM"

# Bumped whenever the fingerprint inputs change, so entries written by an older
# Orbit are ignored rather than silently orphaned under a key nobody computes
# any more. v2 added low_memory, MTP and expert-usage to the fingerprint. v3
# changed the MEASUREMENT (warm-up pass and fewest-threads tie-break): a v2
# entry may hold a winner chosen from a cold first candidate, so it is
# re-measured once rather than trusted.
PROFILE_FORMAT_VERSION = "orbit-server-profile-v3"

# Only these are ever WRITTEN to the cache. A cache entry is a record of what
# was measured, never of what was resolved -- see `store_cached_profile`.
MEASURABLE_FIELDS = ("threads", "threads_batch")

# Bounds every resolved value is clamped into, whatever produced it. A cached
# profile from an older Orbit, a hand-edited JSON file and a benchmark that
# measured something absurd all pass through here.
MIN_THREADS = 1
MAX_THREADS = 32
MIN_BATCH = 32
MAX_BATCH = 2048
MIN_UBATCH = 16
MIN_CACHE_RAM_MIB = 512


@dataclass(frozen=True)
class HostTopology:
    """What the machine says about itself. Measured facts only, no policy."""

    cpu_model: str = ""
    physical_cores: int = 1
    logical_cpus: int = 1
    total_ram_mib: int = 0
    available_ram_mib: int = 0
    swap_total_mib: int = 0


@dataclass(frozen=True)
class ServerProfile:
    """One resolved startup profile. `source` says how it was decided."""

    threads: int
    threads_batch: int
    batch: int
    ubatch: int
    cache_ram_mib: int
    source: str = "heuristic"
    # Per-field provenance, so `--show-profile` can tell an operator which of
    # their values survived and which Orbit chose. Absent fields read as
    # whatever `source` says.
    field_sources: "dict[str, str]" = None  # type: ignore[assignment]

    def with_sources(self, sources: "dict[str, str]") -> "ServerProfile":
        return replace(self, field_sources=dict(sources))


def _read_int(text: str) -> int | None:
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value


def _meminfo_mib(key: str, meminfo: str) -> int:
    match = re.search(rf"^{re.escape(key)}:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    return int(match.group(1)) // 1024 if match else 0


def detect_topology(
    *, meminfo_text: str | None = None, cpuinfo_text: str | None = None
) -> HostTopology:
    """Read the machine. Every failure degrades to a smaller number, never a
    larger one, because every consumer of this treats bigger as more permission.
    """
    try:
        meminfo = (
            meminfo_text
            if meminfo_text is not None
            else Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        )
    except OSError:
        meminfo = ""
    try:
        cpuinfo = (
            cpuinfo_text
            if cpuinfo_text is not None
            else Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        )
    except OSError:
        cpuinfo = ""

    logical = os.cpu_count() or 1
    # Physical cores from the (physical id, core id) pairs Linux publishes. A
    # kernel that omits them -- and every non-Linux host -- falls back to the
    # logical count, which over-reports on SMT and is why the thread candidates
    # never simply take it.
    pairs = set()
    physical_id = ""
    for line in cpuinfo.splitlines():
        if line.startswith("physical id"):
            physical_id = line.split(":", 1)[-1].strip()
        elif line.startswith("core id"):
            pairs.add((physical_id, line.split(":", 1)[-1].strip()))
    physical = len(pairs) if pairs else logical

    model = ""
    match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if match:
        model = match.group(1).strip()
    if not model:
        model = platform.processor() or platform.machine() or ""

    return HostTopology(
        cpu_model=model,
        physical_cores=max(1, physical),
        logical_cpus=max(1, logical),
        total_ram_mib=_meminfo_mib("MemTotal", meminfo),
        available_ram_mib=_meminfo_mib("MemAvailable", meminfo),
        swap_total_mib=_meminfo_mib("SwapTotal", meminfo),
    )


def thread_candidates(topology: HostTopology) -> "list[int]":
    """The bounded set worth measuring, built from topology rather than named.

    Four shapes, because each is a real answer on some machine: half the cores
    (a hybrid package where only the performance cores are worth using), three
    quarters, all of them, and the logical count (SMT, where oversubscribing the
    physical cores sometimes wins). Duplicates collapse, so a small or uniform
    machine measures fewer candidates rather than the same one repeatedly.

    The reference 6 is included when it fits, so the qualified NUC profile is
    always among the things a NUC-like box actually compares.
    """
    physical = max(1, topology.physical_cores)
    logical = max(physical, topology.logical_cpus)
    raw = {
        max(1, physical // 2),
        max(1, (physical * 3) // 4),
        physical,
        logical,
    }
    if physical >= FALLBACK_THREADS:
        raw.add(FALLBACK_THREADS)
    bounded = sorted(
        {min(MAX_THREADS, max(MIN_THREADS, value)) for value in raw if value > 0}
    )
    return bounded


def compute_cache_ram_mib(
    topology: HostTopology,
    *,
    model_bytes: int = 0,
    ctx_tokens: int = 8192,
) -> int:
    """A conservative prompt-cache budget, computed and never benchmarked.

    Benchmarking this would find the largest cache that has not yet swapped,
    which is exactly the value that swaps later under a workload the benchmark
    did not run. So it is arithmetic, and every term is subtracted before the
    remainder is shared:

      total RAM
        - the model, which is resident for the life of the process
        - an OS reserve, so the machine stays usable
        - a KV/context reserve for the configured window
        - an Orbit checkpoint reserve (route-prefix anchors are ~78 MiB each
          and a server can hold more than one lineage)
        - a safety margin on what is left

    and only half of what survives that is offered as cache, because the other
    half is the headroom that keeps a long session off swap. Every failure path
    -- unreadable meminfo, an unknown model size, arithmetic that goes negative
    -- returns the floor. This fails toward a smaller cache by construction.
    """
    total = topology.total_ram_mib
    if total <= 0:
        return MIN_CACHE_RAM_MIB

    model_mib = max(0, int(model_bytes // (1024 * 1024)))
    # 2 GiB or a sixteenth of RAM, whichever is larger: a 128 GiB server should
    # not hand the OS the same absolute reserve as a 16 GiB laptop.
    os_reserve = max(2048, total // 16)
    # The KV window, at a deliberately generous 128 KiB per token. An
    # underestimate here is paid in swap; an overestimate only in a smaller
    # cache, which is the direction this is allowed to be wrong in. The earlier
    # form divided bytes-per-token by a MiB and so contributed a flat 512 MiB
    # from ctx 2048 to ctx 1M -- arithmetic that looked like it scaled and did
    # not.
    kv_reserve = 512 + (max(0, ctx_tokens) * 128) // 1024
    checkpoint_reserve = 512
    margin = max(1024, total // 20)

    remaining = total - model_mib - os_reserve - kv_reserve - checkpoint_reserve - margin
    if remaining <= 0:
        return MIN_CACHE_RAM_MIB
    return max(MIN_CACHE_RAM_MIB, remaining // 2)


def heuristic_profile(
    topology: HostTopology,
    *,
    model_bytes: int = 0,
    ctx_tokens: int = 8192,
) -> ServerProfile:
    """The no-measurement answer, and the floor under every other path.

    Threads track physical cores rather than logical ones, capped, and a
    memory-poor machine is capped harder still: threads cost working set, and a
    box that cannot hold the model comfortably is not helped by more of them.
    Batch and ubatch stay at the qualified pair unless the machine is clearly
    smaller than the reference, because nothing has measured an alternative.
    """
    physical = max(1, topology.physical_cores)
    logical = max(physical, topology.logical_cpus)
    total_gib = topology.total_ram_mib // 1024

    if physical <= 2:
        threads = min(2, logical)
    elif total_gib and total_gib < 16:
        threads = min(4, physical)
    else:
        threads = min(physical, MAX_THREADS)
    threads = max(MIN_THREADS, threads)

    if physical <= 2 or (total_gib and total_gib < 16):
        batch, ubatch = 128, 64
    else:
        batch, ubatch = FALLBACK_BATCH, FALLBACK_UBATCH

    return ServerProfile(
        threads=threads,
        threads_batch=threads,
        batch=batch,
        ubatch=ubatch,
        cache_ram_mib=compute_cache_ram_mib(
            topology, model_bytes=model_bytes, ctx_tokens=ctx_tokens
        ),
        source="heuristic",
    )


def fallback_profile(topology: HostTopology | None = None) -> ServerProfile:
    """The last resort: the qualified reference numbers, unconditionally.

    Used when calibration failed AND the heuristic cannot be trusted either --
    an unreadable machine. It is deliberately not derived from anything.
    """
    return ServerProfile(
        threads=FALLBACK_THREADS,
        threads_batch=FALLBACK_THREADS,
        batch=FALLBACK_BATCH,
        ubatch=FALLBACK_UBATCH,
        cache_ram_mib=MIN_CACHE_RAM_MIB
        if topology is None
        else compute_cache_ram_mib(topology),
        source="fallback",
    )


def clamp_profile(profile: ServerProfile) -> ServerProfile:
    """Bring any profile inside the bounds, whatever produced it.

    A cached file, a hand-edited JSON and a benchmark result all pass through
    here, so a corrupt or malicious value cannot reach the backend. ubatch is
    additionally never larger than batch, which llama.cpp requires and which a
    hand-edited file is the likeliest way to violate.
    """
    threads = min(MAX_THREADS, max(MIN_THREADS, int(profile.threads)))
    threads_batch = min(MAX_THREADS, max(MIN_THREADS, int(profile.threads_batch)))
    batch = min(MAX_BATCH, max(MIN_BATCH, int(profile.batch)))
    ubatch = min(batch, max(MIN_UBATCH, int(profile.ubatch)))
    cache_ram = max(MIN_CACHE_RAM_MIB, int(profile.cache_ram_mib))
    return replace(
        profile,
        threads=threads,
        threads_batch=threads_batch,
        batch=batch,
        ubatch=ubatch,
        cache_ram_mib=cache_ram,
    )


# -- fingerprint and cache -------------------------------------------------

def profile_fingerprint(
    topology: HostTopology,
    *,
    model_sha256: str = "",
    model_arch: str = "",
    model_quant: str = "",
    backend_id: str = "",
    ctx_tokens: int = 8192,
    low_memory: bool = False,
    mtp_enabled: bool = False,
    expert_usage_enabled: bool = False,
) -> str:
    """What a stored measurement is a measurement OF.

    Every term is something that, if it changed, would make the stored numbers
    a measurement of a different situation: the CPU and its core counts, the RAM
    class, the exact model bytes, its architecture and quantization, the backend
    build, and the context size the measurement ran at. RAM is bucketed to GiB
    rather than taken exactly, because `MemTotal` moves slightly across kernels
    and a profile should not be discarded for that.
    """
    parts = [
        PROFILE_FORMAT_VERSION,
        topology.cpu_model,
        f"cores={topology.physical_cores}",
        f"cpus={topology.logical_cpus}",
        f"ram_gib={topology.total_ram_mib // 1024}",
        f"model={model_sha256}",
        f"arch={model_arch}",
        f"quant={model_quant}",
        f"backend={backend_id}",
        f"ctx={ctx_tokens}",
        # `--low-memory` moves peak RSS by more than 10 GiB on a verified
        # profile. A measurement taken in one mode says nothing about the
        # other -- both the thread winner and the cache arithmetic change --
        # so the two modes must not share a cache entry.
        f"low_memory={int(bool(low_memory))}",
        # Both change what the benchmark would have measured. MTP alters KV
        # allocation at context creation; expert-usage counters add per-token
        # instrumentation inside the CPU backend -- that is, inside the thing
        # being timed. A measurement taken without them does not describe a
        # process running with them.
        f"mtp={int(bool(mtp_enabled))}",
        f"experts={int(bool(expert_usage_enabled))}",
    ]
    import hashlib

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def profile_cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """`~/.cache/orbit/server-profiles`, or `$XDG_CACHE_HOME`'s equivalent.

    Outside the repository on purpose: a measurement describes this machine, and
    committing one would hand another machine a profile it never measured.
    """
    env = os.environ if environ is None else environ
    root = env.get("XDG_CACHE_HOME") or ""
    base = Path(root) if root else Path(env.get("HOME", "~")).expanduser() / ".cache"
    return base / "orbit" / "server-profiles"


def cache_path_for(fingerprint: str, environ: Mapping[str, str] | None = None) -> Path:
    return profile_cache_dir(environ) / f"{fingerprint}.json"


def load_cached_profile(
    fingerprint: str,
    environ: Mapping[str, str] | None = None,
    *,
    evict_stale: bool = True,
) -> "dict[str, int] | None":
    """The stored MEASUREMENT for this machine+model, or None.

    Returns only the measured fields, because only those were ever written --
    see `store_cached_profile`. The caller contributes them like any other
    precedence level, so a stored measurement can fill `threads` while `batch`
    continues to come from the heuristic and can still change when that
    heuristic does.

    Anything unreadable, malformed, truncated or of a different format version
    reads as absent. A corrupt cache must cost a recalibration, never a crash
    and never a wrong profile.
    """
    path = cache_path_for(fingerprint, environ)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # The ordinary post-bump state: nothing under the current key yet,
        # while the previous version's entry sits beside it. Sweep here too,
        # or the orphan outlives the recalibration that replaces it.
        if evict_stale:
            evict_foreign_versions(environ)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("format_version") != PROFILE_FORMAT_VERSION:
        # Self-cleaning, and not only for this file: the version is the first
        # term of the fingerprint, so a bump changes the KEY and an older entry
        # lives under a name no current Orbit ever computes. Evicting only the
        # file just opened would therefore never fire on a real bump and every
        # calibrated machine would keep an orphan per version forever. The
        # sweep below is what actually keeps the directory bounded by live
        # combinations; it is bounded itself by the handful of entries a
        # machine accumulates.
        #
        # Not on the preview path, though. `--show-profile` reads this too, and
        # a command that only reports must not change the filesystem -- the
        # same property `--show-profile --recalibrate` was fixed to respect.
        if evict_stale:
            try:
                path.unlink()
            except OSError:
                pass
            evict_foreign_versions(environ)
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    # Absent is not the same as corrupt. Because only CALIBRATED fields are
    # written, a run that measured `threads_batch` while the operator pinned
    # `threads` stores exactly one key -- and an all-or-nothing read rejected
    # that file forever, so the sweep ran again on every single start while the
    # write it produced was a no-op. A missing field is skipped; only a field
    # that is present and unusable condemns the file.
    measured: dict[str, int] = {}
    for field in MEASURABLE_FIELDS:
        if field not in payload:
            continue
        try:
            value = int(payload[field])
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        # Clamped on the way in: a hand-edited or older-Orbit file must not be
        # able to hand the backend a thread count it would refuse.
        measured[field] = min(MAX_THREADS, max(MIN_THREADS, value))
    return measured or None


def evict_foreign_versions(environ: Mapping[str, str] | None = None) -> int:
    """Unlink every cache entry written under another format version.

    Called from the calibrating (non-preview) load path only. Unreadable or
    malformed files are left alone -- they are not this Orbit's to judge -- and
    every failure is swallowed, because housekeeping must never cost a start.
    Returns how many were removed, for tests and logs.
    """
    removed = 0
    try:
        entries = list(profile_cache_dir(environ).glob("*.json"))
    except OSError:
        return 0
    for entry in entries:
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        version = payload.get("format_version")
        if isinstance(version, str) and version != PROFILE_FORMAT_VERSION:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def store_cached_profile(
    fingerprint: str,
    measured: "Mapping[str, int]",
    *,
    environ: Mapping[str, str] | None = None,
    measurements: object = None,
) -> Path | None:
    """Write the MEASURED fields atomically, or give up quietly.

    `measured` is what calibration produced -- never the resolved profile. That
    distinction is the whole point: storing the resolution would persist the
    operator's one-off `--batch 1024`, or whatever the heuristic happened to
    say that day, into `~/.cache` and replay it on every later start as though
    a benchmark had chosen it. An experiment typed once must not become a
    permanent setting, and a heuristic improvement must still reach a machine
    that has calibrated before.

    Temp file in the destination directory then `os.replace`, so a reader never
    observes a half-written profile and a crash mid-write leaves the previous
    one intact. A cache that cannot be written is not an error worth failing a
    server start over -- the profile is still resolved, it is simply measured
    again next time.
    """
    directory = profile_cache_dir(environ)
    payload = {
        "format_version": PROFILE_FORMAT_VERSION,
        "fingerprint": fingerprint,
    }
    for field in MEASURABLE_FIELDS:
        if field in measured:
            payload[field] = int(measured[field])
    if measurements is not None:
        payload["measurements"] = measurements
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(directory), delete=False, suffix=".tmp"
        )
        try:
            with handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, str(cache_path_for(fingerprint, environ)))
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    except (OSError, ValueError):
        return None
    return cache_path_for(fingerprint, environ)


def discard_cached_profile(
    fingerprint: str, environ: Mapping[str, str] | None = None
) -> bool:
    try:
        cache_path_for(fingerprint, environ).unlink()
        return True
    except OSError:
        return False


# -- precedence ------------------------------------------------------------

_FIELDS = ("threads", "threads_batch", "batch", "ubatch", "cache_ram_mib")

_ENV_BY_FIELD = {
    "threads": ENV_THREADS,
    "threads_batch": ENV_THREADS_BATCH,
    "batch": ENV_BATCH,
    "ubatch": ENV_UBATCH,
    "cache_ram_mib": ENV_CACHE_RAM,
}


def env_overrides(environ: Mapping[str, str] | None = None) -> "dict[str, int]":
    """Explicit environment values, ignoring anything unparseable.

    A malformed `ORBIT_THREADS=fast` is dropped rather than defaulted: the
    operator said something, it did not parse, and inventing a number for them
    would be worse than resolving the field normally. The startup line reports
    what was actually used, so a dropped value is visible.
    """
    env = os.environ if environ is None else environ
    found: dict[str, int] = {}
    for field, name in _ENV_BY_FIELD.items():
        raw = env.get(name)
        if raw is None:
            continue
        value = _read_int(raw)
        if value is not None and value > 0:
            found[field] = value
    return found


@dataclass(frozen=True)
class Resolution:
    """The resolved profile plus everything `--show-profile` needs to explain it."""

    profile: ServerProfile
    fingerprint: str
    cache_path: Path | None
    calibrated: bool = False
    calibration_error: str | None = None
    measurements: object = None


def resolve_profile(
    *,
    cli: "Mapping[str, int | None]",
    topology: HostTopology,
    environ: Mapping[str, str] | None = None,
    user_profile: "Mapping[str, int] | None" = None,
    model_bytes: int = 0,
    model_sha256: str = "",
    model_arch: str = "",
    model_quant: str = "",
    backend_id: str = "",
    ctx_tokens: int = 8192,
    low_memory: bool = False,
    mtp_enabled: bool = False,
    expert_usage_enabled: bool = False,
    calibrator: object = None,
    recalibrate: bool = False,
    allow_calibration: bool = True,
) -> Resolution:
    """Walk the precedence chain, filling only what is still unset.

    The chain is the contract, and the loop below is deliberately the whole of
    it: each level contributes its fields to `chosen` only where `chosen` has
    none, so an explicit value can never be revisited by a later level. A field
    named on the CLI is therefore never measured -- which matters beyond
    correctness, because measuring it would spend the calibration budget
    answering a question the operator already answered.
    """
    chosen: dict[str, int] = {}
    sources: dict[str, str] = {}

    def contribute(values: "Mapping[str, int | None]", source: str) -> None:
        for field in _FIELDS:
            if field in chosen:
                continue
            value = values.get(field)
            if value is None:
                continue
            # One validation policy for every level. `env_overrides` already
            # drops non-positive values; applying the same rule here means a
            # `--threads 0` is passed over rather than silently clamped to 1
            # and then reported as `(cli)`, which would tell the operator they
            # got a value they did not ask for and could not have wanted.
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed <= 0:
                continue
            chosen[field] = parsed
            sources[field] = source

    contribute(cli, "cli")
    contribute(env_overrides(environ), "env")
    if user_profile:
        contribute(user_profile, "user-profile")

    fingerprint = profile_fingerprint(
        topology,
        model_sha256=model_sha256,
        model_arch=model_arch,
        model_quant=model_quant,
        backend_id=backend_id,
        ctx_tokens=ctx_tokens,
        low_memory=low_memory,
        mtp_enabled=mtp_enabled,
        expert_usage_enabled=expert_usage_enabled,
    )

    # Only when a measurement can actually replace what is discarded.
    # `--show-profile --recalibrate` previews; a preview that deleted the
    # operator's stored measurement and then reported "no cached measurement
    # for this machine yet" would be destroying state and misreporting it in
    # one step.
    if recalibrate and allow_calibration:
        discard_cached_profile(fingerprint, environ)

    calibrated = False
    calibration_error: str | None = None
    measurements: object = None
    cache_path: Path | None = None

    missing = [field for field in _FIELDS if field not in chosen]
    if missing and not recalibrate:
        cached = load_cached_profile(
            fingerprint, environ, evict_stale=allow_calibration
        )
        if cached:
            contribute(cached, "cached")
            cache_path = cache_path_for(fingerprint, environ)

    # Only ever measured for fields nobody has supplied AND that a benchmark
    # can actually answer. batch and ubatch are context-creation parameters --
    # they are fixed before the context that would be measured exists -- and
    # cache_ram is arithmetic by policy, not a benchmark. So the only thing
    # that can put a calibrator on the critical path is an unresolved thread
    # count, which is also what keeps a cached run from paying for a sweep that
    # would decline every field it was offered.
    missing = [field for field in _FIELDS if field not in chosen]
    measurable = [f for f in missing if f in MEASURABLE_FIELDS]
    if measurable and allow_calibration and calibrator is not None:
        try:
            result = calibrator(topology=topology, fields=tuple(measurable))
        except Exception as exc:  # calibration is best-effort, always
            calibration_error = f"{type(exc).__name__}: {exc}"
            result = None
        if result is None:
            if calibration_error is None:
                calibration_error = "calibration produced no usable result"
        elif not result[0]:
            # A calibrator that measured nothing has not calibrated. Treating
            # it as success would set `calibrated` with no field to show for
            # it, write a contentless cache entry, and print a header that
            # disagrees with every per-field source.
            calibration_error = "calibration returned no measured fields"
        else:
            values, measurements = result
            contribute(values, "calibrated")
            calibrated = True

    heuristic = heuristic_profile(
        topology, model_bytes=model_bytes, ctx_tokens=ctx_tokens
    )
    contribute(asdict(heuristic), "heuristic")
    # Belt and braces. `heuristic_profile` always supplies all five fields, so
    # nothing should reach this -- it exists so that a future heuristic which
    # declines to answer degrades to the qualified reference numbers instead of
    # raising a KeyError below. `"fallback"` appearing in `field_sources` would
    # mean that happened.
    contribute(asdict(fallback_profile(topology)), "fallback")

    # The header names what the MEASUREMENT covered, not the whole profile:
    # calibration resolves threads, while batch, ubatch and cache_ram stay
    # heuristic, and a bare "auto-calibrated" would claim all five were
    # measured. Per-field sources carry the exact story either way.
    measured_fields = [f for f in _FIELDS if sources.get(f) == "calibrated"]
    cached_fields = [f for f in _FIELDS if sources.get(f) == "cached"]
    if measured_fields:
        source = f"auto-calibrated ({', '.join(measured_fields)})"
    elif cached_fields:
        source = f"cached auto-calibrated ({', '.join(cached_fields)})"
    elif all(sources.get(field) in ("cli", "env", "user-profile") for field in _FIELDS):
        source = "explicit"
    else:
        source = "heuristic"

    profile = clamp_profile(
        ServerProfile(
            threads=chosen["threads"],
            threads_batch=chosen["threads_batch"],
            batch=chosen["batch"],
            ubatch=chosen["ubatch"],
            cache_ram_mib=chosen["cache_ram_mib"],
            source=source,
        )
    ).with_sources(sources)

    if calibrated:
        cache_path = store_cached_profile(
            fingerprint,
            {f: chosen[f] for f in MEASURABLE_FIELDS if sources.get(f) == "calibrated"},
            environ=environ,
            measurements=measurements,
        )

    return Resolution(
        profile=profile,
        fingerprint=fingerprint,
        cache_path=cache_path,
        calibrated=calibrated,
        calibration_error=calibration_error,
        measurements=measurements,
    )


def render_profile_lines(resolution: Resolution) -> "list[str]":
    """The startup block. One line per field, each saying where it came from."""
    profile = resolution.profile
    sources = profile.field_sources or {}
    lines = [f"profile: {profile.source}"]
    for field, label in (
        ("threads", "threads"),
        ("threads_batch", "threads_batch"),
        ("batch", "batch"),
        ("ubatch", "ubatch"),
    ):
        lines.append(f"{label}: {getattr(profile, field)} ({sources.get(field, '?')})")
    # Labelled advisory at the point of display, because it is the one line
    # here that does not configure the running server.
    lines.append(
        f"cache_ram: {profile.cache_ram_mib} MiB "
        f"({sources.get('cache_ram_mib', '?')}; advisory, "
        f"not consumed by the native server)"
    )
    if resolution.cache_path is not None:
        lines.append(f"profile cache: {resolution.cache_path}")
    if resolution.calibration_error:
        lines.append(f"calibration: skipped ({resolution.calibration_error})")
    return lines


def main(argv: "list[str] | None" = None) -> int:
    """`python3 -m orbit.native_server.server_profile` -- the heuristic, printed.

    Exists so `scripts/suggest-server-profile.sh` can be a presentation wrapper
    over this module rather than a second implementation of the same policy.
    The script used to re-derive thread and cache numbers in shell, and its
    output drifted from what `orbit server` actually did -- it printed
    `export THREADS=...` for a server that read no environment at all. There is
    now one source of truth, and the script prints what it says.

    This is the HEURISTIC only. It never measures and never reads the profile
    cache, because a suggestion an operator is meant to review should not depend
    on a benchmark they did not watch run. `orbit server --show-profile` is the
    command that reports the real resolved profile.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="orbit-server-profile",
        description="Print the conservative heuristic server profile for this host.",
    )
    parser.add_argument("--shell", action="store_true",
                        help="Print an env block suitable for `eval`.")
    parser.add_argument("--ctx", type=int, default=8192)
    args = parser.parse_args(argv)

    topology = detect_topology()
    profile = heuristic_profile(topology, ctx_tokens=args.ctx)
    if args.shell:
        print(f"# CPU: {topology.cpu_model}")
        print(f"# physical_cores: {topology.physical_cores}")
        print(f"# logical_cpus: {topology.logical_cpus}")
        print(f"# memory_gib: {topology.total_ram_mib // 1024}")
        print("# profile: conservative heuristic (no measurement)")
        print("# ORBIT_CACHE_RAM is advisory: Orbit's native server has no")
        print("# cache-RAM knob. It is here for an external llama-server.")
        print("# `orbit server` already resolves this for you and can measure")
        print("# a better one; these exports are for overriding it explicitly.")
        print(f"export {ENV_THREADS}={profile.threads}")
        print(f"export {ENV_THREADS_BATCH}={profile.threads_batch}")
        print(f"export {ENV_BATCH}={profile.batch}")
        print(f"export {ENV_UBATCH}={profile.ubatch}")
        print(f"export {ENV_CACHE_RAM}={profile.cache_ram_mib}")
        return 0
    print(f"cpu: {topology.cpu_model}")
    print(f"physical_cores: {topology.physical_cores}")
    print(f"logical_cpus: {topology.logical_cpus}")
    print(f"memory_gib: {topology.total_ram_mib // 1024}")
    print(f"threads: {profile.threads}")
    print(f"threads_batch: {profile.threads_batch}")
    print(f"batch: {profile.batch}")
    print(f"ubatch: {profile.ubatch}")
    print(f"cache_ram_mib: {profile.cache_ram_mib}  "
          f"# advisory; not consumed by the native server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENV_BATCH",
    "ENV_CACHE_RAM",
    "ENV_THREADS",
    "ENV_THREADS_BATCH",
    "ENV_UBATCH",
    "FALLBACK_BATCH",
    "FALLBACK_THREADS",
    "FALLBACK_UBATCH",
    "HostTopology",
    "PROFILE_FORMAT_VERSION",
    "Resolution",
    "ServerProfile",
    "cache_path_for",
    "clamp_profile",
    "compute_cache_ram_mib",
    "detect_topology",
    "discard_cached_profile",
    "env_overrides",
    "evict_foreign_versions",
    "fallback_profile",
    "heuristic_profile",
    "load_cached_profile",
    "main",
    "profile_cache_dir",
    "profile_fingerprint",
    "render_profile_lines",
    "resolve_profile",
    "store_cached_profile",
    "thread_candidates",
]
