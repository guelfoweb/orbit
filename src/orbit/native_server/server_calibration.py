"""Measuring the thread counts, rather than guessing them from core count.

Core count is not the answer on a heterogeneous CPU. A modern laptop package
reports performance and efficiency cores as one number, ggml splits a batch
evenly and then waits on the slowest thread in it, so "use all of them" can be
slower than using the fast half. That is not a thing arithmetic can decide, so
this runs a bounded benchmark and lets the machine answer.

Three constraints shape the whole of it.

**It must measure inference, not cache.** The route-prefix prewarm restores 768
tokens and costs ~24 s at startup; a calibration that ran through the ordinary
request path would be timing a `memcpy` on one candidate and a prefill on the
next. So this decodes raw synthetic tokens through `llama_decode` directly,
clears the KV between every candidate, and reports evaluated tokens beside wall
time so a reader can see that each candidate really did the same work.

**It must load the model once.** Thread counts are context parameters, but
`llama_set_n_threads` retunes a live context, so every candidate runs against
the same resident weights. Reloading 20 GiB per candidate would make the tuning
cost more than it could ever save.

**It must not be able to make the machine worse.** A candidate that grows swap
is rejected outright rather than scored, however fast it looked -- swap during a
benchmark means swap during a session, and a number measured while thrashing is
not a number. Everything is bounded: a fixed token count, a fixed candidate
list, one pass each, and a wall-clock ceiling that abandons the whole exercise
rather than delaying a server start indefinitely.
"""

from __future__ import annotations

import math
import time
from ctypes import POINTER, byref, cast, sizeof
from dataclasses import dataclass, asdict
from pathlib import Path

from .server_profile import HostTopology, thread_candidates

# The synthetic prompt's size, and the decode run's. 512 prefill tokens is
# enough for the rate to settle past batch-scheduling noise and short enough
# that eight candidates stay inside the ceiling; 24 decode tokens is the same
# trade for the memory-bound phase, which is far slower per token.
CALIBRATION_PREFILL_TOKENS = 512
CALIBRATION_DECODE_TOKENS = 24

# The reference turn this optimises for, and the single most consequential
# constant here. Scoring by modelled turn latency rather than by a raw rate is
# deliberate: prefill and decode trade AGAINST each other across thread counts
# -- measured on a 16-core hybrid laptop, prefill rose 37 -> 68 tok/s from 6 to
# 12 threads while decode fell 19 -> 8 -- so "fastest prefill" and "fastest
# decode" name different profiles and only a turn shape can choose between them.
#
# These are EVALUATED tokens, not prompt tokens, and the difference decides the
# answer. A tools-on CHAT turn on the qualified profile measures 978 prompt of
# which 768 are restored from the route-prefix anchor: 210 evaluated, 59
# generated. Scoring against the 978 would optimise for a prefill the server
# does not actually perform and pick a thread count whose decode is half as
# fast. An analysis step evaluates more, but it also generates more, and decode
# degrades faster than prefill improves -- so the same winner holds there.
REFERENCE_PREFILL_TOKENS = 210
REFERENCE_DECODE_TOKENS = 60

# Whole-run ceiling. Calibration is a convenience; it must never be the reason a
# server takes minutes to start. Exceeded means abandoned, not extended.
CALIBRATION_BUDGET_SECONDS = 90.0

# Swap growth above this is treated as thrashing rather than noise.
SWAP_GROWTH_TOLERANCE_MIB = 16

# How much better a candidate with MORE threads must score to displace one
# with fewer. Measured on the Dell (16-core hybrid): with the weights warm, 6
# and 8 threads scored within 1-8% of each other across three orderings, yet
# a real cache-restored CHAT turn took 10.6 s at 6 threads and 20.6 s at 8 --
# the synthetic shape (512-token prefill, 24-token decode, no cache restore,
# no other load) flatters larger counts relative to a served turn, where
# extra threads land on the slowest cores and every step waits for them.
# Repeat measurements of one candidate varied by ~5%. So within this margin
# the benchmark cannot tell the candidates apart, and the fewer threads are
# the safer answer: less contention, less power, and the count with a
# qualified corpus behind it. A larger count still wins when it earns it by
# more than the noise.
TIE_MARGIN = 0.10

# Why a warm-up runs first. The model is mmap'd; loading it does not touch
# every page. The FIRST candidate measured after load pays that page-in --
# on a fresh boot from NVMe it measured 17,957 major faults and 25 tok/s of
# prefill where the same thread count, warm, measured 41 tok/s -- and so the
# candidate that happens to run first loses to whichever runs second. That
# is how a fresh-boot calibration cached 8 threads on a machine whose warm
# winner is 6. One untimed pass over the same tokens pays the cost before
# anything is scored; it is recorded in the table, marked, and can never win.
WARMUP_REJECTION = "warmup"


@dataclass(frozen=True)
class CandidateMeasurement:
    threads: int
    prefill_tokens: int
    decode_tokens: int
    prefill_seconds: float
    decode_seconds: float
    prefill_tps: float
    decode_tps: float
    rss_mib: int
    swap_used_mib: int
    swap_growth_mib: int
    score: float
    rejected: str | None = None


def _swap_used_mib() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    total = free = 0
    for line in text.splitlines():
        if line.startswith("SwapTotal:"):
            total = int(line.split()[1]) // 1024
        elif line.startswith("SwapFree:"):
            free = int(line.split()[1]) // 1024
    return max(0, total - free)


def _rss_mib() -> int:
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) // 1024
    return 0


def _turn_score(prefill_tps: float, decode_tps: float) -> float:
    """Inverse modelled latency of the reference turn. Higher is better.

    Zero when either rate is zero, so a candidate that failed to produce a
    measurement can never win by division.
    """
    if prefill_tps <= 0 or decode_tps <= 0:
        return 0.0
    seconds = (
        REFERENCE_PREFILL_TOKENS / prefill_tps + REFERENCE_DECODE_TOKENS / decode_tps
    )
    return 0.0 if seconds <= 0 else 1.0 / seconds


def measure_candidate(client, threads: int, *, tokens: "list[int]") -> CandidateMeasurement:
    """One candidate: clear, retune, prefill, decode, measure.

    The KV is cleared before AND after. Before, so the prefill is real work
    rather than a restore; after, so nothing this did is visible to the first
    real request.
    """
    lib = client.lib.lib
    ctx = client._session.ctx_tgt
    if not ctx:
        raise RuntimeError("native client not loaded")

    from ..native_llama.bindings import llama_token

    swap_before = _swap_used_mib()
    mem = lib.llama_get_memory(ctx)
    if mem:
        lib.llama_memory_clear(mem, True)
    lib.llama_set_n_threads(ctx, int(threads), int(threads))

    array = (llama_token * len(tokens))(*tokens)

    # Everything from here is wrapped, so a candidate that raises still clears
    # what it put in the KV. The client would in practice clear it anyway --
    # `load()` invalidates the committed sequence, so the first real prompt
    # takes the full-clear path -- but that is an invariant in another module,
    # and this function reaches around the client to drive the context
    # directly. Local cleanup keeps the guarantee local.
    try:
        # Prefill, in ubatch-sized steps, exactly as the request path would.
        step = max(1, int(client.config.ubatch_size or 128))
        started = time.perf_counter()
        processed = 0
        while processed < len(tokens):
            n = min(step, len(tokens) - processed)
            ptr = cast(byref(array, processed * sizeof(llama_token)), POINTER(llama_token))
            if lib.llama_decode(ctx, lib.llama_batch_get_one(ptr, n)) != 0:
                raise RuntimeError(f"prefill decode failed at {processed}")
            processed += n
        lib.llama_synchronize(ctx)
        prefill_seconds = max(1e-9, time.perf_counter() - started)

        # Decode, one token at a time, which is the shape real generation has.
        # The token value is irrelevant to the timing -- no sampling happens
        # here, so reusing the last prompt token keeps the run deterministic.
        single = (llama_token * 1)(tokens[-1])
        started = time.perf_counter()
        for _ in range(CALIBRATION_DECODE_TOKENS):
            ptr = cast(single, POINTER(llama_token))
            if lib.llama_decode(ctx, lib.llama_batch_get_one(ptr, 1)) != 0:
                raise RuntimeError("decode failed")
        lib.llama_synchronize(ctx)
        decode_seconds = max(1e-9, time.perf_counter() - started)
    finally:
        if mem:
            lib.llama_memory_clear(mem, True)

    swap_after = _swap_used_mib()
    growth = max(0, swap_after - swap_before)
    prefill_tps = len(tokens) / prefill_seconds
    decode_tps = CALIBRATION_DECODE_TOKENS / decode_seconds
    rejected = (
        f"swap_growth_{growth}_mib"
        if growth > SWAP_GROWTH_TOLERANCE_MIB
        else None
    )
    return CandidateMeasurement(
        threads=int(threads),
        prefill_tokens=len(tokens),
        decode_tokens=CALIBRATION_DECODE_TOKENS,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        rss_mib=_rss_mib(),
        swap_used_mib=swap_after,
        swap_growth_mib=growth,
        # A rejected candidate scores zero rather than carrying a real score
        # that a later change might accidentally start comparing.
        score=0.0 if rejected else _turn_score(prefill_tps, decode_tps),
        rejected=rejected,
    )


def calibration_tokens(client, count: int = CALIBRATION_PREFILL_TOKENS) -> "list[int]":
    """A deterministic token run of the requested length.

    Built by tokenizing fixed ASCII and then cycling it, so the content is the
    same on every machine and every candidate, and no chat template, tool schema
    or route prompt is involved -- this must not accidentally measure the same
    tokens the prefix anchor holds.
    """
    seed = client.tokenize("orbit calibration sequence 0123456789 ") or [1]
    tokens: list[int] = []
    while len(tokens) < count:
        tokens.extend(seed)
    return tokens[:count]


def calibrate_threads(
    client,
    *,
    topology: HostTopology,
    fields: "tuple[str, ...]" = ("threads", "threads_batch"),
    candidates: "list[int] | None" = None,
    budget_seconds: float = CALIBRATION_BUDGET_SECONDS,
    on_event=None,
) -> "tuple[dict[str, int], list[dict]] | None":
    """Measure the candidates and return the winner, or None if nothing is safe.

    Assumes it is the only load on the machine. Two servers starting at once
    will each measure the other's contention and cache the result -- the swap
    rejection catches that on a memory-tight box and nothing catches it on a
    roomy one. Starting servers sequentially is the whole remedy; a lock was
    not added because the failure is a slightly-wrong thread count, not
    corruption, and the cache write itself is atomic.

    None means "fall back", and every failure path produces it: no candidates,
    no tokens, every candidate failing to decode, or every candidate rejected
    for swap. The caller never has to distinguish those to stay correct -- it
    falls back either way -- but the measurement table is returned alongside
    so the reason is recoverable.

    The budget bounds the sweep from the SECOND timed candidate on: the warm-up
    and the first candidate always run, so wall time is at most
    `budget_seconds` plus one candidate, and a start that exhausted the ceiling
    still leaves one real measurement behind rather than none.
    """
    # Threads are the only thing this can measure. Asked for anything else --
    # because the operator named the thread counts and left batch unset -- it
    # declines rather than spending a 47-second sweep to answer a question it
    # has no answer for. batch and ubatch are context-creation parameters and
    # cannot be retuned on a live context at all.
    if "threads" not in fields and "threads_batch" not in fields:
        return None
    picks = candidates if candidates is not None else thread_candidates(topology)
    if not picks:
        return None
    try:
        tokens = calibration_tokens(client)
    except Exception:
        return None
    if not tokens:
        return None

    started = time.perf_counter()
    measurements: list[CandidateMeasurement] = []
    # Untimed in effect: measured like a candidate so it walks the exact same
    # code path (and pays the same page-in), then excluded from scoring. Its
    # wall time counts toward the budget like any other work. A swap
    # rejection it earned is kept beside the marker: the box thrashed, and the
    # row should say so. A warm-up that fails part-way has not necessarily
    # paid the page-in, so the first timed candidate may be cold again; the
    # `warmup:error:` row is what makes that diagnosable afterwards.
    try:
        warm = measure_candidate(client, picks[0], tokens=tokens)
        marker = (
            WARMUP_REJECTION if warm.rejected is None
            else f"{WARMUP_REJECTION}:{warm.rejected}"
        )
        measurements.append(
            CandidateMeasurement(**{**asdict(warm), "score": 0.0, "rejected": marker})
        )
    except Exception as exc:
        measurements.append(
            CandidateMeasurement(
                threads=int(picks[0]), prefill_tokens=len(tokens),
                decode_tokens=CALIBRATION_DECODE_TOKENS,
                prefill_seconds=0.0, decode_seconds=0.0,
                prefill_tps=0.0, decode_tps=0.0, rss_mib=_rss_mib(),
                swap_used_mib=_swap_used_mib(), swap_growth_mib=0,
                score=0.0, rejected=f"{WARMUP_REJECTION}:error:{type(exc).__name__}",
            )
        )
    if on_event:
        on_event(measurements[-1])
    for index, threads in enumerate(picks):
        # The first timed candidate always runs. A warm-up that alone consumed
        # the ceiling (a slow disk, a contended page cache) would otherwise
        # produce no usable row at all, cache nothing, and repeat the same
        # sweep on every start with no signal; one bounded candidate more
        # leaves a real measurement behind instead. From the second candidate
        # on, exceeded means abandoned.
        if index and time.perf_counter() - started > budget_seconds:
            break
        try:
            measurement = measure_candidate(client, threads, tokens=tokens)
        except Exception as exc:
            measurement = CandidateMeasurement(
                threads=int(threads), prefill_tokens=len(tokens),
                decode_tokens=CALIBRATION_DECODE_TOKENS,
                prefill_seconds=0.0, decode_seconds=0.0,
                prefill_tps=0.0, decode_tps=0.0, rss_mib=_rss_mib(),
                swap_used_mib=_swap_used_mib(), swap_growth_mib=0,
                score=0.0, rejected=f"error:{type(exc).__name__}",
            )
        measurements.append(measurement)
        # Every row reaches the sink, failed ones included, so a listener's
        # count always matches the table's.
        if on_event:
            on_event(measurement)

    table = [asdict(m) for m in measurements]
    usable = [m for m in measurements if m.rejected is None and m.score > 0]
    if not usable:
        return None
    top = max(m.score for m in usable)
    # Fewest threads among those the benchmark cannot distinguish from the
    # top score. See TIE_MARGIN for the measurement behind this. The boundary
    # is inclusive, and `isclose` keeps it so when `top * 0.9` lands an ulp
    # off a score that was written as exactly that.
    floor = top * (1.0 - TIE_MARGIN)
    best = min(
        (m for m in usable if m.score >= floor or math.isclose(m.score, floor)),
        key=lambda m: m.threads,
    )
    # Only the fields actually asked for: an operator who set `--threads 8` and
    # left `--threads-batch` unset gets the measured batch count without their
    # explicit value being contradicted in the returned mapping.
    measured = {"threads": best.threads, "threads_batch": best.threads}
    return ({k: v for k, v in measured.items() if k in fields}, table)


def restore_threads(client, threads: int, threads_batch: int) -> None:
    """Put the context back on the resolved counts after measuring.

    Calibration leaves the context tuned to whatever candidate ran last, which
    is not necessarily the winner. Without this the server would serve requests
    on the final candidate's thread count while reporting the winner's.
    """
    ctx = getattr(getattr(client, "_session", None), "ctx_tgt", None)
    if not ctx:
        return
    client.lib.lib.llama_set_n_threads(ctx, int(threads), int(threads_batch))


__all__ = [
    "TIE_MARGIN",
    "WARMUP_REJECTION",
    "CALIBRATION_BUDGET_SECONDS",
    "CALIBRATION_DECODE_TOKENS",
    "CALIBRATION_PREFILL_TOKENS",
    "CandidateMeasurement",
    "calibrate_threads",
    "calibration_tokens",
    "measure_candidate",
    "restore_threads",
]
