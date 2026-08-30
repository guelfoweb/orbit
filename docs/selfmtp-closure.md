# Self-MTP: final engineering decision

Closes the Ornith-1.5 self-MTP performance line. Self-MTP remains available as
an explicit opt-in; it is not enabled by default and is not recommended for
default enablement on this artifact and host class.

## Verdict

| dimension | result |
|---|---|
| Functional | **PASS** |
| Production `/chat` | **PASS** |
| Persistent target KV | **PASS** |
| Persistent draft KV | **PASS** |
| `pending_h` persistence and alignment | **PASS** |
| Strict rolling KV reuse | **PASS** |
| Output correctness | **PASS** |
| Memory | **PASS**, 35.70 GiB (B) / 35.89 GiB (C) peak |
| **Performance** | **FAIL** |
| **Auto-MTP default** | **NO** |

Self-MTP works exactly as designed. It is still not worth enabling by default on
this hardware. Those two statements are both true and must not be collapsed:
functional qualification and performance qualification are separate verdicts.

## The measured result

From the qualified benchmark, measured at Orbit `10444388` and recorded at
`a9fa2c31` (the commit between them touches only the harness and the checkpoint;
`git diff 10444388 a9fa2c31 -- src/` is empty), model
`Ornith-1.5-35B-Q4_K_M.gguf`
(`42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`),
recorded in `workdir/bench/checkpoint/`:

* **B** = current artifact, MTP off
* **C** = current artifact, self-MTP on, persistent pair and strict cache reuse

Long fixture (~230–280 generated tokens per turn), 3 measured turns × 3
repetitions per variant, identical configuration, no code change between them:

| metric | B | C | delta |
|---|---|---|---|
| median generation | 8.272 tok/s | 4.399 tok/s | **−46.8 %** |
| median prefill | 1663.1 ms | 0.0 ms | −100 % |
| median TTFT | 1720.5 ms | 1876.9 ms | +9.1 % |
| median turn wall | 33.77 s | 54.59 s | +61.7 % |
| peak RSS | 36 556 MB | 36 752 MB | +0.5 % |

**Reuse rate 9/9 = 100 %**, `pair_canonical` true on every measured turn, output
correct in both variants. The predeclared gate `C/B >= 1.10` **fails** at 0.5318.

## Why it is slower

Not a reuse failure. Reuse was perfect on every measured turn. Two compounding
causes, both read from counters rather than inferred:

1. **Acceptance is 0.360 on natural prose.** Roughly two of every three drafted
   tokens are rejected, and each rejection still pays for a target verify. The
   short scripted fixture reached 0.833, so acceptance is strongly
   content-dependent and prose is the unfavourable case.
2. **The host is compute-bound.** Speculative decoding trades extra parallel
   compute for fewer sequential steps. On a CPU-only 6-core i7-10710U the extra
   draft and verify decodes are the binding constraint, so the trade is a loss.
   This is a property of the deployment, not of the implementation.

## Why the older MTP benchmark was invalid for this question

`scripts/bench_mtp_throughput.py` builds a fresh `ChatRuntime` per case and
calls `ask_chat` once. Every measured turn is therefore a FIRST turn: no
committed identity exists, so no resident claim can be derived and resident
reuse can never activate. It measures MTP **without** reuse and would report
that as the feature's performance.

Any self-MTP performance claim must come from a multi-turn harness that proves
reuse on the measured turns. `scripts/bench_selfmtp_resident.py` does this, and
records `resident_reuse_active`, `pair_canonical` and `cached_tokens` per turn so
a turn that falls cold is visible rather than silently averaged in. **A
throughput number without a reuse rate is not evidence.**

## Policy

* Current Ornith is supported normally; normal decode is the default path.
* `--mtp` remains available as an opt-in, qualified feature.
* Normal Orbit startup does **not** enable self-MTP for this artifact
  (`use_mtp_experimental` defaults to `False`).
* No opt-out is added, because nothing is opted in.

## Reopening this line

Requires **new evidence**, not intuition. Any of:

* an upstream llama.cpp change affecting QWEN35MOE MTP cost;
* a materially different CPU or backend (GPU offload, a machine where the draft
  is not competing for the same cores);
* a measured implementation change expected to remove a proven bottleneck —
  most plausibly the 0.360 acceptance ratio, which is the largest single lever.

Any reopened measurement must re-run the multi-turn protocol with reuse rate
reported alongside throughput, on a quiet machine. The original run carried
unremovable competing workload (~76 °C, an unrelated process at ~17 % CPU); the
balanced interleave de-biased the comparison but could not eliminate variance,
and the exact percentages should be treated as approximate.
