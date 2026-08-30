# Self-MTP final benchmark checkpoint

First valid performance comparison of Orbit self-MTP with the persistent pair
and strict cache reuse, against MTP off, on the same current artifact.

## Identity

| item | value |
|---|---|
| Orbit HEAD | `10444388f648f50b2762dc89b1910b84249e2ac9` |
| llama.cpp upstream | `379ac6673b5cd75c7b4e07d1521c50f1e093878c` (b9551) |
| `source_tree_sha256` | `6c361f995c0734e8b59fb6255d1e55007ee3072dddda9f0d51921a9aaa94b793` |
| `patchset_v2_sha256` | `8ed0808ca95bf4988e26e6021c6207e1210fac504954bfa450bd04604215e32b` |
| `patchset_sha256` (legacy) | `aa3bb3d637ea963320df82dd588a0a85982936c192b7629d424daba0d1604441` |
| shim `.so` | `40d7512f16d44c255a100675a95f3cb7033ae0c66759e03d38c18683d15d9e3d` |
| model | `Ornith-1.5-35B-Q4_K_M.gguf`, `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f` |

Merged in this mission: `e99bcca6` (EOG canonical identity, PR #275),
`10444388` (/props self-MTP observability, PR #276).

## Hardware and configuration

Intel Core i7-10710U @ 1.10 GHz, 6 cores / 12 threads, 63 GB RAM, CPU-only
(`gpu_layers=0`), governor `powersave`.

Identical for both variants, from `NativeClientConfig` defaults — no overrides,
no code changes between B and C: ctx 8192, threads 6, threads_batch 6, batch
256, ubatch 128, one slot, temperature 0, thinking off.

* **B** = `orbit server --model <artifact>` (MTP off)
* **C** = `orbit server --model <artifact> --mtp` (self-MTP on)

## Protocol

Multi-turn, because the pre-existing `bench_mtp_throughput.py` builds a fresh
`ChatRuntime` per case and calls `ask_chat` once — every measured turn is a
FIRST turn, so no committed identity exists and resident reuse can never
activate. That harness would have measured MTP-without-reuse and reported it as
the feature. `scripts/bench_selfmtp_resident.py` keeps one conversation alive so
measured turns satisfy `0 < committed_len < prompt_len`.

* 4-turn conversation; turn 1 is an unmeasured warm-up establishing identity.
* 3 measured turns per repetition, 3 repetitions per variant, 2 variants = 6
  model loads per fixture.
* Balanced interleave (`B,C` / `C,B` / `B,C`) so neither variant systematically
  occupies the hotter thermal position.
* 45 s settle between servers; each server fully stopped before the next starts.
* Raw rows preserved. No run replaced, no fixture dropped after seeing results.

Two fixtures, both reported in full:

* **short** — 4 scripted one-line answers, ~7 generated tokens/turn.
* **long** — 4 prose paragraphs, ~230–280 generated tokens/turn. This is the
  fixture the throughput gate is applied to; the short one measures per-step
  overhead rather than throughput.

The long fixture was declared **before** the short fixture's repetitions 2–3
were inspected, specifically so it could not be post-hoc selection.

## Results — long fixture (gate applies here)

| metric | B (MTP off) | C (self-MTP) | delta |
|---|---|---|---|
| median generation | **8.272 tok/s** | **4.399 tok/s** | **−46.8 %** |
| raw generation (9 turns) | 8.53 8.44 8.27 8.38 8.06 7.73 8.31 7.99 7.82 | 4.29 4.35 4.01 4.03 4.40 4.57 4.68 4.72 4.43 | |
| median prefill | 1663.1 ms | 0.0 ms | −100 % |
| median TTFT | 1720.5 ms | 1876.9 ms | +9.1 % |
| median turn wall | 33.77 s | 54.59 s | +61.7 % |
| peak RSS | 36 556 MB | 36 752 MB | +0.5 % |
| generated tokens | 268 256 275 (×3) | 232 237 256 (×3) | |
| cached tokens | 319 636 940 | 42 343 623 | |

**C reuse rate: 9/9 = 100 %.** `pair_canonical` true on every measured turn.
Median acceptance ratio **0.360** (drafted 339 / accepted 126 / draft calls 113).

## Results — short fixture (reported, not gated)

| metric | B | C | delta |
|---|---|---|---|
| median generation | 7.815 tok/s | 3.271 tok/s | −58.1 % |
| reuse rate | n/a | 9/9 = 100 % | |
| median acceptance | n/a | 0.833 | |

## Threshold

Predeclared gate, unchanged since before any data was seen:
`median generation C / B >= 1.10`.

Long fixture: `4.399 / 8.272 = 0.5318` → **FAIL** (−46.8 %, not +10 %).

## Why it is slower

Not a reuse failure — reuse was perfect. Two compounding causes, both visible in
the counters rather than inferred:

1. **Low acceptance on natural prose.** 0.360 on the long fixture: roughly two
   of every three drafted tokens are rejected, and each rejection still costs a
   target verify. The short fixture reached 0.833 on scripted answers, so
   acceptance is heavily content-dependent.
2. **Compute-bound host.** Speculative decoding trades extra parallel compute
   for fewer sequential steps. On a CPU-only 6-core mobile part the extra draft
   and verify decodes are the binding constraint, so the trade is a loss. This
   is a property of the deployment, not of the implementation.

Prefill drops to ~0 ms under C, and cached tokens are consistently lower than B
at the same turn — C's committed identity ends at the physical frontier, which
sits behind B's whole-prompt cache. Neither offsets the generation cost.

## Deviations from protocol, declared

* **Competing workload present.** A wine process (~17 % CPU) and chromium ran
  throughout; they were not this session's to terminate. Package temperature sat
  at ~76 °C with load ~1.8 and did not fall below 62 °C during a 10-minute cool
  attempt, so the "cool to threshold, no competing workload" requirement is
  **not** fully satisfied. The balanced interleave neutralises systematic bias
  between variants but not variance.
* **Variance is material, especially on C.** Short fixture: B spans 7.74–9.10
  (±18 %), C spans 2.01–4.32 (±115 %). The long fixture is tighter (B 7.73–8.53,
  C 4.01–4.72) and the conclusion is robust — C's best measured turn never
  approaches B's worst — but the exact percentages carry real uncertainty and
  should not be quoted as precise.
* An initial short-fixture launch died after repetition 1 because `nohup` was
  killed with its parent shell. Its partial data is preserved as
  `raw_short_aborted_rep1.jsonl` and is **not** used in any aggregate; the
  complete 3-repetition run was relaunched under `setsid`.

## Files

* `raw_short.jsonl` — 24 rows, complete short fixture
* `raw_long.jsonl` — 24 rows, complete long fixture
* `raw_short_aborted_rep1.jsonl` — 8 rows, aborted launch, retained for
  completeness, excluded from aggregates
* `SHA256SUMS`
