# QWEN38-ALIGNED-ROUTE-PREWARM-25

Dell, 2026-09-17. Production baseline:
`69225ba98f1c957432ca81c5dc66af5dff89cbe7`. The native gate ran before any
production-source edit. Backend remains vendored llama.cpp `41abbfd`; model,
prompts, tool schemas and inference parameters are unchanged. The model was
resolved through Orbit's canonical resolver to `/srv/orbit-models`.

## Boundary cause and complete ROUTE equivalence

Fourteen actual first-user variants, including empty, whitespace, Unicode and
long text, share exactly **1048 token IDs**. User-dependent tokens begin at
index 1048. The qualified runtime calls native decode in **64-token** pieces:
`min(progress_step=64, batch=256)`. Ubatch is 128. The longest unchanged native
call boundary within the invariant prefix is therefore **1024**, derived from
the token LCP, not hard-coded by model name.

For both `hi` and `What is the capital of France? Answer with one word.`, the
probe captures natural boundaries during uninterrupted cold `complete_chat`,
restores full sequence state, evaluates the exact remaining suffix and compares
complete serialized state, fresh logits and greedy output. Off-boundary probes
extend the preceding natural checkpoint with a shorter native call. Every
newly tested prefix serialized/restored byte-exactly; this alone is insufficient.

| Prefix | Complete state equals cold, both users | Max logit delta, hi / France |
|---:|:---:|---:|
| 896 | yes | 0 / 0 |
| 960 | yes | 0 / 0 |
| 992 | no | 1.697340 / 2.185175 |
| 1024 | yes | 0 / 0 |
| 1025 | no | 1.953153 / 2.641029 |
| 1032 | no | 1.535442 / 1.450384 |
| 1040 | no | 1.419582 / 1.611893 |
| 1048, historical rejected control; not rerun | no | 2.031175 / 2.686543 |

960 succeeds despite not being divisible by batch or ubatch. Holding the
1024 endpoint and configuration constant but replacing its last 64-token
call with **32+32** fails complete-state equality, with logit deltas
**1.946169 / 2.380957**. Thus the rule is preservation of the qualified native
call sequence, not numerical divisibility by batch/ubatch. The active fused
GDN path has no 64-token chunk rule; the experiment does not attribute drift
exclusively to recurrent operations. Other graph operations also see different
tensor shapes. Do not generalize to unqualified call layouts.

At 1024, both complete ROUTE states and logits match cold exactly, and all
five greedy route-output tokens match. Sequence serialization excludes logits;
comparisons use newly computed suffix logits, never stale post-restore output.

The actual synchronous startup blob also matches the natural-boundary probe:
`e26caccdb672ba0f1c20dfc307830582abb32ee53d4b2dd8e0ecd5316ea6fd41`.
The first real terminal ROUTE before/after has the same complete-state SHA
`f328e8fcb402685e792dace9b6034c6149b832ca9eecd38fc7f5c0be70565e6c`
and logits SHA
`61c37faea5daf12d1505f50e563a3910a0e920a2acc03929dee470313487c4f3`.
The second fresh-user complete-state SHA in the native equivalence gate is
`de79dd697600cb10e2b9d8a5b5253bac432e436b6370f09b4877e2a8039ac3ca`.

## Smallest implementation and identity

Previously Flash Next lacked a fixed-prefix planner/startup branch and had
`route_prefix_reuse_supported=False`; the existing lineages also required
Q4_K_M file type 15 instead of its IQ1_M type 31. Startup reported
`model_profile_ineligible`.

Flash Next now uses the existing synchronous ChatML capture, existing Qwen
configuration/switch and existing default Qwen checkpoint slot. Qwen3.6 and
Flash Next cannot occupy one loaded client simultaneously. There is no new
scheduler or cache subsystem. Capture/restore uses the existing full sequence
primitive, preserving attention KV, recurrent and indexer state. No partial
`seq_rm`, padding, approximate prefix or prompt rewriting is introduced.

The shared derivation optionally rounds the token-ID LCP down to a native-call
boundary, validates the actual prompt's exact prefix, and rejects a shorter or
incompatible request. Existing fixed-count derivations retain their behavior.
Eligibility pins ctx4096, threads10/10, batch256, ubatch128, progress64, GPU0,
repack off, low-memory mode off, mmap/lazy on and MTP off. Other configurations fail closed.

The existing `compute_prefix_anchor_key` covers model metadata, all shard paths
and file identities, tokenizer/template, route-system/tool contract, exact token
and invariant-text hashes, backend build/library, state-format lineage, ctx,
batch/ubatch/progress step, threads and relevant load/repack/MTP settings.
No new fingerprint system or persisted state cache is added. Missing shards
reject reuse. A changed batch/ubatch invalidates the checkpoint.

Reset, cancellation, failed completion, model reload and close invalidate the
Qwen prefix. Failed restore clears even partially written native state before
cold fallback. Cancellation during decode or serialization publishes nothing.
A cache-missing request's lazy capture polls its latched disconnect callback at
every native call; review caught and fixed the queued-disconnect case where
request start had cleared an earlier cancellation event. Startup uses the
existing SIGINT cancellation window. Rolling/shadow checkpoints retain priority
and separate storage. Qwen ANALYSIS prewarm remains disabled; Ornith is unchanged.

Existing switches: `ORBIT_QWEN_ROUTE_PREFIX_REUSE=0` disables this prefix reuse;
`ORBIT_KV_PREFIX_PREWARM=off` disables startup capture. The latter still permits
normal lazy capture on the first eligible request.

## Cost and bounded live acceptance

Fresh server processes, same qualified parameters, existing cache diagnostics,
no OS page-cache flush. These are descriptive Dell timings, not a matched
page-cache throughput experiment.

| First `hi` turn | Before | After |
|---|---:|---:|
| ROUTE input | 1058 | 1058 |
| ROUTE cached | 0 | 1024 |
| ROUTE evaluated | 1058 | 34 |
| ROUTE cache fraction | 0% | 96.79% |
| ROUTE prefill | 132.828 s | 4.606 s |
| Full terminal first turn | 144.250 s | 16.921 s |

Route and final generated token IDs are identical. The separate two-message
native reproduction had 1058/1069 input, zero cached, 156.825/99.068 seconds
prefill; safe1024 restores evaluated 34/45 suffix tokens with zero state/logit
error. Twenty-four invariant tokens intentionally remain in the suffix.

**Synchronous prewarm moves computation to startup.** Measured prewarm wall
146.387 s (native prefill 146.007 s, 16 calls), blob 146,400,596 bytes
(139.62 MiB). Time through prewarm was 169.639 s versus 22.112 s through the old
skipped prewarm. RSS rose 8.72→25.73 GiB, peak 25.93 GiB; anonymous RSS rose 132.02
MiB. Most RSS growth was newly resident mapped model pages. Host file cache
rose 11.02→27.01 GiB; process storage reads increased 49,484,738,560 bytes
(46.09 GiB). These host/cache observations include warming and system activity.

No total-compute reduction or end-to-end startup-plus-request gain is claimed.
The measured startup-plus-first-turn sums were 166.36 s before and 186.56 s after;
starting page-cache warmth differed substantially. Prefix work remains 1024
startup tokens + 34 request tokens, rather than 1058 request tokens, with added
checkpoint capture/restore cost.

The bounded after run was short turn, second short turn, a naturally completed
465-token long final reply, 120 seconds idle, then another short turn:

- second ROUTE:1092 input,1058 rolling cached,34 evaluated; full turn10.179 s;
- third ROUTE:1133 input,1092 cached,41 evaluated; long reply completed normally;
- shadow:1133→1603 tokens,470 decoded in30.837 s;
- fourth ROUTE:1624 input,1603 shadow cached,21 evaluated; full turn12.995 s.

As in baseline, later routes can be aborted by the terminal when they emit
prose instead of a route decision, followed by `chat_final_retry`. These are
not eight ordinary stopped generations. Existing cancellation invalidates the
fixed prewarm after its first restore while rolling/shadow reuse continues.
Shutdown completed cleanly.

## Qualification evidence

Machine-local records (never committed):
`workdir/diag/qwen38_aligned_prewarm_25/`. They include native boundary and
call-split harnesses/results, all rendered prompts/token IDs, measured before/
after server and terminal records, seven applied/caught isolated mutants,
focused/cross-sample/full-suite logs and the reviewed source hash manifest.
The rejected1048 control remains in the mission24 diagnostic directory.

Deterministic coverage includes derivation, configuration/identity mismatches,
all-shard changes, complete hybrid state, cold call layout, reset/cancel/close,
latched disconnect, serialization cancellation, partial restore failure,
rolling/shadow composition, existing startup dispatch and ANALYSIS rejection.
No tokenizer or template fixture was re-baselined; the profile capability
expectation changed because the qualified feature is now supported.

Final gates: focused 482 tests RC=0; cross-sample 11 RC=0; full non-live
discovery 6012 tests, 45 skips, 585.356 s, actual child RC=0. Seven isolated
mutants applied and caught, with an 18-test green control. Compileall and
`git diff --check` passed. Independent adversarial review: BLOCKER 0 / MAJOR 0
after correcting the callback cancellation finding. The final production
source/test hashes remained unchanged throughout full-suite qualification.

Decision: **QWEN38_ALIGNED_PREWARM_FIXED**. This is a user-visible first-request
latency improvement by synchronous startup cost shifting. It is not a claim
of reduced total compute or an overall startup-plus-request speedup.
