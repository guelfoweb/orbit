# AGENTS.md

## Role

This file is the AUTHORITATIVE engineering handoff for Orbit. It is written to be
sufficient for a fresh Claude Code or Codex session with NO prior chat history to
continue Orbit safely, including moving development between machines (the
NUC → Dell migration is closed; the Dell is the active workstation). It
preserves established facts, decisions, rejected
approaches (so they are not reopened), and prioritised next steps. When this file
and any older handoff text disagree, the ACTUAL repository (HEAD, tests, source)
is authoritative; then this file; then release notes.

## Current State & Machine-Migration Handoff

Last reconciled against the repository on 2026-09-13 (mission
POST-RC38-STATE-RECONCILE-1). The NUC → Dell migration closed earlier the same
day (DELL-MIGRATION-CLOSURE-1): code, model, corpus and diagnostics are all
present and verified on the Dell.

### Baseline (the anchor for a migration)
- `main == origin/main`, tracked tree clean — verify live with
  `git rev-parse main origin/main` and `git status` (this phrasing stays valid as
  `main` advances, unlike a pinned HEAD SHA). The now-closed NUC → Dell migration
  was anchored at the rewritten baseline commit `c1e4662` (a historical anchor, not
  current HEAD). Untracked `workdir/` scratch is expected and is NOT dirt to clean
  up — read the staging warning at the end of the RC38 entry before `git add`.
- Published release: `v0.0.1-rc38` (annotated tag → `95ba0d5`; GitHub pre-release
  at https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc38, no attached
  assets by convention). Package version stays `0.0.1`; releases are RC tags.
  `95ba0d5` = the rc38 doc/release commit; its qualified PRODUCTION code is
  byte-identical to its parent `9181c40`.
- **`main` has advanced beyond the released `v0.0.1-rc38` tag**; `git describe`
  reports the current offset. **Post-release research HAS started** — see
  "Post-RC38 (unreleased on `main`)" in Release State. Three post-rc38 commits
  change production behaviour (`fa67e5a`, `47d6b4b`, `c1e4662`); the remaining
  post-rc38 commits are documentation. The rc38 entry below no longer describes
  everything `main` does; read both.
- No in-progress mission. The repository is at a clean, qualified state: each
  post-rc38 commit was qualified, adversarially reviewed to BLOCKER 0 / MAJOR 0
  and pushed on its own branch before merge. Do NOT reopen qualification or
  optimisation without new measured evidence.

### Active workstation: the Dell (machine provenance)
- **The Dell is now the primary post-release workstation.** The NUC remains the
  machine every historical benchmark in this file was measured on.
- Dell profile: `dell` — Dell Pro 5 14 P514260 (laptop, mobo 0W3WT8, UEFI 2.1.5
  2026-05-21); Intel Core Ultra 7 366H, 16 cores, L2 24 MiB, 400–4900 MHz;
  30 GiB usable RAM (32 GB class) + 2 GiB swapfile; Intel integrated GPU on the
  `xe` driver (`Intel(R) Graphics (PTL)` — Panther Lake — Mesa 25.2.8, OpenGL
  4.6). Vulkan: the device reports `apiVersion 1.4.318`, while the installed
  loader/instance is only `1.3.275` — quote the device version when scoping GPU
  work, and note the loader may need upgrading to reach it. 953.9 GB NVMe root;
  Linux 7.0.0-31-generic x86_64; Python 3.12.3.
- **Qualified CPU-only Ornith execution has already been observed on this Dell**
  (operator-attested; the matching live IBAN re-verification record is
  `workdir/diag/verify_iban/` — `RC=0 elapsed=1215.1s model_calls=11 actions=3
  report=True`, decode sha `5d51e76599…`. That record carries no host field, so
  it corroborates rather than proves the host).
  Do NOT infer from the 30 GiB figure that Ornith cannot run here: that inference
  was drawn once during reconciliation from the NUC's ~35.8 GiB peak RSS and was
  wrong. That RSS is a NUC measurement, not a portable requirement.
- **Keep machine provenance separate.** Every performance number already recorded
  in this file (multisample bench, prefill/decode rates, RSS, prewarm timings,
  route/final prefix timings) is a NUC number. Do not overwrite them with Dell
  numbers; record Dell measurements as a separate profile when a benchmark
  mission produces them.

### What is NOT in git (machine-local) — transport status
The runtime is fully in git. The following are machine-local and are REQUIRED for
continuity (analysis qualification, benchmarks, reruns). All are now **PRESENT on
the Dell**:
- `models/` (gitignored): the GGUF model files. The qualified analysis model is
  Ornith-1.5-35B-A3B Q4_K_M at
  `models/ornith-ai--Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf`, sha256
  `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f` — VERIFIED
  byte-for-byte on the Dell (21 713 463 040 B). Re-download from Hugging Face
  (`orbit download …`) or copy the file; verify the sha256.
- `workdir/samples/` (the 6 frozen malware corpus items — NEVER committed):
  6/6 present and hash-exact on the Dell. Copy them by hand; do not fetch fresh
  copies (SHA must match the oracles). See the corpus table in the RC38 entry.
- `workdir/diag/` (the frozen oracles, per-mission diagnostics, the multisample
  bench, and the release-corpus-closure reconciliation): present, 788 files /
  21 MB. Key subtrees all present: `corpus_expansion_4` (Office DOC oracle),
  `corpus_4b863c7`, `corpus_mine_hta`, `iban`, `verify_iban`, `js_fold`,
  `vba_byteoffset`, `vba_autoexec`, `multisample_bench`, `release_corpus_closure`,
  `fattura_*`, `end_to_end_perf`, `evidence_kind`, `office_preflight`,
  `report_coverage`, `source_churn`, `kv_*`.
- `workdir/campaign/` (the live ANALYSIS campaign bundles and replay/scoring
  tooling): present, 946 files / 21 MB.
- The native libraries under `src/orbit/native_llama/vendor/lib/*.so*` are build
  OUTPUTS; the vendored SOURCE is tracked, so rebuild them on the new machine with
  `python3 scripts/build_native.py` rather than copying binaries. Built and
  present on the Dell (llama.cpp `b9551`), together with the six MTP shim
  binaries and both Orbit bridges.

### Machine-local research artifacts that were NOT transported (expected, not a defect)
These live OUTSIDE the repository and outside `workdir/`. They gate unit-test
SKIPS and nothing else — no failure, no corpus or ANALYSIS capability depends on
them. Recorded so a future session does not mistake the higher skip count for a
regression.

**Read the two numbers carefully, they are different quantities:** the four
bullets below sum to **77**, which is the ABSOLUTE number of skips on the Dell —
measured by collecting skip reasons, these eight modules account for 100% of the
Dell's skips, and every reason is an artifact-absence reason. The **delta against
the NUC's recorded 8 skips is 69**. The NUC was therefore not fully provisioned
either: its 8 skips were a subset of these same 77 (the self-MTP shim alone is 7
of them, and it lives in `/tmp`, so it does not survive a reboot on any host).
- `~/LAB/llama.cpp` — upstream llama.cpp worktree; gates 32 skips in
  `test_llama_provenance_v2` (vendor provenance attestation).
- `~/LAB/orbit-checkpoints/` — the preserved checkpoint/corpus store; gates 37
  skips across `test_lossless_ac_scoring` (18), `test_oracle_monotonicity` (15),
  `test_completion_shadow_integration` (2), `test_completion_shadow_scorer` (1)
  and `test_artifact_capabilities` (1, captured Ornith GGUF metadata).
- `/tmp/selfmtp-build/liborbit-persistent-mtp.so` — a scratch build output; gates
  7 skips in `test_selfmtp_ownership`. Transient even on the NUC (it does not
  survive a reboot).
- `models/unsloth--Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` — a second verified
  model; gates 1 skip in `test_history_serialization`.

**Git credentials and author identity were also not transported — both are now
RESOLVED on the Dell, but check them first on any future machine.** Initially the
Dell had no GitHub authentication of any kind (HTTPS `origin`, no
`credential.helper`, no `~/.netrc`, no SSH key, `gh auth status` not logged in,
no `GH_TOKEN`), so `git fetch` worked on the public repo while `git push` failed
with `could not read Username for 'https://github.com'`. Resolved by
`gh auth login`: `gh` is authenticated as `guelfoweb` with `repo` scope, and
`~/.gitconfig` now delegates `https://github.com` credentials to
`!/usr/bin/gh auth git-credential`, so `git push` works. Note the helper lives in
the USER gitconfig — `git config --get credential.helper` from inside the repo
still prints nothing, so check `git config --show-origin --get-all
credential.helper` or simply `git push --dry-run` rather than concluding it is
unset. Git author identity was likewise unset (`guelfoweb@dell.(none)`); it is
set repo-locally to `Gianni Amato <guelfoweb@gmail.com>` to match the existing
history, with no `--global` change.

### Cross-machine reproduction check (cheap, no rerun of the full campaign)
**This repository runs unittest, not pytest.** There is no CI, no `conftest.py`,
no pytest configuration and pytest is not a dependency (see
`docs/checkpoints/qrel1-python-source-freshness.md`). The canonical full-suite
command is:

```
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Expect **0 failures, 0 errors, RC=0**; the test count grows with every mission
(5462 at rc38, 5588 at `47d6b4b`, 5622 after DELL-RUNTIME-AND-IBAN-CLOSURE-1 —
all measured on the Dell). The skip count is host-dependent and is NOT a
pass/fail signal: 8 recorded on the NUC, **77 on the Dell**. All 77 are accounted for by the out-of-repo research artifacts listed
above; the 69-skip difference is which of those artifacts each host happened to
have. A green run is `OK` with a real child RC of 0 — never read the result from
a shell pipeline.

Then run one deterministic decode check on the transported IBAN.js (expect the
`js_fromcharcode_offset` stage sha256 `5d51e7659955a754…`) and one Office check
(`extract_office_vba` on the frozen `.doc` → module `ThisDocument`, 55 039 chars,
source sha256 `d034bd8381f4663a…`). If those hold, the qualified state has
reproduced; do NOT rerun the six-sample Ornith campaign for documentation.

### Machine-independent setup assumptions (new Linux box)
1. Linux x86_64, CPU-only, Python ≥ 3.11 (developed on 3.12), CMake present.
   RAM: the NUC has 64 GiB and measured ~35.8 GiB Ornith peak RSS there; the Dell
   runs the qualified profile on 30 GiB. Treat RAM as a per-host measurement, not
   a fixed threshold.
2. `git clone`, `python3 -m venv .venv && . .venv/bin/activate && pip install -e .`
3. `python3 scripts/build_native.py` to build the vendored llama.cpp/ggml `.so`s
   (see the two-manifest-hash gotcha in KV notes if you edit vendored sources).
4. Transport `models/`, `workdir/samples/`, `workdir/diag/`, `workdir/campaign/`
   as above; verify SHAs.
5. `TMPDIR=/tmp` for all runs (a project convention; some sandbox/temp paths assume it).
6. Qualified analysis server profile: `orbit server --ctx 8192 --threads 6
   --threads-batch 6 --batch 256 --ubatch 128 --think off` (MTP off — the default).
   Passing all four tuning flags explicitly is deliberate: since `47d6b4b` an
   omitted field would be auto-calibrated per machine, and qualification runs on
   the qualified numbers, not on measured ones.
7. Performance is host-specific: re-measure with the qualification harness before
   quoting any number; absolute wall time is contention-sensitive.

The authoritative capability matrix, qualified model/config, full corpus table
with SHAs and oracle locations, qualification & safety contracts, performance
baseline, TECHNICAL_STOP decisions, and current limitations are in the **RC38**
Release State entry below.

## Permanent Principles

- Correctness, stability, reliability, and simplicity come before performance.
- Orbit remains Python-first: prefer the standard library and small, readable, debuggable code.
- Primary target: CPU-only Gemma 4 26B-A4B Q4_0 through native `orbit server`.
- Runtime owns behavior; backend owns inference.
- Do not add hardcoded semantic fixes in routing or the tool loop.
- Deterministic guardrails are allowed only for safety, validation, bounded retry, and diagnostics.
- Do not trade correctness for theoretical speedups.
- Benchmarks and tests override intuition.
- `workdir/` is a public fixture: do not touch or stage `workdir/.miktex/` or `workdir/doc/`.
- Do not create tags or releases unless explicitly requested.

## Release State

### RC13

- Focus: MTP diagnostics.
- Added MTP diagnostics for throughput, config, timing, and validate efficiency.
- MTP is stable, but it did not prove robustly throughput-positive on CPU-only systems.

### RC14

- Focus: KV/final evidence diagnostics and compact final evidence.
- `cached=4` on final/retry was explained as prompt-view divergence from `route -> final`, not as a backend/cache bug.
- Slim compact final evidence metadata reduced evaluated tokens in small `final_from_tool` outputs.
- Multi-card `chat_final` remains a technical stop without reliable lineage/intent.

### RC15

- Focus: evidence lineage.
- `EvidenceRecord` includes `evidence_sequence`, `tool_call_id`, `user_turn_id`, and `produced_by_phase`.
- `producer_model_call_id` remains `null`.
- No active evidence selection or compaction.
- `dual_shell` confirms that `current_turn`-only selection is unsafe.
- Lineage smokes must use clean temporary workdirs, not contaminated persistent stores.

### RC16

- Dedicated final budget for `system_info`.
- CPU-first documentation and optional MTP guidance.
- Metadata header for `orbit bench-core`.
- Profiling guidance and conservative server-profile guidance.
- Draft MTP model download moved out of the base install flow; it is optional.

### RC17

- Published historical baseline: `v0.0.1-rc17`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc17
- Release notes commit: `358ed995fbc0607ebbda15099a8c568223ddb752`.
- Tag object: `8af6bd36d8bf0cd0e450e10af80bf8e5038fc408`.
- Tag commit: `358ed995fbc0607ebbda15099a8c568223ddb752`.
- Prerelease: yes. Latest: false.
- Includes #122, #123, #124, #125, and #126.
- Focus: post-RC16 agent guidance, MTP README clarification, conversation reuse route guidance, and smoke-result notes.
- RC17 validation: MTP shim build PASS, full unit PASS with 985 tests, `simple_chat --mtp-required` PASS, `git diff --check` PASS.
- RC17 MTP sanity: `mtp_enabled=true`, `mtp_initialized=true`, `mtp_failure_reason=null`, `in_flight=false`, `multimodal_available=true`, `mtp_last_completion.success=true`, `mtp_config.n_max=3`.

### RC18

- Published predecessor: `v0.0.1-rc18`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc18
- Release notes commit: `230db4341737380afb240cd701861d2ee350df7e`.
- Tag object: `3b327da630ed6d53f442a969c32e962e563589dc`.
- Tag commit: `230db4341737380afb240cd701861d2ee350df7e`.
- Prerelease: yes. Latest: false.
- Includes #127, #128, #129, #130, #131, #132, #133, #134, #135, and #136.
- Focus: compact web-error final handling, correct failed-search reporting, reduced `final_from_tool` instructions, compact evidence prompt metadata, and related guidance updates.
- RC18 validation: MTP shim build PASS, `compileall` PASS, full unit PASS with 989 tests, `git diff --check` PASS, and MTP strict smoke PASS.
- RC18 MTP sanity: `mtp_enabled=true`, `mtp_initialized=true`, `mtp_failure_reason=null`, `multimodal_available=true`, `mtp_last_completion.success=true`, `mtp_config.n_max=3`.
- `cached=4` remains expected and unresolved; RC18 reduces evaluated tokens but does not change route/final prompt divergence.
- No deterministic wall-time improvement is claimed.

### RC19

- Published predecessor: `v0.0.1-rc19`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc19
- Release notes commit: `eb68ad30f9539a731730f7c94397731db9fcbd28`.
- Tag object: `9437d96a05811b5b7fac7dfde3bd37499a1334ae`.
- Tag commit: `eb68ad30f9539a731730f7c94397731db9fcbd28`.
- Prerelease: yes. Latest: false.
- Includes #137, #138, #139, and #140.
- Focus: off-by-default experimental `final_from_tool` prefix reuse and repeatable OFF/ON benchmark, lifecycle, recovery, MTP-guard, and RSS/PID coverage.
- RC19 validation: MTP shim build PASS, `compileall` PASS, `tests.test_bench_core` PASS with 6 tests, `tests.test_smoke_harness` PASS with 44 tests, full unit discovery PASS with 1,022 tests, and `git diff --check` PASS.
- RC19 runtime sanity: experiment OFF PASS with default `cached=4`; experiment ON PASS with first-call capture and subsequent `cached=43` restore; MTP guard PASS with zero final-prefix capture/restore while MTP remained healthy.
- The experiment remains OFF by default. `ORBIT_FINAL_PREFIX_EXPERIMENT=1` enables eligible native `final_from_tool` reuse, with an exact net reduction of 39 evaluated tokens relative to default behavior.
- Default `cached=4` remains unchanged and unresolved. Experimental logits differ from cold full-prefill because segmentation changes; restore is bit-exact against an identically segmented baseline.
- No deterministic wall-time improvement is claimed. Non-stream timeout may require explicit `/cancel`, and bounded RSS allocator variation remains documented.

### RC20

- Published predecessor: `v0.0.1-rc20`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc20
- Release notes commit: `73ec0215258ce9aa8c4c5f69ea650edc2aa103c3`.
- Tag object: `6dc3740a8e87a9d56adf211c235fc13236609e32`.
- Tag commit: `73ec0215258ce9aa8c4c5f69ea650edc2aa103c3`.
- Prerelease: yes. Latest: false. The GitHub `releases/latest` endpoint does not resolve to RC20.
- Includes #142, #143, #144, #145, #146, and #147.
- Focus: structurally covered CHAT evidence omission, diagnostic route-output classification and benchmark aggregation, route/argument technical-stop guidance, and aligned 64-token `final_from_tool` prefix reuse enabled by default.
- RC20 final-prefix behavior: the first eligible native final captures the exact 64-token checkpoint; subsequent eligible finals restore `cached=64`. `ORBIT_FINAL_PREFIX_REUSE=0` is the immediate stable kill switch and restores non-reuse behavior with `cached=4` on the measured smoke.
- The stable variable overrides `ORBIT_FINAL_PREFIX_EXPERIMENT`; legacy-only configurations remain compatible, invalid stable values disable safely, and MTP, tools-off, thinking-enabled, route, chat, tool-call, retry, and repair paths remain ineligible.
- Exact-prefix validation: 58 content tokens plus six Gemma template/control tokens, next dynamic token 105, no padding, 55/55 bit-exact cold/segmented/restore probes, and maximum logits difference `0.0`.
- RC20 post-merge runtime sanity: default reuse PASS with six correct stop completions, one capture, five `cached=64` restores, and zero fallback; kill switch PASS with two `cached=4` completions and zero capture/restore; strict MTP PASS with healthy MTP and zero final-prefix activity.
- RC20 validation: MTP shim build PASS; prompt/final-policy/completion-budget PASS with 60 tests; evidence/runtime/tool-message PASS with 213 tests; resolver PASS with 3 tests; backend/native/protocol PASS with 118 tests; smoke harness PASS with 54 tests; full unit discovery PASS with 1,067 tests; `compileall` PASS; `git diff --check` PASS.
- Restored calls evaluate 36 fewer tokens than previous production, with cumulative evaluated-token break-even on the second eligible final. CPU timing remains workload- and output-dependent; no deterministic wall-time improvement is claimed.

### RC21

- Published predecessor: `v0.0.1-rc21`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc21
- Release notes commit: `b19c9ef1cf82dd07d6aeb70ea1e21c3e16bfc5eb`.
- Tag object: `11419be25bb87920be24fbede42d1dd6a3a19a82`.
- Tag commit: `b19c9ef1cf82dd07d6aeb70ea1e21c3e16bfc5eb`.
- Prerelease: yes. Draft: false. Latest: false. The GitHub `releases/latest` endpoint does not resolve to RC21.
- Includes #148, #149, and #150.
- Focus: canonical runtime tool-call validation enabled by default, deterministic value-preserving formal healing enabled by default, and process-isolated native-backend compatibility observability.
- `ORBIT_TOOL_CALL_CANONICAL_GATE=0` restores the legacy validation path. Invalid values disable the gate safely. The canonical contract rejects duplicate keys, extra arguments, missing required fields, wrong types, invalid ranges, unavailable tools, and policy, permission, or operational-limit denials before execution.
- `ORBIT_TOOL_CALL_HEALING=0` disables formal healing immediately. The fixed whitelist contains only known-envelope removal, trailing-comma removal, complete JSON-string `arguments` decoding, and registered-wrapper unwrapping. Repaired calls must preserve the exact tool name, keys, types, values, and argument count before passing the same canonical contract, guardrails, and executor path.
- Ambiguity, multiple candidates, incomplete delimiters or strings, `finish_reason=length`, timeout, cancel, schema failure, policy denial, permission denial, and operational-limit denial remain fail-closed. No semantic correction, aliasing, fuzzy matching, tool substitution, defaults, clamps, argument invention/removal/renaming, or nudge retry exists.
- The process-isolated generation comparator records versioned corpus, protocol, runtime configuration, model, renderer, tokenizer, exact 64-token prefix, MTP, tools, thinking, affinity, and thread identities. The verified manifest is observational and does not gate startup or inference.
- RC21 benchmark sanity used two distinct native-server processes. Both completed 8/8 evaluable scenarios with eight model calls, zero tool executions, and zero finalizations. Exact-tool match `0.833333`, unwanted-attempt `0.5`, and budget-truncation `0.125` remained visible; wrong-tool, unwanted-tool, and truncation are not formal-healing categories.
- RC21 validation: focused canonical/healing/comparator/capability/harness PASS with 151 tests; full unit discovery PASS with 1,165 tests; MTP shim build PASS; `compileall` PASS; `git diff --check` PASS. Default final-prefix PASS with capture then `cached=64`; combined kill switches PASS with `cached=4` and zero prefix activity; strict MTP/mmproj PASS with usable MTP and zero final-prefix activity.
- No semantic-healing, success-rate, or deterministic performance claim is made. Do not expand the repair whitelist or add a nudge retry without natural, repeatable malformed production-budget samples and separate safety evidence.

### RC22

- Published predecessor: `v0.0.1-rc22`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc22
- Release-notes commit: `2c40a0bf1a33aecac5fc259f60133f8bf1da02e8`.
- Tag object: `e44f8021b369ca522d05d3c49e29456e3047be47`.
- Tag commit: `2c40a0bf1a33aecac5fc259f60133f8bf1da02e8`.
- Prerelease: yes. Draft: false. Latest: false. The GitHub `releases/latest` endpoint remains on the stable release channel and does not resolve to RC22.
- Includes merge commit `c2be0ef` from #151.
- Focus: default-on post-tool final prose reuse while preserving canonical tool validation, deterministic formal healing, MTP, and aligned final-prefix reuse.
- `ORBIT_POST_TOOL_FINAL_REUSE` is enabled by default. `ORBIT_POST_TOOL_FINAL_REUSE=0` is the immediate kill switch; `1` explicitly enables it; invalid values disable it safely.
- Eligible results reuse the exact original `ChatResult.content` only after a stopped terminal `post_tool_route` with no new tool call, retry, pending error or guardrail, technical markup, or additional step. Uncertain cases use normal `final_from_tool`.
- Process-isolated validation recorded 50/50 correct stop reuses, 50 model calls eliminated, 65,189 evaluated tokens saved, median savings of 1,084 evaluated tokens and approximately 107.5 seconds per reuse in the measured workload, with zero false positives or skipped tools.
- Eligibility overhead was approximately 8.55 microseconds at p95. CPU timing remains workload-, output-, process-, and thermal-dependent; no deterministic speedup claim is made.
- Validation: focused tests PASS with 387 tests; full unit discovery PASS with 1,213 tests; process-isolated comparator PASS on 10/10 scenario pairs; MTP shim build PASS; `compileall` PASS; `git diff --check` PASS. Existing final-prefix `cached=64`/kill-switch `cached=4`, canonical/healing, MTP/mmproj, cancel, timeout, and reset gates remain unchanged and passing.

### RC23

- Published predecessor: `v0.0.1-rc23`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc23
- Release-notes commit: `2aada5ca15b6dc3c3ad945a0fd9591c580eecbed`.
- Tag object: `af89a85b66d86d362d1a6abe480cea1a4bf2c7f4`.
- Tag commit: `2aada5ca15b6dc3c3ad945a0fd9591c580eecbed`.
- Prerelease: yes. Draft: false. Latest: false. The GitHub `releases/latest` endpoint remains on the stable release channel and does not resolve to RC23.
- Includes merge commit `0a446a2` from #152.
- Focus: fail-closed mtmd ABI hardening and reproducible llama.cpp vendor provenance, with no vendor revision upgrade or inference-behavior change.
- Python now passes only primitives and opaque handles through the mandatory co-located Orbit mtmd bridge. The bridge constructs upstream mtmd structures from the active build headers, while unknown or mismatched ABI layouts fail before mmproj initialization.
- Bridge, sidecar, runtime libraries, compiler identity, build flags, headers, bridge source, and provenance are revision-bound. Core ctypes structures that remain passed by value are checked through `sizeof`, `alignof`, and relevant `offsetof` gates before use.
- The production vendor remains llama.cpp `b9551`, upstream commit `379ac6673b5cd75c7b4e07d1521c50f1e093878c`. The recorded source-tree hash is `4adb967e643363e7dc4d01d632b3a8471e0df2ec84ff304d364dc182f63e7ee1`; the 60-path Orbit patchset hash is `dea2f205ed2a73d09ad203e08ba85545474742dbb0191f4f1a9b3a86beb4b435`.
- `LLAMA_BUILD_COMMIT` and `LLAMA_BUILD_NUMBER` are explicit vendor metadata and are not derived from the parent Orbit repository. The staged b10068 candidate is not included in RC23.
- RC23 validation: focused ABI/native tests PASS with 105 tests; full unit discovery PASS with 1,223 tests; all six MTP helpers rebuilt from staging; real vision and audio mmproj inputs PASS; MTP initialization/completion PASS; final-prefix capture and `cached=64` restore PASS; cancel, timeout, reset, and restart coverage PASS; artificial ABI mismatch fails safely without a crash; `compileall` PASS; `git diff --check` PASS.
- RC23 makes no performance claim. Future vendor revisions require a separate process-isolated compatibility and performance comparison through the hardened bridge.

### RC24

- Published predecessor: `v0.0.1-rc24`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc24
- Release-notes commit: `41dbc63f19cd5d88a2b0f1aee1d44a29557bc431`.
- Tag object: `52ce691fe724d3c8c6272345f24e07bcdfe684b7`.
- Tag commit: `41dbc63f19cd5d88a2b0f1aee1d44a29557bc431`.
- Prerelease: yes. Draft: false. Latest: false. The GitHub
  `releases/latest` endpoint does not resolve to RC24.
- Includes squash merge `f424e9a` from #156 plus the documentation-only
  technical-stop merges after RC23.
- Focus: convergence on one production tool loop, explicit no-mutation safety,
  Gemma 4 26B-A4B Q4_0 CPU-first guidance, download progress, exact active-model
  reporting, and simpler installed CLI examples.
- RC24 validation: focused pre-merge selection PASS with 475 tests; independent
  focused review PASS with 496 tests and two skips; full discovery PASS with
  1,251 tests and three skips; MTP helper/native rebuild PASS; strict 26B target,
  draft MTP, and mmproj smoke PASS; final-prefix default `cached=64` and kill
  switch `cached=4` PASS; post-tool final reuse PASS; lifecycle and safety
  coverage PASS; `compileall` and `git diff --check` PASS.
- RC24 does not claim deterministic performance improvement or resolution of
  broad multi-deliverable agentic workflows. The production llama.cpp vendor
  remains `b9551` at upstream commit
  `379ac6673b5cd75c7b4e07d1521c50f1e093878c`.

### RC25

- Published predecessor: `v0.0.1-rc25`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc25
- Release-notes commit: `452a8583a7f5eb5cb3fc077d2462801949245719`.
- Tag object: `76b8f4160fd8e1b212422e8531a7a73d499541fd`.
- Tag commit: `452a8583a7f5eb5cb3fc077d2462801949245719`.
- Prerelease: yes. Draft: false. Latest: false.
- Focus: deterministic long-document display, complete literal search,
  bounded multilingual concept search, tokenizer-proven full-document
  admission, and explicit file, scan, and semantic coverage.
- Literal search scans one stable snapshot with zero model calls. Concept
  search uses at most one bounded multilingual planner and one verifier over
  exact evidence windows. Semantic chunking remains a technical stop.
- RC25 validation: document-search tests PASS with 42 tests; shared runtime,
  path, and evidence tests PASS with 329 tests; full discovery PASS with 1,349
  tests; native and MTP helper rebuild PASS; strict Gemma target, draft MTP,
  and mmproj smoke PASS; production corpus PASS 12/12; final-prefix default
  `cached=64` and kill switch `cached=4` PASS; `compileall` and
  `git diff --check` PASS.
- RC25 makes no complete semantic-recall or deterministic performance claim.
  See `docs/FULL_DOCUMENT_READING.md`,
  `docs/DETERMINISTIC_DOCUMENT_SEARCH.md`, and
  `docs/releases/v0.0.1-rc25.md`.

### RC26

- Published predecessor: `v0.0.1-rc26`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc26
- Prerelease: yes. Draft: false. Latest: false.
- Includes squash merges `792b1d0` from #160, `7d1ccd8` from #161, and
  `5addc1d` from #162.
- Focus: the exact verified Qwen 3.6 35B-A3B Q4_K_M native profile,
  Qwen-specific route-prefix reuse, complete Gemma final projections, and
  turn/session token observability.
- Qwen route reuse captures the complete hybrid state at the validated
  768-token boundary inside the 810-token invariant prefix. Cold, segmented,
  captured, and restored logits were byte-identical with maximum absolute
  difference `0.0`.
- Gemma validation retained the 12/12 strict production corpus, default
  final-prefix `cached=64`, kill-switch `cached=4`, strict target/draft/mmproj
  MTP, post-tool reuse, and document-search behavior.
- RC26 final validation: independent reviews PASS; full discovery PASS with
  1,417 tests; native runtime and packaged MTP helper rebuild PASS; Qwen and
  Gemma real-model gates PASS; `compileall` and `git diff --check` PASS.
- Qwen MTP, Qwen multimodal input, and other Qwen templates, quantizations, or
  variants remain unsupported. CPU timing measurements are descriptive and do
  not establish a universal speedup.
- See `docs/QWEN_3_6_COMPATIBILITY.md` and
  `docs/releases/v0.0.1-rc26.md`.

### RC27

- Published predecessor: `v0.0.1-rc27`.
- Focus: model-driven atomic generation of one bounded UTF-8 text artifact,
  content-only native generation, and ephemeral read-only verification.
- The model selects `write_artifact`, destination, overwrite behavior, parent
  creation, content, verification, and final answer. Runtime does not infer
  artifact intent, choose content, or repair generated bytes.
- Publication requires a stopped generation, valid UTF-8, the 4,096-token and
  64 KiB limits, stable path identity, and an atomic same-filesystem commit.
  `verify_artifact` becomes available only after publication and cannot mutate
  or select another path.
- Recovery is conservative: active and unknown owners are always preserved.
  Cleanup requires positive proof that the owner is inactive plus valid
  manifest, UID, boot, inode, path, symlink, and workdir checks. Age alone
  never authorizes deletion.
- RC27 validation: third independent review PASS; recovery matrix 10/10;
  concurrent-parent reproducers 2/2; focused artifact/tool-loop tests 89/89;
  full discovery 1,533/1,533; Qwen Snake 5/5 plus the general UTF-8 artifact
  corpus; Gemma production corpus 12/12; Qwen route-prefix logits equivalence
  with maximum absolute difference `0.0`; Gemma final-prefix `cached=64`, kill
  switch `cached=4`, strict target/draft/mmproj MTP, native and helper rebuild,
  `compileall`, and `git diff --check` PASS.
- Limits: one UTF-8 text file per request, no binary output, no hidden
  multi-file planning, and model-dependent tool selection. Crash residue with
  unknown ownership may require manual inspection. CPU timings are
  descriptive and no deterministic speedup is claimed.
- See `docs/ARTIFACT_GENERATION.md` and
  `docs/releases/v0.0.1-rc27.md`.

### RC28

- Published predecessor: `v0.0.1-rc28`.
- Includes squash merge `7b03221` from #164.
- Focus: exact verified native support for
  `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` through the dedicated
  `orbit-qwen3-coder-native-v1` profile.
- Authorization is bound to reviewed GGUF architecture, model, file type,
  tokenizer, expert layout, and embedded-template metadata. Filename matching
  alone never enables the profile, and unverified variants fail before
  inference.
- The profile uses its native ChatML and XML tool protocol. Generative artifact
  content uses a constrained JSON-string transport with strict UTF-8 decoding;
  malformed framing, invalid UTF-8, and length termination fail closed without
  trimming, normalization, semantic repair, or hidden retries.
- Supported capabilities include chat, tools, tool history and results,
  generative `write_artifact`, read-only `verify_artifact`, existing-file
  modification, and normal coding/tool workflows. Qwen3-Coder MTP, multimodal
  input, Qwen 3.6 route-prefix reuse, arbitrary exact-copy guarantees, empty
  artifacts, and unverified variants remain unsupported.
- RC28 validation: independent PR review PASS; Qwen3-Coder production corpus
  8/8; focused tests 344/344; full discovery 1,566/1,566; native build,
  `compileall`, and `git diff --check` PASS. Qwen 3.6 artifact/tool behavior and
  route-prefix reuse, Gemma artifact behavior, final-prefix `cached=64`, and
  strict target/draft/mmproj MTP remained passing and isolated.
- Local CPU measurements were approximately 18.8 tok/s for synthetic long
  prefill, 11.35 tok/s for synthetic decode, and 31.1 GiB peak RSS. The
  production corpus measured median prefill 28.16 tok/s, median decode 8.16
  tok/s, and 31.24 GiB peak RSS. These workloads are not directly comparable,
  and no universal performance claim is made.
- See `docs/QWEN3_CODER_COMPATIBILITY.md` and
  `docs/releases/v0.0.1-rc28.md`.

### RC29

- Published predecessor: `v0.0.1-rc29`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc29
- Includes squash merges `0549dea` from #165 and `ce7d802` from #166.
- Focus: fail-closed native runtime-family loading, exact-profile
  Qwen3-Coder route-prefix reuse, and synchronous Qwen3-Coder startup prewarm.
- Native loading now validates the complete dependency prefix before mapping,
  rejects dependencies that resolve outside the selected runtime root, and
  prevents a second libllama/libggml family from entering the process. Valid
  aliases of the same canonical family remain accepted, with no global
  environment mutation.
- The verified `orbit-qwen3-coder-native-v1` profile captures complete sequence
  state at the unpadded 768-token boundary inside its 789-token invariant route
  prefix. The process-local checkpoint is 75,507,864 bytes and has the dedicated
  `qwen3-coder-route-prefix-v1` identity. Cold, segmented, and restored logits
  were byte-identical with maximum absolute difference `0.0`.
- Native tools-on startup captures that checkpoint before server readiness, so
  the first real request restores `cached=768`. `ORBIT_KV_PREFIX_PREWARM=off`
  disables startup capture only; `ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE=0`
  disables Qwen3-Coder capture and restore entirely.
- RC29 validation: Qwen3-Coder production corpus 8/8; focused tests 148/148;
  full discovery 1,596/1,596 with four expected skips in the release worktree;
  Qwen 3.6 route-prefix `cached=768`; Gemma final-prefix `cached=64`; strict
  Gemma target/draft/mmproj MTP; `compileall`; and `git diff --check` PASS.
- Local measurements observed a warm route prefill of about 1.49 seconds versus
  24.40 seconds cold, with about 24 seconds moved to startup. The checkpoint
  added about 91.6 MB ready-state RSS. Timings are descriptive and
  hardware-dependent; reset invalidates the process-local checkpoint and the
  next route recaptures cold.
- Qwen3-Coder MTP and multimodal input remain unsupported. Tools remain on,
  routing remains model-driven, and Qwen 3.6 and Gemma checkpoint identities
  and behavior remain separate.
- See `docs/QWEN3_CODER_COMPATIBILITY.md`,
  `docs/NATIVE_ROUTE_PREFIX_STARTUP_PREWARM.md`, and
  `docs/releases/v0.0.1-rc29.md`.

### RC30

- Published predecessor: `v0.0.1-rc30`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc30
- Includes squash merges `b9dbd54` from #167, `286fe53` from #168,
  `dbfca70` from #169, `a09862e` from #170, `d6f9cb1` from #171,
  `fad6d06` from #172, `728ab09` from #173, and `69511da` from #174.
- Focus: the observer-only Qualification Harness v1, safe recovery from
  transient native capability-discovery failures, and fail-closed recognition
  of full-document read intent.
- Qualification uses strict versioned JSON fixtures and canonical `PASS`,
  `FAIL`, `TECHNICAL_STOP`, and `NOT_APPLICABLE` results. Common, agentic,
  optimization-parity, lifecycle/failure, and full-document capability suites
  validate deterministic production evidence without an LLM judge or a
  universal score. Production runtime and backend modules do not import
  `orbit.qualification`.
- Same-model optimization comparisons require correctness and operational
  parity before reporting performance. Baseline and candidate processes remain
  isolated, invalid parity suppresses performance deltas, and volatile
  available RAM remains descriptive rather than part of stable hardware
  identity.
- Native capability discovery no longer caches transient `/props` transport,
  timeout, or startup failures as a permanent non-native result. Valid native,
  confirmed non-native, and malformed successful responses retain the reviewed
  fail-closed cache semantics, with no polling loop or checkpoint change.
- Full-document intent recognition now accepts the qualified missing whole-read
  wording while masking quoted, fenced, and structured inert text. Oversized
  documents fail closed with `coverage=none` and required-context evidence;
  fit-capable documents retain complete analysis and cleanup behavior.
- RC30 validation: qualification tests 83/83; affected backend, document,
  profile, and runtime tests 357/357; full discovery 1,684/1,684 with six
  expected skips; `compileall`; and `git diff --check` PASS. Real core
  qualification passed 4/4 for Gemma, Qwen 3.6, and Qwen3-Coder. Qwen3-Coder
  prewarm restored `cached=768`, Qwen 3.6 route reuse restored `cached=768`,
  Gemma final-prefix restored `cached=64`, and strict Gemma target/draft/mmproj
  MTP passed with acceptance ratio `0.9167`.
- Full-document qualification passed 3/3. At context 8,192 the oversized case
  recorded `coverage=none` with 47,477 required tokens, while the fit case
  recorded complete coverage and clean snapshot cleanup.
- Qualification timing and RSS data remain descriptive. The harness does not
  provide TTFT estimation, statistical significance, cross-model rankings, or
  a universal model score.
- See `docs/releases/v0.0.1-rc30.md`.

### RC31

- Published predecessor: `v0.0.1-rc31`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc31
- Includes squash merges `190903d` from #175 and `ff74ee3` from #176.
- Focus: exact-profile Qwen 3.6 shell-tool prefix reuse and a concise README
  for new users. RC30 remains the predecessor containing Qualification Harness
  v1, transient native capability-discovery recovery, and fail-closed
  full-document intent recognition.
- The verified `orbit-qwen36-native-v1` profile keeps one separate process-local
  checkpoint for the exact tools-on, thinking-off,
  `exec_shell_full_command` tool-call path. It captures complete hybrid state
  at the unpadded 384-token boundary inside the 439-token invariant prefix.
- The checkpoint identity is `qwen36-shell-tool-prefix-v1`; it is separate from
  the 768-token Qwen route checkpoint and never applies to other tools,
  artifact verification, final generation, Qwen3-Coder, or Gemma.
  `ORBIT_QWEN36_SHELL_TOOL_PREFIX_REUSE=0` is the dedicated kill switch.
- RC31 validation: qualification tests 83/83; affected backend, document,
  profile, runtime, and shell-prefix tests 422/422; full discovery 1,701/1,701
  with six expected skips; native runtime and MTP helper rebuild; `compileall`;
  and `git diff --check` PASS.
- Real checks restored Qwen 3.6 route `cached=768` and shell `cached=384` per
  warm shell call with zero fallback. Qwen3-Coder startup prewarm captured once
  and restored `cached=768`; Gemma final-prefix restored `cached=64`; strict
  Gemma target/draft/mmproj MTP completed with acceptance ratio `0.9167`.
- Full-document qualification passed 3/3. At context 8,192 the oversized case
  reported `coverage=none` with 46,982 required tokens, while the fit case
  reported complete coverage and clean snapshot cleanup.
- The README now contains only the project summary, supported-model benchmark
  table, requirements, installation, and quick-start instructions. No runtime
  or backend behavior depends on the documentation change.
- See `docs/releases/v0.0.1-rc31.md`.

### RC32

- Published predecessor: `v0.0.1-rc32`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc32
- Includes squash merges `a474614` from #177, `92aae65` from #178, and
  `0c394d8` from #179.
- Focus: zero-setup interactive native-server startup. In a TTY, `orbit server`
  performs one bounded discovery pass, displays verified and informational
  local-model status, and permits selection of only available or missing
  verified models.
- Available verified selections retain normal exact profile authorization. A
  missing verified selection may use the existing Orbit downloader after
  confirmation; the canonical destination, registry identity, and exact
  detected profile must all match before server bootstrap.
- Unsupported and unverified local GGUFs remain informational and cannot be
  selected. Filename and registry metadata never authorize inference. Download,
  verification, interruption, or bootstrap failure cannot silently select or
  start another model.
- Explicit `--model`, explicit `--model-id`, non-TTY startup, and direct
  `orbit download` retain their previous behavior. No inference, routing,
  tokenizer, prompt, or backend behavior changed.
- RC32 validation: focused discovery, selector, download, and server tests
  105/105; native/profile tests 362/362 with three expected skips; CLI tests
  31/31; affected download and startup tests 255/255; full discovery
  1,747/1,747 with six expected skips; JSON validation, `compileall`, and
  `git diff --check` PASS.
- Real checks selected and started an available verified model, exercised the
  missing-model download/rediscovery/verification/start lifecycle without a
  multi-gigabyte transfer, and kept all failure paths fail-closed. Qwen 3.6
  route and shell reuse restored `cached=768` and `cached=384`; Qwen3-Coder
  startup prewarm restored `cached=768`; Gemma final-prefix restored
  `cached=64`; strict Gemma target/draft/mmproj MTP remained usable.
- See `docs/releases/v0.0.1-rc32.md`.

### RC33

- Published predecessor: `v0.0.1-rc33`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc33
- Includes squash merges `c9bc5fa` from #180, `3ec480d` from #181,
  `c671b29` from #182, `1184f7b` from #183, `ba5c580` from #184,
  `bc5f36a` from #185, `bfced71` from #186, `9b26da2` from #187,
  `71f7cc1` from #188, `a128bed` from #189, and `369829a` from #190.
- Focus: a compact inline CLI, deterministic slash commands, authoritative
  live progress and TTFT metrics, and qualified Qwen3-Coder memory/cache
  behavior.
- The CLI adds `/read`, `/search`, `/ls`, and `/models`, keeps non-TTY output
  deterministic, and reports live prefill/decode progress without estimates or
  extra model work. Exact `backend_ttft_ms` and `stream_ttft_ms` remain separate
  authoritative metrics.
- Qwen3-Coder route-prefix state now survives only the qualified internal
  chat/tools cache-mode transition while explicit reset, cancel, reload, close,
  identity mismatch, and restore errors remain invalidating.
- Verified Qwen3-Coder supports opt-in `--low-memory`; on the qualified NUC10 it
  reduced peak RSS from about 31.3 GiB to 18.3 GiB. Standard remains the default,
  and interactive startup offers the choice only for the exact verified profile.
- Opt-in backend-owned MoE expert-usage diagnostics are observational and
  disabled by default. The README benchmark table was refreshed from repeated
  measured runs on the documented host.
- RC33 validation: Qualification Harness 83/83; full unit discovery 1,839/1,839;
  vendor provenance, `compileall`, and `git diff --check` PASS. Real checks
  retained Gemma final-prefix `cached=64` and strict MTP/mmproj, Qwen 3.6 route
  `cached=768` and shell `cached=384`, and Qwen3-Coder route `cached=768` in
  standard and qualified low-memory operation.
- See `docs/releases/v0.0.1-rc33.md`.

### RC34

- Current published baseline: `v0.0.1-rc34`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc34
- Qualified production code: `183add43145a5cc3f8cb4f413d6777894949e383`; the
  release commit adds only `docs/releases/v0.0.1-rc34.md` and this entry.
- Includes squash merges from #191 through #218, ending with `183add4` from
  #218, `feef345` from #213, `6effa5a` from #217, `6e87211` from #216,
  `d35fd8a` from #215, `c6f92b7` from #214, `1f5b79c` from #212, `4c74928`
  from #211, `fef1d4f` from #210, `c205d52` from #209, and `8657488` from #207.
- Focus: deterministic context management, evidence-grounded finalization, two
  more verified native models, and exact multi-turn chat correctness.
- Admission is decided before every model call, and a session that cannot carry
  the conversation answers from verified evidence instead of a truncated window.
- Ornith 1.5 35B-A3B and Qwen3.8-27B are verified native profiles. Strict
  append-only KV continuation reuses a token-exact resident prefix and never
  attempts partial cache removal; bounded diagnostics report the first refusal
  without logging prompt text.
- Multi-turn windows keep the user turn ahead of the assistant, stored sessions
  are checked for resumability before becoming the active chat, and recovered
  attempts keep their measured tokens and report totals as partial.
- A deliberate route stop is no longer counted as a failed backend call, so
  ordinary conversational turns stop reporting a failure that never happened.
- RC34 validation: full unit discovery 2,172/2,172; `compileall` and
  `git diff --check` PASS; model discovery 5/5 VERIFIED and AVAILABLE with no
  duplicate rows; Qualification Harness PASS; Ornith multi-turn, tools,
  invalid-session rejection, `/reset`, and `/clear` PASS; one real Ornith
  native-stream smoke reported 2 calls, zero failed attempts, finish `stop`.
- Known limit carried into RC34: ordinary conversational chat does not reuse
  route KV between turns, because route and final prompts are different prompt
  families sharing only a few leading tokens. Expected under the current prompt
  topology, not a cache defect.
- See `docs/releases/v0.0.1-rc34.md`.

### RC35–RC37 (analysis workflow)

- rc35 (`9864c3b`): explicit CHAT/ANALYSIS workflow modes; sandboxed
  `execute_analysis`; attested evidence; `/report`; opt-in bounded autonomous
  mode; rolling KV reuse across chat and analysis turns.
- rc36 (`8d8f31f`): correctness hotfix — a grounded analysis report can no longer
  cite a value the analysis has already corrected.
- rc37 (`ff5541f`): UX (the analysis progress line), native-library isolation,
  and documentation; the analysis runtime itself largely unchanged.
- See `docs/releases/v0.0.1-rc35.md`, `…rc36.md`, `…rc37.md`.

### RC38 (deterministic deobfuscation + Office/VBA + qualified malware corpus)

- Current qualified production code: `9181c4020915cfd1905ce6caf182f57f9d84547e`.
  It is the PARENT of the rc38 release commit `95ba0d5` (the commit the
  `v0.0.1-rc38` tag points at), not that release commit itself. Consolidates the
  post-rc37 ANALYSIS arc (#260 → `9181c40`) into a
  qualified release candidate. The release commit `95ba0d5` changes only
  `docs/releases/v0.0.1-rc38.md`, this `AGENTS.md` entry, `MANIFEST.in` (the
  `prune workdir` hardening) and `README.md` — nothing under `src/` or `tests/`,
  so the production code at `95ba0d5` is byte-identical to `9181c40` and there is
  no production behaviour change.
- Qualified model: Ornith-1.5-35B-A3B Q4_K_M, sha
  `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`; config
  ctx 8192 / threads 6 / threads_batch 6 / batch 256 / ubatch 128 / think off /
  MTP off.

- **Capability matrix (capability → source seam → tests → status):**
  - structured ANALYSIS controller (no free-form autonomy) → `analysis_controller.py`,
    `analysis_runtime.py` → `test_analysis_controller*` → MERGED.
  - exact EvidenceStore / provenance → `evidence.py` → `test_evidence.py` (65),
    `test_evidence_authority.py` (67) → MERGED.
  - network deny in autonomous ANALYSIS → `analysis_network_policy.py`
    (execution-time deny) + `analysis_sandbox.py` (`--unshare-all` namespace) →
    `test_analysis_network_policy.py` (14), `test_analysis_sandbox.py` (48) → MERGED.
  - FINISH completeness invariant (no silent no-report; completion witness) →
    `analysis_runtime.py` → `test_analysis_finish_completeness.py` (15) → MERGED.
  - KV Stage A repair reuse / Stage B STEP reuse / FINISH control-history reuse →
    `kv_diag.py`, `analysis_runtime.py` → `test_analysis_rolling_kv*` → MERGED.
  - source-churn suppression → `analysis_runtime.py` → `test_analysis_duplicate_suppression.py` → MERGED.
  - large-source bootstrap / bounded-range analysis → `analysis_runtime.py` →
    `test_analysis_large_source.py` (12) → MERGED.
  - progress UX → `terminal/repl.py` + `runtime/analysis_progress.py` → `test_analysis_progress_display.py`,
    `test_progress_line_separation.py` → MERGED.
  - report deterministic coverage → `analysis_runtime.py` `deterministic_sections` →
    `test_analysis_report_deterministic_coverage.py` (7) → MERGED.
  - evidence kind fidelity (`analysis_action`, not `fetch`) → `evidence.py` →
    `test_evidence_kind_fidelity.py` (11) → MERGED (`9181c40`).
  - OLE2/CFB Office preflight + MS-OVBA extraction → `analysis_ole.py` →
    `test_analysis_ole.py` (36) → MERGED (`ce1f750`).
  - VBScript/PowerShell Chr-offset → `analysis_deobfuscate.py`
    `find_vbscript_chr_stages` → `test_analysis_deobfuscate.py` (72) → MERGED (`442d008`).
  - JS string-array ProgID folding → `analysis_deobfuscate.py`
    `find_js_stringarray_fold_stages` → `test_analysis_js_stringarray_fold.py` (18) → MERGED (`f2f59a4`).
  - JS `String.fromCharCode` constant offset → `analysis_deobfuscate.py`
    `find_js_fromcharcode_stages` → `test_analysis_fromcharcode.py` (43) → MERGED (`1fc5722`).
  - VBA byte-offset + StrReverse → `analysis_vba_eval.py` + `analysis_deobfuscate.py`
    `find_vba_byteoffset_stages` → `test_analysis_vba_eval.py` (56) → MERGED (`131c3fd`).
  - Office/VBA autoexec relationships → `analysis_vba_autoexec.py` →
    `test_analysis_vba_autoexec.py` (30) → MERGED (`c247187`).

- **Qualified malware corpus (6 frozen samples, local-only / UNTRACKED — must be
  transported by hand to a new machine; SHA must match the oracles).
  Status on the Dell: 6/6 PRESENT and hash-exact (verified 2026-09-13):**

  | sample (workdir/samples/) | bytes | sha256[:16] | status | oracle |
  |---|---:|---|---|---|
  | Fattura981033956.js | 7706 | `b7cfd5fdeb16d7b5` | FULL | `workdir/diag/fattura_*`, `end_to_end_perf/trajectory_79f9ae8.json` |
  | peXF7I6W.ps1 (YPS) | 1701 | `5eba3e4538cffbde` | FULL | `workdir/diag/arc1_impl/final_YPS.json` |
  | mine.hta | 50114 | `6840b6d84f7c7190` | FULL | `workdir/diag/corpus_mine_hta/oracle.md` |
  | 4b863c7be268…js | 45316 | `e1a3a8937909e56d` | CORPUS_PASS 4/4 | `workdir/diag/corpus_4b863c7/oracle.md`, `js_fold/` |
  | IBAN.js | 7963 | `86e23fa673271308` | FULL | `workdir/diag/iban/oracle.json` (see SHA note below) |
  | 99eb1d90…74d0809.doc | 217600 | `99eb1d90eb5f0d01` | PASS 7/7 | `workdir/diag/corpus_expansion_4/{ORACLE.md,BLIND_FROZEN.json}` |

  Multisample bench: `workdir/diag/multisample_bench/` (BENCH_REPORT.md,
  metrics.json). Release-closure reconciliation:
  `workdir/diag/release_corpus_closure/`. Live IBAN re-verification (2026-09-13,
  7/7, decode sha `5d51e76599…`): `workdir/diag/verify_iban/`. NB: the IBAN
  byte-SHA drifted from its oracle's recorded `f74ee186…` on a re-save; the
  CONTENT is unchanged (decodes to the same `decoded_mmgclz`) — see the IBAN
  reconciliation bullet below and `workdir/diag/iban/oracle.json`'s
  `sha_reconciliation` block.

  **`Fattura981033956_origin.js` is NOT a corpus item.** It is the 19 948-byte
  non-qualified companion of the 7 706-byte qualified `Fattura981033956.js`
  (see `workdir/diag/comment_view/COMMENT_STRIPPED_VIEW_RESULTS.md`, which lists
  it explicitly as non-qualified). Both live in `workdir/samples/`. A partial
  transport that carries only `_origin` and renames it will fail
  `test_analysis_source_coverage` / `test_analysis_cover_runtime` with
  `924fb89c… != b7cfd5fd…` — this happened once during the Dell migration.
  `workdir/samples/trivial_greeting_demo.js` (244 B) is likewise a benign
  non-corpus fixture, required by `test_oracle_monotonicity`.

- **Corpus qualification contract (established):** deterministic runtime work
  (transforms, EvidenceStore/provenance, canonical IOC, report grounding) must be
  COMPLETE and CORRECT; the model narrative/trajectory may vary within BOUNDED
  HONESTY — no fabricated IOC escaping as authoritative, no false RESOLVED,
  correct caveats, no unsupported material behaviour. The deterministic appendix
  preserves authoritative facts even when model prose is selective. Orbit does
  NOT guarantee model-output determinism.

- **Safety contract (verified by tests/source):** malware/macros/decoded scripts
  are never executed; decoded stages are inert data; ANALYSIS outbound network is
  denied; C2/payload URLs are evidence only; the sandbox isolates each action on
  a read-only artifact copy; cancellation is contained, not swallowed; hostile
  OLE/VBA parsing is bounded and fail-closed; the deterministic evaluators use no
  eval/exec/script engine; payload retrieval is not part of autonomous ANALYSIS.

- **Performance baseline (multisample bench, one CPU-only NUC):** 79 model calls,
  24 actions, 163,891 prompt / 69,615 cached = 42.5% reuse, 21,317 generated;
  phase cost FINISH > REPORT > PLAN > STEP > COVER (REPORT is the largest
  uncached recurring consumer). Rate-derived prefill/decode ~57%/43%; prefill
  ~26–32 tok/s, decode ~8.8 tok/s (matches the Ornith baseline). Absolute wall is
  contention-sensitive on this box and is NOT a comparable number.

- **TECHNICAL_STOP / rejected (do not reopen without NEW measured evidence):**
  further REPORT exact-KV reuse (only the ~384-tok system head reuses = ~2.4% of
  corpus eval, below the material bar); REPORT lossless compaction (~0.3% safe);
  REPORT excerpt reduction (safe frontier one step above a factual cliff); REPORT
  output shortening (budget-bound); comment stripping (~0% on the real corpus);
  suspicious-API lexical inventory (model overrode the hints); sub-threshold KV
  prewarm candidates; MTP production default stays OFF (measured CPU cost);
  SSD/expert-streaming research-only. Reconciled again in
  `workdir/diag/release_corpus_closure/PART_B_report_reuse.md`.

- **Known limitations (rc38):** static only (no execution, no remote retrieval,
  no dynamic sandbox); unsupported JS/VBA obfuscation families fail closed; the
  Office autoexec taxonomy is the qualified Word/Excel core events only; a model
  run may reach the call ceiling while deterministic evidence is complete; the
  deterministic preflight enforces MAX_INPUT_CHARS / per-family bounds; CPU-only,
  performance dominated by model inference.

- **IBAN oracle SHA reconciliation:** IBAN.js was re-saved after its oracle was
  frozen, so the OUTER file SHA drifted (oracle recorded `f74ee186…`; current
  local `86e23fa6…`, 7963 bytes); the analysis CONTENT is unchanged (decodes to
  the exact `decoded_mmgclz` sha `5d51e76599…`). The original qualified SHA is
  preserved and a `sha_reconciliation` block was added in `workdir/diag/iban/
  oracle.json` (diagnostic only; no production change).

- **rc38 validation (measured on the NUC):** full suite 5,454 passed / 8 skipped /
  0 failed (RC=0); CLI `--version` → `orbit 0.0.1`; `main == origin/main`, tracked
  tree clean. Frozen corpus qualification evidence is authoritative — no fresh
  six-sample Ornith campaign was rerun for documentation.
- **rc38 re-validation on the Dell (2026-09-13, same HEAD):** full suite 5,462
  tests, `OK (skipped=77)`, 0 failures / 0 errors, real child RC=0. Total test
  count is identical to the NUC's 5,454 + 8, so there is no test drift; only the
  skip count moves, for the out-of-repo reasons listed in the migration section.
- **QREL-1 (source-fresh qualification) — measured decomposition.** QREL-1 is run
  through `scripts/qualify_fresh.py`, which relaunches a fresh interpreter with a
  private per-run `PYTHONPYCACHEPREFIX` so no stale `.pyc` can execute:
  `python3 scripts/qualify_fresh.py -m unittest discover -s tests -p "test_qualif*.py" -q`
  → **114 tests, OK, RC=0**, being the Qualification Harness (`test_qualification_*`,
  **83**) plus the source-freshness suite (`test_qualify_fresh`, **31**).
  The earlier "QREL-1 qualification tests 92 passed" figure recorded here for rc38
  is unsourced — no run record in `workdir/diag/` or `workdir/campaign/` contains
  it, and neither QREL-1 subset nor their union measures 92. It has been replaced
  by the reproducible 83 + 31 = 114 decomposition. No production code is involved.
- **Release-artifact malware safety:** the 6 real corpus samples are NEVER
  git-tracked (untracked workdir scratch), so a build from a clean checkout
  cannot include them. `MANIFEST.in` was hardened to `prune workdir` + explicit
  `include` of only the 6 benign tracked fixtures (demo dropper, vuln-service
  demo, edit/text fixtures), so even a build from a DIRTY working tree — which
  locally holds the untracked malware corpus — packages no real malware (sdists
  are built from the filesystem, not from git, so the prior broad
  `recursive-include workdir *.js` glob was a footgun). Verified: the manifest
  file-list resolves to exactly those 6 benign files, 0 corpus samples.
  (`MANIFEST.in` is packaging metadata, not runtime behaviour.)
  **Second layer added 2026-09-13 (`.gitignore`):** `MANIFEST.in` protects the
  sdist but did nothing to stop an accidental `git add`, and the corpus/diag
  directories were only ever *untracked*, not ignored. `.gitignore` now carries
  `/workdir/samples/*` with the two benign fixtures re-admitted by name
  (`!/workdir/samples/suspicious_dropper_demo.js`,
  `!/workdir/samples/vulnerable_service.py`), plus `/workdir/diag/` and
  `/workdir/campaign/`. Verified: all 8 local samples ignored, both benign
  fixtures still tracked and not ignored, diag/campaign ignored, and no tracked
  file anywhere in the repo is shadowed by the new rules. The build backend is
  plain `setuptools.build_meta` with no VCS file finder, so `.gitignore` cannot
  change sdist contents and the `MANIFEST.in` guarantee is untouched.

  **Scope of that protection — it is NOT a blanket `git add` safety net.** Only
  `samples/`, `diag/` and `campaign/` are covered. `workdir/` as a whole is
  deliberately not ignored, because `workdir/bench/checkpoint/`, `workdir/media/`,
  `workdir/text/` and the edit fixtures contain legitimately TRACKED files — but
  those same directories also hold untracked, unignored local scratch, so
  `git add workdir/media` or `git add -A` is still unsafe. Known untracked
  residue at the time of writing: `workdir/media/*.png` (analysis screenshots —
  `sample_to_analyze.png` is by name a rendering of a sample under analysis),
  `workdir/doc/` (a personal project document), `workdir/bench/server_mtp_*.log`
  and `selfmtp_resident*.jsonl`, `workdir/__init__.py`, and `workdir/.miktex/`
  (which the Permanent Principles already say never to touch or stage). Stage
  `workdir/` paths by explicit filename, never by directory or with `-A`.
  `workdir/samples/trivial_greeting_demo.js` (244 B, benign) is ignored along
  with the corpus and is NOT re-admitted, even though `test_oracle_monotonicity`
  needs it: like the corpus it is transported by hand, so transport it with the
  six samples.

- See `docs/releases/v0.0.1-rc38.md`.

### RC39 (runtime / diagnostics / terminal-UX qualification over rc38)

- Published pre-release: `v0.0.1-rc39` (annotated tag; GitHub pre-release; no
  attached assets by convention). Package version stays `0.0.1`. Qualified
  production code: `a526c3622a32dcf7102a35f817851e513e42e9bb`; the release commit
  adds only `docs/releases/v0.0.1-rc39.md` and this entry, so its production tree
  is byte-identical to its parent.
- Bundles the qualified post-rc38 work (RC39-RELEASE-QUALIFICATION-1): measured
  server startup profile / auto-calibration (`47d6b4b`, `c1e4662`); deterministic
  zero-action report render + the empty-PLAN fix (`fa67e5a`, `c1e4662`); Ornith
  route-prefix `/props` observability (#332); model-aware read-only
  `--show-profile` (#334); terminal model-status colours (#333); terminal-only
  Markdown report rendering (#335); the R1 status-accessor extraction to
  `client_status.py` (#336). No model, prompt, controller, KV, cache or report
  behaviour change.
- Qualification: full unit discovery RC=0 (0 fail / 0 error, host-dependent skips
  only); QREL-1 31/31; deterministic corpus gate green incl. the IBAN
  `js_fromcharcode_offset` stage sha256
  `5d51e7659955a754d55a83bce9157d8999864ee30a4f3cd5dc752ed0191a7de0` and C2
  `https://productoslili.cl/cv/cr2.exe`; compileall + `git diff --check` clean; no
  import cycle; packaging verified from a clean `git archive` (no malware /
  scratchpad / backups / vendor build outputs) with an install + CLI smoke.
  Independent release review BLOCKER 0 / MAJOR 0. The heavy six-sample live Ornith
  campaign was NOT rerun (no analysis/model behaviour changed since rc38; the
  deterministic reproduction gate is the qualified cheap check).
- Technical stop carried forward: Intel `xe` iGPU / GPU work (decode ~3x slower
  than CPU, thermally infeasible on the reference laptop) — Orbit stays CPU-only,
  `gpu_layers=0`.
- Next unreleased-development baseline: `main` at/after the rc39 release commit;
  there is no unreleased production work beyond it.
- See `docs/releases/v0.0.1-rc39.md`.

### Post-RC38 (bundled into rc39; historical)

The commits below landed after the `v0.0.1-rc38` tag and their production work is
now RELEASED in `v0.0.1-rc39` (see the RC39 entry above). This section is retained
as the historical per-commit record (including the Dell power-audit diagnostics).
Newest last:

- `b465ed6` *docs: close the Dell migration handoff* — documentation only.
- `fa67e5a` *fix(analysis): an empty plan is asked once more when the runtime
  already decoded something* — **production behaviour change**, see the
  invariants below.
- `c9ace69` *docs: record the Dell CPU CHAT cache baseline for the GPU mission*
  — documentation only; closed CHAT-FIRST-TURN-CACHE-REUSE-1 as OUTCOME A (no
  regression, no production change). The baseline table it produced is under
  Suggested Next Objectives; evidence in `workdir/diag/chat_prefix_reuse/`.
- `47d6b4b` *feat(server): resolve a measured startup profile instead of
  shipping one machine's numbers* — **production behaviour change**; `orbit
  server` now resolves `threads`/`threads_batch`/`batch`/`ubatch`/`cache_ram`
  through a precedence chain and may spend ~47 s once per fingerprint measuring
  them. Full contract in "Server Startup Profile (auto-calibration)"; evidence
  in `workdir/diag/autocalibration/`.
- `c1e4662` *fix(server): calibrate on warm weights and prefer fewer threads;
  render a zero-step report* (#324) — **production behaviour change**; the
  DELL-RUNTIME-AND-IBAN-CLOSURE-1 entry below.

**Empty-plan re-ask invariant (`fa67e5a`) — what a future session must not
undo.** An empty PLAN stays legitimate: a model that has enough evidence may
plan nothing and the run reports from what it holds. The change is narrow. When
the deterministic transform preflight (which runs in `__post_init__`, before any
model call) has already established something, `plan_analysis` spends ONE extra
PLAN call that states what the runtime holds and offers the same empty plan back
as a first-class answer. The bound is four non-redundant clauses, each load-bearing:

1. a plan that already has questions gets no re-ask;
2. an iteration must remain to carry it (`attempt + 1 >= attempts`; the loop caps
   `attempts` at 2);
3. a call must survive it (`max_calls < 3`, mirroring COVER's spare-call rule) —
   the re-ask must never replace an honest close with a question nothing can act on;
4. exactly one per RUN, via the run-scoped `_empty_plan_re_asked`. PLAN is
   dispatched twice when admission withdraws the evidence-first message, so a
   phase-local flag would allow two. Do not "simplify" it to phase scope.

With no deterministic evidence `_deterministic_evidence_summary()` is empty and
nothing changes — one PLAN call, honest closure, no invented work. The same
commit made `AnalysisController.adopt_plan` atomic: it used to append each
question as it validated, so a plan REFUSED on a later entry left its earlier
questions adopted and a repair could plan on top of them. Depth-0 ids now come
from a local counter because `_next_id` reads a record that must not be written
before the commit. Qualification: 28 new tests, mutation gate 11/11 applied and
CAUGHT (`workdir/diag/empty_plan_recovery/mutation_gate.py`), controller suites
198, QREL-1 114 OK.

This is NOT a licence to add replanning. No new model call beyond the existing
bounded plan accounting, no sample-specific condition, no infinite replan, no
REPORT prose parsed to reopen analysis.

**DELL-RUNTIME-AND-IBAN-CLOSURE-1 (2026-09-13, merged as `c1e4662`, #324) — two
symptoms, two causes, two production fixes.** Evidence: `workdir/diag/dell_runtime_iban_closure/`
(`RESULTS.md` is the record; `mutation_gate.py` 12/12 CAUGHT).

- *Decode fell to ~10 tok/s on the auto profile.* The Dell had REBOOTED at 14:05;
  the first `orbit server` on that boot calibrated with the model not yet paged in,
  measured its FIRST candidate (6 threads) at 25 tok/s prefill under 17,957 major
  faults, and cached **8 threads** as the winner. A served turn at 8 threads took
  20.6 s against 10.6 s at 6 (`/props.native_threads` proved the context really ran
  8). The first candidate after load pays that page-in on EVERY calibration on this
  box, not only after a reboot (a prefetched page cache reproduced 22,534 faults:
  a 21.7 GB mmap and a 22 GB process do not both fit beside 12 GB of other cache).
  Warm, 6 and 8 score within 1-8 % of each other (repeat noise ~5 %) while the
  real turn is 2× slower at 8. Fix in `server_calibration.py`: one untimed
  warm-up pass over the same tokens before anything is scored (recorded with
  `rejected="warmup"`, counted against the budget, can never win), and the
  FEWEST threads among candidates within `TIE_MARGIN = 0.10` of the top score.
  `--recalibrate` now caches 6/6 on the Dell.
- *Separately, this boot runs under a package power cap* (RAPL MMIO PL2 = 17 W,
  effective min with the 65 W MSR value): explicit 6/6 measures 33-38 tok/s
  prefill / 12-16 decode instead of the 52 / 18.5 recorded on the previous boot,
  with 0 involuntary context switches, 3 major faults and no swap traffic during a
  turn — a frequency ceiling (busy cores 2.3-2.5 GHz; 4 P-cores 2.9 GHz; one core
  alone still 4.3 GHz). Neither CPU pinning nor the `performance` platform profile
  lifts it; the remaining levers need root or the operator (adapter / Dell thermal
  mode / thermald adaptive). **P4 host state — Orbit was NOT modified to compensate,
  and the "decode ≥ 17" class is not reachable while the cap stands.** The desktop
  `gpu-xe.sh` applet installed the same day is not the cause (measured).
- *IBAN "PLAN=[] shows only `no open question requires an action`".* Not the
  transform, not the re-ask, not report assembly: for a zero-action run `report()`
  already returns the deterministic-only report with the decoded body, the stage
  sha and the C2 as a verified indicator, at zero model calls. The REPL's
  `_ask_analysis` treated `run.last_step is None` as "nothing to render", printed
  the stop reason, rewound the analyst turn and returned — dropping that report
  (I4). Fix in `repl.py`: a zero-step run WITH a `final_report` falls through to
  the ordinary rendering and keeps its turn; without one the old behaviour stands.
  Reproduced against the real REPL and the real `report()` on IBAN.js
  (`zero_step_terminal_before.txt` 3 lines → `_after.txt` 41 lines). The one
  live run the mission allowed planned normally (plan_calls 1, 3 actions, report
  with C2), so the empty-plan path is covered by the offline reproduction and the
  unit suites, not by a live PLAN=[].
- *Diagnostic read-back added:* `NativeLlamaClient.native_thread_counts()`
  (`llama_n_threads`/`llama_n_threads_batch`), published as
  `/props.native_threads[_batch]` beside the resolved `threads`, and logged as
  `orbit-server native threads: N/N (<stage>)` after model load, after profile
  resolution (post `restore_threads`) and before bind. Never fails a start.
- **DELL-POWER-PROFILE-AUDIT-1 (2026-09-13, docs only) — the 17 W cap audited.**
  Read-only host audit plus reversible A/B, no Orbit change; record in
  `workdir/diag/dell_power_audit/RESULTS.md`. Durable facts: the Dell's RAPL
  **MMIO** package domain carries PL2 = 17 W on this boot while the MSR domain says
  65 W and the platform itself declares PL2 = 56.25 W (`processor_thermal`
  00:04.0 `power_limits/`); the effective cap is the lower value. Standalone
  native 6/6 measures **prefill ~41 / decode ~15.3 tok/s**, stable across three
  probes and a 2×768-token sustained run (quarters 15.5 → 15.1, package ≤ 73 °C,
  no throttle events, no swap growth during decode). `powerprofilesctl set
  performance` switches EPP, the SoC power slider and the dell-pc profile
  correctly and changes NOTHING (PL2 stays 17 W, decode +1.6 %); a
  power-saver → balanced cycle does not re-evaluate PL2 either. The charger is a
  65 W USB-C PD (20 V × 3.25 A), no adapter warning exists, and the only
  observable difference from the boot that measured 52 / 18.5 is that that boot
  started on battery (`AC Adapter (off-line)`) and this one on AC. Owner of the
  17 W value unresolved without root (firmware/EC at boot, or thermald
  `--adaptive`); package watts are root-only on this kernel. Verdict
  **CPU_BASELINE_NOT_READY** for the GPU benchmark until the operator recipe in
  RESULTS.md (turbostat witness, reboot-on-battery reproduction, thermald A/B,
  volatile PL2 write-back test) settles it. Quote 41 / 15 for this boot; do not
  compare a GPU run against 52 / 18.5 until the cap is gone.
- **DELL-POWER-CAP-ROOT-WITNESS-1 (2026-09-13, docs only) — the cap is CAUSAL and
  firmware-owned.** Operator-run root witnesses (`dell_power_audit/root_witness/`):
  RAPL energy sampled every 2 s shows the package pinned at **16.85–17.00 W** through
  prefill and decode (4.5 W idle) — the MMIO PL2 is the effective limit. With ONLY
  PL2 written from 17 W to the platform-declared 56.25 W (volatile), the identical
  probe measured **prefill 53–60 / decode 19.3–20.4 tok/s** at 37–40 W package —
  the historical 47–55 / 17–20 — versus 42 / 15.4 at 17 W: causality proven.
  Nothing re-asserted 17 W during a 30 s watch, the run, or a `systemctl restart
  thermald` (thermald excluded as owner); BIOS Thermal Management reads
  `Optimized`; OS profile / SoC slider / dell-pc / EPP never moved it. Owner:
  firmware/EC, set once at boot (this boot started on AC; the boot that measured
  52 / 18.5 started on battery — the reboot-on-battery reproduction is now
  RESOLVED, see DELL-POWER-BATTERY-BOOT-REPRO-1 below). Caution: at 56.25 W the package hit TjMax (100 °C) in ~40 s of a
  2×256-token run and logged 14 throttle events; sustained behaviour is PL1 45 W
  + fan and is NOT yet validated. Persistent fix, in policy order: BIOS Thermal
  Management `UltraPerformance` / BIOS-EC update (firmware route) → last resort a
  volatile RAPL write at boot (prefer `45000000` = PL1; rollback = write
  `17000000` or reboot). [SUPERSEDED by DELL-POWER-BATTERY-BOOT-REPRO-1: BIOS
  Thermal Management was subsequently set to `UltraPerformance` and is DISPROVEN as
  the fix — the persistent fix is a safe PL2 found by the ladder in that entry, not a
  thermal-mode change.] PL2 was restored to 17 W at mission end; the gate stays
  **CPU_BASELINE_NOT_READY** until a persistent state is chosen and
  `native_probe.py sustained 768 2` holds ≥ 18 tok/s below TjMax.
- **DELL-POWER-BATTERY-BOOT-REPRO-1 (2026-09-13, docs only) — the boot-on-battery
  discriminator is RESOLVED: the 17 W cap is set by BOOT POWER SOURCE, but the
  uncapped envelope is thermally infeasible as-is.** After a shutdown and a boot
  started ON BATTERY (kernel `ACPI: AC: AC Adapter [AC] (off-line)` at the 17:36
  boot), AC then re-attached, the MMIO package limits came up **PL1 23 W / PL2 65 W**
  — the 17 W short-term cap ABSENT. Contrast the same day's AC boot `probe_boot1722`:
  MMIO **[45, 17]**, decode 14.94, prefill 36.7. Both boots were measured with BIOS
  Thermal Management ALREADY set to `UltraPerformance` (operator-attested), so
  UltraPerformance neither lifts the AC-boot cap nor tames the battery-boot burst.
  Same machine/model/profile; the only
  difference is the power source latched at boot, so the 17 W PL2 is firmware/EC state
  chosen once at boot by power source (C1 proven — the discriminator the entry above
  left pending; the good 52 / 18.5 boot had started on battery). On the uncapped boot
  `native_probe.py sustained 768 2` ran twice from a cold 46 °C package
  (`probe_sustained_batboot1736{,b}.json`): **prefill 55.3 / 55.4 tok/s** (uncapped, vs
  ~41 capped), busy cores median ~3.5 GHz / p90 4.27–4.30 GHz (vs ~2.5 capped) — but the
  package reached **TjMax (98–100 °C) inside the untimed warm-up (512 prefill + 32 decode)
  plus the first timed prefill**, the 95 °C safety guard aborted decode after 1 token both
  runs, and hardware `package_throttle_count` went 0 → 34. **Decode never sustained; the
  ≥ 18 tok/s gate criterion was NOT met.** Lifting the cap (battery boot) is necessary but
  NOT sufficient — with BIOS Thermal Management already `UltraPerformance` and the OS
  `balanced` platform profile, the 65 W PL2 burst still hits TjMax on this thin chassis
  before any sustained decode, and a battery boot is NOT persistent either (the next AC
  boot re-imposes 17 W). The gate stays **CPU_BASELINE_NOT_READY**; the remaining blocker
  is purely THERMAL and reduces to ONE operator-assisted experiment, NO Orbit code: on a
  NORMAL AC boot, PL1 untouched, find the LOWEST safe PL2 that restores useful decode
  without reaching thermal limits — ladder PL2 **25 → 30 → 35 → 40 W** (only as far as
  needed), one candidate at a time, abort at ≥ 95 °C, reject any candidate with a material
  `package_throttle_count` increase, restore the original 17 W after every test, and stop
  as soon as decode ≥ 18 tok/s at a stable temperature. Only after a safe candidate passes
  `native_probe.py sustained 768 2` should persistence be considered, preferably through a
  supported Dell/firmware mechanism; a systemd RAPL write at boot is the last resort.
  Evidence: `workdir/diag/dell_power_audit/RESULTS.md`.
- **DELL-POWER-PL2-LADDER-1 (2026-09-13, docs only) — the lowest safe PL2 is
  30 W; the ladder is CLOSED and the CPU baseline is READY.** Operator-run on a
  NORMAL AC boot (MMIO PL1 45 / PL2 17 as found, ~51 °C idle, throttle 0), PL1
  untouched, root RAPL writes, 17 W restored after the test, no Orbit change. At
  volatile MMIO **PL2 = 30 W**, `native_probe.py sustained 768 2`
  (`probe_sustained.json`, 205 samples) measured **prefill 52.06 / decode 18.56
  tok/s** (cycles 18.73 / 18.39; quarters flat-to-rising every cycle), busy cores
  3052 / 3492 / 3999 MHz (p10/med/p90), package 71 → **92** → 86 °C — no thermal
  abort (95 °C guard, 3 °C headroom). By the "only as far as needed" rule 30 W is
  the FIRST rung meeting the ≥ 18 tok/s target, so **35 / 40 W were NOT tested**
  and must not be: 30 W already peaks 3 °C under the guard. Two closing witnesses,
  both acceptable: (1) `throttle_after == throttle_before` exactly ({pkg 0, core 0}
  both) — no throttle event; (2) the swap 1527 → 1961 MiB (+434) is model-load /
  residency, NOT decode-window paging — the `before` snapshot predates
  `client.load()`, the delta matches the 35.3 s mmap of the 21.7 GB GGUF evicting
  cold desktop pages, and the timed window shows no paging signature (flat/rising
  decode quarters, RSS FELL 26448 → 25536 MiB cycle-to-cycle, MemAvailable 13 GiB
  after). The probe samples only temp+MHz per tick, so witness (2) is a behavioural
  inference, not a per-tick swap series, but conclusive. **Verdict:
  CPU_BASELINE_READY at PL2 = 30 W (VOLATILE).** Caveats: 30 W is a volatile RAPL
  write restored to 17 W after the test (the next AC boot re-imposes 17 W — NOT
  persistent yet); a GPU comparison must run under the SAME PL2 = 30 W (or note the
  power state) so CPU and GPU are matched; persistence, if wanted, is a later step
  (supported Dell/firmware mechanism preferred, systemd RAPL write at boot the last
  resort). This is a standalone-native probe number (512 prefill + 768 decode
  synthetic), a separate harness from the CHAT-turn baseline below — do not conflate
  them, and keep it a Dell profile, never overwriting NUC history.
  Evidence: `workdir/diag/dell_power_audit/RESULTS.md`, `probe_sustained.json`.
- RESOLVED (SERVER-SHOW-PROFILE-MODEL-RESOLUTION-1): `--show-profile` is now
  model-aware and read-only. It resolves the model the same way a real start
  would — an explicit path/id, the shared interactive selection (`_choose_verified_model`,
  the same chooser real startup uses), or the default when non-interactive — and
  prints `model:`/`path:` so the profile is never an unlabelled generic one. A
  model absent locally is reported as a heuristic preview (no download); the
  preview never loads a context, calibrates, prewarms, binds a socket or touches
  the profile cache. Precedence is unchanged (CLI > ORBIT_* env > user profile >
  cached measured > calibration-at-startup > heuristic) because it reuses
  `_resolve_startup_profile`. The old behaviour (previewing the absent default
  model's 16-thread heuristic without naming it) is gone; the fingerprint is the
  same `_model_identity_for_profile` real startup uses.
- RESOLVED (TERMINAL-MARKDOWN-REPORT-RENDERING-1): the terminal-only Markdown
  contract is in force and enforced by tests. `report.text` is the canonical
  artifact and stays byte-identical for API, files, sessions, redirects and any
  non-TTY output; ONLY the interactive REPL presentation renders Markdown, via
  `orbit.terminal.markdown_report.render_report` (headings, bold, inline code,
  ordered/unordered lists, fenced code kept verbatim, plain paragraphs). It is
  used only in the terminal layer (`repl.py`), never in API/session/file paths;
  it sanitises untrusted report bytes BEFORE adding any escape and keeps every
  marker, so stripping the escapes returns the sanitised text exactly (rendering
  changes appearance and nothing else). `supports_ansi()` gates styling, so
  NO_COLOR, a dumb terminal, a pipe and a redirect all yield raw Markdown with no
  escape; malformed/unsupported Markdown (e.g. an unclosed fence) degrades to
  readable plain text; URLs/IoCs/hashes/decoded payloads are reproduced
  byte-exact. Do NOT route rendered terminal text back into runtime state, and do
  NOT render outside the interactive TTY presentation layer.
- RUNTIME-DECOMPOSITION-CAMPAIGN-1 (deep-refactor gate = OPEN): the behaviour-preserving
  decomposition of `native_llama/client.py` ran R1-R6. **R1 MERGED** (#336): the eight read-only
  status/metrics accessors (`*_route_prefix_reuse_status`, `final_prefix_experiment_status`,
  `compatibility_diagnostics`, `model_load_status`, `moe_expert_usage_status`) moved to the new
  `native_llama/client_status.py`; client keeps thin delegates (bodies byte-identical, /props
  unchanged, no cycle). client.py 5416 -> 5262 LOC. **R2-R6 TECHNICAL_STOP**, because each target
  responsibility is ALREADY owned by a dedicated module and the client residual is correctly
  RUNTIME-owned orchestration: R2 capability logic -> `capabilities.py` + `artifact_capabilities.py`
  (residual `supports_vision/audio` is load+decode-coupled); R3 prompt/template -> `chat_template.py`
  + `chat_bridge.py` + `serialize_profile_messages` + `*_route_prefix` (residual `apply_chat_template`
  is renderer dispatch that mutates render state); R4 admission -> profile object + session-coupled
  skip-guards (`native_request_in_flight`/`active_context_present`/`prefill_in_flight`); R5 session/MTP
  lifecycle -> `mtp_session_lifecycle.py` + `persistent_mtp.py` (client owns its own session/handles
  intrinsically); R6 rolling-KV/cache -> `rolling_route_anchor.py` + `prefix_anchor.py` +
  `rolling_anchor_store.py` + `committed_identity.py` (residual is decode-coupled capture/restore
  orchestration; separating it would drag ctx/session state or change cache identity). Forcing any
  R2-R6 extraction would drag inference/session state, duplicate client state, or invent an artificial
  module -- all disallowed. Evidence: `workdir/diag/runtime_decomposition_campaign/`
  (BASELINE.md, EXTRACTION_MAP.md, FINAL_ARCHITECTURE.md). Public API preserved (31 public methods; method set unchanged from baseline). Do NOT
  re-attempt R2-R6 without new structural evidence; decomposing the completion/decode path is a
  separate future mission with its own hot-path qualification.

- OPEN-PR-TRIAGE-1: the six long-standing open PRs (retained only because they were open during
  the remote-branch cleanup) were audited against current main and all CLOSED; open-PR count is 0.
  No code was ported and main is unchanged. #50 route-outcome diagnostics and #91 dynamic completion
  budget were DUPLICATE (already on main: docs/ROUTE_OUTCOME_OBSERVABILITY.md + emit_route_outcome
  via 426a969; completion_budget.py resolve_max_tokens via 79bc17d). #45 KV phase-0 baseline was a
  SUPERSEDED docs-only report (KV route-prefix arc complete, REPORT exact-KV is TECHNICAL_STOP). #33
  was OBSOLETE (its suggest-server-profile.sh was rewritten into a server_profile wrapper; its
  final_policy findings-fix gate was never adopted). #1 (promote MTP boundary-split to default) and
  #3 (commit built .so binaries + MTP debug reuse) were UNSAFE against the MTP-default-OFF invariant
  and the build-output/technical-stop policy. All six remote branches were deleted (head tips
  recorded in workdir/diag/open_pr_triage/head_tips.txt for recovery).
- STALE-BRANCH-FINAL-SWEEP-1: the three remaining non-main remote branches were then all
  deleted (D/B classifications): baseline/opt-in-agent-mode-26b (agent mode is a decided
  removal; its #155 no-mutation policy already on main), compact-previous-shell-evidence
  (#97 closed as a correctness-regression experiment), smoke-harness-settled-mtp-props
  (#103 closed as a failed harness-semantics experiment; settled props already on main).
  No code ported; tips recorded in workdir/diag/stale_branch_final_sweep/RECOVERY_INVENTORY.md.
  **origin now holds only `main` plus the release tags.**
- RESOLVED (IBAN-EVIDENCE-GROUNDING-CLOSURE-1): two evidence-grounding defects on the
  IBAN.js path, both fixed at generic runtime seams (never hardcoded, no fuzzy match, no
  budget change, no PLAN forcing, deterministic authority preserved). Root causes and closures:
  - **Defect A — redundant re-derivation of a deterministic stage (root cause R4, action
    admission).** The transform preflight decodes IBAN.js to one authoritative stage (MMGCLZ,
    564 chars, sha `5d51e765…`) and restores it evidence-first into PLAN, so the answer is
    already in hand; but action admission suppressed only exact-CODE duplicates
    (`observation_fingerprint`) and source-reacquisition (output == the covered/delivered
    SOURCE). Nothing recognised an action whose OUTPUT reproduces a recorded deterministic
    STAGE, so a re-decode was admitted and counted. Fix: `_transform_reacquisition` in
    `analysis_runtime.py` generalises "establishes nothing new" to the transform stages —
    a successful, complete, unaltered, artifact-free stdout that is byte-exact (`classify_output`)
    or source-dominated (`classify_dominated`) against a recorded stage is suppressed
    (`action_executed=False`, not counted, NO_PROGRESS pointing at the stage's evidence id),
    exactly parallel to source reacquisition one seam inward. It triggers on a CORRECT
    re-derivation and does NOT depend on the observed bad-arithmetic failure (a failure keeps
    `stderr` non-empty and reaches the model as the failure it is).
  - **Defect B — report misstated `GetSpecialFolder(2)` as the Windows folder (root cause:
    runtime never surfaced the platform-constant meaning; the model filled it from memory).**
    WSH `Scripting.FileSystemObject.GetSpecialFolder(n)` is a closed, documented enum
    {0=WindowsFolder, 1=SystemFolder, 2=TemporaryFolder(%TEMP%)}. Fix (two runtime-owned seams,
    never model memory): (1) `folder_semantics()` surfaces the mapping for any defined constant
    literally present in an authoritative decoded stage as a deterministic fact inside
    `deterministic_sections()` — fronted by `DETERMINISTIC_AUTHORITY_PREAMBLE` ("quote from here,
    not memory"); undefined indices fail closed; (2) `_flag_special_folder_contradictions`
    prepends a correction (mirroring the unsupported-indicator notice) when the report itself
    writes `GetSpecialFolder(n)` beside the wrong enum name, stating the fixed mapping rather
    than rewriting prose. Acceptable renderings: `%TEMP%`/temporary folder OR the verbatim
    expression; `<Windows>\…` for index 2 is prevented at the fact seam.
  Tests: `tests/test_analysis_transform_reacquisition.py` (12) and
  `tests/test_analysis_special_folder_semantics.py` (14); causal mutation confirmed both fixes
  load-bearing. Frozen decode identity unchanged (stage sha `5d51e765…`, C2
  `https://productoslili.cl/cv/cr2.exe`); no corpus sample other than IBAN uses GetSpecialFolder,
  so blast radius is IBAN-only. Do NOT hardcode the sample constants, add fuzzy matching, or raise
  the action budget; the seams are generic to any deterministic stage and any WSH special folder.
  Live single-smoke on Ornith (CPU-only Dell, model sha `42739874…` verified out of band) after the
  fix: `RC=0 elapsed=225.4s model_calls=2 actions=0 report=True stop="no open question requires an
  action"` — the qualified zero-action path (down from the pre-fix `elapsed=1215.1s model_calls=11
  actions=3` in `workdir/diag/verify_iban/`). The report renders the deterministic fact
  `GetSpecialFolder(2) = TemporaryFolder (%TEMP%)` and the verbatim `fso.GetSpecialFolder(2) +
  "/TKFSIK.exe"`, never the Windows folder; C2 and stage sha present; no false contradiction banner.
  Record: `workdir/diag/iban_evidence_grounding/live_smoke.json` (machine-local).
- RESOLVED / TECHNICAL_STOP (MINE-HTA-IOC-GROUNDING-CLOSURE-1): three defects from a mine.hta run
  (sample sha `6840b6d84f7c7190…`, unchanged from the frozen corpus). All fixes are generic runtime
  seams -- no sample strings/domains hardcoded, no domain allowlist, no fuzzy matching, no budget
  change, no forcing of questions/actions, IOC extraction not weakened.
  - **Defect A -- XHTML namespace published as a verified IOC (root cause I2, indicator admission).**
    `<html xmlns="http://www.w3.org/1999/xhtml">` was read by `uris_in` as an ordinary URI and
    published under Verified indicators. Fix in `analysis_indicators.py`: `uris_in` skips a URI that
    is the value of an `xmlns`/`xmlns:prefix` declaration, recognised by SYNTAX
    (`_is_namespace_declaration`) -- the attribute must be a WHOLE token (`(?<![\w.\-:])` so
    `data-xmlns`/`myxmlns` are not declarations) and the value must be flush against a quote (so a
    stray `xmlns=` token then a spaced URL is not one). The SAME URI outside xmlns syntax stays an
    indicator; fail-closed toward keeping the IOC when a truncated look-back cannot see the boundary.
  - **Defect B -- report doubted an invocation the decoded stage proves (root cause G1, deterministic
    fact omitted from grounding).** The recovered PowerShell stage defines `function ROmYsTcn` and
    ends with the bare call `ROmYsTcn;`, yet the report hedged "not whether that routine is actually
    invoked". Fix in `analysis_runtime.py`: `_stage_entry_invocations` recognises a function a decoded
    stage both defines and calls argumentless as a bare statement (string/comment-masked via
    `_code_mask`; `End Sub`/`End Function` closers and keyword names excluded). `stage_invocations()`
    surfaces it as a deterministic fact in `deterministic_sections()`, stating INNER-stage invocation
    as established while keeping OUTER-container reach a separate/unresolved question (no full-chain
    overclaim). `_flag_invocation_contradictions` (both report-grounding paths) flags a report that
    denies the inner call -- sentence-scoped, suppressed on outer-container hedge language, never on a
    sentence that shows the call, so the nuanced-correct report the fact block invites is not flagged.
  - **Defect C -- two "the completion state could not be read" blocks after successful actions
    (classification C6 bounded model variance -> TECHNICAL_STOP).** The model ran an action, then
    failed to emit a usable `finish_analysis_question` even after the one allowed repair (saved run:
    `workdir/diag/multisample_bench/minehta`, Q2 blocked, `control_repairs=3`). The runtime CANNOT
    infer resolution from an unusable finish without a false RESOLVED, so the bounded block is the
    honest outcome; there is no safe generic parser/controller seam. Kept as-is; containment pinned by
    `tests/test_analysis_control_repair_bound.py` (BLOCKED + exact reason + no false RESOLVED + bounded
    dispatch). Downstream, defects A/B and the existing transform-reacquisition suppression remove the
    redundant questions/actions that create these finish opportunities.
  Tests: `tests/test_analysis_namespace_indicator.py`, `tests/test_analysis_stage_invocation.py`, and
  a containment class in `tests/test_analysis_control_repair_bound.py`; causal mutation confirms A/B
  seams load-bearing. Independent review returned BLOCKER 1 / MAJOR 2 on the first pass (namespace
  over-match dropping real endpoints; contradiction false-positive on outer-hedge prose; invocation
  false-positive inside strings/comments) -- all fixed and regression-tested before merge. Frozen
  decode identity unchanged (mine.hta stage0 sha `0c6b4253cbd1eb8b`, C2-bearing stage sha
  `e2214909b2e7c671`, C2 `https://wall5tghf6fdg.api.opensourcesaas.org/ZOdcfNuo/myxwr5cli.bat`); no
  corpus sample other than mine.hta self-invokes, and namespace exclusion only removes XML/OOXML
  schema URIs that were never real IOCs.
  Live single-smoke on Ornith (CPU-only Dell) after the fix:
  `RC=0 elapsed=32.1s model_calls=2 actions=0 repairs=0 report=True stop="no open question requires
  an action"` -- the zero-action path (down from the pathological ~757s / 10-13 calls / 3-4 actions /
  2 completion failures). Report: XHTML namespace absent from Verified indicators; C2
  `https://wall5tghf6fdg.api.opensourcesaas.org/ZOdcfNuo/myxwr5cli.bat` present; the invocation fact
  `defines function ROmYsTcn and invokes it (ROmYsTcn;)` with the outer-container reach kept separate;
  no fabricated IOC, no false RESOLVED, no completion-state failure, no false contradiction banner.
  Record: `workdir/diag/mine_hta_ioc_grounding/live_smoke.json` (machine-local).
- RESOLVED (OFFICE-EXTRACTED-VBA-EVIDENCE-CLOSURE-1): the frozen Word sample
  (`99eb1d90…`, sha unchanged) regressed to 18 calls / 6 actions / 1287s with repeated
  UnicodeDecodeError (reading the raw binary .doc as UTF-8), a hallucinated
  `orbit_tools.get_evidence`, and a report leaving the Document_Open->payload chain unresolved.
  - **First causal divergence: O2 (extracted module present but not represented as actionable/readable
    source).** The Office preflight extracts `ThisDocument` (55039 chars, sha `d034bd8381f4663a`) as
    authoritative EVIDENCE and recognises the `Document_Open` autoexec entry, but `covered_source_text`
    and `delivered_source_text` are both None -- no source authority is tied to the extracted module.
    The sandbox `read_file` targets the raw binary artifact, so a "read the source" action decodes the
    binary as UTF-8 and fails; source-reacquisition suppression (gated on covered/delivered) cannot
    fire; and the static Document_Open->execution chain was never surfaced, so the model ran actions
    to establish it. The 234-char decoded PowerShell stage (sha `f1fa67e3`, C2 `http://185.189.58.222/x.exe`,
    `PHfW.exe`, Start-Process) is small and already restored evidence-first.
  - **Fix (all generic, no sample strings, no sandbox change, no budget change):**
    1. **Static execution-reach** (`analysis_vba_autoexec.find_office_execution_reach` +
       `OfficeExecutionReach`): for a recognised autoexec procedure, detect whether its OWN body
       (bounded by its `End Sub` OR the next procedure declaration, whichever is first) statically
       contains an execution sink (`Shell`/`.Run`/`ShellExecute`), outside comments/strings. Surfaced
       in the PLAN bootstrap and the report grounding (`office_events_appendix` -> `deterministic_sections`)
       as "Document_Open reaches a Shell execution call at line 750" -- a static reach, never a claim the
       macro ran or the document was opened, and (after review) NOT a claim the call's argument is the
       decoded stage.
    2. **Extracted-source dominance**: `_transform_reacquisition` generalised (via
       `_extracted_source_authorities`) to treat each extracted VBA module source as an authority, so a
       sandbox action that only reproduces the already-extracted source is suppressed (byte-exact or
       dominated) and not counted -- raw-binary reacquisition of source Orbit already holds is not the
       preferred path. Fail-closed guards intact: a FAILED read (the UnicodeDecodeError path, stderr
       non-empty) is never suppressed; the raw binary is never an authority, so legitimate binary
       inspection is untouched.
  - **Evidence/tool contract (section 3):** the `orbit_tools.read_evidence` stub already fail-closes
    with a message pointing at `evidence:<id>` conversation rehydration (no EvidenceStore in the
    sandbox). Added: `orbit_tools.read_file` no longer surfaces a bare `UnicodeDecodeError` on binary
    bytes -- it raises an actionable error saying the path is a binary artifact, not to re-read it as
    text, and that extracted source is reachable as `evidence:<id>` (bytes are never silently decoded;
    legitimate binary inspection is untouched). This is what a live smoke showed the model needed: the
    repeated raw-`.doc` UTF-8 read loop lands on the one move that makes progress.
  Tests: `tests/test_analysis_office_execution_reach.py`, `tests/test_analysis_office_source_dominance.py`;
  causal mutation confirms the reach detector, its body-boundary, and the office-source authority all
  load-bearing. Independent review returned BLOCKER 0 / MAJOR 2 first pass (cross-procedure false reach
  on colon-packed/unclosed entries; grounding over-claiming the sink argument is the decoded stage) --
  both fixed and regression-tested before merge. Frozen identities unchanged (module sha
  `d034bd8381f4663a`, stage sha `f1fa67e3…`, C2 `http://185.189.58.222/x.exe`, `PHfW.exe`, Start-Process).

## RC24 Tool-Loop Convergence

- Orbit now has one production tool loop. The former opt-in agent path,
  `--agent`, `--no-agent`, action-review model call, agent prompts, exact
  `apply_patch` tool, and mandatory agent verification state were removed before
  the first stable release.
- A process-isolated Gemma 4 26B-A4B Q4_0 comparison found no important,
  repeatable benefit unique to the second loop. After correcting one invalid
  repeated-action fixture, both modes completed 15/16 common scenarios. The
  normal loop failed code repair twice; agent mode passed once and failed once,
  so the apparent advantage was not repeatable.
- On the common cohort, the normal loop used 39 model calls, 18 proposed tool
  calls, 8,734 evaluated tokens, 927 output tokens, and 465.448 seconds. Agent
  mode used 61 model calls, 28 proposed tool calls, 52,277 evaluated tokens,
  1,688 output tokens, and 2,061.623 seconds. CPU wall time is descriptive, not
  a deterministic speed claim.
- Agent mode also proposed an unwanted mutation for inert tool-like JSON. The
  shared no-mutation policy prevented the filesystem change, but the proposal
  and extra calls were a correctness regression. The unique code-repair result
  did not survive a fresh-process repetition.
- Shared protections remain in the single loop: canonical validation,
  deterministic formal healing, exact repeated-call rejection, mutation
  epochs, permissions, lifecycle cleanup, and the explicit no-mutation policy
  from #155. A successful mutation reopens prior observations in the new epoch
  while the exact successful mutation remains blocked.
- The no-mutation classifier inspects only active latest-user prose. Quoted
  strings, code, JSON values, Markdown blockquotes, explicitly introduced data
  payloads, and tool output are not policy input. Global and unsupported mixed
  constraints deny generic shell before dispatch under canonical-gate ON or
  OFF; structured read-only tools remain available.
- Compatibility removal is intentional: Orbit is still in `0.0.1` release-
  candidate development and no stable contract included the agent flags. Old
  `--agent` and `--no-agent` invocations now fail as unknown CLI options rather
  than selecting a hidden compatibility path. JSON `agent` data has no runtime
  field or effect.
- The removed path does not migrate planning, review, hidden retries,
  decomposition, source-query experiments, or semantic decisions into the
  normal loop. Broad multi-deliverable source audits remain a model/template
  limitation.
- Post-removal validation completed 16/16 scenarios with a terminal `stop` and
  preserved every expected artifact. An intermediate removal accidentally
  dropped shared inert-data route guidance and caused one unwanted shell
  proposal; review restored the existing normal-loop guidance, and the targeted
  inert-JSON rerun used no tool and passed. A separate ten-case safety corpus
  preserved all artifacts; nine stopped normally, while one no-tool quoted-text
  explanation reached its output budget.
- Final focused regression tests passed 436/436 and full discovery passed
  1,251/1,251
  after obsolete agent/apply-patch suites were removed. The six MTP helpers
  rebuilt successfully. The 26B target, draft, and mmproj initialized together;
  strict MTP completed correctly. Final-prefix ON captured and restored
  `cached=64`; its kill switch retained `cached=4`. A process-isolated
  post-tool-reuse comparison preserved correctness and removed one model call
  and 529 evaluated tokens in the measured eligible case. These measurements
  are regression evidence, not deterministic performance claims.
- Twelve exact local prompts sampled from `docs/PROMPTS.md` initially produced
  11/12 strict passes. All eleven tool workflows passed. The original tools-off
  `grep` explanation reached the fixed 256-token chat-phase cap and ended
  incomplete. Process-isolated runs reproduced the exact same 256-token output
  and hash on RC23, pre-convergence main, and the convergence candidate, proving
  that this was not a convergence regression. The corpus now requests one
  concise sentence; the same three revisions produced the same complete
  26-token response with `finish_reason=stop`, one model call, and zero tools.
  No runtime budget or behavior changed.
- See `docs/TOOL_LOOP_CONVERGENCE_VALIDATION.md` for the inventory, comparison,
  removal decision, compatibility rationale, and release gates.

## Post-RC24 Chat Output Budget

- A post-RC24 fix removes the hidden 256-token cap from user-visible chat
  output. An explicit CLI, JSON, or REPL `max_tokens` value is now authoritative
  for chat responses, within the existing validated configuration range.
- Internal route, tool-call, retry, repair, and final-from-tool budgets keep
  their existing bounded caps. The change does not affect routing, tool
  selection, backend decoding, or continuation behavior.
- A real Gemma 4 26B-A4B Q4_0 smoke for `tell me about Google Deepmind` with
  tools enabled and `max_tokens=2048` used a five-token route decision, then a
  zero-tool 879-token chat final with `finish_reason=stop`. Before the fix, the
  final was truncated at 256 tokens and required `/continue`.
- Focused tests passed 266/266; full discovery passed 1,253/1,253;
  `compileall` and `git diff --check` passed. The longer response demonstrates
  correct budget control, not a performance improvement.

## Post-Tool Final Prose Reuse

- `ORBIT_POST_TOOL_FINAL_REUSE` is enabled by default. Setting it to `0` is the immediate kill switch; setting it to `1` explicitly enables it; invalid values disable the feature safely. Diagnostics report only the bounded effective state and source (`default` or `stable`).
- The optimization applies only after a terminal `post_tool_route` result: the model stopped normally, no new tool call or retry is present, the terminal tool result is complete, no error or guardrail is pending, and the original prose is non-empty, complete, and free of tool markup, control markup, or technical JSON. Any uncertainty falls back to the normal `final_from_tool` call.
- When eligible, runtime returns the exact original `ChatResult.content`; it does not construct, summarize, correct, or semantically reinterpret an answer. Tool selection, arguments, execution, evidence, finish reason, and correctness remain unchanged in the validated cases.
- Process-isolated Gemma 4 12B validation recorded 50/50 comparable OFF cases and 50/50 correct, stop ON reuses, with 50 model calls avoided and zero false positives or skipped tools. The measured aggregate saving was 65,189 evaluated tokens; the median per reuse was 1,084 evaluated tokens and approximately 107.5 seconds of wall time in that workload.
- Eligibility analysis overhead measured approximately 8.55 microseconds at p95. The wall-time result is workload-, output-, process-, and thermal-dependent; no deterministic speedup claim is made.
- Read, system, web, error, synthesis, multi-tool, cancellation, timeout, and reset cases retain normal fallback when the structural eligibility conditions are not met. This feature does not change routing, tool selection, executor behavior, MTP, final-prefix reuse, or retry policy.

## Native mtmd ABI and Vendor Provenance

- Python no longer exposes upstream mtmd structures. The mandatory co-located `liborbit-mtmd-bridge` accepts primitive values and opaque handles, constructs `mtmd_context_params` and `mtmd_input_text` from the compiled headers, and adapts reviewed bitmap return profiles.
- The bridge rejects unknown context, input-text, bitmap, or capability ABI layouts before mmproj initialization. Core ctypes structures passed by value are checked against bridge-reported `sizeof`, `alignof`, and relevant `offsetof` values.
- Bridge reuse is revision-bound. Its identity covers compiler/version, bridge flags, native CMake configuration, relevant source/header hashes, every co-located runtime-library hash, the bridge artifact hash, upstream provenance, and the Orbit patchset hash. Missing or mismatched identity fails explicitly.
- Current vendor provenance is upstream `b9551` at `379ac6673b5cd75c7b4e07d1521c50f1e093878c`, source-tree hash `4adb967e643363e7dc4d01d632b3a8471e0df2ec84ff304d364dc182f63e7ee1`, and Orbit patchset hash `dea2f205ed2a73d09ad203e08ba85545474742dbb0191f4f1a9b3a86beb4b435`.
- Native CMake builds receive vendor commit/build metadata explicitly. `LLAMA_COMMIT` must never be inferred from the enclosing Orbit repository.
- This hardening does not update the production llama.cpp revision. See `docs/NATIVE_MTMD_ABI.md`.

## Qwen 3.6 Native Compatibility

- Native Qwen support is restricted to the verified
  `Qwen3.6-35B-A3B-Q4_K_M.gguf` identity. The profile ID is
  `orbit-qwen36-native-v1`, the GGUF architecture is `qwen35moe`, and the
  embedded-template SHA-256 is
  `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`.
- Profile authorization uses GGUF architecture, model identity, tokenizer
  metadata, `general.file_type=15` (`Q4_K_M`), and the exact template hash.
  Filename matching alone never enables the profile. Other Qwen variants,
  templates, and quantizations fail before inference.
- A revision-bound native chat bridge applies the official embedded template,
  passes `enable_thinking` explicitly, and separates reasoning, visible
  content, and Qwen XML tool calls. Structured phases force thinking off;
  reasoning never reaches route or canonical tool parsers.
- Qwen route-prefix reuse is default-on only for the exact verified profile.
  It captures the complete hybrid sequence state at a 768-token batch-aligned
  boundary within the revision-bound invariant route prefix. The checkpoint is
  81,608,684 bytes and is model, quantization, context, template, tokenizer,
  schema, backend-build, and process bound.
- Cold, explicitly segmented, captured, and restored probes produced
  byte-identical logits with maximum absolute difference `0.0`. The measured
  11-route comparison reduced evaluated prompt tokens from 9,094 to 1,414 and
  restored 7,680 tokens with zero fallback and identical outputs, tools, and
  arguments. CPU timings are descriptive, not a universal speed guarantee.
- `ORBIT_QWEN_ROUTE_PREFIX_REUSE=0` is the immediate kill switch. Invalid
  values disable reuse safely. Cancel, timeout, reset, restart, context or
  identity changes, and restore errors invalidate the process-local blob.
- Qwen MTP and Qwen multimodal input are unsupported. Gemma rendering,
  checkpoints, MTP/mmproj, and tool behavior remain separate and unchanged.
- See `docs/QWEN_3_6_COMPATIBILITY.md` for the exact identity, protocol,
  diagnostics, validation evidence, and current limits.

## Qwen3-Coder Native Compatibility

- Native Qwen3-Coder support is restricted to the verified
  `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` identity. The profile ID is
  `orbit-qwen3-coder-native-v1`, the GGUF architecture is `qwen3moe`, and the
  embedded-template SHA-256 is
  `87710339d25b4e789c1d723f93c91ee861a86d305bb3d20a845536f251d6ea8a`.
- Authorization uses exact model, quantization, tokenizer, template, context,
  and expert metadata. Filename matching never enables the profile; drifted
  or unknown variants fail before inference.
- The official embedded ChatML and XML tool protocol provide chat, routing,
  tools, history/results, existing-file modification, and post-tool final
  behavior through the revision-bound native bridge. Thinking is unsupported
  and fails closed.
- Generative artifacts use a backend-owned, structurally pre-opened JSON string
  with a profile-local grammar. The backend reversibly decodes the generated
  value with strict UTF-8. It does not trim, normalize, retry, repair, or
  reinterpret content. Shared runtime publication and read-only verification
  remain unchanged.
- Independent review rejected the original in-band fence because artifact
  content ending in the same fence was ambiguous. The corrected profile uses a
  grammar-constrained JSON string and strict UTF-8 decoding. Its production
  first smoke passed 4/4 and the extended normal-tool-loop corpus passed 8/8,
  including artifact verification, existing-file modification, failure
  recovery, inert payload handling, and autonomous path/format choice.
- Measured CPU performance at context 8192 and six threads was approximately
  18.80 prefill tokens/s, 11.35 decode tokens/s, and 31.14 GiB peak RSS. Timing
  is descriptive and workload-dependent.
- Qwen3-Coder MTP, multimodal input, Qwen 3.6 route-prefix reuse, arbitrary
  exact-copy artifact guarantees, and empty artifacts are unsupported. Gemma
  and Qwen 3.6 profile identities and behavior remain separate.
- See `docs/QWEN3_CODER_COMPATIBILITY.md` and
  `docs/QWEN3_CODER_QUALIFICATION.md`.

## Post-RC28 Qwen3-Coder Route-Prefix Reuse

- The exact verified `orbit-qwen3-coder-native-v1` profile has its own
  default-on route checkpoint. It does not reuse Qwen 3.6 state, identity,
  template assumptions, or kill switch.
- The invariant route prefix is 789 tokens and the unpadded checkpoint boundary
  is token 768. The serialized sequence state is 75,507,864 bytes. Cold,
  segmented, and restored logits were byte-identical with maximum absolute
  difference `0.0` across chat, tool, failure, inert-data, and artifact routes.
- Warm matched routes restored `cached=768`; median route prefill changed from
  22.03 seconds to 1.44 seconds and median route wall time from 22.54 seconds
  to 1.97 seconds. Timing is descriptive and hardware-dependent.
- `ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE=0` is the immediate dedicated kill
  switch. Invalid values disable safely. Cancel, timeout, reset, reload,
  context or identity changes, restore failure, and restart invalidate or omit
  state. `/props` reports bounded state and identity diagnostics.
- Native tools-on startup captures this same checkpoint before server readiness.
  Qualification measured a 21.62-second capture; the first real greeting route
  restored `cached=768`, evaluated 32 dynamic tokens, and took 2.01 seconds,
  versus 800 evaluated tokens and 36.59 seconds without startup prewarm in that
  matched run. Timings are descriptive.
- `ORBIT_KV_PREFIX_PREWARM=off` disables startup capture but leaves lazy
  Qwen3-Coder reuse enabled: the first eligible route captures and later routes
  restore. Prewarm failures accept no partial state and fall back to that cold
  path without a startup retry. An operator SIGINT during Qwen3-Coder prewarm
  exits before the server binds instead of exposing the cold fallback. Qwen3.6
  and Gemma behavior remains unchanged.
- The verified Ornith profile has a second, separate prefix for ANALYSIS: 384
  tokens, its own `ornith15-analysis-prefix-v1` identity and its own checkpoint
  slot, derived from the ANALYSIS system contract and the `execute_analysis`
  schema alone. It is captured lazily by default -- the first analysis step of a
  server's life captures it, later sessions restore it -- and a rolling ANALYSIS
  checkpoint always outranks it.
- `ORBIT_ORNITH_ANALYSIS_PREFIX_PREWARM=1` additionally captures that prefix at
  startup, so the first analysis session restores 384 instead of paying cold
  prefill. It is off by default because it costs the startup that asks for it:
  a measured 11.0-second capture and a 73,736,684-byte resident checkpoint, paid
  by every server start whether or not an analysis ever runs. With it on, a
  measured first analysis step restored 384 of 588 prompt tokens and its prefill
  fell from 18.2 to 6.0 seconds. Timings are descriptive.
- `ORBIT_ORNITH_ANALYSIS_PREFIX_REUSE=0` disables Ornith ANALYSIS capture and
  restore entirely, eager and lazy alike. The two switches answer different
  questions -- whether the prefix may be reused at all, and who pays to capture
  it -- so asking for the eager capture cannot override the kill switch.
- `ORBIT_ANALYSIS_AUTONOMOUS=1` lets an analysis continue by itself while each
  step produces verifiably new evidence, instead of returning to the analyst
  after every step. It stops on natural completion, on stalled progress, on
  repeated failure, or at a hard bound of 12 actions, and every ending except a
  cancellation produces one grounded report. Off by default, and fail-closed:
  any value other than `1` reads as off.

  It is opt-in for cost, not for correctness. A step that measures something
  new is counted as progress whether or not the measurement was worth making, so
  an artifact with almost nothing to derive can still attract many actions.
  Whether that trade is worthwhile is an operator judgement rather than a
  default.

## Atomic Text Artifact Generation

- Non-trivial UTF-8 files use one model-selected `write_artifact` request with
  path, overwrite, and parent-creation arguments. File content is generated in
  one dedicated native phase without tools, shell, JSON, XML, or heredoc
  framing. The runtime never invents or repairs task content.
- Complete content is published only after `finish_reason=stop`, UTF-8 and
  64-KiB validation, stable path attestation, and an atomic same-filesystem
  commit. The exact published path is intrinsic to the pending capability; the
  model then selects one bounded read-only `verify_artifact` check without
  supplying another path. That ephemeral capability is absent from the normal
  tools-on registry. Evidence distinguishes overwrite authorization from the
  actual `publication_action` (`created` or `replaced`) used by final prose.
- Length, cancel, timeout, reset, path race, or generation error before commit
  publishes nothing. Verification never publishes or mutates. A verification
  failure leaves the atomically published file in place and is reported as
  published but unverified.
- Existing parents use an unnamed same-filesystem temporary file when
  supported. An unsupported anonymous open/link falls back to an exclusive
  private mode-`0600` file in the same directory; unsupported regular-file
  `RENAME_NOREPLACE` then uses an atomic no-replace hard link without weakening
  fsync, race checks, or post-publication attestation.
- The destination file is the atomic unit; shared parent trees are never moved
  for publication or rollback. Overwrite requires atomic exchange, and
  unsupported filesystems fail closed rather than using a pathname-check/
  ordinary-rename fallback with a TOCTOU window.
- `create_parents` explicitly authorizes descriptor-relative creation of
  missing directories. Cleanup attempts to remove only exact directories made
  by the request and only while empty; concurrent or pre-existing content is
  never moved or removed. The mutation epoch advances after file publication,
  while successful completion still requires model-selected verification.
- Named private files use process- and inode-bound recovery manifests. Startup
  removes only exact entries whose owner is positively proven inactive;
  active or unknown owners, symlinks, malformed state, changed identities,
  linked files, and other ambiguity are preserved. Age is diagnostic only and
  never authorizes deletion. Absolute zero
  residue after power loss is not claimed because narrow pre-registration and
  publication crash windows cannot be cleaned without risking user data.
  Explicitly created empty parent directories may likewise remain after an
  uncatchable crash when later ownership cannot be proved safely.
- The dedicated 4,096-token content budget does not change route, normal tool,
  chat, or final budgets. One file per request and native backend only are the
  initial bounds. Semantic chunking, hidden retries, deterministic content,
  and incomplete-envelope repair remain prohibited.
- See `docs/ARTIFACT_GENERATION.md` for protocol, atomicity, lifecycle,
  validation evidence, and measured CPU limits.

## MTP

- MTP is optional and experimental.
- It is not the quick-start default.
- It does not guarantee speedup, especially on CPU-only systems.
- Download the draft MTP model only when intentionally testing MTP.
- `n_max=3` remains the best default among observed experiments.
- `target_validate` is compute-bound; graph compute is the dominant cost.
- Two-pass validate and shadow runtime were rejected for correctness risk: KV mutation, speculative state that cannot be cloned safely, and sensitive sampler/KV/frontier cleanup.
- MTP strict smoke and timeout/cancel recovery remain required gates when validating the MTP path.
- Local MTP validation should use MTP enabled, mmproj, and multimodal availability when validating that path.

## KV / Cache / Final Budget

- Eligible native `final_from_tool` calls now use an exact batch-aligned 64-token checkpoint by default. The first eligible final captures; later eligible finals restore `cached=64`.
- `ORBIT_FINAL_PREFIX_REUSE=0` disables reuse immediately. The measured kill-switch path retains expected non-reuse behavior with `cached=4`.
- The former 43-token segmentation remains historical and must not be restored. Its checkpoint implementation was correct, but the non-aligned boundary changed Gemma logits and caused the reproduced read regression.
- Do not generalize the checkpoint beyond the exact validated final prompt family or into route, chat, retry, tool-call, thinking, tools-off, or MTP-owned paths.
- Small `final_from_tool` was improved with compact evidence metadata.
- `system_info` has a dedicated 160-token cap.
- Small `shell`, `grep_search`, and `unknown` finals remain at 96 tokens.
- `/max-tokens` is user-facing; the runtime still applies internal per-phase budgets.

## Tool-Call Early-Stop Shadow Technical Stop

- Measured fact: a one-time benchmark-only streaming scanner observed the
  production Gemma tool envelope, reused active raw normalization, and applied
  the shared canonical contract. It never cancelled generation, executed a
  tool, or started a final. The scanner, harness branch, and tests were removed
  after the technical stop; no early-stop module or flag remains.
- Measured fact: in 11/11 evaluable production-like scenarios, eight canonical
  completion points were observed. Every point was the final generated token:
  median and p95 trailing tokens were both zero, and theoretical avoidable
  decode time was zero. No false completion was observed.
- Conclusion: the production decode loop samples EOG immediately after the
  closed `<tool_call|>` envelope and exits before decoding or counting EOG.
  Active early stopping therefore has no measured token or wall-time
  opportunity.
- Reopen only for a new model or template that naturally emits at
  least two trailing tokens in at least 20% of
  representative canonical-valid calls, with measurable decode cost and zero
  adversarial false completion. See `docs/TOOL_CALL_EARLY_STOP_SHADOW.md`.

## ngram-mod Technical Stop

- Measured fact: the current vendor's server/common implementation is not
  directly usable by Orbit's production one-token Python decode loop. Safe
  integration would need batch target validation, sampler cloning, KV rollback,
  and checkpoint state that the production path does not expose.
- Measured fact: a staging probe benefited a highly repetitive copy workload,
  but non-repetitive medium/long controls proposed no drafts and showed no gain
  outside variability; the draft path also retained additional memory.
- Conclusion: observed usefulness was limited to a highly repetitive sequence
  and did not generalize. No ngram-mod runtime module, flag, or decode path is
  active.
- Reopen the investigation only if upstream provides a stable C API for the
  required speculative operations, or production-like evidence shows frequent,
  repeatable, and materially beneficial repetitive workloads. Promotion would
  still require revision-matched integration, exact greedy output equivalence,
  process-isolated ABBA validation, bounded memory, and complete
  lifecycle/MTP/prefix-reuse validation.

## Discarded Optimization Experiments

- The post-reuse model-call audit found no additional frequent model call that
  could be removed without taking over a semantic decision. Exact final-input
  replay remains unimplemented and stopped pending repeated real opportunities.
- Additive structured `read_file`/`grep_search` schemas increased cold prefill
  and regressed exact argument fidelity. Their schemas, harness branches, and
  tests are not present in production.
- Production tool-schema compaction changed tool selection or exact arguments;
  generic output bounding lacked supporting oversized evidence. Both remain
  technical stops.
- Bounded planning failed its semantic smoke gate with Gemma 4 12B and remains
  inactive. It requires a new model/template to pass exact-plan,
  unsupported-plan, and zero-wrong-plan gates before reconsideration.
- A 768-token tool-mode checkpoint was numerically exact but the
  production-like opportunity sample observed no useful real restore. It also
  retained about 252 MiB. Its runtime integration and
  `ORBIT_TOOL_PREFIX_REUSE` flag were discarded after RC23.
- Preserve only the measured audits in
  `docs/POST_TOOL_MODEL_CALL_AUDIT.md`,
  `docs/STRUCTURED_FILE_TOOLS_SHADOW.md`, and
  `docs/TOOL_CALL_POST_ROUTE_KV_REUSE.md`. Do not restore their experimental
  code without new production evidence and a separate review.

## Evidence Lineage

- `user_turn_id` is useful for provenance, not relevance.
- `tool_call_id` and `evidence_sequence` are useful but insufficient for selection.
- `produced_by_phase` is populated only for known paths.
- `producer_model_call_id` remains `null`.
- The model-guided shadow evidence-selection experiment was negative: the extra model call was too expensive on CPU-only, JSON was unreliable, and `dual_shell` was fragile. The patch was reverted; do not use it now.
- Multi-card `chat_final` compaction remains a technical stop.
- `dual_shell` may require both cards in retry/final; do not reduce without stronger lineage/intent.

## Benchmarking

- `orbit bench-core` is the public regression benchmark.
- The `bench-core` metadata header is ON by default.
- Use `--no-metadata` only when minimal output is needed.
- Metadata includes commit/tag, `base_url`, `workdir`, timeout, `max_tokens`, selected env vars, and best-effort backend `/props`.
- If `/props` does not respond, `backend_props: unavailable` must not fail the benchmark.
- Always record commit/tag, model, ctx, threads, MTP, tools, and prewarm.
- `scripts/suggest-server-profile.sh` is a conservative starting point, not a guarantee of optimal tuning.
- **Since `47d6b4b`, record the RESOLVED profile, not the flags you passed.** Any
  of `threads`/`threads_batch`/`batch`/`ubatch` you leave unspecified is filled
  from a cached measurement, a calibration sweep or the heuristic — so two runs
  launched with identical command lines on the same box can run different
  profiles (the first start sweeps, later starts read the cache). Only the
  thread counts are ever MEASURED; `batch`/`ubatch` are fixed before the context
  a benchmark would need, so they come from cache or heuristic.
  `orbit server --show-profile` prints the resolution without loading a model;
  pin every field explicitly for a controlled benchmark.
- GPU must be measured through an external compatible backend, for example `llama-server --base-url`, not as native `orbit server` performance.
- Native `orbit server` is CPU-first with `gpu_layers=0`.

## Server Startup Profile (auto-calibration)

`orbit server` no longer ships one set of tuning numbers for every machine. It
resolves `threads`, `threads_batch`, `batch`, `ubatch` and `cache_ram` through a
precedence chain, measuring only what nobody supplied:

    explicit CLI > explicit env > stored user profile (reserved, no loader yet)
      > cached auto-calibrated profile > bounded auto-calibration > heuristic

Each level fills only fields still unset, so `--threads 8` alone leaves the rest
resolvable. **An explicitly supplied value is never measured, never cached over,
and never reported as auto.** Source: `src/orbit/native_server/server_profile.py`
(policy, pure) and `server_calibration.py` (the sweep).

- Commands: `orbit server --show-profile` resolves and prints without loading a
  model or measuring; `orbit server --recalibrate` discards the stored
  measurement and takes it again. A preview never deletes anything.
- Env: `ORBIT_THREADS`, `ORBIT_THREADS_BATCH`, `ORBIT_BATCH`, `ORBIT_UBATCH`,
  `ORBIT_CACHE_RAM`. `ORBIT_`-prefixed on purpose — the old suggest script
  printed bare `export THREADS=…` for a server that read no environment at all,
  so its advice had no effect. `scripts/suggest-server-profile.sh` is now a thin
  wrapper over the same Python heuristic; there is one source of truth.
- **`cache_ram` is advisory.** It is computed conservatively and reported, but
  Orbit's native server has no cache-RAM knob to consume it — `--cache-ram` is a
  `llama-server` option. Treat it as advice for an external llama-server.
- Cache: `~/.cache/orbit/server-profiles/<fingerprint>.json`, outside the
  repository because a measurement describes one machine. It records **only
  measured fields**, never the resolved profile — otherwise a one-off
  `--batch 1024` would become a permanent "measurement" and a heuristic
  improvement could never reach a machine that had calibrated. Fingerprint
  covers CPU model, core counts, RAM class, model identity, backend revision,
  ctx, and the `low_memory` / MTP / expert-usage flags; a stale
  `format_version` entry is unlinked on read.
- Calibration is bounded: candidates derived from topology, one pass each, a 90 s
  ceiling, KV cleared before and after every candidate, and it runs after the
  model loads but BEFORE the route-prefix prewarm and before the socket binds —
  so it measures inference, never a checkpoint restore. Any failure (raise,
  timeout, every candidate rejected, unwritable cache) falls back to the
  heuristic and the server still starts. **Nothing auto-enables MTP, GPU, a
  different ctx, quantization or model.**

**Measured on the Dell (16-core hybrid), and the reason this measures rather
than counts:** prefill rises 37 → 68 tok/s from 6 to 16 threads while decode
FALLS 19 → 5.4. Scored against the real cache-restored turn (210 evaluated + 60
generated) the winner is **threads=6** — the existing qualified value — and 16
threads would roughly double turn latency. Scoring the full 978-token prompt
instead would pick 12–16 and regress the machine; that constant is the single
most consequential line in the calibrator. Candidate table:
`workdir/diag/autocalibration/`.

Calibration costs ~47 s once per fingerprint (first start 120 s vs 81 s cached,
on top of a ~24 s prewarm and ~46 s model load); with the warm-up below it is
~75 s and a `--recalibrate` start measured 156 s to healthy. Concurrent server
starts each measure the other's contention — start them sequentially.

**Warm-up and tie-break (DELL-RUNTIME-AND-IBAN-CLOSURE-1).** The sweep first runs
one untimed pass with the first candidate, because the first measurement after
load pays the mmap page-in (17,957-22,534 major faults on the Dell, 25 tok/s
prefill against 41 warm) and would otherwise hand the win to whichever candidate
ran second — which is exactly how a fresh boot cached 8 threads. The row is kept
in the table as `rejected: "warmup"`, scores 0 and counts against the budget. The
winner is then the FEWEST threads among candidates within `TIE_MARGIN` (10 %) of
the top score: warm, 6 and 8 scored within 1-8 % on the Dell while a served turn
was 2× slower at 8, and repeat noise is ~5 %. Read `/props.native_threads` — not
`threads` — when you need to know what the context is running.

## Recommended Gates

- Pre-PR: targeted unit tests for the modified area.
- Always: targeted `compileall` and `git diff --check`.
- Full unit only for pre-release or broad changes.
- If touching budgets/final behavior: smoke `system_info`.
- If touching `bench_core`: smoke the metadata header and `--no-metadata`.
- Evidence lineage smoke: use clean temporary workdirs.
- KV/final smoke: `pwd_followup`.
- MTP gate: `simple_chat --mtp-required` with healthy `/props`.
- Recovery gate: timeout/cancel with `shell20`, then a new `simple_chat --mtp-required`.
- Never use a persistent store for RC evidence-lineage smokes.

## Git & Qualification Workflow (as used by the ANALYSIS arc, #260 → rc38)

The workflow every recent production mission followed, and the one a new session
should follow for any tracked change:

1. Baseline: `main == origin/main`, tracked tree clean. Branch off main
   (`feat/…`, `fix/…`, or `docs/…`); never commit production code straight to main.
2. Qualify on the branch BEFORE merge:
   - focused unit tests for the changed surface, then the full suite
     (`TMPDIR=/tmp PYTHONPATH=src python3 -m unittest discover -s tests -q` —
     unittest, never pytest) with a REAL child return code, never one read from a
     shell pipeline (chunk it if the box's memory-pressure heuristic kills a
     single huge run; 5462 tests / 0 failed at rc38 and 5622 after
     DELL-RUNTIME-AND-IBAN-CLOSURE-1, skips host-dependent: 8 on the NUC, 77 on
     the Dell);
   - a mutation gate for any new deterministic capability — each mutant APPLIED
     and CAUGHT on an isolated tree copy (see the transform/OLE/autoexec/kind
     gates under `workdir/diag/*/mutation_gate.py`);
   - `compileall` and `git diff --check`.
3. Independent adversarial review (a fresh agent that reads the diff/tests, not
   the author's summary): require **BLOCKER 0, MAJOR 0**. Fix in-scope findings
   autonomously and re-review the delta until clean. A real production
   correctness/safety defect in review → stop, do not paper over it.
4. Freeze the reviewed commit SHA. Push the branch; verify the remote branch SHA
   equals the reviewed SHA.
5. Squash-merge to main from the reviewed content; verify the merged tree is
   byte-identical to the reviewed tree (per-file sha compare, as the ANALYSIS
   missions did). Commit source-ONLY — never stage `workdir/` scratch; the malware
   corpus and diagnostics stay untracked. End commit messages with the session
   attribution line when one is provided.
6. Push main; verify `main == origin/main`. Delete the branch (local + remote).
   Preserve diagnostics/oracles under `workdir/diag/`.
7. Releases are RC tags (`v0.0.1-rcNN`, annotated, message `Orbit v0.0.1-rcNN`)
   pointing at the doc/release commit; the GitHub release is a pre-release whose
   body is the verbatim `docs/releases/v0.0.1-rcNN.md`, with NO attached assets.
   Build any verification sdist from a CLEAN `git archive` checkout, never the
   dirty working tree (untracked malware would otherwise be swept in — `MANIFEST.in`
   is now hardened with `prune workdir` + explicit benign includes as a backstop).
   Do not tag/release unless the mission explicitly authorises it.

Live model runs (analysis qualification, benchmarks): one fresh session per
sample via `scripts/live_validate_analysis.py` (controller metrics) or the
KV-instrumented `workdir/diag/multisample_bench/bench_one.py` (adds a per-call
KV trace — the live validation script does NOT install `kv_diag.instrument_backend`).
Kill any leftover `until … do :; done` busy-wait loops before benchmarking (they
spin at ~37% CPU each and contaminate timings). Absolute wall time on a contended
box is not a comparable number; token/cache/rate and correctness are.

## Bounded Multi-Action Planning Shadow

- Bounded multi-action planning is observational only. Production routing,
  tool execution, and finalization remain unchanged.
- `ORBIT_TOOL_PLAN_SHADOW` is OFF by default. The dedicated smoke-harness mode
  performs one real planning inference, validates exactly two literal linear
  steps, and never executes a tool or starts post-tool finalization.
- The initial shadow allowlist is `system_info` and `list_directory`. Network,
  mutation, and unrestricted shell actions are excluded, including shell calls
  that appear read-only but can access external resources or paths.
- The only alternate response is exact `{"type":"unsupported_plan"}`. Model
  IDs, expectations, dependencies, branches, parallelism, and completion choice
  are not part of the minimal schema.
- Every proposed step reuses the canonical tool contract. The analyzer rejects
  multiple plans, duplicate keys, dynamic step references, external text,
  truncation, disabled tools, schema/policy/permission failures, and operational
  limit failures.
- Structural literal arguments do not prove semantic independence from prior
  step output. This is the main technical stop before any opt-in execution.
- Three compact prompt views were tested on nine scenarios each. The best view
  reached 88.9% JSON compliance, 50% exact positive plans, 100% unsupported
  accuracy, and an 11.1% wrong-plan rate. It failed the smoke gate, so five
  repetitions were not run and no model-call reduction is credited.
- See `docs/BOUNDED_TOOL_PLANNING.md` for the minimal schema, prompt comparison,
  token/wall measurements, and closure decision.

## #124, Conversation Reuse Route Guidance

- Released in `v0.0.1-rc17`.
- Problem: the router could call tools again for recaps, summaries, repetitions, or continuations of information already present in the conversation.
- Solution: add one general, model-guided rule only in `ROUTE_SYSTEM_PROMPT`.
- The rule prefers `CHAT` when the user asks to recap/summarize/repeat/continue/explain/compare and existing context is sufficient.
- Tools remain allowed for fresh/current, verify/check, new information, changed file/state, or missing/stale/ambiguous/insufficient context.
- Files touched: `src/orbit/runtime/messages.py`, `tests/test_messages.py`.
- Tests run: `PYTHONPATH=src python3 -m unittest tests.test_messages -q`, `python3 -m compileall -q src/orbit/runtime tests`, `git diff --check`.
- Post-merge route-level smoke: `system_info` recap and read-file recap confirmed `CHAT` / no tool when context is sufficient; refresh/current/check changed/new search still allow tools.
- Smoke limitation: grep recap was only partially confirmed because it ended with `finish_reason=length` and empty output.
- Full E2E is not a lightweight gate on this CPU: A1 `system_info` took about 220s and A2 full E2E was interrupted.
- No regression was observed and no further patch is required.
- Residual limit: this is routing guidance, not a deterministic guarantee; it adds no cache, TTL, fast path, or tool-specific logic.
- Status: closed work. Do not add more conversation-reuse patches without an observed regression.

## #128, Compact Final View for Web Search Errors

- Included in `v0.0.1-rc18`.
- Problem: `web_search` tool errors correctly closed through `final_from_tool`, but could miss the compact web final view and prefill a larger final prompt.
- Solution: `web_search` evidence with `status=error` now uses the compact web final view.
- The final context carries bounded metadata, including query, status, `error_message`, raw ref/hash, and size.
- Full raw web error/output is not reinjected into the compact final prompt.
- `error:` detection is scoped to `web_search` evidence only; generic non-web tool errors are unchanged.
- Files touched: `src/orbit/runtime/chat.py`, `src/orbit/runtime/evidence.py`, `tests/test_evidence.py`, `tests/test_runtime.py`.
- Tests run: targeted evidence/runtime tests PASS with 37 tests, runtime/evidence/tool_message PASS with 198 tests, messages/final_policy/completion_budget PASS with 58 tests, `compileall` PASS, `git diff --check` PASS.
- Full unit discovery previously passed with 988 tests.
- Safety preserved: no route/tool-loop changes, no MTP changes, no cache/KV changes, no global budget changes.

## #130, Web Search Error Final Correctness

- Included in `v0.0.1-rc18`.
- Completes the compact web error final behavior introduced by #128.
- Problem: after #128, a known-query `web_search` error could still lead the final model call to answer from general knowledge as if the search had succeeded.
- Solution: `web_search` evidence with `status=error` now adds `web_search_failed: true` and a narrow final instruction to report the web failure briefly and not answer from general knowledge as if the search succeeded.
- Files touched: `src/orbit/runtime/evidence.py`, `tests/test_evidence.py`, `tests/test_runtime.py`.
- Tests run: `tests.test_evidence tests.test_runtime` PASS with 193 tests, `tests.test_messages tests.test_final_policy tests.test_completion_budget` PASS with 58 tests, `compileall` PASS, `git diff --check` PASS.
- Safety preserved: scoped only to `web_search` with `status=error`; `status=none`, successful web results, and non-web errors are unchanged.
- Full raw web error/output is not reinjected.
- No route/tool-loop changes, no MTP changes, no cache/KV changes, and no global budget changes.

## #132, Reduced final_from_tool Prompt Tokens

- Included in `v0.0.1-rc18`.
- Problem: every `final_from_tool` call evaluated a correct but unnecessarily verbose dedicated system instruction.
- Solution: compact equivalent wording preserves the full contract: answer concisely from tool evidence, do not call tools, do not expose raw tool-call syntax, do not falsely claim lack of access, and report errors briefly.
- Production-tokenizer measurement: the `final_from_tool` system component decreased from 49 to 34 tokens, an exact deterministic reduction of 15 tokens per call.
- No deterministic wall-time improvement is claimed because observed CPU timings were noisy.
- Files touched: `src/orbit/runtime/messages.py`, `tests/test_messages.py`.
- Tests run: messages/final policy/completion budget PASS with 59 tests, evidence/runtime PASS with 193 tests, `compileall` PASS, and `git diff --check` PASS. Full unit discovery previously passed with 989 tests.
- Safety preserved: no route/tool-loop changes, no evidence-selection changes, no MTP changes, no cache/KV changes, and no completion-budget changes.

## #134, Reduced Compact Evidence Prompt Metadata

- Included in `v0.0.1-rc18`.
- Problem: compact model-facing evidence cards still included audit-only provenance fields that were retained elsewhere and were not needed to answer.
- Solution: small compact cards no longer expose `raw_ref`; compact web cards no longer expose `tool`, `raw_ref`, hash, or size.
- Full and medium cards are unchanged. EvidenceStore, raw retrieval, sidecars, route cards, tool messages, evidence identity, hashes, and lineage remain intact outside the model prompt.
- The historical #128 compact web view included raw ref/hash/size; #134 removes those fields only from its model-facing projection without changing #128/#130 web-error correctness.
- `kv_diag_evidence_card_tokens.evidence_id_hash` may be `null` for compact cards without `raw_ref`. This is intentional: `kv_diag_evidence_lineage` independently preserves the hashed `EvidenceRecord.evidence_id`.
- Measured prompt reductions: `system_info` 36 tokens, `read_file` 37, `grep_search` 36, `list_files` 37, `shell_error` 35, `web_none` 72, `web_error` 74, and `web_success` 77.
- Tests run: evidence/runtime/tool-message PASS with 198 tests, messages/final-policy/completion-budget PASS with 59 tests, `compileall` PASS, and `git diff --check` PASS.
- Safety preserved: no route/tool-loop, evidence-selection, MTP, cache/KV, segmentation, completion-budget, system-prompt, or raw-retrieval changes.
- This reduces evaluated dynamic-suffix tokens. It does not fix cache reuse: `cached=4` remains expected from route/final prompt-view divergence.

## #137, Experimental final_from_tool Prefix Reuse

- Included in `v0.0.1-rc19`; not included in `v0.0.1-rc18`.
- Adds an off-by-default experimental path behind `ORBIT_FINAL_PREFIX_EXPERIMENT=1`.
- Eligibility (when enabled):
  - native backend,
  - `tools` enabled,
  - exact `final_from_tool` prompt family and role sequence,
  - exact `FINAL_FROM_TOOL_SYSTEM_PROMPT` alignment,
  - exact validated 43-token prefix,
  - MTP experimental path is not active.
- Default behavior remains unchanged:
  - `cached=4` remains for route/final in production default.
  - normal final path behavior is unchanged when the flag is off.
- Experimental behavior when enabled:
  - eligible final calls can restore validated 43-token prefix;
  - cached tokens reach 43 for eligible finals when active;
  - this is an exact 39 evaluated-token reduction relative to disabled default behavior.
  - this is additive to prior final prompt reductions.
- Safety and lifecycle:
  - mismatch or restore failure falls back to normal prefill,
  - failed restore cannot keep initialized state,
  - cancel, reset, and completion errors invalidate checkpoint state,
  - route, chat, retry/repair, and tool-call phases are not eligible.
- Observability:
  - `/props` reports bounded experiment diagnostics: `enabled`, `initialized`, `prefix_tokens`, `capture_count`, `restore_count`, `fallback_count`, `failure_reason`, `last_used`, and checkpoint size.
- Validation recorded:
  - OFF/ON matrix: 66 finals, `finish_reason=stop` in 66/66.
  - 50 completion mixed stability run: 50/50 correct.
  - 0 fallbacks.
  - cancel invalidation and safe recapture: PASS.
  - full suite PASS with 997 tests; compileall PASS; `git diff --check` PASS.
- Limitations:
  - not enabled by default,
  - experimental logits differ from cold full-prefill due to segmentation differences,
  - restore is bit-exact against an identically segmented baseline,
  - no deterministic wall-time claim is made.

## #139, Repeatable Final Prefix Benchmark Coverage

- Included in `v0.0.1-rc19`; not included in `v0.0.1-rc18`.
- Extends `scripts/orbit_smoke_harness.py` with managed, repeatable OFF/ON validation for `ORBIT_FINAL_PREFIX_EXPERIMENT=1`; it does not change the reuse mechanism or enable it by default.
- Harness coverage includes managed native-server startup, explicit flag propagation to both server and runtime client, deterministic web success/none/error fixtures, capture/restore/fallback counters, and additive JSONL summaries.
- Recorded comparison metadata includes route/final/non-model/total timing, output tokens, run and block order, CPU affinity, managed process identity, and bounded `/props` snapshots.
- Tools-off validation is truthful on both sides: the managed server is configured with tools disabled, the runtime receives no allowed tools, normal chat remains available, and final-prefix capture/restore remain zero.
- First-class lifecycle coverage includes server restart, context change, thinking-mode eligibility, cancel and timeout cleanup, invalidation and recapture, and the MTP guard. The experiment remains ineligible while MTP is active.
- Ordered RSS records cover startup, capture, restores 10/25/50, invalidation, and recapture with PID and block identity. The measured run showed no linear-growth pattern; this is diagnostic evidence, not a general no-leak guarantee.
- Controlled OFF/ON results confirm default `cached=4`, eligible experimental `cached=43`, and an exact net reduction of 39 evaluated tokens. Matched-output cases showed lower final latency, but no deterministic wall-time improvement is claimed.
- Validation recorded: OFF and ON correctness remained stable, web errors did not answer from model memory, no stale evidence or cross-turn contamination was observed, and full unit discovery passed with 1,022 tests; `compileall` and `git diff --check` passed.
- RC19 includes the experiment only as an optional, OFF-by-default feature with its benchmark harness; it is not promoted to default behavior.

## #147, Aligned 64-Token final_from_tool Prefix Reuse

- Included in `v0.0.1-rc20`; not included in `v0.0.1-rc19`.
- The experimental 43-token checkpoint boundary was not safe for default use because it split production prefill before the normal 64-token batch boundary. Gemma output could therefore diverge even though checkpoint capture and restore were correct; the reproduced read fixture expanded from a concise 13-token stop response to a 96-token length termination.
- The final system instruction is now a meaningful 58-content-token policy. With six Gemma template/control tokens, the stable prefix is exactly 64 tokens, ends at the system turn, and contains no padding or filler. The next dynamic token is the user-turn control token 105.
- Stable-prefix text hash: `c3b8e45ac695a87e60146bb8017a98f1b41fc13a708565b58160cce6d419c6f3`. Serialized token-prefix hash: `398338fd38a9c80d54b269e09ae70077ab7323ec1a47920879a24896928cdfc5`.
- Cold production prefill, explicit segmentation at token 64, and checkpoint restore were bit-exact in 55/55 probes across 11 final families: logits hashes, next token, ordered top-10, bounded output, and finish reason matched, with maximum logits difference `0.0`.
- The read regression is resolved: aligned capture and restore both return the same 13-token `finish_reason=stop` answer as cold production. Checkpoints created for the old 43-token identity are rejected and rebuilt.
- Eligible native `final_from_tool` reuse is enabled by default. The first eligible final captures the checkpoint; later eligible finals restore `cached=64`.
- `ORBIT_FINAL_PREFIX_REUSE=0` is the immediate stable kill switch. The stable variable overrides `ORBIT_FINAL_PREFIX_EXPERIMENT`; legacy-only configurations remain compatible, and invalid stable values disable reuse safely with bounded diagnostics.
- Eligibility remains restricted to native `final_from_tool` calls with tools enabled and the exact prompt family. Tools-off, thinking-enabled, MTP-owned, route, chat, tool-call, retry, and repair paths do not use the checkpoint.
- A restored final evaluates 60 fewer tokens than the same new prompt without reuse and 36 fewer tokens than previous production. The first capture costs 28 additional evaluated tokens, so cumulative evaluated-token break-even occurs on the second eligible final.
- Semantic validation passed 40/40 high-information cases with `finish_reason=stop`; output tokens decreased from 845 to 825 without losing requested facts, paths, counts, matches, errors, or values.
- Lifecycle validation recorded 900 restores and 906/906 correct stop completions, zero normal fallbacks, bounded non-linear RSS, safe cancel/timeout/reset/restart invalidation, and no stale or cross-process checkpoint state.
- Post-merge validation on `6330d85` passed six default-reuse tool finals with one capture, five `cached=64` restores, and zero fallback; stable OFF passed two `cached=4` finals with zero capture/restore; strict MTP remained healthy with zero final-prefix activity.
- Validation: prompt/final-policy/completion-budget PASS with 60 tests; evidence/runtime/tool-message PASS with 213 tests; resolver PASS with 3 tests; backend/native/protocol PASS with 118 tests; smoke harness PASS with 54 tests; MTP shim build PASS; full discovery PASS with 1,067 tests; `compileall` PASS; `git diff --check` PASS.
- The evaluated-token reduction is deterministic for eligible restored calls. CPU wall time remains workload-, output-, process-, and thermal-dependent; no deterministic wall-time improvement is claimed.

## #142, Structurally Covered CHAT Evidence Omission

- Included in `v0.0.1-rc20`; not included in `v0.0.1-rc19`.
- Problem: conversation reuse could correctly choose `CHAT` and avoid another tool call, but the CHAT prompt still reinjected large hidden evidence contexts already represented by completed visible assistant answers.
- Solution: the CHAT model projection may omit hidden evidence only when structural coverage is proven. This is structural redundancy handling, not semantic relevance selection; stored evidence and provenance remain unchanged.
- Coverage requires a live-turn evidence sequence and associated user turn, exact evidence IDs, matching tool messages, and a later non-empty visible assistant final with `finish_reason=stop` before another user message. Every evidence record in the projected window must pass, live history and lineage must remain consistent, and the visible projection must be smaller than the existing evidence projection.
- The visible projection preserves bounded visible user and assistant messages in order, includes the current user request exactly once, and excludes hidden tool syntax. It does not modify stored conversation history.
- Conservative fallback retains the existing evidence view after reload, memory compaction, reset, rollback, history truncation, missing lineage, empty/failed/cancelled/length finals, inconsistent ordering, uncertain evidence association, or a non-smaller visible projection.
- The visible CHAT policy preserves concrete facts such as paths, filenames, counts, errors, and matched values. If a requested detail is absent from visible answers, it reports that the detail is unavailable in the visible conversation and does not infer omitted context.
- Measured system recap result: prompt tokens decreased from 841 to 279 and evaluated tokens from 837 to 275, an exact deterministic reduction of 562 evaluated tokens for that measured case. The recap remained `CHAT`, made no repeated tool call, returned a correct answer, and finished with `finish_reason=stop`.
- No deterministic wall-time improvement is claimed. CPU timing remains dependent on output length, process state, and thermal conditions.
- Validation: messages/final-policy/completion-budget PASS with 60 tests; runtime/evidence/tool-message PASS with 213 tests; full discovery PASS with 1,038 tests; `compileall` PASS; `git diff --check` PASS.
- Safety preserved: no route or tool-loop behavior changes, no evidence-store changes, no semantic evidence selection, no MTP or KV/cache changes, no final-prefix changes, and no completion-budget changes.
- General evidence selection remains closed; do not extend this structural projection into a relevance selector without a new reliable signal and separate evidence.

## #143, Diagnostic Route Output Classification

- Included in `v0.0.1-rc20`; not included in `v0.0.1-rc19`.
- Adds five mutually exclusive diagnostic classes for completed route output: `canonical`, `legacy_tolerated`, `direct_prose`, `malformed`, and `control_loop`.
- `canonical` is restricted to one strict parser-accepted JSON object with the canonical field shape. `legacy_tolerated` covers only non-canonical forms already accepted by the existing parser through normalization. `direct_prose` is limited to the existing intentional direct-answer branch. Rejected output remains `malformed` unless it meets the bounded control-loop diagnostic conditions.
- Classification occurs only after the existing parser and direct-answer handling. It does not change parser results, routing, fallback, tool selection, direct prose, or model-call count. Initial and retry route completions are classified independently.
- Diagnostics are additive and bounded: class, static reason, parser-accepted flag, finish reason, and output-token count when available. They do not store raw route text, user requests, evidence content, or mutable aggregate counters.
- Production `control_loop` classification may use the `empty_visible_control_output` surrogate only when visible route output is empty, `finish_reason=length`, and at least 8 completion tokens were generated. This is conservative diagnostic evidence, not proof that the exact raw control-token cycle occurred; empty stop, error, or cancelled output remains `malformed`.
- Validation: `tests.test_command_request` PASS with 57 tests; `tests.test_kv_diag` PASS with 30 tests; messages/runtime/evidence/tool-message PASS with 222 tests; full discovery PASS with 1,049 tests; `compileall` PASS; `git diff --check` PASS.
- This is observability, not a route fix. The next step is to measure class frequencies through the existing benchmark or smoke harness before considering any behavior change. Do not reopen grammar integration or route-contract redesign from these diagnostics alone.

## Route Generation Technical Stop

- The repeated route control-token loop is classified as prompt/model instability. Cold and warm KV, prompt-cache reuse, prefill segmentation, sampler reset, streaming and non-streaming collection, final stream flushing, and the 64-token route budget were excluded as root causes.
- Local prompt edits, role reordering, evidence placement changes, format reminders, and independently designed compact route contracts moved the instability but caused regressions in other CHAT, refresh, verification, or tool-routing cases. No production route-prompt change was retained.
- Global grammar-constrained decoding removed malformed syntax and control loops, but changed intentional direct-prose behavior, added repeated route-generation overhead, and produced parser-valid yet operationally inadequate arguments.
- Evidence-selective grammar and one malformed/control-loop grammar retry were also rejected. Valid JSON did not guarantee a correct semantic decision: observed regressions included verification requests becoming `CHAT`, adjacent comparisons selecting `list_directory`, unsafe path or shell quoting, and omitted directory options.
- Direct one-sentence route answers remain intentional behavior for suitable requests. Grammar guarantees output syntax only; it does not prove route semantics or argument adequacy.
- The current conservative `chat_final_retry` and existing route fallback remain the safe production behavior. Do not continue route micro-patches, grammar integration, or route-contract redesign without new model or backend evidence.

## Tool Argument Validation Technical Stop

- Generic same-family model repair is rejected. Preserving the selected tool name did not prevent semantic rewriting; an empty web-search request was repaired into an invented non-empty query.
- The canonical runtime contract now enforces unequivocal structural invariants such as required fields and exact types, numeric bounds, contradictory flags, unsupported URL schemes, NUL/control characters, duplicate keys, and invalid shell syntax. Existing policy, permission, operational-limit, and executor guardrails remain authoritative after validation.
- Deterministic validation must not infer missing user intent, command adequacy, a requested but absent depth, or corrected quoting for an opaque shell command when no structured path is available.
- Current `read_file` and `grep_search` route behavior is flattened into `exec_shell_full_command`; reliable path and pattern integrity validation would require structured arguments before broader validation could be considered.
- Deterministic formal healing is restricted to syntax-envelope transformations that preserve the exact tool and typed argument values. Do not reopen generic argument repair, semantic argument correction, or command rewriting without a narrower structured representation and new validation evidence.

## Canonical Tool-Call Validation and Healing Technical Stop

- Tool-call healing diagnostics remain shadow-only and OFF by default through `ORBIT_TOOL_CALL_HEALING_SHADOW=0|1`. Shadow candidates are diagnostic objects only and cannot reach normalization, guardrails, policy, an executor, or the normal tool loop.
- The benchmark-only generation mode stops after one production-template tool-mode model result. It does not execute a tool or start `final_from_tool`, and records only bounded, redacted metadata.
- `no_attempt` is a normal non-error outcome. Timeout and cancel results are non-evaluable. `budget_truncation` records `finish_reason=length`, while `truncated_attempt` is reserved for structurally incomplete JSON detected by the scanner; neither is silently treated as a successful repair.
- The current real Gemma 4 12B sample contained 39 evaluable outputs at a 48-token tool-call budget: 34 were strict-valid first pass, and no natural deterministic formal-repair category was observed. No multiple candidate or markup-leakage event was observed.
- The dominant observed failures were semantic or budget-related: the model selected a different tool in the read cases, emitted a tool call for JSON presented as an example, or reached the generation limit. Structural repair cannot safely correct tool selection, decide that an apparent example is non-executable, or reconstruct content lost to a token limit.
- Deterministic formal healing is enabled by default. `ORBIT_TOOL_CALL_HEALING=0` is the immediate kill switch, and invalid values disable it safely. No nudge retry exists. Synthetic replay demonstrates parser mechanics only and is not evidence of production utility.
- The shared strict contract is implemented in the neutral runtime module `orbit.runtime.tool_contract`. It returns one normalized call plus separate schema, permission, policy, and operational-limit outcomes, a terminal decision, and a stable rejection code. Shadow candidates and active post-normalization calls use this API.
- `ORBIT_TOOL_CALL_CANONICAL_GATE=0|1` is the normal-call gate and is enabled by default after the paired Gemma validation. Invalid values disable it safely. `ORBIT_TOOL_CALL_CANONICAL_GATE=0` is the immediate rollback to legacy behavior. ON rejects strict-invalid calls before `execute_tool` without creating values, dropping extras, applying defaults, or applying clamps. The canonical contract reuses the existing shell policy/contract validators rather than copying them.
- Existing command/content normalization remains behaviorally unchanged and runs before the shared contract. Exact canonical backend calls retain their original argument object for strict validation, so extra fields, wrong types, duplicate keys, and out-of-range values cannot be hidden by normalization. The gate rejects multiple calls before tool events or execution; formal healing runs only between active normalization and the same canonical contract.
- The closed healing whitelist contains only known-envelope removal, trailing-comma removal, complete JSON decoding of `arguments`, and registered-wrapper unwrapping. Reinsertion requires the entire output to be one strong tool-template envelope, one complete candidate, unchanged exact tool name, identical typed argument keys/values and count, an idempotent canonical form, and passing schema, permission, policy, and operational-limit outcomes. Length, cancel, timeout, Markdown/examples, external prose, aliases, top-level arguments, inferred delimiters, schema failures, and policy failures are never authorized.
- An authorized repair creates only a normal `tool_calls` structure and cannot call an executor directly. Runtime ordering is now `normalization -> optional formal repair -> canonical contract -> loop guardrails -> executor`. The repair returns its canonical decision, the loop reuses that decision without validating again, and the executor consumes the same attestation before `execute_tool`.
- Permission precedence is centralized in the canonical contract. A name outside the turn allow-list is `tool_not_enabled`; a permitted name without a registered schema is `unknown_tool`. Schema-, permission-, and operational-limit-invalid calls skip argument-reading loop guardrails and cannot reach `execute_tool`; policy outcomes are evaluated once by the contract and may be consumed by the existing bounded runtime guardrails before the executor returns the same attested denial.
- The controlled legacy-divergence replay currently records 16/16 expected outcomes across all four tools: four calls depend on executor defaults, three on clamps, four on ignored extra fields, one discovers a missing required field in the executor, and one is denied by existing policy. Gate ON enables no new execution, and no healing candidate is executed by this replay.
- The paired production-budget Gemma baseline completed 8/8 correct stop scenarios with gate OFF and 8/8 with gate ON. Tool selection and model-call counts matched exactly; neither mode produced a canonical rejection, timeout, duplicate event, or lifecycle difference. Median total step time was 149.57 seconds OFF and 145.65 seconds ON, but CPU timing and output lengths differed and no speedup is claimed.
- Valid OFF/ON calls preserve tool name, argument values, model-call count, tool events, executor result, and lifecycle in the covered matrix. All four runtime schemas now declare required fields, exact types, real ranges, and `additionalProperties=false`; the canonical gate is enabled by default with immediate legacy rollback through `ORBIT_TOOL_CALL_CANONICAL_GATE=0`. No natural production-budget malformed repair sample has been observed, so default healing makes no success-rate or performance claim.
- The measured in-memory p95 was 113.2 microseconds for a canonical strict analysis, 127.2 microseconds for trailing-comma analysis plus equivalence proof, and 10.4 microseconds for strict observation of an active call. These are local diagnostic measurements, not runtime latency guarantees.
- Final gate microbenchmarks measured OFF at 10.65 microseconds p50 / 14.58 p95 and ON at 25.97 p50 / 42.49 p95 for a valid `system_info` call with execution stubbed. Local synchronous JSONL measured 217.1 microseconds p50 / 245.2 p95; a simulated 5 ms sink measured 5.46 ms p50 / 5.64 ms p95. Slow diagnostic storage therefore transfers latency to the request even though diagnostic failures remain non-behavioral.
- Opt-in healing microbenchmarks measured the disabled resolver at 0.82 microseconds p50 / 0.86 p95, a complete strong repair plus certificate at 139.16 p50 / 175.03 p95, and rejection of an external-prose example at 51.87 p50 / 74.19 p95.
- A production-budget generation-only smoke produced eight evaluable outputs: seven strict-valid calls and one `no_attempt` length completion. It produced zero natural formal-repair candidates, zero multiple candidates, and zero markup leakage. The observed failures remained one semantic wrong-tool choice and one unwanted tool call for a JSON example; neither is repairable by this mechanism.
- The complete 40-case production-budget generation corpus was also fully evaluable: 30 valid-first-pass calls, eight `no_attempt` outputs, four budget-length completions, one ambiguous invalid-JSON/tool-selection output with markup leakage, and one schema-invalid extra-argument output on a negative example. It produced zero whitelist repair candidates, zero multiple candidates, zero tool executions, and healthy cleanup in 40/40. Semantic outcomes included five wrong-tool choices and two unwanted tool calls.
- A five-case end-to-end OFF/ON comparison triggered no repairs. OFF completed 5/5 correct stop steps; ON completed 4/5, with the multiple-tool scenario ending in a one-tool `finish_reason=length` result. Since no repair was attempted and model-call state was process-isolated, this is not evidence that a repair transformation caused the difference, but it provides no success-rate benefit and cannot support default promotion.
- Current production-budget evidence totals 48 generation-only attempts and 12 labelled negatives in this validation block. The earlier 39-output sample used a lower 48-token budget and is not counted toward the production-budget threshold. The requested 500 attempts plus 500 negatives remain uncollected; measured warm throughput projects roughly 3.15 hours for 1,000 generation-only calls before cold blocks and end-to-end coverage.
- Final live sanity retained default final-prefix capture followed by `cached=64`, while `ORBIT_FINAL_PREFIX_REUSE=0` retained `cached=4` with zero prefix activity. The strict MTP registry path loaded mmproj, reported an initialized usable MTP session with no failure, completed 2/2 stop responses, and kept final-prefix capture/restore at zero.
- Do not expand the healing whitelist or add a nudge retry without natural, repeatable malformed tool-call events observed at the production budget. Any expansion still requires zero false positives and unsafe acceptance in a sufficiently large real sample; replay-only evidence is insufficient.
- A future tool-selection reliability investigation must be separate from formal healing. It may measure wrong-tool and unwanted-tool decisions, but must not add fuzzy matching, deterministic semantic routing, invented arguments, or tool substitution to this mechanism.

## Native Backend Compatibility Observability

- Native `/props` exposes a bounded `native_backend_capabilities` manifest for the `orbit-gemma4-native-v1` profile. It is observational and must not alter startup eligibility, inference, routing, tools, healing, MTP, or final-prefix behavior.
- The manifest fingerprints the loaded `llama.cpp` build and library, the exact aligned final-prefix tokenizer boundary, and a versioned Gemma 4 renderer fixture suite. It contains hashes and bounded identifiers only; prompts, outputs, evidence, tool arguments, model paths, and environment values are excluded.
- Renderer conformance covers tool declaration/generation, a complete tool round trip, nested and escaped argument shapes, and tool error responses. Golden fixture hashes are checked in explicitly; expectations must never be derived from the implementation under test.
- `verified` means the reviewed backend commit, renderer fixtures, and tokenizer boundary all match. A new backend commit reports `backend_unverified` and remains usable so that its behavior can be measured. Renderer or tokenizer mismatch remains diagnostic and must not be silently treated as verified.
- The generation-only benchmark records a versioned corpus hash, an actual tool-mode protocol hash covering the production system prompt and registered schemas, and a hash of the generation-affecting configuration. Raw corpus prompts and schemas are not copied into comparison output.
- `scripts/compare_tool_call_generation.py` is an offline gate. It accepts an unverified backend revision only when corpus, sample set, protocol, configuration, renderer suite, and tokenizer identity remain comparable. It rejects semantic regressions, markup leakage, multiple candidates, tool execution, finalization, and extra model calls. Timing deltas are informational only.
- Do not make the capability manifest a runtime startup gate without separate evidence and an explicit rollback path. Use it to identify drift, then run the versioned conformance corpus before accepting a new backend or template revision.

## Current Route Priority

- Keep #142: structurally covered evidence omission is effective whenever a valid `CHAT` decision reaches the normal `chat_final` path, and its conservative lifecycle fallbacks remain required.
- Continue observational measurement through the #143 route-output classes and the additive #145 smoke-harness aggregation. Classification must remain diagnostic-only and must not change parser, route, fallback, direct-prose, or tool behavior.
- Treat malformed error-plus-success routes and similar failures as known model-adherence limitations. Preserve their visibility in benchmark results rather than repairing, reclassifying, or hiding them.

## Main Commits

- post-rc38 (unreleased, newest last): `b465ed6` migration-handoff closure,
  `fa67e5a` empty-plan re-ask + atomic `adopt_plan`, `c9ace69` Dell CPU CHAT
  cache baseline, `47d6b4b` measured server startup profile, then
  `c1e4662` DELL-RUNTIME-AND-IBAN-CLOSURE-1 (#324: calibration warm-up +
  tie-break, REPL zero-step report, native thread read-back). See the "Post-RC38 (unreleased on `main`)" entry in Release State.
- rc38 ANALYSIS arc (post-rc37, #260 → `9181c40`): see the RC38 Release State
  entry above for the per-capability merge SHAs (network-deny `35ae411`,
  fromCharCode `1fc5722`, Chr-offset `442d008`, string-array fold `f2f59a4`,
  byte-offset+StrReverse `131c3fd`, OLE/VBA extraction `ce1f750`, autoexec
  `c247187`, evidence-kind fidelity `9181c40`, plus the structured controller,
  completion shadow, source-coverage, and report-grounding fixes).
- rc35–rc37: `ff5541f` (rc37), `8d8f31f` (rc36), `9864c3b` (rc35) — the analysis
  workflow, the report-citation hotfix, and UX/isolation.
- `2aada5c` Add release notes for v0.0.1-rc23
- `0a446a2` Harden mtmd ABI and record vendor provenance (#152)
- `2c40a0b` Add release notes for v0.0.1-rc22
- `c2be0ef` Enable post-tool final prose reuse by default (#151)
- `b19c9ef` Add release notes for v0.0.1-rc21
- `7f46c0c` Add native backend compatibility observability (#150)
- `b7207aa` Add canonical tool-call validation and deterministic healing (#149)
- `73ec021` Add release notes for v0.0.1-rc20
- `6330d85` Enable aligned final tool prefix reuse by default (#147)
- `000fbfe` Record route and argument validation technical stops (#146)
- `3618000` Add route output classification benchmark coverage (#145)
- `6384396` Update agent guidance after route output diagnostics (#144)
- `1dba552` Add diagnostic route output classification (#143)
- `a303a3e` Reduce redundant evidence in conversation reuse prompts (#142)
- `eb68ad3` Add release notes for v0.0.1-rc19
- `2c541e9` Update agent guidance after final prefix benchmark coverage (#140)
- `f4e2226` Add repeatable final prefix benchmark coverage (#139)
- `b02e59a` Update agent guidance after experimental final prefix reuse (#138)
- `a1419d4` Add experimental final tool prefix reuse (#137)
- `230db43` Add release notes for v0.0.1-rc18
- `48b28b3` Update agent guidance after compact evidence reduction (#135)
- `992ba3e` Reduce compact evidence prompt metadata (#134)
- `f171089` Reduce final from tool prompt tokens (#132)
- `0980c3d` Report web search errors without answering from memory (#130)
- `de204cd` Update agent guidance after web search error final view (#129)
- `2bb40b2` Use compact final view for web search errors (#128)
- `ab4dd4f` Normalize agent guidance after v0.0.1-rc17 (#127)
- `358ed99` Add release notes for v0.0.1-rc17
- `f75bb73` Record conversation reuse smoke results (#126)
- `6b11419` Update agent guidance after conversation reuse merge (#125)
- `1d54e9c` Improve route guidance for conversation reuse (#124)
- `3390059` Clarify optional native MTP support (#123)
- `a05a1e9` Add post-RC16 agent guidance (#122)
- `a6133c35` Add release notes for v0.0.1-rc16
- `767ed6e` Document optional MTP model download (#121)
- `400711e` Document bench core metadata and profile guidance (#120)
- `8e830ed` Add bench core metadata header (#119)
- `b700d74` Clarify CPU-first server and MTP guidance (#118)
- `c03533e` Increase system info final budget (#117)
- `91e84e2` Add release notes for v0.0.1-rc15
- `d4991d4` Add user turn lineage to evidence records (#116)
- `d4ae03a` Add evidence lineage diagnostics (#115)

## Suggested Next Objectives

Current state (2026-09-13): rc38 is **RELEASED** — tag `v0.0.1-rc38` → `95ba0d5`,
GitHub pre-release published. The NUC → Dell migration is CLOSED and the Dell is
the active workstation. **Post-release research is UNDER WAY**: `main` is four
commits ahead of the tag at `47d6b4b`, tracked tree clean, no mission in flight.
Closed since the release — do not reopen any of them without new evidence:

- CHAT-FIRST-TURN-CACHE-REUSE-1 → OUTCOME A, no regression, doc only (`c9ace69`);
  it produced the Dell CPU baseline immediately below.
- The empty-plan re-ask and `adopt_plan` atomicity fix (`fa67e5a`).
- The measured server startup profile (`47d6b4b`).
- DELL-RUNTIME-AND-IBAN-CLOSURE-1: cold-first-candidate calibration + REPL zero-step
  report rendering (see the Post-RC38 entry). The Dell currently runs under a 17 W
  package cap (host state, P4): quote 33-38 / 12-16 tok/s for this boot, not the
  52 / 18.5 of the previous one, until the cap is understood — that is an operator
  question (adapter, Dell thermal mode, thermald adaptive), not an Orbit mission.

**Gate: CLEARED — `CPU_BASELINE_READY`.** DELL-POWER-PL2-LADDER-1 resolved the
last thermal blocker: on a normal AC boot, volatile MMIO **PL2 = 30 W** (PL1
untouched) sustains **prefill 52.06 / decode 18.56 tok/s** with no throttle and a
3 °C thermal margin — 30 W is the lowest passing rung and the ladder is closed
(35 / 40 W not tested). The Dell CPU side of DELL-INTEL-GPU-BENCH-1 was therefore
measured, the mission ran, and it closed as `GPU_BACKEND_TECHNICAL_STOP` (see the
DELL-INTEL-GPU-BENCH-1 entry below). History for context: the 17 W AC-boot cap is
firmware/EC state latched at boot by power source (AC boot → PL2 17 W; battery boot
→ PL2 65 W, proven by DELL-POWER-BATTERY-BOOT-REPRO-1); BIOS `UltraPerformance` is
disproven as the fix; the uncapped 65 W battery-boot burst hits TjMax; the 30 W
value is a volatile RAPL write, so it is NOT persistent (the next AC boot re-imposes
17 W). See DELL-POWER-PROFILE-AUDIT-1, DELL-POWER-CAP-ROOT-WITNESS-1,
DELL-POWER-BATTERY-BOOT-REPRO-1 and DELL-POWER-PL2-LADDER-1 in the Post-RC38 entry
and `workdir/diag/dell_power_audit/RESULTS.md`.

The mission below ran under a volatile MMIO **PL2 = 30 W** (PL1 untouched),
restored to 17 W at mission end; both the CPU and the external-backend iGPU legs
ran under that same 30 W as a same-system, same-host-policy comparison (NOT
"power-matched" — the on-package iGPU shares the CPU power/thermal budget).

- **DELL-INTEL-GPU-BENCH-1 (2026-09-13) — COMPLETED; verdict
  `GPU_BACKEND_TECHNICAL_STOP` (classification G4, VULKAN_REGRESSION).** External-backend
  benchmark of the Dell's Intel `xe` iGPU (Vulkan) versus the qualified CPU baseline,
  same Ornith GGUF (sha `42739874…`), same host policy at PL2 = 30 W, external
  llama.cpp pinned to Orbit's backend b9551 (`379ac66`), built outside the repo. Orbit
  was NOT modified. Results:
  - external CPU leg: **51.65 prefill / 18.70 decode tok/s** — equal to the qualified
    CPU baseline, so Orbit adds no measurable CPU overhead;
  - Intel `xe` Vulkan FULL offload (41 layers): **58.8 prefill / 5.71 decode tok/s** —
    decode **−69 %**; prefill +13 % (below the 15 % bar and irrelevant while decode
    dominates);
  - representative 512-prefill + 768-decode raw turn: **CPU 51.0 s vs iGPU 143.2 s
    (2.81× slower)**.
  Full offload was CORRECT (output byte-identical to CPU) and STABLE (78 °C, no
  throttle, no swap thrash), but decode is memory-bandwidth-bound on the UMA iGPU and
  collapses; every offloaded layer only drags decode toward that figure, so no offload
  reaches a useful decode gain. PARTIAL offload (10 layers) hit the **95 °C** safety
  guard and was aborted: the on-package iGPU shares the CPU thermal budget and the
  CPU-only leg already peaks 91 °C at 30 W, leaving no headroom for a dual load. Vulkan
  itself was NOT broken (stable, correct, Intel device only — no llvmpipe), so
  SYCL/Level-Zero is **NOT warranted** (only tested on a G6/Vulkan-specific defect).
  **Orbit stays CPU-only, `gpu_layers=0`; nothing was integrated** — the outcome is a
  TECHNICAL_STOP, not GPU_BACKEND_CANDIDATE_FOR_ORBIT. **Reopen ONLY with materially
  different hardware, driver, or backend evidence.** Constraints honoured: native
  `orbit server` stayed CPU-first with no GPU promise (Anti-Goals); the GPU was
  measured only through an external backend, never as native `orbit server`
  performance (Benchmarking); recorded as a Dell profile without overwriting NUC
  history. Evidence: `workdir/diag/dell_intel_gpu_bench/` (RESULTS.md, per-candidate
  JSON/logs, build metadata; machine-local, gitignored).

### Authoritative Dell CPU CHAT baseline (2026-09-13, `fa67e5a`)

Measured for DELL-INTEL-GPU-BENCH-1 so CPU and GPU are compared under MATCHED
cache states. Ornith-1.5-35B-A3B Q4_K_M, ctx 8192, threads 6/6, batch 256,
ubatch 128, think off, MTP off, tools on. Prompt `hi, who are you?`. Evidence and
raw KV trace: `workdir/diag/chat_prefix_reuse/`.

One CHAT turn is TWO native calls. `cached` below is the route call's, because
that is the only one a prefix serves; `prompt`/`eval` are the turn's sum. The
fast turn is route 940/172 with **768 cached**, plus final 38/38 with 0.

| state | prompt | eval | cached (route) | prefill tok/s | decode tok/s | wall |
|---|---:|---:|---:|---:|---:|---:|
| cold process, FIRST turn | 978 | 210 | 768 | 52.4 | 18.5 | 7.96 s |
| warm process, new session | 978 | 210 | 768 | 54.8 | 19.3 | 7.9 s |
| warm process, new session (repeat) | 978 | 210 | 768 | 55.3 | 19.6 | 7.35 s |
| warm process, 2nd turn same session | 1115 | 175 | 940 † | 47.1 | 17.0 | 7 s |
| **no prefix** (see trigger below) | 978 | 978 | **0** | ~53 | ~19 | **25 s** |

† that 940 is the ROLLING checkpoint reusing the conversation's own tokens, a
different mechanism from the 768 prewarm prefix. Do not read the column as one
thing.

**A cold Dell PROCESS is not a cold CACHE, and the difference is paid off-book.**
The startup prewarm captures the 768-token Ornith CHAT route prefix
(`ornith15-route-prefix-v1`, 77.8 MiB) **before the socket binds** — `app.py:880`
precedes `app.py:900` — at a measured cost of **23.8 s of blocking prefill and
state-save on every server start**, whether or not a chat ever happens.
`ORBIT_KV_PREFIX_PREWARM` defaults to `startup`; the Ornith reuse resolver
defaults to enabled (`ornith_route_prefix.py:45`) and the env var is opt-OUT
(`=0` disables). Note 9a26838's own message says "opt-in", which was wrong when
written — the default has never been False.

**For DELL-INTEL-GPU-BENCH-1 this is the trap to avoid.** An external
`llama-server --base-url` GPU backend has no equivalent prewarm, so comparing it
against the 7.96 s row hands Orbit-CPU a 23.8 s head start that the row does not
show. Either charge the prewarm to the CPU side, compare against the 25 s no-prefix
row, or start the CPU server with `ORBIT_KV_PREFIX_PREWARM=off` — that is the
only clean way to get a server with no prefix, since interrupting the prewarm
with SIGINT exits instead of continuing (`app.py:878-887`). Say which cache is
being matched — a native
`llama_state_seq` checkpoint restore and an external server's own slot prompt
cache are not the same object, so "warm vs warm" needs spelling out.

Raw rates are cache-independent within noise (prefill 47–55 tok/s, decode 17–20
tok/s, n=1 per cell). **These are previous-boot numbers.** After the 14:05 reboot
the same explicit profile measured 33–38 / 12–16 under a 17 W package cap (see the
Post-RC38 entry, DELL-RUNTIME-AND-IBAN-CLOSURE-1); check `/props.native_threads`
and the busy-core frequency before comparing against this table. Latency differences on this box are evaluated-token count,
not throughput — report evaluated tokens beside any wall time.

**What actually costs you the prefix.** Not "an ANALYSIS run": the trigger is a
prompt-cache-MODE change. `_ensure_prompt_cache_mode` (`client.py:1026`) calls
`reset_session_state` on any change, and that invalidates the Ornith route
prefix unconditionally (`client.py:1004-1006`) — the `preserve_*` flags beside it
cover the rolling checkpoint and the Qwen3-Coder prefix, not this one. The mode
string keys off the actual `tools=` payload (`client.py:1438-1441`), and plain
CHAT passes none — the route phase is a contextvar label only (`chat.py:577`), so
route and final are both `chat:thinking=off` and an ordinary turn never resets.
Verified in the trace: `tools_parameter_present: false` on all seven chat route
requests. So the prefix survives ordinary chat indefinitely, and these three
cost the NEXT chat turn:

- an ANALYSIS run (its calls pass `tools=`, then the return to chat is the change);
- any tool-invoking CHAT turn (`chat.py:1184` passes `tools=tools`);
- `/reset` — `app.py:120-128` preserves only the Qwen3-Coder prefix, and that is
  gated on the Qwen3-Coder profile, so Ornith's dies.

**A benchmark run that types `/reset` between iterations lands on the 25 s row
while believing it is on the 7.96 s row.** Starting a fresh client process does
not do this; only the `/reset` command reaches that endpoint.

The cost is exactly one turn: the next route call re-captures lazily (the 768
tokens are a prefix of its own prompt, so the marginal cost is the state save,
not a re-prefill). Measured: CHAT 768 → ANALYSIS → CHAT 0 (25 s) → CHAT 768 →
768 → 768. Investigated under CHAT-FIRST-TURN-CACHE-REUSE-1 and closed as
EXPECTED — no regression, no production change. Do not reopen the invalidation
rule to reclaim ~17 s once per transition.

Observability, RESOLVED (ORNITH-ROUTE-PREFIX-OBSERVABILITY-1): `/props` now
publishes `ornith_route_prefix_reuse` beside the three Qwen prefix-reuse states,
emitting the Ornith `QwenRoutePrefixStatus` fields (enabled/source/config_error,
the initialized flag and capture/restore/fallback/invalidation counts,
`failure_reason`, `last_used`, checkpoint size, and profile/template/tokenizer/
prefix identities via `client.ornith_route_prefix_reuse_status()`), so an Ornith
reuse REFUSAL reason is now observable. The change is additive and observational —
no reuse/cache/route behaviour changed. Per-request reuse is still:
`ORBIT_KV_DIAG=1` (+ `ORBIT_KV_DIAG_FILE`) →
`kv_diag_native_cache.cached_tokens`. Two traps: `kv_diag_route_prefix_anchor`
reports `model_profile_ineligible` for Ornith because it describes the GEMMA
lineage (correct, easily misread as the Ornith refusal); and on a successful
first-turn restore the cache event reads `previous_prompt_tokens: 0` /
`cache_miss_reason: no_previous_prompt` beside `cached_tokens: 768`, because
those fields compare against the previous prompt, not the anchor. Read
`cached_tokens`.

Other post-release research candidates (each a separate mission; do NOT bundle):
- NPU/OpenVINO or DeepSeek/other model research — all post-release, none promised
  for the native CPU server.
- A new corpus item only when a real sample needs a NEW obfuscation family, and
  only if that family can be made deterministic and fail-closed.

Standing guardrails (apply to all post-release work):

1. Keep the formal-healing whitelist fixed; collect natural malformed production-budget events before considering any expansion.
2. Investigate wrong-tool and unwanted-tool reliability only as a separate observational mission, without semantic hardcoding or tool substitution.
3. Use the process-isolated comparator and verified capability manifest before accepting a native backend, renderer, tokenizer, or tool-protocol revision.
4. Run controlled CPU benchmarks with `bench-core` metadata; do not infer speedup from the compatibility comparator.
5. Do not reopen route grammar, evidence selection, or generic argument repair without a new reliable signal and separate evidence.
6. Do not reopen MTP algorithm tuning without new upstream evidence or a strong benchmark.
7. Do not reopen REPORT exact-KV reuse / compaction / excerpt-reduction: each is a standing TECHNICAL_STOP with no new seam (see the RC38 entry).
8. New obfuscation families, additional Office hosts/events, or GPU/SYCL/NPU/DeepSeek work are all post-release; add a family only when a real corpus sample needs it and it can be made deterministic and fail-closed.
9. Consider small UX/documentation improvements only if measurable, isolated, and covered by tests.

## Anti-Goals

- No multi-language rewrite.
- No hardcoded semantic routing.
- No `current_turn`-only evidence selection.
- No MTP default.
- No GPU promise for the native server.
- No release without preflight.
- No benchmark without metadata.
