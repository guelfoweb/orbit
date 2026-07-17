# AGENTS.md

## Role

This file guides engineering agents and future sessions working on Orbit. It preserves the post-`v0.0.1-rc21` project state, separating established facts, decisions, unproven hypotheses, and reasonable next steps.

## Permanent Principles

- Correctness, stability, reliability, and simplicity come before performance.
- Orbit remains Python-first: prefer the standard library and small, readable, debuggable code.
- Primary target: CPU-only Gemma 4 12B through native `orbit server`.
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

- Current published baseline: `v0.0.1-rc21`.
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
- GPU must be measured through an external compatible backend, for example `llama-server --base-url`, not as native `orbit server` performance.
- Native `orbit server` is CPU-first with `gpu_layers=0`.

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

1. Stop and use RC21 as the published baseline.
2. Keep the formal-healing whitelist fixed; collect natural malformed production-budget events before considering any expansion.
3. Investigate wrong-tool and unwanted-tool reliability only as a separate observational mission, without semantic hardcoding or tool substitution.
4. Use the process-isolated comparator and verified capability manifest before accepting a native backend, renderer, tokenizer, or tool-protocol revision.
5. Run controlled CPU benchmarks with `bench-core` metadata; do not infer speedup from the compatibility comparator.
6. Do not reopen route grammar, evidence selection, or generic argument repair without a new reliable signal and separate evidence.
7. Do not reopen MTP algorithm tuning without new upstream evidence or a strong benchmark.
8. Consider small UX/documentation improvements only if measurable, isolated, and covered by tests.

## Anti-Goals

- No multi-language rewrite.
- No hardcoded semantic routing.
- No `current_turn`-only evidence selection.
- No MTP default.
- No GPU promise for the native server.
- No release without preflight.
- No benchmark without metadata.
