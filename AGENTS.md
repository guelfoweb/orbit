# AGENTS.md

## Role

This file guides engineering agents and future sessions working on Orbit. It preserves the post-`v0.0.1-rc17` project state, separating established facts, decisions, unproven hypotheses, and reasonable next steps.

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

- Current published baseline: `v0.0.1-rc17`.
- Release URL: https://github.com/guelfoweb/orbit/releases/tag/v0.0.1-rc17
- Release notes commit: `358ed995fbc0607ebbda15099a8c568223ddb752`.
- Tag object: `8af6bd36d8bf0cd0e450e10af80bf8e5038fc408`.
- Tag commit: `358ed995fbc0607ebbda15099a8c568223ddb752`.
- Prerelease: yes. Latest: false.
- Includes #122, #123, #124, #125, and #126.
- Focus: post-RC16 agent guidance, MTP README clarification, conversation reuse route guidance, and smoke-result notes.
- RC17 validation: MTP shim build PASS, full unit PASS with 985 tests, `simple_chat --mtp-required` PASS, `git diff --check` PASS.
- RC17 MTP sanity: `mtp_enabled=true`, `mtp_initialized=true`, `mtp_failure_reason=null`, `in_flight=false`, `multimodal_available=true`, `mtp_last_completion.success=true`, `mtp_config.n_max=3`.

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

- `cached=4` on `route -> final` is expected: prompts diverge immediately in the system prompt.
- Do not chase `cached=4` with risky redesigns without new evidence.
- Final-prefix checkpoint reuse remains a technical stop: the stable 43-token boundary does not align with production prefill batching, and changing segmentation changes logits.
- The safer path remains reducing evaluated tokens in final/retry prompts.
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

- Post-RC17 change; not included in `v0.0.1-rc17`.
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

- Post-RC17 change; not included in `v0.0.1-rc17`.
- Completes the compact web error final behavior introduced by #128.
- Problem: after #128, a known-query `web_search` error could still lead the final model call to answer from general knowledge as if the search had succeeded.
- Solution: `web_search` evidence with `status=error` now adds `web_search_failed: true` and a narrow final instruction to report the web failure briefly and not answer from general knowledge as if the search succeeded.
- Files touched: `src/orbit/runtime/evidence.py`, `tests/test_evidence.py`, `tests/test_runtime.py`.
- Tests run: `tests.test_evidence tests.test_runtime` PASS with 193 tests, `tests.test_messages tests.test_final_policy tests.test_completion_budget` PASS with 58 tests, `compileall` PASS, `git diff --check` PASS.
- Safety preserved: scoped only to `web_search` with `status=error`; `status=none`, successful web results, and non-web errors are unchanged.
- Full raw web error/output is not reinjected.
- No route/tool-loop changes, no MTP changes, no cache/KV changes, and no global budget changes.

## #132, Reduced final_from_tool Prompt Tokens

- Post-RC17 change; not included in `v0.0.1-rc17`.
- Problem: every `final_from_tool` call evaluated a correct but unnecessarily verbose dedicated system instruction.
- Solution: compact equivalent wording preserves the full contract: answer concisely from tool evidence, do not call tools, do not expose raw tool-call syntax, do not falsely claim lack of access, and report errors briefly.
- Production-tokenizer measurement: the `final_from_tool` system component decreased from 49 to 34 tokens, an exact deterministic reduction of 15 tokens per call.
- No deterministic wall-time improvement is claimed because observed CPU timings were noisy.
- Files touched: `src/orbit/runtime/messages.py`, `tests/test_messages.py`.
- Tests run: messages/final policy/completion budget PASS with 59 tests, evidence/runtime PASS with 193 tests, `compileall` PASS, and `git diff --check` PASS. Full unit discovery previously passed with 989 tests.
- Safety preserved: no route/tool-loop changes, no evidence-selection changes, no MTP changes, no cache/KV changes, and no completion-budget changes.

## #134, Reduced Compact Evidence Prompt Metadata

- Post-RC17 change; not included in `v0.0.1-rc17`.
- Problem: compact model-facing evidence cards still included audit-only provenance fields that were retained elsewhere and were not needed to answer.
- Solution: small compact cards no longer expose `raw_ref`; compact web cards no longer expose `tool`, `raw_ref`, hash, or size.
- Full and medium cards are unchanged. EvidenceStore, raw retrieval, sidecars, route cards, tool messages, evidence identity, hashes, and lineage remain intact outside the model prompt.
- The historical #128 compact web view included raw ref/hash/size; #134 removes those fields only from its model-facing projection without changing #128/#130 web-error correctness.
- `kv_diag_evidence_card_tokens.evidence_id_hash` may be `null` for compact cards without `raw_ref`. This is intentional: `kv_diag_evidence_lineage` independently preserves the hashed `EvidenceRecord.evidence_id`.
- Measured prompt reductions: `system_info` 36 tokens, `read_file` 37, `grep_search` 36, `list_files` 37, `shell_error` 35, `web_none` 72, `web_error` 74, and `web_success` 77.
- Tests run: evidence/runtime/tool-message PASS with 198 tests, messages/final-policy/completion-budget PASS with 59 tests, `compileall` PASS, and `git diff --check` PASS.
- Safety preserved: no route/tool-loop, evidence-selection, MTP, cache/KV, segmentation, completion-budget, system-prompt, or raw-retrieval changes.
- This reduces evaluated dynamic-suffix tokens. It does not fix cache reuse: `cached=4` remains expected from route/final prompt-view divergence.

## Main Commits

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

1. Stop and use RC17 as the stable baseline.
2. Run controlled CPU benchmarks with `bench-core` metadata.
3. Analyze `bench-core` output for regressions or better profiles.
4. Run a lightweight conversation-reuse end-to-end smoke only if a regression or ambiguous behavior appears.
5. Investigate runtime-side `producer_model_call_id` only if needed.
6. Do not reopen evidence selection without a new reliable relevance signal.
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
