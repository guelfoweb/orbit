# CHAT-FINAL-RETRY-HISTORY-27

Baseline: `e20616cb1d2e35ae8dfc032e4fa660160db01835` on the Dell.

Production closure 26 reproduced a conversation failure: after a user declared
the codeword CEDAR and Orbit acknowledged it, recall returned `None`. It also
failed with zero cached tokens. The first lossy transformation was
`ChatRuntime._chat_final_retry_messages`: it selected only the latest user and
one preceding assistant, producing `U2, A1` from `U1, A1, U2`. CEDAR remained in
the assistant acknowledgement, but the user declaration and chronology were
lost. PR #360 had not changed this runtime code.

## Invariant and smallest fix

The retry projection preserves committed user/assistant messages verbatim and
in chronological order, including long assistant answers and session memory.
It copies canonical `self.messages` while excluding tool result messages and
assistant tool-call envelopes. The existing evidence-card helper still supplies
tool evidence. Raw tool result bodies remain excluded.

Route output, route repair instructions, planning output and tool guard/retry
instructions are call-local; they are not committed to `self.messages` before
the final retry. They remain excluded without inspecting prompt text or adding
semantic rules. The normal CHAT policy replacement and context admission remain
unchanged. No cache, backend, profile, tokenizer/template or prompt text changes.

Longer retry prompts are the intentional cost of preserving conversational
meaning. This is a correctness fix, with no performance claim. Other compact
final/repair builders are outside this mission.

## Deterministic proof

Before editing production code, the new streaming regression test failed with
the exact backend request `system, U2, A1` instead of `system, U1, A1, U2`.
It drives the actual route-abort/final-retry path and asserts the captured
messages, rather than trusting a fabricated model answer.

Eight focused tests cover this path, multiple/short histories, full long
assistant content, tool/evidence exclusion, session memory, reset/new runtime
isolation, no duplication and input immutability. Three older expectations were
causally updated: one-user retention and two assistant-truncation expectations
contradicted the corrected invariant. No tokenizer/template fixtures changed.

Six mutations were applied and caught on an isolated copy: old lossy projection,
reversed history, leaked raw tool results, duplicate current user, lost session
memory and truncated committed assistant. The unmodified control passed.

## Live causal validation

Two separate real `orbit server` processes selected Qwen3.8 Flash Next UD-IQ1_M
through the normal model menu and canonical `/srv/orbit-models` resolver.
Both used the unchanged qualified ctx 4096, threads 10/10, batch 256/128,
lazy loading, repack off and MTP off profile; terminal temperature was zero.

- Normal production: declaration and immediate recall both returned `CEDAR`.
  First ROUTE: 1071 input / 1024 cached / 47 evaluated. Recall ROUTE:
  1102 / 1071 / 31. Recall total wall: 10.615 s.
- Cold control: existing prewarm and route-prefix/rolling switches disabled;
  `/session/reset` before each user turn cleared backend state while preserving
  runtime conversation. All four model calls evaluated their entire prompts,
  with zero cached tokens and zero prefix captures/restores. Both answers were
  `CEDAR`; recall total wall was 88.424 s.
- Both recall retries received identical 77-token prompts and generated the
  same two token IDs, `[59964, 905]`. The old corrupted retry had 58 input tokens.
- The required bounded `system_info` smoke passed, reporting the actual Linux
  version through the normal tool/final path. Both servers exited with RC=0.

## Qualification and scope

Focused chat/runtime/history/cache tests: 447 tests, one host-artifact skip,
RC=0. Mandatory ANALYSIS cross-sample gate: 11 tests, RC=0. PR #358 rolling,
#359 shadow and #360 aligned-prewarm tests are included and unchanged.

Full non-live suite: 6020 tests, 45 skips, RC=0 (554.082 s).
The first attempt returned RC=1 because an existing ANALYSIS test searched the
entire appendix for `c2`, which happened to occur in a random evidence UUID
(`bb22dcc2e9d3`). An isolated pristine-baseline/candidate audit reproduced that
same failure with the observed UUID; both passed with a neutral UUID. The
unchanged canonical full-suite rerun passed. No ANALYSIS code or test was edited.

Independent adversarial review: BLOCKER 0 / MAJOR 0, including source/tests,
mutation results, live token diagnostics and the random-ID failure audit.
Targeted compileall and `git diff --check` passed. Diagnostics live under
`workdir/diag/chat_final_retry_history_27/`, including both full-suite attempts,
the pre-fix failure, mutation results, frozen source hashes and live transcripts.
No production closure rerun is performed automatically after this mission.
