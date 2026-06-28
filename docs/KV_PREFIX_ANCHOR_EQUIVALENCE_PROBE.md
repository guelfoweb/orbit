# KV Prefix Anchor Equivalence Probe

Commit analyzed: `6eb3acc`

## Goal

This probe checks a narrower question than full runtime integration:

- can the native backend save a checkpoint for a stable prefix `P`
- restore that checkpoint later
- append the same suffix `S`
- and produce the same next-step behavior as baseline `P+S`

The probe is intentionally isolated from the production runtime path.

It does **not** change routing, prompt construction, tool selection, final
policy, or cache behavior in normal Orbit execution.

## Scope

The probe is implemented in `src/orbit/native_llama/prefix_anchor_probe.py`.

It uses only metadata-safe inputs and outputs:

- token counts
- checkpoint size
- token ids for the immediate next token
- logits row hashes
- restore used / restore failed

It does **not** log:

- raw prompt text
- raw tokens
- user content
- tool output
- file/web content

## Method

Given:

- `prefix_text`
- `full_text`

the probe first verifies that `tokenize(prefix_text)` is a true token-prefix of
`tokenize(full_text)`.

If that boundary is valid:

1. decode prefix tokens `P`
2. capture native seq checkpoint for `P`
3. decode suffix tokens `S`
4. record:
   - baseline next token
   - baseline logits hash
5. clear the context memory
6. restore checkpoint `P`
7. decode the same suffix tokens `S`
8. record:
   - restored next token
   - restored logits hash

The probe passes only if restore succeeds and the post-suffix behavior matches.

## APIs Used

The probe relies on native llama.cpp APIs already exposed through bindings:

- `llama_state_seq_get_size`
- `llama_state_seq_get_data`
- `llama_state_seq_set_data`
- `llama_batch_init`
- `llama_batch_free`
- `llama_decode`
- `llama_synchronize`
- `llama_get_logits_ith`
- `llama_vocab_n_tokens`

## What The Probe Proves

If the probe passes for a chosen synthetic prefix/suffix pair, it shows:

- checkpoint capture/restore can be behaviorally equivalent for that isolated
  token prefix and suffix
- the mismatch is not automatically caused by restore itself

This is a stronger result than the earlier route-wide no-go, but it is still
not enough to wire prefix-anchor into the standard runtime path.

## Limits

This probe still does **not** prove:

- that shared-lineage restore in the production route path is safe
- that later phases remain unaffected after a restore-enabled route pass
- that the rendered route prompt always exposes a token-prefix boundary usable
  for capture/restore

So a passing probe is a prerequisite for the next phase, not final proof.

## Next Step If The Probe Passes

The next acceptable patch would be a bounded runtime experiment for the
tools-on route phase only, behind a default-off flag, with:

- prefix boundary validation
- restore fallback to baseline
- strict regression checks on:
  - route outcome
  - model call count
  - retry/repair count
  - file/web/fetch evidence paths

## Next Step If The Probe Fails

If baseline `P+S` and restore(`P`)+`S` diverge in this isolated probe, then the
blocker is lower-level:

- sequence checkpoint semantics are not equivalent enough in practice, or
- additional native state outside the saved seq payload is required.

## Local Result In This Worktree

One real local probe was executed against the standard native backend in a
dedicated subprocess, using a neutral synthetic prefix and suffix with no
person names or user data.

Observed result:

- split boundary valid at token level
- prefix tokens: `11`
- suffix tokens: `10`
- full tokens: `21`
- checkpoint size: `3786160`
- restore used: `true`
- baseline next token == restored next token
- baseline logits hash == restored logits hash

This means the isolated `seq_id=0` checkpoint/restore path can reproduce the
same next-step behavior for at least one small synthetic case.

That is enough to reject the old blanket no-go.

It is **not** enough to claim production safety yet, because the standard
runtime still needs a route-wide boundary plus a shared-lineage safety story
before any runtime hookup is acceptable.
