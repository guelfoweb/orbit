# KV Route Prefix Token Boundary

Commit analyzed: `57c6a3e`

## Goal

Verify whether the route tools-on text boundary is also a backend-visible token
boundary.

The text boundary is already available from `RoutePromptSegments`:

- `stable_prefix_text`
- `dynamic_suffix_text`
- `full_prompt_text`

The required invariant for a future prefix-anchor runtime experiment is:

```text
tokenize(full_prompt_text).startswith(tokenize(stable_prefix_text))
```

If this invariant does not hold, restoring a checkpoint for
`stable_prefix_text` would not be equivalent to pre-filling the full prompt.

## Implementation

This phase adds an isolated token-boundary probe:

- `probe_route_boundary_token_prefix(...)`
- `RouteBoundaryTokenProbeResult`

The probe accepts a `RoutePromptSegments` value and a tokenizer function. That
keeps it independent from production runtime behavior while still allowing the
same function to be used with the real native tokenizer when available.

The probe reports metadata only:

- `route_boundary_token_prefix_ok`
- `stable_prefix_token_count`
- `full_prompt_token_count`
- `token_lcp_with_stable_prefix`
- `divergence_index`
- `stable_prefix_hash`
- `full_prompt_hash`

It does not report:

- raw prompt text
- raw tokens
- user content
- tool output
- file or web content

## Unit Coverage

The unit tests cover both outcomes with synthetic tokenizers:

- valid token-prefix boundary
- invalid token-prefix boundary with LCP and divergence index

The tests also verify that metadata does not contain raw prompt text or raw
token arrays.

## Local Native Tokenizer Result

The real native tokenizer check was attempted in this worktree, but the native
runtime library was not available:

```text
libllama.so not found
```

Therefore this report does **not** claim a real backend-tokenizer PASS or FAIL.

Current status:

- text boundary: PASS
- isolated token-boundary probe: PASS
- real native tokenizer boundary: not executed in this worktree

## Implication

The next route-prefix-anchor step is blocked until this probe is run against the
real native tokenizer in an environment where the native library and target
model are available.

If the real tokenizer result is PASS:

- next step is a bounded route-prefix checkpoint/restore experiment behind
  `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`

If the real tokenizer result is FAIL:

- next step is a boundary/tokenization fix that preserves the exact rendered
  prompt semantics
- no checkpoint/restore runtime hookup should be attempted before that fix

## Production Behavior

No production runtime path uses this probe.

No prompt, routing, tool selection, final policy, evidence policy, backend
cache behavior, or visible output behavior changes in this phase.
