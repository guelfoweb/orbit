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

## Initial Local Native Tokenizer Attempt

The first real native tokenizer check was attempted in this worktree, but the
native runtime library was not available:

```text
libllama.so not found
```

Initial status:

- text boundary: PASS
- isolated token-boundary probe: PASS
- real native tokenizer boundary: not executed in this worktree

## Real Tokenizer Attempt

Worktree:

```text
/home/guelfoweb/LAB/orbit-kv-route-anchor-runtime
```

Environment discovery:

```text
ORBIT_LLAMA_LIB_DIR: unset
ORBIT_LLAMA_ROOT: unset
```

`resolve_paths()` failed before model loading:

```text
libllama.so not found.
Searched: /home/guelfoweb/LAB/orbit-kv-route-anchor-runtime/src/orbit/native_llama/vendor/lib/libllama.so.
Provide ORBIT_LLAMA_LIB_DIR, --llama-root, or ORBIT_LLAMA_ROOT, or package native libraries under orbit/native_llama/vendor/lib.
```

Scenarios requested for real-tokenizer probing:

| Scenario | Result |
| --- | --- |
| short chat route | not executed, native library unavailable |
| trivial route | not executed, native library unavailable |
| medium no-tool route | not executed, native library unavailable |
| listing route | not executed, native library unavailable |
| file-read route | not executed, native library unavailable |
| web/fetch route | not executed, native library unavailable |

Boundary verdict for that attempt:

```text
BLOCKED_BY_ENVIRONMENT
```

This is not a technical FAIL of the boundary. It means the real tokenizer could
not be loaded in this worktree.

## Real Tokenizer Attempt With Configured Native Library

The tokenizer check was rerun with the native library configured explicitly:

```text
ORBIT_LLAMA_LIB_DIR=/home/guelfoweb/LAB/orbit/src/orbit/native_llama/vendor/lib
```

Resolved environment:

| Field | Value |
| --- | --- |
| Native library | `/home/guelfoweb/LAB/orbit/src/orbit/native_llama/vendor/lib/libllama.so` |
| Model id | `gemma4-12b-it-q4km` |
| Tokenizer API | `NativeLlamaClient.tokenize` |

The probe used neutral synthetic route prompts and recorded metadata only.

| Scenario | token prefix ok | stable tokens | full tokens | LCP with stable | divergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| short chat route | true | 693 | 707 | 693 | none |
| trivial route | true | 693 | 712 | 693 | none |
| medium no-tool route | true | 693 | 722 | 693 | none |
| listing route | true | 693 | 711 | 693 | none |
| file-read route | true | 693 | 712 | 693 | none |
| web route | true | 693 | 714 | 693 | none |
| fetch route | true | 693 | 715 | 693 | none |

Metadata hashes:

| Scenario | stable prefix hash | full prompt hash |
| --- | --- | --- |
| short chat route | `16216b5e9e8f642e170bba51c2b94f13` | `74dbcc3158d62908f8c13857156935a8` |
| trivial route | `16216b5e9e8f642e170bba51c2b94f13` | `31b7ccd1158d9b7cb04029d6302bcb88` |
| medium no-tool route | `16216b5e9e8f642e170bba51c2b94f13` | `b1835d8363ca448eb7690b12681d56e1` |
| listing route | `16216b5e9e8f642e170bba51c2b94f13` | `c5443a543919c9cd40ebc526111a41a5` |
| file-read route | `16216b5e9e8f642e170bba51c2b94f13` | `9aa39b271c9244d9b3ee37d2753835b3` |
| web route | `16216b5e9e8f642e170bba51c2b94f13` | `e3772f6fd62762d655cb9f74fc1db15d` |
| fetch route | `16216b5e9e8f642e170bba51c2b94f13` | `8a0a01742957f3fdd55099ed22b1c650` |

Boundary verdict for the configured real native tokenizer:

```text
PASS
```

The stable route prefix is a true token-prefix of the rendered full route prompt
for all tested scenarios.

## Implication

The next route-prefix-anchor step is now a bounded route-prefix
checkpoint/restore runtime experiment behind
`ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`.

If a future scenario fails this token-prefix invariant:

- next step is a boundary/tokenization fix that preserves the exact rendered
  prompt semantics
- no checkpoint/restore runtime hookup should be attempted before that fix

## Production Behavior

No production runtime path uses this probe.

No prompt, routing, tool selection, final policy, evidence policy, backend
cache behavior, or visible output behavior changes in this phase.
