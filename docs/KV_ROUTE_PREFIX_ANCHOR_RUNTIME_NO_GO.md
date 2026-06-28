# KV Route Prefix Anchor Runtime No-Go

Commit analyzed: `806968a`

## Goal

Evaluate whether Orbit can safely wire a real native KV prefix-anchor into the
production route path for tools-on requests.

The intended scope was deliberately narrow:

- route phase only
- tools-on only
- no chat-final, final-from-tool, or tool-call hookup
- no file, web, fetch, or listing special-casing
- no semantic prompt changes
- no runtime pre-classification before the route pass

This route-wide anchor would be model-guided if implemented correctly: the
runtime would restore only the stable route prefix, then append the same dynamic
suffix and let the model make the same route decision.

## What Is Valid In Principle

A route-wide prefix anchor is not deterministic routing.

The route phase is common to tools-on requests. If the backend can restore the
same stable route prefix and then process the same dynamic suffix, the model
still decides whether to answer directly, return `CHAT`, or select a tool.

So the conceptual target is valid:

1. stable route prefix is already decoded in native KV state
2. dynamic route suffix is decoded normally
3. the model produces the route decision
4. downstream file, web, fetch, listing, and chat behavior stays unchanged

## Route Path Inspection

The current tools-on route path builds route messages in
`src/orbit/runtime/chat.py`:

- the user turn is appended to `self.messages`
- `with_command_system_prompt(self.messages)` replaces or inserts the route
  system prompt
- the backend receives the full route message list

The native client then renders and tokenizes the full prompt in
`src/orbit/native_llama/client.py`:

- `complete_chat(...)` calls `apply_chat_template(...)`
- `apply_chat_template(...)` returns one rendered prompt string
- `complete_prompt(...)` calls `_complete_prompt_standard(...)`
- `_complete_prompt_standard(...)` tokenizes the entire prompt
- `_prepare_memory_for_prompt(...)` compares the whole prompt against
  `cached_prompt_tokens`

At that point the backend has only one token list for the entire rendered
prompt. It does not receive a separate stable prefix token range.

## Real Blocker

The blocker is not that the route comes before the model decision.

The blocker is that Orbit does not currently expose a safe backend-visible split
between:

- stable route prefix tokens
- dynamic route suffix tokens

The runtime has a logical message-level layout, but the native production path
operates on the fully rendered chat-template prompt. Chat-template rendering can
add framing tokens around roles and message boundaries. A message-level prefix
hash is not enough to prove a token-prefix boundary in the final rendered
backend prompt.

Without a backend-visible token split, a route anchor hookup would need to infer
or reconstruct the split after rendering. That is exactly the kind of fragile
prompt-shape dependency this line has been avoiding.

## Shared-Lineage Risk

The standard native path also uses one active lineage:

- one `ctx_tgt`
- one sequence id in production
- one `cached_prompt_tokens` list
- one active memory state

The isolated equivalence probe proves that checkpoint(`P`) + suffix(`S`) can
match baseline `P+S` for a synthetic case. It does not prove that restoring a
route checkpoint inside this shared production lineage is safe across:

- later route calls
- route retry paths
- direct route final answers
- `CHAT` handoff
- final-from-tool handoff
- continuation readiness
- subsequent prompt-cache behavior

This is especially important because `_prepare_memory_for_prompt(...)` mutates
the same native memory and `cached_prompt_tokens` state based on the LCP of the
full prompt.

## Why No Runtime Patch Was Applied

A safe implementation would need all of these at once:

- a validated token-prefix boundary for the rendered route prompt
- capture only after decoding that exact prefix
- restore only when the same rendered prefix identity is proven
- suffix decode that is byte/token equivalent to baseline
- correct update of `cached_prompt_tokens`
- no contamination of later phases sharing the same native context
- fallback to baseline on any mismatch

The first item is not present in the production interface today. The native
backend accepts full messages or a full rendered prompt, not a verified
`prefix_tokens + suffix_tokens` route request.

So this phase does not connect prefix-anchor to runtime behavior.

## Required Minimal Next Patch

The next safe patch is not route optimization yet. It is a boundary-validation
patch.

A minimal acceptable next step would add an internal, feature-gated probe that:

1. renders the route prompt exactly as production does
2. constructs the candidate stable route prefix exactly as production would
3. tokenizes both with the same native tokenizer
4. verifies that prefix tokens are a true prefix of full route tokens
5. reports only metadata:
   - prefix hash
   - prefix token count
   - suffix token count
   - split valid true/false
   - failure reason

That probe must not alter the runtime path.

Only after that boundary is measured as stable should Orbit attempt a bounded
route-prefix-anchor runtime experiment behind `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`.

## Verdict

No-go for runtime hookup in this phase.

Reason: the route-wide anchor is conceptually valid, but the production native
path does not yet expose a safe rendered-token prefix/suffix boundary, and
restore would operate inside the shared standard lineage.

No benchmark A/B was run because no runtime patch was applied.
